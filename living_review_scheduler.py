#!/usr/bin/env python3
"""
living_review_scheduler.py — Scheduler de Living Review automatique par scénario GESICA

Fonctionnement :
  - Tourne en boucle (daemon) ou en mode one-shot
  - Pour chaque scénario, exécute les queries PubMed/bioRxiv/medRxiv définies dans SCENARIO_QUERIES
  - Ingère les nouveaux articles (non déjà en base) dans literature_document
  - Met à jour les embeddings des nouveaux chunks
  - Expose un endpoint /living-review/status dans main.py
  - Envoie un résumé des nouvelles publications (optionnel : webhook Slack/email)

Usage :
  python3 living_review_scheduler.py --mode daemon --interval-hours 24
  python3 living_review_scheduler.py --mode once --scenario epidemic-early-warning
  python3 living_review_scheduler.py --mode once --all-scenarios
"""

import argparse
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("living-review")

# ─── Configuration ────────────────────────────────────────────────────────────

def _load_secrets() -> str:
    """Charge la clé OpenAI depuis secrets.env ou l'environnement."""
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        for path in ["/opt/literev-api/secrets.env", Path(__file__).parent / "secrets.env"]:
            try:
                for line in Path(path).read_text().splitlines():
                    if line.startswith("OPENAI_API_KEY="):
                        key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
            except Exception:
                pass
    return key

DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://literev:MyNewStrongPassword!@10.10.1.10:5432/literev"
)

# ─── Queries PubMed par scénario ──────────────────────────────────────────────

SCENARIO_QUERIES: dict[str, dict] = {
    "epidemic-early-warning": {
        "label": "Alerte précoce épidémique",
        "pubmed_queries": [
            '(influenza OR "respiratory syncytial virus" OR gastroenteritis) AND (surveillance OR "early warning" OR forecasting) AND (emergency medical services OR EMS OR prehospital)',
            '("epidemic threshold" OR "sentinel surveillance" OR "syndromic surveillance") AND (machine learning OR SARIMAX OR Prophet) AND (2022:2026[dp])',
            '("Réseau Sentinelles" OR "ILI surveillance" OR "influenza-like illness") AND (prediction OR forecast OR "time series")',
        ],
        "biorxiv_terms": ["epidemic forecasting", "influenza surveillance EMS", "syndromic surveillance machine learning"],
        "days_lookback": 30,
    },
    "demand-forecasting": {
        "label": "Prévision de la demande EMS",
        "pubmed_queries": [
            '("emergency medical services" OR EMS OR ambulance) AND (demand OR call OR volume) AND (forecasting OR prediction OR "time series") AND (2020:2026[dp])',
            '("ambulance demand" OR "EMS call volume") AND (machine learning OR "random forest" OR XGBoost OR LightGBM OR Prophet)',
            '(weather OR temperature OR season) AND ("EMS demand" OR "ambulance calls") AND (prediction OR forecast)',
        ],
        "biorxiv_terms": ["EMS demand forecasting", "ambulance call prediction"],
        "days_lookback": 30,
    },
    "response-time-optimization": {
        "label": "Optimisation du temps de réponse",
        "pubmed_queries": [
            '("response time" OR "dispatch time") AND ("emergency medical services" OR ambulance) AND (optimization OR routing OR GIS) AND (2020:2026[dp])',
            '("cross-border" OR transfrontier OR "mutual aid") AND (EMS OR ambulance OR prehospital) AND (response OR coordination)',
            '("ambulance placement" OR "base location" OR "coverage optimization") AND (emergency OR EMS)',
        ],
        "biorxiv_terms": ["EMS response time optimization", "ambulance routing"],
        "days_lookback": 30,
    },
    "cardiac-arrest-prediction": {
        "label": "Prédiction arrêt cardiaque OHCA",
        "pubmed_queries": [
            '("out-of-hospital cardiac arrest" OR OHCA) AND (prediction OR forecasting OR "machine learning") AND (2020:2026[dp])',
            '(OHCA OR "cardiac arrest") AND (weather OR temperature OR season OR circadian) AND (incidence OR risk)',
            '("cardiac arrest" OR OHCA) AND ("survival" OR "return of spontaneous circulation" OR ROSC) AND (EMS OR prehospital) AND (2022:2026[dp])',
        ],
        "biorxiv_terms": ["out-of-hospital cardiac arrest prediction", "OHCA machine learning"],
        "days_lookback": 30,
    },
    "heatwave-ems-impact": {
        "label": "Impact canicule sur les EMS",
        "pubmed_queries": [
            '(heatwave OR "heat wave" OR "extreme heat") AND ("emergency medical services" OR EMS OR ambulance) AND (impact OR demand OR calls) AND (2020:2026[dp])',
            '("heat-related illness" OR hyperthermia OR "heat stroke") AND (prehospital OR EMS OR ambulance)',
            '(UTCI OR "universal thermal climate index" OR "wet bulb globe temperature") AND (health OR mortality OR morbidity)',
        ],
        "biorxiv_terms": ["heatwave EMS impact", "extreme heat emergency services"],
        "days_lookback": 30,
    },
    "stroke-detection": {
        "label": "Détection et orientation AVC",
        "pubmed_queries": [
            '(stroke OR "cerebrovascular accident") AND ("door-to-needle" OR "door to needle" OR thrombolysis OR thrombectomy) AND (prehospital OR EMS OR ambulance) AND (2020:2026[dp])',
            '("FAST score" OR "NIHSS" OR "Cincinnati prehospital stroke scale") AND (EMS OR prehospital OR ambulance)',
            '("stroke unit" OR "comprehensive stroke center") AND ("transport time" OR "response time") AND (outcome OR survival)',
        ],
        "biorxiv_terms": ["prehospital stroke detection", "stroke EMS triage"],
        "days_lookback": 30,
    },
    "triage-support": {
        "label": "Aide au triage EMS",
        "pubmed_queries": [
            '(triage OR "CCMU" OR "NEWS2" OR "Manchester triage") AND ("emergency medical services" OR prehospital OR ambulance) AND (2020:2026[dp])',
            '("artificial intelligence" OR "machine learning" OR "natural language processing") AND (triage OR "clinical decision support") AND (emergency OR prehospital)',
            '("undertriage" OR "overtriage") AND (EMS OR ambulance OR prehospital) AND (accuracy OR sensitivity OR specificity)',
        ],
        "biorxiv_terms": ["AI triage emergency", "prehospital triage machine learning"],
        "days_lookback": 30,
    },
    "undertriage-risk": {
        "label": "Risque de sous-triage",
        "pubmed_queries": [
            '(undertriage OR "under-triage") AND (EMS OR ambulance OR prehospital OR emergency) AND (risk OR prediction OR detection) AND (2018:2026[dp])',
            '("triage accuracy" OR "triage error") AND (prehospital OR EMS) AND (outcome OR mortality OR morbidity)',
            '("missed diagnosis" OR "undertriage") AND (trauma OR cardiac OR stroke) AND (ambulance OR EMS)',
        ],
        "biorxiv_terms": ["undertriage prediction EMS", "triage accuracy prehospital"],
        "days_lookback": 60,
    },
    "trauma-care": {
        "label": "Prise en charge trauma",
        "pubmed_queries": [
            '("injury severity score" OR ISS OR TRISS OR RTS) AND (prehospital OR EMS OR ambulance) AND (outcome OR survival OR mortality) AND (2020:2026[dp])',
            '("damage control" OR "damage control resuscitation") AND (prehospital OR EMS OR trauma) AND (2020:2026[dp])',
            '("major trauma" OR "polytrauma") AND ("trauma center" OR "trauma system") AND (EMS OR ambulance OR prehospital) AND (outcome)',
        ],
        "biorxiv_terms": ["prehospital trauma care", "damage control resuscitation EMS"],
        "days_lookback": 30,
    },
    "mass-casualty": {
        "label": "Événement à victimes multiples",
        "pubmed_queries": [
            '("mass casualty" OR "multiple casualty" OR MCI OR MCE) AND (EMS OR ambulance OR prehospital) AND (triage OR management OR response) AND (2018:2026[dp])',
            '("SALT triage" OR "START triage" OR "METHANE") AND (mass casualty OR disaster OR MCI)',
            '("disaster medicine" OR "mass casualty incident") AND (simulation OR planning OR preparedness) AND (EMS OR prehospital)',
        ],
        "biorxiv_terms": ["mass casualty incident EMS", "SALT triage simulation"],
        "days_lookback": 60,
    },
}

# ─── Fonctions utilitaires ────────────────────────────────────────────────────

def _get_db_conn():
    """Retourne une connexion psycopg."""
    try:
        import psycopg
        return psycopg.connect(DB_URL)
    except Exception as e:
        logger.error(f"Connexion DB impossible : {e}")
        return None


def _external_id_exists(conn, external_id: str, source: str) -> bool:
    """Vérifie si un article est déjà en base."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM literature_document WHERE external_id = %s AND source = %s LIMIT 1",
                (external_id, source)
            )
            return cur.fetchone() is not None
    except Exception:
        return False


def _insert_document(conn, doc: dict) -> Optional[int]:
    """Insère un document dans literature_document et retourne son ID."""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO literature_document
                    (title, abstract, source, external_id, url, publication_year,
                     project_context, scenario_type, evidence_category, source_type,
                     metadata, created_at, updated_at)
                VALUES
                    (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CAST(%s AS jsonb), NOW(), NOW())
                ON CONFLICT (external_id, source) DO NOTHING
                RETURNING id
            """, (
                doc.get("title", "")[:500],
                doc.get("abstract", ""),
                doc.get("source", "pubmed"),
                doc.get("external_id", ""),
                doc.get("url", ""),
                doc.get("year"),
                doc.get("project_context", "gesica"),
                doc.get("scenario_type", ""),
                doc.get("evidence_category", "empirical"),
                doc.get("source_type", "journal_article"),
                json.dumps(doc.get("metadata", {})),
            ))
            row = cur.fetchone()
            conn.commit()
            return row[0] if row else None
    except Exception as e:
        conn.rollback()
        logger.warning(f"Erreur insertion doc '{doc.get('title', '')[:60]}': {e}")
        return None


def _insert_chunk(conn, doc_id: int, content: str, chunk_type: str, metadata: dict) -> bool:
    """Insère un chunk dans document_chunk."""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO document_chunk
                    (document_id, content, chunk_type, chunk_index, metadata, created_at)
                VALUES (%s, %s, %s, 0, CAST(%s AS jsonb), NOW())
            """, (doc_id, content[:8000], chunk_type, json.dumps(metadata)))
            conn.commit()
            return True
    except Exception as e:
        conn.rollback()
        logger.warning(f"Erreur insertion chunk doc_id={doc_id}: {e}")
        return False


# ─── Sources d'ingestion ──────────────────────────────────────────────────────

def fetch_pubmed_new(query: str, days: int = 30, max_results: int = 50) -> list[dict]:
    """Récupère les nouveaux articles PubMed via eutils."""
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    date_from = (datetime.now() - timedelta(days=days)).strftime("%Y/%m/%d")
    date_to = datetime.now().strftime("%Y/%m/%d")
    full_query = f"({query}) AND ({date_from}[PDAT]:{date_to}[PDAT])"

    try:
        # esearch
        r = requests.get(f"{base}/esearch.fcgi", params={
            "db": "pubmed", "term": full_query, "retmax": max_results,
            "retmode": "json", "sort": "date"
        }, timeout=15)
        r.raise_for_status()
        pmids = r.json().get("esearchresult", {}).get("idlist", [])
        if not pmids:
            return []

        # efetch XML
        r2 = requests.post(f"{base}/efetch.fcgi", data={
            "db": "pubmed", "id": ",".join(pmids),
            "retmode": "xml", "rettype": "abstract"
        }, timeout=30)
        r2.raise_for_status()

        import xml.etree.ElementTree as ET
        root = ET.fromstring(r2.text)
        articles = []
        for art in root.findall(".//PubmedArticle"):
            try:
                pmid = art.findtext(".//PMID", "")
                title = art.findtext(".//ArticleTitle", "")
                abstract_parts = [t.text or "" for t in art.findall(".//AbstractText")]
                abstract = " ".join(abstract_parts)
                year_el = art.find(".//PubDate/Year")
                year = int(year_el.text) if year_el is not None and year_el.text else None
                journal = art.findtext(".//Journal/Title", "")
                doi_el = art.find(".//ArticleId[@IdType='doi']")
                doi = doi_el.text if doi_el is not None else ""
                authors = []
                for a in art.findall(".//Author"):
                    ln = a.findtext("LastName", "")
                    fn = a.findtext("ForeName", "")
                    if ln:
                        authors.append(f"{ln} {fn}".strip())
                articles.append({
                    "title": title,
                    "abstract": abstract,
                    "source": "pubmed",
                    "external_id": pmid,
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    "year": year,
                    "source_type": "journal_article",
                    "evidence_category": "empirical",
                    "metadata": {
                        "journal": journal,
                        "doi": doi,
                        "authors": "; ".join(authors[:10]),
                        "living_review": True,
                        "ingested_at": datetime.now(timezone.utc).isoformat(),
                    }
                })
            except Exception:
                continue
        time.sleep(0.35)  # rate limit NCBI
        return articles
    except Exception as e:
        logger.warning(f"Erreur PubMed query '{query[:60]}': {e}")
        return []


def fetch_biorxiv_new(term: str, server: str = "medrxiv", days: int = 30) -> list[dict]:
    """Récupère les nouveaux preprints bioRxiv/medRxiv."""
    date_from = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    date_to = datetime.now().strftime("%Y-%m-%d")
    url = f"https://api.biorxiv.org/details/{server}/{date_from}/{date_to}/0/json"
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()
        papers = data.get("collection", [])
        term_lower = term.lower()
        results = []
        for p in papers:
            title = p.get("title", "")
            abstract = p.get("abstract", "")
            if any(w in title.lower() or w in abstract.lower() for w in term_lower.split()):
                doi = p.get("doi", "")
                results.append({
                    "title": title,
                    "abstract": abstract,
                    "source": server,
                    "external_id": doi or p.get("rel_doi", ""),
                    "url": f"https://doi.org/{doi}" if doi else "",
                    "year": int(p.get("date", "2024")[:4]),
                    "source_type": "preprint",
                    "evidence_category": "empirical",
                    "metadata": {
                        "doi": doi,
                        "authors": p.get("authors", ""),
                        "category": p.get("category", ""),
                        "server": server,
                        "living_review": True,
                        "ingested_at": datetime.now(timezone.utc).isoformat(),
                    }
                })
        return results[:20]
    except Exception as e:
        logger.warning(f"Erreur {server} term='{term}': {e}")
        return []


# ─── Logique principale par scénario ─────────────────────────────────────────

def run_living_review_for_scenario(
    scenario_id: str,
    conn,
    dry_run: bool = False,
    days: int = 30,
) -> dict:
    """
    Lance la living review pour un scénario donné.
    Retourne un résumé des nouvelles publications.
    """
    config = SCENARIO_QUERIES.get(scenario_id)
    if not config:
        return {"error": f"Scénario '{scenario_id}' inconnu"}

    label = config["label"]
    logger.info(f"[{scenario_id}] Living review — {label}")

    new_docs = []
    skipped = 0
    errors = 0

    # PubMed
    for query in config.get("pubmed_queries", []):
        articles = fetch_pubmed_new(query, days=days, max_results=30)
        for art in articles:
            art["scenario_type"] = scenario_id
            art["project_context"] = "gesica"
            if _external_id_exists(conn, art["external_id"], art["source"]):
                skipped += 1
                continue
            if dry_run:
                new_docs.append(art)
                continue
            doc_id = _insert_document(conn, art)
            if doc_id:
                content = f"{art['title']}\n\n{art['abstract']}"
                _insert_chunk(conn, doc_id, content, "title_abstract", {
                    "source": art["source"],
                    "living_review": True,
                })
                new_docs.append(art)
            else:
                errors += 1

    # bioRxiv / medRxiv
    for term in config.get("biorxiv_terms", []):
        for server in ["medrxiv", "biorxiv"]:
            preprints = fetch_biorxiv_new(term, server=server, days=days)
            for p in preprints:
                p["scenario_type"] = scenario_id
                p["project_context"] = "gesica"
                if not p["external_id"] or _external_id_exists(conn, p["external_id"], p["source"]):
                    skipped += 1
                    continue
                if dry_run:
                    new_docs.append(p)
                    continue
                doc_id = _insert_document(conn, p)
                if doc_id:
                    content = f"{p['title']}\n\n{p['abstract']}"
                    _insert_chunk(conn, doc_id, content, "title_abstract", {
                        "source": p["source"],
                        "living_review": True,
                    })
                    new_docs.append(p)
                else:
                    errors += 1

    result = {
        "scenario_id": scenario_id,
        "label": label,
        "run_at": datetime.now(timezone.utc).isoformat(),
        "new_documents": len(new_docs),
        "skipped_existing": skipped,
        "errors": errors,
        "dry_run": dry_run,
        "new_titles": [d["title"][:80] for d in new_docs[:5]],
    }
    logger.info(
        f"[{scenario_id}] Terminé — {len(new_docs)} nouveaux, {skipped} déjà en base, {errors} erreurs"
    )
    return result


def run_all_scenarios(conn, dry_run: bool = False, days: int = 30) -> list[dict]:
    """Lance la living review pour tous les scénarios."""
    results = []
    for scenario_id in SCENARIO_QUERIES:
        result = run_living_review_for_scenario(scenario_id, conn, dry_run=dry_run, days=days)
        results.append(result)
        time.sleep(1)  # politesse entre scénarios
    return results


def save_run_report(results: list[dict], output_dir: str = "/opt/literev-api"):
    """Sauvegarde un rapport JSON de la dernière exécution."""
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_new_documents": sum(r.get("new_documents", 0) for r in results),
        "scenarios": results,
    }
    path = Path(output_dir) / "living_review_last_run.json"
    try:
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        logger.info(f"Rapport sauvegardé : {path}")
    except Exception as e:
        logger.warning(f"Impossible de sauvegarder le rapport : {e}")
    return report


# ─── Daemon mode ─────────────────────────────────────────────────────────────

def run_daemon(interval_hours: int = 24, days: int = 7):
    """Tourne en boucle, exécute la living review toutes les N heures."""
    logger.info(f"Démarrage du daemon living review — intervalle {interval_hours}h")
    while True:
        conn = _get_db_conn()
        if conn:
            try:
                results = run_all_scenarios(conn, days=days)
                report = save_run_report(results)
                total = report["total_new_documents"]
                logger.info(f"Cycle terminé — {total} nouveaux documents au total")
            finally:
                conn.close()
        else:
            logger.error("Impossible de se connecter à la base de données")
        logger.info(f"Prochain cycle dans {interval_hours}h")
        time.sleep(interval_hours * 3600)


# ─── Entrypoint ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Living Review Scheduler GESICA")
    parser.add_argument("--mode", choices=["once", "daemon"], default="once",
                        help="Mode d'exécution : once (une fois) ou daemon (boucle)")
    parser.add_argument("--scenario", default=None,
                        help="ID du scénario à traiter (ex: epidemic-early-warning)")
    parser.add_argument("--all-scenarios", action="store_true",
                        help="Traiter tous les scénarios")
    parser.add_argument("--days", type=int, default=30,
                        help="Nombre de jours de lookback (défaut: 30)")
    parser.add_argument("--interval-hours", type=int, default=24,
                        help="Intervalle en heures pour le mode daemon (défaut: 24)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulation sans écriture en base")
    parser.add_argument("--list-scenarios", action="store_true",
                        help="Lister tous les scénarios disponibles")
    args = parser.parse_args()

    if args.list_scenarios:
        print("\nScénarios disponibles :")
        for sid, cfg in SCENARIO_QUERIES.items():
            print(f"  {sid:35s} — {cfg['label']}")
        return

    if args.mode == "daemon":
        run_daemon(interval_hours=args.interval_hours, days=args.days)
        return

    # Mode once
    conn = _get_db_conn()
    if not conn:
        logger.error("Impossible de se connecter à la base de données")
        return

    try:
        if args.scenario:
            result = run_living_review_for_scenario(
                args.scenario, conn, dry_run=args.dry_run, days=args.days
            )
            save_run_report([result])
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.all_scenarios:
            results = run_all_scenarios(conn, dry_run=args.dry_run, days=args.days)
            report = save_run_report(results)
            total = report["total_new_documents"]
            print(f"\n✅ Living review terminée — {total} nouveaux documents")
            for r in results:
                status = "DRY-RUN" if args.dry_run else "OK"
                print(f"  [{status}] {r['scenario_id']:35s} +{r['new_documents']} nouveaux, {r['skipped_existing']} existants")
        else:
            parser.print_help()
    finally:
        conn.close()


if __name__ == "__main__":
    main()
