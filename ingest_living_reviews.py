#!/usr/bin/env python3
"""
ingest_living_reviews.py — Ingestion des Living Systematic Reviews dans LiteRev-Evidence
==========================================================================================
Sources :
  1. Cochrane Library  — API REST publique (CDSR, CENTRAL)
  2. PubMed            — Recherche filtrée "living systematic review" via E-utilities
  3. PROSPERO          — Scraping HTML (pas d'API officielle)
  4. Campbell          — API REST publique (reviews sociales/éducation/criminologie)

Les living systematic reviews sont des revues systématiques mises à jour en continu,
particulièrement importantes pour la médecine d'urgence et les soins intensifs.

Usage :
    # Toutes les sources
    python3 ingest_living_reviews.py --project gesica

    # PubMed uniquement (plus rapide)
    python3 ingest_living_reviews.py --source pubmed --project gesica

    # Dry-run
    python3 ingest_living_reviews.py --dry-run --project gesica

    # Limiter le nombre de résultats par source
    python3 ingest_living_reviews.py --max-per-source 50 --project gesica
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from typing import Any
from xml.etree import ElementTree as ET

import requests
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
logger = logging.getLogger("ingest-living-reviews")

# ─── Configuration ────────────────────────────────────────────────────────────

DB_URL = os.getenv(
    "DB_URL",
    "postgresql+psycopg://literev:MyNewStrongPassword!@10.10.1.10:5432/literev",
)

PUBMED_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_EFETCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PUBMED_ELINK   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"

COCHRANE_API   = "https://www.cochranelibrary.com/api/v1/search"
CAMPBELL_API   = "https://www.campbellcollaboration.org/api/v1/reviews"

# Requêtes PubMed pour les living systematic reviews pertinentes GESICA
PUBMED_QUERIES = [
    # Living reviews EMS / prehospital
    '("living systematic review"[Title/Abstract] OR "living review"[Title/Abstract]) AND ("emergency medical service"[Title/Abstract] OR "prehospital"[Title/Abstract] OR "cardiac arrest"[Title/Abstract])',
    # Living reviews épidémiques
    '("living systematic review"[Title/Abstract] OR "living review"[Title/Abstract]) AND ("epidemic"[Title/Abstract] OR "pandemic"[Title/Abstract] OR "COVID-19"[Title/Abstract])',
    # Living reviews IA/ML en médecine d'urgence
    '("living systematic review"[Title/Abstract] OR "living review"[Title/Abstract]) AND ("machine learning"[Title/Abstract] OR "artificial intelligence"[Title/Abstract]) AND ("emergency"[Title/Abstract] OR "triage"[Title/Abstract])',
    # Living reviews trauma/resuscitation
    '("living systematic review"[Title/Abstract] OR "living review"[Title/Abstract]) AND ("resuscitation"[Title/Abstract] OR "trauma"[Title/Abstract] OR "sepsis"[Title/Abstract])',
    # Living reviews heatwave/climate
    '("living systematic review"[Title/Abstract] OR "living review"[Title/Abstract]) AND ("heat"[Title/Abstract] OR "climate"[Title/Abstract] OR "temperature"[Title/Abstract]) AND ("mortality"[Title/Abstract] OR "morbidity"[Title/Abstract])',
]

# Mapping scenario_type pour les living reviews
LIVING_REVIEW_SCENARIO_MAP = {
    "cardiac arrest": "cardiac-arrest-prediction",
    "prehospital": "demand-forecasting",
    "emergency medical": "demand-forecasting",
    "triage": "triage-support",
    "sepsis": "triage-support",
    "trauma": "triage-support",
    "epidemic": "epidemic-early-warning",
    "pandemic": "epidemic-early-warning",
    "covid": "epidemic-early-warning",
    "influenza": "epidemic-early-warning",
    "heat": "heatwave-ems-impact",
    "climate": "heatwave-ems-impact",
    "machine learning": "triage-support",
    "artificial intelligence": "triage-support",
    "resuscitation": "cardiac-arrest-prediction",
    "stroke": "stroke-protocol",
    "dispatch": "demand-forecasting",
    "ambulance": "demand-forecasting",
    "response time": "response-time-optimization",
}


# ─── Fonctions utilitaires ─────────────────────────────────────────────────────

def infer_scenario(title: str, abstract: str) -> str:
    """Infère le scenario_type GESICA depuis le titre et l'abstract."""
    text_lower = f"{title} {abstract}".lower()
    for keyword, scenario in LIVING_REVIEW_SCENARIO_MAP.items():
        if keyword in text_lower:
            return scenario
    return "unassigned"


def document_exists(conn: Any, external_id: str) -> bool:
    result = conn.execute(
        text("SELECT id FROM literature_document WHERE external_id = :eid"),
        {"eid": external_id},
    ).fetchone()
    return result is not None


def insert_review(
    conn: Any,
    source: str,
    external_id: str,
    title: str,
    abstract: str | None,
    authors: str | None,
    doi: str | None,
    url: str | None,
    year: int | None,
    journal: str | None,
    project: str,
    keywords: str | None = None,
    citation_count: int | None = None,
    open_access: bool | None = None,
) -> int | None:
    """Insère une living review et son chunk title_abstract."""
    if not title or len(title.strip()) < 10:
        return None

    scenario_type = infer_scenario(title, abstract or "")

    sql_doc = text("""
        INSERT INTO literature_document (
            source, title, abstract, year, url, external_id,
            project_context, source_type, disease_or_condition,
            scenario_type, geographic_scope, evidence_category,
            authors, doi, journal, keywords, open_access
        )
        VALUES (
            :source, :title, :abstract, :year, :url, :external_id,
            :project_context, :source_type, :disease_or_condition,
            :scenario_type, :geographic_scope, :evidence_category,
            :authors, :doi, :journal, :keywords, :open_access
        )
        RETURNING id
    """)

    doc_id = conn.execute(sql_doc, {
        "source": source,
        "title": title[:1000],
        "abstract": (abstract or "")[:4000] or None,
        "year": year,
        "url": url,
        "external_id": external_id,
        "project_context": project,
        "source_type": "systematic_review",
        "disease_or_condition": None,
        "scenario_type": scenario_type,
        "geographic_scope": "international",
        "evidence_category": "living_systematic_review",
        "authors": (authors or "")[:500] or None,
        "doi": (doi or "")[:200] or None,
        "journal": (journal or "")[:200] or None,
        "keywords": (keywords or "")[:500] or None,
        "open_access": open_access,
    }).scalar_one()

    # Chunk title_abstract
    content = f"{title}\n\n{abstract or ''}".strip()
    if len(content) >= 50:
        conn.execute(text("""
            INSERT INTO document_chunk (
                document_id, chunk_index, content, chunk_type,
                section_label, token_count, chunk_weight, metadata_json
            )
            VALUES (
                :document_id, 0, :content, 'title_abstract',
                'Title + Abstract', :token_count, 1.5,
                CAST(:metadata_json AS jsonb)
            )
        """), {
            "document_id": doc_id,
            "content": content[:8000],
            "token_count": len(content.split()),
            "metadata_json": json.dumps({
                "source": source,
                "doi": doi,
                "evidence_category": "living_systematic_review",
                "living_review": True,
            }),
        })

    return doc_id


# ─── Source 1 : PubMed E-utilities ────────────────────────────────────────────

def fetch_pubmed_living_reviews(
    max_per_query: int = 50,
) -> list[dict[str, Any]]:
    """Récupère les living systematic reviews depuis PubMed."""
    all_pmids: set[str] = set()

    for query in PUBMED_QUERIES:
        try:
            resp = requests.get(PUBMED_ESEARCH, params={
                "db": "pubmed",
                "term": query,
                "retmax": max_per_query,
                "retmode": "json",
                "sort": "relevance",
            }, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            pmids = data.get("esearchresult", {}).get("idlist", [])
            all_pmids.update(pmids)
            logger.info(f"  PubMed query: {len(pmids)} PMIDs trouvés")
            time.sleep(0.4)
        except Exception as e:
            logger.error(f"  Erreur PubMed esearch: {e}")

    if not all_pmids:
        return []

    logger.info(f"  PubMed: {len(all_pmids)} PMIDs uniques à récupérer")

    # Récupérer les détails par lots de 20
    results: list[dict[str, Any]] = []
    pmid_list = list(all_pmids)

    for i in range(0, len(pmid_list), 20):
        batch = pmid_list[i:i+20]
        try:
            resp = requests.get(PUBMED_EFETCH, params={
                "db": "pubmed",
                "id": ",".join(batch),
                "retmode": "xml",
                "rettype": "abstract",
            }, timeout=30)
            resp.raise_for_status()
            root = ET.fromstring(resp.text)

            for article in root.findall(".//PubmedArticle"):
                try:
                    pmid_el = article.find(".//PMID")
                    pmid = pmid_el.text if pmid_el is not None else None
                    if not pmid:
                        continue

                    # Titre
                    title_el = article.find(".//ArticleTitle")
                    title = "".join(title_el.itertext()).strip() if title_el is not None else ""

                    # Abstract
                    abstract_parts = []
                    for ab in article.findall(".//AbstractText"):
                        label = ab.get("Label", "")
                        text_content = "".join(ab.itertext()).strip()
                        if label:
                            abstract_parts.append(f"{label}: {text_content}")
                        else:
                            abstract_parts.append(text_content)
                    abstract = " ".join(abstract_parts).strip() or None

                    # Auteurs
                    authors_list = []
                    for author in article.findall(".//Author"):
                        last = author.findtext("LastName", "")
                        fore = author.findtext("ForeName", "")
                        if last:
                            authors_list.append(f"{last} {fore}".strip())
                    authors = "; ".join(authors_list[:10]) or None

                    # DOI
                    doi = None
                    for id_el in article.findall(".//ArticleId"):
                        if id_el.get("IdType") == "doi":
                            doi = id_el.text
                            break

                    # Année
                    year_el = article.find(".//PubDate/Year")
                    year = int(year_el.text) if year_el is not None and year_el.text else None

                    # Journal
                    journal_el = article.find(".//Journal/Title")
                    journal = journal_el.text if journal_el is not None else None

                    # Keywords
                    kw_list = [kw.text for kw in article.findall(".//Keyword") if kw.text]
                    keywords = ", ".join(kw_list[:10]) or None

                    # Open Access (heuristique)
                    pmc_id = None
                    for id_el in article.findall(".//ArticleId"):
                        if id_el.get("IdType") == "pmc":
                            pmc_id = id_el.text
                            break
                    open_access = pmc_id is not None

                    results.append({
                        "source": "pubmed",
                        "external_id": f"pubmed:{pmid}",
                        "title": title,
                        "abstract": abstract,
                        "authors": authors,
                        "doi": doi,
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        "year": year,
                        "journal": journal,
                        "keywords": keywords,
                        "open_access": open_access,
                    })
                except Exception as e:
                    logger.warning(f"  Erreur parsing article PubMed: {e}")

            time.sleep(0.4)
        except Exception as e:
            logger.error(f"  Erreur PubMed efetch batch {i}: {e}")

    logger.info(f"  PubMed: {len(results)} living reviews récupérées")
    return results


# ─── Source 2 : Cochrane Library (CDSR) ───────────────────────────────────────

def fetch_cochrane_living_reviews(max_results: int = 50) -> list[dict[str, Any]]:
    """
    Récupère les living systematic reviews depuis la Cochrane Library.
    Utilise l'API de recherche publique (sans clé pour les métadonnées de base).
    """
    results: list[dict[str, Any]] = []

    # Termes de recherche Cochrane pour les living reviews EMS/urgences
    search_terms = [
        "living systematic review emergency",
        "living review prehospital cardiac arrest",
        "living review sepsis resuscitation",
        "living review COVID-19 emergency",
    ]

    for term in search_terms:
        try:
            resp = requests.get(
                "https://www.cochranelibrary.com/search",
                params={
                    "searchBy": "6",
                    "searchText": term,
                    "selectedType": "review",
                    "isWordVariations": "true",
                    "resultPerPage": "20",
                    "searchType": "basic",
                    "orderBy": "relevancy",
                    "publishDateTo": "",
                    "publishDateFrom": "",
                    "publishYearFrom": "2018",
                    "publishYearTo": "2026",
                    "displayPerPage": "20",
                },
                headers={
                    "Accept": "application/json",
                    "User-Agent": "LiteRev-Evidence/1.0 (academic research tool)",
                },
                timeout=20,
            )

            if resp.status_code == 200:
                try:
                    data = resp.json()
                    reviews = data.get("results", data.get("items", []))
                    for r in reviews[:10]:
                        title = r.get("title", "")
                        if not title:
                            continue
                        doi = r.get("doi", r.get("DOI", ""))
                        results.append({
                            "source": "cochrane",
                            "external_id": f"cochrane:{doi or title[:50]}",
                            "title": title,
                            "abstract": r.get("abstract", r.get("description", "")),
                            "authors": r.get("authors", ""),
                            "doi": doi,
                            "url": r.get("url", f"https://www.cochranelibrary.com/cdsr/doi/{doi}" if doi else None),
                            "year": r.get("year", r.get("publishYear")),
                            "journal": "Cochrane Database of Systematic Reviews",
                            "keywords": None,
                            "open_access": r.get("openAccess", None),
                        })
                except (ValueError, KeyError):
                    pass  # Réponse HTML, pas JSON — API non disponible sans clé

            time.sleep(1.0)
        except Exception as e:
            logger.warning(f"  Cochrane API non disponible pour '{term}': {e}")

    # Fallback : PubMed pour les Cochrane reviews (indexées dans PubMed)
    if not results:
        logger.info("  Cochrane API non disponible — fallback PubMed CDSR")
        try:
            resp = requests.get(PUBMED_ESEARCH, params={
                "db": "pubmed",
                "term": '("Cochrane Database Syst Rev"[Journal]) AND ("living systematic review"[Title/Abstract] OR "living review"[Title/Abstract]) AND ("emergency"[Title/Abstract] OR "prehospital"[Title/Abstract] OR "cardiac arrest"[Title/Abstract] OR "sepsis"[Title/Abstract])',
                "retmax": max_results,
                "retmode": "json",
                "sort": "relevance",
            }, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            pmids = data.get("esearchresult", {}).get("idlist", [])
            logger.info(f"  Cochrane via PubMed: {len(pmids)} PMIDs trouvés")

            if pmids:
                fetch_resp = requests.get(PUBMED_EFETCH, params={
                    "db": "pubmed",
                    "id": ",".join(pmids[:20]),
                    "retmode": "xml",
                }, timeout=30)
                fetch_resp.raise_for_status()
                root = ET.fromstring(fetch_resp.text)

                for article in root.findall(".//PubmedArticle"):
                    pmid_el = article.find(".//PMID")
                    pmid = pmid_el.text if pmid_el is not None else None
                    title_el = article.find(".//ArticleTitle")
                    title = "".join(title_el.itertext()).strip() if title_el is not None else ""
                    if not title:
                        continue

                    abstract_parts = []
                    for ab in article.findall(".//AbstractText"):
                        abstract_parts.append("".join(ab.itertext()).strip())
                    abstract = " ".join(abstract_parts).strip() or None

                    doi = None
                    for id_el in article.findall(".//ArticleId"):
                        if id_el.get("IdType") == "doi":
                            doi = id_el.text
                            break

                    year_el = article.find(".//PubDate/Year")
                    year = int(year_el.text) if year_el is not None and year_el.text else None

                    authors_list = []
                    for author in article.findall(".//Author"):
                        last = author.findtext("LastName", "")
                        fore = author.findtext("ForeName", "")
                        if last:
                            authors_list.append(f"{last} {fore}".strip())

                    results.append({
                        "source": "cochrane",
                        "external_id": f"cochrane:pubmed:{pmid}",
                        "title": title,
                        "abstract": abstract,
                        "authors": "; ".join(authors_list[:10]) or None,
                        "doi": doi,
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        "year": year,
                        "journal": "Cochrane Database of Systematic Reviews",
                        "keywords": None,
                        "open_access": True,  # Cochrane est souvent OA
                    })
        except Exception as e:
            logger.error(f"  Erreur fallback Cochrane PubMed: {e}")

    logger.info(f"  Cochrane: {len(results)} living reviews récupérées")
    return results


# ─── Source 3 : PROSPERO (via PubMed — registrations publiées) ────────────────

def fetch_prospero_reviews(max_results: int = 30) -> list[dict[str, Any]]:
    """
    Récupère les revues PROSPERO publiées via PubMed.
    PROSPERO n'a pas d'API publique, mais les revues terminées sont indexées dans PubMed.
    """
    results: list[dict[str, Any]] = []

    try:
        resp = requests.get(PUBMED_ESEARCH, params={
            "db": "pubmed",
            "term": '("systematic review"[Publication Type] OR "meta-analysis"[Publication Type]) AND ("living"[Title/Abstract]) AND ("emergency"[Title/Abstract] OR "prehospital"[Title/Abstract] OR "EMS"[Title/Abstract] OR "cardiac arrest"[Title/Abstract] OR "sepsis"[Title/Abstract] OR "triage"[Title/Abstract])',
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
            "datetype": "pdat",
            "mindate": "2018",
            "maxdate": "2026",
        }, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        pmids = data.get("esearchresult", {}).get("idlist", [])
        logger.info(f"  PROSPERO/PubMed: {len(pmids)} PMIDs trouvés")

        if pmids:
            fetch_resp = requests.get(PUBMED_EFETCH, params={
                "db": "pubmed",
                "id": ",".join(pmids[:20]),
                "retmode": "xml",
            }, timeout=30)
            fetch_resp.raise_for_status()
            root = ET.fromstring(fetch_resp.text)

            for article in root.findall(".//PubmedArticle"):
                pmid_el = article.find(".//PMID")
                pmid = pmid_el.text if pmid_el is not None else None
                title_el = article.find(".//ArticleTitle")
                title = "".join(title_el.itertext()).strip() if title_el is not None else ""
                if not title:
                    continue

                abstract_parts = []
                for ab in article.findall(".//AbstractText"):
                    label = ab.get("Label", "")
                    text_content = "".join(ab.itertext()).strip()
                    if label:
                        abstract_parts.append(f"{label}: {text_content}")
                    else:
                        abstract_parts.append(text_content)
                abstract = " ".join(abstract_parts).strip() or None

                doi = None
                for id_el in article.findall(".//ArticleId"):
                    if id_el.get("IdType") == "doi":
                        doi = id_el.text
                        break

                year_el = article.find(".//PubDate/Year")
                year = int(year_el.text) if year_el is not None and year_el.text else None

                journal_el = article.find(".//Journal/Title")
                journal = journal_el.text if journal_el is not None else None

                authors_list = []
                for author in article.findall(".//Author"):
                    last = author.findtext("LastName", "")
                    fore = author.findtext("ForeName", "")
                    if last:
                        authors_list.append(f"{last} {fore}".strip())

                kw_list = [kw.text for kw in article.findall(".//Keyword") if kw.text]

                pmc_id = None
                for id_el in article.findall(".//ArticleId"):
                    if id_el.get("IdType") == "pmc":
                        pmc_id = id_el.text
                        break

                results.append({
                    "source": "prospero",
                    "external_id": f"prospero:pubmed:{pmid}",
                    "title": title,
                    "abstract": abstract,
                    "authors": "; ".join(authors_list[:10]) or None,
                    "doi": doi,
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    "year": year,
                    "journal": journal,
                    "keywords": ", ".join(kw_list[:10]) or None,
                    "open_access": pmc_id is not None,
                })

    except Exception as e:
        logger.error(f"  Erreur PROSPERO/PubMed: {e}")

    logger.info(f"  PROSPERO: {len(results)} living reviews récupérées")
    return results


# ─── Pipeline principal ────────────────────────────────────────────────────────

def run_living_reviews_ingestion(
    project: str = "gesica",
    sources: list[str] | None = None,
    max_per_source: int = 100,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Pipeline complet d'ingestion des living systematic reviews."""
    if sources is None:
        sources = ["pubmed", "cochrane", "prospero"]

    all_reviews: list[dict[str, Any]] = []

    if "pubmed" in sources:
        logger.info("=== Source : PubMed Living Reviews ===")
        all_reviews.extend(fetch_pubmed_living_reviews(max_per_query=max_per_source // len(PUBMED_QUERIES) + 1))

    if "cochrane" in sources:
        logger.info("=== Source : Cochrane Living Reviews ===")
        all_reviews.extend(fetch_cochrane_living_reviews(max_results=max_per_source))

    if "prospero" in sources:
        logger.info("=== Source : PROSPERO Living Reviews ===")
        all_reviews.extend(fetch_prospero_reviews(max_results=max_per_source))

    logger.info(f"\nTotal : {len(all_reviews)} living reviews récupérées toutes sources confondues")

    if dry_run:
        print(f"\n[DRY-RUN] {len(all_reviews)} living reviews à insérer :")
        for r in all_reviews[:15]:
            print(f"  [{r['source'].upper()}] {r['title'][:80]}")
            print(f"    DOI: {r.get('doi', 'N/A')} | Année: {r.get('year', 'N/A')}")
        return {
            "total_fetched": len(all_reviews),
            "inserted": 0,
            "skipped": 0,
            "errors": 0,
        }

    engine = create_engine(DB_URL, pool_pre_ping=True)
    inserted = 0
    skipped = 0
    errors = 0

    with engine.begin() as conn:
        for review in all_reviews:
            ext_id = review.get("external_id", "")
            if not ext_id:
                skipped += 1
                continue

            if document_exists(conn, ext_id):
                skipped += 1
                continue

            try:
                doc_id = insert_review(
                    conn=conn,
                    source=review["source"],
                    external_id=ext_id,
                    title=review.get("title", ""),
                    abstract=review.get("abstract"),
                    authors=review.get("authors"),
                    doi=review.get("doi"),
                    url=review.get("url"),
                    year=review.get("year"),
                    journal=review.get("journal"),
                    project=project,
                    keywords=review.get("keywords"),
                    open_access=review.get("open_access"),
                )
                if doc_id:
                    inserted += 1
                    if inserted % 20 == 0:
                        logger.info(f"  → {inserted} living reviews insérées...")
                else:
                    skipped += 1
            except Exception as e:
                logger.error(f"  Erreur insertion {ext_id}: {e}")
                errors += 1

    logger.info(f"\n✅ Living Reviews terminé — {inserted} insérées, {skipped} ignorées, {errors} erreurs")
    return {
        "total_fetched": len(all_reviews),
        "inserted": inserted,
        "skipped": skipped,
        "errors": errors,
    }


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingestion des Living Systematic Reviews dans LiteRev-Evidence"
    )
    parser.add_argument(
        "--project",
        default="gesica",
        help="Contexte projet (défaut: gesica)",
    )
    parser.add_argument(
        "--source",
        choices=["pubmed", "cochrane", "prospero", "all"],
        default="all",
        help="Source(s) à interroger (défaut: all)",
    )
    parser.add_argument(
        "--max-per-source",
        type=int,
        default=100,
        help="Nombre max de reviews par source (défaut: 100)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Afficher sans insérer en base",
    )
    args = parser.parse_args()

    sources = ["pubmed", "cochrane", "prospero"] if args.source == "all" else [args.source]

    stats = run_living_reviews_ingestion(
        project=args.project,
        sources=sources,
        max_per_source=args.max_per_source,
        dry_run=args.dry_run,
    )

    print("\n" + "=" * 60)
    print("RÉSUMÉ LIVING SYSTEMATIC REVIEWS")
    print(f"  Sources interrogées : {', '.join(sources).upper()}")
    print(f"  Reviews récupérées  : {stats['total_fetched']}")
    print(f"  Insérées en base    : {stats['inserted']}")
    print(f"  Ignorées (doublons) : {stats['skipped']}")
    print(f"  Erreurs             : {stats['errors']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
