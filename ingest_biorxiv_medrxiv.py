#!/usr/bin/env python3
"""
ingest_biorxiv_medrxiv.py — Ingestion bioRxiv et medRxiv dans LiteRev-Evidence
================================================================================
Utilise l'API REST publique de bioRxiv/medRxiv (https://api.biorxiv.org)
Aucune clé API requise.

Usage :
    # bioRxiv — 30 derniers jours, mots-clés EMS/prehospital
    python3 ingest_biorxiv_medrxiv.py --server biorxiv --days 30 --project gesica

    # medRxiv — 60 derniers jours, tous les articles
    python3 ingest_biorxiv_medrxiv.py --server medrxiv --days 60 --project gesica

    # Les deux serveurs en une seule passe
    python3 ingest_biorxiv_medrxiv.py --server both --days 30 --project gesica

    # Dry-run (affiche sans insérer)
    python3 ingest_biorxiv_medrxiv.py --server medrxiv --days 7 --dry-run

API Reference : https://api.biorxiv.org/
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import date, timedelta
from typing import Any

import requests
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
logger = logging.getLogger("ingest-biorxiv")

# ─── Configuration ────────────────────────────────────────────────────────────

DB_URL = os.getenv("DB_URL") or os.getenv("DATABASE_URL")
if not DB_URL:
    raise RuntimeError("DB_URL (or DATABASE_URL) environment variable is required")
API_BASE_URL = os.getenv("LITEREV_API_URL", "http://localhost:8000")
WRITE_API_KEY = os.getenv("WRITE_API_KEY", "")

BIORXIV_API = "https://api.biorxiv.org/details/{server}/{date_from}/{date_to}/{cursor}/json"

# Mots-clés pertinents pour GESICA (filtre côté client, pas dans l'API)
GESICA_KEYWORDS = [
    "emergency medical service", "prehospital", "pre-hospital", "ambulance",
    "cardiac arrest", "out-of-hospital", "OHCA", "resuscitation",
    "emergency dispatch", "EMS", "paramedic", "first responder",
    "mass casualty", "triage", "trauma", "stroke", "sepsis",
    "epidemic", "pandemic", "outbreak", "surveillance", "influenza",
    "COVID-19", "SARS-CoV-2", "heatwave", "heat stress",
    "response time", "dispatch", "telemedicine", "telehealth",
    "artificial intelligence", "machine learning", "deep learning",
    "prediction", "forecasting", "early warning",
    "cross-border", "transborder", "Geneva", "Switzerland",
]

# Mapping catégories bioRxiv → scenario_type GESICA
CATEGORY_TO_SCENARIO: dict[str, str] = {
    "epidemiology": "epidemic-early-warning",
    "public and global health": "epidemic-early-warning",
    "emergency medicine": "demand-forecasting",
    "cardiovascular medicine": "cardiac-arrest-prediction",
    "infectious diseases": "epidemic-early-warning",
    "health informatics": "triage-support",
    "health policy": "demand-forecasting",
    "intensive care and critical care medicine": "triage-support",
    "neurology": "stroke-protocol",
    "pharmacology and therapeutics": "medication-management",
}

# ─── Fonctions utilitaires ─────────────────────────────────────────────────────

def fetch_preprints(
    server: str,
    date_from: str,
    date_to: str,
    max_results: int = 500,
) -> list[dict[str, Any]]:
    """Récupère les preprints depuis l'API bioRxiv/medRxiv."""
    all_results: list[dict[str, Any]] = []
    cursor = 0
    page_size = 100  # max par page selon l'API

    while len(all_results) < max_results:
        url = BIORXIV_API.format(
            server=server,
            date_from=date_from,
            date_to=date_to,
            cursor=cursor,
        )
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logger.error(f"Erreur API {server} (cursor={cursor}): {e}")
            break

        collection = data.get("collection", [])
        messages = data.get("messages", [])

        if not collection:
            logger.info(f"  {server}: fin de pagination à cursor={cursor}")
            break

        all_results.extend(collection)
        logger.info(f"  {server}: {len(all_results)} preprints récupérés (cursor={cursor})")

        # Vérifier s'il y a une page suivante
        total = 0
        for msg in messages:
            if isinstance(msg, dict) and "total" in msg:
                total = int(msg["total"])
                break

        if len(all_results) >= total or len(collection) < page_size:
            break

        cursor += page_size
        time.sleep(0.5)  # Respecter le rate limit de l'API

    return all_results


def is_relevant(preprint: dict[str, Any]) -> bool:
    """Filtre les preprints par pertinence pour GESICA."""
    title = (preprint.get("title") or "").lower()
    abstract = (preprint.get("abstract") or "").lower()
    category = (preprint.get("category") or "").lower()
    text_combined = f"{title} {abstract} {category}"

    return any(kw.lower() in text_combined for kw in GESICA_KEYWORDS)


def map_to_scenario(preprint: dict[str, Any]) -> str:
    """Mappe la catégorie bioRxiv vers un scenario_type GESICA."""
    category = (preprint.get("category") or "").lower()
    for cat_key, scenario in CATEGORY_TO_SCENARIO.items():
        if cat_key in category:
            return scenario
    return "unassigned"


def document_exists(conn: Any, external_id: str) -> bool:
    """Vérifie si un document avec cet external_id existe déjà."""
    result = conn.execute(
        text("SELECT id FROM literature_document WHERE external_id = :eid"),
        {"eid": external_id},
    ).fetchone()
    return result is not None


def insert_document(
    conn: Any,
    preprint: dict[str, Any],
    server: str,
    project: str,
) -> int | None:
    """Insère un preprint dans literature_document et crée son chunk title_abstract."""
    doi = preprint.get("doi", "")
    external_id = f"{server}:{doi}" if doi else f"{server}:{preprint.get('biorxiv_doi', '')}"

    if not external_id or external_id == f"{server}:":
        logger.warning(f"  Preprint sans DOI ignoré: {preprint.get('title', 'N/A')[:60]}")
        return None

    # Construire les auteurs
    authors_raw = preprint.get("authors", "")
    authors = authors_raw[:500] if authors_raw else None

    # Construire l'URL
    url = f"https://doi.org/{doi}" if doi else None

    # Année depuis la date de publication
    pub_date = preprint.get("date", "")
    year = int(pub_date[:4]) if pub_date and len(pub_date) >= 4 else None

    # Catégorie → scenario_type
    scenario_type = map_to_scenario(preprint)

    # Déterminer la source_type
    source_type = "preprint"

    # Insérer le document
    sql_doc = text("""
        INSERT INTO literature_document (
            source, title, abstract, year, url, external_id,
            project_context, source_type, disease_or_condition,
            scenario_type, geographic_scope, evidence_category,
            authors, doi, journal
        )
        VALUES (
            :source, :title, :abstract, :year, :url, :external_id,
            :project_context, :source_type, :disease_or_condition,
            :scenario_type, :geographic_scope, :evidence_category,
            :authors, :doi, :journal
        )
        RETURNING id
    """)

    category = preprint.get("category", "")
    abstract = (preprint.get("abstract") or "").strip()

    doc_id = conn.execute(sql_doc, {
        "source": server,
        "title": (preprint.get("title") or "")[:1000],
        "abstract": abstract[:4000] if abstract else None,
        "year": year,
        "url": url,
        "external_id": external_id,
        "project_context": project,
        "source_type": source_type,
        "disease_or_condition": category[:200] if category else None,
        "scenario_type": scenario_type,
        "geographic_scope": "international",
        "evidence_category": "preprint",
        "authors": authors,
        "doi": doi[:200] if doi else None,
        "journal": f"{server.capitalize()} preprint",
    }).scalar_one()

    # Créer le chunk title_abstract
    title = (preprint.get("title") or "").strip()
    content = f"{title}\n\n{abstract}".strip()

    if content and len(content) >= 50:
        sql_chunk = text("""
            INSERT INTO document_chunk (
                document_id, chunk_index, content, chunk_type,
                section_label, token_count, chunk_weight, metadata_json
            )
            VALUES (
                :document_id, 0, :content, 'title_abstract',
                'Title + Abstract', :token_count, 1.0,
                CAST(:metadata_json AS jsonb)
            )
        """)
        conn.execute(sql_chunk, {
            "document_id": doc_id,
            "content": content[:8000],
            "token_count": len(content.split()),
            "metadata_json": json.dumps({
                "server": server,
                "doi": doi,
                "category": category,
                "pub_date": pub_date,
                "version": preprint.get("version", "1"),
                "type": preprint.get("type", "preprint"),
            }),
        })

    return doc_id


# ─── Pipeline principal ────────────────────────────────────────────────────────

def run_ingestion(
    server: str,
    days: int,
    project: str,
    dry_run: bool = False,
    filter_relevant: bool = True,
    max_results: int = 500,
) -> dict[str, int]:
    """Pipeline complet d'ingestion pour un serveur (biorxiv ou medrxiv)."""
    date_to = date.today()
    date_from = date_to - timedelta(days=days)

    logger.info(f"{'[DRY-RUN] ' if dry_run else ''}Ingestion {server.upper()} — {date_from} → {date_to}")

    preprints = fetch_preprints(
        server=server,
        date_from=str(date_from),
        date_to=str(date_to),
        max_results=max_results,
    )

    logger.info(f"  {len(preprints)} preprints récupérés depuis {server}")

    if filter_relevant:
        relevant = [p for p in preprints if is_relevant(p)]
        logger.info(f"  {len(relevant)} preprints pertinents après filtrage GESICA")
    else:
        relevant = preprints

    if dry_run:
        for p in relevant[:10]:
            print(f"  [DRY] {p.get('title', 'N/A')[:80]}")
            print(f"        DOI: {p.get('doi', 'N/A')} | Cat: {p.get('category', 'N/A')}")
        return {"fetched": len(preprints), "relevant": len(relevant), "inserted": 0, "skipped": 0}

    engine = create_engine(DB_URL, pool_pre_ping=True)
    inserted = 0
    skipped = 0
    errors = 0

    with engine.begin() as conn:
        for preprint in relevant:
            doi = preprint.get("doi", "")
            external_id = f"{server}:{doi}" if doi else None

            if not external_id:
                skipped += 1
                continue

            if document_exists(conn, external_id):
                skipped += 1
                continue

            try:
                doc_id = insert_document(conn, preprint, server, project)
                if doc_id:
                    inserted += 1
                    if inserted % 50 == 0:
                        logger.info(f"  → {inserted} insérés...")
                else:
                    skipped += 1
            except Exception as e:
                logger.error(f"  Erreur insertion {external_id}: {e}")
                errors += 1

    logger.info(
        f"  ✅ {server.upper()} terminé — {inserted} insérés, {skipped} ignorés, {errors} erreurs"
    )
    return {
        "fetched": len(preprints),
        "relevant": len(relevant),
        "inserted": inserted,
        "skipped": skipped,
        "errors": errors,
    }


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingestion bioRxiv/medRxiv dans LiteRev-Evidence"
    )
    parser.add_argument(
        "--server",
        choices=["biorxiv", "medrxiv", "both"],
        default="both",
        help="Serveur à interroger (défaut: both)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Nombre de jours en arrière (défaut: 30)",
    )
    parser.add_argument(
        "--project",
        default="gesica",
        help="Contexte projet (défaut: gesica)",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=500,
        help="Nombre max de preprints par serveur (défaut: 500)",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Désactiver le filtre par mots-clés GESICA (ingère tout)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Afficher sans insérer en base",
    )
    args = parser.parse_args()

    servers = ["biorxiv", "medrxiv"] if args.server == "both" else [args.server]
    total_stats: dict[str, int] = {"fetched": 0, "relevant": 0, "inserted": 0, "skipped": 0, "errors": 0}

    for srv in servers:
        stats = run_ingestion(
            server=srv,
            days=args.days,
            project=args.project,
            dry_run=args.dry_run,
            filter_relevant=not args.no_filter,
            max_results=args.max_results,
        )
        for k in total_stats:
            total_stats[k] += stats.get(k, 0)

    print("\n" + "=" * 60)
    print(f"RÉSUMÉ TOTAL ({', '.join(servers).upper()})")
    print(f"  Preprints récupérés : {total_stats['fetched']}")
    print(f"  Pertinents GESICA   : {total_stats['relevant']}")
    print(f"  Insérés en base     : {total_stats['inserted']}")
    print(f"  Ignorés (doublons)  : {total_stats['skipped']}")
    print(f"  Erreurs             : {total_stats['errors']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
