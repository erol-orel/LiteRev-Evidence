#!/usr/bin/env python3
"""
extract_pico_batch.py - Extraction PICO en batch pour le corpus GESICA
======================================================================
Utilise le modèle du rôle `bulk` (llm_usage.model_for, réglable par LLM_MODEL_BULK) et
le contrat PICO de l'application (api/pico_map) : le même prompt, le même texte lu (le
résumé, plus les méthodes et les résultats du texte intégral quand il existe) et les
mêmes champs que les autres rédacteurs. Il codait en dur un nom de modèle, un plafond de
400 jetons et un prompt à lui, sans le résultat principal que le brief et le digest lisent.

Usage :
    python3 extract_pico_batch.py --dry-run          # Aperçu sans modification
    python3 extract_pico_batch.py --execute          # Extraction complète
    python3 extract_pico_batch.py --execute --limit 50   # Limiter à 50 articles
    python3 extract_pico_batch.py --execute --scenario triage-support  # Par scénario
    python3 extract_pico_batch.py --execute --reprocess  # Réextraire même si déjà fait

Sortie JSON par article (api/pico_map.normalise_pico) :
    {
        "P": "Population étudiée",
        "I": "Intervention ou exposition",
        "C": "Comparateur (ou 'Not specified')",
        "O": "Outcome(s) principal(aux)",
        "study_design": "un terme du vocabulaire contrôlé (api/pico_map.PICO_DESIGNS)",
        "key_finding": "le résultat principal, en une phrase (vide si l'article n'en rapporte pas)",
        "finding_direction": "increase | decrease | no_difference | mixed | descriptive | none",
        "pico_confidence": 0.0-1.0,
        "pico_notes": "Remarques éventuelles",
        "pico_source": "abstract | fulltext"
    }
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import create_engine, text

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] pico: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("pico")

# ── Configuration ─────────────────────────────────────────────────────────────
# AVANT de lire DB_URL : ce script cherchait OPENAI_API_KEY ligne par ligne dans trois
# fichiers choisis ici, mais exigeait DB_URL du shell. Un seul chargeur pour tout le
# dépôt (env_files), lu une fois, et les deux variables viennent du même endroit.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # racine du dépôt
from env_files import CANONICAL as _CANONICAL  # noqa: E402
from env_files import load_env as _load_env  # noqa: E402

log.info(f"Fichiers d'environnement lus : {', '.join(_load_env()) or 'aucun'}")

# ── DB ────────────────────────────────────────────────────────────────────────
DB_URL = os.environ.get("DATABASE_URL") or os.environ.get("DB_URL")
if not DB_URL:
    raise RuntimeError("DATABASE_URL (or DB_URL) environment variable is required")
engine = create_engine(DB_URL, pool_pre_ping=True)
import llm_usage as _llm_usage  # noqa: E402
_llm_usage.configure(engine)              # comptabilise les appels de ce script

try:
    from llm_usage import MeteredOpenAI as OpenAI
    _api_key = os.environ.get("OPENAI_API_KEY")
    if not _api_key:
        raise RuntimeError(
            "OPENAI_API_KEY non définie.\n"
            "Solutions :\n"
            f"  1. L'ajouter au fichier de configuration du service : {_CANONICAL}\n"
            "  2. OPENAI_API_KEY=sk-... python3 extract_pico_batch.py\n"
            "Les fichiers lus, dans l'ordre : python3 scripts/env_audit.py"
        )
    client = OpenAI(api_key=_api_key, purpose="extract_pico_batch", timeout=90.0)
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    log.warning("openai non installé - pip3 install openai")
except RuntimeError as e:
    log.error(str(e))
    sys.exit(1)

# ── Le contrat PICO de l'application ─────────────────────────────────────────
# Le prompt, le texte lu, l'appel (modèle du rôle `bulk`, plafond, nouvelle demande plus
# courte si la réponse est coupée) et la normalisation des champs : les mêmes que ceux
# des autres rédacteurs. Module pur, importé sans démarrer l'API.
from api.pico_map import PicoUnreadable, extract_pico, pico_fulltext_chunks  # noqa: E402


def extract_pico_for_article(doc_id: int, title: str, abstract: str) -> Optional[dict]:
    """Appelle le LLM pour extraire le PICO d'un article (et son résultat principal)."""
    if not OPENAI_AVAILABLE:
        return None
    try:
        with engine.connect() as conn:
            chunks = pico_fulltext_chunks(conn, doc_id)
        pico, _source = extract_pico(client, title, abstract, chunks, label=f"PICO script {doc_id}")
        return pico
    except PicoUnreadable as e:
        log.warning("Réponse inutilisable : %s", e)
        return None
    except Exception as e:
        log.error("Erreur LLM : %s", e)
        return None


def run_extraction(dry_run: bool, limit: Optional[int], scenario: Optional[str], reprocess: bool):
    """Extrait le PICO pour tous les articles éligibles."""
    
    # Construire la requête de sélection
    where_clauses = ["project_context = 'gesica'"]
    params: dict = {}
    
    if not reprocess:
        where_clauses.append("pico_json IS NULL")
    
    if scenario:
        where_clauses.append("scenario_type = :scenario")
        params["scenario"] = scenario
    
    # Exclure les articles sans abstract
    where_clauses.append("(abstract IS NOT NULL AND LENGTH(abstract) > 50)")
    
    where_sql = " AND ".join(where_clauses)
    limit_sql = f"LIMIT {limit}" if limit else ""
    
    select_sql = text(f"""
        SELECT id, title, abstract, scenario_type
        FROM literature_document
        WHERE {where_sql}
        ORDER BY id
        {limit_sql}
    """)
    
    with engine.connect() as conn:
        rows = conn.execute(select_sql, params).mappings().fetchall()
    
    total = len(rows)
    log.info("Articles à traiter : %d", total)
    
    if dry_run:
        log.info("[DRY-RUN] Exemple du premier article :")
        if rows:
            r = rows[0]
            log.info("  ID: %d | Scénario: %s", r["id"], r["scenario_type"])
            log.info("  Titre: %s", r["title"][:80])
            log.info("  Abstract: %s...", (r["abstract"] or "")[:100])
        log.info("[DRY-RUN] Aucune modification effectuée.")
        return
    
    if not OPENAI_AVAILABLE:
        log.error("openai non disponible - installez-le avec : pip3 install openai")
        sys.exit(1)
    
    # Extraction en batch avec rate limiting
    success = 0
    failed = 0
    skipped = 0
    
    for i, row in enumerate(rows):
        doc_id = row["id"]
        title = row["title"] or ""
        abstract = row["abstract"] or ""
        
        if not title and not abstract:
            skipped += 1
            continue
        
        log.info("[%d/%d] Extraction PICO pour doc #%d (%s)...",
                 i + 1, total, doc_id, row["scenario_type"])
        
        pico = extract_pico_for_article(doc_id, title, abstract)

        if pico:
            with engine.begin() as conn:
                conn.execute(text("""
                    UPDATE literature_document
                    SET pico_json = CAST(:pico AS jsonb),
                        pico_extracted_at = :ts
                    WHERE id = :doc_id
                """), {
                    "pico": json.dumps(pico),
                    "ts": datetime.now(timezone.utc),
                    "doc_id": doc_id,
                })
            success += 1
            log.info("  ✓ P: %s | O: %s | Design: %s | Résultat: %s | Conf: %.2f",
                     str(pico.get("P") or "")[:50], str(pico.get("O") or "")[:50],
                     pico["study_design"], pico["finding_direction"], pico["pico_confidence"])
        else:
            failed += 1
            log.warning("  ✗ Extraction échouée pour doc #%d", doc_id)
        
        # Rate limiting : 1 requête / 0.5s (120 req/min max)
        if i < total - 1:
            time.sleep(0.5)
    
    log.info("=== Terminé : %d succès, %d échecs, %d ignorés sur %d articles ===",
             success, failed, skipped, total)


def main():
    parser = argparse.ArgumentParser(description="Extraction PICO en batch pour GESICA")
    parser.add_argument("--dry-run", action="store_true",
                        help="Aperçu sans modification")
    parser.add_argument("--execute", action="store_true",
                        help="Lancer l'extraction")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limiter le nombre d'articles traités")
    parser.add_argument("--scenario", type=str, default=None,
                        help="Traiter uniquement un scénario spécifique")
    parser.add_argument("--reprocess", action="store_true",
                        help="Réextraire même si PICO déjà présent")
    args = parser.parse_args()
    
    if not args.dry_run and not args.execute:
        parser.print_help()
        sys.exit(1)
    
    # Vérifier la connexion DB
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        log.info("Connexion DB OK")
    except Exception as e:
        log.error("Connexion DB échouée : %s", e)
        sys.exit(1)
    
    run_extraction(
        dry_run=args.dry_run,
        limit=args.limit,
        scenario=args.scenario,
        reprocess=args.reprocess,
    )


if __name__ == "__main__":
    main()
