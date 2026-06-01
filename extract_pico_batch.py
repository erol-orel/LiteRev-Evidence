#!/usr/bin/env python3
"""
extract_pico_batch.py — Extraction PICO en batch pour le corpus GESICA
======================================================================
Utilise gpt-4.1-mini (via l'API OpenAI configurée dans l'environnement)
pour extraire les éléments PICO de chaque article à partir du titre + abstract.

Usage :
    python3 extract_pico_batch.py --dry-run          # Aperçu sans modification
    python3 extract_pico_batch.py --execute          # Extraction complète
    python3 extract_pico_batch.py --execute --limit 50   # Limiter à 50 articles
    python3 extract_pico_batch.py --execute --scenario triage-support  # Par scénario
    python3 extract_pico_batch.py --execute --reprocess  # Réextraire même si déjà fait

Sortie JSON par article :
    {
        "P": "Population étudiée",
        "I": "Intervention ou exposition",
        "C": "Comparateur (ou 'Non spécifié')",
        "O": "Outcome(s) principal(aux)",
        "study_design": "RCT | Observational | Systematic Review | ...",
        "pico_confidence": 0.0-1.0,
        "pico_notes": "Remarques éventuelles"
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

# ── DB ────────────────────────────────────────────────────────────────────────
DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg://literev:MyNewStrongPassword!@10.10.1.10:5432/literev",
)
engine = create_engine(DB_URL, pool_pre_ping=True)

# ── OpenAI ────────────────────────────────────────────────────────────────────
try:
    from openai import OpenAI
    client = OpenAI()  # Utilise OPENAI_API_KEY + base_url de l'environnement
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    log.warning("openai non installé — pip3 install openai")

# ── Prompt PICO ───────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a systematic review expert specializing in emergency medicine, 
pre-hospital care, and disaster medicine. Your task is to extract PICO elements from 
scientific article abstracts.

Extract the following elements and return ONLY valid JSON (no markdown, no explanation):
{
  "P": "Population studied (patients, setting, demographics)",
  "I": "Intervention or exposure studied",
  "C": "Comparator or control group (write 'Not specified' if absent)",
  "O": "Primary outcome(s) measured",
  "study_design": "One of: RCT | Quasi-experimental | Cohort | Case-control | Cross-sectional | Systematic review | Meta-analysis | Narrative review | Case report | Simulation | Modelling | Other",
  "pico_confidence": <float 0.0-1.0, confidence in extraction quality>,
  "pico_notes": "Any important caveats or missing information (empty string if none)"
}

Rules:
- Be concise (max 2 sentences per field)
- If the abstract is too short or uninformative, set pico_confidence < 0.4
- Always respond in English regardless of input language
- Return ONLY the JSON object, nothing else"""

def extract_pico_for_article(title: str, abstract: str) -> Optional[dict]:
    """Appelle le LLM pour extraire le PICO d'un article."""
    if not OPENAI_AVAILABLE:
        return None
    
    # Tronquer l'abstract si trop long (max ~2000 tokens)
    abstract_truncated = abstract[:3000] if abstract else ""
    
    user_content = f"Title: {title}\n\nAbstract: {abstract_truncated}"
    
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.1,
            max_tokens=400,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        pico = json.loads(raw)
        
        # Validation minimale
        required = {"P", "I", "C", "O", "study_design", "pico_confidence"}
        if not required.issubset(pico.keys()):
            log.warning("PICO incomplet : %s", list(pico.keys()))
            return None
        
        # Normaliser pico_confidence
        pico["pico_confidence"] = float(pico.get("pico_confidence", 0.5))
        pico["pico_notes"] = pico.get("pico_notes", "")
        
        return pico
    
    except json.JSONDecodeError as e:
        log.warning("JSON invalide : %s", e)
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
        log.error("openai non disponible — installez-le avec : pip3 install openai")
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
        
        pico = extract_pico_for_article(title, abstract)
        
        if pico:
            with engine.begin() as conn:
                conn.execute(text("""
                    UPDATE literature_document
                    SET pico_json = :pico,
                        pico_extracted_at = :ts
                    WHERE id = :doc_id
                """), {
                    "pico": json.dumps(pico),
                    "ts": datetime.now(timezone.utc),
                    "doc_id": doc_id,
                })
            success += 1
            log.info("  ✓ P: %s | O: %s | Design: %s | Conf: %.2f",
                     pico["P"][:50], pico["O"][:50],
                     pico["study_design"], pico["pico_confidence"])
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
