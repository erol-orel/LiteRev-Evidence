"""Batch LLM enrichment: PICO, metadata, full text.

Extracted from main.py (LiteRev API); `main` re-exports everything for the scripts,
tools and tests.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from fastapi import Depends, HTTPException
from sqlalchemy import text

from .core import app, engine, logger, require_api_key

# ─── Enrichissement LLM Batch ────────────────────────────────────────────────

@app.post("/pico/extract")
def extract_pico_batch(
    scenario_id: Optional[str] = None,
    limit: int = 100000,
    _: None = Depends(require_api_key),
):
    """
    Extrait le PICO pour un lot d'articles (par scénario ou tout le corpus).
    Traite uniquement les articles sans PICO ou avec un PICO de faible confiance.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise HTTPException(status_code=503, detail="Clé OpenAI non configurée")

    # Récupérer les articles sans PICO
    with engine.connect() as conn:
        if scenario_id:
            rows = conn.execute(text("""
                SELECT ld.id, ld.title, ld.abstract
                FROM literature_document ld
                JOIN article_scenarios asn ON asn.document_id = ld.id AND asn.scenario_id = :sid
                WHERE ld.project_context = 'literev'
                  AND (ld.pico_json IS NULL OR (ld.pico_json->>'pico_confidence')::float < 0.5)
                  AND COALESCE(asn.screening_status, ld.screening_status) IS DISTINCT FROM 'excluded'  -- porte de screening (C1)
                  AND COALESCE(ld.pico_attempts, 0) < 3  -- borne les échecs déterministes (token-bleed)
                ORDER BY ld.id
                LIMIT :lim
            """), {"sid": scenario_id, "lim": limit}).mappings().fetchall()
        else:
            rows = conn.execute(text("""
                SELECT id, title, abstract
                FROM literature_document
                WHERE project_context = 'literev'
                  AND (pico_json IS NULL OR (pico_json->>'pico_confidence')::float < 0.5)
                  AND abstract IS NOT NULL AND length(abstract) > 50
                  AND COALESCE(pico_attempts, 0) < 3  -- borne les échecs déterministes (token-bleed)
                ORDER BY id
                LIMIT :lim
            """), {"lim": limit}).mappings().fetchall()

    extracted = 0
    skipped = 0
    errors = 0

    system_prompt = (
        "You are a systematic review expert. "
        "Extract PICO elements and return ONLY valid JSON:\n"
        '{"P":"Population","I":"Intervention","C":"Comparator or Not specified",'
        '"O":"Outcome(s)","study_design":"RCT|Cohort|Systematic review|etc",'
        '"pico_confidence":0.0-1.0,"pico_notes":""}\n'
        "Be concise (max 2 sentences per field). Return ONLY the JSON."
    )

    try:
        from llm_usage import MeteredOpenAI as _OAI
        from datetime import datetime, timezone
        _client = _OAI(api_key=openai_key, timeout=90.0)

        for row in rows:
            article_id = row["id"]
            title = row["title"] or ""
            abstract = row["abstract"] or ""
            if not abstract or len(abstract) < 50:
                skipped += 1
                continue
            # Appel LLM isolé : une erreur d'API (quota/réseau) est TRANSITOIRE
            # → on ne compte PAS de tentative (réessai quand l'API est saine).
            try:
                response = _client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Title: {title}\n\nAbstract: {abstract[:3000]}"},
                    ],
                    temperature=0,
                    seed=42,
                    max_tokens=800,  # 400 tronquait le JSON des articles verbeux → JSON invalide
                    response_format={"type": "json_object"},
                )
            except Exception as e:
                logger.warning(f"PICO batch API error article {article_id}: {e}")
                errors += 1
                continue  # transitoire — ne PAS consommer une tentative
            # On a une RÉPONSE → on COMPTE la tentative quoi qu'il arrive (borne le
            # token-bleed : une sortie déterministe malformée ne sera pas ré-extraite
            # à l'infini). Remplissage tolérant des clés plutôt que rejet en boucle.
            try:
                pico = json.loads(response.choices[0].message.content)
                if not isinstance(pico, dict):
                    raise ValueError("réponse PICO non-dict")
                for _k in ("P", "I", "C", "O"):
                    pico.setdefault(_k, "")
                pico.setdefault("study_design", "non précisé")
                try:
                    pico["pico_confidence"] = float(pico.get("pico_confidence", 0.3))
                except (TypeError, ValueError):
                    pico["pico_confidence"] = 0.3
                pico["pico_notes"] = pico.get("pico_notes", "")
                with engine.begin() as conn:
                    conn.execute(text("""
                        UPDATE literature_document
                        SET pico_json = CAST(:pico AS jsonb), pico_extracted_at = :ts,
                            pico_attempts = COALESCE(pico_attempts, 0) + 1
                        WHERE id = :article_id
                    """), {
                        "pico": json.dumps(pico),
                        "ts": datetime.now(timezone.utc),
                        "article_id": article_id,
                    })
                extracted += 1
            except Exception as e:
                logger.warning(f"PICO batch parse error article {article_id}: {e}")
                # Réponse reçue mais JSON invalide → COMPTE la tentative (borne le token-bleed).
                try:
                    with engine.begin() as conn:
                        conn.execute(text(
                            "UPDATE literature_document SET pico_attempts = COALESCE(pico_attempts, 0) + 1 WHERE id = :aid"
                        ), {"aid": article_id})
                except Exception:
                    pass
                errors += 1
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur LLM batch: {str(e)}")

    return {
        "extracted": extracted,
        "skipped": skipped,
        "errors": errors,
        "message": f"{extracted} articles enrichis, {skipped} ignorés, {errors} erreurs",
    }


@app.post("/metadata/extract")
def extract_metadata_batch(
    scenario_id: Optional[str] = None,
    limit: int = 100000,
    _: None = Depends(require_api_key),
):
    """
    Enrichit les métadonnées (type d'étude, année, journal) via LLM pour un lot d'articles.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise HTTPException(status_code=503, detail="Clé OpenAI non configurée")

    with engine.connect() as conn:
        if scenario_id:
            rows = conn.execute(text("""
                SELECT ld.id, ld.title, ld.abstract, ld.source, ld.year
                FROM literature_document ld
                JOIN article_scenarios asn ON asn.document_id = ld.id AND asn.scenario_id = :sid
                WHERE ld.project_context = 'literev'
                  AND (ld.metadata_json IS NULL OR ld.metadata_json = '{}'::jsonb)
                ORDER BY ld.id
                LIMIT :lim
            """), {"sid": scenario_id, "lim": limit}).mappings().fetchall()
        else:
            rows = conn.execute(text("""
                SELECT id, title, abstract, source, year
                FROM literature_document
                WHERE project_context = 'literev'
                  AND (metadata_json IS NULL OR metadata_json = '{}'::jsonb)
                  AND abstract IS NOT NULL AND length(abstract) > 30
                ORDER BY id
                LIMIT :lim
            """), {"lim": limit}).mappings().fetchall()

    extracted = 0
    skipped = 0
    errors = 0

    system_prompt = (
        "You are a biomedical librarian. Extract metadata from this article and return ONLY valid JSON:\n"
        '{"study_type":"RCT|Cohort|Case-control|Cross-sectional|Systematic review|Meta-analysis|Case report|Editorial|Other",'
        '"sample_size":null,"country":"ISO2 or null","setting":"hospital|prehospital|community|other|null",'
        '"primary_outcome":"brief description or null","funding":"public|industry|mixed|not reported",'
        '"bias_risk":"low|moderate|high|unclear","metadata_confidence":0.0-1.0}\n'
        "Return ONLY the JSON."
    )

    try:
        from llm_usage import MeteredOpenAI as _OAI
        _client = _OAI(api_key=openai_key, timeout=90.0)

        for row in rows:
            article_id = row["id"]
            title = row["title"] or ""
            abstract = row["abstract"] or ""
            if not title:
                skipped += 1
                continue
            try:
                response = _client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Title: {title}\n\nAbstract: {abstract[:2000]}"},
                    ],
                    temperature=0.1,
                    max_tokens=300,
                    response_format={"type": "json_object"},
                )
                metadata = json.loads(response.choices[0].message.content)
                metadata["metadata_confidence"] = float(metadata.get("metadata_confidence", 0.5))
                with engine.begin() as conn:
                    conn.execute(text("""
                        UPDATE literature_document
                        SET metadata_json = CAST(:meta AS jsonb)
                        WHERE id = :article_id
                    """), {
                        "meta": json.dumps(metadata),
                        "article_id": article_id,
                    })
                extracted += 1
            except Exception as e:
                logger.warning(f"Metadata batch error article {article_id}: {e}")
                errors += 1
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur LLM batch: {str(e)}")

    return {
        "extracted": extracted,
        "skipped": skipped,
        "errors": errors,
        "message": f"{extracted} articles enrichis, {skipped} ignorés, {errors} erreurs",
    }


@app.post("/fulltext/fetch")
def fetch_fulltext_batch(
    scenario_id: Optional[str] = None,
    limit: int = 100000,
    _: None = Depends(require_api_key),
):
    """
    Tente de récupérer le texte intégral (via DOI/URL) pour un lot d'articles.
    Utilise Unpaywall + CrossRef pour les accès ouverts.
    """
    import urllib.request

    with engine.connect() as conn:
        if scenario_id:
            rows = conn.execute(text("""
                SELECT ld.id, ld.title, ld.doi, ld.url
                FROM literature_document ld
                JOIN article_scenarios asn ON asn.document_id = ld.id AND asn.scenario_id = :sid
                WHERE ld.project_context = 'literev'
                  AND (ld.has_fulltext IS NULL OR ld.has_fulltext = false)
                  AND ld.doi IS NOT NULL
                ORDER BY ld.id
                LIMIT :lim
            """), {"sid": scenario_id, "lim": limit}).mappings().fetchall()
        else:
            rows = conn.execute(text("""
                SELECT id, title, doi, url
                FROM literature_document
                WHERE project_context = 'literev'
                  AND (has_fulltext IS NULL OR has_fulltext = false)
                  AND doi IS NOT NULL
                ORDER BY id
                LIMIT :lim
            """), {"lim": limit}).mappings().fetchall()

    fetched = 0
    not_available = 0
    errors = 0

    for row in rows:
        article_id = row["id"]
        doi = row["doi"]
        if not doi:
            not_available += 1
            continue
        try:
            # Tenter Unpaywall
            unpaywall_url = f"https://api.unpaywall.org/v2/{doi}?email=literev@gesica.ch"
            req = urllib.request.Request(unpaywall_url, headers={"User-Agent": "LiteRev/1.0"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read())
            oa_url = None
            if data.get("is_oa") and data.get("best_oa_location"):
                oa_url = data["best_oa_location"].get("url_for_pdf") or data["best_oa_location"].get("url")
            if oa_url:
                with engine.begin() as conn:
                    conn.execute(text("""
                        UPDATE literature_document
                        SET has_fulltext = true, url = :url
                        WHERE id = :article_id
                    """), {"url": oa_url, "article_id": article_id})
                fetched += 1
            else:
                not_available += 1
        except Exception as e:
            logger.warning(f"Fulltext fetch error article {article_id}: {e}")
            errors += 1

    return {
        "fetched": fetched,
        "not_available": not_available,
        "errors": errors,
        "message": f"{fetched} textes intégraux récupérés, {not_available} non disponibles, {errors} erreurs",
    }


@app.get("/enrichment/status")
def get_enrichment_status(scenario_id: Optional[str] = None):
    """Retourne le statut d'enrichissement (PICO, métadonnées, fulltext) pour un scénario ou tout le corpus."""
    with engine.connect() as conn:
        if scenario_id:
            row = conn.execute(text("""
                SELECT
                    COUNT(*) as total,
                    COUNT(ld.pico_json) as with_pico,
                    COUNT(CASE WHEN ld.metadata_json IS NOT NULL AND ld.metadata_json != '{}'::jsonb THEN 1 END) as with_metadata,
                    COUNT(CASE WHEN ld.has_fulltext = true THEN 1 END) as with_fulltext
                FROM literature_document ld
                WHERE ld.project_context = 'literev'
                  AND (
                    EXISTS (SELECT 1 FROM article_scenarios asn WHERE asn.document_id = ld.id AND asn.scenario_id = :sid)
                  )
            """), {"sid": scenario_id}).mappings().fetchone()
        else:
            row = conn.execute(text("""
                SELECT
                    COUNT(*) as total,
                    COUNT(pico_json) as with_pico,
                    COUNT(CASE WHEN metadata_json IS NOT NULL AND metadata_json != '{}'::jsonb THEN 1 END) as with_metadata,
                    COUNT(CASE WHEN has_fulltext = true THEN 1 END) as with_fulltext
                FROM literature_document
                WHERE project_context = 'literev'
            """)).mappings().fetchone()

    total = row["total"] or 1
    return {
        "scenario_id": scenario_id,
        "total": row["total"],
        "pico": {"count": row["with_pico"], "pct": round(row["with_pico"] / total * 100, 1)},
        "metadata": {"count": row["with_metadata"], "pct": round(row["with_metadata"] / total * 100, 1)},
        "fulltext": {"count": row["with_fulltext"], "pct": round(row["with_fulltext"] / total * 100, 1)},
    }
