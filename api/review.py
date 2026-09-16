"""PRISMA flow, screening progress, PICO per article and in bulk.

Extracted from main.py (LiteRev API); `main` re-exports everything for the scripts,
tools and tests.
"""
from __future__ import annotations

import json
import os
from typing import Any

from fastapi import Depends, HTTPException, Query
from sqlalchemy import text

from .core import app, engine, require_api_key
from .scenario_store import _get_user_scenario_or_404
from .search import _load_prisma_identification, _reconcile_prisma_identification
from .double_blind import _write_ars_screening

# ── Proxy endpoints : rediriger les appels /gesica/scenarios/{usr-*}/... ──────
# Les endpoints existants (screening, pico, evidence-brief, clustering, rag, etc.)
# valident maintenant l'ID via la DB (user_scenarios is_system=TRUE).
# Pour les scénarios utilisateur (usr-*), on intercepte avant ce check.

@app.get("/user-scenarios/{scenario_id}/screening-progress")
def get_user_scenario_screening_progress(scenario_id: str) -> dict[str, Any]:
    """Progression du screening PRISMA pour un scénario utilisateur."""
    _get_user_scenario_or_404(scenario_id)
    with engine.connect() as conn:
        stats = conn.execute(text("""
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN d.is_duplicate = TRUE THEN 1 ELSE 0 END) AS duplicates,
                -- included/excluded/pending comptés sur le sous-ensemble NON dupliqué
                -- (même base que `unique = total - duplicates`), sinon un doublon
                -- marqué inclus/exclu faisait dépasser 100 % / rendait `pending` négatif.
                SUM(CASE WHEN (d.is_duplicate IS NULL OR d.is_duplicate = FALSE) AND COALESCE(ars.screening_status, d.screening_status) = 'included' THEN 1 ELSE 0 END) AS included,
                SUM(CASE WHEN (d.is_duplicate IS NULL OR d.is_duplicate = FALSE) AND COALESCE(ars.screening_status, d.screening_status) = 'excluded' THEN 1 ELSE 0 END) AS excluded,
                SUM(CASE WHEN (d.is_duplicate IS NULL OR d.is_duplicate = FALSE) AND (COALESCE(ars.screening_status, d.screening_status) IS NULL OR COALESCE(ars.screening_status, d.screening_status) = 'pending') THEN 1 ELSE 0 END) AS pending
            FROM article_scenarios ars
            JOIN literature_document d ON d.id = ars.document_id
            WHERE ars.scenario_id = :sid
        """), {"sid": scenario_id}).mappings().first()
    total = int(stats["total"] or 0)
    duplicates = int(stats["duplicates"] or 0)
    unique = total - duplicates
    included = int(stats["included"] or 0)
    excluded = int(stats["excluded"] or 0)
    screened = included + excluded
    pct = round(screened / unique * 100, 1) if unique > 0 else 0
    return {
        "scenario_id": scenario_id,
        "total_in_db": total,
        "total": total,
        "duplicates": duplicates,
        "unique_articles": unique,
        "screened": screened,
        "included": included,
        "excluded": excluded,
        "awaiting": unique - screened,
        "pending": unique - screened,
        "progress_pct": pct,
        "screening_complete": pct >= 100,
    }


@app.get("/user-scenarios/{scenario_id}/pico-stats")
def get_user_scenario_pico_stats(scenario_id: str) -> dict[str, Any]:
    """Statistiques PICO pour un scénario utilisateur."""
    _get_user_scenario_or_404(scenario_id)
    with engine.connect() as conn:
        counts = conn.execute(text("""
            SELECT
                COUNT(*) AS total,
                COUNT(*) FILTER (WHERE d.pico_json IS NOT NULL) AS with_pico,
                COUNT(*) FILTER (WHERE d.pico_json IS NULL) AS without_pico,
                ROUND(AVG((d.pico_json->>'pico_confidence')::float)
                    FILTER (WHERE d.pico_json IS NOT NULL)::numeric, 2) AS avg_confidence
            FROM article_scenarios ars
            JOIN literature_document d ON d.id = ars.document_id
            WHERE ars.scenario_id = :sid
              AND (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)
        """), {"sid": scenario_id}).mappings().fetchone()
        designs = conn.execute(text("""
            SELECT
                COALESCE(d.pico_json->>'study_design', 'Non extrait') AS design,
                COUNT(*) AS n
            FROM article_scenarios ars
            JOIN literature_document d ON d.id = ars.document_id
            WHERE ars.scenario_id = :sid AND d.pico_json IS NOT NULL
              AND (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)
            GROUP BY 1 ORDER BY 2 DESC
        """), {"sid": scenario_id}).mappings().fetchall()
    total = counts["total"] if counts else 0
    with_pico = counts["with_pico"] if counts else 0
    return {
        "scenario_id": scenario_id,
        "total": total,
        "with_pico": with_pico,
        "without_pico": counts["without_pico"] if counts else 0,
        "coverage_pct": round((with_pico / total * 100) if total > 0 else 0, 1),
        "avg_confidence": float(counts["avg_confidence"]) if counts and counts["avg_confidence"] else None,
        "study_design_distribution": [{"design": d["design"], "count": d["n"]} for d in designs],
    }


@app.get("/user-scenarios/{scenario_id}/prisma")
def get_user_scenario_prisma(
    scenario_id: str,
    threshold: float = Query(None),
) -> dict[str, Any]:
    """Flow PRISMA modernisé pour un scénario utilisateur.

    Retourne 4 étapes : identification → pré-screening IA → curation manuelle → synthèse.
    Le seuil de similarité sémantique sépare sélection automatique et borderline.
    """
    row = _get_user_scenario_or_404(scenario_id)

    # Effective threshold (param > saved setting > default 0.45)
    with engine.connect() as conn:
        ss = conn.execute(text(
            "SELECT similarity_threshold FROM scenario_settings WHERE scenario_id=:sid"
        ), {"sid": scenario_id}).first()
    eff_threshold = threshold if threshold is not None else (
        float(ss[0]) if ss and ss[0] is not None else 0.45
    )

    with engine.connect() as conn:
        stats = conn.execute(text("""
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN d.source = 'pubmed'   THEN 1 ELSE 0 END) AS pubmed,
                SUM(CASE WHEN d.source = 'pmc'      THEN 1 ELSE 0 END) AS pmc,
                SUM(CASE WHEN d.source = 'openalex' THEN 1 ELSE 0 END) AS openalex,
                SUM(CASE WHEN d.source = 'europepmc' THEN 1 ELSE 0 END) AS europepmc,
                SUM(CASE WHEN d.source = 'crossref' THEN 1 ELSE 0 END) AS crossref,
                SUM(CASE WHEN d.source = 'preprint' THEN 1 ELSE 0 END) AS preprint,
                SUM(CASE WHEN d.source = 'medrxiv'  THEN 1 ELSE 0 END) AS medrxiv,
                SUM(CASE WHEN d.source = 'biorxiv'  THEN 1 ELSE 0 END) AS biorxiv,
                SUM(CASE WHEN d.source = 'semantic_scholar' THEN 1 ELSE 0 END) AS semantic_scholar,
                SUM(CASE WHEN d.source = 'doaj'     THEN 1 ELSE 0 END) AS doaj,
                SUM(CASE WHEN d.source = 'clinicaltrials' THEN 1 ELSE 0 END) AS clinicaltrials,
                SUM(CASE WHEN d.source = 'core'     THEN 1 ELSE 0 END) AS core,
                SUM(CASE WHEN d.source = 'arxiv'    THEN 1 ELSE 0 END) AS arxiv,
                SUM(CASE WHEN d.source = 'openaire' THEN 1 ELSE 0 END) AS openaire,
                SUM(CASE WHEN d.source = 'db_cache' THEN 1 ELSE 0 END) AS db_cache,
                SUM(CASE WHEN d.is_duplicate = TRUE THEN 1 ELSE 0 END) AS duplicates,
                -- Étapes POST-identification (screening/éligibilité/preuves) comptées
                -- sur le sous-ensemble NON dupliqué — cohérent avec toutes les autres
                -- surfaces "pertinentes" (evidence-brief, RAG, cartes). L'identification
                -- ci-dessus (total/by_source/duplicates) reste sur le corpus ENTIER.
                -- semantic split at effective threshold
                SUM(CASE WHEN (d.is_duplicate IS NULL OR d.is_duplicate = FALSE) AND COALESCE(ars.similarity_score, 0) >= :thr THEN 1 ELSE 0 END) AS above_threshold,
                SUM(CASE WHEN (d.is_duplicate IS NULL OR d.is_duplicate = FALSE) AND COALESCE(ars.similarity_score, 0) <  :thr THEN 1 ELSE 0 END) AS below_threshold,
                -- manual curation
                SUM(CASE WHEN (d.is_duplicate IS NULL OR d.is_duplicate = FALSE) AND COALESCE(ars.screening_status, d.screening_status) = 'included' THEN 1 ELSE 0 END) AS manually_included,
                SUM(CASE WHEN (d.is_duplicate IS NULL OR d.is_duplicate = FALSE) AND COALESCE(ars.screening_status, d.screening_status) = 'excluded' THEN 1 ELSE 0 END) AS manually_excluded,
                SUM(CASE WHEN (d.is_duplicate IS NULL OR d.is_duplicate = FALSE) AND (COALESCE(ars.screening_status, d.screening_status) IS NULL OR COALESCE(ars.screening_status, d.screening_status) = 'pending')
                         THEN 1 ELSE 0 END) AS pending,
                -- manually included but below threshold (override)
                SUM(CASE WHEN (d.is_duplicate IS NULL OR d.is_duplicate = FALSE) AND COALESCE(ars.screening_status, d.screening_status) = 'included'
                           AND COALESCE(ars.similarity_score, 0) < :thr THEN 1 ELSE 0 END) AS manually_rescued,
                -- manually excluded above threshold (veto)
                SUM(CASE WHEN (d.is_duplicate IS NULL OR d.is_duplicate = FALSE) AND COALESCE(ars.screening_status, d.screening_status) = 'excluded'
                           AND COALESCE(ars.similarity_score, 0) >= :thr THEN 1 ELSE 0 END) AS manually_vetoed,
                -- full text — RESTREINT à l'ensemble de preuves (≥ seuil OU inclus
                -- manuellement, hors exclus), pas au corpus entier : sinon le "X of Y"
                -- du PRISMA pouvait dépasser Y (le fameux "5 of 4").
                SUM(CASE WHEN (d.is_duplicate IS NULL OR d.is_duplicate = FALSE) AND EXISTS (
                    SELECT 1 FROM document_chunk c
                    WHERE c.document_id = d.id AND c.chunk_type = 'fulltext_section'
                ) AND COALESCE(ars.screening_status, d.screening_status) IS DISTINCT FROM 'excluded'
                  AND (COALESCE(ars.similarity_score, 0) >= :thr OR COALESCE(ars.screening_status, d.screening_status) = 'included')
                  THEN 1 ELSE 0 END) AS with_fulltext,
                -- embeddings
                SUM(CASE WHEN EXISTS (
                    SELECT 1 FROM document_chunk c
                    WHERE c.document_id = d.id AND c.embedding IS NOT NULL
                ) THEN 1 ELSE 0 END) AS embedded
            FROM article_scenarios ars
            JOIN literature_document d ON d.id = ars.document_id
            WHERE ars.scenario_id = :sid
        """), {"sid": scenario_id, "thr": eff_threshold}).mappings().first()

    total           = int(stats["total"] or 0)
    duplicates      = int(stats["duplicates"] or 0)
    unique          = max(0, total - duplicates)   # après retrait des doublons
    above           = int(stats["above_threshold"] or 0)
    below           = int(stats["below_threshold"] or 0)
    man_included    = int(stats["manually_included"] or 0)
    man_excluded    = int(stats["manually_excluded"] or 0)
    pending         = int(stats["pending"] or 0)
    man_rescued     = int(stats["manually_rescued"] or 0)   # below threshold but manually included
    man_vetoed      = int(stats["manually_vetoed"] or 0)    # above threshold but manually excluded
    with_fulltext   = int(stats["with_fulltext"] or 0)
    embedded        = int(stats["embedded"] or 0)

    # Evidence = (above threshold NOT vetoed) + manually rescued
    evidence_total  = (above - man_vetoed) + man_rescued
    screening_done  = (man_included + man_excluded) > 0

    # ── Identification : chiffres de la RECHERCHE quand ils existent ─────────
    # Un populate/rebuild récent a stocké ce que chaque source a ramené, les doublons
    # (recoupements entre sources + lignes fusionnées) et les retraits pour d'autres
    # raisons (cf. _prisma_identification_figures). Sans eux (scénario antérieur), on
    # retombe sur le corpus, qui est DÉJÀ dédupliqué : ses « doublons » sont le flag
    # is_duplicate, posé par aucun runtime — d'où l'ancien « 0 » permanent, signalé
    # ici par figures_from="corpus" pour que l'interface le dise.
    # « Passés au screening » = le corpus tel qu'il est MAINTENANT, hors doublons — la
    # même référence que /counts et l'onglet Corpus. Les chiffres de la recherche sont
    # RÉCONCILIÉS avec lui (_reconcile_prisma_identification) : ce qui a été ajouté ou
    # retiré depuis la recherche apparaît sur sa propre ligne, et l'arithmétique
    # identifiés − doublons − retraits (+ ajoutés − retirés depuis) = screening tient
    # toujours. Avant : `total` (liens, doublons compris) affiché à côté de retraits
    # calculés pour un autre corpus → 3 623 − 731 − 401 ≠ 3 602.
    _figures = _load_prisma_identification(scenario_id)
    if _figures:
        _rec = _reconcile_prisma_identification(_figures, unique)
        _identification = {
            "total_records": int(_rec.get("records_identified") or 0),
            "by_source": {str(k): int(v or 0) for k, v in (_figures.get("records_by_source") or {}).items()},
            "duplicates_removed": int(_rec.get("duplicates_removed") or 0),
            "duplicate_records_across_sources": int(_figures.get("duplicate_records_across_sources") or 0),
            "duplicate_rows_in_database": int(_figures.get("duplicate_rows_in_database") or 0),
            "unique_records": int(_rec.get("unique_records") or 0),
            "removed_no_abstract": int(_rec.get("removed_no_abstract") or 0),
            "removed_not_matching": int(_rec.get("removed_not_matching") or 0),
            "removed_other_reasons": int(_rec.get("removed_other_reasons") or 0),
            "removed_before_screening": int(_rec.get("removed_before_screening") or 0),
            "added_after_search": int(_rec.get("added_after_search") or 0),
            "removed_after_search": int(_rec.get("removed_after_search") or 0),
            "records_screened_at_search": int(_rec.get("records_screened_at_search") or 0),
            "records_screened": unique,
            "embedded": embedded,
            "figures_from": "search_run",
            "computed_at": _figures.get("computed_at"),
            "federation_incomplete": bool(_figures.get("federation_incomplete")),
        }
    else:
        _identification = {
            "total_records": total,
            "by_source": {
                "pubmed":    int(stats["pubmed"] or 0),
                "pmc":       int(stats["pmc"] or 0),
                "openalex":  int(stats["openalex"] or 0),
                "europepmc": int(stats["europepmc"] or 0),
                "crossref":  int(stats["crossref"] or 0),
                "preprint":  int(stats["preprint"] or 0),
                "medrxiv":   int(stats["medrxiv"] or 0),
                "biorxiv":   int(stats["biorxiv"] or 0),
                "semantic_scholar": int(stats["semantic_scholar"] or 0),
                "doaj":      int(stats["doaj"] or 0),
                "clinicaltrials": int(stats["clinicaltrials"] or 0),
                "core":      int(stats["core"] or 0),
                "arxiv":     int(stats["arxiv"] or 0),
                "openaire":  int(stats["openaire"] or 0),
                "db_cache":  int(stats["db_cache"] or 0),
            },
            "duplicates_removed": duplicates,
            "unique_records": unique,
            "removed_no_abstract": 0,
            "removed_not_matching": 0,
            "removed_other_reasons": 0,
            "removed_before_screening": 0,
            "added_after_search": 0,
            "removed_after_search": 0,
            "records_screened_at_search": unique,
            "records_screened": unique,
            "embedded": embedded,
            "figures_from": "corpus",
        }

    return {
        "scenario_id": scenario_id,
        "scenario_title": row["name"],
        "identification": _identification,
        "semantic_screening": {
            "threshold": eff_threshold,
            "above_threshold": above,
            "below_threshold": below,
            "method": "cosine similarity (text-embedding-3-small)",
        },
        "full_text": {
            # Numérateur ET dénominateur sur le MÊME ensemble (les preuves) : with_fulltext
            # est déjà restreint à l'ensemble de preuves, donc le % se rapporte à evidence_total
            # (auparavant : numérateur sous-ensemble / dénominateur corpus entier → trompeur).
            "with_fulltext": with_fulltext,
            "without_fulltext": max(0, evidence_total - with_fulltext),
            "pct": round(with_fulltext / evidence_total * 100, 1) if evidence_total > 0 else 0.0,
            "note": "Texte intégral via PMC / EuropePMC / Unpaywall / Semantic Scholar",
        },
        "manual_curation": {
            "included": man_included,
            "excluded": man_excluded,
            "pending": pending,
            "screening_complete": screening_done,
            "manually_rescued": man_rescued,
            "manually_vetoed": man_vetoed,
        },
        "evidence": {
            "total": evidence_total,
            "ai_auto_selected": above - man_vetoed,
            "manually_rescued": man_rescued,
            "with_fulltext": with_fulltext,
            "screening_complete": screening_done,
        },
        # Keep legacy fields for backward compatibility
        "screening": {
            "records_screened": unique,   # après retrait des doublons (≠ total identifié)
            "records_excluded_title_abstract": man_excluded,
            "records_included_screening": man_included,
            "records_awaiting_screening": pending,
        },
        "eligibility": {
            # Éligibilité = ensemble de preuves évalué en texte intégral. `above` seul
            # (≥ seuil) excluait les rescapés manuels sous le seuil et pouvait rendre
            # `not_retrieved` négatif (with_fulltext > above). On borne sur evidence_total.
            "fulltext_assessed": evidence_total,
            "fulltext_retrieved": with_fulltext,
            "fulltext_not_retrieved": max(0, evidence_total - with_fulltext),
            "fulltext_excluded": 0,
        },
        "included": {
            "total_included": man_included if screening_done else evidence_total,
            "awaiting_assessment": pending,
            "screening_complete": screening_done,
            "note": "" if screening_done else "Screening manuel non encore effectué.",
        },
    }


@app.get("/user-scenarios/{scenario_id}/pico-bulk")
def get_user_scenario_pico_bulk(scenario_id: str, limit: int = 100000, offset: int = 0) -> dict[str, Any]:
    """Tous les articles d'un scénario utilisateur avec leur PICO extrait."""
    _get_user_scenario_or_404(scenario_id)
    # Endpoint ouvert : borne limit/offset pour éviter qu'un ?limit=100000000 matérialise
    # toute la jointure en RAM/JSON (DoS mémoire). Plafond généreux (le front pagine).
    limit = max(1, min(int(limit), 5000))
    offset = max(0, int(offset))
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT d.id, d.title, d.abstract, d.year, d.source, d.authors, d.doi, d.journal,
                   d.study_design, d.pico_json, d.pico_extracted_at, COALESCE(ars.screening_status, d.screening_status) AS screening_status
            FROM article_scenarios ars
            JOIN literature_document d ON d.id = ars.document_id
            WHERE ars.scenario_id = :sid AND d.is_duplicate IS NOT TRUE
            ORDER BY
                CASE WHEN d.pico_json IS NOT NULL THEN 0 ELSE 1 END,
                d.year DESC NULLS LAST, d.id DESC
            LIMIT :limit OFFSET :offset
        """), {"sid": scenario_id, "limit": limit, "offset": offset}).mappings().fetchall()
        total_row = conn.execute(text("""
            SELECT COUNT(*) AS total,
                   COUNT(*) FILTER (WHERE d.pico_json IS NOT NULL) AS with_pico
            FROM article_scenarios ars
            JOIN literature_document d ON d.id = ars.document_id
            WHERE ars.scenario_id = :sid AND d.is_duplicate IS NOT TRUE
        """), {"sid": scenario_id}).mappings().fetchone()
    articles = []
    for r in rows:
        pico = r["pico_json"] if r["pico_json"] else None
        articles.append({
            "id": r["id"],
            "title": r["title"],
            "year": r["year"],
            "source": r["source"],
            "authors": r["authors"],
            "doi": r["doi"],
            "journal": r["journal"],
            "study_design": pico.get("study_design") if pico else r["study_design"],
            "pico_confidence": float(pico.get("pico_confidence", 0)) if pico else None,
            "P": pico.get("P") if pico else None,
            "I": pico.get("I") if pico else None,
            "C": pico.get("C") if pico else None,
            "O": pico.get("O") if pico else None,
            "pico_notes": pico.get("pico_notes") if pico else None,
            "has_pico": pico is not None,
            "pico_extracted_at": r["pico_extracted_at"].isoformat() if r["pico_extracted_at"] else None,
            "screening_status": r["screening_status"],
        })
    return {
        "scenario_id": scenario_id,
        "total": int(total_row["total"]) if total_row else 0,
        "with_pico": int(total_row["with_pico"]) if total_row else 0,
        "offset": offset,
        "limit": limit,
        "articles": articles,
    }


@app.post("/user-scenarios/{scenario_id}/articles/{article_id}/screen")
def screen_user_scenario_article(
    scenario_id: str,
    article_id: int,
    status: str,
    reason: str | None = None,
    notes: str | None = None,
    _: None = Depends(require_api_key),
) -> dict[str, Any]:
    """Screening PRISMA pour un article d'un scénario utilisateur."""
    _get_user_scenario_or_404(scenario_id)
    if status not in ("included", "excluded", "pending"):
        raise HTTPException(status_code=422, detail="status doit être 'included', 'excluded' ou 'pending'")
    with engine.connect() as conn:
        exists = conn.execute(text("""
            SELECT 1 FROM article_scenarios WHERE document_id = :doc_id AND scenario_id = :sid
        """), {"doc_id": article_id, "sid": scenario_id}).first()
    if not exists:
        raise HTTPException(status_code=404, detail="Article non trouvé dans ce scénario utilisateur")
    with engine.begin() as conn:
        row = conn.execute(text("""
            UPDATE literature_document
            SET screening_status = :status, screening_reason = :reason, screening_notes = :notes
            WHERE id = :article_id AND project_context = 'literev'
            RETURNING id
        """), {"status": status, "reason": reason, "notes": notes, "article_id": article_id}).first()
        # Migration 2 dual-write: also record the decision on the per-scenario row
        _write_ars_screening(conn, scenario_id, article_id, status, reason, notes)
    if not row:
        raise HTTPException(status_code=404, detail="Article non trouvé")
    return {"id": row[0], "status": status, "updated": True}


@app.post("/user-scenarios/{scenario_id}/articles/{article_id}/pico/extract")
def extract_user_scenario_article_pico(scenario_id: str, article_id: int, _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Extraction PICO à la demande pour UN article d'un scénario utilisateur.

    Implémentation réelle (les endpoints GESICA `.../pico/extract` délèguent ici).
    Utilise le texte intégral si disponible (meilleure qualité), sinon le résumé ;
    max_tokens=800 (400 tronquait le JSON verbeux) et remplissage tolérant des clés.
    """
    _get_user_scenario_or_404(scenario_id)
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise HTTPException(status_code=503, detail="Clé OpenAI non configurée")
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT ld.id, ld.title, ld.abstract
            FROM literature_document ld
            JOIN article_scenarios asn ON asn.document_id = ld.id AND asn.scenario_id = :sid
            WHERE ld.id = :aid
            LIMIT 1
        """), {"sid": scenario_id, "aid": article_id}).mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="Article non trouvé dans ce scénario")
    title = row["title"] or ""
    abstract = row["abstract"] or ""
    body_text, body_label, pico_source = abstract[:3000], "Abstract", "abstract"
    with engine.connect() as _ftc:
        _ft = _ftc.execute(text("""
            SELECT string_agg(content, chr(10) || chr(10) ORDER BY chunk_index) AS ft
            FROM document_chunk
            WHERE document_id = :id AND chunk_type = 'fulltext_section'
        """), {"id": article_id}).scalar()
    if _ft and len(_ft) > len(abstract):
        body_text, body_label, pico_source = _ft[:14000], "Full text", "fulltext"
    if not body_text or len(body_text.strip()) < 30:
        raise HTTPException(status_code=422, detail="Article sans texte exploitable pour l'extraction PICO")
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
        resp = _client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Title: {title}\n\n{body_label}: {body_text}"},
            ],
            temperature=0,
            seed=42,
            max_tokens=800,
            response_format={"type": "json_object"},
        )
        pico = json.loads(resp.choices[0].message.content)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Échec de l'extraction PICO: {str(e)[:200]}")
    if not isinstance(pico, dict):
        raise HTTPException(status_code=502, detail="Réponse PICO invalide (JSON attendu)")
    for _k in ("P", "I", "C", "O"):
        pico.setdefault(_k, "")
    pico.setdefault("study_design", "non précisé")
    try:
        pico["pico_confidence"] = float(pico.get("pico_confidence", 0.3))
    except (TypeError, ValueError):
        pico["pico_confidence"] = 0.3
    pico["pico_notes"] = pico.get("pico_notes", "")
    pico["pico_source"] = pico_source
    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE literature_document
            SET pico_json = CAST(:pico AS jsonb),
                pico_extracted_at = :ts,
                pico_attempts = COALESCE(pico_attempts, 0) + 1
            WHERE id = :aid
        """), {"pico": json.dumps(pico), "ts": datetime.now(timezone.utc), "aid": article_id})
    return {"article_id": article_id, "pico": pico, "extracted": True, "pico_source": pico_source}


@app.get("/user-scenarios/{scenario_id}/articles/{article_id}/pico")
def get_user_scenario_article_pico(scenario_id: str, article_id: int) -> dict[str, Any]:
    """PICO d'un article dans un scénario utilisateur (lecture).

    Implémentation réelle (les endpoints GESICA `.../pico` délèguent ici)."""
    _get_user_scenario_or_404(scenario_id)
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT ld.id, ld.title, ld.pico_json, ld.pico_extracted_at
            FROM literature_document ld
            JOIN article_scenarios asn ON asn.document_id = ld.id AND asn.scenario_id = :sid
            WHERE ld.id = :aid
            LIMIT 1
        """), {"sid": scenario_id, "aid": article_id}).mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="Article non trouvé dans ce scénario")
    pico = row["pico_json"] if isinstance(row["pico_json"], dict) else None
    return {
        "article_id": article_id,
        "title": row["title"],
        "pico": pico,
        "extracted": pico is not None,
        "pico_extracted_at": row["pico_extracted_at"].isoformat() if row.get("pico_extracted_at") else None,
    }
