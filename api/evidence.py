"""Evidence briefs: structured, LLM-generated, PDF.

Extracted from main.py (LiteRev API); `main` re-exports everything for the scripts,
tools and tests.
"""
from __future__ import annotations

from typing import Any

from fastapi import Depends, Query
from sqlalchemy import text

from .core import _job_is_active, _msg, app, engine, logger, require_api_key
from .documents import _GRADE_LEVEL_CASE, _STUDY_DESIGN_CASE, _llm_lang_directive
from .scenario_store import _get_scenario_threshold, _get_user_scenario_or_404
from .gesica import _get_scenario_name
from .relevance import _evidence_fingerprint, _get_above_threshold_articles

@app.get("/user-scenarios/{scenario_id}/evidence-brief")
def get_user_scenario_evidence_brief(scenario_id: str) -> dict[str, Any]:
    """Evidence Brief d'un scénario utilisateur (délègue au constructeur générique)."""
    _get_user_scenario_or_404(scenario_id)
    return _build_evidence_brief(scenario_id)


def _build_evidence_brief(scenario_id: str) -> dict[str, Any]:
    """Construit l'Evidence Brief d'un scénario, indépendamment de son type (user
    ou GESICA) : un seul helper générique. Toutes les statistiques (designs,
    sources, niveaux de preuve, couverture, citations) sont calculées sur le
    SOUS-ENSEMBLE PERTINENT (au-dessus du seuil sémantique)."""
    eff_thr = _get_scenario_threshold(scenario_id)
    with engine.connect() as conn:
        # `relevant*` = sous-ensemble PERTINENT (au-dessus du seuil sémantique) sur
        # lequel l'Evidence Brief / le modèle s'appuient ; `total`/`with_*` couvrent
        # le corpus complet (pour le contexte).
        corpus_stats = conn.execute(text("""
            SELECT
                -- total/with_pico/with_fulltext/included/excluded/pending comptés hors
                -- doublons (comme la sortie PDF « (uniques) » et la carte scénario) →
                -- headline cohérent entre le brief à l'écran, le PDF et le tableau de bord.
                COUNT(*) FILTER (WHERE d.is_duplicate IS NOT TRUE) AS total,
                COUNT(*) FILTER (WHERE d.is_duplicate IS TRUE) AS duplicates,
                COUNT(*) FILTER (WHERE d.is_duplicate IS NOT TRUE AND d.pico_json IS NOT NULL) AS with_pico,
                COUNT(*) FILTER (WHERE d.is_duplicate IS NOT TRUE AND COALESCE(ars.screening_status, d.screening_status) = 'included') AS included,
                COUNT(*) FILTER (WHERE d.is_duplicate IS NOT TRUE AND COALESCE(ars.screening_status, d.screening_status) = 'excluded') AS excluded,
                COUNT(*) FILTER (WHERE d.is_duplicate IS NOT TRUE AND (COALESCE(ars.screening_status, d.screening_status) = 'pending' OR COALESCE(ars.screening_status, d.screening_status) IS NULL)) AS pending,
                COUNT(*) FILTER (WHERE d.is_duplicate IS NOT TRUE AND EXISTS (
                    SELECT 1 FROM document_chunk c
                    WHERE c.document_id = d.id AND c.chunk_type = 'fulltext_section'
                )) AS with_fulltext,
                COUNT(*) FILTER (WHERE d.is_duplicate IS NOT TRUE
                    AND COALESCE(ars.screening_status, d.screening_status) IS DISTINCT FROM 'excluded'
                    AND (COALESCE(ars.screening_status, d.screening_status) = 'included' OR COALESCE(ars.similarity_score, 0) >= :thr)) AS relevant,
                COUNT(*) FILTER (WHERE d.is_duplicate IS NOT TRUE
                    AND COALESCE(ars.screening_status, d.screening_status) IS DISTINCT FROM 'excluded'
                    AND (COALESCE(ars.screening_status, d.screening_status) = 'included' OR COALESCE(ars.similarity_score, 0) >= :thr)
                    AND d.pico_json IS NOT NULL) AS relevant_with_pico,
                COUNT(*) FILTER (WHERE d.is_duplicate IS NOT TRUE
                    AND COALESCE(ars.screening_status, d.screening_status) IS DISTINCT FROM 'excluded'
                    AND (COALESCE(ars.screening_status, d.screening_status) = 'included' OR COALESCE(ars.similarity_score, 0) >= :thr)
                    AND EXISTS (SELECT 1 FROM document_chunk c
                        WHERE c.document_id = d.id AND c.chunk_type = 'fulltext_section')) AS relevant_with_fulltext,
                -- Couverture & citations : calculées sur le SOUS-ENSEMBLE PERTINENT
                -- (au-dessus du seuil), pas sur le corpus complet.
                MIN(d.year) FILTER (WHERE d.is_duplicate IS NOT TRUE
                    AND COALESCE(ars.screening_status, d.screening_status) IS DISTINCT FROM 'excluded'
                    AND (COALESCE(ars.screening_status, d.screening_status) = 'included' OR COALESCE(ars.similarity_score, 0) >= :thr)
                    AND d.year BETWEEN 1800 AND EXTRACT(YEAR FROM CURRENT_DATE)::int) AS year_min,
                MAX(d.year) FILTER (WHERE d.is_duplicate IS NOT TRUE
                    AND COALESCE(ars.screening_status, d.screening_status) IS DISTINCT FROM 'excluded'
                    AND (COALESCE(ars.screening_status, d.screening_status) = 'included' OR COALESCE(ars.similarity_score, 0) >= :thr)
                    AND d.year BETWEEN 1800 AND EXTRACT(YEAR FROM CURRENT_DATE)::int) AS year_max,
                AVG(d.citation_count) FILTER (WHERE d.is_duplicate IS NOT TRUE
                    AND COALESCE(ars.screening_status, d.screening_status) IS DISTINCT FROM 'excluded'
                    AND (COALESCE(ars.screening_status, d.screening_status) = 'included' OR COALESCE(ars.similarity_score, 0) >= :thr)
                    AND d.citation_count IS NOT NULL) AS avg_citations,
                MAX(d.citation_count) FILTER (WHERE d.is_duplicate IS NOT TRUE
                    AND COALESCE(ars.screening_status, d.screening_status) IS DISTINCT FROM 'excluded'
                    AND (COALESCE(ars.screening_status, d.screening_status) = 'included' OR COALESCE(ars.similarity_score, 0) >= :thr)) AS max_citations
            FROM article_scenarios ars
            JOIN literature_document d ON d.id = ars.document_id
            WHERE ars.scenario_id = :sid
        """), {"sid": scenario_id, "thr": eff_thr}).mappings().fetchone()

        top_articles = conn.execute(text("""
            SELECT d.id, d.title, d.abstract, d.year, d.journal, d.authors, d.doi,
                   d.study_design, d.pico_json, d.citation_count, COALESCE(ars.screening_status, d.screening_status) AS screening_status,
                   d.quality_score, ars.similarity_score
            FROM article_scenarios ars
            JOIN literature_document d ON d.id = ars.document_id
            WHERE ars.scenario_id = :sid
              AND d.is_duplicate IS NOT TRUE AND d.abstract IS NOT NULL
              AND COALESCE(ars.screening_status, d.screening_status) IS DISTINCT FROM 'excluded'
              AND (COALESCE(ars.screening_status, d.screening_status) = 'included' OR COALESCE(ars.similarity_score, 0) >= :thr)
            ORDER BY
                CASE WHEN COALESCE(ars.screening_status, d.screening_status) = 'included' THEN 0 ELSE 1 END,
                d.citation_count DESC NULLS LAST, d.year DESC NULLS LAST
            LIMIT 15
        """), {"sid": scenario_id, "thr": eff_thr}).mappings().fetchall()

        # Toutes les distributions ci-dessous sont calculées sur le SOUS-ENSEMBLE
        # PERTINENT (au-dessus du seuil sémantique), pas sur le corpus complet :
        # elles doivent sommer au nombre de « pertinents » affiché (ex. 51), pas
        # au total du corpus (ex. 100).
        study_designs = conn.execute(text(f"""
            WITH b AS (
                SELECT lower(coalesce(
                    nullif(trim(ld.study_design), ''),
                    nullif(trim(ld.pico_json->>'study_design'), ''), '')) AS d
                FROM article_scenarios ars
                JOIN literature_document ld ON ld.id = ars.document_id
                WHERE ars.scenario_id = :sid AND ld.is_duplicate IS NOT TRUE
                  AND COALESCE(ars.screening_status, ld.screening_status) IS DISTINCT FROM 'excluded'
                  AND (COALESCE(ars.screening_status, ld.screening_status) = 'included' OR COALESCE(ars.similarity_score, 0) >= :thr)
            )
            SELECT {_STUDY_DESIGN_CASE} AS design, COUNT(*) AS n
            FROM b GROUP BY 1 ORDER BY 2 DESC
        """), {"sid": scenario_id, "thr": eff_thr}).mappings().fetchall()

        year_dist = conn.execute(text("""
            SELECT d.year, COUNT(*) AS n
            FROM article_scenarios ars
            JOIN literature_document d ON d.id = ars.document_id
            WHERE ars.scenario_id = :sid AND d.is_duplicate IS NOT TRUE
              AND COALESCE(ars.screening_status, d.screening_status) IS DISTINCT FROM 'excluded'
              AND (COALESCE(ars.screening_status, d.screening_status) = 'included' OR COALESCE(ars.similarity_score, 0) >= :thr)
              AND d.year >= 1800 AND d.year <= EXTRACT(YEAR FROM CURRENT_DATE)::int
            GROUP BY d.year ORDER BY d.year ASC
        """), {"sid": scenario_id, "thr": eff_thr}).mappings().fetchall()

        source_dist = conn.execute(text("""
            SELECT d.source, COUNT(*) AS n
            FROM article_scenarios ars
            JOIN literature_document d ON d.id = ars.document_id
            WHERE ars.scenario_id = :sid AND d.is_duplicate IS NOT TRUE
              AND COALESCE(ars.screening_status, d.screening_status) IS DISTINCT FROM 'excluded'
              AND (COALESCE(ars.screening_status, d.screening_status) = 'included' OR COALESCE(ars.similarity_score, 0) >= :thr)
            GROUP BY d.source ORDER BY n DESC LIMIT 8
        """), {"sid": scenario_id, "thr": eff_thr}).mappings().fetchall()

        # Niveau de preuve = GRADE strict basé sur le DEVIS (cf. _GRADE_LEVEL_CASE),
        # pas sur le quality_score composite : sinon une cohorte/cas-témoins bien
        # citée passait « Modérée » voire « Forte », ce qui contredit GRADE (toute
        # étude observationnelle part en certitude faible).
        evidence_levels = conn.execute(text(f"""
            WITH b AS (
                SELECT lower(coalesce(
                    nullif(trim(ld.study_design), ''),
                    nullif(trim(ld.pico_json->>'study_design'), ''), '')) AS d
                FROM article_scenarios ars
                JOIN literature_document ld ON ld.id = ars.document_id
                WHERE ars.scenario_id = :sid AND ld.is_duplicate IS NOT TRUE
                  AND COALESCE(ars.screening_status, ld.screening_status) IS DISTINCT FROM 'excluded'
                  AND (COALESCE(ars.screening_status, ld.screening_status) = 'included' OR COALESCE(ars.similarity_score, 0) >= :thr)
            )
            SELECT {_GRADE_LEVEL_CASE} AS level, COUNT(*) AS n
            FROM b GROUP BY 1 ORDER BY 2 DESC
        """), {"sid": scenario_id, "thr": eff_thr}).mappings().fetchall()

    pico_table = []
    for r in top_articles:
        pj = r["pico_json"] or {}
        pico_table.append({
            "id": r["id"],
            "title": (r["title"] or "")[:120],
            "year": r["year"],
            "journal": r["journal"],
            "citation_count": r["citation_count"],
            "study_design": r["study_design"] or pj.get("study_design", ""),
            "screening_status": r["screening_status"],
            "similarity_score": round(float(r["similarity_score"]), 3) if r["similarity_score"] else None,
            "pico": {
                "population": pj.get("population", pj.get("P", "")),
                "intervention": pj.get("intervention", pj.get("I", "")),
                "comparator": pj.get("comparator", pj.get("C", "")),
                "outcome": pj.get("outcome", pj.get("O", "")),
                "study_design": pj.get("study_design", ""),
                "key_finding": pj.get("key_finding", pj.get("conclusion", "")),
                "limitations": pj.get("limitations", ""),
                "evidence_level": pj.get("evidence_level", ""),
            }
        })

    return {
        "scenario_id": scenario_id,
        "generated_at": __import__('datetime').datetime.now().isoformat(),
        "corpus_stats": {
            "total": int(corpus_stats["total"] or 0),
            "duplicates": int(corpus_stats["duplicates"] or 0),
            "with_pico": int(corpus_stats["with_pico"] or 0),
            "with_fulltext": int(corpus_stats["with_fulltext"] or 0),
            "relevant": int(corpus_stats["relevant"] or 0),
            "relevant_with_pico": int(corpus_stats["relevant_with_pico"] or 0),
            "relevant_with_fulltext": int(corpus_stats["relevant_with_fulltext"] or 0),
            "threshold": eff_thr,
            "included": int(corpus_stats["included"] or 0),
            "excluded": int(corpus_stats["excluded"] or 0),
            "pending": int(corpus_stats["pending"] or 0),
            "year_min": corpus_stats["year_min"],
            "year_max": corpus_stats["year_max"],
            "avg_citations": round(float(corpus_stats["avg_citations"]), 1) if corpus_stats["avg_citations"] else None,
            "max_citations": int(corpus_stats["max_citations"]) if corpus_stats["max_citations"] else None,
            "pico_coverage_pct": round(
                100 * int(corpus_stats["with_pico"] or 0) / max(int(corpus_stats["total"] or 1), 1), 1
            ),
        },
        "double_blind_stats": {"reviewer_1_done": 0, "reviewer_2_done": 0, "both_done": 0, "agreements": 0, "conflicts": 0},
        "top_articles": [
            {
                "id": r["id"],
                "title": r["title"],
                "year": r["year"],
                "journal": r["journal"],
                "authors": r["authors"],
                "doi": r["doi"],
                "study_design": r["study_design"] or (r["pico_json"].get("study_design") if r["pico_json"] else None),
                "citation_count": r["citation_count"],
                "screening_status": r["screening_status"],
                "quality_score": round(float(r["quality_score"]), 2) if r["quality_score"] else None,
                "similarity_score": round(float(r["similarity_score"]), 3) if r["similarity_score"] else None,
                "abstract_excerpt": (r["abstract"] or "")[:500],
                "pico_summary": {
                    "population": r["pico_json"].get("population", r["pico_json"].get("P", "")) if r["pico_json"] else "",
                    "intervention": r["pico_json"].get("intervention", r["pico_json"].get("I", "")) if r["pico_json"] else "",
                    "outcome": r["pico_json"].get("outcome", r["pico_json"].get("O", "")) if r["pico_json"] else "",
                    "key_finding": r["pico_json"].get("key_finding", r["pico_json"].get("conclusion", "")) if r["pico_json"] else "",
                } if r["pico_json"] else None,
            }
            for r in top_articles
        ],
        "pico_table": pico_table,
        "study_design_distribution": [{"design": d["design"], "count": int(d["n"])} for d in study_designs],
        "year_distribution": [{"year": d["year"], "count": int(d["n"])} for d in year_dist],
        "source_distribution": [{"source": s["source"], "count": int(s["n"])} for s in source_dist],
        "evidence_level_distribution": [{"level": e["level"], "count": int(e["n"])} for e in evidence_levels],
    }


@app.get("/user-scenarios/{scenario_id}/evidence-brief/pdf")
def get_user_scenario_evidence_brief_pdf(scenario_id: str):
    """PDF Evidence Brief pour un scénario utilisateur."""
    row = _get_user_scenario_or_404(scenario_id)
    # Injecter le scénario utilisateur dans user_scenarios avec is_system=True temporairement
    # n'est plus nécessaire : get_evidence_brief_pdf lit maintenant depuis la DB via _get_db_gesica_scenario_or_404
    # On crée une entrée temporaire dans user_scenarios si nécessaire
    # Pour les user_scenarios, on appelle directement la logique PDF avec les données du scénario
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    import io as _io

    # Comme le panneau Evidences à l'écran : tout porte sur le SOUS-ENSEMBLE
    # PERTINENT (≥ seuil sémantique, via article_scenarios), pas le corpus complet.
    eff_thr = _get_scenario_threshold(scenario_id)
    with engine.connect() as _conn:
        corpus_stats = _conn.execute(text("""
            SELECT
                COUNT(*) FILTER (WHERE d.is_duplicate IS NOT TRUE) AS total,
                COUNT(*) FILTER (WHERE d.is_duplicate IS NOT TRUE
                    AND COALESCE(ars.screening_status, d.screening_status) IS DISTINCT FROM 'excluded'
                    AND (COALESCE(ars.screening_status, d.screening_status) = 'included' OR COALESCE(ars.similarity_score, 0) >= :thr)) AS relevant,
                MIN(d.year) FILTER (WHERE d.is_duplicate IS NOT TRUE
                    AND COALESCE(ars.screening_status, d.screening_status) IS DISTINCT FROM 'excluded'
                    AND (COALESCE(ars.screening_status, d.screening_status) = 'included' OR COALESCE(ars.similarity_score, 0) >= :thr)
                    AND d.year BETWEEN 1800 AND EXTRACT(YEAR FROM CURRENT_DATE)::int) AS year_min,
                MAX(d.year) FILTER (WHERE d.is_duplicate IS NOT TRUE
                    AND COALESCE(ars.screening_status, d.screening_status) IS DISTINCT FROM 'excluded'
                    AND (COALESCE(ars.screening_status, d.screening_status) = 'included' OR COALESCE(ars.similarity_score, 0) >= :thr)
                    AND d.year BETWEEN 1800 AND EXTRACT(YEAR FROM CURRENT_DATE)::int) AS year_max,
                COUNT(*) FILTER (WHERE d.is_duplicate IS NOT TRUE
                    AND COALESCE(ars.screening_status, d.screening_status) IS DISTINCT FROM 'excluded'
                    AND (COALESCE(ars.screening_status, d.screening_status) = 'included' OR COALESCE(ars.similarity_score, 0) >= :thr)
                    AND COALESCE(ars.screening_status, d.screening_status) = 'included') AS included,
                COUNT(*) FILTER (WHERE d.is_duplicate IS NOT TRUE
                    AND COALESCE(ars.screening_status, d.screening_status) IS DISTINCT FROM 'excluded'
                    AND (COALESCE(ars.screening_status, d.screening_status) = 'included' OR COALESCE(ars.similarity_score, 0) >= :thr)
                    AND d.pico_json IS NOT NULL) AS with_pico
            FROM article_scenarios ars
            JOIN literature_document d ON d.id = ars.document_id
            WHERE d.project_context = 'literev' AND ars.scenario_id = :sid
        """), {"sid": scenario_id, "thr": eff_thr}).mappings().first()
        top_articles = _conn.execute(text("""
            SELECT d.title, d.year, d.journal, d.authors,
                   COALESCE((d.pico_json->>'study_design'), d.study_design, 'N/A') AS design
            FROM article_scenarios ars
            JOIN literature_document d ON d.id = ars.document_id
            WHERE d.project_context = 'literev' AND ars.scenario_id = :sid
              AND d.is_duplicate IS NOT TRUE AND d.abstract IS NOT NULL
              AND COALESCE(ars.screening_status, d.screening_status) IS DISTINCT FROM 'excluded'
              AND (COALESCE(ars.screening_status, d.screening_status) = 'included' OR COALESCE(ars.similarity_score, 0) >= :thr)
            ORDER BY d.quality_score DESC NULLS LAST, d.year DESC NULLS LAST
            LIMIT 100000
        """), {"sid": scenario_id, "thr": eff_thr}).mappings().all()
        study_designs = _conn.execute(text("""
            SELECT
                COALESCE((d.pico_json->>'study_design'), d.study_design, 'Non classifié') AS design,
                COUNT(*) AS n
            FROM article_scenarios ars
            JOIN literature_document d ON d.id = ars.document_id
            WHERE d.project_context = 'literev' AND ars.scenario_id = :sid
              AND d.is_duplicate IS NOT TRUE
              AND COALESCE(ars.screening_status, d.screening_status) IS DISTINCT FROM 'excluded'
              AND (COALESCE(ars.screening_status, d.screening_status) = 'included' OR COALESCE(ars.similarity_score, 0) >= :thr)
            GROUP BY 1 ORDER BY 2 DESC LIMIT 8
        """), {"sid": scenario_id, "thr": eff_thr}).mappings().all()

    _buf = _io.BytesIO()
    _doc = SimpleDocTemplate(_buf, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    _styles = getSampleStyleSheet()
    _dark_green = colors.HexColor("#1a3a2a")
    _brand_green = colors.HexColor("#22c55e")
    _light_text = colors.HexColor("#374151")
    _title_style = ParagraphStyle("UT", parent=_styles["Title"], fontSize=22, textColor=_dark_green, spaceAfter=6, fontName="Helvetica-Bold")
    _h2_style = ParagraphStyle("UH2", parent=_styles["Heading2"], fontSize=13, textColor=_dark_green, spaceBefore=14, spaceAfter=4, fontName="Helvetica-Bold")
    _body_style = ParagraphStyle("UB", parent=_styles["Normal"], fontSize=9, textColor=_light_text, spaceAfter=4, leading=14)
    _small_style = ParagraphStyle("US", parent=_styles["Normal"], fontSize=7, textColor=colors.HexColor("#6b7280"), spaceAfter=2)

    _story = []
    _story.append(Paragraph("LiteRev : Evidence to Scenario", _small_style))
    _story.append(Paragraph(f"Evidence Brief : {row['name']}", _title_style))
    _story.append(Paragraph(f"Scénario utilisateur · Généré le {__import__('datetime').datetime.now().strftime('%d/%m/%Y à %H:%M')}", _small_style))
    _story.append(HRFlowable(width="100%", thickness=2, color=_brand_green, spaceAfter=12))

    _total = int(corpus_stats["total"] or 0)
    _relevant = int(corpus_stats["relevant"] or 0)
    _included = int(corpus_stats["included"] or 0)
    _with_pico = int(corpus_stats["with_pico"] or 0)
    _year_min = corpus_stats["year_min"] or "N/A"
    _year_max = corpus_stats["year_max"] or "N/A"

    _story.append(Paragraph("Corpus documentaire", _h2_style))
    _stats_data = [["Indicateur", "Valeur"], ["Articles du corpus (uniques)", str(_total)],
                   [f"Articles pertinents utilisés (≥ {eff_thr:.2f})", str(_relevant)],
                   ["Articles inclus", str(_included) if _included > 0 else "En attente"], ["Pertinents avec PICO", str(_with_pico)],
                   ["Période (pertinents)", f"{_year_min} – {_year_max}"]]
    _st = Table(_stats_data, colWidths=[10*cm, 6*cm])
    _st.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), _dark_green), ("TEXTCOLOR", (0,0), (-1,0), colors.white),
                              ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,-1), 9),
                              ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#f9fafb"), colors.white]),
                              ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#e5e7eb")), ("PADDING", (0,0), (-1,-1), 6)]))
    _story.append(_st)

    if study_designs:
        _story.append(Paragraph("Distribution par type d'étude", _h2_style))
        _dd = [["Type d'étude", "Nombre"]] + [[str(d["design"]), str(d["n"])] for d in study_designs]
        _dt = Table(_dd, colWidths=[10*cm, 6*cm])
        _dt.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), _dark_green), ("TEXTCOLOR", (0,0), (-1,0), colors.white),
                                  ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,-1), 9),
                                  ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#f9fafb"), colors.white]),
                                  ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#e5e7eb")), ("PADDING", (0,0), (-1,-1), 6)]))
        _story.append(_dt)

    if top_articles:
        _story.append(Paragraph("Articles les plus pertinents", _h2_style))
        for _i, _art in enumerate(top_articles, 1):
            _story.append(Paragraph(f"<b>{_i}. {(_art['title'] or 'Sans titre')[:120]}</b>",
                                    ParagraphStyle("at", parent=_body_style, fontSize=9, textColor=_dark_green)))
            _story.append(Paragraph((_art['authors'] or '')[:80], _small_style))
            _story.append(Paragraph(f"{_art['year'] or 'N/A'} · {_art['journal'] or 'Journal inconnu'} · {_art['design']}", _small_style))
            _story.append(Spacer(1, 4))

    _story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e5e7eb"), spaceBefore=16))
    _story.append(Paragraph("Ce document a été généré automatiquement par LiteRev.", _small_style))
    _doc.build(_story)
    _buf.seek(0)
    from fastapi.responses import Response as _Resp
    return _Resp(content=_buf.read(), media_type="application/pdf",
                 headers={"Content-Disposition": f'attachment; filename="evidence_brief_{scenario_id}.pdf"'})


# ─── EVIDENCE BRIEF LLM AUTOMATIQUE ──────────────────────────────────────────

_BRIEF_GENERATION_JOBS: dict[str, dict] = {}


def _generate_evidence_brief_llm(scenario_id: str, force: bool = False, lang: str | None = None) -> dict[str, Any]:
    """
    Génère un Evidence Brief narratif complet via LLM à partir des articles
    au-dessus du seuil de similarité (ou validés humainement).
    Sauvegarde le résultat dans scenario_settings.evidence_brief_json.
    """
    import json as _json
    from datetime import datetime, timezone
    from llm_usage import MeteredOpenAI as _OAI

    threshold = _get_scenario_threshold(scenario_id)
    # Résumés/PICO/texte intégral pour les 30 articles du contexte seulement ; les
    # autres lignes (légères) servent aux statistiques et à l'empreinte du corpus.
    articles = _get_above_threshold_articles(scenario_id, threshold, include_fulltext=True,
                                             fulltext_query=_get_scenario_name(scenario_id),
                                             fulltext_top_docs=30, fulltext_char_cap=2800,
                                             full_rows=30)

    if not articles:
        return {"error": _msg(lang, "Aucun article au-dessus du seuil pour générer le brief.",
                              "No article above the threshold to generate the brief.")}

    # Cache par EMPREINTE DU CORPUS (+ seuil + langue), et non plus par âge (< 24h).
    # On ne régénère que si le sous-ensemble pertinent, le seuil ou la langue ont
    # changé - l'ancien cache « 24h » régénérait un corpus inchangé ET servait un
    # brief périmé (mauvaise langue / corpus modifié) tant qu'il avait moins de 24h.
    # v4 : le prompt porte désormais le digest du corpus COMPLET (et plus seulement les
    # 30 articles reproduits). Le suffixe invalide une fois les briefs écrits sans lui.
    _brief_fp = _evidence_fingerprint([a["id"] for a in articles], threshold, lang, "brief-v4-full-corpus-digest")
    if not force:
        with engine.connect() as conn:
            row = conn.execute(text("""
                SELECT evidence_brief_json FROM scenario_settings WHERE scenario_id = :sid
            """), {"sid": scenario_id}).mappings().first()
        if row and row["evidence_brief_json"]:
            _cached = dict(row["evidence_brief_json"])
            if _cached.get("_corpus_fingerprint") == _brief_fp:
                return _cached

    scenario_name = _get_scenario_name(scenario_id)

    # Préparer le contexte : top 30 articles. Trois niveaux de texte, du plus
    # structuré au plus brut : PICO (structure) + abstract (récit) + extrait du
    # TEXTE INTÉGRAL quand disponible (méthodes/résultats que l'abstract résume).
    # Le LLM voit ainsi le texte réel des articles, pas seulement leur PICO.
    context_articles = []
    for a in articles[:30]:
        pj = a.get("pico_json") or {}
        _ca = {
            "title": a.get("title", ""),
            "year": a.get("year"),
            "journal": a.get("journal", ""),
            "citation_count": a.get("citation_count"),
            "study_design": a.get("study_design") or pj.get("study_design", ""),
            "screening_status": a.get("screening_status"),
            "P": pj.get("population", pj.get("P", "")),
            "I": pj.get("intervention", pj.get("I", "")),
            "C": pj.get("comparator", pj.get("C", "")),
            "O": pj.get("outcome", pj.get("O", "")),
            "key_finding": pj.get("key_finding", pj.get("conclusion", "")),
            "abstract": (a.get("abstract") or "")[:1500],
        }
        _ft = (a.get("fulltext") or "").strip()
        if _ft:
            _ca["fulltext_excerpt"] = _ft
        context_articles.append(_ca)

    context_str = _json.dumps(context_articles, ensure_ascii=False, indent=2)

    # Le brief parle au nom du corpus ENTIER : le digest (agrégats calculés sur TOUS les
    # articles pertinents, sans échantillonnage) précède les articles reproduits, qui ne
    # sont là que pour citer. Avant, le modèle lisait 30 articles sur 2 732 et annonçait
    # pourtant le total : les distributions décrites étaient celles de son échantillon.
    from .digest import corpus_digest, digest_coverage_note, digest_to_prompt
    _digest = corpus_digest(scenario_id, threshold)
    _digest_block = digest_to_prompt(_digest)
    _coverage = digest_coverage_note(_digest, len(context_articles))

    # Stats corpus (sur TOUS les articles pertinents : `has_pico` est renseigné pour
    # chaque ligne, `pico_json` seulement pour les 30 du contexte).
    total = len(articles)
    included = sum(1 for a in articles if a.get("screening_status") == "included")
    with_pico = sum(1 for a in articles if a.get("has_pico") or a.get("pico_json"))
    years = [a["year"] for a in articles if a.get("year")]
    year_range = f"{min(years)}-{max(years)}" if years else "N/A"

    study_designs = {}
    for a in articles:
        pj = a.get("pico_json") or {}
        d = a.get("study_design") or pj.get("study_design", "Non classifié")
        study_designs[d] = study_designs.get(d, 0) + 1

    top_designs = sorted(study_designs.items(), key=lambda x: -x[1])[:5]

    # Plafond GRADE déterministe d'après les devis présents : on empêche le LLM de
    # surclasser un corpus observationnel. En GRADE strict, toute étude
    # observationnelle (cohorte, cas-témoins, transversale, série de cas) démarre
    # en certitude FAIBLE ; seuls les essais randomisés et leurs synthèses peuvent
    # soutenir une certitude élevée.
    _designs_blob = " ".join(study_designs.keys()).lower()
    if any(k in _designs_blob for k in ("randomi", "rct", "meta-analysis", "méta-analyse",
                                        "meta analysis", "systematic review", "revue systématique")):
        _grade_ceiling = ("Forte possible (essais randomisés / synthèses d'essais présents), "
                          "à pondérer selon la cohérence et le risque de biais")
    elif any(k in _designs_blob for k in ("controlled trial", "clinical trial", "quasi-exper",
                                          "quasi exper", "non-randomi", "non randomi")):
        _grade_ceiling = "Modérée au mieux (essais contrôlés non randomisés / quasi-expérimental)"
    else:
        _grade_ceiling = ("Faible (corpus observationnel : GRADE plafonne la certitude à faible, "
                          "sauf upgrade explicitement justifié)")

    system_prompt = """Tu es un expert en sciences de la santé et en revue systématique de la littérature scientifique.
Tu génères des Evidence Briefs complets, rigoureux et structurés.
Tu dois produire un JSON structuré avec tous les champs demandés.
Sois précis, factuel, et base-toi exclusivement sur les articles fournis.
Ne pas utiliser de tiret cadratin (em dash). Utiliser des tirets simples (-) si nécessaire.""" + _llm_lang_directive(lang)

    user_prompt = f"""Génère un Evidence Brief complet pour le scénario de recherche : "{scenario_name}"

{_digest_block or f"Corpus : {total} articles ({year_range}), {with_pico} avec PICO extrait, {included} validés humainement."}
Designs d'étude principaux : {', '.join(f'{d} ({n})' for d, n in top_designs)}.

{_coverage}

Articles reproduits ({len(context_articles)} les mieux établis, pour citer et illustrer) :
{context_str}

RÈGLES GRADE (strict) pour « evidence_level » et « grade_recommendation » :
- Le niveau de preuve part du DEVIS d'étude. Essais randomisés et méta-analyses/revues
  systématiques d'essais → certitude potentiellement Forte. TOUTE étude observationnelle
  (cohorte, cas-témoins, transversale, série/rapport de cas) démarre en certitude FAIBLE.
  Ne JAMAIS classer des cas-témoins ou des cohortes en preuve « Forte ».
- « evidence_level » reflète la MEILLEURE preuve du corpus, plafonnée par le devis.
  Plafond estimé d'après les devis présents : {_grade_ceiling}.
- « grade_recommendation » suit le niveau de preuve : A uniquement si preuves Fortes et
  cohérentes (essais/méta-analyses) ; B si preuves modérées ; C si preuves faibles
  (corpus observationnel) ; D si très faibles / avis d'experts ; GPP pour une bonne
  pratique sans preuve directe.

Génère un JSON avec EXACTEMENT ces champs :
{{
  "executive_summary": "Résumé exécutif en 3-4 phrases synthétisant les principales conclusions",
  "clinical_context": "Contexte clinique et importance du sujet (2-3 paragraphes)",
  "key_findings": ["Finding 1", "Finding 2", "Finding 3", "Finding 4", "Finding 5"],
  "recommended_actions": ["Action 1", "Action 2", "Action 3", "Action 4"],
  "evidence_synthesis": "Synthèse narrative détaillée des évidences (4-6 paragraphes)",
  "population_summary": "Résumé des populations étudiées",
  "intervention_summary": "Résumé des interventions/expositions étudiées",
  "outcome_summary": "Résumé des outcomes mesurés",
  "methodological_quality": "Évaluation de la qualité méthodologique globale",
  "limitations": ["Limite 1", "Limite 2", "Limite 3"],
  "research_gaps": ["Gap 1", "Gap 2", "Gap 3"],
  "clinical_implications": "Implications cliniques pratiques (2-3 paragraphes)",
  "implementation_recommendations": ["Recommandation 1", "Recommandation 2", "Recommandation 3"],
  "evidence_level": "Niveau de preuve global (Fort/Modéré/Faible/Insuffisant)",
  "grade_recommendation": "Grade de recommandation (A/B/C/D/GPP)",
  "future_research": "Directions pour la recherche future",
  "key_references": [
    {{"title": "...", "year": ..., "journal": "...", "key_contribution": "..."}}
  ]
}}
Retourne UNIQUEMENT le JSON valide."""

    try:
        client = _OAI(timeout=90.0)
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=3000,
            response_format={"type": "json_object"},
        )
        brief = _json.loads(response.choices[0].message.content)

        # Ajouter les métadonnées
        brief["_meta"] = {
            "scenario_id": scenario_id,
            "scenario_name": scenario_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "articles_used": total,
            "articles_above_threshold": total,
            "threshold": threshold,
            "human_validated": included,
            "year_range": year_range,
            "study_designs": dict(top_designs),
            "auto_generated": True,
            "model": "gpt-4.1",
        }

        # Empreinte du corpus : sert de clé de cache « ne pas régénérer si inchangé »
        # (relue en tête de fonction). Stockée DANS le brief pour éviter une colonne.
        brief["_corpus_fingerprint"] = _brief_fp

        # Sauvegarder en DB
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO scenario_settings (scenario_id, evidence_brief_json, brief_generated_at, updated_at)
                VALUES (:sid, CAST(:brief AS jsonb), NOW(), NOW())
                ON CONFLICT (scenario_id) DO UPDATE
                SET evidence_brief_json = CAST(:brief AS jsonb),
                    brief_generated_at = NOW(),
                    updated_at = NOW()
            """), {"sid": scenario_id, "brief": _json.dumps(brief)})

        logger.info(f"Evidence Brief LLM généré pour {scenario_id}: {len(context_articles)} articles.")
        return brief

    except Exception as e:
        logger.error(f"Evidence Brief LLM {scenario_id}: {e}", exc_info=True)
        return {"error": str(e)}


@app.post("/scenarios/{scenario_id}/evidence-brief/generate")
def generate_evidence_brief(scenario_id: str, force: bool = False, lang: str | None = Query(None), _: None = Depends(require_api_key)) -> dict[str, Any]:
    """
    Déclenche la génération asynchrone de l'Evidence Brief LLM.
    Fonctionne pour GESICA et user_scenarios.
    """
    import threading, time

    if _job_is_active(_BRIEF_GENERATION_JOBS.get(scenario_id)):
        return {"status": "already_running", "scenario_id": scenario_id}

    _BRIEF_GENERATION_JOBS[scenario_id] = {"status": "running", "started_at": time.time()}

    def _run():
        try:
            result = _generate_evidence_brief_llm(scenario_id, force=force, lang=lang)
            if "error" in result:
                _BRIEF_GENERATION_JOBS[scenario_id] = {"status": "error", "error": result["error"]}
            else:
                _BRIEF_GENERATION_JOBS[scenario_id] = {"status": "done", "generated_at": result.get("_meta", {}).get("generated_at")}
        except Exception as e:
            # Without this, an exception (or a killed thread) leaves status stuck
            # at "running" and every retry returns "already_running" forever.
            logger.error(f"Evidence Brief job {scenario_id}: {e}", exc_info=True)
            _BRIEF_GENERATION_JOBS[scenario_id] = {"status": "error", "error": str(e)}

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "started", "scenario_id": scenario_id}


@app.get("/scenarios/{scenario_id}/evidence-brief/generate/status")
def get_brief_generation_status(scenario_id: str) -> dict[str, Any]:
    """Statut du job de génération du brief LLM."""
    return _BRIEF_GENERATION_JOBS.get(scenario_id, {"status": "idle"})


@app.get("/scenarios/{scenario_id}/evidence-brief/llm")
def get_llm_evidence_brief(scenario_id: str, lang: str | None = Query(None)) -> dict[str, Any]:
    """
    Retourne le brief LLM généré (depuis le cache DB) DANS LA LANGUE demandée.
    Le cache n'est servi que si son empreinte (corpus + seuil + langue) correspond ;
    sinon (corpus modifié OU langue différente) on déclenche une régénération et on
    retourne un statut pending. Sans `lang`, le brief reste en français (défaut).
    """
    threshold = _get_scenario_threshold(scenario_id)
    articles = _get_above_threshold_articles(scenario_id, threshold, full_rows=0)   # ids/statuts seulement
    if not articles:
        return {"status": "empty", "message": _msg(lang,
                "Aucun article au-dessus du seuil. Ajoutez des articles ou abaissez le seuil de similarité.",
                "No article above the threshold. Add articles or lower the similarity threshold.")}

    _want_fp = _evidence_fingerprint([a["id"] for a in articles], threshold, lang, "brief-v3-relevant-fulltext")

    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT evidence_brief_json, brief_generated_at
            FROM scenario_settings WHERE scenario_id = :sid
        """), {"sid": scenario_id}).mappings().first()

    if row and row["evidence_brief_json"]:
        brief = dict(row["evidence_brief_json"])
        # Ne servir le cache que s'il correspond au corpus ET à la langue demandés.
        # Un brief français ne doit pas être resservi quand l'UI passe en anglais.
        if brief.get("_corpus_fingerprint") == _want_fp:
            brief["_cached"] = True
            brief["_generated_at"] = row["brief_generated_at"].isoformat() if row["brief_generated_at"] else None
            return brief

    # Si un job précédent a ÉCHOUÉ, renvoyer l'erreur au lieu de relancer la
    # génération à chaque appel : sinon un échec persistant (mauvaise sortie LLM,
    # quota) boucle en regénération à chaque poll côté front. Le bouton
    # « régénérer » (POST force) reste la voie de reprise explicite.
    _bj = _BRIEF_GENERATION_JOBS.get(scenario_id, {})
    if _bj.get("status") == "error":
        return {"status": "error", "message": _bj.get("error", "Échec de la génération du brief.")}

    # Pas de brief en cache valide (absent, corpus changé, ou autre langue) :
    # déclencher la génération DANS LA LANGUE demandée.
    generate_evidence_brief(scenario_id, lang=lang)
    return {"status": "generating", "message": _msg(lang, "Génération en cours, réessayez dans 30 secondes.",
                                                    "Generating, try again in 30 seconds.")}
