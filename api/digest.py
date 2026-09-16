"""Corpus digest: what the WHOLE relevant subset of a scenario says, in a compact block.

House rule of this project: every evidence extraction reads ALL the relevant articles
(above the similarity threshold, or included by a reviewer), never a sample. A corpus of
several thousand abstracts cannot be pasted into one prompt, so the rule is honoured the
only way that scales, map then reduce:

  map     per article, once, cached on the article row: PICO (`pico_json`, extracted for
          every article by the pipeline and the background worker) and typed concepts
          (`concepts_json`);
  reduce  this module aggregates those per-article facts over the ENTIRE relevant subset,
          in SQL, with no LLM and no sampling.

A generator then writes its narrative over the digest, which covers every article, plus
the verbatim text of the best few for quotation. Before this, the brief spoke for 30
articles out of 2,732 while announcing the full count.

Everything here is deterministic: the same corpus gives the same digest.
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import text

from .core import engine, logger
from .scenario_store import _get_scenario_threshold

# Sous-ensemble PERTINENT : même porte que partout ailleurs (jamais les exclus ; inclus
# manuellement OU au-dessus du seuil). Une seule définition, reprise par chaque agrégat.
_RELEVANT = """
    FROM literature_document d
    JOIN article_scenarios ars ON ars.document_id = d.id
    WHERE ars.scenario_id = :sid
      AND d.is_duplicate IS NOT TRUE
      AND COALESCE(ars.screening_status, d.screening_status) IS DISTINCT FROM 'excluded'
      AND (COALESCE(ars.screening_status, d.screening_status) = 'included'
           OR COALESCE(ars.similarity_score, 0) >= :thr)
"""

# Combien de modalités on garde par distribution : de quoi décrire un corpus sans noyer
# le prompt. Les totaux, eux, portent TOUJOURS sur l'ensemble complet.
_TOP_N = 12
_TOP_CONCEPTS_PER_TYPE = 8


def _rows(conn, sql: str, sid: str, thr: float, **extra) -> list[dict]:
    return [dict(r) for r in conn.execute(text(sql), {"sid": sid, "thr": thr, **extra}).mappings().all()]


def corpus_digest(scenario_id: str, threshold: float | None = None) -> dict[str, Any]:
    """Portrait chiffré du corpus PERTINENT COMPLET : volumétrie, couverture, années,
    devis, pays, journaux, concepts typés, et les articles les mieux établis.

    Aucun échantillonnage : chaque compteur est calculé sur tous les articles pertinents.
    Les distributions sont tronquées aux modalités les plus fréquentes, mais leur total
    reste celui du corpus entier (champ `n_articles`)."""
    thr = _get_scenario_threshold(scenario_id) if threshold is None else float(threshold)
    out: dict[str, Any] = {"scenario_id": scenario_id, "threshold": thr}
    try:
        with engine.connect() as conn:
            head = _rows(conn, f"""
                SELECT COUNT(*) AS n_articles,
                       COUNT(*) FILTER (WHERE COALESCE(ars.screening_status, d.screening_status) = 'included') AS n_included,
                       COUNT(*) FILTER (WHERE d.pico_json IS NOT NULL) AS n_with_pico,
                       COUNT(*) FILTER (WHERE d.concepts_json IS NOT NULL) AS n_with_concepts,
                       COUNT(*) FILTER (WHERE d.has_fulltext IS TRUE) AS n_with_fulltext,
                       COUNT(*) FILTER (WHERE d.abstract IS NOT NULL) AS n_with_abstract,
                       MIN(d.year) AS year_min, MAX(d.year) AS year_max,
                       ROUND(AVG(d.quality_score)::numeric, 3) AS mean_quality,
                       SUM(COALESCE(d.citation_count, 0)) AS total_citations
                {_RELEVANT}
            """, scenario_id, thr)
            # AVG renvoie un Decimal : coercé en float pour rester sérialisable en JSON.
            _h = dict(head[0]) if head else {}
            if _h.get("mean_quality") is not None:
                _h["mean_quality"] = float(_h["mean_quality"])
            out.update(_h)

            out["by_year"] = _rows(conn, f"""
                SELECT d.year AS value, COUNT(*) AS n
                {_RELEVANT} AND d.year IS NOT NULL
                GROUP BY d.year ORDER BY d.year DESC LIMIT 20
            """, scenario_id, thr)

            # Devis d'étude : le PICO d'abord (extrait par article), sinon la colonne.
            out["by_design"] = _rows(conn, f"""
                SELECT LOWER(TRIM(COALESCE(NULLIF(d.pico_json->>'study_design', ''), d.study_design))) AS value,
                       COUNT(*) AS n
                {_RELEVANT}
                  AND COALESCE(NULLIF(d.pico_json->>'study_design', ''), d.study_design) IS NOT NULL
                GROUP BY 1 HAVING LOWER(TRIM(COALESCE(NULLIF(d.pico_json->>'study_design', ''), d.study_design)))
                                  NOT IN ('non précisé', 'not specified', 'unknown', 'n/a')
                ORDER BY n DESC LIMIT :top
            """, scenario_id, thr, top=_TOP_N)

            out["by_country"] = _rows(conn, f"""
                SELECT UPPER(TRIM(d.country)) AS value, COUNT(*) AS n
                {_RELEVANT} AND d.country IS NOT NULL AND LENGTH(TRIM(d.country)) = 2
                GROUP BY 1 ORDER BY n DESC LIMIT :top
            """, scenario_id, thr, top=_TOP_N)

            out["by_journal"] = _rows(conn, f"""
                SELECT TRIM(d.journal) AS value, COUNT(*) AS n
                {_RELEVANT} AND d.journal IS NOT NULL AND LENGTH(TRIM(d.journal)) > 1
                GROUP BY 1 ORDER BY n DESC LIMIT :top
            """, scenario_id, thr, top=_TOP_N)

            # Concepts typés (concepts_json, un par article, normalisés par le LLM une
            # seule fois) : la vue la plus fidèle de ce dont TOUT le corpus parle.
            try:
                concepts = _rows(conn, """
                    SELECT c->>'t' AS type, c->>'en' AS label, COUNT(*) AS n
                    FROM literature_document d
                    JOIN article_scenarios ars ON ars.document_id = d.id
                    CROSS JOIN LATERAL jsonb_array_elements(d.concepts_json->'concepts') AS c
                    WHERE ars.scenario_id = :sid
                      AND d.is_duplicate IS NOT TRUE
                      AND COALESCE(ars.screening_status, d.screening_status) IS DISTINCT FROM 'excluded'
                      AND (COALESCE(ars.screening_status, d.screening_status) = 'included'
                           OR COALESCE(ars.similarity_score, 0) >= :thr)
                      AND jsonb_typeof(d.concepts_json->'concepts') = 'array'
                      AND c->>'en' IS NOT NULL
                    GROUP BY 1, 2 ORDER BY n DESC
                """, scenario_id, thr)
            except Exception as _ce:                         # base sans concepts_json
                logger.info(f"corpus_digest concepts {scenario_id}: {_ce}")
                concepts = []
            by_type: dict[str, list] = {}
            for c in concepts:
                lst = by_type.setdefault(c["type"] or "topic", [])
                if len(lst) < _TOP_CONCEPTS_PER_TYPE:
                    lst.append({"label": c["label"], "n": c["n"]})
            out["concepts"] = by_type
    except Exception as e:                                   # noqa: BLE001 - jamais bloquant
        logger.warning(f"corpus_digest {scenario_id}: {e}")
        out.setdefault("n_articles", 0)
    out["complete"] = True                                   # calculé sur TOUT le corpus pertinent
    return out


def digest_to_prompt(digest: dict, max_chars: int = 2600) -> str:
    """Le digest en un bloc de texte compact pour un prompt. Dit d'emblée sur combien
    d'articles il porte, pour que le modèle ne parle jamais au nom d'un échantillon."""
    if not digest or not digest.get("n_articles"):
        return ""
    d = digest
    lines = [f"CORPUS COMPLET: {d['n_articles']} articles pertinents "
             f"(seuil {d.get('threshold')}), dont {d.get('n_included') or 0} inclus par un relecteur, "
             f"{d.get('n_with_pico') or 0} avec PICO extrait, {d.get('n_with_fulltext') or 0} avec texte integral."]
    if d.get("year_min") and d.get("year_max"):
        lines.append(f"Annees: {d['year_min']} a {d['year_max']}.")
    if d.get("mean_quality") is not None:
        lines.append(f"Qualite moyenne: {d['mean_quality']}; citations cumulees: {d.get('total_citations') or 0}.")

    def _dist(label, rows, fmt=lambda r: f"{r['value']} ({r['n']})"):
        if rows:
            lines.append(f"{label}: " + ", ".join(fmt(r) for r in rows if r.get("value")))

    _dist("Devis d'etude", d.get("by_design"))
    _dist("Pays", d.get("by_country"))
    _dist("Revues", (d.get("by_journal") or [])[:6])
    _dist("Annees recentes", (d.get("by_year") or [])[:8], lambda r: f"{r['value']}: {r['n']}")
    for t, items in (d.get("concepts") or {}).items():
        if items:
            lines.append(f"Concepts [{t}]: " + ", ".join(f"{i['label']} ({i['n']})" for i in items))
    block = "\n".join(lines)
    return block[:max_chars]


def digest_coverage_note(digest: dict, n_verbatim: int) -> str:
    """La phrase que chaque prompt porte : le digest couvre tout, le verbatim illustre."""
    n = (digest or {}).get("n_articles") or 0
    return (f"Les chiffres ci-dessus portent sur la TOTALITE des {n} articles pertinents. "
            f"Les {n_verbatim} articles reproduits ensuite en sont les mieux etablis "
            f"(qualite, citations) et servent a citer et a illustrer: tes conclusions "
            f"doivent rester coherentes avec les chiffres du corpus complet, jamais "
            f"tirees de ces seuls {n_verbatim} articles.")
