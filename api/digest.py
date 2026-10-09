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
from .study_design import raw_design_sql

#: Le devis brut d'un article, écrit une seule fois.
_raw_design_d = raw_design_sql("d")

from .scenario_store import (_get_scenario_threshold, relevant_gate_sql,
                             screening_status_sql)

# Sous-ensemble PERTINENT : même porte que partout ailleurs (jamais les exclus ; inclus
# manuellement OU au-dessus du seuil). Une seule définition, reprise par chaque agrégat.
_RELEVANT = f"""
    FROM literature_document d
    JOIN article_scenarios ars ON ars.document_id = d.id
    WHERE ars.scenario_id = :sid
      AND {relevant_gate_sql('d', 'ars', ':thr')}
"""

# Combien de modalités on garde par distribution : de quoi décrire un corpus sans noyer
# le prompt. Les totaux, eux, portent TOUJOURS sur l'ensemble complet.
_TOP_N = 12
_TOP_CONCEPTS_PER_TYPE = 8


def _rows(conn, sql: str, sid: str, thr: float, **extra) -> list[dict]:
    return [dict(r) for r in conn.execute(text(sql), {"sid": sid, "thr": thr, **extra}).mappings().all()]


# ─────────────────────────────────────────────────────────────────────────────
# Gap matrix: what the corpus has NOT studied
# ─────────────────────────────────────────────────────────────────────────────
# A commercial report we compared against prints a matrix of theme against dimension with
# a paper count in each cell and "Potential gap" where a cell is empty. It is the most
# useful figure in the document and the least trustworthy one: that report synthesises 50
# of its 216 eligible papers, so a cell reading "No papers" means none of the fifty, and
# an empty cell is as likely to be a sampling artefact as a gap in the literature.
#
# The same figure computed over the WHOLE relevant subset says something a sample cannot:
# that no article in this corpus pairs these two things. Which is why this lives here,
# beside the digest, in SQL, with no LLM: a gap asserted by a model is an opinion, and a
# gap counted over every relevant article is a finding about the corpus.
#
# The axes are two concept TYPES from `concepts_json` (intervention, outcome, pathogen,
# vector, population, ...), extracted once per article by the map step. Intervention
# against outcome is the PICO cross-tab a reviewer actually wants: it answers "which
# intervention has nobody measured against which outcome".

#: How many labels each axis keeps. The matrix is for reading, and a 40x40 grid is not
#: read. The COUNTS are over the full corpus; only the axes are truncated, and
#: `rows_shown` / `rows_total` say so.
_MATRIX_MAX_LABELS = 10


def concept_matrix(scenario_id: str, row_type: str, col_type: str,
                   threshold: float | None = None,
                   max_labels: int = _MATRIX_MAX_LABELS) -> dict[str, Any]:
    """Articles pairing each `row_type` concept with each `col_type` concept.

    Every relevant article is counted. An article is counted once per (row, col) pair it
    carries, and `DISTINCT` guards the case of an article carrying the same label twice.

    What an empty cell means, exactly, and what it does not: no article in this corpus
    carries both labels. It is not evidence that the pairing is unstudied in the
    literature, and it cannot be, because the corpus is one search over a few sources. It
    also cannot see articles whose concepts were never extracted, which is why
    `coverage` reports how many relevant articles the matrix could read at all. A gap
    figure that hides its own denominator is the thing this is meant to replace."""
    thr = _get_scenario_threshold(scenario_id) if threshold is None else float(threshold)
    out: dict[str, Any] = {"scenario_id": scenario_id, "threshold": thr,
                           "row_type": row_type, "col_type": col_type}
    try:
        with engine.connect() as conn:
            cov = _rows(conn, f"""
                SELECT COUNT(*) AS relevant,
                       COUNT(*) FILTER (
                           WHERE jsonb_typeof(d.concepts_json->'concepts') = 'array') AS with_concepts
                {_RELEVANT}
            """, scenario_id, thr)
            out["coverage"] = dict(cov[0]) if cov else {"relevant": 0, "with_concepts": 0}

            # Les types disponibles, pour que l'appelant puisse choisir ses axes sans les
            # deviner : un corpus sans « vector » ne doit pas proposer cet axe.
            # La porte `_RELEVANT` n'est pas réutilisable ici : elle commence par son
            # propre FROM et le CROSS JOIN LATERAL doit s'insérer avant le WHERE. La
            # condition est donc répétée, à l'identique, plutôt que fabriquée par
            # bricolage de chaîne.
            out["available_types"] = _rows(conn, f"""
                SELECT c->>'t' AS value, COUNT(DISTINCT d.id) AS n
                FROM literature_document d
                JOIN article_scenarios ars ON ars.document_id = d.id
                CROSS JOIN LATERAL jsonb_array_elements(d.concepts_json->'concepts') AS c
                WHERE ars.scenario_id = :sid
                  AND {relevant_gate_sql('d', 'ars', ':thr')}
                  AND jsonb_typeof(d.concepts_json->'concepts') = 'array'
                  AND c->>'t' IS NOT NULL
                GROUP BY 1 ORDER BY n DESC
            """, scenario_id, thr)

            pairs = _rows(conn, f"""
                SELECT r->>'en' AS row_label, c->>'en' AS col_label,
                       COUNT(DISTINCT d.id) AS n
                FROM literature_document d
                JOIN article_scenarios ars ON ars.document_id = d.id
                CROSS JOIN LATERAL jsonb_array_elements(d.concepts_json->'concepts') AS r
                CROSS JOIN LATERAL jsonb_array_elements(d.concepts_json->'concepts') AS c
                WHERE ars.scenario_id = :sid
                  AND {relevant_gate_sql('d', 'ars', ':thr')}
                  AND jsonb_typeof(d.concepts_json->'concepts') = 'array'
                  AND r->>'t' = :row_type AND c->>'t' = :col_type
                  AND r->>'en' IS NOT NULL AND c->>'en' IS NOT NULL
                GROUP BY 1, 2
            """, scenario_id, thr, row_type=row_type, col_type=col_type)
        out.update(build_matrix(pairs, max_labels=max_labels))
        out["complete"] = True
    except Exception as e:                                   # noqa: BLE001 - jamais bloquant
        logger.warning(f"concept_matrix {scenario_id} {row_type}x{col_type}: {e}")
        out.update({"complete": False, "error": str(e)[:300], "rows": [], "cols": [],
                    "cells": [], "gaps": []})
    return out


def build_matrix(pairs, max_labels: int = _MATRIX_MAX_LABELS) -> dict[str, Any]:
    """The grid, from (row_label, col_label, n) triples. Pure, so it is testable.

    Axes are ordered by how many articles carry the label, descending, and truncated to
    `max_labels`. `rows_total` keeps the untruncated count, because an axis silently cut
    to its ten biggest labels would make the matrix look more complete than it is.

    `gaps` lists the empty cells of the SHOWN grid, which is the only region where an
    empty cell is informative: outside it, a zero may simply be a label that was cut."""
    row_totals: dict[str, int] = {}
    col_totals: dict[str, int] = {}
    counts: dict[tuple[str, str], int] = {}
    for p in pairs:
        row, col, n = p["row_label"], p["col_label"], int(p["n"] or 0)
        if not row or not col:
            continue
        counts[(row, col)] = counts.get((row, col), 0) + n
        row_totals[row] = row_totals.get(row, 0) + n
        col_totals[col] = col_totals.get(col, 0) + n

    def _axis(totals):
        return [label for label, _ in sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))]

    all_rows, all_cols = _axis(row_totals), _axis(col_totals)
    rows, cols = all_rows[:max_labels], all_cols[:max_labels]
    cells = [{"row": r, "col": c, "n": counts.get((r, c), 0)} for r in rows for c in cols]
    return {
        "rows": [{"label": r, "n": row_totals[r]} for r in rows],
        "cols": [{"label": c, "n": col_totals[c]} for c in cols],
        "rows_total": len(all_rows), "cols_total": len(all_cols),
        "cells": cells,
        "gaps": [{"row": c["row"], "col": c["col"]} for c in cells if c["n"] == 0],
        "pairs_observed": len([c for c in cells if c["n"] > 0]),
        "cells_shown": len(cells),
    }


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
                       COUNT(*) FILTER (WHERE {screening_status_sql('d', 'ars')} = 'included') AS n_included,
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

            # Devis d'étude : l'expression COMMUNE (cf. api/study_design.raw_design_sql),
            # qui traite les marqueurs d'absence comme une absence des deux côtés.
            out["by_design"] = _rows(conn, f"""
                SELECT LOWER({_raw_design_d}) AS value, COUNT(*) AS n
                {_RELEVANT}
                GROUP BY 1 HAVING LOWER({_raw_design_d}) <> ''
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
                concepts = _rows(conn, f"""
                    SELECT c->>'t' AS type, c->>'en' AS label, COUNT(*) AS n
                    FROM literature_document d
                    JOIN article_scenarios ars ON ars.document_id = d.id
                    CROSS JOIN LATERAL jsonb_array_elements(d.concepts_json->'concepts') AS c
                    WHERE ars.scenario_id = :sid
                      AND {relevant_gate_sql('d', 'ars', ':thr')}
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
        out["complete"] = True                               # calculé sur TOUT le corpus pertinent
    except Exception as e:                                   # noqa: BLE001 - jamais bloquant
        # `complete` était posé HORS du try : une agrégation en échec renvoyait
        # {"n_articles": 0, "complete": True}, les générateurs retombaient en silence sur
        # leurs 20 à 30 articles reproduits, et la note de couverture annonçait « la
        # TOTALITE des 0 articles ». Le seul mode de panne qui viole la règle de la maison
        # se déclarait conforme.
        logger.warning(f"corpus_digest {scenario_id}: {e}")
        out.setdefault("n_articles", 0)
        out["complete"] = False
        out["error"] = str(e)[:300]
    return out


def digest_to_prompt(digest: dict, max_chars: int = 2600) -> str:
    """Le digest en un bloc de texte compact pour un prompt. Dit d'emblée sur combien
    d'articles il porte, pour que le modèle ne parle jamais au nom d'un échantillon."""
    # Un digest incomplet ne doit RIEN affirmer : mieux vaut un bloc vide (le générateur
    # reprend alors son repli explicite) qu'un total faux présenté comme exhaustif.
    if not digest or not digest.get("n_articles") or not digest.get("complete"):
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
    """La phrase que chaque prompt porte : le digest couvre tout, le verbatim illustre.

    Si le digest est INCOMPLET, la phrase s'inverse au lieu de disparaître. Elle
    affirmait « la TOTALITE des 0 articles » au-dessus d'un bloc de chiffres que
    `digest_to_prompt` venait justement de supprimer : le modèle lisait une garantie
    d'exhaustivité portant sur rien, alors qu'il n'avait sous les yeux que les articles
    reproduits, c'est-à-dire l'échantillon que la règle de la maison interdit. Dans ce
    cas il faut le lui DIRE, et lui interdire de généraliser, plutôt que se taire."""
    if not digest or not digest.get("complete") or not digest.get("n_articles"):
        return (f"ATTENTION : l'agregation du corpus complet a echoue. Tu ne disposes que "
                f"des {n_verbatim} articles reproduits ci-dessous. N'enonce AUCUN total, "
                f"AUCUNE proportion et AUCUNE tendance d'ensemble : limite-toi a ce que "
                f"ces {n_verbatim} articles etablissent, et dis explicitement que la vue "
                f"d'ensemble du corpus n'etait pas disponible.")
    n = digest.get("n_articles") or 0
    return (f"Les chiffres ci-dessus portent sur la TOTALITE des {n} articles pertinents. "
            f"Les {n_verbatim} articles reproduits ensuite en sont les mieux etablis "
            f"(qualite, citations) et servent a citer et a illustrer: tes conclusions "
            f"doivent rester coherentes avec les chiffres du corpus complet, jamais "
            f"tirees de ces seuls {n_verbatim} articles.")
