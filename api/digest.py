"""Corpus digest: what the WHOLE relevant subset of a scenario says, in a compact block.

House rule of this project: every evidence extraction reads ALL the relevant articles
(above the similarity threshold, or included by a reviewer), never a sample. A corpus of
several thousand abstracts cannot be pasted into one prompt, so the rule is honoured the
only way that scales, map then reduce:

  map     per article, once, cached on the article row: PICO (`pico_json`, extracted for
          every article by the pipeline and the background worker, with the article's
          main finding and its direction, see api/pico_map) and typed concepts
          (`concepts_json`);
  reduce  this module aggregates those per-article facts over the ENTIRE relevant subset,
          in SQL, with no LLM and no sampling.

A generator then writes its narrative over the digest, which covers every article, plus
the verbatim text of a few for quotation: the closest to the question, in the corpus's
relevance order. Before this, the brief spoke for 30 articles out of 2,732 while
announcing the full count. And until the findings were counted here, whatever it said
about RESULTS still came from those 30: the digest counted designs and concepts, never
what the studies found.

Everything here is deterministic: the same corpus gives the same digest.
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import text

from .core import engine, logger
from .pico_map import FINDING_DIRECTIONS
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
#: Outcomes dont les directions de résultat sont détaillées. Les totaux par direction,
#: eux, portent sur tous les articles pertinents, et le nombre d'outcomes est dit.
_TOP_FINDING_OUTCOMES = 8

#: Taille par défaut d'un bloc de prompt. 2 600 coupait le digest au milieu d'une ligne,
#: si bien que les derniers types de concepts n'atteignaient jamais le modèle.
PROMPT_BLOCK_CHARS = 6000


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


def _findings(conn, scenario_id: str, thr: float) -> dict[str, Any]:
    """Ce que les études ont TROUVÉ, compté sur TOUS les articles pertinents.

    La direction du résultat principal de chaque article (`pico_json.finding_direction`,
    lue une fois par article, cf. api/pico_map) : au total, puis pour les outcomes les plus
    fréquents. Les outcomes sont les concepts typés « outcome » de `concepts_json` quand le
    corpus en porte ; sinon le texte O du PICO, ramené à un libellé normalisé (minuscules,
    première proposition, sans ponctuation). `n_read` dit combien d'articles ont été lus
    pour leur résultat : les autres ne disent rien, ni dans un sens ni dans l'autre."""
    dirs = ",\n".join(f"COUNT(*) FILTER (WHERE d.pico_json->>'finding_direction' = '{k}') AS f_{k}"
                      for k in FINDING_DIRECTIONS)
    head = _rows(conn, f"""
        SELECT COUNT(*) FILTER (WHERE d.pico_json->>'finding_direction' IS NOT NULL) AS n_read,
               COUNT(*) FILTER (WHERE NULLIF(TRIM(d.pico_json->>'key_finding'), '') IS NOT NULL)
                   AS n_with_finding,
               {dirs}
        {_RELEVANT}
    """, scenario_id, thr)[0]
    out: dict[str, Any] = {
        "n_read": int(head["n_read"] or 0),
        "n_with_finding": int(head["n_with_finding"] or 0),
        "by_direction": {k: int(head[f"f_{k}"] or 0) for k in FINDING_DIRECTIONS},
    }

    def _by_label(rows: list[dict]) -> list[dict]:
        return [{"label": r["label"], "n": int(r["n"]), "n_read": int(r["n_read"] or 0),
                 "by_direction": {k: int(r[f"f_{k}"] or 0) for k in FINDING_DIRECTIONS}}
                for r in rows]

    # Un article compte UNE fois par libellé (DISTINCT), même s'il porte le concept deux fois.
    per_concept = ",\n".join(
        f"COUNT(DISTINCT d.id) FILTER (WHERE d.pico_json->>'finding_direction' = '{k}') AS f_{k}"
        for k in FINDING_DIRECTIONS)
    try:
        with conn.begin_nested():                            # base sans concepts_json : repli
            rows = _rows(conn, f"""
                SELECT c->>'en' AS label, COUNT(DISTINCT d.id) AS n,
                       COUNT(DISTINCT d.id) FILTER (
                           WHERE d.pico_json->>'finding_direction' IS NOT NULL) AS n_read,
                       {per_concept},
                       COUNT(*) OVER () AS n_labels
                FROM literature_document d
                JOIN article_scenarios ars ON ars.document_id = d.id
                CROSS JOIN LATERAL jsonb_array_elements(d.concepts_json->'concepts') AS c
                WHERE ars.scenario_id = :sid
                  AND {relevant_gate_sql('d', 'ars', ':thr')}
                  AND jsonb_typeof(d.concepts_json->'concepts') = 'array'
                  AND c->>'t' = 'outcome' AND NULLIF(TRIM(c->>'en'), '') IS NOT NULL
                GROUP BY 1 ORDER BY n DESC, 1 LIMIT :top
            """, scenario_id, thr, top=_TOP_FINDING_OUTCOMES)
    except Exception as _ce:                                 # noqa: BLE001
        logger.info(f"corpus_digest outcome concepts {scenario_id}: {_ce}")
        rows = []
    out["outcome_basis"] = "concepts"
    if not rows:
        # Aucun concept « outcome » : le texte O du PICO, à un libellé près.
        label = ("NULLIF(LEFT(TRIM(regexp_replace(regexp_replace(lower(split_part(split_part("
                 "split_part(COALESCE(NULLIF(TRIM(d.pico_json->>'O'), ''), d.pico_json->>'outcome', ''),"
                 " ';', 1), ',', 1), '(', 1)), '[^[:alnum:][:space:]]+', ' ', 'g'),"
                 " '[[:space:]]+', ' ', 'g')), 80), '')")
        per_label = ",\n".join(f"COUNT(*) FILTER (WHERE o.dir = '{k}') AS f_{k}"
                               for k in FINDING_DIRECTIONS)
        rows = _rows(conn, f"""
            SELECT o.label, COUNT(*) AS n, COUNT(*) FILTER (WHERE o.dir IS NOT NULL) AS n_read,
                   {per_label},
                   COUNT(*) OVER () AS n_labels
            FROM (SELECT d.pico_json->>'finding_direction' AS dir, {label} AS label
                  {_RELEVANT}) o
            WHERE o.label IS NOT NULL
            GROUP BY 1 ORDER BY n DESC, 1 LIMIT :top
        """, scenario_id, thr, top=_TOP_FINDING_OUTCOMES)
        out["outcome_basis"] = "pico_outcome"
    out["by_outcome"] = _by_label(rows)
    out["n_outcome_labels"] = int(rows[0]["n_labels"]) if rows else 0
    return out


def corpus_digest(scenario_id: str, threshold: float | None = None) -> dict[str, Any]:
    """Portrait chiffré du corpus PERTINENT COMPLET : volumétrie, couverture, années,
    devis, pays, journaux, concepts typés, et ce que les études ont trouvé (direction du
    résultat principal de chaque article, au total et par outcome).

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
                       -- Une moyenne sur les seuls articles NOTÉS, et leur nombre. La
                       -- colonne a longtemps porté un défaut 0.0 : 0 y veut dire « jamais
                       -- calculé » (même convention que api/corpus.py). Sur HPAI_last, 861
                       -- zéros sur 1 000 faisaient une « qualité moyenne 0.08 » que le brief
                       -- recopiait en conclusion.
                       ROUND((AVG(d.quality_score) FILTER (WHERE d.quality_score > 0))::numeric, 3)
                           AS mean_quality,
                       COUNT(*) FILTER (WHERE d.quality_score > 0) AS n_with_quality,
                       -- Un nombre de citations absent est INCONNU, pas nul : la somme ne
                       -- porte que sur les articles qui en ont un, et leur nombre est dit.
                       SUM(d.citation_count) AS total_citations,
                       COUNT(d.citation_count) AS n_with_citations
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
            labels_per_type: dict[str, int] = {}
            for c in concepts:
                lst = by_type.setdefault(c["type"] or "topic", [])
                labels_per_type[c["type"] or "topic"] = labels_per_type.get(c["type"] or "topic", 0) + 1
                if len(lst) < _TOP_CONCEPTS_PER_TYPE:
                    lst.append({"label": c["label"], "n": c["n"]})
            out["concepts"] = by_type
            # Combien de libellés chaque type porte en tout : le prompt dit « (+N more) »
            # au lieu de laisser croire que les huit premiers sont tout le corpus.
            out["concepts_total"] = labels_per_type

        # Les résultats ont leur propre connexion : une requête de concepts en échec (base
        # sans concepts_json) a annulé la transaction de la première. Un décompte en échec
        # est DIT dans le prompt, jamais rendu comme « aucun résultat ».
        try:
            with engine.connect() as fconn:
                out["findings"] = _findings(fconn, scenario_id, thr)
        except Exception as _fe:                             # noqa: BLE001 - jamais bloquant
            logger.warning(f"corpus_digest findings {scenario_id}: {_fe}")
            out["findings"] = {"error": str(_fe)[:300]}
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


def _quality_and_citation_lines(d: dict) -> list[str]:
    """Le score de qualité et les citations, chacun avec ce qu'il couvre.

    La ligne d'avant, « Qualite moyenne: 0.08; citations cumulees: 0 », donnait au modèle
    deux chiffres sans dénominateur ni définition, et il en tirait une conclusion : « la
    qualité méthodologique d'ensemble est faible, avec un score moyen de 0.08 et zéro
    citation ». Le 0.08 était une moyenne sur 861 articles jamais notés ; le zéro, une
    somme de nombres de citations que personne n'avait collectés. Ici, un chiffre vient
    avec sa définition et le nombre d'articles qu'il couvre, et une donnée absente est
    dite absente."""
    n = d.get("n_articles") or 0
    lines: list[str] = []
    nq = d.get("n_with_quality") or 0
    if d.get("mean_quality") is not None and nq:
        lines.append(f"Score de qualite composite (0 a 1; devis 50 %, effectif 18 %, citations "
                     f"12 %, recence 12 %, acces ouvert 8 %; pas une evaluation GRADE): moyenne "
                     f"{d['mean_quality']} sur {nq} articles notes / {n}.")
    else:
        lines.append("Score de qualite composite: non calcule pour ce corpus.")
    nc = d.get("n_with_citations") or 0
    if nc:
        lines.append(f"Citations: {d.get('total_citations') or 0} au total sur {nc} articles "
                     f"dont le nombre est connu / {n}.")
    else:
        lines.append("Citations: inconnues pour tous les articles (non collectees), pas nulles.")
    return lines


def _more_marker(labels: list[str]) -> str:
    shown = ", ".join(labels[:6]) + (", ..." if len(labels) > 6 else "")
    return f"(+{len(labels)} more, left out for length: {shown})"


def fit_prompt_sections(sections: list[tuple[str, str]], max_chars: int = PROMPT_BLOCK_CHARS,
                        mandatory: int = 1) -> str:
    """Les sections d'un bloc de prompt, ENTIÈRES, dans l'ordre où on les donne.

    Les blocs étaient coupés à `max_chars` caractères au milieu d'une ligne : la fin d'une
    liste, puis des sections entières disparaissaient sans un mot, et avec elles la
    dernière ligne, qui portait l'instruction de citer le dénominateur. Ici, les
    `mandatory` premières sections passent toujours (l'en-tête et les instructions, que
    l'appelant place en tête pour cela) ; les suivantes passent entières tant qu'elles
    tiennent, et celles qui ne tiennent pas sont nommées sur une dernière ligne
    « (+N more ...) ». `sections` : des paires (nom, texte), le texte vide est ignoré. PUR."""
    sections = [(label, t) for label, t in sections if t]
    if sum(len(t) + 1 for _, t in sections) - 1 <= max_chars:
        return "\n".join(t for _, t in sections)
    kept = [i for i in range(min(mandatory, len(sections)))]
    used = sum(len(sections[i][1]) + 1 for i in kept)
    for i in range(len(kept), len(sections)):
        if used + len(sections[i][1]) + 1 <= max_chars:
            kept.append(i)
            used += len(sections[i][1]) + 1

    def _marker() -> str:
        return _more_marker([sections[i][0] for i in range(len(sections)) if i not in kept])

    # La ligne « (+N more) » doit tenir aussi : on retire, au besoin, les dernières
    # sections facultatives retenues.
    while len(kept) > mandatory and used + len(_marker()) > max_chars:
        used -= len(sections[kept.pop()][1]) + 1
    return "\n".join([sections[i][1] for i in kept] + [_marker()])


#: Les directions de résultat, telles que le prompt les nomme.
_DIRECTION_LABELS = {"increase": "hausse", "decrease": "baisse", "no_difference": "sans difference",
                     "mixed": "mixte", "descriptive": "descriptif", "none": "aucun resultat"}


def _findings_sections(d: dict) -> list[tuple[str, str]]:
    """Ce que les études ont trouvé, en une ou deux lignes, sur TOUS les articles pertinents.

    Rien si le digest ne porte pas ce décompte (forme ancienne) ; une ligne qui le DIT s'il
    a échoué ou si aucun article n'a encore été lu pour son résultat, pour que le modèle
    n'en déduise pas « aucun résultat » ni ne parte des seuls articles reproduits."""
    f = d.get("findings")
    if not isinstance(f, dict):
        return []
    n = d.get("n_articles") or 0
    if f.get("error"):
        return [("Resultats", "RESULTATS: le decompte des resultats sur le corpus complet a echoue: "
                              "n'enonce aucune tendance des resultats du corpus.")]
    read = int(f.get("n_read") or 0)
    if not read:
        return [("Resultats", f"RESULTATS: aucun des {n} articles pertinents n'a encore ete lu pour "
                              f"son resultat principal: n'enonce aucune tendance des resultats du "
                              f"corpus, et ne la deduis pas des seuls articles reproduits.")]
    by_dir = f.get("by_direction") or {}
    line = (f"RESULTATS sur les {n} articles pertinents (resultat principal de chaque article, lu "
            f"une fois et compte ici sur tous): {read} lus, dont {f.get('n_with_finding') or 0} "
            f"rapportent un resultat. Direction: "
            + ", ".join(f"{_DIRECTION_LABELS[k]} {by_dir.get(k, 0)}" for k in FINDING_DIRECTIONS) + ".")
    if read < n:
        line += f" {n - read} articles ne sont pas encore lus pour leur resultat: ne conclus rien sur eux."
    out = [("Resultats", line)]
    rows = f.get("by_outcome") or []
    if rows:
        basis = ("concepts d'outcome" if f.get("outcome_basis") == "concepts"
                 else "outcome du PICO, libelle normalise")

        def _one(r: dict) -> str:
            parts = [f"{_DIRECTION_LABELS[k]} {r['by_direction'][k]}" for k in FINDING_DIRECTIONS
                     if r["by_direction"].get(k)]
            if r["n"] > r["n_read"]:
                parts.append(f"non lus {r['n'] - r['n_read']}")
            return f"{r['label']} ({r['n']} articles: {', '.join(parts)})"

        more = int(f.get("n_outcome_labels") or len(rows)) - len(rows)
        out.append(("Resultats par outcome",
                    f"Directions par outcome ({basis}, les {len(rows)} plus frequents): "
                    + "; ".join(_one(r) for r in rows)
                    + (f" (+{more} more)" if more > 0 else "") + "."))
    return out


def digest_to_prompt(digest: dict, max_chars: int = PROMPT_BLOCK_CHARS) -> str:
    """Le digest en un bloc de texte compact pour un prompt. Dit d'emblée sur combien
    d'articles il porte, pour que le modèle ne parle jamais au nom d'un échantillon.

    Coupé par sections ENTIÈRES (`fit_prompt_sections`), jamais au milieu d'une ligne, dans
    l'ordre de ce qui compte : l'en-tête, ce que les études ont trouvé, puis les
    distributions, les concepts, et en dernier les revues et les années récentes."""
    # Un digest incomplet ne doit RIEN affirmer : mieux vaut un bloc vide (le générateur
    # reprend alors son repli explicite) qu'un total faux présenté comme exhaustif.
    if not digest or not digest.get("n_articles") or not digest.get("complete"):
        return ""
    d = digest
    sections: list[tuple[str, str]] = [(
        "Corpus",
        f"CORPUS COMPLET: {d['n_articles']} articles pertinents "
        f"(seuil {d.get('threshold')}), dont {d.get('n_included') or 0} inclus par un relecteur, "
        f"{d.get('n_with_pico') or 0} avec PICO extrait, {d.get('n_with_fulltext') or 0} avec texte integral.")]
    sections += _findings_sections(d)
    if d.get("year_min") and d.get("year_max"):
        sections.append(("Annees", f"Annees: {d['year_min']} a {d['year_max']}."))
    sections.append(("Qualite et citations", "\n".join(_quality_and_citation_lines(d))))

    def _dist(label, rows, fmt=lambda r: f"{r['value']} ({r['n']})"):
        if rows:
            sections.append((label, f"{label}: " + ", ".join(fmt(r) for r in rows if r.get("value"))))

    _dist("Devis d'etude", d.get("by_design"))
    _dist("Pays", d.get("by_country"))
    totals = d.get("concepts_total") or {}
    for t, items in (d.get("concepts") or {}).items():
        if items:
            more = int(totals.get(t) or len(items)) - len(items)
            sections.append((f"Concepts [{t}]", f"Concepts [{t}]: "
                             + ", ".join(f"{i['label']} ({i['n']})" for i in items)
                             + (f" (+{more} more)" if more > 0 else "")))
    _dist("Revues", (d.get("by_journal") or [])[:6])
    _dist("Annees recentes", (d.get("by_year") or [])[:8], lambda r: f"{r['value']}: {r['n']}")
    return fit_prompt_sections(sections, max_chars, mandatory=1)


#: Comment les articles reproduits sont choisis, dit tel que le code le fait. Le prompt les
#: appelait « les mieux établis (qualité, citations) » alors qu'ils sont pris dans l'ordre
#: de pertinence du corpus (`relevance_order_sql`) : le modèle pouvait leur prêter une
#: autorité qu'ils n'ont pas, et s'y appuyer plutôt que sur le corpus complet.
VERBATIM_BY_RELEVANCE = ("les plus proches de la question du scenario (ordre de pertinence du "
                         "corpus: score de rerank, puis similarite semantique)")
#: Pour l'assistant : des extraits pris par proximité avec la question posée.
VERBATIM_BY_QUESTION = ("les plus proches de la question posee (similarite semantique, un "
                        "extrait par article)")


def digest_coverage_note(digest: dict, n_verbatim: int,
                         chosen_by: str = VERBATIM_BY_RELEVANCE) -> str:
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
            f"Les {n_verbatim} articles reproduits ensuite en sont {chosen_by}; ils ne sont "
            f"choisis ni pour leur qualite ni pour leurs citations, et ne forment pas un "
            f"echantillon representatif. Ils servent a citer et a illustrer: tes conclusions, y "
            f"compris sur les resultats, doivent rester coherentes avec les chiffres du corpus "
            f"complet, jamais tirees de ces seuls {n_verbatim} articles.")
