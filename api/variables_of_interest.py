"""The register of variables of interest: one list of what a review asks of every paper.

WHY IT EXISTS. The same kind of fact about a paper is produced in five places and shown in
five others, each with its own name and its own denominator. The year and the journal sit on
the article row; the study design and the evidence ceiling are folded from two extraction
passes (`api/study_design`); the population, the exposure, the outcome and the main finding
come from the PICO pass; the country and the NUTS region are resolved at read time from the
place text (`api/geography`); the theme comes from the clustering projection; the concepts
from the concept pass; the per-covariate rows from the extraction, named by the codebook. A
reviewer who asks "what do we actually know about these 819 papers, and how many of them
answered each question" has no screen that says so, and only five of these can narrow a
corpus, for no reason other than that nobody wrote the button for the others.

WHAT THIS MODULE IS. One read-only assembly, over the WHOLE relevant subset, of every
variable the review holds: the standard ones below, which cost nothing because their answer
is already in the database or already computed, and the scenario's own, which are the groups
of its codebook. Per variable: who fills it, what type it is, how many papers carry a value,
the denominator that count is out of, and the values with their counts.

WHAT IT IS NOT. It computes nothing new, it calls no model, it writes nothing, and it reads
no cache of its own. Each family is counted in its own try/except and a family that cannot be
read becomes a blank row with its reason, never an error page: this is a panel, and a panel
that fails must still show the rest.

THE DENOMINATOR IS NEVER IMPLIED. A coverage count is meaningless without the set it is out
of, and the three sets differ: `relevant` (the gate: not a duplicate, not excluded, included
by a reviewer or above the threshold with an abstract), `extracted` (the relevant papers the
extraction has read, which is fewer), `corpus` (every paper the search brought back). Each
variable says which one it used.

House rule (CLAUDE.md): every count here is over ALL the relevant articles, in SQL, with no
sampling. The value lists are capped for display and say how many values they stand for; the
counts behind them are not.
"""
from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Query
from sqlalchemy import text

from .codebook import get_codebook, get_index, normalise, sheet_key
from .core import _norm_lang, app, engine, logger
from .digest import _rows
from .extraction import COVERAGE_KEYS, _OBS_FROM
from .geography import geography
from .pico_map import FINDING_DIRECTIONS
from .pooling import display_names, display_path
from .scenario_store import (_get_scenario_threshold, _get_user_scenario_or_404,
                             relevant_gate_sql, screening_status_sql)
from .study_design import design_label, level_label
from .subsets import _cluster_membership, _design_membership

# ─────────────────────────────────────────────────────────────────────────────
# The vocabulary of the register itself
# ─────────────────────────────────────────────────────────────────────────────
#: The families, in reading order. A family is a reason to look, not a storage location.
FAMILIES = ("record", "decision", "design", "place", "topic", "measure", "own")

#: What a variable costs to have. `free`: the answer is already a column or is computed from
#: one. `already_run`: it comes from a pass the pipeline runs on every relevant paper anyway.
#: `one_pass`: asking it of a paper that was never asked costs one reading of that paper.
COST_FREE, COST_ALREADY_RUN, COST_ONE_PASS = "free", "already_run", "one_pass"

#: The five types a variable may have. Deliberately five: no formulas, no nested objects.
CAT, NUM, BOOL, DATE, TEXT = "category", "number", "boolean", "date", "text"

#: Which set a count is out of. Printed next to every count, never implied.
OF_RELEVANT, OF_EXTRACTED, OF_CORPUS = "relevant", "extracted", "corpus"

#: How many values of a variable are listed. The COUNTS are over everything; only the list
#: is cut, and `n_values` says how many values it stands for.
_MAX_VALUES = 10
#: How many of the scenario's own variables are returned. A codebook may hold thousands of
#: nodes; the groups are what this page lists, and there are never many of them.
_MAX_OWN = 120


def _v(key: str, family: str, source: str, cost: str, vtype: str, shown: tuple[str, ...],
       of: str = OF_RELEVANT, filterable: bool = True, crossable: bool = True,
       covers: tuple[str, ...] = ()) -> dict[str, Any]:
    """One row of the standard register. `shown` names the panels that show this fact today,
    so the pass that removes a duplicate knows where to look. `covers` names the keys a row
    stands for when several variables share a line (the three NUTS levels, the study dates):
    the count is then the number of papers that carry at least one of them."""
    return {"key": key, "family": family, "source": source, "cost": cost, "type": vtype,
            "shown_in": list(shown), "denominator_of": of,
            "filterable": filterable, "crossable": crossable,
            "covers": list(covers) or None}


#: The standard register: present in every scenario, nothing to define, nothing to upload.
#: Fifteen of these cost nothing at all; the rest come from passes that already run.
STANDARD: tuple[dict[str, Any], ...] = (
    # ── A. The record ───────────────────────────────────────────────────────
    _v("year", "record", "metadata", COST_FREE, NUM, ("corpus",)),
    _v("journal", "record", "metadata", COST_FREE, CAT, ("corpus",)),
    _v("language", "record", "metadata", COST_FREE, CAT, ("corpus",)),
    _v("publication_type", "record", "metadata", COST_FREE, CAT, ("corpus",)),
    _v("source_db", "record", "search", COST_FREE, CAT, ("prisma",)),
    _v("open_access", "record", "metadata", COST_FREE, BOOL, ("corpus",)),
    _v("has_fulltext", "record", "fulltext", COST_FREE, BOOL, ("enrichment",)),
    _v("citations", "record", "metadata", COST_FREE, NUM, ("corpus",)),
    _v("has_doi", "record", "metadata", COST_FREE, BOOL, ("corpus",), crossable=False),
    # ── B. The review's own decisions ───────────────────────────────────────
    _v("screening", "decision", "reviewers", COST_FREE, CAT, ("review", "prisma")),
    _v("above_threshold", "decision", "scoring", COST_FREE, BOOL, ("seuil",),
       of=OF_CORPUS, crossable=False),
    _v("scope", "decision", "selection", COST_FREE, CAT, ("subset", "prisma"), crossable=False),
    _v("kappa", "decision", "double_blind", COST_FREE, CAT, ("doubleBlind",)),
    # Le MOTIF d'une décision, pas un marqueur d'import : une décision importée d'une revue
    # précédente arrive dans `screening_status` avec son motif, sans drapeau à part. Le motif
    # est donc la variable, et le PRISMA le regroupe déjà.
    _v("screening_reason", "decision", "reviewers", COST_FREE, CAT, ("review", "prisma")),
    # ── C. Design and evidence ──────────────────────────────────────────────
    _v("design", "design", "pico_pass", COST_ALREADY_RUN, CAT, ("corpus", "pico")),
    _v("grade_level", "design", "derived", COST_FREE, CAT, ("corpus",)),
    _v("pico", "design", "pico_pass", COST_ALREADY_RUN, TEXT, ("pico",),
       filterable=False, crossable=False,
       covers=("population", "intervention", "comparator", "outcome")),
    _v("key_finding", "design", "pico_pass", COST_ALREADY_RUN, TEXT, ("pico", "evidence"),
       filterable=False, crossable=False),
    _v("finding_direction", "design", "pico_pass", COST_ALREADY_RUN, CAT, ("evidence",)),
    # ── D. Place and time ───────────────────────────────────────────────────
    _v("country", "place", "place_resolver", COST_FREE, CAT, ("geography",), of=OF_EXTRACTED),
    _v("europe", "place", "place_resolver", COST_FREE, BOOL, ("geography", "extraction"),
       of=OF_EXTRACTED),
    _v("nuts", "place", "nuts_list", COST_FREE, CAT, ("geography",), of=OF_EXTRACTED,
       covers=("nuts1", "nuts2", "nuts3")),
    _v("study_period", "place", "extraction", COST_ALREADY_RUN, DATE, ("extraction",),
       of=OF_EXTRACTED, crossable=False, covers=("study_start", "study_end")),
    # ── E. Topic ────────────────────────────────────────────────────────────
    _v("cluster", "topic", "clustering", COST_ALREADY_RUN, CAT, ("clustering",)),
    _v("concepts", "topic", "concept_pass", COST_ALREADY_RUN, CAT, ("knowledgeGraph",)),
    _v("disease", "topic", "disease_folder", COST_FREE, CAT, ("extraction", "pooled"),
       of=OF_EXTRACTED),
    # ── F. What was read, and the quantities ────────────────────────────────
    _v("read_from", "measure", "extraction", COST_ALREADY_RUN, CAT, ("extraction",),
       of=OF_EXTRACTED, crossable=False),
    _v("counts", "measure", "extraction", COST_ALREADY_RUN, NUM, ("extraction", "pooled"),
       of=OF_EXTRACTED, covers=("risk_pop", "positive", "percent_positive")),
    _v("math_model", "measure", "extraction", COST_ALREADY_RUN, BOOL, ("extraction",),
       of=OF_EXTRACTED),
    _v("epi_params", "measure", "parameter_pass", COST_ALREADY_RUN, NUM, ("epiParams", "seir"),
       filterable=False),
)

#: The keys of the standard register, for the tests and for the frontend's locale parity.
STANDARD_KEYS = tuple(v["key"] for v in STANDARD)


# ─────────────────────────────────────────────────────────────────────────────
# Counting, family by family
# ─────────────────────────────────────────────────────────────────────────────
def _head_counts(conn, scenario_id: str, thr: float) -> dict[str, int]:
    """One query over the relevant subset: how many papers carry a value, per variable whose
    answer is a column or a cached JSON field. Every count is a COUNT(*) FILTER over the same
    gate, so they are all out of the same set and can be read side by side."""
    screened = screening_status_sql("d", "ars")
    sql = f"""
        SELECT COUNT(*) AS n_relevant,
               COUNT(*) FILTER (WHERE d.year IS NOT NULL) AS year,
               COUNT(*) FILTER (WHERE COALESCE(TRIM(d.journal), '') <> '') AS journal,
               COUNT(*) FILTER (WHERE COALESCE(TRIM(d.language), '') <> '') AS language,
               COUNT(*) FILTER (WHERE COALESCE(TRIM(d.publication_type), '') <> '') AS publication_type,
               COUNT(*) FILTER (WHERE COALESCE(TRIM(d.source), '') <> '') AS source_db,
               COUNT(*) FILTER (WHERE d.open_access IS NOT NULL) AS open_access,
               COUNT(*) FILTER (WHERE d.has_fulltext IS TRUE) AS has_fulltext,
               COUNT(*) FILTER (WHERE d.citation_count IS NOT NULL) AS citations,
               COUNT(*) FILTER (WHERE COALESCE(TRIM(d.doi), '') <> '') AS has_doi,
               COUNT(*) FILTER (WHERE {screened} IN ('included', 'excluded')) AS screening,
               COUNT(*) FILTER (WHERE COALESCE(TRIM(ars.screening_reason), '') <> '') AS screening_reason,
               COUNT(*) FILTER (WHERE COALESCE(TRIM(ars.kappa_final_status), '') <> '') AS kappa,
               COUNT(*) FILTER (WHERE jsonb_typeof(d.pico_json) = 'object') AS pico,
               COUNT(*) FILTER (WHERE COALESCE(d.pico_json->>'key_finding', '') <> '') AS key_finding,
               COUNT(*) FILTER (WHERE COALESCE(d.pico_json->>'finding_direction', '') <> '') AS finding_direction,
               COUNT(*) FILTER (WHERE jsonb_typeof(d.concepts_json->'concepts') = 'array') AS concepts,
               COUNT(*) FILTER (WHERE jsonb_typeof(d.epi_params_json) = 'object') AS epi_params,
               COUNT(*) FILTER (WHERE jsonb_typeof(d.extraction_json) = 'object') AS n_extracted,
               COUNT(*) FILTER (WHERE d.extraction_json->>'source' IN ('fulltext', 'abstract')) AS read_from,
               COUNT(*) FILTER (WHERE d.extraction_json->'ref'->>'math_model' IS NOT NULL) AS math_model,
               COUNT(*) FILTER (WHERE d.extraction_json->'ref'->>'risk_pop' IS NOT NULL
                                   OR d.extraction_json->'ref'->>'positive' IS NOT NULL
                                   OR d.extraction_json->'ref'->>'percent_positive' IS NOT NULL) AS counts,
               COUNT(*) FILTER (WHERE COALESCE(d.extraction_json->'ref'->>'study_start', '') <> ''
                                   OR COALESCE(d.extraction_json->'ref'->>'study_end', '') <> '') AS study_period,
               COUNT(*) FILTER (WHERE EXISTS (
                   SELECT 1 FROM jsonb_array_elements(
                       CASE WHEN jsonb_typeof(d.extraction_json->'observations') = 'array'
                            THEN d.extraction_json->'observations' ELSE '[]'::jsonb END) AS ob
                   WHERE COALESCE(ob->>'disease', '') <> '')) AS disease
        FROM literature_document d
        JOIN article_scenarios ars ON ars.document_id = d.id
        WHERE ars.scenario_id = :sid AND {relevant_gate_sql('d', 'ars', ':thr')}
    """
    rows = _rows(conn, sql, scenario_id, thr)
    return {k: int(v or 0) for k, v in (rows[0] if rows else {}).items()}


def _corpus_total(conn, scenario_id: str) -> int:
    """Every paper the search put in this scenario, duplicates aside. The denominator of the
    one variable that is about the corpus and not about the relevant subset."""
    rows = _rows(conn, """
        SELECT COUNT(*) AS n
        FROM literature_document d
        JOIN article_scenarios ars ON ars.document_id = d.id
        WHERE ars.scenario_id = :sid AND (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)
    """, scenario_id, 0.0)
    return int(rows[0]["n"]) if rows else 0


#: The categorical variables whose values are a column or a cached field, counted in one
#: query. `(key, expression, fallback)`: the fallback names the papers with no value, which
#: is itself a value a reviewer filters on.
_VALUE_SOURCES: tuple[tuple[str, str], ...] = (
    ("language", "COALESCE(NULLIF(TRIM(d.language), ''), '')"),
    ("publication_type", "COALESCE(NULLIF(TRIM(d.publication_type), ''), '')"),
    ("source_db", "COALESCE(NULLIF(TRIM(d.source), ''), '')"),
    ("journal", "COALESCE(NULLIF(TRIM(d.journal), ''), '')"),
    ("screening", f"COALESCE(NULLIF({screening_status_sql('d', 'ars')}, ''), 'unjudged')"),
    ("kappa", "COALESCE(NULLIF(TRIM(ars.kappa_final_status), ''), '')"),
    ("screening_reason", "COALESCE(NULLIF(TRIM(ars.screening_reason), ''), '')"),
    ("finding_direction", "COALESCE(NULLIF(TRIM(d.pico_json->>'finding_direction'), ''), '')"),
    ("read_from", "COALESCE(NULLIF(TRIM(d.extraction_json->>'source'), ''), '')"),
    ("open_access", "CASE WHEN d.open_access IS TRUE THEN 'yes' WHEN d.open_access IS FALSE THEN 'no' ELSE '' END"),
    ("has_fulltext", "CASE WHEN d.has_fulltext IS TRUE THEN 'yes' ELSE 'no' END"),
    ("math_model", "COALESCE(NULLIF(TRIM(d.extraction_json->'ref'->>'math_model'), ''), '')"),
    ("year", "COALESCE(d.year::text, '')"),
)


def _value_counts(conn, scenario_id: str, thr: float) -> dict[str, list[dict[str, Any]]]:
    """The values of each column-backed categorical variable, with the number of papers, over
    the relevant subset. One query with one branch per variable: Postgres scans the subset
    once per branch, which is cheap on a few thousand rows, and the caller gets everything in
    one round trip instead of thirteen."""
    gate = relevant_gate_sql("d", "ars", ":thr")
    branches = []
    for key, expr in _VALUE_SOURCES:
        branches.append(f"""
            SELECT '{key}' AS var, {expr} AS value, COUNT(*) AS n
            FROM literature_document d
            JOIN article_scenarios ars ON ars.document_id = d.id
            WHERE ars.scenario_id = :sid AND {gate}
            GROUP BY 2
        """)
    sql = " UNION ALL ".join(branches) + " ORDER BY 1, 3 DESC, 2"
    out: dict[str, list[dict[str, Any]]] = {}
    for r in _rows(conn, sql, scenario_id, thr):
        # Une valeur vide n'est pas une valeur : elle dit « cet article ne la porte pas ».
        # Elle est comptée dans le dénominateur, jamais listée comme modalité.
        if not (r["value"] or "").strip():
            continue
        out.setdefault(r["var"], []).append({"value": r["value"], "n": int(r["n"])})
    return out


def _cut(values: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """The values to show and how many there are in all. The list is cut; the count is not."""
    shown = values[:_MAX_VALUES]
    for v in shown:
        v.setdefault("label", v["value"])
    return shown, len(values)


def _disease_values(conn, scenario_id: str, thr: float) -> list[dict[str, Any]]:
    """The diseases of the extracted rows, one entry per disease however it is spelt.

    "HPAI H5N1", "influenza A(H5N1)" and "H5N1" are one disease, and the pooled estimates
    already fold them that way (`api/diseases`): the register must not show them as three."""
    from .diseases import disease_key

    gate = relevant_gate_sql("d", "ars", ":thr")
    ids: dict[str, set] = {}
    for r in _rows(conn, f"""
        SELECT o->>'disease' AS disease, array_agg(DISTINCT d.id) AS ids
        {_OBS_FROM.format(gate=gate)} AND NULLIF(TRIM(o->>'disease'), '') IS NOT NULL
        GROUP BY 1
    """, scenario_id, thr):
        ids.setdefault(disease_key(r["disease"]), set()).update(r["ids"] or ())
    return [{"value": k, "label": k, "n": len(v)}
            for k, v in sorted(ids.items(), key=lambda kv: (-len(kv[1]), kv[0]))]


# ─────────────────────────────────────────────────────────────────────────────
# The scenario's own variables: the groups of its codebook
# ─────────────────────────────────────────────────────────────────────────────
def _own_variables(conn, scenario_id: str, thr: float, lang: str) -> list[dict[str, Any]]:
    """One variable per codebook GROUP, with the papers that answered it and its values.

    A group the codebook declares and no paper answered is listed with zero: that is the gap
    map, and a blank there is a result. A group a paper wrote that the codebook maps to
    nothing is listed too, marked `unmapped`, because it is work waiting to be folded in: on
    the HPAI corpus "wild bird" sits there, with a hundred papers behind it, as a value of
    Species written where a group belongs.

    The counting is exact: the ids of the papers behind each group and each value are unioned,
    so a paper that words the same value two ways is counted once."""
    nodes = (get_codebook(scenario_id) or {}).get("nodes") or []
    index = get_index(scenario_id)
    names = display_names(index, lang)

    declared: list[tuple[str, str]] = []
    for n in nodes:
        key = (n["sheet"], n["l1"])
        if key not in declared:
            declared.append(key)

    gate = relevant_gate_sql("d", "ars", ":thr")
    rows = _rows(conn, f"""
        SELECT o->>'sheet' AS sheet, o->>'group' AS grp, o->>'covariate' AS cov,
               array_agg(DISTINCT d.id) AS ids
        {_OBS_FROM.format(gate=gate)} AND NULLIF(TRIM(o->>'group'), '') IS NOT NULL
        GROUP BY 1, 2, 3
    """, scenario_id, thr)

    per_group: dict[tuple[str, str], set] = {}
    per_value: dict[tuple[str, str], dict[str, set]] = {}
    for r in rows:
        sheet = sheet_key(r["sheet"]) or (r["sheet"] or "")
        pos = normalise(index, sheet, r["grp"], r["cov"])
        gkey = pos["group_key"] or (r["grp"] or "").strip().lower()
        if not gkey:
            continue
        ids = set(r["ids"] or ())
        per_group.setdefault((sheet, gkey), set()).update(ids)
        vkey = pos["covariate_key"]
        if vkey:
            per_value.setdefault((sheet, gkey), {}).setdefault(vkey, set()).update(ids)

    out: list[dict[str, Any]] = []
    for sheet, gkey in declared + [k for k in per_group if k not in declared]:
        ids = per_group.get((sheet, gkey)) or set()
        vals = [{"value": v, "label": display_path(names, sheet, gkey, v, lang).split(" > ")[-1],
                 "n": len(vids)}
                for v, vids in sorted((per_value.get((sheet, gkey)) or {}).items(),
                                      key=lambda kv: (-len(kv[1]), kv[0]))]
        shown, n_values = _cut(vals)
        unmapped = (sheet, gkey) not in declared
        out.append({
            "key": f"{sheet}.{gkey}", "family": "own", "source": "codebook",
            "cost": COST_ONE_PASS, "type": CAT, "shown_in": ["extraction", "pooled"],
            "denominator_of": OF_EXTRACTED, "filterable": True, "crossable": True,
            "covers": None,
            "name": (gkey.replace("_", " ") if unmapped
                     else display_path(names, sheet, gkey, None, lang)),
            "sheet": sheet, "group": gkey, "unmapped": unmapped,
            "n_with_value": len(ids), "values": shown, "n_values": n_values,
        })
    out.sort(key=lambda v: (-v["n_with_value"], v["name"]))
    return out[:_MAX_OWN]


# ─────────────────────────────────────────────────────────────────────────────
# The assembly
# ─────────────────────────────────────────────────────────────────────────────
def variables_of_interest(scenario_id: str, lang: str = "en",
                          threshold: float | None = None) -> dict[str, Any]:
    """The register of a scenario: the standard variables and the scenario's own, each with
    its coverage over the whole relevant subset and its values.

    Never raises. A family whose source cannot be read loses its counts and is named in
    `incomplete`, with the reason, and the rest of the register is served."""
    thr = _get_scenario_threshold(scenario_id) if threshold is None else float(threshold)
    lang = _norm_lang(lang) or "en"
    out: dict[str, Any] = {"scenario_id": scenario_id, "lang": lang, "threshold": thr,
                           "families": list(FAMILIES), "incomplete": {}, "notes": {}}

    head: dict[str, int] = {}
    values: dict[str, list[dict[str, Any]]] = {}
    n_corpus = 0
    try:
        with engine.connect() as conn:
            head = _head_counts(conn, scenario_id, thr)
            n_corpus = _corpus_total(conn, scenario_id)
            values = _value_counts(conn, scenario_id, thr)
            values["disease"] = _disease_values(conn, scenario_id, thr)
    except Exception as e:                                   # noqa: BLE001 - never blocks the panel
        logger.warning(f"variables_of_interest head {scenario_id}: {e}")
        out["incomplete"]["record"] = str(e)[:200]

    n_relevant = int(head.get("n_relevant") or 0)
    n_extracted = int(head.get("n_extracted") or 0)
    out.update({"n_relevant": n_relevant, "n_extracted": n_extracted, "n_corpus": n_corpus})

    # Les trois dimensions qui demandent plus qu'une colonne. Chacune dans son propre
    # try/except : la projection des thèmes, le résolveur de lieux et le pliage des devis
    # échouent indépendamment, et l'un qui tombe ne doit pas emporter les deux autres.
    designs: list[dict[str, Any]] = []
    levels: list[dict[str, Any]] = []
    try:
        _, _, meta = _design_membership(scenario_id, thr)
        designs = [{"value": d["value"], "label": design_label(d["value"], lang), "n": int(d["n"])}
                   for d in meta.get("designs") or []]
        levels = [{"value": lv["value"], "label": level_label(lv["value"], lang), "n": int(lv["n"])}
                  for lv in meta.get("levels") or []]
    except Exception as e:                                   # noqa: BLE001
        logger.warning(f"variables_of_interest designs {scenario_id}: {e}")
        out["incomplete"]["design"] = str(e)[:200]

    countries: list[dict[str, Any]] = []
    n_placed = n_europe = n_nuts = n_geo_papers = 0
    try:
        geo = geography(scenario_id)
        n_geo_papers = int(geo.get("n_papers") or 0)
        n_placed = int(geo.get("n_resolved") or 0)
        n_europe = int(geo.get("n_in_europe") or 0)
        for c in geo.get("countries") or []:
            countries.append({"value": c["iso2"], "label": c["name"], "n": int(c["n_papers"])})
            n_nuts += sum(int(g.get("n_papers") or 0) for g in c.get("regions") or [])
    except Exception as e:                                   # noqa: BLE001
        logger.warning(f"variables_of_interest places {scenario_id}: {e}")
        out["incomplete"]["place"] = str(e)[:200]

    narrowed_out = 0
    try:
        from .subsets import scope_state
        narrowed_out = int((scope_state(scenario_id) or {}).get("excluded_by_scope") or 0)
    except Exception as e:                                   # noqa: BLE001
        logger.warning(f"variables_of_interest scope {scenario_id}: {e}")
        out["incomplete"]["decision"] = str(e)[:200]

    clusters: list[dict[str, Any]] = []
    n_clustered = 0
    try:
        cluster_of, cluster_names, _ = _cluster_membership(scenario_id)
        counts: dict[int, int] = {}
        for cid in cluster_of.values():
            counts[cid] = counts.get(cid, 0) + 1
        n_clustered = len(cluster_of)
        clusters = [{"value": str(cid), "label": cluster_names.get(cid, str(cid)), "n": n}
                    for cid, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    except HTTPException as e:
        # Pas de clustering en cache n'est pas une panne du registre : c'est une projection
        # qui n'a pas encore été calculée. La variable existe, elle ne compte encore rien, et
        # la raison est dite. La marquer « incomplete » ferait passer l'état normal d'un
        # scénario tout neuf pour une erreur.
        out["notes"]["cluster"] = str(e.detail)[:300]
    except Exception as e:                                   # noqa: BLE001
        logger.warning(f"variables_of_interest clusters {scenario_id}: {e}")
        out["incomplete"]["topic"] = str(e)[:200]

    #: Les listes de modalités qui ne viennent pas d'une colonne, par clé de variable.
    computed: dict[str, list[dict[str, Any]]] = {
        "design": designs, "grade_level": levels, "country": countries, "cluster": clusters,
        "europe": [{"value": "europe", "label": "Europe", "n": n_europe},
                   {"value": "elsewhere", "label": "elsewhere", "n": max(n_placed - n_europe, 0)}],
        "above_threshold": [{"value": "above", "label": "above", "n": n_relevant},
                            {"value": "below", "label": "below", "n": max(n_corpus - n_relevant, 0)}],
        "scope": [{"value": "in_scope", "label": "in scope", "n": n_relevant},
                  {"value": "narrowed_out", "label": "narrowed out", "n": narrowed_out}],
    }
    #: Les couvertures qui ne se comptent pas en SQL sur une colonne.
    overrides: dict[str, int] = {
        "design": sum(d["n"] for d in designs), "grade_level": sum(lv["n"] for lv in levels),
        "country": n_placed, "europe": n_europe, "nuts": n_nuts, "cluster": n_clustered,
        "above_threshold": n_relevant, "scope": n_relevant,
    }
    denominators = {OF_RELEVANT: n_relevant, OF_EXTRACTED: n_extracted, OF_CORPUS: n_corpus}

    variables: list[dict[str, Any]] = []
    for spec in STANDARD:
        key = spec["key"]
        shown, n_values = _cut(list(computed.get(key) or values.get(key) or []))
        n_with = overrides.get(key, head.get(key))
        variables.append({**spec, "name": None,
                          "n_with_value": (int(n_with) if n_with is not None else None),
                          "denominator": denominators.get(spec["denominator_of"], n_relevant),
                          "values": shown, "n_values": n_values})

    own: list[dict[str, Any]] = []
    try:
        with engine.connect() as conn:
            own = _own_variables(conn, scenario_id, thr, lang)
    except Exception as e:                                   # noqa: BLE001
        logger.warning(f"variables_of_interest own {scenario_id}: {e}")
        out["incomplete"]["own"] = str(e)[:200]
    for v in own:
        v["denominator"] = n_extracted

    out["variables"] = variables + own
    out["n_standard"] = len(variables)
    out["n_own"] = len(own)
    out["n_unmapped"] = sum(1 for v in own if v.get("unmapped"))
    # Le nombre d'articles que la géographie a pu lire, pour que « 546 sur 818 » se vérifie.
    out["n_with_place_read"] = n_geo_papers
    out["coverage_keys"] = list(COVERAGE_KEYS)
    out["finding_directions"] = list(FINDING_DIRECTIONS)
    out["complete"] = not out["incomplete"]
    return out


@app.get("/user-scenarios/{scenario_id}/variables-of-interest")
def get_variables_of_interest(scenario_id: str, lang: str = Query("en", max_length=8),
                              threshold: float | None = Query(None, ge=0.0, le=1.0)) -> dict[str, Any]:
    """The register of this review's variables of interest, counted over the whole relevant
    subset. Read only: nothing is extracted, nothing is written, no model is called."""
    _get_user_scenario_or_404(scenario_id)
    return variables_of_interest(scenario_id, lang=lang, threshold=threshold)
