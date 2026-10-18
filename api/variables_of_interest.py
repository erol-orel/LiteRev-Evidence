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
from fastapi.responses import Response
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
#: How many UNMAPPED groups are listed. The groups a codebook declares are never cut: they
#: are the questions the review asks, and a cap that hid one would turn the gap map into a
#: list of what happens to be present. What the model writes besides them is a long tail, so
#: it is cut here, with its true length and the papers behind it said in the payload.
_MAX_UNMAPPED_SHOWN = 30


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
    # Le côté du seuil et la portée ne se filtrent PAS: ils sont la porte et le mécanisme
    # eux-mêmes. « Garder ce qui est au-dessus du seuil » est déjà ce que fait la porte, et
    # « garder ce qui est hors de portée » demanderait de garder des articles exclus.
    _v("above_threshold", "decision", "scoring", COST_FREE, BOOL, ("seuil",),
       of=OF_CORPUS, filterable=False, crossable=False),
    _v("scope", "decision", "selection", COST_FREE, CAT, ("subset", "prisma"),
       filterable=False, crossable=False),
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
    # La région se filtrera quand le résolveur rendra le code NUTS par article; il ne rend
    # aujourd'hui que le pays, et promettre un filtre que rien ne résout serait pire que
    # l'absence du filtre.
    _v("nuts", "place", "nuts_list", COST_FREE, CAT, ("geography",), of=OF_EXTRACTED,
       filterable=False, covers=("nuts1", "nuts2", "nuts3")),
    _v("study_period", "place", "extraction", COST_ALREADY_RUN, DATE, ("extraction",),
       of=OF_EXTRACTED, filterable=False, crossable=False,
       covers=("study_start", "study_end")),
    # ── E. Topic ────────────────────────────────────────────────────────────
    _v("cluster", "topic", "clustering", COST_ALREADY_RUN, CAT, ("clustering",)),
    _v("concepts", "topic", "concept_pass", COST_ALREADY_RUN, CAT, ("knowledgeGraph",)),
    _v("disease", "topic", "disease_folder", COST_FREE, CAT, ("extraction", "pooled"),
       of=OF_EXTRACTED),
    # ── F. What was read, and the quantities ────────────────────────────────
    _v("read_from", "measure", "extraction", COST_ALREADY_RUN, CAT, ("extraction",),
       of=OF_EXTRACTED, crossable=False),
    # Trois nombres sur une ligne: une condition porte sur UNE variable, donc ils se
    # filtreront séparément le jour où ils seront trois lignes.
    _v("counts", "measure", "extraction", COST_ALREADY_RUN, NUM, ("extraction", "pooled"),
       of=OF_EXTRACTED, filterable=False, covers=("risk_pop", "positive", "percent_positive")),
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
#: query, and read per article by `variable_membership`. One table for both, so a variable
#: can never be counted one way on the page and filtered another way by the selection.
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

#: Lues PAR ARTICLE seulement. Leur liste de modalités n'apprendrait rien (des centaines
#: d'entiers distincts pour les citations), mais une condition sur elles a du sens.
_MEMBERSHIP_ONLY: tuple[tuple[str, str], ...] = (
    ("citations", "COALESCE(d.citation_count::text, '')"),
    ("has_doi", "CASE WHEN COALESCE(TRIM(d.doi), '') <> '' THEN 'yes' ELSE 'no' END"),
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

    def _row(sheet: str, gkey: str, unmapped: bool) -> dict[str, Any]:
        ids = per_group.get((sheet, gkey)) or set()
        vals = [{"value": v, "label": display_path(names, sheet, gkey, v, lang).split(" > ")[-1],
                 "n": len(vids)}
                for v, vids in sorted((per_value.get((sheet, gkey)) or {}).items(),
                                      key=lambda kv: (-len(kv[1]), kv[0]))]
        shown, n_values = _cut(vals)
        return {
            "key": f"{sheet}.{gkey}", "family": "own", "source": "codebook",
            "cost": COST_ONE_PASS, "type": CAT, "shown_in": ["extraction", "pooled"],
            "denominator_of": OF_EXTRACTED, "filterable": True, "crossable": True,
            "covers": None,
            "name": (gkey.replace("_", " ") if unmapped
                     else display_path(names, sheet, gkey, None, lang)),
            "sheet": sheet, "group": gkey, "unmapped": unmapped,
            "n_with_value": len(ids), "values": shown, "n_values": n_values,
        }

    # Les groupes DÉCLARÉS par le codebook passent toujours, en entier : ils sont la
    # question que la revue pose, et un plafond qui en cache un transformerait la carte
    # des manques en liste des présents. Sur un vrai corpus le modèle écrit en plus une
    # longue traîne de libellés de groupe que le codebook ne sait pas placer (102 sur la
    # revue HPAI) : eux sont plafonnés, et le total est dit plutôt que coupé en silence,
    # avec le nombre d'articles derrière, parce que c'est ce nombre qui dit si la traîne
    # mérite une séance de rattachement.
    mine = sorted((_row(s, g, False) for s, g in declared),
                  key=lambda v: (-v["n_with_value"], v["name"]))
    strays = sorted((_row(s, g, True) for s, g in per_group if (s, g) not in declared),
                    key=lambda v: (-v["n_with_value"], v["name"]))
    stray_ids: set = set()
    for sheet, gkey in per_group:
        if (sheet, gkey) not in declared:
            stray_ids |= per_group[(sheet, gkey)]
    totals = {
        "n_declared": len(mine),
        "n_unmapped": len(strays),
        "n_unmapped_shown": min(len(strays), _MAX_UNMAPPED_SHOWN),
        # Les articles qui portent au moins un groupe non rattachable : la mesure du
        # travail qui reste, et non le nombre d'étiquettes.
        "n_papers_unmapped": len(stray_ids),
    }
    return mine + strays[:_MAX_UNMAPPED_SHOWN], totals


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
    n_placed = n_europe = n_eu_no_country = n_nuts = n_geo_papers = 0
    try:
        geo = geography(scenario_id)
        n_geo_papers = int(geo.get("n_papers") or 0)
        n_placed = int(geo.get("n_resolved") or 0)
        n_europe = int(geo.get("n_in_europe") or 0)
        n_eu_no_country = int(geo.get("n_in_europe_without_country") or 0)
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
        # « Ailleurs » n'est PAS « placés moins européens ». Un article peut être européen
        # sans nommer un pays (« Europe », « Germany and the Netherlands ») : sur la revue
        # HPAI, 43 des 164 européens sont dans ce cas, et ils ne figurent donc pas parmi les
        # 503 articles à pays nommé. Retrancher les 164 des 503 les enlevait deux fois et
        # donnait 339 là où la vérité est 382, ce que la sélection par pays comptait déjà
        # correctement de son côté. Les trois seaux s'additionnent maintenant au nombre
        # d'articles lus, ce qui rend l'erreur impossible à refaire sans le voir.
        "europe": [{"value": "europe", "label": "Europe", "n": n_europe},
                   {"value": "elsewhere", "label": "elsewhere",
                    "n": max(n_placed - (n_europe - n_eu_no_country), 0)},
                   {"value": "no_place", "label": "no place stated",
                    "n": max(n_geo_papers - n_europe
                             - max(n_placed - (n_europe - n_eu_no_country), 0), 0)}],
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
    own_totals: dict[str, int] = {"n_declared": 0, "n_unmapped": 0, "n_unmapped_shown": 0,
                                  "n_papers_unmapped": 0}
    try:
        with engine.connect() as conn:
            own, own_totals = _own_variables(conn, scenario_id, thr, lang)
    except Exception as e:                                   # noqa: BLE001
        logger.warning(f"variables_of_interest own {scenario_id}: {e}")
        out["incomplete"]["own"] = str(e)[:200]
    for v in own:
        v["denominator"] = n_extracted

    out["variables"] = variables + own
    out["n_standard"] = len(variables)
    out["n_own"] = own_totals["n_declared"]
    # Le total VRAI des groupes que le codebook ne sait pas placer, pas le nombre affiché,
    # et les articles derrière eux : « 102 étiquettes sur 318 articles » dit s'il faut une
    # séance de rattachement, ce qu'une liste coupée à trente ne dirait pas.
    out["n_unmapped"] = own_totals["n_unmapped"]
    out["n_unmapped_shown"] = own_totals["n_unmapped_shown"]
    out["n_papers_unmapped"] = own_totals["n_papers_unmapped"]
    # Le nombre d'articles que la géographie a pu lire, pour que « 503 sur 818 » se vérifie.
    out["n_with_place_read"] = n_geo_papers
    out["coverage_keys"] = list(COVERAGE_KEYS)
    out["finding_directions"] = list(FINDING_DIRECTIONS)
    out["complete"] = not out["incomplete"]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# One variable, read per article, for the selection
# ─────────────────────────────────────────────────────────────────────────────
#: Les variables que la SÉLECTION résout par un chemin à elle, dans api/subsets : leur
#: sémantique est particulière (le bruit du clustering, l'Europe sans pays unique, « devis
#: non précisé » qui est une réponse) et éprouvée. Nommées ici pour que le registre et la
#: sélection ne se contredisent jamais sur qui traite quoi.
ROUTED_TO_SUBSETS = {"cluster": "clusters", "design": "designs", "grade_level": "levels",
                     "country": "countries", "europe": "countries", "concepts": "concepts"}

#: Les types lus comme des nombres : la condition porte alors un intervalle, pas une liste.
_NUMBER_KEYS = {"year", "citations"}


def filterable_keys() -> set[str]:
    """Les clés sur lesquelles une condition de sélection a un sens: les variables standard
    marquées filtrables, plus les groupes du codebook (`sheet.group`, reconnus au point)."""
    return {v["key"] for v in STANDARD if v["filterable"]}


def variable_membership(scenario_id: str, variable: str,
                        threshold: float | None = None) -> dict[str, Any]:
    """Qui porte quelle valeur, pour UNE variable, sur tout le sous-ensemble pertinent.

    Renvoie `of` ({article: valeur}, ou {article: set(valeurs)} quand un article peut en
    porter plusieurs), `kind` ("value", "multi" ou "number"), `labels` ({valeur: nom lisible})
    et `present` (les valeurs effectivement portées). `judgeable` reste implicite: un article
    absent de `of` n'a pas de valeur, donc la condition ne sait pas le juger et ne vote ni
    pour ni contre, exactement comme une projection plafonnée.

    Les cinq dimensions de ROUTED_TO_SUBSETS ne passent PAS ici: la sélection les résout
    par leur chemin propre. Demander l'une d'elles est donc une erreur de programmation,
    signalée comme telle plutôt que silencieusement mal répondue."""
    if variable in ROUTED_TO_SUBSETS:
        raise ValueError(f"{variable} is resolved by api/subsets, not here")
    thr = _get_scenario_threshold(scenario_id) if threshold is None else float(threshold)
    gate = relevant_gate_sql("d", "ars", ":thr")
    out: dict[str, Any] = {"variable": variable, "kind": "value", "of": {}, "labels": {},
                           "present": []}

    # ── Un groupe du codebook: `sheet.group`. Un article peut en porter plusieurs valeurs
    #    (deux professions, trois espèces), donc `multi`.
    if "." in variable:
        sheet, _, gkey = variable.partition(".")
        index = get_index(scenario_id)
        names = display_names(index, "en")
        of: dict[int, set] = {}
        labels: dict[str, str] = {}
        with engine.connect() as conn:
            for r in _rows(conn, f"""
                SELECT o->>'sheet' AS sheet, o->>'group' AS grp, o->>'covariate' AS cov,
                       array_agg(DISTINCT d.id) AS ids
                {_OBS_FROM.format(gate=gate)} AND NULLIF(TRIM(o->>'group'), '') IS NOT NULL
                GROUP BY 1, 2, 3
            """, scenario_id, thr):
                s = sheet_key(r["sheet"]) or (r["sheet"] or "")
                if s != sheet:
                    continue
                pos = normalise(index, s, r["grp"], r["cov"])
                if (pos["group_key"] or (r["grp"] or "").strip().lower()) != gkey:
                    continue
                # Une ligne dont la VALEUR ne se rattache à rien porte quand même son
                # groupe: l'article a répondu à la question, et l'écarter d'une sélection
                # sur ce groupe le rendrait injugeable alors qu'il a parlé.
                vkey = pos["covariate_key"] or ""
                if not vkey:
                    continue
                labels.setdefault(vkey, display_path(names, sheet, gkey, vkey, "en").split(" > ")[-1])
                for i in (r["ids"] or ()):
                    of.setdefault(int(i), set()).add(vkey)
        out.update({"kind": "multi", "of": of, "labels": labels,
                    "present": sorted({v for vs in of.values() for v in vs})})
        return out

    # ── La maladie: pliée comme le pooling la plie, et multiple par nature.
    if variable == "disease":
        from .diseases import disease_key

        of_d: dict[int, set] = {}
        with engine.connect() as conn:
            for r in _rows(conn, f"""
                SELECT o->>'disease' AS disease, array_agg(DISTINCT d.id) AS ids
                {_OBS_FROM.format(gate=gate)} AND NULLIF(TRIM(o->>'disease'), '') IS NOT NULL
                GROUP BY 1
            """, scenario_id, thr):
                key = disease_key(r["disease"])
                for i in (r["ids"] or ()):
                    of_d.setdefault(int(i), set()).add(key)
        present = sorted({v for vs in of_d.values() for v in vs})
        out.update({"kind": "multi", "of": of_d, "labels": {k: k for k in present},
                    "present": present})
        return out

    # ── Une colonne ou un champ caché, par la MÊME expression que le comptage.
    expr = dict(_VALUE_SOURCES + _MEMBERSHIP_ONLY).get(variable)
    if expr is None:
        raise ValueError(f"unknown variable {variable!r}")
    number = variable in _NUMBER_KEYS
    of_v: dict[int, Any] = {}
    with engine.connect() as conn:
        for r in _rows(conn, f"""
            SELECT d.id AS id, {expr} AS v
            FROM literature_document d
            JOIN article_scenarios ars ON ars.document_id = d.id
            WHERE ars.scenario_id = :sid AND {gate}
        """, scenario_id, thr):
            raw = (r["v"] or "").strip()
            if not raw:
                continue
            if number:
                try:
                    of_v[int(r["id"])] = int(float(raw))
                except (TypeError, ValueError):
                    continue
            else:
                of_v[int(r["id"])] = raw
    present = sorted({str(v) for v in of_v.values()})
    out.update({"kind": "number" if number else "value", "of": of_v,
                "labels": {v: v for v in present}, "present": present})
    return out


def condition_verdict(of: dict[int, Any], kind: str, wanted: set | None = None,
                      lo: float | None = None, hi: float | None = None):
    """La fonction de verdict d'UNE condition: `in`, `out` ou `unknown` par article. PURE.

    `unknown` veut dire « cette condition ne sait rien de cet article », jamais « non »:
    un article sans valeur ne vote ni pour ni contre, et c'est la même règle que pour le
    clustering plafonné. Confondre les deux viderait un corpus sur la foi d'une absence."""
    if kind == "multi":
        want = wanted or set()

        def _multi(i: int) -> str:
            vs = of.get(i)
            if not vs:
                return "unknown"
            return "in" if (vs & want) else "out"
        return _multi
    if kind == "number":
        def _num(i: int) -> str:
            v = of.get(i)
            if v is None:
                return "unknown"
            if lo is not None and v < lo:
                return "out"
            if hi is not None and v > hi:
                return "out"
            return "in"
        return _num
    want_v = wanted or set()

    def _value(i: int) -> str:
        v = of.get(i)
        if v is None:
            return "unknown"
        return "in" if str(v) in want_v else "out"
    return _value


# ─────────────────────────────────────────────────────────────────────────────
# Crossing two variables
# ─────────────────────────────────────────────────────────────────────────────
#: Combien de modalités par axe. Au-delà, une grille ne se lit plus, et la longueur vraie
#: de chaque axe est dite à côté pour qu'une troncature ne passe pas pour un corpus pauvre.
_CROSS_MAX_LABELS = 12


def _as_sets(of: dict[int, Any], kind: str) -> dict[int, set]:
    """Une carte article -> valeurs, quelle que soit sa forme. Un nombre devient sa propre
    valeur: croiser l'année avec autre chose a un sens, et une année EST une modalité."""
    if kind == "multi":
        return {i: set(v) for i, v in of.items() if v}
    return {i: {str(v)} for i, v in of.items() if v is not None}


def cross_counts(of_rows: dict[int, set], of_cols: dict[int, set],
                 max_labels: int = _CROSS_MAX_LABELS) -> dict[str, Any]:
    """La grille de deux variables, en articles DISTINCTS. PURE.

    Le total d'un axe est le nombre d'articles qui portent cette modalité, jamais la somme
    de ses cellules: un article qui porte deux professions et un pays compte une fois pour
    ce pays, et une somme de cellules le compterait deux fois. La carte des concepts, qui
    additionne, le dit dans sa propre note; ici les deux lectures ne peuvent pas diverger.

    La population de la grille est l'ensemble des articles que les DEUX variables savent
    juger, et les trois autres cas sont comptés à part: un lecteur doit savoir ce que la
    grille n'a pas pu placer avant de lire une case vide comme un manque."""
    both = set(of_rows) & set(of_cols)
    cells: dict[tuple[str, str], set] = {}
    row_ids: dict[str, set] = {}
    col_ids: dict[str, set] = {}
    for i in both:
        for r in of_rows[i]:
            row_ids.setdefault(r, set()).add(i)
            for c in of_cols[i]:
                cells.setdefault((r, c), set()).add(i)
        for c in of_cols[i]:
            col_ids.setdefault(c, set()).add(i)

    def _axis(ids: dict[str, set]) -> list[str]:
        return [k for k, _ in sorted(ids.items(), key=lambda kv: (-len(kv[1]), kv[0]))]

    all_rows, all_cols = _axis(row_ids), _axis(col_ids)
    rows, cols = all_rows[:max_labels], all_cols[:max_labels]
    grid = [{"row": r, "col": c, "n": len(cells.get((r, c)) or ())} for r in rows for c in cols]
    return {
        "rows": [{"value": r, "n": len(row_ids[r])} for r in rows],
        "cols": [{"value": c, "n": len(col_ids[c])} for c in cols],
        "rows_total": len(all_rows), "cols_total": len(all_cols),
        "cells": grid,
        "n_both": len(both),
        "n_rows_only": len(set(of_rows) - set(of_cols)),
        "n_cols_only": len(set(of_cols) - set(of_rows)),
        "pairs_observed": sum(1 for c in grid if c["n"] > 0),
        "cells_shown": len(grid),
    }


def membership_for_crossing(scenario_id: str, variable: str, threshold: float,
                            lang: str = "en") -> tuple[dict[int, set], dict[str, str]]:
    """(article -> valeurs, {valeur: nom lisible}) pour N'IMPORTE QUELLE variable du
    registre, les cinq que la sélection résout elle-même comprises.

    Elles passent par les MÊMES fonctions que la sélection et que les panneaux (le pliage
    des devis, le résolveur de lieux, le cache du clustering), de sorte qu'une case de la
    grille ne peut pas compter autrement que la barre qui montre la même chose ailleurs."""
    from .study_design import design_label, level_label
    from .subsets import EUROPE_KEY, _cluster_membership, _country_membership, _design_membership

    if variable in ("design", "grade_level"):
        design_of, level_of, _ = _design_membership(scenario_id, threshold)
        of = design_of if variable == "design" else level_of
        fn = design_label if variable == "design" else level_label
        return ({i: {v} for i, v in of.items()},
                {v: fn(v, lang) for v in set(of.values())})
    if variable in ("country", "europe"):
        from .geography import COUNTRY_NAME

        country_of, _ = _country_membership(scenario_id, threshold)
        if variable == "europe":
            from .geography import NUTS_ISO

            eu = set(NUTS_ISO) | {EUROPE_KEY}
            return ({i: {"europe" if v in eu else "elsewhere"} for i, v in country_of.items()},
                    {"europe": "Europe", "elsewhere": "elsewhere"})
        return ({i: {v} for i, v in country_of.items()},
                {v: (COUNTRY_NAME.get(v, v) if v != EUROPE_KEY else "Europe (no single country)")
                 for v in set(country_of.values())})
    if variable == "cluster":
        cluster_of, names, _ = _cluster_membership(scenario_id)
        return ({i: {str(c)} for i, c in cluster_of.items()},
                {str(c): n for c, n in names.items()})
    got = variable_membership(scenario_id, variable, threshold)
    return _as_sets(got["of"], got["kind"]), dict(got["labels"])


def crosstab(scenario_id: str, rows: str, cols: str, lang: str = "en",
             threshold: float | None = None) -> dict[str, Any]:
    """Deux variables du registre croisées sur tout le sous-ensemble pertinent.

    Jamais d'exception: une variable que sa source ne sait pas lire (un clustering non
    calculé) rend une grille vide qui dit pourquoi, parce qu'un panneau qui tombe en entier
    n'apprend rien à personne."""
    thr = _get_scenario_threshold(scenario_id) if threshold is None else float(threshold)
    lang = _norm_lang(lang) or "en"
    out: dict[str, Any] = {"scenario_id": scenario_id, "rows_variable": rows,
                           "cols_variable": cols, "threshold": thr, "lang": lang}
    if rows == cols:
        raise HTTPException(status_code=422, detail="Croisez deux variables différentes.")
    crossable = {v["key"] for v in STANDARD if v["crossable"]}
    for key in (rows, cols):
        if "." not in key and key not in crossable:
            raise HTTPException(
                status_code=404,
                detail=f"La variable « {key} » ne se croise pas. Croisables : "
                       f"{', '.join(sorted(crossable))}, ou un groupe du codebook (sheet.group).")
    try:
        of_r, labels_r = membership_for_crossing(scenario_id, rows, thr, lang)
        of_c, labels_c = membership_for_crossing(scenario_id, cols, thr, lang)
    except HTTPException as e:
        out.update({"complete": False, "note": str(e.detail)[:300], "cells": [],
                    "rows": [], "cols": []})
        return out
    except Exception as e:                                   # noqa: BLE001
        logger.warning(f"crosstab {scenario_id} {rows}x{cols}: {e}")
        out.update({"complete": False, "error": str(e)[:200], "cells": [], "rows": [], "cols": []})
        return out
    grid = cross_counts(of_r, of_c)
    for r in grid["rows"]:
        r["label"] = labels_r.get(r["value"], r["value"])
    for c in grid["cols"]:
        c["label"] = labels_c.get(c["value"], c["value"])
    out.update(grid)
    out["complete"] = True
    return out


@app.get("/user-scenarios/{scenario_id}/variables-of-interest/crosstab")
def get_variables_crosstab(scenario_id: str, rows: str = Query(..., max_length=80),
                           cols: str = Query(..., max_length=80),
                           lang: str = Query("en", max_length=8)) -> dict[str, Any]:
    """Deux variables croisées, en articles distincts, sur tout le sous-ensemble pertinent."""
    _get_user_scenario_or_404(scenario_id)
    return crosstab(scenario_id, rows, cols, lang)


@app.get("/user-scenarios/{scenario_id}/variables-of-interest/export")
def export_variables_of_interest(scenario_id: str, lang: str = Query("en", max_length=8)):
    """Le registre en CSV: une ligne par variable, sa couverture et son dénominateur.

    C'est la carte des manques sous la forme qu'un partenaire peut coller dans son tableur:
    « voilà les questions que la revue pose, et combien d'articles y répondent »."""
    import csv as _csv
    import io as _io

    _get_user_scenario_or_404(scenario_id)
    reg = variables_of_interest(scenario_id, lang=lang)
    buf = _io.StringIO()
    w = _csv.writer(buf)
    w.writerow(["variable", "name", "family", "filled_by", "cost", "type",
                "papers_with_a_value", "out_of", "denominator", "n_values",
                "values_shown", "unmapped"])
    for v in reg["variables"]:
        w.writerow([v["key"], v.get("name") or "", v["family"], v["source"], v["cost"],
                    v["type"], "" if v["n_with_value"] is None else v["n_with_value"],
                    v["denominator"], v["denominator_of"], v["n_values"],
                    " | ".join(f"{x['label']}={x['n']}" for x in v["values"]),
                    "yes" if v.get("unmapped") else ""])
    return Response(content="﻿" + buf.getvalue(), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition":
                             f'attachment; filename="variables-{scenario_id}.csv"'})


@app.get("/user-scenarios/{scenario_id}/variables-of-interest")
def get_variables_of_interest(scenario_id: str, lang: str = Query("en", max_length=8),
                              threshold: float | None = Query(None, ge=0.0, le=1.0)) -> dict[str, Any]:
    """The register of this review's variables of interest, counted over the whole relevant
    subset. Read only: nothing is extracted, nothing is written, no model is called."""
    _get_user_scenario_or_404(scenario_id)
    return variables_of_interest(scenario_id, lang=lang, threshold=threshold)
