"""What a scenario is supposed to contain, and whether it does.

Every output of this application had its own storage, its own freshness rule and its own
silent failure mode, and no single place said what a finished scenario holds. Thirteen
pipeline steps, a dozen panels each fetching for itself, and the only detector of a
missing output was a reviewer noticing in front of partners. That is how the evidence
report could be deleted by the tab that displays it for weeks without anything failing:
nothing was ever asked whether it was there.

So the list lives here, once, in `OUTPUTS`, and `scenario_outputs()` answers it over SQL:
present, stale, partial or missing, when it was built, and from how many papers. No model
call, nothing written, and the whole thing is two queries.

Two shapes of output, and they are measured differently:

  - an ARTEFACT is built once and stored (the corpus portrait, the narrative report, the
    clusters, the concept graph, the recommended actions, the variable spec). It is
    MISSING when nothing is stored and STALE when it was built before the corpus last
    moved. A stale artefact is still served everywhere: an output may be old, it must
    never be absent.
  - a COVERAGE is a fact cached on each article (its score, its full text, its PICO, its
    extraction, its concepts, its parameters, its place). It is never stale, it is
    PARTIAL: n of the N papers that should have it. The gap is the work left, and it is
    the number a reviewer needs before deciding to spend.

`tests/test_a_scenario_says_what_it_is_missing.py` pins the contract itself: every output
is declared with a readable name in both languages, a cost, the pipeline stage that
produces it and a way to rebuild it; every stored JSON column of `scenario_settings` is
either declared an output or listed as deliberately not one; and the pipeline's step list
covers every stage the declaration names. A new output cannot be added half wired,
because the suite fails.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import Query
from sqlalchemy import text

from .core import _norm_lang, app, engine, logger
from .digest import _rows
from .scenario_store import (CAP_MODEL, DEFAULT_SIMILARITY_THRESHOLD,
                             _get_scenario_threshold, _get_user_scenario_or_404,
                             capabilities_for, normalise_kind, relevant_gate_sql)

# ─────────────────────────────────────────────────────────────────────────────
# The vocabulary of the contract
# ─────────────────────────────────────────────────────────────────────────────
#: Les deux formes d'un résultat. Un ARTEFACT est construit une fois et rangé ; une
#: COUVERTURE est un fait mis en cache sur chaque article. Le premier peut être périmé,
#: la seconde ne peut qu'être partielle.
ARTEFACT, COVERAGE = "artefact", "coverage"

#: Ce que coûte de le refaire. `free` ne dépense rien (du SQL, ou des téléchargements) ;
#: `one_call` est un appel de modèle pour tout le scénario ; `one_pass` est un appel PAR
#: ARTICLE qui n'en a pas, donc la seule classe qu'il faut chiffrer avant de la lancer.
FREE, ONE_CALL, ONE_PASS = "free", "one_call", "one_pass"

#: Les états. `not_applicable` n'est pas un manque : une revue de littérature n'a pas de
#: modèle, et le dire est plus utile que de laisser une case vide. `unknown` non plus n'est
#: pas un manque : c'est la mesure qui a échoué, et la page le dit au lieu de tomber.
READY, PARTIAL, STALE, MISSING, EMPTY, NOT_APPLICABLE, UNKNOWN = (
    "ready", "partial", "stale", "missing", "empty", "not_applicable", "unknown")

#: Pourquoi un artefact n'est pas simplement « prêt ». Un CODE, et non une phrase : la
#: raison s'affiche sur une page qui a deux langues, et une phrase anglaise rendue telle
#: quelle dans l'interface française est une chaîne d'interface qui a échappé aux
#: fichiers de traduction. L'interface les traduit (`readiness.why.*`).
WHY_CORPUS_MOVED = "corpus_moved"           # le corpus a bougé après sa construction
WHY_NO_DATE = "no_date"                     # rien n'a noté quand il a été construit
WHY_DATE_UNREADABLE = "date_unreadable"     # une date qu'on ne sait pas lire
WHY_NOT_MEASURED = "not_measured"           # la lecture a échoué : on ne sait pas
WHY_CODES = (WHY_CORPUS_MOVED, WHY_NO_DATE, WHY_DATE_UNREADABLE, WHY_NOT_MEASURED)


def _artefact(key: str, column: str, stamp: str, stage: str, cost: str,
              name_en: str, name_fr: str, rebuild: str, tab: str,
              needs: str | None = None) -> dict[str, Any]:
    return {"key": key, "shape": ARTEFACT, "column": column, "stamp": stamp,
            "stage": stage, "cost": cost, "name_en": name_en, "name_fr": name_fr,
            "rebuild": rebuild, "tab": tab, "needs": needs}


def _coverage(key: str, expr: str, over: str, stage: str, cost: str,
              name_en: str, name_fr: str, rebuild: str, tab: str,
              needs: str | None = None) -> dict[str, Any]:
    """`expr` compte les articles qui PORTENT le fait ; `over` dit sur quel ensemble."""
    return {"key": key, "shape": COVERAGE, "expr": expr, "over": over,
            "stage": stage, "cost": cost, "name_en": name_en, "name_fr": name_fr,
            "rebuild": rebuild, "tab": tab, "needs": needs}


#: Les deux dénominateurs d'une couverture. `corpus` = tout ce que la recherche a ramené
#: (le score se mesure là, sinon la mesure serait circulaire : c'est lui qui fait la
#: pertinence) ; `relevant` = le sous-ensemble pertinent, pour tout le reste.
OF_CORPUS, OF_RELEVANT = "corpus", "relevant"

#: LA LISTE. Chaque ligne est un résultat qu'un scénario devrait porter, avec ce qui le
#: produit, ce qu'il coûte à refaire et où il se lit.
OUTPUTS: tuple[dict[str, Any], ...] = (
    # ── Ce qui est construit une fois et rangé ──────────────────────────────
    _artefact("brief_portrait", "evidence_brief_json", "evidence_brief_at",
              "evidence", FREE,
              "Corpus portrait", "Portrait du corpus",
              "POST /user-scenarios/{id}/pipeline", "evidence"),
    _artefact("brief_report", "evidence_brief_llm_json", "brief_generated_at",
              "evidence", ONE_CALL,
              "Evidence report", "Rapport de preuves",
              "POST /scenarios/{id}/evidence-brief/generate", "evidence"),
    _artefact("clustering", "clustering_json", "clustering_generated_at",
              "clustering", FREE,
              "Themes and clusters", "Thèmes et clusters",
              "POST /user-scenarios/{id}/pipeline", "viz"),
    _artefact("knowledge_graph", "knowledge_graph_json", "kg_generated_at",
              "knowledge_graph", FREE,
              "Similarity graph", "Graphe de similarité",
              "POST /user-scenarios/{id}/pipeline", "viz"),
    _artefact("concept_graph", "concept_graph_json", "concept_graph_generated_at",
              "knowledge_graph", ONE_PASS,
              "Concept map", "Carte des concepts",
              "POST /user-scenarios/{id}/pipeline", "viz"),
    _artefact("actions", "recommended_actions_json", "actions_generated_at",
              "actions", ONE_CALL,
              "Recommended actions", "Actions recommandées",
              "POST /user-scenarios/{id}/pipeline", "review"),
    _artefact("variables", "variables_json", "variables_generated_at",
              "variables", ONE_CALL,
              "Variables and model spec", "Variables et spec du modèle",
              "POST /scenarios/{id}/variables/generate", "variables",
              needs=CAP_MODEL),
    # Produit à la demande dans son onglet, et non par le pipeline : d'où `on_demand`.
    _artefact("seir", "seir_projection_json", "seir_projection_generated_at",
              "on_demand", FREE,
              "SEIR projection", "Projection SEIR",
              "POST /scenarios/{id}/seir/projection", "variables",
              needs=CAP_MODEL),

    # ── Ce qui est mis en cache article par article ─────────────────────────
    _coverage("scores", "ars.similarity_score IS NOT NULL", OF_CORPUS,
              "semantic_scoring", FREE,
              "Relevance scores", "Scores de pertinence",
              "POST /scenarios/{id}/rerank", "review"),
    _coverage("rerank", "ars.rerank_score IS NOT NULL", OF_CORPUS,
              "cross_encoder", FREE,
              "Cross-encoder scores", "Scores du cross-encoder",
              "POST /scenarios/{id}/rerank?missing_only=true", "review"),
    _coverage("fulltext", "d.has_fulltext IS TRUE", OF_RELEVANT,
              "fulltext", FREE,
              "Full texts", "Textes intégraux",
              "POST /fulltext/fetch", "enrichment"),
    _coverage("pico", "jsonb_typeof(d.pico_json) = 'object'", OF_RELEVANT,
              "pico", ONE_PASS,
              "PICO of each paper", "PICO de chaque article",
              "POST /pico/extract", "evidence"),
    _coverage("extraction", "jsonb_typeof(d.extraction_json) = 'object'", OF_RELEVANT,
              "extraction", ONE_PASS,
              "Extraction of each paper", "Extraction de chaque article",
              "POST /user-scenarios/{id}/extraction/run", "evidence"),
    _coverage("concepts", "jsonb_typeof(d.concepts_json->'concepts') = 'array'",
              OF_RELEVANT, "knowledge_graph", ONE_PASS,
              "Concepts of each paper", "Concepts de chaque article",
              "POST /user-scenarios/{id}/pipeline", "viz"),
    _coverage("epi_params", "jsonb_typeof(d.epi_params_json) = 'object'", OF_RELEVANT,
              "extraction", ONE_PASS,
              "Epidemiological parameters", "Paramètres épidémiologiques",
              "POST /scenarios/{id}/epidemic-parameters/extract", "evidence"),
    _coverage("place", "COALESCE(TRIM(d.country), '') <> ''", OF_RELEVANT,
              # Le pays est rempli par la passe MÉTADONNÉES, pas par l'extraction :
              # c'est elle qu'il faut relancer, et le dire juste évite d'envoyer un
              # relecteur appuyer sur le mauvais bouton.
              "metadata", ONE_PASS,
              "Country of each study", "Pays de chaque étude",
              "POST /metadata/extract", "evidence"),
)

OUTPUT_KEYS = tuple(o["key"] for o in OUTPUTS)

#: Les colonnes JSON de `scenario_settings` qui ne sont PAS un résultat, avec la raison.
#: Une colonne absente des deux listes fait échouer la suite : c'est le mécanisme qui
#: empêche d'ajouter un résultat sans le déclarer, et donc sans que personne ne puisse
#: voir qu'il manque.
NOT_AN_OUTPUT = {
    "codebook_json": "an input: the vocabulary the review asks for",
    "variables_proposal_json": "a draft of the spec, not the spec",
    "variables_i18n": "a translation cache of an output declared above",
    "seir_observed_json": "data a reviewer uploads, not something we produce",
}


# ─────────────────────────────────────────────────────────────────────────────
# Reading the state
# ─────────────────────────────────────────────────────────────────────────────
def _naive(t: Any) -> datetime | None:
    """Une date comparable, ou None si ce n'en est pas une.

    Deux colonnes de la même base peuvent être l'une avec fuseau et l'autre sans, et
    Python refuse alors de les comparer. Les dix dates que ce module rapproche ont été
    posées par autant de migrations sur quatre ans, et c'est arrivé : en production
    `article_scenarios.assigned_at` est en TIMESTAMPTZ (la colonne y préexistait, et
    `ADD COLUMN IF NOT EXISTS` ne l'a pas convertie) alors que
    `scenario_settings.similarity_threshold_set_at` est naïve. Le `max()` des deux levait
    TypeError, donc HTTP 500, sur tout scénario dont le seuil avait été réglé, et sur
    ceux-là seulement : les deux seuls scénarios qui répondaient étaient ceux dont le
    seuil n'avait jamais été touché. Le panneau qui doit dire ce qui manque ne peut pas
    tomber parce que deux dates ne s'accordent pas."""
    if not isinstance(t, datetime):
        return None
    return t.replace(tzinfo=None) if t.tzinfo is not None else t


def _counts(conn, scenario_id: str, thr: float) -> dict[str, int]:
    """Un seul passage pour toutes les couvertures, sur les deux dénominateurs."""
    gate = relevant_gate_sql("d", "ars", ":thr")
    cov = []
    for o in OUTPUTS:
        if o["shape"] != COVERAGE:
            continue
        scope = gate if o["over"] == OF_RELEVANT else "TRUE"
        cov.append(f"COUNT(*) FILTER (WHERE ({scope}) AND ({o['expr']})) AS {o['key']}")
    sql = f"""
        SELECT COUNT(*) AS n_corpus,
               COUNT(*) FILTER (WHERE {gate}) AS n_relevant,
               MAX(ars.assigned_at) AS last_arrival,
               MAX(ars.screened_at) AS last_decision,
               {', '.join(cov)}
        FROM literature_document d
        JOIN article_scenarios ars ON ars.document_id = d.id
        WHERE ars.scenario_id = :sid AND (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)
    """
    rows = _rows(conn, sql, scenario_id, thr)
    return dict(rows[0]) if rows else {}


def _settings(conn, scenario_id: str) -> dict[str, Any]:
    """La présence et la date de chaque artefact, sans rapatrier un seul octet de leur
    contenu : certains pèsent des centaines de kilooctets."""
    cols = []
    for o in OUTPUTS:
        if o["shape"] != ARTEFACT:
            continue
        cols.append(f"jsonb_typeof({o['column']}) = 'object' AS has_{o['key']}")
        cols.append(f"{o['stamp']} AS at_{o['key']}")
    rows = _rows(conn, f"""
        SELECT {', '.join(cols)}, similarity_threshold_set_at
        FROM scenario_settings WHERE scenario_id = :sid
    """, scenario_id, 0.0)
    return dict(rows[0]) if rows else {}


def _state_of(o: dict[str, Any], settings: dict[str, Any], counts: dict[str, int],
              touched, caps: frozenset, unread: set[str] | None = None) -> dict[str, Any]:
    """L'état d'UN résultat. PUR (les deux lectures sont déjà faites).

    `unread` nomme les formes dont la lecture a échoué : leurs lignes valent `unknown`,
    jamais « manquant ». Un panneau qui annonce un résultat absent parce qu'il n'a pas su
    le mesurer envoie refaire un travail déjà fait, ce qui est pire que de se taire."""
    out: dict[str, Any] = {k: o[k] for k in ("key", "shape", "stage", "cost", "tab")}
    out["needs"] = o["needs"]
    if o["needs"] and o["needs"] not in caps:
        return {**out, "state": NOT_APPLICABLE, "n": None, "of": None,
                "generated_at": None, "why": None, "rebuild": None}
    out["rebuild"] = o["rebuild"]
    if unread and o["shape"] in unread:
        return {**out, "state": UNKNOWN, "n": None, "of": None,
                "generated_at": None, "why": WHY_NOT_MEASURED}

    if o["shape"] == ARTEFACT:
        at = settings.get(f"at_{o['key']}")
        when = _naive(at)
        has = bool(settings.get(f"has_{o['key']}"))
        state, because = READY, None
        if not has:
            state = MISSING
        elif when and touched and when < touched:
            state, because = STALE, WHY_CORPUS_MOVED
        elif not when:
            state, because = STALE, (WHY_DATE_UNREADABLE if at else WHY_NO_DATE)
        return {**out, "state": state, "n": None, "of": None,
                "generated_at": when.isoformat() if when else None, "why": because}

    n = int(counts.get(o["key"]) or 0)
    of = int(counts.get("n_relevant" if o["over"] == OF_RELEVANT else "n_corpus") or 0)
    state = EMPTY if of == 0 else READY if n >= of else MISSING if n == 0 else PARTIAL
    return {**out, "state": state, "n": n, "of": of, "over": o["over"],
            "generated_at": None, "why": None}


def scenario_outputs(scenario_id: str, lang: str = "en",
                     threshold: float | None = None) -> dict[str, Any]:
    """Every output this scenario should hold, and whether it does.

    Read only: two SQL queries, no model call, nothing written. The point is that a
    missing output becomes a fact on a screen instead of a surprise in a meeting."""
    row = _get_user_scenario_or_404(scenario_id)
    kind = normalise_kind(row.get("kind") if isinstance(row, dict) else None)
    caps = capabilities_for(kind)
    english = _norm_lang(lang).startswith("en")

    # Chaque lecture peut échouer sans emporter les autres : un panneau de complétude qui
    # tombe est précisément le silence qu'il doit supprimer.
    counts: dict[str, Any] = {}
    settings: dict[str, Any] = {}
    errors: dict[str, str] = {}
    unread: set[str] = set()

    if threshold is not None:
        thr = float(threshold)
    else:
        try:
            thr = _get_scenario_threshold(scenario_id)
        except Exception as e:                               # noqa: BLE001
            logger.warning(f"scenario_outputs threshold {scenario_id}: {e}")
            errors["threshold"] = str(e)[:300]
            thr = DEFAULT_SIMILARITY_THRESHOLD

    with engine.connect() as conn:
        for shape, name, read in ((COVERAGE, "coverage", _counts),
                                  (ARTEFACT, "artefact", _settings)):
            try:
                got = read(conn, scenario_id, thr) if read is _counts else read(conn, scenario_id)
            except Exception as e:                           # noqa: BLE001
                logger.warning(f"scenario_outputs {name} {scenario_id}: {e}")
                errors[name] = str(e)[:300]
                unread.add(shape)
            else:
                if shape == COVERAGE:
                    counts = got
                else:
                    settings = got

    touched = max([t for t in (_naive(counts.get("last_arrival")),
                               _naive(counts.get("last_decision")),
                               _naive(settings.get("similarity_threshold_set_at"))) if t],
                  default=None)
    rows = []
    for o in OUTPUTS:
        state = _state_of(o, settings, counts, touched, caps, unread)
        state["name"] = o["name_en"] if english else o["name_fr"]
        rows.append(state)

    tally = {s: sum(1 for r in rows if r["state"] == s)
             for s in (READY, PARTIAL, STALE, MISSING, EMPTY, NOT_APPLICABLE, UNKNOWN)}
    # « À faire » ne compte que ce qu'une action changerait : un résultat sans objet
    # (corpus vide) ou hors de la nature du scénario n'est pas un manque.
    todo = [r["key"] for r in rows if r["state"] in (PARTIAL, STALE, MISSING)]
    return {"scenario_id": scenario_id, "kind": kind,
            "n_corpus": int(counts.get("n_corpus") or 0),
            "n_relevant": int(counts.get("n_relevant") or 0),
            "corpus_touched_at": touched.isoformat() if touched else None,
            "outputs": rows, "tally": tally, "todo": todo,
            # `complete` ne peut pas être vrai quand une mesure a échoué : on ne sait pas.
            "complete": not todo and not errors,
            "errors": errors,
            # Ce qu'une mise à jour coûterait, par classe : un relecteur décide avec ça.
            "cost_of_todo": {c: sum(1 for r in rows if r["key"] in todo and r["cost"] == c)
                             for c in (FREE, ONE_CALL, ONE_PASS)}}


@app.get("/user-scenarios/{scenario_id}/outputs")
def get_scenario_outputs(scenario_id: str, lang: str = Query("en", max_length=8),
                         threshold: float | None = Query(None, ge=0.0, le=1.0)) -> dict[str, Any]:
    """What this scenario should contain, and what it actually holds.

    One list, two SQL queries, no model call, nothing written. Each output says whether it
    is ready, partial, stale, missing, without object or outside this scenario's nature,
    when it was built, out of how many papers, what it costs to rebuild and where it is
    read. `todo` is the subset an action would change.

    A read that fails costs its own rows, not the answer: they come back `unknown`, the
    reason lands in `errors`, and `complete` is then false. A panel built to end a silence
    must not be able to become one."""
    return scenario_outputs(scenario_id, lang=lang, threshold=threshold)
