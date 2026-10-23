"""A scenario says what it is missing, so nobody has to notice it in a meeting.

Every output of this application had its own storage, its own freshness rule and its own
silent failure mode, and nothing anywhere said what a finished scenario holds. That is how
the evidence report came to be deleted by the very tab that displays it, for weeks, with no
test failing and no screen complaining: nothing was ever asked whether it was there. The
detector was a reviewer in front of partners, which is the most expensive detector there is.

So `api/outputs.py` declares the list once and answers it in SQL. These tests pin the
CONTRACT rather than the answers, because the contract is what stops the next output from
shipping half wired:

  1. every declared output is well formed: a readable name in both languages, a cost, the
     pipeline stage that builds it, a tab where it is read, and a way to rebuild it;
  2. every stage named is a real step of the full pipeline, or explicitly on demand;
  3. every rebuild route exists in the application;
  4. EVERY JSON column of `scenario_settings` is either declared an output or listed as
     deliberately not one. This is the enforcement: `evidence_brief_llm_json` could only
     be added today because the payload-size guard made its omission fail, and this does
     the same for the question "can anyone see whether it is there?";
  5. an artefact is missing when nothing is stored and stale when the corpus moved after
     it was built; a coverage is partial, never stale;
  6. what a scenario lacks is counted, priced by class, and `complete` means nothing is
     left that an action would change.
"""
import json
import pathlib
from datetime import datetime, timezone

import pytest

pytest.importorskip("fastapi")

import main  # noqa: E402
from api.outputs import (  # noqa: E402
    ARTEFACT,
    COVERAGE,
    EMPTY,
    FREE,
    MISSING,
    NOT_AN_OUTPUT,
    NOT_APPLICABLE,
    ONE_CALL,
    ONE_PASS,
    OUTPUTS,
    OUTPUT_KEYS,
    PARTIAL,
    READY,
    STALE,
    UNKNOWN,
    WHY_CODES,
    WHY_CORPUS_MOVED,
    WHY_NOT_MEASURED,
    scenario_outputs,
)
from api.scenario_store import CAP_MODEL  # noqa: E402

SID = "usr-outputs-test"
IDS = (9951, 9952, 9953)


# ── The contract itself ─────────────────────────────────────────────────────
def test_every_output_is_well_formed():
    """A row the page cannot render is a blank chip, which is worse than no chip: it reads
    as "nothing to see" when it means "nobody knows"."""
    assert len(set(OUTPUT_KEYS)) == len(OUTPUT_KEYS), "duplicate output key"
    for o in OUTPUTS:
        assert o["shape"] in (ARTEFACT, COVERAGE), o
        assert o["cost"] in (FREE, ONE_CALL, ONE_PASS), o
        assert o["name_en"] and o["name_fr"], o["key"]
        assert o["name_en"] != o["name_fr"] or len(o["name_en"]) <= 5, (
            f"{o['key']} reads the same in both languages, which usually means one was "
            "forgotten")
        assert o["tab"], o["key"]
        assert o["rebuild"].startswith("POST "), o["key"]
        assert o["needs"] in (None, CAP_MODEL), o["key"]
        if o["shape"] == ARTEFACT:
            assert o["column"].endswith("_json") and o["stamp"], o["key"]
        else:
            assert o["expr"] and o["over"] in ("corpus", "relevant"), o["key"]


def test_every_stage_is_a_real_pipeline_step():
    """An output whose stage is not in the pipeline is an output nothing produces. The one
    exception is declared as `on_demand`, which says so out loud."""
    import inspect

    import api.pipeline as pipeline

    source = inspect.getsource(pipeline)
    start = source.index("STEP_ORDER = [")
    steps = set(json.loads(source[start + len("STEP_ORDER = "):source.index("]", start) + 1]
                           .replace("'", '"')))
    assert "extraction" in steps and "evidence" in steps, steps
    for o in OUTPUTS:
        assert o["stage"] in steps or o["stage"] == "on_demand", (
            f"{o['key']} says it is built at stage {o['stage']!r}, which the pipeline does "
            f"not have: it would never be built automatically")


def test_every_rebuild_route_exists():
    """A rebuild instruction nobody can follow is worse than none: it sends a reviewer to
    an endpoint that answers 404."""
    paths = {r.path for r in main.app.routes if getattr(r, "path", "")}
    for o in OUTPUTS:
        route = o["rebuild"].split(" ", 1)[1].split("?")[0]
        template = route.replace("{id}", "{scenario_id}")
        assert template in paths, f"{o['key']} points at {route}, which is not a route"


def test_every_stored_json_column_is_classified():
    """THE enforcement. A new JSON column on scenario_settings is either an output someone
    can see the absence of, or explicitly not one, with a reason. Nothing in between: the
    whole bug class begins with a thing that is stored and never asked about."""
    import api.model_spec as model_spec
    import inspect

    ddl = inspect.getsource(model_spec)
    columns = set()
    for line in ddl.splitlines():
        if "scenario_settings ADD COLUMN IF NOT EXISTS" in line and "JSONB" in line:
            columns.add(line.split("IF NOT EXISTS")[1].split()[0])
    assert len(columns) >= 8, f"the DDL reader found too few columns: {columns}"

    declared = {o["column"] for o in OUTPUTS if o["shape"] == ARTEFACT}
    for column in sorted(columns):
        assert column in declared or column in NOT_AN_OUTPUT, (
            f"{column} is stored and classified nowhere. Declare it in OUTPUTS so its "
            f"absence is visible, or in NOT_AN_OUTPUT with the reason it is not a result.")
    for column in NOT_AN_OUTPUT:
        assert column not in declared, f"{column} is in both lists"
        assert NOT_AN_OUTPUT[column], f"{column} needs a reason"


def test_the_costly_outputs_are_the_ones_that_read_every_paper():
    """The classes have to mean something: `one_pass` is a call per paper and is the only
    class a reviewer must be shown a price for before pressing anything."""
    per_paper = {o["key"] for o in OUTPUTS if o["cost"] == ONE_PASS}
    # `place` is in here, which surprises until you look: a study's place is read FROM the
    # paper by the extraction pass, one model call each, not sitting in a column.
    assert {"extraction", "pico", "concepts", "epi_params", "place"} <= per_paper
    # And what is pure SQL or a plain download is never priced as if it were a call.
    free = {o["key"] for o in OUTPUTS if o["cost"] == FREE}
    assert {"brief_portrait", "scores", "fulltext"} <= free
    # The clustering is NOT free, which surprises until you look: the maths costs nothing
    # (cached embeddings, UMAP, HDBSCAN) but every group gets a written summary, one call
    # each, a dozen or so. A button announced as free that spends a dozen calls is exactly
    # what this classification exists to prevent.
    assert {o["key"]: o for o in OUTPUTS}["clustering"]["cost"] == ONE_CALL
    # Neither is the cross-encoder, for the same reason and found the same way: it is
    # Cohere's, and Cohere bills the query and its documents in batches of about a hundred
    # papers. The four partner reviews were short 9,772 of these scores, offered as a free
    # rebuild, which is a hundred or so billed calls behind a button that said it cost
    # nothing. `scores` above stays free: that one is pgvector, in the database.
    assert {o["key"]: o for o in OUTPUTS}["rerank"]["cost"] == ONE_PASS
    assert "rerank" not in free, "the cross-encoder bills, so it is never announced free"


# ── Over the database ───────────────────────────────────────────────────────
def _engine_ok() -> bool:
    try:
        with main.engine.connect():
            return True
    except Exception:                                        # noqa: BLE001
        return False


@pytest.fixture()
def seeded(db_conn):
    """Three papers: two relevant, one below the threshold. One of the relevant two has
    been extracted and has a country; neither has a PICO. No artefact at all."""
    if not _engine_ok():
        pytest.skip("main.engine cannot reach the database")
    from conftest import ensure_document_columns

    with db_conn.cursor() as cur:
        cur.execute("SELECT to_regclass('user_scenarios') IS NULL")
        if cur.fetchone()[0]:
            main._ensure_user_scenarios_table()
        cur.execute("CREATE TABLE IF NOT EXISTS literature_document (id BIGINT PRIMARY KEY)")
        cur.execute("CREATE TABLE IF NOT EXISTS article_scenarios ("
                    "scenario_id TEXT, document_id BIGINT, PRIMARY KEY (scenario_id, document_id))")
        created = ensure_document_columns(cur)
        for col, typ in (("title", "TEXT"), ("abstract", "TEXT"), ("country", "TEXT"),
                         ("has_fulltext", "BOOLEAN"), ("pico_json", "JSONB"),
                         ("extraction_json", "JSONB"), ("concepts_json", "JSONB"),
                         ("epi_params_json", "JSONB"),
                         ("is_duplicate", "BOOLEAN DEFAULT FALSE")):
            cur.execute(f"ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS {col} {typ}")
        for col, typ in (("similarity_score", "DOUBLE PRECISION"),
                         ("rerank_score", "DOUBLE PRECISION"),
                         ("assigned_at", "TIMESTAMP"), ("screened_at", "TIMESTAMP")):
            cur.execute(f"ALTER TABLE article_scenarios ADD COLUMN IF NOT EXISTS {col} {typ}")
        cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM scenario_settings WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))
        cur.execute("DELETE FROM literature_document WHERE id = ANY(%s)", (list(IDS),))
        cur.execute("INSERT INTO user_scenarios (id, name, query, mode, filters, pinned, kind) "
                    "VALUES (%s, 'Outputs', 'avian influenza', 'boolean', '{}', TRUE, 'review')",
                    (SID,))
        cur.execute("""
            INSERT INTO literature_document (id, title, abstract, source, is_duplicate,
                                             project_context, country, extraction_json)
            VALUES (9951, 'Read', 'A cross-sectional study of exposure among poultry cullers, with counts by sex.', 'pubmed', FALSE, 'literev',
                    'Italy', %s::jsonb),
                   -- `country` SANS extraction : la colonne de métadonnées est remplie,
                   -- mais personne n'a lu le lieu dans l'article. Le panneau Géographie
                   -- ne place pas celui-là, donc le registre ne doit pas le compter : si
                   -- cette ligne revenait à lire `d.country`, le lieu dirait 2 sur 2.
                   (9952, 'Not read', 'A cohort study of exposure at live bird markets, with seropositivity by stall.', 'pubmed', FALSE, 'literev',
                    'China', NULL),
                   (9953, 'Below', 'An unrelated paper about something else entirely, long enough to be scored.', 'pubmed', FALSE, 'literev', NULL, NULL)
        """, (json.dumps({"source": "abstract", "ref": {"location": "Lombardy, Italy"},
                          "observations": []}),))
        cur.execute("""
            INSERT INTO article_scenarios (scenario_id, document_id, similarity_score,
                                           rerank_score, assigned_at)
            VALUES (%s, 9951, 0.90, 0.8, NOW() - INTERVAL '2 days'),
                   (%s, 9952, 0.80, NULL, NOW() - INTERVAL '2 days'),
                   (%s, 9953, 0.10, NULL, NOW() - INTERVAL '2 days')
        """, (SID, SID, SID))
        cur.execute("INSERT INTO scenario_settings (scenario_id, similarity_threshold) "
                    "VALUES (%s, 0.45) ON CONFLICT (scenario_id) DO UPDATE "
                    "SET similarity_threshold = 0.45", (SID,))
    yield db_conn
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM scenario_settings WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))
        cur.execute("DELETE FROM literature_document WHERE id = ANY(%s)", (list(IDS),))
        if created:
            cur.execute("DROP TABLE document_chunk")


def _by_key(out: dict) -> dict:
    return {o["key"]: o for o in out["outputs"]}


def test_a_scenario_with_nothing_built_says_so_output_by_output(seeded):
    out = scenario_outputs(SID, lang="en")
    assert out["n_corpus"] == 3 and out["n_relevant"] == 2
    o = _by_key(out)
    # Artefacts: nothing stored, so missing, and named as a person reads them.
    assert o["brief_report"]["state"] == MISSING
    assert o["brief_report"]["name"] == "Evidence report"
    assert o["clustering"]["state"] == MISSING
    # Coverages: counted out of their own denominator.
    assert (o["scores"]["n"], o["scores"]["of"]) == (3, 3), "scores are of the corpus"
    assert o["scores"]["state"] == READY
    assert (o["rerank"]["n"], o["rerank"]["of"]) == (1, 3)
    assert o["rerank"]["state"] == PARTIAL
    assert (o["extraction"]["n"], o["extraction"]["of"]) == (1, 2), "of the relevant"
    assert o["extraction"]["state"] == PARTIAL
    assert o["pico"]["state"] == MISSING and o["pico"]["n"] == 0
    # Le lieu est celui que l'EXTRACTION a noté, celui que la carte lit. 9952 porte un
    # `country` de métadonnées et aucune extraction : il n'est pas placé. Cette ligne a
    # compté `d.country` pendant un jour et annonçait « 0 sur 881 » sur les quatre revues
    # des partenaires pendant que la carte en plaçait 630 sur 879.
    assert (o["place"]["n"], o["place"]["of"]) == (1, 2)
    assert o["place"]["stage"] == "extraction", "c'est la lecture qui le remplit"
    # And the summary agrees with the rows.
    assert out["complete"] is False
    assert set(out["todo"]) == {k for k, v in o.items()
                                if v["state"] in (PARTIAL, STALE, MISSING)}
    assert out["tally"][MISSING] >= 1 and out["tally"][PARTIAL] >= 2


def test_a_literature_review_is_not_missing_a_model(seeded):
    """`not_applicable` is not a gap. A review has no model, and saying that is more use
    than an empty square a reviewer reads as a failure."""
    out = scenario_outputs(SID, lang="en")
    o = _by_key(out)
    assert o["variables"]["state"] == NOT_APPLICABLE
    assert o["seir"]["state"] == NOT_APPLICABLE
    assert o["variables"]["rebuild"] is None, "nothing to press, so nothing offered"
    assert "variables" not in out["todo"] and "seir" not in out["todo"]


def test_an_artefact_built_before_the_corpus_moved_is_stale_not_ready(seeded):
    """The rule that matters for trust: an output older than the corpus is reported as
    old, with the reason, instead of being served as current."""
    with seeded.cursor() as cur:
        # Built two days ago, and a paper arrived yesterday.
        cur.execute("UPDATE scenario_settings SET clustering_json = %s::jsonb, "
                    "clustering_generated_at = NOW() - INTERVAL '2 days' "
                    "WHERE scenario_id = %s", (json.dumps({"clusters": []}), SID))
        cur.execute("UPDATE article_scenarios SET assigned_at = NOW() - INTERVAL '1 day' "
                    "WHERE scenario_id = %s AND document_id = 9952", (SID,))
    o = _by_key(scenario_outputs(SID, lang="en"))
    assert o["clustering"]["state"] == STALE
    assert o["clustering"]["why"] == WHY_CORPUS_MOVED
    assert o["clustering"]["generated_at"], "and it says when it was built"


def test_an_artefact_built_after_the_last_change_is_ready(seeded):
    with seeded.cursor() as cur:
        cur.execute("UPDATE scenario_settings SET clustering_json = %s::jsonb, "
                    "clustering_generated_at = NOW() WHERE scenario_id = %s",
                    (json.dumps({"clusters": [1]}), SID))
    o = _by_key(scenario_outputs(SID, lang="en"))
    assert o["clustering"]["state"] == READY and o["clustering"]["why"] is None


def test_what_is_missing_is_priced_by_class(seeded):
    """So that "bring it up to date" can say what it will spend before it spends it."""
    out = scenario_outputs(SID, lang="en")
    assert set(out["cost_of_todo"]) == {FREE, ONE_CALL, ONE_PASS}
    assert sum(out["cost_of_todo"].values()) == len(out["todo"])
    assert out["cost_of_todo"][ONE_PASS] >= 2, "pico and the others read every paper"


def test_an_empty_scenario_reports_no_object_rather_than_a_gap(seeded):
    """A corpus with nothing relevant has nothing to extract. That is `empty`, and it is
    not work: a list of fourteen red dots on a scenario nobody has searched yet would
    teach reviewers to ignore the panel."""
    with seeded.cursor() as cur:
        cur.execute("UPDATE article_scenarios SET similarity_score = 0.01 "
                    "WHERE scenario_id = %s", (SID,))
    out = scenario_outputs(SID, lang="en")
    o = _by_key(out)
    assert out["n_relevant"] == 0
    assert o["extraction"]["state"] == "empty" and o["extraction"]["of"] == 0
    assert "extraction" not in out["todo"]


def test_a_measurement_that_fails_reads_unknown_and_never_missing(seeded, monkeypatch):
    """The panel exists to end a silence, so it must not become one. When a read fails,
    its rows say `unknown` with the reason, the other half is still measured, and
    `complete` cannot be true: "missing" would send someone to redo work already done,
    which is worse than saying nothing."""
    import api.outputs as outputs

    def _boom(*a, **kw):                                     # noqa: ANN002, ANN003
        raise RuntimeError("column the_pipeline_forgot does not exist")

    monkeypatch.setattr(outputs, "_counts", _boom)
    out = outputs.scenario_outputs(SID, lang="en")
    o = _by_key(out)
    assert o["extraction"]["state"] == "unknown"
    assert o["extraction"]["why"] == WHY_NOT_MEASURED
    assert o["brief_report"]["state"] == MISSING, "the artefacts were still read"
    assert out["complete"] is False
    assert "the_pipeline_forgot" in out["errors"]["coverage"]
    assert "extraction" not in out["todo"], "unknown is not work: we do not know yet"


def test_two_dates_that_disagree_about_the_timezone_do_not_fell_the_panel(seeded):
    """The eight dates this module compares were added by eight migrations over four
    years, and one of them carrying a timezone is enough for Python to refuse the
    comparison. The panel loses that staleness, never the page."""
    import api.outputs as outputs

    aware = {"at_clustering": datetime(2025, 1, 1, tzinfo=timezone.utc),
             "has_clustering": True}
    naive = datetime(2025, 6, 1)                              # the corpus moved later
    o = outputs._state_of(
        next(x for x in OUTPUTS if x["key"] == "clustering"),
        aware, {}, naive, frozenset(), None)
    assert o["state"] == STALE and o["why"] == WHY_CORPUS_MOVED
    assert o["generated_at"] == "2025-01-01T00:00:00", "and the date is still readable"


def test_a_threshold_that_cannot_be_read_falls_back_and_says_so(seeded, monkeypatch):
    """The threshold decides what counts as relevant, so every coverage denominator hangs
    on it. When it cannot be read the page still renders, on the default, and names the
    failure rather than quietly reporting figures from a threshold nobody chose."""
    import api.outputs as outputs

    def _boom(*a, **kw):                                      # noqa: ANN002, ANN003
        raise RuntimeError("scenario_settings is being migrated")

    monkeypatch.setattr(outputs, "_get_scenario_threshold", _boom)
    out = outputs.scenario_outputs(SID, lang="en")
    assert out["n_corpus"] == 3, "the corpus was still counted"
    assert "being migrated" in out["errors"]["threshold"]
    assert out["complete"] is False


def _readiness_block(lang: str) -> str:
    """Le bloc `readiness` du fichier de traduction, par appariement d'accolades."""
    src = pathlib.Path(f"frontend/src/i18n/locales/{lang}.ts").read_text(encoding="utf-8")
    i = src.index("readiness: {")
    depth, j = 0, src.index("{", i)
    for k in range(j, len(src)):
        depth += 1 if src[k] == "{" else -1 if src[k] == "}" else 0
        if depth == 0:
            return src[j:k]
    raise AssertionError("unbalanced readiness block")


@pytest.mark.parametrize("lang", ["en", "fr"])
def test_every_state_and_reason_has_a_label_in_both_languages(lang):
    """A state the interface cannot name renders as its own key, in both languages, which
    is how `unknown` shipped with "scenarioDetail.readiness.state.unknown" in a cell. The
    vocabulary lives in Python, so Python checks that the labels exist."""
    block = _readiness_block(lang)
    for code in (READY, PARTIAL, STALE, MISSING, EMPTY, NOT_APPLICABLE, UNKNOWN):
        assert f"{code}:" in block, f"{lang}: no label for the state '{code}'"
    for code in WHY_CODES:
        assert f"{code}:" in block, f"{lang}: no label for the reason '{code}'"


def test_the_names_are_served_in_the_language_asked(seeded):
    en = _by_key(scenario_outputs(SID, lang="en"))
    fr = _by_key(scenario_outputs(SID, lang="fr"))
    assert en["brief_report"]["name"] == "Evidence report"
    assert fr["brief_report"]["name"] == "Rapport de preuves"


# ── The endpoint ────────────────────────────────────────────────────────────
def test_the_endpoint_is_read_only_and_spends_nothing(seeded, monkeypatch):
    from fastapi.testclient import TestClient

    import llm_usage

    def _explode(*a, **kw):                                  # noqa: ANN002, ANN003
        raise AssertionError("the readiness panel spent a model call")

    monkeypatch.setattr(llm_usage, "MeteredOpenAI", _explode)
    r = TestClient(main.app).get(f"/user-scenarios/{SID}/outputs", params={"lang": "en"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["outputs"]) == len(OUTPUTS)
    assert body["n_relevant"] == 2 and body["complete"] is False
    assert body["corpus_touched_at"], "and when the corpus last moved"


def test_an_unknown_scenario_is_a_404():
    from fastapi.testclient import TestClient

    r = TestClient(main.app).get("/user-scenarios/usr-no-such-thing/outputs")
    assert r.status_code == 404


# ── Rebuilding what is missing ──────────────────────────────────────────────
HDR = {"X-API-Key": "test-write-key"}


@pytest.fixture(autouse=True)
def _no_job_left_behind():
    """Les travaux de rebuild vivent en mémoire : un test qui en laisse un « en cours »
    ferait répondre « déjà en cours » au suivant."""
    import api.outputs as outputs

    outputs._REBUILD_JOBS.pop(SID, None)
    yield
    outputs._REBUILD_JOBS.pop(SID, None)


def test_every_rebuildable_output_is_an_artefact_with_a_declared_cost():
    """The button can only rebuild what is built once and stored. A coverage has its own
    endpoint, its own progress and its own invoice, and an "update everything" that
    quietly starts one model call per paper is the opposite of what the panel is for."""
    from api.outputs import REBUILDABLE

    by_key = {o["key"]: o for o in OUTPUTS}
    for key in REBUILDABLE:
        assert key in by_key, f"{key} is rebuildable but not declared"
        assert by_key[key]["shape"] == ARTEFACT, f"{key} is a coverage: it has its own route"
    # And every artefact is rebuildable: one that is not would be declared missing with
    # no way to act on it, which is the silence this whole module exists to end.
    assert {o["key"] for o in OUTPUTS if o["shape"] == ARTEFACT} == set(REBUILDABLE)


def test_a_free_rebuild_refuses_what_costs_a_model_call_and_says_so(seeded, monkeypatch):
    """`spend=free` is the default, and it is the point: a reviewer pressing "update" must
    not discover afterwards that it spent. What it refuses it names, with the class that
    would be needed."""
    import api.outputs as outputs

    started: list[list[str]] = []
    monkeypatch.setattr(outputs.threading, "Thread",
                        lambda target, args, daemon: type("T", (), {"start": lambda s: started.append(list(args[1]))})())
    from fastapi.testclient import TestClient

    r = TestClient(main.app).post(f"/user-scenarios/{SID}/outputs/rebuild", headers=HDR)
    assert r.status_code == 200, r.text
    body = r.json()
    costly = {x["key"]: x for x in body["refused"]}
    assert "brief_report" in costly and costly["brief_report"]["cost"] == ONE_CALL
    assert "concept_graph" in costly and costly["concept_graph"]["cost"] == ONE_PASS
    assert all(x["why"] for x in body["refused"]), "a refusal with no reason teaches nothing"
    # The free ones did start, and nothing else.
    assert started and set(started[0]) <= {"brief_portrait", "knowledge_graph"}


def test_a_rebuild_never_touches_a_coverage(seeded, monkeypatch):
    """Asking for the extraction by name is refused with its own route, rather than
    silently starting a reading pass over every paper."""
    import api.outputs as outputs

    monkeypatch.setattr(outputs.threading, "Thread",
                        lambda target, args, daemon: type("T", (), {"start": lambda s: None})())
    from fastapi.testclient import TestClient

    r = TestClient(main.app).post(f"/user-scenarios/{SID}/outputs/rebuild",
                                  params={"keys": "extraction", "spend": "one_pass"}, headers=HDR)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "nothing_to_do"
    assert body["refused"][0]["key"] == "extraction"
    assert "own endpoint" in body["refused"][0]["why"]
    assert body["refused"][0]["rebuild"], "and it says where to press instead"


def test_a_rebuild_of_an_unknown_output_is_refused_by_name(seeded):
    from fastapi.testclient import TestClient

    r = TestClient(main.app).post(f"/user-scenarios/{SID}/outputs/rebuild",
                                  params={"keys": "nonsense"}, headers=HDR)
    assert r.status_code == 422 and "nonsense" in r.text


def test_a_rebuild_needs_the_key(seeded):
    """It writes, so it is keyed like every other write."""
    from fastapi.testclient import TestClient

    r = TestClient(main.app).post(f"/user-scenarios/{SID}/outputs/rebuild")
    assert r.status_code == 401, r.text


def test_a_rebuild_waits_for_a_full_pipeline_instead_of_racing_it(seeded, monkeypatch):
    """Both write the same columns. Letting them run together is how an artefact ends up
    describing whichever corpus finished last, which is the bug this module was born of."""
    from api.scenarios import _user_scenario_pipeline_jobs
    from fastapi.testclient import TestClient

    _user_scenario_pipeline_jobs[SID] = {"overall_status": "running", "current_step": "pico"}
    try:
        r = TestClient(main.app).post(f"/user-scenarios/{SID}/outputs/rebuild", headers=HDR)
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "pipeline_running"
    finally:
        _user_scenario_pipeline_jobs.pop(SID, None)


def test_the_rebuild_status_is_readable_before_anything_ran(seeded):
    from fastapi.testclient import TestClient

    r = TestClient(main.app).get(f"/user-scenarios/{SID}/outputs/rebuild/status")
    assert r.status_code == 200 and r.json()["status"] == "not_started"


def test_a_rebuild_whose_thread_died_does_not_block_the_next_one(seeded, monkeypatch):
    """The lock lives in memory, so a thread that dies without writing its end would
    forbid rebuilding this scenario until the next restart: a block nobody discovers until
    they need it. Past the deadline the stuck job is abandoned and the next one starts."""
    import api.outputs as outputs

    monkeypatch.setattr(outputs.threading, "Thread",
                        lambda target, args, daemon: type("T", (), {"start": lambda s: None})())
    from fastapi.testclient import TestClient

    c = TestClient(main.app)
    first = c.post(f"/user-scenarios/{SID}/outputs/rebuild", headers=HDR)
    assert first.json()["status"] == "started"
    assert c.post(f"/user-scenarios/{SID}/outputs/rebuild", headers=HDR).json()["status"] \
        == "already_running", "two rebuilds at once write the same columns"

    outputs._REBUILD_JOBS[SID]["started_at"] -= outputs._REBUILD_STALE_S + 1
    again = c.post(f"/user-scenarios/{SID}/outputs/rebuild", headers=HDR)
    assert again.json()["status"] == "started"


def test_every_name_the_rebuild_calls_actually_exists():
    """A lazy import inside a background thread is invisible until it runs, and then only
    in a log. Three such names were undefined in the pipeline for weeks: the cross-encoder
    step, the PICO step and the metadata step each raised NameError on every full run,
    caught by the step's own try and filed as "error", so a quarter of the corpus carried
    metadata and nobody knew why. The list is repeated here on purpose: it is the contract
    that a rename has to break the suite rather than a production run."""
    import importlib

    for mod, name in (("api.evidence", "_build_evidence_brief"),
                      ("api.evidence", "store_evidence_brief"),
                      ("api.evidence", "_generate_evidence_brief_llm"),
                      ("api.clustering", "_run_clustering_background"),
                      ("api.knowledge_graph", "_precompute_user_kg"),
                      ("api.knowledge_graph", "_precompute_concept_graph"),
                      ("api.actions", "_generate_recommended_actions"),
                      ("api.variables", "_generate_variables_from_pico"),
                      ("api.seir", "_precompute_seir_projection")):
        got = getattr(importlib.import_module(mod), name, None)
        assert callable(got), f"{mod}.{name} is called by the rebuild and does not exist"


def test_the_rebuild_calls_one_function_per_artefact():
    """And the map covers them: a new artefact added to the declaration without a way to
    rebuild it would be reported missing with nothing to press."""
    from api.outputs import _rebuild_steps

    steps = _rebuild_steps("usr-none", "en")
    for o in OUTPUTS:
        if o["shape"] == ARTEFACT:
            assert callable(steps.get(o["key"])), f"{o['key']} has no rebuild"
