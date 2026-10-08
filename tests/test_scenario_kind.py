"""Not every research question ends in a model.

Many end in a synthesis: what the literature establishes, with what certainty, and
what is missing. For those, the predictive half of the application is not merely
decoration, it costs a model pass per scenario that invents candidate predictor
variables nobody will ever fit (api/pipeline.py, step "variables").

So a scenario declares its nature, and the nature decides. Two properties matter
more than any other and are pinned here:

- the cut is one-way. The review half is a strict subset the predictive half is
  built on, so a gate only ever WITHHOLDS the predictive half. Nothing is ever
  taken from a predictive scenario, and NULL in the column means "everything", so
  no row that already exists changes behaviour;
- the pooled epidemiological parameters stay with the review. They read without a
  model, without an LLM call, over the whole relevant corpus, and a quality
  weighted R0 carrying the provenance of each study IS a review deliverable. Filing
  them with the forecast would take them from a parameter meta-analysis, which is
  precisely a review.
"""
import pytest

pytest.importorskip("fastapi")

import main  # noqa: E402
from api.scenario_store import (  # noqa: E402
    CAP_FIELD_DATA,
    CAP_MODEL,
    CAPABILITIES,
    DEFAULT_KIND,
    KIND_PREDICTIVE,
    KIND_REVIEW,
    KINDS,
    capabilities_for,
    capability_refusal,
    kind_has,
    normalise_kind,
)


# ── the vocabulary (pure) ────────────────────────────────────────────────────

def test_there_are_two_natures_and_no_third_state():
    assert KINDS == (KIND_REVIEW, KIND_PREDICTIVE)


def test_an_unset_nature_means_everything_as_before():
    """The column is nullable and NULL is the old behaviour: no backfill, no surprise."""
    assert DEFAULT_KIND == KIND_PREDICTIVE
    assert normalise_kind(None) == KIND_PREDICTIVE
    assert capabilities_for(None) == frozenset(CAPABILITIES)


def test_an_unreadable_value_opens_rather_than_closes():
    """A typo in the database must not silently hide half of someone's workspace."""
    for junk in ("", "  ", "reviewww", "PREDICTIVE?", "null"):
        assert normalise_kind(junk) == KIND_PREDICTIVE


def test_a_nature_written_in_any_case_is_still_read():
    assert normalise_kind(" Review ") == KIND_REVIEW
    assert normalise_kind("PREDICTIVE") == KIND_PREDICTIVE


def test_the_cut_only_ever_subtracts():
    """Review capabilities are a strict subset of predictive ones."""
    review, predictive = capabilities_for(KIND_REVIEW), capabilities_for(KIND_PREDICTIVE)
    assert review < predictive
    assert predictive == frozenset(CAPABILITIES)


def test_only_two_things_are_ever_withheld():
    """Everything else belongs to both natures, so there is nothing else to check."""
    assert set(CAPABILITIES) == {CAP_MODEL, CAP_FIELD_DATA}


def test_a_review_withholds_the_model_half_and_the_field_reports():
    assert kind_has(KIND_REVIEW, CAP_MODEL) is False
    assert kind_has(KIND_REVIEW, CAP_FIELD_DATA) is False


def test_a_predictive_scenario_is_never_short_of_anything():
    for cap in CAPABILITIES:
        assert kind_has(KIND_PREDICTIVE, cap) is True
        assert kind_has(None, cap) is True


def test_an_unknown_capability_is_a_programming_error_not_a_refusal():
    """Asking the wrong question must fail loudly, not answer "no"."""
    with pytest.raises(ValueError):
        kind_has(KIND_REVIEW, "epidemic_parameters")


def test_the_pooled_epidemiological_parameters_are_not_a_gated_capability():
    """They read with no model and no LLM call, and a pooled R0 with its provenance
    is a review deliverable. Filing them with the forecast would take them from a
    parameter meta-analysis, which is a review."""
    assert "epidemic_parameters" not in CAPABILITIES
    assert "parameters" not in " ".join(CAPABILITIES)


# ── the refusal ──────────────────────────────────────────────────────────────

def test_a_refusal_speaks_the_shape_the_api_already_uses():
    """`applicable: false` plus a stable reason_code, as api/seir.py answers. Not a
    4xx: the question simply does not call for this half, which is not a failure."""
    r = capability_refusal("s1", CAP_MODEL)
    assert r["applicable"] is False
    assert r["reason_code"] == "review_scenario"
    assert r["scenario_id"] == "s1" and r["capability"] == CAP_MODEL
    assert isinstance(r["reason"], str) and len(r["reason"]) > 40


def test_a_refusal_says_the_change_is_reversible_and_destroys_nothing():
    assert "supprim" in capability_refusal("s1", CAP_MODEL)["reason"]


# ── the database ─────────────────────────────────────────────────────────────

SID = "kind-check"


def _engine_ok() -> bool:
    try:
        with main.engine.connect():
            return True
    except Exception:
        return False


@pytest.fixture()
def scenario(db_conn):
    if not _engine_ok():
        pytest.skip("main.engine cannot reach the database")
    with db_conn.cursor() as cur:
        # Le détail compte le corpus : les tables doivent exister, même vides, sinon
        # c'est le comptage qui échoue et non la nature que ce fichier teste.
        cur.execute("DROP TABLE IF EXISTS document_search")
        cur.execute("DROP TABLE IF EXISTS document_chunk, article_scenarios, literature_document CASCADE")
        cur.execute(
            "CREATE TABLE literature_document ("
            "id bigint PRIMARY KEY, title text NOT NULL, source text NOT NULL DEFAULT 'pubmed',"
            "abstract text, year int, is_duplicate boolean DEFAULT false, screening_status text,"
            "journal text, created_at timestamptz DEFAULT now(),"
            "project_context text DEFAULT 'literev')")
        cur.execute(
            "CREATE TABLE document_chunk ("
            "id bigint GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,"
            "document_id bigint NOT NULL REFERENCES literature_document(id) ON DELETE CASCADE,"
            "chunk_index int DEFAULT 0, content text, chunk_type text, embedding text)")
        cur.execute(
            "CREATE TABLE article_scenarios (scenario_id text, document_id bigint,"
            "similarity_score double precision, rerank_score double precision,"
            "screening_status text, PRIMARY KEY (scenario_id, document_id))")
        cur.execute("SELECT to_regclass('user_scenarios') IS NULL")
        if cur.fetchone()[0]:
            main._ensure_user_scenarios_table()
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))
        cur.execute(
            "INSERT INTO user_scenarios (id, name, query, mode, filters) "
            "VALUES (%s, 'Kind', 'rsv AND surveillance', 'boolean', '{}')", (SID,))
    yield db_conn
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))


def test_a_scenario_created_before_the_column_existed_keeps_everything(scenario):
    assert main.scenario_kind(SID) == KIND_PREDICTIVE
    assert main.scenario_can(SID, CAP_MODEL) is True


def test_a_scenario_that_does_not_exist_is_not_refused_by_the_gate(scenario):
    """A capability check must not be the thing that reports a missing scenario."""
    assert main.scenario_kind("no-such-scenario") == KIND_PREDICTIVE


def test_the_nature_is_switched_through_the_patch_and_takes_effect(scenario):
    main.patch_user_scenario(SID, main.UserScenarioPatch(kind=KIND_REVIEW))
    assert main.scenario_kind(SID) == KIND_REVIEW
    assert main.scenario_can(SID, CAP_MODEL) is False
    main.patch_user_scenario(SID, main.UserScenarioPatch(kind=KIND_PREDICTIVE))
    assert main.scenario_can(SID, CAP_MODEL) is True


def test_switching_back_is_allowed_and_nothing_is_deleted_on_the_way(scenario, db_conn):
    """The spec survives a round trip through the review nature: changing one's mind
    must not cost the work already paid for."""
    with db_conn.cursor() as cur:
        cur.execute("SELECT to_regclass('scenario_settings') IS NULL")
        if cur.fetchone()[0]:
            main._ensure_scenario_settings_table()
        cur.execute(
            "INSERT INTO scenario_settings (scenario_id, variables_json) VALUES (%s, %s) "
            "ON CONFLICT (scenario_id) DO UPDATE SET variables_json = EXCLUDED.variables_json",
            (SID, '{"model_spec": {"outcome": "kept"}}'))
    main.patch_user_scenario(SID, main.UserScenarioPatch(kind=KIND_REVIEW))
    main.patch_user_scenario(SID, main.UserScenarioPatch(kind=KIND_PREDICTIVE))
    with db_conn.cursor() as cur:
        cur.execute("SELECT variables_json FROM scenario_settings WHERE scenario_id = %s", (SID,))
        kept = cur.fetchone()[0]
    assert kept and kept["model_spec"]["outcome"] == "kept"
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM scenario_settings WHERE scenario_id = %s", (SID,))


def test_an_unknown_nature_is_refused_rather_than_stored(scenario):
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as e:
        main.patch_user_scenario(SID, main.UserScenarioPatch(kind="prediction"))
    assert e.value.status_code == 400
    assert main.scenario_kind(SID) == KIND_PREDICTIVE       # et rien n'a bougé


def test_the_detail_payload_carries_the_nature_and_its_capabilities(scenario):
    d = main.get_user_scenario_detail(SID)
    assert d["kind"] == KIND_PREDICTIVE
    assert sorted(d["capabilities"]) == sorted(CAPABILITIES)
    main.patch_user_scenario(SID, main.UserScenarioPatch(kind=KIND_REVIEW))
    d = main.get_user_scenario_detail(SID)
    assert d["kind"] == KIND_REVIEW and d["capabilities"] == []


def test_generating_variables_on_a_review_answers_not_applicable(scenario):
    """And answers it without starting the thread that would have paid for it."""
    main.patch_user_scenario(SID, main.UserScenarioPatch(kind=KIND_REVIEW))
    out = main.generate_scenario_variables(SID)
    assert out["status"] == "not_applicable"
    assert out["applicable"] is False and out["reason_code"] == "review_scenario"
