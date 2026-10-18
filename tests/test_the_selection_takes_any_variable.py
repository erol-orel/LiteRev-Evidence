"""Narrowing the corpus on ANY variable of the register, not just the five named dimensions.

The mechanism was already right and is untouched: a narrowing is a screening decision, so it
is written where screening is written, it shows in PRISMA with its reason, and it is undone
by removing only the rows it wrote. What changes is how many questions can ask for it. Five
dimensions had a button (theme, concept, design, evidence level, country); the other twenty
variables of the register had none, for no reason but that nobody wrote them.

Four things are worth pinning here.

1. A CONDITION THAT CANNOT JUDGE AN ARTICLE NEITHER SAVES NOR CONDEMNS IT. A paper with no
   value for the variable is `unknown`, exactly as a paper outside the capped clustering is.
   Confounding "no value" with "no" would empty a corpus on the strength of an absence, which
   is the one mistake this whole module exists to avoid.

2. THE FIVE NAMED DIMENSIONS KEEP THEIR OWN SEMANTICS. Asking for a country as a condition
   and asking for it as `countries` must do exactly the same thing, Europe's expansion
   included, because they are the same resolution and not two implementations.

3. A CONDITION READS THE VARIABLE THE WAY THE PAGE COUNTED IT. The register and the
   selection share one expression per variable, so a reviewer can never filter on a reading
   different from the one they were shown.

4. THE REASON NAMES THE VARIABLE AND ITS VALUES IN CLEAR. A methods section has to be able
   to say what was dropped; "scope: hors de language en" is readable, an opaque key is not.
"""
import json

import pytest

pytest.importorskip("fastapi")

import main  # noqa: E402
from api.subsets import SUBSET_REASON_PREFIX, describe_selection, plan_subset  # noqa: E402
from api.variables_of_interest import (  # noqa: E402
    ROUTED_TO_SUBSETS,
    condition_verdict,
    filterable_keys,
)
from conftest import ensure_document_columns  # noqa: E402

SID = "usr-selection-test"
IDS = (9801, 9802, 9803, 9804)


# ── Pure: one condition's verdict ───────────────────────────────────────────
def test_a_paper_with_no_value_is_unknown_not_out():
    """The rule the whole module rests on. Paper 3 has no language recorded: the condition
    has nothing to say about it, so it does not vote, and the paper survives on the other
    conditions' verdicts or on the `unassigned` rule."""
    verdict = condition_verdict({1: "en", 2: "fr"}, "value", {"en"})
    assert verdict(1) == "in"
    assert verdict(2) == "out"
    assert verdict(3) == "unknown"


def test_a_multi_valued_variable_matches_on_any_of_its_values():
    """A paper reports two occupations. Asking for one of them keeps the paper: a group's
    values are not exclusive, and requiring the whole set would answer almost never."""
    of = {1: {"poultry_worker", "veterinarian"}, 2: {"farmer"}, 3: set()}
    verdict = condition_verdict(of, "multi", {"poultry_worker"})
    assert verdict(1) == "in"
    assert verdict(2) == "out"
    assert verdict(3) == "unknown", "an empty set is no answer, not a negative one"


def test_a_number_takes_an_interval_and_an_open_bound_stays_open():
    of = {1: 2014, 2: 2015, 3: 2026}
    verdicts = lambda lo, hi: [condition_verdict(of, "number", None, lo, hi)(i) for i in (1, 2, 3)]
    assert verdicts(2015, None) == ["out", "in", "in"], "a bound is inclusive"
    assert verdicts(None, 2015) == ["in", "in", "out"]
    assert verdicts(2015, 2025) == ["out", "in", "out"]
    assert condition_verdict(of, "number", None, 2015, None)(4) == "unknown"


def test_conditions_and_the_named_dimensions_combine_as_one_selection():
    """A generic condition is one more voice in the same vote, not a second mechanism.
    Under `all` a paper must satisfy every condition that can judge it; the paper the
    language condition cannot judge is carried by the cluster alone."""
    cluster_of = {1: 0, 2: 0, 3: 0}
    lang = condition_verdict({1: "en", 2: "fr"}, "value", {"en"})
    plan = plan_subset([1, 2, 3], cluster_of=cluster_of, clusters_wanted={0},
                       conditions=[("language", lang)], combine="all")
    assert plan["keep_ids"] == [1, 3]
    assert plan["exclude_ids"] == [2]
    assert plan["by_dimension"]["language"] == {"in": 1, "out": 1, "unknown": 1}
    assert plan["by_dimension"]["clusters"] == {"in": 3, "out": 0, "unknown": 0}


def test_a_selection_of_conditions_alone_is_a_selection():
    """Before this, a payload with no cluster, concept, design, level or country was refused.
    A condition on a variable is a dimension like any other and stands on its own."""
    lang = condition_verdict({1: "en", 2: "fr"}, "value", {"en"})
    plan = plan_subset([1, 2], conditions=[("language", lang)])
    assert plan["keep"] == 1 and plan["exclude"] == 1


def test_the_reason_names_every_condition_in_clear():
    reason = describe_selection(None, None, None, "all", "keep", threshold=0.4887,
                                conditions=["language en, fr", "year à partir de 2015"])
    assert reason.startswith(f"{SUBSET_REASON_PREFIX} (seuil 0.4887) hors de ")
    assert "language en, fr" in reason and "year à partir de 2015" in reason
    assert " et " in reason, "under 'all' the conditions are joined by 'and'"


def test_a_routed_variable_is_offered_but_never_resolved_twice():
    """The five named dimensions are offered on the page like any other variable, and the
    generic resolver refuses them: exactly one resolution per variable, so "country" cannot
    mean one thing in a condition and another in the country box."""
    from api.variables_of_interest import variable_membership

    assert set(ROUTED_TO_SUBSETS) == {"cluster", "design", "grade_level", "country",
                                      "europe", "concepts"}
    offered = filterable_keys()
    for var in ("cluster", "design", "grade_level", "country", "europe"):
        assert var in offered, f"{var} must still be offered to a reviewer"
        with pytest.raises(ValueError, match="api/subsets"):
            variable_membership("usr-whatever", var)


def test_the_gate_and_the_mechanism_itself_are_never_filterable():
    """"Keep what is above the threshold" is what the gate already does, and "keep what is
    out of scope" would ask to keep excluded papers. Neither is a filter."""
    keys = filterable_keys()
    assert "above_threshold" not in keys and "scope" not in keys


# ── Over the database ───────────────────────────────────────────────────────
def _engine_ok() -> bool:
    try:
        with main.engine.connect():
            return True
    except Exception:                                        # noqa: BLE001
        return False


@pytest.fixture()
def seeded(db_conn):
    if not _engine_ok():
        pytest.skip("main.engine cannot reach the database")
    with db_conn.cursor() as cur:
        cur.execute("SELECT to_regclass('user_scenarios') IS NULL")
        if cur.fetchone()[0]:
            main._ensure_user_scenarios_table()
        cur.execute("CREATE TABLE IF NOT EXISTS literature_document (id BIGINT PRIMARY KEY)")
        cur.execute("CREATE TABLE IF NOT EXISTS article_scenarios ("
                    "scenario_id TEXT, document_id BIGINT, PRIMARY KEY (scenario_id, document_id))")
        created_chunk_table = ensure_document_columns(cur)
        for col, typ in (("title", "TEXT"), ("abstract", "TEXT"), ("year", "INTEGER"),
                         ("language", "TEXT"), ("journal", "TEXT"), ("doi", "TEXT"),
                         ("citation_count", "INTEGER"), ("extraction_json", "JSONB"),
                         ("is_duplicate", "BOOLEAN DEFAULT FALSE")):
            cur.execute(f"ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS {col} {typ}")
        for col, typ in (("similarity_score", "DOUBLE PRECISION"), ("screening_reason", "TEXT"),
                         ("screening_notes", "TEXT"), ("screened_at", "TIMESTAMP")):
            cur.execute(f"ALTER TABLE article_scenarios ADD COLUMN IF NOT EXISTS {col} {typ}")
        cur.execute("ALTER TABLE document_chunk ADD COLUMN IF NOT EXISTS embedding TEXT")
        cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM scenario_settings WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))
        cur.execute("DELETE FROM literature_document WHERE id = ANY(%s)", (list(IDS),))
        cur.execute("INSERT INTO user_scenarios (id, name, query, mode, filters, pinned) "
                    "VALUES (%s, 'Selection', 'avian influenza', 'boolean', '{}', TRUE)", (SID,))
        ex = json.dumps({"source": "fulltext", "version": 1, "ref": {"location": "Italy"},
                         "coverage": {}, "observations": [
                             {"sheet": "human_exp", "group": "occupation",
                              "covariate": "poultry worker", "disease": "HPAI H5N1",
                              "quote": "q", "quote_verified": True}]})
        ex2 = json.dumps({"source": "abstract", "version": 1, "ref": {"location": "China"},
                          "coverage": {}, "observations": [
                              {"sheet": "human_exp", "group": "occupation",
                               "covariate": "veterinarian", "disease": "HPAI H5N1",
                               "quote": "q", "quote_verified": True}]})
        cur.execute("""
            INSERT INTO literature_document
              (id, title, abstract, source, year, language, journal, doi, citation_count,
               is_duplicate, project_context, extraction_json)
            VALUES
              (9801, 'English 2024 poultry', %(a)s, 'pubmed', 2024, 'en', 'Viruses', '10.1/a', 9,
               FALSE, 'literev', %(x1)s::jsonb),
              (9802, 'French 2012 vets', %(a)s, 'pubmed', 2012, 'fr', 'Virologie', NULL, 1,
               FALSE, 'literev', %(x2)s::jsonb),
              (9803, 'No language, no year', %(a)s, 'pubmed', NULL, NULL, NULL, NULL, NULL,
               FALSE, 'literev', NULL),
              (9804, 'English 2020, nothing read', %(a)s, 'pubmed', 2020, 'en', 'Viruses', '10.1/d', 3,
               FALSE, 'literev', NULL)
        """, {"a": "a study of exposure among poultry workers", "x1": ex, "x2": ex2})
        cur.execute("INSERT INTO article_scenarios (scenario_id, document_id, similarity_score) "
                    "VALUES (%s, 9801, 0.90), (%s, 9802, 0.85), (%s, 9803, 0.80), (%s, 9804, 0.75)",
                    (SID,) * 4)
        cur.execute("INSERT INTO scenario_settings (scenario_id, similarity_threshold) VALUES (%s, 0.45) "
                    "ON CONFLICT (scenario_id) DO UPDATE SET similarity_threshold = 0.45", (SID,))
    yield db_conn
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM scenario_settings WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))
        cur.execute("DELETE FROM literature_document WHERE id = ANY(%s)", (list(IDS),))
        if created_chunk_table:
            cur.execute("DROP TABLE document_chunk")


def _client():
    from fastapi.testclient import TestClient
    return TestClient(main.app)


HDR = {"X-API-Key": "test-write-key"}


def _preview(payload: dict) -> dict:
    r = _client().post(f"/user-scenarios/{SID}/subset/preview", json=payload)
    assert r.status_code == 200, r.text
    return r.json()


def test_a_condition_on_a_column_narrows_the_corpus(seeded):
    """Four relevant papers: two English, one French, one with no language. Keeping English
    drops the French one and leaves the one with no language alone."""
    out = _preview({"conditions": [{"variable": "language", "values": ["en"]}]})
    assert out["relevant"] == 4
    assert out["by_dimension"]["language"] == {"in": 2, "out": 1, "unknown": 1}
    assert out["keep"] == 3 and out["exclude"] == 1
    assert out["undecided"] == 1, "the paper with no language was judged by nothing"


def test_a_year_condition_takes_an_interval(seeded):
    out = _preview({"conditions": [{"variable": "year", "min": 2015}]})
    assert out["by_dimension"]["year"] == {"in": 2, "out": 1, "unknown": 1}
    assert "year à partir de 2015" in out["reason"]


def test_a_condition_on_a_codebook_group_narrows_the_corpus(seeded):
    """The scenario's own variables filter like any other. Two papers reported an occupation,
    one a poultry worker and one a veterinarian; the two papers nobody read cannot be judged."""
    out = _preview({"conditions": [{"variable": "human_exp.occupation",
                                    "values": ["poultry_worker"]}]})
    assert out["by_dimension"]["human_exp.occupation"] == {"in": 1, "out": 1, "unknown": 2}
    assert out["keep"] == 3 and out["exclude"] == 1


def test_two_conditions_cross(seeded):
    """The question the register exists for: English AND from 2015 onwards. Paper 9801 and
    9804 pass both; 9802 fails both; 9803 is judged by neither and is kept."""
    out = _preview({"conditions": [{"variable": "language", "values": ["en"]},
                                   {"variable": "year", "min": 2015}]})
    assert out["keep"] == 3 and out["exclude"] == 1
    out_any = _preview({"conditions": [{"variable": "language", "values": ["fr"]},
                                       {"variable": "year", "min": 2015}],
                        "combine": "any"})
    assert out_any["keep"] == 4, "under 'any', French OR recent keeps everything judged"


def test_dropping_the_unjudged_is_the_callers_explicit_choice(seeded):
    kept = _preview({"conditions": [{"variable": "language", "values": ["en"]}]})
    dropped = _preview({"conditions": [{"variable": "language", "values": ["en"]}],
                        "unassigned": "exclude"})
    assert kept["keep"] == 3 and dropped["keep"] == 2
    assert dropped["undecided"] == 1, "still reported, never absorbed in silence"


def test_a_country_condition_and_the_named_country_dimension_agree(seeded):
    """Same resolution, one behaviour. Asking for Italy as a condition routes to the country
    dimension, so the two payloads give the same plan down to the dimension tally."""
    as_condition = _preview({"conditions": [{"variable": "country", "values": ["IT"]}]})
    as_named = _preview({"countries": ["IT"]})
    for key in ("keep", "exclude", "undecided", "relevant"):
        assert as_condition[key] == as_named[key], key
    assert as_condition["by_dimension"] == as_named["by_dimension"]


def test_an_unfilterable_variable_is_refused_with_the_list_of_those_that_are(seeded):
    r = _client().post(f"/user-scenarios/{SID}/subset/preview",
                       json={"conditions": [{"variable": "above_threshold", "values": ["above"]}]})
    assert r.status_code == 404
    assert "ne se filtre pas" in r.json()["detail"]
    assert "language" in r.json()["detail"], "it says what can be filtered"


def test_a_value_absent_from_the_corpus_is_refused_rather_than_emptying_it(seeded):
    """Asking for a value no paper carries would exclude everything judged. The corpus says
    what it holds instead, which is the difference between a typo and a disaster."""
    r = _client().post(f"/user-scenarios/{SID}/subset/preview",
                       json={"conditions": [{"variable": "language", "values": ["de"]}]})
    assert r.status_code == 404
    body = r.json()["detail"]
    assert "de" in body and "en" in body


def test_a_number_asked_as_a_list_says_what_it_wants(seeded):
    r = _client().post(f"/user-scenarios/{SID}/subset/preview",
                       json={"conditions": [{"variable": "year", "values": ["2024"]}]})
    assert r.status_code == 422 and "min" in r.json()["detail"]


def test_applying_a_condition_writes_the_exclusion_and_undo_takes_only_it(seeded, db_conn):
    """The mechanism is unchanged: the narrowing is a screening exclusion carrying its
    reason, and the undo removes only the rows the selection wrote. A reviewer's own
    exclusion, written here by hand, survives it."""
    with db_conn.cursor() as cur:
        cur.execute("UPDATE article_scenarios SET screening_status = 'excluded', "
                    "screening_reason = 'wrong population' WHERE scenario_id = %s AND document_id = 9804",
                    (SID,))
    client = _client()
    r = client.post(f"/user-scenarios/{SID}/subset/apply",
                    json={"conditions": [{"variable": "language", "values": ["en"]}]}, headers=HDR)
    assert r.status_code == 200, r.text
    with db_conn.cursor() as cur:
        cur.execute("SELECT document_id, screening_status, screening_reason FROM article_scenarios "
                    "WHERE scenario_id = %s AND screening_status = 'excluded' ORDER BY document_id", (SID,))
        rows = cur.fetchall()
    by_id = {r[0]: (r[1], r[2]) for r in rows}
    assert 9802 in by_id and by_id[9802][1].startswith(SUBSET_REASON_PREFIX)
    assert "language en" in by_id[9802][1]
    assert by_id[9804] == ("excluded", "wrong population"), "the reviewer's own reason is untouched"

    r = client.post(f"/user-scenarios/{SID}/subset/undo", headers=HDR)
    assert r.status_code == 200, r.text
    with db_conn.cursor() as cur:
        cur.execute("SELECT document_id FROM article_scenarios WHERE scenario_id = %s "
                    "AND screening_status = 'excluded' ORDER BY document_id", (SID,))
        left = [r[0] for r in cur.fetchall()]
    assert left == [9804], "only the selection's own exclusion was undone"
