"""A model's inputs, joined to what the extraction actually read.

A predictive scenario's variables each carry `evidence_count`: the relevant articles whose
title, abstract or PICO contains the variable's name as a whole word. Before any paper has
been read that is the best number available, and it is honest about what it is: a count of
MENTIONS.

Once the extraction has run, the review knows something stronger. The codebook asked every
paper for a named list of variables, and the extraction recorded which ones each paper
reports. So "occupation" as a model input and "occupation" as a question the review asked
are the same thing seen from two sides, and joining them turns a word that appears in an
abstract into a figure a paper reports.

What these tests pin:

  - the join is by VOCABULARY, not by regular expression: `find_variable` maps a predictor's
    own wording onto a codebook group, and an ambiguous match is refused rather than
    guessed, because a wrong variable cannot be un-said;
  - the count is over the WHOLE relevant subset in SQL, with the papers READ as its
    denominator, named as such (house rule);
  - `evidence_count` is never touched. Two numbers measuring two different things both stay,
    each under its own name, and a variable no codebook asks for keeps its word count rather
    than reading as zero evidence;
  - the page can show it without paying for a generation: the endpoint re-reads the spec
    already stored and writes nothing back.
"""
import json

import pytest

pytest.importorskip("fastapi")

import main  # noqa: E402
from api.codebook import DEFAULT_NODES, Index, find_variable, validate_nodes  # noqa: E402
from api.variables_of_interest import (  # noqa: E402
    REGISTER_EVIDENCE_RULE,
    _join_terms,
    register_evidence,
)
from conftest import ensure_document_columns  # noqa: E402

SID = "usr-modelev-test"
IDS = (9801, 9802, 9803, 9804)


def _index() -> Index:
    return Index(validate_nodes([dict(n) for n in DEFAULT_NODES]))


# ── Pure: the join itself ───────────────────────────────────────────────────
def test_a_predictor_named_like_the_question_finds_it():
    ix = _index()
    assert find_variable(ix, "occupation") == ("human_exp", "occupation")
    assert find_variable(ix, "Sex") == ("human_susc", "sex_gender")
    assert find_variable(ix, "biosecurity measures") == ("env", "biosecurity")


def test_a_predictor_named_in_a_sentence_still_finds_it():
    """A model words its variables for a person: "occupation of the cases", "mean daily
    temperature". The join has to survive the words around the name."""
    ix = _index()
    assert find_variable(ix, "Occupation of the cases") == ("human_exp", "occupation")
    assert find_variable(ix, "Mean daily temperature") == ("env", "climate")
    assert find_variable(ix, "Reported contact with poultry") == ("human_exp", "contact_type")


def test_a_name_two_different_questions_could_claim_is_refused():
    """`species` is declared on the animal sheet and on the vector sheet. Joining to either
    would attach a count to the wrong question, and nothing on the screen would say so, so
    the join refuses and the word count stays."""
    ix = _index()
    assert find_variable(ix, "species") is None
    assert find_variable(ix, "Species of the affected animals") is None


def test_a_name_no_question_asks_about_joins_to_nothing():
    ix = _index()
    assert find_variable(ix, "hospital bed capacity") is None
    assert find_variable(ix, "gross domestic product per head") is None
    assert find_variable(ix, "") is None and find_variable(ix) is None


def test_the_variables_own_name_is_tried_before_its_keywords():
    """The order matters: a keyword list is a net, the name is the intent. A variable named
    "age" with "occupation" among its keywords is an age variable."""
    ix = _index()
    assert find_variable(ix, "age", "occupation") == ("human_susc", "age")
    assert find_variable(ix, "nothing here", "occupation") == ("human_exp", "occupation")


def test_the_terms_of_a_variable_read_its_name_then_its_keywords():
    terms = _join_terms({"name": "Occupation of the cases", "keywords": ["job", "x", "job"],
                         "synonyms": "profession", "definition": "What the person does"})
    assert terms[0] == "Occupation of the cases", "the name first"
    assert "job" in terms and terms.count("job") == 1
    assert "x" not in terms, "two characters is noise, not a term"
    assert len(terms) <= 12


# ── Over the database ───────────────────────────────────────────────────────
def _engine_ok() -> bool:
    try:
        with main.engine.connect():
            return True
    except Exception:                                        # noqa: BLE001
        return False


def _extraction(obs: list[tuple[str, str, str]]) -> str:
    return json.dumps({
        "source": "fulltext", "version": 1, "ref": {"location": "Italy"},
        "observations": [{"sheet": s, "group": g, "covariate": c, "disease": "HPAI H5N1",
                          "value": 1, "quote": "q", "quote_verified": True}
                         for s, g, c in obs],
    })


#: Le spec stocké : quatre prédicteurs, deux qui se rattachent et deux qui ne peuvent pas.
_SPEC = {
    "predictor_variables": [
        {"name": "Occupation of the cases", "type": "categorical", "keywords": ["job"],
         "importance": "high", "evidence_count": 3, "evidence_level": "3/3"},
        {"name": "Mean daily temperature", "type": "numeric", "importance": "medium",
         "evidence_count": 1, "evidence_level": "1/3"},
        {"name": "Reported occupation group", "type": "categorical", "importance": "low",
         "evidence_count": 2, "evidence_level": "2/3"},
        {"name": "Hospital bed capacity", "type": "numeric", "importance": "high",
         "evidence_count": 0, "evidence_level": "0/3"},
    ],
    "_meta": {"evidence_count_denominator": 3},
}


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
                         ("pico_json", "JSONB"), ("extraction_json", "JSONB"),
                         ("is_duplicate", "BOOLEAN DEFAULT FALSE")):
            cur.execute(f"ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS {col} {typ}")
        for col, typ in (("similarity_score", "DOUBLE PRECISION"), ("screening_reason", "TEXT"),
                         ("screened_at", "TIMESTAMP")):
            cur.execute(f"ALTER TABLE article_scenarios ADD COLUMN IF NOT EXISTS {col} {typ}")
        cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM scenario_settings WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))
        cur.execute("DELETE FROM literature_document WHERE id = ANY(%s)", (list(IDS),))
        cur.execute("INSERT INTO user_scenarios (id, name, query, mode, filters, pinned) "
                    "VALUES (%s, 'Model evidence', 'avian influenza', 'boolean', '{}', TRUE)",
                    (SID,))
        # 9801 and 9802 read; 9803 relevant but never read; 9804 BELOW the threshold, and
        # read, so its observations must never reach a count.
        cur.execute("""
            INSERT INTO literature_document
              (id, title, abstract, source, year, is_duplicate, project_context, extraction_json)
            VALUES (9801, 'Cullers in Lombardy', %(a)s, 'pubmed', 2024, FALSE, 'literev', %(x1)s::jsonb),
                   (9802, 'Markets and weather', %(a)s, 'pubmed', 2023, FALSE, 'literev', %(x2)s::jsonb),
                   (9803, 'Never read', %(a)s, 'pubmed', 2022, FALSE, 'literev', NULL),
                   (9804, 'Below the threshold', %(a)s, 'pubmed', 2021, FALSE, 'literev', %(x4)s::jsonb)
        """, {"a": "exposure among poultry workers",
              "x1": _extraction([("human_exp", "occupation", "poultry worker"),
                                 ("human_exp", "occupation", "veterinarian")]),
              "x2": _extraction([("human_exp", "occupation", "poultry worker"),
                                 ("env", "climate", "temperature")]),
              "x4": _extraction([("human_exp", "occupation", "poultry worker")])})
        cur.execute("""
            INSERT INTO article_scenarios (scenario_id, document_id, similarity_score)
            VALUES (%s, 9801, 0.90), (%s, 9802, 0.85), (%s, 9803, 0.80), (%s, 9804, 0.10)
        """, (SID,) * 4)
        cur.execute("INSERT INTO scenario_settings (scenario_id, similarity_threshold, variables_json) "
                    "VALUES (%s, 0.45, %s::jsonb) ON CONFLICT (scenario_id) DO UPDATE SET "
                    "similarity_threshold = 0.45, variables_json = EXCLUDED.variables_json",
                    (SID, json.dumps(_SPEC)))
    yield db_conn
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM scenario_settings WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))
        cur.execute("DELETE FROM literature_document WHERE id = ANY(%s)", (list(IDS),))
        if created_chunk_table:
            cur.execute("DROP TABLE document_chunk")


def test_a_joined_variable_counts_the_papers_that_reported_it(seeded):
    """Two papers were read and both report an occupation, so 2 of 2, with the values the
    papers gave. That is a different statement from "three abstracts contain the word"."""
    variables = json.loads(json.dumps(_SPEC))
    out = register_evidence(SID, variables, threshold=0.45, lang="en")
    assert out["denominator"] == 2, "the denominator is the papers READ, not the relevant ones"
    assert out["matched"] == 3, "two occupation variables and the temperature one"
    pv = {v["name"]: v for v in variables["predictor_variables"]}

    occ = pv["Occupation of the cases"]
    assert occ["register_variable"] == "human_exp.occupation"
    assert occ["register_count"] == 2 and occ["register_level"] == "2/2"
    assert occ["register_matched_by"] == "codebook"
    assert {v["value"]: v["n"] for v in occ["register_values"]} == {"poultry_worker": 2,
                                                                   "veterinarian": 1}
    assert occ["register_name"], "the question is named as the codebook names it"

    temp = pv["Mean daily temperature"]
    assert temp["register_variable"] == "env.climate" and temp["register_count"] == 1


def test_a_variable_the_review_never_asked_about_keeps_its_word_count(seeded):
    """The honest answer is "the register does not know", not zero. A zero would read as
    "no paper supports this variable", which is a claim the join cannot make."""
    variables = json.loads(json.dumps(_SPEC))
    register_evidence(SID, variables, threshold=0.45, lang="en")
    beds = next(v for v in variables["predictor_variables"]
                if v["name"] == "Hospital bed capacity")
    assert beds["register_variable"] is None
    assert beds["register_count"] is None and beds["register_level"] is None
    assert beds["register_matched_by"] is None and beds["register_values"] == []
    # And the measure that did exist is untouched, on every variable.
    assert beds["evidence_count"] == 0 and beds["evidence_level"] == "0/3"
    assert [v["evidence_count"] for v in variables["predictor_variables"]] == [3, 1, 2, 0]


def test_two_predictors_on_the_same_question_get_the_same_count(seeded):
    """"Occupation of the cases" and "Reported occupation group" are one question asked
    twice. Neither may be counted twice, and neither may be left out."""
    variables = json.loads(json.dumps(_SPEC))
    register_evidence(SID, variables, threshold=0.45, lang="en")
    both = [v for v in variables["predictor_variables"]
            if v.get("register_variable") == "human_exp.occupation"]
    assert len(both) == 2
    assert {v["register_count"] for v in both} == {2}


def test_a_paper_below_the_threshold_never_reaches_a_count(seeded):
    """The house rule on this page too: the subset is the relevant one. 9804 reports an
    occupation and sits below the threshold, so the occupation count stays at 2."""
    variables = json.loads(json.dumps(_SPEC))
    out = register_evidence(SID, variables, threshold=0.45, lang="en")
    assert out["denominator"] == 2
    occ = next(v for v in variables["predictor_variables"]
               if v["register_variable"] == "human_exp.occupation")
    assert occ["register_count"] == 2
    # Lowering the threshold lets it in, which proves the gate is what excluded it.
    wider = json.loads(json.dumps(_SPEC))
    out2 = register_evidence(SID, wider, threshold=0.05, lang="en")
    assert out2["denominator"] == 3
    occ2 = next(v for v in wider["predictor_variables"]
                if v["register_variable"] == "human_exp.occupation")
    assert occ2["register_count"] == 3


def test_a_review_with_no_variables_is_not_an_error(seeded):
    out = register_evidence(SID, {}, threshold=0.45, lang="en")
    assert out == {"denominator": 0, "matched": 0, "n_variables": 0, "matched_variables": []}


def test_the_rule_says_what_each_of_the_two_numbers_counts():
    """Both are in the answer, so both have to be readable. A page that showed one number
    without saying which one it is would be worse than showing neither."""
    assert "register_count" in REGISTER_EVIDENCE_RULE
    assert "REPORTS" in REGISTER_EVIDENCE_RULE and "MENTIONS" in REGISTER_EVIDENCE_RULE
    assert "evidence_count" in REGISTER_EVIDENCE_RULE
    assert "—" not in REGISTER_EVIDENCE_RULE


# ── The endpoint ────────────────────────────────────────────────────────────
def _client():
    from fastapi.testclient import TestClient
    return TestClient(main.app)


def test_the_page_can_read_the_join_without_paying_for_a_generation(seeded):
    """The four partner reviews already have their variables. Re-reading the stored spec
    through the codebook costs one SQL pass and no model call, so the join is visible
    tonight rather than after a regeneration."""
    r = _client().get(f"/user-scenarios/{SID}/variables-of-interest/model-evidence",
                      params={"lang": "en"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok" and body["n_variables"] == 4 and body["matched"] == 3
    assert body["denominator"] == 2
    assert body["evidence_count_denominator"] == 3, "the other measure's denominator too"
    rows = {v["name"]: v for v in body["variables"]}
    assert rows["Occupation of the cases"]["register_count"] == 2
    assert rows["Occupation of the cases"]["mentions"] == 3, "the two numbers side by side"
    assert rows["Hospital bed capacity"]["register_variable"] is None
    assert rows["Hospital bed capacity"]["mentions"] == 0
    assert "MENTIONS" in body["rule"]


def test_the_rows_carry_their_position_so_a_translated_page_still_joins(seeded):
    """The page shows the variables in the language the reviewer reads, and the names are
    translated. The join is therefore by POSITION in the stored spec, and the rows come back
    in that order, with the position written on each one."""
    body = _client().get(f"/user-scenarios/{SID}/variables-of-interest/model-evidence").json()
    assert [v["index"] for v in body["variables"]] == [0, 1, 2, 3]
    assert [v["name"] for v in body["variables"]] == [pv["name"] for pv in
                                                      _SPEC["predictor_variables"]]


def test_reading_the_join_writes_nothing_back_to_the_spec(seeded):
    """A page opened twice must not rewrite what a reviewer validated."""
    client = _client()
    client.get(f"/user-scenarios/{SID}/variables-of-interest/model-evidence")
    with seeded.cursor() as cur:
        cur.execute("SELECT variables_json FROM scenario_settings WHERE scenario_id = %s", (SID,))
        stored = cur.fetchone()[0]
    assert stored == _SPEC, "the stored spec is untouched by a read"


def test_a_scenario_with_no_spec_says_so_rather_than_answering_zero(seeded):
    with seeded.cursor() as cur:
        cur.execute("UPDATE scenario_settings SET variables_json = NULL, "
                    "variables_proposal_json = NULL WHERE scenario_id = %s", (SID,))
    r = _client().get(f"/user-scenarios/{SID}/variables-of-interest/model-evidence")
    assert r.status_code == 200
    assert r.json()["status"] == "no_variables" and r.json()["variables"] == []


def test_an_unknown_scenario_is_a_404():
    r = _client().get("/user-scenarios/usr-no-such-thing/variables-of-interest/model-evidence")
    assert r.status_code == 404
