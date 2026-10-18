"""The register decides what the extraction asks, and adding a variable is a priced choice.

The codebook used to STEER the extraction and not drive it: `vocabulary_prompt` told the
model what to call a group it had decided to report, so a reviewer could add "biosecurity
level" to the codebook and see nothing change, because nothing asked for it. The register's
own variables are now asked by name, with the values the codebook knows.

That makes the register cost money, which is why the rest of this file is about cost.

THE PROPERTY THAT MATTERS MOST: the pipeline's bill does not change. Adding a variable does
not make a single paper eligible for re-reading on any automatic path. The gap is priced by
a read-only endpoint, and spent only by a run that is asked for it in so many words. A
corpus of several thousand papers must never start re-reading itself because somebody typed
a word into a codebook.

THE SECOND PROPERTY: an unknown is not a no. A paper read before the record existed may well
have been asked the variable, and re-reading a corpus on a maybe is how a one-time cost
becomes a recurring one. Those papers are counted on their own line and never swept in.
"""
import json

import pytest

pytest.importorskip("fastapi")

import main  # noqa: E402
from api.codebook import (  # noqa: E402
    DEFAULT_NODES,
    asked_variable_keys,
    asked_variables_prompt,
    codebook_template,
)
from api.extraction import (  # noqa: E402
    _missing_variables,
    _needs_extraction,
    _vars_asked_of,
    variable_gap,
)
from conftest import ensure_document_columns  # noqa: E402

SID = "usr-asked-test"
IDS = (9901, 9902, 9903, 9904)


# ── Pure: what the register asks ────────────────────────────────────────────
def test_the_asked_keys_are_the_registers_own_rows():
    """One list for two uses that must never disagree: what the prompt asks, and what each
    extraction records as having been asked. A key is a row of the Variables of interest
    page, which is why it carries the page's shape."""
    keys = asked_variable_keys(DEFAULT_NODES)
    assert "human_exp.occupation" in keys and "animal.species" in keys
    assert keys == sorted(set(keys)), "sorted and deduplicated, so two runs compare"
    assert len(keys) == 19


def test_the_prompt_asks_for_each_variable_by_name_with_its_values():
    block = asked_variables_prompt(DEFAULT_NODES, "en")
    assert "VARIABLES THIS REVIEW ASKS FOR" in block
    assert "Occupation (sheet human_exp, group occupation)" in block
    assert "Poultry worker" in block, "the values the codebook knows are named"
    assert "never infer" in block, "a variable the paper is silent about stays out"


def test_the_prompt_says_when_it_had_to_cut_rather_than_implying_the_list_is_whole():
    """A hundred-group codebook does not fit in a prompt beside the paper's text. The block
    is bounded, and a bounded list that does not say so is a lie to the model."""
    big = []
    for i in range(120):
        big.append({"sheet": "env", "l1": f"group_{i:03d}", "l2": None, "l3": None,
                    "synonyms": [], "label_en": f"Group {i}", "label_fr": None})
    block = asked_variables_prompt(big, "en")
    assert "more variables are not listed here" in block
    assert len(block) < 5000


def test_a_starter_asks_for_what_it_declares():
    """The link that makes a starter worth loading: pick healthcare capacity and the
    extraction starts asking about beds and staff."""
    block = asked_variables_prompt(codebook_template("healthcare_capacity"), "en")
    for word in ("Beds", "Staff", "Surge capacity"):
        assert word in block
    assert "Poultry worker" not in block


# ── Pure: the cost rules ────────────────────────────────────────────────────
def test_an_unknown_record_is_not_an_empty_one():
    assert _vars_asked_of({"vars_asked": ["a.b"]}) == ["a.b"]
    assert _vars_asked_of({}) is None, "read before the record existed"
    assert _vars_asked_of({"vars_asked": "a.b"}) is None, "a malformed record is unknown"
    assert _vars_asked_of(None) is None


def test_a_paper_whose_record_is_unknown_is_never_counted_as_missing():
    """The rule that keeps a one-time cost from becoming a recurring one."""
    assert _missing_variables({}, ["a.b", "c.d"]) == []
    assert _missing_variables({"vars_asked": ["a.b"]}, ["a.b", "c.d"]) == ["c.d"]
    assert _missing_variables({"vars_asked": ["a.b", "c.d"]}, ["a.b"]) == [], \
        "removing a variable costs nothing"


def test_adding_a_variable_makes_no_paper_eligible_on_the_automatic_path():
    """`_needs_extraction` is what the pipeline and the plain run use. A paper that was
    asked fewer variables than the register asks today is NOT eligible by it: the only way
    to spend on that gap is the explicit flag."""
    row = {"extraction_attempts": 0, "has_fulltext_chunk": True, "abstract": "x" * 200,
           "extraction_json": {"v": 1, "source": "fulltext", "vars_asked": ["human_susc.age"]}}
    assert _needs_extraction(row) is False
    assert _missing_variables(row["extraction_json"], ["human_susc.age", "animal.species"]) \
        == ["animal.species"], "the gap is real, and still not a reason to spend"


# ── Over the database ───────────────────────────────────────────────────────
def _engine_ok() -> bool:
    try:
        with main.engine.connect():
            return True
    except Exception:                                        # noqa: BLE001
        return False


def _ex(vars_asked=None) -> str:
    body = {"source": "fulltext", "v": 1, "ref": {"location": "Italy"}, "coverage": {},
            "observations": [{"sheet": "human_susc", "group": "age", "covariate": "children",
                              "disease": "HPAI H5N1", "quote": "q", "quote_verified": True}]}
    if vars_asked is not None:
        body["vars_asked"] = vars_asked
    return json.dumps(body)


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
        for col, typ in (("title", "TEXT"), ("abstract", "TEXT"), ("extraction_json", "JSONB"),
                         ("extraction_attempts", "INTEGER DEFAULT 0"),
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
                    "VALUES (%s, 'Asked', 'avian influenza', 'boolean', '{}', TRUE)", (SID,))
        # 9901 was asked only age, so the rest of the register is a real gap for it.
        # 9902 was asked everything the register asks.
        # 9903 was read before the record existed: unknown, never swept in.
        # 9904 has no extraction at all: the ordinary path already covers it.
        cur.execute("""
            INSERT INTO literature_document
              (id, title, abstract, source, is_duplicate, project_context, extraction_json)
            VALUES (9901, 'Asked age only', %(a)s, 'pubmed', FALSE, 'literev', %(x1)s::jsonb),
                   (9902, 'Asked everything', %(a)s, 'pubmed', FALSE, 'literev', %(x2)s::jsonb),
                   (9903, 'Read before the record', %(a)s, 'pubmed', FALSE, 'literev', %(x3)s::jsonb),
                   (9904, 'Never read', %(a)s, 'pubmed', FALSE, 'literev', NULL)
        """, {"a": "a study of exposure among poultry workers, with ages",
              "x1": _ex(["human_susc.age"]),
              "x2": _ex(asked_variable_keys(DEFAULT_NODES)),
              "x3": _ex(None)})
        cur.execute("INSERT INTO article_scenarios (scenario_id, document_id, similarity_score) "
                    "VALUES (%s, 9901, 0.9), (%s, 9902, 0.9), (%s, 9903, 0.9), (%s, 9904, 0.9)",
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


def test_the_bill_says_what_a_re_read_would_cost_and_what_it_cannot_know(seeded):
    gap = variable_gap(SID)
    assert gap["complete"] is True
    assert gap["n_relevant"] == 4 and gap["n_extracted"] == 3
    assert gap["n_papers_missing_a_variable"] == 1, "only 9901, which recorded a short list"
    assert gap["n_calls_estimate"] == 1, "one call per paper; the text is already cached"
    assert gap["n_papers_unknown"] == 1, "9903 was read before the record existed"
    assert len(gap["variables_asked_now"]) == 19
    by = {v["key"]: v for v in gap["by_variable"]}
    assert by["animal.species"]["n_papers"] == 1
    assert by["animal.species"]["name"] == "Species", "named as the codebook names it"
    assert "human_susc.age" not in by, "9901 was asked age, so it is not in the bill"


def test_the_bill_is_read_only_and_changes_nothing(seeded, db_conn):
    with db_conn.cursor() as cur:
        cur.execute("SELECT id, extraction_json FROM literature_document "
                    "WHERE id = ANY(%s) ORDER BY id", (list(IDS),))
        before = cur.fetchall()
    variable_gap(SID)
    with db_conn.cursor() as cur:
        cur.execute("SELECT id, extraction_json FROM literature_document "
                    "WHERE id = ANY(%s) ORDER BY id", (list(IDS),))
        assert cur.fetchall() == before


def test_the_gap_endpoint_prices_it_without_an_api_key(seeded):
    from fastapi.testclient import TestClient

    r = TestClient(main.app).get(f"/user-scenarios/{SID}/extraction/variable-gap")
    assert r.status_code == 200
    assert r.json()["n_papers_missing_a_variable"] == 1


def test_loading_a_starter_changes_what_is_asked_without_re_reading_anything(seeded):
    """Loading a vocabulary is cheap to try: the bill moves, nothing is spent. Here the
    healthcare starter asks about beds and staff, which no paper was ever asked, so 9901 and
    9902 both appear in the bill, and 9903 still does not."""
    from fastapi.testclient import TestClient

    client = TestClient(main.app)
    before = variable_gap(SID)["n_papers_missing_a_variable"]
    r = client.post(f"/user-scenarios/{SID}/codebook/template/healthcare_capacity",
                    headers={"X-API-Key": "test-write-key"})
    assert r.status_code == 200, r.text
    after = variable_gap(SID)
    assert after["n_papers_missing_a_variable"] == 2 > before
    assert after["n_papers_unknown"] == 1
    keys = {v["key"] for v in after["by_variable"]}
    assert "env.beds" in keys and "env.staff" in keys
