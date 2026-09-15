"""The relevant-articles fetch behind the LLM generators (brief, variables, actions)
returned every relevant article with its abstract and PICO — 25,000 rows and 120 MB
for a large scenario — while each generator reads 20 to 30 of them. `full_rows=N`
keeps every row (ids, statuses, `has_pico`: enough for counts and the corpus
fingerprint) but ships abstract/PICO for the top N only; `require_pico` restricts to
articles with an extracted PICO."""
import json

import pytest

pytest.importorskip("fastapi")

import main  # noqa: E402
from conftest import ensure_document_columns  # noqa: E402

SID = "usr-relevant-articles-test"
IDS = (9201, 9202, 9203, 9204)


def _engine_ok() -> bool:
    try:
        with main.engine.connect():
            return True
    except Exception:
        return False


@pytest.fixture()
def seeded(db_conn):
    if not _engine_ok():
        pytest.skip("main.engine cannot reach the database")
    with db_conn.cursor() as cur:
        cur.execute("SELECT to_regclass('user_scenarios') IS NULL")
        if cur.fetchone()[0]:
            main._ensure_user_scenarios_table()
        created_chunk_table = ensure_document_columns(cur)
        cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM literature_document WHERE id = ANY(%s)", (list(IDS),))
        cur.execute("INSERT INTO user_scenarios (id, name, query, mode, filters, pinned) "
                    "VALUES (%s, 'Relevant', 'flu', 'boolean', '{}', TRUE)", (SID,))
        pico = json.dumps({"population": "adults", "study_design": "RCT"})
        cur.execute(
            "INSERT INTO literature_document (id, title, source, abstract, is_duplicate, project_context, pico_json, study_design) VALUES "
            "(9201, 'Top', 'pubmed', 'abstract of the top article', false, 'literev', %s::jsonb, 'RCT'),"
            "(9202, 'Second', 'pubmed', 'abstract of the second article', false, 'literev', NULL, 'cohort'),"
            "(9203, 'Third with pico', 'pubmed', 'abstract of the third article', false, 'literev', %s::jsonb, NULL),"
            "(9204, 'Below threshold', 'pubmed', 'never relevant', false, 'literev', NULL, NULL)",
            (pico, pico))
        cur.execute("INSERT INTO article_scenarios (scenario_id, document_id, similarity_score) VALUES "
                    "(%s, 9201, 0.9), (%s, 9202, 0.8), (%s, 9203, 0.7), (%s, 9204, 0.1)", (SID, SID, SID, SID))
    yield db_conn
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM literature_document WHERE id = ANY(%s)", (list(IDS),))
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))
        if created_chunk_table:
            cur.execute("DROP TABLE document_chunk")


def test_full_rows_keeps_every_row_but_heavy_fields_only_on_top_n(seeded):
    arts = main._get_above_threshold_articles(SID, 0.45, full_rows=1)
    assert [a["id"] for a in arts] == [9201, 9202, 9203]          # ordered, below-threshold excluded
    assert arts[0]["abstract"] == "abstract of the top article" and arts[0]["pico_json"]
    assert arts[1]["abstract"] is None and arts[1]["pico_json"] is None
    assert [a["has_pico"] for a in arts] == [True, False, True]   # counts stay exact
    assert arts[1]["study_design"] == "cohort"                    # light columns for every row


def test_default_returns_everything_as_before(seeded):
    arts = main._get_above_threshold_articles(SID, 0.45)
    assert all(a["abstract"] for a in arts) and len(arts) == 3
    assert main._get_above_threshold_articles(SID, 0.45, full_rows=0)[0]["abstract"] is None


def test_require_pico_filters_to_articles_with_an_extracted_pico(seeded):
    arts = main._get_above_threshold_articles(SID, 0.45, full_rows=20, require_pico=True)
    assert [a["id"] for a in arts] == [9201, 9203]
    assert all(a["pico_json"] for a in arts)
