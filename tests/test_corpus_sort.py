"""Sorting the corpus by year happens on the server, over the whole corpus.

The search page downloads the 200 most relevant articles of a corpus that can hold
thousands, and sorted THOSE by year in the browser: "most recent first" showed the most
recent of the 200 most relevant, not the most recent articles of the corpus. The corpus
endpoint now takes `sort`, and the page asks it.
"""
import pytest

pytest.importorskip("fastapi")

import main  # noqa: E402
from conftest import ensure_document_columns  # noqa: E402

SID = "usr-corpus-sort"
# id: (year, similarity). The most relevant article is the oldest; the newest is the
# least relevant, so a sort over the top two by relevance would never show it.
ROWS = {9801: (2001, 0.95), 9802: (2010, 0.90), 9803: (2024, 0.20), 9804: (None, 0.85),
        9805: (2015, 0.50)}


@pytest.fixture()
def seeded(db_conn):
    try:
        with main.engine.connect():
            pass
    except Exception:                                        # noqa: BLE001
        pytest.skip("main.engine cannot reach the database")
    with db_conn.cursor() as cur:
        cur.execute("SELECT to_regclass('user_scenarios') IS NULL")
        if cur.fetchone()[0]:
            main._ensure_user_scenarios_table()
        # A fresh database, or one an earlier test left without them (several drop them).
        cur.execute("CREATE TABLE IF NOT EXISTS literature_document (id BIGINT PRIMARY KEY)")
        cur.execute("CREATE TABLE IF NOT EXISTS article_scenarios ("
                    "scenario_id TEXT, document_id BIGINT, PRIMARY KEY (scenario_id, document_id))")
        created_chunk_table = ensure_document_columns(cur)
        for col, typ in (("title", "TEXT"), ("abstract", "TEXT"), ("is_duplicate", "BOOLEAN DEFAULT FALSE")):
            cur.execute(f"ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS {col} {typ}")
        cur.execute("ALTER TABLE article_scenarios ADD COLUMN IF NOT EXISTS similarity_score DOUBLE PRECISION")
        cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))
        cur.execute("DELETE FROM literature_document WHERE id = ANY(%s)", (list(ROWS),))
        cur.execute("INSERT INTO user_scenarios (id, name, query, mode, filters, pinned) "
                    "VALUES (%s, 'Sort', 'q', 'boolean', '{}', TRUE)", (SID,))
        for i, (year, sim) in ROWS.items():
            cur.execute("INSERT INTO literature_document (id, title, abstract, year, source, is_duplicate, "
                        "project_context) VALUES (%s, %s, 'an abstract', %s, 'pubmed', false, 'literev')",
                        (i, f"Article {i}", year))
            cur.execute("INSERT INTO article_scenarios (scenario_id, document_id, similarity_score) "
                        "VALUES (%s, %s, %s)", (SID, i, sim))
    yield db_conn
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))
        cur.execute("DELETE FROM literature_document WHERE id = ANY(%s)", (list(ROWS),))
        if created_chunk_table:
            cur.execute("DROP TABLE document_chunk")


def _ids(client, **params):
    r = client.get(f"/user-scenarios/{SID}/corpus", params=params)
    assert r.status_code == 200, r.text
    return [a["id"] for a in r.json()["articles"]]


def test_the_year_sort_runs_over_the_whole_corpus(seeded):
    from fastapi.testclient import TestClient

    c = TestClient(main.app)
    assert _ids(c, limit=2) == [9801, 9802]                         # relevance: the default
    assert _ids(c, limit=2, sort="year_desc") == [9803, 9805]       # the newest, wherever it ranks
    assert _ids(c, limit=2, sort="year_asc") == [9801, 9802]
    # Undated articles come last both ways, and paging follows the same order.
    assert _ids(c, sort="year_desc")[-1] == 9804 and _ids(c, sort="year_asc")[-1] == 9804
    assert _ids(c, limit=2, offset=2, sort="year_desc") == [9802, 9801]


def test_an_unknown_sort_is_refused(seeded):
    from fastapi.testclient import TestClient

    r = TestClient(main.app).get(f"/user-scenarios/{SID}/corpus", params={"sort": "citations"})
    assert r.status_code == 422
