"""The PICO tab reads the relevant subset, not the whole corpus.

It used to ask for every article of the scenario, 5,000 at a time, up to forty pages. On
the ISS strategy that is 52,765 articles and about 66 MB of JSON; on the COVID exposure
review, 86,716 and over 100 MB. Each page is fast (two to three seconds against
production), so nothing looked broken in the logs: the browser simply never finished
parsing and holding a hundred megabytes of rows, and the tab stayed on "Loading PICO
data..." forever.

It was also pointless. The PICO extraction only ever runs on the relevant articles, so of
those 52,765 rows about 51,700 carried `has_pico: false` and nothing else: fifty thousand
empty rows downloaded to show a thousand full ones.

The endpoint therefore takes a scope, and defaults to the subset the PICO exists for. The
whole corpus stays one explicit parameter away, and the response always says how big the
corpus is, so the tab can state what it is showing out of what.
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")

import main  # noqa: E402
from conftest import ensure_document_columns  # noqa: E402

SID = "usr-pico-scope-test"
IDS = (9801, 9802, 9803, 9804)


def _engine_ok() -> bool:
    try:
        with main.engine.connect():
            return True
    except Exception:
        return False


def _pico(label: str) -> str:
    return json.dumps({"P": f"population {label}", "I": "exposure", "C": "none",
                       "O": "infection", "study_design": "Cohort", "pico_confidence": 0.8})


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
        for col, typ in (("title", "TEXT"), ("abstract", "TEXT"), ("journal", "TEXT"),
                         ("authors", "TEXT"), ("doi", "TEXT"),
                         ("pico_extracted_at", "TIMESTAMP"),
                         ("is_duplicate", "BOOLEAN DEFAULT FALSE")):
            cur.execute(f"ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS {col} {typ}")
        cur.execute("ALTER TABLE article_scenarios ADD COLUMN IF NOT EXISTS "
                    "similarity_score DOUBLE PRECISION")
        cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM scenario_settings WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))
        cur.execute("DELETE FROM literature_document WHERE id = ANY(%s)", (list(IDS),))
        cur.execute("INSERT INTO user_scenarios (id, name, query, mode, filters, pinned) "
                    "VALUES (%s, 'PICO scope', 'h5n1', 'boolean', '{}', TRUE)", (SID,))
        abstract = "a long enough abstract for the gate to accept this record"
        cur.execute(
            "INSERT INTO literature_document (id, title, abstract, source, is_duplicate, project_context) VALUES "
            "(9801, 'Above the threshold, read', %s, 'pubmed', false, 'literev'),"
            "(9802, 'Above the threshold, unread', %s, 'pubmed', false, 'literev'),"
            "(9803, 'Below the threshold', %s, 'pubmed', false, 'literev'),"
            "(9804, 'Excluded by a reviewer', %s, 'pubmed', false, 'literev')",
            (abstract,) * 4)
        cur.execute("UPDATE literature_document SET pico_json = %s::jsonb WHERE id = 9801", (_pico("A"),))
        # Below the threshold and excluded: both carry a PICO from an earlier pass, and
        # neither may be counted in the relevant scope.
        cur.execute("UPDATE literature_document SET pico_json = %s::jsonb WHERE id = 9803", (_pico("B"),))
        cur.execute("UPDATE literature_document SET pico_json = %s::jsonb WHERE id = 9804", (_pico("C"),))
        cur.execute("INSERT INTO article_scenarios (scenario_id, document_id, similarity_score, screening_status) VALUES "
                    "(%s, 9801, 0.90, NULL), (%s, 9802, 0.80, NULL),"
                    "(%s, 9803, 0.10, NULL), (%s, 9804, 0.95, 'excluded')", (SID,) * 4)
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


def _bulk(params: str = ""):
    r = _client().get(f"/user-scenarios/{SID}/pico-bulk{params}")
    assert r.status_code == 200, r.text
    return r.json()


def test_by_default_it_serves_the_relevant_subset(seeded):
    body = _bulk()
    assert body["scope"] == "relevant"
    assert {a["id"] for a in body["articles"]} == {9801, 9802}
    assert body["total"] == 2 and body["with_pico"] == 1
    # And it says how big the corpus is, so the tab can state one out of the other.
    assert body["corpus_total"] == 4
    assert body["truncated"] is False


def test_the_whole_corpus_is_one_parameter_away(seeded):
    body = _bulk("?scope=all")
    assert body["scope"] == "all"
    assert {a["id"] for a in body["articles"]} == set(IDS)
    assert body["total"] == 4 and body["corpus_total"] == 4
    # Three of the four carry a PICO, including the ones the relevant scope leaves out.
    assert body["with_pico"] == 3


def test_a_scope_nobody_defined_is_refused_rather_than_guessed(seeded):
    r = _client().get(f"/user-scenarios/{SID}/pico-bulk?scope=europe")
    assert r.status_code == 422 and "scope" in r.text


def test_the_pages_stay_inside_the_scope(seeded):
    """Paging must not fall back to the whole corpus half way through: the second page
    of a relevant scope is still relevant."""
    first = _bulk("?scope=relevant&limit=1")
    assert first["returned"] == 1 and first["truncated"] is True
    assert first["next_offset"] == 1
    second = _bulk(f"?scope=relevant&limit=1&offset={first['next_offset']}")
    assert second["returned"] == 1 and second["truncated"] is False
    assert {first["articles"][0]["id"], second["articles"][0]["id"]} == {9801, 9802}


def test_the_tab_asks_for_the_relevant_subset_and_can_ask_for_everything():
    """The endpoint's default is useless if the tab keeps asking for the whole corpus."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    api_ts = (root / "frontend" / "src" / "lib" / "api.ts").read_text(encoding="utf-8")
    assert 'scope: PicoScope = "relevant"' in api_ts, "the client's default is not the subset"
    assert "&scope=${scope}" in api_ts, "the scope never reaches the endpoint"
    page = (root / "frontend" / "src" / "components" / "ScenarioDetailPage.tsx").read_text(encoding="utf-8")
    assert "fetchAllScenarioPico(scenarioId, 40, picoScope)" in page
    assert "corpus_total" in page, "the tab never says what it is showing out of"
