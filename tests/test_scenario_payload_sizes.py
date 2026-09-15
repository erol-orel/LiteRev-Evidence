"""Payload-size regressions found by scripts/bench_scenario.py on a 25,000-article
scenario:

- GET /scenarios/{id}/settings returned `SELECT *` of scenario_settings, i.e. the
  whole clustering / knowledge-graph / brief caches (2.5 MB) to read one threshold;
- GET /user-scenarios/{id}/corpus?limit=10000 (search results page) shipped 10,000
  full abstracts (27 MB) for a 600-character excerpt.
"""
import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

import main  # noqa: E402

SID = "usr-payload-size-test"


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
        for tbl, ensure in (("user_scenarios", main._ensure_user_scenarios_table),
                            ("scenario_settings", getattr(main, "_ensure_scenario_settings_table", None))):
            cur.execute("SELECT to_regclass(%s) IS NULL", (tbl,))
            if cur.fetchone()[0] and ensure:
                ensure()
        # Columns the corpus endpoint reads but a suite-bootstrapped database lacks
        # (CI has no pgvector, so schema.sql is not applied there and the documents
        # table comes from a minimal fixture): the boot DDL that production relies on,
        # plus the schema.sql base columns.
        cur.execute("SELECT to_regclass('document_chunk') IS NULL")
        created_chunk_table = cur.fetchone()[0]
        if created_chunk_table:                       # dropped again at teardown
            cur.execute("""
                CREATE TABLE document_chunk (
                    id BIGSERIAL PRIMARY KEY, document_id BIGINT NOT NULL,
                    chunk_index INTEGER NOT NULL DEFAULT 0, content TEXT NOT NULL DEFAULT '',
                    chunk_type TEXT, created_at TIMESTAMP DEFAULT now())""")
        for ensure_name in ("_ensure_bibliographic_columns", "_ensure_double_blind_columns", "_ensure_dedup_columns"):
            ensure = getattr(main, ensure_name, None)
            if ensure:
                ensure()
        for col, typ in (("created_at", "TIMESTAMP DEFAULT now()"), ("url", "TEXT"), ("pmid", "TEXT"),
                         ("year", "INTEGER"), ("source", "TEXT"), ("keywords", "TEXT"), ("language", "TEXT"),
                         ("open_access", "BOOLEAN"), ("sample_size", "INTEGER")):
            cur.execute(f"ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS {col} {typ}")
        for col, typ in (("rerank_score", "FLOAT"), ("screening_status", "TEXT"), ("reviewer_1_status", "VARCHAR(20)")):
            cur.execute(f"ALTER TABLE article_scenarios ADD COLUMN IF NOT EXISTS {col} {typ}")
        cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM scenario_settings WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))
        cur.execute("DELETE FROM literature_document WHERE id IN (9101, 9102)")
        cur.execute(
            "INSERT INTO user_scenarios (id, name, query, mode, filters, pinned, article_count, populate_status, pipeline_status) "
            "VALUES (%s, 'Payload size', 'influenza', 'boolean', '{}', TRUE, 2, 'done', 'done')", (SID,))
        long_abstract = "word " * 400                       # 2,000 characters
        cur.execute("INSERT INTO literature_document (id, title, source, abstract, is_duplicate) VALUES "
                    "(9101, 'A', 'pubmed', %s, false), (9102, 'B', 'pubmed', %s, false)", (long_abstract, long_abstract))
        cur.execute("INSERT INTO article_scenarios (scenario_id, document_id, similarity_score) VALUES "
                    "(%s, 9101, 0.9), (%s, 9102, 0.8)", (SID, SID))
        big_blob = json.dumps({"clusters": [{"points": [{"x": i, "y": i} for i in range(20000)]}]})
        cur.execute(
            "INSERT INTO scenario_settings (scenario_id, similarity_threshold, clustering_json, updated_at) "
            "VALUES (%s, 0.6, %s::jsonb, NOW())", (SID, big_blob))
    yield db_conn
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM scenario_settings WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))
        cur.execute("DELETE FROM literature_document WHERE id IN (9101, 9102)")
        if created_chunk_table:
            cur.execute("DROP TABLE document_chunk")


def test_settings_do_not_ship_the_cached_artifacts(seeded):
    from fastapi.testclient import TestClient
    r = TestClient(main.app).get(f"/scenarios/{SID}/settings")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["similarity_threshold"] == 0.6
    assert "clustering_json" not in body and "knowledge_graph_json" not in body
    assert body["cached"]["clustering"] is True and body["cached"]["evidence_brief"] is False
    assert len(r.content) < 2_000, len(r.content)            # was ~2.5 MB with the blob


def test_settings_default_when_no_row():
    if not _engine_ok():
        pytest.skip("main.engine cannot reach the database")
    from fastapi.testclient import TestClient
    body = TestClient(main.app).get("/scenarios/usr-does-not-exist/settings").json()
    assert body["similarity_threshold"] == main.DEFAULT_SIMILARITY_THRESHOLD
    assert body["cached"] == {"evidence_brief": False, "variables": False, "clustering": False,
                              "knowledge_graph": False, "recommended_actions": False}


def test_corpus_abstracts_can_be_truncated_for_excerpt_views(seeded):
    from fastapi.testclient import TestClient
    client = TestClient(main.app)
    full = client.get(f"/user-scenarios/{SID}/corpus?limit=10").json()
    assert full["abstract_truncated"] is False
    assert all(len(a["abstract"]) == 2000 for a in full["articles"])
    short = client.get(f"/user-scenarios/{SID}/corpus?limit=10&abstract_chars=600").json()
    assert short["abstract_truncated"] is True
    assert all(len(a["abstract"]) == 600 for a in short["articles"])
    assert short["total"] == full["total"] == 2                # counts unaffected
    assert client.get(f"/user-scenarios/{SID}/corpus?abstract_chars=-1").status_code == 422
