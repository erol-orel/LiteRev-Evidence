"""The PRISMA exclusion reasons add up to the exclusion total.

The card lists WHY the excluded were excluded, the twelve most frequent reasons first. It
stopped there, with no line for the rest: a review with fifteen distinct reasons showed
twelve lines summing to less than the "excluded" figure printed just above them, and a
reader could not tell whether the difference was a bug or a reason nobody wrote down. The
rest now has its own line, so the list sums to the total.
"""
import pytest

pytest.importorskip("fastapi")

import main  # noqa: E402
from conftest import ensure_document_columns  # noqa: E402

SID = "usr-prisma-reasons"
IDS = list(range(9420, 9437))           # 17 articles, a range no other test seeds


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
        cur.execute("SELECT to_regclass('scenario_settings') IS NULL")
        if cur.fetchone()[0]:
            main._ensure_scenario_settings_table()
        # A fresh database, or one an earlier test left without them (several drop them).
        cur.execute("CREATE TABLE IF NOT EXISTS literature_document (id BIGINT PRIMARY KEY)")
        cur.execute("CREATE TABLE IF NOT EXISTS article_scenarios ("
                    "scenario_id TEXT, document_id BIGINT, PRIMARY KEY (scenario_id, document_id))")
        created_chunk_table = ensure_document_columns(cur)
        for col, typ in (("title", "TEXT"), ("abstract", "TEXT"), ("is_duplicate", "BOOLEAN DEFAULT FALSE")):
            cur.execute(f"ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS {col} {typ}")
        for col, typ in (("similarity_score", "DOUBLE PRECISION"), ("screening_reason", "TEXT"),
                         ("screened_at", "TIMESTAMP")):
            cur.execute(f"ALTER TABLE article_scenarios ADD COLUMN IF NOT EXISTS {col} {typ}")
        cur.execute("ALTER TABLE document_chunk ADD COLUMN IF NOT EXISTS embedding TEXT")
        cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM scenario_settings WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))
        cur.execute("DELETE FROM literature_document WHERE id = ANY(%s)", (IDS,))
        cur.execute("INSERT INTO user_scenarios (id, name, query, mode, filters, pinned) "
                    "VALUES (%s, 'Reasons', 'q', 'boolean', '{}', TRUE)", (SID,))
        for i in IDS:
            cur.execute("INSERT INTO literature_document (id, title, abstract, source, is_duplicate, "
                        "project_context) VALUES (%s, %s, 'an abstract', 'pubmed', false, 'literev')",
                        (i, f"Article {i}"))
        # Fifteen excluded: two share "wrong population", thirteen have a reason each, and
        # one has none. Two more are kept, to show they are not counted.
        reasons = ["wrong population", "wrong population"] + [f"reason {k:02d}" for k in range(12)] + [None]
        for i, reason in zip(IDS, reasons):
            cur.execute("INSERT INTO article_scenarios (scenario_id, document_id, similarity_score, "
                        "screening_status, screening_reason) VALUES (%s, %s, 0.9, 'excluded', %s)",
                        (SID, i, reason))
        for i in IDS[len(reasons):]:
            cur.execute("INSERT INTO article_scenarios (scenario_id, document_id, similarity_score) "
                        "VALUES (%s, %s, 0.9)", (SID, i))
    yield db_conn
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM scenario_settings WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))
        cur.execute("DELETE FROM literature_document WHERE id = ANY(%s)", (IDS,))
        if created_chunk_table:
            cur.execute("DROP TABLE document_chunk")


def test_the_listed_reasons_sum_to_the_exclusion_total(seeded):
    from fastapi.testclient import TestClient

    prisma = TestClient(main.app).get(f"/user-scenarios/{SID}/prisma").json()
    mc = prisma["manual_curation"]
    rows = mc["excluded_by_reason"]
    assert mc["excluded"] == 15
    assert sum(r["articles"] for r in rows) == 15, rows
    # Twelve reasons, then the rest on one line that says how many reasons it groups.
    assert len(rows) == 13
    assert rows[0] == {"reason": "wrong population", "articles": 2}
    other = rows[-1]
    assert other["other"] is True and other["articles"] == 2 and other["reasons"] == 2
    assert other["reason"] == "(autres motifs : 2)"


def test_a_short_list_has_no_other_line(seeded):
    from fastapi.testclient import TestClient

    with seeded.cursor() as cur:
        cur.execute("UPDATE article_scenarios SET screening_reason = 'wrong population' "
                    "WHERE scenario_id = %s AND screening_status = 'excluded'", (SID,))
    rows = TestClient(main.app).get(f"/user-scenarios/{SID}/prisma").json()[
        "manual_curation"]["excluded_by_reason"]
    assert rows == [{"reason": "wrong population", "articles": 15}]
