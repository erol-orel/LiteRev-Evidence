"""Integration test: `_build_where`'s scenario filter, executed against a real
Postgres, returns Migration-1 "Way B" (article_scenarios membership) — including
documents cross-scored into a scenario they were not ingested under.

Skips automatically when no Postgres is reachable (see conftest.db_conn), so CI
without a DB service still passes on the pure-logic tests.
"""
import re

import main


def _seed(conn):
    with conn.cursor() as c:
        c.execute("DROP TABLE IF EXISTS article_scenarios, literature_document CASCADE")
        c.execute(
            "CREATE TABLE literature_document ("
            "id int PRIMARY KEY, scenario_type text, is_duplicate boolean, project_context text)"
        )
        c.execute(
            "CREATE TABLE article_scenarios ("
            "scenario_id text, document_id int, PRIMARY KEY (scenario_id, document_id))"
        )
        c.execute(
            "INSERT INTO literature_document VALUES "
            "(1,'sc-a',false,'literev'),(2,'sc-a',false,'literev'),(3,'sc-b',false,'literev')"
        )
        # Migration-1 backfill: membership mirrors scenario_type…
        c.execute("INSERT INTO article_scenarios SELECT scenario_type, id FROM literature_document")
        # …then doc 3 (ingested under sc-b) is ALSO scored into sc-a.
        c.execute("INSERT INTO article_scenarios VALUES ('sc-a', 3)")


def _ids_for(conn, filters):
    """Run main._build_where(filters) against the seeded table, return doc ids.

    _build_where emits SQLAlchemy ':name' binds; convert to psycopg '%(name)s'.
    """
    where_sql, params = main._build_where(filters)
    pg_sql = re.sub(r":(\w+)", r"%(\1)s", where_sql)
    with conn.cursor() as c:
        c.execute(f"SELECT d.id FROM literature_document d WHERE 1=1 {pg_sql} ORDER BY d.id", params)
        return [row[0] for row in c.fetchall()]


def test_scenario_filter_returns_way_b_membership(db_conn):
    _seed(db_conn)
    # Way B = every doc scored into sc-a: 1,2 (ingested) + 3 (cross-scored).
    # Way A (legacy d.scenario_type = 'sc-a') would wrongly return only [1, 2].
    assert _ids_for(db_conn, {"scenario_type": "sc-a"}) == [1, 2, 3]


def test_scenario_filter_excludes_non_members(db_conn):
    _seed(db_conn)
    # Only doc 3 is a member of sc-b.
    assert _ids_for(db_conn, {"scenario_type": "sc-b"}) == [3]
