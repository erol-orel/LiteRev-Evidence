"""The automatic PICO is paid for one article at a time: only the relevant ones.

The background worker summarized every article linked to a scenario, relevant or not.
On HPAI_last that was 15,547 PICO summaries for 1,075 relevant articles: fourteen calls
paid for one that serves, and a new scenario of thirty thousand records started that
spend the moment its search ended. The PICO is read only on the relevant subset (digest,
brief, extraction), so the batch now goes through the gate the rest of the application
shares, with each scenario's own threshold read inside the statement.
"""
import pytest

pytest.importorskip("fastapi")

import main  # noqa: E402
from api.scenario_store import scenario_scope_sql  # noqa: E402

from sqlalchemy import text  # noqa: E402


# ── the statement (pure) ─────────────────────────────────────────────────────

def test_the_batch_goes_through_the_shared_relevance_gate():
    """Not a fourth copy of the condition: the one `scenario_scope_sql` writes."""
    sql = main.background_pico_sql()
    gate = scenario_scope_sql("relevant", doc="literature_document", link="ars", sid="ars.scenario_id")
    assert gate in sql


def test_each_scenario_brings_its_own_threshold():
    """The threshold is read per link, inside the statement: an article relevant in one
    scenario and not in another is summarized once, for the one that needs it."""
    sql = main.background_pico_sql()
    assert "WHERE ss.scenario_id = ars.scenario_id" in sql
    assert ":thr" not in sql and ":sid" not in sql


def test_what_the_batch_already_bounded_stays_bounded():
    sql = main.background_pico_sql()
    assert "COALESCE(pico_attempts, 0) < :max_attempts" in sql
    assert "pico_fulltext_attempted IS NOT TRUE" in sql
    assert "project_context = 'literev'" in sql
    assert "LIMIT :lim" in sql


def test_the_worker_runs_this_statement_and_no_other():
    """The worker used to carry its own SELECT inline; a copy left behind would keep the
    old spend running under the new name."""
    import inspect
    import api.schema_boot as sb
    src = inspect.getsource(sb.startup_event)
    assert "background_pico_sql()" in src
    assert "FROM literature_document\n                        WHERE project_context" not in src


# ── the batch (database) ─────────────────────────────────────────────────────

S1, S2 = "bg-pico-a", "bg-pico-b"


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
        cur.execute("DROP TABLE IF EXISTS document_search")
        cur.execute("DROP TABLE IF EXISTS document_chunk, article_scenarios, literature_document CASCADE")
        cur.execute(
            "CREATE TABLE literature_document ("
            "id bigint PRIMARY KEY, title text NOT NULL, source text NOT NULL DEFAULT 'pubmed',"
            "abstract text, is_duplicate boolean DEFAULT false, has_fulltext boolean,"
            "pico_json jsonb, pico_attempts int DEFAULT 0, pico_fulltext_attempted boolean,"
            "project_context text DEFAULT 'literev')")
        cur.execute(
            "CREATE TABLE article_scenarios (scenario_id text, document_id bigint,"
            "similarity_score double precision, rerank_score double precision,"
            "screening_status text, PRIMARY KEY (scenario_id, document_id))")
        cur.execute("SELECT to_regclass('scenario_settings') IS NULL")
        if cur.fetchone()[0]:
            main._ensure_scenario_settings_table()
        cur.execute("DELETE FROM scenario_settings WHERE scenario_id IN (%s, %s)", (S1, S2))
        # S1: a threshold a person chose (0.50). S2: none chosen, the starting 0.45 applies.
        cur.execute(
            "INSERT INTO scenario_settings (scenario_id, similarity_threshold, similarity_threshold_set_at) "
            "VALUES (%s, 0.50, now())", (S1,))
        long = "an abstract long enough for the worker to bother reading it at all"
        cur.execute(
            "INSERT INTO literature_document (id, title, abstract, is_duplicate, has_fulltext, pico_json,"
            " pico_attempts, pico_fulltext_attempted) VALUES "
            "(1, 'above the threshold', %(a)s, false, false, NULL, 0, NULL),"
            "(2, 'below the threshold', %(a)s, false, false, NULL, 0, NULL),"
            "(3, 'a duplicate', %(a)s, true, false, NULL, 0, NULL),"
            "(4, 'set aside by a reviewer', %(a)s, false, false, NULL, 0, NULL),"
            "(5, 'rescued by a reviewer', %(a)s, false, false, NULL, 0, NULL),"
            "(6, 'in no scenario', %(a)s, false, false, NULL, 0, NULL),"
            "(7, 'below in S1, above in S2', %(a)s, false, false, NULL, 0, NULL),"
            "(8, 'already summarized', %(a)s, false, false, '{\"pico_source\": \"abstract\"}', 1, NULL),"
            "(9, 'full text arrived since', %(a)s, false, true, '{\"pico_source\": \"abstract\"}', 1, NULL),"
            "(10, 'abstract too short', 'short', false, false, NULL, 0, NULL),"
            "(11, 'failed three times', %(a)s, false, false, NULL, 3, NULL),"
            "(12, 'below the threshold, full text arrived', %(a)s, false, true,"
            " '{\"pico_source\": \"abstract\"}', 1, NULL)",
            {"a": long})
        cur.execute(
            "INSERT INTO article_scenarios (scenario_id, document_id, similarity_score, screening_status) VALUES "
            "(%(s1)s, 1, 0.90, NULL), (%(s1)s, 2, 0.40, NULL), (%(s1)s, 3, 0.95, NULL),"
            "(%(s1)s, 4, 0.95, 'excluded'), (%(s1)s, 5, 0.01, 'included'),"
            "(%(s1)s, 7, 0.40, NULL), (%(s2)s, 7, 0.46, NULL),"
            "(%(s1)s, 8, 0.90, NULL), (%(s1)s, 9, 0.90, NULL), (%(s1)s, 10, 0.90, NULL),"
            "(%(s1)s, 11, 0.90, NULL), (%(s1)s, 12, 0.40, NULL)",
            {"s1": S1, "s2": S2})
    yield db_conn
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM scenario_settings WHERE scenario_id IN (%s, %s)", (S1, S2))


def _batch(limit=50, max_attempts=3):
    with main.engine.connect() as conn:
        rows = conn.execute(text(main.background_pico_sql()),
                            {"lim": limit, "max_attempts": max_attempts}).mappings().fetchall()
    return [r["id"] for r in rows]


def test_only_relevant_articles_are_summarized(seeded):
    assert set(_batch()) == {1, 5, 7, 9}


def test_below_the_threshold_nothing_is_paid_for(seeded):
    """Neither a first summary (2) nor a re-reading once the full text arrives (12)."""
    got = set(_batch())
    assert 2 not in got and 12 not in got


def test_a_reviewer_decision_outranks_the_score_both_ways(seeded):
    got = set(_batch())
    assert 5 in got        # included by hand, below the threshold
    assert 4 not in got    # excluded, far above it


def test_relevant_in_one_scenario_is_enough(seeded):
    """Article 7 sits below S1's chosen 0.50 and above S2's starting 0.45."""
    assert 7 in _batch()


def test_lowering_a_threshold_brings_the_article_in_at_the_next_cycle(seeded, db_conn):
    assert 2 not in _batch()
    with db_conn.cursor() as cur:
        cur.execute("UPDATE scenario_settings SET similarity_threshold = 0.35, similarity_threshold_set_at = now() "
                    "WHERE scenario_id = %s", (S1,))
    got = set(_batch())
    assert {2, 12} <= got


def test_articles_never_summarized_come_first(seeded):
    ids = _batch()
    assert ids.index(9) == len(ids) - 1      # the only re-reading, after the first summaries
