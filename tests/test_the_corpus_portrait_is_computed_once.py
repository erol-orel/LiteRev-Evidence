"""The Evidence tab's corpus portrait is computed once, not on every visit.

Six aggregates over the whole corpus: ten seconds on the 52,765 articles of the ISS
strategy, fourteen on the 86,716 of the COVID exposure review, measured against
production. The tab ran them again every time it was opened, which is what "Loading
corpus statistics..." was, every visit, for as long as it took.

They depend only on the corpus, so they are cached beside the clustering, the knowledge
graph and the recommended actions: the pipeline writes the cache after a threshold is
applied, anything that changes the corpus clears it with the others, and `refresh=1`
forces a recomputation.
"""
from __future__ import annotations

import inspect

import pytest

pytest.importorskip("fastapi")

import main  # noqa: E402
from api import evidence as E  # noqa: E402
from api import pipeline as P  # noqa: E402
from api.scenario_store import FREE_CACHE_RESET  # noqa: E402

SID = "usr-brief-cache-test"


def test_the_cache_is_cleared_because_the_portrait_costs_nothing():
    """If it were not in that list, a threshold change would serve the PREVIOUS corpus's
    figures under the new one, which is worse than a slow tab. It is in the FREE list
    rather than kept and dated because it is SQL: the tab recomputes it on finding it
    absent, so dropping it is how it refreshes. The narrative report beside it, which
    costs a model call, is in neither list and dates itself."""
    assert "evidence_brief_json = NULL" in FREE_CACHE_RESET
    assert "evidence_brief_at = NULL" in FREE_CACHE_RESET


def test_the_endpoint_serves_the_cache_and_can_be_forced():
    src = inspect.getsource(E.get_user_scenario_evidence_brief)
    assert "cached_evidence_brief(scenario_id)" in src
    assert "store_evidence_brief(scenario_id, out)" in src
    assert "refresh" in src, "there is no way to recompute on demand"
    # The cache is read BEFORE the expensive build, not after it.
    assert src.index("cached_evidence_brief") < src.index("_build_evidence_brief")


def test_the_pipeline_fills_the_cache_so_the_tab_never_waits():
    src = inspect.getsource(P._run_user_scenario_full_pipeline)
    step = src[src.index('update_step("evidence", "running")'):src.index('update_step("evidence", "done")')]
    assert "store_evidence_brief(scenario_id, _build_evidence_brief(scenario_id))" in step


def test_neither_half_of_the_cache_ever_raises():
    """A cache that cannot be read or written must not break the answer it accompanies:
    both halves swallow their exception and say so in the log."""
    for fn in (E.cached_evidence_brief, E.store_evidence_brief):
        src = inspect.getsource(fn)
        assert "except Exception" in src and "logger.warning" in src, fn.__name__


@pytest.fixture()
def scenario(db_conn):
    try:
        with main.engine.connect():
            pass
    except Exception:
        pytest.skip("main.engine cannot reach the database")
    with db_conn.cursor() as cur:
        cur.execute("SELECT to_regclass('user_scenarios') IS NULL")
        if cur.fetchone()[0]:
            main._ensure_user_scenarios_table()
        # The stamp reads these, and a stamp it cannot compute forbids serving the cache.
        cur.execute("CREATE TABLE IF NOT EXISTS article_scenarios ("
                    "scenario_id TEXT, document_id BIGINT, PRIMARY KEY (scenario_id, document_id))")
        for col, typ in (("screening_status", "TEXT"), ("screened_at", "TIMESTAMP"),
                         ("reviewer_1_status", "VARCHAR(20)"), ("reviewer_2_status", "VARCHAR(20)")):
            cur.execute(f"ALTER TABLE article_scenarios ADD COLUMN IF NOT EXISTS {col} {typ}")
        cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM scenario_settings WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))
        cur.execute("INSERT INTO user_scenarios (id, name, query, mode, filters, pinned) "
                    "VALUES (%s, 'Brief cache', 'h5n1', 'boolean', '{}', TRUE)", (SID,))
        cur.execute("INSERT INTO article_scenarios (scenario_id, document_id) "
                    "VALUES (%s, 1), (%s, 2)", (SID, SID))
    yield db_conn
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM scenario_settings WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))


def test_what_is_stored_comes_back(scenario):
    assert E.cached_evidence_brief(SID) is None               # nothing yet
    assert E.store_evidence_brief(SID, {"corpus_stats": {"total": 7}, "generated_at": "now"})
    got = E.cached_evidence_brief(SID)
    assert got is not None
    assert got["corpus_stats"]["total"] == 7
    assert got["cached_at"], "a cache without a date cannot be judged stale"

    # And clearing the corpus-derived caches clears it with the others.
    with main.engine.begin() as conn:
        from sqlalchemy import text as _t
        conn.execute(_t(f"UPDATE scenario_settings SET {FREE_CACHE_RESET} "
                        f"WHERE scenario_id = :sid"), {"sid": SID})
    assert E.cached_evidence_brief(SID) is None


def test_an_empty_cache_is_not_a_hit(scenario):
    """An empty object is not a portrait: serving it would show a corpus of nothing."""
    assert E.store_evidence_brief(SID, {})
    assert E.cached_evidence_brief(SID) is None
