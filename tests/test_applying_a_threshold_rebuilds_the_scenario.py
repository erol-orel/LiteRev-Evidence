"""Applying a threshold rebuilds what the threshold makes possible, and nothing else.

Choosing the relevant subset is the moment every downstream step becomes computable: the
full text of those articles, their PICO, their structured extraction, the clustering, the
knowledge graph, the brief, the variables, the recommended actions. Until now PATCH
/scenarios/{id}/settings only invalidated the caches, so a reviewer who moved the slider
had to press every button again, and the panels sat empty in the meantime.

The one thing it must NEVER do is search. The pipeline begins with an ingestion step, so
launching it on a scenario that was never populated would turn "apply a threshold" into
"query ten databases". The gate is therefore the corpus, not the name of the scenario.
"""
from __future__ import annotations

import inspect

import pytest

import main
from api import pipeline as P
from api import relevance as R

SID = "usr-threshold-rebuild-test"


def test_the_settings_patch_launches_the_pipeline_behind_the_corpus_gate():
    src = inspect.getsource(R.update_scenario_settings)
    assert "_launch_full_pipeline" in src, "a threshold no longer rebuilds anything"
    assert "_corpus_already_ingested(scenario_id)" in src, "the rebuild is not gated on the corpus"
    # The gate comes FIRST: the launch is inside it, not beside it.
    assert src.index("_corpus_already_ingested(scenario_id)") < src.index("_launch_full_pipeline(scenario_id")
    assert '"no_corpus"' in src, "a scenario with no corpus must say so rather than start a search"
    assert "similarity_threshold" in src and "rerank_threshold" in src


def test_a_pipeline_whose_threshold_moved_while_it_ran_starts_again_once():
    """Moving the slider during a run must not leave the outputs on the old subset, and
    must not loop either: the relaunch is conditional on the threshold having changed."""
    src = inspect.getsource(P._run_user_scenario_full_pipeline)
    assert "threshold_at_start" in src or "_thr_at_start" in src
    assert "_launch_full_pipeline" in src, "the pipeline never picks up a threshold that moved"
    assert "abs(float(_thr_now) - float(_thr_at_start))" in src, "the relaunch is not conditional"


def test_an_unknown_scenario_has_no_corpus_and_starts_nothing():
    assert R._corpus_already_ingested("usr-does-not-exist-at-all") is False


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
        cur.execute("CREATE TABLE IF NOT EXISTS article_scenarios ("
                    "scenario_id TEXT, document_id BIGINT, PRIMARY KEY (scenario_id, document_id))")
        cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))
        cur.execute("INSERT INTO user_scenarios (id, name, query, mode, filters, pinned) "
                    "VALUES (%s, 'Threshold rebuild', 'h5n1', 'boolean', '{}', TRUE)", (SID,))
    yield db_conn
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))


def test_the_corpus_gate_wants_a_finished_search_and_articles(scenario):
    # Never populated: nothing to rebuild.
    assert R._corpus_already_ingested(SID) is False

    with scenario.cursor() as cur:
        cur.execute("UPDATE user_scenarios SET populate_status = 'done' WHERE id = %s", (SID,))
    # Search finished, but the corpus is empty: still nothing.
    assert R._corpus_already_ingested(SID) is False

    with scenario.cursor() as cur:
        cur.execute("INSERT INTO article_scenarios (scenario_id, document_id) VALUES (%s, 1)", (SID,))
    assert R._corpus_already_ingested(SID) is True

    # A search still running is not a corpus to rebuild either: the pipeline doing it
    # will reach the enrichment steps by itself.
    with scenario.cursor() as cur:
        cur.execute("UPDATE user_scenarios SET populate_status = 'running' WHERE id = %s", (SID,))
    assert R._corpus_already_ingested(SID) is False
