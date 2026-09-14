"""A search keeps running on the server when the person leaves the page; the interface
must be able to find it again from anywhere.

GET /activity lists what is running across all user scenarios, from the persisted
statuses (populate_status 'running' is now set at launch, pipeline_status
'running'/'starting' by the pipeline), enriched with the current step from the
in-memory jobs. The header polls it and offers "view results" / "open scenario".
"""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

import main  # noqa: E402

RUN_SEARCH = "activity-search-running"
RUN_PIPE = "activity-pipeline-running"
IDLE = "activity-idle"


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
        cur.execute("DELETE FROM user_scenarios WHERE id IN (%s, %s, %s)", (RUN_SEARCH, RUN_PIPE, IDLE))
        cur.execute(
            "INSERT INTO user_scenarios (id, name, query, mode, filters, pinned, article_count, populate_status, pipeline_status, pipeline_step) VALUES "
            "(%s, 'Running search', 'influenza AND wastewater', 'boolean', '{}', FALSE, 120, 'running', NULL, NULL),"
            "(%s, 'Running pipeline', 'RSV early warning', 'boolean', '{}', TRUE, 4000, 'done', 'running', 'fulltext'),"
            "(%s, 'Idle', 'measles', 'boolean', '{}', TRUE, 10, 'done', 'done', 'done')",
            (RUN_SEARCH, RUN_PIPE, IDLE))
    main._user_scenario_populate_jobs[RUN_SEARCH] = {"status": "running", "phase": "federation"}
    yield db_conn
    main._user_scenario_populate_jobs.pop(RUN_SEARCH, None)
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM user_scenarios WHERE id IN (%s, %s, %s)", (RUN_SEARCH, RUN_PIPE, IDLE))


def test_running_searches_and_pipelines_are_listed_with_their_step(seeded):
    from fastapi.testclient import TestClient
    r = TestClient(main.app).get("/activity")
    assert r.status_code == 200, r.text
    body = r.json()
    items = {i["scenario_id"]: i for i in body["running"] if i["scenario_id"] in (RUN_SEARCH, RUN_PIPE, IDLE)}
    assert set(items) == {RUN_SEARCH, RUN_PIPE}          # the idle one is not activity
    assert items[RUN_SEARCH]["kind"] == "search"
    assert items[RUN_SEARCH]["step"] == "federation"     # from the in-memory job
    assert items[RUN_SEARCH]["pinned"] is False and items[RUN_SEARCH]["query"] == "influenza AND wastewater"
    assert items[RUN_PIPE]["kind"] == "pipeline"
    assert items[RUN_PIPE]["step"] == "fulltext"         # persisted step, no in-memory job
    assert items[RUN_PIPE]["article_count"] == 4000
    assert body["count"] >= 2 and body["checked_at"]
