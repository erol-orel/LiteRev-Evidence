"""A corpus build (populate job) that crashes before its own error handling must not
stay "running" forever.

`_run_user_scenario_populate` reports its failures itself, but only once inside its
body: an exception raised earlier (its imports, for one - the browser smoke test hit
`ModuleNotFoundError: requests` on a bare API) killed the thread, and the in-memory
job kept saying "running" while the search page polled it without end. The launcher
now wraps the thread target. Pure: no database needed (the status update is
best-effort and logged when the database is unreachable).
"""
import time

import main
from conftest import patch_app  # noqa: E402


def test_launch_populate_job_marks_a_crash_as_error(monkeypatch):
    sid = "usr-populate-guard-test"

    def _boom(*_a, **_k):
        raise ModuleNotFoundError("No module named 'requests'")

    patch_app(monkeypatch, "_run_user_scenario_populate", _boom)
    main._user_scenario_populate_jobs.pop(sid, None)

    assert main._launch_populate_job(sid, "influenza AND surveillance", {}, 50, include_live=False) == "started"

    deadline = time.time() + 5
    while time.time() < deadline and main._user_scenario_populate_jobs.get(sid, {}).get("status") != "error":
        time.sleep(0.05)
    job = main._user_scenario_populate_jobs.pop(sid)
    assert job["status"] == "error"
    assert "requests" in job["error"]
    assert job["ingested"] == 0
