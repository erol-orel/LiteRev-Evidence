"""Everything the pipeline caches is produced in the language of the interface that
started it — the pin, the "rebuild everything" button, the search.

Seen on September 16: a scenario pinned under the English toggle, "Pipeline finished",
and yet the Variables & Model, Visualization and knowledge-graph tabs computed at their
first opening. The pipeline had no notion of language: cluster summaries were left for
the first opening, the brief and the variables were generated in French (then
regenerated or translated at the first English view) and the recommended actions were
not a step at all. The launchers now forward the toggle language to the pipeline, whose
steps use it, and the pipeline generates the actions too. Pure: the workers are stubbed.
"""
import time

import main
from conftest import patch_app  # noqa: E402

SID = "usr-pipeline-lang-test"


def _wait(pred, seconds=5.0):
    deadline = time.time() + seconds
    while time.time() < deadline and not pred():
        time.sleep(0.02)
    return pred()


def test_the_full_pipeline_launcher_forwards_the_language(monkeypatch):
    calls = []
    patch_app(monkeypatch, "_run_user_scenario_full_pipeline", lambda *a, **k: calls.append(a))
    patch_app(monkeypatch, "_get_user_scenario_or_404",
              lambda sid: {"id": sid, "query": "chikungunya AND europe", "filters": {}})
    main._user_scenario_pipeline_jobs.pop(SID, None)

    assert main._launch_full_pipeline(SID, 50, lang="en") == "started"
    assert _wait(lambda: calls)
    scenario_id, query, filters, max_results, lang = calls[0]
    assert (scenario_id, query, filters, max_results, lang) == (SID, "chikungunya AND europe", {}, 50, "en")
    job = main._user_scenario_pipeline_jobs.pop(SID)
    assert job["lang"] == "en"
    assert "actions" in job["steps"]                        # the actions are a pipeline step now

    # No language → French, the historical default (never "None").
    main._user_scenario_pipeline_jobs.pop(SID, None)
    assert main._launch_full_pipeline(SID, 50) == "started"
    assert _wait(lambda: len(calls) == 2)
    assert calls[1][4] == "fr"
    main._user_scenario_pipeline_jobs.pop(SID, None)


def test_the_populate_launcher_forwards_the_language(monkeypatch):
    calls = []
    patch_app(monkeypatch, "_run_user_scenario_populate", lambda *a, **k: calls.append(a))
    main._user_scenario_populate_jobs.pop(SID, None)

    assert main._launch_populate_job(SID, "chikungunya", {}, 50, include_live=False, lang="EN") == "started"
    assert _wait(lambda: calls)
    assert calls[0][-1] == "en"                             # normalised, last positional argument
    main._user_scenario_populate_jobs.pop(SID, None)


def test_the_pipeline_endpoint_and_the_pin_forward_the_toggle_language(monkeypatch):
    from fastapi.testclient import TestClient

    seen = []
    patch_app(monkeypatch, "_launch_full_pipeline",
              lambda sid, max_results=0, lang=None: seen.append((sid, lang)) or "started")
    patch_app(monkeypatch, "_get_user_scenario_or_404",
              lambda sid: {"id": sid, "query": "q", "filters": {}, "name": "n", "pinned": True})
    client = TestClient(main.app)
    r = client.post(f"/user-scenarios/{SID}/pipeline?lang=en", headers={"X-API-Key": "test-write-key"})
    assert r.status_code == 200, r.text
    assert seen[-1] == (SID, "en")
