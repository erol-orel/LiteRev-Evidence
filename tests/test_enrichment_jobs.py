"""An enrichment batch runs in the background and says where it is.

The three POSTs (PICO, metadata, open-access links) looped over model calls INSIDE the
request, up to 90 s each. A batch of a few hundred articles outlived the proxy timeout:
the screen showed an error while the server went on working, and paying, unseen. A batch
now starts in the background, one at a time per kind and scenario, and is followed with
GET /enrichment/jobs. `wait=true` keeps the old synchronous answer for the scripts that
call these POSTs and read their JSON.

The batches themselves are stubbed: what is under test is the plumbing around them.
"""
import threading
import time

import pytest

pytest.importorskip("fastapi")

import main  # noqa: E402
from conftest import patch_app  # noqa: E402

HDR = {"X-API-Key": "test-write-key"}
SID = "usr-enrich-jobs"


@pytest.fixture()
def client(monkeypatch):
    from fastapi.testclient import TestClient

    import api.enrichment as E

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with E._enrich_jobs_lock:
        E._enrich_jobs.clear()
    yield TestClient(main.app)
    with E._enrich_jobs_lock:
        E._enrich_jobs.clear()


def _wait_done(client, kind, sid=SID, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get("/enrichment/jobs", params={"scenario_id": sid}).json()["jobs"][kind]
        if job and job["status"] != "running":
            return job
        time.sleep(0.02)
    raise AssertionError(f"{kind} still running after {timeout} s")


def test_a_batch_starts_in_the_background_and_reports_its_progress(client, monkeypatch):
    seen = {}

    def fake(scenario_id=None, limit=None, scope=None, progress=None):
        seen.update(scenario_id=scenario_id, limit=limit, scope=scope)
        for i in range(3):
            progress(i, 3)
        progress(3, 3)
        return {"extracted": 3, "skipped": 0, "errors": 0, "message": "3 articles enrichis"}

    patch_app(monkeypatch, "extract_pico_batch", fake)
    r = client.post("/pico/extract", params={"scenario_id": SID, "scope": "relevant"}, headers=HDR)
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "started" and r.json()["job"]["kind"] == "pico"
    job = _wait_done(client, "pico")
    assert job["status"] == "done" and job["processed"] == 3 and job["total"] == 3
    assert job["result"]["extracted"] == 3 and job["error"] is None
    # No limit means every pending article, not 100000.
    assert seen == {"scenario_id": SID, "limit": None, "scope": "relevant"}
    assert job["limit"] is None and job["elapsed_s"] >= 0


def test_wait_true_keeps_the_synchronous_answer_for_the_scripts(client, monkeypatch):
    patch_app(monkeypatch, "extract_metadata_batch",
              lambda scenario_id=None, limit=None, scope=None, progress=None:
              {"extracted": 2, "skipped": 1, "errors": 0, "message": "2 articles enrichis",
               "limit_seen": limit})
    r = client.post("/metadata/extract", params={"scenario_id": SID, "limit": 100000, "wait": "true"},
                    headers=HDR)
    assert r.status_code == 200
    assert r.json() == {"extracted": 2, "skipped": 1, "errors": 0,
                        "message": "2 articles enrichis", "limit_seen": 100000}
    # Nothing was left behind in the background.
    assert client.get("/enrichment/jobs", params={"scenario_id": SID}).json()["jobs"]["metadata"] is None


def test_one_batch_at_a_time_per_kind_and_scenario(client, monkeypatch):
    release = threading.Event()

    def slow(scenario_id=None, limit=None, scope=None, progress=None):
        progress(0, 10)
        release.wait(5)
        return {"fetched": 0, "not_available": 0, "errors": 0, "message": "0 liens"}

    patch_app(monkeypatch, "fetch_fulltext_batch", slow)
    first = client.post("/fulltext/fetch", params={"scenario_id": SID, "limit": 1000}, headers=HDR).json()
    second = client.post("/fulltext/fetch", params={"scenario_id": SID, "limit": 1000}, headers=HDR).json()
    assert first["status"] == "started" and second["status"] == "running"
    assert second["job"]["limit"] == 1000 and second["job"]["total"] in (None, 10)
    # Another scenario is another job.
    other = client.post("/fulltext/fetch", params={"scenario_id": "usr-other", "limit": 5}, headers=HDR).json()
    assert other["status"] == "started"
    release.set()
    assert _wait_done(client, "fulltext")["status"] == "done"
    assert _wait_done(client, "fulltext", sid="usr-other")["status"] == "done"


def test_a_failing_batch_says_why_instead_of_vanishing(client, monkeypatch):
    def broken(scenario_id=None, limit=None, scope=None, progress=None):
        raise RuntimeError("quota exceeded")

    patch_app(monkeypatch, "extract_pico_batch", broken)
    client.post("/pico/extract", params={"scenario_id": SID}, headers=HDR)
    job = _wait_done(client, "pico")
    assert job["status"] == "error" and "quota exceeded" in job["error"]


def test_bad_input_is_refused_before_anything_starts(client, monkeypatch):
    patch_app(monkeypatch, "extract_pico_batch",
              lambda **kw: pytest.fail("the batch must not start"))
    r = client.post("/pico/extract", params={"scenario_id": SID, "scope": "everything"}, headers=HDR)
    assert r.status_code == 400
    monkeypatch.setenv("OPENAI_API_KEY", "")
    r = client.post("/pico/extract", params={"scenario_id": SID}, headers=HDR)
    assert r.status_code == 503
    assert client.post("/pico/extract", params={"scenario_id": SID}).status_code in (401, 403)
    assert client.get("/enrichment/jobs", params={"scenario_id": SID}).json()["jobs"]["pico"] is None


class _Rows:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return self

    def fetchall(self):
        return self._rows


class _Engine:
    """Two candidate articles, and a record of the writes: no database needed."""

    def __init__(self, rows):
        self.rows, self.writes = rows, []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def connect(self):
        return self

    begin = connect

    def execute(self, sql, params=None):
        if str(sql).lstrip().upper().startswith("UPDATE"):
            self.writes.append(params)
            return _Rows([])
        return _Rows(self.rows)


def test_the_open_access_batch_says_it_found_links_not_texts(monkeypatch):
    """It stores an Unpaywall LINK and no text, yet it answered "N full texts retrieved"."""
    import io
    import json as _json

    import api.enrichment as E

    fake = _Engine([{"id": 1, "title": "Open", "doi": "10.1/open", "url": None},
                    {"id": 2, "title": "Closed", "doi": "10.1/closed", "url": None}])
    monkeypatch.setattr(E, "engine", fake)

    def urlopen(req, timeout=None):                     # Unpaywall, without the network
        is_open = "10.1/open" in req.full_url
        body = {"is_oa": True, "best_oa_location": {"url_for_pdf": "https://oa.example/1.pdf"}} \
            if is_open else {"is_oa": False}
        return io.BytesIO(_json.dumps(body).encode())

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    seen = []
    out = main.fetch_fulltext_batch(scenario_id=SID, limit=None, scope="all",
                                    progress=lambda done, total: seen.append((done, total)))
    assert (out["fetched"], out["not_available"], out["errors"]) == (1, 1, 0)
    assert out["message"] == "1 liens d'accès ouvert trouvés, 1 non disponibles, 0 erreurs"
    assert "textes intégraux" not in out["message"]
    assert fake.writes == [{"url": "https://oa.example/1.pdf", "article_id": 1}]
    assert seen[-1] == (2, 2)                            # the progress a job reports
