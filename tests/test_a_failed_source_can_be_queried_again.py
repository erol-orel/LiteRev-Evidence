"""A source that failed can be queried again, and only it.

A search that loses a source loses whatever only that database holds, and the only remedy
was to re-run the whole search: on the HPAI_team scenario, 40,046 records and twenty
minutes to recover one refused request. The answers of the sources that DID answer are
already kept (`source_query_cache`, SOURCE_CACHE_TTL_S) and replayed without a network
call, while a failed source is never cached, so a re-run queries the failing ones and
nothing else. The card now says which failed and offers that run.

Also here: Europe PMC's long queries. Sent in a URL, their server answers 503 above
roughly 5,000 encoded bytes (measured on three production scenarios); the same query in
the body of a POST has no such limit.
"""
import pytest

pytest.importorskip("fastapi")

import main  # noqa: E402

SID = "retry-failed-sources"
KEY = {"X-API-Key": "test-write-key"}


# ── Pure: how Europe PMC is called ───────────────────────────────────────────
class _FakeResponse:
    def __init__(self, status_code=200):
        self.status_code = status_code


class _FakeRequests:
    """Records the calls a fetcher makes, and answers what the test asks for."""

    def __init__(self, post_status=200):
        self.calls: list[tuple[str, dict]] = []
        self._post_status = post_status

    def get(self, url, params=None, timeout=None, headers=None):
        self.calls.append(("get", {"url": url, "params": params, "headers": headers}))
        return _FakeResponse(200)

    def post(self, url, data=None, timeout=None, headers=None):
        self.calls.append(("post", {"url": url, "data": data, "headers": headers}))
        return _FakeResponse(self._post_status)


def test_a_short_europe_pmc_query_goes_in_the_url():
    req = _FakeRequests()
    main.epmc_send(req, {"query": "avian influenza AND poultry", "format": "json"})
    assert [c[0] for c in req.calls] == ["get"]
    assert req.calls[0][1]["params"]["query"].startswith("avian influenza")


def test_a_long_europe_pmc_query_goes_in_the_body():
    req = _FakeRequests()
    long_query = " OR ".join(f'"a term number {i}"' for i in range(400))
    assert len(long_query) > main.EPMC_POST_ABOVE_ENCODED
    main.epmc_send(req, {"query": long_query, "format": "json"})
    assert [c[0] for c in req.calls] == ["post"], "a long query must not travel in the URL"
    sent = req.calls[0][1]
    assert sent["data"]["query"] == long_query
    assert sent["headers"]["Content-Type"] == "application/x-www-form-urlencoded"


def test_a_server_that_refuses_the_post_is_asked_in_the_url_again():
    req = _FakeRequests(post_status=405)
    main.epmc_send(req, {"query": "x" * 9000, "format": "json"})
    assert [c[0] for c in req.calls] == ["post", "get"], "no source is lost to an unsupported method"


# ── With a database: the endpoints ───────────────────────────────────────────
def _engine_ok() -> bool:
    try:
        with main.engine.connect():
            return True
    except Exception:
        return False


FIGURES = {
    "method": "populate",
    "records_by_source": {"pubmed": 9813, "doaj": 1319, "europepmc": 0, "openalex": 18832},
    "records_identified": 40046,
    "source_outcomes": {"pubmed": "ok", "openalex": "ok", "doaj": "error", "europepmc": "error",
                        "arxiv": "skipped"},
    "sources_failed": ["doaj", "europepmc"],
    "sources_cut_off": [],
    "sources_skipped": ["arxiv"],
    "source_error_reasons": {"doaj": "400 Client Error: Bad Request", "europepmc": "503 Server Error"},
    "records_screened": 27869,
}


@pytest.fixture()
def seeded(db_conn, monkeypatch):
    if not _engine_ok():
        pytest.skip("main.engine cannot reach the database")
    from conftest import patch_app
    launched: list[dict] = []
    patch_app(monkeypatch, "_launch_populate_job",
              lambda sid, query, filters, max_results, include_live=True, lang=None,
              force_live=False, auto_pipeline=True: launched.append(
                  {"sid": sid, "include_live": include_live, "force_live": force_live,
                   "auto_pipeline": auto_pipeline}) or "started")
    with db_conn.cursor() as cur:
        cur.execute("SELECT to_regclass('user_scenarios') IS NULL")
        if cur.fetchone()[0]:
            main._ensure_user_scenarios_table()
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))
        cur.execute("INSERT INTO user_scenarios (id, name, query, mode, filters)"
                    " VALUES (%s, 'Retry', 'avian influenza AND occupational', 'boolean', '{}')", (SID,))
    main._store_prisma_identification(SID, FIGURES)
    main._user_scenario_populate_jobs.pop(SID, None)
    yield db_conn, launched
    main._user_scenario_populate_jobs.pop(SID, None)
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))


def _client():
    from fastapi.testclient import TestClient
    return TestClient(main.app)


def test_the_card_says_which_sources_failed_and_why(seeded):
    r = _client().get(f"/user-scenarios/{SID}/sources/failed")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["failed"] == ["doaj", "europepmc"], "a skipped source is not a failed one"
    assert body["reasons"]["europepmc"].startswith("503")
    assert body["retryable"] is True and body["search_running"] is False
    assert body["cache_ttl_hours"] > 0


def test_querying_the_failing_sources_again_needs_the_key_and_starts_one_run(seeded):
    _conn, launched = seeded
    c = _client()
    assert c.post(f"/user-scenarios/{SID}/sources/retry-failed").status_code == 401
    assert launched == []
    r = c.post(f"/user-scenarios/{SID}/sources/retry-failed", headers=KEY)
    assert r.status_code == 200, r.text
    assert r.json()["retrying"] == ["doaj", "europepmc"]
    assert len(launched) == 1
    run = launched[0]
    # The cache must be ON: that is what makes the run query the failing sources only.
    assert run["force_live"] is False and run["include_live"] is True
    # A retry fills the corpus; it does not re-run the extraction.
    assert run["auto_pipeline"] is False


def test_nothing_failed_is_not_a_retry(seeded, db_conn):
    main._store_prisma_identification(SID, {**FIGURES, "source_outcomes": {"pubmed": "ok"},
                                            "sources_failed": [], "sources_cut_off": [],
                                            "source_error_reasons": {}})
    c = _client()
    assert c.get(f"/user-scenarios/{SID}/sources/failed").json()["failed"] == []
    r = c.post(f"/user-scenarios/{SID}/sources/retry-failed", headers=KEY)
    assert r.status_code == 422


def test_a_search_already_running_is_not_disturbed(seeded):
    _conn, launched = seeded
    main._user_scenario_populate_jobs[SID] = {"status": "running", "ingested": 12}
    c = _client()
    state = c.get(f"/user-scenarios/{SID}/sources/failed").json()
    assert state == c.get(f"/user-scenarios/{SID}/sources/failed").json(), "reading it changes nothing"
    assert state["search_running"] is True and state["retryable"] is False
    assert c.post(f"/user-scenarios/{SID}/sources/retry-failed", headers=KEY).status_code == 409
    assert launched == []


def test_an_unknown_scenario_is_a_404():
    if not _engine_ok():
        pytest.skip("main.engine cannot reach the database")
    c = _client()
    assert c.get("/user-scenarios/usr-nope-nope/sources/failed").status_code == 404
    assert c.post("/user-scenarios/usr-nope-nope/sources/retry-failed", headers=KEY).status_code == 404
