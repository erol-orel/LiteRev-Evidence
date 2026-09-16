"""scripts/preflight_demo.py — the demo readiness report, on canned API answers.

Pure: `get`/`post` are fakes, `sleep` is a no-op, the audit is skipped."""
import importlib.util
import os

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "preflight_demo.py")
_spec = importlib.util.spec_from_file_location("preflight_demo", _PATH)
preflight = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(preflight)

SID = "usr-demo"


def _healthy():
    return {"status": "ok", "database": "ok", "schema": {"ok": True},
            "lexical_search": {"engine": "fts"},
            "process": {"rss_mb": 400.0, "rss_peak_mb": 900.0, "threads": 12, "uptime_s": 7200},
            "rate_limit": {"general_per_min": 600, "expensive_per_min": 30}}


class FakeApi:
    """Answers by path prefix; records the POSTs; a brief that is generated on demand."""

    def __init__(self, health=None, degraded=False, brief_status="ready", polls_until_ready=2):
        self.health = health or _healthy()
        self.degraded = degraded
        self.brief_status = brief_status
        self.polls_until_ready = polls_until_ready
        self.posts: list[str] = []
        self.brief_polls = 0

    def get(self, path):
        if path == "/health":
            return 200, self.health
        if "/detail" in path:
            return 200, {"title": "Flu surveillance", "corpus_stats": {"total": 40}}
        if path.endswith("/counts"):
            return 200, {"consistent": True}
        if "/clustering/status" in path:
            return 200, {"status": "done", "clusters": [{}, {}], "n_docs": 40, "lang": "en", "from_cache": True}
        if "/clustering" in path:
            return 200, {"status": "running"}
        if "/knowledge-graph" in path:
            return 200, {"nodes": [1, 2, 3], "edges": [1]}
        if "/concept-graph" in path:
            return 200, {"kind": "concepts", "nodes": [1, 2], "edges": [1], "source": "llm",
                         "enriching": False, "n_missing_concepts": 0}
        if path.endswith("/evidence-brief"):
            return 200, {"corpus_stats": {"total": 40}}
        if "/evidence-brief/llm" in path:
            if self.brief_status == "ready":
                return 200, {"summary": "…", "lang": "en"}
            if self.brief_status == "generating":
                self.brief_polls += 1
                if self.brief_polls > self.polls_until_ready:
                    return 200, {"summary": "…", "lang": "en"}
                return 200, {"status": "generating"}
            return 200, {"status": self.brief_status}
        if "/variables" in path:
            return 200, {"variables": [], "status": "validated"}
        if "/recommended-actions" in path:
            return 200, {"status": "ready", "actions": ["a"]}
        if "/embedding-status" in path:
            return 200, {"status": "ready", "corpus_total": 40, "title_abstract_chunks": {"embedded_docs": 40},
                         "ranking": {"scored": 40}, "score_availability": {"cohere_configured": True}}
        if "/seir/projection" in path:
            return 200, {"applicable": True}
        return 200, {"status": "ok"}

    def post(self, path, payload):
        self.posts.append(path)
        if path == "/search-strategy":
            return 200, {"general": "x", "degraded": self.degraded}
        if "/evidence-brief/generate" in path:
            self.brief_status = "generating"
            return 200, {"status": "started"}
        return 200, {}


def _run(api, **kw):
    pf = preflight.Preflight(api.get, api.post, [SID], lang="en", sleep=lambda s: None, audit=None, **kw)
    rc = pf.run()
    levels = {check: level for level, check, _ in pf.rows}
    return rc, levels, pf


def test_a_ready_api_is_green():
    rc, levels, pf = _run(FakeApi(), api_key="k")
    assert rc == 0
    assert all(level == "OK" for level in levels.values()), levels
    assert "usr-demo: clustering" in levels and "usr-demo: LLM brief" in levels


def test_schema_and_openai_problems_fail():
    health = _healthy()
    health["schema"] = {"ok": False, "missing_tables": ["article_scenarios"], "ddl_failures": 1}
    rc, levels, _ = _run(FakeApi(health=health, degraded=True), api_key="k")
    assert rc == 1
    assert levels["health: schema"] == "FAIL"
    assert levels["OpenAI"] == "FAIL"


def test_missing_brief_is_generated_with_a_key_and_only_reported_without():
    api = FakeApi(brief_status="empty")
    rc, levels, _ = _run(api, api_key="k")
    assert any("/evidence-brief/generate" in p for p in api.posts)
    assert levels["usr-demo: LLM brief"] == "OK" and rc == 0

    api = FakeApi(brief_status="empty")
    rc, levels, _ = _run(api, api_key=None)
    assert not any("/evidence-brief/generate" in p for p in api.posts)
    assert levels["usr-demo: LLM brief"] == "WARN" and rc == 0


def test_a_generation_that_never_finishes_is_a_warning_and_a_failure_is_a_fail():
    api = FakeApi(brief_status="generating", polls_until_ready=10**6)
    rc, levels, _ = _run(api, api_key="k", wait_s=0)
    assert levels["usr-demo: LLM brief"] == "WARN"
    api = FakeApi(brief_status="error")
    rc, levels, _ = _run(api, api_key="k")
    assert levels["usr-demo: LLM brief"] == "FAIL" and rc == 1


def test_slow_endpoints_and_a_recent_restart_are_warnings(monkeypatch):
    health = _healthy()
    health["process"]["uptime_s"] = 60
    api = FakeApi(health=health)
    rc, levels, _ = _run(api, api_key="k", slow_ms=-1)          # every read is "slow"
    assert rc == 0
    assert levels["health: uptime"] == "WARN"
    assert levels["usr-demo: PRISMA"] == "WARN"
