"""A "Failed to fetch" in the browser must be diagnosable from the server: /health
reports the process memory, threads, uptime and DB pool, and every request slower
than SLOW_REQUEST_MS is logged with its route, duration, status and size."""
import logging

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

import main  # noqa: E402
from conftest import patch_app  # noqa: E402


def _engine_ok() -> bool:
    try:
        with main.engine.connect():
            return True
    except Exception:
        return False


def test_health_reports_process_stats():
    if not _engine_ok():
        pytest.skip("main.engine cannot reach the database")
    from fastapi.testclient import TestClient
    body = TestClient(main.app).get("/health").json()
    proc = body["process"]
    assert proc["uptime_s"] >= 0
    assert proc["rss_mb"] > 0 and proc["rss_peak_mb"] >= proc["rss_mb"]
    assert proc["threads"] >= 1
    assert isinstance(proc.get("db_pool"), str)       # SQLAlchemy pool status line


def test_process_stats_never_raises_without_procfs(monkeypatch):
    import builtins
    real_open = builtins.open

    def _no_proc(path, *a, **k):
        if str(path).startswith("/proc/"):
            raise FileNotFoundError(path)
        return real_open(path, *a, **k)
    monkeypatch.setattr(builtins, "open", _no_proc)
    out = main._process_stats()
    assert "uptime_s" in out and "rss_mb" not in out


def test_slow_requests_are_logged_with_route_and_duration(monkeypatch, caplog):
    if not _engine_ok():
        pytest.skip("main.engine cannot reach the database")
    from fastapi.testclient import TestClient
    client = TestClient(main.app)
    patch_app(monkeypatch, "_SLOW_REQUEST_MS", 0)          # everything is "slow"
    with caplog.at_level(logging.WARNING, logger="literev-api"):
        r = client.get("/activity")
    assert r.status_code == 200
    slow = [m for m in caplog.messages if m.startswith("slow request:")]
    assert slow, caplog.messages
    assert "GET /activity" in slow[0] and " ms status=200 bytes=" in slow[0]
    # Fast requests stay quiet at the default threshold.
    caplog.clear()
    patch_app(monkeypatch, "_SLOW_REQUEST_MS", 60_000)
    with caplog.at_level(logging.WARNING, logger="literev-api"):
        client.get("/activity")
    assert not [m for m in caplog.messages if m.startswith("slow request:")]
