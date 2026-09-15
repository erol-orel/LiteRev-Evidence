"""The per-IP rate limits can be raised from the environment for an event where a whole
room shares one public IP, and /health says which values are in force."""
import main


def test_env_int_reads_valid_values_and_falls_back(monkeypatch):
    monkeypatch.setenv("LITEREV_TEST_INT", "12")
    assert main._env_int("LITEREV_TEST_INT", 5) == 12
    monkeypatch.setenv("LITEREV_TEST_INT", "abc")
    assert main._env_int("LITEREV_TEST_INT", 5) == 5
    monkeypatch.setenv("LITEREV_TEST_INT", "-3")
    assert main._env_int("LITEREV_TEST_INT", 5, minimum=1) == 1
    monkeypatch.delenv("LITEREV_TEST_INT", raising=False)
    assert main._env_int("LITEREV_TEST_INT", 7) == 7


def test_default_limits_are_the_historical_ones():
    assert main.RATE_LIMIT_GENERAL_PER_MIN == 600
    assert main.RATE_LIMIT_EXPENSIVE_PER_MIN == 30
    assert main.general_limiter.requests_limit == 600
    assert main.expensive_limiter.requests_limit == 30
    assert main.general_limiter.window_seconds == 60


def test_health_reports_the_limits_in_force():
    import pytest
    from fastapi.testclient import TestClient

    try:
        with main.engine.connect():
            pass
    except Exception:
        pytest.skip("main.engine cannot reach the database")
    body = TestClient(main.app).get("/health").json()
    assert body["rate_limit"] == {"general_per_min": 600, "expensive_per_min": 30}
