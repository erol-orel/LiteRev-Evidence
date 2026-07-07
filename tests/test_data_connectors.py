"""Pure-logic tests for data_connectors.py — no network.

The only side effect in the module is `_http_get_json`; every test monkeypatches it
with a canned Open-Meteo response and asserts the tidy-row assembly + aggregation.
"""
import json

import pytest

import data_connectors as dc


# ── region resolution ────────────────────────────────────────────────────────
def test_resolve_region_alias_and_latlon():
    assert dc.resolve_region({"region": "geneva"}) == (46.2044, 6.1432)
    assert dc.resolve_region({"region": "LAUSANNE"}) == (46.5160, 6.6291)
    # explicit lat/lon overrides the alias
    assert dc.resolve_region({"region": "geneva", "lat": 47.0, "lon": 7.0}) == (47.0, 7.0)
    # unknown alias → Geneva fallback
    assert dc.resolve_region({"region": "narnia"}) == (46.2044, 6.1432)
    assert dc.resolve_region({}) == (46.2044, 6.1432)


def test_require_dates():
    with pytest.raises(ValueError):
        dc._require_dates({"start_date": "2024-01-01"})   # missing end
    assert dc._require_dates({"start_date": "2024-01-01", "end_date": "2024-01-31"}) == \
           ("2024-01-01", "2024-01-31")


# ── hourly → daily aggregation ───────────────────────────────────────────────
def test_aggregate_hourly_to_daily_mean_sum_max_skip_none():
    times = ["2024-01-01T00:00", "2024-01-01T12:00", "2024-01-02T00:00", "2024-01-02T06:00"]
    vals = [10.0, 20.0, 4.0, None]                       # None is skipped
    assert dc._aggregate_hourly_to_daily(times, vals, "mean") == {"2024-01-01": 15.0, "2024-01-02": 4.0}
    assert dc._aggregate_hourly_to_daily(times, vals, "sum") == {"2024-01-01": 30.0, "2024-01-02": 4.0}
    assert dc._aggregate_hourly_to_daily(times, vals, "max") == {"2024-01-01": 20.0, "2024-01-02": 4.0}
    assert dc._aggregate_hourly_to_daily([], []) == {}


# ── weather connector ────────────────────────────────────────────────────────
def test_open_meteo_weather_tidy_rows(monkeypatch):
    canned = {
        "daily": {
            "time": ["2024-01-01", "2024-01-02"],
            "temperature_2m_mean": [3.1, 4.2],
            "temperature_2m_max": [6.0, 7.0],
            "temperature_2m_min": [0.0, 1.0],
            "precipitation_sum": [2.5, 0.0],
            "wind_speed_10m_max": [18.0, 22.0],
        },
        "hourly": {
            "time": ["2024-01-01T00:00", "2024-01-01T12:00", "2024-01-02T00:00"],
            "relative_humidity_2m": [80, 60, 90],
        },
    }
    monkeypatch.setattr(dc, "_http_get_json", lambda url, timeout=20: canned)
    rows = dc.fetch_series("open-meteo-weather", {"region": "geneva",
                                                  "start_date": "2024-01-01", "end_date": "2024-01-02"})
    assert [r["date"] for r in rows] == ["2024-01-01", "2024-01-02"]
    assert rows[0]["temp_mean"] == 3.1 and rows[0]["precip_sum"] == 2.5 and rows[0]["wind_max"] == 18.0
    assert rows[0]["relative_humidity_mean"] == 70.0        # (80+60)/2
    assert rows[1]["relative_humidity_mean"] == 90.0
    # JSON-serialisable (goes over the wire)
    json.dumps(rows)


def test_open_meteo_weather_skips_missing_values(monkeypatch):
    canned = {"daily": {"time": ["2024-03-01"], "temperature_2m_mean": [None],
                        "precipitation_sum": [1.0]}, "hourly": {"time": [], "relative_humidity_2m": []}}
    monkeypatch.setattr(dc, "_http_get_json", lambda url, timeout=20: canned)
    rows = dc.fetch_series("open-meteo-weather", {"start_date": "2024-03-01", "end_date": "2024-03-01"})
    assert rows[0]["date"] == "2024-03-01"
    assert "temp_mean" not in rows[0]                       # None dropped, not written as null
    assert rows[0]["precip_sum"] == 1.0


# ── air-quality connector ────────────────────────────────────────────────────
def test_open_meteo_air_quality_daily_means(monkeypatch):
    canned = {"hourly": {
        "time": ["2024-01-01T00:00", "2024-01-01T12:00", "2024-01-02T00:00"],
        "pm2_5": [10.0, 20.0, 5.0], "pm10": [15.0, 25.0, 8.0],
        "nitrogen_dioxide": [30.0, 30.0, 12.0], "ozone": [40.0, 60.0, 50.0],
    }}
    monkeypatch.setattr(dc, "_http_get_json", lambda url, timeout=20: canned)
    rows = dc.fetch_series("open-meteo-air-quality", {"region": "lausanne",
                                                      "start_date": "2024-01-01", "end_date": "2024-01-02"})
    assert rows[0]["date"] == "2024-01-01"
    assert rows[0]["pm2_5"] == 15.0 and rows[0]["no2"] == 30.0 and rows[0]["o3"] == 50.0
    assert rows[1]["pm2_5"] == 5.0


# ── registry ─────────────────────────────────────────────────────────────────
def test_list_connectors_is_json_safe_and_hides_callable():
    meta = dc.list_connectors()
    ids = {m["id"] for m in meta}
    assert {"open-meteo-weather", "open-meteo-air-quality"} <= ids
    for m in meta:
        assert "fetch" not in m                            # callable not leaked
        assert m["variables"] and all("machine_name" in v for v in m["variables"])
        assert m["commercial_ok"] is False                 # Open-Meteo free tier caveat surfaced
    json.dumps(meta)                                        # fully serialisable


def test_fetch_series_unknown_connector_raises():
    with pytest.raises(KeyError):
        dc.fetch_series("does-not-exist", {"start_date": "2024-01-01", "end_date": "2024-01-02"})
