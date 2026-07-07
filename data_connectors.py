"""data_connectors.py — Phase-2 public-data connectors.

Each connector fetches a REAL, machine-readable public data source and returns a
TIDY daily time series — a list of {"date": "YYYY-MM-DD", <machine_name>: value, …}
rows — that can be joined into a scenario's modeling dataset on the date key,
instead of asking the user to upload a CSV.

Design goals (mirrors model_trainer.py):
  - PURE + testable: the only side effect is one HTTP GET, isolated behind the
    `_http_get_json` seam so unit tests monkeypatch it (no network in tests).
  - Honest metadata: every connector declares its real provider, licence, whether
    the free tier is commercial-OK, its geographic granularity, and the exact
    variables it supplies (machine_name / label / unit / dtype). No mislabelling
    (learning from the "MeteoSwiss"/"INSEE" mislabels in the legacy terrain tab).
  - Registry: `CONNECTORS[id] -> Connector`. The app maps a scenario's public
    columns to a connector + params, fetches, and assembles the dataset.

Connectors shipped here (verified pluggable, 2026-07):
  - open-meteo-weather      → historical daily temperature/precip/wind/humidity
  - open-meteo-air-quality  → daily PM2.5 / PM10 / NO2 / O3 (Copernicus CAMS)

Both are point sources (query by lat/lon → any Romandie city). The FOPH respiratory
open-data + EAWAG wastewater connectors are the next additions (CSV sources).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable


# ── HTTP seam (monkeypatched in tests) ───────────────────────────────────────
def _http_get_json(url: str, timeout: int = 20) -> dict:
    """The ONLY network call in this module. Isolated so tests can replace it."""
    import requests
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


# ── Region resolution ────────────────────────────────────────────────────────
# Romandie / Léman anchor points. A connector accepts either explicit lat/lon or a
# named region alias; unknown/blank aliases fall back to Geneva.
_REGION_COORDS: dict[str, tuple[float, float]] = {
    "geneva": (46.2044, 6.1432), "geneve": (46.2044, 6.1432), "genève": (46.2044, 6.1432),
    "lausanne": (46.5160, 6.6291), "vaud": (46.5160, 6.6291),
    "sion": (46.2290, 7.3620), "valais": (46.2290, 7.3620),
    "neuchatel": (46.9930, 6.9310), "neuchâtel": (46.9930, 6.9310),
    "fribourg": (46.8020, 7.1510), "jura": (47.3660, 7.3440),
}


def resolve_region(params: dict) -> tuple[float, float]:
    """(lat, lon) from explicit coords or a Romandie region alias (default Geneva)."""
    lat, lon = params.get("lat"), params.get("lon")
    if lat is not None and lon is not None:
        return float(lat), float(lon)
    alias = str(params.get("region", "geneva") or "geneva").strip().lower()
    return _REGION_COORDS.get(alias, _REGION_COORDS["geneva"])


def _require_dates(params: dict) -> tuple[str, str]:
    start, end = params.get("start_date"), params.get("end_date")
    if not start or not end:
        raise ValueError("start_date and end_date (YYYY-MM-DD) are required")
    return str(start), str(end)


def _aggregate_hourly_to_daily(times: list[str], values: list, how: str = "mean") -> dict[str, float]:
    """Collapse an hourly series (times like '2024-01-01T13:00') to per-day values.
    Skips None entries. `how` ∈ {mean, sum, max, min}."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for t, v in zip(times or [], values or []):
        if v is None or not t:
            continue
        buckets[t[:10]].append(v)
    out: dict[str, float] = {}
    for day, vs in buckets.items():
        if not vs:
            continue
        if how == "sum":
            out[day] = sum(vs)
        elif how == "max":
            out[day] = max(vs)
        elif how == "min":
            out[day] = min(vs)
        else:
            out[day] = sum(vs) / len(vs)
    return out


# ── Open-Meteo: historical weather ───────────────────────────────────────────
_OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
_WEATHER_DAILY = ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
                  "precipitation_sum", "wind_speed_10m_max"]
_WEATHER_ALIAS = {
    "temperature_2m_mean": "temp_mean", "temperature_2m_max": "temp_max",
    "temperature_2m_min": "temp_min", "precipitation_sum": "precip_sum",
    "wind_speed_10m_max": "wind_max",
}


def _fetch_open_meteo_weather(params: dict) -> list[dict]:
    """Daily historical weather for a point + date range. Humidity is aggregated
    from the hourly relative-humidity series (no reliable daily mean in the API)."""
    lat, lon = resolve_region(params)
    start, end = _require_dates(params)
    url = (f"{_OPEN_METEO_ARCHIVE}?latitude={lat}&longitude={lon}"
           f"&start_date={start}&end_date={end}"
           f"&daily={','.join(_WEATHER_DAILY)}&hourly=relative_humidity_2m"
           f"&timezone=Europe%2FZurich")
    data = _http_get_json(url)
    daily = data.get("daily", {}) or {}
    times = daily.get("time", []) or []
    rows: dict[str, dict] = {t: {"date": t} for t in times}
    for var in _WEATHER_DAILY:
        series = daily.get(var) or []
        for t, val in zip(times, series):
            if val is not None:
                rows[t][_WEATHER_ALIAS[var]] = val
    hourly = data.get("hourly", {}) or {}
    hum = _aggregate_hourly_to_daily(hourly.get("time", []), hourly.get("relative_humidity_2m", []), "mean")
    for t, val in hum.items():
        if t in rows:
            rows[t]["relative_humidity_mean"] = round(val, 1)
    return [rows[t] for t in times]


# ── Open-Meteo: air quality (Copernicus CAMS) ────────────────────────────────
_OPEN_METEO_AQ = "https://air-quality-api.open-meteo.com/v1/air-quality"
_AQ_HOURLY = ["pm2_5", "pm10", "nitrogen_dioxide", "ozone"]
_AQ_ALIAS = {"pm2_5": "pm2_5", "pm10": "pm10", "nitrogen_dioxide": "no2", "ozone": "o3"}


def _fetch_open_meteo_air_quality(params: dict) -> list[dict]:
    """Daily-mean air quality (PM2.5/PM10/NO2/O3) from hourly CAMS data for a point."""
    lat, lon = resolve_region(params)
    start, end = _require_dates(params)
    url = (f"{_OPEN_METEO_AQ}?latitude={lat}&longitude={lon}"
           f"&hourly={','.join(_AQ_HOURLY)}&start_date={start}&end_date={end}"
           f"&timezone=Europe%2FZurich")
    data = _http_get_json(url)
    hourly = data.get("hourly", {}) or {}
    times = hourly.get("time", []) or []
    by_date: dict[str, dict] = {}
    for var in _AQ_HOURLY:
        for day, val in _aggregate_hourly_to_daily(times, hourly.get(var, []), "mean").items():
            by_date.setdefault(day, {"date": day})[_AQ_ALIAS[var]] = round(val, 2)
    return [by_date[d] for d in sorted(by_date)]


# ── Connector registry ───────────────────────────────────────────────────────
@dataclass
class Connector:
    id: str
    name: str
    provider: str
    license: str
    geo: str                        # "point" | "national" | "catchment"
    variables: list[dict]           # [{machine_name, label, unit, dtype}]
    fetch: Callable[[dict], list[dict]] = field(repr=False)
    commercial_ok: bool = True
    params_schema: dict = field(default_factory=dict)
    notes: str = ""

    def metadata(self) -> dict:
        """JSON-safe description (everything except the fetch callable)."""
        return {
            "id": self.id, "name": self.name, "provider": self.provider,
            "license": self.license, "geo": self.geo, "variables": self.variables,
            "commercial_ok": self.commercial_ok, "params_schema": self.params_schema,
            "notes": self.notes,
        }


_POINT_PARAMS = {
    "region": "Romandie alias (geneva|lausanne|sion|neuchatel|fribourg|jura) OR pass lat+lon",
    "lat": "latitude (optional, overrides region)", "lon": "longitude (optional)",
    "start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD",
}

CONNECTORS: dict[str, Connector] = {
    "open-meteo-weather": Connector(
        id="open-meteo-weather",
        name="Open-Meteo — Weather (historical daily)",
        provider="Open-Meteo (ERA5 / Copernicus)",
        license="CC BY 4.0 — free tier NON-commercial",
        geo="point",
        commercial_ok=False,
        variables=[
            {"machine_name": "temp_mean", "label": "Mean temperature", "unit": "°C", "dtype": "float"},
            {"machine_name": "temp_max", "label": "Max temperature", "unit": "°C", "dtype": "float"},
            {"machine_name": "temp_min", "label": "Min temperature", "unit": "°C", "dtype": "float"},
            {"machine_name": "relative_humidity_mean", "label": "Mean relative humidity", "unit": "%", "dtype": "float"},
            {"machine_name": "precip_sum", "label": "Precipitation", "unit": "mm", "dtype": "float"},
            {"machine_name": "wind_max", "label": "Max wind speed", "unit": "km/h", "dtype": "float"},
        ],
        fetch=_fetch_open_meteo_weather,
        params_schema=_POINT_PARAMS,
        notes="Already used live by /terrain/meteo. Point source: any Romandie coords. "
              "Free tier is non-commercial — a production deployment needs a paid/self-hosted plan.",
    ),
    "open-meteo-air-quality": Connector(
        id="open-meteo-air-quality",
        name="Open-Meteo — Air Quality (Copernicus CAMS)",
        provider="Open-Meteo (Copernicus CAMS)",
        license="CC BY 4.0 — free tier NON-commercial",
        geo="point",
        commercial_ok=False,
        variables=[
            {"machine_name": "pm2_5", "label": "PM2.5", "unit": "µg/m³", "dtype": "float"},
            {"machine_name": "pm10", "label": "PM10", "unit": "µg/m³", "dtype": "float"},
            {"machine_name": "no2", "label": "Nitrogen dioxide", "unit": "µg/m³", "dtype": "float"},
            {"machine_name": "o3", "label": "Ozone", "unit": "µg/m³", "dtype": "float"},
        ],
        fetch=_fetch_open_meteo_air_quality,
        params_schema=_POINT_PARAMS,
        notes="~11 km CAMS grid, queried by lat/lon. Free tier is non-commercial.",
    ),
}


def list_connectors() -> list[dict]:
    """JSON-safe metadata for every registered connector."""
    return [c.metadata() for c in CONNECTORS.values()]


def fetch_series(connector_id: str, params: dict) -> list[dict]:
    """Fetch a connector's tidy daily series. Raises KeyError for an unknown id."""
    c = CONNECTORS.get(connector_id)
    if c is None:
        raise KeyError(connector_id)
    return c.fetch(params or {})
