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


# ── HTTP seams (monkeypatched in tests) ──────────────────────────────────────
def _http_get_json(url: str, timeout: int = 20) -> dict:
    """JSON network call. Isolated so tests can replace it (no network in tests)."""
    import requests
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _http_get_text(url: str, timeout: int = 30) -> str:
    """Text/CSV network call (bulk CSV connectors). Isolated for tests."""
    import requests
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text


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


# ── EAWAG wastewater (bulk CSV, Romandie catchments) ─────────────────────────
_EAWAG_CSV_URL = "https://raw.githubusercontent.com/EawagPHH/RespiratoryVirusesWastewater/main/LatestRESP6Data-filtered.csv"
_EAWAG_WWTP = {   # Romandie alias → treatment-plant name used in the dataset
    "geneva": "STEP Aire", "geneve": "STEP Aire", "genève": "STEP Aire",
    "lausanne": "STEP Vidy", "vaud": "STEP Vidy",
    "neuchatel": "STEP Neuchatel", "neuchâtel": "STEP Neuchatel", "jura": "STEP Porrentruy",
}
_EAWAG_TARGET_COL = {"IAV": "flu_a_load", "IBV": "flu_b_load", "RSV": "rsv_load", "SARS": "sars_cov2_load"}


def _parse_eawag_csv(csv_text: str, wwtp: str, targets: list | None = None) -> list[dict]:
    """EAWAG RespiratoryVirusesWastewater CSV (wwtp,sample_date,Target,Protein,Load)
    → tidy daily rows for ONE plant, pivoted long→wide by virus Target. PURE/testable."""
    import csv as _csv
    import io as _io
    want = {str(t).upper() for t in (targets or _EAWAG_TARGET_COL)}
    by_date: dict[str, dict] = {}
    try:
        reader = _csv.DictReader(_io.StringIO(csv_text))
    except Exception:
        return []
    for r in reader:
        if (r.get("wwtp") or "").strip() != wwtp:
            continue
        tgt = (r.get("Target") or "").strip().upper()
        if tgt not in want:
            continue
        date = (r.get("sample_date") or "").strip()[:10]
        raw = r.get("Load")
        try:
            load = float(raw) if raw not in (None, "") else None
        except (TypeError, ValueError):
            load = None
        if not date or load is None:
            continue
        by_date.setdefault(date, {"date": date})[_EAWAG_TARGET_COL.get(tgt, f"{tgt.lower()}_load")] = load
    return [by_date[d] for d in sorted(by_date)]


def _fetch_eawag(params: dict) -> list[dict]:
    alias = str(params.get("region", "geneva") or "geneva").strip().lower()
    wwtp = params.get("wwtp") or _EAWAG_WWTP.get(alias, "STEP Aire")
    rows = _parse_eawag_csv(_http_get_text(params.get("url") or _EAWAG_CSV_URL),
                            wwtp, params.get("targets"))
    s, e = params.get("start_date"), params.get("end_date")
    if s or e:
        rows = [r for r in rows if (not s or r["date"] >= str(s)) and (not e or r["date"] <= str(e))]
    return rows


# ── FOPH / BAG respiratory open data (opendata.swiss CSV) — BETA ─────────────
# Exact resource/columns not reachable to verify from here (egress-blocked during
# research). Resolves the CSV via CKAN (or FOPH_SENTINELLA_CSV_URL) and parses
# GENERICALLY (a date/week column + numeric columns). Firm up once a live fetch
# confirms the schema; the connector still works, just with detected column names.
_FOPH_CKAN = "https://opendata.swiss/api/3/action/package_show?id=influenza1"
_GENERIC_DATE_KEYS = {"date", "week", "yearweek", "year_week", "time", "datum", "temporal", "woche"}


def _parse_generic_csv(csv_text: str) -> list[dict]:
    """Best-effort CSV → tidy rows: pick a date/week column, keep numeric columns."""
    import csv as _csv
    import io as _io
    try:
        rows = list(_csv.DictReader(_io.StringIO(csv_text)))
    except Exception:
        return []
    cols = [c for c in (rows[0].keys() if rows else []) if c]
    if not cols:
        return []
    dcol = next((c for c in cols if c.strip().lower() in _GENERIC_DATE_KEYS), cols[0])
    out: list[dict] = []
    for r in rows:
        d = (r.get(dcol) or "").strip()
        if not d:
            continue
        row = {"date": d}
        for c, v in r.items():
            if c == dcol or v in (None, ""):
                continue
            try:
                row[c.strip().lower().replace(" ", "_")] = float(v)
            except (TypeError, ValueError):
                pass
        if len(row) > 1:
            out.append(row)
    return out


def _fetch_foph_ckan_csv(ckan_id: str, env_var: str, params: dict) -> list[dict]:
    """Résout le 1er CSV d'un dataset opendata.swiss (CKAN, ou l'override <env_var>),
    le parse génériquement (date/semaine + colonnes numériques) et filtre par dates.
    Partagé par les connecteurs FOPH (schéma non vérifié en direct → parseur générique)."""
    import os as _os
    url = params.get("url") or _os.getenv(env_var)
    if not url:
        try:
            pkg = _http_get_json(f"https://opendata.swiss/api/3/action/package_show?id={ckan_id}")
            for res in (((pkg or {}).get("result") or {}).get("resources") or []):
                if str(res.get("format", "")).lower() == "csv" and (res.get("download_url") or res.get("url")):
                    url = res.get("download_url") or res.get("url")
                    break
        except Exception:
            url = None
    if not url:
        return []
    rows = _parse_generic_csv(_http_get_text(url))
    s, e = params.get("start_date"), params.get("end_date")
    if s or e:
        rows = [r for r in rows if (not s or r["date"] >= str(s)) and (not e or r["date"] <= str(e))]
    return rows


def _fetch_foph_respiratory(params: dict) -> list[dict]:
    return _fetch_foph_ckan_csv("influenza1", "FOPH_SENTINELLA_CSV_URL", params)


def _fetch_foph_wastewater(params: dict) -> list[dict]:
    # Live wastewater feed (influenza + RSV) — complements the frozen EAWAG archive.
    return _fetch_foph_ckan_csv("abwassermonitoring-influenza-und-rsv", "FOPH_WASTEWATER_CSV_URL", params)


_POINT_PARAMS = {
    "region": "Romandie alias (geneva|lausanne|sion|neuchatel|fribourg|jura) OR pass lat+lon",
    "lat": "latitude (optional, overrides region)", "lon": "longitude (optional)",
    "start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD",
}
_CSV_PARAMS = {
    "region": "Romandie alias → WWTP (EAWAG: geneva→STEP Aire, lausanne→STEP Vidy, …)",
    "wwtp": "explicit treatment-plant name (EAWAG, optional)",
    "targets": "virus targets IAV|IBV|RSV|SARS (EAWAG, optional)",
    "url": "override CSV URL (optional)",
    "start_date": "YYYY-MM-DD (optional filter)", "end_date": "YYYY-MM-DD (optional filter)",
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
    "eawag-wastewater": Connector(
        id="eawag-wastewater",
        name="EAWAG — Respiratory-virus wastewater (Romandie catchments)",
        provider="EAWAG / FOPH (national wastewater programme)",
        license="CC BY 4.0",
        geo="catchment",
        commercial_ok=True,
        variables=[
            {"machine_name": "flu_a_load", "label": "Influenza A wastewater load", "unit": "gc/day", "dtype": "float"},
            {"machine_name": "flu_b_load", "label": "Influenza B wastewater load", "unit": "gc/day", "dtype": "float"},
            {"machine_name": "rsv_load", "label": "RSV wastewater load", "unit": "gc/day", "dtype": "float"},
            {"machine_name": "sars_cov2_load", "label": "SARS-CoV-2 wastewater load", "unit": "gc/day", "dtype": "float"},
        ],
        fetch=_fetch_eawag,
        params_schema=_CSV_PARAMS,
        notes="Catchment-level (STEP Aire=Geneva, STEP Vidy=Lausanne, Neuchâtel, Porrentruy). "
              "ARCHIVE frozen since 2024-03 → historical backfill; use the live FOPH feed for current data.",
    ),
    "foph-respiratory": Connector(
        id="foph-respiratory",
        name="FOPH — Respiratory-virus surveillance (Sentinella ILI/ARI) — BETA",
        provider="Federal Office of Public Health (opendata.swiss influenza1)",
        license="opendata.swiss terms of use",
        geo="national",
        commercial_ok=True,
        variables=[],   # detected at fetch time — schema unverified (see notes)
        fetch=_fetch_foph_respiratory,
        params_schema=_CSV_PARAMS,
        notes="BETA — CSV schema not verified live: columns are auto-detected. Set "
              "FOPH_SENTINELLA_CSV_URL or let CKAN resolution pick the resource. NATIONAL granularity "
              "(no cantonal split). Supplies the weekly Sentinella ILI/ARI outcome once the schema is confirmed.",
    ),
    "foph-wastewater": Connector(
        id="foph-wastewater",
        name="FOPH — Wastewater influenza + RSV (live) — BETA",
        provider="Federal Office of Public Health (opendata.swiss Abwassermonitoring)",
        license="opendata.swiss terms of use",
        geo="catchment",
        commercial_ok=True,
        variables=[],   # detected at fetch time — schema unverified (see notes)
        fetch=_fetch_foph_wastewater,
        params_schema=_CSV_PARAMS,
        notes="BETA — LIVE influenza/RSV wastewater feed; complements the frozen EAWAG archive. "
              "CSV schema not verified live: columns auto-detected. Set FOPH_WASTEWATER_CSV_URL "
              "or let CKAN resolve the 'Abwassermonitoring Influenza und RSV' dataset.",
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
