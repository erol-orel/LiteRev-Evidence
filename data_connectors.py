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


# ── FOPH wastewater respiratory-virus monitoring (opendata.swiss CSV) ─────────
# CONFIRMED live via a prod fetch of opendata.swiss package "influenza1": 70k+ rows
# since 2022 with columns date, value, valuemean7d, conc, flow, pop. This is FOPH
# WASTEWATER influenza monitoring (normalized viral load) — NOT clinical Sentinella
# ILI/ARI. The raw feed multiplexes many treatment plants, so rows are aggregated
# per date into ONE national series (mean load/concentration, summed pop/flow) — a
# downstream join therefore can never receive duplicate dates. Pass a column filter
# (e.g. {"geoRegion": "GE"}) to narrow to a single region/plant before aggregation.
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


_FOPH_WW_CKAN_ID = "influenza1"                 # opendata.swiss package (confirmed live)
_FOPH_WW_SUM_COLS = {"pop", "flow"}             # extensive quantities → sum across plants
# every other numeric column (value, valuemean7d, conc, …) is a rate/concentration → mean
_FOPH_WW_RESERVED = {"url", "start_date", "end_date", "region", "wwtp", "targets", "lat", "lon"}


def _agg_foph_by_date(rows: list[dict]) -> list[dict]:
    """Collapse multiplexed FOPH wastewater rows to ONE row per date (so a downstream
    join can't duplicate dates): sum for extensive columns (pop, flow), mean for every
    other signal (value, valuemean7d, conc). PURE/testable."""
    from collections import defaultdict as _dd
    bucket: dict[str, dict[str, list]] = _dd(lambda: _dd(list))
    for r in rows:
        d = r.get("date")
        if not d:
            continue
        for k, v in r.items():
            if k != "date" and isinstance(v, (int, float)):
                bucket[d][k].append(v)
    out = []
    for d in sorted(bucket):
        row: dict = {"date": d}
        for k, vals in bucket[d].items():
            if not vals:
                continue
            row[k] = round(sum(vals), 4) if k in _FOPH_WW_SUM_COLS else round(sum(vals) / len(vals), 4)
        out.append(row)
    return out


def _parse_foph_wastewater(csv_text: str, filters: dict | None = None) -> list[dict]:
    """FOPH opendata.swiss wastewater CSV → tidy daily rows. Applies optional equality
    `filters` on any raw column (e.g. {"geoRegion": "GE"}), keeps the numeric columns
    (value, valuemean7d, conc, flow, pop), and aggregates multiplexed plants to one row
    per date. PURE/testable."""
    import csv as _csv
    import io as _io
    try:
        raw = list(_csv.DictReader(_io.StringIO(csv_text)))
    except Exception:
        return []
    cols = [c for c in (raw[0].keys() if raw else []) if c]
    if not cols:
        return []
    dcol = next((c for c in cols if (c or "").strip().lower() in _GENERIC_DATE_KEYS), cols[0]).strip().lower()
    flt = {str(k).strip().lower(): str(v).strip().lower()
           for k, v in (filters or {}).items() if v not in (None, "")}
    kept: list[dict] = []
    for r in raw:
        norm = {(k or "").strip().lower(): v for k, v in r.items()}
        if any((norm.get(fk) or "").strip().lower() != fv for fk, fv in flt.items()):
            continue
        d = (norm.get(dcol) or "").strip()
        if not d:
            continue
        row: dict = {"date": d[:10] if (len(d) >= 10 and d[4:5] == "-") else d}
        for k, v in norm.items():
            if k == dcol or v in (None, ""):
                continue
            try:
                row[k.replace(" ", "_")] = float(v)
            except (TypeError, ValueError):
                pass
        if len(row) > 1:
            kept.append(row)
    dates = [r["date"] for r in kept]
    if len(dates) != len(set(dates)):          # multiplexed across plants → aggregate
        return _agg_foph_by_date(kept)
    kept.sort(key=lambda x: x["date"])
    return kept


def _fetch_foph_wastewater(params: dict) -> list[dict]:
    """Live FOPH influenza-wastewater series (opendata.swiss 'influenza1'). Resolves the
    CSV via FOPH_WASTEWATER_CSV_URL, an explicit url=, or the CKAN package, filters by
    any extra column param (e.g. geoRegion=GE) + date window, aggregated per date."""
    import os as _os
    url = params.get("url") or _os.getenv("FOPH_WASTEWATER_CSV_URL")
    if not url:
        try:
            pkg = _http_get_json(
                f"https://opendata.swiss/api/3/action/package_show?id={_FOPH_WW_CKAN_ID}")
            for res in (((pkg or {}).get("result") or {}).get("resources") or []):
                if str(res.get("format", "")).lower() == "csv" and (res.get("download_url") or res.get("url")):
                    url = res.get("download_url") or res.get("url")
                    break
        except Exception:
            url = None
    if not url:
        return []
    filters = {k: v for k, v in params.items() if k not in _FOPH_WW_RESERVED}
    rows = _parse_foph_wastewater(_http_get_text(url), filters)
    s, e = params.get("start_date"), params.get("end_date")
    if s or e:
        rows = [r for r in rows if (not s or r["date"] >= str(s)) and (not e or r["date"] <= str(e))]
    return rows


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
_FOPH_PARAMS = {
    "url": "override CSV URL (optional; else resolved from opendata.swiss 'influenza1' or FOPH_WASTEWATER_CSV_URL)",
    "<column>=<value>": "optional equality filter on any raw CSV column (e.g. geoRegion=GE) to select one region/plant before per-date aggregation",
    "start_date": "YYYY-MM-DD (optional filter)", "end_date": "YYYY-MM-DD (optional filter)",
}


# ── FOPH Sentinella — clinical influenza-like illness (ILI), opendata.swiss ────
# The CLINICAL ILI signal (Sentinella sentinel-GP network) — the influenza-ILI
# OUTCOME, distinct from the wastewater viral-load connector above. FOPH publishes
# weekly ILI consultation incidence; the exact opendata.swiss resource id shifts
# each season, so resolution order is: explicit url= → FOPH_SENTINELLA_CSV_URL env
# → CKAN package_search. ISO-week rows are converted to a real date (Monday of the
# ISO week) so this weekly series resamples/join-aligns exactly like the daily
# weather/air-quality connectors. This is the live ILI outcome that was previously
# "not yet wired → manual upload", enabling a weather → influenza-ILI model.
_SENTINELLA_CKAN_QUERY = "sentinella grippe ILI"
_SENTINELLA_VALUE_HINTS = ("inzidenz", "incidence", "inz", "rate", "ili", "ari",
                           "grippe", "influenza", "faelle", "cases", "consult", "value", "wert")
_SENTINELLA_WEEK_KEYS = {"iso_week", "isoweek", "week_iso", "yearweek", "year_week",
                         "week", "woche", "sos_woche", "semaine"}
_SENTINELLA_RESERVED = {"url", "start_date", "end_date", "region", "value_col",
                        "age_group", "lat", "lon"}


def _isoweek_to_date(token) -> str | None:
    """'2023-W05' / '2023W05' / '202305' / '2023-05' (ISO year-week) → 'YYYY-MM-DD'
    (Monday of that ISO week). A plain 'YYYY-MM-DD' passes through unchanged. None if
    unparseable. PURE — lets a weekly ILI series align onto the daily connectors' grid."""
    import re as _re
    from datetime import date as _date
    s = str(token or "").strip()
    if not s:
        return None
    if _re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):              # already an ISO date
        return s
    m = _re.fullmatch(r"(\d{4})[-_ ]?[wW]?(\d{1,2})", s)    # ISO year + week
    if not m:
        return None
    y, w = int(m.group(1)), int(m.group(2))
    if not (1 <= w <= 53):
        return None
    try:
        return _date.fromisocalendar(y, w, 1).isoformat()  # Monday of the ISO week
    except ValueError:
        return None


def _guess_ili_value_column(rows: list[dict]) -> str | None:
    """Pick the ILI incidence column among a tidy row's numeric keys: prefer a name
    matching the ILI hints, else the first numeric column. PURE."""
    if not rows:
        return None
    keys = [k for k in rows[0].keys() if k != "date"]
    for k in keys:
        if any(h in k for h in _SENTINELLA_VALUE_HINTS):
            return k
    return keys[0] if keys else None


def _parse_sentinella_ili(csv_text: str, value_col: str | None = None,
                          filters: dict | None = None) -> list[dict]:
    """FOPH Sentinella CSV → tidy weekly rows {date, ili_incidence}. Detects a
    week/date column, converts ISO-week to the week's Monday date, applies optional
    equality `filters` on any raw column (e.g. {"age_group": "all"}), maps the chosen
    incidence column (auto-detected when value_col is None) to `ili_incidence`, and
    averages any duplicate weeks. PURE/testable."""
    import csv as _csv
    import io as _io
    try:
        raw = list(_csv.DictReader(_io.StringIO(csv_text)))
    except Exception:
        return []
    if not raw:
        return []
    cols = [c for c in raw[0].keys() if c]
    if not cols:
        return []

    def _n(c):
        return (c or "").strip().lower().replace(" ", "_")
    dcol = next((c for c in cols if _n(c) in _SENTINELLA_WEEK_KEYS or _n(c) in _GENERIC_DATE_KEYS), cols[0])
    dcol_n = _n(dcol)
    flt = {_n(k): str(v).strip().lower() for k, v in (filters or {}).items() if v not in (None, "")}
    tidy: list[dict] = []
    for r in raw:
        norm = {_n(k): v for k, v in r.items()}
        if any((norm.get(fk) or "").strip().lower() != fv for fk, fv in flt.items()):
            continue
        d = _isoweek_to_date(r.get(dcol))
        if not d:
            continue
        row = {"date": d}
        for k, v in norm.items():
            if k == dcol_n or v in (None, ""):
                continue
            try:
                row[k] = float(v)
            except (TypeError, ValueError):
                pass
        if len(row) > 1:
            tidy.append(row)
    val = _n(value_col) if value_col else _guess_ili_value_column(tidy)
    if not val:
        return []
    by_date: dict[str, list[float]] = {}
    for r in tidy:
        if val in r:
            by_date.setdefault(r["date"], []).append(r[val])
    merged = [{"date": d, "ili_incidence": round(sum(v) / len(v), 4)} for d, v in by_date.items()]
    merged.sort(key=lambda x: x["date"])
    return merged


def _fetch_sentinella_ili(params: dict) -> list[dict]:
    """Live FOPH Sentinella clinical-ILI series. Resolves the CSV via url=, the
    FOPH_SENTINELLA_CSV_URL env var, or an opendata.swiss CKAN search; parses to
    weekly {date, ili_incidence}; applies extra column filters + a date window."""
    import os as _os
    url = params.get("url") or _os.getenv("FOPH_SENTINELLA_CSV_URL")
    if not url:
        try:
            res = _http_get_json(
                "https://opendata.swiss/api/3/action/package_search?rows=5&q="
                + _SENTINELLA_CKAN_QUERY.replace(" ", "%20"))
            for pkg in (((res or {}).get("result") or {}).get("results") or []):
                for r in (pkg.get("resources") or []):
                    if str(r.get("format", "")).lower() == "csv" and (r.get("download_url") or r.get("url")):
                        url = r.get("download_url") or r.get("url")
                        break
                if url:
                    break
        except Exception:
            url = None
    if not url:
        return []
    filters = {k: v for k, v in params.items() if k not in _SENTINELLA_RESERVED}
    rows = _parse_sentinella_ili(_http_get_text(url), params.get("value_col"), filters)
    s, e = params.get("start_date"), params.get("end_date")
    if s or e:
        rows = [r for r in rows if (not s or r["date"] >= str(s)) and (not e or r["date"] <= str(e))]
    return rows


_SENTINELLA_PARAMS = {
    "url": "override CSV URL (optional; else FOPH_SENTINELLA_CSV_URL env, or opendata.swiss search)",
    "value_col": "name of the ILI incidence column to use (optional; auto-detected)",
    "<column>=<value>": "optional equality filter on any raw column (e.g. age_group=all, region=CH)",
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
    "foph-wastewater": Connector(
        id="foph-wastewater",
        name="FOPH — Influenza wastewater (normalized viral load, live)",
        provider="Federal Office of Public Health (opendata.swiss 'influenza1')",
        license="opendata.swiss terms of use",
        geo="national",
        commercial_ok=True,
        variables=[
            {"machine_name": "value", "label": "Influenza viral load (normalized)", "unit": "gc/day/person", "dtype": "float"},
            {"machine_name": "valuemean7d", "label": "Influenza viral load, 7-day mean", "unit": "gc/day/person", "dtype": "float"},
            {"machine_name": "conc", "label": "Influenza concentration in wastewater", "unit": "gc/L", "dtype": "float"},
            {"machine_name": "flow", "label": "Wastewater flow (summed over plants)", "unit": "L/day", "dtype": "float"},
            {"machine_name": "pop", "label": "Catchment population covered (summed)", "unit": "people", "dtype": "float"},
        ],
        fetch=_fetch_foph_wastewater,
        params_schema=_FOPH_PARAMS,
        notes="LIVE influenza WASTEWATER monitoring (opendata.swiss 'influenza1'), weekly since 2022 "
              "(70k+ rows confirmed) — NOT clinical Sentinella ILI/ARI. The raw feed multiplexes "
              "treatment plants; rows are aggregated per date into a national series (mean load/conc, "
              "summed pop/flow). Pass a column filter (e.g. geoRegion=GE) to narrow to one region, or "
              "set FOPH_WASTEWATER_CSV_URL. Clinical ILI/ARI is served by 'foph-sentinella-ili'.",
    ),
    "foph-sentinella-ili": Connector(
        id="foph-sentinella-ili",
        name="FOPH Sentinella — Influenza-like illness (clinical ILI, weekly)",
        provider="Federal Office of Public Health — Sentinella sentinel-GP network (opendata.swiss)",
        license="opendata.swiss terms of use",
        geo="national",
        commercial_ok=True,
        variables=[
            {"machine_name": "ili_incidence", "label": "Influenza-like illness consultation incidence",
             "unit": "per 100k / per 1000 consultations", "dtype": "float"},
        ],
        fetch=_fetch_sentinella_ili,
        params_schema=_SENTINELLA_PARAMS,
        notes="CLINICAL influenza-like-illness OUTCOME (Sentinella sentinel-GP network), "
              "complementary to the wastewater viral-load signal. Weekly ISO-week series "
              "converted to the week's Monday date so it join-aligns with the daily weather / "
              "air-quality connectors after weekly resampling. The opendata.swiss resource id "
              "shifts by season → set FOPH_SENTINELLA_CSV_URL or pass url=/value_col= to pin the "
              "source. This is the live ILI target for a weather → influenza-ILI predictive model.",
    ),
}


# ── SEIR projection (modèle compartimental paramétré par la littérature) ─────
# PAS une source réseau : la « fetch » intègre localement le modèle SEIR-family
# (seir_model) à partir des paramètres épidémiologiques EXTRAITS du corpus, transmis
# dans params["epidemic_parameters"] (= bloc model_spec.epidemic_parameters). Renvoie
# la trajectoire MÉDIANE en lignes journalières, prête à rejoindre le dataset du
# scénario comme n'importe quel connecteur. Les bandes d'incertitude (IC) sont
# exposées séparément par l'endpoint de projection (une colonne de dataset = une série).
SEIR_CONNECTOR_ID = "seir-projection"
_SEIR_VARS = [
    {"machine_name": "seir_incidence",   "label": "Incidence modélisée (nouvelles infections/j)", "unit": "cas/jour", "dtype": "float"},
    {"machine_name": "seir_prevalence",  "label": "Prévalence modélisée (infectieux)",             "unit": "cas",      "dtype": "float"},
    {"machine_name": "seir_cumulative",  "label": "Infections cumulées",                           "unit": "cas",      "dtype": "float"},
    {"machine_name": "seir_deaths",      "label": "Décès cumulés (si létalité connue)",            "unit": "décès",    "dtype": "float"},
    {"machine_name": "seir_susceptible", "label": "Susceptibles (compartiment S)",                 "unit": "personnes", "dtype": "float"},
    {"machine_name": "seir_exposed",     "label": "Exposés / latents (compartiment E)",            "unit": "personnes", "dtype": "float"},
    {"machine_name": "seir_recovered",   "label": "Rétablis / immuns (compartiment R)",            "unit": "personnes", "dtype": "float"},
]


def _seir_days_from_params(params: dict, default: int = 365) -> int:
    """Horizon (jours) : fenêtre start_date→end_date si fournie, sinon `days`, sinon défaut.
    Borné à [1, 3650] pour rester raisonnable."""
    start, end = params.get("start_date"), params.get("end_date")
    if start and end:
        from datetime import date
        try:
            y1, m1, d1 = (int(x) for x in str(start)[:10].split("-"))
            y2, m2, d2 = (int(x) for x in str(end)[:10].split("-"))
            n = (date(y2, m2, d2) - date(y1, m1, d1)).days
            if n > 0:
                return min(n, 3650)
        except Exception:
            pass
    try:
        return max(1, min(int(params.get("days") or default), 3650))
    except (TypeError, ValueError):
        return default


def _fetch_seir_projection(params: dict) -> list[dict]:
    """Trajectoire MÉDIANE du modèle SEIR-family, en lignes journalières tidy. Aucun
    appel réseau : intégration locale. `params["epidemic_parameters"]` = bloc
    model_spec.epidemic_parameters (issu du corpus) ; `population` / `initial_infected`
    en contexte (défauts 1e6 / 10) ; horizon via start/end ou `days`. Liste vide si
    aucun paramètre exploitable (scénario non épidémique)."""
    import seir_model
    from datetime import date, timedelta
    epi = params.get("epidemic_parameters") or {}
    dists = seir_model.params_to_distributions((epi.get("params") or {}))
    if not dists:
        return []  # rien d'extrait / scénario non transmissible → pas de projection
    try:
        dists["population"] = float(params.get("population") or 1_000_000)
    except (TypeError, ValueError):
        dists["population"] = 1_000_000.0
    try:
        dists["initial_infected"] = float(params.get("initial_infected") or 10)
    except (TypeError, ValueError):
        dists["initial_infected"] = 10.0
    days = _seir_days_from_params(params)
    try:
        n_samples = max(1, min(int(params.get("n_samples") or 200), 1000))
    except (TypeError, ValueError):
        n_samples = 200
    ens = seir_model.simulate_ensemble(dists, days=days, n_samples=n_samples)

    d0 = None
    start = str(params.get("start_date") or "")[:10]
    try:
        y, m, d = (int(x) for x in start.split("-"))
        d0 = date(y, m, d)
    except Exception:
        d0 = None

    inc, prev = ens["incidence"]["median"], ens["prevalence"]["median"]
    cum, dth = ens["cumulative"]["median"], ens["deaths"]["median"]
    sus, exp, rec = ens["susceptible"]["median"], ens["exposed"]["median"], ens["recovered"]["median"]
    rows: list[dict] = []
    for i, day in enumerate(ens["days"]):
        dt = (d0 + timedelta(days=int(day))).isoformat() if d0 else str(day)
        rows.append({
            "date": dt,
            "seir_incidence": round(inc[i], 3),
            "seir_prevalence": round(prev[i], 3),
            "seir_cumulative": round(cum[i], 3),
            "seir_deaths": round(dth[i], 3),
            "seir_susceptible": round(sus[i], 3),
            "seir_exposed": round(exp[i], 3),
            "seir_recovered": round(rec[i], 3),
        })
    return rows


CONNECTORS[SEIR_CONNECTOR_ID] = Connector(
    id=SEIR_CONNECTOR_ID,
    name="Projection SEIR (paramétrée par la littérature)",
    provider="LiteRev — modèle compartimental",
    license="interne (modèle) ; paramètres tracés vers les articles source",
    geo="model",
    commercial_ok=True,
    variables=_SEIR_VARS,
    fetch=_fetch_seir_projection,
    params_schema={
        "epidemic_parameters": "bloc model_spec.epidemic_parameters (injecté par le scénario)",
        "population": "entier (défaut 1 000 000)",
        "initial_infected": "entier (défaut 10)",
        "start_date": "YYYY-MM-DD (origine des dates)",
        "end_date": "YYYY-MM-DD (ou 'days')",
        "n_samples": "taille de l'ensemble (défaut 200)",
    },
    notes="Intègre localement un modèle SEIR/SEIRD/SEIRS (auto-sélectionné selon les "
          "paramètres disponibles) paramétré par R0/période infectieuse/incubation/CFR/"
          "immunité extraits du corpus. Colonne dataset = trajectoire médiane ; les "
          "bandes d'incertitude sont fournies par l'endpoint de projection.",
)


def list_connectors() -> list[dict]:
    """JSON-safe metadata for every registered connector."""
    return [c.metadata() for c in CONNECTORS.values()]


def fetch_series(connector_id: str, params: dict) -> list[dict]:
    """Fetch a connector's tidy daily series. Raises KeyError for an unknown id."""
    c = CONNECTORS.get(connector_id)
    if c is None:
        raise KeyError(connector_id)
    return c.fetch(params or {})
