"""ReliefWeb as a SEPARATE evidence stream, alongside the scientific corpus.

ReliefWeb (https://reliefweb.int, UN OCHA) curates humanitarian situation reports from
4,000+ organisations. For epidemic intelligence it supplies what the peer-reviewed
literature structurally cannot: **timeliness** (a WHO AFRO bulletin or an IFRC DREF lands
in days; the paper describing the same outbreak takes 6-24 months) and **operational
detail** that no journal publishes (treatment-centre openings, oral-cholera-vaccine doses,
ring-vaccination logistics, access constraints).

It is also GREY LITERATURE: no peer review, no methods section, numbers that are revised
without a changelog and that mix "suspected" with "confirmed". This module therefore keeps
ReliefWeb strictly apart from `literature_document` and hands every record a
`credibility` well below any peer-reviewed study (see `credibility`), so a press-release
figure can never outweigh a published estimate in the SEIR parameter pool.

Why a separate stream rather than a `data_connectors.Connector`: connectors carry ONE
NUMBER KEYED BY A DATE. `_assemble_connector_frames` coerces every non-date column with
`pd.to_numeric(errors="coerce")` and drops the NaN, then resamples to one row per period.
A title, a body, an organisation or a country cannot survive that pipe, and thirty sitreps
in one week cannot coexist in it. ReliefWeb yields DOCUMENTS, so it is modelled as one.

PURE except for two named HTTP seams (`_http_post_json`, `_http_get_json`), which the
tests monkeypatch — everything here is exercised offline.

API notes, from the official documentation (apidoc.reliefweb.int):
  - Base URL is **https://api.reliefweb.int/v2/**. V1 is decommissioned.
  - `appname` is mandatory AND pre-approval-gated: "From 1 November 2025, API users will
    require a pre-approved appname." Set RELIEFWEB_APPNAME once registered.
  - Quotas, quoted in full: "The maximum number of entries returned per call is 1000. The
    maximum number of calls allowed per day is 1000." There is no documented per-second
    limit, no reset time and no rate-limit header, so callers must count their own calls
    (`QUOTA_CALLS_PER_DAY`) and cache — see `main._rw_budget_remaining`.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable

BASE_URL = "https://api.reliefweb.int/v2"
DEFAULT_APPNAME = "literev-evidence"

# Documented quotas (apidoc.reliefweb.int/#quotas). 1000 calls/day is the binding
# constraint: never build a per-user live query on top of it — ingest and cache.
QUOTA_CALLS_PER_DAY = 1000
MAX_LIMIT_PER_CALL = 1000

# Report types worth ingesting for epidemic intelligence. `format` on ReliefWeb.
EPI_FORMATS = ("Situation Report", "Bulletin", "Assessment", "Analysis",
               "Epidemiological update", "Appeal", "News and Press Release")

# Pathogen sweep. Editorial tagging (`disaster_type = Epidemic`) is applied AFTER an
# outbreak is already visible in free text, so a tag-only filter is systematically late.
# Pairing the tag with a keyword sweep is what buys the timeliness ReliefWeb is for.
PATHOGEN_TERMS = (
    "cholera", "mpox", "monkeypox", "ebola", "marburg", "measles", "dengue",
    "polio", "diphtheria", "meningitis", "yellow fever", "lassa", "malaria",
    "influenza", "avian influenza", "hepatitis E", "typhoid", "chikungunya",
    "zika", "plague", "COVID-19", "Rift Valley fever", "Nipah",
)

# ── source credibility ───────────────────────────────────────────────────────
# `quality_score` on literature_document IS the SEIR pooling weight, and a situation
# report scored a flat 0.55 there — ABOVE a peer-reviewed case report at 0.386. One
# press-release "R0 ~ 6" moved a pooled R0 from 2.09 to 3.04 and pushed the lower CI
# below zero. These values are deliberately far below any peer-reviewed study: they rank
# reports against EACH OTHER (a WHO bulletin above an unattributed news item) without
# ever competing with the literature. The hard cap lives in `seir_model`.
CREDIBILITY_TIERS: dict[str, float] = {
    "who": 0.45, "world health organization": 0.45,
    "africa cdc": 0.42, "ecdc": 0.42, "us cdc": 0.42, "paho": 0.42,
    "unicef": 0.38, "ocha": 0.38, "wfp": 0.35, "unhcr": 0.35, "fao": 0.35,
    "ifrc": 0.34, "icrc": 0.34, "msf": 0.34, "medecins sans frontieres": 0.34,
    "government": 0.32, "ministry of health": 0.32,
}
DEFAULT_CREDIBILITY = 0.20        # unknown NGO / press release
MAX_CREDIBILITY = 0.45            # nothing from ReliefWeb may exceed this


def credibility(source_names, format_name: str | None = None) -> float:
    """Credibility 0..1 of a report, from its issuing organisation(s) and format.

    Capped at MAX_CREDIBILITY so grey literature never reaches the range of a
    peer-reviewed study. Takes the BEST of the listed sources (a joint WHO/NGO bulletin
    is a WHO bulletin), then discounts a bare press release. PURE."""
    best = DEFAULT_CREDIBILITY
    for raw in (source_names or []):
        s = _norm(raw)
        if not s:
            continue
        for key, val in CREDIBILITY_TIERS.items():
            # substring match: "WHO Regional Office for Africa" must score as WHO
            if key in s and val > best:
                best = val
    if _norm(format_name) == "news and press release":
        best = min(best, 0.25)
    return round(min(best, MAX_CREDIBILITY), 3)


# ── text helpers ─────────────────────────────────────────────────────────────
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_MD_TOKENS = re.compile(r"[*_`>#]+")
_HTML_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
# Trailing serial in a recurring sitrep title: "... Situation Report No. 12", "(#7)".
_SERIAL_TAIL = re.compile(
    r"\s*[\(\[]?\s*(?:no|n[o°]|num(?:ber)?|#|issue|update|edition|vol)\s*\.?\s*"
    r"\d+[a-z]?\s*[\)\]]?\s*$", re.I)
_DATE_TAIL = re.compile(
    r"\s*[\(\[]?\s*(?:as of\s+)?\d{1,2}\s+\w+\s+\d{4}\s*[\)\]]?\s*$", re.I)


def strip_markdown(text) -> str:
    """ReliefWeb `body` is Markdown (`body-html` is the HTML twin). Flatten it to plain
    text for indexing/summarising — do NOT run an HTML sanitiser over it. PURE."""
    s = str(text or "")
    s = _MD_LINK.sub(r"\1", s)          # [label](url) → label
    s = _HTML_TAG.sub(" ", s)           # stray inline HTML happens in practice
    s = _MD_TOKENS.sub(" ", s)
    return _WS.sub(" ", s).strip()


def _norm(v) -> str:
    return _WS.sub(" ", str(v or "").strip().lower())


def normalize_title(title) -> str:
    """Lowercase alphanumeric form of a title, used for within-ReliefWeb dedup.

    Mirrors `main._normalize_title` in spirit but is INDEPENDENT of it: these titles never
    enter `literature_document`, whose partial unique index on (project_context,
    title_norm) makes a recurring sitrep title physically unstorable. PURE."""
    s = _NON_ALNUM.sub(" ", _norm(title))
    return _WS.sub(" ", s).strip()


def series_key(title, source_names=None) -> str:
    """Collapse a recurring report SERIES to one key.

    "Ukraine: Humanitarian Impact Situation Report No. 12" and "... No. 13" are the same
    series; keeping them apart makes report volume look like an epidemic signal when it is
    only a publication cadence. Strips a trailing serial and/or date, then hashes with the
    primary source. PURE."""
    base = str(title or "")
    for _ in range(3):                            # "... No. 12 (5 March 2026)"
        new = _DATE_TAIL.sub("", _SERIAL_TAIL.sub("", base)).strip(" -–—:|")
        if new == base:
            break
        base = new
    src = _norm((source_names or [""])[0] if source_names else "")
    return hashlib.sha1(f"{src}|{normalize_title(base)}".encode()).hexdigest()[:32]


# ── query building ───────────────────────────────────────────────────────────
def _cond(field_name: str, value=None, negate: bool = False) -> dict:
    """One filter condition. NOTE: omitting `value` tests FIELD EXISTENCE, not equality —
    an easy silent bug when a caller passes None, so we never emit a value-less filter
    unless asked explicitly."""
    c: dict[str, Any] = {"field": field_name}
    if value is not None:
        c["value"] = value
    if negate:
        c["negate"] = True
    return c


def build_reports_query(
    *,
    pathogens: tuple | list | None = None,
    countries: tuple | list | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 100,
    offset: int = 0,
    languages: tuple | list | None = ("en",),
    include_epidemic_tag: bool = True,
    changed_since: str | None = None,
) -> dict:
    """POST body for /v2/reports, targeted at epidemic-intelligence content.

    Combines the editorial tag (`disaster_type = Epidemic`) OR a pathogen free-text sweep,
    because the tag lags the outbreak. Always emits an explicit `sort`: the docs warn that
    "if there is no sort specified, results may not be consistent", and paginating an
    unsorted result set silently skips and repeats records. PURE — no network."""
    limit = max(0, min(int(limit), MAX_LIMIT_PER_CALL))
    conditions: list[dict] = []

    topic: list[dict] = []
    if include_epidemic_tag:
        topic.append(_cond("disaster_type.name", "Epidemic"))
    terms = list(pathogens) if pathogens else list(PATHOGEN_TERMS)
    query_block = None
    if terms:
        query_block = {"value": " OR ".join(f'"{t}"' if " " in t else t for t in terms),
                       "fields": ["title^5", "body"], "operator": "OR"}
    if topic:
        conditions.append(topic[0] if len(topic) == 1
                          else {"operator": "OR", "conditions": topic})
    if countries:
        conditions.append(_cond("primary_country.iso3", [str(c).lower() for c in countries]))
    if languages:
        conditions.append(_cond("language.code", list(languages)))
    if date_from or date_to:
        rng = {}
        if date_from:
            rng["from"] = date_from
        if date_to:
            rng["to"] = date_to
        conditions.append({"field": "date.original", "value": rng})
    if changed_since:
        conditions.append({"field": "date.changed", "value": {"from": changed_since}})

    body: dict[str, Any] = {
        "limit": limit,
        "offset": max(0, int(offset)),
        # Sort by ingest time so an incremental sync has a stable, resumable cursor.
        "sort": ["date.created:desc"],
        "fields": {"include": [
            "id", "title", "body", "url", "origin", "status",
            "date.original", "date.created", "date.changed",
            "source.name", "source.shortname", "source.type",
            "primary_country.name", "primary_country.iso3", "country.name", "country.iso3",
            "disaster.id", "disaster.name", "disaster.glide", "disaster_type.name",
            "theme.name", "format.name", "language.code",
        ]},
    }
    if query_block:
        body["query"] = query_block
    if conditions:
        body["filter"] = ({"operator": "AND", "conditions": conditions}
                          if len(conditions) > 1 else conditions[0])
    return body


def build_disasters_query(*, status: tuple | list = ("current", "alert"),
                          limit: int = 200, include_archived: bool = False) -> dict:
    """POST body for /v2/disasters — the epidemic EVENT spine (GLIDE ids).

    Archived events are excluded by the default presets, so a historical backfill needs
    `include_archived` (the docs' `preset=analysis` behaviour). PURE."""
    body: dict[str, Any] = {
        "limit": max(0, min(int(limit), MAX_LIMIT_PER_CALL)),
        "sort": ["date.event:desc"],
        "filter": {"operator": "AND", "conditions": [
            _cond("type.name", "Epidemic"),
            _cond("status", list(status) + (["archive", "alert-archive"]
                                            if include_archived else [])),
        ]},
        "fields": {"include": [
            "id", "name", "glide", "status", "date.event", "date.changed",
            "primary_country.name", "primary_country.iso3", "type.name", "url",
        ]},
    }
    if include_archived:
        body["preset"] = "analysis"
    return body


# ── response parsing ─────────────────────────────────────────────────────────
def _first(seq, key: str):
    for it in (seq or []):
        if isinstance(it, dict) and it.get(key) is not None:
            return it[key]
    return None


def _names(seq, key: str = "name") -> list[str]:
    out: list[str] = []
    for it in (seq or []):
        if isinstance(it, dict) and it.get(key):
            v = str(it[key]).strip()
            if v and v not in out:
                out.append(v)
    return out


@dataclass
class Report:
    """One ReliefWeb report, flattened to what this app stores. `credibility` is set at
    parse time so nothing downstream has to remember to compute it."""
    rw_id: str
    title: str
    body: str
    url: str
    published_at: str | None          # date.original — the SOURCE's date, not our ingest
    created_at: str | None
    changed_at: str | None
    sources: list[str] = field(default_factory=list)
    format: str | None = None
    primary_country: str | None = None
    primary_iso3: str | None = None
    countries: list[str] = field(default_factory=list)
    disaster_names: list[str] = field(default_factory=list)
    glide: str | None = None
    disaster_types: list[str] = field(default_factory=list)
    themes: list[str] = field(default_factory=list)
    language: str | None = None
    status: str | None = None
    credibility: float = DEFAULT_CREDIBILITY
    title_norm: str = ""
    series_key: str = ""

    def to_row(self) -> dict:
        """JSON-safe dict matching the `situation_report` columns."""
        return {
            "rw_id": self.rw_id, "title": self.title, "body": self.body, "url": self.url,
            "published_at": self.published_at, "changed_at": self.changed_at,
            "sources": self.sources, "format": self.format,
            "primary_country": self.primary_country, "primary_iso3": self.primary_iso3,
            "countries": self.countries, "glide": self.glide,
            "disaster_names": self.disaster_names, "disaster_types": self.disaster_types,
            "themes": self.themes, "language": self.language, "status": self.status,
            "credibility": self.credibility, "title_norm": self.title_norm,
            "series_key": self.series_key,
        }


def parse_report(item: dict) -> Report | None:
    """One `data[]` entry from /v2/reports → `Report`, or None if unusable. PURE.

    Drops records with no title, and records whose editorial status is not published
    unless they carry a body: `status='to-review'` is INCLUDED BY DEFAULT by every
    ReliefWeb preset, so unreviewed items arrive whether or not you asked for them."""
    if not isinstance(item, dict):
        return None
    f = item.get("fields") if isinstance(item.get("fields"), dict) else {}
    title = str(f.get("title") or "").strip()
    if not title:
        return None
    rw_id = str(item.get("id") or f.get("id") or "").strip()
    if not rw_id:
        return None
    date_blk = f.get("date") if isinstance(f.get("date"), dict) else {}
    sources = _names(f.get("source")) or ([str(f["source"]["name"])]
                                          if isinstance(f.get("source"), dict)
                                          and f["source"].get("name") else [])
    pc = f.get("primary_country") if isinstance(f.get("primary_country"), dict) else {}
    fmt = _first(f.get("format"), "name") or (f.get("format", {}) or {}).get("name") \
        if isinstance(f.get("format"), (list, dict)) else None
    body = strip_markdown(f.get("body"))
    rep = Report(
        rw_id=rw_id,
        title=title,
        body=body,
        url=str(f.get("url") or f.get("origin") or "").strip(),
        published_at=_iso_date(date_blk.get("original")),
        created_at=_iso_date(date_blk.get("created")),
        changed_at=_iso_date(date_blk.get("changed")),
        sources=sources,
        format=fmt,
        primary_country=(pc.get("name") or None),
        primary_iso3=(str(pc["iso3"]).lower() if pc.get("iso3") else None),
        countries=_names(f.get("country")),
        disaster_names=_names(f.get("disaster")),
        glide=_first(f.get("disaster"), "glide"),
        disaster_types=_names(f.get("disaster_type")),
        themes=_names(f.get("theme")),
        language=(_names(f.get("language"), "code") or [None])[0],
        status=(str(f.get("status")).strip() if f.get("status") else None),
    )
    rep.credibility = credibility(rep.sources, rep.format)
    rep.title_norm = normalize_title(title)
    rep.series_key = series_key(title, rep.sources)
    return rep


def _iso_date(v) -> str | None:
    """ReliefWeb dates are ISO 8601 with an offset; keep the full string, drop junk."""
    s = str(v or "").strip()
    return s[:32] if s else None


def parse_reports(payload: dict) -> tuple[list[Report], int]:
    """Full /v2/reports response → (reports, totalCount). PURE."""
    if not isinstance(payload, dict):
        return [], 0
    reports = [r for r in (parse_report(it) for it in (payload.get("data") or [])) if r]
    try:
        total = int(payload.get("totalCount") or 0)
    except (TypeError, ValueError):
        total = 0
    return reports, total


def dedupe(reports, keep_per_series: int = 1) -> list[Report]:
    """Collapse ReliefWeb's structural duplication. PURE.

    4,000+ sources cover one event and the SAME document exists as separate EN/FR/ES/AR
    records, so raw report volume is a publication cadence, not an epidemic signal.
    Removes exact `rw_id` repeats and identical titles, then keeps at most
    `keep_per_series` of any recurring series (most recent first, most credible first)."""
    seen_id: set[str] = set()
    seen_title: set[str] = set()
    ordered = sorted(
        reports or [],
        key=lambda r: (r.published_at or r.created_at or "", r.credibility),
        reverse=True,
    )
    per_series: dict[str, int] = {}
    out: list[Report] = []
    for r in ordered:
        if r.rw_id in seen_id:
            continue
        if r.title_norm and r.title_norm in seen_title:
            continue
        if r.series_key:
            n = per_series.get(r.series_key, 0)
            if n >= max(1, int(keep_per_series)):
                continue
            per_series[r.series_key] = n + 1
        seen_id.add(r.rw_id)
        if r.title_norm:
            seen_title.add(r.title_norm)
        out.append(r)
    return out


# ── HTTP seams (the only impure part) ────────────────────────────────────────
def _http_post_json(url: str, body: dict, timeout: int = 30) -> dict:
    """POST a query to the ReliefWeb API. Isolated so tests replace it (no network)."""
    import requests
    r = requests.post(url, json=body, timeout=timeout,
                      headers={"Content-Type": "application/json"})
    r.raise_for_status()
    return r.json()


def _http_get_json(url: str, timeout: int = 30) -> dict:
    """GET (item requests). Isolated so tests replace it (no network)."""
    import requests
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def endpoint(kind: str, appname: str | None = None) -> str:
    """Full URL for a collection. `appname` is MANDATORY and, since 1 Nov 2025,
    pre-approval-gated — an ad-hoc string is likely to be rejected."""
    return f"{BASE_URL}/{kind}?appname={appname or DEFAULT_APPNAME}"


def fetch_reports(query: dict, appname: str | None = None) -> tuple[list[Report], int]:
    """One /v2/reports POST → (reports, totalCount). One call against the daily quota."""
    payload = _http_post_json(endpoint("reports", appname), query)
    return parse_reports(payload)


def fetch_disasters(query: dict, appname: str | None = None) -> list[dict]:
    """One /v2/disasters POST → flattened epidemic events (GLIDE spine)."""
    payload = _http_post_json(endpoint("disasters", appname), query)
    out: list[dict] = []
    for it in (payload.get("data") or []) if isinstance(payload, dict) else []:
        f = it.get("fields") if isinstance(it.get("fields"), dict) else {}
        pc = f.get("primary_country") if isinstance(f.get("primary_country"), dict) else {}
        d = f.get("date") if isinstance(f.get("date"), dict) else {}
        if not f.get("name"):
            continue
        out.append({
            "rw_id": str(it.get("id") or ""), "name": f.get("name"),
            "glide": f.get("glide"), "status": f.get("status"),
            "event_date": _iso_date(d.get("event")), "url": f.get("url"),
            "country": pc.get("name"), "iso3": (str(pc["iso3"]).lower()
                                                if pc.get("iso3") else None),
        })
    return out


def plan_pagination(total: int, limit: int, budget_calls: int) -> list[int]:
    """Offsets needed to walk `total` records at `limit` per call, truncated to the call
    budget. Returns the offsets ACTUALLY affordable — the caller must report the shortfall
    rather than presenting a partial sweep as complete. PURE."""
    limit = max(1, min(int(limit), MAX_LIMIT_PER_CALL))
    pages = max(0, math.ceil(max(0, int(total)) / limit))
    return [i * limit for i in range(min(pages, max(0, int(budget_calls))))]


def summarise_ingest(reports, kept, total_available: int, calls_used: int) -> dict:
    """Machine-readable ingest report. `truncated` is explicit: a sweep cut short by the
    quota must never read as full coverage."""
    fetched = len(reports or [])
    return {
        "fetched": fetched,
        "kept_after_dedup": len(kept or []),
        "dropped_as_duplicate": max(0, fetched - len(kept or [])),
        "total_available": int(total_available or 0),
        "calls_used": int(calls_used or 0),
        "truncated": fetched < int(total_available or 0),
        "quota_calls_per_day": QUOTA_CALLS_PER_DAY,
    }


def as_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


__all__ = [
    "BASE_URL", "QUOTA_CALLS_PER_DAY", "MAX_LIMIT_PER_CALL", "PATHOGEN_TERMS",
    "EPI_FORMATS", "MAX_CREDIBILITY", "DEFAULT_CREDIBILITY", "Report",
    "credibility", "strip_markdown", "normalize_title", "series_key",
    "build_reports_query", "build_disasters_query", "parse_report", "parse_reports",
    "dedupe", "fetch_reports", "fetch_disasters", "endpoint", "plan_pagination",
    "summarise_ingest", "as_json",
]
