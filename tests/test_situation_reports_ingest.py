"""The ReliefWeb ingest (POST /reliefweb/ingest, api/situation_reports.py).

What it did before (audit items 70 to 74):
  - English only, so the French, Spanish and Arabic reports that are often the first were
    never ingested;
  - 100 reports per call and 3 calls, so at most 300 per ingest, while the quota counts
    calls and allows 1000 reports in each;
  - the Epidemic tag in `filter` and the pathogen terms in `query`, which the API ANDs:
    the untagged early reports the terms exist for were excluded;
  - on any error the pages already received were thrown away, and the calls were counted
    against the daily quota twice (`except` and `finally`).

HTTP and the database are stubbed: nothing leaves the machine.
"""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi import HTTPException  # noqa: E402
from pydantic import ValidationError  # noqa: E402

import main  # noqa: E402,F401  (loads the API modules in order)
import reliefweb_source as rw  # noqa: E402
from api import situation_reports as SR  # noqa: E402
from conftest import patch_app  # noqa: E402


def _item(rid, title, source="WHO", date="2026-03-05T00:00:00+00:00", lang="fr"):
    return {"id": rid, "fields": {
        "title": title, "body": "Cas en hausse.", "url": f"https://reliefweb.int/report/{rid}",
        "date": {"original": date, "created": date, "changed": date},
        "source": [{"name": source}], "format": [{"name": "Situation Report"}],
        "primary_country": {"name": "DRC", "iso3": "COD"}, "language": [{"code": lang}],
    }}


def _branch(body) -> str:
    return rw.BRANCH_TERMS if "query" in body else rw.BRANCH_TAG


class _ReliefWeb:
    """ReliefWeb answered from memory. `records[branch]` is every record that branch
    matches; pages are cut from it by the body's offset and limit. `fail_at` is the call
    number (1-based) that raises."""

    def __init__(self, tag, terms, fail_at=None):
        self.records = {rw.BRANCH_TAG: tag, rw.BRANCH_TERMS: terms}
        self.fail_at = fail_at
        self.bodies: list[dict] = []

    def post(self, url, body, timeout=60):
        self.bodies.append(body)
        if self.fail_at and len(self.bodies) == self.fail_at:
            raise RuntimeError("HTTP 503 from ReliefWeb")
        recs = self.records[_branch(body)]
        page = recs[body["offset"]:body["offset"] + body["limit"]]
        return {"totalCount": len(recs), "count": len(page), "data": page}

    def branches_called(self):
        return [_branch(b) for b in self.bodies]


@pytest.fixture
def ingest(monkeypatch):
    noted: list[int] = []
    stored: dict = {}
    patch_app(monkeypatch, "RELIEFWEB_APPNAME", "unit-test")
    patch_app(monkeypatch, "_rw_budget_remaining", lambda: 1000)
    patch_app(monkeypatch, "_rw_note_calls", lambda n: noted.append(n))

    def upsert(reports):
        stored["reports"] = list(reports)
        return len(stored["reports"]), 0

    patch_app(monkeypatch, "_rw_upsert_reports", upsert)
    patch_app(monkeypatch, "_rw_link_to_scenario", lambda sid, ids: len(ids))

    def run(api, **payload):
        monkeypatch.setattr(rw, "_http_post_json", api.post)
        return SR.reliefweb_ingest(SR.ReliefWebIngestIn(**payload))

    return run, noted, stored


def test_the_sweep_is_the_tag_or_the_terms_in_every_language(ingest):
    run, noted, stored = ingest
    api = _ReliefWeb(tag=[_item("1", "RDC : choléra, rapport de situation n° 1")],
                     terms=[_item("2", "Kasaï : flambée de choléra", lang="fr"),
                            _item("3", "Brote de cólera en Haití", source="PAHO", lang="es")])
    out = run(api)
    assert sorted(api.branches_called()) == [rw.BRANCH_TAG, rw.BRANCH_TERMS]
    # Both halves were read: the untagged early reports are in.
    assert sorted(r.rw_id for r in stored["reports"]) == ["1", "2", "3"]
    assert out["total_available"] == 3 and out["fetched"] == 3 and out["truncated"] is False
    assert out["branches"][rw.BRANCH_TAG] == {"total": 1, "fetched": 1, "complete": True}
    assert out["branches"][rw.BRANCH_TERMS] == {"total": 2, "fetched": 2, "complete": True}
    # Every language, the largest page the API allows.
    for body in api.bodies:
        assert body["limit"] == rw.MAX_LIMIT_PER_CALL == 1000
        conds = body["filter"]["conditions"] if "conditions" in body["filter"] else [body["filter"]]
        assert not any(c.get("field") == "language.code" for c in conds)
    assert out["languages"] == "all"
    # Counted once: two calls, one note.
    assert noted == [2] and out["calls_used"] == 2 and out["error"] is None


def test_a_language_can_still_be_asked_for(ingest):
    run, _noted, _stored = ingest
    api = _ReliefWeb(tag=[], terms=[])
    out = run(api, languages=["fr"])
    for body in api.bodies:
        conds = body["filter"]["conditions"] if "conditions" in body["filter"] else [body["filter"]]
        assert {"field": "language.code", "value": ["fr"]} in conds
    assert out["languages"] == ["fr"]


def test_the_two_searches_share_the_call_budget_in_turn(ingest):
    """A tag search with many pages cannot spend the whole budget before the terms search
    has had a single call, and a search the budget did not let finish is said."""
    run, noted, stored = ingest
    api = _ReliefWeb(tag=[_item(str(i), f"Sitrep {i}", source=f"Org {i}") for i in range(6)],
                     terms=[_item("t1", "Untagged early report", source="MSF")])
    out = run(api, limit=2, max_calls=3)
    assert api.branches_called() == [rw.BRANCH_TAG, rw.BRANCH_TERMS, rw.BRANCH_TAG]
    assert [b["offset"] for b in api.bodies] == [0, 0, 2]
    assert out["branches"][rw.BRANCH_TERMS]["complete"] is True
    assert out["branches"][rw.BRANCH_TAG] == {"total": 6, "fetched": 4, "complete": False}
    assert out["truncated"] is True and out["total_available"] == 7 and out["fetched"] == 5
    assert noted == [3]


def test_a_budget_of_one_call_says_the_terms_were_never_searched(ingest):
    """The arithmetic on the known totals alone would read complete here."""
    run, _noted, _stored = ingest
    api = _ReliefWeb(tag=[_item("1", "Tagged")], terms=[_item("2", "Untagged")])
    out = run(api, max_calls=1)
    assert api.branches_called() == [rw.BRANCH_TAG]
    assert out["branches"][rw.BRANCH_TERMS] == {"total": None, "fetched": 0, "complete": False}
    assert out["truncated"] is True


def test_an_error_keeps_what_was_fetched_and_counts_each_call_once(ingest):
    run, noted, stored = ingest
    api = _ReliefWeb(tag=[_item("1", "Tagged sitrep"), _item("2", "Another tagged one", source="OCHA")],
                     terms=[_item("3", "Untagged")], fail_at=2)
    out = run(api)
    assert sorted(r.rw_id for r in stored["reports"]) == ["1", "2"], "the pages received are kept"
    assert out["inserted"] == 2 and "503" in out["error"] and out["truncated"] is True
    # Two calls were made (the failed one may have been counted by ReliefWeb): noted
    # once, not once in `except` and again in `finally`.
    assert noted == [2]


def test_an_error_before_any_page_is_a_502_counted_once(ingest):
    run, noted, stored = ingest
    with pytest.raises(HTTPException) as exc:
        run(_ReliefWeb(tag=[], terms=[], fail_at=1))
    assert exc.value.status_code == 502
    assert noted == [1] and "reports" not in stored


def test_a_whole_series_can_be_kept(ingest):
    run, _noted, stored = ingest
    series = [_item(str(i), f"DRC: Cholera Situation Report No. {i}",
                    date=f"2026-03-{i:02d}T00:00:00+00:00") for i in range(1, 4)]
    run(_ReliefWeb(tag=series, terms=[]), keep_per_series=0)
    assert len(stored["reports"]) == 3
    run(_ReliefWeb(tag=series, terms=[]))
    assert [r.rw_id for r in stored["reports"]] == ["3"], "the default keeps the newest"


def test_the_payload_defaults_and_bounds():
    p = SR.ReliefWebIngestIn()
    assert p.languages is None and p.limit == 1000 and p.max_calls == 3
    assert p.keep_per_series == 1
    with pytest.raises(ValidationError):
        SR.ReliefWebIngestIn(keep_per_series=-1)
