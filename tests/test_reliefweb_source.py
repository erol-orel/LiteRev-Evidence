"""Pure tests for reliefweb_source.py - no network, no database, stdlib only.

The HTTP seams are monkeypatched with a recorded-shape fixture. This matters more than
usual here: the live API was unreachable from the build environment (the egress proxy
answers 403 to CONNECT for api.reliefweb.int), so every claim about request/response
shape comes from the official documentation and is pinned here rather than assumed.
"""
import json

import pytest

import reliefweb_source as rw


# ── a response shaped like the documented /v2/reports envelope ───────────────
def _item(rid, title, *, sources=("WHO",), fmt="Situation Report", iso3="cod",
          country="Democratic Republic of the Congo", original="2026-03-05T00:00:00+00:00",
          body="## Cholera\n\nCases **rose** to 1,200. See [the bulletin](http://x/y).",
          lang="en", glide="EP-2026-000042-COD", status="published"):
    return {
        "id": rid,
        "fields": {
            "title": title,
            "body": body,
            "url": f"https://reliefweb.int/report/{rid}",
            "status": status,
            "date": {"original": original, "created": original, "changed": original},
            "source": [{"name": s, "shortname": s} for s in sources],
            "format": [{"name": fmt}],
            "primary_country": {"name": country, "iso3": iso3},
            "country": [{"name": country, "iso3": iso3}],
            "disaster": [{"id": 1, "name": "DRC: Cholera Outbreak", "glide": glide}],
            "disaster_type": [{"name": "Epidemic"}],
            "theme": [{"name": "Health"}],
            "language": [{"code": lang}],
        },
    }


def _payload(items, total=None):
    return {"totalCount": total if total is not None else len(items),
            "count": len(items), "data": items}


# ── query building ───────────────────────────────────────────────────────────
def test_reports_query_is_json_safe_and_always_sorted():
    q = rw.build_reports_query(countries=["COD"], date_from="2026-01-01", limit=250)
    json.dumps(q)                                  # goes out as a POST body
    # Unsorted pagination silently skips and repeats records - the docs warn about it.
    assert q["sort"] == ["date.created:desc"]
    assert q["limit"] == 250 and q["offset"] == 0
    assert "title" in q["fields"]["include"] and "date.original" in q["fields"]["include"]


def _conds(q):
    f = q.get("filter") or {}
    return f["conditions"] if "conditions" in f else ([f] if f else [])


def test_reports_query_combines_the_editorial_tag_with_a_keyword_sweep():
    """Tagging lags the outbreak, so the tag alone is systematically late: the sweep is
    the tag OR the pathogen terms. The API ANDs `filter` and `query`, so one body holding
    both returned only reports tagged AND naming a pathogen - the untagged early reports
    the terms exist for were excluded. The OR is two searches, made disjoint."""
    assert rw.reports_branches() == (rw.BRANCH_TAG, rw.BRANCH_TERMS)
    tag = rw.build_reports_query(branch=rw.BRANCH_TAG)
    terms = rw.build_reports_query(branch=rw.BRANCH_TERMS)
    # The tag branch: the editorial tag, no free-text query.
    assert "query" not in tag
    assert any(c.get("field") == "disaster_type.name" and c.get("value") == "Epidemic"
               and not c.get("negate") for c in _conds(tag))
    # The terms branch: the sweep, and NOT the tag, so no report is fetched twice and the
    # two totals add up to the union.
    assert terms["query"]["fields"] == ["title^5", "body"]
    assert "cholera" in terms["query"]["value"] and "mpox" in terms["query"]["value"]
    assert '"yellow fever"' in terms["query"]["value"]    # multiword terms are quoted
    assert any(c.get("field") == "disaster_type.name" and c.get("value") == "Epidemic"
               and c.get("negate") is True for c in _conds(terms))
    # No body ever ANDs a positive tag with the terms again.
    for q in (tag, terms):
        assert not ("query" in q and any(c.get("field") == "disaster_type.name"
                                         and not c.get("negate") for c in _conds(q)))


def test_without_the_tag_the_sweep_is_the_terms_alone():
    assert rw.reports_branches(include_epidemic_tag=False) == (rw.BRANCH_TERMS,)
    q = rw.build_reports_query(include_epidemic_tag=False, pathogens=["cholera"])
    assert q["query"]["value"] == "cholera"
    assert not any(c.get("field") == "disaster_type.name" for c in _conds(q))
    with pytest.raises(ValueError):
        rw.build_reports_query(branch=rw.BRANCH_TAG, include_epidemic_tag=False)
    with pytest.raises(ValueError):
        rw.build_reports_query(branch="both")


def test_every_language_and_the_largest_page_by_default():
    """English-only excluded the French, Spanish and Arabic reports that are often the
    first (a WHO AFRO bulletin, a ministry alert); a language can still be asked for. The
    quota counts calls, so a page of 100 bought a tenth of the reports for the same cost."""
    for branch in rw.reports_branches():
        q = rw.build_reports_query(branch=branch)
        assert q["limit"] == rw.MAX_LIMIT_PER_CALL == 1000
        assert not any(c.get("field") == "language.code" for c in _conds(q))
    q = rw.build_reports_query(languages=["fr", "en"], branch=rw.BRANCH_TAG)
    assert {"field": "language.code", "value": ["fr", "en"]} in _conds(q)


def test_reports_query_never_emits_a_value_less_filter():
    """A filter with a field but NO value tests EXISTENCE, not equality - it would
    silently match every record that merely has the field."""
    def walk(node):
        if isinstance(node, dict):
            if "field" in node and "conditions" not in node:
                assert "value" in node, f"value-less filter on {node['field']}"
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    for branch in rw.reports_branches():
        q = rw.build_reports_query(countries=None, languages=None, pathogens=["ebola"],
                                   branch=branch)
        walk(q.get("filter", {}))


def test_reports_query_limit_is_capped_at_the_documented_maximum():
    assert rw.build_reports_query(limit=99999)["limit"] == rw.MAX_LIMIT_PER_CALL == 1000
    assert rw.build_reports_query(limit=-5)["limit"] == 0


def test_country_and_date_and_changed_filters():
    q = rw.build_reports_query(countries=["COD", "SDN"], date_from="2026-01-01",
                               date_to="2026-06-30", changed_since="2026-06-01",
                               languages=["en"])
    conds = q["filter"]["conditions"]
    by_field = {c["field"]: c for c in conds if "field" in c}
    assert by_field["primary_country.iso3"]["value"] == ["cod", "sdn"]
    assert by_field["date.original"]["value"] == {"from": "2026-01-01", "to": "2026-06-30"}
    assert by_field["date.changed"]["value"] == {"from": "2026-06-01"}
    assert by_field["language.code"]["value"] == ["en"]
    # The same filters on both halves of the sweep: only the tag condition differs.
    tag = rw.build_reports_query(countries=["COD", "SDN"], date_from="2026-01-01",
                                 date_to="2026-06-30", changed_since="2026-06-01",
                                 languages=["en"], branch=rw.BRANCH_TAG)
    strip = lambda cs: [c for c in cs if c.get("field") != "disaster_type.name"]  # noqa: E731
    assert strip(tag["filter"]["conditions"]) == strip(conds)


def test_disasters_query_needs_the_analysis_preset_for_archived_events():
    """Archived epidemics are excluded by default - a historical backfill misses them."""
    live = rw.build_disasters_query()
    arch = rw.build_disasters_query(include_archived=True)
    assert "preset" not in live
    assert arch["preset"] == "analysis"
    statuses = [c["value"] for c in arch["filter"]["conditions"] if c["field"] == "status"][0]
    assert "archive" in statuses and "alert-archive" in statuses


# ── parsing ──────────────────────────────────────────────────────────────────
def test_parse_report_flattens_the_documented_shape():
    r = rw.parse_report(_item("3999001", "DRC: Cholera Situation Report No. 12"))
    assert r.rw_id == "3999001"
    assert r.primary_iso3 == "cod" and r.primary_country.startswith("Democratic")
    assert r.glide == "EP-2026-000042-COD"
    assert r.disaster_types == ["Epidemic"] and r.themes == ["Health"]
    assert r.language == "en" and r.format == "Situation Report"
    assert r.sources == ["WHO"]
    json.dumps(r.to_row())                         # must land in a JSONB column


def test_body_markdown_is_flattened_not_sanitised_as_html():
    r = rw.parse_report(_item("1", "T"))
    assert "**" not in r.body and "##" not in r.body
    assert "the bulletin" in r.body and "http://x/y" not in r.body   # link text kept
    assert "Cases rose to 1,200" in r.body


def test_parse_report_rejects_unusable_records():
    assert rw.parse_report(None) is None
    assert rw.parse_report({}) is None
    assert rw.parse_report({"id": "1", "fields": {"title": "  "}}) is None   # no title
    assert rw.parse_report({"fields": {"title": "x"}}) is None               # no id


def test_parse_reports_returns_total_for_pagination():
    reports, total = rw.parse_reports(_payload([_item("1", "A"), _item("2", "B")], total=57))
    assert len(reports) == 2 and total == 57
    assert rw.parse_reports({}) == ([], 0)
    assert rw.parse_reports({"data": [], "totalCount": "junk"}) == ([], 0)


# ── credibility: grey literature must never look like a study ────────────────
def test_credibility_is_capped_far_below_a_peer_reviewed_study():
    """quality_score IS the SEIR pooling weight; a flat 0.55 outranked a case report."""
    who = rw.credibility(["World Health Organization"])
    ngo = rw.credibility(["Some Local NGO"])
    press = rw.credibility(["Some Local NGO"], "News and Press Release")
    assert who == rw.MAX_CREDIBILITY == 0.45
    assert ngo == rw.DEFAULT_CREDIBILITY == 0.20
    assert press < ngo or press <= 0.25
    # nothing from ReliefWeb may reach the range of a peer-reviewed study
    for names in (["WHO"], ["Africa CDC"], ["UNICEF"], ["Nobody"], [], None):
        assert rw.credibility(names) <= rw.MAX_CREDIBILITY < 0.5


def test_credibility_takes_the_best_source_and_matches_substrings():
    assert rw.credibility(["WHO Regional Office for Africa"]) == 0.45
    assert rw.credibility(["Random Blog", "Africa CDC"]) == 0.42   # best of the two
    assert rw.parse_report(_item("1", "T", sources=("WHO",))).credibility == 0.45
    assert rw.parse_report(_item("2", "T", sources=("Tiny NGO",))).credibility == 0.20


# ── dedup: report volume is a publication cadence, not a signal ──────────────
def test_series_key_collapses_a_recurring_numbered_report():
    a = rw.series_key("Ukraine: Humanitarian Impact Situation Report No. 12", ["OCHA"])
    b = rw.series_key("Ukraine: Humanitarian Impact Situation Report No. 13", ["OCHA"])
    c = rw.series_key("Ukraine: Humanitarian Impact Situation Report (#14)", ["OCHA"])
    d = rw.series_key("Sudan: Cholera Situation Report No. 3", ["OCHA"])
    assert a == b == c
    assert a != d
    # same series from a different organisation is a different series
    assert a != rw.series_key("Ukraine: Humanitarian Impact Situation Report No. 12", ["WHO"])


def test_series_key_strips_a_trailing_date_too():
    assert (rw.series_key("DRC Cholera Update No. 4 (5 March 2026)", ["WHO"])
            == rw.series_key("DRC Cholera Update No. 5 (12 March 2026)", ["WHO"]))


def test_dedupe_removes_id_title_and_series_repeats_keeping_the_newest():
    items = [
        _item("1", "DRC: Cholera Situation Report No. 10", original="2026-03-01T00:00:00+00:00"),
        _item("2", "DRC: Cholera Situation Report No. 11", original="2026-03-08T00:00:00+00:00"),
        _item("3", "DRC: Cholera Situation Report No. 12", original="2026-03-15T00:00:00+00:00"),
        _item("4", "Sudan: Measles Outbreak Assessment", original="2026-03-02T00:00:00+00:00"),
        _item("1", "DRC: Cholera Situation Report No. 10", original="2026-03-01T00:00:00+00:00"),
    ]
    reports, _ = rw.parse_reports(_payload(items))
    kept = rw.dedupe(reports)
    titles = {r.title for r in kept}
    assert len(kept) == 2                                   # one sitrep series + Sudan
    assert "DRC: Cholera Situation Report No. 12" in titles  # newest of the series
    assert "Sudan: Measles Outbreak Assessment" in titles


def test_dedupe_can_keep_several_per_series_when_asked():
    reports, _ = rw.parse_reports(_payload([
        _item(str(i), f"DRC: Cholera Situation Report No. {i}",
              original=f"2026-03-{i:02d}T00:00:00+00:00") for i in range(1, 6)]))
    assert len(rw.dedupe(reports, keep_per_series=3)) == 3


def test_dedupe_can_keep_a_whole_series():
    """The older reports of a series carry the figures of their own date: an outbreak's
    history. 0 (or None) keeps them all; the default still keeps the newest only."""
    reports, _ = rw.parse_reports(_payload([
        _item(str(i), f"DRC: Cholera Situation Report No. {i}",
              original=f"2026-03-{i:02d}T00:00:00+00:00") for i in range(1, 6)]))
    assert len(rw.dedupe(reports, keep_per_series=0)) == 5
    assert len(rw.dedupe(reports, keep_per_series=None)) == 5
    assert len(rw.dedupe(reports)) == 1


def test_dedupe_keeps_the_same_title_from_two_organisations():
    """The title alone was the key: two organisations' reports sharing a title collapsed
    into one. The same title from the same organisation still does."""
    reports, _ = rw.parse_reports(_payload([
        _item("1", "Cholera Outbreak Response", sources=("WHO",),
              original="2026-03-01T00:00:00+00:00"),
        _item("2", "Cholera Outbreak Response", sources=("UNICEF",),
              original="2026-03-01T00:00:00+00:00"),
        _item("3", "Cholera outbreak response!", sources=("WHO",),
              original="2026-02-01T00:00:00+00:00"),
    ]))
    kept = rw.dedupe(reports, keep_per_series=0)
    assert sorted(r.rw_id for r in kept) == ["1", "2"]
    assert {r.sources[0] for r in kept} == {"WHO", "UNICEF"}


def test_dedupe_of_nothing_is_nothing():
    assert rw.dedupe([]) == [] and rw.dedupe(None) == []


# ── quota arithmetic: a truncated sweep must not read as complete ────────────
def test_plan_pagination_respects_the_call_budget():
    assert rw.plan_pagination(total=450, limit=100, budget_calls=10) == [0, 100, 200, 300, 400]
    assert rw.plan_pagination(total=450, limit=100, budget_calls=2) == [0, 100]   # truncated
    assert rw.plan_pagination(total=0, limit=100, budget_calls=5) == []
    assert rw.plan_pagination(total=100, limit=100, budget_calls=0) == []


def test_summarise_ingest_says_out_loud_when_coverage_is_partial():
    s = rw.summarise_ingest(reports=[1] * 200, kept=[1] * 150,
                            total_available=980, calls_used=2)
    assert s["truncated"] is True
    assert s["dropped_as_duplicate"] == 50 and s["kept_after_dedup"] == 150
    assert s["quota_calls_per_day"] == 1000
    full = rw.summarise_ingest(reports=[1] * 20, kept=[1] * 20, total_available=20, calls_used=1)
    assert full["truncated"] is False


# ── the HTTP seams are the only impure part, and are replaceable ─────────────
def test_fetch_reports_goes_through_the_seam_and_hits_v2(monkeypatch):
    seen = {}

    def fake_post(url, body, timeout=30):
        seen["url"], seen["body"] = url, body
        return _payload([_item("77", "DRC: Cholera Situation Report No. 1")], total=1)

    monkeypatch.setattr(rw, "_http_post_json", fake_post)
    reports, total = rw.fetch_reports(rw.build_reports_query(limit=5), appname="unit-test")
    assert total == 1 and reports[0].rw_id == "77"
    # v1 is decommissioned; the appname is mandatory and travels in the URL for POST too
    assert seen["url"].startswith("https://api.reliefweb.int/v2/reports?appname=unit-test")
    assert "/v1/" not in seen["url"]


def test_fetch_disasters_flattens_the_event_spine(monkeypatch):
    monkeypatch.setattr(rw, "_http_post_json", lambda url, body, timeout=30: {"data": [
        {"id": 5, "fields": {"name": "DRC: Cholera Outbreak - Mar 2026",
                             "glide": "EP-2026-000042-COD", "status": "current",
                             "date": {"event": "2026-03-01T00:00:00+00:00"},
                             "primary_country": {"name": "DRC", "iso3": "COD"},
                             "url": "https://reliefweb.int/disaster/ep-2026-000042-cod"}},
        {"id": 6, "fields": {"glide": "x"}},          # no name → dropped
    ]})
    events = rw.fetch_disasters(rw.build_disasters_query(), appname="unit-test")
    assert len(events) == 1
    assert events[0]["glide"] == "EP-2026-000042-COD" and events[0]["iso3"] == "cod"


def test_endpoint_requires_an_appname():
    assert rw.endpoint("reports") == f"{rw.BASE_URL}/reports?appname={rw.DEFAULT_APPNAME}"
    assert rw.endpoint("disasters", "my-approved-name").endswith("appname=my-approved-name")


def test_module_makes_no_network_call_on_import():
    """Everything above ran without a socket; guard against a future import-time fetch."""
    import importlib
    mod = importlib.reload(rw)
    assert mod.BASE_URL.startswith("https://api.reliefweb.int/v2")
