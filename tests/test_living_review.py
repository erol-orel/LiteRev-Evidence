"""The living review: every record of its window, through the library's own ingest, linked
to the scenario its preset belongs to, and run detached with a status anyone can read.

What it did before (audit items 67, 68, 86 to 90 and 96):
  - the PubMed queries ended their year ranges in 2026, so from 1 January 2027 they
    would have found nothing new;
  - each query kept the 200 newest records of its window, with no paging and no total;
  - titles and abstracts stopped at their first inline tag (`findtext`, `.text`), and the
    year came from PubDate/Year alone;
  - preprints came from the first page of the bioRxiv API, matched as substrings, dated
    2024 when undated and cut at 20;
  - rows were written as 'gesica' or 'user' and linked to no scenario, so enrichment and
    extraction skipped them, and a later search ingested the same paper a second time;
  - `?days=` was overridden by each preset's own window, and the API killed a run after
    600 seconds without a report.

HTTP is stubbed everywhere: nothing here leaves the machine.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")

from sqlalchemy import text  # noqa: E402

import main  # noqa: E402
import living_review_scheduler as L  # noqa: E402
from api import living_review as LR  # noqa: E402
from api import sources as S  # noqa: E402


# ── fakes ────────────────────────────────────────────────────────────────────

class _Resp:
    def __init__(self, payload=None, content: bytes = b"", status: int = 200):
        self._payload, self.content, self.status_code = payload, content, status
        self.text = content.decode("utf-8") if content else ""

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _article_xml(pmid: str, *, year_xml: str = "<Year>2026</Year>", doi: str | None = None) -> str:
    doi_xml = f'<ArticleId IdType="doi">{doi}</ArticleId>' if doi else ""
    return f"""<PubmedArticle><MedlineCitation><PMID>{pmid}</PMID><Article>
      <Journal><JournalIssue><PubDate>{year_xml}</PubDate></JournalIssue>
        <Title>Prehospital Emergency Care</Title></Journal>
      <ArticleTitle>Effect of <i>H5N1</i> on CO<sub>2</sub> in record {pmid}</ArticleTitle>
      <Abstract><AbstractText Label="BACKGROUND">Ambulance <b>calls</b> rose by 12<sup>%</sup>.</AbstractText>
        <AbstractText>A second <i>part</i>.</AbstractText></Abstract>
      <AuthorList><Author><LastName>Doe</LastName><ForeName>Jane</ForeName></Author>
        <Author><CollectiveName>The <i>EMS</i> Group</CollectiveName></Author></AuthorList>
    </Article></MedlineCitation>
    <PubmedData><ArticleIdList><ArticleId IdType="pubmed">{pmid}</ArticleId>{doi_xml}</ArticleIdList>
      <ReferenceList><Reference><ArticleIdList><ArticleId IdType="doi">10.9999/a-reference</ArticleId>
      </ArticleIdList></Reference></ReferenceList></PubmedData></PubmedArticle>"""


class _FakePubMed:
    """eutils answered from memory: esearch pages through `pmids`, efetch returns XML."""

    def __init__(self, pmids):
        self.pmids = list(pmids)
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, endpoint, params, timeout=30):
        self.calls.append((endpoint, dict(params)))
        if endpoint == "esearch.fcgi":
            s, n = int(params["retstart"]), int(params["retmax"])
            return _Resp({"esearchresult": {"count": str(len(self.pmids)),
                                            "idlist": self.pmids[s:s + n]}})
        ids = str(params["id"]).split(",")
        body = "<PubmedArticleSet>" + "".join(_article_xml(i) for i in ids) + "</PubmedArticleSet>"
        return _Resp(content=body.encode("utf-8"))

    def retstarts(self):
        return [int(p["retstart"]) for e, p in self.calls if e == "esearch.fcgi"]


class _FakeEPMC:
    """The Europe PMC preprint facet answered from memory, `n` records, cursor paging."""

    def __init__(self, n: int):
        self.n = n
        self.calls: list[dict] = []

    def __call__(self, params, timeout=30):
        self.calls.append(dict(params))
        start = 0 if params["cursorMark"] == "*" else int(params["cursorMark"])
        size = int(params["pageSize"])
        items = [{"id": f"PPR{i}", "source": "PPR", "doi": f"10.1101/2026.10.01.{i:06d}",
                  "title": f"Preprint {i} on prehospital triage",
                  "abstractText": "<p>An abstract, <i>long</i> enough to keep.</p>",
                  "pubYear": "2026", "authorString": "Doe J, Roe R"}
                 for i in range(start, min(start + size, self.n))]
        nxt = str(start + size) if start + size < self.n else params["cursorMark"]
        return {"hitCount": self.n, "resultList": {"result": items}, "nextCursorMark": nxt}


def _empty_search(*_a, **_k):
    return {"query": "q", "ids": [], "records": [], "total": 0, "fetched": 0,
            "truncated": False, "error": None}


# ── 1. the queries no longer expire on 1 January 2027 ────────────────────────

def test_no_preset_query_closes_its_year_range():
    """`2026[dp]` as an upper bound found nothing new from 1 January 2027, and already
    excluded in-press 2027 issues. The entry-date window bounds recency."""
    for sid, cfg in L.SCENARIO_QUERIES.items():
        assert "biorxiv_terms" not in cfg, f"{sid}: the substring sweep is gone"
        for q in cfg["pubmed_queries"]:
            for lo, hi in re.findall(r"(\d{4}):(\d{4})\[dp\]", q):
                assert hi == "3000", f"{sid}: {q}"
            assert "2026[dp]" not in q


# ── 2. every record of the window, and the total ─────────────────────────────

def test_pubmed_pages_through_every_hit_of_the_window(monkeypatch):
    fake = _FakePubMed([str(30_000_000 + i) for i in range(1234)])
    monkeypatch.setattr(L, "_eutils", fake)
    s = L.search_pubmed("(stroke) AND (2020:3000[dp])", days=30, max_results=5000)
    assert s["total"] == 1234 and s["fetched"] == 1234 and len(set(s["ids"])) == 1234
    assert s["truncated"] is False and s["error"] is None
    assert fake.retstarts() == [0, 500, 1000]
    assert all("[EDAT]" in p["term"] for e, p in fake.calls)


def test_the_guard_stops_the_paging_and_says_so(monkeypatch):
    fake = _FakePubMed([str(31_000_000 + i) for i in range(1234)])
    monkeypatch.setattr(L, "_eutils", fake)
    s = L.search_pubmed("stroke", days=30, max_results=1000)
    assert s["fetched"] == 1000 and s["total"] == 1234 and s["truncated"] is True
    assert fake.retstarts() == [0, 500]


def test_the_guard_is_generous_and_still_configurable(monkeypatch):
    import importlib
    import inspect
    assert 'os.environ.get("LIVING_REVIEW_MAX_RESULTS", "5000")' in inspect.getsource(L)
    try:
        monkeypatch.setenv("LIVING_REVIEW_MAX_RESULTS", "750")
        assert importlib.reload(L).MAX_RESULTS_PER_QUERY == 750
        monkeypatch.setenv("LIVING_REVIEW_MAX_RESULTS", "not a number")
        assert importlib.reload(L).MAX_RESULTS_PER_QUERY == 5000
    finally:
        monkeypatch.undo()
        importlib.reload(L)


def test_a_failed_page_is_reported_not_swallowed(monkeypatch):
    def boom(endpoint, params, timeout=30):
        raise RuntimeError("HTTP 500")
    monkeypatch.setattr(L, "_eutils", boom)
    s = L.search_pubmed("stroke", days=30)
    assert s["ids"] == [] and s["truncated"] is True and "500" in s["error"]


# ── 3. the whole title and abstract, and the year of a MedlineDate ───────────

def test_titles_and_abstracts_are_read_whole():
    xml = "<PubmedArticleSet>" + _article_xml("123", doi="10.1/abc") + "</PubmedArticleSet>"
    [rec] = L.parse_pubmed_articles(xml.encode("utf-8"))
    assert rec["title"] == "Effect of H5N1 on CO2 in record 123"
    assert rec["abstract"] == "Ambulance calls rose by 12%. A second part."
    assert rec["external_id"] == "pmid:123" and rec["year"] == 2026
    # The article's DOI, not the DOI of one of its references.
    assert rec["doi"] == "10.1/abc"
    assert rec["authors"] == "Doe Jane; The EMS Group"
    [no_doi] = L.parse_pubmed_articles(
        ("<PubmedArticleSet>" + _article_xml("124") + "</PubmedArticleSet>").encode("utf-8"))
    assert no_doi["doi"] is None


@pytest.mark.parametrize("year_xml,expected", [
    ("<MedlineDate>1998 Dec-1999 Jan</MedlineDate>", 1998),
    ("<MedlineDate>2026 Spring</MedlineDate>", 2026),
    ("<Year>2025</Year><Month>Oct</Month>", 2025),
    ("<Season>Spring</Season>", None),
])
def test_the_year_comes_from_medline_dates_too(year_xml, expected):
    xml = "<PubmedArticleSet>" + _article_xml("9", year_xml=year_xml) + "</PubmedArticleSet>"
    [rec] = L.parse_pubmed_articles(xml)
    assert rec["year"] == expected


# ── 4. preprints: the Europe PMC search the populate uses ────────────────────

def test_preprints_come_from_the_europe_pmc_facet_with_the_same_query(monkeypatch):
    fake = _FakeEPMC(250)
    monkeypatch.setattr(L, "_epmc_search", fake)
    p = L.search_preprints('("cardiac arrest" OR OHCA) AND (EMS) AND (2020:3000[dp])', days=30)
    q = fake.calls[0]["query"]
    assert q.startswith('(("cardiac arrest" OR OHCA) AND (EMS)) AND (SRC:PPR)')
    assert re.search(r"FIRST_PDATE:\[\d{4}-\d{2}-\d{2} TO \d{4}-\d{2}-\d{2}\]", q)
    assert "[dp]" not in q and "3000" not in q
    assert all(c["resultType"] == "core" and "sort" not in c for c in fake.calls)
    # Every page, not the first one: 250 records over three pages of 100.
    assert [c["cursorMark"] for c in fake.calls] == ["*", "100", "200"]
    assert p["total"] == 250 and p["fetched"] == 250 and len(p["records"]) == 250
    assert p["truncated"] is False
    rec = p["records"][0]
    assert rec["external_id"] == "PPR0" and rec["year"] == 2026
    assert rec["abstract"] == "An abstract, long enough to keep."
    assert rec["url"] == "https://europepmc.org/article/PPR/PPR0"


def test_the_preprint_guard_is_said(monkeypatch):
    monkeypatch.setattr(L, "_epmc_search", _FakeEPMC(250))
    p = L.search_preprints("triage", days=30, max_results=150)
    assert p["fetched"] == 150 and p["total"] == 250 and p["truncated"] is True


# ── 5. the library's own ingest, and a link to the scenario ──────────────────

PRESET = "lr-test-preset"


@pytest.fixture
def preset(monkeypatch):
    monkeypatch.setitem(L.SCENARIO_QUERIES, PRESET, {
        "label": "Living review test", "days_lookback": 45,
        "pubmed_queries": ['("stroke") AND (2020:3000[dp])'],
    })
    return PRESET


def _pubmed_record(pmid: str) -> dict:
    return {"pmid": pmid, "external_id": f"pmid:{pmid}", "title": f"Stroke triage record {pmid}",
            "abstract": "An abstract long enough to be kept by the ingest.", "year": 2026,
            "doi": f"10.5555/lr.{pmid}", "authors": "Doe Jane", "journal": "J",
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"}


def _preprint_record(i: int) -> dict:
    return {"external_id": f"PPR{i}", "title": f"Preprint {i} on stroke triage",
            "abstract": "An abstract long enough to be kept.", "year": 2026,
            "doi": f"10.1101/2026.10.{i}", "url": f"https://europepmc.org/article/PPR/PPR{i}",
            "authors": None, "journal": None}


def test_records_go_through_the_library_ingest_and_are_linked(monkeypatch, preset):
    ingested, links, changed = [], [], []

    def fake_ingest(**kw):
        ingested.append(kw)
        return (900 + len(ingested), True)

    monkeypatch.setattr(S, "_ingest_doc_direct", fake_ingest)
    monkeypatch.setattr(L, "_existing_scenario", lambda sid: {"id": sid, "filters": {}})
    monkeypatch.setattr(L, "search_pubmed", lambda q, days, max_results=0: {
        "ids": ["1", "2"], "total": 2, "fetched": 2, "truncated": False, "error": None})
    monkeypatch.setattr(L, "_known_pmids", lambda ids: {"2": 77})       # already held
    monkeypatch.setattr(L, "fetch_pubmed_records", lambda ids: ([_pubmed_record(i) for i in ids], None))
    monkeypatch.setattr(L, "search_preprints", lambda q, days, max_results=0: {
        "query": "pp", "records": [_preprint_record(5)], "total": 1, "fetched": 1,
        "truncated": False, "error": None})
    monkeypatch.setattr(L, "_link_documents", lambda sid, ids: links.append((sid, sorted(ids))) or len(ids))
    monkeypatch.setattr(L, "_corpus_changed", lambda sid: changed.append(sid))

    out = L.run_living_review_for_scenario(preset)

    assert [k["project_context"] for k in ingested] == ["literev", "literev"]
    pm, pp = ingested
    assert (pm["source"], pm["source_type"], pm["external_id"]) == ("pubmed", "article", "pmid:1")
    assert (pp["source"], pp["source_type"], pp["external_id"]) == ("preprint", "preprint", "PPR5")
    # The held record is linked without being downloaded again.
    assert links == [(preset, [77, 901, 902])]
    assert changed == [preset]
    assert out["new_documents"] == 2 and out["already_in_library"] == 1
    assert out["linked_to_scenario"] == 3 and out["linked_scenario_id"] == preset
    assert out["scenario_missing"] is False
    assert [s["source"] for s in out["searches"]] == ["pubmed", "preprints"]


def test_a_record_two_queries_return_is_handled_once(monkeypatch):
    """Downloaded once, counted once: the second query would have found it "already in
    the library", since the first had just put it there."""
    monkeypatch.setitem(L.SCENARIO_QUERIES, "lr-two-queries", {
        "label": "Two queries", "days_lookback": 30,
        "pubmed_queries": ["(stroke)", "(thrombolysis)"]})
    hits = {"(stroke)": ["1", "2"], "(thrombolysis)": ["2", "3"]}
    fetched, ingested = [], []
    monkeypatch.setattr(L, "_existing_scenario", lambda sid: None)
    monkeypatch.setattr(L, "search_pubmed", lambda q, days, max_results=0: {
        "ids": hits[q], "total": 2, "fetched": 2, "truncated": False, "error": None})
    monkeypatch.setattr(L, "_known_pmids", lambda ids: {})
    monkeypatch.setattr(L, "fetch_pubmed_records",
                        lambda ids: fetched.extend(ids) or ([_pubmed_record(i) for i in ids], None))
    monkeypatch.setattr(L, "search_preprints", lambda q, days, max_results=0: {
        "query": q, "records": [_preprint_record(7)], "total": 1, "fetched": 1,
        "truncated": False, "error": None})
    monkeypatch.setattr(S, "_ingest_doc_direct",
                        lambda **kw: ingested.append(kw["external_id"]) or (len(ingested), True))
    out = L.run_living_review_for_scenario("lr-two-queries")
    assert fetched == ["1", "2", "3"]
    assert ingested == ["pmid:1", "pmid:2", "pmid:3", "PPR7"]
    assert out["new_documents"] == 4 and out["already_in_library"] == 0


def test_a_preset_without_a_scenario_is_ingested_reported_and_linked_to_nothing(monkeypatch, preset, tmp_path):
    monkeypatch.setattr(S, "_ingest_doc_direct", lambda **kw: (950, True))
    monkeypatch.setattr(L, "_existing_scenario", lambda sid: None)
    monkeypatch.setattr(L, "search_pubmed", lambda q, days, max_results=0: {
        "ids": ["1"], "total": 1, "fetched": 1, "truncated": False, "error": None})
    monkeypatch.setattr(L, "_known_pmids", lambda ids: {})
    monkeypatch.setattr(L, "fetch_pubmed_records", lambda ids: ([_pubmed_record(i) for i in ids], None))
    monkeypatch.setattr(L, "search_preprints", _empty_search)

    def no_link(*a, **k):
        raise AssertionError("nothing to link to")

    monkeypatch.setattr(L, "_link_documents", no_link)
    out = L.run_living_review_for_scenario(preset)
    assert out["scenario_missing"] is True and out["linked_to_scenario"] == 0
    assert out["new_documents"] == 1
    report = L.save_run_report([out], output_dir=str(tmp_path))
    assert report["presets_without_scenario"] == [preset]
    assert json.loads((tmp_path / L.REPORT_FILE).read_text())["presets_without_scenario"] == [preset]


def test_the_three_presets_without_a_scenario_are_the_known_ones():
    """undertriage-risk, trauma-care and mass-casualty match no GESICA scenario; the other
    seven do. Pinned so that a renamed scenario does not silently stop being linked."""
    from gesica_scenario_enriched_metadata import GESICA_ENRICHED
    assert sorted(set(L.SCENARIO_QUERIES) - set(GESICA_ENRICHED)) == [
        "mass-casualty", "trauma-care", "undertriage-risk"]


def test_a_scenario_that_leaves_preprints_out_gets_none(monkeypatch, preset):
    seen = []
    monkeypatch.setattr(L, "_existing_scenario",
                        lambda sid: {"id": sid, "filters": {"exclude_preprints": True}})
    monkeypatch.setattr(L, "search_pubmed", lambda q, days, max_results=0: seen.append(q) or _empty_search())
    monkeypatch.setattr(L, "_known_pmids", lambda ids: {})

    def no_preprints(*a, **k):
        raise AssertionError("the preprint facet must not be searched")

    monkeypatch.setattr(L, "search_preprints", no_preprints)
    out = L.run_living_review_for_scenario(preset)
    assert seen and all(q.endswith("NOT preprint[pt]") for q in seen)
    assert out["preprints_excluded"] is True


def test_a_dry_run_writes_nothing(monkeypatch, preset):
    def no_write(**kw):
        raise AssertionError("a dry run must not ingest")

    monkeypatch.setattr(S, "_ingest_doc_direct", no_write)
    monkeypatch.setattr(L, "_existing_scenario", lambda sid: {"id": sid, "filters": {}})
    monkeypatch.setattr(L, "search_pubmed", lambda q, days, max_results=0: {
        "ids": ["1", "2"], "total": 2, "fetched": 2, "truncated": False, "error": None})
    monkeypatch.setattr(L, "_known_pmids", lambda ids: {})
    monkeypatch.setattr(L, "fetch_pubmed_records", lambda ids: ([_pubmed_record(i) for i in ids], None))
    monkeypatch.setattr(L, "search_preprints", _empty_search)
    monkeypatch.setattr(L, "_in_library", lambda rec: rec["external_id"] == "pmid:2")
    monkeypatch.setattr(L, "_link_documents", no_write)
    out = L.run_living_review_for_scenario(preset, dry_run=True)
    assert out["new_documents"] == 1 and out["already_in_library"] == 1
    assert out["linked_to_scenario"] == 0 and out["dry_run"] is True


# ── 6. the caller's window is the one used ───────────────────────────────────

def test_the_callers_window_wins_over_the_presets(monkeypatch, preset):
    seen = []
    monkeypatch.setattr(L, "_existing_scenario", lambda sid: None)
    monkeypatch.setattr(L, "_known_pmids", lambda ids: {})
    monkeypatch.setattr(L, "search_pubmed", lambda q, days, max_results=0: seen.append(days) or _empty_search())
    monkeypatch.setattr(L, "search_preprints", lambda q, days, max_results=0: seen.append(days) or _empty_search())
    assert L.run_living_review_for_scenario(preset)["window_days"] == 45
    assert set(seen) == {45}
    seen.clear()
    assert L.run_living_review_for_scenario(preset, days=120)["window_days"] == 120
    assert set(seen) == {120}
    assert L._window_days(None) == L.DEFAULT_LOOKBACK_DAYS == 30
    assert L._window_days(0, 60) == 1


@pytest.mark.parametrize("argv,days", [(["--all-scenarios"], None),
                                       (["--all-scenarios", "--days", "90"], 90)])
def test_the_command_line_passes_days_only_when_given(monkeypatch, argv, days):
    got = {}
    monkeypatch.setattr(sys, "argv", ["living_review_scheduler.py", *argv])
    monkeypatch.setattr(L, "run_once", lambda **kw: got.update(kw) or 0)
    assert L.main() == 0
    assert got["days"] == days and got["all_scenarios"] is True


def test_a_user_scenario_runs_alone_through_scenario(monkeypatch, tmp_path):
    monkeypatch.setenv("LIVING_REVIEW_REPORT_DIR", str(tmp_path))
    monkeypatch.setattr(L, "_db_ready", lambda: None)
    got = {}
    monkeypatch.setattr(L, "run_user_scenarios",
                        lambda **kw: got.update(kw) or [{"scenario_id": kw["only"], "new_documents": 0}])
    assert L.run_once(scenario="usr-0123456789ab") == 0
    assert got["only"] == "usr-0123456789ab"


# ── 7. the run's state, written by the run ───────────────────────────────────

def test_a_database_that_does_not_answer_is_an_error_not_a_silent_success(monkeypatch, tmp_path):
    monkeypatch.setenv("LIVING_REVIEW_REPORT_DIR", str(tmp_path))

    def down():
        raise RuntimeError("connection refused")

    monkeypatch.setattr(L, "_db_ready", down)
    assert L.run_once(all_scenarios=True) == 1
    state = L.read_run_state()
    assert state["state"] == "error" and "connection refused" in state["error"]
    assert state["pid"] == os.getpid() and state["finished_at"]


def test_a_run_writes_running_then_done(monkeypatch, tmp_path):
    monkeypatch.setenv("LIVING_REVIEW_REPORT_DIR", str(tmp_path))
    monkeypatch.setattr(L, "_db_ready", lambda: None)
    during = {}

    def fake_all(**kw):
        during.update(L.read_run_state())
        return [{"scenario_id": "mass-casualty", "new_documents": 3, "linked_to_scenario": 0,
                 "scenario_missing": True, "truncated": False}]

    monkeypatch.setattr(L, "run_all_scenarios", fake_all)
    assert L.run_once(all_scenarios=True, days=14) == 0
    assert during["state"] == "running" and during["days"] == 14
    state = L.read_run_state()
    assert state["state"] == "done" and state["total_new_documents"] == 3
    assert state["presets_without_scenario"] == ["mass-casualty"]
    assert json.loads((tmp_path / L.REPORT_FILE).read_text())["total_new_documents"] == 3


# ── 8. the API: detached, honoured `days`, one run at a time, a status ───────

class _FakeProc:
    """A scheduler process that runs until released."""
    launched: list = []

    def __init__(self, cmd, **kw):
        self.cmd, self.kw, self.pid = cmd, kw, 4_000_000 + len(_FakeProc.launched)
        self.returncode = None
        self._done = threading.Event()
        _FakeProc.launched.append(self)

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self._done.wait(10)
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def finish(self, rc=0):
        self.returncode = rc
        self._done.set()


@pytest.fixture
def api(monkeypatch, tmp_path):
    monkeypatch.setenv("LIVING_REVIEW_REPORT_DIR", str(tmp_path))
    monkeypatch.setattr(LR, "_LAUNCHED", {})
    _FakeProc.launched = []
    monkeypatch.setattr(subprocess, "Popen", _FakeProc)

    def no_run(*a, **k):
        raise AssertionError("the run is detached, never waited for with a timeout")

    monkeypatch.setattr(subprocess, "run", no_run)
    yield tmp_path
    for p in _FakeProc.launched:
        p.finish()
    # The watchers write the run's end: let them do it while LIVING_REVIEW_REPORT_DIR
    # still points here, and not into the repository once the test has restored it.
    for t in threading.enumerate():
        if t.name == "living-review-watch":
            t.join(timeout=10)


def test_days_reaches_the_scheduler_only_when_given(api):
    out = LR.living_review_run(scenario_id="all", days=None, dry_run=True, lang="en", _=None)
    assert out["status"] == "started"
    cmd = _FakeProc.launched[-1].cmd
    assert "--days" not in cmd and "--all-scenarios" in cmd and "--dry-run" in cmd
    assert _FakeProc.launched[-1].kw.get("start_new_session") is True
    _FakeProc.launched[-1].finish()
    LR._LAUNCHED.clear()
    LR.living_review_run(scenario_id="stroke-detection", days=90, dry_run=False, lang="en", _=None)
    cmd = _FakeProc.launched[-1].cmd
    assert cmd[cmd.index("--days") + 1] == "90"
    assert cmd[cmd.index("--scenario") + 1] == "stroke-detection"


def test_a_second_run_waits_for_the_first(api):
    first = LR.living_review_run(scenario_id="all", days=None, dry_run=False, lang="en", _=None)
    second = LR.living_review_run(scenario_id="all", days=None, dry_run=False, lang="en", _=None)
    assert first["status"] == "started" and second["status"] == "already_running"
    assert len(_FakeProc.launched) == 1
    assert second["run"]["state"] == "starting" and second["run"]["pid"] == first["pid"]


def test_the_status_reports_the_run_in_progress_and_its_end(api):
    out = LR.living_review_run(scenario_id="all", days=None, dry_run=False, lang="en", _=None)
    proc = _FakeProc.launched[-1]
    status = LR.living_review_status(lang="en")
    assert status["run"]["state"] == "starting" and status["status"] == "running"
    # The scheduler writes its own state, then its report and its end.
    L.write_run_state({"state": "running", "pid": proc.pid, "scenario": "all"})
    assert LR.living_review_status(lang="en")["run"]["state"] == "running"
    (api / L.REPORT_FILE).write_text(json.dumps({"total_new_documents": 4, "scenarios": []}))
    L.write_run_state({"state": "done", "pid": proc.pid, "scenario": "all"})
    proc.finish(0)
    status = LR.living_review_status(lang="en")
    assert status["total_new_documents"] == 4 and status["run"]["state"] == "done"
    assert out["pid"] == proc.pid


def test_a_run_whose_process_vanished_reads_interrupted(api):
    """Killed by a restart, the scheduler cannot write its end: "running" would be false."""
    if not hasattr(os, "fork"):
        pytest.skip("no fork on this platform")
    dead = os.fork()
    if dead == 0:                                        # the child: gone at once
        os._exit(0)
    os.waitpid(dead, 0)
    L.write_run_state({"state": "running", "pid": dead, "scenario": "all"})
    assert LR.living_review_status(lang="en")["run"]["state"] == "interrupted"
    # Our own pid is alive, but it is not a scheduler: still interrupted.
    L.write_run_state({"state": "running", "pid": os.getpid(), "scenario": "all"})
    assert LR.living_review_status(lang="en")["run"]["state"] == "interrupted"


def test_the_watcher_records_a_scheduler_that_died_without_a_word(api):
    proc = _FakeProc(["python", "living_review_scheduler.py"])
    proc.finish(3)
    LR._watch_run(proc, {"scenario": "all", "days": None, "dry_run": False, "log": "x.log"})
    state = L.read_run_state()
    assert state["state"] == "error" and state["returncode"] == 3 and state["pid"] == proc.pid
    # A scheduler that wrote its own end keeps it.
    proc2 = _FakeProc(["python", "living_review_scheduler.py"])
    L.write_run_state({"state": "done", "pid": proc2.pid, "total_new_documents": 2})
    proc2.finish(0)
    LR._watch_run(proc2, {"scenario": "all"})
    assert L.read_run_state()["total_new_documents"] == 2


def test_the_trigger_runs_the_user_scenario_it_was_asked_for(api, monkeypatch):
    from fastapi import HTTPException

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class _Engine:
        def connect(self):
            return _Conn()

    def not_gesica(sid, conn=None):
        raise HTTPException(status_code=404)

    monkeypatch.setattr(LR, "engine", _Engine())
    monkeypatch.setattr(LR, "_get_db_gesica_scenario_or_404", not_gesica)
    monkeypatch.setattr(LR, "_get_user_scenario_or_404",
                        lambda sid: {"id": sid, "name": "Mine", "query": "stroke"})
    out = LR.trigger_living_review(scenario_id="usr-0123456789ab", dry_run=False, days=None,
                                   lang="en", _=None)
    assert out["status"] == "triggered" and out["run"]["pid"]
    cmd = _FakeProc.launched[-1].cmd
    assert cmd[cmd.index("--scenario") + 1] == "usr-0123456789ab"
    assert "--user-scenarios" not in cmd


# ── 9. against the database: the link and its date ───────────────────────────

def _db_or_skip():
    try:
        with main.engine.connect() as c:
            c.execute(text("SELECT 1"))
    except Exception as e:                               # noqa: BLE001
        pytest.skip(f"no reachable Postgres ({type(e).__name__})")


@pytest.fixture
def scenario_row(preset):
    _db_or_skip()
    main._ensure_user_scenarios_table()      # article_scenarios, and its assigned_at default

    def _clean():
        with main.engine.begin() as c:
            c.execute(text("DELETE FROM article_scenarios WHERE scenario_id = :s"), {"s": PRESET})
            c.execute(text("DELETE FROM scenario_settings WHERE scenario_id = :s"), {"s": PRESET})
            c.execute(text("DELETE FROM user_scenarios WHERE id = :s"), {"s": PRESET})

    _clean()
    with main.engine.begin() as c:
        c.execute(text("INSERT INTO user_scenarios (id, name, query) VALUES (:s, 'LR test', 'stroke')"),
                  {"s": PRESET})
        c.execute(text("INSERT INTO scenario_settings (scenario_id, clustering_json) "
                       "VALUES (:s, CAST('{}' AS jsonb))"), {"s": PRESET})
    yield PRESET
    _clean()


def test_the_links_are_written_with_their_date_and_only_once(monkeypatch, scenario_row):
    ids = iter(range(990_001, 990_100))
    monkeypatch.setattr(S, "_ingest_doc_direct", lambda **kw: (next(ids), True))
    monkeypatch.setattr(L, "search_pubmed", lambda q, days, max_results=0: {
        "ids": ["1"], "total": 1, "fetched": 1, "truncated": False, "error": None})
    monkeypatch.setattr(L, "_known_pmids", lambda ids_: {})
    monkeypatch.setattr(L, "fetch_pubmed_records", lambda p: ([_pubmed_record(i) for i in p], None))
    monkeypatch.setattr(L, "search_preprints", _empty_search)

    out = L.run_living_review_for_scenario(scenario_row)
    assert out["linked_to_scenario"] == 1
    with main.engine.connect() as c:
        rows = c.execute(text("SELECT document_id, similarity_score, assigned_at FROM article_scenarios "
                              "WHERE scenario_id = :s"), {"s": scenario_row}).all()
        cache, moved = c.execute(text("SELECT clustering_json, corpus_moved_at FROM "
                                      "scenario_settings WHERE scenario_id = :s"),
                                 {"s": scenario_row}).first()
    assert [(r[0], r[1]) for r in rows] == [(990_001, None)]
    assert rows[0][2] is not None, "the link carries the date it was made"
    # The watch brought in new papers, so every view computed on the previous corpus
    # describes the previous corpus. The clustering is not destroyed for it (its summaries
    # cost a model call each) but the move is dated, which is what stops it passing for
    # the current one.
    assert cache is not None, "a paid view was destroyed by a watch run"
    assert moved is not None, "the views computed on the old corpus would pass for current"

    # The same paper again: the ingest finds it, the link exists, nothing changes.
    monkeypatch.setattr(S, "_ingest_doc_direct", lambda **kw: (990_001, False))
    again = L.run_living_review_for_scenario(scenario_row)
    assert again["linked_to_scenario"] == 0 and again["already_in_library"] == 1


def _ingestion_works() -> bool:
    probe = "pmid:livingreviewprobe"
    try:
        did, _new = S._ingest_doc_direct(
            source="pubmed", title="Living review ingestion probe, inserted then deleted",
            abstract="A probe document for the living review test.", year=2026,
            url=None, external_id=probe, doi=None)
    except Exception:                                    # noqa: BLE001
        return False
    try:
        with main.engine.begin() as c:
            c.execute(text("DELETE FROM document_chunk WHERE document_id = :d"), {"d": did})
            c.execute(text("DELETE FROM literature_document WHERE external_id = :e"), {"e": probe})
    except Exception:                                    # noqa: BLE001
        pass
    return did is not None


def test_a_real_run_writes_one_literev_row_and_links_it(monkeypatch, scenario_row):
    if not _ingestion_works():
        pytest.skip("ingestion does not work on this schema: nothing to measure here")
    pmid = str(int(time.time() * 1000))[-9:]
    monkeypatch.setattr(L, "_eutils", _FakePubMed([pmid]))
    monkeypatch.setattr(L, "search_preprints", _empty_search)
    try:
        first = L.run_living_review_for_scenario(scenario_row)
        second = L.run_living_review_for_scenario(scenario_row)
        with main.engine.connect() as c:
            rows = c.execute(text("SELECT id, project_context, title FROM literature_document "
                                  "WHERE external_id = :e"), {"e": f"pmid:{pmid}"}).all()
            linked = c.execute(text("SELECT COUNT(*) FROM article_scenarios a JOIN literature_document d "
                                    "ON d.id = a.document_id WHERE a.scenario_id = :s AND d.external_id = :e"),
                               {"s": scenario_row, "e": f"pmid:{pmid}"}).scalar()
        assert len(rows) == 1 and rows[0][1] == "literev"
        assert rows[0][2] == f"Effect of H5N1 on CO2 in record {pmid}"
        assert first["new_documents"] == 1 and first["linked_to_scenario"] == 1
        # Known by its PMID: linked already, not downloaded again, not a second row.
        assert second["new_documents"] == 0 and second["already_in_library"] == 1
        assert second["linked_to_scenario"] == 0 and linked == 1
    finally:
        with main.engine.begin() as c:
            c.execute(text("DELETE FROM document_chunk WHERE document_id IN (SELECT id FROM "
                           "literature_document WHERE external_id = :e)"), {"e": f"pmid:{pmid}"})
            c.execute(text("DELETE FROM literature_document WHERE external_id = :e"), {"e": f"pmid:{pmid}"})
