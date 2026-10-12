"""A PubMed query that announces more than 10,000 records is read whole, in year slices.

eutils never returns an identifier past the 10,000th of one PubMed search (retstart +
retmax <= 10,000). A query announcing 23,000 records brought the first 10,000, and the
PRISMA card presented them as the search. Decision of the product owner: above 10,000, the
SAME query is run restricted to publication-year ranges (`AND (YYYY:YYYY[dp])`), each range
split in two until it announces 10,000 or fewer; every range is read, the identifiers are
merged (a record published online in December and in print in January answers both years
and counts once), and the PRISMA total stays the count of the whole query, not the sum of
the slices. A single year still over 10,000 keeps its 10,000 most relevant, and says so.

The fake eutils below holds its records in memory with their publication years, applies the
year clauses of the term it receives, ranks by a fixed pseudo-relevance, and refuses, like
eutils, anything past its ceiling. Nothing leaves the machine.
"""
from __future__ import annotations

import hashlib
import re
import time
import xml.sax.saxutils as _sx

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")

from sqlalchemy import text  # noqa: E402

import main  # noqa: E402
from api import sources as S  # noqa: E402

# 23,000 records from 1950 to 2026, more each decade, as a real literature grows. Every
# 25th record of a year (but the last) is also dated the next year: e-published in one,
# printed in the next, so PubMed's [dp] finds it in both.
GROWTH = ({y: 20 for y in range(1950, 1980)} | {y: 150 for y in range(1980, 2000)}
          | {y: 400 for y in range(2000, 2010)} | {y: 800 for y in range(2010, 2020)}
          | {y: 1100 for y in range(2020, 2026)} | {2026: 800})
assert sum(GROWTH.values()) == 23_000


def _records(by_year: dict[int, int], straddle_every: int = 25, first: int = 30_000_000) -> dict[str, tuple]:
    out: dict[str, tuple] = {}
    pmid = first
    last = max(by_year)
    for year in sorted(by_year):
        for i in range(by_year[year]):
            pmid += 1
            two = straddle_every and i % straddle_every == 0 and year < last
            out[str(pmid)] = (year, year + 1) if two else (year,)
    return out


_CLAUSE = re.compile(r"\b(AND|NOT) \((\d{4}):(\d{4})\[dp\]\)")


class FakeEutilsSearch:
    """esearch answered from memory: the year clauses of the term are applied, the order is
    a fixed pseudo-relevance, and nothing is served past `ceiling`, like eutils."""

    def __init__(self, records: dict[str, tuple], ceiling: int = 10_000):
        self.records = records
        self.ceiling = ceiling
        self.calls: list[tuple[str, int, int]] = []
        self.past_ceiling: list[tuple[str, int, int]] = []
        self.rank = sorted(records, key=lambda p: hashlib.sha1(p.encode()).hexdigest())

    def matching(self, term: str) -> list[str]:
        clauses = [(op, int(lo), int(hi)) for op, lo, hi in _CLAUSE.findall(term)]

        def keep(years) -> bool:
            for op, lo, hi in clauses:
                hit = any(lo <= y <= hi for y in years)
                if (op == "AND" and not hit) or (op == "NOT" and hit):
                    return False
            return True
        return [p for p in self.rank if keep(self.records[p])]

    def __call__(self, term: str, retstart: int, retmax: int) -> dict:
        self.calls.append((term, int(retstart), int(retmax)))
        found = self.matching(term)
        if int(retstart) + int(retmax) > self.ceiling:
            self.past_ceiling.append((term, int(retstart), int(retmax)))
            return {"ERROR": f"retstart+retmax past {self.ceiling}", "count": str(len(found))}
        return {"count": str(len(found)), "idlist": found[int(retstart):int(retstart) + int(retmax)]}


def _render(lo, hi):
    return S.pubmed_years_term("(poultry worker) AND (h5n1)", lo, hi)


def _split(fake, total=None, cap=100_000, **kw):
    total = len(fake.matching(_render(kw.get("year_min"), kw.get("year_max")))) if total is None else total
    return S.pubmed_esearch_by_years(fake, _render, total=total, cap=cap, this_year=2026, **kw)


# ── the slices, pure ─────────────────────────────────────────────────────────

def test_a_slice_is_the_same_query_restricted_to_its_years():
    q = "(h5n1) AND (poultry)"
    assert S.pubmed_years_term(q, None, None) == q
    assert S.pubmed_years_term(q, 1995, 2010) == f"({q}) AND (1995:2010[dp])"
    assert S.pubmed_years_term(q, 2023, None) == f"({q}) AND (2023:3000[dp])"
    # Open towards the past by exclusion: no floor year that would leave older records out.
    assert S.pubmed_years_term(q, None, 1962) == f"({q}) NOT (1963:3000[dp])"
    assert [S.pubmed_years_label(*r) for r in ((2021, 2021), (1995, 2010), (None, 1962), (2023, None))] \
        == ["2021", "1995-2010", "≤1962", "≥2023"]


def test_the_scenario_years_are_read_tolerantly():
    assert S.pubmed_year_bounds({"year_min": 2015, "year_max": "2024"}) == (2015, 2024)
    assert S.pubmed_year_bounds({"year_min": "2015"}) == (2015, None)
    assert S.pubmed_year_bounds({"year_max": 2020.0}) == (None, 2020)
    assert S.pubmed_year_bounds({"year_min": "", "year_max": None}) == (None, None)
    assert S.pubmed_year_bounds({"year_min": "soon"}) == (None, None)
    assert S.pubmed_year_bounds({"year_min": 2024, "year_max": 2015}) == (None, None)   # inverted
    assert S.pubmed_year_bounds(None) == (None, None)


def test_under_the_per_source_cap_every_slice_gives_all_it_can():
    assert S.pubmed_slice_quotas([9000, 8000, 12000], cap=100_000, total=28_000) == [9000, 8000, 10_000]


def test_over_the_per_source_cap_the_cap_is_shared_in_proportion():
    quotas = S.pubmed_slice_quotas([3000, 6000, 9000], cap=6000, total=18_000)
    assert sum(quotas) == 6000 and quotas == [1000, 2000, 3000]
    # A slice held to the ceiling gives its unused share to the others.
    quotas = S.pubmed_slice_quotas([30_000, 1000, 1000], cap=12_000, total=32_000)
    assert quotas[0] == 10_000 and sum(quotas) == 12_000 and all(q <= 1000 for q in quotas[1:])
    assert S.pubmed_slice_quotas([7, 3], cap=5, total=10) == S.pubmed_slice_quotas([7, 3], cap=5, total=10)


# ── the split, against the fake eutils ───────────────────────────────────────

def test_a_query_announcing_23000_is_read_whole_in_slices_under_the_ceiling():
    fake = FakeEutilsSearch(_records(GROWTH))
    out = _split(fake)
    assert out["complete"] and out["error"] is None and not out["stopped"]
    # Every record, each once, though about 900 of them answer two years.
    assert len(out["ids"]) == len(set(out["ids"])) == 23_000
    assert set(out["ids"]) == set(fake.records)
    # Every slice fetched, and none past what eutils serves.
    assert not fake.past_ceiling, fake.past_ceiling[:3]
    slices = out["slices"]
    assert len(slices) >= 3
    assert all(s["count"] <= 10_000 and s["kept"] == s["count"] and not s["capped"] for s in slices), slices
    # The slices cover every year once, oldest first, from an open past to an open future.
    assert slices[0]["from"] is None and slices[-1]["to"] is None
    for a, b in zip(slices, slices[1:]):
        assert a["to"] + 1 == b["from"], (a, b)
    # The sum of the slices counts the records that answer two years twice: it is NOT the
    # total, which stays the count of the whole query.
    assert sum(s["count"] for s in slices) > 23_000
    assert out["total"] == 23_000


def test_the_slices_are_the_same_query_with_their_years():
    fake = FakeEutilsSearch(_records(GROWTH))
    _split(fake)
    for term, _start, _n in fake.calls:
        assert term.startswith("((poultry worker) AND (h5n1)) ")
        assert len(_CLAUSE.findall(term)) == 1, term


def test_a_single_year_over_the_ceiling_keeps_its_most_relevant_and_says_so():
    fake = FakeEutilsSearch(_records({2019: 3000, 2020: 12_000, 2021: 4000}, straddle_every=0))
    out = _split(fake)
    capped = [s for s in out["slices"] if s["capped"]]
    assert [s["years"] for s in capped] == ["2020"], out["slices"]
    assert capped[0]["count"] == 12_000 and capped[0]["kept"] == 10_000
    assert len(out["ids"]) == 3000 + 10_000 + 4000
    # The 10,000 kept are that year's most relevant: the first 10,000 eutils ranks.
    kept_2020 = {p for p in out["ids"] if fake.records[p] == (2020,)}
    assert kept_2020 == set(fake.matching(_render(2020, 2020))[:10_000])
    assert not fake.past_ceiling


def test_over_the_per_source_cap_every_slice_keeps_its_share_by_relevance():
    fake = FakeEutilsSearch(_records(GROWTH))
    out = _split(fake, cap=15_000)
    assert 14_500 <= len(out["ids"]) <= 15_000
    # Every era is represented, not the newest only.
    years = [fake.records[p][0] for p in out["ids"]]
    assert min(years) < 1980 and max(years) == 2026
    # Within each slice, its most relevant records: the first ones eutils ranks.
    kept = set(out["ids"])
    for s in out["slices"]:
        assert set(fake.matching(_render(s["from"], s["to"]))[:s["kept"]]) <= kept, s


def test_the_scenario_years_bound_every_slice():
    fake = FakeEutilsSearch(_records(GROWTH))
    out = _split(fake, year_min=2005, year_max=2024)
    assert out["slices"][0]["from"] == 2005 and out["slices"][-1]["to"] == 2024
    for term, _s, _n in fake.calls:
        [(op, lo, hi)] = _CLAUSE.findall(term)
        assert op == "AND" and 2005 <= int(lo) <= int(hi) <= 2024, term
    assert set(out["ids"]) == set(fake.matching(_render(2005, 2024)))


def test_the_budget_stops_the_split_and_says_so():
    fake = FakeEutilsSearch(_records(GROWTH))
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] > 3
    out = _split(fake, stop=stop)
    assert out["stopped"] and not out["complete"]
    assert out["ids"] == [] and out["slices"] == []


def test_a_slice_that_fails_is_said_and_the_others_are_kept():
    fake = FakeEutilsSearch(_records(GROWTH))
    reads = {"n": 0}

    def flaky(term, retstart, retmax):
        if retmax:                                   # reading a slice, not counting it
            reads["n"] += 1
            if reads["n"] == 3:
                raise RuntimeError("HTTP 502")
        return fake(term, retstart, retmax)
    out = S.pubmed_esearch_by_years(flaky, _render, total=23_000, cap=100_000, this_year=2026)
    assert out["error"] and "502" in out["error"] and not out["complete"]
    assert 0 < len(out["ids"]) < 23_000


def test_a_split_that_cannot_be_planned_returns_nothing_and_says_why():
    def broken(term, retstart, retmax):
        raise RuntimeError("HTTP 500")
    out = S.pubmed_esearch_by_years(broken, _render, total=23_000, cap=100_000, this_year=2026)
    assert out["ids"] == [] and "500" in out["error"] and not out["complete"]


def test_an_esearch_error_in_the_body_is_an_error_not_an_empty_slice():
    def refuses(term, retstart, retmax):
        return {"ERROR": "Invalid query", "count": "0"}
    out = S.pubmed_esearch_by_years(refuses, _render, total=23_000, cap=100_000, this_year=2026)
    assert "Invalid query" in (out["error"] or "") and out["ids"] == []


# ── the corpus build, end to end against fake sources ────────────────────────
# The same mechanism at a hundredth of the scale: 230 records under a ceiling of 100. Through
# the real ingestion (about 8 ms a record here), 23,000 records would take minutes for each
# test; the split itself is exercised at full scale above.

SID = "usr-pubmedyears1"
HPAI = ('("poultry worker" OR veterinarians OR seroprevalence) AND '
        '("avian influenza" OR h5n1 OR "bird flu")')
_ABSTRACT = "Exposure of poultry workers to avian influenza viruses, long enough to keep. " * 2
SMALL_GROWTH = ({y: 3 for y in range(1990, 2000)} | {y: 5 for y in range(2000, 2010)}
                | {y: 8 for y in range(2010, 2020)} | {y: 10 for y in range(2020, 2027)})
SMALL_CEILING = 100
assert sum(SMALL_GROWTH.values()) == 230


class _Resp:
    def __init__(self, status=200, payload=None, body=""):
        self.status_code = status
        self._payload = payload if payload is not None else {}
        self.text = body
        self.content = body.encode()
        self.headers = {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeFederation:
    """PubMed from `FakeEutilsSearch` (esearch) and from memory (efetch); every other source
    answers that it found nothing."""

    def __init__(self, tag: str, records: dict[str, tuple], ceiling: int):
        self.tag = tag
        self.pubmed = FakeEutilsSearch(records, ceiling=ceiling)
        self.esearch: list[dict] = []
        self.efetched: list[str] = []

    def title(self, pmid: str) -> str:
        return f"Avian influenza in poultry workers, ysplit-{self.tag} record {pmid}"

    def get(self, url, params=None, **kw):
        url, params = str(url), dict(params or {})
        if "eutils" in url and "esearch" in url:
            self.esearch.append(params)
            res = self.pubmed(str(params.get("term", "")), int(params.get("retstart") or 0),
                              int(params.get("retmax") or 0))
            return _Resp(200, {"esearchresult": res})
        if "semanticscholar" in url:
            return _Resp(200, {"data": [], "total": 0, "token": None})
        if "europepmc" in url or "ebi.ac.uk" in url:
            return _Resp(200, {"resultList": {"result": []}, "hitCount": 0, "nextCursorMark": None})
        if "crossref" in url:
            return _Resp(200, {"message": {"items": [], "total-results": 0}})
        if "openalex" in url:
            return _Resp(200, {"results": [], "meta": {"count": 0, "next_cursor": None}})
        if "doaj.org" in url:
            return _Resp(200, {"results": [], "total": 0})
        if "clinicaltrials" in url:
            return _Resp(200, {"studies": [], "totalCount": 0})
        if "arxiv" in url:
            return _Resp(200, body="<feed xmlns='http://www.w3.org/2005/Atom'></feed>")
        if "openaire" in url:
            return _Resp(200, {"results": [], "header": {}})
        return _Resp(200, {})

    def post(self, url, data=None, json=None, **kw):              # noqa: A002 - le nom de requests
        if "efetch" in str(url):
            ids = str((data or {}).get("id", "")).split(",")
            self.efetched.extend(ids)
            body = "".join(
                f"<PubmedArticle><MedlineCitation><PMID>{p}</PMID><Article>"
                f"<Journal><JournalIssue><PubDate><Year>{self.pubmed.records[p][-1]}</Year></PubDate>"
                f"</JournalIssue><Title>Fake Journal</Title></Journal>"
                f"<ArticleTitle>{_sx.escape(self.title(p))}</ArticleTitle>"
                f"<Abstract><AbstractText>{_ABSTRACT}</AbstractText></Abstract>"
                f"</Article></MedlineCitation></PubmedArticle>"
                for p in ids if p in self.pubmed.records)
            return _Resp(200, body=f"<PubmedArticleSet>{body}</PubmedArticleSet>")
        return _Resp(200, {"results": [], "totalHits": 0})

    def terms(self) -> list[str]:
        return [str(p.get("term", "")) for p in self.esearch]


def _ingestion_works() -> bool:
    from api.sources import _ingest_doc_direct
    probe = "crossref:probe-yearsplit-0001"
    try:
        did, _new = _ingest_doc_direct(
            source="crossref", title="Ingestion probe for the PubMed year split test",
            abstract="A probe document, inserted then deleted by the test.", year=2021,
            url=None, external_id=probe, doi=None)
    except Exception:                                    # noqa: BLE001
        return False
    try:
        with main.engine.begin() as c:
            if did is not None:
                c.execute(text("DELETE FROM document_chunk WHERE document_id = :d"), {"d": did})
            c.execute(text("DELETE FROM literature_document WHERE external_id = :e"), {"e": probe})
    except Exception:                                    # noqa: BLE001
        pass
    return did is not None


@pytest.fixture
def federation(monkeypatch):
    if not _ingestion_works():
        pytest.skip("ingestion does not work on this schema: nothing to measure here")
    import requests as _real_requests

    from api import pipeline as P
    from api import relevance as R

    main._ensure_user_scenarios_table()
    tag = str(int(time.time() * 1000))[-8:]
    state: dict = {}

    def build(records: dict[str, tuple], ceiling: int = SMALL_CEILING):
        fake = _FakeFederation(tag, records, ceiling)
        monkeypatch.setattr(_real_requests, "get", fake.get)
        monkeypatch.setattr(_real_requests, "post", fake.post)
        monkeypatch.setattr(P, "_ncbi_get", lambda url, params, timeout=30, **k: fake.get(url, params))
        monkeypatch.setattr(P, "NCBI_ESEARCH_CEILING", ceiling)
        state["fake"] = fake
        return fake

    monkeypatch.setenv("CORE_API_KEY", "")
    monkeypatch.setattr(P, "_NCBI_MIN_INTERVAL", 0.0)
    monkeypatch.setattr(P, "FETCH_RETRY_BACKOFF_S", (0.0, 0.0, 0.0))
    # The production value: LIVE_MAX_PER_SOURCE set very high to lift the standard cap (the
    # populate brings `max_results` down to it).
    monkeypatch.setattr(P, "LIVE_MAX_PER_SOURCE", 100_000)
    monkeypatch.setattr(R, "_run_semantic_rerank_inline", lambda *a, **k: 0)
    monkeypatch.setattr(R, "_run_cross_encoder_rerank", lambda *a, **k: {})
    monkeypatch.setattr(P, "_run_clustering_background", lambda *a, **k: None)
    monkeypatch.setattr(P, "_precompute_user_kg", lambda *a, **k: None)

    def _clean():
        with main.engine.begin() as c:
            c.execute(text("DELETE FROM article_scenarios WHERE scenario_id = :s"), {"s": SID})
            c.execute(text("DELETE FROM user_scenarios WHERE id = :s"), {"s": SID})
            c.execute(text("DELETE FROM source_query_cache"))
            c.execute(text("DELETE FROM document_chunk WHERE document_id IN (SELECT id FROM "
                           "literature_document WHERE title LIKE :t)"), {"t": f"%ysplit-{tag}%"})
            c.execute(text("DELETE FROM literature_document WHERE title LIKE :t"), {"t": f"%ysplit-{tag}%"})

    _clean()
    with main.engine.begin() as c:
        c.execute(text("INSERT INTO user_scenarios (id, name, query, article_count) "
                       "VALUES (:s, 'pubmed years', :q, 0)"), {"s": SID, "q": HPAI})

    def run(filters=None, **kw):
        P._run_user_scenario_populate(SID, HPAI, filters or {}, max_results=kw.pop("max_results", 100_000),
                                      include_live=True, auto_pipeline=False,
                                      force_live=kw.pop("force_live", True),
                                      exhaustive=kw.pop("exhaustive", False), **kw)
        job = P._user_scenario_populate_jobs.get(SID) or {}
        return job.get("prisma_identification") or {}

    yield build, run
    _clean()


def test_the_corpus_build_reads_every_slice_and_counts_each_record_once(federation):
    build, run = federation
    fake = build(_records(SMALL_GROWTH, straddle_every=4))
    fig = run()
    terms = fake.terms()
    # The whole query first, then its slices: every one under the ceiling, none past it.
    assert "[dp]" not in terms[0]
    assert not fake.pubmed.past_ceiling, fake.pubmed.past_ceiling[:3]
    slices = (fig.get("year_slices") or {}).get("pubmed") or []
    assert len(slices) >= 3, fig.get("year_slices")
    # Every record of the query, each once, though about one in four answers two years.
    assert set(fake.efetched) == set(fake.pubmed.records)
    assert len(fake.efetched) == len(set(fake.efetched)) == 230
    assert (fig.get("records_by_source") or {}).get("pubmed") == 230, fig.get("records_by_source")
    # The PRISMA total is the whole query's, not the sum of the slices.
    assert (fig.get("source_totals") or {}).get("pubmed") == 230
    assert "pubmed" not in (fig.get("sources_capped") or [])
    assert (fig.get("source_outcomes") or {}).get("pubmed") == "ok"
    assert (fig.get("year_slices_capped") or {}) == {}
    # And the card reads it from the stored figures.
    from fastapi.testclient import TestClient
    with TestClient(main.app) as c:
        ident = c.get(f"/user-scenarios/{SID}/prisma").json()["identification"]
    assert ident["year_slices"] == {"pubmed": slices}


def test_a_query_under_the_ceiling_is_not_split(federation):
    build, run = federation
    fake = build(_records({2019: 30, 2020: 40}))
    fig = run()
    assert len(fake.esearch) == 1, fake.terms()
    assert "[dp]" not in fake.terms()[0]
    assert (fig.get("year_slices") or {}) == {}
    assert (fig.get("records_by_source") or {}).get("pubmed") == 70


def test_a_capped_search_is_not_split(federation):
    """Under a per-source cap the ceiling already covers, the most relevant are the first
    page of the whole query: nothing to split."""
    build, run = federation
    fake = build(_records(SMALL_GROWTH))
    fig = run(max_results=50)
    assert len(fake.esearch) == 1 and "[dp]" not in fake.terms()[0]
    assert (fig.get("records_by_source") or {}).get("pubmed") == 50
    assert "pubmed" in (fig.get("sources_capped") or [])


def test_every_slice_keeps_the_preprint_exclusion_and_long_terms_travel_by_post(federation):
    build, run = federation
    fake = build(_records(SMALL_GROWTH))
    run({"exclude_preprints": True})
    terms = fake.terms()
    assert len(terms) > 3 and all(t.endswith("NOT preprint[pt]") for t in terms), terms
    # Each slice goes through `_ncbi_get`, which posts what does not fit in a URL.
    from api import pipeline as P
    import inspect
    src = inspect.getsource(P._run_user_scenario_populate)
    body = src[src.index("def _fetch_pubmed"):src.index("def _fetch_openalex")]
    assert "pubmed_esearch_by_years(" in body and "_ncbi_get(" in body


def test_the_scenario_years_reach_pubmed_and_bound_the_slices(federation):
    build, run = federation
    fake = build(_records(SMALL_GROWTH))
    fig = run({"year_min": 2000, "year_max": 2019})
    terms = fake.terms()
    # The whole query carries the scenario's years, and its total is theirs, not all years'.
    assert terms[0].endswith("AND (2000:2019[dp])"), terms[0]
    in_years = fake.pubmed.matching(S.pubmed_years_term("x", 2000, 2019))
    assert SMALL_CEILING < len(in_years) < 230
    assert (fig.get("source_totals") or {}).get("pubmed") == len(in_years)
    assert len(terms) > 1
    for t in terms[1:]:
        [(op, lo, hi)] = _CLAUSE.findall(t)
        assert op == "AND" and 2000 <= int(lo) <= int(hi) <= 2019, t
    assert set(fake.efetched) == set(in_years)


def test_a_search_replayed_from_the_cache_still_says_it_was_split(federation):
    build, run = federation
    build(_records(SMALL_GROWTH))
    first = run()
    again = run(force_live=False)
    assert (again.get("source_outcomes") or {}).get("pubmed") == "cached"
    assert again.get("year_slices") == first.get("year_slices") != {}
    assert (again.get("source_totals") or {}).get("pubmed") == 230


# ── the living review, the other reader of PubMed's esearch ──────────────────

def test_the_living_review_reads_a_long_window_in_year_slices(monkeypatch):
    import living_review_scheduler as L
    fake = FakeEutilsSearch(_records(GROWTH))

    def eutils(endpoint, params, timeout=30):
        assert endpoint == "esearch.fcgi"
        res = fake(params["term"], int(params["retstart"]), int(params["retmax"]))
        return type("R", (), {"json": lambda self: {"esearchresult": res},
                              "raise_for_status": lambda self: None})()
    monkeypatch.setattr(L, "_eutils", eutils)
    s = L.search_pubmed("(stroke)", days=3650, max_results=30_000)
    assert s["total"] == 23_000 and s["fetched"] == 23_000 and set(s["ids"]) == set(fake.records)
    assert len(s["ids"]) == len(set(s["ids"]))
    assert s["truncated"] is False and s["error"] is None
    assert len(s["year_slices"]) >= 3 and not fake.past_ceiling
    assert all("[EDAT]" in t for t, _a, _b in fake.calls)


def test_the_living_review_under_its_guard_is_not_split(monkeypatch):
    import living_review_scheduler as L
    fake = FakeEutilsSearch(_records(GROWTH))

    def eutils(endpoint, params, timeout=30):
        res = fake(params["term"], int(params["retstart"]), int(params["retmax"]))
        return type("R", (), {"json": lambda self: {"esearchresult": res},
                              "raise_for_status": lambda self: None})()
    monkeypatch.setattr(L, "_eutils", eutils)
    s = L.search_pubmed("(stroke)", days=3650, max_results=5000)
    assert s["fetched"] == 5000 and s["total"] == 23_000 and s["truncated"] is True
    assert "year_slices" not in s
    assert not any("[dp]" in t for t, _a, _b in fake.calls)
