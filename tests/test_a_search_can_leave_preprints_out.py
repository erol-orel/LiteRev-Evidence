"""A search can leave preprints out, and bioRxiv and medRxiv are no longer searched directly.

Asked while preparing the HPAI review for the Italian team: "why keep bioRxiv and medRxiv
if they do not work or are retrievable via Europe PMC? And a toggle that lets users remove
preprints from the search would be good."

- The bioRxiv API has no keyword search, only date windows. The native fetcher scanned the
  last 45 days and filtered on the client: 0 records on HPAI_last, where the Europe PMC
  preprint facet, which indexes both servers, returned 457. It is out of the federation;
  the bioRxiv API is still used for full texts, by DOI.
- The preprint facet sorted by date: at the per-source cap it kept the most recent
  preprints, not the most relevant, unlike every other source that can rank.
- "Exclude preprints" is stored in the scenario's filters. The preprint servers (the Europe
  PMC preprint facet, arXiv) are not searched, PubMed and Europe PMC exclude preprints in
  the query itself, the local library matches none, and the preprints other databases
  return (OpenAlex, Semantic Scholar, Crossref...) leave the corpus before screening, on a
  PRISMA line of their own.
- What a preprint is lives in one SQL predicate. A preprint whose published version was
  merged onto the same row by the title dedup is a published article and stays.
"""
from __future__ import annotations

import pathlib
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")

from sqlalchemy import text  # noqa: E402

import main  # noqa: E402
from api.search import (_build_where, _prisma_identification_figures,  # noqa: E402
                        _reconcile_prisma_identification, excludes_preprints, preprint_sql)

ROOT = pathlib.Path(__file__).resolve().parent.parent


# ── the filter (pure) ────────────────────────────────────────────────────────

def test_the_filter_reads_true_only_when_asked():
    assert excludes_preprints({"exclude_preprints": True})
    assert excludes_preprints({"exclude_preprints": "true"})
    assert not excludes_preprints({"exclude_preprints": False})
    assert not excludes_preprints({"projectContext": "literev"})
    assert not excludes_preprints(None)


def test_the_local_library_matches_no_preprint_when_they_are_excluded():
    where, _params = _build_where({"exclude_preprints": True, "projectContext": "literev"})
    assert where.startswith(" AND ") and f"NOT {preprint_sql('d')}" in where
    # Without the option the filters are what they were: no clause at all.
    assert _build_where({"projectContext": "literev"}) == ("", {})
    assert _build_where({"exclude_preprints": False}) == ("", {})


def test_the_predicate_has_no_bind_parameter_or_percent_sign():
    """It is pasted into `text()` SQL: a `:name` would become a bind parameter, and a `%`
    would have to be doubled for the psycopg paramstyle."""
    import re
    sql = preprint_sql("d")
    assert not re.search(r"(?<![:\w]):\w", sql), sql
    assert "%" not in sql


# ── the PRISMA figures (pure) ────────────────────────────────────────────────

def test_the_preprints_removed_have_their_own_line_and_the_arithmetic_holds():
    f = _prisma_identification_figures(
        {"pubmed": 40, "semantic_scholar": 30}, 60, 0, 40,
        source_outcomes={"_fetch_pubmed": "ok", "_fetch_semantic_scholar": "ok"},
        removed_no_abstract=5, removed_preprints=7, removed_not_matching=8,
        preprints_excluded=True, preprints_excluded_at_source=["pubmed", "_fetch_europepmc"],
        preprint_sources_not_searched=["_fetch_preprints", "arxiv"])
    assert f["removed_preprints"] == 7
    assert f["preprints_excluded"] is True
    assert f["preprints_excluded_at_source"] == ["europepmc", "pubmed"]
    assert f["preprint_sources_not_searched"] == ["arxiv", "preprints"]
    removed = (f["removed_no_abstract"] + f["removed_preprints"] + f["removed_not_matching"]
               + f["removed_other_reasons"])
    assert removed == f["removed_before_screening"]
    assert f["records_identified"] - f["duplicates_removed"] - removed == f["records_screened"]


def test_the_preprint_line_is_bounded_by_what_is_left_to_explain():
    f = _prisma_identification_figures({"pubmed": 10}, 10, 0, 6, removed_no_abstract=3,
                                       removed_preprints=5, removed_not_matching=5)
    assert (f["removed_no_abstract"], f["removed_preprints"], f["removed_not_matching"],
            f["removed_other_reasons"]) == (3, 1, 0, 0)


def test_a_search_with_preprints_says_nothing_about_them():
    f = _prisma_identification_figures({"pubmed": 10}, 10, 0, 10)
    assert f["preprints_excluded"] is False and f["removed_preprints"] == 0
    assert f["preprints_excluded_at_source"] == [] and f["preprint_sources_not_searched"] == []


def test_the_reconciled_figures_count_the_preprints_among_the_removals():
    f = _prisma_identification_figures({"pubmed": 20}, 20, 0, 10, removed_no_abstract=4,
                                       removed_preprints=6, preprints_excluded=True)
    r = _reconcile_prisma_identification(f, 10)
    assert r["removed_preprints"] == 6
    assert r["removed_before_screening"] == 10


# ── the federation (source pinned) ───────────────────────────────────────────

def _populate_source() -> str:
    import inspect
    from api.pipeline import _run_user_scenario_populate
    return inspect.getsource(_run_user_scenario_populate)


def test_biorxiv_and_medrxiv_are_no_longer_searched_directly():
    src = _populate_source()
    assert "_fetch_biorxiv_medrxiv" not in src
    assert "api.biorxiv.org/details/{_server}/{_win}" not in src
    import api.sources as S
    for gone in ("_parse_biorxiv", "_live_fetch_biorxiv", "_live_fetch_medrxiv",
                 "_live_fetch_preprint_server"):
        assert not hasattr(S, gone), f"{gone} is dead code"
    # The full-text step still asks the bioRxiv API for a preprint, by DOI.
    from api.pipeline import _run_user_scenario_full_pipeline
    import inspect
    assert "api.biorxiv.org/details/{_srv}/{_doi}/na/json" in inspect.getsource(_run_user_scenario_full_pipeline)


def test_the_preprint_facet_ranks_by_relevance_and_says_its_total():
    src = _populate_source()
    body = src[src.index("def _fetch_preprints"):src.index("def _ingest_parsed")]
    assert "P_PDATE_D" not in body and '"sort"' not in body, "the facet still sorts by date"
    assert '_note_total("preprints"' in body


# ── the interface (source pinned) ────────────────────────────────────────────

def test_the_search_form_has_the_toggle_and_sends_it():
    app = (ROOT / "frontend" / "src" / "App.tsx").read_text(encoding="utf-8")
    assert "checked={excludePreprints}" in app
    # Le mode de la recherche part avec elle (exhaustive par défaut), et plus aucun
    # plafond : le serveur applique le sien.
    assert "populateUserScenario(sid, { includeLive, excludePreprints, exhaustive: exhaustiveSearch })" in app
    assert "exclude_preprints: true" in app
    assert '"bioRxiv", "medRxiv"' not in app
    api_ts = (ROOT / "frontend" / "src" / "lib" / "api.ts").read_text(encoding="utf-8")
    assert "params.set('exclude_preprints'" in api_ts
    page = (ROOT / "frontend" / "src" / "components" / "ScenarioDetailPage.tsx").read_text(encoding="utf-8")
    assert "ident.removed_preprints" in page and "ident.preprints_excluded" in page


# ── the predicate, evaluated by Postgres ─────────────────────────────────────

_ROWS = [
    # (label, source, source_type, doi, url, journal, is a preprint)
    ("medrxiv doi via openalex", "openalex", "article", "10.1101/2020.03.05.20031773", None, None, True),
    ("old biorxiv doi", "crossref", "article", "10.1101/123456", None, None, True),
    ("2026 medrxiv doi via openalex", "openalex", "article", "10.64898/2026.08.06.26359901", None, None, True),
    ("genome research", "pubmed", "article", "10.1101/gr.275365.121", None, "Genome research", False),
    ("cold spring harbor perspectives", "crossref", "article", "10.1101/cshperspect.a012345", None, None, False),
    ("research square", "openalex", "article", "10.21203/rs.3.rs-123/v1", None, None, True),
    ("ssrn", "crossref", "article", "10.2139/ssrn.3550000", None, None, True),
    ("europe pmc preprint without doi", "preprint", "preprint", None,
     "https://europepmc.org/article/PPR/PPR123", None, True),
    ("europe pmc preprint from an unlisted server", "preprint", "preprint", "10.32388/abc", None, None, True),
    ("arxiv without doi", "arxiv", "preprint", None, "http://arxiv.org/abs/2101.00001v1", None, True),
    ("arxiv with the published version's doi", "arxiv", "preprint", "10.1103/PhysRevLett.1",
     "http://arxiv.org/abs/2101.00001v1", None, False),
    ("a preprint indexed by pubmed", "pubmed", "article", "10.1101/2021.01.01.21249000", None,
     "medRxiv : the preprint server for health sciences", True),
    ("a preprint merged with its published version", "preprint", "preprint",
     "10.1101/2020.03.05.20031773", None, "Emerging infectious diseases", False),
    ("a journal article", "pubmed", "article", "10.3201/eid2601.190001", None, "Emerg Infect Dis", False),
    ("semantic scholar without doi", "semantic_scholar", "article", None,
     "https://www.semanticscholar.org/paper/x", None, False),
    ("core record of an arxiv pdf", "core", "article", None, "https://arxiv.org/pdf/2101.00001", None, True),
    ("ssrn by its journal name", "crossref", "article", None, None, "SSRN Electronic Journal", True),
    ("upper-case doi", "openalex", "article", "10.1101/2020.03.05.20031773".upper(), None, None, True),
    ("preprints.org", "openalex", "article", "10.20944/preprints202003.0001.v1", None, None, True),
    ("arxiv datacite doi", "openalex", "article", "10.48550/arXiv.2101.00001", None, None, True),
]


def _db_or_skip():
    try:
        with main.engine.connect() as c:
            c.execute(text("SELECT 1"))
    except Exception as e:                               # noqa: BLE001
        pytest.skip(f"no reachable Postgres ({type(e).__name__})")


def test_the_predicate_tells_preprints_from_published_articles():
    _db_or_skip()
    values = ", ".join(f"(:l{i}, :s{i}, :t{i}, :d{i}, :u{i}, :j{i})" for i in range(len(_ROWS)))
    params = {}
    for i, (label, s, t, d, u, j, _e) in enumerate(_ROWS):
        params.update({f"l{i}": label, f"s{i}": s, f"t{i}": t, f"d{i}": d, f"u{i}": u, f"j{i}": j})
    sql = (f"SELECT label, {preprint_sql('d')} AS pp FROM (VALUES {values}) "
           f"AS d(label, source, source_type, doi, url, journal)")
    with main.engine.connect() as c:
        got = {r.label: r.pp for r in c.execute(text(sql), params)}
    wrong = [(label, got[label], expected) for (label, *_r, expected) in _ROWS if got[label] != expected]
    assert not wrong, wrong


# ── the search, end to end against fake sources ──────────────────────────────

_ABSTRACT = "Exposure of poultry workers to avian influenza viruses, long enough to keep. " * 2
HPAI = ('("poultry worker" OR veterinarians OR seroprevalence) AND '
        '("avian influenza" OR h5n1 OR "bird flu")')
SID = "usr-nopreprints1"


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


class _FakeSources:
    """Every source answered from memory. Semantic Scholar returns one preprint (a medRxiv
    DOI) and one journal article; Europe PMC one article, and one preprint from its preprint
    facet. Titles carry `tag` so the test deletes exactly what it inserted."""

    def __init__(self, tag: str):
        self.tag = tag
        self.calls: list[tuple[str, dict]] = []

    def title(self, what: str) -> str:
        return f"Avian influenza in poultry worker cohorts, nopp-{self.tag} {what}"

    def get(self, url, params=None, **kw):
        url, params = str(url), dict(params or {})
        self.calls.append((url, params))
        if "semanticscholar" in url:
            data = [
                {"paperId": f"nopp{self.tag}pre", "title": self.title("s2 preprint"),
                 "abstract": _ABSTRACT, "year": 2021,
                 "externalIds": {"DOI": f"10.1101/2021.02.03.{self.tag}"}},
                {"paperId": f"nopp{self.tag}art", "title": self.title("s2 article"),
                 "abstract": _ABSTRACT, "year": 2021,
                 "externalIds": {"DOI": f"10.5555/nopp.{self.tag}.s2"}},
            ]
            return _Resp(200, {"data": data, "total": 2, "token": None})
        if "europepmc" in url or "ebi.ac.uk" in url:
            q = str(params.get("query", ""))
            if "AND (SRC:PPR)" in q:
                res = [{"id": f"PPR{self.tag}", "source": "PPR", "doi": f"10.1101/2022.05.06.{self.tag}",
                        "title": self.title("epmc preprint"), "abstractText": _ABSTRACT,
                        "pubYear": "2022"}]
            else:
                res = [{"pmcid": f"PMCnopp{self.tag}", "doi": f"10.5555/nopp.{self.tag}.ep",
                        "title": self.title("epmc article"), "abstractText": _ABSTRACT,
                        "pubYear": "2020"}]
            return _Resp(200, {"resultList": {"result": res}, "hitCount": 1, "nextCursorMark": None})
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
        if "eutils" in url:
            return _Resp(200, {"esearchresult": {"count": "0", "idlist": []}})
        return _Resp(200, {})

    def post(self, url, **kw):
        self.calls.append((str(url), dict(kw.get("params") or kw.get("data") or {})))
        return _Resp(200, {"results": [], "totalHits": 0})

    def urls(self, needle: str) -> list[tuple[str, dict]]:
        return [(u, p) for u, p in self.calls if needle in u]


def _ingestion_works() -> bool:
    from api.sources import _ingest_doc_direct
    probe = "crossref:probe-nopreprints-0001"
    try:
        did, _new = _ingest_doc_direct(
            source="crossref", title="Ingestion probe for the preprint exclusion test",
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
    from api import sources as S

    main._ensure_user_scenarios_table()
    tag = str(int(time.time() * 1000))[-8:]
    fake = _FakeSources(tag)
    monkeypatch.setattr(_real_requests, "get", fake.get)
    monkeypatch.setattr(_real_requests, "post", fake.post)
    monkeypatch.setattr(S, "_ncbi_get", lambda url, params, timeout=30: fake.get(url, params))
    monkeypatch.setattr(P, "_ncbi_get", lambda url, params, timeout=30: fake.get(url, params))
    monkeypatch.setenv("CORE_API_KEY", "")
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
                           "literature_document WHERE title LIKE :t)"), {"t": f"%nopp-{tag}%"})
            c.execute(text("DELETE FROM literature_document WHERE title LIKE :t"), {"t": f"%nopp-{tag}%"})

    _clean()
    with main.engine.begin() as c:
        c.execute(text("INSERT INTO user_scenarios (id, name, query, article_count) "
                       "VALUES (:s, 'no preprints', :q, 0)"), {"s": SID, "q": HPAI})

    def run(filters, **kw):
        P._run_user_scenario_populate(SID, HPAI, filters, max_results=50, include_live=kw.pop("live", True),
                                      auto_pipeline=False, force_live=True, **kw)
        return P._user_scenario_populate_jobs.get(SID) or {}

    def corpus_titles() -> set[str]:
        """The titles of THIS run's records in the corpus (the library may hold others)."""
        with main.engine.connect() as c:
            rows = c.execute(text("SELECT d.title FROM article_scenarios a JOIN literature_document d "
                                  "ON d.id = a.document_id WHERE a.scenario_id = :s AND d.title LIKE :t"),
                             {"s": SID, "t": f"%nopp-{tag}%"}).all()
        return {r[0] for r in rows}

    yield run, fake, corpus_titles
    _clean()


def test_a_search_without_preprints_leaves_them_out_and_says_so(federation):
    run, fake, corpus_titles = federation
    job = run({"exclude_preprints": True})
    fig = job.get("prisma_identification") or {}
    # The preprint servers are not searched at all.
    assert not fake.urls("arxiv"), fake.urls("arxiv")
    assert not [p for u, p in fake.urls("ebi.ac.uk") + fake.urls("europepmc")
                if "AND (SRC:PPR)" in str(p.get("query", ""))]
    # PubMed and Europe PMC exclude preprints in the query itself.
    epmc = [p for u, p in fake.urls("ebi.ac.uk") + fake.urls("europepmc")]
    assert epmc and all(str(p.get("query", "")).endswith("NOT SRC:PPR") for p in epmc), epmc
    esearch = [p for u, p in fake.urls("esearch.fcgi")]
    assert esearch and all(str(p.get("term", "")).endswith("NOT preprint[pt]") for p in esearch), esearch
    # The preprint Semantic Scholar returned is not in the corpus; the articles are.
    titles = corpus_titles()
    assert not any("s2 preprint" in t for t in titles), titles
    assert any("s2 article" in t for t in titles) and any("epmc article" in t for t in titles), titles
    # The card says it, on a line of its own, and the arithmetic holds.
    assert fig.get("preprints_excluded") is True
    assert fig.get("removed_preprints") == 1, fig
    assert fig.get("preprint_sources_not_searched") == ["arxiv", "preprints"]
    assert fig.get("preprints_excluded_at_source") == ["europepmc", "pubmed"]
    assert "preprints" not in (fig.get("source_outcomes") or {})
    assert "arxiv" not in (fig.get("source_outcomes") or {})
    assert (fig["records_identified"] - fig["duplicates_removed"] - fig["removed_before_screening"]
            == fig["records_screened"]), fig
    assert job.get("exclude_preprints") is True


def test_a_search_with_preprints_is_unchanged(federation):
    run, fake, corpus_titles = federation
    job = run({})
    fig = job.get("prisma_identification") or {}
    assert fake.urls("arxiv")
    facet = [p for u, p in fake.urls("ebi.ac.uk") + fake.urls("europepmc")
             if "AND (SRC:PPR)" in str(p.get("query", ""))]
    assert facet and all("sort" not in p for p in facet), facet       # by relevance
    assert not [p for u, p in fake.urls("esearch.fcgi") if "preprint[pt]" in str(p.get("term", ""))]
    titles = corpus_titles()
    assert any("s2 preprint" in t for t in titles), titles
    assert any("epmc preprint" in t for t in titles), titles
    assert fig.get("preprints_excluded") is False and fig.get("removed_preprints") == 0
    assert (fig.get("source_totals") or {}).get("preprints") == 1


def test_the_local_library_alone_leaves_preprints_out_too(federation):
    run, fake, corpus_titles = federation
    run({})                                    # fills the library with both kinds
    job = run({"exclude_preprints": True}, live=False)
    titles = corpus_titles()
    assert not any("preprint" in t for t in titles), titles
    assert any("article" in t for t in titles), titles
    assert job.get("exclude_preprints") is True


def test_the_option_is_stored_on_the_scenario_and_removed_cleanly(federation):
    from api.scenario_store import set_scenario_exclude_preprints
    set_scenario_exclude_preprints(SID, True)
    with main.engine.connect() as c:
        f = c.execute(text("SELECT filters FROM user_scenarios WHERE id = :s"), {"s": SID}).scalar()
    assert f.get("exclude_preprints") is True
    set_scenario_exclude_preprints(SID, False)
    with main.engine.connect() as c:
        f = c.execute(text("SELECT filters FROM user_scenarios WHERE id = :s"), {"s": SID}).scalar()
    assert "exclude_preprints" not in f, f   # absent, not false: the source cache key is unchanged
