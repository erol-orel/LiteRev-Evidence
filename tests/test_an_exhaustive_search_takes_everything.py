"""An exhaustive search takes every record the strategy matches, and the card says what it did.

Asked for the HPAI review, after measuring what the 2 000-per-source cap left out on the
production scenario HPAI_last: Europe PMC 1 996 of 13 810, OpenAlex 1 981 of 43 181,
Semantic Scholar 1 961 of 3 341 (and not the most relevant ones: its bulk endpoint returns
Boolean matches in record-ID order), DOAJ 428 of the 514 the whole strategy matches,
because DOAJ does not page through a long query and received a shortened one.

So a search mode, stored on the scenario:

- every database that applies the Boolean strategy returns everything (a guard rail, not a
  sample, at EXHAUSTIVE_MAX_PER_SOURCE), with a longer federation budget;
- Crossref, which ranks keywords instead of applying the strategy, keeps the standard cap;
- OpenAlex is searched in titles and abstracts: its default search also reads full texts;
- DOAJ gets the whole strategy split into queries short enough to page, whose union is the
  strategy (the distributive law, exclusions kept in every part).

And the PRISMA card: the cache of source answers kept the records but not the total each
source had announced, so a search replayed from the cache no longer knew a source had been
capped. HPAI_last's card named no capped source while four were.
"""
from __future__ import annotations

import collections
import random
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")

from sqlalchemy import text  # noqa: E402

import main  # noqa: E402
from api.search import (DOAJ_MAX_QUERY_CHARS, _clean_boolean, _parse_boolean_ast,  # noqa: E402
                        _prisma_identification_figures, _split_boolean, _tokenize_boolean)

EXPOSED = ["occupational exposure", "occupational diseases", "occupational", "veterinarians",
           "veterinarian", "veterinary staff", "veterinary worker", "farmers", "farm worker",
           "farmworker", "poultry worker", "poultry farmer", "dairy worker", "cullers",
           "culling personnel", "abattoirs", "slaughterhouse worker", "live bird market",
           "animal handler", "first responder", "emergency responders", "health personnel",
           "healthcare worker", "laboratory worker", "exposed worker", "exposed person",
           "human exposure", "seroprevalence", "seropositive", "seropositivity", "serosurvey",
           "personal protective equipment", "risk perception"]
VIRUS = ["influenza in birds", "avian influenza", "avian flu", "bird flu", "hpai", "h5n1",
         "h5n5", "h5n6", "h5n8", "h5nx", "h7n9", "h9n2", "h3n8"]


def _block(terms):
    return "(" + " OR ".join(f'"{t}"' if " " in t else t for t in terms) + ")"


HPAI = f"{_block(EXPOSED)} AND {_block(VIRUS)}"


# ── the split (pure) ─────────────────────────────────────────────────────────

def _norm(term: str) -> str:
    return " ".join(str(term).rstrip("*").lower().split())


def _eval(ast, doc: set) -> bool:
    kind = ast[0]
    if kind == "term":
        return _norm(ast[1]) in doc
    if kind == "not":
        return not _eval(ast[1], doc)
    if kind == "and":
        return all(_eval(c, doc) for c in ast[1])
    return any(_eval(c, doc) for c in ast[1])


def _terms(ast) -> list[str]:
    if ast[0] == "term":
        return [_norm(ast[1])]
    if ast[0] == "not":
        return _terms(ast[1])
    return [t for c in ast[1] for t in _terms(c)]


def _ast(query: str):
    return _parse_boolean_ast(_tokenize_boolean(query))


def test_the_hpai_strategy_does_not_fit_doaj_and_splits_into_parts_that_do():
    assert len(_clean_boolean(HPAI)) > DOAJ_MAX_QUERY_CHARS
    parts = _split_boolean(HPAI, DOAJ_MAX_QUERY_CHARS)
    assert len(parts) >= 2
    assert all(len(p) <= DOAJ_MAX_QUERY_CHARS for p in parts), [len(p) for p in parts]
    # No synonym is dropped: every term of the strategy is in some part.
    found = {t for p in parts for t in _terms(_ast(p))}
    assert set(_terms(_ast(HPAI))) <= found


def test_the_union_of_the_parts_is_the_strategy():
    """The distributive law, checked on documents: a record matches the strategy if and
    only if it matches one of the parts."""
    original = _ast(HPAI)
    parts = [_ast(p) for p in _split_boolean(HPAI, DOAJ_MAX_QUERY_CHARS)]
    vocabulary = sorted(set(_terms(original)))
    rng = random.Random(20261010)
    docs = [set(rng.sample(vocabulary, rng.randint(0, 6))) for _ in range(3000)]
    # and every single pairing of an exposure term with a virus term
    docs += [{_norm(a), _norm(b)} for a in EXPOSED for b in VIRUS]
    for doc in docs:
        assert _eval(original, doc) == any(_eval(p, doc) for p in parts), sorted(doc)


def test_an_exclusion_stays_attached_to_every_part():
    q = "(alpha OR beta OR gamma OR delta OR epsilon) AND virus NOT ferret"
    parts = _split_boolean(q, 40)
    assert len(parts) >= 2
    assert all("NOT ferret" in p for p in parts), parts
    original = _ast(q)
    for doc in ({"alpha", "virus"}, {"delta", "virus", "ferret"}, {"epsilon"}, {"beta", "virus"}):
        assert _eval(original, doc) == any(_eval(_ast(p), doc) for p in parts)


def test_a_strategy_that_fits_is_one_part_and_the_clean_form():
    q = '("poultry worker" OR cullers) AND (h5n1 OR "avian influenza")'
    assert _split_boolean(q, DOAJ_MAX_QUERY_CHARS) == [_clean_boolean(q)]


def test_no_split_when_a_single_term_cannot_fit_or_too_many_parts_are_needed():
    assert _split_boolean('"a very long phrase that cannot fit" AND b', 10) == []
    assert _split_boolean(HPAI, 60, max_parts=4) == []
    assert _split_boolean("", 700) == []


# ── the figures (pure) ───────────────────────────────────────────────────────

def _figures(**kw):
    base = dict(records_by_source={"europepmc": 13810, "crossref": 2000, "openalex": 3100},
                unique_records=15000, duplicate_rows_removed=0, corpus_total=14000,
                source_outcomes={"_fetch_europepmc": "ok", "_fetch_crossref": "ok",
                                 "_fetch_openalex": "ok"},
                source_totals={"europepmc": 13810, "crossref": 2634278, "openalex": 3100})
    base.update(kw)
    return _prisma_identification_figures(**base)


def test_in_an_exhaustive_search_only_a_source_over_its_own_cap_is_capped():
    f = _figures(per_source_cap=20000, search_mode="exhaustive", source_caps={"crossref": 2000},
                 title_abstract_sources=["_fetch_openalex"], unranked_sources=["semantic_scholar"],
                 split_queries={"doaj": ["(a) AND (x)", "(b) AND (x)"]})
    assert f["sources_capped"] == ["crossref"]
    assert f["search_mode"] == "exhaustive"
    assert f["source_caps"] == {"crossref": 2000}
    assert f["title_abstract_sources"] == ["openalex"]
    assert f["unranked_sources"] == ["semantic_scholar"]
    assert f["split_queries"] == {"doaj": ["(a) AND (x)", "(b) AND (x)"]}


def test_a_standard_search_is_described_as_before():
    f = _figures(per_source_cap=2000)
    assert f["search_mode"] == "standard"
    assert f["sources_capped"] == ["crossref", "europepmc", "openalex"]
    assert f["source_caps"] == {} and f["split_queries"] == {}
    assert f["title_abstract_sources"] == [] and f["unranked_sources"] == []


# ── the federation (database, fake APIs) ─────────────────────────────────────

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


_ABSTRACT = "Exposure of poultry workers to avian influenza viruses, long enough to keep. " * 2


def _ingestion_works() -> bool:
    from api.sources import _ingest_doc_direct
    probe = "crossref:probe-exhaustive-0001"
    try:
        did, _new = _ingest_doc_direct(
            source="crossref", title="Ingestion probe for the exhaustive search test",
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


class _FakeSources:
    """Every source of the federation, answered from memory. Records carry `tag` in their
    titles so the test can delete exactly what it inserted."""

    def __init__(self, tag: str, epmc_total: int = 0, epmc_rows: int = 0):
        self.tag = tag
        self.calls: list[tuple[str, dict]] = []
        self.epmc_total, self.epmc_rows = epmc_total, epmc_rows

    def _title(self, what: str) -> str:
        return f"Avian influenza and exposed workers, exh-{self.tag} {what}"

    def get(self, url, params=None, **kw):
        url, params = str(url), dict(params or {})
        self.calls.append((url, params))
        if "semanticscholar" in url and "bulk" in url:
            page = {None: 1, "t2": 2, "t3": 3}.get(params.get("token"), 4)
            data = [{"paperId": f"exh{self.tag}p{page}i{i}", "title": self._title(f"s2 {page}.{i}"),
                     "abstract": _ABSTRACT, "year": 2022, "externalIds": {}} for i in range(100)]
            return _Resp(200, {"data": data if page <= 3 else [], "total": 300,
                               "token": {1: "t2", 2: "t3"}.get(page)})
        if "crossref" in url:
            rows = int(params.get("rows") or 20)
            off = int(params.get("offset") or 0)
            items = [{"DOI": f"10.5555/exh.{self.tag}.cr.{off + i}", "title": [self._title(f"cr {off + i}")],
                      "abstract": _ABSTRACT, "created": {"date-parts": [[2023]]}} for i in range(rows)]
            return _Resp(200, {"message": {"items": items, "total-results": 1_000_000}})
        if "openalex" in url:
            # Titre et résumé (filtre, curseur) : les notices 0 et 1, 2 au total. Recherche
            # par défaut (texte intégral, par pages) : 1 et 2, 43 181 au total. La notice 1
            # est rendue par les deux passages.
            def _work(i):
                return {"id": f"https://openalex.org/Wexh{self.tag}{i}", "title": self._title(f"oa {i}"),
                        "abstract_inverted_index": {"exposure": [0], "of": [1], "poultry": [2],
                                                    "workers": [3], "to": [4], "influenza": [5]},
                        "publication_year": 2021, "doi": None}
            if "filter" in params:
                first = params.get("cursor") == "*"
                return _Resp(200, {"results": [_work(0), _work(1)] if first else [],
                                   "meta": {"count": 2, "next_cursor": "c2" if first else None}})
            first = str(params.get("page", "1")) == "1" and params.get("cursor") in (None, "*")
            return _Resp(200, {"results": [_work(1), _work(2)] if first else [],
                               "meta": {"count": 43181, "next_cursor": None}})
        if "doaj.org" in url:
            if str(params.get("page", 1)) != "1":
                return _Resp(200, {"results": []})
            part = str(abs(hash(url)) % 10_000)
            results = [{"id": f"exh{self.tag}-{part}-{i}",
                        "bibjson": {"title": self._title(f"doaj {part}.{i}"), "abstract": _ABSTRACT,
                                    "year": "2021"}} for i in range(3)]
            results.append({"id": f"exh{self.tag}-shared",
                            "bibjson": {"title": self._title("doaj shared"), "abstract": _ABSTRACT,
                                        "year": "2021"}})
            return _Resp(200, {"results": results, "total": 4})
        if "europepmc" in url or "ebi.ac.uk" in url:
            if "SRC:PPR" in str(params.get("query", "")) or not self.epmc_rows:
                return _Resp(200, {"resultList": {"result": []}, "hitCount": 0})
            res = [{"pmcid": f"PMCexh{self.tag}{i}", "doi": f"10.5555/exh.{self.tag}.ep.{i}",
                    "title": self._title(f"ep {i}"), "abstractText": _ABSTRACT, "pubYear": "2020"}
                   for i in range(self.epmc_rows)]
            return _Resp(200, {"resultList": {"result": res}, "hitCount": self.epmc_total,
                               "nextCursorMark": None})
        if "clinicaltrials" in url:
            return _Resp(200, {"studies": []})
        if "arxiv" in url:
            return _Resp(200, body="<feed xmlns='http://www.w3.org/2005/Atom'></feed>")
        if "openaire" in url:
            return _Resp(200, {"results": [], "header": {}})
        if "biorxiv" in url:
            return _Resp(200, {"collection": [], "messages": [{"total": 0}]})
        if "core.ac.uk" in url:
            return _Resp(200, {"results": [], "totalHits": 0})
        if "eutils" in url:
            return _Resp(200, {"esearchresult": {"count": "0", "idlist": []}})
        return _Resp(200, {})

    def post(self, url, **kw):
        self.calls.append((str(url), dict(kw.get("params") or {})))
        if "core.ac.uk" in str(url):
            return _Resp(200, {"results": [], "totalHits": 0})
        return _Resp(200, {})

    def urls(self, needle: str) -> list[tuple[str, dict]]:
        return [(u, p) for u, p in self.calls if needle in u]


SID = "usr-exhaustive01"


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
    monkeypatch.setattr(P, "EXHAUSTIVE_MAX_PER_SOURCE", 500)
    monkeypatch.setenv("CORE_API_KEY", "")
    monkeypatch.setattr(R, "_run_semantic_rerank_inline", lambda *a, **k: 0)
    monkeypatch.setattr(R, "_run_cross_encoder_rerank", lambda *a, **k: {})
    monkeypatch.setattr(P, "_run_clustering_background", lambda *a, **k: None)
    monkeypatch.setattr(P, "_precompute_user_kg", lambda *a, **k: None)
    with main.engine.begin() as c:
        c.execute(text("DELETE FROM article_scenarios WHERE scenario_id = :s"), {"s": SID})
        c.execute(text("DELETE FROM user_scenarios WHERE id = :s"), {"s": SID})
        c.execute(text("DELETE FROM source_query_cache"))
        c.execute(text("INSERT INTO user_scenarios (id, name, query, article_count) "
                       "VALUES (:s, 'exhaustive', :q, 0)"), {"s": SID, "q": HPAI})

    def run(**kw):
        P._run_user_scenario_populate(SID, HPAI, {}, max_results=kw.pop("max_results", 150),
                                      include_live=True, auto_pipeline=False, **kw)
        return P._user_scenario_populate_jobs.get(SID) or {}

    yield run, fake
    with main.engine.begin() as c:
        c.execute(text("DELETE FROM article_scenarios WHERE scenario_id = :s"), {"s": SID})
        c.execute(text("DELETE FROM user_scenarios WHERE id = :s"), {"s": SID})
        c.execute(text("DELETE FROM source_query_cache"))
        c.execute(text("DELETE FROM document_chunk WHERE document_id IN (SELECT id FROM literature_document "
                       "WHERE title LIKE :t)"), {"t": f"%exh-{tag}%"})
        c.execute(text("DELETE FROM literature_document WHERE title LIKE :t"), {"t": f"%exh-{tag}%"})


def test_an_exhaustive_search_takes_every_record_the_strategy_matches(federation):
    run, fake = federation
    job = run(force_live=True, exhaustive=True)
    fig = job.get("prisma_identification") or {}
    by = fig.get("records_by_source") or {}
    assert job.get("search_mode") == "exhaustive" or fig.get("search_mode") == "exhaustive"
    # Semantic Scholar: all three bulk pages, where the standard cap of 150 stops at two.
    assert by.get("semantic_scholar") == 300, by
    # Crossref ranks keywords: it keeps the standard cap.
    assert by.get("crossref") == 150, by
    assert all(int(p.get("rows") or 0) <= 150 for _, p in fake.urls("crossref"))
    # OpenAlex: every title/abstract match (cursor paging), THEN the standard pass, so an
    # exhaustive search never finds less than a standard one; a record both return counts once.
    oa = fake.urls("openalex")
    assert oa and oa[0][1].get("filter", "").startswith("title_and_abstract.search:"), oa[0]
    assert oa[0][1].get("cursor") == "*" and "page" not in oa[0][1] and "search" not in oa[0][1]
    assert any(p.get("search") and p.get("page") == 1 and p.get("sort") == "relevance_score:desc"
               for _, p in oa), oa
    assert by.get("openalex") == 3, by
    # Its total is the exhaustive pass's: the standard pass is an addition, not the search.
    assert (fig.get("source_totals") or {}).get("openalex") == 2
    assert "openalex" not in (fig.get("sources_capped") or [])
    # DOAJ: the whole strategy in several queries, a record found by two of them counted once.
    doaj_queries = {u for u, p in fake.urls("doaj.org")}
    assert len(doaj_queries) >= 2, doaj_queries
    assert by.get("doaj") == 3 * len(doaj_queries) + 1, by
    assert fig.get("search_mode") == "exhaustive"
    assert fig.get("title_abstract_sources") == ["openalex"]
    assert fig.get("unranked_sources") == ["semantic_scholar"]
    assert len((fig.get("split_queries") or {}).get("doaj") or []) == len(doaj_queries)
    assert (fig.get("source_caps") or {}).get("crossref") == 150
    assert "doaj" not in (fig.get("keyword_fallback_sources") or [])


def test_a_standard_search_is_unchanged(federation):
    run, fake = federation
    job = run(force_live=True, exhaustive=False)
    fig = job.get("prisma_identification") or {}
    by = fig.get("records_by_source") or {}
    assert by.get("semantic_scholar") == 200, by           # stops past the cap, as before
    oa = fake.urls("openalex")
    assert oa and "search" in oa[0][1] and "filter" not in oa[0][1] and oa[0][1].get("page") == 1
    assert not any("filter" in p for _, p in oa)
    assert by.get("openalex") == 2, by
    assert "openalex" in (fig.get("sources_capped") or [])  # 43 181 announced, 150 kept
    assert len({u for u, p in fake.urls("doaj.org")}) == 1   # the shortened query, as before
    assert "doaj" in (fig.get("keyword_fallback_sources") or [])
    assert fig.get("search_mode") == "standard"
    # The bulk endpoint's order is said even in a standard search: capped, it keeps an
    # arbitrary subset.
    assert fig.get("unranked_sources") == ["semantic_scholar"]


def test_the_mode_is_read_from_the_scenario(federation):
    from api.scenario_store import scenario_search_mode, set_scenario_search_mode
    run, fake = federation
    assert scenario_search_mode(SID) == "standard"
    set_scenario_search_mode(SID, "exhaustive")
    assert scenario_search_mode(SID) == "exhaustive"
    job = run(force_live=True)                              # no explicit mode
    assert (job.get("prisma_identification") or {}).get("search_mode") == "exhaustive"
    with pytest.raises(ValueError):
        set_scenario_search_mode(SID, "everything")


def test_a_cached_replay_still_says_a_source_was_capped(federation):
    run, fake = federation
    fake.epmc_total, fake.epmc_rows = 13810, 60
    first = run(force_live=True, exhaustive=False, max_results=50)
    fig1 = first.get("prisma_identification") or {}
    assert "europepmc" in (fig1.get("sources_capped") or []), fig1.get("sources_capped")
    calls_before = len(fake.urls("europepmc")) + len(fake.urls("ebi.ac.uk"))
    second = run(exhaustive=False, max_results=50)        # same query: replayed from the cache
    fig2 = second.get("prisma_identification") or {}
    assert (fig2.get("source_outcomes") or {}).get("europepmc") == "cached"
    assert len(fake.urls("europepmc")) + len(fake.urls("ebi.ac.uk")) == calls_before
    assert (fig2.get("source_totals") or {}).get("europepmc") == 13810
    assert "europepmc" in (fig2.get("sources_capped") or []), fig2.get("sources_capped")


def test_the_search_and_the_pipeline_score_with_the_same_query_vector():
    """The full pipeline's scoring step had its own copy of the scoring SQL, and embedded
    `query[:2000]`, the RAW query with its PubMed tags, where the search embeds
    `embedding_text_for_query` (the whole query, tags stripped). One scenario had two query
    vectors depending on the last path taken: on HPAI_last every one of the 6 453 articles
    both paths scored moved (-0.024 on average), and a threshold set to keep 1 000 articles
    kept 288 after a re-run. One scoring function now serves both."""
    import inspect

    from api import pipeline as P
    from api import relevance as R

    src = inspect.getsource(P._run_user_scenario_full_pipeline)
    assert "_run_semantic_rerank_inline(scenario_id, query)" in src
    assert "input=query" not in src
    assert "embedding_text_for_query(query)" in inspect.getsource(R._run_semantic_rerank_inline)


def test_an_exhaustive_search_never_replays_a_capped_answer():
    from api.pipeline import _source_query_hash
    standard = _source_query_hash(HPAI, {}, 2000)
    assert standard == _source_query_hash(HPAI, {}, 2000, mode="standard")
    assert standard != _source_query_hash(HPAI, {}, 2000, mode="exhaustive")


def test_counts_are_not_inflated_by_the_fake(federation):
    """Guard on the harness itself: every DOAJ part answers its own three records plus the
    shared one, so a regression in the dedup shows up as a count, not as a silent pass."""
    run, fake = federation
    run(force_live=True, exhaustive=True)
    parts = collections.Counter(u for u, p in fake.urls("doaj.org"))
    assert all(n == 1 for n in parts.values()), parts
