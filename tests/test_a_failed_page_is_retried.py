"""Une page qui échoue se réessaie, un nombre borné de fois, avant que la source ne finisse
en échec.

Neuf boucles de pagination sur onze (OpenAlex, Crossref, Europe PMC et sa facette
préprints, DOAJ, ClinicalTrials.gov, arXiv, OpenAIRE, et l'esearch de PubMed) s'arrêtaient
au premier délai dépassé, au premier 429 ou 5xx : la source finissait en `error` avec les
pages d'avant, et un lot efetch de PubMed (200 notices) en échec était sauté. Le helper
commun réessaie trois fois (2 s, 4 s, 8 s ; `Retry-After` respecté jusqu'à un plafond),
jamais au-delà du budget de la fédération, et la source reste en `error` après la dernière
tentative. Ces tests pilotent la fédération réelle contre des API simulées.
"""
import ast
import collections
import inspect
import textwrap
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")

from sqlalchemy import text  # noqa: E402

import main  # noqa: E402
from search_tables import ensure_search_tables  # noqa: E402

QUERY ='("poultry worker" OR veterinarian) AND ("avian influenza" OR h5n1)'
SID = "usr-retrypages001"
_ABSTRACT = "Seroprevalence of H5N1 antibodies among exposed poultry workers, long enough to keep."
_AUTHORS = [("Smith", "Jane"), ("Doe", "John"), ("Roe", "Ann"), ("Poe", "Edgar"), ("Lee", "Kim"),
            ("Ng", "Andrew"), ("Wu", "Lan"), ("Li", "Mei")]


def _pubmed_xml(tag: str) -> str:
    authors = "".join(f"<Author><LastName>{last}</LastName><ForeName>{first}</ForeName></Author>"
                      for last, first in _AUTHORS)
    arts = []
    for i in range(3):
        arts.append(
            f"<PubmedArticle><MedlineCitation><PMID>9{tag}{i}</PMID><Article>"
            f"<Journal><JournalIssue><PubDate><MedlineDate>1998 Spring</MedlineDate></PubDate></JournalIssue>"
            f"<Title>J Occup Med</Title></Journal>"
            f"<ArticleTitle>Avian influenza among poultry workers, retry-{tag} {i}</ArticleTitle>"
            f"<Abstract><AbstractText>{_ABSTRACT}</AbstractText></Abstract>"
            f"<AuthorList>{authors}</AuthorList></Article></MedlineCitation></PubmedArticle>")
    return "<PubmedArticleSet>" + "".join(arts) + "</PubmedArticleSet>"


class _Resp:
    def __init__(self, status=200, payload=None, body="", headers=None):
        self.status_code = status
        self._payload = payload if payload is not None else {}
        self.text = body
        self.content = body.encode()
        self.headers = headers or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FlakySources:
    """Chaque source répond « rien trouvé », sauf ce qu'un test règle : DOAJ sert trois
    pages (la deuxième en 503 une fois), OpenAlex répond 503 à chaque fois, et l'efetch de
    PubMed échoue une fois avant de rendre ses trois notices."""

    def __init__(self, tag: str):
        self.tag = tag
        self.calls: collections.Counter = collections.Counter()
        self.doaj_fail_left = 1
        self.efetch_fail_left = 1
        self.openalex_status = 503

    def get(self, url, params=None, **kw):
        url, params = str(url), dict(params or {})
        if "doaj.org" in url:
            page = int(params.get("page") or 1)
            self.calls[("doaj", page)] += 1
            if page == 2 and self.doaj_fail_left:
                self.doaj_fail_left -= 1
                return _Resp(503, body="Service Unavailable")
            n = 100 if page < 3 else 50
            return _Resp(200, {"total": 250, "results": [
                {"id": f"retry{self.tag}-{page}-{i}", "bibjson": {
                    "title": f"Avian influenza among poultry workers, retry-{self.tag} doaj {page}.{i}",
                    "abstract": _ABSTRACT, "year": "2021"}} for i in range(n)]})
        if "openalex" in url:
            self.calls["openalex"] += 1
            return _Resp(self.openalex_status, body="Service Unavailable")
        if "eutils" in url and "esearch" in url:
            self.calls["esearch"] += 1
            return _Resp(200, {"esearchresult": {"count": "3",
                                                 "idlist": [f"9{self.tag}{i}" for i in range(3)]}})
        if "crossref" in url:
            return _Resp(200, {"message": {"items": [], "total-results": 0}})
        if "europepmc" in url or "ebi.ac.uk" in url:
            return _Resp(200, {"resultList": {"result": []}, "hitCount": 0})
        if "clinicaltrials" in url:
            return _Resp(200, {"studies": [], "totalCount": 0})
        if "arxiv" in url:
            return _Resp(200, body="<feed xmlns='http://www.w3.org/2005/Atom'></feed>")
        if "openaire" in url:
            return _Resp(200, {"results": [], "header": {}})
        if "semanticscholar" in url:
            return _Resp(200, {"data": [], "total": 0})
        return _Resp(200, {})

    def post(self, url, data=None, json=None, **kw):              # noqa: A002 - le nom de requests
        url = str(url)
        if "efetch" in url:
            self.calls["efetch"] += 1
            if self.efetch_fail_left:
                self.efetch_fail_left -= 1
                return _Resp(503, body="Service Unavailable")
            return _Resp(200, body=_pubmed_xml(self.tag))
        return _Resp(200, {"results": [], "totalHits": 0})


def _ingestion_works() -> bool:
    from api.sources import _ingest_doc_direct
    probe = "crossref:probe-retry-0001"
    try:
        did, _new = _ingest_doc_direct(
            source="crossref", title="Ingestion probe for the retry test, long enough",
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
def federation(db_conn, monkeypatch):
    import requests as _real_requests

    from api import pipeline as P
    from api import relevance as R

    created_chunk_table = ensure_search_tables(db_conn.cursor())
    tag = str(int(time.time() * 1000))[-7:]
    fake = _FlakySources(tag)
    monkeypatch.setattr(_real_requests, "get", fake.get)
    monkeypatch.setattr(_real_requests, "post", fake.post)
    monkeypatch.setattr(P, "_ncbi_get", lambda url, params, timeout=30, use_key=True: fake.get(url, params))
    monkeypatch.setenv("CORE_API_KEY", "")
    monkeypatch.setattr(P, "FETCH_RETRY_BACKOFF_S", (0.0, 0.0, 0.0))
    monkeypatch.setattr(R, "_run_semantic_rerank_inline", lambda *a, **k: 0)
    monkeypatch.setattr(R, "_run_cross_encoder_rerank", lambda *a, **k: {})
    monkeypatch.setattr(P, "_run_clustering_background", lambda *a, **k: None)
    monkeypatch.setattr(P, "_precompute_user_kg", lambda *a, **k: None)

    def _clean():
        with main.engine.begin() as c:
            c.execute(text("DELETE FROM article_scenarios WHERE scenario_id = :s"), {"s": SID})
            c.execute(text("DELETE FROM user_scenarios WHERE id = :s"), {"s": SID})
            c.execute(text("DELETE FROM source_query_cache"))
        try:
            with main.engine.begin() as c:
                c.execute(text("DELETE FROM document_chunk WHERE document_id IN (SELECT id FROM "
                               "literature_document WHERE title LIKE :t)"), {"t": f"%retry-{tag}%"})
                c.execute(text("DELETE FROM literature_document WHERE title LIKE :t"), {"t": f"%retry-{tag}%"})
        except Exception:                                # noqa: BLE001 - schéma sans document_chunk
            pass

    _clean()
    with main.engine.begin() as c:
        c.execute(text("INSERT INTO user_scenarios (id, name, query, article_count) "
                       "VALUES (:s, 'retries', :q, 0)"), {"s": SID, "q": QUERY})

    def run():
        P._run_user_scenario_populate(SID, QUERY, {}, max_results=300, include_live=True,
                                      auto_pipeline=False, force_live=True)
        return P._user_scenario_populate_jobs.get(SID) or {}

    yield run, fake, tag
    _clean()
    if created_chunk_table:
        with db_conn.cursor() as cur:
            cur.execute("DROP TABLE document_chunk")


def test_a_503_then_a_200_keeps_every_page(federation):
    run, fake, _tag = federation
    job = run()
    assert fake.calls[("doaj", 1)] == 1 and fake.calls[("doaj", 2)] == 2, dict(fake.calls)
    assert fake.calls[("doaj", 3)] == 1, "la pagination doit reprendre après la page réessayée"
    assert (job.get("source_outcomes") or {}).get("doaj") != "error", job.get("source_outcomes")


def test_three_failures_end_in_error_and_the_reason_is_kept(federation):
    from api import pipeline as P
    run, fake, _tag = federation
    job = run()
    assert fake.calls["openalex"] == P.FETCH_RETRY_ATTEMPTS == 3, dict(fake.calls)
    assert (job.get("source_outcomes") or {}).get("openalex") == "error"
    reasons = (job.get("prisma_identification") or {}).get("source_error_reasons") or {}
    assert "503" in reasons.get("openalex", ""), reasons


def test_a_pubmed_batch_failing_once_is_retried_not_skipped(federation):
    run, fake, tag = federation
    job = run()
    assert fake.calls["efetch"] == 2, dict(fake.calls)
    assert (job.get("source_outcomes") or {}).get("pubmed") != "error", job.get("source_outcomes")
    if not _ingestion_works():
        pytest.skip("l'ingestion ne marche pas sur ce schéma : les notices ne sont pas mesurables ici")
    with main.engine.connect() as c:
        rows = c.execute(text("SELECT title, year, authors FROM literature_document WHERE title LIKE :t "
                              "AND source = 'pubmed' ORDER BY title"), {"t": f"%retry-{tag}%"}).mappings().all()
    assert len(rows) == 3, "le lot réessayé doit entrer en entier"
    # MedlineDate (« 1998 Spring ») lu, et TOUS les auteurs gardés, pas les six premiers.
    assert all(r["year"] == 1998 for r in rows)
    assert all(r["authors"].count(";") == len(_AUTHORS) - 1 and "Li Mei" in r["authors"] for r in rows)


def test_no_retry_waits_past_the_federation_budget(federation, monkeypatch):
    """Un budget de deux secondes et des attentes de trente : la tentative suivante n'a pas
    lieu, la source finit en échec tout de suite au lieu de dormir au-delà du budget."""
    from api import pipeline as P
    run, fake, _tag = federation
    monkeypatch.setattr(P, "POPULATE_FEDERATION_BUDGET", 2.0)
    monkeypatch.setattr(P, "EXHAUSTIVE_FEDERATION_BUDGET", 2.0)
    monkeypatch.setattr(P, "FETCH_RETRY_BACKOFF_S", (30.0, 30.0, 30.0))
    t0 = time.monotonic()
    job = run()
    assert time.monotonic() - t0 < 25, "une attente a dépassé le budget"
    assert fake.calls["openalex"] == 1, dict(fake.calls)
    assert (job.get("source_outcomes") or {}).get("openalex") in ("error", "cut_by_budget")


def test_every_page_loop_goes_through_the_retry_helper():
    """Compté sur l'arbre : chaque fetcher passe ses pages par `_with_retries` (Semantic
    Scholar et CORE sans les 429, qu'ils attendent à leur façon ; l'efetch de PubMed à
    travers `_efetch_once`)."""
    from api import pipeline as P
    tree = ast.parse(textwrap.dedent(inspect.getsource(P._run_user_scenario_populate)))
    fetchers = {n.name: n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name.startswith("_fetch_")}
    for name in ("_fetch_pubmed", "_fetch_openalex", "_fetch_crossref", "_fetch_europepmc",
                 "_fetch_preprints", "_fetch_doaj", "_fetch_clinicaltrials", "_fetch_arxiv",
                 "_fetch_openaire", "_fetch_core", "_fetch_semantic_scholar"):
        assert name in fetchers, name
        calls = {ast.unparse(c.func) for c in ast.walk(fetchers[name]) if isinstance(c, ast.Call)}
        assert "_with_retries" in calls, f"{name} ne réessaie pas une page en échec"
