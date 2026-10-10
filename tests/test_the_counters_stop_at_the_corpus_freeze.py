"""Les compteurs servis s'arrêtent au gel du corpus, comme le corpus.

Confirmé par la relecture adversariale (deux lentilles, trois verdicts sur trois pour l'une),
et reproduit par elle sur la vraie fédération : une source coupée par le budget continue de
paginer en arrière-plan (`executor.shutdown(wait=False)`, voulu). Sa page tardive n'entre
plus dans le corpus ni dans les chiffres PRISMA (`_link_to_scenario` refuse après le gel),
mais `_inc` continuait d'écrire `ingested` et `sources` dans le job, et le statut final
publiait le dict vivant lui-même. Le statut servi après « done » affichait alors
« Sources interrogées : ... (crossref: 160) » au-dessus d'un `total_found` de 100, et
continuait de monter après la publication.

Ce test pilote la vraie fédération contre un `requests` simulé : Crossref sert une première
page pleine, puis sa seconde page arrive APRÈS l'échéance du budget ; le scoring est
ralenti pour que la publication finale vienne après cette page. On lit le statut publié,
puis on le relit plus tard.
"""
from __future__ import annotations

import collections
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")

from sqlalchemy import text  # noqa: E402

from api.core import engine  # noqa: E402

BUDGET_S = 3.0
LATE_S = 1.0
SCORING_S = 3.0
P1_TITLED = 40
P2_ROWS = 25


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


def _empty_for(url: str) -> _Resp:
    if "openalex" in url:
        return _Resp(200, {"results": [], "meta": {"count": 0}})
    if "europepmc" in url:
        return _Resp(200, {"resultList": {"result": []}, "hitCount": 0})
    if "doaj.org" in url:
        return _Resp(200, {"results": []})
    if "clinicaltrials" in url:
        return _Resp(200, {"studies": []})
    if "arxiv" in url:
        return _Resp(200, body="<feed xmlns='http://www.w3.org/2005/Atom'></feed>")
    if "openaire" in url:
        return _Resp(200, {"results": [], "header": {}})
    if "biorxiv" in url:
        return _Resp(200, {"collection": [], "messages": [{"total": 0}]})
    if "semanticscholar" in url:
        return _Resp(200, {"data": [], "total": 0})
    if "core.ac.uk" in url:
        return _Resp(200, {"results": [], "totalHits": 0})
    if "eutils" in url:
        return _Resp(200, {"esearchresult": {"count": "0", "idlist": []}})
    return _Resp(200, {})


def _crossref_page(tag: str, page: int, size: int, titled: int) -> _Resp:
    items = [{
        "DOI": f"10.5555/freeze.{tag}.p{page}.{i}",
        "title": ([f"Avian influenza exposure among poultry workers, record {tag} {page}.{i}"]
                  if i < titled else []),
        "abstract": "An abstract long enough to pass the thirty character rule, twice over. " * 2,
        "created": {"date-parts": [[2023]]},
    } for i in range(size)]
    return _Resp(200, {"message": {"items": items, "total-results": 1_000_000}})


def _ingestion_works() -> bool:
    """La base de CI n'a pas toutes les colonnes de la production : rien ne s'y ingère, et
    les compteurs y restent à zéro quoi qu'il arrive. On essaie une ingestion sonde."""
    from api.sources import _ingest_doc_direct
    probe = "crossref:probe-freeze-0001"
    try:
        did, _new = _ingest_doc_direct(
            source="crossref", title="Ingestion probe for the corpus freeze test",
            abstract="A probe document, inserted then deleted by the test.", year=2021,
            url=None, external_id=probe, doi=None)
    except Exception:                                    # noqa: BLE001
        return False
    try:
        with engine.begin() as c:
            if did is not None:
                c.execute(text("DELETE FROM document_chunk WHERE document_id = :d"), {"d": did})
            c.execute(text("DELETE FROM literature_document WHERE external_id = :e"), {"e": probe})
    except Exception:                                    # noqa: BLE001
        pass
    return did is not None


@pytest.fixture
def late_crossref(monkeypatch):
    if not _ingestion_works():
        pytest.skip("l'ingestion ne marche pas sur ce schéma : les compteurs ne sont pas mesurables ici")
    import requests as _real_requests

    from api import pipeline as P
    from api import relevance as R
    from api import sources as S

    tag = str(int(time.time() * 1000))[-8:]
    calls: collections.Counter = collections.Counter()
    t0: list[float | None] = [None]
    page2_served: list[float | None] = [None]

    def _fake_get(url, **kw):
        key = str(url).split("?")[0]
        calls[key] += 1
        if "crossref" in str(url):
            if t0[0] is None:
                t0[0] = time.time()
            if calls[key] == 1:
                return _crossref_page(tag, 1, 1000, P1_TITLED)
            time.sleep(max(0.0, t0[0] + BUDGET_S + LATE_S - time.time()))
            page2_served[0] = time.time()
            return _crossref_page(tag, 2, P2_ROWS, P2_ROWS)
        return _empty_for(str(url))

    def _fake_post(url, **kw):
        calls[str(url).split("?")[0]] += 1
        return _empty_for(str(url))

    monkeypatch.setattr(_real_requests, "get", _fake_get)
    monkeypatch.setattr(_real_requests, "post", _fake_post)
    monkeypatch.setattr(S, "_ncbi_get", lambda url, params, timeout=30: _fake_get(url))
    monkeypatch.setattr(P, "_ncbi_get", lambda url, params, timeout=30: _fake_get(url))
    monkeypatch.setattr(P, "POPULATE_FEDERATION_BUDGET", BUDGET_S)

    frozen_at: list[float | None] = [None]
    real_corpus_ids = P._boolean_corpus_ids

    def _corpus_ids_timed(*a, **k):
        frozen_at[0] = frozen_at[0] or time.time()        # le gel a lieu juste avant
        return real_corpus_ids(*a, **k)

    monkeypatch.setattr(P, "_boolean_corpus_ids", _corpus_ids_timed)

    def _slow_scoring(*a, **k):
        time.sleep(SCORING_S)
        return 0

    monkeypatch.setattr(R, "_run_semantic_rerank_inline", _slow_scoring)

    sid = "usr-freezecount01"
    with engine.begin() as c:
        c.execute(text("DELETE FROM article_scenarios WHERE scenario_id = :s"), {"s": sid})
        c.execute(text("DELETE FROM user_scenarios WHERE id = :s"), {"s": sid})
        c.execute(text("DELETE FROM source_query_cache"))
        c.execute(text("INSERT INTO user_scenarios (id, name, query, article_count) "
                       "VALUES (:s, :n, :q, 0)"), {"s": sid, "n": "freeze counters", "q": "hpai"})

    def _run():
        P._run_user_scenario_populate(sid, '"avian influenza" AND "poultry workers"', {},
                                      max_results=2000, include_live=True, force_live=True)
        return P._user_scenario_populate_jobs.get(sid) or {}

    yield _run, sid, page2_served, frozen_at
    with engine.begin() as c:
        c.execute(text("DELETE FROM article_scenarios WHERE scenario_id = :s"), {"s": sid})
        c.execute(text("DELETE FROM user_scenarios WHERE id = :s"), {"s": sid})
        c.execute(text("DELETE FROM document_chunk WHERE document_id IN (SELECT id FROM literature_document "
                       "WHERE doi LIKE '10.5555/freeze.%')"))
        c.execute(text("DELETE FROM literature_document WHERE doi LIKE '10.5555/freeze.%'"))


def test_the_published_counters_do_not_move_after_the_freeze(late_crossref):
    import copy
    run, sid, page2_served, frozen_at = late_crossref
    job = run()
    published = copy.deepcopy(job)
    # Laisser la page tardive finir ses insertions, puis relire le statut.
    time.sleep(2.0)
    from api import pipeline as P
    later = P._user_scenario_populate_jobs.get(sid) or {}

    assert page2_served[0] is not None, "la seconde page n'a jamais été servie : le scénario n'est pas joué"
    if frozen_at[0] is None or page2_served[0] <= frozen_at[0]:
        pytest.skip("la page tardive est arrivée avant le gel sur cette machine : rien à mesurer")

    figures = published.get("prisma_identification") or {}
    frozen_crossref = int((figures.get("records_by_source") or {}).get("crossref") or 0)
    assert frozen_crossref == P1_TITLED, figures.get("records_by_source")

    shown = int((published.get("sources") or {}).get("crossref") or 0)
    assert shown <= frozen_crossref, (
        f"le statut publié détaille crossref: {shown} au-dessus d'un total gelé de {frozen_crossref}")
    assert f"crossref: {shown}" in (published.get("message") or ""), published.get("message")

    for key in ("ingested", "errors", "sources", "total_found", "message"):
        assert later.get(key) == published.get(key), (
            f"`{key}` a encore bougé après la publication : {published.get(key)!r} -> {later.get(key)!r}")
