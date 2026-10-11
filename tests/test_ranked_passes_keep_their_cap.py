"""Les passages classés par pertinence gardent leur plafond, quel que soit LIVE_MAX_PER_SOURCE.

En production, LIVE_MAX_PER_SOURCE est réglé très haut pour lever le plafond d'une recherche
standard, et l'écran envoyait de toute façon 2 000. Depuis que l'écran laisse le serveur
décider, ce plafond s'appliquait aussi aux deux passages qui ne s'arrêtent jamais d'eux-mêmes,
parce qu'ils classent tout par pertinence : Crossref (des mots-clés, 2,6 millions de réponses
sur la question HPAI) et le passage « meilleures correspondances en texte intégral »
d'OpenAlex. Ils paginaient jusqu'au refus de l'API (400 au-delà de 10 000 résultats), les deux
sources finissaient en échec, et OpenAlex versait au corpus des milliers d'articles qui ne
répondent au booléen que dans leur texte intégral. Ces tests pilotent la fédération réelle
contre des API simulées qui ne tarissent jamais.
"""
import collections
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")

from sqlalchemy import text  # noqa: E402

import main  # noqa: E402
from search_tables import ensure_search_tables  # noqa: E402

QUERY = '("poultry worker" OR veterinarian) AND ("avian influenza" OR h5n1)'
SID = "usr-rankedcap0001"
# Le défaut de RANKED_PASS_MAX : ce que Crossref et le texte intégral d'OpenAlex gardaient
# quand l'écran envoyait 2 000.
RANKED_DEFAULT = 2000
_ABSTRACT = "Seroprevalence of H5N1 antibodies among exposed poultry workers, long enough to keep."


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


class _EndlessSources:
    """Crossref et OpenAlex rendent des pages pleines sans fin, et refusent (400), comme
    les vraies API, tout décalage ou numéro de page au-delà de 10 000 résultats. Les autres
    sources ne trouvent rien."""

    def __init__(self, tag: str):
        self.tag = tag
        self.calls: collections.Counter = collections.Counter()
        self.crossref_offsets: list[int] = []
        self.openalex_pages: list[tuple[int, int]] = []

    def get(self, url, params=None, **kw):
        url, params = str(url), dict(params or {})
        if "crossref" in url:
            offset, rows = int(params.get("offset") or 0), int(params.get("rows") or 20)
            self.crossref_offsets.append(offset)
            if offset + rows > 10000:
                return _Resp(400, body="Bad Request")
            return _Resp(200, {"message": {"total-results": 2634984, "items": [
                {"DOI": f"10.9999/ranked-{self.tag}-{offset + i}",
                 "title": [f"Avian influenza and poultry workers, ranked-{self.tag} crossref {offset + i}"],
                 "abstract": _ABSTRACT, "issued": {"date-parts": [[2021]]}} for i in range(rows)]}})
        if "openalex" in url:
            self.calls["openalex"] += 1
            per_page = int(params.get("per-page") or 25)
            if "page" in params:                       # le passage classé, par numéro de page
                page = int(params["page"])
                self.openalex_pages.append((page, per_page))
                if page * per_page > 10000:
                    return _Resp(400, body="Bad Request")
                start = (page - 1) * per_page
                return _Resp(200, {"meta": {"count": 500000}, "results": [
                    {"id": f"https://openalex.org/W{self.tag}{start + i}",
                     "title": f"Poultry workers and avian influenza, ranked-{self.tag} oa {start + i}",
                     "abstract_inverted_index": {"Exposed": [0], "workers": [1], "tested": [2]},
                     "publication_year": 2020} for i in range(per_page)]})
            # Le passage par curseur (titre et résumé) : une page, puis plus rien.
            return _Resp(200, {"meta": {"count": 3, "next_cursor": None}, "results": []})
        if "eutils" in url and "esearch" in url:
            return _Resp(200, {"esearchresult": {"count": "0", "idlist": []}})
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
        if "doaj.org" in url:
            return _Resp(200, {"total": 0, "results": []})
        return _Resp(200, {})

    def post(self, url, data=None, json=None, **kw):              # noqa: A002 - le nom de requests
        return _Resp(200, {"results": [], "totalHits": 0})


@pytest.fixture
def federation(db_conn, monkeypatch):
    import requests as _real_requests

    from api import pipeline as P
    from api import relevance as R

    created_chunk_table = ensure_search_tables(db_conn.cursor())
    tag = str(int(time.time() * 1000))[-7:]
    fake = _EndlessSources(tag)
    monkeypatch.setattr(_real_requests, "get", fake.get)
    monkeypatch.setattr(_real_requests, "post", fake.post)
    monkeypatch.setattr(P, "_ncbi_get", lambda url, params, timeout=30, use_key=True: fake.get(url, params))
    monkeypatch.setenv("CORE_API_KEY", "")
    monkeypatch.setattr(P, "FETCH_RETRY_BACKOFF_S", (0.0, 0.0, 0.0))
    # La valeur de production : LIVE_MAX_PER_SOURCE réglé très haut pour lever le plafond
    # d'une recherche standard (le populate y ramène `max_results`).
    monkeypatch.setattr(P, "LIVE_MAX_PER_SOURCE", 100000)
    monkeypatch.setattr(R, "_run_semantic_rerank_inline", lambda *a, **k: 0)
    monkeypatch.setattr(R, "_run_cross_encoder_rerank", lambda *a, **k: {})
    monkeypatch.setattr(P, "_run_clustering_background", lambda *a, **k: None)
    monkeypatch.setattr(P, "_precompute_user_kg", lambda *a, **k: None)
    monkeypatch.setattr(P, "_ingest_doc_direct", lambda **kw: (None, False))

    def _clean():
        with main.engine.begin() as c:
            c.execute(text("DELETE FROM article_scenarios WHERE scenario_id = :s"), {"s": SID})
            c.execute(text("DELETE FROM user_scenarios WHERE id = :s"), {"s": SID})
            c.execute(text("DELETE FROM source_query_cache"))

    _clean()
    with main.engine.begin() as c:
        c.execute(text("INSERT INTO user_scenarios (id, name, query, article_count) "
                       "VALUES (:s, 'ranked caps', :q, 0)"), {"s": SID, "q": QUERY})

    def run(exhaustive: bool):
        # 100 000 : la valeur de LIVE_MAX_PER_SOURCE en production, que l'écran ne ramène
        # plus à 2 000.
        P._run_user_scenario_populate(SID, QUERY, {}, max_results=100000, include_live=True,
                                      auto_pipeline=False, force_live=True, exhaustive=exhaustive)
        return P._user_scenario_populate_jobs.get(SID) or {}

    yield run, fake, P
    _clean()
    if created_chunk_table:
        with db_conn.cursor() as cur:
            cur.execute("DROP TABLE document_chunk")


@pytest.mark.parametrize("exhaustive", [False, True])
def test_crossref_stops_at_its_ranked_cap_and_never_errors(federation, exhaustive):
    run, fake, P = federation
    job = run(exhaustive)
    assert fake.crossref_offsets, "Crossref n'a pas été interrogé"
    assert max(fake.crossref_offsets) < RANKED_DEFAULT, fake.crossref_offsets
    assert (job.get("source_outcomes") or {}).get("crossref") != "error", job.get("source_outcomes")


@pytest.mark.parametrize("exhaustive", [False, True])
def test_the_openalex_full_text_pass_stops_at_its_ranked_cap(federation, exhaustive):
    run, fake, P = federation
    job = run(exhaustive)
    fetched = sum(per for _page, per in fake.openalex_pages)
    assert 0 < fetched <= RANKED_DEFAULT, fake.openalex_pages
    assert all(page * per <= 10000 for page, per in fake.openalex_pages), fake.openalex_pages
    assert (job.get("source_outcomes") or {}).get("openalex") != "error", job.get("source_outcomes")


@pytest.mark.parametrize("exhaustive", [False, True])
def test_the_prisma_card_says_which_passes_keep_their_own_cap(federation, monkeypatch, exhaustive):
    run, _fake, P = federation
    job = run(exhaustive)
    ident = job.get("prisma_identification") or {}
    caps = ident.get("source_caps") or {}
    assert caps.get("crossref") == RANKED_DEFAULT, caps
    # En standard, le seul passage d'OpenAlex est le passage classé ; en exhaustif, son
    # passage titre et résumé a le plafond commun.
    assert caps.get("openalex") == (None if exhaustive else RANKED_DEFAULT), caps
    # Et l'écran de recherche annonce, avant la recherche, ce que la carte dira après.
    from api import sources as S
    monkeypatch.setattr(S, "LIVE_MAX_PER_SOURCE", P.LIVE_MAX_PER_SOURCE)
    limits = S.search_limits()
    assert ident.get("per_source_cap") == limits["exhaustive_per_source" if exhaustive
                                                 else "standard_per_source"], (ident, limits)
    assert caps.get("crossref") == limits["ranked_per_source"], (caps, limits)


def test_the_ranked_cap_never_passes_the_apis_own_bound(monkeypatch):
    import importlib

    from api import search as S
    monkeypatch.setenv("RANKED_PASS_MAX", "50000")
    try:
        assert importlib.reload(S).RANKED_PASS_MAX == 10000
    finally:
        monkeypatch.delenv("RANKED_PASS_MAX", raising=False)
        importlib.reload(S)
