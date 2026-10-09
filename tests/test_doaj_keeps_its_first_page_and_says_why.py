"""DOAJ sert la première page d'un booléen long et répond 502 à toutes les suivantes.

Reproduit depuis le serveur de production sur la requête du scénario HPAI_last (894
caractères propres) : page 1 en 200 (100 notices), pages 2 et 3 en 502, en 100 comme en
50 par page ; une requête courte pagine normalement. La fédération gardait bien les 100
notices et marquait la source en échec, mais la raison servie à la carte était l'URL de
1 300 caractères de l'appel, coupée bien avant le numéro de page : « 502 Server Error:
Bad Gateway for url: https://doaj.org/api/search/articles/%28occupationa ».

Ce test PILOTE la fédération réelle contre un `requests` simulé qui répond cent notices
puis 502, et lit ce que la recherche enregistre : les notices gardées, l'issue, la raison.
"""
from __future__ import annotations

import collections

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")

from sqlalchemy import text  # noqa: E402

from api.core import engine  # noqa: E402


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


def _doaj_page(n: int, start: int) -> dict:
    return {"total": 514, "results": [
        {"id": f"doaj-{start + i}", "bibjson": {
            "title": f"Avian influenza among poultry workers, record {start + i}",
            "abstract": "Seroprevalence of H5N1 antibodies in exposed farmers.",
            "year": "2021", "identifier": [{"type": "doi", "id": f"10.9999/doaj.{start + i}"}]}}
        for i in range(n)]}


def _empty_for(url: str) -> _Resp:
    if "openalex" in url:
        return _Resp(200, {"results": [], "meta": {"count": 0}})
    if "crossref" in url:
        return _Resp(200, {"message": {"items": []}})
    if "europepmc" in url:
        return _Resp(200, {"resultList": {"result": []}})
    if "clinicaltrials" in url:
        return _Resp(200, {"studies": []})
    if "arxiv" in url:
        return _Resp(200, body="<feed xmlns='http://www.w3.org/2005/Atom'></feed>")
    if "openaire" in url:
        return _Resp(200, {"results": [], "header": {}})
    if "biorxiv" in url:
        return _Resp(200, {"collection": [], "messages": [{"total": 0}]})
    if "semanticscholar" in url:
        return _Resp(200, {"data": []})
    if "core.ac.uk" in url:
        return _Resp(200, {"results": []})
    if "eutils" in url:
        return _Resp(200, {"esearchresult": {"count": "0", "idlist": []}})
    return _Resp(200, {})


@pytest.fixture
def federation(monkeypatch):
    import requests as _real_requests

    from api import pipeline as P
    from api import sources as S

    calls: collections.Counter = collections.Counter()

    def _fake_get(url, **kw):
        calls[str(url).split("?")[0]] += 1
        if "doaj.org" in str(url):
            page = int((kw.get("params") or {}).get("page") or 1)
            if page == 1:
                return _Resp(200, _doaj_page(100, 0))
            return _Resp(502, body="Bad Gateway")
        return _empty_for(str(url))

    def _fake_post(url, **kw):
        calls[str(url).split("?")[0]] += 1
        return _empty_for(str(url))

    monkeypatch.setattr(_real_requests, "get", _fake_get)
    monkeypatch.setattr(_real_requests, "post", _fake_post)
    monkeypatch.setattr(S, "_ncbi_get", lambda url, params, timeout=30: _fake_get(url))
    monkeypatch.setattr(P, "_ncbi_get", lambda url, params, timeout=30: _fake_get(url))

    sid = "usr-doajpage00001"
    with engine.begin() as c:
        c.execute(text("DELETE FROM article_scenarios WHERE scenario_id = :s"), {"s": sid})
        c.execute(text("DELETE FROM user_scenarios WHERE id = :s"), {"s": sid})
        c.execute(text("DELETE FROM source_query_cache"))
        c.execute(text("INSERT INTO user_scenarios (id, name, query, article_count) "
                       "VALUES (:s, :n, :q, 0)"), {"s": sid, "n": "doaj page", "q": "hpai"})

    def _run():
        P._run_user_scenario_populate(sid, '"avian influenza" AND "poultry workers"', {},
                                      max_results=300, include_live=True, force_live=True)
        return P._user_scenario_populate_jobs.get(sid) or {}

    yield _run, calls, sid
    with engine.begin() as c:
        c.execute(text("DELETE FROM article_scenarios WHERE scenario_id = :s"), {"s": sid})
        c.execute(text("DELETE FROM user_scenarios WHERE id = :s"), {"s": sid})
        c.execute(text("DELETE FROM literature_document WHERE external_id LIKE 'doaj:doaj-%'"))


def test_doaj_gets_a_query_short_enough_to_paginate():
    """Mesuré depuis le serveur : 894 caractères, 502 dès la page 2 ; 796, pagine. DOAJ
    reçoit donc sa propre réduction au-delà de DOAJ_MAX_QUERY_CHARS, dite à la carte."""
    import ast
    import inspect
    import textwrap

    from api import pipeline as P
    from api.search import DOAJ_MAX_QUERY_CHARS, _shorten_boolean, _strip_field_tags

    assert 400 <= DOAJ_MAX_QUERY_CHARS <= 796, "la limite doit rester sous le plus long qui a paginé"
    hpai = ('( "Occupational Exposure"[mh] OR "occupational*"[tiab] OR "Veterinarians"[mh] OR "farm worker*"[tiab] '
            'OR "poultry worker*"[tiab] OR "Abattoirs"[mh] OR "slaughterhouse worker*"[tiab] OR "live bird market*"[tiab] '
            'OR "Health Personnel"[mh] OR "healthcare worker*"[tiab] OR "laboratory worker*"[tiab] OR "seroprevalence"[tiab] '
            'OR "Personal Protective Equipment"[mh] OR "risk perception"[tiab] OR "Health Knowledge, Attitudes, Practice"[mh] ) '
            'AND ( "Influenza in Birds"[mh] OR "Influenza A Virus, H5N1 Subtype"[mh] OR "avian influenza"[tiab] '
            'OR "H5N1"[tiab] OR "H5N6"[tiab] OR "H5N8"[tiab] OR "H7N9"[tiab] OR "H9N2"[tiab] OR "2.3.4.4b"[tiab] )')
    short = _shorten_boolean(" ".join(_strip_field_tags(hpai).split()), DOAJ_MAX_QUERY_CHARS)
    assert short and len(short) <= DOAJ_MAX_QUERY_CHARS and " AND " in short

    tree = ast.parse(textwrap.dedent(inspect.getsource(P._run_user_scenario_populate)))
    assigned: dict[str, list[str]] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign):
            for tgt in n.targets:
                elts = tgt.elts if isinstance(tgt, ast.Tuple) else [tgt]
                for i, e in enumerate(elts):
                    if isinstance(e, ast.Name):
                        value = n.value.elts[i] if (isinstance(tgt, ast.Tuple) and isinstance(n.value, ast.Tuple)) else n.value
                        assigned.setdefault(e.id, []).append(ast.unparse(value))
    assert any("_shorten_boolean(_portable_bool, DOAJ_MAX_QUERY_CHARS)" in v for v in assigned.get("_doaj_q", [])), assigned.get("_doaj_q")
    fetch = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_fetch_doaj")
    src = ast.unparse(fetch)
    assert "_doaj_q" in src and "boolean_native=_doaj_native" in src, "le fetcher DOAJ n'envoie pas sa propre requête"
    subs = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Subscript) and ast.unparse(n.targets[0]).startswith("_fallback_queries[")]
    assert any("'doaj'" in s and "_doaj_q" in s for s in subs), "la carte ne reçoit pas la requête de DOAJ"


DOAJ_URL = "https://doaj.org/api/search/articles/%22avian%20influenza%22%20AND%20%22poultry%20workers%22"


def test_the_fetcher_stops_at_the_first_502_and_the_reason_names_the_page(federation):
    run, calls, _sid = federation
    job = run()
    figures = job.get("prisma_identification") or {}
    assert (job.get("source_outcomes") or {}).get("doaj") == "error", (
        "une pagination refusée n'est ni « a répondu » ni « aucun résultat »")
    reason = (figures.get("source_error_reasons") or {}).get("doaj") or ""
    assert "502" in reason and "page 2" in reason and "100" in reason, reason
    assert "doaj.org/api/search/articles/%28" not in reason, "la raison est encore l'URL coupée"
    # Deux appels, pas un de plus : rien ne tente une troisième page après le 502.
    assert calls[DOAJ_URL] == 2, dict(calls)


def _ingestion_works() -> bool:
    """La base de CI est amorcée par l'application, sans toutes les colonnes de la base de
    production (`d.year`, `document_chunk.created_at`...) : rien ne s'y ingère, donc le
    compte par source y reste à zéro quoi que fasse le fetcher. Plutôt que de deviner les
    colonnes qui manquent, on ESSAIE d'ingérer un document sonde par le même chemin que
    la fédération, et on le retire. La mesure du compte gardé n'a de sens que là où cela
    marche."""
    from api.sources import _ingest_doc_direct
    probe = "doaj:probe-ingestion-0001"
    try:
        did, _new = _ingest_doc_direct(
            source="doaj", title="Ingestion probe for the DOAJ pagination test",
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


def test_the_first_page_is_kept_in_the_table(federation):
    if not _ingestion_works():
        pytest.skip("l'ingestion ne marche pas sur ce schéma : le compte par source n'est pas mesurable ici")
    run, _calls, _sid = federation
    job = run()
    figures = job.get("prisma_identification") or {}
    assert (figures.get("records_by_source") or {}).get("doaj") == 100, (
        "les cent notices de la première page doivent rester au tableau")
