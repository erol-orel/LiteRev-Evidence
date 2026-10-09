"""Le 429 est réellement borné : on COMPTE les requêtes, on ne relit pas le source.

La relecture adversariale a fait l'expérience que j'aurais dû faire : elle a supprimé le
`_429 += 1` des trois appelants, ce qui rend la boucle de nouveau infinie, et mes tests
sont restés verts. L'un est passé au rouge, mais pour une raison incidente, une expression
régulière sur le TEXTE du source dont le décalage avait changé. Reproduit, et confirmé.

C'est la pathologie que ce dépôt redoute le plus : un test qui laisse passer un code
incorrect et se casse sur un code correct. Mes six premières assertions sur la limitation
de débit lisaient le texte du module ; aucune ne lisait un comportement. Celles-ci
PILOTENT la fédération réelle contre un `requests` simulé et comptent les appels, parce
que la borne ne vit pas dans `_wait_out_rate_limit` mais chez ses trois appelants, qui lui
passent le compteur.

La forme de ce harnais vient de la relecture, qui s'en était servie pour mesurer
« exactement 3 requêtes en 6,5 s ».
"""
from __future__ import annotations

import collections

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")

from sqlalchemy import text  # noqa: E402

from api.core import engine  # noqa: E402


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


def _empty_for(url: str) -> _Resp:
    """Chaque source répond « rien », sauf celle qu'un test veut malmener."""
    if "openalex" in url:
        return _Resp(200, {"results": [], "meta": {"count": 0}})
    if "crossref" in url:
        return _Resp(200, {"message": {"items": []}})
    if "europepmc" in url:
        return _Resp(200, {"resultList": {"result": []}})
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
        return _Resp(200, {"data": []})
    if "core.ac.uk" in url:
        return _Resp(200, {"results": []})
    if "eutils" in url:
        return _Resp(200, {"esearchresult": {"count": "0", "idlist": []}})
    return _Resp(200, {})


S2_BULK = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
S2_PLAIN = "https://api.semanticscholar.org/graph/v1/paper/search"


@pytest.fixture
def federation(monkeypatch, request):
    """Pilote la vraie fédération, sans réseau, et rend les compteurs d'appels.

    `request.param` est le `Retry-After` à renvoyer avec chaque 429 (None pour aucun)."""
    import requests as _real_requests

    from api import pipeline as P
    from api import sources as S

    calls: collections.Counter = collections.Counter()
    retry_after = getattr(request, "param", None)

    def _fake_get(url, **kw):
        calls[str(url).split("?")[0]] += 1
        if "semanticscholar" in str(url):
            hdrs = {"Retry-After": str(retry_after)} if retry_after is not None else {}
            return _Resp(429, {}, headers=hdrs)
        return _empty_for(str(url))

    def _fake_post(url, **kw):
        calls[str(url).split("?")[0]] += 1
        return _empty_for(str(url))

    monkeypatch.setattr(_real_requests, "get", _fake_get)
    monkeypatch.setattr(_real_requests, "post", _fake_post)
    monkeypatch.setattr(S, "_ncbi_get", lambda url, params, timeout=30: _fake_get(url))
    monkeypatch.setattr(P, "_ncbi_get", lambda url, params, timeout=30: _fake_get(url))

    sid = "usr-ratelimit0001"
    with engine.begin() as c:
        c.execute(text("DELETE FROM article_scenarios WHERE scenario_id = :s"), {"s": sid})
        c.execute(text("DELETE FROM user_scenarios WHERE id = :s"), {"s": sid})
        c.execute(text("DELETE FROM source_query_cache"))
        c.execute(text("INSERT INTO user_scenarios (id, name, query, article_count) "
                       "VALUES (:s, :n, :q, 0)"), {"s": sid, "n": "rate limit", "q": "hpai"})

    def _run():
        P._run_user_scenario_populate(sid, '"avian influenza"', {}, max_results=300,
                                      include_live=True, force_live=True)
        return P._user_scenario_populate_jobs.get(sid) or {}

    yield _run, calls, sid
    with engine.begin() as c:
        c.execute(text("DELETE FROM article_scenarios WHERE scenario_id = :s"), {"s": sid})
        c.execute(text("DELETE FROM user_scenarios WHERE id = :s"), {"s": sid})


#: Le plafond de tentatives, lu depuis le code pour que le test suive un changement
#: délibéré au lieu d'épingler un 3 en dur.
def _attempts() -> int:
    import inspect

    from api import pipeline as P
    src = inspect.getsource(P._run_user_scenario_populate)
    for line in src.splitlines():
        if "_RATE_LIMIT_ATTEMPTS =" in line:
            return int(line.split("=")[1].strip())
    raise AssertionError("_RATE_LIMIT_ATTEMPTS a disparu")


def test_a_source_that_always_answers_429_is_asked_a_bounded_number_of_times(federation):
    """LE test que la relecture a montré manquant.

    Supprimer `_429 += 1` chez les appelants rend la boucle infinie jusqu'au budget :
    ce compteur d'appels le voit, là où une relecture du texte du module ne le voyait pas.
    """
    run, calls, _sid = federation
    run()
    n = calls[S2_BULK] + calls[S2_PLAIN]
    assert n > 0, "Semantic Scholar n'a pas été interrogée du tout"
    # Une marge : le fetcher peut emprunter l'un ou l'autre point d'entrée, et une
    # pagination légitime recommence le compte à chaque page. Sans borne, ce nombre
    # s'envole jusqu'au budget de fédération (une quarantaine d'appels et plus).
    assert n <= _attempts() * 2, (
        f"{n} requêtes à une source qui répond 429, pour un plafond de {_attempts()} "
        "tentatives : la borne du 429 ne tient pas, et la recherche tiendra son fil "
        "jusqu'au budget de fédération")


def test_a_rate_limited_source_ends_as_an_error_not_as_empty(federation):
    """L'issue, mesurée sur un run réel, et non lue dans le source."""
    run, _calls, _sid = federation
    job = run()
    outcomes = job.get("source_outcomes") or {}
    assert outcomes.get("semantic_scholar") == "error", (
        f"une source limitée en débit est annoncée « {outcomes.get('semantic_scholar')} » ; "
        "`empty` serait une affirmation sur la littérature")
    # Et la raison voyage avec l'issue.
    reasons = (job.get("prisma_identification") or {}).get("source_error_reasons") or {}
    assert "429" in (reasons.get("semantic_scholar") or ""), (
        f"la raison ne dit pas que c'était un 429 : {reasons.get('semantic_scholar')!r}")


def test_the_coverage_line_names_the_rate_limited_source(federation):
    run, _calls, _sid = federation
    job = run()
    msg = job.get("message") or ""
    assert "semantic_scholar" in msg or "semanticscholar" in msg.lower(), (
        f"la ligne de couverture ne nomme pas la source en échec : {msg!r}")
    assert "échec" in msg or "error" in msg


def test_the_other_eleven_sources_are_not_held_up_by_the_throttled_one(federation):
    """Le fil de la source limitée ne doit pas retarder la recherche entière."""
    run, calls, _sid = federation
    job = run()
    outcomes = job.get("source_outcomes") or {}
    assert len(outcomes) >= 12, f"seulement {len(outcomes)} sources déclarées"
    # Les autres ont bien répondu (vide, ici), donc elles ont été interrogées.
    searched = (job.get("prisma_identification") or {}).get("sources_searched")
    assert searched is not None and searched >= 9, (
        f"seulement {searched} sources interrogées : une source limitée en débit a "
        "entraîné les autres")
    assert calls["https://api.openalex.org/works"] >= 1
    assert calls["https://www.ebi.ac.uk/europepmc/webservices/rest/search"] >= 1


@pytest.mark.parametrize("federation", [1], indirect=True)
def test_a_retry_after_header_is_honoured_without_breaking_the_bound(federation):
    """`Retry-After: 1` doit être respecté, et ne pas multiplier les tentatives."""
    run, calls, _sid = federation
    job = run()
    n = calls[S2_BULK] + calls[S2_PLAIN]
    assert n <= _attempts() * 2, f"{n} requêtes avec un Retry-After de 1 s"
    assert (job.get("source_outcomes") or {}).get("semantic_scholar") == "error"


@pytest.mark.parametrize("federation", ["Wed, 21 Oct 2026 07:28:00 GMT"], indirect=True)
def test_an_http_date_retry_after_does_not_break_the_loop(federation):
    """Un `Retry-After` en date HTTP n'est pas un nombre : il doit retomber sur le défaut,
    sans lever hors du try ni rendre la boucle infinie."""
    run, calls, _sid = federation
    job = run()
    n = calls[S2_BULK] + calls[S2_PLAIN]
    assert n <= _attempts() * 2, f"{n} requêtes avec un Retry-After en date HTTP"
    assert (job.get("source_outcomes") or {}).get("semantic_scholar") == "error"
