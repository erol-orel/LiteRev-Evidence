"""Une stratégie longue n'est plus amputée : en recherche exhaustive, chaque base que l'URL
limite la reçoit ENTIÈRE, découpée en requêtes dont l'union est le booléen.

Avant : au-delà de 1 200 caractères, OpenAlex, DOAJ, CORE, ClinicalTrials.gov, arXiv et
OpenAIRE recevaient la requête réduite (chaque bloc OU tronqué à ses premiers termes, les
exclusions retirées) ; CORE, qui reçoit sa requête dans le corps d'un POST JSON, la
recevait réduite aussi ; PubMed ne recevait l'union « (MeSH) OR (booléen) » que sous
1 900 caractères, sinon la seule requête MeSH (35 notices contre 306) ; la limite se
mesurait en caractères et non sur l'URL encodée ; et une facette OU trop longue pour
être fédérée ne le disait qu'au journal. Ces tests pilotent la fédération réelle contre
des API simulées et lisent ce que chaque base a reçu.
"""
import json
import pathlib
import urllib.parse

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")

from sqlalchemy import text  # noqa: E402

import main  # noqa: E402
from api.search import (  # noqa: E402
    FACET_WIDEN_MAX_CHARS,
    OPENAIRE_MAX_OPERATORS,
    URL_QUERY_MAX_ENCODED,
    _boolean_to_arxiv,
    _clean_boolean,
    _count_operators,
    _shorten_boolean,
    _split_boolean,
    _url_encoded_len,
    _widen_boolean_for_or_facets,
)

from search_tables import ensure_search_tables  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: Une stratégie de revue : quarante-cinq expositions en OU, ET un bloc de virus. Sa forme
#: propre dépasse la limite d'URL encodée, chacune de ses moitiés la tient.
_EXPOSURES = [f'"poultry worker exposure cohort number {i:02d}"' for i in range(1, 46)]
LONG_Q = "(" + " OR ".join(_EXPOSURES) + ') AND ("avian influenza" OR h5n1 OR h7n9)'
MESH_Q = '"Influenza in Birds"[Mesh] AND "Occupational Exposure"[Mesh]'
#: Courte, mais plus de quatre opérateurs : OpenAIRE ne la prend pas en une fois.
SHORT_Q = '("poultry worker" OR veterinarian OR "farm worker" OR slaughterhouse) AND ("avian influenza" OR h5n1)'


# ── la mesure et le découpage (purs) ─────────────────────────────────────────

def test_the_url_limit_is_measured_on_the_encoded_query():
    q = '("a b" OR c) AND d'
    assert _url_encoded_len(q) == len(urllib.parse.quote_plus(q)) > len(q)
    # Sous la limite en caractères, au-delà une fois encodée : les guillemets y prennent
    # trois octets chacun, et c'est l'URL encodée que les serveurs refusent.
    q = " OR ".join(f'"x{i:03d} y"' for i in range(120))
    clean = _clean_boolean(q)
    assert len(clean) < URL_QUERY_MAX_ENCODED < _url_encoded_len(clean)
    assert _shorten_boolean(q, URL_QUERY_MAX_ENCODED) == clean, "mesurée en caractères, elle passait"
    short = _shorten_boolean(q, URL_QUERY_MAX_ENCODED, measure=_url_encoded_len)
    assert short and short != clean and _url_encoded_len(short) <= URL_QUERY_MAX_ENCODED
    assert _url_encoded_len(_clean_boolean(LONG_Q)) > URL_QUERY_MAX_ENCODED


def test_every_part_fits_and_the_parts_hold_every_term():
    parts = _split_boolean(LONG_Q, URL_QUERY_MAX_ENCODED, measure=_url_encoded_len)
    assert len(parts) >= 2 and all(_url_encoded_len(p) <= URL_QUERY_MAX_ENCODED for p in parts)
    joined = " ".join(parts)
    assert all(e.strip('"') in joined for e in _EXPOSURES), "un terme a disparu au découpage"
    ax = _split_boolean(LONG_Q, URL_QUERY_MAX_ENCODED, render=_boolean_to_arxiv, measure=_url_encoded_len)
    assert ax and all(p.count("all:") >= 2 and _url_encoded_len(p) <= URL_QUERY_MAX_ENCODED for p in ax)
    oa = _split_boolean(SHORT_Q, URL_QUERY_MAX_ENCODED, measure=_url_encoded_len,
                        max_operators=OPENAIRE_MAX_OPERATORS)
    assert len(oa) == 2 and all(_count_operators(p) <= OPENAIRE_MAX_OPERATORS for p in oa), oa


def test_an_or_facet_that_cannot_be_federated_is_listed():
    long_facet = " OR ".join(f'"facet term number {i:03d}"' for i in range(200))
    assert len(long_facet) > FACET_WIDEN_MAX_CHARS
    skipped: list = []
    boolean, pubmed, n_or = _widen_boolean_for_or_facets(
        "h5n1", "h5n1[tiab]", [{"kind": "boolean", "text": "h5n1"},
                               {"kind": "boolean", "text": "ferrets", "op": "or"},
                               {"kind": "boolean", "text": long_facet, "op": "or"}],
        "union", skipped=skipped)
    assert skipped == [long_facet] and n_or == 1 and "ferrets" in boolean and "facet term" not in boolean
    figures = main._prisma_identification_figures({"pubmed": 3}, 3, 0, 3, facets_skipped=skipped)
    assert figures["facets_skipped"] == [long_facet]


def test_the_card_names_the_facets_left_out_in_both_languages():
    page = (ROOT / "frontend" / "src" / "components" / "ScenarioDetailPage.tsx").read_text(encoding="utf-8")
    assert "ident.facets_skipped" in page and "prisma.facetsSkipped" in page
    for loc in ("fr", "en"):
        txt = (ROOT / "frontend" / "src" / "i18n" / "locales" / f"{loc}.ts").read_text(encoding="utf-8")
        assert "facetsSkipped:" in txt and "localMatchCapped:" in txt, loc


# ── la fédération (base de données, API simulées) ────────────────────────────

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


class _EmptySources:
    """Chaque base répond « rien trouvé » ; on ne lit que ce qu'elle a reçu."""

    def __init__(self):
        self.gets: list[tuple[str, dict]] = []
        self.posts: list[tuple[str, dict]] = []

    def get(self, url, params=None, **kw):
        url = str(url)
        self.gets.append((url, dict(params or {})))
        if "openalex" in url:
            return _Resp(200, {"results": [], "meta": {"count": 0, "next_cursor": None}})
        if "crossref" in url:
            return _Resp(200, {"message": {"items": [], "total-results": 0}})
        if "europepmc" in url or "ebi.ac.uk" in url:
            return _Resp(200, {"resultList": {"result": []}, "hitCount": 0})
        if "doaj.org" in url:
            return _Resp(200, {"results": [], "total": 0})
        if "clinicaltrials" in url:
            return _Resp(200, {"studies": [], "totalCount": 0})
        if "arxiv" in url:
            return _Resp(200, body="<feed xmlns='http://www.w3.org/2005/Atom'></feed>")
        if "openaire" in url:
            return _Resp(200, {"results": [], "header": {}})
        if "semanticscholar" in url:
            return _Resp(200, {"data": [], "total": 0})
        if "eutils" in url:
            return _Resp(200, {"esearchresult": {"count": "0", "idlist": []}})
        return _Resp(200, {})

    def post(self, url, data=None, json=None, **kw):              # noqa: A002 - le nom de requests
        url = str(url)
        self.posts.append((url, dict(json or data or {})))
        if "eutils" in url:
            return _Resp(200, {"esearchresult": {"count": "0", "idlist": []}})
        return _Resp(200, {"results": [], "totalHits": 0})

    def params(self, needle: str, key: str) -> list:
        return [p.get(key) for u, p in self.gets if needle in u and key in p]


SID = "usr-longquery0001"


@pytest.fixture
def federation(db_conn, monkeypatch):
    import requests as _real_requests

    from api import pipeline as P
    from api import relevance as R

    created_chunk_table = ensure_search_tables(db_conn.cursor())
    fake = _EmptySources()
    monkeypatch.setattr(_real_requests, "get", fake.get)
    monkeypatch.setattr(_real_requests, "post", fake.post)
    monkeypatch.setenv("CORE_API_KEY", "test-core-key")
    monkeypatch.setattr(R, "_run_semantic_rerank_inline", lambda *a, **k: 0)
    monkeypatch.setattr(R, "_run_cross_encoder_rerank", lambda *a, **k: {})
    monkeypatch.setattr(P, "_run_clustering_background", lambda *a, **k: None)
    monkeypatch.setattr(P, "_precompute_user_kg", lambda *a, **k: None)

    def _reset(general: str, pubmed: str):
        with main.engine.begin() as c:
            c.execute(text("DELETE FROM article_scenarios WHERE scenario_id = :s"), {"s": SID})
            c.execute(text("DELETE FROM user_scenarios WHERE id = :s"), {"s": SID})
            c.execute(text("DELETE FROM source_query_cache"))
            c.execute(text("INSERT INTO user_scenarios (id, name, query, article_count, search_strategy) "
                           "VALUES (:s, 'long query', 'avian influenza in poultry workers', 0, CAST(:st AS jsonb))"),
                      {"s": SID, "st": json.dumps({"general": general, "pubmed": pubmed,
                                                   "explanation": "", "synonyms": []})})

    def run(general: str = LONG_Q, pubmed: str = MESH_Q, exhaustive: bool = True):
        _reset(general, pubmed)
        fake.gets.clear()
        fake.posts.clear()
        P._run_user_scenario_populate(SID, "avian influenza in poultry workers", {}, max_results=50,
                                      include_live=True, auto_pipeline=False, force_live=True,
                                      exhaustive=exhaustive)
        return (P._user_scenario_populate_jobs.get(SID) or {}).get("prisma_identification") or {}

    yield run, fake
    with main.engine.begin() as c:
        c.execute(text("DELETE FROM article_scenarios WHERE scenario_id = :s"), {"s": SID})
        c.execute(text("DELETE FROM user_scenarios WHERE id = :s"), {"s": SID})
        c.execute(text("DELETE FROM source_query_cache"))
    if created_chunk_table:
        with db_conn.cursor() as cur:
            cur.execute("DROP TABLE document_chunk")


def test_an_exhaustive_search_splits_the_strategy_for_every_url_limited_database(federation):
    run, fake = federation
    fig = run(exhaustive=True)
    parts = _split_boolean(LONG_Q, URL_QUERY_MAX_ENCODED, measure=_url_encoded_len)
    split = fig.get("split_queries") or {}
    assert split.get("openalex") == parts and split.get("clinicaltrials") == parts, split
    assert split.get("arxiv") and split.get("doaj"), split
    # Ce que chaque base a REÇU : les morceaux, jamais la requête réduite.
    assert set(fake.params("clinicaltrials", "query.term")) == set(parts)
    assert set(fake.params("openalex", "search")) == set(parts)
    assert {f.split(":", 1)[1] for f in fake.params("openalex", "filter")} == {p.replace(",", " ") for p in parts}
    assert set(fake.params("arxiv", "search_query")) == set(split["arxiv"])
    # Leurs notices sont natives : la carte ne les range plus parmi les stratégies dégradées.
    fallback = set(fig.get("keyword_fallback_sources") or [])
    assert not fallback & {"openalex", "clinicaltrials", "arxiv", "doaj", "core"}, fallback
    # OpenAIRE : quatre opérateurs par morceau en demanderaient trop, elle garde sa réduction.
    assert "openaire" in fallback and "openaire" not in split
    assert all(_count_operators(q) <= OPENAIRE_MAX_OPERATORS for q in fake.params("openaire", "search"))


def test_core_gets_the_whole_strategy_in_both_modes(federation):
    run, fake = federation
    whole = _clean_boolean(LONG_Q)
    for exhaustive in (True, False):
        fig = run(exhaustive=exhaustive)
        core = [p for u, p in fake.posts if "core.ac.uk" in u]
        assert core and all(p.get("q") == whole for p in core), (exhaustive, core[:1])
        assert "core" not in (fig.get("keyword_fallback_sources") or []), exhaustive


def test_a_standard_search_still_sends_the_reduced_query_and_says_so(federation):
    run, fake = federation
    fig = run(exhaustive=False)
    short = _shorten_boolean(LONG_Q, URL_QUERY_MAX_ENCODED, measure=_url_encoded_len)
    assert set(fake.params("clinicaltrials", "query.term")) == {short}
    assert set(fake.params("openalex", "search")) == {short}
    assert (fig.get("split_queries") or {}) == {}
    assert {"openalex", "clinicaltrials", "arxiv", "doaj", "openaire"} <= set(fig.get("keyword_fallback_sources") or [])


def test_pubmed_always_gets_the_union_and_a_long_one_goes_out_as_post(federation):
    run, fake = federation
    run(exhaustive=False)
    esearch_posts = [p for u, p in fake.posts if "esearch.fcgi" in u]
    esearch_gets = [p for u, p in fake.gets if "esearch.fcgi" in u]
    assert esearch_posts and not esearch_gets, "une union de plus de 1 800 caractères part dans le corps"
    term = esearch_posts[0]["term"]
    assert term.startswith(f"({MESH_Q}) OR (") and "poultry worker exposure cohort number 45" in term


def test_openaire_gets_a_short_strategy_split_by_operators(federation):
    run, fake = federation
    fig = run(general=SHORT_Q, pubmed=SHORT_Q, exhaustive=True)
    parts = _split_boolean(SHORT_Q, URL_QUERY_MAX_ENCODED, measure=_url_encoded_len,
                           max_operators=OPENAIRE_MAX_OPERATORS)
    assert (fig.get("split_queries") or {}).get("openaire") == parts
    assert set(fake.params("openaire", "search")) == set(parts)
    assert "openaire" not in (fig.get("keyword_fallback_sources") or [])
