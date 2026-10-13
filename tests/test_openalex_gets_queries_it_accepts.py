"""OpenAlex receives only queries it accepts, and a part it refuses is cut in two and asked again.

On 12 October 2026 the exposure/transmission strategy of the T2.4 HPAI scenario, split in four
parts at the URL limit, lost its largest part: 60 operators, 1,548 characters, refused with a
400, while the parts of 23 to 47 operators went through. The 400 raised, OpenAlex ended in
`error`, the parts after it were never asked, and the PRISMA card marked the source as failed.
Now the parts are bounded in operators as well as in length, and a refused one is bisected
(`_bisect_boolean`) and asked again, up to OPENALEX_BISECT_DEPTH times."""
import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")

from sqlalchemy import text  # noqa: E402

import main  # noqa: E402
from api.search import (  # noqa: E402
    OPENALEX_MAX_OPERATORS,
    OPENALEX_SPLIT_MAX_PARTS,
    URL_QUERY_MAX_ENCODED,
    _bisect_boolean,
    _clean_boolean,
    _count_operators,
    _split_boolean,
    _url_encoded_len,
)
from search_tables import ensure_search_tables  # noqa: E402
from test_a_long_query_is_split_not_cut import _EmptySources, _Resp  # noqa: E402

#: Sixty exposures and a virus block: about 63 operators, short enough for one URL.
_EXP = [f"exposure{i:02d}" for i in range(60)]
WIDE_Q = "(" + " OR ".join(_EXP) + ') AND ("avian influenza" OR h5n1 OR h7n9) NOT review'


# ── the pure half ────────────────────────────────────────────────────────────

def test_a_refused_query_is_cut_in_two_whose_union_is_the_query():
    halves = _bisect_boolean(_clean_boolean(WIDE_Q))
    assert len(halves) == 2
    for h in halves:                                  # each keeps the other blocks and the exclusion
        assert "h5n1" in h and "NOT review" in h
    # The widest block is shared out, every term in exactly one half.
    owners = [[t for t in _EXP if f"{t} " in f"{h} " or f"{t})" in h] for h in halves]
    assert sorted(owners[0] + owners[1]) == sorted(_EXP) and not set(owners[0]) & set(owners[1])
    assert _bisect_boolean('"one single term"') == [] and _bisect_boolean("") == []


def test_the_openalex_parts_are_bounded_in_operators_as_well_as_length():
    clean = _clean_boolean(WIDE_Q)
    assert _url_encoded_len(clean) <= URL_QUERY_MAX_ENCODED < 10_000
    assert _count_operators(clean) > OPENALEX_MAX_OPERATORS
    parts = _split_boolean(WIDE_Q, URL_QUERY_MAX_ENCODED, measure=_url_encoded_len,
                           max_operators=OPENALEX_MAX_OPERATORS, max_parts=OPENALEX_SPLIT_MAX_PARTS)
    assert len(parts) >= 2 and all(_count_operators(p) <= OPENALEX_MAX_OPERATORS for p in parts)


# ── the federation, against an OpenAlex that refuses wide queries ─────────────

class _PickyOpenAlex(_EmptySources):
    """OpenAlex refusing any query of more than `limit` operators, as it did on 12 October."""

    def __init__(self, limit: int):
        super().__init__()
        self.limit = limit
        self.refused: list[str] = []

    def get(self, url, params=None, **kw):
        if "openalex" not in str(url):
            return super().get(url, params, **kw)
        params = dict(params or {})
        self.gets.append((str(url), params))
        q = params.get("search") or str(params.get("filter") or "").split(":", 1)[-1]
        if _count_operators(q) > self.limit:
            self.refused.append(q)
            return _Resp(400, {"error": "Invalid query parameters"})
        return _Resp(200, {"results": [], "meta": {"count": 0, "next_cursor": None}})


SID = "usr-openalexops01"


@pytest.fixture
def picky(db_conn, monkeypatch):
    import requests as _real_requests

    from api import pipeline as P
    from api import relevance as R

    created_chunk_table = ensure_search_tables(db_conn.cursor())
    monkeypatch.setenv("CORE_API_KEY", "test-core-key")
    monkeypatch.setattr(R, "_run_semantic_rerank_inline", lambda *a, **k: 0)
    monkeypatch.setattr(R, "_run_cross_encoder_rerank", lambda *a, **k: {})
    monkeypatch.setattr(P, "_run_clustering_background", lambda *a, **k: None)
    monkeypatch.setattr(P, "_precompute_user_kg", lambda *a, **k: None)

    def run(limit: int, query: str = WIDE_Q):
        fake = _PickyOpenAlex(limit)
        monkeypatch.setattr(_real_requests, "get", fake.get)
        monkeypatch.setattr(_real_requests, "post", fake.post)
        with main.engine.begin() as c:
            c.execute(text("DELETE FROM article_scenarios WHERE scenario_id = :s"), {"s": SID})
            c.execute(text("DELETE FROM user_scenarios WHERE id = :s"), {"s": SID})
            c.execute(text("DELETE FROM source_query_cache"))
            c.execute(text("INSERT INTO user_scenarios (id, name, query, article_count, search_strategy) "
                           "VALUES (:s, 'wide', 'avian influenza exposure', 0, CAST(:st AS jsonb))"),
                      {"s": SID, "st": json.dumps({"general": query, "pubmed": query,
                                                   "explanation": "", "synonyms": []})})
        P._run_user_scenario_populate(SID, "avian influenza exposure", {}, max_results=50, include_live=True,
                                      auto_pipeline=False, force_live=True)
        return fake, (P._user_scenario_populate_jobs.get(SID) or {}).get("prisma_identification") or {}

    yield run
    with main.engine.begin() as c:
        c.execute(text("DELETE FROM article_scenarios WHERE scenario_id = :s"), {"s": SID})
        c.execute(text("DELETE FROM user_scenarios WHERE id = :s"), {"s": SID})
        c.execute(text("DELETE FROM source_query_cache"))
    if created_chunk_table:
        with db_conn.cursor() as cur:
            cur.execute("DROP TABLE document_chunk")


def test_a_query_that_fits_the_url_but_has_too_many_operators_is_split(picky):
    fake, fig = picky(limit=OPENALEX_MAX_OPERATORS)
    asked = [p.get("search") or p.get("filter") for u, p in fake.gets if "openalex" in u]
    assert asked and not fake.refused
    assert (fig.get("source_outcomes") or {}).get("openalex") in ("ok", "empty"), fig.get("source_outcomes")
    parts = (fig.get("split_queries") or {}).get("openalex") or []
    assert len(parts) >= 2 and all(_count_operators(p) <= OPENALEX_MAX_OPERATORS for p in parts)
    assert "openalex" not in (fig.get("keyword_fallback_sources") or [])


def test_a_refused_part_is_cut_in_two_and_asked_again_and_the_source_does_not_fail(picky):
    # Stricter than our bound: every first part is refused, its halves go through.
    fake, fig = picky(limit=20)
    assert fake.refused, "the stricter OpenAlex must have refused the first parts"
    assert (fig.get("source_outcomes") or {}).get("openalex") in ("ok", "empty"), fig.get("source_outcomes")
    parts = (fig.get("split_queries") or {}).get("openalex") or []
    assert parts and all(_count_operators(p) <= 20 for p in parts), [(_count_operators(p)) for p in parts]
    asked = {p.get("search") or str(p.get("filter") or "").split(":", 1)[-1] for u, p in fake.gets if "openalex" in u}
    assert {p.replace(",", " ") for p in parts} <= {a.replace(",", " ") for a in asked}
