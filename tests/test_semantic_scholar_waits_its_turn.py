"""Every Semantic Scholar call waits its turn: with the server's key, one request a second for
the whole API, the searches and the full-text download together.

On 13 October 2026, the key installed, the exposure search of the T2.4 HPAI scenario still
stopped on five 429 in a row after 299 of 10,257 records. Semantic Scholar allows a key one
request a second on all its endpoints together, and two searches and a full-text download were
sharing it, each pacing itself (0.3 s between pages) without knowing of the others."""
import json
import threading
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")

from sqlalchemy import text  # noqa: E402

import main  # noqa: E402
from api import sources as S  # noqa: E402
from conftest import patch_app  # noqa: E402
from search_tables import ensure_search_tables  # noqa: E402
from test_a_long_query_is_split_not_cut import _EmptySources, _Resp  # noqa: E402

KEY = "s2-test-key-0123456789"


# ── the pacer ────────────────────────────────────────────────────────────────

def test_without_the_key_nothing_waits_and_no_header_is_sent(monkeypatch):
    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
    monkeypatch.setattr(S, "S2_MIN_INTERVAL_S", 5.0)
    t0 = time.monotonic()
    assert [S.s2_headers_paced() for _ in range(3)] == [{}, {}, {}]
    assert time.monotonic() - t0 < 1.0


def test_with_the_key_callers_on_several_threads_leave_one_interval_apart(monkeypatch):
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", KEY)
    monkeypatch.setattr(S, "S2_MIN_INTERVAL_S", 0.2)
    monkeypatch.setattr(S, "_S2_LAST", [0.0])
    left: list[tuple[float, dict]] = []
    lock = threading.Lock()

    def call():
        h = S.s2_headers_paced()
        with lock:
            left.append((time.monotonic(), h))
    threads = [threading.Thread(target=call) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    times = sorted(t for t, _h in left)
    assert [h for _t, h in left] == [{"x-api-key": KEY}] * 5
    # Five callers at once leave over four intervals, not together (0.12: room for a slow machine).
    assert min(b - a for a, b in zip(times, times[1:])) >= 0.12 and times[-1] - times[0] >= 0.6


# ── every call site asks the pacer, once per request ─────────────────────────

@pytest.fixture
def paced(monkeypatch):
    """The pacer replaced by a counter: each request must carry the header of its own call."""
    asked: list[str] = []

    def fake_pacer():
        asked.append(f"paced-{len(asked) + 1}")
        return {"x-api-key": asked[-1]}
    patch_app(monkeypatch, "s2_headers_paced", fake_pacer)
    return asked


def test_the_full_text_lookup_waits_its_turn(paced, monkeypatch):
    from api import pipeline as P
    calls: list[tuple[str, dict]] = []

    def fake_get(url, params=None, timeout=20, headers=None):
        calls.append((str(url), dict(headers or {})))
        return None
    patch_app(monkeypatch, "_ft_get", fake_get)
    patch_app(monkeypatch, "_ft_resolve_pmcid", lambda *a, **k: None)
    monkeypatch.delenv("CORE_API_KEY", raising=False)
    P._fulltext_of({"external_id": "s2:0b8a1d0257d1784a5f8b3fe7ecb21d3e21e4fc5c", "doi": None, "pmid": None})
    s2 = [h for u, h in calls if "semanticscholar" in u]
    assert s2 == [{"x-api-key": "paced-1"}] and paced == ["paced-1"]


class _TwoPagesOfS2(_EmptySources):
    """Semantic Scholar's bulk search answers two pages; every other base answers nothing."""

    def __init__(self):
        super().__init__()
        self.s2: list[dict] = []

    def get(self, url, params=None, **kw):
        if "semanticscholar" not in str(url):
            return super().get(url, params, **kw)
        self.s2.append(dict(kw.get("headers") or {}))
        page = len(self.s2)
        return _Resp(200, {"total": 2, "token": "next" if page == 1 else None,
                           "data": [{"paperId": f"p{page}", "title": f"H5N1 exposure study {page}",
                                     "abstract": "Poultry workers exposed to H5N1. " * 3, "year": 2020}]})


SID = "usr-s2pacer0001"
QUERY = '(h5n1 OR "avian influenza") AND exposure'


@pytest.fixture
def two_pages(db_conn, monkeypatch):
    import requests as _real_requests

    from api import pipeline as P
    from api import relevance as R

    created_chunk_table = ensure_search_tables(db_conn.cursor())
    monkeypatch.setattr(R, "_run_semantic_rerank_inline", lambda *a, **k: 0)
    monkeypatch.setattr(R, "_run_cross_encoder_rerank", lambda *a, **k: {})
    monkeypatch.setattr(P, "_run_clustering_background", lambda *a, **k: None)
    monkeypatch.setattr(P, "_precompute_user_kg", lambda *a, **k: None)
    fake = _TwoPagesOfS2()
    monkeypatch.setattr(_real_requests, "get", fake.get)
    monkeypatch.setattr(_real_requests, "post", fake.post)
    with main.engine.begin() as c:
        c.execute(text("DELETE FROM article_scenarios WHERE scenario_id = :s"), {"s": SID})
        c.execute(text("DELETE FROM user_scenarios WHERE id = :s"), {"s": SID})
        c.execute(text("DELETE FROM source_query_cache"))
        c.execute(text("INSERT INTO user_scenarios (id, name, query, article_count, search_strategy) "
                       "VALUES (:s, 'pacer', 'avian influenza exposure', 0, CAST(:st AS jsonb))"),
                  {"s": SID, "st": json.dumps({"general": QUERY, "pubmed": QUERY,
                                               "explanation": "", "synonyms": []})})
    yield fake, P
    with main.engine.begin() as c:
        c.execute(text("DELETE FROM article_scenarios WHERE scenario_id = :s"), {"s": SID})
        c.execute(text("DELETE FROM user_scenarios WHERE id = :s"), {"s": SID})
        c.execute(text("DELETE FROM source_query_cache"))
    if created_chunk_table:
        with db_conn.cursor() as cur:
            cur.execute("DROP TABLE document_chunk")


def test_each_page_of_the_search_waits_its_turn(paced, two_pages):
    fake, P = two_pages
    P._run_user_scenario_populate(SID, "avian influenza exposure", {}, max_results=50, include_live=True,
                                  auto_pipeline=False, force_live=True, exhaustive=True)
    assert len(fake.s2) == 2, fake.s2
    assert fake.s2 == [{"x-api-key": "paced-1"}, {"x-api-key": "paced-2"}] and paced == ["paced-1", "paced-2"]
