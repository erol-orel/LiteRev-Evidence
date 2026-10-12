"""Every OpenAlex call carries the server's key when it holds one, and no key reaches a log line
or a stored error.

On 12 October 2026 OpenAlex answered 429 "Insufficient budget" to the exposure search of the
T2.4 HPAI scenario after 1,594 records: without a key, a call draws on a small daily budget
shared by every client behind the server's IP address, and the day's searches had spent it.
A free key gives the server a budget of its own. OpenAlex takes it in the URL (`api_key=`),
and an HTTP error from requests quotes its URL: the reason a source failed is stored with the
search and served by its PRISMA card, so it passes through `redact_keys` first."""
import json
import logging
import re
from pathlib import Path
from urllib.parse import quote, urlencode

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")

from sqlalchemy import text  # noqa: E402

import main  # noqa: E402
from api.core import logger as api_logger, redact_keys  # noqa: E402
from api.sources import OPENALEX_MAILTO, openalex_params  # noqa: E402
from conftest import patch_app  # noqa: E402
from search_tables import ensure_search_tables  # noqa: E402
from test_a_long_query_is_split_not_cut import _EmptySources, _Resp  # noqa: E402

KEY = "oa-test-key-0123456789"
API = Path(__file__).resolve().parents[1] / "api"


# ── the pure half ────────────────────────────────────────────────────────────

def test_the_key_goes_with_every_call_only_when_the_server_holds_one(monkeypatch):
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    p = openalex_params({"search": "h5n1", "per-page": 200})
    assert p == {"mailto": OPENALEX_MAILTO, "search": "h5n1", "per-page": 200}
    monkeypatch.setenv("OPENALEX_API_KEY", f"  {KEY} ")
    p = openalex_params({"search": "h5n1"})
    assert p["api_key"] == KEY and p["mailto"] == OPENALEX_MAILTO and p["search"] == "h5n1"
    assert openalex_params() == {"mailto": OPENALEX_MAILTO, "api_key": KEY}


def test_a_key_is_redacted_from_an_error_text_raw_or_encoded(monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", KEY)
    msg = ("429 Client Error: Too Many Requests for url: "
           f"https://api.openalex.org/works?api_key={KEY}&search=h5n1&mailto=x%40y.org")
    clean = redact_keys(msg)
    assert KEY not in clean and "api_key=<redacted>" in clean and "search=h5n1" in clean
    # The value anywhere else, as typed or as a URL encodes it.
    odd = "oa+key/with=signs-0123"
    monkeypatch.setenv("OPENALEX_API_KEY", odd)
    for form in (odd, quote(odd, safe="")):
        assert odd not in redact_keys(f"key {form} in a sentence")
        assert quote(odd, safe="") not in redact_keys(f"key {form} in a sentence")
    assert redact_keys(None) == "" and redact_keys("nothing secret") == "nothing secret"


def test_a_key_never_reaches_a_line_of_the_api_log(monkeypatch, caplog):
    monkeypatch.setenv("OPENALEX_API_KEY", KEY)
    with caplog.at_level(logging.WARNING, logger="literev-api"):
        api_logger.warning("OpenAlex populate usr-x: %s",
                           f"403 Client Error: Forbidden for url: https://api.openalex.org/works?api_key={KEY}")
    assert caplog.records and KEY not in caplog.text and "api_key=<redacted>" in caplog.text


def test_every_openalex_call_of_the_api_goes_through_openalex_params():
    """A guard on the code itself: an OpenAlex URL with no `openalex_params` near it is a call
    that goes without the key (and without `mailto`)."""
    missing = []
    for path in sorted(API.glob("*.py")):
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if "api.openalex.org" not in line or line.lstrip().startswith("#"):
                continue
            window = "\n".join(lines[max(0, i - 12):i + 6])
            if "openalex_params(" not in window:
                missing.append(f"{path.name}:{i + 1}")
    assert not missing, f"OpenAlex called without openalex_params: {missing}"


def test_the_full_text_download_sends_the_semantic_scholar_and_openalex_keys(monkeypatch):
    from api import pipeline as P

    monkeypatch.setenv("OPENALEX_API_KEY", KEY)
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "s2-test-key-0123456789")
    calls: list[tuple[str, dict, dict]] = []

    def fake_get(url, params=None, timeout=20, headers=None):
        calls.append((str(url), dict(params or {}), dict(headers or {})))
        return None

    patch_app(monkeypatch, "_ft_get", fake_get)
    patch_app(monkeypatch, "_ft_resolve_pmcid", lambda *a, **k: None)
    text_, source, fails = P._fulltext_of({"external_id": "W2741809807", "doi": "10.1000/xyz", "pmid": "123"})
    assert text_ is None and source is None
    s2 = [c for c in calls if "semanticscholar" in c[0]]
    oa = [c for c in calls if "openalex" in c[0]]
    assert s2 and s2[0][2].get("x-api-key") == "s2-test-key-0123456789"
    assert oa and oa[0][1].get("api_key") == KEY and oa[0][1].get("mailto") == OPENALEX_MAILTO
    # Without the Semantic Scholar key, no empty header is sent.
    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY")
    calls.clear()
    P._fulltext_of({"external_id": "W2741809807", "doi": "10.1000/xyz", "pmid": "123"})
    assert "x-api-key" not in [c for c in calls if "semanticscholar" in c[0]][0][2]


# ── the search, against an OpenAlex that refuses every call ───────────────────

class _ForbiddenResp(_Resp):
    """A refusal whose error, like requests', quotes the URL it was asked, key included."""

    def __init__(self, url: str, params: dict):
        super().__init__(403, {"error": "Forbidden"})
        self._url = f"{url}?{urlencode(params)}"

    def raise_for_status(self):
        import requests
        raise requests.HTTPError(f"403 Client Error: Forbidden for url: {self._url}")


class _RefusingOpenAlex(_EmptySources):
    def get(self, url, params=None, **kw):
        if "openalex" not in str(url):
            return super().get(url, params, **kw)
        params = dict(params or {})
        self.gets.append((str(url), params))
        # The key first, so that the stored reason, cut at 200 characters, would carry it.
        ordered = {"api_key": params.get("api_key"), **{k: v for k, v in params.items() if k != "api_key"}}
        return _ForbiddenResp(str(url), ordered)


SID = "usr-openalexkey01"
QUERY = '(h5n1 OR "avian influenza") AND exposure'


@pytest.fixture
def refusing(db_conn, monkeypatch):
    import requests as _real_requests

    from api import pipeline as P
    from api import relevance as R

    created_chunk_table = ensure_search_tables(db_conn.cursor())
    monkeypatch.setenv("CORE_API_KEY", "test-core-key")
    monkeypatch.setenv("OPENALEX_API_KEY", KEY)
    monkeypatch.setattr(R, "_run_semantic_rerank_inline", lambda *a, **k: 0)
    monkeypatch.setattr(R, "_run_cross_encoder_rerank", lambda *a, **k: {})
    monkeypatch.setattr(P, "_run_clustering_background", lambda *a, **k: None)
    monkeypatch.setattr(P, "_precompute_user_kg", lambda *a, **k: None)
    fake = _RefusingOpenAlex()
    monkeypatch.setattr(_real_requests, "get", fake.get)
    monkeypatch.setattr(_real_requests, "post", fake.post)
    with main.engine.begin() as c:
        c.execute(text("DELETE FROM article_scenarios WHERE scenario_id = :s"), {"s": SID})
        c.execute(text("DELETE FROM user_scenarios WHERE id = :s"), {"s": SID})
        c.execute(text("DELETE FROM source_query_cache"))
        c.execute(text("INSERT INTO user_scenarios (id, name, query, article_count, search_strategy) "
                       "VALUES (:s, 'key', 'avian influenza exposure', 0, CAST(:st AS jsonb))"),
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


def test_the_search_sends_the_key_and_the_stored_reason_of_a_refusal_does_not_carry_it(refusing, caplog):
    fake, P = refusing
    with caplog.at_level(logging.WARNING, logger="literev-api"):
        P._run_user_scenario_populate(SID, "avian influenza exposure", {}, max_results=50, include_live=True,
                                      auto_pipeline=False, force_live=True, exhaustive=True)
    asked = [p for u, p in fake.gets if "openalex" in u]
    assert asked and all(p.get("api_key") == KEY and p.get("mailto") == OPENALEX_MAILTO for p in asked)
    fig = (P._user_scenario_populate_jobs.get(SID) or {}).get("prisma_identification") or {}
    assert (fig.get("source_outcomes") or {}).get("openalex") == "error", fig.get("source_outcomes")
    reason = (fig.get("source_error_reasons") or {}).get("openalex") or ""
    assert "403" in reason and KEY not in reason and re.search(r"api_key=<redacted>", reason), reason
    # Nor in the figures stored with the scenario, nor in the log.
    with main.engine.connect() as c:
        has_column = c.execute(text("SELECT 1 FROM information_schema.columns WHERE table_name = 'user_scenarios' "
                                    "AND column_name = 'prisma_identification'")).first()
        stored = (c.execute(text("SELECT prisma_identification::text FROM user_scenarios WHERE id = :s"),
                            {"s": SID}).scalar() if has_column else "")
    assert KEY not in (stored or "")
    assert KEY not in caplog.text
