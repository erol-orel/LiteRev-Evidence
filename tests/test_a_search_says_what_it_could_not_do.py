"""Trois petites choses que la recherche taisait, et le budget de sa fédération.

- Le panneau de recherche en direct fondait deux notices de même titre exact, sans
  longueur minimale : deux « Editorial » sans DOI n'en faisaient qu'un.
- L'endpoint classique de Semantic Scholar ne rend pas plus de 1 000 notices, mais la
  carte jugeait « plafonnée » contre le plafond commun de 2 000.
- Les correspondances locales sont lues avec un LIMIT sans ORDER BY : atteint, le corpus
  serait un sous-ensemble arbitraire, et rien ne le disait.
- Le budget de la fédération passe de 180 à 600 secondes.
"""
import inspect
import logging
import os

import pytest

pytest.importorskip("fastapi")

import main  # noqa: E402
from api.sources import _live_dedup_key  # noqa: E402
from conftest import patch_app  # noqa: E402
from search_tables import ensure_search_tables  # noqa: E402


@pytest.fixture()
def tables(db_conn):
    created_chunk_table = ensure_search_tables(db_conn.cursor())
    yield db_conn
    if created_chunk_table:
        with db_conn.cursor() as cur:
            cur.execute("DROP TABLE document_chunk")


def test_the_live_panel_merges_only_long_titles_or_equal_dois():
    assert _live_dedup_key({"title": "Editorial"}) is None
    assert _live_dedup_key({"title": "Correction."}) is None
    long_title = "Avian influenza among poultry workers: a cohort"
    assert _live_dedup_key({"title": long_title}) == _live_dedup_key({"title": long_title.upper() + "!"})
    assert _live_dedup_key({"title": "Editorial", "doi": "10.1/A"}) == _live_dedup_key({"doi": "10.1/a "})
    assert _live_dedup_key({"title": long_title, "doi": "10.1/a"}) != _live_dedup_key({"title": long_title})


def test_the_classic_semantic_scholar_path_is_capped_at_its_own_limit():
    figures = main._prisma_identification_figures(
        {"semantic_scholar": 1000}, 1000, 0, 1000, per_source_cap=2000,
        source_outcomes={"_fetch_semantic_scholar": "ok"}, source_totals={"semantic_scholar": 1500},
        source_caps={"semantic_scholar": 1000})
    assert figures["sources_capped"] == ["semantic_scholar"], "1 500 annoncées, 1 000 rendues : plafonnée"
    from api import pipeline as P
    src = inspect.getsource(P._run_user_scenario_populate)
    assert '_source_caps_run["semantic_scholar"] = min(int(max_results), 1000)' in src
    assert "dict(_source_caps_run)" in src


def test_a_local_match_at_its_limit_says_so(tables, monkeypatch, caplog):
    from api import search as S
    db_conn = tables
    monkeypatch.setenv("LEXICAL_SEARCH_ENGINE", "like")
    patch_app(monkeypatch, "LOCAL_MATCH_LIMIT", 2)
    abstract = "Numbatpox in captive marsupials, a long enough abstract for the test."
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM literature_document WHERE id BETWEEN 9890 AND 9899")
        for doc_id in (9890, 9891, 9892):
            cur.execute("INSERT INTO literature_document (id, title, abstract, year, source, project_context, "
                        "is_duplicate) VALUES (%s, %s, %s, 2024, 'crossref', 'literev', false)",
                        (doc_id, f"Numbatpox report {doc_id}", abstract))
    try:
        with caplog.at_level(logging.WARNING, logger="literev-api"):
            ids = S._boolean_corpus_ids("numbatpox", {})
        assert len(ids) == 2 and ids.capped is True
        assert "limit of 2 matches was reached" in caplog.text
        multi = S._multi_query_corpus_ids([{"kind": "boolean", "text": "numbatpox"},
                                           {"kind": "boolean", "text": "marsupials", "op": "or"}], "union", {})
        assert multi.capped is True
        assert S._boolean_corpus_ids("numbatpox AND nothingmatchesthis", {}).capped is False
        figures = main._prisma_identification_figures({"db_cache": 2}, 2, 0, 2, local_match_capped=True)
        assert figures["local_match_capped"] is True
    finally:
        with db_conn.cursor() as cur:
            cur.execute("DELETE FROM literature_document WHERE id BETWEEN 9890 AND 9899")


def test_the_facet_preview_says_when_its_match_was_capped(tables, monkeypatch):
    from fastapi.testclient import TestClient
    db_conn = tables
    monkeypatch.setenv("LEXICAL_SEARCH_ENGINE", "like")
    patch_app(monkeypatch, "LOCAL_MATCH_LIMIT", 1)
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM literature_document WHERE id BETWEEN 9895 AND 9899")
        for doc_id in (9895, 9896):
            cur.execute("INSERT INTO literature_document (id, title, abstract, year, source, project_context, "
                        "is_duplicate) VALUES (%s, %s, 'Quollpox in the wild, an abstract long enough.', "
                        "2024, 'crossref', 'literev', false)", (doc_id, f"Quollpox report {doc_id}"))
    try:
        body = TestClient(main.app).post(
            "/search-facets", headers={"X-API-Key": "test-write-key"},
            json={"sub_queries": [{"kind": "boolean", "text": "quollpox"},
                                  {"kind": "boolean", "text": "quollpox AND wild", "op": "or"}],
                  "combinator": "union"}).json()
        assert body["capped"] is True and all(f["capped"] for f in body["facets"]), body
    finally:
        with db_conn.cursor() as cur:
            cur.execute("DELETE FROM literature_document WHERE id BETWEEN 9895 AND 9899")


def test_the_federation_budget_defaults_to_ten_minutes():
    from api import core
    src = inspect.getsource(core)
    assert 'os.getenv("POPULATE_FEDERATION_BUDGET", "600")' in src
    assert "POPULATE_FEDERATION_BUDGET = 600.0" in src
    if os.getenv("POPULATE_FEDERATION_BUDGET") is None:
        assert core.POPULATE_FEDERATION_BUDGET == 600.0
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = open(os.path.join(root, ".env.example"), encoding="utf-8").read()
    assert "POPULATE_FEDERATION_BUDGET=600" in env and "POPULATE_FEDERATION_BUDGET=180" not in env
