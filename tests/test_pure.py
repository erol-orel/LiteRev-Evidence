"""Pure-logic unit tests — no database or network required.

These cover the exact helpers behind recent production bugs:
- `_job_is_active`  — the stale-job guard that stops "already_running" lock-outs.
- `_llm_lang_directive` — FR/EN LLM output-language switch.
- `_build_where` — the /search filter, incl. the Migration-1 Way-B scoping.
"""
import time

import main


# ── _job_is_active (stale-job guard) ─────────────────────────────────────────
def test_job_is_active_fresh_running():
    assert main._job_is_active({"status": "running", "started_at": time.time()}) is True


def test_job_is_active_stale_running():
    # A "running" entry older than the staleness window is treated as dead.
    assert main._job_is_active({"status": "running", "started_at": 0}) is False


def test_job_is_active_none_or_missing():
    assert main._job_is_active(None) is False
    assert main._job_is_active({}) is False


def test_job_is_active_not_running_status():
    assert main._job_is_active({"status": "done", "started_at": time.time()}) is False
    assert main._job_is_active({"status": "error", "started_at": time.time()}) is False


# ── _llm_lang_directive (FR/EN output language) ──────────────────────────────
def test_lang_directive_english():
    assert "ENGLISH" in main._llm_lang_directive("en")
    assert "ENGLISH" in main._llm_lang_directive("EN-GB")


def test_lang_directive_defaults_to_french():
    for value in ("fr", None, "", "de", "es"):
        assert "français" in main._llm_lang_directive(value).lower()


def test_lang_directive_non_string_falls_back_to_french():
    # Regression for "'Query' object has no attribute 'strip'": a non-string —
    # e.g. a FastAPI Query object leaked by an internal direct call to an endpoint
    # whose param defaults to Query(...) — must NOT crash, and falls back to French.
    from fastapi import Query
    assert "français" in main._llm_lang_directive(Query(None)).lower()
    assert "français" in main._llm_lang_directive(object()).lower()


# ── _build_where (/search filter) ────────────────────────────────────────────
def test_build_where_empty():
    assert main._build_where({}) == ("", {})
    assert main._build_where(None) == ("", {})


def test_build_where_simple_equality():
    where_sql, params = main._build_where({"source": "pubmed"})
    assert "d.source = :source" in where_sql
    assert params["source"] == "pubmed"


def test_build_where_scenario_type_is_way_b_membership():
    # Migration 1: a scenario_type filter must scope by article_scenarios
    # membership (Way B), NOT the legacy global d.scenario_type column.
    where_sql, params = main._build_where({"scenario_type": "sc-a"})
    assert "article_scenarios" in where_sql
    assert "EXISTS" in where_sql
    assert "d.scenario_type = :scenario_type" not in where_sql  # not the legacy equality
    assert params["scenario_type"] == "sc-a"


def test_build_where_project_context_normalized():
    # Legacy project contexts collapse to 'literev'.
    _where_sql, params = main._build_where({"project_context": "gesica"})
    assert params["project_context"] == "literev"


def test_build_where_ignores_blank_values():
    where_sql, params = main._build_where({"source": "", "disease_or_condition": None})
    assert where_sql == ""
    assert params == {}


# ── _normalize_sub_queries (multi-query cleaning) ────────────────────────────
def test_normalize_sub_queries_filters_and_defaults():
    raw = [
        {"kind": "boolean", "text": "  cardiac arrest  "},  # trimmed
        {"kind": "natural", "text": "bystander CPR"},
        {"kind": "weird", "text": "x"},                     # unknown kind → natural
        {"kind": "boolean", "text": "   "},                  # blank text → dropped
        {"text": "no kind"},                                 # missing kind → natural
        "not a dict",                                        # ignored
    ]
    assert main._normalize_sub_queries(raw) == [
        {"kind": "boolean", "text": "cardiac arrest"},
        {"kind": "natural", "text": "bystander CPR"},
        {"kind": "natural", "text": "x"},
        {"kind": "natural", "text": "no kind"},
    ]


def test_normalize_sub_queries_non_list():
    assert main._normalize_sub_queries(None) == []
    assert main._normalize_sub_queries("cardiac") == []
    assert main._normalize_sub_queries({}) == []


# ── _multi_query_corpus_ids (union / intersection of doc-id sets) ─────────────
def _patch_local_ids(monkeypatch, mapping):
    """Stub _search_local_doc_ids so each sub-query text maps to a fixed id set."""
    def _fake(query, mode, filters, limit=10_000, threshold=0.45):
        return list(mapping.get(query, []))
    monkeypatch.setattr(main, "_search_local_doc_ids", _fake)


def test_multi_query_union(monkeypatch):
    _patch_local_ids(monkeypatch, {"A": [1, 2, 3], "B": [3, 4]})
    sub = [{"kind": "boolean", "text": "A"}, {"kind": "natural", "text": "B"}]
    assert sorted(main._multi_query_corpus_ids(sub, "union", {})) == [1, 2, 3, 4]


def test_multi_query_intersection(monkeypatch):
    _patch_local_ids(monkeypatch, {"A": [1, 2, 3], "B": [3, 4]})
    sub = [{"kind": "boolean", "text": "A"}, {"kind": "natural", "text": "B"}]
    assert main._multi_query_corpus_ids(sub, "intersection", {}) == [3]


def test_multi_query_intersection_disjoint_is_empty(monkeypatch):
    _patch_local_ids(monkeypatch, {"A": [1, 2], "B": [3, 4]})
    sub = [{"kind": "boolean", "text": "A"}, {"kind": "boolean", "text": "B"}]
    assert main._multi_query_corpus_ids(sub, "intersection", {}) == []


def test_multi_query_unknown_combinator_is_intersection(monkeypatch):
    # Anything that isn't "union" narrows (intersection = the safe default).
    _patch_local_ids(monkeypatch, {"A": [1, 2, 3], "B": [3, 4]})
    sub = [{"kind": "boolean", "text": "A"}, {"kind": "natural", "text": "B"}]
    assert main._multi_query_corpus_ids(sub, "banana", {}) == [3]


def test_multi_query_all_blank_is_empty(monkeypatch):
    _patch_local_ids(monkeypatch, {})
    assert main._multi_query_corpus_ids([{"kind": "boolean", "text": "  "}], "union", {}) == []
