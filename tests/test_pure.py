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
    """Stub _search_local_doc_ids so each sub-query text maps to a fixed id set,
    and stub the NL→boolean translator to identity (a natural facet is translated
    to boolean before matching — the identity stub keeps the mapping keyed on the
    raw text AND avoids any network call in the unit test)."""
    monkeypatch.setattr(main, "_generate_search_strategy",
                        lambda q: {"general": q, "pubmed": q}, raising=True)
    def _fake(query, mode, filters, limit=10_000, threshold=0.45):
        # Membership is now ALWAYS lexical/boolean — no facet uses semantic mode.
        assert mode == "boolean", f"corpus membership must be boolean, got mode={mode!r}"
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


def test_multi_query_unknown_combinator_is_union(monkeypatch):
    # Anything that isn't "intersection" broadens (union = the default).
    _patch_local_ids(monkeypatch, {"A": [1, 2, 3], "B": [3, 4]})
    sub = [{"kind": "boolean", "text": "A"}, {"kind": "natural", "text": "B"}]
    assert sorted(main._multi_query_corpus_ids(sub, "banana", {})) == [1, 2, 3, 4]


def test_multi_query_all_blank_is_empty(monkeypatch):
    _patch_local_ids(monkeypatch, {})
    assert main._multi_query_corpus_ids([{"kind": "boolean", "text": "  "}], "union", {}) == []


def test_multi_query_natural_facet_is_translated_to_boolean(monkeypatch):
    # The heart of the fix: a NATURAL sub-query is translated to boolean (via
    # _generate_search_strategy) and matched LEXICALLY — the semantic threshold
    # never decides corpus membership. A boolean sub-query is used verbatim.
    seen_modes: list[str] = []

    def _fake_translate(q):
        # natural text "flu surges" → an explicit boolean; anything else unchanged
        return {"general": '("influenza" OR ILI)' if q == "flu surges" else q}

    def _fake_local(query, mode, filters, limit=10_000, threshold=0.45):
        seen_modes.append(mode)
        table = {'("influenza" OR ILI)': [1, 2], "cardiac AND arrest": [2, 3]}
        return list(table.get(query, []))

    monkeypatch.setattr(main, "_generate_search_strategy", _fake_translate, raising=True)
    monkeypatch.setattr(main, "_search_local_doc_ids", _fake_local, raising=True)

    sub = [{"kind": "natural", "text": "flu surges"},
           {"kind": "boolean", "text": "cardiac AND arrest"}]
    # natural facet resolved via its TRANSLATED boolean → {1,2}; boolean facet → {2,3}
    assert sorted(main._multi_query_corpus_ids(sub, "union", {})) == [1, 2, 3]
    assert main._multi_query_corpus_ids(sub, "intersection", {}) == [2]
    # every membership lookup ran in boolean mode — semantic never touches the corpus
    assert set(seen_modes) == {"boolean"}


def test_multi_query_translation_failure_falls_back_to_raw_text(monkeypatch):
    # If NL→boolean translation raises (or no OpenAI key), the raw text is used as
    # the boolean — membership still works, just without synonym expansion.
    def _boom(_q):
        raise RuntimeError("no openai key")
    monkeypatch.setattr(main, "_generate_search_strategy", _boom, raising=True)
    monkeypatch.setattr(main, "_search_local_doc_ids",
                        lambda query, mode, filters, limit=10_000, threshold=0.45:
                        [9] if (query == "flu" and mode == "boolean") else [], raising=True)
    out = main._multi_query_corpus_ids([{"kind": "natural", "text": "flu"}], "union", {})
    assert out == [9]


# ── _evidence_fingerprint (corpus/threshold/lang cache key) ───────────────────
def test_evidence_fingerprint_is_order_independent():
    assert main._evidence_fingerprint([3, 1, 2], 0.45, "en", "brief") == \
           main._evidence_fingerprint([1, 2, 3], 0.45, "en", "brief")


def test_evidence_fingerprint_is_sensitive():
    base = main._evidence_fingerprint([1, 2, 3], 0.45, "en", "brief")
    assert base != main._evidence_fingerprint([1, 2, 3], 0.45, "fr", "brief")   # lang
    assert base != main._evidence_fingerprint([1, 2, 3], 0.60, "en", "brief")   # threshold
    assert base != main._evidence_fingerprint([1, 2], 0.45, "en", "brief")      # id set
    assert base != main._evidence_fingerprint([1, 2, 3], 0.45, "en", "vars")    # ctx version


def test_evidence_fingerprint_none_lang_is_french_bucket():
    assert main._evidence_fingerprint([1], 0.45, None, "x") == \
           main._evidence_fingerprint([1], 0.45, "fr", "x")


# ── _truncate_display_name + name capping (the 422-on-long-boolean-query bug) ─
def test_truncate_display_name_short_unchanged():
    assert main._truncate_display_name("influenza forecast") == "influenza forecast"
    assert main._truncate_display_name("  spaced  ") == "spaced"        # stripped
    assert main._truncate_display_name(None) is None                    # non-str passthrough


def test_truncate_display_name_long_is_capped_with_ellipsis():
    long_q = "(influenza OR RSV) AND (surveillance OR nowcasting) " * 20   # >>255 chars
    out = main._truncate_display_name(long_q)
    assert len(out) <= 255            # fits VARCHAR(255)
    assert out.endswith("…")


def test_user_scenario_in_caps_overlong_name_instead_of_422():
    # A long boolean query used as the scenario NAME must NOT raise (was a 422).
    long_name = "(influenza OR ILI OR ARI OR RSV OR \"SARS-CoV-2\") AND (surveillance OR nowcasting OR forecast) " * 5
    m = main.UserScenarioIn(name=long_name, query=long_name)
    assert len(m.name) <= 255         # name truncated to fit the column
    assert m.query == long_name       # query (TEXT) keeps the full text


def test_user_scenario_patch_caps_overlong_name():
    long_name = "x" * 400
    m = main.UserScenarioPatch(name=long_name)
    assert len(m.name) <= 255
    # None stays None (name is optional on PATCH)
    assert main.UserScenarioPatch().name is None


# ── _sentiweb_latest_value (the /terrain/epidemic silent-fallback bug) ────────
def test_sentiweb_parse_picks_latest_week_inc100():
    # Real Sentiweb shape: {"data": [{"week", "inc100", "inc"}, …]} — pick max week's inc100.
    res = {"data": [
        {"week": 202405, "inc100": 120.0, "inc": 9000},
        {"week": 202407, "inc100": 155.5, "inc": 12000},   # most recent
        {"week": 202406, "inc100": 140.0, "inc": 10000},
    ]}
    assert main._sentiweb_latest_value(res) == 155.5


def test_sentiweb_parse_falls_back_to_inc_then_none():
    # inc100 missing → use inc; nothing usable / wrong shapes → None (no crash)
    assert main._sentiweb_latest_value({"data": [{"week": 202401, "inc": 88}]}) == 88.0
    assert main._sentiweb_latest_value({"data": []}) is None
    assert main._sentiweb_latest_value([]) is None
    assert main._sentiweb_latest_value({"nope": 1}) is None
    # the OLD broken assumption (a root list of dicts) is now handled too
    assert main._sentiweb_latest_value([{"week": 202410, "inc100": 42.0}]) == 42.0


# ── New corpus-source parsers (PR B1): raw API payload → tidy docs ────────────
def test_parse_semantic_scholar():
    payload = {"data": [
        {"paperId": "abc", "title": "Flu nowcasting", "abstract": "We model ILI.",
         "year": 2024, "externalIds": {"DOI": "10.1/x"}, "url": "https://s2/abc"},
        {"paperId": None, "title": "no id"},          # dropped (needs paperId)
        {"title": "no paper id key"},                   # dropped
    ]}
    docs = main._parse_semantic_scholar(payload)
    assert len(docs) == 1
    d = docs[0]
    assert d["external_id"] == "s2:abc" and d["doi"] == "10.1/x" and d["year"] == 2024
    assert d["title"] == "Flu nowcasting" and d["source_type"] == "article"
    assert main._parse_semantic_scholar({}) == [] and main._parse_semantic_scholar({"data": None}) == []


def test_parse_doaj():
    payload = {"results": [
        {"id": "d1", "bibjson": {"title": "OA respiratory study", "abstract": "abs", "year": "2023",
                                  "identifier": [{"type": "doi", "id": "10.2/y"}],
                                  "link": [{"url": "https://doaj/d1"}]}},
        {"bibjson": {"title": "no id"}},                # dropped (needs id)
    ]}
    docs = main._parse_doaj(payload)
    assert len(docs) == 1
    assert docs[0]["external_id"] == "doaj:d1" and docs[0]["doi"] == "10.2/y"
    assert docs[0]["year"] == 2023 and docs[0]["url"] == "https://doaj/d1"


def test_parse_clinicaltrials():
    payload = {"studies": [
        {"protocolSection": {
            "identificationModule": {"nctId": "NCT01", "briefTitle": "RSV vaccine trial"},
            "descriptionModule": {"briefSummary": "A trial."},
            "statusModule": {"startDateStruct": {"date": "2022-05-01"}}}},
        {"protocolSection": {"identificationModule": {"nctId": None, "briefTitle": "x"}}},  # dropped
    ]}
    docs = main._parse_clinicaltrials(payload)
    assert len(docs) == 1
    d = docs[0]
    assert d["external_id"] == "nct:NCT01" and d["year"] == 2022
    assert d["url"] == "https://clinicaltrials.gov/study/NCT01" and d["source_type"] == "clinical_trial"
    assert d["doi"] is None


def test_parse_core():
    payload = {"results": [
        {"id": 555, "title": "CORE OA paper", "abstract": "abs", "yearPublished": 2021,
         "doi": "10.3/z", "downloadUrl": "https://core/pdf"},
        {"id": None, "title": "no id"},                 # dropped
    ]}
    docs = main._parse_core(payload)
    assert len(docs) == 1
    assert docs[0]["external_id"] == "core:555" and docs[0]["doi"] == "10.3/z"
    assert docs[0]["year"] == 2021 and docs[0]["url"] == "https://core/pdf"
    # malformed / empty are safe
    for bad in ({}, {"results": None}, {"results": ["nope"]}):
        assert main._parse_core(bad) == []


# ── PR B2 parsers: arXiv (XML), bioRxiv (scan+filter), OpenAIRE (nested) ──────
def test_parse_arxiv():
    xml = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
      <entry>
        <id>http://arxiv.org/abs/2401.01234v1</id>
        <title>Deep learning for   ILI nowcasting</title>
        <summary>We forecast influenza.</summary>
        <published>2024-01-15T00:00:00Z</published>
        <arxiv:doi>10.9/arxiv.x</arxiv:doi>
      </entry>
      <entry><id></id><title>no id → dropped</title></entry>
    </feed>"""
    docs = main._parse_arxiv(xml)
    assert len(docs) == 1
    d = docs[0]
    assert d["external_id"] == "arxiv:2401.01234v1" and d["year"] == 2024
    assert d["title"] == "Deep learning for ILI nowcasting"        # whitespace collapsed
    assert d["doi"] == "10.9/arxiv.x" and d["source_type"] == "preprint"
    assert main._parse_arxiv("not xml at all") == []               # malformed → []


def test_parse_biorxiv_filters_by_keywords():
    payload = {"collection": [
        {"doi": "10.1101/aaa", "title": "Respiratory virus wastewater",
         "abstract": "influenza surveillance", "date": "2025-03-01"},
        {"doi": "10.1101/bbb", "title": "Unrelated", "abstract": "quantum widgets", "date": "2025-02-01"},
        {"doi": None, "title": "no doi", "abstract": "influenza respiratory wastewater"},
    ]}
    terms = ["respiratory", "influenza", "wastewater"]
    docs = main._parse_biorxiv(payload, terms, "biorxiv")
    assert len(docs) == 1                                          # only the matching, DOI-bearing one
    assert docs[0]["external_id"] == "biorxiv:10.1101/aaa" and docs[0]["year"] == 2025
    # no terms → keep all DOI-bearing entries
    assert len(main._parse_biorxiv(payload, [], "medrxiv")) == 2


def test_openaire_text_helper():
    assert main._openaire_text("x") == "x"
    assert main._openaire_text({"$": "y"}) == "y"
    assert main._openaire_text([{"$": "z"}, {"$": "w"}]) == "z"
    assert main._openaire_text(None) is None


def test_parse_openaire_nested_and_malformed():
    payload = {"response": {"results": {"result": [
        {"header": {"dri:objIdentifier": {"$": "oai:xxx"}},
         "metadata": {"oaf:entity": {"oaf:result": {
             "title": [{"$": "Respiratory  forecasting"}, {"$": "alt"}],
             "description": {"$": "abstract text"},
             "dateofacceptance": {"$": "2023-06-01"},
             "pid": [{"@classid": "doi", "$": "10.5/oa"}]}}}},
        {"metadata": {"oaf:entity": {"oaf:result": {"title": None}}}},   # dropped (no title)
    ]}}}
    docs = main._parse_openaire(payload)
    assert len(docs) == 1
    d = docs[0]
    assert d["external_id"] == "doi:10.5/oa" and d["doi"] == "10.5/oa"
    assert d["title"] == "Respiratory forecasting" and d["year"] == 2023
    # no DOI → falls back to objIdentifier
    p2 = {"response": {"results": {"result": [
        {"header": {"dri:objIdentifier": {"$": "oai:zzz"}},
         "metadata": {"oaf:entity": {"oaf:result": {"title": {"$": "No DOI paper"}}}}}]}}}
    assert main._parse_openaire(p2)[0]["external_id"] == "openaire:oai:zzz"
    for bad in ({}, {"response": None}, {"response": {"results": None}}):
        assert main._parse_openaire(bad) == []


# ── Slice 2: _assemble_connector_frames (resample + align + join) ─────────────
def test_assemble_connector_frames_resamples_aligns_and_joins():
    import pandas as pd
    # Daily weather + wastewater sampled on specific dates → resample weekly, align, join.
    weather = pd.DataFrame({"date": ["2025-01-01", "2025-01-02", "2025-01-08", "2025-01-09"],
                            "temp_mean": [1.0, 3.0, 5.0, 7.0]})
    waste = pd.DataFrame({"date": ["2025-01-01", "2025-01-08"], "rsv_load": [100.0, 200.0]})
    frames = {"open-meteo-weather": weather, "eawag-wastewater": waste}
    mappings = [
        {"template_column": "temperature", "connector_id": "open-meteo-weather", "connector_variable": "temp_mean"},
        {"template_column": "rsv_ww", "connector_id": "eawag-wastewater", "connector_variable": "rsv_load"},
    ]
    df, filled = main._assemble_connector_frames(frames, mappings, "W", "week")
    assert set(filled) == {"temperature", "rsv_ww"}
    assert "week" in df.columns and {"temperature", "rsv_ww"} <= set(df.columns)
    assert len(df) == 2                                   # two aligned weekly buckets
    assert list(df["temperature"]) == [2.0, 6.0]         # weekly means of daily temps
    assert list(df["rsv_ww"]) == [100.0, 200.0]


def test_assemble_keeps_date_when_no_datetime_col():
    import pandas as pd
    frames = {"c": pd.DataFrame({"date": ["2025-01-01", "2025-01-08"], "v": [1.0, 2.0]})}
    df, filled = main._assemble_connector_frames(
        frames, [{"template_column": "x", "connector_id": "c", "connector_variable": "v"}], "W", None)
    assert "date" in df.columns and filled == ["x"]


def test_assemble_empty_when_nothing_maps():
    import pandas as pd
    frames = {"c": pd.DataFrame({"date": ["2025-01-01"], "v": [1.0]})}
    # mapping points at a variable the frame doesn't have → nothing assembled
    df, filled = main._assemble_connector_frames(
        frames, [{"template_column": "x", "connector_id": "c", "connector_variable": "missing"}], "W", None)
    assert df is None and filled == []


def test_agg_for_column_inference():
    assert main._agg_for_column("precip_sum") == "sum"
    assert main._agg_for_column("rainfall") == "sum"
    assert main._agg_for_column("rsv_load") == "last"
    assert main._agg_for_column("wastewater_viral") == "last"
    assert main._agg_for_column("temp_mean") == "mean"
    assert main._agg_for_column("relative_humidity_mean") == "mean"


def test_assemble_per_variable_aggregation():
    import pandas as pd
    # one weekly bucket, 3 daily rows: precip → SUM, wastewater load → LAST, temp → MEAN
    wx = pd.DataFrame({"date": ["2025-01-01", "2025-01-02", "2025-01-03"],
                       "temp_mean": [2.0, 4.0, 6.0], "precip_sum": [1.0, 2.0, 3.0]})
    ww = pd.DataFrame({"date": ["2025-01-01", "2025-01-02", "2025-01-03"], "rsv_load": [10.0, 20.0, 30.0]})
    frames = {"w": wx, "e": ww}
    mappings = [
        {"template_column": "t", "connector_id": "w", "connector_variable": "temp_mean"},
        {"template_column": "p", "connector_id": "w", "connector_variable": "precip_sum"},
        {"template_column": "r", "connector_id": "e", "connector_variable": "rsv_load"},
    ]
    df, _ = main._assemble_connector_frames(frames, mappings, "W", None)
    assert len(df) == 1
    assert df["t"].iloc[0] == 4.0     # mean(2,4,6)
    assert df["p"].iloc[0] == 6.0     # sum(1,2,3) — not mean
    assert df["r"].iloc[0] == 30.0    # last(10,20,30) — not mean


# ── _looks_boolean (auto-detect boolean vs natural, no manual toggle) ─────────
def test_looks_boolean_detects_operators_and_syntax():
    assert main._looks_boolean('ambulance AND (demand OR forecasting)') is True
    assert main._looks_boolean('"influenza-like illness" OR RSV') is True
    assert main._looks_boolean('influenza AND pneumonia') is True
    assert main._looks_boolean('cancer NOT lung') is True
    assert main._looks_boolean('(surveillance)') is True
    assert main._looks_boolean('grippe 2019:2026[dp]') is True


def test_looks_boolean_treats_natural_language_as_natural():
    assert main._looks_boolean('prévision de la demande d\'ambulances') is False
    assert main._looks_boolean('early warning of respiratory surges') is False
    assert main._looks_boolean('which indicators predict flu weeks in advance') is False
    assert main._looks_boolean('') is False
    # lowercase 'and'/'or' inside prose is NOT a boolean operator
    assert main._looks_boolean('forecasting and early detection of surges') is False


def test_normalize_sub_queries_auto_detects_when_kind_absent():
    out = main._normalize_sub_queries([
        {"text": "ambulance AND demand"},                 # no kind → detected boolean
        {"kind": "auto", "text": "early warning of flu"},  # auto → detected natural
        {"kind": "natural", "text": "cancer AND lung"},    # explicit override wins
    ])
    assert [o["kind"] for o in out] == ["boolean", "natural", "natural"]


# ── /search-facets preview (per-facet lexical counts + union/intersection) ────
def test_post_search_facets_counts_union_and_intersection(monkeypatch):
    # deterministic doc-id sets per boolean query; no DB, no LLM
    corpus = {"A": {1, 2, 3}, "B": {3, 4}}
    monkeypatch.setattr(main, "_search_local_doc_ids",
                        lambda q, mode, filters, limit=0: list(corpus.get(q, set())))
    monkeypatch.setattr(main, "_generate_search_strategy", lambda text: {"general": "B"})
    payload = main.FacetPreviewIn(
        sub_queries=[{"kind": "boolean", "text": "A"},
                     {"kind": "natural", "text": "some prose"}],   # → translated to "B"
        combinator="union")
    res = main.post_search_facets(payload)
    assert [f["count"] for f in res["facets"]] == [3, 2]
    assert res["facets"][1]["boolean"] == "B"          # translation surfaced
    assert res["union"] == 4 and res["intersection"] == 1
    assert res["combined"] == 4                          # union chosen
    res_i = main.post_search_facets(main.FacetPreviewIn(
        sub_queries=payload.sub_queries, combinator="intersection"))
    assert res_i["combined"] == 1                        # intersection chosen


def test_generate_search_strategy_returns_cached_copy():
    # A cached (valid) translation is returned WITHOUT touching the LLM/network, and as
    # a COPY — mutating the caller's dict must not corrupt the cache.
    main._STRATEGY_CACHE.clear()
    main._STRATEGY_CACHE["flu forecasting"] = {"general": "CACHED", "degraded": False}
    res = main._generate_search_strategy("  flu forecasting  ")   # trimmed → same key
    assert res["general"] == "CACHED"
    res["general"] = "MUTATED"
    assert main._STRATEGY_CACHE["flu forecasting"]["general"] == "CACHED"
    main._STRATEGY_CACHE.clear()


def test_strategy_key_is_deterministic_and_normalized():
    # The persistent-cache key must collapse case + whitespace so the SAME natural
    # question always maps to the SAME stored boolean (fixes "Main 57 vs #1 56").
    assert main._strategy_key("  Public   Health? ") == "public health?"
    assert main._strategy_key("Public Health?") == main._strategy_key("public   health?")
    assert main._strategy_key("") == "" and main._strategy_key(None) == ""


# ── boolean parser: groups respected + PubMed field tags stripped ─────────────
def _bool_leaves(ast):
    if ast is None:
        return []
    if ast[0] == "term":
        return [ast[1]]
    if ast[0] == "not":
        return _bool_leaves(ast[1])
    out = []
    for ch in ast[1]:
        out += _bool_leaves(ch)
    return out


def _ast_of(q):
    return main._parse_boolean_ast(main._tokenize_boolean(q))


def test_boolean_parser_respects_parenthesized_groups():
    # THE regression: (A OR B) AND (C OR D) must be AND of two OR-groups, NOT
    # "A AND C AND (B OR D)" (the old flat parser's bug that caused 109 → 5).
    ast = _ast_of('(A OR B) AND (C OR D)')
    assert ast == ("and", [
        ("or", [("term", "a"), ("term", "b")]),
        ("or", [("term", "c"), ("term", "d")]),
    ])


def test_boolean_parser_strips_pubmed_field_tags():
    ast = _ast_of('ranking[Title/Abstract] OR "ranking system"[Title/Abstract]')
    leaves = _bool_leaves(ast)
    assert leaves == ["ranking", "ranking system"]
    # the tag text must NOT leak in as a phantom required term
    for junk in ("titleabstract", "title", "abstract", "meshterms", "mesh", "terms"):
        assert junk not in leaves


def test_boolean_parser_users_three_group_query():
    q = ('("Public Health Schools"[MeSH Terms] OR "public health schools"[Title/Abstract] '
         'OR "schools of public health"[Title/Abstract] OR "public health education"[Title/Abstract]) '
         'AND (ranking[Title/Abstract] OR criteria[Title/Abstract] OR "ranking criteria"[Title/Abstract] '
         'OR "ranking system"[Title/Abstract] OR evaluation[Title/Abstract] OR assessment[Title/Abstract]) '
         'AND (global[Title/Abstract] OR worldwide[Title/Abstract] OR international[Title/Abstract] '
         'OR "in the world"[Title/Abstract])')
    ast = _ast_of(q)
    assert ast[0] == "and" and len(ast[1]) == 3          # three AND-groups, not flattened
    assert [g[0] for g in ast[1]] == ["or", "or", "or"]   # each group is an OR of alternatives
    assert [len(g[1]) for g in ast[1]] == [4, 6, 4]
    leaves = _bool_leaves(ast)
    assert "schools of public health" in leaves and "in the world" in leaves and "worldwide" in leaves
    for junk in ("titleabstract", "meshterms", "mesh", "terms"):
        assert junk not in leaves
    # SQL binds real phrases, never the tag text
    params: dict = {}
    sql = main._build_boolean_match_sql_from_query(q, params)
    vals = set(params.values())
    assert "%ranking%" in vals and "%in the world%" in vals and "%global%" in vals
    assert not any("titleabstract" in v or "meshterms" in v for v in vals)
    assert sql.count(" AND ") >= 2                        # three groups AND-ed together


def test_boolean_to_arxiv_transforms_terms_and_groups():
    # each term → all:"phrase", groups + AND/OR preserved (for the arXiv search_query)
    got = main._boolean_to_arxiv(_ast_of('("public health schools" OR ranking) AND global'))
    assert got == '((all:"public health schools" OR all:"ranking") AND all:"global")'
    assert main._boolean_to_arxiv(_ast_of('influenza surveillance')) == '(all:"influenza" AND all:"surveillance")'
    # NOT → None (arXiv ANDNOT is binary, not unary) so the caller falls back to all:<keywords>
    assert main._boolean_to_arxiv(_ast_of('cancer NOT benign')) is None


def test_boolean_to_s2_bulk_syntax():
    # space = AND, ` | ` = OR, quotes = phrase, parens = group
    got = main._boolean_to_s2(_ast_of('("public health schools" OR ranking) AND global'))
    assert got == '(("public health schools" | ranking) global)'
    assert main._boolean_to_s2(_ast_of('influenza')) == 'influenza'
    assert main._boolean_to_s2(_ast_of('cancer NOT benign')) is None   # NOT → fallback to keyword


def test_parse_openaire_graph_defensive():
    payload = {"results": [
        {"id": "oai:123", "mainTitle": "Global ranking of schools",
         "descriptions": ["We ranked schools worldwide."],
         "publicationDate": "2021-05-01",
         "pids": [{"scheme": "doi", "value": "10.1234/abc"}]},
        {"id": "oai:456", "mainTitle": "Dict-shaped description",
         "descriptions": [{"value": "desc as dict"}], "publicationDate": "2019"},
        {"mainTitle": "no id, skipped"},                      # missing id → dropped
    ]}
    docs = main._parse_openaire_graph(payload)
    assert len(docs) == 2
    assert docs[0]["title"] == "Global ranking of schools"
    assert docs[0]["abstract"] == "We ranked schools worldwide."
    assert docs[0]["year"] == 2021 and docs[0]["doi"] == "10.1234/abc"
    assert docs[0]["external_id"] == "openaire:oai:123"
    assert docs[1]["abstract"] == "desc as dict" and docs[1]["year"] == 2019
    assert main._parse_openaire_graph({}) == []


def test_boolean_parser_and_or_not_and_fallback():
    assert _ast_of('cancer AND lung') == ("and", [("term", "cancer"), ("term", "lung")])
    assert _ast_of('cancer OR tumour') == ("or", [("term", "cancer"), ("term", "tumour")])
    assert _ast_of('cancer NOT benign') == ("and", [("term", "cancer"), ("not", ("term", "benign"))])
    assert _ast_of('influenza surveillance') == ("and", [("term", "influenza"), ("term", "surveillance")])
    assert main._build_boolean_match_sql_from_query("", {}) == "TRUE"   # empty → matches all (filtered elsewhere)


# ── _normalize_title (cross-source dedup key: DOI → title) ────────────────────
def test_normalize_title_canonicalizes_for_dedup():
    # same paper, different punctuation/case/spacing from two sources → identical key
    a = main._normalize_title("Global Ranking of Schools of Public Health: A Systematic Review")
    b = main._normalize_title("  global ranking of schools of public health — a systematic review  ")
    assert a == b == "global ranking of schools of public health a systematic review"
    # SQL backfill uses the SAME normalization, so on-ingest and backfilled keys agree
    assert main._normalize_title("SARS-CoV-2: Omicron (BA.5)") == "sars cov 2 omicron ba 5"
    assert main._normalize_title(None) == "" and main._normalize_title("!!!") == ""
