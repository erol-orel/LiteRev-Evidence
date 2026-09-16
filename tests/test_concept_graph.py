"""The concept map: typed concepts per article, co-occurrence links, chains, gaps.

Pure: `_build_concept_graph` works on rows already read from the database. The endpoint
test stubs the row reader and the cache."""
import main
from conftest import patch_app  # noqa: E402


def _row(i, year=2024, concepts=None, country=None, design=None, keywords=None, pico=None, meta=None, q=0.5):
    return {
        "id": i, "title": f"Article {i}", "year": year, "quality": q, "doi": f"10.1/{i}", "pmid": None,
        "country": country, "study_design": design, "pico_json": pico, "metadata_json": meta,
        "keywords": keywords, "similarity": 0.9,
        "concepts_json": {"v": 1, "concepts": concepts} if concepts is not None else None,
    }


def _c(t, en, fr=None):
    return {"t": t, "en": en, "fr": fr or en}


def test_designs_and_places_are_normalised_from_free_text():
    assert main._normalise_design("Systematic review and meta-analysis") == "systematic_review"
    assert main._normalise_design("Randomized controlled trial") == "rct"
    assert main._normalise_design("Retrospective cohort") == "cohort"
    assert main._normalise_design("Narrative review") == "review"
    assert main._normalise_design("non précisé") is None
    concepts = main._article_concepts(_row(1, country="it", pico={"study_design": "Case series"},
                                           meta={"country": "FR", "setting": "hospital"},
                                           keywords="Humans; Aedes; Chikungunya virus; Italy"))
    types_keys = {(t, k) for t, k, _ in concepts}
    assert ("place", "IT") in types_keys and ("place", "FR") in types_keys
    assert ("design", "case report / series") in types_keys
    assert ("setting", "hospital") in types_keys
    assert ("topic", "humans") not in types_keys                    # MeSH check tag dropped
    assert ("topic", "aedes") in types_keys


def test_the_graph_links_concepts_cited_by_the_same_articles():
    chik = _c("pathogen", "chikungunya virus", "virus du chikungunya")
    aedes = _c("vector", "Aedes albopictus")
    italy = _c("place", "IT")
    auto = _c("outcome", "autochthonous transmission", "transmission autochtone")
    death = _c("outcome", "mortality", "mortalité")
    rows = [
        _row(1, 2025, [chik, aedes, italy, auto]),
        _row(2, 2025, [chik, aedes, auto]),
        _row(3, 2024, [chik, aedes, italy]),
        _row(4, 2024, [chik, death]),
        _row(5, 2023, [chik, death]),
        _row(6, 2023, [aedes]),
    ]
    g = main._build_concept_graph(rows, n_total=6)
    by = {(n["type"], n["label"]["en"]): n for n in g["nodes"]}
    assert by[("pathogen", "chikungunya virus")]["count"] == 5
    assert by[("pathogen", "chikungunya virus")]["label"]["fr"] == "virus du chikungunya"
    assert by[("vector", "Aedes albopictus")]["count"] == 4
    assert g["latest_year"] == 2025 and by[("pathogen", "chikungunya virus")]["new_count"] == 2
    w = {(e["source"], e["target"]): e["weight"] for e in g["edges"]}
    chik_id, aedes_id = by[("pathogen", "chikungunya virus")]["id"], by[("vector", "Aedes albopictus")]["id"]
    assert w[tuple(sorted((chik_id, aedes_id)))] == 3
    assert all(g["nodes"][e["source"]]["type"] != g["nodes"][e["target"]]["type"] for e in g["edges"])
    # chain pathogen › vector › outcome, and a gap: Aedes albopictus was never studied with mortality
    assert g["triples"][0]["count"] == 2
    gap_pairs = {tuple(sorted(x["nodes"])) for x in g["gaps"]}
    assert tuple(sorted((aedes_id, by[("outcome", "mortality")]["id"]))) in gap_pairs
    assert str(1) in g["articles"] and g["articles"]["1"]["doi"] == "10.1/1"
    assert g["source"] == "llm" and g["n_missing_concepts"] == 0


def test_without_llm_concepts_the_graph_still_stands_on_structured_fields():
    rows = [_row(i, 2024, None, country="IT", design="Cohort study", keywords="dengue; climate") for i in range(1, 6)]
    g = main._build_concept_graph(rows)
    types = {n["type"] for n in g["nodes"]}
    assert types == {"place", "design", "topic"}
    assert g["source"] == "structured" and g["n_missing_concepts"] == 5


def test_the_endpoint_serves_the_cache_and_recomputes_without_it(monkeypatch):
    from fastapi.testclient import TestClient

    saved = {}
    patch_app(monkeypatch, "_get_user_scenario_or_404", lambda sid: {"id": sid, "query": "q"})
    patch_app(monkeypatch, "_concept_rows", lambda sid: ([_row(1, 2024, [_c("pathogen", "dengue virus")]),
                                                         _row(2, 2024, [_c("pathogen", "dengue virus")])], 2))
    patch_app(monkeypatch, "_load_viz_cache", lambda sid, col, ttl=None: None)
    patch_app(monkeypatch, "_save_viz_cache", lambda sid, col, payload: saved.setdefault(col, payload))
    monkeypatch.setenv("OPENAI_API_KEY", "")                      # no background enrichment
    client = TestClient(main.app)
    r = client.get("/user-scenarios/usr-x/concept-graph")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["kind"] == "concepts" and body["nodes"][0]["label"]["en"] == "dengue virus"
    assert body["enriching"] is False and "concepts" in saved

    cached = dict(saved["concepts"], n_missing_concepts=0)
    patch_app(monkeypatch, "_load_viz_cache", lambda sid, col, ttl=None: cached)
    patch_app(monkeypatch, "_concept_rows", lambda sid: (_ for _ in ()).throw(AssertionError("must not recompute")))
    r = client.get("/gesica/scenarios/influenza-surveillance/concept-graph")
    assert r.status_code == 200 and r.json()["nodes"][0]["label"]["en"] == "dengue virus"


def test_the_seir_default_projection_comes_from_the_cache_when_fresh(monkeypatch):
    computed = []
    written = {}
    patch_app(monkeypatch, "_seir_projection_payload",
              lambda sid, *a, **k: computed.append(sid) or {"applicable": False, "scenario_id": sid,
                                                              "reason_code": "no_parameters", "reason": "x"})
    patch_app(monkeypatch, "_seir_cache_write", lambda sid, p: written.setdefault(sid, p))
    patch_app(monkeypatch, "_seir_cache_read", lambda sid: None)
    out = main._default_seir_projection("usr-s")
    assert out["from_cache"] is False and computed == ["usr-s"] and "usr-s" in written

    patch_app(monkeypatch, "_seir_cache_read", lambda sid: dict(written[sid]))
    out2 = main._default_seir_projection("usr-s")
    assert out2["from_cache"] is True and computed == ["usr-s"]      # served, not recomputed
    out3 = main._precompute_seir_projection("usr-s")
    assert out3["from_cache"] is False and computed == ["usr-s", "usr-s"]


def test_the_auto_pipeline_flag_defaults_on_and_reads_the_environment(monkeypatch):
    monkeypatch.delenv("AUTO_PIPELINE_AFTER_SEARCH", raising=False)
    assert main._auto_pipeline_after_search() is True
    monkeypatch.setenv("AUTO_PIPELINE_AFTER_SEARCH", "0")
    assert main._auto_pipeline_after_search() is False
