"""Exports of the relevant articles (api/exports.py): formatters are pure, the endpoint
serves a file with the right type and name. The relevance query is stubbed."""
import json

import main
from conftest import patch_app  # noqa: E402


def _articles():
    return [
        {"id": 7, "title": "Autochthonous chikungunya in Italy {2024}", "authors": "Rossi Maria; Bianchi Luca",
         "year": 2024, "journal": "Eurosurveillance", "doi": "10.2807/x", "pmid": "39000001",
         "study_design": "Cohort", "similarity_score": 0.8123456, "quality_score": 0.71, "citation_count": 3,
         "screening_status": "included", "abstract": "Line one.\nLine two.", "keywords": "Aedes; Italy",
         "pico_json": {"P": "residents", "I": "vector control", "C": "none", "O": "cases"}},
        {"id": 8, "title": "Second article", "authors": None, "year": None, "journal": None, "doi": None,
         "similarity_score": None, "abstract": None, "pico_json": None},
    ]


def test_rows_and_tabular_formats():
    rows = main.export_rows(_articles())
    assert [r["rank"] for r in rows] == [1, 2]
    assert rows[0]["url"] == "https://doi.org/10.2807/x" and rows[0]["similarity_score"] == 0.8123
    assert rows[0]["pico_population"] == "residents" and rows[1]["pico_population"] == ""
    csv_text = main.to_csv(rows)
    assert csv_text.startswith("﻿rank,id,title,authors,year")
    assert "Autochthonous chikungunya in Italy {2024}" in csv_text and csv_text.count("\r\n") == 3
    data = json.loads(main.to_json(rows, {"scenario": "s"}))
    assert data["meta"]["scenario"] == "s" and data["articles"][1]["title"] == "Second article"
    xlsx = main.to_xlsx(rows, "Test")
    assert xlsx[:2] == b"PK"                                   # a zip, hence a real workbook
    md = main.to_markdown(rows, "Chikungunya")
    assert md.startswith("# Chikungunya") and "1. **Autochthonous" in md


def test_ris_and_bibtex_records():
    rows = main.export_rows(_articles())
    ris = main.to_ris(rows)
    assert ris.count("TY  - JOUR") == 2 and ris.count("ER  - ") == 2
    assert "AU  - Rossi Maria\r\nAU  - Bianchi Luca" in ris
    assert "DO  - 10.2807/x" in ris and "AB  - Line one. Line two." in ris and "KW  - Aedes" in ris
    bib = main.to_bibtex(rows)
    assert bib.startswith("@article{rossi2024_7,")
    assert "title = {Autochthonous chikungunya in Italy \\{2024\\}}" in bib   # braces escaped
    assert "author = {Rossi Maria and Bianchi Luca}" in bib
    assert "@article{anonnd_8," in bib


def test_the_endpoint_serves_a_named_file(monkeypatch):
    from fastapi.testclient import TestClient

    patch_app(monkeypatch, "_get_user_scenario_or_404", lambda sid: {"id": sid, "name": "Chikungunya Europe", "query": "q"})
    patch_app(monkeypatch, "_get_above_threshold_articles", lambda sid, threshold=None, **k: _articles())
    patch_app(monkeypatch, "_export_extra_fields", lambda ids: {7: {"id": 7, "pmid": "39000001", "source": "pubmed", "url": None, "keywords": "Aedes"}})
    client = TestClient(main.app)
    r = client.get("/user-scenarios/usr-x/relevant/export?format=ris")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("application/x-research-info-systems")
    assert r.headers["content-disposition"] == 'attachment; filename="chikungunya-europe_relevant-articles_2.ris"'
    assert r.headers["x-article-count"] == "2" and "TY  - JOUR" in r.text

    r = client.get("/gesica/scenarios/influenza-surveillance/relevant/export?format=xlsx&include_abstract=false")
    assert r.status_code == 200 and r.content[:2] == b"PK"
    assert client.get("/user-scenarios/usr-x/relevant/export?format=docx").status_code == 400


# ── Exports of SUBSETS: clusters, concepts, an explicit selection ────────────
def _client(monkeypatch):
    from fastapi.testclient import TestClient

    patch_app(monkeypatch, "_get_user_scenario_or_404",
              lambda sid: {"id": sid, "name": "Chikungunya Europe", "query": "q"})
    patch_app(monkeypatch, "_export_extra_fields", lambda ids: {})
    patch_app(monkeypatch, "articles_by_ids", lambda sid, ids: [
        a for a in _articles() if int(a["id"]) in {int(i) for i in ids}])
    return TestClient(main.app)


def _cluster_cache():
    return {
        "lang": "fr", "n_docs": 3000, "n_docs_total": 8200,
        "clusters": [
            {"cluster_id": 0, "cluster_name": "Vector control", "n_docs": 2,
             "top_words": ["aedes", "control"], "summary": "Sur la lutte antivectorielle.",
             "points": [{"id": 7, "x": 0.1, "y": 0.2}, {"id": 8, "x": 0.3, "y": 0.4}]},
            {"cluster_id": -1, "cluster_name": "Bruit", "is_noise": True, "points": []},
        ],
    }


def test_a_cluster_exports_its_articles_and_declares_its_bounds(monkeypatch):
    """A cluster export is complete FOR THAT CLUSTER (the cache keeps every point), but
    clustering itself runs on the most relevant CLUSTER_MAX_DOCS only. The file has to
    say so, or it reads as "every article on this theme"."""
    client = _client(monkeypatch)
    patch_app(monkeypatch, "_load_viz_cache", lambda sid, col, ttl=None: _cluster_cache())

    r = client.get("/user-scenarios/usr-x/clusters/0/export?format=json")
    assert r.status_code == 200, r.text
    assert r.headers["x-export-subset"] == "cluster"
    assert r.headers["x-article-count"] == "2"
    assert "vector-control" in r.headers["content-disposition"]
    meta = json.loads(r.text)["meta"]
    assert meta["subset"] == "cluster" and meta["cluster_id"] == 0
    assert meta["n_docs_clustered"] == 3000 and meta["n_docs_eligible"] == 8200
    # The bound is stated in the file itself, not only in the interface.
    assert "3000" in meta["coverage"] and "8200" in meta["coverage"]
    assert "pas le corpus entier" in meta["coverage"]


def test_an_unknown_cluster_and_a_missing_cache_are_both_explained(monkeypatch):
    client = _client(monkeypatch)
    patch_app(monkeypatch, "_load_viz_cache", lambda sid, col, ttl=None: _cluster_cache())
    r = client.get("/user-scenarios/usr-x/clusters/42/export")
    assert r.status_code == 404 and "42" in r.json()["detail"]

    patch_app(monkeypatch, "_load_viz_cache", lambda sid, col, ttl=None: None)
    r = client.get("/user-scenarios/usr-x/clusters/0/export")
    assert r.status_code == 404 and "Clusters" in r.json()["detail"]


def _concept_graph_full():
    """What _build_concept_graph returns with full_articles=True: nodes carry ALL ids."""
    return {
        "n_with_concepts": 120, "n_total": 150,
        "nodes": [
            {"id": 0, "type": "pathogen", "label": {"en": "chikungunya virus", "fr": "virus du chikungunya"},
             "count": 2, "articles": [7, 8], "articles_listed": 2},
            {"id": 1, "type": "vector", "label": {"en": "Aedes albopictus", "fr": "Aedes albopictus"},
             "count": 1, "articles": [7], "articles_listed": 1},
        ],
    }


def test_a_concept_export_is_not_capped_at_the_display_limit(monkeypatch):
    """Nodes served to the interface carry at most 40 articles. An export that inherited
    that cap would be a silent sample, so the graph is recomputed with full_articles."""
    client = _client(monkeypatch)
    seen = {}

    def _fake_build(rows, **kw):
        seen.update(kw)
        return _concept_graph_full()

    patch_app(monkeypatch, "_concept_rows", lambda sid: ([], 150))
    patch_app(monkeypatch, "_build_concept_graph", _fake_build)

    r = client.get("/user-scenarios/usr-x/concepts/export"
                   "?concepts=pathogen:chikungunya virus&format=json")
    assert r.status_code == 200, r.text
    assert seen.get("full_articles") is True, "the export must not reuse the capped payload"
    meta = json.loads(r.text)["meta"]
    assert meta["subset"] == "concepts" and meta["concepts"] == ["chikungunya virus"]
    assert meta["n_with_concepts"] == 120
    assert r.headers["x-article-count"] == "2"


def test_a_concept_selection_unions_or_intersects(monkeypatch):
    client = _client(monkeypatch)
    patch_app(monkeypatch, "_concept_rows", lambda sid: ([], 150))
    patch_app(monkeypatch, "_build_concept_graph", lambda rows, **kw: _concept_graph_full())

    q = "concepts=pathogen:chikungunya virus|vector:Aedes albopictus"
    r_any = client.get(f"/user-scenarios/usr-x/concepts/export?{q}&mode=any&format=json")
    r_all = client.get(f"/user-scenarios/usr-x/concepts/export?{q}&mode=all&format=json")
    assert r_any.headers["x-article-count"] == "2"      # union: 7 and 8
    assert r_all.headers["x-article-count"] == "1"      # intersection: only 7
    assert " OU " in json.loads(r_any.text)["meta"]["coverage"]
    assert " ET " in json.loads(r_all.text)["meta"]["coverage"]


def test_an_unknown_concept_is_named_rather_than_silently_dropped(monkeypatch):
    client = _client(monkeypatch)
    patch_app(monkeypatch, "_concept_rows", lambda sid: ([], 150))
    patch_app(monkeypatch, "_build_concept_graph", lambda rows, **kw: _concept_graph_full())
    r = client.get("/user-scenarios/usr-x/concepts/export?concepts=pathogen:zika virus")
    assert r.status_code == 404 and "zika virus" in r.json()["detail"]
    # A malformed selector says what the shape should be.
    r = client.get("/user-scenarios/usr-x/concepts/export?concepts=justalabel")
    assert r.status_code == 400 and "type:label" in r.json()["detail"]


def test_the_generic_id_export_is_scoped_to_the_scenario(monkeypatch):
    """This is what serves the RAG sources and any hand-picked set. An id that does not
    belong to the scenario must not come out, and the file must say some were dropped."""
    client = _client(monkeypatch)
    r = client.get("/user-scenarios/usr-x/articles/export?ids=7,8,99999&label=RAG answer&format=json")
    assert r.status_code == 200, r.text
    assert r.headers["x-export-subset"] == "selection"
    assert r.headers["x-article-count"] == "2"
    assert r.headers["x-missing-ids"] == "1"
    meta = json.loads(r.text)["meta"]
    assert meta["requested"] == 3 and meta["returned"] == 2 and meta["missing_ids"] == [99999]
    assert "écartés" in meta["coverage"]
    assert "rag-answer" in r.headers["content-disposition"]

    assert client.get("/user-scenarios/usr-x/articles/export?ids=").status_code == 400
    assert client.get("/user-scenarios/usr-x/articles/export?ids=abc").status_code == 400
