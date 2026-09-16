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
