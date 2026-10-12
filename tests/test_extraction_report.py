"""The extraction report (api/extraction_report.py): one document model, three files.

What matters about a report is that it carries its method and its limits as well as its numbers,
that the three formats say the same thing, and that the Word and PDF files really open. The Word
file is checked by reading it back with python-docx and the PDF with pypdf, where they are installed."""
import io
import zipfile
from xml.dom import minidom

import pytest

pytest.importorskip("fastapi")

import main  # noqa: E402
from api import extraction_report as R  # noqa: E402
from test_extraction import HDR, SID, extracted, seeded  # noqa: E402,F401
from test_pooling import pool_corpus  # noqa: E402,F401


def _client():
    from fastapi.testclient import TestClient
    return TestClient(main.app)


def _get(fmt="md", lang="en"):
    return _client().get(f"/user-scenarios/{SID}/extraction/report", params={"format": fmt, "lang": lang})


# ── Formatting ──────────────────────────────────────────────────────────────
def test_numbers_are_formatted_for_a_reader():
    assert R.pct(0.2236) == "22.4%" and R.pct(0.5) == "50%" and R.pct(None) == "-" and R.pct(1.0) == "100%"
    assert R.fmt_or(1.23456) == "1.23" and R.fmt_or(0.047312) == "0.0473" and R.fmt_or(250.4) == "250" and R.fmt_or(None) == "-"
    assert R._interval(0.1, 0.35, R.pct) == "10% to 35%" and R._interval(0.1, None, R.pct) == "-"


def test_markdown_tables_survive_a_pipe_in_a_cell():
    doc = {"title": "T", "subtitle": "S", "meta": "m", "blocks": [("table", ["a|b", "c"], [["x|y", "z"]])]}
    md = R.report_to_markdown(doc)
    assert "| a/b | c |" in md and "| x/y | z |" in md


def test_the_word_file_escapes_markup_and_stays_well_formed():
    doc = {"title": "A & B <draft>", "subtitle": 'He said "no"', "meta": "m",
           "blocks": [("h2", "Tom & Jerry"), ("table", ["a<b", "c"], [["1 & 2", "<x>"]]), ("bullets", ["x < y"])]}
    data = R.report_to_docx(doc)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        assert z.testzip() is None
        xml = z.read("word/document.xml").decode("utf-8")
        minidom.parseString(xml)                      # raises if the XML is not well formed
        minidom.parseString(z.read("docProps/core.xml"))
    assert "A &amp; B &lt;draft&gt;" in xml and "1 &amp; 2" in xml


# ── The endpoint ────────────────────────────────────────────────────────────
def test_the_markdown_report_carries_the_method_the_numbers_and_the_limits(pool_corpus):
    r = _get("md")
    assert r.status_code == 200 and r.headers["x-snapshot"] and len(r.headers["x-snapshot"]) == 10
    md = r.text
    for needle in ("# Structured extraction report", "## How this was made", "## What the papers report", "## Review status",
                   "## Pooled proportions", "## Comparisons between groups", "## Limits to keep in mind", "## Studies behind each pooled estimate",
                   "Snapshot " + r.headers["x-snapshot"]):
        assert needle in md, needle
    # The method: the subset, the extraction, the model and prompt, the codebook, the statistics.
    assert "above the similarity threshold 0.45" in md and "from the full text" in md and "prompt" in md
    assert "the default codebook" in md and "DerSimonian-Laird" in md and "Hartung-Knapp-Sidik-Jonkman" in md
    assert "No row has been reviewed yet" in md
    # The numbers: the pooled male proportion, with its studies and the odds ratio against female.
    assert "sex gender > male" in md and "| A(H5N1) |" in md and "male against female" in md
    # The disease is named once whatever its spelling, says so, and shows what the papers wrote.
    assert "A disease is recognised whatever its spelling" in md
    assert "sex gender > male (A(H5N1)). As written: HPAI A(H5N1) (6)" in md
    assert "sex gender > male (not stated)" in md and "(not stated))" not in md
    # The limits: an abstract-only paper and unreviewed rows are said, not hidden.
    assert "read from the abstract only" in md and "have not been reviewed" in md
    assert chr(0x2014) not in md


def test_the_french_report_is_in_french(pool_corpus):
    md = _get("md", "fr").text
    for needle in ("# Rapport d'extraction structurée", "## Comment cela a été fait", "## Proportions combinées", "## Limites à garder en tête",
                   "Aucune ligne n'a encore été relue", "quelle que soit sa graphie", "Tel qu'écrit : HPAI A(H5N1) (6)",
                   "sex gender > male (non précisée)"):
        assert needle in md, needle
    assert "How this was made" not in md and chr(0x2014) not in md


def test_the_word_report_opens_and_holds_the_tables(pool_corpus):
    docx = pytest.importorskip("docx")
    r = _get("docx")
    assert r.status_code == 200 and "wordprocessingml" in r.headers["content-type"] and r.headers["content-disposition"].endswith('.docx"')
    d = docx.Document(io.BytesIO(r.content))
    text_ = "\n".join(p.text for p in d.paragraphs)
    assert "Structured extraction report" in text_ and "How this was made" in text_ and "Limits to keep in mind" in text_
    assert len(d.tables) >= 5
    first = [c.text for c in d.tables[0].rows[0].cells]
    assert first == ["Item", "Papers", "Share", "From full text"]
    pooled = next(t for t in d.tables if t.rows[0].cells[0].text == "Label")
    assert any("sex gender > male" in row.cells[0].text for row in pooled.rows)


def test_the_pdf_report_is_a_real_pdf_with_figures(pool_corpus):
    r = _get("pdf")
    assert r.status_code == 200 and r.content.startswith(b"%PDF") and r.headers["content-type"] == "application/pdf"
    assert len(r.content) > 8000
    pypdf = pytest.importorskip("pypdf")
    reader = pypdf.PdfReader(io.BytesIO(r.content))
    text_ = "\n".join((p.extract_text() or "") for p in reader.pages)
    assert "Structured extraction report" in text_ and "Pooled proportions" in text_ and "DerSimonian-Laird" in text_
    # The forest plots are drawn: the label of the pooled row is in the figure.
    assert text_.count("Pooled") >= 3


def test_the_snapshot_follows_what_the_figures_rest_on(pool_corpus):
    a = _get("md").headers["x-snapshot"]
    assert _get("md").headers["x-snapshot"] == a                       # stable for the same data
    obs = _client().get(f"/user-scenarios/{SID}/articles/9711/extraction").json()["extraction"]["observations"][0]["obs_key"]
    _client().post(f"/user-scenarios/{SID}/articles/9711/extraction/review", headers=HDR,
                   json={"obs_key": obs, "reviewer": "Ana", "status": "accepted"})
    b = _get("md")
    assert b.headers["x-snapshot"] != a and "reviewed by Ana" in b.text
    with pool_corpus.cursor() as cur:
        cur.execute("DELETE FROM extraction_review WHERE document_id = 9711")


def test_the_errors_are_the_right_ones(seeded, extracted):
    assert _client().get(f"/user-scenarios/{SID}/extraction/report", params={"format": "odt"}).status_code == 422
    assert _client().get("/user-scenarios/nope/extraction/report").status_code == 404


def test_a_scenario_with_nothing_extracted_has_nothing_to_report(seeded):
    r = _client().get(f"/user-scenarios/{SID}/extraction/report")
    assert r.status_code == 409 and "Nothing is extracted" in r.json()["detail"]


# ── Every cut list says how much more there is, and where ───────────────────
@pytest.fixture()
def report_corpus(pool_corpus):
    """Two countries (Germany with a region), one paper read condensed, one from a cut text."""
    import json
    with pool_corpus.cursor() as cur:
        for i, place in ((9711, "Berlin, Germany"), (9712, "Rome, Italy")):
            cur.execute("UPDATE literature_document SET extraction_json = jsonb_set(extraction_json, '{ref}', %s::jsonb) "
                        "WHERE id = %s", (json.dumps({"location": place}), i))
        cur.execute("UPDATE literature_document SET extraction_json = extraction_json || '{\"condensed\": true}'::jsonb "
                    "WHERE id = 9713")
        cur.execute("UPDATE literature_document SET extraction_json = extraction_json || "
                    "'{\"truncated\": true, \"max_chars\": 150000}'::jsonb WHERE id = 9714")
    return pool_corpus


def test_the_report_says_how_much_more_each_cut_list_holds(report_corpus, monkeypatch):
    for name in ("_COUNTRIES_LISTED", "_REGIONS_LISTED", "_FEW_LISTED", "_COMPARISONS_LISTED"):
        monkeypatch.setattr(R, name, 0 if name != "_COUNTRIES_LISTED" else 1)
    md = _get("md").text
    assert "1 more countries are not listed here" in md and "sheet COUNTRIES" in md
    assert "1 more regions are not listed here" in md and "| Germany | DE | 1 | 2 | - |" in md
    assert "more labels with fewer than 3 studies are not listed here" in md and "sheet POOLED" in md
    assert "1 more comparisons are not listed here" in md and "sheet COMPARISONS" in md
    # Nothing is said when nothing is cut.
    monkeypatch.setattr(R, "_COUNTRIES_LISTED", 25)
    monkeypatch.setattr(R, "_COMPARISONS_LISTED", 40)
    md = _get("md").text
    assert "more countries" not in md and "more comparisons" not in md and "male against female" in md


def test_the_limits_say_how_many_papers_were_read_condensed_or_cut(report_corpus):
    md = _get("md").text
    assert "1 articles held more data than one answer of the model can carry" in md
    assert "1 articles were read from a text cut at the size limit (300,000 characters per article today)" in md
    fr = _get("md", "fr").text
    assert "1 articles ont été lus dans un texte coupé à la taille limite (300 000 caractères" in fr
    assert "forme condensée" in fr
    assert chr(0x2014) not in md and chr(0x2014) not in fr


def test_the_excel_export_holds_the_full_lists(report_corpus, monkeypatch):
    openpyxl = pytest.importorskip("openpyxl")
    from api import pooling as P
    monkeypatch.setattr(P, "_MAX_PAIRS", 0)
    monkeypatch.setattr(P, "_MAX_GROUPS", 1)
    wb = openpyxl.load_workbook(io.BytesIO(_client().get(f"/user-scenarios/{SID}/extraction/export").content))
    assert {"COUNTRIES", "POOLED", "COMPARISONS"} <= set(wb.sheetnames)
    full = P.pooled_estimates(SID, full=True)
    assert wb["POOLED"].max_row - 1 == len(full["pooled"]) == full["groups_total"] > 1
    assert wb["COMPARISONS"].max_row - 1 == len(full["comparisons"]) == full["comparisons_total"] >= 1
    pooled_head = [c.value for c in wb["POOLED"][1]]
    flags = {r[pooled_head.index("Pooled")].value for r in wb["POOLED"].iter_rows(min_row=2)}
    assert flags == {"Y", "N"}, "the labels with too few studies are listed too"
    # Each pool and each comparison says how the papers behind it wrote its disease.
    for sheet in ("POOLED", "COMPARISONS"):
        head = [c.value for c in wb[sheet][1]]
        assert head[-1] == "Disease as written (studies)"
        assert any(str(r[-1].value or "").startswith("HPAI A(H5N1) (") for r in wb[sheet].iter_rows(min_row=2)), sheet
    countries = [(r[0].value, r[5].value) for r in wb["COUNTRIES"].iter_rows(min_row=2)]
    assert ("Germany", None) in countries and ("Germany", "DE3") in countries and ("Italy", None) in countries
    readme = " ".join(str(r[0].value) for r in wb["README"].iter_rows())
    assert "COUNTRIES: every country" in readme and "COMPARISONS: every comparison" in readme
