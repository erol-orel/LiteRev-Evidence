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
    md = R.to_markdown(doc)
    assert "| a/b | c |" in md and "| x/y | z |" in md


def test_the_word_file_escapes_markup_and_stays_well_formed():
    doc = {"title": "A & B <draft>", "subtitle": 'He said "no"', "meta": "m",
           "blocks": [("h2", "Tom & Jerry"), ("table", ["a<b", "c"], [["1 & 2", "<x>"]]), ("bullets", ["x < y"])]}
    data = R.to_docx(doc)
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
    assert "sex gender > male" in md and "hpai a h5n1" in md and "male against female" in md
    # The limits: an abstract-only paper and unreviewed rows are said, not hidden.
    assert "read from the abstract only" in md and "have not been reviewed" in md
    assert chr(0x2014) not in md


def test_the_french_report_is_in_french(pool_corpus):
    md = _get("md", "fr").text
    for needle in ("# Rapport d'extraction structurée", "## Comment cela a été fait", "## Proportions combinées", "## Limites à garder en tête",
                   "Aucune ligne n'a encore été relue"):
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
