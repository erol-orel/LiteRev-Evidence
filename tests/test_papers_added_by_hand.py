"""Papers added by hand to a scenario from uploaded files (api/additions.py).

A search reaches the databases LiteRev can query; a review also holds papers those
databases do not index (the ISS team's Embase-only records) or papers a reviewer has
on their desk. They are uploaded as PDFs or reference files, go through the same
ingestion as a search result, join the corpus, survive later searches, and are counted
by the PRISMA card on their own line.

Pure tests for the parsers and the metadata guessed from a PDF's text; database tests
for the endpoint, the survival of the links, and the kept-link counts.
"""
import io
import os
import shutil

import pytest

pytest.importorskip("fastapi")

from api import additions as A  # noqa: E402

SID = "papers-by-hand"
KEY = {"X-API-Key": "test-write-key"}

RIS = """TY  - JOUR
TI  - Seroprevalence of avian influenza among poultry cullers in Lombardy
AU  - Rossi, Maria
AU  - Bianchi, Luca
PY  - 2019///
JO  - Zoonoses and Public Health
DO  - https://doi.org/10.5555/byhand.{tag}.1
AB  - We measured antibodies against H5N1 in 240 cullers after the 2017 epidemic.
  The seroprevalence was low but not null.
ER  -

TY  - JOUR
T1  - Environmental persistence of H5N8 in poultry manure in winter
Y1  - 2021/01/15
N2  - Virus survival in manure was measured at three temperatures.
T2  - Veterinary Research
ER  -

TY  - JOUR
TI  - A
ER  -
"""

BIBTEX = """@article{rossi2019,
  title = {Seroprevalence of avian influenza among poultry cullers in {Lombardy}},
  author = {Rossi, Maria and Bianchi, Luca},
  journal = {Zoonoses and Public Health},
  year = {2019},
  doi = {10.5555/byhand.x.1},
  abstract = "Antibodies against H5N1 in cullers \\& veterinarians.",
}
@comment{ignored}
@inproceedings{doe2020,
  title={Biosecurity practices on {Belgian} farms},
  booktitle={Proceedings of the Zoonoses Congress},
  year=2020,
  pmid={31234567}
}
"""

NBIB = """PMID- 31234567
DP  - 2019 Mar 4
TI  - Occupational exposure to highly pathogenic avian influenza among
      veterinarians in Italy.
AB  - Background: veterinarians handle infected flocks. Methods: a survey.
      Results: exposure was frequent.
AU  - Rossi M
AU  - Bianchi L
JT  - Eurosurveillance
AID - 10.5555/byhand.nbib.1 [doi]
LID - 10.5555/byhand.nbib.1 [doi]

PMID- 31234568
DP  - 2020
TI  - Second record without a DOI.
JT  - Some Journal
"""

CSV = ("Title;Abstract;DOI;Publication Year;Authors\n"
       "Knowledge and attitudes of farmers towards HPAI in Bavaria;A survey of 300 farmers.;10.5555/byhand.csv.1;2018;Weber K\n"
       ";no title here;10.5555/none;2018;\n")

PDF_PAGE = """Zoonoses and Public Health 2020; 67: 123-130
doi: 10.5555/byhand.pdf.1

Protective equipment use among poultry workers during the 2017 Italian
H5N8 epidemic

Maria Rossi1, Luca Bianchi2

Abstract
Highly pathogenic avian influenza H5N8 spread through poultry farms in northern Italy in 2017. We surveyed 412
workers involved in culling operations about the protective equipment they used, their training and their
perception of the risk. Most wore gloves, fewer wore respirators, and training predicted adherence.

Keywords: avian influenza, occupational exposure, personal protective equipment

1. Introduction
The 2017 epidemic was the largest in Italy since 1999.
"""


# ── Pure: the parsers ─────────────────────────────────────────────────────────
def test_a_ris_file_is_read_record_by_record():
    recs = A.parse_ris(RIS.replace("{tag}", "t"))
    assert [r["title"] for r in recs] == [
        "Seroprevalence of avian influenza among poultry cullers in Lombardy",
        "Environmental persistence of H5N8 in poultry manure in winter"]       # "A" has no usable title
    first, second = recs
    assert first["doi"] == "10.5555/byhand.t.1" and first["year"] == 2019
    assert first["authors"] == "Rossi, Maria, Bianchi, Luca" and first["journal"] == "Zoonoses and Public Health"
    assert first["abstract"].endswith("The seroprevalence was low but not null.")   # continuation line kept
    assert first["url"] == "https://doi.org/10.5555/byhand.t.1"
    assert second["year"] == 2021 and second["abstract"].startswith("Virus survival") and second["journal"] == "Veterinary Research"


def test_a_bibtex_file_is_read_with_its_braces_and_authors():
    recs = A.parse_bibtex(BIBTEX)
    assert len(recs) == 2
    assert recs[0]["title"] == "Seroprevalence of avian influenza among poultry cullers in Lombardy"
    assert recs[0]["authors"] == "Rossi, Maria, Bianchi, Luca"
    assert recs[0]["abstract"] == "Antibodies against H5N1 in cullers & veterinarians."
    assert recs[0]["doi"] == "10.5555/byhand.x.1" and recs[0]["year"] == 2019
    assert recs[1]["title"] == "Biosecurity practices on Belgian farms"
    assert recs[1]["journal"] == "Proceedings of the Zoonoses Congress" and recs[1]["pmid"] == "31234567"
    assert recs[1]["url"] == "https://pubmed.ncbi.nlm.nih.gov/31234567/"


def test_a_pubmed_nbib_file_is_read_with_its_continuation_lines():
    recs = A.parse_nbib(NBIB)
    assert len(recs) == 2
    assert recs[0]["title"] == "Occupational exposure to highly pathogenic avian influenza among veterinarians in Italy."
    assert recs[0]["pmid"] == "31234567" and recs[0]["year"] == 2019 and recs[0]["doi"] == "10.5555/byhand.nbib.1"
    assert recs[0]["abstract"].startswith("Background: veterinarians") and "Results: exposure" in recs[0]["abstract"]
    assert recs[0]["authors"] == "Rossi M, Bianchi L" and recs[0]["journal"] == "Eurosurveillance"
    assert recs[1]["pmid"] == "31234568" and recs[1]["doi"] is None and recs[1]["year"] == 2020


def test_a_table_is_read_by_its_column_names_whatever_the_delimiter():
    recs = A.parse_csv(CSV.encode("utf-8"))
    assert len(recs) == 1                                   # the row without a title is skipped
    r = recs[0]
    assert r["title"].startswith("Knowledge and attitudes") and r["doi"] == "10.5555/byhand.csv.1"
    assert r["year"] == 2018 and r["authors"] == "Weber K" and r["abstract"] == "A survey of 300 farmers."
    assert A.map_columns(["PMID", "Article Title", "Source title", "Year"]) == {
        "pmid": "PMID", "title": "Article Title", "journal": "Source title", "year": "Year"}
    with pytest.raises(ValueError):
        A.parse_csv(b"doi,year\n10.1/x,2020\n")


def test_the_kind_of_a_file_comes_from_its_content_then_its_name():
    assert A.detect_kind("paper.txt", b"%PDF-1.4 ...") == "pdf"
    assert A.detect_kind("refs.ris", b"TY  - JOUR") == "ris"
    assert A.detect_kind("refs.txt", b"TY  - JOUR\nTI  - x\nER  -") == "ris"
    assert A.detect_kind("refs.txt", b"PMID- 123\nTI  - x") == "nbib"
    assert A.detect_kind("refs.txt", b"@article{k,\n title={x}}") == "bibtex"
    assert A.detect_kind("export.txt", b"Title,Abstract,DOI\nx,y,z") == "csv"
    assert A.detect_kind("x.xlsx", b"PK\x03\x04...") == "xlsx"
    assert A.detect_kind("notes.txt", b"just some prose") is None


def test_a_record_needs_a_title_and_keeps_only_what_is_plausible():
    assert A.make_record("abc") is None
    r = A.make_record("A plausible title", year="1750", doi="doi:not-a-doi", pmid="PMID: 12345")
    assert r["year"] is None and r["doi"] is None and r["pmid"] == "12345"
    assert A.make_record("A plausible title", year="2024 Jan")["year"] == 2024


# ── Pure: what a PDF's text says ─────────────────────────────────────────────
def test_the_metadata_of_a_pdf_is_guessed_from_its_first_page():
    m = A.guess_pdf_metadata(PDF_PAGE)
    assert m["title"] == "Protective equipment use among poultry workers during the 2017 Italian H5N8 epidemic"
    assert m["doi"] == "10.5555/byhand.pdf.1"
    assert m["abstract"].startswith("Highly pathogenic avian influenza H5N8 spread")
    assert m["abstract"].endswith("training predicted adherence.")       # cut before the keywords
    assert m["year"] == 2017 or m["year"] == 2020
    assert A.guess_pdf_metadata("")["title"] is None


# ── With a database ──────────────────────────────────────────────────────────
def _engine_ok() -> bool:
    import main
    try:
        with main.engine.connect():
            return True
    except Exception:
        return False


@pytest.fixture()
def seeded(db_conn, monkeypatch):
    """A user scenario, the tables the ingestion writes, and no background work: the
    scoring trigger and the training are recorded, not run."""
    import main
    from conftest import patch_app
    from search_tables import ensure_search_tables
    if not _engine_ok():
        pytest.skip("main.engine cannot reach the database")
    scored: list[str] = []
    retrained: list[str] = []
    patch_app(monkeypatch, "score_new_links", lambda sid: scored.append(sid) or True)
    patch_app(monkeypatch, "schedule_retrain", lambda sid: retrained.append(sid) or True)
    # Crossref is not called in the tests: the PDF keeps what its text says.
    patch_app(monkeypatch, "crossref_by_doi", lambda doi: None)
    with db_conn.cursor() as cur:
        created_chunk_table = ensure_search_tables(cur)
        # The ingestion INSERTs without an id, as production does: a suite-bootstrapped
        # table is created as a bare `id BIGINT PRIMARY KEY`, with no sequence behind it.
        # The default is put back as it was at teardown: leaving one behind made the other
        # integration tests of the session take a path they do not expect.
        cur.execute("SELECT column_default FROM information_schema.columns"
                    " WHERE table_name = 'literature_document' AND column_name = 'id'")
        previous_default = (cur.fetchone() or [None])[0]
        if not previous_default:
            cur.execute("CREATE SEQUENCE IF NOT EXISTS papers_by_hand_id_seq")
            cur.execute("SELECT setval('papers_by_hand_id_seq',"
                        " GREATEST(1000000, COALESCE((SELECT MAX(id) FROM literature_document), 0) + 1))")
            cur.execute("ALTER TABLE literature_document ALTER COLUMN id"
                        " SET DEFAULT nextval('papers_by_hand_id_seq')")
        for col, typ in (("reviewer_2_status", "VARCHAR(20)"), ("kappa_final_status", "VARCHAR(20)"),
                         ("kappa_resolved", "BOOLEAN DEFAULT FALSE"), ("learned_score", "DOUBLE PRECISION"),
                         ("rerank_score", "DOUBLE PRECISION")):
            cur.execute(f"ALTER TABLE article_scenarios ADD COLUMN IF NOT EXISTS {col} {typ}")
        for col, typ in (("chunk_weight", "DOUBLE PRECISION DEFAULT 1.0"), ("metadata_json", "JSONB DEFAULT '{}'::jsonb"),
                         ("token_count", "INTEGER")):
            cur.execute(f"ALTER TABLE document_chunk ADD COLUMN IF NOT EXISTS {col} {typ}")
        cur.execute("SELECT to_regclass('scenario_settings') IS NULL")
        if cur.fetchone()[0]:
            main._ensure_scenario_settings_table()
        main._ensure_additions_table()
        _clean_up(cur)
        cur.execute("INSERT INTO user_scenarios (id, name, query, mode, filters)"
                    " VALUES (%s, 'Papers by hand', 'avian influenza AND occupational', 'boolean', '{}')", (SID,))
    yield db_conn, scored, retrained
    with db_conn.cursor() as cur:
        _clean_up(cur)
        if not previous_default:
            cur.execute("ALTER TABLE literature_document ALTER COLUMN id DROP DEFAULT")
            cur.execute("DROP SEQUENCE IF EXISTS papers_by_hand_id_seq")
        if created_chunk_table:
            cur.execute("DROP TABLE document_chunk")


def _clean_up(cur) -> None:
    """Everything this test creates, before it runs and after: the scenario, its links and
    additions, the row it seeds by hand, and every document an upload ingested (they all
    carry `source = 'upload'`). Run BEFORE as well, so a row left by an interrupted run
    does not make the next one read it as "already in the library"."""
    cur.execute("DELETE FROM scenario_additions WHERE scenario_id = %s", (SID,))
    cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (SID,))
    cur.execute("DELETE FROM document_chunk WHERE document_id IN"
                " (SELECT id FROM literature_document WHERE source = 'upload' OR id = 990001)")
    cur.execute("DELETE FROM article_scenarios WHERE document_id IN"
                " (SELECT id FROM literature_document WHERE source = 'upload' OR id = 990001)")
    cur.execute("DELETE FROM literature_document WHERE source = 'upload' OR id = 990001")
    cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))


def _client():
    import main
    from fastapi.testclient import TestClient
    return TestClient(main.app)


def _links(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT a.document_id, a.screening_status, d.doi, d.source FROM article_scenarios a"
                    " JOIN literature_document d ON d.id = a.document_id WHERE a.scenario_id = %s ORDER BY a.document_id", (SID,))
        return cur.fetchall()


def test_an_uploaded_reference_file_joins_the_corpus_and_is_remembered(seeded):
    conn, scored, retrained = seeded
    import main
    c = _client()
    tag = "ris"
    r = c.post(f"/user-scenarios/{SID}/articles/upload", headers=KEY,
               files=[("files", ("team.ris", RIS.replace("{tag}", tag).encode(), "application/x-research-info-systems"))])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["files"][0]["kind"] == "ris" and body["files"][0]["records"] == 2
    assert body["totals"] == {"records": 2, "added": 2, "new_documents": 2, "already_in_corpus": 0, "skipped": 0,
                              "included": 0, "kept_decision": 0, "fulltext_stored": 0, "files_failed": 0}
    assert body["scoring_started"] is True and scored == [SID] and retrained == []
    links = _links(conn)
    assert len(links) == 2 and all(status is None for _, status, _, _ in links)
    assert {doi for _, _, doi, _ in links} == {f"10.5555/byhand.{tag}.1", None}
    assert all(src == "upload" for *_, src in links)
    # Remembered, with the file it came from.
    with conn.cursor() as cur:
        cur.execute("SELECT filename FROM scenario_additions WHERE scenario_id = %s", (SID,))
        assert {f for (f,) in cur.fetchall()} == {"team.ris"}
    listed = c.get(f"/user-scenarios/{SID}/articles/added").json()
    assert listed["total"] == 2 and all(a["in_corpus"] for a in listed["articles"])
    # The same file again: nothing is duplicated, the papers are already in the corpus.
    r2 = c.post(f"/user-scenarios/{SID}/articles/upload", headers=KEY,
                files=[("files", ("team.ris", RIS.replace("{tag}", tag).encode(), "text/plain"))])
    assert r2.json()["totals"]["already_in_corpus"] == 2 and r2.json()["totals"]["added"] == 0
    assert len(_links(conn)) == 2
    # A later search that does not match them keeps them, and the PRISMA card counts them.
    ids = [d for d, *_ in links]
    main._set_scenario_corpus(SID, [ids[0]])                       # the search matched one of the two
    assert [d for d, *_ in _links(conn)] == ids
    assert main._kept_link_counts(SID, [ids[0]]) == {"reviewed": 0, "added": 1, "previous": 0}
    figs = main._prisma_identification_figures({"db_cache": 1}, 1, 0, 2, kept_added_links=1)
    assert figs["kept_added_links"] == 1 and figs["records_screened"] == 2
    assert figs["records_identified"] - figs["duplicates_removed"] - figs["removed_before_screening"] + 1 == 2


def test_include_marks_the_uploaded_papers_as_included_by_hand_and_teaches_the_queue(seeded):
    conn, scored, retrained = seeded
    import main
    c = _client()
    with conn.cursor() as cur:                         # one of them is already in the library and excluded
        cur.execute("INSERT INTO literature_document (id, title, abstract, source, project_context, doi, title_norm)"
                    " VALUES (990001, 'Knowledge and attitudes of farmers towards HPAI in Bavaria', 'x', 'pubmed',"
                    " 'literev', '10.5555/byhand.csv.1', %s)",
                    (main._normalize_title("Knowledge and attitudes of farmers towards HPAI in Bavaria"),))
        cur.execute("INSERT INTO article_scenarios (scenario_id, document_id, screening_status) VALUES (%s, 990001, 'excluded')", (SID,))
    r = c.post(f"/user-scenarios/{SID}/articles/upload?include=true", headers=KEY,
               files=[("files", ("refs.csv", CSV.encode("utf-8"), "text/csv")),
                      ("files", ("pubmed.nbib", NBIB.encode("utf-8"), "text/plain"))])
    assert r.status_code == 200, r.text
    t = r.json()["totals"]
    assert (t["records"], t["added"], t["already_in_corpus"], t["new_documents"]) == (3, 2, 1, 2)
    assert t["included"] == 2 and t["kept_decision"] == 1, "the exclusion a reviewer made is not overturned"
    assert retrained == [SID] and scored == [SID]
    by_doi = {doi: status for _, status, doi, _ in _links(conn)}
    assert by_doi["10.5555/byhand.csv.1"] == "excluded"
    assert by_doi["10.5555/byhand.nbib.1"] == "included"
    with conn.cursor() as cur:
        cur.execute("SELECT pmid FROM literature_document WHERE doi = '10.5555/byhand.nbib.1'")
        assert cur.fetchone()[0] == "31234567", "the PMID of an nbib record is kept on the row"
        cur.execute("SELECT screening_notes FROM article_scenarios WHERE scenario_id = %s AND screening_status = 'included'", (SID,))
        assert all(n.startswith("Added by hand from ") for (n,) in cur.fetchall())
    # Included by hand: a search that no longer matches them keeps them as reviewed links.
    main._set_scenario_corpus(SID, [990001])
    assert len(_links(conn)) == 3
    assert main._kept_link_counts(SID, [990001]) == {"reviewed": 2, "added": 0, "previous": 0}


def test_the_upload_needs_the_key_and_says_what_it_cannot_read(seeded):
    c = _client()
    r = c.post(f"/user-scenarios/{SID}/articles/upload", files=[("files", ("x.ris", b"TY  - JOUR\nER  -", "text/plain"))])
    assert r.status_code == 401
    r = c.post(f"/user-scenarios/{SID}/articles/upload", headers=KEY,
               files=[("files", ("notes.txt", b"just some prose", "text/plain"))])
    assert r.status_code == 200
    f = r.json()["files"][0]
    assert f["kind"] is None and "not recognised" in f["error"] and r.json()["totals"]["files_failed"] == 1
    r = c.post("/user-scenarios/no-such-scenario/articles/upload", headers=KEY,
               files=[("files", ("x.ris", b"TY  - JOUR\nER  -", "text/plain"))])
    assert r.status_code == 404


def _minimal_pdf(lines: list[str]) -> bytes:
    """A one-page PDF with Helvetica text, one line per entry; an empty entry is a gap
    that pdftotext renders as a blank line."""
    ops = ["BT", "/F1 11 Tf", "50 760 Td", "13 TL"]
    for ln in lines:
        ops.append("T*" if not ln else f"({ln}) Tj T*")
    ops.append("ET")
    content = "\n".join(ops)
    objs = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        "/Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(content)} >>\nstream\n{content}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = "%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{o}\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n" + "".join(f"{off:010d} 00000 n \n" for off in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    return out.encode("latin-1")


def test_a_pdf_is_read_and_its_text_kept(seeded):
    if not shutil.which("pdftotext"):
        try:
            import pypdf  # noqa: F401
        except ImportError:
            pytest.skip("neither pdftotext nor pypdf: the PDF path cannot run here")
    conn, scored, _ = seeded
    lines = [ln.replace("(", "").replace(")", "") for ln in PDF_PAGE.splitlines()]
    pdf = _minimal_pdf(lines + [""] + [f"Body line {i} of the article about poultry workers and equipment." for i in range(60)])
    c = _client()
    r = c.post(f"/user-scenarios/{SID}/articles/upload", headers=KEY,
               files=[("files", ("paper.pdf", pdf, "application/pdf"))])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["files"][0]["kind"] == "pdf" and body["totals"]["added"] == 1, body
    art = body["articles"][0]
    assert art["title"].startswith("Protective equipment use among poultry workers"), art
    assert art["metadata_from"] == "text" and art["doi"] == "10.5555/byhand.pdf.1"
    assert art["fulltext_stored"] is True and body["totals"]["fulltext_stored"] == 1
    with conn.cursor() as cur:
        cur.execute("SELECT abstract, has_fulltext FROM literature_document WHERE id = %s", (art["id"],))
        abstract, has_ft = cur.fetchone()
        assert abstract.startswith("Highly pathogenic avian influenza H5N8 spread") and has_ft is True
        cur.execute("SELECT COUNT(*) FROM document_chunk WHERE document_id = %s AND chunk_type = 'fulltext_section'", (art["id"],))
        assert cur.fetchone()[0] >= 1
    assert scored == [SID]


def test_a_database_without_the_additions_table_still_searches(monkeypatch):
    """The predicate is on the search path: on a database that has no `scenario_additions`
    (older than the feature, or a DDL that could not run), it must read FALSE rather than
    make every search fail on a missing relation."""
    import main
    from api import search as S
    assert "scenario_additions" in main.added_by_hand_sql("a")        # pure, whatever the database
    monkeypatch.setattr(S, "_ADDITIONS_TABLE", [False])
    monkeypatch.setattr(S, "engine", None)                            # any probe failure
    assert S._added_by_hand_predicate("a") == "FALSE"


def test_pdf_text_says_when_no_reader_is_available(monkeypatch):
    import subprocess
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()))
    monkeypatch.setitem(__import__("sys").modules, "pypdf", None)
    with pytest.raises(A.PdfReaderMissing):
        A.pdf_text(b"%PDF-1.4")
