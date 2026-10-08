"""Structured extraction in the shape of the T2.4 / T4.5 data extraction template
(api/extraction.py).

Pure tests cover the parsing, the quote check, the template's column titles, who is
(re)extracted and the workbook; the integration tests run the whole map step against the
test database with a stubbed LLM: every relevant article is read (above the threshold or
included by hand, never the excluded), the result is cached on the article row, the full
text replaces an abstract-only pass, and a failing article stops being retried."""
import io
import json
import types

import pytest

pytest.importorskip("fastapi")

import main  # noqa: E402
from api import extraction  # noqa: E402
from conftest import ensure_document_columns, patch_app  # noqa: E402

SID = "usr-extraction-test"
OTHER = "usr-extraction-other"
IDS = (9701, 9702, 9703, 9704, 9705, 9706)
HDR = {"X-API-Key": "test-write-key"}

PAPER = ("Outbreak of HPAI A(H5N1) in poultry and cats, Germany. Seventeen exposed persons were "
         "identified. Among veterinary authority staff, 7 of 7 were not vaccinated against "
         "seasonal influenza. No antibodies against H5N1 were detected in 11 sera.")


def _raw(**over):
    base = {
        "ref": {"description": "Outbreak investigation in a poultry holding.", "article_type": "outbreak report",
                "location": "Sigmaringen, Germany", "risk_pop": "17", "positive": 0, "math_model": False},
        "coverage": {"sex_gender": False, "ppe": True, "vaccination": True},
        "observations": [{
            "sheet": "human_exp", "transmission_mode": "contact", "disease": "HPAI A(H5N1)",
            "group": "occupation", "covariate": "veterinary authority staff", "value": 7,
            "descr": "number exposed", "n_cases": 0, "pop_risk": "7",
            "original_name": "Table 4", "page_section": "Table 4", "source_kind": "Table",
            "quote": "7 of 7 were not vaccinated against seasonal influenza"}],
    }
    base.update(over)
    return base


# ── The template, verbatim ──────────────────────────────────────────────────
def test_the_columns_are_the_templates_titles_to_the_letter():
    ref = [t for t, _ in extraction.REF_COLUMNS]
    assert len(ref) == 19 and ref[0] == "ID" and ref[-1] == "Exclusion"
    # The workbook spells two titles with a trailing and a doubled space: kept as they are.
    assert "study area NUTS level 1 " in ref and "Number  of positive (cases)" in ref
    names = {k: v[0] for k, v in extraction.SHEET_COLUMNS.items()}
    assert names == {"human_susc": "HUMAN_COV_SUSC", "human_exp": "HUMAN_COV_EXP", "env": "ENV_COV",
                     "animal": "ANIMALorRESERVOIR_COV", "vector": "VECTOR_COV"}
    hum = [t for t, _ in extraction.SHEET_COLUMNS["human_exp"][1]]
    assert hum == ["ID", "TRANSMISSION MODE", "DISEASE", "COV_HUM_GROUP", "COVARIATE_hum", "Value_hum",
                   "descr_hum", "notes_hum", "N_cases_hum", "Pop_risk_Hum", "Variable original name",
                   "Page/section", "Extraction from Table/Figure/text"]
    assert [t for t, _ in extraction.SHEET_COLUMNS["env"][1]][3] == "COVARIATE_env"
    assert [t for t, _ in extraction.SHEET_COLUMNS["animal"][1]][3] == "COV_ANIM_GROUP"
    assert [t for t, _ in extraction.SHEET_COLUMNS["vector"][1]][3] == "COV_VEC_GROUP"


# ── Parsing ─────────────────────────────────────────────────────────────────
def test_parsing_keeps_real_rows_flags_the_unverifiable_and_drops_the_unusable():
    raw = _raw()
    raw["observations"] += [
        {"sheet": "nonsense", "covariate": "x"},                                  # unknown sheet
        {"sheet": "env", "covariate": ""},                                       # no covariate
        {"sheet": "animal", "covariate": "Felis catus", "value": "about half",   # text value
         "descr": "cats", "quote": "an excerpt the paper never contained"},
        "not an object",
    ]
    out = extraction.parse_extraction(raw, PAPER)
    assert [o["covariate"] for o in out["observations"]] == ["veterinary authority staff", "Felis catus"]
    first, second = out["observations"]
    assert first["value"] == 7 and first["pop_risk"] == 7 and first["n_cases"] == 0
    assert first["source_kind"] == "table" and first["quote_verified"] is True
    # A value that is not a number is kept, as text, instead of being lost.
    assert second["value"] is None and second["descr"] == "about half. cats"
    # A quote the text does not contain is KEPT and flagged, not silently filtered.
    assert second["quote_verified"] is False
    assert out["ref"]["risk_pop"] == 17 and out["ref"]["math_model"] is False
    # Coverage: true only when the model said true, false otherwise (never unknown).
    assert out["coverage"]["ppe"] is True and out["coverage"]["sex_gender"] is False
    assert set(out["coverage"]) == set(extraction.COVERAGE_KEYS)


def test_numbers_are_coerced_and_junk_is_not_a_number():
    f = extraction._to_number
    assert f("12,5") == 12.5 and f(" 3 ") == 3 and f(4.0) == 4 and isinstance(f(4.0), int)
    assert f("1 280") == 1280 and f(0) == 0
    assert f(True) is None and f(None) is None and f("n/a") is None and f("<10") is None
    assert f(float("nan")) is None and f(float("inf")) is None


def test_the_quote_check_ignores_line_breaks_and_case_but_not_words_or_figures():
    assert extraction.quote_in_text("7 of 7 were NOT vaccinated\nagainst seasonal influenza", PAPER)
    assert not extraction.quote_in_text("8 of 8 were not vaccinated", PAPER)
    assert not extraction.quote_in_text("", PAPER) and not extraction.quote_in_text(None, PAPER)


def test_the_number_of_observations_is_bounded_and_a_non_object_is_refused():
    many = _raw(observations=[{"sheet": "env", "covariate": f"c{i}"} for i in range(500)])
    assert len(extraction.parse_extraction(many, "")["observations"]) == extraction._MAX_OBSERVATIONS
    with pytest.raises(ValueError):
        extraction.parse_extraction([], "")


def test_the_prompt_forbids_the_em_dash_and_asks_for_quotes():
    p = extraction._EXTRACTION_SYSTEM
    assert chr(0x2014) not in p and "em dash" in p
    assert "verbatim" in p and "quote" in p


# ── Who gets (re)extracted ──────────────────────────────────────────────────
def test_an_article_is_extracted_once_then_again_only_for_a_reason():
    abstract = "x" * 200
    cur = {"v": extraction.EXTRACTION_VERSION, "source": "abstract"}
    assert extraction._needs_extraction({"abstract": abstract}) is True
    assert extraction._needs_extraction({"abstract": abstract, "extraction_json": cur}) is False
    # The full text arrived after an abstract-only pass: read it again.
    assert extraction._needs_extraction({"abstract": abstract, "has_fulltext": True,
                                         "extraction_json": cur}) is True
    assert extraction._needs_extraction({"abstract": abstract, "has_fulltext": True,
                                         "extraction_json": {**cur, "source": "fulltext"}}) is False
    # A newer extraction version makes everything eligible again.
    assert extraction._needs_extraction({"abstract": abstract,
                                         "extraction_json": {**cur, "v": extraction.EXTRACTION_VERSION - 1}}) is True
    # Nothing to read, or too many failures: left alone.
    assert extraction._needs_extraction({"abstract": "short"}) is False
    assert extraction._needs_extraction({"abstract": abstract, "extraction_attempts": 3}) is False


# ── One LLM call ────────────────────────────────────────────────────────────
def _fake_client(content=None, finish="stop", boom=None, seen=None):
    def create(**kw):
        if seen is not None:
            seen.append(kw)
        if boom:
            raise boom
        if content is None:
            text_ = json.loads(kw["messages"][1]["content"])["text"]
            body = _raw()
            body["observations"][0]["quote"] = text_.split(".")[1].strip()[:60]
        else:
            body = content
        msg = types.SimpleNamespace(content=json.dumps(body))
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg, finish_reason=finish)])
    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))


def test_one_article_goes_through_one_call_and_comes_back_stamped():
    seen: list = []
    out = extraction.extract_article(_fake_client(_raw(), seen=seen), {"id": 1}, PAPER, "abstract", False, "HPAI")
    assert len(seen) == 1 and seen[0]["temperature"] == 0 and seen[0]["response_format"] == {"type": "json_object"}
    assert json.loads(seen[0]["messages"][1]["content"])["disease_of_interest"] == "HPAI"
    assert out["v"] == extraction.EXTRACTION_VERSION and out["source"] == "abstract"
    assert out["truncated"] is False and out["n_chars"] == len(PAPER)
    assert out["observations"][0]["quote_verified"] is True


def test_a_truncated_or_failed_call_raises_instead_of_caching_an_empty_answer():
    with pytest.raises(Exception):
        extraction.extract_article(_fake_client(_raw(), finish="length"), {"id": 1}, PAPER, "abstract", False)
    with pytest.raises(RuntimeError):
        extraction.extract_article(_fake_client(boom=RuntimeError("down")), {"id": 1}, PAPER, "abstract", False)


# ── The template workbook ───────────────────────────────────────────────────
def _articles():
    ex = extraction.parse_extraction(_raw(), PAPER)
    ex.update({"v": 1, "source": "abstract", "truncated": False})
    ex["observations"].append({**ex["observations"][0], "sheet": "env", "covariate": "free ranging",
                               "page_section": "Table 2"})
    ex["ref"]["math_model"] = True
    return [{"id": 77, "authors": "Dressler A; Wagner-Wiening C", "doi": "10.2807/x", "year": 2026,
             "extraction_json": ex, "screening_status": None},
            {"id": 78, "authors": "", "doi": None, "year": None, "extraction_json": None}]


def test_the_workbook_has_the_templates_sheets_columns_and_the_review_columns_on_the_right():
    openpyxl = pytest.importorskip("openpyxl")
    data = extraction.build_workbook(_articles(), "1 of the 2 relevant articles are extracted.", "AB")
    wb = openpyxl.load_workbook(io.BytesIO(data))
    assert wb.sheetnames == ["README", "REF", "HUMAN_COV_SUSC", "HUMAN_COV_EXP", "ENV_COV",
                             "ANIMALorRESERVOIR_COV", "VECTOR_COV"]
    assert "1 of the 2 relevant" in wb["README"]["A2"].value
    ref = [c.value for c in wb["REF"][1]]
    assert ref == [t for t, _ in extraction.REF_COLUMNS]
    row = {h: c.value for h, c in zip(ref, wb["REF"][2])}
    assert row["ID"] == "AB77" and row["first author name"] == "Dressler"
    assert row["reference"] == "https://doi.org/10.2807/x" and row["Mathematical model (Y/N)"] == "Y"
    assert row["Number of exposed (population at risk)"] == 17
    assert wb["REF"].max_row == 2                       # the unextracted article is not exported
    exp = wb["HUMAN_COV_EXP"]
    head = [c.value for c in exp[1]]
    assert head[:13] == [t for t, _ in extraction.SHEET_COLUMNS["human_exp"][1]]
    assert head[13:] == ["Quote", "Quote found in text", "Extracted from"]
    vals = {h: c.value for h, c in zip(head, exp[2])}
    assert vals["COVARIATE_hum"] == "veterinary authority staff" and vals["Value_hum"] == 7
    assert vals["Quote found in text"] == "Y" and vals["Extracted from"] == "abstract"
    # The environment sheet has no page column in the template: it is added on the right.
    env_head = [c.value for c in wb["ENV_COV"][1]]
    assert env_head[-4:] == ["Page/section", "Quote", "Quote found in text", "Extracted from"]
    assert wb["ENV_COV"][2][env_head.index("Page/section")].value == "Table 2"
    assert wb["HUMAN_COV_SUSC"].max_row == 1


def test_the_long_csv_has_one_row_per_observation_with_its_sheet():
    import csv
    rows = list(csv.DictReader(io.StringIO(extraction.build_long_csv(_articles(), "AB"))))
    assert [(r["sheet"], r["covariate"]) for r in rows] == [
        ("human_exp", "veterinary authority staff"), ("env", "free ranging")]
    assert rows[0]["id"] == "AB77" and rows[0]["quote"]


def test_the_first_author_is_the_family_name_when_the_stored_form_says_so():
    f = extraction._first_author
    assert f("Dressler A; Wagner-Wiening C") == "Dressler"
    assert f("Dressler, Aparna; Köster, Judith") == "Dressler"
    assert f("Aparna Dressler") == "Aparna Dressler"            # ambiguous: left for the reviewer
    assert f("") == "" and f(None) == ""


# ── Integration: the whole map step, LLM stubbed ────────────────────────────
def _engine_ok() -> bool:
    try:
        with main.engine.connect():
            return True
    except Exception:
        return False


@pytest.fixture()
def seeded(db_conn):
    if not _engine_ok():
        pytest.skip("main.engine cannot reach the database")
    with db_conn.cursor() as cur:
        cur.execute("SELECT to_regclass('user_scenarios') IS NULL")
        if cur.fetchone()[0]:
            main._ensure_user_scenarios_table()
        cur.execute("CREATE TABLE IF NOT EXISTS literature_document (id BIGINT PRIMARY KEY)")
        cur.execute("CREATE TABLE IF NOT EXISTS article_scenarios ("
                    "scenario_id TEXT, document_id BIGINT, PRIMARY KEY (scenario_id, document_id))")
        created_chunk_table = ensure_document_columns(cur)
        for col, typ in (("title", "TEXT"), ("abstract", "TEXT"),
                         ("is_duplicate", "BOOLEAN DEFAULT FALSE")):
            cur.execute(f"ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS {col} {typ}")
        for col, typ in (("similarity_score", "DOUBLE PRECISION"), ("screening_reason", "TEXT"),
                         ("screening_notes", "TEXT"), ("screened_at", "TIMESTAMP")):
            cur.execute(f"ALTER TABLE article_scenarios ADD COLUMN IF NOT EXISTS {col} {typ}")
        cur.execute("ALTER TABLE document_chunk ADD COLUMN IF NOT EXISTS embedding TEXT")
        main._ensure_extraction_columns()
        for sid in (SID, OTHER):
            cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (sid,))
            cur.execute("DELETE FROM scenario_settings WHERE scenario_id = %s", (sid,))
            cur.execute("DELETE FROM user_scenarios WHERE id = %s", (sid,))
        cur.execute("DELETE FROM document_chunk WHERE document_id = ANY(%s)", (list(IDS),))
        cur.execute("DELETE FROM literature_document WHERE id = ANY(%s)", (list(IDS),))
        for sid, name in ((SID, "Avian influenza"), (OTHER, "Another scenario")):
            cur.execute("INSERT INTO user_scenarios (id, name, query, mode, filters, pinned) "
                        "VALUES (%s, %s, 'h5n1', 'boolean', '{}', TRUE)", (sid, name))
        abstract = "Outbreak of avian influenza in a poultry holding. " + PAPER
        for i, t in zip(IDS, ("Above threshold", "Above threshold, full text", "Included by hand",
                              "Below threshold", "Excluded by a reviewer", "Fails")):
            cur.execute("INSERT INTO literature_document (id, title, abstract, authors, doi, year, source, "
                        "is_duplicate, project_context) VALUES (%s,%s,%s,'Dressler A',%s,2026,'pubmed',"
                        "false,'literev')", (i, t, abstract, f"10.1/{i}"))
        cur.execute("INSERT INTO article_scenarios (scenario_id, document_id, similarity_score, screening_status) VALUES "
                    "(%s, 9701, 0.90, NULL), (%s, 9702, 0.85, NULL), (%s, 9703, 0.10, 'included'),"
                    "(%s, 9704, 0.10, NULL), (%s, 9705, 0.95, 'excluded'), (%s, 9706, 0.80, NULL)", (SID,) * 6)
        cur.execute("INSERT INTO article_scenarios (scenario_id, document_id, similarity_score) "
                    "VALUES (%s, 9704, 0.99)", (OTHER,))
        for sid in (SID, OTHER):
            cur.execute("INSERT INTO scenario_settings (scenario_id, similarity_threshold) VALUES (%s, 0.45) "
                        "ON CONFLICT (scenario_id) DO UPDATE SET similarity_threshold = 0.45", (sid,))
        cur.execute("UPDATE literature_document SET has_fulltext = TRUE WHERE id = 9702")
        cur.execute("INSERT INTO document_chunk (document_id, chunk_index, content, chunk_type) VALUES "
                    "(9702, 1, %s, 'fulltext_section'), (9702, 0, %s, 'fulltext_section')",
                    ("Table 4. Veterinary authority staff, 7, none vaccinated. " + "z" * 600, "Methods. " + PAPER))
    yield db_conn
    with db_conn.cursor() as cur:
        for sid in (SID, OTHER):
            cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (sid,))
            cur.execute("DELETE FROM scenario_settings WHERE scenario_id = %s", (sid,))
            cur.execute("DELETE FROM user_scenarios WHERE id = %s", (sid,))
        cur.execute("DELETE FROM document_chunk WHERE document_id = ANY(%s)", (list(IDS),))
        cur.execute("DELETE FROM literature_document WHERE id = ANY(%s)", (list(IDS),))
        if created_chunk_table:
            cur.execute("DROP TABLE document_chunk")


def _run(client):
    extraction._jobs[SID] = {"running": True, "total": 0, "done": 0, "failed": 0}
    return extraction._extract_scenario(SID, "Avian influenza")


def _stored(db_conn, i):
    with db_conn.cursor() as cur:
        cur.execute("SELECT extraction_json, extraction_attempts FROM literature_document WHERE id = %s", (i,))
        return cur.fetchone()


def test_every_relevant_article_is_extracted_once_and_never_the_excluded(seeded, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = _fake_client()
    boom = _fake_client(boom=RuntimeError("boom"))
    real_create = client.chat.completions.create

    def create(**kw):                                       # article 9706 always fails
        if "Fails" in kw["messages"][1]["content"] or json.loads(kw["messages"][1]["content"])["text"].startswith("Fails"):
            return boom.chat.completions.create(**kw)
        return real_create(**kw)
    client.chat.completions.create = create
    patch_app(monkeypatch, "_llm_client", lambda: client)

    res = _run(client)
    # Relevant: 9701, 9702 (above), 9703 (included by hand), 9706 (above). Not 9704, 9705.
    assert res == {"total": 4, "done": 3, "failed": 1}
    for i in (9701, 9702, 9703):
        ex, attempts = _stored(seeded, i)
        assert ex["v"] == extraction.EXTRACTION_VERSION and ex["observations"] and attempts == 0
    assert _stored(seeded, 9702)[0]["source"] == "fulltext"        # it had a full text
    assert _stored(seeded, 9701)[0]["source"] == "abstract"
    assert _stored(seeded, 9704)[0] is None and _stored(seeded, 9705)[0] is None
    ex6, attempts6 = _stored(seeded, 9706)
    assert ex6 is None and attempts6 == 1

    # The full text is read in chunk order, with the title first.
    st = extraction.extraction_status(SID)
    assert st["n_relevant"] == 4 and st["n_extracted"] == 3 and st["n_from_fulltext"] == 1
    assert st["n_from_abstract"] == 2 and st["n_pending"] == 1 and st["n_observations"] == 3

    # Second run: the three cached ones are NOT read again, only the failing one.
    assert _run(client) == {"total": 1, "done": 0, "failed": 1}
    assert _run(client) == {"total": 1, "done": 0, "failed": 1}
    # Three failures: left alone, and said so.
    assert _run(client) == {"total": 0, "done": 0, "failed": 0}
    assert extraction.extraction_status(SID)["n_given_up"] == 1


def test_a_full_text_that_arrives_later_replaces_the_abstract_pass(seeded, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    patch_app(monkeypatch, "_llm_client", lambda: _fake_client())
    _run(None)
    assert _stored(seeded, 9703)[0]["source"] == "abstract"
    with seeded.cursor() as cur:
        cur.execute("UPDATE literature_document SET has_fulltext = TRUE WHERE id = 9703")
        cur.execute("INSERT INTO document_chunk (document_id, chunk_index, content, chunk_type) "
                    "VALUES (9703, 0, %s, 'fulltext_section')", ("Methods. " + PAPER + " " + "y" * 600,))
    assert _run(None)["total"] >= 1
    assert _stored(seeded, 9703)[0]["source"] == "fulltext"


def test_the_article_text_is_capped_per_prompt_and_says_so(seeded, monkeypatch):
    monkeypatch.setattr(extraction, "EXTRACTION_MAX_CHARS", 700)
    with main.engine.connect() as conn:
        text_, source, truncated = extraction._article_text(conn, {"id": 9702, "title": "T", "abstract": "a"})
    assert source == "fulltext" and truncated is True and text_.startswith("T\n\nMethods.")
    assert len(text_) <= 700 + 4


def test_the_endpoints_status_article_export_and_run(seeded, monkeypatch):
    openpyxl = pytest.importorskip("openpyxl")
    from fastapi.testclient import TestClient
    c = TestClient(main.app)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    patch_app(monkeypatch, "_llm_client", lambda: _fake_client())
    _run(None)

    st = c.get(f"/user-scenarios/{SID}/extraction/status").json()
    assert st["n_relevant"] == 4 and st["n_extracted"] == 4 and st["n_pending"] == 0

    art = c.get(f"/user-scenarios/{SID}/articles/9701/extraction").json()
    assert art["extraction"]["observations"][0]["covariate"] == "veterinary authority staff"
    # An article that is not in this scenario is not served through it.
    assert c.get(f"/user-scenarios/{OTHER}/articles/9701/extraction").status_code == 404
    assert c.get("/user-scenarios/nope/extraction/status").status_code == 404

    r = c.get(f"/user-scenarios/{SID}/extraction/export?id_prefix=ISS")
    assert r.status_code == 200 and "spreadsheetml" in r.headers["content-type"]
    assert "4 of the 4 relevant" in r.headers["x-coverage"]
    wb = openpyxl.load_workbook(io.BytesIO(r.content))
    assert [row[0].value for row in wb["REF"].iter_rows(min_row=2)] == ["ISS9701", "ISS9702", "ISS9703", "ISS9706"]
    csv_r = c.get(f"/user-scenarios/{SID}/extraction/export?format=csv")
    assert csv_r.status_code == 200 and "human_exp" in csv_r.text
    assert c.get(f"/user-scenarios/{SID}/extraction/export?format=pdf").status_code == 400

    # Starting a run needs the write key, and answers honestly when nothing is left or no LLM.
    assert c.post(f"/user-scenarios/{SID}/extraction/run").status_code in (401, 403)
    monkeypatch.setenv("OPENAI_API_KEY", "")
    assert c.post(f"/user-scenarios/{SID}/extraction/run", headers=HDR).json()["status"] == "no_llm"
