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
        cur.execute("ALTER TABLE document_chunk ADD COLUMN IF NOT EXISTS metadata_json JSONB")
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


# ── The reduce half: counts over every relevant article ─────────────────────
def _ex(source, coverage, observations):
    return json.dumps({"v": 1, "source": source, "truncated": False,
                       "ref": {}, "coverage": {k: True for k in coverage}, "observations": observations})


def _obs(group, covariate, n_cases, pop_risk, verified, sheet="human_susc"):
    return {"sheet": sheet, "group": group, "covariate": covariate, "value": None, "descr": None,
            "n_cases": n_cases, "pop_risk": pop_risk, "quote": "q", "quote_verified": verified}


@pytest.fixture()
def extracted(seeded):
    """9701 and 9702 read from the full text, 9703 from the abstract, 9706 not read at all;
    9704 (below the threshold) and 9705 (excluded) carry extractions that must NOT count."""
    rows = {
        9701: _ex("fulltext", ["sex_gender"], [_obs("sex", "Male", 3, 10, True),
                                               _obs("sex", "female", 2, 20, False)]),
        9702: _ex("fulltext", ["sex_gender", "age"], [_obs("sex", "male ", 5, 50, True),
                                                      _obs("occupation", "farmer", None, None, True, "human_exp")]),
        9703: _ex("abstract", ["ppe"], []),
        9704: _ex("fulltext", ["sex_gender", "age", "ppe"], [_obs("sex", "male", 99, 999, True)]),
        9705: _ex("fulltext", ["sex_gender"], [_obs("sex", "male", 99, 999, True)]),
    }
    with seeded.cursor() as cur:
        for i, j in rows.items():
            cur.execute("UPDATE literature_document SET extraction_json = %s::jsonb WHERE id = %s", (j, i))
    return seeded


def test_the_digest_counts_every_relevant_extracted_article_and_only_those(extracted):
    d = extraction.extraction_digest(SID)
    assert d["complete"] is True
    assert (d["n_relevant"], d["n_extracted"], d["n_unread"]) == (4, 3, 1)
    assert (d["n_fulltext"], d["n_abstract"]) == (2, 1)
    # The below-threshold and the excluded article are out of every count.
    assert d["coverage"]["sex_gender"] == 2 and d["coverage"]["age"] == 1 and d["coverage"]["ppe"] == 1
    assert d["coverage_fulltext"]["sex_gender"] == 2 and d["coverage_fulltext"]["ppe"] == 0
    assert d["coverage"]["vector"] == 0
    by_sheet = {r["sheet"]: r for r in d["by_sheet"]}
    assert by_sheet["human_susc"] == {"sheet": "human_susc", "n_rows": 3, "n_articles": 2, "n_quote_found": 2}
    assert by_sheet["human_exp"]["n_rows"] == 1
    assert {"sheet": "human_susc", "value": "sex", "n": 2} in d["top_groups"]
    # Crude counts: only rows with cases AND population AND a quote found; labels folded.
    assert d["crude_counts"] == [{"sheet": "human_susc", "covariate": "male", "n_studies": 2,
                                  "n_cases": 8.0, "pop_risk": 60.0}]
    assert "Not a pooled estimate" in d["crude_counts_note"]


def test_the_digest_follows_the_threshold_it_is_given(extracted):
    low = extraction.extraction_digest(SID, threshold=0.0)
    assert low["n_relevant"] == 5                       # 9704 (below 0.45) now counts; 9705 never does
    assert low["coverage"]["sex_gender"] == 3


def test_the_prompt_block_states_its_denominator_and_its_source(extracted):
    block = extraction.extraction_to_prompt(extraction.extraction_digest(SID))
    assert block.startswith("EXTRACTION STRUCTUREE: 3 des 4 articles pertinents")
    assert "2 depuis le texte integral, 1 depuis le resume seul" in block
    assert "1 articles pertinents ne sont pas encore extraits" in block
    assert "sexe ou genre 2 (dont 2 en texte integral)" in block
    assert "male [human_susc] 2 etudes, 8 cas sur 60" in block and "pas une estimation poolee" in block
    assert "denominateur (3 articles extraits sur 4)" in block
    assert chr(0x2014) not in block


def test_the_prompt_block_says_nothing_when_there_is_nothing_to_say():
    assert extraction.extraction_to_prompt(None) == ""
    assert extraction.extraction_to_prompt({"complete": True, "n_extracted": 0, "n_relevant": 9}) == ""
    # An aggregation that failed must assert nothing, not a total of zero.
    assert extraction.extraction_to_prompt({"complete": False, "n_extracted": 5, "n_relevant": 9}) == ""
    none_reported = {"complete": True, "n_extracted": 2, "n_relevant": 2, "n_fulltext": 2, "n_abstract": 0,
                     "n_unread": 0, "coverage": {k: 0 for k in extraction.COVERAGE_KEYS},
                     "coverage_fulltext": {k: 0 for k in extraction.COVERAGE_KEYS}}
    assert "aucun des elements suivis" in extraction.extraction_to_prompt(none_reported)


def test_the_coverage_endpoint_serves_the_same_digest(extracted):
    from fastapi.testclient import TestClient
    c = TestClient(main.app)
    body = c.get(f"/user-scenarios/{SID}/extraction/coverage").json()
    assert body["n_extracted"] == 3 and body["coverage"]["sex_gender"] == 2
    assert c.get("/user-scenarios/nope/extraction/coverage").status_code == 404


def test_both_assistant_paths_put_the_extraction_digest_in_their_prompt():
    import inspect

    from api import assistant
    for fn in (assistant.ask_stream_filtered, assistant.user_scenario_rag_assistant):
        src = inspect.getsource(fn)
        assert "extraction_digest" in src and "extraction_to_prompt" in src, fn.__name__
        assert "EXTRACTION STRUCTUREE" in src, f"{fn.__name__} does not tell the model to quote the denominator"


def test_the_scenario_assistant_really_receives_the_block(extracted, monkeypatch):
    """End to end through the endpoint, LLM stubbed: what the model is sent contains the
    extraction counts and the instruction to quote the denominator."""
    import llm_usage
    from fastapi.testclient import TestClient
    seen: list = []

    class _Fake:
        def __init__(self, *a, **kw):
            def _boom(**_):
                raise RuntimeError("no embeddings here")        # falls back to the lexical branch
            def _chat(**kw):
                seen.append(kw["messages"])
                msg = types.SimpleNamespace(content="ok")
                return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])
            self.embeddings = types.SimpleNamespace(create=_boom)
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=_chat))

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(llm_usage, "MeteredOpenAI", _Fake)
    out = TestClient(main.app).post(f"/user-scenarios/{SID}/rag",
                                    json={"question": "vaccinated veterinary"}).json()
    assert out["answer"] == "ok", out
    system, user = seen[0][0]["content"], seen[0][1]["content"]
    assert "EXTRACTION STRUCTUREE: 3 des 4 articles pertinents" in user
    assert "denominateur" in system and "EXTRACTION STRUCTUREE" in system
