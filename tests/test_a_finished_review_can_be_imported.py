"""A review already screened elsewhere can be brought into a scenario (api/review_import.py).

The ISS team screened 1,524 records in Rayyan before LiteRev ever saw them. Those
decisions are work: re-doing them by hand was the only way to put them into a scenario.
A screening export now carries them over, with the reason each exclusion gives.

Two rules are what these tests pin. A model never decides a paper's fate: the file is read
by column name, and the only question a model may answer is which column is which. And
nothing is overwritten in silence: a paper this scenario already decided otherwise is
reported as a conflict and left as it was unless the caller asks for the file to win, and
every import can be undone.
"""
import io
import json

import pytest

pytest.importorskip("fastapi")

import main  # noqa: E402

SID = "import-decisions"
KEY = {"X-API-Key": "test-write-key"}

# Their export: one included, one excluded with a reason, one maybe, one they included
# that this corpus does not hold, and one row with nothing to decide.
EXPORT = (
    "Key,Title,Authors,DOI,PMID,Year,Decision,Exclusion Reason\n"
    "r1,Seroprevalence among poultry cullers in Lombardy,Rossi M,10.5555/imp.1,,2019,Included,\n"
    "r2,A recombinant vaccine candidate in mice,Bianchi L,10.5555/imp.2,,2020,Exclude: wrong population,Non-target population\n"
    "r3,Environmental persistence of H5N8 in manure,Weber K,10.5555/imp.3,,2021,Maybe,\n"
    "r4,An Embase-only paper this corpus never had,Verdi A,10.5555/imp.4,,2018,Included,\n"
    "r5,A row with no decision at all,Neri P,10.5555/imp.5,,2017,,\n"
)

# The same decisions under headers no one recognises: only the mapping differs.
ODD_EXPORT = (
    "ref;libelle;identifiant;verdict_du_relecteur;pourquoi\n"
    "1;Seroprevalence among poultry cullers in Lombardy;10.5555/imp.1;Inclus;\n"
    "2;A recombinant vaccine candidate in mice;10.5555/imp.2;Exclu;Mauvaise population\n"
)


# ── Pure: reading a decision ─────────────────────────────────────────────────
def test_a_cell_is_read_the_way_screening_tools_write_it():
    d = main.decision_of
    assert d("Included") == d("include") == d("yes") == d("1") == d("Keep") == "included"
    assert d("Excluded") == d("exclude") == d("no") == d("0") == d("Rejected") == "excluded"
    assert d("Exclude: wrong population") == "excluded"
    assert d("Included (full text)") == "included"
    assert d("excluded - no abstract") == "excluded"
    assert d("Inclus") == "included" and d("Exclu") == "excluded"


def test_a_cell_that_decides_nothing_decides_nothing():
    d = main.decision_of
    for value in ("", None, "maybe", "Unsure", "pending", "conflict", "a sentence about the paper"):
        assert d(value) is None, value


def test_a_rayyan_note_carries_the_decision_and_its_reason():
    """Rayyan's export, the one a team most often arrives with, puts the decision inside
    its notes field, under no column of its own: read as a plain reason column, every row
    of the ISS team's 1,524 came back undecided."""
    r = main.rayyan_decision
    assert r('RAYYAN-INCLUSION: {"Claudia"=>"Excluded"} | RAYYAN-EXCLUSION-REASONS: wrong study design') \
        == ("excluded", "wrong study design")
    assert r('RAYYAN-INCLUSION: {"Claudia"=>"Included"}') == ("included", None)
    assert r('RAYYAN-INCLUSION: {"Claudia"=>"Included"} | RAYYAN-LABELS: persistence') == ("included", None)
    # Several reviewers are read as this project reads its own votes: agreement decides.
    assert r('RAYYAN-INCLUSION: {"A"=>"Included", "B"=>"Included"}') == ("included", None)
    assert r('RAYYAN-INCLUSION: {"A"=>"Included", "B"=>"Excluded"}')[0] is None
    assert r('RAYYAN-INCLUSION: {"A"=>"Maybe"}')[0] is None
    assert r("a plain note") == (None, None) and r(None) == (None, None)


def test_a_rayyan_row_never_carries_a_reviewer_name_into_the_corpus():
    """The note holds who voted. The corpus needs the decision, not the person."""
    row = ("key,title,doi,notes\n"
           'k1,A paper about poultry workers,10.5555/imp.9,"RAYYAN-INCLUSION: {""Claudia""=>""Excluded""}'
           ' | RAYYAN-EXCLUSION-REASONS: Extra EU"\n')
    recs, _header, cols = main.records_with_decisions(row.encode(), "csv", None)
    assert cols.get("reason") == "notes" and "decision" not in cols
    assert len(recs) == 1
    assert recs[0]["decision"] == "excluded"
    assert recs[0]["reason"] == "Extra EU", "the reason Rayyan names, not the whole note"
    assert "Claudia" not in str(recs[0]["reason"]) and "RAYYAN" not in str(recs[0]["reason"])


def test_the_decision_and_reason_columns_are_found_by_name():
    cols = main.map_decision_columns(["Key", "Title", "Authors", "DOI", "Decision", "Exclusion Reason", "Year"])
    assert cols["decision"] == "Decision" and cols["reason"] == "Exclusion Reason"
    assert cols["title"] == "Title" and cols["doi"] == "DOI"
    assert "decision" not in main.map_decision_columns(["ref", "libelle", "verdict_du_relecteur"])


# ── With a database ──────────────────────────────────────────────────────────
def _engine_ok() -> bool:
    try:
        with main.engine.connect():
            return True
    except Exception:
        return False


DOCS = [
    (970001, "Seroprevalence among poultry cullers in Lombardy", "10.5555/imp.1"),
    (970002, "A recombinant vaccine candidate in mice", "10.5555/imp.2"),
    (970003, "Environmental persistence of H5N8 in manure", "10.5555/imp.3"),
    (970005, "A row with no decision at all", "10.5555/imp.5"),
]


@pytest.fixture()
def seeded(db_conn, monkeypatch):
    """The scenario, four of their five papers in its corpus, and no background work."""
    from conftest import patch_app
    from search_tables import ensure_search_tables
    if not _engine_ok():
        pytest.skip("main.engine cannot reach the database")
    retrained: list[str] = []
    patch_app(monkeypatch, "schedule_retrain", lambda sid: retrained.append(sid) or True)
    patch_app(monkeypatch, "score_new_links", lambda sid: True)
    # No model is ever asked in these tests: a mapping is given, or absent.
    patch_app(monkeypatch, "suggest_mapping", lambda header, rows, lang=None: {})
    with db_conn.cursor() as cur:
        created_chunk_table = ensure_search_tables(cur)
        for col, typ in (("reviewer_2_status", "VARCHAR(20)"), ("kappa_final_status", "VARCHAR(20)"),
                         ("kappa_resolved", "BOOLEAN DEFAULT FALSE"), ("learned_score", "DOUBLE PRECISION"),
                         ("rerank_score", "DOUBLE PRECISION")):
            cur.execute(f"ALTER TABLE article_scenarios ADD COLUMN IF NOT EXISTS {col} {typ}")
        for col, typ in (("chunk_weight", "DOUBLE PRECISION DEFAULT 1.0"),
                         ("metadata_json", "JSONB DEFAULT '{}'::jsonb"), ("token_count", "INTEGER")):
            cur.execute(f"ALTER TABLE document_chunk ADD COLUMN IF NOT EXISTS {col} {typ}")
        cur.execute("SELECT column_default FROM information_schema.columns"
                    " WHERE table_name = 'literature_document' AND column_name = 'id'")
        previous_default = (cur.fetchone() or [None])[0]
        if not previous_default:
            cur.execute("CREATE SEQUENCE IF NOT EXISTS import_decisions_id_seq")
            cur.execute("SELECT setval('import_decisions_id_seq',"
                        " GREATEST(1000000, COALESCE((SELECT MAX(id) FROM literature_document), 0) + 1))")
            cur.execute("ALTER TABLE literature_document ALTER COLUMN id"
                        " SET DEFAULT nextval('import_decisions_id_seq')")
        main._ensure_import_tables()
        main._ensure_additions_table()
        _clean(cur)
        cur.execute("INSERT INTO user_scenarios (id, name, query, mode, filters)"
                    " VALUES (%s, 'Import', 'avian influenza', 'boolean', '{}')", (SID,))
        for doc_id, title, doi in DOCS:
            cur.execute("INSERT INTO literature_document (id, title, abstract, source, project_context,"
                        " doi, title_norm) VALUES (%s, %s, 'An abstract.', 'pubmed', 'literev', %s, %s)",
                        (doc_id, title, doi, main._normalize_title(title)))
            cur.execute("INSERT INTO article_scenarios (scenario_id, document_id, similarity_score)"
                        " VALUES (%s, %s, 0.7)", (SID, doc_id))
    yield db_conn, retrained
    with db_conn.cursor() as cur:
        _clean(cur)
        if not previous_default:
            cur.execute("ALTER TABLE literature_document ALTER COLUMN id DROP DEFAULT")
            cur.execute("DROP SEQUENCE IF EXISTS import_decisions_id_seq")
        if created_chunk_table:
            cur.execute("DROP TABLE document_chunk")


def _clean(cur) -> None:
    cur.execute("DELETE FROM scenario_decision_import WHERE scenario_id = %s", (SID,))
    cur.execute("DELETE FROM scenario_additions WHERE scenario_id = %s", (SID,))
    cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (SID,))
    cur.execute("DELETE FROM document_chunk WHERE document_id IN (SELECT id FROM literature_document"
                " WHERE source = 'upload' OR doi LIKE '10.5555/imp.%%')")
    cur.execute("DELETE FROM article_scenarios WHERE document_id IN (SELECT id FROM literature_document"
                " WHERE source = 'upload' OR doi LIKE '10.5555/imp.%%')")
    cur.execute("DELETE FROM literature_document WHERE source = 'upload' OR doi LIKE '10.5555/imp.%%'")
    cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))


def _client():
    from fastapi.testclient import TestClient
    return TestClient(main.app)


def _file(content: str, name: str = "rayyan.csv"):
    return {"file": (name, io.BytesIO(content.encode()), "text/csv")}


def _status(conn, doc_id: int):
    with conn.cursor() as cur:
        cur.execute("SELECT screening_status, screening_reason FROM article_scenarios"
                    " WHERE scenario_id = %s AND document_id = %s", (SID, doc_id))
        return cur.fetchone()


def test_the_preview_says_what_would_happen_and_writes_nothing(seeded):
    conn, retrained = seeded
    r = _client().post(f"/user-scenarios/{SID}/decisions/preview", headers=KEY, files=_file(EXPORT))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mapping"]["decision"] == "Decision" and body["mapping"]["reason"] == "Exclusion Reason"
    counts = body["counts"]
    assert counts["records"] == 5
    assert counts["apply"] == 2                    # the included and the excluded, both in the corpus
    assert counts["add_and_include"] == 1          # the one their review has and this corpus does not
    assert counts["no_decision"] == 2              # the maybe, and the empty cell
    by_action = {row["title"][:20]: row["action"] for row in body["plan"]}
    assert by_action["Seroprevalence among"] == "apply"
    assert by_action["Environmental persis"] == "no_decision"
    assert all(_status(conn, d)[0] is None for d, _t, _x in DOCS), "a preview decides nothing"
    assert retrained == []


def test_importing_applies_the_decisions_with_their_reasons(seeded):
    conn, retrained = seeded
    r = _client().post(f"/user-scenarios/{SID}/decisions/import", headers=KEY, files=_file(EXPORT))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["applied"] == 2 and body["added"] == 1
    assert _status(conn, 970001) == ("included", None)
    assert _status(conn, 970002) == ("excluded", "Non-target population"), "the reason comes across"
    assert _status(conn, 970003)[0] is None, "a maybe is not a decision"
    assert _status(conn, 970005)[0] is None
    # Their inclusion the corpus did not hold is now in it, included by hand.
    with conn.cursor() as cur:
        cur.execute("SELECT a.screening_status FROM article_scenarios a JOIN literature_document d"
                    " ON d.id = a.document_id WHERE a.scenario_id = %s AND d.doi = '10.5555/imp.4'", (SID,))
        assert cur.fetchone() == ("included",)
    assert retrained and retrained[0] == SID, "imported decisions teach the queue at once"


def test_a_decision_this_review_already_made_is_never_overwritten_in_silence(seeded):
    conn, _retrained = seeded
    with conn.cursor() as cur:                     # a reviewer here included what their file excludes
        cur.execute("UPDATE article_scenarios SET screening_status = 'included',"
                    " screening_reason = 'our own reading' WHERE scenario_id = %s AND document_id = 970002", (SID,))
    c = _client()
    body = c.post(f"/user-scenarios/{SID}/decisions/import", headers=KEY, files=_file(EXPORT)).json()
    assert body["counts"]["conflict"] == 1
    assert [x["id"] for x in body["conflicts"]] == [970002]
    assert _status(conn, 970002) == ("included", "our own reading"), "left exactly as it was"
    # Asked for, the file wins, and the conflict is counted as such.
    body2 = c.post(f"/user-scenarios/{SID}/decisions/import?overwrite=true", headers=KEY,
                   files=_file(EXPORT)).json()
    assert body2["counts"]["overwrite_conflict"] == 1
    assert _status(conn, 970002) == ("excluded", "Non-target population")


def test_an_import_can_be_undone(seeded):
    conn, _retrained = seeded
    c = _client()
    before = _status(conn, 970001)
    done = c.post(f"/user-scenarios/{SID}/decisions/import", headers=KEY, files=_file(EXPORT)).json()
    assert _status(conn, 970001)[0] == "included"
    listed = c.get(f"/user-scenarios/{SID}/decisions/imports").json()["imports"]
    assert listed[0]["id"] == done["import_id"] and listed[0]["undone_at"] is None
    undone = c.post(f"/user-scenarios/{SID}/decisions/undo", headers=KEY)
    assert undone.status_code == 200, undone.text
    assert _status(conn, 970001) == before, "put back exactly as it was"
    assert _status(conn, 970002)[0] is None
    assert c.post(f"/user-scenarios/{SID}/decisions/undo", headers=KEY).status_code == 404
    assert c.get(f"/user-scenarios/{SID}/decisions/imports").json()["imports"][0]["undone_at"] is not None


def test_unknown_headers_are_imported_with_a_mapping_a_person_confirms(seeded):
    conn, _retrained = seeded
    c = _client()
    # Without a mapping nothing is read at all, and the preview SAYS so, with the header it
    # saw: that is what the screen offers a mapping from, instead of refusing the file.
    plain = c.post(f"/user-scenarios/{SID}/decisions/preview", headers=KEY,
                   files=_file(ODD_EXPORT, "export.csv")).json()
    assert plain["unreadable"] and plain["mapping"] == {} and plain["counts"]["records"] == 0
    assert plain["header"] == ["ref", "libelle", "identifiant", "verdict_du_relecteur", "pourquoi"]
    # With the mapping (what a model proposes, and a person confirms), it reads normally.
    mapping = json.dumps({"title": "libelle", "doi": "identifiant",
                          "decision": "verdict_du_relecteur", "reason": "pourquoi"})
    r = c.post(f"/user-scenarios/{SID}/decisions/import", headers=KEY,
               files=_file(ODD_EXPORT, "export.csv"), data={"mapping": mapping})
    assert r.status_code == 200, r.text
    assert r.json()["applied"] == 2
    assert _status(conn, 970001)[0] == "included"
    assert _status(conn, 970002) == ("excluded", "Mauvaise population")


def test_the_model_is_asked_for_the_mapping_only_when_the_headers_are_unknown(seeded, monkeypatch):
    from conftest import patch_app
    asked: list[list] = []
    patch_app(monkeypatch, "suggest_mapping",
              lambda header, rows, lang=None: asked.append(list(header)) or {"decision": "verdict_du_relecteur"})
    c = _client()
    c.post(f"/user-scenarios/{SID}/decisions/preview", headers=KEY, files=_file(EXPORT))
    assert asked == [], "a header that names its columns needs no model"
    body = c.post(f"/user-scenarios/{SID}/decisions/preview", headers=KEY,
                  files=_file(ODD_EXPORT, "export.csv")).json()
    assert asked and "verdict_du_relecteur" in asked[0]
    assert body["suggested_mapping"] == {"decision": "verdict_du_relecteur"}
    assert body["counts"]["apply"] == 0, "a suggestion is not applied until a person passes it back"


def test_the_import_needs_the_key_and_refuses_what_it_cannot_read(seeded):
    c = _client()
    assert c.post(f"/user-scenarios/{SID}/decisions/import", files=_file(EXPORT)).status_code == 401
    r = c.post(f"/user-scenarios/{SID}/decisions/preview", headers=KEY,
               files={"file": ("paper.pdf", io.BytesIO(b"%PDF-1.4 ..."), "application/pdf")})
    assert r.status_code == 422 and "no decision column" in r.json()["detail"].lower()
    r = c.post(f"/user-scenarios/{SID}/decisions/preview", headers=KEY,
               files={"file": ("notes.txt", io.BytesIO(b"just prose"), "text/plain")})
    assert r.status_code == 422
    assert c.post("/user-scenarios/usr-nope/decisions/import", headers=KEY,
                  files=_file(EXPORT)).status_code == 404
