"""Everything the register shows can be taken away as a file.

A reviewer told us there was no export on the variables of interest tab. There was one: a
grey text link in the top right corner of one of the three tables. The other two, the
crossed grid and the selection, had none at all, so the only way to get the figure a
partner asks for was to retype it. That is the work this tool exists to remove, and a
feature nobody can find is not a feature.

So the three exports are pinned here, and with them the two properties that make an
exported file trustworthy on its own:

  1. IT CONTAINS WHAT THE SCREEN SHOWS. The grid's CSV is the matrix, with the same axis
     totals, and it repeats the caveat that an axis total counts distinct papers and is
     therefore not the sum of its cells. The selection's file holds the papers the
     selection KEEPS, the same ones the preview counts, because both come from the same
     plan.
  2. IT SAYS WHAT IT IS. The selection's file carries the selection that produced it and
     states that nothing was applied to the scenario: a list of articles that travels
     without saying where it came from is not data.

And the one thing an export must never do: write. Exporting a selection must leave every
screening decision exactly as it was, or a reviewer who clicked "download" would silently
narrow the corpus of every other panel.
"""
import json

import pytest

pytest.importorskip("fastapi")

import main  # noqa: E402
from conftest import ensure_document_columns  # noqa: E402

SID = "usr-register-export"
IDS = (9731, 9732, 9733)


def _engine_ok() -> bool:
    try:
        with main.engine.connect():
            return True
    except Exception:
        return False


def _extraction(place: str, occupation: str) -> str:
    return json.dumps({
        "source": "abstract",
        "observations": [{"place": place, "population_group": "human_exp",
                          "variable": "occupation", "value": occupation}],
    })


@pytest.fixture()
def seeded(db_conn):
    """Three papers, two of them relevant, with a country and an occupation each, so the
    grid has something to cross and the selection something to keep."""
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
        for col, typ in (("title", "TEXT"), ("abstract", "TEXT"), ("year", "INTEGER"),
                         ("journal", "TEXT"), ("doi", "TEXT"), ("country", "TEXT"),
                         ("pico_json", "JSONB"), ("extraction_json", "JSONB"),
                         ("is_duplicate", "BOOLEAN DEFAULT FALSE")):
            cur.execute(f"ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS {col} {typ}")
        for col, typ in (("similarity_score", "DOUBLE PRECISION"),
                         ("screening_reason", "TEXT"), ("screened_at", "TIMESTAMP")):
            cur.execute(f"ALTER TABLE article_scenarios ADD COLUMN IF NOT EXISTS {col} {typ}")
        cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM scenario_settings WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))
        cur.execute("DELETE FROM literature_document WHERE id = ANY(%s)", (list(IDS),))
        cur.execute("INSERT INTO user_scenarios (id, name, query, mode, filters, pinned) "
                    "VALUES (%s, 'Register export', 'avian influenza', 'boolean', '{}', TRUE)",
                    (SID,))
        cur.execute("""
            INSERT INTO literature_document
              (id, title, abstract, source, year, journal, doi, country, is_duplicate,
               project_context, pico_json, extraction_json)
            VALUES
              (9731, 'Poultry workers in Lombardy', %(a)s, 'pubmed', 2024, 'Eurosurveillance',
               '10.1/a', 'IT', FALSE, 'literev', %(p1)s::jsonb, %(x1)s::jsonb),
              (9732, 'Market vendors in Guangdong', %(a)s, 'openalex', 2023, 'EID',
               '10.1/b', 'CN', FALSE, 'literev', %(p2)s::jsonb, %(x2)s::jsonb),
              (9733, 'Off topic', %(a)s, 'pubmed', 2010, NULL, NULL, NULL, FALSE,
               'literev', NULL, NULL)
        """, {"a": "a study of exposure among poultry workers, with counts by sex",
              "p1": json.dumps({"study_design": "Cross-sectional study"}),
              "p2": json.dumps({"study_design": "Cohort study"}),
              "x1": _extraction("Lombardy, Italy", "poultry worker"),
              "x2": _extraction("Guangdong, China", "market vendor")})
        cur.execute("""
            INSERT INTO article_scenarios (scenario_id, document_id, similarity_score)
            VALUES (%s, 9731, 0.90), (%s, 9732, 0.85), (%s, 9733, 0.10)
        """, (SID, SID, SID))
        cur.execute("INSERT INTO scenario_settings (scenario_id, similarity_threshold) "
                    "VALUES (%s, 0.45) ON CONFLICT (scenario_id) DO UPDATE "
                    "SET similarity_threshold = 0.45", (SID,))
    yield db_conn
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM scenario_settings WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))
        cur.execute("DELETE FROM literature_document WHERE id = ANY(%s)", (list(IDS),))
        if created_chunk_table:
            cur.execute("DROP TABLE document_chunk")


def _client():
    from fastapi.testclient import TestClient

    return TestClient(main.app)


# ── The register's own map ──────────────────────────────────────────────────
def test_the_register_exports_one_row_per_variable_and_opens_in_excel(seeded):
    """The BOM is the whole reason this assertion exists: without it Excel reads UTF-8 as
    latin-1 and a French label opens as mojibake. It moved into a shared helper, so it is
    pinned here rather than trusted."""
    r = _client().get(f"/user-scenarios/{SID}/variables-of-interest/export", params={"lang": "en"})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/csv")
    body = r.text
    assert body.startswith("﻿"), "no BOM: Excel would mangle every accent"
    head = body.lstrip("﻿").splitlines()[0]
    assert head.startswith("variable,name,family")
    assert "papers_with_a_value" in head and "denominator" in head


# ── The crossed grid ────────────────────────────────────────────────────────
def test_the_grid_exports_the_matrix_with_its_totals(seeded):
    """The figure a partner copies into a report, in the shape they paste it into: the
    matrix itself, then its axis totals, then what the grid could not place."""
    r = _client().get(f"/user-scenarios/{SID}/variables-of-interest/crosstab/export",
                      params={"rows": "country", "cols": "design", "lang": "en"})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/csv")
    assert 'filename="crosstab-country-x-design-' in r.headers["content-disposition"]
    body = r.text.lstrip("﻿")
    lines = [ln for ln in body.splitlines() if ln.strip()]
    assert lines[0].startswith("country \\ design"), lines[0]
    assert "Cross-sectional study" in lines[0] or "Cohort study" in lines[0]
    # Les deux pays sont des lignes, chacun avec son total.
    assert any(ln.startswith("Italy") for ln in lines), body
    assert any(ln.startswith("China") for ln in lines), body
    assert any(ln.startswith("all (country)") for ln in lines), body
    assert "papers both variables can judge,2" in body
    # Et la mise en garde qui empêche de lire la dernière ligne comme une somme.
    assert "not the sum of its cells" in body


def test_a_grid_that_cannot_be_built_exports_its_reason(seeded):
    """The clustering may not be computed. The file then says why, in one line, instead of
    the request failing: someone who clicked download is owed an answer."""
    r = _client().get(f"/user-scenarios/{SID}/variables-of-interest/crosstab/export",
                      params={"rows": "cluster", "cols": "design", "lang": "en"})
    assert r.status_code == 200, r.text
    assert r.text.strip(), "an empty file says nothing at all"


# ── The selection ───────────────────────────────────────────────────────────
def test_the_selection_exports_the_papers_it_keeps(seeded):
    """The same plan the preview counts, so the file cannot disagree with the number the
    reviewer read before clicking."""
    sel = {"conditions": [{"variable": "country", "values": ["IT"]}], "combine": "all"}
    c = _client()
    plan = c.post(f"/user-scenarios/{SID}/subset/preview", json=sel)
    assert plan.status_code == 200, plan.text
    kept = plan.json()["keep"]

    r = c.post(f"/user-scenarios/{SID}/subset/export", json=sel, params={"format": "json"})
    assert r.status_code == 200, r.text
    doc = r.json()
    assert len(doc["articles"]) == kept == 1
    assert doc["articles"][0]["id"] == 9731
    assert r.headers["x-article-count"] == str(kept)


def test_an_exported_selection_says_which_selection_it_is(seeded):
    """A list of articles that travels without its definition is not data. The file names
    the selection, counts what it kept out of the relevant subset, and states that nothing
    was applied to the scenario."""
    sel = {"conditions": [{"variable": "country", "values": ["IT", "CN"]}], "combine": "any"}
    r = _client().post(f"/user-scenarios/{SID}/subset/export", json=sel,
                       params={"format": "json"})
    assert r.status_code == 200, r.text
    meta = r.json()["meta"]
    assert meta["applied"] is False
    assert meta["kept"] == 2 and meta["relevant"] == 2
    assert meta["selection"] and "coverage" in meta
    assert "Aucun découpage n'a été appliqué" in meta["coverage"]


def test_exporting_a_selection_writes_nothing(seeded):
    """The property that matters most. Applying a selection excludes the rest of the
    corpus from this scenario and accumulates; exporting must leave every decision exactly
    as it was, or a download would narrow every other panel without saying so."""
    with seeded.cursor() as cur:
        cur.execute("SELECT document_id, screening_status, screening_reason "
                    "FROM article_scenarios WHERE scenario_id = %s ORDER BY document_id", (SID,))
        before = cur.fetchall()

    sel = {"conditions": [{"variable": "country", "values": ["IT"]}], "combine": "all"}
    r = _client().post(f"/user-scenarios/{SID}/subset/export", json=sel)
    assert r.status_code == 200, r.text

    with seeded.cursor() as cur:
        cur.execute("SELECT document_id, screening_status, screening_reason "
                    "FROM article_scenarios WHERE scenario_id = %s ORDER BY document_id", (SID,))
        assert cur.fetchall() == before, "the export changed a screening decision"


def test_a_selection_without_a_condition_is_refused(seeded):
    """Exporting "everything" is the relevant export, which already exists. An empty
    selection here would quietly mean something else, so it is refused by name."""
    r = _client().post(f"/user-scenarios/{SID}/subset/export", json={"combine": "all"})
    assert r.status_code == 422, r.text
    assert "condition" in r.text.lower()
