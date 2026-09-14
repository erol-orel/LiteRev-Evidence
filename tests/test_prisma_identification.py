"""PRISMA "identification" must count what the search RETURNED, not the de-duplicated corpus.

Production, every scenario: "records identified — before de-duplication: 2,766",
"duplicates removed: 0". The box counted corpus documents flagged `is_duplicate`, a flag
no runtime sets. The real de-duplication happens at ingestion (a paper returned by
OpenAlex and PubMed becomes one row; the second arrival is absorbed without a trace)
and at linking (_dedup_scenario_links, whose count was only logged). So 2,766 was the
number AFTER de-duplication, labelled as before.

The fix counts records per source during the search (the local database included),
the distinct documents behind them, and stores the figures on the scenario. Pure tests
pin the arithmetic; the database tests check the endpoint reads the stored figures and
says so when it has to fall back to the corpus.
"""
import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

import main  # noqa: E402


# ── arithmetic (pure) ────────────────────────────────────────────────────────

def test_one_paper_from_three_sources_is_two_duplicates():
    f = main._prisma_identification_figures(
        {"db_cache": 1, "openalex": 1, "pubmed": 1}, unique_records=1,
        duplicate_rows_removed=0, corpus_total=1)
    assert f["records_identified"] == 3
    assert f["duplicates_removed"] == 2
    assert f["unique_records"] == 1
    assert f["removed_other_reasons"] == 0
    assert f["records_screened"] == 1


def test_the_production_shape():
    """Local matches + three live sources, overlaps, three rows merged by the link
    dedup, and a few records dropped for lacking an abstract."""
    f = main._prisma_identification_figures(
        {"db_cache": 557, "openalex": 1673, "europepmc": 907, "pubmed": 89},
        unique_records=2766, duplicate_rows_removed=3, corpus_total=2740,
        method="populate", federation_incomplete=True,
        removed_no_abstract=15, removed_not_matching=8)
    assert f["records_identified"] == 3226
    assert f["duplicate_records_across_sources"] == 460      # 3226 records → 2766 documents
    assert f["duplicate_rows_in_database"] == 3
    assert f["duplicates_removed"] == 463
    assert f["unique_records"] == 2763
    assert f["removed_before_screening"] == 23               # 2763 unique, 2740 screened…
    assert f["removed_no_abstract"] == 15                    # …of which: no abstract,
    assert f["removed_not_matching"] == 8                    # keyword-source records off-query,
    assert f["removed_other_reasons"] == 0                   # and nothing left unexplained
    assert f["records_screened"] == 2740
    assert f["method"] == "populate" and f["federation_incomplete"] is True
    assert f["computed_at"].endswith("+00:00")
    # PRISMA arithmetic holds: identified − duplicates − removals = screened
    assert (f["records_identified"] - f["duplicates_removed"] - f["removed_no_abstract"]
            - f["removed_not_matching"] - f["removed_other_reasons"]) == f["records_screened"]


def test_the_september_14_run():
    """The run that prompted the breakdown: 38 458 records, 4 086 duplicates, 34 372
    unique, 3 861 removed, 30 511 screened — the person reading it wanted to know how
    many of the 3 861 were the no-abstract rule."""
    f = main._prisma_identification_figures(
        {"db_cache": 18367, "openalex": 9000, "europepmc": 7000, "pubmed": 4091},
        unique_records=34372, duplicate_rows_removed=0, corpus_total=30511,
        removed_no_abstract=1266, removed_not_matching=2595)
    assert f["records_identified"] == 38458
    assert f["duplicates_removed"] == 4086
    assert f["unique_records"] == 34372
    assert f["removed_before_screening"] == 3861
    assert (f["removed_no_abstract"], f["removed_not_matching"], f["removed_other_reasons"]) == (1266, 2595, 0)
    assert f["records_screened"] == 30511


def test_inconsistent_inputs_never_go_negative_or_drop_zero_sources():
    f = main._prisma_identification_figures(
        {"db_cache": 5, "core": 0, "arxiv": None}, unique_records=9,
        duplicate_rows_removed=20, corpus_total=50, removed_no_abstract=99, removed_not_matching=99)
    assert f["records_by_source"] == {"db_cache": 5}          # empty sources are not "searched"
    assert f["duplicates_removed"] == 5                       # never more than identified
    assert f["unique_records"] == 0 and f["removed_before_screening"] == 0
    assert f["removed_no_abstract"] == 0 and f["removed_not_matching"] == 0   # bounded by what is left to explain
    # buckets that overshoot are clipped in order, and the residual absorbs the rest
    g = main._prisma_identification_figures({"db_cache": 10}, 10, 0, 4, removed_no_abstract=4, removed_not_matching=4)
    assert (g["removed_no_abstract"], g["removed_not_matching"], g["removed_other_reasons"]) == (4, 2, 0)
    h = main._prisma_identification_figures({"db_cache": 10}, 10, 0, 4)
    assert (h["removed_no_abstract"], h["removed_not_matching"], h["removed_other_reasons"]) == (0, 0, 6)
    empty = main._prisma_identification_figures({}, 0, 0, 0)
    assert empty["records_identified"] == 0 and empty["duplicates_removed"] == 0


# ── database ─────────────────────────────────────────────────────────────────

SID_RUN = "prisma-with-figures"
SID_OLD = "prisma-before-accounting"


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
        cur.execute("DROP TABLE IF EXISTS document_search")
        cur.execute("DROP TABLE IF EXISTS document_chunk, article_scenarios, literature_document CASCADE")
        cur.execute(
            "CREATE TABLE literature_document ("
            "id bigint PRIMARY KEY, title text NOT NULL, source text NOT NULL, abstract text,"
            "year int, journal text, is_duplicate boolean DEFAULT false, screening_status text,"
            "project_context text DEFAULT 'literev')")
        cur.execute(
            "CREATE TABLE document_chunk ("
            "id bigint GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,"
            "document_id bigint NOT NULL REFERENCES literature_document(id) ON DELETE CASCADE,"
            "chunk_index int DEFAULT 0, content text, chunk_type text, embedding text)")
        cur.execute(
            "CREATE TABLE article_scenarios (scenario_id text, document_id bigint,"
            "similarity_score double precision, screening_status text,"
            "PRIMARY KEY (scenario_id, document_id))")
        cur.execute("SELECT to_regclass('user_scenarios') IS NULL")
        if cur.fetchone()[0]:
            main._ensure_user_scenarios_table()
        cur.execute("SELECT to_regclass('scenario_settings') IS NULL")
        if cur.fetchone()[0]:
            main._ensure_scenario_settings_table()
        cur.execute("DELETE FROM user_scenarios WHERE id IN (%s, %s)", (SID_RUN, SID_OLD))
        cur.execute("INSERT INTO user_scenarios (id, name, query, mode, filters) VALUES (%s, %s, %s, 'boolean', '{}')",
                    (SID_RUN, "With figures", "influenza AND wastewater"))
        cur.execute("INSERT INTO user_scenarios (id, name, query, mode, filters) VALUES (%s, %s, %s, 'boolean', '{}')",
                    (SID_OLD, "Before accounting", "influenza AND wastewater"))
        cur.execute(
            "INSERT INTO literature_document (id, title, source, abstract, is_duplicate) VALUES "
            "(1, 'A', 'openalex', 'abstract one long enough to count', false),"
            "(2, 'B', 'pubmed', 'abstract two long enough to count', false),"
            "(3, 'C', 'pubmed', 'abstract three long enough to count', true)")
        cur.execute("INSERT INTO article_scenarios (scenario_id, document_id, similarity_score) VALUES "
                    "(%s, 1, 0.9), (%s, 2, 0.2), (%s, 1, 0.9), (%s, 2, 0.2), (%s, 3, 0.7)",
                    (SID_RUN, SID_RUN, SID_OLD, SID_OLD, SID_OLD))
    yield db_conn
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM user_scenarios WHERE id IN (%s, %s)", (SID_RUN, SID_OLD))


def _prisma(sid):
    from fastapi.testclient import TestClient
    r = TestClient(main.app).get(f"/gesica/scenarios/{sid}/prisma")
    assert r.status_code == 200, r.text
    return r.json()["identification"]


def test_stored_figures_survive_a_round_trip(seeded):
    figures = main._prisma_identification_figures({"db_cache": 557, "openalex": 1673}, 2000, 3, 1990)
    main._store_prisma_identification(SID_RUN, figures)
    loaded = main._load_prisma_identification(SID_RUN)
    assert loaded == json.loads(json.dumps(figures))
    assert main._load_prisma_identification(SID_OLD) is None
    assert main._load_prisma_identification("no-such-scenario") is None


def test_the_endpoint_reports_the_search_run(seeded):
    main._store_prisma_identification(SID_RUN, main._prisma_identification_figures(
        {"db_cache": 557, "openalex": 1673, "europepmc": 907, "pubmed": 89}, 2766, 3, 2740,
        removed_no_abstract=15, removed_not_matching=8))
    ident = _prisma(SID_RUN)
    assert ident["figures_from"] == "search_run"
    assert ident["total_records"] == 3226                    # what the sources returned
    assert ident["duplicates_removed"] == 463
    assert ident["unique_records"] == 2763
    assert ident["removed_before_screening"] == 23
    assert ident["removed_no_abstract"] == 15
    assert ident["removed_not_matching"] == 8
    assert ident["removed_other_reasons"] == 0
    assert ident["by_source"] == {"db_cache": 557, "openalex": 1673, "europepmc": 907, "pubmed": 89}
    assert ident["records_screened"] == 2                    # the corpus as it stands NOW
    assert ident["computed_at"]


def test_without_figures_the_endpoint_falls_back_and_says_so(seeded):
    ident = _prisma(SID_OLD)
    assert ident["figures_from"] == "corpus"
    assert ident["total_records"] == 3                       # the de-duplicated corpus, as before
    assert ident["duplicates_removed"] == 1                  # only the is_duplicate flag, as before
    assert ident["unique_records"] == 2
    assert ident["by_source"]["pubmed"] == 2 and ident["by_source"]["openalex"] == 1
