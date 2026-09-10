"""Full-text corpus membership (lexical_search.py) — the compiler, the triggers, the
backfill, and the semantics a searcher will notice.

Production numbers that motivate this: the LIKE path took 55 to 240 s per boolean
query on 346 152 documents, and `%ai%` matched inside "chain" (238 438 "AI" papers).
Pure tests cover the compilation; the database tests run against the CI Postgres
service and skip cleanly without one. They own the corpus tables the way the other
integration tests do and reinstall the module's DDL on them, because earlier tests
recreate those tables without triggers.
"""
import os
import re

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("fastapi")

import lexical_search as lex  # noqa: E402
import main  # noqa: E402


def _ast(q):
    return main._parse_boolean_ast(main._tokenize_boolean(q))


# ── compilation (pure) ───────────────────────────────────────────────────────

def test_the_whole_boolean_is_one_tsquery_with_bound_terms():
    params: dict = {}
    sql = lex.ast_to_tsquery_sql(_ast('(mpox OR monkeypox) AND ("case count" OR incidence)'), params)
    assert sql.count("phraseto_tsquery") == 4
    assert " && " in sql and " || " in sql
    assert set(params.values()) == {"mpox", "monkeypox", "case count", "incidence"}
    for term in params.values():
        assert term not in sql                      # bound, never interpolated


def test_not_compiles_to_the_tsquery_negation():
    """Per-document semantics make `!!` exact here: one vector per document, so
    `mpox & !benign` means "mpox somewhere, benign nowhere" — the same rule the LIKE
    path enforced with a correlated NOT EXISTS."""
    sql = lex.ast_to_tsquery_sql(_ast("mpox NOT benign"), {})
    assert "(!! phraseto_tsquery" in sql and " && " in sql


def test_trailing_star_is_a_prefix_query():
    params: dict = {}
    sql = lex.ast_to_tsquery_sql(_ast("influenz* AND wastewater"), params)
    assert "to_tsquery('english', quote_literal(CAST(:tq_0 AS text)) || ':*')" in sql
    assert params["tq_0"] == "influenz"             # the star is syntax, not text
    assert params["tq_1"] == "wastewater"


def test_an_empty_ast_matches_nothing_never_everything():
    assert lex.ast_to_tsquery_sql(_ast(""), {}) is None
    assert lex.match_sql(_ast("[tiab]"), {}) is None
    assert main._build_boolean_match_sql_from_query("", {}) == "FALSE"


def test_tokenizer_keeps_accents_and_truncation_only_at_the_end():
    """`cathéters` used to become `cathters`, which matched nothing on either path."""
    assert main._tokenize_boolean("cathéters veineux") == [("TERM", "cathéters"), ("TERM", "veineux")]
    assert main._tokenize_boolean("forecast* AND fore*cast") == [
        ("TERM", "forecast*"), ("AND", None), ("TERM", "forecast")]
    assert main._tokenize_boolean('"emergency medical services*"') == [("TERM", "emergency medical services*")]
    assert main._tokenize_boolean("*") == []


def test_other_compilers_drop_the_star():
    params: dict = {}
    like = main._build_boolean_match_sql_from_query("forecast*", params)
    assert params["bq_0"] == "%forecast%" and "*" not in like
    assert main._boolean_to_arxiv(_ast("forecast*")) == 'all:"forecast"'
    assert main._boolean_to_s2(_ast("forecast* OR demand")) == "(forecast | demand)"
    # a phrase with a trailing star is a phrase (no prefix inside phraseto_tsquery)
    params = {}
    sql = lex.ast_to_tsquery_sql(_ast('"emergency medical services*"'), params)
    assert "phraseto_tsquery" in sql and params["tq_0"] == "emergency medical services"


def test_engine_choice_reads_the_rollback_switch(monkeypatch):
    monkeypatch.setenv("LEXICAL_SEARCH_ENGINE", "LIKE")
    assert lex.engine_choice() == "like" and lex.use_fts() is False
    monkeypatch.setenv("LEXICAL_SEARCH_ENGINE", "nonsense")
    assert lex.engine_choice() == "auto"
    monkeypatch.delenv("LEXICAL_SEARCH_ENGINE")
    assert lex.engine_choice() == "auto"


def test_schema_sql_carries_the_same_objects():
    """schema.sql is what a new deployment applies first; it must not drift from the
    module that is the source of truth for the boot DDL and the migration."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sql = open(os.path.join(root, "schema.sql"), encoding="utf-8").read()
    for name in ("document_search", "literev_document_tsv", "literev_document_search_doc_trg",
                 "literev_document_search_chunk_trg", *lex.TRIGGER_NAMES):
        assert name in sql, name
    assert f"to_tsvector('{lex.TS_CONFIG}'" in sql
    assert "chunk_type IS DISTINCT FROM 'title_abstract'" in sql
    assert "ON DELETE CASCADE" in sql[sql.index("CREATE TABLE IF NOT EXISTS document_search"):]


# ── database ─────────────────────────────────────────────────────────────────

ABS = {
    "ems": "Forecasting daily ambulance demand for emergency medical services in the region.",
    "chain": "A blockchain approach to hospital supply chains with plain html reporting only.",
    "flu": "Seasonal influenza surveillance in wastewater: SARS-CoV-2 and influenza A signals.",
    "cath": "Étude des cathéters veineux périphériques et des voies veineuses chez l'adulte.",
    "words": "Emergency services medical staffing during heat waves and pollution episodes.",
}


def _engine_ok() -> bool:
    try:
        with main.engine.connect():
            return True
    except Exception:
        return False


def _reset_state():
    lex._STATE.update(ready=False, missing=None, stale=None, checked_at=None,
                      missing_checked_at=None, last_error=None)


@pytest.fixture()
def corpus(db_conn, monkeypatch):
    if not _engine_ok():
        pytest.skip("main.engine cannot reach the database")
    with db_conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS document_search")
        cur.execute("DROP TABLE IF EXISTS document_chunk, article_scenarios, literature_document CASCADE")
        cur.execute(
            "CREATE TABLE literature_document ("
            "id bigint PRIMARY KEY, title text NOT NULL, source text NOT NULL DEFAULT 'test',"
            "abstract text, year int, is_duplicate boolean DEFAULT false,"
            "project_context text DEFAULT 'literev')")
        cur.execute(
            "CREATE TABLE document_chunk ("
            "id bigint GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,"
            "document_id bigint NOT NULL REFERENCES literature_document(id) ON DELETE CASCADE,"
            "chunk_index int NOT NULL DEFAULT 0, content text NOT NULL, chunk_type text)")
        cur.execute("CREATE TABLE article_scenarios (scenario_id text, document_id bigint, "
                    "similarity_score double precision, PRIMARY KEY (scenario_id, document_id))")
    # The real boot function, on tables that exist this time. On CI the session's
    # import of main ran it against an EMPTY database (no literature_document yet, so
    # the table, its indexes and the triggers failed and were recorded); this rerun
    # must both create the objects and clear that record, exactly as a boot would.
    main._ensure_document_search()
    assert lex.DDL_FAILURES == [], lex.DDL_FAILURES
    _reset_state()
    monkeypatch.setenv("LEXICAL_SEARCH_ENGINE", "auto")
    with db_conn.cursor() as cur:
        for i, (key, abstract) in enumerate(ABS.items(), start=1):
            cur.execute("INSERT INTO literature_document (id, title, abstract, year) VALUES (%s, %s, %s, 2024)",
                        (i, f"Doc {key}", abstract))
    yield db_conn
    _reset_state()


def _row(conn, doc_id):
    with conn.cursor() as cur:
        cur.execute("SELECT tsv::text, stale FROM document_search WHERE document_id = %s", (doc_id,))
        return cur.fetchone()


def _search(q):
    return set(main._search_local_doc_ids(q, "boolean", {}, limit=1000))


def test_document_triggers_compute_the_vector_at_once(corpus):
    tsv, stale = _row(corpus, 1)
    assert stale is False and "'ambul'" in tsv and "'forecast'" in tsv      # stemmed
    with corpus.cursor() as cur:
        cur.execute("UPDATE literature_document SET abstract = abstract || ' Helicopter dispatch.' WHERE id = 1")
    tsv, stale = _row(corpus, 1)
    assert "'helicopt'" in tsv and stale is False


def test_chunk_triggers_mark_stale_and_the_worker_recomputes(corpus):
    with corpus.cursor() as cur:
        cur.execute("INSERT INTO document_chunk (document_id, content, chunk_type) "
                    "VALUES (1, 'Doc ems title and abstract copy', 'title_abstract')")
    assert _row(corpus, 1)[1] is False                 # a title_abstract copy is not body text
    with corpus.cursor() as cur:
        cur.execute("INSERT INTO document_chunk (document_id, chunk_index, content, chunk_type) "
                    "VALUES (1, 0, 'Methods: a gradient boosting model was trained.', 'fulltext_section')")
    tsv, stale = _row(corpus, 1)
    assert stale is True and "'boost'" not in tsv      # marked, not yet recomputed
    r = lex.refresh_once()
    assert r["refreshed"] == 1 and r["stale"] == 0
    tsv, stale = _row(corpus, 1)
    assert stale is False and "'boost'" in tsv and "'ambul'" in tsv
    # an abstract edit recomputes with the chunk text still included
    with corpus.cursor() as cur:
        cur.execute("UPDATE literature_document SET abstract = abstract || ' Extra.' WHERE id = 1")
    tsv, _ = _row(corpus, 1)
    assert "'boost'" in tsv and "'extra'" in tsv
    # deleting the chunk marks stale; the recompute drops its words
    with corpus.cursor() as cur:
        cur.execute("DELETE FROM document_chunk WHERE document_id = 1 AND chunk_type = 'fulltext_section'")
    assert _row(corpus, 1)[1] is True
    lex.refresh_once()
    assert "'boost'" not in _row(corpus, 1)[0]


def test_deleting_a_document_takes_its_row_along(corpus):
    with corpus.cursor() as cur:
        cur.execute("INSERT INTO document_chunk (document_id, content, chunk_type) "
                    "VALUES (2, 'body', 'fulltext_section')")
        cur.execute("DELETE FROM literature_document WHERE id = 2")   # cascades chunks + row
    assert _row(corpus, 2) is None


def test_backfill_fills_missing_rows_and_readiness_follows(corpus):
    with corpus.cursor() as cur:
        cur.execute("DELETE FROM document_search WHERE document_id IN (2, 3)")   # pre-existing docs
    _reset_state()
    assert lex.is_ready() is False and lex._STATE["missing"] == 2
    r = lex.refresh_once(batch=1, max_batches=1)      # bounded: one document per pass
    assert r["backfilled"] == 1 and r["missing"] == 1 and r["ready"] is False
    r = lex.refresh_once()
    assert r["backfilled"] == 1 and r["missing"] == 0 and r["ready"] is True
    assert lex.is_ready() is True
    assert "'blockchain'" in _row(corpus, 2)[0]


def test_until_the_backfill_is_complete_the_search_falls_back_to_like(corpus):
    """The deploy-time guarantee: never a partial corpus. `ai` inside "chain" and
    "daily" is the LIKE fingerprint — full text cannot match it."""
    with corpus.cursor() as cur:
        cur.execute("DELETE FROM document_search WHERE document_id = 5")
    _reset_state()
    assert _search("ai") == {1, 2}                     # LIKE: %ai% inside "daily", "blockchain"
    lex.refresh_once()
    _reset_state()
    assert _search("ai") == set()                      # full text: no such word anywhere


def test_and_holds_across_chunks_per_document(corpus):
    with corpus.cursor() as cur:
        cur.execute("INSERT INTO document_chunk (document_id, chunk_index, content, chunk_type) VALUES "
                    "(5, 0, 'We used ambulance dispatch logs.', 'fulltext_section'),"
                    "(5, 1, 'A forecasting horizon of seven days.', 'fulltext_section')")
    lex.refresh_once()
    _reset_state()
    assert {1, 5} <= _search("ambulance AND forecast")          # doc 5: two different chunks
    assert 5 in _search('"ambulance dispatch"')                 # phrase inside a chunk
    assert _search("ambulance NOT forecast") == set()           # NOT is per document too


def test_what_the_searcher_will_notice(corpus):
    lex.refresh_once()
    _reset_state()
    assert _search("forecasting") == {1} == _search("forecast") == _search("forecasts")   # stemming
    assert _search("ml") == set()                                # no longer inside "html"
    assert _search('"emergency medical services"') == {1}       # a phrase, not a substring…
    assert 5 not in _search('"emergency medical services"')     # …so word order matters
    assert _search("emergency AND medical AND services") == {1, 5}
    assert _search("influenz*") == {3}                           # prefix
    assert _search("sars-cov-2") == {3}                          # hyphenated token
    assert _search("cathéters") == {4}                           # accents survive
    assert _search("the") == set()                               # stop words match nothing alone…
    assert _search("the AND ambulance") == _search("ambulance")  # …and are dropped inside AND
    assert lex.stopword_terms(["the", "ambulance", "of"]) == ["the", "of"]


def test_health_reports_the_engine(corpus):
    from fastapi.testclient import TestClient
    lex.refresh_once()
    body = TestClient(main.app).get("/health").json()
    ls = body["lexical_search"]
    assert ls["engine"] == "fts" and ls["ready"] is True and ls["missing"] == 0
    assert ls["choice"] == "auto" and ls["ddl_failures"] == []


def test_the_search_log_line_names_the_engine(corpus, caplog):
    import logging
    lex.refresh_once()
    _reset_state()
    with caplog.at_level(logging.INFO, logger="literev-api"):
        _search("influenza")
    assert re.search(r"lexical search \[fts\] 1 docs in \d+ ms", caplog.text), caplog.text
