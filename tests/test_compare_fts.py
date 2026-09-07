"""The FTS comparison tool must compile queries FAITHFULLY, or its verdict is noise.

This script exists to answer "does full-text search return more papers or fewer" on real
data. That answer is only worth having if the tsquery it builds actually means the same
thing as the boolean the user typed — so these tests check the compilation, and where a
database is available, that PostgreSQL agrees with the intended semantics.
"""
import os
import sys
import uuid

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("fastapi")

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import compare_fts as cf  # noqa: E402
import main  # noqa: E402


def _ast(q):
    return main._parse_boolean_ast(main._tokenize_boolean(q))


# ── compilation ──────────────────────────────────────────────────────────────
def test_the_whole_boolean_collapses_into_one_tsquery():
    """The entire performance argument: term count stops mattering, because AND/OR
    become tsquery operators inside a single index condition."""
    params: dict = {}
    sql = cf.ast_to_tsquery_sql(_ast("(mpox OR monkeypox) AND (forecast OR incidence)"), params)
    assert sql.count("phraseto_tsquery") == 4
    assert " && " in sql and " || " in sql
    assert set(params.values()) == {"mpox", "monkeypox", "forecast", "incidence"}


def test_terms_are_bound_never_interpolated():
    params: dict = {}
    sql = cf.ast_to_tsquery_sql(_ast('"case count" OR mpox'), params)
    assert "case count" not in sql and "mpox" not in sql
    assert "case count" in params.values()


def test_negation_is_refused_inside_the_tsquery():
    """`!!term` at document level would accept a document whose title lacks the term
    while one of its chunks contains it — the per-document semantics the current
    compiler is careful about. The caller must subtract sets instead."""
    with pytest.raises(cf.UnsupportedQuery):
        cf.ast_to_tsquery_sql(_ast("mpox NOT benign"), {})


def test_negation_becomes_a_set_subtraction():
    params: dict = {}
    sql = cf.fts_doc_ids_sql(_ast("mpox NOT benign"), params)
    assert " EXCEPT " in sql
    # Two TERMS — one positive, one negated. Each appears four times in the text because
    # every branch embeds its tsquery once for the documents arm and once for the chunks
    # arm of the UNION; the parameter count is what says how many terms there really are.
    assert len(params) == 2, params
    assert sql.count("phraseto_tsquery") == 4


def test_an_only_negative_query_starts_from_the_whole_corpus():
    sql = cf.fts_doc_ids_sql(_ast("NOT benign"), {})
    assert " EXCEPT " in sql and "FROM literature_document ld)" in sql


def test_both_branches_are_searched_so_membership_stays_per_document():
    """A term in any chunk must count, exactly as the correlated EXISTS does today."""
    sql = cf.fts_doc_ids_sql(_ast("mpox"), {})
    assert "FROM literature_document ld" in sql and "FROM document_chunk dc" in sql
    assert " UNION " in sql


def test_query_expressions_match_the_indexed_expressions():
    """An alias is fine; a changed COALESCE or config is not — the planner would stop
    using the index with no error, which is how the trigram indexes can silently fail."""
    assert cf.doc_tsv() in cf.BUILD_DDL[0]
    assert cf.chunk_tsv() in cf.BUILD_DDL[1]
    sql = cf.fts_doc_ids_sql(_ast("mpox"), {})
    assert cf.doc_tsv("ld") in sql and cf.chunk_tsv("dc") in sql
    # same shape, only the alias differs
    assert cf.doc_tsv("ld").replace("ld.", "") == cf.doc_tsv()


# ── PostgreSQL has to agree ──────────────────────────────────────────────────
@pytest.fixture()
def pg():
    import sqlalchemy as sa
    from sqlalchemy.engine import make_url
    raw = os.getenv("DB_URL") or ""
    if not raw:
        pytest.skip("DB_URL not set")
    try:
        eng = sa.create_engine(make_url(raw))
        with eng.connect() as c:
            c.execute(sa.text("SELECT 1"))
    except Exception as e:
        pytest.skip(f"no reachable database: {e}")
    return eng


def _matches(pg, text_body, query):
    """Does `text_body` match `query` under the compiled tsquery?"""
    import sqlalchemy as sa
    params: dict = {}
    q = cf.ast_to_tsquery_sql(_ast(query), params)
    with pg.connect() as c:
        return c.execute(sa.text(
            f"SELECT to_tsvector('{cf.TS_CONFIG}', :body) @@ {q}"),
            {**params, "body": text_body}).scalar()


def test_stemming_is_what_gains_papers(pg):
    """The predicted gain, verified rather than assumed."""
    assert _matches(pg, "We forecast the outbreak", "forecasting")
    assert _matches(pg, "Weekly incidences were recorded", "incidence")


def test_substring_false_positives_disappear(pg):
    """The predicted loss — and it is a precision WIN. Today
    `LIKE '%incidence%'` matches 'coincidence'; full text does not."""
    assert "incidence" in "coincidence"                    # the current behaviour
    assert not _matches(pg, "A remarkable coincidence occurred", "incidence")


def test_a_quoted_phrase_stays_a_phrase(pg):
    assert _matches(pg, "the case counts rose", '"case count"')
    assert not _matches(pg, "in this case we counted separately", '"case count"')


def test_and_or_semantics_survive_the_translation(pg):
    q = "(mpox OR monkeypox) AND forecast"
    assert _matches(pg, "mpox forecast for Europe", q)
    assert _matches(pg, "monkeypox forecasting study", q)
    assert not _matches(pg, "mpox surveillance only", q)   # second group unmet
    assert not _matches(pg, "influenza forecast", q)       # first group unmet


def test_a_stopword_only_term_is_reported_not_silently_dropped(pg):
    """`phraseto_tsquery('english','the')` is EMPTY and matches nothing, so a stopword
    term would silently empty an AND. Pinned so the script's output can be trusted."""
    import sqlalchemy as sa
    with pg.connect() as c:
        empty = c.execute(sa.text(
            f"SELECT phraseto_tsquery('{cf.TS_CONFIG}', 'the')::text")).scalar()
    assert empty == "", f"expected an empty tsquery, got {empty!r}"
    assert not _matches(pg, "the study of the thing", "the")
