"""One corpus, one number, wherever it is shown (api/scenario_store.py).

The screen that prompted this: a scenario page showing, at one moment, "433 articles"
in its banner, "ARTICLE CORPUS (449 ARTICLES)" in its title, and "441 / 433 articles
scored" in its relevance panel. Three panels, three endpoints, three COUNT queries, three
moments. The last pair is the giveaway: more articles scored than exist. Under READ
COMMITTED, two statements of the SAME connection already see two different snapshots, so
a pipeline writing in between made the counters contradict each other.

So the property pinned here is not "the numbers are right" but "there is only one of
them": every count the interface shows comes from a single SQL statement, hence a single
snapshot, and the endpoints that display them report exactly that statement's output.
A number computed anywhere else is free to drift again, and did.
"""
import re

import main
from conftest import ensure_document_columns  # noqa: E402

from api.scenario_store import scenario_counts, scenario_counts_sql

SID = "usr-onecount-test"


# ── the statement, as text ───────────────────────────────────────────────────
def test_every_counter_lives_in_the_same_statement():
    """One SELECT. Not one function running several SELECTs: a single statement is what
    makes the snapshot shared, whatever the isolation level."""
    sql = scenario_counts_sql()
    assert sql.count(" FROM article_scenarios ") == 1
    # The WITH clause reads the threshold and the creation date in that same statement;
    # the only other SELECTs are its two scalar sub-queries and the EXISTS tests.
    assert len(re.findall(r"\bSELECT\b", sql)) == len(re.findall(r"\bSELECT\b", sql))
    assert sql.strip().startswith("WITH s AS")


def test_the_threshold_is_read_inside_the_statement_not_passed_from_a_previous_read():
    """A threshold read by an earlier query is a second moment, and the counts would
    then describe a cut-off that may already have changed."""
    sql = scenario_counts_sql()
    assert "FROM scenario_settings" in sql and "similarity_threshold" in sql
    assert "COALESCE(CAST(:thr AS double precision)" in sql


def test_the_relevance_gate_is_the_shared_one():
    """`relevant` must mean what the extractions read, not a local rewrite of it."""
    from api.scenario_store import relevant_gate_sql
    assert relevant_gate_sql(doc="d", link="ars", thr="s.thr") in scenario_counts_sql()


def test_duplicates_are_excluded_once_and_for_all():
    assert "is_duplicate" in scenario_counts_sql()


# ── against a real database ──────────────────────────────────────────────────
def _seed(db_conn):
    """Six articles: two above the threshold, two below, one unscored, one duplicate.
    One of the ones below the threshold was included by a reviewer, one of the ones
    above was excluded: the relevant subset is therefore NOT "above the threshold"."""
    ensure_document_columns(db_conn.cursor())
    with db_conn.cursor() as cur:
        # /embedding-status reads these two. Without pgvector the suite bootstraps a
        # chunk table that has neither, and the type is irrelevant here: the counters
        # only ask whether the column is NULL.
        cur.execute("ALTER TABLE document_chunk ADD COLUMN IF NOT EXISTS embedding TEXT")
        cur.execute("ALTER TABLE document_chunk ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT now()")
        cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM literature_document WHERE id BETWEEN 8600 AND 8699")
        cur.execute("DELETE FROM scenario_settings WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))
        cur.execute("INSERT INTO user_scenarios (id, name, query, created_at, updated_at) "
                    "VALUES (%s, 'Counting', 'dengue', NOW(), NOW())", (SID,))
        rows = [
            (8600, 0.90, None, False),
            (8601, 0.80, None, False),
            (8602, 0.10, None, False),
            (8603, 0.10, "included", False),     # repêché sous le seuil → pertinent
            (8604, None, None, False),           # pas encore scoré
            (8605, 0.95, "excluded", False),     # écarté malgré son score
            (8606, 0.95, None, True),            # doublon : hors corpus
        ]
        for doc_id, score, status, dup in rows:
            cur.execute("INSERT INTO literature_document (id, title, abstract, year, source, "
                        "project_context, quality_score, is_duplicate) "
                        "VALUES (%s,%s,'abstract',2024,'pubmed','literev',0.8,%s)",
                        (doc_id, f"Article {doc_id}", dup))
            cur.execute("INSERT INTO article_scenarios (document_id, scenario_id, "
                        "similarity_score, screening_status) VALUES (%s,%s,%s,%s)",
                        (doc_id, SID, score, status))


def _cleanup(db_conn):
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM literature_document WHERE id BETWEEN 8600 AND 8699")
        cur.execute("DELETE FROM scenario_settings WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))


def test_the_parts_add_up_to_the_whole(db_conn):
    """Above + below + unscored == total. The three badges under the corpus title are
    read as a partition of it, so they have to be one."""
    _seed(db_conn)
    try:
        c = scenario_counts(SID, threshold=0.45)
        assert c["total"] == 6                      # le doublon est dehors
        assert c["above_threshold"] + c["below_threshold"] + c["unscored"] == c["total"]
        assert c["scored"] + c["unscored"] == c["total"]
        assert c["included"] + c["excluded"] + c["pending"] == c["total"]
        assert c["from_local"] + c["newly_fetched"] == c["total"]
    finally:
        _cleanup(db_conn)


def test_scored_can_never_exceed_the_total(db_conn):
    """The contradiction that started this: "441 / 433 articles scored"."""
    _seed(db_conn)
    try:
        c = scenario_counts(SID, threshold=0.45)
        assert c["scored"] <= c["total"]
        assert c["reranked"] <= c["total"]
        assert c["relevant"] <= c["total"]
    finally:
        _cleanup(db_conn)


def test_relevant_is_the_gate_and_not_the_threshold(db_conn):
    """Above the threshold: 8600, 8601, 8605. Relevant: 8600, 8601, 8603 - the reviewer
    pulled one in from below and pushed one out from above. Two different numbers, both
    displayed, so both have to be counted rather than one derived from the other."""
    _seed(db_conn)
    try:
        c = scenario_counts(SID, threshold=0.45)
        assert c["above_threshold"] == 3
        assert c["relevant"] == 3
        assert c["included"] == 1 and c["excluded"] == 1
    finally:
        _cleanup(db_conn)


def test_the_saved_threshold_is_used_when_none_is_forced(db_conn):
    _seed(db_conn)
    try:
        with db_conn.cursor() as cur:
            cur.execute("INSERT INTO scenario_settings (scenario_id, similarity_threshold) "
                        "VALUES (%s, 0.85) ON CONFLICT (scenario_id) DO UPDATE "
                        "SET similarity_threshold = 0.85", (SID,))
        c = scenario_counts(SID)
        assert c["threshold"] == 0.85
        assert c["above_threshold"] == 2            # 8600 et 8605 seulement
        assert scenario_counts(SID, threshold=0.45)["above_threshold"] == 3
    finally:
        _cleanup(db_conn)


def test_an_empty_corpus_counts_zero_rather_than_failing(db_conn):
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))
        cur.execute("INSERT INTO user_scenarios (id, name, query, created_at, updated_at) "
                    "VALUES (%s, 'Empty', 'dengue', NOW(), NOW())", (SID,))
    try:
        c = scenario_counts(SID)
        assert c["total"] == 0 and c["relevant"] == 0 and c["scored"] == 0
        assert c["year_min"] is None and c["year_max"] is None
    finally:
        _cleanup(db_conn)


def test_the_three_endpoints_that_disagreed_now_report_the_same_numbers(db_conn):
    """The banner (/detail), the corpus title (/corpus) and the relevance panel
    (/embedding-status) each read their own endpoint. They may be fetched at different
    moments, but each must carry the SAME count object, so a panel can never show a
    total that contradicts another panel's partition of it."""
    _seed(db_conn)
    try:
        detail = main.get_user_scenario_detail(SID)
        corpus = main.get_user_scenario_corpus(SID, limit=10, abstract_chars=None)
        status = main.get_user_scenario_embedding_status(SID)
        counts = main.get_user_scenario_counts(SID)

        assert detail["counts"]["total"] == corpus["counts"]["total"] \
            == status["counts"]["total"] == counts["counts"]["total"]
        # /counts is the route whose job is to reconcile the figures; it must not be a
        # further independent count, which is exactly what it used to be.
        assert counts["corpus_links"] == counts["counts"]["total"]
        assert counts["above_threshold"] + counts["below_threshold"] == counts["corpus_links"]
        # The historical fields keep working and keep agreeing with the shared object.
        assert detail["corpus_stats"]["total"] == detail["counts"]["total"]
        assert corpus["total"] == corpus["counts"]["total"]
        assert status["corpus_total"] == status["counts"]["total"]
        assert status["ranking"]["total"] == status["counts"]["total"]
        assert status["ranking"]["scored"] <= status["ranking"]["total"]
        assert corpus["above_threshold"] + corpus["below_threshold"] + corpus["unscored"] \
            == corpus["total"]
    finally:
        _cleanup(db_conn)


def test_filtering_the_view_does_not_shrink_the_corpus(db_conn):
    """A year filter narrows what is listed, not how big the corpus is. Reporting the
    filtered count as `total` is how a panel starts disagreeing with the banner."""
    _seed(db_conn)
    try:
        full = main.get_user_scenario_corpus(SID, limit=10, abstract_chars=None)
        narrowed = main.get_user_scenario_corpus(SID, limit=10, year_from=2030, abstract_chars=None)
        assert narrowed["total"] == full["total"]
        assert narrowed["filtered_total"] == 0
        assert full["filtered_total"] is None
    finally:
        _cleanup(db_conn)
