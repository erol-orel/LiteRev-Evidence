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


# ── chercher DANS le corpus ──────────────────────────────────────────────────
# Un corpus de plusieurs milliers d'articles se parcourt par pages de cent : filtrer la
# page affichée ne chercherait que dans ces cent-là.

def test_the_terms_of_a_search_are_cumulative_and_folded():
    from api.scenario_store import corpus_search_terms
    assert corpus_search_terms("Dengue Vaccine") == ["%dengue%", "%vaccine%"]
    assert corpus_search_terms('"severe dengue" Lévy') == ["%severe dengue%", "%levy%"]
    assert corpus_search_terms("   ") == []
    # % and _ are LIKE wildcards: typed by a user they are literal characters.
    assert corpus_search_terms("100%_sure") == [r"%100\%\_sure%"]


def test_the_search_looks_at_the_fields_a_reviewer_remembers():
    from api.scenario_store import corpus_search_sql
    sql = corpus_search_sql("cq_0", doc="d")
    for field in ("d.title", "d.abstract", "d.authors", "d.journal", "d.keywords",
                  "d.doi", "d.pmid"):
        assert field in sql
    assert "lower(" in sql and "translate(" in sql


def _seed_searchable(db_conn):
    ensure_document_columns(db_conn.cursor())
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM literature_document WHERE id BETWEEN 8600 AND 8699")
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))
        cur.execute("INSERT INTO user_scenarios (id, name, query, created_at, updated_at) "
                    "VALUES (%s, 'Search', 'dengue', NOW(), NOW())", (SID,))
        rows = [
            (8610, "Dengue vaccine efficacy", "A trial of the vaccine.", "Lévy A", 0.90, None),
            (8611, "Dengue surveillance in Asia", "Wastewater monitoring.", "Roe B", 0.90, None),
            (8612, "Malaria vaccine efficacy", "Another trial.", "Doe C", 0.10, None),
            (8613, "Dengue vaccine safety", "Safety follow-up.", "Poe D", 0.10, "excluded"),
        ]
        for doc_id, title, abstract, authors, score, status in rows:
            cur.execute("INSERT INTO literature_document (id, title, abstract, authors, year, source, "
                        "project_context, quality_score) "
                        "VALUES (%s,%s,%s,%s,2024,'pubmed','literev',0.8)",
                        (doc_id, title, abstract, authors))
            cur.execute("INSERT INTO article_scenarios (document_id, scenario_id, "
                        "similarity_score, screening_status) VALUES (%s,%s,%s,%s)",
                        (doc_id, SID, score, status))


def test_searching_the_corpus_reaches_articles_beyond_the_displayed_page(db_conn):
    _seed_searchable(db_conn)
    try:
        # limit=1: a client-side filter would have had one article to look at.
        got = main.get_user_scenario_corpus(SID, limit=1, q="dengue vaccine", abstract_chars=None)
        assert got["filtered_total"] == 2            # 8610 et 8613
        assert got["total"] == 4                     # le corpus ne rétrécit pas
        assert len(got["articles"]) == 1
    finally:
        _cleanup(db_conn)


def test_a_search_can_be_restricted_to_the_relevant_articles(db_conn):
    """« au moins les pertinents » : la porte commune, donc jamais un article exclu."""
    _seed_searchable(db_conn)
    try:
        all_hits = main.get_user_scenario_corpus(SID, limit=50, q="dengue", abstract_chars=None)
        relevant = main.get_user_scenario_corpus(SID, limit=50, q="dengue", relevant_only=True,
                                                 threshold=0.45, abstract_chars=None)
        assert all_hits["filtered_total"] == 3       # 8610, 8611, 8613
        assert relevant["filtered_total"] == 2       # 8613 est exclu par un relecteur
        assert {a["id"] for a in relevant["articles"]} == {8610, 8611}
    finally:
        _cleanup(db_conn)


def test_a_search_ignores_accents_in_either_direction(db_conn):
    _seed_searchable(db_conn)
    try:
        assert main.get_user_scenario_corpus(SID, limit=5, q="levy",
                                             abstract_chars=None)["filtered_total"] == 1
        assert main.get_user_scenario_corpus(SID, limit=5, q="Lévy",
                                             abstract_chars=None)["filtered_total"] == 1
    finally:
        _cleanup(db_conn)


def test_an_empty_search_is_not_a_filter(db_conn):
    _seed_searchable(db_conn)
    try:
        got = main.get_user_scenario_corpus(SID, limit=50, q="   ", abstract_chars=None)
        assert got["filtered_total"] is None and got["total"] == 4
    finally:
        _cleanup(db_conn)


# ── la requête sauvegardée est classée pour ce qu'elle est ───────────────────
# Le sélecteur de l'interface vaut « booléen » par défaut : une question en langage
# naturel était donc rangée sous BOOLÉEN (1) / NATUREL (0), alors que la recherche,
# elle, l'avait traduite avant d'interroger les bases.

def _detail_of(db_conn, query: str, mode: str):
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))
        cur.execute("INSERT INTO user_scenarios (id, name, query, mode, created_at, updated_at) "
                    "VALUES (%s, 'Kind', %s, %s, NOW(), NOW())", (SID, query, mode))
    return main.get_user_scenario_detail(SID)


def test_a_natural_language_question_is_not_filed_as_a_boolean_query(db_conn):
    try:
        d = _detail_of(db_conn, "What are the early warning indicators for respiratory infections?", "boolean")
        assert d["boolean_queries"] == []
        assert d["nl_queries"] == ["What are the early warning indicators for respiratory infections?"]
    finally:
        _cleanup(db_conn)


def test_a_real_boolean_query_is_filed_as_one_whatever_the_stored_mode(db_conn):
    try:
        d = _detail_of(db_conn, '("respiratory infection"[tiab]) AND wastewater', "hybrid")
        assert d["boolean_queries"] == ['("respiratory infection"[tiab]) AND wastewater']
        assert d["nl_queries"] == []
    finally:
        _cleanup(db_conn)


def test_the_same_heuristic_as_the_search_itself(db_conn):
    """Classer autrement ici que dans la recherche remettrait les deux en désaccord."""
    from api.search import _looks_boolean
    for q in ("dengue AND vaccine", '"severe dengue"', "(a OR b)", "x[tiab]",
              "What works to prevent dengue?", "dengue vaccine efficacy"):
        try:
            d = _detail_of(db_conn, q, "boolean")
            assert bool(d["boolean_queries"]) is _looks_boolean(q), q
        finally:
            _cleanup(db_conn)
