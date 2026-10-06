"""Which pairs of concepts no relevant article studies together (api/digest.py).

The figure this replaces: a commercial report prints a theme-by-dimension matrix with a
paper count per cell and "Potential gap" where a cell is empty. It is the most useful page
in that document and the least trustworthy, because the report synthesises 50 of its 216
eligible papers. A cell reading "No papers" means none of the fifty, so an empty cell is
as likely to be a sampling artefact as a gap in the literature.

Counted over the WHOLE relevant subset, the same cell says something a sample cannot: no
article in this corpus pairs these two things. So these tests pin two properties above all
others: the count covers every relevant article, and the matrix reports the articles it
could NOT read, because a gap figure that hides its own denominator is the thing being
replaced.
"""
import json

import main
from conftest import ensure_document_columns  # noqa: E402

from api.digest import build_matrix

SID = "usr-gapmatrix-test"


# ── the grid, as a pure function ─────────────────────────────────────────────
def _pairs(*triples):
    return [{"row_label": r, "col_label": c, "n": n} for r, c, n in triples]


def test_the_grid_is_every_combination_of_the_two_axes():
    got = build_matrix(_pairs(("vaccination", "severe dengue", 5),
                              ("larviciding", "case fatality", 2)))
    assert [r["label"] for r in got["rows"]] == ["vaccination", "larviciding"]
    assert [c["label"] for c in got["cols"]] == ["severe dengue", "case fatality"]
    # Two observed pairs out of four possible: the other two are the point of the figure.
    assert got["pairs_observed"] == 2 and got["cells_shown"] == 4
    assert sorted((g["row"], g["col"]) for g in got["gaps"]) == [
        ("larviciding", "severe dengue"), ("vaccination", "case fatality")]


def test_axes_are_ordered_by_how_much_of_the_corpus_carries_the_label():
    got = build_matrix(_pairs(("rare", "o", 1), ("common", "o", 50), ("mid", "o", 7)))
    assert [r["label"] for r in got["rows"]] == ["common", "mid", "rare"]
    assert [r["n"] for r in got["rows"]] == [50, 7, 1]


def test_a_tie_is_broken_by_label_so_the_grid_does_not_move_between_calls():
    """A matrix that reorders itself on identical data cannot be cited in a paper."""
    first = build_matrix(_pairs(("b", "o", 3), ("a", "o", 3), ("c", "o", 3)))
    second = build_matrix(_pairs(("c", "o", 3), ("b", "o", 3), ("a", "o", 3)))
    assert [r["label"] for r in first["rows"]] == ["a", "b", "c"]
    assert first["rows"] == second["rows"]


def test_an_axis_is_truncated_for_reading_but_says_how_much_it_hid():
    """A 40x40 grid is not read. An axis silently cut to its ten biggest labels would
    make the matrix look more complete than it is, so the untruncated count stays."""
    got = build_matrix(_pairs(*[(f"row{i:02d}", "o", 100 - i) for i in range(25)]),
                       max_labels=10)
    assert len(got["rows"]) == 10 and got["rows_total"] == 25
    assert got["rows"][0]["label"] == "row00"


def test_a_gap_is_only_claimed_inside_the_shown_grid():
    """Outside it, a zero may be a label that was cut rather than a pairing nobody
    studied, and reporting those as gaps would invent findings."""
    got = build_matrix(_pairs(("r1", "c1", 4), ("r2", "c2", 3), ("r3", "c3", 2)),
                       max_labels=2)
    shown = {(c["row"], c["col"]) for c in got["cells"]}
    assert all((g["row"], g["col"]) in shown for g in got["gaps"])
    assert "r3" not in [r["label"] for r in got["rows"]]
    assert all(g["row"] != "r3" and g["col"] != "c3" for g in got["gaps"])


def test_the_same_pair_arriving_twice_is_added_up():
    got = build_matrix(_pairs(("r", "c", 2), ("r", "c", 3)))
    assert got["cells"] == [{"row": "r", "col": "c", "n": 5}]
    assert got["gaps"] == []


def test_a_label_that_is_null_or_blank_is_not_an_axis():
    got = build_matrix(_pairs(("r", "c", 1), (None, "c", 9), ("r", "", 9)))
    assert [r["label"] for r in got["rows"]] == ["r"]
    assert [c["label"] for c in got["cols"]] == ["c"]


def test_an_empty_corpus_produces_an_empty_grid_not_a_grid_of_gaps():
    """Nothing extracted must not read as "every pairing is a gap"."""
    got = build_matrix([])
    assert got["rows"] == [] and got["cols"] == [] and got["cells"] == []
    assert got["gaps"] == [] and got["pairs_observed"] == 0


# ── against a real database ──────────────────────────────────────────────────
def _seed(db_conn):
    """Four articles: three relevant with concepts, one excluded, one relevant without
    concepts (which the matrix cannot read and must therefore declare)."""
    ensure_document_columns(db_conn.cursor())
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM literature_document WHERE id BETWEEN 8300 AND 8399")
        cur.execute("INSERT INTO user_scenarios (id, name, query, created_at, updated_at) "
                    "VALUES (%s, 'Gaps', 'dengue', NOW(), NOW()) ON CONFLICT (id) DO NOTHING",
                    (SID,))

        def _concepts(intervention, outcome):
            return json.dumps({"v": 1, "concepts": [
                {"t": "intervention", "en": intervention},
                {"t": "outcome", "en": outcome}]})

        rows = [
            (8300, _concepts("vaccination", "severe dengue"), 0.80, None),
            (8301, _concepts("vaccination", "severe dengue"), 0.80, None),
            (8302, _concepts("larviciding", "case fatality"), 0.80, None),
            # Excluded by a reviewer: must not be counted, though its concepts would
            # otherwise fill the empty corner of the grid.
            (8303, _concepts("vaccination", "case fatality"), 0.95, "excluded"),
            # Relevant, but nothing extracted: invisible to the matrix, and the reason
            # `coverage` exists.
            (8304, None, 0.80, None),
        ]
        for doc_id, concepts, score, status in rows:
            cur.execute("INSERT INTO literature_document (id, title, abstract, year, source, "
                        "project_context, quality_score) VALUES (%s,%s,'a',2024,'pubmed','literev',0.8)",
                        (doc_id, f"Article {doc_id}"))
            if concepts:
                cur.execute("UPDATE literature_document SET concepts_json = %s WHERE id = %s",
                            (concepts, doc_id))
            cur.execute("INSERT INTO article_scenarios (document_id, scenario_id, "
                        "similarity_score, screening_status) VALUES (%s,%s,%s,%s)",
                        (doc_id, SID, score, status))


def _cleanup(db_conn):
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM literature_document WHERE id BETWEEN 8300 AND 8399")
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))


def test_the_matrix_counts_the_whole_relevant_subset(db_conn):
    _seed(db_conn)
    try:
        m = main.concept_matrix(SID, "intervention", "outcome", threshold=0.45)
        assert m["complete"] is True
        cells = {(c["row"], c["col"]): c["n"] for c in m["cells"]}
        assert cells[("vaccination", "severe dengue")] == 2
        assert cells[("larviciding", "case fatality")] == 1
        # The reviewer excluded the only article pairing vaccination with case fatality,
        # so that cell is a gap. Screening decisions have to reach this figure.
        assert cells[("vaccination", "case fatality")] == 0
        assert {("vaccination", "case fatality"), ("larviciding", "severe dengue")} == {
            (g["row"], g["col"]) for g in m["gaps"]}
    finally:
        _cleanup(db_conn)


def test_the_matrix_says_how_many_relevant_articles_it_could_not_read(db_conn):
    """THE honesty requirement. Four relevant articles, three with concepts: the fourth is
    invisible here, and a gap figure that hides its denominator is the thing this
    replaces."""
    _seed(db_conn)
    try:
        m = main.concept_matrix(SID, "intervention", "outcome", threshold=0.45)
        assert m["coverage"] == {"relevant": 4, "with_concepts": 3}
    finally:
        _cleanup(db_conn)


def test_the_available_types_are_offered_so_the_axes_need_not_be_guessed(db_conn):
    _seed(db_conn)
    try:
        m = main.concept_matrix(SID, "intervention", "outcome", threshold=0.45)
        assert [(t["value"], t["n"]) for t in m["available_types"]] == [
            ("intervention", 3), ("outcome", 3)]
    finally:
        _cleanup(db_conn)


def test_an_axis_type_that_does_not_exist_is_an_empty_grid_not_an_error(db_conn):
    _seed(db_conn)
    try:
        m = main.concept_matrix(SID, "intervention", "no-such-type", threshold=0.45)
        assert m["complete"] is True and m["cells"] == [] and m["gaps"] == []
    finally:
        _cleanup(db_conn)


def test_the_endpoint_picks_its_own_axes(db_conn):
    """Useful without knowing what the map step extracted for this corpus."""
    from fastapi.testclient import TestClient

    _seed(db_conn)
    try:
        got = TestClient(main.app).get(f"/user-scenarios/{SID}/evidence-gaps").json()
        assert got["row_type"] == "intervention" and got["col_type"] == "outcome"
        assert got["coverage"]["relevant"] == 4
    finally:
        _cleanup(db_conn)
