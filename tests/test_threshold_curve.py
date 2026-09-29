"""The threshold curve: "what threshold keeps 100 articles", answered on the real scores.

Two halves, deliberately.

The PURE half pins the arithmetic that is easy to get subtly wrong and impossible to see
from the interface: ties (a round target is rarely reachable), reviewer-included articles
(they pass whatever the threshold is, so they count toward the target without moving it),
and the rounding of the threshold itself.

The INTEGRATION half exercises the endpoint against a real database, and above all checks
the PROMISE: setting the threshold to a point of the curve really does leave exactly the
announced number of relevant articles, counted by the app's own relevance gate rather
than by the curve's own arithmetic. A curve that agrees only with itself is worthless.
"""
import pytest

pytest.importorskip("fastapi")

import main  # noqa: E402
from api.relevance import _floor4, threshold_curve  # noqa: E402
from conftest import ensure_document_columns  # noqa: E402

SID = "usr-threshold-curve-test"
IDS = (9401, 9402, 9403, 9404, 9405, 9406, 9407, 9408, 9409)


# ── Pure: the arithmetic ─────────────────────────────────────────────────────
def test_a_tie_reports_what_it_really_keeps_not_the_number_asked():
    """Asking for 2 on a corpus whose articles 2 to 6 share one score keeps 6, because
    the gate is `>= threshold` and cannot split a tie. The point must say 6 and
    `exact: False`: reporting the request back as if it were the answer is how a corpus
    silently becomes three times the size the methods section claims."""
    rows = [(0.9, True)] + [(0.7, False)] * 5
    (point,) = threshold_curve(rows, 0, targets=(2,))
    assert point["requested"] == 2
    assert point["kept"] == 6 and point["kept_scored"] == 6
    assert point["exact"] is False


def test_a_reachable_target_is_marked_exact():
    rows = [(0.9, True), (0.8, False), (0.7, True)]
    (point,) = threshold_curve(rows, 0, targets=(2,))
    assert (point["kept"], point["exact"], point["threshold"]) == (2, True, 0.8)


def test_reviewer_included_articles_count_toward_the_target_without_moving_it():
    """Included-by-hand articles are in the relevant subset at any threshold. Aiming at 5
    with 3 of them must therefore look for 2 more among the scored ones, not 5: ignoring
    them overshoots the target by exactly their number, every time."""
    rows = [(0.9, True), (0.8, False), (0.7, True), (0.6, False), (0.5, True)]
    (point,) = threshold_curve(rows, 3, targets=(5,))
    assert point["kept_scored"] == 2 and point["kept"] == 5 and point["exact"] is True
    assert point["threshold"] == 0.8


def test_a_target_already_covered_by_the_included_articles_yields_no_point():
    """10 articles included by hand, target 5: no threshold can keep fewer than 10, so
    there is nothing to propose. An empty curve is the honest answer."""
    assert threshold_curve([(0.9, True), (0.5, False)], 10, targets=(5,)) == []


def test_a_target_larger_than_the_corpus_yields_no_point():
    assert threshold_curve([(0.9, True), (0.5, False)], 0, targets=(50,)) == []


def test_an_empty_corpus_has_no_curve():
    assert threshold_curve([], 0, targets=(25, 100)) == []


def test_the_threshold_is_floored_so_the_promised_articles_are_really_kept():
    """The returned threshold is truncated DOWNWARD to 4 decimals. Rounding to the
    nearest would, one time in two, land just above the boundary article's score and cut
    one more article than the count displayed: the number shown would be wrong by the
    very article that defines it."""
    assert _floor4(0.71236) == 0.7123 and round(0.71236, 4) == 0.7124   # nearest would overshoot
    rows = [(0.9, True), (0.71236, False), (0.5, True)]
    (point,) = threshold_curve(rows, 0, targets=(2,))
    assert point["threshold"] == 0.7123
    assert point["kept"] == 2                      # ... and the boundary article survives it
    assert 0.71236 >= point["threshold"]


def test_the_parameter_column_says_what_each_threshold_would_cut():
    """The column that makes a threshold defensible in a methods section: how many
    articles reporting an epidemiological parameter it keeps, and how many it throws
    away. Kept + cut is the corpus total, at every point."""
    rows = [(0.9, True), (0.8, False), (0.7, True), (0.6, False), (0.5, True)]
    curve = threshold_curve(rows, 0, targets=(1, 3, 5))
    assert [(p["with_parameter_kept"], p["with_parameter_cut"]) for p in curve] == [
        (1, 2), (2, 1), (3, 0)]
    assert all(p["with_parameter_kept"] + p["with_parameter_cut"] == 3 for p in curve)


def test_the_current_threshold_is_on_the_curve_as_a_point_nobody_asked_for():
    """Where the slider stands today must appear next to the proposals, or there is
    nothing to compare them with. It carries `requested: None` and `exact: None`: it is
    not a missed target, it is not a target at all."""
    rows = [(0.9, True), (0.8, False), (0.7, True)]
    curve = threshold_curve(rows, 0, targets=(1,), current=0.75)
    assert [p["threshold"] for p in curve] == [0.9, 0.75]       # sorted, strictest first
    here = curve[1]
    assert here["requested"] is None and here["exact"] is None and here["kept"] == 2


def test_the_current_threshold_is_not_repeated_when_a_target_already_lands_on_it():
    """... but the point must still say it is where the slider stands. It was added only
    once, labelled by the target, so on a corpus where a ladder target happened to land on
    the current threshold the interface had nothing left to mark "you are here" with."""
    rows = [(0.9, True), (0.8, False), (0.7, True)]
    curve = threshold_curve(rows, 0, targets=(2,), current=0.8)
    assert [p["threshold"] for p in curve] == [0.8]
    assert curve[0]["requested"] == 2
    assert curve[0]["is_current"] is True
    # ... and exactly one point carries it.
    assert sum(1 for p in threshold_curve(rows, 0, targets=(1, 2, 3), current=0.8)
               if p["is_current"]) == 1


def test_no_point_is_current_when_no_threshold_was_given():
    rows = [(0.9, True), (0.8, False)]
    assert all(p["is_current"] is False for p in threshold_curve(rows, 0, targets=(1, 2)))


def test_the_target_asked_for_wins_the_label_over_one_from_the_standard_ladder():
    """Two targets can land on the same threshold when a tie swallows them both, and a
    point carries one label. Whichever comes FIRST in `targets` keeps it, which is why
    the endpoint puts the user's own target at the head: labelled with the ladder's 25
    instead, the answer to "I asked for 30" would read as "30 is out of reach"."""
    rows = [(0.9, True)] + [(0.4, False)] * 40          # 25 and 30 both fall in the tie
    by_ladder = threshold_curve(rows, 0, targets=(25, 30))
    assert [p["requested"] for p in by_ladder] == [25]          # 30 collapses into 25
    by_request = threshold_curve(rows, 0, targets=(30, 25))     # as the endpoint orders it
    assert [p["requested"] for p in by_request] == [30]
    assert by_request[0]["kept"] == 41 and by_request[0]["exact"] is False


def test_the_curve_is_monotone_and_never_repeats_a_threshold():
    """Lowering the threshold can only keep more. Two targets that fall inside the same
    tie produce the same threshold and must collapse into one point."""
    rows = [(0.9, True), (0.7, False), (0.7, True), (0.7, False), (0.3, True)]
    curve = threshold_curve(rows, 0, targets=(2, 3, 4, 5))
    thresholds = [p["threshold"] for p in curve]
    assert thresholds == sorted(set(thresholds), reverse=True)
    assert [p["kept"] for p in curve] == sorted(p["kept"] for p in curve)


# ── Integration: the endpoint, and the promise it makes ──────────────────────
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
        # Self-sufficient: earlier tests in the suite leave `literature_document` in
        # whatever shape they needed (one of them recreates it with four columns), so
        # this fixture asks for the columns it reads instead of inheriting them.
        cur.execute("CREATE TABLE IF NOT EXISTS literature_document (id BIGINT PRIMARY KEY)")
        cur.execute("CREATE TABLE IF NOT EXISTS article_scenarios ("
                    "scenario_id TEXT, document_id BIGINT, PRIMARY KEY (scenario_id, document_id))")
        created_chunk_table = ensure_document_columns(cur)
        for col, typ in (("title", "TEXT"), ("abstract", "TEXT"),
                         ("is_duplicate", "BOOLEAN DEFAULT FALSE")):
            cur.execute(f"ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS {col} {typ}")
        cur.execute("ALTER TABLE article_scenarios ADD COLUMN IF NOT EXISTS "
                    "similarity_score DOUBLE PRECISION")
        # Setup clears EVERYTHING it will write, teardown included: when a fixture errors
        # half way its teardown never runs, and the next run must not inherit the debris.
        cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM literature_document WHERE id = ANY(%s)", (list(IDS),))
        cur.execute("DELETE FROM scenario_settings WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))
        cur.execute("INSERT INTO user_scenarios (id, name, query, mode, filters, pinned) "
                    "VALUES (%s, 'Threshold curve', 'dengue', 'boolean', '{}', TRUE)", (SID,))
        # Three of the candidates report a parameter (9401, 9403, 9405); 9404 ties with
        # 9403; 9406 is excluded, 9407 included by hand, 9408 a duplicate, 9409 unscored.
        cur.execute(
            "INSERT INTO literature_document (id, title, source, abstract, is_duplicate, project_context) VALUES "
            "(9401, 'Transmission', 'pubmed', 'the basic reproduction number was 2.4', false, 'literev'),"
            "(9402, 'Surveillance', 'pubmed', 'a descriptive surveillance report', false, 'literev'),"
            "(9403, 'Severity',    'pubmed', 'the case fatality rate reached 1.2%%', false, 'literev'),"
            "(9404, 'Vectors',     'pubmed', 'breeding sites in urban settings', false, 'literev'),"
            "(9405, 'Natural history', 'pubmed', 'the incubation period was 5 days', false, 'literev'),"
            "(9406, 'Excluded',    'pubmed', 'off topic entirely', false, 'literev'),"
            "(9407, 'Hand picked', 'pubmed', 'a report the reviewer insisted on', false, 'literev'),"
            "(9408, 'Duplicate',   'pubmed', 'the basic reproduction number again', true, 'literev'),"
            "(9409, 'Never scored','pubmed', 'ingested after the ranking ran', false, 'literev')")
        cur.execute(
            "INSERT INTO article_scenarios (scenario_id, document_id, similarity_score, screening_status) VALUES "
            "(%s, 9401, 0.91, NULL), (%s, 9402, 0.82, NULL), (%s, 9403, 0.73, NULL),"
            "(%s, 9404, 0.73, NULL), (%s, 9405, 0.44, NULL), (%s, 9406, 0.95, 'excluded'),"
            "(%s, 9407, 0.05, 'included'), (%s, 9408, 0.99, NULL), (%s, 9409, NULL, NULL)",
            (SID,) * 9)
        cur.execute("INSERT INTO scenario_settings (scenario_id, similarity_threshold) VALUES (%s, 0.45) "
                    "ON CONFLICT (scenario_id) DO UPDATE SET similarity_threshold = 0.45", (SID,))
    yield db_conn
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM literature_document WHERE id = ANY(%s)", (list(IDS),))
        cur.execute("DELETE FROM scenario_settings WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))
        if created_chunk_table:
            cur.execute("DROP TABLE document_chunk")


def _client():
    from fastapi.testclient import TestClient
    return TestClient(main.app)


def test_the_endpoint_counts_the_corpus_the_way_the_rest_of_the_app_does(seeded):
    """Excluded and duplicate articles are out whatever the threshold; the hand-included
    one is in whatever the threshold; the unscored one counts as 0, exactly as the gate
    `COALESCE(similarity_score, 0) >= threshold` does everywhere else. The curve must
    describe the corpus the app will actually build, not an idealised one."""
    body = _client().get(f"/scenarios/{SID}/threshold-curve").json()
    assert body["candidates"] == 6            # 9401-9405 and the unscored 9409
    assert body["included"] == 1              # 9407, whatever the threshold
    assert body["unscored"] == 1              # 9409, counted as score 0
    assert body["corpus"] == 7
    assert body["with_parameter_total"] == 3  # 9401, 9403, 9405; the duplicate is not one
    assert body["current_threshold"] == 0.45
    assert body["scoring_in_progress"] is False


def test_a_requested_target_comes_back_as_a_usable_threshold(seeded):
    body = _client().get(f"/scenarios/{SID}/threshold-curve?target=3").json()
    sug = body["suggestion"]
    assert sug is not None and sug["requested"] == 3
    assert sug["kept"] == 3 and sug["kept_scored"] == 2 and sug["exact"] is True
    assert 0.73 < sug["threshold"] <= 0.82
    assert sug["with_parameter_kept"] == 1 and sug["with_parameter_cut"] == 2
    assert sug in body["curve"]


def test_a_target_the_corpus_cannot_reach_says_so_instead_of_answering_next_to_it(seeded):
    """Asking for 500 on a corpus of 7 must not quietly hand back the 7-article point as
    if it were the answer. `reachable` then says what the threshold CAN do, so the
    interface has something to offer instead of a bare refusal."""
    body = _client().get(f"/scenarios/{SID}/threshold-curve?target=500").json()
    assert body["suggestion"] is None
    assert body["reachable"] == {"min": 2, "max": 7}     # 1 hand-included + 1 .. + all 6


def test_a_target_below_the_hand_included_floor_is_unreachable_too(seeded):
    """One article is included by hand, so no threshold can leave a single article. The
    answer is "out of reach", not the 2-article point dressed up as a 1-article one."""
    body = _client().get(f"/scenarios/{SID}/threshold-curve?target=1").json()
    assert body["suggestion"] is None
    assert body["reachable"]["min"] == 2


def test_the_target_asked_for_is_answered_even_when_the_ladder_covers_it(seeded):
    """The standard ladder starts at 25 and this corpus holds 7, so nothing on the ladder
    applies; a target of 2 must still come back answered rather than swallowed."""
    body = _client().get(f"/scenarios/{SID}/threshold-curve?target=2").json()
    assert body["suggestion"] is not None and body["suggestion"]["kept"] == 2
    assert main.corpus_digest(SID, body["suggestion"]["threshold"])["n_articles"] == 2


def test_setting_the_threshold_to_a_point_really_leaves_that_many_articles(seeded):
    """THE promise, checked against the app's own relevance gate rather than against the
    curve's own arithmetic: for every point offered, the full-corpus digest - the same
    SQL the brief, the variables and the exports count with - must find exactly `kept`
    relevant articles at that threshold."""
    points = []
    for target in (None, 2, 3, 4, 5, 6, 7):
        qs = "" if target is None else f"?target={target}"
        points += _client().get(f"/scenarios/{SID}/threshold-curve{qs}").json()["curve"]
    assert points, "the fixture must produce at least one point"
    for point in points:
        digest = main.corpus_digest(SID, point["threshold"])
        assert digest["complete"] is True
        assert digest["n_articles"] == point["kept"], (
            f"threshold {point['threshold']} promises {point['kept']} articles, "
            f"the corpus gate finds {digest['n_articles']}")


def test_a_tie_in_the_database_is_reported_as_a_tie(seeded):
    """9403 and 9404 share 0.73: asking for 4 articles (1 included + 3 scored) cannot be
    met, and the endpoint must say 5 rather than repeat 4 back."""
    body = _client().get(f"/scenarios/{SID}/threshold-curve?target=4").json()
    sug = body["suggestion"]
    assert sug["requested"] == 4 and sug["exact"] is False and sug["kept"] == 5
    assert main.corpus_digest(SID, sug["threshold"])["n_articles"] == 5


def test_an_unknown_scenario_is_a_404_not_an_empty_curve(seeded):
    assert _client().get("/scenarios/usr-does-not-exist/threshold-curve").status_code == 404


def test_a_nonsense_target_is_refused(seeded):
    assert _client().get(f"/scenarios/{SID}/threshold-curve?target=0").status_code == 422
