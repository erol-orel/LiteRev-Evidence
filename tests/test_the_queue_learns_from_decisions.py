"""The double-blind queue learns from the reviewers' decisions (api/learning.py).

On the ISS team's HPAI review, LiteRev's relevance score needed 66% of their 1,518
records screened to meet 90% of the 73 papers they included: the score measures
closeness to the question and knows nothing of their criteria. A model trained on their
own decisions needed 35%. So every decision now retrains a model on the scenario's
decided articles, every article gets a `learned_score`, the queue follows it, and the
articles the threshold left out are listed when the model would include them.

Pure tests for what decides (the label of a row, the representation, the training), and
database tests for the rest: the scores written, the queue's order, the panel's figures,
and the decision endpoints that schedule the training.
"""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("psycopg")

import main  # noqa: E402
from api import learning  # noqa: E402

SID = "queue-learns"
R1, R2 = "R-4444", "R-5555"
KEY = {"X-API-Key": "test-write-key"}

EXPOSURE = ("Occupational exposure of poultry workers to avian influenza H5N1: a seroprevalence survey "
            "of farm workers, cullers and veterinarians with protective equipment use and risk factors.")
VACCINE = ("A recombinant vaccine candidate protects mice against lethal influenza challenge: "
           "immunogenicity, antibody titres and survival in a murine model of infection.")


def _docs():
    """Ten articles: six decided (three exposure papers included, three vaccine papers
    excluded), two undecided above the threshold (one of each kind), two below."""
    out = []
    for i in range(3):
        out.append((800 + i, f"Poultry worker exposure study {i}", EXPOSURE, 0.9, "included"))
        out.append((803 + i, f"Vaccine mouse study {i}", VACCINE, 0.9, "excluded"))
    out.append((806, "Seroprevalence among cullers of H5N1 poultry", EXPOSURE, 0.85, None))
    out.append((807, "Murine vaccine immunogenicity trial", VACCINE, 0.85, None))
    out.append((808, "Farm workers and avian influenza exposure, below the threshold", EXPOSURE, 0.1, None))
    out.append((809, "Antibody titres after vaccination in mice, below the threshold", VACCINE, 0.1, None))
    return out


# ── Pure ─────────────────────────────────────────────────────────────────────
def test_a_row_teaches_what_the_review_decided():
    label = learning.decision_label
    assert label("included", None, None, None) == 1
    assert label("excluded", "included", "included", None) == 0      # the review's status wins
    assert label(None, None, None, "included") == 1                  # an arbitration
    assert label("pending", "included", None, None) == 1             # one vote is a decision in progress
    assert label(None, None, "excluded", None) == 0
    assert label(None, "included", "excluded", None) is None         # a disagreement teaches nothing
    assert label(None, None, None, None) is None
    assert label("pending", "pending", None, None) is None


def test_embeddings_are_read_only_when_nearly_every_candidate_has_one():
    choose = learning.choose_representation
    assert choose(6, 6, 100, 100) == "embeddings"
    assert choose(6, 6, 100, 95) == "embeddings"
    assert choose(6, 6, 100, 90) == "tfidf"                          # too many would get no score
    assert choose(6, 5, 100, 100) == "tfidf"                         # a decided article without one
    assert choose(0, 0, 0, 0) == "tfidf"


def test_training_waits_for_decisions_of_both_kinds():
    m = learning.MIN_PER_CLASS
    assert learning.enough_to_train(m, m)
    assert not learning.enough_to_train(m - 1, 50)
    assert not learning.enough_to_train(50, 0)


def test_the_model_puts_the_like_of_what_was_included_first():
    np = pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    pos = rng.normal([1, 0, 0], 0.1, size=(12, 3))
    neg = rng.normal([0, 1, 0], 0.1, size=(12, 3))
    x_lab = np.vstack([pos, neg])
    y = [1] * 12 + [0] * 12
    x_all = np.array([[0.9, 0.1, 0.0], [0.1, 0.9, 0.0], [0.5, 0.5, 0.0]])
    scores, auc = learning.train_scores(x_lab, y, x_all, sparse=False)
    assert scores[0] > scores[2] > scores[1]
    assert auc is not None and auc > 0.95                            # 12 of each: measured
    few, auc_few = learning.train_scores(x_lab[:6].tolist() + x_lab[12:18].tolist(), [1] * 6 + [0] * 6,
                                         x_all, sparse=False)
    assert few[0] > few[1] and auc_few is None                       # too few to measure


def test_a_pgvector_literal_is_read_as_floats():
    assert learning.parse_vector("[0.5,-1,2e-3]") == [0.5, -1.0, 0.002]
    assert learning.parse_vector(" [] ") == []


# ── With a database ──────────────────────────────────────────────────────────
def _engine_ok() -> bool:
    try:
        with main.engine.connect():
            return True
    except Exception:
        return False


@pytest.fixture()
def seeded(db_conn, monkeypatch):
    if not _engine_ok():
        pytest.skip("main.engine cannot reach the database")
    from conftest import patch_app
    scheduled: list[str] = []
    # The decision endpoints train in a background thread; here they only record the
    # call, and each test trains when it wants to, in the request.
    patch_app(monkeypatch, "schedule_retrain", lambda sid: scheduled.append(sid) or True)
    with db_conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS document_search")
        cur.execute("DROP TABLE IF EXISTS document_chunk, article_scenarios, literature_document CASCADE")
        cur.execute(
            "CREATE TABLE literature_document ("
            "id bigint PRIMARY KEY, title text NOT NULL, source text NOT NULL,"
            "abstract text, doi text, year int, journal text, external_id text,"
            "title_norm text, is_duplicate boolean DEFAULT false, screening_status text,"
            "project_context text DEFAULT 'literev')")
        cur.execute(
            "CREATE TABLE article_scenarios (scenario_id text, document_id bigint,"
            "similarity_score double precision, rerank_score double precision,"
            "learned_score double precision,"
            "screening_status text, screening_reason text, screening_notes text,"
            "screened_at timestamp, reviewer_1_status varchar(20), reviewer_1_reason text,"
            "reviewer_2_status varchar(20), reviewer_2_reason text,"
            "kappa_resolved boolean DEFAULT false, kappa_final_status varchar(20),"
            "cluster_id integer, cluster_label text,"
            "PRIMARY KEY (scenario_id, document_id))")
        # The embedding column is text here: CI's Postgres has no pgvector, and the
        # loader reads the column as text whatever its type.
        cur.execute(
            "CREATE TABLE document_chunk (id serial PRIMARY KEY, document_id bigint,"
            "chunk_index int, content text, chunk_type text, embedding text)")
        cur.execute("SELECT to_regclass('user_scenarios') IS NULL")
        if cur.fetchone()[0]:
            main._ensure_user_scenarios_table()
        cur.execute("SELECT to_regclass('scenario_settings') IS NULL")
        if cur.fetchone()[0]:
            main._ensure_scenario_settings_table()
        main._ensure_learning_tables()
        cur.execute("DELETE FROM scenario_learning WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM scenario_reviewer WHERE scenario_id = %s", (SID,))
        # The settings row outlives the tables above, and the audit sample is one of its
        # columns: left on by a run that switched it on, it draws every below-threshold
        # article into the blind sample and the queue tests read a different queue than
        # they were written for. The seed owns this scenario's settings, so it clears them.
        cur.execute("DELETE FROM scenario_settings WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))
        cur.execute("INSERT INTO user_scenarios (id, name, query, mode, filters)"
                    " VALUES (%s, 'The queue learns', 'avian influenza', 'boolean', '{}')", (SID,))
        cur.executemany(
            "INSERT INTO literature_document (id, title, source, abstract, year) VALUES (%s, %s, 'pubmed', %s, 2024)",
            [(d, t, a) for d, t, a, _, _ in _docs()])
        # The six decided articles were judged by both reviewers, who agreed: the review's
        # status is set and neither reviewer has them in the queue any more.
        cur.executemany(
            "INSERT INTO article_scenarios (scenario_id, document_id, similarity_score, screening_status,"
            " reviewer_1_status, reviewer_2_status, kappa_resolved, kappa_final_status)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            [(SID, d, sim, status, status, status, status is not None, status)
             for d, _, _, sim, status in _docs()])
        db_conn.commit()
    yield db_conn, scheduled
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM scenario_learning WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM scenario_reviewer WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM scenario_settings WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))
        db_conn.commit()


def _client():
    from fastapi.testclient import TestClient
    return TestClient(main.app)


def _scores(conn) -> dict[int, float | None]:
    with conn.cursor() as cur:
        cur.execute("SELECT document_id, learned_score FROM article_scenarios WHERE scenario_id = %s", (SID,))
        return {int(d): s for d, s in cur.fetchall()}


def _queue(c, code: str):
    r = c.get(f"/user-scenarios/{SID}/double-blind/queue?reviewer_code={code}&limit=10")
    assert r.status_code == 200, r.text
    return r.json()


def test_too_few_decisions_keep_the_relevance_order(seeded):
    conn, _ = seeded
    pytest.importorskip("sklearn")
    with conn.cursor() as cur:                      # two included, one excluded: not enough
        cur.execute("UPDATE article_scenarios SET screening_status = NULL, reviewer_1_status = NULL,"
                    " reviewer_2_status = NULL, kappa_resolved = FALSE, kappa_final_status = NULL"
                    " WHERE scenario_id = %s AND document_id IN (802, 804, 805)", (SID,))
        conn.commit()
    status = main.retrain(SID)
    assert status["trained"] is False and status["reason"] == "too_few_decisions"
    assert (status["n_included"], status["n_excluded"]) == (2, 1)
    assert all(s is None for s in _scores(conn).values())
    q = _queue(_client(), R1)
    assert q["ordering"] == "relevance" and q["learning"]["trained"] is False
    assert q["learning"]["min_per_class"] == learning.MIN_PER_CLASS


def test_the_queue_follows_the_decisions_once_trained(seeded):
    conn, _ = seeded
    pytest.importorskip("sklearn")
    status = main.retrain(SID)
    assert status["trained"] is True and status["representation"] == "tfidf"
    assert (status["n_included"], status["n_excluded"], status["n_scored"]) == (3, 3, 10)
    scores = _scores(conn)
    assert all(s is not None for s in scores.values()), "every article of the scenario is scored"
    assert scores[806] > scores[807], "the exposure paper, like the included ones, scores higher"
    assert scores[808] > scores[809]
    q = _queue(_client(), R1)
    assert q["ordering"] == "learned"
    undecided = [a["id"] for a in q["articles"] if not a["audit_sample"]]
    # 808 sits below the threshold and the model would include it, so it is IN the list to
    # screen, between the two relevant ones by learned score. It used to be named in the
    # card and left in the corpus for the reviewer to find, which on a real review meant
    # a hundred articles counted and never read.
    assert undecided == [806, 808, 807], "the relevant ones and the suggestion, by learned score"
    by_id = {a["id"]: a for a in q["articles"]}
    assert by_id[808]["below_threshold"] is True, "and it says it came in below the threshold"
    assert by_id[806]["below_threshold"] is False and by_id[807]["below_threshold"] is False
    assert q["remaining_suggested"] == 1 and q["remaining_relevant"] == 2
    assert q["articles"][0]["learned_score"] == pytest.approx(scores[806])
    info = q["learning"]
    assert info["remaining_relevant"] == 2
    assert 0 < info["expected_relevant_left"] < 2
    # The exposure paper below the threshold is the one the model would include.
    assert info["below_threshold_suggested"] == 1
    assert [s["id"] for s in info["suggested"]] == [808]
    r = _client().get(f"/user-scenarios/{SID}/learned-ranking")
    assert r.status_code == 200 and r.json()["suggested"][0]["id"] == 808


def test_the_audit_sample_is_never_dressed_up_as_a_suggestion(seeded):
    """The audit sample and the model's suggestions are drawn from the same pool: the
    below-threshold articles nobody judged. The sample exists to check the threshold
    blind, so an article in it must never be marked as something the model likes, or its
    kappa stops being an independent check. With the sample at 100 percent there is
    nothing left to suggest, and the queue says so rather than double-counting."""
    pytest.importorskip("sklearn")
    main.retrain(SID)
    c = _client()
    r = c.patch(f"/scenarios/{SID}/settings", json={"double_blind_below_sample": 100}, headers=KEY)
    assert r.status_code == 200, r.text
    try:
        q = _queue(c, R1)
        for a in q["articles"]:
            assert not (a["audit_sample"] and a["below_threshold"]), \
                f"article {a['id']} is in the blind audit sample and marked as a suggestion"
        assert q["remaining_suggested"] == 0, "the sample already screens them, blind"
        assert q["remaining_audit"] > 0
    finally:
        # The setting lives in the database and outlives this test, its file and this
        # run: left on, it puts every below-threshold article of this scenario into the
        # audit sample, and the test above that expects a suggestion in the queue fails
        # on the NEXT run, not this one.
        c.patch(f"/scenarios/{SID}/settings", json={"double_blind_below_sample": 0}, headers=KEY)


def test_every_suggestion_can_be_read_not_only_counted(seeded):
    """The card shows five suggestions and the count said ninety-seven, so on a real
    review the rest could be counted and never read. `limit` asks for more, and the export
    takes them all, in the model's own order with both scores beside each other."""
    pytest.importorskip("sklearn")
    main.retrain(SID)
    c = _client()
    state = c.get(f"/user-scenarios/{SID}/learned-ranking", params={"limit": 500}).json()
    assert len(state["suggested"]) == state["below_threshold_suggested"], "all of them, not a page"
    r = c.get(f"/user-scenarios/{SID}/learned-ranking/export")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/csv")
    body = r.content.decode("utf-8-sig")
    assert "rank,id,title,year,learned_score,similarity_score,url" in body
    assert str(state["suggested"][0]["id"]) in body
    assert body.count("\n") >= 3 + len(state["suggested"]) - 1, "a line per suggestion, plus the header"
    # The limit is bounded: a caller asking for a million does not get a million rows.
    assert len(c.get(f"/user-scenarios/{SID}/learned-ranking",
                     params={"limit": 10 ** 9}).json()["suggested"]) <= main.SUGGESTED_LIMIT_MAX


def test_embeddings_are_read_when_the_scenario_has_them(seeded):
    conn, _ = seeded
    pytest.importorskip("sklearn")
    pytest.importorskip("numpy")
    with conn.cursor() as cur:
        rows = []
        for i, (d, _, abstract, _, _) in enumerate(_docs()):
            v = [1.0, 0.05 * i, 0.0] if abstract == EXPOSURE else [0.05 * i, 1.0, 0.0]
            rows.append((d, "[" + ",".join(str(x) for x in v) + "]"))
        cur.executemany("INSERT INTO document_chunk (document_id, chunk_index, content, chunk_type, embedding)"
                        " VALUES (%s, 0, 'x', 'title_abstract', %s)", rows)
        conn.commit()
    status = main.retrain(SID)
    assert status["representation"] == "embeddings" and status["n_scored"] == 10
    scores = _scores(conn)
    assert scores[806] > 0.5 > scores[807]


def test_a_decision_schedules_the_training_once_it_is_written(seeded):
    conn, scheduled = seeded
    c = _client()
    r = c.post(f"/user-scenarios/{SID}/double-blind/decision", headers=KEY,
               json={"article_id": 806, "status": "included", "reviewer_code": R1})
    assert r.status_code == 200, r.text
    assert scheduled == [SID]
    r = c.post(f"/user-scenarios/{SID}/articles/807/screen?status=excluded", headers=KEY)
    assert r.status_code == 200, r.text
    assert scheduled == [SID, SID]
    c.post(f"/user-scenarios/{SID}/double-blind/decision", headers=KEY,
           json={"article_id": 806, "status": "excluded", "reviewer_code": R2})
    r = c.post(f"/user-scenarios/{SID}/double-blind/resolve?article_id=806&final_status=included", headers=KEY)
    assert r.status_code == 200, r.text
    assert scheduled[-1] == SID and len(scheduled) == 4
    refused = c.post(f"/user-scenarios/{SID}/double-blind/decision", headers=KEY,
                     json={"article_id": 806, "status": "included", "reviewer_code": R1})
    assert refused.status_code == 409 and len(scheduled) == 4, "a refused vote trains nothing"


def test_a_single_vote_already_teaches_the_model(seeded):
    """One reviewer's vote counts before the second has voted: the queue should move
    from the first decisions, not once both have gone through the whole list."""
    conn, _ = seeded
    pytest.importorskip("sklearn")
    with conn.cursor() as cur:
        cur.execute("UPDATE article_scenarios SET screening_status = NULL, kappa_resolved = FALSE,"
                    " kappa_final_status = NULL, reviewer_2_status = NULL"
                    " WHERE scenario_id = %s AND document_id IN (800, 801, 802)", (SID,))
        cur.execute("UPDATE article_scenarios SET screening_status = NULL, kappa_resolved = FALSE,"
                    " kappa_final_status = NULL, reviewer_1_status = NULL"
                    " WHERE scenario_id = %s AND document_id IN (803, 804, 805)", (SID,))
        conn.commit()
    status = main.retrain(SID)
    assert status["trained"] and (status["n_included"], status["n_excluded"]) == (3, 3)


def test_the_retrain_endpoint_needs_the_key_and_returns_the_state(seeded):
    pytest.importorskip("sklearn")
    c = _client()
    assert c.post(f"/user-scenarios/{SID}/learned-ranking/retrain").status_code in (401, 403)
    r = c.post(f"/user-scenarios/{SID}/learned-ranking/retrain", headers=KEY)
    assert r.status_code == 200, r.text
    assert r.json()["trained"] is True and r.json()["remaining_relevant"] == 2
    assert c.get("/user-scenarios/no-such-scenario/learned-ranking").status_code == 404


def test_one_training_runs_at_a_time_and_a_decision_during_it_runs_it_again(monkeypatch):
    """The scheduler, without a database: a second call while one runs makes the worker
    loop once more at the end, never start a second thread."""
    import threading
    runs: list[str] = []
    gate = threading.Event()

    def fake_retrain(sid):
        runs.append(sid)
        if len(runs) == 1:
            gate.wait(5)
        return {}

    monkeypatch.setattr(learning, "retrain", fake_retrain)
    assert learning.schedule_retrain("s") is True
    import time
    for _ in range(50):                               # until the first run is inside retrain
        if runs:
            break
        time.sleep(0.02)
    assert learning.schedule_retrain("s") is False     # one in flight: queued, not started
    assert learning.schedule_retrain("s") is False
    gate.set()
    for _ in range(100):
        if "s" not in learning._running:
            break
        time.sleep(0.02)
    assert runs == ["s", "s"], "run once more at the end, not once per decision"
    assert "s" not in learning._running and "s" not in learning._again
