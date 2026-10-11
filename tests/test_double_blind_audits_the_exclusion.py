"""The double blind can audit what the machine threw out, and its kappa says what it covers.

The queue only ever offered the relevant set: articles below the threshold were never
shown to the two reviewers, so the automatic exclusion was never checked, and the kappa
measured agreement on the machine's selection alone, with nothing on the panel saying so.

A scenario option (`double_blind_below_sample`, 0 = off, 100 when switched on) now adds a
random sample of articles from below the threshold, with an abstract, the same for both
reviewers, each marked as such. The kappa is reported per stratum.
"""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("psycopg")

import main  # noqa: E402

SID = "usr-dbaudit-0001"
R1, R2 = "R-6101", "R-6102"
KEY = {"X-API-Key": "test-write-key"}
RELEVANT = [800, 801, 802, 803]
BELOW = list(range(810, 817))           # seven articles below the threshold, with an abstract


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
            "id bigint PRIMARY KEY, title text NOT NULL, source text NOT NULL DEFAULT 'pubmed',"
            "abstract text, doi text, year int, journal text, is_duplicate boolean DEFAULT false,"
            "screening_status text, project_context text DEFAULT 'literev')")
        cur.execute(
            "CREATE TABLE article_scenarios (scenario_id text, document_id bigint,"
            "similarity_score double precision, rerank_score double precision,"
            "screening_status text, screening_reason text, screening_notes text,"
            "screened_at timestamp, reviewer_1_status varchar(20), reviewer_1_reason text,"
            "reviewer_2_status varchar(20), reviewer_2_reason text,"
            "kappa_resolved boolean DEFAULT false, kappa_final_status varchar(20),"
            "PRIMARY KEY (scenario_id, document_id))")
        cur.execute("SELECT to_regclass('user_scenarios') IS NULL")
        if cur.fetchone()[0]:
            main._ensure_user_scenarios_table()
        cur.execute("SELECT to_regclass('scenario_settings') IS NULL")
        if cur.fetchone()[0]:
            main._ensure_scenario_settings_table()
        for t, col in (("user_scenarios", "id"), ("scenario_settings", "scenario_id"),
                       ("scenario_reviewer", "scenario_id"), ("double_blind_audit", "scenario_id")):
            cur.execute(f"DELETE FROM {t} WHERE {col} = %s", (SID,))
        cur.execute("INSERT INTO user_scenarios (id, name, query, mode, filters)"
                    " VALUES (%s, 'Audit', 'avian influenza', 'boolean', '{}')", (SID,))
        abstract = "an abstract long enough to be screened by two reviewers"
        rows = ([(i, abstract, 0.9 - k * 0.05, None, False) for k, i in enumerate(RELEVANT)]
                + [(i, abstract, 0.1 + k * 0.03, None, False) for k, i in enumerate(BELOW)]
                + [(820, None, 0.2, None, False),          # below, no abstract
                   (821, "short", 0.2, None, False),       # below, no usable abstract
                   (830, abstract, 0.2, "excluded", False),  # below, excluded by a reviewer
                   (840, abstract, 0.2, None, True)])      # below, a duplicate
        for i, abs_, sim, status, dup in rows:
            cur.execute("INSERT INTO literature_document (id, title, abstract, is_duplicate)"
                        " VALUES (%s, %s, %s, %s)", (i, f"Paper {i}", abs_, dup))
            cur.execute("INSERT INTO article_scenarios (scenario_id, document_id, similarity_score,"
                        " screening_status) VALUES (%s, %s, %s, %s)", (SID, i, sim, status))
    yield db_conn
    with db_conn.cursor() as cur:
        for t, col in (("user_scenarios", "id"), ("scenario_settings", "scenario_id"),
                       ("scenario_reviewer", "scenario_id"), ("double_blind_audit", "scenario_id")):
            cur.execute(f"DELETE FROM {t} WHERE {col} = %s", (SID,))


def _client():
    from fastapi.testclient import TestClient
    return TestClient(main.app)


def _queue(c, code, limit=200):
    r = c.get(f"/user-scenarios/{SID}/double-blind/queue?reviewer_code={code}&limit={limit}")
    assert r.status_code == 200, r.text
    return r.json()


def _sample(c, n):
    r = c.patch(f"/scenarios/{SID}/settings", json={"double_blind_below_sample": n}, headers=KEY)
    assert r.status_code == 200, r.text
    return r.json()


def _vote(c, doc, status, code):
    r = c.post(f"/user-scenarios/{SID}/double-blind/decision", headers=KEY,
               json={"article_id": doc, "status": status, "reviewer_code": code})
    assert r.status_code == 200, r.text
    return r.json()


def test_without_the_option_the_queue_is_the_relevant_set(seeded):
    q = _queue(_client(), R1)
    assert [a["id"] for a in q["articles"]] == RELEVANT
    assert q["remaining"] == 4 and q["remaining_audit"] == 0 and q["audit_sample_size"] == 0
    assert not any(a["audit_sample"] for a in q["articles"])


def test_the_option_adds_a_marked_sample_from_below_the_threshold(seeded):
    c = _client()
    assert _sample(c, 100)["double_blind_below_sample"] == 100
    q = _queue(c, R1)
    audit = {a["id"] for a in q["articles"] if a["audit_sample"]}
    assert audit == set(BELOW), "only articles below the threshold, with an abstract, never decided"
    assert {a["id"] for a in q["articles"] if not a["audit_sample"]} == set(RELEVANT)
    assert q["remaining"] == 11 and q["remaining_relevant"] == 4 and q["remaining_audit"] == 7
    # Interleaved in proportion: the first page already holds both kinds.
    first = _queue(c, R1, limit=3)["articles"]
    assert {a["audit_sample"] for a in first} == {True, False}


def test_the_sample_is_the_same_for_both_reviewers_and_holds_under_their_votes(seeded):
    c = _client()
    _sample(c, 3)
    picked = [a["id"] for a in _queue(c, R1)["articles"] if a["audit_sample"]]
    assert len(picked) == 3
    assert [a["id"] for a in _queue(c, R2)["articles"] if a["audit_sample"]] == picked
    for doc, status in zip(picked, ("included", "excluded", "excluded")):
        _vote(c, doc, status, R1)
    assert [a["id"] for a in _queue(c, R2)["articles"] if a["audit_sample"]] == picked
    for doc, status in zip(picked, ("included", "excluded", "excluded")):
        _vote(c, doc, status, R2)
    # Decided by the two reviewers: the sample does not refill with other articles.
    for code in (R1, R2):
        assert _queue(c, code)["remaining_audit"] == 0


def test_the_kappa_is_reported_per_stratum_and_says_what_it_covers(seeded):
    c = _client()
    _sample(c, 3)
    a1, a2 = [a["id"] for a in _queue(c, R1)["articles"] if a["audit_sample"]][:2]
    for doc, s1, s2 in ((800, "included", "included"), (801, "included", "excluded"),
                        (a1, "included", "included"), (a2, "excluded", "excluded")):
        _vote(c, doc, s1, R1)
        _vote(c, doc, s2, R2)
    k = c.get(f"/user-scenarios/{SID}/double-blind/kappa").json()
    rel, aud = k["strata"]["relevant"], k["strata"]["audit_sample"]
    assert rel["n_evaluated"] == 2 and rel["conflicts"] == 1
    # a1, included by both, is now in the relevant set; it stays in the audit stratum.
    assert aud["n_evaluated"] == 2 and aud["conflicts"] == 0 and aud["kappa"] == 1.0
    assert k["n_evaluated"] == 2 and k["stratum"] == "relevant"      # the top-level figures
    assert "sous le seuil" in aud["covers"].lower() and "sélection de la machine" in rel["covers"]
    assert k["audit_sample_size"] == 3
    conflicts = c.get(f"/user-scenarios/{SID}/double-blind/conflicts").json()
    assert [(x["id"], x["audit_sample"]) for x in conflicts] == [(801, False)]


def test_switching_the_sample_off_keeps_its_votes_where_they_were(seeded):
    c = _client()
    _sample(c, 3)
    a1 = [a["id"] for a in _queue(c, R1)["articles"] if a["audit_sample"]][0]
    _vote(c, a1, "excluded", R1)
    _vote(c, a1, "excluded", R2)
    _sample(c, 0)
    assert _queue(c, R1)["remaining_audit"] == 0
    assert c.get(f"/user-scenarios/{SID}/double-blind/kappa").json()["strata"]["audit_sample"]["n_evaluated"] == 1


def test_the_option_is_validated(seeded):
    c = _client()
    for bad in (-1, 5000, "many"):
        r = c.patch(f"/scenarios/{SID}/settings", json={"double_blind_below_sample": bad}, headers=KEY)
        assert r.status_code == 422, (bad, r.text)
    _sample(c, 100)
    assert c.get(f"/scenarios/{SID}/settings").json()["double_blind_below_sample"] == 100
