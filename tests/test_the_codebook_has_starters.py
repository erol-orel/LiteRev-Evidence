"""The starter vocabularies: a new review does not begin by deleting avian influenza.

The default codebook is the One Health one, and it is right for the reviews it was written
for: poultry worker, live bird market, duck, biosecurity. It is the wrong starting point for
a review of hospital capacity or of how long a virus survives on a surface, and a reviewer
who meets it there has to delete most of it before adding anything.

So it becomes one starter among four, loaded in one click, and a scenario created from now
on begins with the generic one, which holds only what is true of any epidemiological review.

The constraint that shapes all of this: DEFAULT_NODES must not change. A scenario that never
stored a codebook reads it at serve time, so changing it would silently rewrite the labels
of every review already running, the four partner ones included. The tests below pin that
boundary as hard as they pin the starters themselves.
"""
import pytest

pytest.importorskip("fastapi")

import main  # noqa: E402
from api.codebook import (  # noqa: E402
    DEFAULT_NODES,
    DEFAULT_TEMPLATE,
    TEMPLATES,
    Index,
    codebook_template,
    normalise,
    template_list,
    validate_nodes,
)


# ── Pure: the starters themselves ───────────────────────────────────────────
def test_every_starter_passes_the_same_validation_as_an_uploaded_file():
    """A starter is a codebook. If one would be refused as an import, it has no business
    being offered as a starting point."""
    assert len(TEMPLATES) == 4
    for t in TEMPLATES:
        nodes = codebook_template(t["key"])
        assert nodes == validate_nodes([dict(n) for n in nodes]), t["key"]
        assert len(nodes) >= 15, f"{t['key']} is too thin to start from"


def test_an_unknown_starter_says_which_ones_exist():
    with pytest.raises(ValueError, match="generic_epi"):
        codebook_template("no_such_template")


def test_the_generic_starter_has_no_disease_in_it():
    """The point of it. No species, no poultry worker, no live bird market: a review of
    dengue or of hospital capacity must not have to delete them first."""
    nodes = codebook_template("generic_epi")
    keys = {(n["sheet"], n["l1"], n.get("l2")) for n in nodes}
    flat = " ".join(f"{n['l1']} {n.get('l2') or ''} {' '.join(n['synonyms'])}" for n in nodes)
    assert not any(l1 == "species" for _, l1, _ in keys), "species is a One Health question"
    for word in ("poultry", "live bird market", "duck", "chicken", "biosecurity", "culling"):
        assert word not in flat, f"{word!r} is specific to avian influenza"
    # And it does hold what every epidemiological review asks.
    groups = {(n["sheet"], n["l1"]) for n in nodes}
    for group in (("human_susc", "sex_gender"), ("human_susc", "age"), ("human_susc", "outcome"),
                  ("human_exp", "occupation"), ("human_exp", "contact_type"),
                  ("env", "setting"), ("animal", "test")):
        assert group in groups, group


def test_the_one_health_starter_is_exactly_the_old_default():
    """It is not a new vocabulary, it is the one that was there, now named and loadable.
    Were it to drift, a review that loads it would stop matching the reviews that read the
    default, and the two would pool different labels from the same corpus."""
    assert codebook_template("one_health") == validate_nodes([dict(n) for n in DEFAULT_NODES])


def test_the_healthcare_capacity_starter_answers_the_question_it_was_asked_for():
    """The codebook the ISS reviewers asked for: beds, intensive care, staff, surge."""
    nodes = codebook_template("healthcare_capacity")
    groups = {n["l1"] for n in nodes}
    assert {"beds", "staff", "equipment", "surge", "access"} <= groups
    index = Index(nodes)
    # The words a paper actually uses must land on the right node.
    assert normalise(index, "env", "beds", "ICU beds")["l2"] == "icu_beds"
    assert normalise(index, "env", "staff", "nurse to patient ratio")["l2"] == "nurses"
    assert normalise(index, "env", "equipment", "mechanical ventilation")["l2"] == "ventilators"


def test_the_persistence_starter_reads_a_survival_experiment():
    nodes = codebook_template("env_persistence")
    index = Index(nodes)
    assert normalise(index, "env", "matrix", "stainless steel")["l2"] == "surface"
    assert normalise(index, "env", "condition", "relative humidity")["l2"] == "humidity"
    assert normalise(index, "env", "persistence", "half life")["l2"] == "survival_time"
    assert normalise(index, "animal", "test", "TCID50")["l2"] == "culture"


def test_the_default_for_a_new_scenario_is_the_generic_one():
    assert DEFAULT_TEMPLATE == "generic_epi"
    listed = {t["key"]: t for t in template_list("en")}
    assert listed["generic_epi"]["is_default_for_new"] is True
    assert sum(1 for t in listed.values() if t["is_default_for_new"]) == 1
    assert listed["one_health"]["n_groups"] >= 15
    fr = {t["key"]: t for t in template_list("fr")}
    assert fr["generic_epi"]["name"] != listed["generic_epi"]["name"], "named in both languages"


def test_the_read_default_is_untouched_so_running_reviews_do_not_move():
    """The boundary that protects the four partner reviews. They stored no codebook, so they
    read DEFAULT_NODES every time it is served; a starter that replaced it would rewrite
    their pooled labels without a single screen saying so."""
    assert len(DEFAULT_NODES) == 73
    groups = {(n["sheet"], n["l1"]) for n in DEFAULT_NODES}
    assert ("animal", "species") in groups and ("env", "biosecurity") in groups
    index = Index([dict(n) for n in DEFAULT_NODES])
    assert normalise(index, "human_exp", "occupation", "poultry farmer")["l2"] == "poultry_worker"


# ── Over the database ───────────────────────────────────────────────────────
def _engine_ok() -> bool:
    try:
        with main.engine.connect():
            return True
    except Exception:                                        # noqa: BLE001
        return False


SID = "usr-starter-test"


@pytest.fixture()
def scenario(db_conn):
    if not _engine_ok():
        pytest.skip("main.engine cannot reach the database")
    with db_conn.cursor() as cur:
        cur.execute("SELECT to_regclass('user_scenarios') IS NULL")
        if cur.fetchone()[0]:
            main._ensure_user_scenarios_table()
        cur.execute("DELETE FROM scenario_settings WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))
        cur.execute("INSERT INTO user_scenarios (id, name, query, mode, filters, pinned) "
                    "VALUES (%s, 'Starters', 'dengue', 'boolean', '{}', TRUE)", (SID,))
    yield db_conn
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM scenario_settings WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))


HDR = {"X-API-Key": "test-write-key"}


def _client():
    from fastapi.testclient import TestClient
    return TestClient(main.app)


def test_the_starters_are_listed_for_the_picker():
    r = _client().get("/codebook/templates", params={"lang": "en"})
    assert r.status_code == 200
    body = r.json()
    assert body["default_for_new"] == "generic_epi"
    keys = [t["key"] for t in body["templates"]]
    assert keys == ["generic_epi", "one_health", "healthcare_capacity", "env_persistence"]
    for t in body["templates"]:
        assert t["name"] and t["note"] and t["n_nodes"] > 0 and t["n_groups"] > 0


def test_loading_a_starter_replaces_the_codebook_and_says_how_much_it_wrote(scenario):
    client = _client()
    r = client.post(f"/user-scenarios/{SID}/codebook/template/healthcare_capacity", headers=HDR)
    assert r.status_code == 200, r.text
    assert r.json()["template"] == "healthcare_capacity"
    n = r.json()["n_nodes"]
    got = client.get(f"/user-scenarios/{SID}/codebook").json()
    assert got["source"] == "custom" and got["n_nodes"] == n
    assert {node["l1"] for node in got["nodes"]} >= {"beds", "staff", "surge"}
    # It REPLACES: the One Health vocabulary is gone, not merged into it.
    assert not any(node["l1"] == "species" for node in got["nodes"])


def test_an_unknown_starter_is_a_404_not_a_wiped_codebook(scenario):
    client = _client()
    client.post(f"/user-scenarios/{SID}/codebook/template/generic_epi", headers=HDR)
    before = client.get(f"/user-scenarios/{SID}/codebook").json()["n_nodes"]
    r = client.post(f"/user-scenarios/{SID}/codebook/template/nope", headers=HDR)
    assert r.status_code == 404
    assert client.get(f"/user-scenarios/{SID}/codebook").json()["n_nodes"] == before


def test_a_scenario_created_now_starts_from_the_generic_starter(db_conn):
    """The ask: the same starting page for everyone, whatever the review is about."""
    if not _engine_ok():
        pytest.skip("main.engine cannot reach the database")
    client = _client()
    r = client.post("/user-scenarios", headers=HDR,
                    json={"name": "Brand new", "query": "dengue vaccine", "mode": "boolean",
                          "filters": {}, "result_count": 0, "pinned": False})
    assert r.status_code == 201, r.text
    new_id = r.json()["id"]
    try:
        got = client.get(f"/user-scenarios/{new_id}/codebook").json()
        assert got["source"] == "custom", "the starter was stored, not merely read"
        assert got["n_nodes"] == len(codebook_template("generic_epi"))
        assert not any(n["l1"] == "species" for n in got["nodes"])
    finally:
        with db_conn.cursor() as cur:
            cur.execute("DELETE FROM scenario_settings WHERE scenario_id = %s", (new_id,))
            cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (new_id,))
            cur.execute("DELETE FROM user_scenarios WHERE id = %s", (new_id,))


def test_a_scenario_that_stored_nothing_still_reads_the_one_health_default(scenario):
    """The four partner reviews are in this state. Nothing about them moves."""
    got = _client().get(f"/user-scenarios/{SID}/codebook").json()
    assert got["source"] == "default" and got["n_nodes"] == len(DEFAULT_NODES)
