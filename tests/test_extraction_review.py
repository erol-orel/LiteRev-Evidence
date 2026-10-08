"""Reviewing the extraction (api/extraction_review.py): accept, edit or reject each
observation, per reviewer, in a table of its own.

Pure tests pin the logic that must not drift: a stable key per observation, how several
reviewers' decisions combine (a disagreement is a CONFLICT, never settled silently), and
Cohen's kappa against a value worked out by hand. The integration tests check the part a
reviewer relies on: the model's output is never overwritten, a decision survives, and the
exports carry the corrections and leave the rejected rows out."""
import io
import json

import pytest

pytest.importorskip("fastapi")

import main  # noqa: E402
from api import extraction_review as rv  # noqa: E402
from test_extraction import HDR, SID, extracted, seeded  # noqa: E402,F401


def _o(sheet="human_susc", cov="Male", quote="3 of 10 were male"):
    return {"sheet": sheet, "covariate": cov, "quote": quote}


# ── Keys ────────────────────────────────────────────────────────────────────
def test_the_key_follows_sheet_covariate_and_quote_and_nothing_else():
    a = rv.observation_keys([_o()])[0]
    assert len(a) == 12
    # The value, the group and the codebook do not move it: an edit must not orphan its own review.
    assert rv.observation_keys([{**_o(), "value": 9, "group": "sex"}])[0] == a
    assert rv.observation_keys([_o(cov="MALES ")])[0] == rv.observation_keys([_o(cov="male")])[0]
    assert rv.observation_keys([_o(quote="a different quote")])[0] != a
    assert rv.observation_keys([_o(sheet="human_exp")])[0] != a


def test_two_identical_rows_get_two_keys():
    k = rv.observation_keys([_o(), _o(), _o()])
    assert len(set(k)) == 3 and k[1].endswith("-2") and k[2].endswith("-3")


# ── What a reviewer may change ──────────────────────────────────────────────
def test_edits_are_validated_and_only_known_fields_pass():
    assert rv.clean_edits({"value": "12,5", "n_cases": 3, "pop_risk": ""}) == {"value": 12.5, "n_cases": 3, "pop_risk": None}
    assert rv.clean_edits({"covariate": "  male   adults "}) == {"covariate": "male adults"}
    assert rv.clean_edits({"sheet": "env"}) == {"sheet": "env"}
    for bad in ({}, None, {"quote": "x"}, {"sheet": "nope"}, {"value": "abc"}, {"value": float("nan")}):
        with pytest.raises(ValueError):
            rv.clean_edits(bad)


def test_the_reviewer_name_is_bounded():
    assert rv.clean_reviewer("  Ana  Maria ") == "Ana Maria"
    for bad in ("", "A", None, "x" * 41):
        with pytest.raises(ValueError):
            rv.clean_reviewer(bad)


def test_applying_edits_does_not_change_the_original():
    o = {"value": 3, "covariate": "male"}
    e = rv.apply_edits(o, {"value": 4})
    assert e == {"value": 4, "covariate": "male"} and o["value"] == 3
    assert rv.apply_edits(o, None) == o


# ── Combining reviewers ─────────────────────────────────────────────────────
def _r(who, status, at="2026-01-01T00:00:00", edits=None):
    return {"reviewer": who, "status": status, "updated_at": at, "edits": edits}


def test_nothing_reviewed_is_unreviewed():
    assert rv.combine([])["status"] == "unreviewed"


def test_agreeing_reviewers_give_one_status():
    assert rv.combine([_r("Ana", "accepted"), _r("Ben", "accepted")])["status"] == "accepted"
    assert rv.combine([_r("Ana", "rejected"), _r("Ben", "rejected")])["status"] == "rejected"
    assert rv.combine([_r("Ana", "accepted"), _r("Ben", "edited", edits={"value": 5})])["status"] == "edited"


def test_a_disagreement_is_a_conflict_and_is_never_settled_silently():
    c = rv.combine([_r("Ana", "accepted"), _r("Ben", "rejected")])
    assert c["status"] == "conflict" and c["conflict"] is True and c["reviewers"] == ["Ana", "Ben"]
    assert rv.combine([_r("Ana", "edited", edits={"value": 1}), _r("Ben", "rejected")])["status"] == "conflict"


def test_the_latest_edit_is_the_one_applied():
    c = rv.combine([_r("Ana", "edited", "2026-01-01T00:00:00", {"value": 1}),
                    _r("Ben", "edited", "2026-02-01T00:00:00", {"value": 2})])
    assert c["status"] == "edited" and c["edits"] == {"value": 2}


# ── Agreement ───────────────────────────────────────────────────────────────
def test_kappa_matches_a_hand_computed_value():
    # 10 observations: 4 both keep, 3 both reject, 2 first keeps and second rejects, 1 the reverse.
    # observed 0.7; chance = 0.6*0.5 + 0.4*0.5 = 0.5; kappa = (0.7 - 0.5) / (1 - 0.5) = 0.4
    pairs = [(True, True)] * 4 + [(False, False)] * 3 + [(True, False)] * 2 + [(False, True)]
    assert rv.cohen_kappa(pairs) == {"n_common": 10, "observed": 0.7, "kappa": 0.4}


def test_kappa_edge_cases_are_undefined_not_wrong():
    assert rv.cohen_kappa([]) == {"n_common": 0, "observed": None, "kappa": None}
    assert rv.cohen_kappa([(True, True)] * 5)["kappa"] is None          # no disagreement to measure
    assert rv.cohen_kappa([(True, True), (False, False)])["kappa"] == 1.0
    assert rv.cohen_kappa([(True, False), (False, True)])["kappa"] == -1.0


def test_agreement_is_computed_for_every_pair_over_what_both_reviewed():
    by_obs = {1: [_r("Ana", "accepted"), _r("Ben", "accepted"), _r("Cy", "rejected")],
              2: [_r("Ana", "rejected"), _r("Ben", "rejected")],
              3: [_r("Ana", "accepted")]}                                 # only Ana: in no pair
    out = {tuple(a["reviewers"]): a for a in rv.agreement(by_obs)}
    assert out[("Ana", "Ben")]["n_common"] == 2 and out[("Ana", "Ben")]["kappa"] == 1.0
    assert out[("Ana", "Cy")]["n_common"] == 1 and out[("Ben", "Cy")]["n_common"] == 1
    assert len(out) == 3


# ── Endpoints ───────────────────────────────────────────────────────────────
def _client():
    from fastapi.testclient import TestClient
    return TestClient(main.app)


def _obs(c, aid=9701):
    return c.get(f"/user-scenarios/{SID}/articles/{aid}/extraction").json()["extraction"]["observations"]


def _post(c, aid, body, headers=HDR):
    return c.post(f"/user-scenarios/{SID}/articles/{aid}/extraction/review", json=body, headers=headers)


@pytest.fixture()
def clean_reviews(extracted):
    with extracted.cursor() as cur:
        cur.execute("DELETE FROM extraction_review WHERE document_id BETWEEN 9701 AND 9706")
    yield extracted
    with extracted.cursor() as cur:
        cur.execute("DELETE FROM extraction_review WHERE document_id BETWEEN 9701 AND 9706")


def test_a_decision_needs_the_key_a_valid_body_and_a_real_observation(clean_reviews):
    c = _client()
    key = _obs(c)[0]["obs_key"]
    assert _post(c, 9701, {"obs_key": key, "reviewer": "Ana", "status": "accepted"}, {}).status_code in (401, 403)
    for body in ({"obs_key": key, "reviewer": "A", "status": "accepted"},
                 {"obs_key": key, "reviewer": "Ana", "status": "maybe"},
                 {"obs_key": key, "reviewer": "Ana", "status": "edited"},
                 {"obs_key": key, "reviewer": "Ana", "status": "edited", "edits": {"quote": "x"}}):
        assert _post(c, 9701, body).status_code == 422, body
    assert _post(c, 9701, {"obs_key": "nope", "reviewer": "Ana", "status": "accepted"}).status_code == 404
    # An article that is not in the scenario cannot be reviewed through it.
    assert _post(c, 99999, {"obs_key": key, "reviewer": "Ana", "status": "accepted"}).status_code == 404


def test_accepting_editing_and_rejecting_never_touch_the_stored_extraction(clean_reviews):
    c = _client()
    before = _obs(c)[0]
    key = before["obs_key"]
    assert before["review_status"] == "unreviewed" and before["effective"]["n_cases"] == 3
    assert before["value"] is None

    r = _post(c, 9701, {"obs_key": key, "reviewer": "Ana", "status": "edited", "edits": {"value": 4, "n_cases": 4}}).json()
    assert r["observation"]["review_status"] == "edited" and r["observation"]["effective"]["value"] == 4
    after = _obs(c)[0]
    assert after["value"] is None and after["n_cases"] == 3                    # the model's reading is still there
    assert after["effective"]["n_cases"] == 4 and after["effective"]["value"] == 4
    with clean_reviews.cursor() as cur:
        cur.execute("SELECT extraction_json->'observations'->0->>'n_cases' FROM literature_document WHERE id = 9701")
        assert cur.fetchone()[0] == "3"

    # A second reviewer disagrees: a conflict, with both decisions visible.
    r = _post(c, 9701, {"obs_key": key, "reviewer": "Ben", "status": "rejected", "note": "wrong group"}).json()
    o = r["observation"]
    assert o["review_status"] == "conflict" and {x["reviewer"] for x in o["reviews"]} == {"Ana", "Ben"}
    # Ben changes his mind, then clears: back to Ana's edit alone.
    assert _post(c, 9701, {"obs_key": key, "reviewer": "Ben", "status": "accepted"}).json()["observation"]["review_status"] == "edited"
    assert _post(c, 9701, {"obs_key": key, "reviewer": "Ben", "status": "clear"}).json()["observation"]["review_status"] == "edited"
    assert _post(c, 9701, {"obs_key": key, "reviewer": "Ana", "status": "clear"}).json()["observation"]["review_status"] == "unreviewed"


def test_an_edit_that_moves_the_label_is_read_through_the_codebook(clean_reviews):
    c = _client()
    key = _obs(c)[0]["obs_key"]
    o = _post(c, 9701, {"obs_key": key, "reviewer": "Ana", "status": "edited",
                        "edits": {"covariate": "Women"}}).json()["observation"]
    assert o["label_path"] == "sex_gender > male"                    # the model's label, as stored
    assert o["effective"]["label_path"] == "sex_gender > female"     # the reviewer's, through the codebook


def test_bulk_accept_takes_the_verified_rows_and_leaves_decided_ones_alone(clean_reviews):
    c = _client()
    obs = _obs(c)                                                    # 9701: one verified row, one not
    url = f"/user-scenarios/{SID}/articles/9701/extraction/review/bulk"
    assert c.post(url, json={"reviewer": "Ana", "status": "accepted", "verified_only": True}).status_code in (401, 403)
    r = c.post(url, json={"reviewer": "Ana", "status": "accepted", "verified_only": True}, headers=HDR).json()
    assert (r["n_recorded"], r["n_skipped"]) == (1, 0)
    statuses = {o["obs_key"]: o["review_status"] for o in _obs(c)}
    assert sorted(statuses.values()) == ["accepted", "unreviewed"]
    # Ana rejects one by hand; a second bulk run does not overwrite it.
    unreviewed = next(k for k, v in statuses.items() if v == "unreviewed")
    _post(c, 9701, {"obs_key": unreviewed, "reviewer": "Ana", "status": "rejected"})
    again = c.post(url, json={"reviewer": "Ana", "status": "accepted", "obs_keys": list(statuses)}, headers=HDR).json()
    assert again["n_recorded"] == 0 and again["n_skipped"] == 2
    assert c.post(url, json={"reviewer": "Ana", "status": "edited"}, headers=HDR).status_code == 422
    assert c.post(url, json={"reviewer": "Ana", "status": "accepted"}, headers=HDR).status_code == 422


def test_the_summary_counts_every_observation_and_scores_agreement(clean_reviews):
    c = _client()
    k0, k1 = (o["obs_key"] for o in _obs(c, 9701))
    k2 = _obs(c, 9702)[0]["obs_key"]
    _post(c, 9701, {"obs_key": k0, "reviewer": "Ana", "status": "accepted"})
    _post(c, 9701, {"obs_key": k0, "reviewer": "Ben", "status": "accepted"})
    _post(c, 9701, {"obs_key": k1, "reviewer": "Ana", "status": "accepted"})
    _post(c, 9701, {"obs_key": k1, "reviewer": "Ben", "status": "rejected"})
    _post(c, 9702, {"obs_key": k2, "reviewer": "Ana", "status": "rejected"})
    s = c.get(f"/user-scenarios/{SID}/extraction/review/summary").json()
    # The relevant extracted articles: 9701 (2 rows), 9702 (2 rows), 9703 (0 rows): 4 observations.
    assert s["n_observations"] == 4
    assert s["counts"] == {"unreviewed": 1, "accepted": 1, "edited": 0, "rejected": 1, "conflict": 1}
    assert s["n_reviewed"] == 3 and s["share_reviewed"] == 0.75
    assert {r["reviewer"]: r["n_decisions"] for r in s["reviewers"]} == {"Ana": 3, "Ben": 2}
    ab = s["agreement"][0]
    assert ab["reviewers"] == ["Ana", "Ben"] and ab["n_common"] == 2 and ab["observed"] == 0.5
    assert c.get("/user-scenarios/nope/extraction/review/summary").status_code == 404


def test_a_decision_whose_observation_changed_is_stale_not_applied(clean_reviews):
    c = _client()
    key = _obs(c)[0]["obs_key"]
    _post(c, 9701, {"obs_key": key, "reviewer": "Ana", "status": "rejected"})
    with clean_reviews.cursor() as cur:       # a re-extraction reads the paper differently
        cur.execute("UPDATE literature_document SET extraction_json = jsonb_set(extraction_json, "
                    "'{observations,0,quote}', '\"a quote the new reading has\"') WHERE id = 9701")
    body = c.get(f"/user-scenarios/{SID}/articles/9701/extraction").json()
    assert [o["review_status"] for o in body["extraction"]["observations"]][0] == "unreviewed"
    assert [s["obs_key"] for s in body["stale_reviews"]] == [key]
    assert c.get(f"/user-scenarios/{SID}/extraction/review/summary").json()["n_stale_decisions"] == 1


def test_the_list_carries_each_articles_review_counts(clean_reviews):
    c = _client()
    k0, k1 = (o["obs_key"] for o in _obs(c, 9701))
    _post(c, 9701, {"obs_key": k0, "reviewer": "Ana", "status": "accepted"})
    _post(c, 9701, {"obs_key": k1, "reviewer": "Ana", "status": "rejected"})
    _post(c, 9701, {"obs_key": k1, "reviewer": "Ben", "status": "accepted"})
    a = {x["id"]: x for x in c.get(f"/user-scenarios/{SID}/extraction/articles").json()["articles"]}
    assert (a[9701]["n_reviewed"], a[9701]["n_rejected"], a[9701]["n_conflict"]) == (2, 0, 1)
    assert (a[9702]["n_reviewed"], a[9706]["n_reviewed"]) == (0, 0)


def test_the_exports_apply_corrections_and_leave_the_rejected_out(clean_reviews):
    openpyxl = pytest.importorskip("openpyxl")
    c = _client()
    k0, k1 = (o["obs_key"] for o in _obs(c, 9701))
    _post(c, 9701, {"obs_key": k0, "reviewer": "Ana", "status": "edited", "edits": {"value": 4}})
    _post(c, 9701, {"obs_key": k1, "reviewer": "Ana", "status": "rejected"})

    def rows(url):
        wb = openpyxl.load_workbook(io.BytesIO(c.get(url).content))
        head = [x.value for x in wb["HUMAN_COV_SUSC"][1]]
        return [{h: x.value for h, x in zip(head, r)} for r in wb["HUMAN_COV_SUSC"].iter_rows(min_row=2)]

    mine = [r for r in rows(f"/user-scenarios/{SID}/extraction/export") if r["ID"] == "LR9701"]
    assert [r["COVARIATE_hum"] for r in mine] == ["Male"]            # the rejected "female" row is out
    assert mine[0]["Value_hum"] == 4 and mine[0]["Review status"] == "edited" and mine[0]["Reviewed by"] == "Ana"
    withrej = [r for r in rows(f"/user-scenarios/{SID}/extraction/export?include_rejected=true") if r["ID"] == "LR9701"]
    assert sorted(r["Review status"] for r in withrej) == ["edited", "rejected"]
    csv_text = c.get(f"/user-scenarios/{SID}/extraction/export?format=csv").text
    assert "review_status" in csv_text.splitlines()[0] and ",edited,Ana," in csv_text


def test_the_annotated_dataset_holds_only_what_a_reviewer_stood_behind(clean_reviews):
    c = _client()
    k0, k1 = (o["obs_key"] for o in _obs(c, 9701))
    _post(c, 9701, {"obs_key": k0, "reviewer": "Ana", "status": "edited", "edits": {"value": 4}})
    _post(c, 9701, {"obs_key": k1, "reviewer": "Ana", "status": "rejected"})
    r = c.get(f"/user-scenarios/{SID}/extraction/dataset")
    assert r.status_code == 200 and r.headers["x-rows"] == "1"
    line = json.loads(r.text.strip())
    assert line["observation"]["value"] == 4 and line["observation"]["label_path"] == "sex_gender > male"
    assert line["review"] == {"status": "edited", "reviewers": ["Ana"]}
    assert line["quote"] and line["article"]["id"] == 9701 and "prompt_sha" in line["extraction"]
    everything = c.get(f"/user-scenarios/{SID}/extraction/dataset?include_unreviewed=true")
    assert int(everything.headers["x-rows"]) == 3                    # 1 reviewed + the 2 unreviewed rows of 9702 (the rejected one stays out)
