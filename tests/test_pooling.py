"""Pooled estimates (api/pooling.py).

The statistics are pinned to values computed with `statsmodels` and `scipy` (the BCG vaccine
trials are the usual benchmark for a random-effects model), and cross-checked live against them
where they are installed. The module itself imports neither. The integration tests check what
decides whether a pooled number can be trusted: what is counted, what is left out and why, and
that nothing is pooled across diseases unless asked."""
import json
import math

import pytest

pytest.importorskip("fastapi")

import main  # noqa: E402
from api import pooling as P  # noqa: E402
from test_extraction import SID, extracted, seeded  # noqa: E402,F401

# 13 trials: cases and non-cases in the vaccinated, then in the controls.
BCG = [(4, 119, 11, 128), (6, 300, 29, 274), (3, 228, 11, 209), (62, 13536, 248, 12619), (33, 5036, 47, 5761),
       (180, 1361, 372, 1079), (8, 2537, 10, 619), (505, 87886, 499, 87892), (29, 7470, 45, 7232),
       (17, 1699, 65, 1600), (186, 50448, 141, 27197), (5, 2493, 3, 2338), (27, 16886, 29, 17825)]


def _bcg_effects():
    return zip(*(P.log_odds_ratio(a, a + b, c, c + d) for a, b, c, d in BCG))


# ── The statistics, against reference values ────────────────────────────────
def test_random_effects_pooling_matches_statsmodels_on_the_bcg_trials():
    ys, vs = _bcg_effects()
    r = P.pool_effects(list(ys), list(vs))
    assert r["k"] == 13 and r["df"] == 12
    assert r["mu"] == pytest.approx(-0.74739, abs=1e-4)
    assert r["tau2"] == pytest.approx(0.36634, abs=1e-4)
    assert r["Q"] == pytest.approx(163.1649, abs=1e-3)
    assert r["I2"] == pytest.approx(92.645, abs=1e-2)
    assert r["p_Q"] < 1e-20
    # Hartung-Knapp interval on 12 degrees of freedom, and the prediction interval on 11.
    assert r["ci"][0] == pytest.approx(-1.16629, abs=1e-4) and r["ci"][1] == pytest.approx(-0.32849, abs=1e-4)
    assert r["pi"][0] == pytest.approx(-2.1452, abs=1e-3) and r["pi"][1] == pytest.approx(0.6504, abs=1e-3)
    assert sum(r["weights_pct"]) == pytest.approx(100.0)


def test_pooled_proportions_match_statsmodels():
    res = P.pool_proportions([dict(x=3, n=10), dict(x=12, n=40), dict(x=5, n=60), dict(x=0, n=15), dict(x=22, n=50)])
    assert res["pooled"]["p"] == pytest.approx(0.2236, abs=1e-4)
    assert res["pooled"]["ci_low"] == pytest.approx(0.0665, abs=1e-3) and res["pooled"]["ci_high"] == pytest.approx(0.5383, abs=1e-3)
    assert res["heterogeneity"]["tau2"] == pytest.approx(0.7770, abs=1e-3) and res["heterogeneity"]["I2"] == pytest.approx(78.7, abs=0.1)
    assert res["heterogeneity"]["band"] == "considerable"
    assert sum(s["weight_pct"] for s in res["studies"]) == pytest.approx(100.0)
    assert 0 < res["pooled"]["pi_low"] < res["pooled"]["ci_low"] < res["pooled"]["ci_high"] < res["pooled"]["pi_high"] < 1


def test_the_distribution_helpers_agree_with_scipy():
    stats = pytest.importorskip("scipy.stats")
    assert max(abs(P.t975(d) - stats.t.ppf(0.975, d)) for d in range(1, 400)) < 2e-4
    for df in (1, 2, 5, 12, 40):
        for x in (0.3, 2, 9, 30, 90):
            assert P.chi2_sf(x, df) == pytest.approx(stats.chi2.sf(x, df), abs=1e-10)


def test_wilson_and_the_pooling_agree_with_statsmodels_live():
    pytest.importorskip("statsmodels")
    np = pytest.importorskip("numpy")
    from statsmodels.stats.meta_analysis import combine_effects
    from statsmodels.stats.proportion import proportion_confint
    for n in (1, 5, 17, 200):
        for x in sorted({0, 1, n // 2, n}):
            lo, hi = P.wilson(x, n)
            ref = proportion_confint(x, n, alpha=0.05, method="wilson")
            assert (lo, hi) == pytest.approx(ref, abs=1e-7)
    ys, vs = _bcg_effects()
    sm = combine_effects(np.array(ys), np.array(vs), method_re="chi2", use_t=False)
    r = P.pool_effects(list(ys), list(vs))
    assert r["mu"] == pytest.approx(sm.mean_effect_re) and r["tau2"] == pytest.approx(sm.tau2) and r["Q"] == pytest.approx(sm.q)


# ── Edge cases that go wrong in a naive version ─────────────────────────────
def test_identical_studies_have_no_heterogeneity_and_the_interval_is_never_narrower_than_dl():
    r = P.pool_effects([0.5, 0.5, 0.5], [0.1, 0.1, 0.1])
    assert r["tau2"] == 0 and r["I2"] == 0 and r["Q"] == pytest.approx(0)
    lo, hi = r["ci"]
    assert hi - lo >= 2 * 1.959 * r["se_dl"] - 1e-9            # the max(1, q) safeguard


def test_two_studies_pool_but_have_no_prediction_interval():
    r = P.pool_effects([0.1, 0.9], [0.05, 0.05])
    assert r["k"] == 2 and r["pi"] is None
    with pytest.raises(ValueError):
        P.pool_effects([0.1], [0.05])
    with pytest.raises(ValueError):
        P.pool_effects([0.1, 0.2], [0.05, 0.0])


def test_a_study_with_no_cases_or_all_cases_stays_finite():
    for x, n in ((0, 20), (20, 20)):
        y, v = P.logit_proportion(x, n)
        assert math.isfinite(y) and math.isfinite(v) and v > 0
    y, v = P.log_odds_ratio(0, 10, 4, 12)
    assert math.isfinite(y) and v > 0
    # An ordinary study is not corrected.
    assert P.logit_proportion(5, 20)[0] == pytest.approx(math.log(5 / 15))


def test_wilson_stays_inside_zero_and_one_at_the_edges():
    assert P.wilson(0, 10)[0] == 0 and 0 < P.wilson(0, 10)[1] < 0.35
    assert P.wilson(10, 10)[1] == 1 and 0.65 < P.wilson(10, 10)[0] < 1


def test_heterogeneity_in_words():
    assert [P.heterogeneity_band(x) for x in (0, 24.9, 25, 49, 50, 74, 75, 99)] == [
        "low", "low", "moderate", "moderate", "substantial", "substantial", "considerable", "considerable"]


# ── Which rows may be counted ───────────────────────────────────────────────
def test_only_whole_counts_with_cases_not_above_the_population_are_usable():
    assert P.row_counts({"n_cases": 3, "pop_risk": 10}) == (3, 10)
    assert P.row_counts({"n_cases": 0, "pop_risk": 7}) == (0, 7)
    assert P.row_counts({"n_cases": "4", "pop_risk": 9.0}) == (4, 9)
    assert P.row_counts({"n_cases": None, "pop_risk": 10}) == "missing"
    assert P.row_counts({"n_cases": 3}) == "missing"
    for bad in ({"n_cases": 11, "pop_risk": 10}, {"n_cases": 3.5, "pop_risk": 10}, {"n_cases": -1, "pop_risk": 10},
                {"n_cases": 0, "pop_risk": 0}, {"n_cases": True, "pop_risk": 10}, {"n_cases": "x", "pop_risk": 10}):
        assert P.row_counts(bad) == "invalid", bad


def _row(aid, x, n, label="male", disease="HPAI", group="sex_gender", sheet="human_susc", status="accepted"):
    return {"article_id": aid, "title": f"Paper {aid}", "year": 2020 + aid % 5, "first_author": f"A{aid}", "x": x, "n": n,
            "review_status": status, "obs": {"sheet": sheet, "group_key": group, "covariate_key": label, "disease": disease,
                                             "label_path": f"{group} > {label}", "matched": True}}


def test_one_article_gives_one_row_per_label_and_the_largest_population_is_kept():
    rows = [_row(1, 3, 10), _row(1, 6, 25), _row(2, 4, 12)]
    groups, dropped = P.build_groups(rows)
    (only,) = groups.values()
    assert dropped == 1 and {r["article_id"]: r["n"] for r in only} == {1: 25, 2: 12}


def test_diseases_are_not_pooled_together_unless_asked():
    rows = [_row(i, 2 + i, 20 + i, disease="HPAI") for i in range(1, 4)] + [_row(i, 1 + i, 30 + i, disease="COVID-19") for i in range(4, 7)]
    split = P.pooled_groups(rows)
    assert sorted((g["disease"], g["k"]) for g in split) == [("covid 19", 3), ("hpai", 3)]
    merged = P.pooled_groups(rows, split_disease=False)
    assert len(merged) == 1 and merged[0]["k"] == 6 and merged[0]["disease"] is None


def test_fewer_than_three_studies_get_no_pooled_estimate_but_keep_their_studies():
    g = P.pooled_groups([_row(1, 3, 10), _row(2, 5, 30)])[0]
    assert g["pooled"] is None and g["heterogeneity"] is None and g["reason"] == "too_few_studies"
    assert [round(s["p"], 2) for s in g["studies"]] == [0.3, 0.17] and all(0 <= s["ci_low"] <= s["ci_high"] <= 1 for s in g["studies"])
    assert (g["k"], g["n_total"], g["events_total"]) == (2, 40, 8)
    assert P.pooled_groups([_row(i, 2, 20 + i) for i in range(1, 4)], min_studies=4)[0]["pooled"] is None
    assert P.pooled_groups([_row(i, 2, 20 + i) for i in range(1, 4)], min_studies=3)[0]["pooled"] is not None


def test_comparisons_use_only_papers_reporting_both_labels_and_follow_the_codebook_order():
    rows = []
    for i in range(1, 5):
        rows += [_row(i, 4 + i, 30), _row(i, 3 + i, 32, label="female")]
    rows.append(_row(9, 2, 10))                                       # male only: not in the comparison
    order = {("human_susc", "sex_gender", "male"): 0, ("human_susc", "sex_gender", "female"): 1}
    (cmp_,) = P.pooled_comparisons(rows, order)
    assert (cmp_["a"], cmp_["b"], cmp_["k"]) == ("male", "female", 4)
    assert cmp_["pooled"]["or"] > 1 and cmp_["pooled"]["ci_low"] < cmp_["pooled"]["or"] < cmp_["pooled"]["ci_high"]
    # The codebook order decides the direction, not the alphabet.
    (rev,) = P.pooled_comparisons(rows, {("human_susc", "sex_gender", "female"): 0, ("human_susc", "sex_gender", "male"): 1})
    assert rev["a"] == "female" and rev["pooled"]["or"] == pytest.approx(1 / cmp_["pooled"]["or"], rel=0.02)
    assert P.pooled_comparisons(rows[:4], order) == []                # two papers: nothing to pool


# ── Rows that give the composition of a sample, not groups with cases ───────
# The male against female rows of avian influenza papers in HPAI_last (Whitlow 2025, Charania
# 2014, Rehman 2023, Hinjoy 2023): each pair is one sample split by sex. The fifth is two people
# short of its total, as when a few people's sex was not recorded.
SPLIT = [(142, 81, 225), (80, 26, 106), (119, 81, 200), (162, 184, 346), (110, 118, 230)]
SEX = {("human_susc", "sex_gender", "male"): 0, ("human_susc", "sex_gender", "female"): 1}


def test_complementary_shares_of_one_sample_are_composition():
    assert all(P.is_composition_pair(men, n, women, n) for men, women, n in SPLIT)
    # Within 2 people or 2% of the total, whichever is larger, on either side of it.
    assert P.is_composition_pair(110, 225, 111, 225) and not P.is_composition_pair(110, 225, 110, 225)
    assert P.is_composition_pair(24, 50, 24, 50) and not P.is_composition_pair(24, 50, 23, 50)
    assert P.is_composition_pair(30, 50, 22, 50) and not P.is_composition_pair(30, 50, 23, 50)
    assert P.is_composition_pair(0, 12, 12, 12)                      # everyone studied was a woman
    # Never more than 5% of the total: in a small sample the counts must add up (nearly) exactly.
    assert P.is_composition_pair(9, 30, 20, 30) and not P.is_composition_pair(9, 30, 19, 30)
    assert P.is_composition_pair(3, 8, 5, 8) and not P.is_composition_pair(2, 4, 1, 4)
    assert not P.is_composition_pair(1, 1, 1, 1)                     # one cat and one dog, both positive


def test_a_real_two_group_contrast_is_not_composition():
    # Each group has its own size: 12 cases among 40 men, 10 among 45 women.
    assert not P.is_composition_pair(12, 40, 10, 45)
    # Counts that add up to a total, but not the same total: two groups, whatever their sum.
    assert not P.is_composition_pair(142, 225, 81, 223)
    # The same size by design (two arms of 100), with counts well short of it.
    assert not P.is_composition_pair(5, 100, 20, 100)


def test_three_labels_can_split_one_sample_where_no_two_of_them_do():
    ages = {"child": 40, "adult": 160, "elderly": 25}
    assert P.composition_labels(ages, 225) == {"child", "adult", "elderly"}
    assert not any(P.is_composition_pair(ages[a], 225, ages[b], 225) for a in ages for b in ages if a < b)
    # A label that breaks the sum does not hide the pair that splits the total.
    assert P.composition_labels({"male": 142, "female": 81, "non_binary": 30}, 225) == {"male", "female"}
    assert P.composition_labels({"male": 142}, 225) == set()
    assert P.composition_labels({"child": 4, "adult": 16}, 225) == set()


def test_pairs_that_split_one_sample_give_no_odds_ratio_and_are_counted():
    rows = []
    for aid, (men, women, n) in enumerate(SPLIT, 1):
        rows += [_row(aid, men, n), _row(aid, women, n, label="female")]
    assert P.pooled_comparisons(rows, SEX) == []
    groups, _ = P.build_groups(rows)
    assert P.composition_pairs(groups) == 5
    # The proportions keep the rows and mark them: a "male 63%" here is the share of men studied.
    male = next(g for g in P.pooled_groups(rows) if g["label"] == "male")
    assert (male["k"], male["k_composition"]) == (5, 5) and male["pooled"] is not None
    assert all(s["composition"] for s in male["studies"])


def test_a_comparison_keeps_its_real_contrasts_and_says_how_many_papers_split_one_sample():
    rows = []
    for aid in range(1, 5):                                           # four papers with two groups and their cases
        rows += [_row(aid, 4 + aid, 30), _row(aid, 3 + aid, 32, label="female")]
    for aid, (men, women, n) in enumerate(SPLIT[:2], 11):             # two that split one sample
        rows += [_row(aid, men, n), _row(aid, women, n, label="female")]
    (cmp_,) = P.pooled_comparisons(rows, SEX)
    assert (cmp_["k"], cmp_["excluded_composition"]) == (4, 2)
    assert {s["article_id"] for s in cmp_["studies"]} == {1, 2, 3, 4}
    ref = P.pool_odds_ratios([dict(x1=4 + i, n1=30, x2=3 + i, n2=32) for i in range(1, 5)])
    assert cmp_["pooled"]["or"] == pytest.approx(ref["pooled"]["or"])
    assert P.composition_pairs(P.build_groups(rows)[0]) == 2
    # Every pair that holds a row giving the composition is left out, not only the pair that splits:
    # three age bands of one sample give three pairs.
    ages = [_row(21, 40, 225, label="child", group="age"), _row(21, 160, 225, label="adult", group="age"),
            _row(21, 25, 225, label="elderly", group="age")]
    assert P.composition_pairs(P.build_groups(ages)[0]) == 3


def test_a_paper_s_count_of_cases_is_kept_over_its_table_of_who_was_studied():
    # The paper gives who was studied (142 men and 81 women among 225) AND the cases among the men
    # and among the women. The table of who was studied has the larger denominator; it used to win.
    rows = [_row(1, 142, 225), _row(1, 81, 225, label="female"), _row(1, 3, 142), _row(1, 2, 81, label="female")]
    for aid in range(2, 4):
        rows += [_row(aid, 4 + aid, 30), _row(aid, 3 + aid, 32, label="female")]
    groups, dropped = P.build_groups(rows)
    kept = {k[2]: next(r for r in items if r["article_id"] == 1) for k, items in groups.items()}
    assert dropped == 2
    assert (kept["male"]["x"], kept["male"]["n"], kept["male"]["composition"]) == (3, 142, False)
    assert (kept["female"]["x"], kept["female"]["n"], kept["female"]["composition"]) == (2, 81, False)
    (cmp_,) = P.pooled_comparisons(rows, SEX)
    assert (cmp_["k"], cmp_["excluded_composition"]) == (3, 0)
    assert next(s for s in cmp_["studies"] if s["article_id"] == 1)["x1"] == 3
    # Where the paper gives nothing else, the row that splits the sample is the one there is.
    alone = [_row(1, 142, 225), _row(1, 81, 225, label="female"), _row(1, 3, 142)]
    female = next(r for k, items in P.build_groups(alone)[0].items() if k[2] == "female" for r in items)
    assert (female["n"], female["composition"]) == (225, True)


# ── Endpoint ────────────────────────────────────────────────────────────────
MALE = [(12, 40), (5, 30), (20, 60), (8, 25), (15, 50), (9, 35)]
FEMALE = [(10, 45), (6, 38), (14, 55), (7, 30), (11, 48), (8, 40)]
EXTRA = list(range(9711, 9717))


def _doc(i, male, female, verified=True, extra_rows=()):
    obs = []
    for lab, (x, n) in (("Males", male), ("Females", female)):
        obs.append({"sheet": "human_susc", "group": "sex", "covariate": lab, "value": None, "n_cases": x, "pop_risk": n,
                    "disease": "HPAI A(H5N1)", "quote": f"{lab} {x} of {n} study {i}", "quote_verified": verified})
    return json.dumps({"v": 1, "source": "fulltext", "coverage": {}, "observations": obs + list(extra_rows)})


@pytest.fixture()
def pool_corpus(extracted):
    with extracted.cursor() as cur:
        for k, i in enumerate(EXTRA):
            cur.execute("INSERT INTO literature_document (id, title, abstract, authors, year, source, is_duplicate, project_context, "
                        "extraction_json) VALUES (%s,%s,'a','Smith J; Doe A',2021,'pubmed',false,'literev',%s::jsonb)",
                        (i, f"Pool paper {i}", _doc(i, MALE[k], FEMALE[k])))
            cur.execute("INSERT INTO article_scenarios (scenario_id, document_id, similarity_score) VALUES (%s,%s,0.9)", (SID, i))
        cur.execute("DELETE FROM extraction_review WHERE document_id BETWEEN 9701 AND 9720")
    yield extracted
    with extracted.cursor() as cur:
        cur.execute("DELETE FROM extraction_review WHERE document_id BETWEEN 9701 AND 9720")
        cur.execute("DELETE FROM article_scenarios WHERE document_id = ANY(%s)", (EXTRA,))
        cur.execute("DELETE FROM literature_document WHERE id = ANY(%s)", (EXTRA,))


def _get(path="", **params):
    from fastapi.testclient import TestClient
    return TestClient(main.app).get(f"/user-scenarios/{SID}/extraction/pooled{path}", params=params)


def _group(body, label, disease):
    return next((g for g in body["pooled"] if g["label"] == label and g["disease"] == disease), None)


def test_the_endpoint_pools_the_whole_relevant_corpus_within_a_disease(pool_corpus):
    body = _get().json()
    male = _group(body, "male", "hpai a h5n1")
    assert male["k"] == 6 and male["mapped"] and male["label_path"] == "sex_gender > male"
    assert male["events_total"] == sum(x for x, _ in MALE) and male["n_total"] == sum(n for _, n in MALE)
    ref = P.pool_proportions([dict(x=x, n=n) for x, n in MALE])
    assert male["pooled"]["p"] == pytest.approx(ref["pooled"]["p"]) and male["heterogeneity"]["I2"] == ref["heterogeneity"]["I2"]
    assert [s["first_author"] for s in male["studies"]] == ["Smith"] * 6
    # The two seeded papers have no disease stated: a group of their own, too small to pool.
    stated = _group(body, "male", "(not stated)")
    assert stated["k"] == 2 and stated["pooled"] is None and stated["reason"] == "too_few_studies"
    cmp_ = next(c for c in body["comparisons"] if c["disease"] == "hpai a h5n1")
    assert (cmp_["a"], cmp_["b"], cmp_["k"]) == ("male", "female", 6)
    assert body["filters"] == {"reviewed_only": False, "verified_only": True, "split_disease": True, "min_studies": 3}


def test_diseases_merge_only_when_asked_and_the_filters_are_applied(pool_corpus):
    merged = _get(split_disease="false").json()
    assert next(g for g in merged["pooled"] if g["label"] == "male")["k"] == 8
    assert _get(min_studies=7).json()["comparisons"] == []
    assert _get(min_studies=1).status_code == 422


def test_unverified_quotes_are_left_out_unless_asked_and_counted(pool_corpus):
    with pool_corpus.cursor() as cur:
        cur.execute("UPDATE literature_document SET extraction_json = %s::jsonb WHERE id = 9711", (_doc(9711, MALE[0], FEMALE[0], verified=False),))
    body = _get().json()
    assert _group(body, "male", "hpai a h5n1")["k"] == 5
    # The male and female rows of 9711 (2), plus the seeded paper's "female" row that never had its quote found.
    assert body["excluded"]["quote_not_found"] == 3
    assert _group(_get(verified_only="false").json(), "male", "hpai a h5n1")["k"] == 6


def test_reviews_decide_what_counts(pool_corpus):
    from fastapi.testclient import TestClient
    from test_extraction import HDR
    c = TestClient(main.app)
    obs = c.get(f"/user-scenarios/{SID}/articles/9712/extraction").json()["extraction"]["observations"]
    male_key = next(o["obs_key"] for o in obs if o["covariate"] == "Males")
    post = lambda who, status, **kw: c.post(f"/user-scenarios/{SID}/articles/9712/extraction/review", headers=HDR,
                                            json={"obs_key": male_key, "reviewer": who, "status": status, **kw})
    # A correction replaces the model's count in the pool.
    post("Ana", "edited", edits={"n_cases": 6, "pop_risk": 32})
    body = _get().json()
    s = next(s for s in _group(body, "male", "hpai a h5n1")["studies"] if s["article_id"] == 9712)
    assert (s["x"], s["n"], s["review_status"]) == (6, 32, "edited")
    # A rejection removes the row; a disagreement removes it too and says so.
    post("Ana", "rejected")
    body = _get().json()
    assert _group(body, "male", "hpai a h5n1")["k"] == 5 and body["excluded"]["rejected"] == 1
    post("Ben", "accepted")
    body = _get().json()
    assert _group(body, "male", "hpai a h5n1")["k"] == 5 and body["excluded"]["conflict"] == 1
    # reviewed_only keeps just the rows a person stood behind.
    post("Ben", "clear"); post("Ana", "accepted")
    only = _get(reviewed_only="true").json()
    assert _group(only, "male", "hpai a h5n1")["k"] == 1 and only["excluded"]["not_reviewed"] > 0


def test_a_duplicate_row_in_one_article_is_counted_once_and_reported(pool_corpus):
    dup = {"sheet": "human_susc", "group": "sex", "covariate": "Men", "n_cases": 20, "pop_risk": 90, "disease": "HPAI A(H5N1)",
           "quote": "Men 20 of 90 in the pooled cohort", "quote_verified": True}
    with pool_corpus.cursor() as cur:
        cur.execute("UPDATE literature_document SET extraction_json = %s::jsonb WHERE id = 9713", (_doc(9713, MALE[2], FEMALE[2], extra_rows=[dup]),))
    body = _get().json()
    s = next(s for s in _group(body, "male", "hpai a h5n1")["studies"] if s["article_id"] == 9713)
    assert s["n"] == 90 and body["n_duplicate_rows_dropped"] == 1           # the larger population is kept


def test_invalid_and_missing_counts_are_reported_not_silently_dropped(pool_corpus):
    bad = [{"sheet": "human_susc", "group": "sex", "covariate": "Girls", "n_cases": 30, "pop_risk": 10, "disease": "HPAI",
            "quote": "Girls 30 of 10", "quote_verified": True},
           {"sheet": "human_susc", "group": "sex", "covariate": "Boys", "n_cases": None, "pop_risk": 10, "disease": "HPAI",
            "quote": "Boys 10 persons", "quote_verified": True}]
    with pool_corpus.cursor() as cur:
        cur.execute("UPDATE literature_document SET extraction_json = %s::jsonb WHERE id = 9714", (_doc(9714, MALE[3], FEMALE[3], extra_rows=bad),))
    ex = _get().json()["excluded"]
    assert ex["invalid_counts"] >= 1 and ex["missing_counts"] >= 1


def test_unknown_scenario_is_a_404():
    from fastapi.testclient import TestClient
    assert TestClient(main.app).get("/user-scenarios/nope/extraction/pooled").status_code == 404


SPLIT_IDS = list(range(9721, 9721 + len(SPLIT)))


@pytest.fixture()
def composition_corpus(extracted):
    """Five avian influenza papers whose male and female rows split one sample (SPLIT)."""
    with extracted.cursor() as cur:
        for i, (men, women, n) in zip(SPLIT_IDS, SPLIT):
            cur.execute("INSERT INTO literature_document (id, title, abstract, authors, year, source, is_duplicate, project_context, "
                        "extraction_json) VALUES (%s,%s,'a','Whitlow K; Doe A',2025,'pubmed',false,'literev',%s::jsonb)",
                        (i, f"Split paper {i}", _doc(i, (men, n), (women, n))))
            cur.execute("INSERT INTO article_scenarios (scenario_id, document_id, similarity_score) VALUES (%s,%s,0.9)", (SID, i))
    yield extracted
    with extracted.cursor() as cur:
        cur.execute("DELETE FROM extraction_review WHERE document_id = ANY(%s)", (SPLIT_IDS,))
        cur.execute("DELETE FROM article_scenarios WHERE document_id = ANY(%s)", (SPLIT_IDS,))
        cur.execute("DELETE FROM literature_document WHERE id = ANY(%s)", (SPLIT_IDS,))


def test_the_hpai_pairs_that_split_one_sample_no_longer_give_an_odds_ratio(composition_corpus):
    body = _get().json()
    assert [c for c in body["comparisons"] if c["disease"] == "hpai a h5n1"] == []
    # Counted in what was left out, as pairs: the rows themselves still count in the proportions.
    assert body["excluded"]["composition"] == 5
    male = _group(body, "male", "hpai a h5n1")
    assert (male["k"], male["k_composition"]) == (5, 5) and all(s["composition"] for s in male["studies"])
    assert male["events_total"] == sum(men for men, _, _ in SPLIT) and body["n_rows_used"] >= 10


def test_a_comparison_beside_them_keeps_its_real_contrasts_and_counts_what_it_left_out(pool_corpus, composition_corpus):
    body = _get().json()
    cmp_ = next(c for c in body["comparisons"] if c["disease"] == "hpai a h5n1")
    assert (cmp_["a"], cmp_["b"], cmp_["k"], cmp_["excluded_composition"]) == ("male", "female", 6, 5)
    assert sorted(s["article_id"] for s in cmp_["studies"]) == EXTRA
    ref = P.pool_odds_ratios([dict(x1=m[0], n1=m[1], x2=f[0], n2=f[1]) for m, f in zip(MALE, FEMALE)])
    assert cmp_["pooled"]["or"] == pytest.approx(ref["pooled"]["or"])
    male = _group(body, "male", "hpai a h5n1")
    assert (male["k"], male["k_composition"]) == (11, 5)
    assert body["excluded"]["composition"] == 5
