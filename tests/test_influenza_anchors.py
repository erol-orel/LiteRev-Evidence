"""The influenza anchors, exercised end to end — no database, no network, no API key.

`epidemic_parameters` is produced by an LLM call that no test can make, and every SEIR
curve in the app is drawn from it. These tests cover the half that IS reachable: given an
extraction shaped the way the prompt asks for, does the rest of the chain behave, and does
the checker actually catch the ways an extraction goes wrong?

The failure modes below are not hypothetical categories. Each is a specific thing an LLM
does: a case fatality ratio in percent, a period in hours, a provenance id for an article
that was never in the corpus, a point estimate with the per-study observations omitted.
"""
import math

import influenza_anchors as anchors
import seir_model as sm
from influenza_anchors import (ANCHORS, CORPUS_IDS, INFLUENZA_EXTRACTION, QUALITY_BY_ID,
                               check_extraction)

POPULATION = 1_000_000.0


def _normalised():
    return sm.normalize_extracted_parameters(
        INFLUENZA_EXTRACTION, set(CORPUS_IDS), QUALITY_BY_ID)


def _simulate(params: dict, **over):
    """Deterministic single run from a normalised `params` block (central values only)."""
    kw = {k: v["value"] for k, v in params.items() if k in sm._PARAM_FIELDS}
    kw.update(over)
    return sm.simulate(sm.SeirParams(population=POPULATION, initial_infected=10.0, **kw),
                       days=365)


# ── the fixture itself ───────────────────────────────────────────────────────
def test_every_anchor_band_contains_its_own_reference_value():
    """A typo in a band would silently disarm every check that uses it."""
    for name, a in ANCHORS.items():
        assert a.low < a.high, f"{name}: empty band"
        assert a.holds(a.typical), f"{name}: typical {a.typical} outside [{a.low}, {a.high}]"
        for factor, label in a.confusions:
            assert not a.holds(a.typical * factor), (
                f"{name}: the {label} check is useless — a correct value times {factor} "
                "still lands inside the band, so the confusion is undetectable")


def test_a_faithful_extraction_raises_no_complaint():
    assert check_extraction(INFLUENZA_EXTRACTION, CORPUS_IDS) == []


# ── the chain below the LLM ──────────────────────────────────────────────────
def test_quality_weighted_pooling_lands_inside_every_band():
    """Observations -> pool_weighted -> params. The reviews outweigh the small studies."""
    got = _normalised()
    assert got["applicable"] and got["disease"]
    for name, blk in got["params"].items():
        assert ANCHORS[name].holds(blk["value"]), (
            f"{name} pooled to {blk['value']}, outside {ANCHORS[name].source}")
        assert set(blk["provenance"]) <= set(CORPUS_IDS)

    # R0: the systematic review (weight 0.88, reporting 1.28 and 1.46) must pull the pool
    # BELOW the plain mean of the four observations, which the small outbreak study at
    # 2.10 would otherwise inflate.
    r0 = got["params"]["r0"]["value"]
    assert 1.2 < r0 < 1.8, r0
    assert r0 < (1.28 + 1.46 + 1.60 + 2.10) / 4.0
    # One paper reporting twice is still ONE study.
    assert got["params"]["r0"]["n_studies"] == 3


def test_a_single_study_parameter_keeps_the_models_own_estimate():
    """Under two studies the pool is refused, so the narrative value has to survive."""
    blk = _normalised()["params"]["immunity_duration_days"]
    assert blk["value"] == 365.0 and blk["n_studies"] == 1


def test_the_extraction_projects_a_recognisable_influenza_epidemic():
    res = _simulate(_normalised()["params"])
    s = res["summary"]
    assert s["model"] == "SEIRDS"                 # incubation + deaths + waning immunity
    assert s["r0_source"] == "literature", (
        "the whole chain exists to avoid presenting the hard-coded 2.5 as sourced")
    assert ANCHORS["r0"].holds(s["r0"])
    assert 0.3 < s["attack_rate"] < 1.0, s["attack_rate"]
    assert 0 < s["peak_incidence_day"] < 365


def test_deaths_track_the_extracted_case_fatality_ratio():
    """The invariant that makes a percent/proportion error impossible to miss."""
    params = _normalised()["params"]
    cfr = params["cfr"]["value"]
    res = _simulate(params)
    ratio = res["deaths"][-1] / res["cumulative"][-1]
    assert math.isclose(ratio, cfr, rel_tol=0.01), (ratio, cfr)

    # Same extraction, CFR read as a percentage: ~100x the deaths, on a curve that looks
    # every bit as plausible. Nothing in the engine can tell these two apart — only the
    # anchor can, which is why it exists.
    hundred_fold = _simulate(params, cfr=cfr * 100.0)
    assert hundred_fold["deaths"][-1] > 50 * res["deaths"][-1]


def test_the_uncertainty_ensemble_stays_finite_on_real_intervals():
    """Real confidence intervals are wide; no draw may diverge or be silently dropped."""
    dists = sm.params_to_distributions(_normalised()["params"])
    res = sm.simulate_ensemble({**dists, "population": POPULATION, "initial_infected": 10.0},
                               days=365, n_samples=200)
    assert res["n_dropped"] == 0, f"{res['n_dropped']} draws diverged on literature CIs"
    assert res["model"] == "SEIRDS"
    band = res["summary"]["r0"]
    assert band["lower"] <= band["median"] <= band["upper"]


def test_the_serial_interval_is_extracted_but_never_simulated():
    """It is normalised and displayed, yet absent from _PARAM_FIELDS.

    Pinned so nobody reads it off the SEIR tab believing it shaped the curve — and so that
    wiring it in later has to be a deliberate change to this test.
    """
    for name in anchors.NOT_SIMULATED:
        assert name in _normalised()["params"]
        assert name not in sm._PARAM_FIELDS


def test_epidemic_parameters_never_become_predictor_variables():
    """R0 as a feature is a constant column; incidence as a feature is legitimate."""
    for p in ("Basic reproduction number", "R0", "Taux de létalité (CFR)",
              "Incubation period", "Serial interval", "Durée de la période infectieuse"):
        assert sm.is_seir_parameter(p), p
        assert sm.seir_feature_column(p) is None, p
    for out, col in (("Weekly ILI incidence", "seir_incidence"),
                     ("Prévalence des cas actifs", "seir_prevalence"),
                     ("Cumulative infections", "seir_cumulative")):
        assert not sm.is_seir_parameter(out)
        assert sm.seir_feature_column(out) == col
    for neither in ("Mean temperature", "Vaccination coverage", "Hospital bed occupancy"):
        assert not sm.is_seir_parameter(neither)
        assert sm.seir_feature_column(neither) is None


# ── what the checker must catch ──────────────────────────────────────────────
def _mutate(param: str, **fields) -> dict:
    out = {k: (dict(v) if isinstance(v, dict) else v)
           for k, v in INFLUENZA_EXTRACTION.items()}
    out[param] = {**out[param], **fields}
    return out


def test_a_case_fatality_ratio_in_percent_is_caught():
    bad = _mutate("cfr", value=2.0, ci_low=1.0, ci_high=3.0, unit="proportion")
    assert any("PERCENTAGE" in c for c in check_extraction(bad, CORPUS_IDS))


def test_a_case_fatality_ratio_honestly_labelled_percent_is_not_flagged():
    """`_is_percent_unit` converts it downstream, so the checker must not cry wolf."""
    ok = _mutate("cfr", value=0.02, ci_low=0.01, ci_high=0.09, unit="%")
    assert [c for c in check_extraction(ok, CORPUS_IDS) if "cfr" in c] == []
    converted = sm.normalize_extracted_parameters(ok, set(CORPUS_IDS))["params"]["cfr"]
    assert math.isclose(converted["value"], 0.0002)


def test_a_period_given_in_hours_is_caught_and_named():
    bad = _mutate("incubation_period_days", value=34.0, ci_low=31.0, ci_high=36.0)
    hit = [c for c in check_extraction(bad, CORPUS_IDS) if "incubation" in c]
    assert hit and "hours" in hit[0], hit


def test_a_hallucinated_magnitude_is_caught():
    bad = _mutate("r0", value=15.0, ci_low=12.0, ci_high=18.0)
    assert any("outside the plausible band" in c and "r0" in c
               for c in check_extraction(bad, CORPUS_IDS))


def test_fabricated_provenance_is_caught():
    """The id is dropped by _clean_provenance, so the code itself never complains."""
    bad = _mutate("r0", provenance=[101, 999])
    assert any("999" in c and "no article in the corpus" in c
               for c in check_extraction(bad, CORPUS_IDS))
    kept = sm.normalize_extracted_parameters(bad, set(CORPUS_IDS))["params"]["r0"]
    assert 999 not in kept["provenance"]


def test_an_observation_attributed_outside_the_corpus_is_caught():
    bad = _mutate("r0", observations=[{"article_id": 777, "value": 1.3}])
    assert any("777" in c for c in check_extraction(bad, CORPUS_IDS))


def test_a_point_estimate_with_no_observations_is_caught():
    """Without observations the model's own averaging stands, unweighted by quality."""
    bad = _mutate("infectious_period_days", observations=[])
    assert any("observations" in c and "infectious_period_days" in c
               for c in check_extraction(bad, CORPUS_IDS))


def test_a_missing_r0_is_reported_as_fatal_to_the_projection():
    """Gate 2 refuses to serve a curve without it; the checker must say so plainly."""
    bad = {k: v for k, v in INFLUENZA_EXTRACTION.items() if k != "r0"}
    hit = [c for c in check_extraction(bad, CORPUS_IDS) if c.startswith("r0")]
    assert hit and "no projection is served" in hit[0]

    dists = sm.params_to_distributions(
        sm.normalize_extracted_parameters(bad, set(CORPUS_IDS), QUALITY_BY_ID)["params"])
    assert not ({"r0", "beta"} & set(dists))          # exactly what gate 2 tests


def test_an_inverted_confidence_interval_is_caught():
    bad = _mutate("r0", ci_low=2.5, ci_high=1.1)
    assert any("ci_low" in c for c in check_extraction(bad, CORPUS_IDS))
    kept = sm.normalize_extracted_parameters(bad, set(CORPUS_IDS))["params"]["r0"]
    assert kept["ci_low"] is None and kept["ci_high"] is None


def test_applicable_false_is_reported_as_gating_everything_off():
    bad = {**INFLUENZA_EXTRACTION, "applicable": False}
    assert any("applicable is false" in c for c in check_extraction(bad, CORPUS_IDS))


def test_the_checker_says_what_it_cannot_check_on_a_normalised_block():
    """The normalised block has no observations left; silence there would be a lie."""
    out = check_extraction(_normalised(), CORPUS_IDS)
    assert any(c.startswith("NOTE:") for c in out)
    assert not any("observations" in c for c in out if not c.startswith("NOTE:"))
    # band checks still ran on the real shape
    assert not any("outside the plausible band" in c for c in out)


def test_provenance_is_only_judged_when_the_corpus_is_known():
    """Without corpus_ids a fabricated id is invisible here — as it is to the code."""
    bad = _mutate("r0", provenance=[101, 999])
    assert not any("999" in c for c in check_extraction(bad))
