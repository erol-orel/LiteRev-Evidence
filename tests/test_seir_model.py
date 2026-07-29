"""Pure-logic tests for seir_model.py — no database, no network, stdlib only.

The strongest correctness check is the SIR **final-size relation**: for a closed
epidemic the attack rate z satisfies z = 1 - exp(-R0 * z). We also cover model
auto-selection, the SEIR latent-period delay, SEIRD mortality accounting, and the
uncertainty ensemble (band ordering + reproducibility).
"""
import math

import seir_model as sm


def _final_size(r0: float) -> float:
    z = 0.5
    for _ in range(500):
        z = 1.0 - math.exp(-r0 * z)
    return z


# ── deterministic dynamics ───────────────────────────────────────────────────
def test_sir_final_size_relation():
    # Attack rate must match the analytic final size to within 2 points.
    for r0 in (1.5, 2.5, 3.5):
        res = sm.simulate(
            sm.SeirParams(r0=r0, infectious_period_days=6,
                          population=1_000_000, initial_infected=10),
            days=500,
        )
        assert abs(res["summary"]["attack_rate"] - _final_size(r0)) < 0.02


def test_subcritical_no_epidemic():
    # R0 < 1 → the outbreak dies out: negligible attack rate, incidence never grows.
    res = sm.simulate(
        sm.SeirParams(r0=0.8, infectious_period_days=6,
                      population=1_000_000, initial_infected=100),
        days=200,
    )
    assert res["summary"]["attack_rate"] < 0.02
    assert res["incidence"][0] >= max(res["incidence"])


def test_conservation_of_population():
    # incidence, prevalence, cumulative, deaths stay finite & non-negative; the
    # cumulative curve is monotincreasing and bounded by N.
    res = sm.simulate(
        sm.SeirParams(r0=2.5, infectious_period_days=6, cfr=0.02,
                      population=500_000, initial_infected=10),
        days=400,
    )
    n = 500_000
    assert all(v >= -1e-6 for v in res["prevalence"])
    assert all(res["cumulative"][k] <= res["cumulative"][k + 1] + 1e-6
               for k in range(len(res["cumulative"]) - 1))
    assert res["cumulative"][-1] <= n + 1.0


def test_r_eff_starts_near_r0_and_falls():
    res = sm.simulate(
        sm.SeirParams(r0=3.0, infectious_period_days=5,
                      population=1_000_000, initial_infected=10),
        days=400,
    )
    assert abs(res["r_eff"][0] - 3.0) < 0.01          # R_eff(0) ≈ R0 (S/N ≈ 1)
    assert res["r_eff"][-1] < res["r_eff"][0]         # depletion of susceptibles


# ── model auto-selection ─────────────────────────────────────────────────────
def test_model_auto_selection():
    cases = [
        (dict(r0=2), "SIR"),
        (dict(r0=2, incubation_period_days=3), "SEIR"),
        (dict(r0=2, incubation_period_days=3, cfr=0.02), "SEIRD"),
        (dict(r0=2, incubation_period_days=3, immunity_duration_days=180), "SEIRS"),
        (dict(r0=2, incubation_period_days=3, cfr=0.02, immunity_duration_days=180), "SEIRDS"),
        (dict(r0=2, cfr=0.01), "SIRD"),
    ]
    for kw, expected in cases:
        res = sm.simulate(sm.SeirParams(population=1e6, initial_infected=10, **kw), days=50)
        assert res["model"] == expected, f"{kw} → {res['model']} (expected {expected})"


def test_beta_derived_from_r0_and_direct_beta_equivalent():
    # Supplying beta directly must equal supplying the matching R0 = beta / gamma.
    gamma = 1.0 / 6.0
    a = sm.simulate(sm.SeirParams(r0=2.4, infectious_period_days=6, population=1e6, initial_infected=10), days=300)
    b = sm.simulate(sm.SeirParams(beta=2.4 * gamma, infectious_period_days=6, population=1e6, initial_infected=10), days=300)
    assert abs(a["summary"]["attack_rate"] - b["summary"]["attack_rate"]) < 1e-3


def test_missing_r0_and_beta_uses_fallback():
    # Neither β nor R0 → prudent R0=2.5 fallback (still a real epidemic, no crash).
    res = sm.simulate(sm.SeirParams(infectious_period_days=6, population=1e6, initial_infected=10), days=300)
    assert abs(res["summary"]["r0"] - 2.5) < 1e-6
    assert res["summary"]["attack_rate"] > 0.5


# ── epidemiological structure ────────────────────────────────────────────────
def test_seir_peak_later_than_sir_same_final_size():
    sir = sm.simulate(sm.SeirParams(r0=2.5, infectious_period_days=6, population=1e6, initial_infected=10), days=500)
    seir = sm.simulate(sm.SeirParams(r0=2.5, infectious_period_days=6, incubation_period_days=4,
                                     population=1e6, initial_infected=10), days=500)
    assert seir["summary"]["peak_prevalence_day"] > sir["summary"]["peak_prevalence_day"]
    # a latent compartment reshapes timing, not the closed-epidemic final size
    assert abs(seir["summary"]["attack_rate"] - sir["summary"]["attack_rate"]) < 0.01


def test_seird_deaths_track_cfr():
    cfr = 0.03
    res = sm.simulate(sm.SeirParams(r0=2.5, infectious_period_days=6, cfr=cfr,
                                    population=1_000_000, initial_infected=10), days=600)
    # essentially everyone who was infected has left I by the end → deaths ≈ CFR·cumulative
    assert abs(res["summary"]["total_deaths"] - cfr * res["cumulative"][-1]) < 0.01 * res["cumulative"][-1] + 1
    assert res["deaths"][0] <= res["deaths"][-1]  # monotone non-decreasing


# ── uncertainty ensemble ─────────────────────────────────────────────────────
def _ens_params():
    return {
        "r0": sm.ParamDist(2.5, 2.0, 3.2),
        "infectious_period_days": sm.ParamDist(6, 5, 8),
        "incubation_period_days": sm.ParamDist(4, 3, 5),
        "cfr": 0.02,
        "population": 1_000_000,
        "initial_infected": 10,
    }


def test_ensemble_band_ordering_and_model():
    ens = sm.simulate_ensemble(_ens_params(), days=300, n_samples=120, seed=7)
    assert ens["model"] == "SEIRD"
    assert ens["n_samples"] == 120
    for key in ("incidence", "prevalence", "cumulative", "deaths", "r_eff"):
        b = ens[key]
        assert all(b["lower"][t] <= b["median"][t] <= b["upper"][t] for t in range(len(b["median"])))


def test_ensemble_reproducible_with_seed():
    a = sm.simulate_ensemble(_ens_params(), days=200, n_samples=80, seed=42)
    b = sm.simulate_ensemble(_ens_params(), days=200, n_samples=80, seed=42)
    assert a["incidence"]["median"] == b["incidence"]["median"]
    assert a["summary"]["r0"] == b["summary"]["r0"]


def test_ensemble_fixed_params_collapse_to_point_estimate():
    # No CIs anywhere → every draw identical → zero-width bands == the deterministic run.
    params = {"r0": 2.5, "infectious_period_days": 6, "population": 1e6, "initial_infected": 10}
    ens = sm.simulate_ensemble(params, days=150, n_samples=25, seed=1)
    det = sm.simulate(sm.SeirParams(r0=2.5, infectious_period_days=6, population=1e6, initial_infected=10), days=150)
    lo, up = ens["incidence"]["lower"], ens["incidence"]["upper"]
    # zero-width band (identical draws) — up to percentile-interpolation float rounding
    assert all(abs(up[t] - lo[t]) < 1e-6 for t in range(len(lo)))
    for t in range(len(det["incidence"])):
        assert abs(ens["incidence"]["median"][t] - det["incidence"][t]) < 1e-6


def test_param_dist_sd_from_ci():
    d = sm.ParamDist(2.5, 2.0, 3.0)
    assert abs(d.sd() - (3.0 - 2.0) / (2 * 1.959963984540054)) < 1e-9
    assert sm.ParamDist(2.5).sd() == 0.0  # no CI → fixed


# ── literature-extracted parameters → model inputs ───────────────────────────
def _raw_block():
    return {
        "applicable": True,
        "population_disease": "Influenza A (H3N2)",
        "r0": {"value": "1.3", "ci_low": 1.1, "ci_high": 1.6, "unit": "ratio",
               "n_studies": 4, "provenance": [11, 22, 999, "33"]},
        "infectious_period_days": {"value": 4, "provenance": [11]},
        "incubation_period_days": {"value": 2, "ci_low": 3, "ci_high": 1, "provenance": [22]},
        "cfr": {"value": 1.4, "unit": "proportion", "provenance": [11]},
        "immunity_duration_days": {"value": None},
        "serial_interval_days": {"value": 3.0, "ci_low": 2.5, "ci_high": 3.6, "provenance": [22]},
    }


def test_normalize_coercion_and_provenance_filter():
    norm = sm.normalize_extracted_parameters(_raw_block(), valid_ids={11, 22, 33})
    assert norm["applicable"] is True
    assert norm["disease"] == "Influenza A (H3N2)"
    # invalid id 999 dropped, string "33" coerced to 33
    assert norm["params"]["r0"]["provenance"] == [11, 22, 33]
    assert norm["params"]["r0"]["value"] == 1.3
    assert norm["cited"] == [11, 22, 33]


def test_normalize_drops_nulls_clamps_cfr_and_bad_ci():
    norm = sm.normalize_extracted_parameters(_raw_block(), valid_ids={11, 22, 33})
    p = norm["params"]
    assert "immunity_duration_days" not in p                 # value null → dropped entirely
    assert p["cfr"]["value"] == 1.0                          # proportion clamped to [0,1]
    # incoherent CI (low 3 > high 1) → CI discarded, value kept as a point estimate
    assert "incubation_period_days" in p
    assert p["incubation_period_days"]["ci_low"] is None and p["incubation_period_days"]["ci_high"] is None


def test_normalize_not_applicable_or_empty():
    # explicit not-applicable → applicable False even if numbers present
    off = sm.normalize_extracted_parameters({"applicable": False, "r0": {"value": 2.0}}, valid_ids=set())
    assert off["applicable"] is False and "r0" in off["params"]
    # non-dict / missing → empty, safe
    assert sm.normalize_extracted_parameters(None)["params"] == {}
    assert sm.normalize_extracted_parameters("nope")["applicable"] is False


def test_params_to_distributions_types():
    norm = sm.normalize_extracted_parameters(_raw_block(), valid_ids={11, 22, 33})
    dists = sm.params_to_distributions(norm["params"])
    assert isinstance(dists["r0"], sm.ParamDist)             # has CI → distribution
    assert dists["r0"].ci_low == 1.1 and dists["r0"].ci_high == 1.6
    assert dists["infectious_period_days"] == 4.0            # no CI → fixed float
    assert dists["incubation_period_days"] == 2.0           # CI discarded → fixed float


def test_extracted_params_feed_ensemble_end_to_end():
    norm = sm.normalize_extracted_parameters(_raw_block(), valid_ids={11, 22, 33})
    dists = sm.params_to_distributions(norm["params"])
    dists.update({"population": 1_000_000, "initial_infected": 50})
    ens = sm.simulate_ensemble(dists, days=180, n_samples=40, seed=3)
    assert ens["model"] == "SEIRD"                          # incubation + CFR present
    b = ens["summary"]["r0"]
    assert b["lower"] <= b["median"] <= b["upper"]


# ── geography → population (Follow-up A) ──────────────────────────────────────
def test_population_for_geography_exact_and_case_insensitive():
    assert sm.population_for_geography("France") == 68_000_000
    assert sm.population_for_geography("  france ") == 68_000_000     # trimmed + lowercased
    assert sm.population_for_geography("SWITZERLAND") == 8_800_000
    assert sm.population_for_geography("Suisse") == 8_800_000         # FR alias


def test_population_for_geography_segments_first_match_wins():
    # multi-segment label → first recognised segment (by order) is used
    assert sm.population_for_geography("Geneva, Switzerland") == 500_000
    assert sm.population_for_geography("multi-country / France") == 68_000_000
    assert sm.population_for_geography("Vaud; Valais") == 815_000


def test_population_for_geography_no_false_substring():
    # "us" must NOT match inside "australia"; unknown/empty → None (no guessing)
    assert sm.population_for_geography("australia") == 26_000_000
    assert sm.population_for_geography("Neverland") is None
    assert sm.population_for_geography("") is None
    assert sm.population_for_geography(None) is None


# ── quality-weighted pooling (Follow-up B) ────────────────────────────────────
def test_pool_weighted_weights_toward_higher_quality():
    # two studies: a low value at high quality, a high value at low quality → the
    # pooled mean is pulled toward the high-quality study (below the plain average 2.5).
    obs = [{"article_id": 1, "value": 2.0}, {"article_id": 2, "value": 3.0}]
    pooled = sm.pool_weighted(obs, {1: 9.0, 2: 1.0})
    assert 2.0 < pooled["value"] < 2.5
    assert pooled["n_studies"] == 2
    assert sorted(pooled["provenance"]) == [1, 2]
    assert pooled["ci_low"] is not None and pooled["ci_low"] < pooled["value"] < pooled["ci_high"]


def test_pool_weighted_single_and_empty():
    one = sm.pool_weighted([{"article_id": 5, "value": 1.7}], {5: 4.0})
    assert one["value"] == 1.7 and one["n_studies"] == 1
    assert one["ci_low"] is None and one["ci_high"] is None    # single study → no dispersion CI
    assert sm.pool_weighted([], {}) is None                    # nothing numeric → None
    assert sm.pool_weighted([{"article_id": 1, "value": "abc"}], {1: 2.0}) is None


def test_pool_weighted_defaults_weight_when_quality_missing():
    # no quality map → equal weights → plain arithmetic mean
    pooled = sm.pool_weighted([{"article_id": 1, "value": 2.0}, {"article_id": 2, "value": 4.0}])
    assert abs(pooled["value"] - 3.0) < 1e-9


# ── normalize with per-study observations + quality weighting ─────────────────
def _obs_block():
    return {
        "applicable": True,
        "population_disease": "Measles",
        "r0": {"value": 12.0, "ci_low": 10.0, "ci_high": 14.0, "unit": "ratio",
               "n_studies": 1, "provenance": [1],
               "observations": [{"article_id": 1, "value": 15.0},
                                {"article_id": 2, "value": 18.0},
                                {"article_id": 99, "value": 999.0}]},  # 99 not in valid_ids
    }


def test_normalize_pooling_overrides_llm_value_when_two_studies():
    norm = sm.normalize_extracted_parameters(_obs_block(), valid_ids={1, 2}, quality_by_id={1: 5.0, 2: 5.0})
    r0 = norm["params"]["r0"]
    # invalid obs (id 99) filtered; equal weights → mean of 15 and 18 = 16.5, NOT the LLM 12.0
    assert abs(r0["value"] - 16.5) < 1e-9
    assert r0["n_studies"] == 2
    assert sorted(r0["provenance"]) == [1, 2]
    assert r0["ci_low"] < r0["value"] < r0["ci_high"]


def test_normalize_ignores_observations_without_quality_map():
    # backward-compatible: no quality_by_id → observations ignored, LLM value kept
    norm = sm.normalize_extracted_parameters(_obs_block(), valid_ids={1, 2})
    assert norm["params"]["r0"]["value"] == 12.0


def test_normalize_single_observation_falls_back_to_llm_value():
    blk = {
        "applicable": True,
        "r0": {"value": 12.0, "provenance": [1],
               "observations": [{"article_id": 1, "value": 15.0}]},  # only one → no pool override
    }
    norm = sm.normalize_extracted_parameters(blk, valid_ids={1}, quality_by_id={1: 5.0})
    assert norm["params"]["r0"]["value"] == 12.0                # < 2 studies → LLM estimate retained


def test_normalize_pooling_works_when_llm_value_absent():
    # param with ONLY observations (no central "value") still yields a pooled estimate
    blk = {
        "applicable": True,
        "cfr": {"value": None, "unit": "proportion", "provenance": [1, 2],
                "observations": [{"article_id": 1, "value": 0.02},
                                 {"article_id": 2, "value": 0.04}]},
    }
    norm = sm.normalize_extracted_parameters(blk, valid_ids={1, 2}, quality_by_id={1: 3.0, 2: 3.0})
    assert "cfr" in norm["params"]
    assert abs(norm["params"]["cfr"]["value"] - 0.03) < 1e-9
