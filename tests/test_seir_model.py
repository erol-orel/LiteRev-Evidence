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


# ── vaccination (V) & quarantine (Q) compartments ────────────────────────────
def _seir_base(**kw):
    return sm.simulate(sm.SeirParams(r0=2.5, infectious_period_days=6, incubation_period_days=3,
                                     population=1e6, initial_infected=10, **kw), days=250)


def test_vq_off_reproduces_base_model_exactly():
    # The V/Q extension must be a no-op when both are absent (or zero): identical name,
    # identical attack rate — this is what preserves the final-size guarantees above.
    base = _seir_base()
    none_off = _seir_base(vaccination_rate=None, quarantine_rate=None)
    zero_off = _seir_base(vaccination_rate=0.0, vaccine_efficacy=0.9, quarantine_rate=0.0)
    for other in (none_off, zero_off):
        assert other["model"] == base["model"] == "SEIR"
        assert abs(other["summary"]["attack_rate"] - base["summary"]["attack_rate"]) < 1e-12
    assert base["summary"]["total_vaccinated"] == 0.0
    assert base["summary"]["peak_quarantine"] == 0.0


def test_model_naming_with_v_and_q():
    cases = [
        (dict(r0=2, incubation_period_days=3, vaccination_rate=0.01), "SVEIR"),
        (dict(r0=2, vaccination_rate=0.01), "SVIR"),
        (dict(r0=2, incubation_period_days=3, quarantine_rate=0.1), "SEIQR"),
        (dict(r0=2, quarantine_rate=0.1), "SIQR"),
        (dict(r0=2, incubation_period_days=3, vaccination_rate=0.01, quarantine_rate=0.1), "SVEIQR"),
        (dict(r0=2, incubation_period_days=3, vaccination_rate=0.01, quarantine_rate=0.1,
              cfr=0.02, immunity_duration_days=180), "SVEIQRDS"),
    ]
    for kw, expected in cases:
        res = sm.simulate(sm.SeirParams(population=1e6, initial_infected=10, **kw), days=40)
        assert res["model"] == expected, f"{kw} → {res['model']} (expected {expected})"


def test_vaccination_reduces_attack_rate():
    base = _seir_base()
    vac = _seir_base(vaccination_rate=0.01, vaccine_efficacy=0.9)
    assert vac["model"] == "SVEIR"
    assert vac["summary"]["attack_rate"] < base["summary"]["attack_rate"]
    assert vac["summary"]["total_vaccinated"] > 0.0
    # higher efficacy → stronger protection (nu = rate · efficacy) → lower attack rate
    low_eff = _seir_base(vaccination_rate=0.01, vaccine_efficacy=0.3)
    assert vac["summary"]["attack_rate"] < low_eff["summary"]["attack_rate"]


def test_quarantine_reduces_peak_prevalence():
    base = _seir_base()
    # mild isolation: epidemic still occurs but with a lower, later peak
    qua = _seir_base(quarantine_rate=0.05)
    assert qua["model"] == "SEIQR"
    assert qua["summary"]["peak_quarantine"] > 0.0
    assert qua["summary"]["peak_prevalence"] < base["summary"]["peak_prevalence"]
    assert qua["summary"]["attack_rate"] < base["summary"]["attack_rate"]
    # strong isolation drives R_eff below 1 → outbreak fails to take off
    strong = _seir_base(quarantine_rate=0.5)
    assert strong["summary"]["attack_rate"] < 0.02


def test_vq_population_conservation():
    # S+E+I+R+D+V+Q must equal N at every recorded step (C is a separate accumulator).
    n = 1e6
    res = sm.simulate(sm.SeirParams(r0=3.0, infectious_period_days=5, incubation_period_days=2,
                                    cfr=0.02, vaccination_rate=0.008, vaccine_efficacy=0.8,
                                    quarantine_rate=0.15, population=n, initial_infected=50), days=300)
    for t in range(len(res["days"])):
        total = (res["susceptible"][t] + res["exposed"][t] + res["prevalence"][t]
                 + res["recovered"][t] + res["deaths"][t]
                 + res["vaccinated"][t] + res["quarantined"][t])
        assert abs(total - n) < 1.0, f"day {res['days'][t]}: {total} != {n}"


def test_ensemble_reports_vq_bands():
    ens = sm.simulate_ensemble(
        {"r0": sm.ParamDist(2.4, 2.0, 3.0), "infectious_period_days": 6, "incubation_period_days": 3,
         "vaccination_rate": 0.01, "vaccine_efficacy": 0.85, "population": 1e6, "initial_infected": 10},
        days=150, n_samples=40, seed=7)
    assert ens["model"] == "SVEIR"
    for key in ("vaccinated", "quarantined"):
        b = ens[key]
        assert all(b["lower"][t] <= b["median"][t] <= b["upper"][t] for t in range(len(b["median"])))
    assert ens["summary"]["total_vaccinated"]["median"] > 0.0


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


def test_population_in_text_prefers_most_local():
    # a COVID-Suisse-Romande query resolves to Romandie (2M), not Switzerland (8.8M) or world
    hit = sm.population_for_geography_in_text(
        "prediction du nombre de personnes hospitalisées par le COVID-19 en suisse romande")
    assert hit is not None and hit[0] == 2_000_000
    hit2 = sm.population_for_geography_in_text("COVID-19 hospitalizations in Switzerland")
    assert hit2 is not None and hit2[0] == 8_800_000
    # most local wins when several are named
    hit3 = sm.population_for_geography_in_text("ICU beds in Geneva, Switzerland")
    assert hit3 is not None and hit3[0] == 500_000


def test_population_in_text_none_and_word_boundary():
    assert sm.population_for_geography_in_text("machine learning for sepsis") is None
    assert sm.population_for_geography_in_text("") is None
    assert sm.population_for_geography_in_text("hospitalisations quotidiennes") is None  # no geo word
    assert sm.population_for_geography_in_text("a cohort in the USA")[0] == 335_000_000


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


# ── SEIR as a submodel: which predictor variables it derives ─────────────────
def test_seir_feature_column_maps_outputs():
    assert sm.seir_feature_column("Incidence hebdomadaire COVID-19") == "seir_incidence"
    assert sm.seir_feature_column("nouvelles infections / jour") == "seir_incidence"
    assert sm.seir_feature_column("Prévalence des cas infectieux") == "seir_prevalence"
    assert sm.seir_feature_column("Nombre de cas actifs") == "seir_prevalence"
    assert sm.seir_feature_column("seir_deaths") == "seir_deaths"
    assert sm.seir_feature_column("Décès cumulés dus au COVID") == "seir_deaths"


def test_seir_feature_column_priority_and_none():
    # "cumulative incidence" must resolve to cumulative (checked before incidence)
    assert sm.seir_feature_column("Incidence cumulée") == "seir_cumulative"
    assert sm.seir_feature_column("Taux d'attaque") == "seir_cumulative"
    # non-epidemic / unrelated variables → not SEIR-derivable
    assert sm.seir_feature_column("Température maximale") is None
    assert sm.seir_feature_column("Taux de vaccination") is None
    assert sm.seir_feature_column("Occupation des lits hospitaliers") is None
    assert sm.seir_feature_column("") is None


def test_is_seir_parameter_detects_params():
    assert sm.is_seir_parameter("R0 de base")
    assert sm.is_seir_parameter("Taux de reproduction effectif")
    assert sm.is_seir_parameter("CFR (létalité)")
    assert sm.is_seir_parameter("Période d'incubation (jours)")
    assert sm.is_seir_parameter("Infectious period")
    assert sm.is_seir_parameter("Intervalle sériel")


def test_is_seir_parameter_no_false_positives():
    # short ambiguous tokens are whole-word only — "r0"/"cfr" must not match inside words
    assert not sm.is_seir_parameter("start_date")       # contains "rt"? no bare "rt" keyword anyway
    assert not sm.is_seir_parameter("comfort_index")    # "rt"/"r0" not present as words
    assert not sm.is_seir_parameter("Nombre d'hospitalisations")
    assert not sm.is_seir_parameter("Incidence hebdomadaire")   # an output, not a parameter
    assert not sm.is_seir_parameter("Température")
    assert not sm.is_seir_parameter("")


def test_seir_param_and_output_are_disjoint_for_typical_vars():
    # a case-fatality variable is a PARAMETER, not a "deaths" output (param check wins upstream)
    assert sm.is_seir_parameter("Case fatality rate")
    # bare "décès" is an output, not a parameter
    assert not sm.is_seir_parameter("Décès quotidiens")
    assert sm.seir_feature_column("Décès quotidiens") == "seir_deaths"


# ── expanded SEIR+ vocabulary (compartments, flows, parameters) ──────────────
def test_seir_feature_column_compartments():
    assert sm.seir_feature_column("Population susceptible") == "seir_susceptible"
    assert sm.seir_feature_column("Nombre de personnes exposées") == "seir_exposed"
    assert sm.seir_feature_column("Latently infected individuals") == "seir_exposed"
    assert sm.seir_feature_column("Personnes rétablies") == "seir_recovered"
    assert sm.seir_feature_column("Recovered / immune population") == "seir_recovered"
    assert sm.seir_feature_column("Immunité de la population") == "seir_recovered"


def test_seir_feature_column_expanded_flow_vocab():
    assert sm.seir_feature_column("Cas confirmés quotidiens") == "seir_incidence"
    assert sm.seir_feature_column("Reported cases") == "seir_incidence"
    assert sm.seir_feature_column("Taux d'incidence") == "seir_incidence"
    assert sm.seir_feature_column("Total infections to date") == "seir_cumulative"
    assert sm.seir_feature_column("Final size of the epidemic") == "seir_cumulative"
    assert sm.seir_feature_column("Cas symptomatiques") == "seir_prevalence"
    assert sm.seir_feature_column("Number of infectious individuals") == "seir_prevalence"


def test_is_seir_parameter_expanded():
    for v in ("Rt effectif", "Effective reproduction number", "Basic reproduction number R0",
              "Taux de transmission", "Transmission coefficient", "Contact rate",
              "Recovery rate", "Taux de guérison", "Generation time", "Generation interval",
              "Temps de génération", "Waning immunity", "Immunité décroissante",
              "Infection fatality ratio (IFR)"):
        assert sm.is_seir_parameter(v), v


def test_is_seir_parameter_keeps_external_covariates():
    # covariates the engine does NOT model must stay real features (not dropped as params)
    for v in ("Taux de vaccination", "Vaccination coverage", "Couverture vaccinale",
              "Mobilité de la population", "Occupation des lits", "Température moyenne",
              "Densité de population", "Proportion de tests positifs"):
        assert not sm.is_seir_parameter(v), v
        assert sm.seir_feature_column(v) is None, v


def test_param_precedence_for_immunity_duration():
    # "durée d'immunité" is a PARAMETER; the caller (_attach_model_spec) checks
    # is_seir_parameter BEFORE seir_feature_column, so it is excluded, not routed to R.
    assert sm.is_seir_parameter("Durée d'immunité (jours)")


def test_simulate_exposes_compartments():
    res = sm.simulate(
        sm.SeirParams(r0=2.5, infectious_period_days=6, incubation_period_days=4,
                      cfr=0.02, population=1_000_000, initial_infected=10),
        days=120,
    )
    n = 1_000_000
    for key in ("susceptible", "exposed", "recovered"):
        assert key in res and len(res[key]) == len(res["days"])
        assert all(0.0 - 1e-6 <= v <= n + 1.0 for v in res[key])
    # S starts near the whole population and decreases; R starts at 0 and grows
    assert res["susceptible"][0] > 0.9 * n
    assert res["susceptible"][-1] < res["susceptible"][0]
    assert res["recovered"][0] < 1.0 and res["recovered"][-1] > res["recovered"][0]


def test_ensemble_aggregates_compartments():
    ens = sm.simulate_ensemble(_ens_params(), days=120, n_samples=40, seed=5)
    for key in ("susceptible", "exposed", "recovered"):
        b = ens[key]
        assert all(b["lower"][t] <= b["median"][t] <= b["upper"][t] for t in range(len(b["median"])))


# ── calibration to an observed series + overlay scaling ──────────────────────
def _synthetic_observed(r0, col, scale, days_range):
    truth = sm.simulate(sm.SeirParams(r0=r0, infectious_period_days=6, incubation_period_days=3,
                                      population=1_000_000, initial_infected=20), days=200)
    return [(d, truth[col][d] * scale) for d in days_range], truth


def test_calibrate_recovers_r0_despite_unit_scale():
    obs, _ = _synthetic_observed(2.7, "incidence", 0.0003, range(12, 130, 5))
    base = {"infectious_period_days": 6, "incubation_period_days": 3,
            "population": 1_000_000, "initial_infected": 20}
    fit = sm.calibrate_to_observed(obs, base, column="incidence", days=200)
    assert fit["ok"] and abs(fit["fitted_r0"] - 2.7) < 0.15
    assert fit["r2"] > 0.98
    assert abs(fit["scale"] - 0.0003) / 0.0003 < 0.1     # amplitude scale recovered


def test_calibrate_needs_enough_points():
    r = sm.calibrate_to_observed([(1, 2.0), (2, 3.0)], {"population": 1e6})
    assert r["ok"] is False and r["n_points"] == 2


def test_align_observed_dates_to_day_offsets():
    a = sm.align_observed([{"date": "2023-02-01", "value": 3},
                           {"date": "2023-01-25", "value": 1},
                           {"date": "2023-02-08", "value": 9}])
    assert a["start_date"] == "2023-01-25"
    assert a["points"] == [(0.0, 1.0), (7.0, 3.0), (14.0, 9.0)]
    # numeric days pass through, junk dropped
    b = sm.align_observed([{"day": 2, "value": 5}, {"day": 0, "value": 1}, {"date": "x", "value": 9}])
    assert b["points"] == [(0.0, 1.0), (2.0, 5.0)]


def test_scale_observed_into_model_units():
    obs, truth = _synthetic_observed(2.5, "incidence", 0.002, range(20, 120, 7))
    base = {"r0": 2.5, "infectious_period_days": 6, "incubation_period_days": 3,
            "population": 1_000_000, "initial_infected": 20}
    s = sm.scale_observed_to_model(obs, base, "incidence", days=180)
    assert abs(s["scale"] - 0.002) / 0.002 < 0.05 and s["r2"] > 0.99
    d0 = int(s["points"][0]["day"])
    assert abs(s["points"][0]["value"] - truth["incidence"][d0]) / truth["incidence"][d0] < 0.05
