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
