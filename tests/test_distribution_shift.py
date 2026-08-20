"""Tests for the distributional-shift guardrail (GESICA "distributional shift : problème").

PSI between the early and late half of the time-ordered data flags features whose
distribution drifts over time; it rides along in the model's guardrails so the dashboard
surfaces it. Skips cleanly if the ML wheels are absent.
"""
import pytest

for _mod in ("numpy", "pandas", "sklearn", "optuna"):
    pytest.importorskip(_mod)

import numpy as np
import pandas as pd

import model_trainer as mt


def test_psi_zero_for_same_distribution_and_large_for_shift():
    rng = np.random.RandomState(0)
    a = rng.normal(0, 1, 2000)
    b = rng.normal(0, 1, 2000)
    assert mt._psi(a, b) < 0.05                      # same law → ~0
    c = rng.normal(4, 1, 2000)
    assert mt._psi(a, c) > 0.25                      # shifted mean → significant
    assert mt._psi([1.0] * 5, [2.0] * 5) is None     # too small → None


def _df(n=300, seed=0):
    rng = np.random.RandomState(seed)
    drift = np.concatenate([rng.normal(0, 1, n // 2), rng.normal(4, 1, n - n // 2)])
    stable = rng.normal(0, 1, n)
    return pd.DataFrame({
        "t": pd.date_range("2024-01-01", periods=n, freq="D").astype(str),
        "drift": drift, "stable": stable,
        "y": 2 * stable + rng.normal(0, 1, n),
    })


_USED = [{"machine_name": "drift", "dtype": "float"},
         {"machine_name": "stable", "dtype": "float"},
         {"machine_name": "t", "dtype": "datetime"}]


def test_distribution_shift_flags_the_drifting_feature_only():
    df = _df()
    chk = mt.distribution_shift(df[["t", "drift", "stable"]], _USED)
    assert chk["key"] == "distribution_shift" and chk["status"] in ("warn", "fail")
    assert chk["ordered_by"] == "t"
    psi = {f["feature"]: f["psi"] for f in chk["features"]}
    assert psi["drift"] >= 0.25 and psi["stable"] < 0.25
    assert "drift" in chk["detail"]


def test_distribution_shift_ok_when_stable():
    df = _df()
    chk = mt.distribution_shift(df[["stable"]], [{"machine_name": "stable", "dtype": "float"}])
    assert chk["status"] == "ok"


def test_distribution_shift_short_data_is_safe():
    df = _df(n=20)
    chk = mt.distribution_shift(df[["stable"]], [{"machine_name": "stable", "dtype": "float"}])
    assert chk["status"] == "ok" and "min. 40" in chk["detail"]


def test_guardrails_carry_the_drift_check_after_training():
    df = _df()
    spec = {"outcome": {"machine_name": "y", "task_type": "regression"},
            "features": [{"machine_name": "drift", "dtype": "float"},
                         {"machine_name": "stable", "dtype": "float"},
                         {"machine_name": "t", "dtype": "datetime"}],
            "algorithm": {"family": "random_forest", "metric": "rmse",
                          "cv": {"strategy": "kfold", "folds": 4}}}
    res = mt.train_model(df, spec, n_trials=4)
    keys = [c["key"] for c in res["guardrails"]["checks"]]
    assert "distribution_shift" in keys
