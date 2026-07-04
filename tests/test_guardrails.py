"""Correctness tests for the model guardrails + local explanation (model_trainer).

Guardrails run after training (informative, never alter the model): target-leak
detection, class-imbalance reporting, CV stability, and a bootstrap CI on the test
metric. explain_prediction attributes a prediction to its features by ablation.

These lock in: an injected leak is caught (and clean data is NOT flagged), a skewed
class distribution warns, the bootstrap CI brackets the point metric on non-degenerate
data, and local contributions track a known linear signal.
"""
import pytest

for _mod in ("numpy", "pandas", "sklearn", "scipy", "optuna"):
    pytest.importorskip(_mod)

import numpy as np
import pandas as pd

import model_trainer as mt


def _reg_spec(features):
    return {
        "outcome": {"task_type": "regression", "machine_name": "target"},
        "features": [{"machine_name": n, "dtype": "float"} for n in features],
        "algorithm": {"family": "linear_regression", "metric": "rmse", "cv": {"strategy": "kfold", "folds": 5}},
    }


def _reg_df(seed=0, n=400):
    rng = np.random.RandomState(seed)
    x1, x2, x3 = (rng.normal(0, 1, n) for _ in range(3))
    y = 3 * x1 - 2 * x2 + 0.5 * x3 + rng.normal(0, 1.0, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "target": y})


def _check(res, key):
    return next((c for c in res["guardrails"]["checks"] if c["key"] == key), None)


# ── leakage detection ────────────────────────────────────────────────────────
def test_leakage_detected_in_regression():
    df = _reg_df()
    df["x_leak"] = df["target"] + np.random.RandomState(9).normal(0, 1e-6, len(df))
    res = mt.train_model(df, _reg_spec(["x1", "x2", "x3", "x_leak"]), n_trials=5)
    lk = _check(res, "leakage")
    assert lk["status"] == "fail"
    assert "x_leak" in [l["feature"] for l in lk["leaks"]]


def test_clean_regression_has_no_leak():
    res = mt.train_model(_reg_df(), _reg_spec(["x1", "x2", "x3"]), n_trials=5)
    assert _check(res, "leakage")["status"] == "ok"


def test_leakage_detected_in_classification():
    rng = np.random.RandomState(1)
    n = 400
    a, b = rng.normal(0, 1, n), rng.normal(0, 1, n)
    label = (a - b > 0).astype(int)
    df = pd.DataFrame({"a": a, "b": b, "leaky": label.astype(float), "out": np.where(label == 1, "oui", "non")})
    spec = {
        "outcome": {"task_type": "classification", "machine_name": "out"},
        "features": [{"machine_name": c, "dtype": "float"} for c in ("a", "b", "leaky")],
        "algorithm": {"family": "logistic_regression", "metric": "roc_auc", "cv": {"strategy": "stratified_kfold", "folds": 4}},
    }
    res = mt.train_model(df, spec, n_trials=5)
    lk = _check(res, "leakage")
    assert lk["status"] == "fail" and "leaky" in [l["feature"] for l in lk["leaks"]]


# ── class imbalance ──────────────────────────────────────────────────────────
def test_class_imbalance_flagged():
    rng = np.random.RandomState(2)
    n = 500
    a, b = rng.normal(0, 1, n), rng.normal(0, 1, n)
    out = ["oui" if v > 0 else "non" for v in (a - b - 2.2)]     # ~5% positive
    df = pd.DataFrame({"a": a, "b": b, "out": out})
    spec = {
        "outcome": {"task_type": "classification", "machine_name": "out"},
        "features": [{"machine_name": "a", "dtype": "float"}, {"machine_name": "b", "dtype": "float"}],
        "algorithm": {"family": "logistic_regression", "metric": "roc_auc", "cv": {"strategy": "stratified_kfold", "folds": 4}},
    }
    res = mt.train_model(df, spec, n_trials=5)
    cb = _check(res, "class_balance")
    assert cb["status"] == "warn"
    assert cb["statistic"] < 0.10
    assert set(cb["distribution"].keys()) == {"oui", "non"}


def test_balanced_classes_ok():
    rng = np.random.RandomState(3)
    n = 400
    a, b = rng.normal(0, 1, n), rng.normal(0, 1, n)
    df = pd.DataFrame({"a": a, "b": b, "out": ["oui" if v > 0 else "non" for v in (a - b)]})
    spec = {
        "outcome": {"task_type": "classification", "machine_name": "out"},
        "features": [{"machine_name": "a", "dtype": "float"}, {"machine_name": "b", "dtype": "float"}],
        "algorithm": {"family": "logistic_regression", "metric": "roc_auc", "cv": {"strategy": "stratified_kfold", "folds": 4}},
    }
    res = mt.train_model(df, spec, n_trials=5)
    assert _check(res, "class_balance")["status"] == "ok"


# ── bootstrap metric CI ──────────────────────────────────────────────────────
def test_metric_ci_brackets_point_estimate():
    # Non-degenerate signal (noise sigma=1): the bootstrap 95% CI must contain the
    # full-test point RMSE, and be a real interval (low < high).
    res = mt.train_model(_reg_df(), _reg_spec(["x1", "x2", "x3"]), n_trials=5)
    ci = res["guardrails"]["metric_ci"]
    pt = res["metrics"]["rmse"]
    assert ci["metric"] == "rmse"
    assert ci["low"] < ci["high"]
    assert ci["low"] <= pt <= ci["high"]


def test_cv_stability_present():
    res = mt.train_model(_reg_df(), _reg_spec(["x1", "x2", "x3"]), n_trials=5)
    cv = _check(res, "cv_stability")
    assert cv is not None and cv["status"] in ("ok", "warn")
    assert "mean" in cv and "std" in cv


# ── local explanation ────────────────────────────────────────────────────────
def test_explain_prediction_tracks_linear_signal():
    res = mt.train_model(_reg_df(), _reg_spec(["x1", "x2", "x3"]), n_trials=5)
    bg = res["explain_background"]
    assert set(bg.keys()) == {"x1", "x2", "x3"}
    # A row with large positive x1 (coef +3) → x1 is the dominant positive contributor.
    row = pd.DataFrame([{"x1": 3.0, "x2": 0.0, "x3": 0.0}])
    exp = mt.explain_prediction(res["pipeline"], row, [{"machine_name": c} for c in ("x1", "x2", "x3")],
                                "regression", bg)
    top = exp["contributions"][0]
    assert top["feature"] == "x1"
    assert top["contribution"] > 0                              # positive x1 pushes prediction up
