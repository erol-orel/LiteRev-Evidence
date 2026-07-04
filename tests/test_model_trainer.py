"""Correctness tests for the Model tab's real training (model_trainer.py) and the
upload validator guards. No DB or network — model_trainer is a pure DataFrame→dict
module; the validator is a pure function (main is import-only, env set by conftest).

These lock in the audit: reported metrics match an INDEPENDENT recomputation, the
model recovers a known signal, regression assumption diagnostics fire correctly,
accent/space-insensitive column matching works, and the upload readiness guards
block un-trainable files instead of returning a false "training started".
"""
import numpy as np
import pandas as pd
import pytest

import model_trainer as mt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics as M


def _reg_spec(family="linear_regression"):
    return {
        "outcome": {"task_type": "regression", "machine_name": "target"},
        "features": [{"machine_name": n, "dtype": "float"} for n in ("x1", "x2", "x3")],
        "algorithm": {"family": family, "metric": "rmse", "cv": {"strategy": "kfold", "folds": 5}},
    }


def _linear_df(seed=0, n=400):
    rng = np.random.RandomState(seed)
    x1, x2, x3 = (rng.normal(0, 1, n) for _ in range(3))
    y = 3 * x1 - 2 * x2 + 0.5 * x3 + rng.normal(0, 1.0, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "target": y})


# ── the core correctness proof: reported metric == independent recomputation ──
def test_regression_rmse_matches_independent_recomputation():
    df = _linear_df()
    res = mt.train_model(df, _reg_spec(), n_trials=5)

    Xtr, Xte, ytr, yte = train_test_split(
        df[["x1", "x2", "x3"]], df["target"].astype(float),
        test_size=0.2, random_state=42, shuffle=True)
    pre = ColumnTransformer([("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                               ("sc", StandardScaler())]), ["x1", "x2", "x3"])])
    gt = Pipeline([("pre", pre), ("est", LinearRegression())]).fit(Xtr, ytr)
    gt_rmse = float(np.sqrt(M.mean_squared_error(yte, gt.predict(Xte))))

    assert abs(res["metrics"]["rmse"] - gt_rmse) < 1e-6       # reported == ground truth
    assert res["metrics"]["r2"] > 0.9                          # strong signal recovered
    assert 0.85 < res["metrics"]["rmse"] < 1.20                # ~ irreducible noise sigma=1


def test_importances_rank_by_true_signal_and_sum_to_one():
    res = mt.train_model(_linear_df(), _reg_spec(), n_trials=5)
    iv = res["importances_by_variable"]
    assert iv["x1"] > iv["x2"] > iv["x3"]                      # matches |beta|: 3 > 2 > 0.5
    assert abs(sum(iv.values()) - 1.0) < 1e-6


def test_cv_metric_is_positive_and_pretty_named():
    res = mt.train_model(_linear_df(), _reg_spec(), n_trials=5)
    assert res["metrics"]["cv_rmse"] > 0                       # not the negative neg_* scorer


# ── regression assumption diagnostics (Gauss-Markov) ─────────────────────────
def test_regression_carries_assumption_diagnostics():
    res = mt.train_model(_linear_df(), _reg_spec(), n_trials=5)
    keys = {c["key"] for c in res["assumptions"]["checks"]}
    assert {"autocorrelation", "normality", "homoscedasticity", "multicollinearity"} <= keys
    assert res["assumptions"]["applies"] is True              # linear model → OLS assumptions apply


def test_clean_data_passes_assumptions():
    res = mt.train_model(_linear_df(), _reg_spec(), n_trials=5)
    st = {c["key"]: c["status"] for c in res["assumptions"]["checks"]}
    assert st["homoscedasticity"] == "ok"
    assert st["multicollinearity"] == "ok"


def test_diagnostics_flag_multicollinearity_and_heteroscedasticity():
    rng = np.random.RandomState(1)
    n = 500
    x1 = rng.normal(0, 1, n)
    x2 = x1 + rng.normal(0, 0.01, n)          # near-perfect collinearity with x1
    x3 = rng.normal(0, 1, n)
    y = 3 * x1 - 2 * x3 + np.abs(x1) * 3.0 * rng.normal(0, 1, n)   # funnel heteroscedasticity
    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "target": y})
    res = mt.train_model(df, _reg_spec(), n_trials=5)
    st = {c["key"]: c["status"] for c in res["assumptions"]["checks"]}
    assert st["multicollinearity"] in ("warn", "fail")
    assert st["homoscedasticity"] == "warn"


def test_classification_has_no_regression_assumptions():
    rng = np.random.RandomState(2)
    n = 300
    df = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n)})
    df["out"] = ["oui" if v > 0 else "non" for v in (df.a - df.b)]
    res = mt.train_model(df, {
        "outcome": {"task_type": "classification", "machine_name": "out"},
        "features": [{"machine_name": "a", "dtype": "float"}, {"machine_name": "b", "dtype": "float"}],
        "algorithm": {"family": "logistic_regression", "metric": "roc_auc",
                      "cv": {"strategy": "stratified_kfold", "folds": 4}},
    }, n_trials=5)
    assert "assumptions" not in res
    assert res["metrics"]["roc_auc"] > 0.85


# ── accent/space-insensitive column matching ─────────────────────────────────
def test_slug_column_matching_handles_accents_and_spaces():
    rng = np.random.RandomState(0)
    n = 200
    df = pd.DataFrame({
        "Température max (°C)": rng.normal(0, 1, n),
        "Âge du patient": rng.normal(0, 1, n),
    })
    df["Issue clinique"] = 2 * df["Température max (°C)"] - df["Âge du patient"] + rng.normal(0, 1, n)
    spec = {
        "outcome": {"task_type": "regression", "machine_name": "issue_clinique"},
        "features": [{"machine_name": "temperature_max_c", "dtype": "float"},
                     {"machine_name": "age_du_patient", "dtype": "float"}],
        "algorithm": {"family": "linear_regression", "metric": "rmse", "cv": {"strategy": "kfold", "folds": 5}},
    }
    res = mt.train_model(df, spec, n_trials=3)
    assert set(res["features_used"]) == {"temperature_max_c", "age_du_patient"}


# ── metric labelling: cross-task metric is corrected to the real scorer ──────
def test_metric_is_authoritative_when_spec_metric_is_cross_task():
    rng = np.random.RandomState(2)
    n = 300
    df = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n)})
    df["out"] = ["oui" if v > 0 else "non" for v in (df.a - df.b)]
    # spec asks for "rmse" on a CLASSIFICATION task (inconsistent)
    res = mt.train_model(df, {
        "outcome": {"task_type": "classification", "machine_name": "out"},
        "features": [{"machine_name": "a", "dtype": "float"}, {"machine_name": "b", "dtype": "float"}],
        "algorithm": {"family": "logistic_regression", "metric": "rmse",
                      "cv": {"strategy": "stratified_kfold", "folds": 4}},
    }, n_trials=5)
    assert res["metric"] == "roc_auc"           # corrected to the real scorer
    assert res["requested_metric"] == "rmse"    # original preserved for transparency
    assert res["metric"] in res["metrics"]      # so the UI card shows the right value


# ── guards ───────────────────────────────────────────────────────────────────
def test_missing_target_raises():
    df = _linear_df().drop(columns=["target"])
    with pytest.raises(ValueError):
        mt.train_model(df, _reg_spec(), n_trials=3)


def test_too_few_rows_raises():
    with pytest.raises(ValueError):
        mt.train_model(_linear_df(n=10), _reg_spec(), n_trials=3)


def test_reproducible_same_seed():
    a = mt.train_model(_linear_df(), _reg_spec("gradient_boosting"), n_trials=8)
    b = mt.train_model(_linear_df(), _reg_spec("gradient_boosting"), n_trials=8)
    assert a["metrics"]["rmse"] == b["metrics"]["rmse"]


# ── upload validator readiness guards (pure function in main) ────────────────
def test_validator_can_train_guards():
    import main
    tmpl = {"target_column": "y", "columns": [
        {"name": "y", "role": "outcome", "dtype": "float"},
        {"name": "x1", "role": "feature", "dtype": "float"}]}
    ok = main._validate_dataset_against_template(["y", "x1"], tmpl, {"y": "numeric", "x1": "numeric"}, n_rows=100)
    assert ok["readiness"]["can_train"] is True

    few = main._validate_dataset_against_template(["y", "x1"], tmpl, {"y": "numeric", "x1": "numeric"}, n_rows=5)
    assert few["readiness"]["can_train"] is False              # too few rows blocks (was True before)

    bad = main._validate_dataset_against_template(["y", "x1"], tmpl, {"y": "numeric", "x1": "other"}, n_rows=100)
    assert bad["readiness"]["can_train"] is False              # non-numeric-in-numeric blocks


def test_validator_matches_accented_headers():
    import main
    tmpl = {"target_column": "temperature_c", "columns": [
        {"name": "temperature_c", "role": "outcome", "dtype": "float"},
        {"name": "x1", "role": "feature", "dtype": "float"}]}
    rep = main._validate_dataset_against_template(
        ["Température (°C)", "X1"], tmpl, {"temperature_c": "numeric", "x1": "numeric"}, n_rows=100)
    assert rep["target_present"] is True
    assert rep["n_features_present"] == 1
