"""Tests for the Extremal / Quantile Regression Forest (GESICA "Extremal Random Forest").

It must predict a high conditional quantile (the surge) rather than the mean: on noisy
data its coverage (share of actuals at or below the prediction) sits near the target
quantile, well above a mean model's ~0.5, and it plugs into the Pipeline/CV/joblib
machinery. Skips cleanly if the ML wheels are absent.
"""
import pytest

for _mod in ("numpy", "pandas", "sklearn", "optuna"):
    pytest.importorskip(_mod)

import io
import numpy as np
import pandas as pd
import joblib

import model_trainer as mt


def _noisy_df(n=400, seed=0):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({"x1": rng.normal(0, 1, n), "x2": rng.normal(0, 1, n)})
    df["y"] = 10 + 3 * df.x1 + 2 * df.x2 + rng.normal(0, 2.0, n)
    return df


def test_weighted_quantile_matches_numpy_on_uniform_weights():
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    w = np.ones_like(v)
    # weighted median with uniform weights ≈ ordinary median
    assert abs(mt._weighted_quantile(v, w, 0.5) - 3.0) < 1e-9
    assert mt._weighted_quantile(v, w, 0.99) >= 4.0


def test_estimator_predicts_high_quantile_above_mean():
    df = _noisy_df()
    est = mt.QuantileRandomForestRegressor(quantile=0.9, n_estimators=150, random_state=0).fit(
        df[["x1", "x2"]].values, df["y"].values)
    p = est.predict(df[["x1", "x2"]].values)
    coverage = float((df["y"].values <= p).mean())
    assert 0.80 <= coverage <= 0.99          # ~90% of actuals fall at/below the prediction
    assert p.mean() > df["y"].mean()          # the surge sits above the average


_BASE = {
    "outcome": {"machine_name": "y", "task_type": "regression", "unit": "count"},
    "features": [{"machine_name": "x1", "dtype": "float"}, {"machine_name": "x2", "dtype": "float"}],
}


def test_train_model_extremal_rf_scores_with_pinball_and_covers_the_tail():
    df = _noisy_df()
    res = mt.train_model(df, {**_BASE, "algorithm": {
        "family": "extremal_rf", "quantile": 0.9, "cv": {"strategy": "kfold", "folds": 4}}}, n_trials=6)
    assert res["family"] == "extremal_rf"
    assert res["metric"] == "pinball"                          # optimised on pinball, not rmse
    m = res["metrics"]
    assert m["target_quantile"] == 0.9
    assert 0.80 <= m["coverage"] <= 1.0 and "pinball" in m and "cv_pinball" in m
    # vs a mean model: extremal coverage is much higher than random forest's ~0.5
    rf = mt.train_model(df, {**_BASE, "algorithm": {
        "family": "random_forest", "metric": "rmse", "cv": {"strategy": "kfold", "folds": 4}}}, n_trials=4)
    cov_rf = float((df["y"].values <= rf["pipeline"].predict(df[["x1", "x2"]])).mean())
    assert m["coverage"] > cov_rf


def test_extremal_rf_pipeline_is_picklable_and_cloneable():
    from sklearn.base import clone
    df = _noisy_df()
    res = mt.train_model(df, {**_BASE, "algorithm": {"family": "extremal_rf", "quantile": 0.9}}, n_trials=4)
    buf = io.BytesIO(); joblib.dump(res["pipeline"], buf)
    reloaded = joblib.load(io.BytesIO(buf.getvalue()))
    assert len(reloaded.predict(df[["x1", "x2"]].head(3))) == 3
    clone(res["pipeline"])


def test_extremal_rf_classification_falls_back_to_random_forest():
    # quantile forest is regression-only → a classification spec must remap, not crash
    assert mt._effective_family("extremal_rf", "classification") == "random_forest"
    assert mt._effective_family("extremal_rf", "regression") == "extremal_rf"
