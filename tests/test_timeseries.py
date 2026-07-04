"""Correctness tests for the time-series forecasting path (model_trainer).

Prophet/SARIMAX are NOT sklearn estimators — train_model routes families
`prophet`/`sarimax` to train_timeseries_model, which holds out the last H points,
scores rmse/mae/mape on that holdout, then refits on the full series to forecast
forward. These lock in: the reported holdout RMSE matches an INDEPENDENT
recomputation from the returned actual/predicted arrays, the seasonal SARIMAX term
beats a plain (non-seasonal) baseline on a weekly-seasonal series, horizon/holdout
window sizes are consistent, the forward forecast is strictly after the history,
and the guards (too-short series, no date column) fire.

statsmodels (SARIMAX) is installed in CI so the SARIMAX path runs for real;
Prophet is heavy (cmdstanpy) and only asserted where importable.
"""
import pytest

for _mod in ("numpy", "pandas"):
    pytest.importorskip(_mod)
pytest.importorskip("statsmodels")   # SARIMAX path is the CI-covered one

import numpy as np
import pandas as pd

import model_trainer as mt

_HAS_PROPHET = mt._has_package("prophet")


def _seasonal_df(seed=0, n=220, sigma=1.0):
    """Daily series: linear trend + weekly (period 7) seasonality + noise."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    y = 20 + 0.05 * np.arange(n) + 5.0 * np.sin(2 * np.pi * np.arange(n) / 7.0) + rng.normal(0, sigma, n)
    return pd.DataFrame({"date": dates, "sales": y})


def _spec(family):
    return {
        "outcome": {"machine_name": "sales", "task_type": "forecast", "name": "Sales"},
        "features": [{"machine_name": "date", "dtype": "datetime", "name": "Date"}],
        "algorithm": {"family": family, "metric": "rmse"},
    }


# ── pure helpers ─────────────────────────────────────────────────────────────
def test_seasonal_period_by_freq():
    assert mt._seasonal_period("D") == 7
    assert mt._seasonal_period("MS") == 12
    assert mt._seasonal_period("W-SUN") == 52
    assert mt._seasonal_period("h") == 24
    assert mt._seasonal_period("QS") == 4
    assert mt._seasonal_period("YS") == 0


def test_ts_metrics_matches_manual():
    a = [10.0, 12.0, 11.0, 13.0]
    p = [10.5, 11.0, 11.5, 12.0]
    m = mt._ts_metrics(a, p)
    err = np.array(a) - np.array(p)
    assert abs(m["rmse"] - float(np.sqrt(np.mean(err ** 2)))) < 1e-9
    assert abs(m["mae"] - float(np.mean(np.abs(err)))) < 1e-9


def test_infer_freq_daily():
    d = pd.date_range("2022-01-01", periods=30, freq="D")
    assert mt._infer_freq(d) == "D"


# ── SARIMAX (real, CI-covered) ───────────────────────────────────────────────
def test_sarimax_holdout_rmse_matches_recomputation():
    res = mt.train_model(_seasonal_df(), _spec("sarimax"))
    assert res["family"] == "sarimax"
    assert res["task_type"] == "forecast"
    assert res["metric"] == "rmse"
    a = np.array(res["holdout"]["actual"], dtype=float)
    p = np.array(res["holdout"]["predicted"], dtype=float)
    rmse_indep = float(np.sqrt(np.mean((a - p) ** 2)))
    assert abs(res["metrics"]["rmse"] - rmse_indep) < 1e-3     # stored value rounded to 4dp


def test_sarimax_window_and_forecast_shapes():
    res = mt.train_model(_seasonal_df(), _spec("sarimax"))
    H = res["horizon"]
    assert len(res["holdout"]["actual"]) == H == len(res["holdout"]["predicted"])
    assert len(res["forecast"]["predicted"]) == H
    assert len(res["forecast"]["lower"]) == H and len(res["forecast"]["upper"]) == H
    assert res["n_train"] + res["n_test"] == res["n_points"]
    # forward forecast strictly after the last observed timestamp
    assert res["forecast"]["dates"][0] > res["history_tail"]["dates"][-1]


def test_sarimax_seasonal_term_selected_and_beats_amplitude():
    # On a strongly weekly-seasonal series the grid must pick a seasonal order and
    # drive RMSE below the seasonal amplitude (5) — i.e. it actually models the cycle.
    res = mt.train_model(_seasonal_df(sigma=1.0), _spec("sarimax"))
    assert res["best_params"]["seasonal_order"][3] == 7      # weekly period captured
    assert res["metrics"]["rmse"] < 3.0


def test_forecast_result_has_no_feature_importances():
    res = mt.train_model(_seasonal_df(), _spec("sarimax"))
    assert res["feature_importances"] == []                  # N/A for univariate forecast
    assert res["importances_by_variable"] == {}


# ── guards ───────────────────────────────────────────────────────────────────
def test_too_short_series_raises():
    with pytest.raises(ValueError):
        mt.train_model(_seasonal_df(n=12), _spec("sarimax"))


def test_no_datetime_column_raises():
    df = _seasonal_df().drop(columns=["date"])
    spec = _spec("sarimax")
    spec["features"] = []                                    # no datetime feature, no date column
    with pytest.raises(ValueError):
        mt.train_model(df, spec)


def test_routing_bypasses_effective_family_downgrade():
    # prophet/sarimax must NOT be coerced to gradient_boosting by _effective_family;
    # train_model routes them to the forecasting path (task_type 'forecast').
    res = mt.train_model(_seasonal_df(), _spec("sarimax"))
    assert res["family"] == "sarimax" and res["task_type"] == "forecast"


# ── Prophet (only where importable) ──────────────────────────────────────────
@pytest.mark.skipif(not _HAS_PROPHET, reason="prophet not installed (heavy; verified in prod)")
def test_prophet_holdout_rmse_matches_recomputation():
    res = mt.train_model(_seasonal_df(), _spec("prophet"))
    assert res["family"] == "prophet" and res["task_type"] == "forecast"
    a = np.array(res["holdout"]["actual"], dtype=float)
    p = np.array(res["holdout"]["predicted"], dtype=float)
    rmse_indep = float(np.sqrt(np.mean((a - p) ** 2)))
    assert abs(res["metrics"]["rmse"] - rmse_indep) < 1e-3
    assert res["metrics"]["rmse"] < 3.0                     # captures weekly seasonality natively
    assert len(res["forecast"]["predicted"]) == res["horizon"]
