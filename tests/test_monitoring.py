"""Tests for the classification monitor honoring literature alert thresholds.

Previously the classification live-monitor used fixed 0.33/0.66 probability bands,
ignoring the evidence-derived alert_thresholds. Now it mirrors the regression path:
literature thresholds first (probability ranges), fixed bands only as a fallback.
"""
import pytest

for _m in ("numpy", "pandas", "sklearn", "scipy", "optuna"):
    pytest.importorskip(_m)

import numpy as np
import pandas as pd

import model_trainer as mt


def _clf_pipeline():
    rng = np.random.RandomState(0)
    n = 300
    a, b = rng.normal(0, 1, n), rng.normal(0, 1, n)
    df = pd.DataFrame({"a": a, "b": b, "out": ["oui" if v > 0 else "non" for v in (a - b)]})
    spec = {
        "outcome": {"task_type": "classification", "machine_name": "out", "positive_class": "oui"},
        "features": [{"machine_name": "a", "dtype": "float"}, {"machine_name": "b", "dtype": "float"}],
        "algorithm": {"family": "logistic_regression", "metric": "roc_auc", "cv": {"strategy": "stratified_kfold", "folds": 4}},
    }
    res = mt.train_model(df, spec, n_trials=5)
    return res["pipeline"], res["classes"], df[["a", "b"]].head(20)


def test_classification_monitor_uses_literature_thresholds():
    pipe, classes, recent = _clf_pipeline()
    at = {"green": {"range": "< 0.10"}, "orange": {"range": "0.10-0.30"}, "red": {"range": "> 0.30"}}
    mon = mt.compute_monitoring(pipe, recent, "classification", classes=classes,
                                positive_class="oui", alert_thresholds=at)
    assert mon["kind"] == "probability"
    assert mon["bands_source"] == "literature"
    assert abs(mon["bands"]["orange"] - 0.10) < 1e-9
    assert abs(mon["bands"]["red"] - 0.30) < 1e-9


def test_classification_monitor_percentage_thresholds_normalized():
    pipe, classes, recent = _clf_pipeline()
    at = {"green": {"range": "< 10"}, "orange": {"range": "10-30"}, "red": {"range": "> 30"}}
    mon = mt.compute_monitoring(pipe, recent, "classification", classes=classes,
                                positive_class="oui", alert_thresholds=at)
    assert abs(mon["bands"]["orange"] - 0.10) < 1e-9      # 10% → 0.10
    assert abs(mon["bands"]["red"] - 0.30) < 1e-9         # 30% → 0.30


def test_classification_monitor_falls_back_to_fixed_bands():
    pipe, classes, recent = _clf_pipeline()
    mon = mt.compute_monitoring(pipe, recent, "classification", classes=classes,
                                positive_class="oui", alert_thresholds=None)
    assert mon["bands"] == {"orange": 0.33, "red": 0.66}
    assert mon["bands_source"] == "default_probability"
