"""Built-in demo scenario — a real influenza dataset + a trained model.

Makes the real-dataset trial (`scripts/trial_weather_influenza_ili.py`) visible IN
the app as a proper, openable scenario, instead of only a script + committed files.
This module is PURE and importable without the web app (no FastAPI / DB import): it
builds the model_spec, points at the committed real dataset, and trains the model via
the app's own `model_trainer`. The DB insert that persists the scenario lives in
`main._seed_demo_scenarios`, which calls into here.

The dataset is the committed real EAWAG Swiss respiratory-virus wastewater series
(`scripts/trial_output/ch-influenza_dataset.csv`): weekly influenza-A activity plus
co-circulating RSV / influenza-B / SARS-CoV-2 and a seasonal (climatic) cycle. So the
demo trains a genuine model on genuine surveillance data, deterministically, offline.
"""
from __future__ import annotations

import os

# Stable id → the startup seed is idempotent (matches the app's usr-<12 hex> format).
DEMO_SCENARIO_ID = "usr-deadbeef0001"
DEMO_SCENARIO_NAME = "Influenza & environment — Switzerland (demo)"
DEMO_SCENARIO_QUERY = (
    "Influenza-like illness and its environmental / co-circulating drivers in "
    "Switzerland — trained on real EAWAG wastewater surveillance"
)
_DATASET_REL = os.path.join("scripts", "trial_output", "ch-influenza_dataset.csv")

_MODEL_SPEC_SCHEMA = "model_spec/1.0"

_OUTCOME = {
    "name": "Influenza-A activity",
    "machine_name": "flu_a_load",
    "task_type": "regression",
    "unit": "gc/day",
    "positive_class": None,
    "description": "Weekly influenza-A wastewater load (a validated influenza-activity signal).",
}
_FEATURES = [
    {"name": "Influenza-B activity", "machine_name": "flu_b_load", "dtype": "float",
     "role": "feature", "source": "public_api", "importance": "high"},
    {"name": "RSV activity", "machine_name": "rsv_load", "dtype": "float",
     "role": "feature", "source": "public_api", "importance": "medium"},
    {"name": "SARS-CoV-2 activity", "machine_name": "sars_cov2_load", "dtype": "float",
     "role": "feature", "source": "public_api", "importance": "medium"},
    {"name": "Seasonal cycle (sin)", "machine_name": "season_sin", "dtype": "float",
     "role": "feature", "source": "user", "importance": "high"},
    {"name": "Seasonal cycle (cos)", "machine_name": "season_cos", "dtype": "float",
     "role": "feature", "source": "user", "importance": "high"},
]
_ALGORITHM = {"family": "random_forest", "metric": "r2",
              "cv": {"strategy": "kfold", "folds": 5}}


def _data_template() -> dict:
    cols = [{"name": _OUTCOME["machine_name"], "role": "outcome", "dtype": "float",
             "required": True, "source": "public_api"}]
    for f in _FEATURES:
        cols.append({"name": f["machine_name"], "role": "feature", "dtype": f["dtype"],
                     "required": f.get("importance") == "high", "source": f["source"]})
    return {"columns": cols, "target_column": _OUTCOME["machine_name"], "datetime_column": "date"}


def demo_model_spec() -> dict:
    """The model_spec stored under scenario_settings.variables_json['model_spec']."""
    return {
        "schema": _MODEL_SPEC_SCHEMA,
        "version": 1,
        "outcome": dict(_OUTCOME),
        "features": [dict(f) for f in _FEATURES],
        "algorithm": dict(_ALGORITHM),
        "data_template": _data_template(),
        "epidemic_parameters": {"applicable": False, "disease": "Influenza", "params": {}},
        "provenance_index": {},
        "is_demo": True,
    }


def demo_variables_json() -> dict:
    """The full scenario_settings.variables_json for the demo (model_spec + variable list)."""
    return {
        "model_spec": demo_model_spec(),
        "outcome": dict(_OUTCOME),
        "predictor_variables": [dict(f) for f in _FEATURES],
        "is_demo": True,
    }


def dataset_path(repo_root: str) -> str:
    return os.path.join(repo_root, _DATASET_REL)


def load_dataset(repo_root: str):
    import pandas as pd
    df = pd.read_csv(dataset_path(repo_root))
    return df.dropna(subset=[_OUTCOME["machine_name"]]).reset_index(drop=True)


def train_demo(repo_root: str, n_trials: int = 20):
    """Load the committed real dataset and train via the app's real trainer.
    Returns (df, spec, result) where result is a `model_trainer.train_model` dict
    (including the fitted `pipeline`)."""
    import model_trainer
    df = load_dataset(repo_root)
    spec = demo_model_spec()
    result = model_trainer.train_model(df, spec, n_trials=n_trials)
    return df, spec, result
