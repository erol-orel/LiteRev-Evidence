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

# Paramètres épidémiologiques de la grippe SAISONNIÈRE, aux ordres de grandeur publiés
# et admis. Le bloc était auparavant `applicable: False` avec `params: {}` — de sorte que
# le SEIR de l'UNIQUE scénario livré (et qui porte sur la grippe) affichait « aucun modèle
# SEIR », c'est-à-dire exactement l'inverse de la démonstration voulue.
#
# HONNÊTETÉ : `provenance: []` et `n_studies: 0` sont VOLONTAIRES. Ces valeurs ne sont pas
# extraites du corpus de ce scénario (il n'y en a pas) : ce sont des valeurs de
# démonstration. On ne fabrique pas d'ids d'articles pour faire croire à une traçabilité —
# c'est précisément le travers que le reste de ce correctif supprime. `is_demo` le dit à
# l'UI, qui peut étiqueter la projection comme démonstrative.
def _demo_epidemic_parameters() -> dict:
    def _p(value, lo, hi, unit=""):
        return {"value": value, "ci_low": lo, "ci_high": hi, "unit": unit,
                "n_studies": 0, "provenance": []}
    return {
        "applicable": True,
        "disease": "Influenza (seasonal)",
        "is_demo": True,
        "note": ("Valeurs de démonstration : ordres de grandeur publiés pour la grippe "
                 "saisonnière, non extraits d'un corpus. Aucune provenance revendiquée."),
        "params": {
            "r0": _p(1.3, 1.2, 1.4),
            "incubation_period_days": _p(2.0, 1.4, 2.6, "days"),
            "infectious_period_days": _p(4.0, 3.0, 5.0, "days"),
            "cfr": _p(0.001, 0.0005, 0.002, "proportion"),
        },
        "cited": [],
    }


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
        "epidemic_parameters": _demo_epidemic_parameters(),
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
