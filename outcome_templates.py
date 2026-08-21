"""Predefined OUTCOME templates for the GESICA operational targets.

The notes emphasise "bien définir les outcomes": these are ready-made, well-specified
outcomes (emergency-department overload, incoming-call volume, call surge, bed occupancy)
that a user can apply to a scenario's model in one click, then upload a matching hospital
extract and train — instead of hand-defining the target each time. Pure/importable
(no DB, no FastAPI); `main` builds the data_template + persists.

Each template carries:
  - outcome     : name, machine_name, task_type, unit, positive_class
  - algorithm   : family + metric (+ quantile for the extremal/surge template)
  - features    : suggested predictor columns to include (source=user; the operator fills them)
  - alert_thresholds : optional monitor bands (regression), else None (classification auto-bands)
"""
from __future__ import annotations

from typing import Any

TEMPLATES: list[dict[str, Any]] = [
    {
        "id": "ed_overload",
        "name": "Surcharge des urgences",
        "description": "Prédire si le service d'urgence sera en surcharge (lits disponibles sous le seuil critique).",
        "outcome": {"name": "Surcharge urgences", "machine_name": "ed_overload",
                    "task_type": "classification", "unit": "", "positive_class": "surcharge"},
        "algorithm": {"family": "gradient_boosting", "metric": "average_precision"},
        "features": [
            {"name": "Lits disponibles", "machine_name": "available_beds", "dtype": "int"},
            {"name": "Taux d'occupation", "machine_name": "occupancy_rate", "dtype": "float"},
            {"name": "Passages aux urgences (24 h)", "machine_name": "ed_visits", "dtype": "int"},
            {"name": "Appels entrants (24 h)", "machine_name": "incoming_calls", "dtype": "int"},
        ],
        "alert_thresholds": None,
    },
    {
        "id": "bed_occupancy",
        "name": "Taux d'occupation des lits",
        "description": "Prédire le taux de remplissage des lits (0–100 %).",
        "outcome": {"name": "Taux d'occupation", "machine_name": "occupancy_rate",
                    "task_type": "regression", "unit": "%", "positive_class": None},
        "algorithm": {"family": "gradient_boosting", "metric": "rmse"},
        "features": [
            {"name": "Passages aux urgences (24 h)", "machine_name": "ed_visits", "dtype": "int"},
            {"name": "Admissions (24 h)", "machine_name": "admissions", "dtype": "int"},
            {"name": "Sorties (24 h)", "machine_name": "discharges", "dtype": "int"},
        ],
        "alert_thresholds": {
            "green": {"range": "< 85", "label": "Normal"},
            "orange": {"range": "85–95", "label": "Tendu"},
            "red": {"range": "> 95", "label": "Saturation"},
        },
    },
    {
        "id": "call_volume",
        "name": "Volume d'appels entrants",
        "description": "Prédire le nombre d'appels entrants au centre de régulation médicale.",
        "outcome": {"name": "Appels entrants", "machine_name": "incoming_calls",
                    "task_type": "count", "unit": "appels/jour", "positive_class": None},
        "algorithm": {"family": "gradient_boosting", "metric": "mae"},
        "features": [
            {"name": "Jour de la semaine", "machine_name": "day_of_week", "dtype": "category"},
            {"name": "Température moyenne", "machine_name": "temp_mean", "dtype": "float"},
            {"name": "Pollution (PM2.5)", "machine_name": "pm2_5", "dtype": "float"},
            {"name": "Incidence grippale (ILI)", "machine_name": "ili_incidence", "dtype": "float"},
        ],
        "alert_thresholds": None,
    },
    {
        "id": "call_surge",
        "name": "Pic d'appels (surcharge de la centrale)",
        "description": "Anticiper les PICS d'appels (quantile haut), pas la moyenne — via la forêt extrémale (extremal_rf).",
        "outcome": {"name": "Pic d'appels", "machine_name": "incoming_calls",
                    "task_type": "regression", "unit": "appels/jour", "positive_class": None},
        "algorithm": {"family": "extremal_rf", "metric": "pinball", "quantile": 0.9},
        "features": [
            {"name": "Jour de la semaine", "machine_name": "day_of_week", "dtype": "category"},
            {"name": "Température moyenne", "machine_name": "temp_mean", "dtype": "float"},
            {"name": "Pollution (PM2.5)", "machine_name": "pm2_5", "dtype": "float"},
            {"name": "Incidence grippale (ILI)", "machine_name": "ili_incidence", "dtype": "float"},
        ],
        "alert_thresholds": None,
    },
]

_BY_ID = {t["id"]: t for t in TEMPLATES}


def as_list() -> list[dict]:
    """JSON-safe catalogue for the picker (id, name, description, outcome, algorithm, feature names)."""
    return [
        {
            "id": t["id"], "name": t["name"], "description": t["description"],
            "outcome": dict(t["outcome"]), "algorithm": dict(t["algorithm"]),
            "features": [dict(f) for f in t["features"]],
        }
        for t in TEMPLATES
    ]


def get(template_id: str) -> dict | None:
    t = _BY_ID.get(str(template_id or "").strip())
    return {k: (v if not isinstance(v, list) else [dict(x) for x in v])
            for k, v in t.items()} if t else None
