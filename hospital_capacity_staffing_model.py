"""
hospital_capacity_staffing_model.py
Prévision Capacité Hospitalière & Staffing EMS
Couvre : hospital-capacity-forecasting + staffing-level-prediction
Basé sur : Jones BMJ (2009), Hoot AEM (2007), Asplin AEM (2003),
           Weiss AEM (2004) — NEDOCS, EDWIN scores
"""
from __future__ import annotations
import math
from datetime import datetime, timezone, timedelta
from typing import Any

# ─── Constantes ───────────────────────────────────────────────────────────────
TOTAL_ED_BEDS = 48        # HUG urgences
TOTAL_ICU_BEDS = 32       # HUG réanimation
TOTAL_WARD_BEDS = 320     # HUG médecine/chirurgie
TOTAL_EMS_CREWS = 12      # Équipes EMS disponibles Grand Genève

def _nedocs_score(
    total_ed_beds: int, occupied_ed_beds: int,
    patients_waiting: int, longest_wait_hours: float,
    admits_waiting: int, vent_patients: int,
) -> dict[str, Any]:
    """
    National Emergency Department Overcrowding Score (NEDOCS).
    Weiss SJ et al. Acad Emerg Med. 2004;11(1):38-50.
    Score 0-200 : 0-20 normal, 20-60 busy, 60-100 extremely busy, 100-140 overcrowded, >140 dangerously overcrowded.
    """
    if total_ed_beds == 0:
        return {"nedocs": 0, "level": "INCONNU"}

    score = (
        85.8 * (occupied_ed_beds / total_ed_beds)
        + 600 * (admits_waiting / total_ed_beds)
        + 13.4 * longest_wait_hours
        + 0.93 * patients_waiting
        + 5.64 * vent_patients
        - 20
    )
    score = max(0, round(score, 1))

    if score < 20: level, color = "NORMAL", "green"
    elif score < 60: level, color = "CHARGÉ", "yellow"
    elif score < 100: level, color = "TRÈS CHARGÉ", "orange"
    elif score < 140: level, color = "SATURÉ", "red"
    else: level, color = "DANGEREUSEMENT SATURÉ", "red"

    return {"nedocs": score, "level": level, "color": color}

def _predict_hourly_demand(hour_of_day: int, day_of_week: int, month: int) -> float:
    """
    Modèle de prévision de la demande horaire basé sur les patterns EMS.
    Basé sur : Lowthian Emerg Med J (2011), Hoot AEM (2007).
    """
    # Pattern horaire (pic 10h-12h et 18h-20h)
    hourly_pattern = [
        0.40, 0.35, 0.30, 0.28, 0.30, 0.38,  # 0-5h
        0.55, 0.75, 0.90, 1.05, 1.15, 1.20,  # 6-11h
        1.10, 1.05, 1.00, 1.00, 1.05, 1.15,  # 12-17h
        1.20, 1.15, 1.05, 0.90, 0.75, 0.55,  # 18-23h
    ]
    hourly_factor = hourly_pattern[hour_of_day % 24]

    # Pattern hebdomadaire (lundi=0, dimanche=6)
    weekly_pattern = [1.05, 1.00, 0.95, 0.95, 1.00, 1.10, 1.15]
    weekly_factor = weekly_pattern[day_of_week % 7]

    # Pattern mensuel (hiver = plus de demande)
    monthly_pattern = [1.20, 1.15, 1.05, 0.95, 0.90, 0.85, 0.85, 0.88, 0.92, 0.98, 1.08, 1.18]
    monthly_factor = monthly_pattern[(month - 1) % 12]

    base_demand = 4.5  # appels EMS/heure en moyenne
    return round(base_demand * hourly_factor * weekly_factor * monthly_factor, 1)

def _staffing_recommendation(predicted_demand: float, current_crews: int) -> dict[str, Any]:
    """Recommande le niveau de staffing EMS optimal."""
    # Ratio cible : 1 équipe pour 1.5 appels/heure (inclut temps de transport + dispo)
    required_crews = math.ceil(predicted_demand / 1.5)
    delta = required_crews - current_crews

    if delta <= 0:
        status = "SUFFISANT"
        color = "green"
        action = f"Staffing actuel ({current_crews} équipes) suffisant pour la demande prévue."
    elif delta == 1:
        status = "LÉGÈREMENT INSUFFISANT"
        color = "yellow"
        action = f"Rappeler 1 équipe supplémentaire (total recommandé : {required_crews})."
    elif delta <= 3:
        status = "INSUFFISANT"
        color = "orange"
        action = f"Rappeler {delta} équipes supplémentaires (total recommandé : {required_crews}). Activer le plan de renfort."
    else:
        status = "CRITIQUE"
        color = "red"
        action = f"Déficit de {delta} équipes. Activer le plan ORSEC EMS. Demander renforts interdépartementaux."

    return {
        "required_crews": required_crews,
        "current_crews": current_crews,
        "delta": delta,
        "status": status,
        "color": color,
        "action": action,
    }

def forecast_capacity_and_staffing(
    current_ed_occupancy_pct: float = 75.0,
    current_icu_occupancy_pct: float = 68.0,
    patients_waiting: int = 12,
    longest_wait_hours: float = 2.5,
    admits_waiting: int = 4,
    vent_patients: int = 2,
    current_ems_crews: int = 10,
) -> dict[str, Any]:
    """Prévision complète capacité + staffing pour les 12 prochaines heures."""
    now = datetime.now(timezone.utc)

    # NEDOCS actuel
    occupied_ed = int(TOTAL_ED_BEDS * current_ed_occupancy_pct / 100)
    nedocs = _nedocs_score(TOTAL_ED_BEDS, occupied_ed, patients_waiting, longest_wait_hours, admits_waiting, vent_patients)

    # Prévision horaire sur 12h
    hourly_forecast = []
    for h in range(12):
        future = now + timedelta(hours=h)
        demand = _predict_hourly_demand(future.hour, future.weekday(), future.month)
        staffing = _staffing_recommendation(demand, current_ems_crews)
        hourly_forecast.append({
            "hour": future.strftime("%H:%M"),
            "predicted_calls_per_hour": demand,
            "required_crews": staffing["required_crews"],
            "staffing_status": staffing["status"],
            "staffing_color": staffing["color"],
        })

    # Pic de demande sur 12h
    peak_hour = max(hourly_forecast, key=lambda x: x["predicted_calls_per_hour"])
    max_deficit = max(hourly_forecast, key=lambda x: x["required_crews"] - current_ems_crews)

    # Saturation hospitalière
    if current_ed_occupancy_pct >= 100:
        hosp_status = "SATURATION COMPLÈTE — Déviation des ambulances requise"
        hosp_color = "red"
    elif current_ed_occupancy_pct >= 85:
        hosp_status = "QUASI-SATURATION — Anticipation de déviation"
        hosp_color = "orange"
    elif current_ed_occupancy_pct >= 70:
        hosp_status = "CHARGÉ — Surveillance renforcée"
        hosp_color = "yellow"
    else:
        hosp_status = "CAPACITÉ NORMALE"
        hosp_color = "green"

    return {
        "model": "HospitalCapacityStaffing v1.0",
        "status": "live",
        "generated_at": now.isoformat(),
        "current_status": {
            "ed_occupancy_pct": current_ed_occupancy_pct,
            "icu_occupancy_pct": current_icu_occupancy_pct,
            "nedocs_score": nedocs["nedocs"],
            "nedocs_level": nedocs["level"],
            "nedocs_color": nedocs["color"],
            "hospital_status": hosp_status,
            "hospital_color": hosp_color,
        },
        "staffing_now": _staffing_recommendation(
            _predict_hourly_demand(now.hour, now.weekday(), now.month),
            current_ems_crews
        ),
        "12h_forecast": hourly_forecast,
        "peak_demand": {"hour": peak_hour["hour"], "calls_per_hour": peak_hour["predicted_calls_per_hour"]},
        "max_staffing_deficit": {
            "hour": max_deficit["hour"],
            "required": max_deficit["required_crews"],
            "current": current_ems_crews,
            "deficit": max(0, max_deficit["required_crews"] - current_ems_crews),
        },
        "evidence_base": [
            "NEDOCS — Weiss SJ et al. Acad Emerg Med (2004)",
            "Hoot NR et al. Acad Emerg Med (2007) — ED forecasting",
            "Lowthian JA et al. Emerg Med J (2011) — EMS demand prediction",
            "Jones SS et al. BMJ (2009) — Hospital capacity modelling",
        ]
    }


class HospitalCapacityStaffingModel:
    def predict_demo(self) -> dict[str, Any]:
        return forecast_capacity_and_staffing(
            current_ed_occupancy_pct=88.0,
            current_icu_occupancy_pct=75.0,
            patients_waiting=18,
            longest_wait_hours=3.5,
            admits_waiting=6,
            vent_patients=3,
            current_ems_crews=9,
        )


hospital_capacity_model_singleton = HospitalCapacityStaffingModel()
