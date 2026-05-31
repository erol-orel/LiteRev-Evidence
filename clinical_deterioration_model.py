"""
clinical_deterioration_model.py
Prédiction de la Détérioration Clinique en Transit (MEWS/NEWS2 + Tendance temporelle)
Basé sur : Smith (2001) MEWS, Royal College of Physicians NEWS2 (2017),
           Churpek AUROC NEWS2 (2017), Alam JAMA (2014)
"""
from __future__ import annotations
import math
from datetime import datetime, timezone
from typing import Any

# ─── Paramètres NEWS2 ─────────────────────────────────────────────────────────
# Chaque paramètre physiologique est scoré de 0 à 3 selon les seuils NEWS2 officiels

def _score_rr(rr: float) -> int:
    """Fréquence respiratoire (cycles/min)"""
    if rr <= 8: return 3
    if rr <= 11: return 1
    if rr <= 20: return 0
    if rr <= 24: return 2
    return 3

def _score_spo2_scale1(spo2: float) -> int:
    """SpO2 (%) — Échelle 1 (patients sans BPCO)"""
    if spo2 <= 91: return 3
    if spo2 <= 93: return 2
    if spo2 <= 95: return 1
    return 0

def _score_spo2_scale2(spo2: float, on_oxygen: bool) -> int:
    """SpO2 (%) — Échelle 2 (patients BPCO, cible 88-92%)"""
    if spo2 <= 83: return 3
    if spo2 <= 85: return 2
    if spo2 <= 87: return 1
    if spo2 <= 92: return 0 if not on_oxygen else 0
    if spo2 <= 94: return 1 if on_oxygen else 0
    if spo2 <= 96: return 2 if on_oxygen else 0
    return 3 if on_oxygen else 0

def _score_air_or_oxygen(on_oxygen: bool) -> int:
    return 2 if on_oxygen else 0

def _score_sbp(sbp: float) -> int:
    """Pression artérielle systolique (mmHg)"""
    if sbp <= 90: return 3
    if sbp <= 100: return 2
    if sbp <= 110: return 1
    if sbp <= 219: return 0
    return 3

def _score_hr(hr: float) -> int:
    """Fréquence cardiaque (bpm)"""
    if hr <= 40: return 3
    if hr <= 50: return 1
    if hr <= 90: return 0
    if hr <= 110: return 1
    if hr <= 130: return 2
    return 3

def _score_consciousness(avpu: str) -> int:
    """Niveau de conscience AVPU"""
    avpu_upper = avpu.upper().strip()
    if avpu_upper in ("A", "ALERT", "ALERTE"): return 0
    if avpu_upper in ("C", "CONFUSED", "CONFUS", "NEW CONFUSION"): return 3
    if avpu_upper in ("V", "VOICE", "VERBAL"): return 3
    if avpu_upper in ("P", "PAIN", "DOULEUR"): return 3
    if avpu_upper in ("U", "UNRESPONSIVE", "INCONSCIENT"): return 3
    return 0

def _score_temperature(temp: float) -> int:
    """Température corporelle (°C)"""
    if temp <= 35.0: return 3
    if temp <= 36.0: return 1
    if temp <= 38.0: return 0
    if temp <= 39.0: return 1
    return 2

def compute_news2(
    rr: float, spo2: float, on_oxygen: bool, sbp: float,
    hr: float, avpu: str, temp: float, copd: bool = False
) -> dict[str, Any]:
    """Calcule le score NEWS2 complet et le niveau de risque."""
    s_rr = _score_rr(rr)
    s_spo2 = _score_spo2_scale2(spo2, on_oxygen) if copd else _score_spo2_scale1(spo2)
    s_o2 = _score_air_or_oxygen(on_oxygen)
    s_sbp = _score_sbp(sbp)
    s_hr = _score_hr(hr)
    s_avpu = _score_consciousness(avpu)
    s_temp = _score_temperature(temp)

    total = s_rr + s_spo2 + s_o2 + s_sbp + s_hr + s_avpu + s_temp
    any_3 = any(s == 3 for s in [s_rr, s_spo2, s_sbp, s_hr, s_avpu, s_temp])

    if total <= 4 and not any_3:
        risk = "FAIBLE"
        risk_color = "green"
        action = "Surveillance standard. Réévaluation toutes les 4-6h."
        monitoring_freq = "4-6h"
    elif total <= 6 or any_3:
        risk = "MODÉRÉ"
        risk_color = "orange"
        action = "Alerte médicale urgente. Réévaluation toutes les 30-60 min. Considérer transfert en soins intensifs."
        monitoring_freq = "30-60 min"
    else:
        risk = "ÉLEVÉ"
        risk_color = "red"
        action = "URGENCE MÉDICALE. Appel équipe de réanimation immédiat. Transfert en réanimation."
        monitoring_freq = "Continu"

    return {
        "total_score": total,
        "risk_level": risk,
        "risk_color": risk_color,
        "action": action,
        "monitoring_frequency": monitoring_freq,
        "any_single_extreme": any_3,
        "subscores": {
            "respiratory_rate": {"value": rr, "score": s_rr},
            "spo2": {"value": spo2, "score": s_spo2, "scale": 2 if copd else 1},
            "supplemental_oxygen": {"value": on_oxygen, "score": s_o2},
            "systolic_bp": {"value": sbp, "score": s_sbp},
            "heart_rate": {"value": hr, "score": s_hr},
            "consciousness": {"value": avpu, "score": s_avpu},
            "temperature": {"value": temp, "score": s_temp},
        }
    }

def compute_mews(sbp: float, hr: float, rr: float, temp: float, avpu: str) -> dict[str, Any]:
    """Modified Early Warning Score (MEWS) — version simplifiée préhospitalière."""
    def s_sbp_mews(v):
        if v <= 70: return 3
        if v <= 80: return 2
        if v <= 100: return 1
        if v <= 199: return 0
        return 2

    def s_hr_mews(v):
        if v < 40: return 2
        if v <= 50: return 1
        if v <= 100: return 0
        if v <= 110: return 1
        if v <= 129: return 2
        return 3

    def s_rr_mews(v):
        if v < 9: return 2
        if v <= 14: return 0
        if v <= 20: return 1
        if v <= 29: return 2
        return 3

    def s_temp_mews(v):
        if v < 35: return 2
        if v <= 38.4: return 0
        return 2

    def s_avpu_mews(v):
        v = v.upper().strip()
        if v in ("A", "ALERT"): return 0
        if v in ("V", "VOICE"): return 1
        if v in ("P", "PAIN"): return 2
        return 3

    total = s_sbp_mews(sbp) + s_hr_mews(hr) + s_rr_mews(rr) + s_temp_mews(temp) + s_avpu_mews(avpu)

    if total <= 2:
        risk = "FAIBLE"; color = "green"
    elif total <= 4:
        risk = "MODÉRÉ"; color = "orange"
    else:
        risk = "ÉLEVÉ"; color = "red"

    return {"mews_score": total, "risk_level": risk, "risk_color": color}

def _trend_analysis(vitals_series: list[dict]) -> dict[str, Any]:
    """
    Analyse la tendance des constantes sur les N dernières mesures.
    Détecte une aggravation progressive même si chaque mesure individuelle est normale.
    Basé sur : Alam JAMA (2014) — 'Continuous vital sign monitoring improves detection of deterioration'.
    """
    if len(vitals_series) < 2:
        return {"trend": "INSUFFISANT", "delta_news2": 0, "deterioration_velocity": 0}

    scores = []
    for v in vitals_series:
        r = compute_news2(
            rr=v.get("rr", 16), spo2=v.get("spo2", 98),
            on_oxygen=v.get("on_oxygen", False), sbp=v.get("sbp", 120),
            hr=v.get("hr", 75), avpu=v.get("avpu", "A"), temp=v.get("temp", 37.0)
        )
        scores.append(r["total_score"])

    delta = scores[-1] - scores[0]
    n = len(scores)
    # Vitesse de détérioration (points NEWS2 par mesure)
    velocity = delta / max(n - 1, 1)

    if delta >= 3 or velocity >= 1.5:
        trend = "DÉGRADATION RAPIDE"
        trend_color = "red"
    elif delta >= 1:
        trend = "DÉGRADATION PROGRESSIVE"
        trend_color = "orange"
    elif delta <= -2:
        trend = "AMÉLIORATION"
        trend_color = "green"
    else:
        trend = "STABLE"
        trend_color = "green"

    return {
        "trend": trend,
        "trend_color": trend_color,
        "delta_news2": delta,
        "deterioration_velocity": round(velocity, 2),
        "score_history": scores,
        "n_measurements": n,
    }

def predict_deterioration(
    current_vitals: dict[str, Any],
    vitals_history: list[dict] | None = None,
    transport_duration_min: float = 15.0,
    copd: bool = False,
) -> dict[str, Any]:
    """
    Prédiction complète de détérioration clinique en transit.
    Combine NEWS2, MEWS et analyse de tendance.
    """
    rr = current_vitals.get("rr", 16.0)
    spo2 = current_vitals.get("spo2", 98.0)
    on_oxygen = current_vitals.get("on_oxygen", False)
    sbp = current_vitals.get("sbp", 120.0)
    hr = current_vitals.get("hr", 75.0)
    avpu = current_vitals.get("avpu", "A")
    temp = current_vitals.get("temp", 37.0)

    news2 = compute_news2(rr, spo2, on_oxygen, sbp, hr, avpu, temp, copd)
    mews = compute_mews(sbp, hr, rr, temp, avpu)

    # Analyse de tendance si historique disponible
    history = vitals_history or []
    all_vitals = history + [current_vitals]
    trend = _trend_analysis(all_vitals)

    # Score composite de risque (0-100)
    news2_normalized = min(news2["total_score"] / 20.0, 1.0) * 60
    mews_normalized = min(mews["mews_score"] / 14.0, 1.0) * 20
    trend_bonus = 20 if trend["trend"] == "DÉGRADATION RAPIDE" else \
                  10 if trend["trend"] == "DÉGRADATION PROGRESSIVE" else 0
    composite_risk = round(news2_normalized + mews_normalized + trend_bonus, 1)

    # Prédiction de l'état à l'arrivée (projection linéaire simple)
    projected_score = news2["total_score"] + trend["deterioration_velocity"] * (transport_duration_min / 5)
    projected_score = max(0, min(20, projected_score))

    if projected_score <= 4:
        projected_risk = "FAIBLE"
    elif projected_score <= 6:
        projected_risk = "MODÉRÉ"
    else:
        projected_risk = "ÉLEVÉ"

    # Recommandations
    recommendations = []
    if news2["total_score"] >= 7:
        recommendations.append("Activation de l'équipe de réanimation à l'arrivée (pre-alert)")
    if news2["subscores"]["spo2"]["score"] >= 2:
        recommendations.append("Optimiser l'oxygénothérapie — cible SpO2 ≥ 94%")
    if news2["subscores"]["systolic_bp"]["score"] >= 2:
        recommendations.append("Remplissage vasculaire — surveiller la réponse tensionnelle")
    if trend["trend"] in ("DÉGRADATION RAPIDE", "DÉGRADATION PROGRESSIVE"):
        recommendations.append(f"Tendance à la dégradation détectée (+{trend['delta_news2']} pts NEWS2) — accélérer le transport")
    if transport_duration_min > 20 and news2["risk_level"] == "ÉLEVÉ":
        recommendations.append("Durée de transport > 20 min avec risque élevé — envisager renfort SMUR ou hélitreuillage")
    if not recommendations:
        recommendations.append("Patient stable — surveillance standard en transit")

    return {
        "model": "ClinicalDeteriorationPredictor v1.0",
        "status": "live",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "current_assessment": {
            "news2": news2,
            "mews": mews,
            "composite_risk_score": composite_risk,
            "overall_risk": news2["risk_level"],
        },
        "trend_analysis": trend,
        "transport_projection": {
            "duration_min": transport_duration_min,
            "projected_news2_at_arrival": round(projected_score, 1),
            "projected_risk_at_arrival": projected_risk,
        },
        "recommendations": recommendations,
        "evidence_base": [
            "NEWS2 — Royal College of Physicians (2017)",
            "MEWS — Morgan & Williams, Anaesthesia (1997)",
            "Churpek et al. JAMA (2016) — Rapid detection of deterioration",
            "Alam et al. JAMA (2014) — Continuous vital sign monitoring",
        ]
    }


# ─── Singleton pour l'API ─────────────────────────────────────────────────────
class ClinicalDeteriorationModel:
    """Wrapper singleton pour l'endpoint FastAPI."""

    def predict_demo(self) -> dict[str, Any]:
        """Cas cliniques de démonstration couvrant les 3 niveaux de risque."""
        cases = [
            {
                "name": "Patient 1 — Détresse respiratoire (ÉLEVÉ)",
                "vitals": {"rr": 28, "spo2": 89, "on_oxygen": True, "sbp": 95, "hr": 118, "avpu": "V", "temp": 38.8},
                "history": [
                    {"rr": 18, "spo2": 95, "on_oxygen": False, "sbp": 115, "hr": 88, "avpu": "A", "temp": 37.5},
                    {"rr": 22, "spo2": 92, "on_oxygen": True, "sbp": 105, "hr": 102, "avpu": "A", "temp": 38.2},
                ],
                "transport_min": 18.0,
            },
            {
                "name": "Patient 2 — Douleur thoracique stable (MODÉRÉ)",
                "vitals": {"rr": 18, "spo2": 96, "on_oxygen": False, "sbp": 145, "hr": 95, "avpu": "A", "temp": 37.2},
                "history": [],
                "transport_min": 12.0,
            },
            {
                "name": "Patient 3 — Malaise vagal (FAIBLE)",
                "vitals": {"rr": 14, "spo2": 99, "on_oxygen": False, "sbp": 118, "hr": 62, "avpu": "A", "temp": 36.8},
                "history": [],
                "transport_min": 8.0,
            },
        ]

        results = []
        for case in cases:
            pred = predict_deterioration(
                current_vitals=case["vitals"],
                vitals_history=case["history"],
                transport_duration_min=case["transport_min"],
            )
            results.append({
                "case_name": case["name"],
                "news2_score": pred["current_assessment"]["news2"]["total_score"],
                "risk_level": pred["current_assessment"]["overall_risk"],
                "composite_risk": pred["current_assessment"]["composite_risk_score"],
                "trend": pred["trend_analysis"]["trend"],
                "projected_risk_arrival": pred["transport_projection"]["projected_risk_at_arrival"],
                "recommendations": pred["recommendations"],
                "subscores": pred["current_assessment"]["news2"]["subscores"],
            })

        high_risk = sum(1 for r in results if r["risk_level"] == "ÉLEVÉ")
        moderate_risk = sum(1 for r in results if r["risk_level"] == "MODÉRÉ")

        return {
            "model": "ClinicalDeteriorationPredictor v1.0",
            "status": "live",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "total_patients": len(results),
                "high_risk": high_risk,
                "moderate_risk": moderate_risk,
                "low_risk": len(results) - high_risk - moderate_risk,
                "pre_alert_required": high_risk,
            },
            "cases": results,
            "evidence_base": [
                "NEWS2 — Royal College of Physicians (2017)",
                "MEWS — Morgan & Williams (1997)",
                "Churpek JAMA (2016) — Rapid detection of deterioration",
            ]
        }


clinical_deterioration_model_singleton = ClinicalDeteriorationModel()
