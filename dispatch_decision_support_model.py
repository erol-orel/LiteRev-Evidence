"""
dispatch_decision_support_model.py
Aide à la Décision pour le Dispatch EMS — Allocation optimale des ressources
Couvre : dispatch-decision-support
Basé sur : Lerner (2012), Nicholl BMJ (2007), Pons Prehosp Emerg Care (2005),
           Blackwell AEM (2002) — Tiered response systems
"""
from __future__ import annotations
import math
from datetime import datetime, timezone
from typing import Any

# ─── Ressources EMS disponibles Grand Genève ─────────────────────────────────
EMS_RESOURCES = [
    {"id": "SMUR-HUG-01", "type": "SMUR", "base": "HUG Genève", "lat": 46.1907, "lon": 6.1464, "available": True, "crew": ["médecin", "infirmier IADE"]},
    {"id": "SMUR-HUG-02", "type": "SMUR", "base": "HUG Genève", "lat": 46.1907, "lon": 6.1464, "available": True, "crew": ["médecin", "infirmier IADE"]},
    {"id": "SMUR-CHUV-01", "type": "SMUR", "base": "CHUV Lausanne", "lat": 46.5218, "lon": 6.6343, "available": True, "crew": ["médecin", "infirmier IADE"]},
    {"id": "HELI-REGA-01", "type": "HÉLICOPTÈRE", "base": "Base REGA Genève", "lat": 46.2381, "lon": 6.1089, "available": True, "crew": ["médecin urgentiste", "pilote", "secouriste"]},
    {"id": "AMB-SIS-01", "type": "AMBULANCE", "base": "SIS Genève", "lat": 46.2044, "lon": 6.1432, "available": True, "crew": ["ambulancier diplômé", "ambulancier"]},
    {"id": "AMB-SIS-02", "type": "AMBULANCE", "base": "SIS Genève", "lat": 46.2044, "lon": 6.1432, "available": True, "crew": ["ambulancier diplômé", "ambulancier"]},
    {"id": "AMB-SDIS74-01", "type": "AMBULANCE", "base": "SDIS 74 Annemasse", "lat": 46.1933, "lon": 6.2356, "available": True, "crew": ["ambulancier", "secouriste"]},
    {"id": "AMB-SDIS74-02", "type": "AMBULANCE", "base": "SDIS 74 Thonon", "lat": 46.3700, "lon": 6.4800, "available": True, "crew": ["ambulancier", "secouriste"]},
    {"id": "VSL-01", "type": "VSL", "base": "Genève Centre", "lat": 46.2044, "lon": 6.1432, "available": True, "crew": ["conducteur VSL"]},
    {"id": "MEDIC-01", "type": "MÉDECIN_LIBÉRAL", "base": "Cabinet Carouge", "lat": 46.1833, "lon": 6.1500, "available": True, "crew": ["médecin généraliste"]},
]

# ─── Matrice de décision dispatch ────────────────────────────────────────────
# (priority, category) → liste ordonnée de types de ressources à envoyer
DISPATCH_MATRIX: dict[tuple[int, str], list[str]] = {
    (5, "cardiac-arrest"): ["SMUR", "AMBULANCE", "HÉLICOPTÈRE"],
    (5, "respiratory-arrest"): ["SMUR", "AMBULANCE"],
    (5, "hemorrhage"): ["SMUR", "AMBULANCE"],
    (5, "major-trauma"): ["SMUR", "AMBULANCE", "HÉLICOPTÈRE"],
    (5, "drowning"): ["SMUR", "AMBULANCE", "HÉLICOPTÈRE"],
    (5, "hanging"): ["SMUR", "AMBULANCE"],
    (4, "chest-pain"): ["SMUR", "AMBULANCE"],
    (4, "stroke"): ["SMUR", "AMBULANCE"],
    (4, "respiratory-distress"): ["SMUR", "AMBULANCE"],
    (4, "seizure"): ["SMUR", "AMBULANCE"],
    (4, "unconscious"): ["SMUR", "AMBULANCE"],
    (4, "anaphylaxis"): ["SMUR", "AMBULANCE"],
    (4, "obstetric"): ["SMUR", "AMBULANCE"],
    (3, "fracture"): ["AMBULANCE"],
    (3, "burns"): ["AMBULANCE"],
    (3, "hypoglycemia"): ["AMBULANCE"],
    (3, "abdominal-pain"): ["AMBULANCE"],
    (3, "fever"): ["AMBULANCE"],
    (3, "syncope"): ["AMBULANCE"],
    (3, "poisoning"): ["SMUR", "AMBULANCE"],
    (2, "minor-trauma"): ["VSL"],
    (2, "gastro"): ["MÉDECIN_LIBÉRAL"],
    (2, "headache"): ["MÉDECIN_LIBÉRAL"],
    (1, "advice"): [],
}

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def _estimate_travel_time_min(dist_km: float, resource_type: str) -> float:
    """Estime le temps de trajet selon le type de ressource."""
    speeds = {"HÉLICOPTÈRE": 200.0, "SMUR": 60.0, "AMBULANCE": 55.0, "VSL": 50.0, "MÉDECIN_LIBÉRAL": 40.0}
    speed = speeds.get(resource_type, 50.0)
    mobilization = {"HÉLICOPTÈRE": 5.0, "SMUR": 2.0, "AMBULANCE": 1.5, "VSL": 3.0, "MÉDECIN_LIBÉRAL": 5.0}
    mob = mobilization.get(resource_type, 2.0)
    return mob + (dist_km / speed * 60)

def recommend_dispatch(
    incident_lat: float, incident_lon: float,
    priority: int, category: str,
    available_resources: list[dict] | None = None,
) -> dict[str, Any]:
    """Recommande les ressources optimales pour un incident donné."""
    resources = available_resources or EMS_RESOURCES
    required_types = DISPATCH_MATRIX.get((priority, category),
                     DISPATCH_MATRIX.get((priority, "unclassified"), ["AMBULANCE"]))

    if not required_types:
        return {
            "dispatch_required": False,
            "message": "Conseil médical téléphonique suffisant — aucun vecteur requis",
            "priority": priority,
            "category": category,
        }

    dispatched = []
    for req_type in required_types:
        candidates = [r for r in resources if r["type"] == req_type and r["available"]]
        if not candidates:
            continue
        # Trier par distance à l'incident
        for c in candidates:
            c["_dist_km"] = _haversine_km(c["lat"], c["lon"], incident_lat, incident_lon)
            c["_eta_min"] = _estimate_travel_time_min(c["_dist_km"], c["type"])
        best = min(candidates, key=lambda x: x["_eta_min"])
        dispatched.append({
            "resource_id": best["id"],
            "type": best["type"],
            "base": best["base"],
            "distance_km": round(best["_dist_km"], 1),
            "eta_min": round(best["_eta_min"], 1),
            "crew": best["crew"],
        })

    total_eta = max((d["eta_min"] for d in dispatched), default=0)

    # Évaluation de la performance
    target_times = {5: 8, 4: 12, 3: 20, 2: 30, 1: 60}
    target = target_times.get(priority, 20)
    performance = "DANS LES DÉLAIS" if total_eta <= target else "DÉLAI DÉPASSÉ"
    performance_color = "green" if total_eta <= target else "red"

    return {
        "dispatch_required": True,
        "priority": priority,
        "category": category,
        "dispatched_resources": dispatched,
        "total_resources_dispatched": len(dispatched),
        "estimated_first_arrival_min": total_eta,
        "target_response_time_min": target,
        "performance": performance,
        "performance_color": performance_color,
    }


class DispatchDecisionSupportModel:
    def predict_demo(self) -> dict[str, Any]:
        incidents = [
            {"name": "Arrêt cardiaque — Centre-ville Genève", "lat": 46.2044, "lon": 6.1432, "priority": 5, "category": "cardiac-arrest"},
            {"name": "AVC — Annemasse", "lat": 46.1933, "lon": 6.2356, "priority": 4, "category": "stroke"},
            {"name": "Fracture — Carouge", "lat": 46.1833, "lon": 6.1500, "priority": 3, "category": "fracture"},
            {"name": "Malaise vagal — Thonon", "lat": 46.3700, "lon": 6.4800, "priority": 2, "category": "syncope"},
        ]

        results = []
        for inc in incidents:
            rec = recommend_dispatch(inc["lat"], inc["lon"], inc["priority"], inc["category"])
            rec["incident_name"] = inc["name"]
            results.append(rec)

        on_time = sum(1 for r in results if r.get("performance") == "DANS LES DÉLAIS")

        return {
            "model": "DispatchDecisionSupport v1.0",
            "status": "live",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "total_incidents": len(incidents),
                "on_time_responses": on_time,
                "delayed_responses": len(incidents) - on_time,
                "performance_rate_pct": round(on_time / len(incidents) * 100, 1),
            },
            "incidents": results,
            "evidence_base": [
                "Lerner Prehosp Emerg Care (2012) — Dispatch triage",
                "Nicholl BMJ (2007) — Tiered EMS response",
                "Blackwell AEM (2002) — Dispatch accuracy",
            ]
        }


dispatch_model_singleton = DispatchDecisionSupportModel()
