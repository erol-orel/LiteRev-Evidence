"""
patient_pathway_optimization_model.py
Optimisation du Parcours Patient — Orientation vers le bon établissement
Couvre : patient-pathway-optimization
Basé sur : Nicholl BMJ (2007), Sasser WHO (2005), Lerner (2012),
           Pons Prehosp Emerg Care (2005) — Bypass protocols
"""
from __future__ import annotations
import math
from datetime import datetime, timezone
from typing import Any

# ─── Hôpitaux Grand Genève ────────────────────────────────────────────────────
HOSPITALS = [
    {
        "id": "HUG", "name": "Hôpitaux Universitaires de Genève",
        "lat": 46.1907, "lon": 6.1464, "country": "CH",
        "level": "TRAUMA_CENTER_1", "specialties": ["cardiac", "stroke", "trauma", "burns", "neuro", "pediatric", "obstetric"],
        "cath_lab": True, "stroke_unit": True, "trauma_bay": True, "helipad": True,
        "current_capacity_pct": 78,
    },
    {
        "id": "CHUV", "name": "Centre Hospitalier Universitaire Vaudois",
        "lat": 46.5218, "lon": 6.6343, "country": "CH",
        "level": "TRAUMA_CENTER_1", "specialties": ["cardiac", "stroke", "trauma", "burns", "neuro", "pediatric"],
        "cath_lab": True, "stroke_unit": True, "trauma_bay": True, "helipad": True,
        "current_capacity_pct": 72,
    },
    {
        "id": "HCR", "name": "Hôpital de la Tour (Meyrin)",
        "lat": 46.2381, "lon": 6.0889, "country": "CH",
        "level": "REGIONAL", "specialties": ["cardiac", "stroke", "trauma"],
        "cath_lab": True, "stroke_unit": True, "trauma_bay": False, "helipad": False,
        "current_capacity_pct": 55,
    },
    {
        "id": "ANNEMASSE", "name": "Centre Hospitalier Alpes-Léman (Annemasse)",
        "lat": 46.1933, "lon": 6.2356, "country": "FR",
        "level": "REGIONAL", "specialties": ["trauma", "obstetric", "pediatric"],
        "cath_lab": False, "stroke_unit": False, "trauma_bay": True, "helipad": False,
        "current_capacity_pct": 82,
    },
    {
        "id": "THONON", "name": "Hôpital Intercommunal Pays du Léman (Thonon)",
        "lat": 46.3700, "lon": 6.4800, "country": "FR",
        "level": "REGIONAL", "specialties": ["trauma", "obstetric"],
        "cath_lab": False, "stroke_unit": False, "trauma_bay": False, "helipad": False,
        "current_capacity_pct": 65,
    },
    {
        "id": "SAINTGENIS", "name": "Hôpital de Saint-Genis-Pouilly",
        "lat": 46.2433, "lon": 6.0233, "country": "FR",
        "level": "LOCAL", "specialties": ["trauma"],
        "cath_lab": False, "stroke_unit": False, "trauma_bay": False, "helipad": False,
        "current_capacity_pct": 40,
    },
]

def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def _travel_time_min(dist_km: float, cross_border: bool = False) -> float:
    base = 2.0 + dist_km / 55.0 * 60
    if cross_border:
        base += 3.0  # délai douane
    return round(base, 1)

def _hospital_score(hospital: dict, required_specialties: list[str], dist_km: float, cross_border: bool) -> float:
    """Score composite pour un hôpital donné (plus élevé = meilleur)."""
    # Capacité disponible (0-40 pts)
    capacity_score = (100 - hospital["current_capacity_pct"]) / 100 * 40

    # Spécialités disponibles (0-30 pts)
    matched = sum(1 for s in required_specialties if s in hospital["specialties"])
    specialty_score = (matched / max(len(required_specialties), 1)) * 30

    # Niveau de l'hôpital (0-20 pts)
    level_scores = {"TRAUMA_CENTER_1": 20, "REGIONAL": 12, "LOCAL": 5}
    level_score = level_scores.get(hospital["level"], 5)

    # Pénalité distance (0-10 pts perdu)
    dist_penalty = min(10, dist_km / 5)

    # Pénalité frontière
    border_penalty = 5 if cross_border else 0

    return capacity_score + specialty_score + level_score - dist_penalty - border_penalty

def optimize_patient_pathway(
    incident_lat: float, incident_lon: float,
    clinical_category: str,
    priority: int,
    incident_country: str = "CH",
) -> dict[str, Any]:
    """Recommande le meilleur hôpital de destination selon la pathologie et la disponibilité."""

    # Spécialités requises par catégorie
    specialty_map = {
        "cardiac-arrest": ["cardiac"], "chest-pain": ["cardiac"],
        "stroke": ["stroke", "neuro"], "major-trauma": ["trauma"],
        "burns": ["burns"], "obstetric": ["obstetric"],
        "pediatric": ["pediatric"], "respiratory-distress": ["cardiac"],
        "hemorrhage": ["trauma"], "poisoning": ["neuro"],
    }
    required = specialty_map.get(clinical_category, ["trauma"])

    # Contraintes critiques (bypass si non disponible)
    critical_requirements = {
        "cardiac-arrest": "cath_lab",
        "chest-pain": "cath_lab",
        "stroke": "stroke_unit",
        "major-trauma": "trauma_bay",
    }
    critical_req = critical_requirements.get(clinical_category)

    scored_hospitals = []
    for h in HOSPITALS:
        dist = _haversine_km(incident_lat, incident_lon, h["lat"], h["lon"])
        cross_border = (h["country"] != incident_country)
        eta = _travel_time_min(dist, cross_border)

        # Bypass si contrainte critique non satisfaite
        if critical_req and not h.get(critical_req, False):
            bypass_reason = f"Bypass — {critical_req} non disponible"
            scored_hospitals.append({
                "hospital_id": h["id"], "hospital_name": h["name"],
                "country": h["country"], "level": h["level"],
                "distance_km": round(dist, 1), "eta_min": eta,
                "cross_border": cross_border, "score": 0,
                "recommended": False, "bypass_reason": bypass_reason,
                "capacity_pct": h["current_capacity_pct"],
            })
            continue

        score = _hospital_score(h, required, dist, cross_border)
        scored_hospitals.append({
            "hospital_id": h["id"], "hospital_name": h["name"],
            "country": h["country"], "level": h["level"],
            "distance_km": round(dist, 1), "eta_min": eta,
            "cross_border": cross_border, "score": round(score, 1),
            "recommended": False, "bypass_reason": None,
            "capacity_pct": h["current_capacity_pct"],
            "has_required_specialty": all(s in h["specialties"] for s in required),
        })

    # Trier par score décroissant
    eligible = [h for h in scored_hospitals if h["score"] > 0]
    eligible.sort(key=lambda x: -x["score"])

    if eligible:
        eligible[0]["recommended"] = True
        best = eligible[0]
    else:
        best = min(scored_hospitals, key=lambda x: x["eta_min"])
        best["recommended"] = True

    return {
        "model": "PatientPathwayOptimization v1.0",
        "status": "live",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "incident": {"lat": incident_lat, "lon": incident_lon, "category": clinical_category, "priority": priority},
        "recommended_hospital": best,
        "all_hospitals_scored": scored_hospitals,
        "evidence_base": [
            "Nicholl BMJ (2007) — Bypass protocols",
            "Sasser WHO (2005) — Trauma system design",
            "Pons Prehosp Emerg Care (2005) — Destination protocols",
        ]
    }


class PatientPathwayModel:
    def predict_demo(self) -> dict[str, Any]:
        cases = [
            {"name": "STEMI — Centre-ville Genève", "lat": 46.2044, "lon": 6.1432, "cat": "chest-pain", "p": 4},
            {"name": "AVC — Annemasse", "lat": 46.1933, "lon": 6.2356, "cat": "stroke", "p": 4},
            {"name": "Polytraumatisme — Ferney-Voltaire", "lat": 46.2567, "lon": 6.1067, "cat": "major-trauma", "p": 5},
        ]
        results = []
        for c in cases:
            r = optimize_patient_pathway(c["lat"], c["lon"], c["cat"], c["p"])
            results.append({"case": c["name"], "recommended": r["recommended_hospital"]["hospital_name"],
                           "eta_min": r["recommended_hospital"]["eta_min"],
                           "cross_border": r["recommended_hospital"]["cross_border"]})
        return {
            "model": "PatientPathwayOptimization v1.0", "status": "live",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "cases": results,
        }


patient_pathway_model_singleton = PatientPathwayModel()
