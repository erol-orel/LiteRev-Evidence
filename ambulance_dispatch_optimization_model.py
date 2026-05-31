"""
ambulance_dispatch_optimization_model.py
Optimisation du Positionnement des Ambulances (Dynamic Deployment / VRP)
Couvre : ambulance-dispatch-optimization
Basé sur : Gendreau EJOR (2001), Brotcorne EJOR (2003), Schmid EJOR (2012),
           Maxwell Interfaces (2010) — MEXCLP model
"""
from __future__ import annotations
import math
from datetime import datetime, timezone
from typing import Any

# ─── Zones de couverture Grand Genève ─────────────────────────────────────────
COVERAGE_ZONES = [
    {"id": "GE-CENTRE", "name": "Genève Centre", "lat": 46.2044, "lon": 6.1432, "population": 85000, "demand_weight": 1.4},
    {"id": "GE-CAROUGE", "name": "Carouge", "lat": 46.1833, "lon": 6.1500, "population": 22000, "demand_weight": 1.0},
    {"id": "GE-MEYRIN", "name": "Meyrin", "lat": 46.2381, "lon": 6.0889, "population": 25000, "demand_weight": 0.9},
    {"id": "GE-LANCY", "name": "Lancy", "lat": 46.1833, "lon": 6.1167, "population": 32000, "demand_weight": 1.0},
    {"id": "FR-ANNEMASSE", "name": "Annemasse", "lat": 46.1933, "lon": 6.2356, "population": 35000, "demand_weight": 1.1},
    {"id": "FR-THONON", "name": "Thonon-les-Bains", "lat": 46.3700, "lon": 6.4800, "population": 40000, "demand_weight": 0.8},
    {"id": "FR-SAINTGENIS", "name": "Saint-Genis-Pouilly", "lat": 46.2433, "lon": 6.0233, "population": 12000, "demand_weight": 0.7},
    {"id": "FR-FERNEY", "name": "Ferney-Voltaire", "lat": 46.2567, "lon": 6.1067, "population": 10000, "demand_weight": 0.7},
]

# ─── Bases EMS disponibles ────────────────────────────────────────────────────
EMS_BASES = [
    {"id": "HUG", "name": "HUG Genève", "lat": 46.1907, "lon": 6.1464, "country": "CH", "capacity": 4},
    {"id": "SIS-GE", "name": "SIS Genève Centre", "lat": 46.2044, "lon": 6.1432, "country": "CH", "capacity": 3},
    {"id": "SIS-CAROUGE", "name": "SIS Carouge", "lat": 46.1833, "lon": 6.1500, "country": "CH", "capacity": 2},
    {"id": "SDIS74-ANNEMASSE", "name": "SDIS 74 Annemasse", "lat": 46.1933, "lon": 6.2356, "country": "FR", "capacity": 2},
    {"id": "SDIS74-THONON", "name": "SDIS 74 Thonon", "lat": 46.3700, "lon": 6.4800, "country": "FR", "capacity": 2},
    {"id": "SAMU74-FERNEY", "name": "Antenne SAMU Ferney", "lat": 46.2567, "lon": 6.1067, "country": "FR", "capacity": 1},
]

COVERAGE_RADIUS_KM = 8.0   # Rayon de couverture cible (8 km ≈ 8 min)
MAX_RESPONSE_TIME_MIN = 8.0  # Objectif ORSAN : 8 min pour les urgences vitales

def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def _coverage_matrix() -> dict[str, dict[str, float]]:
    """Calcule la matrice de couverture bases → zones."""
    matrix = {}
    for base in EMS_BASES:
        matrix[base["id"]] = {}
        for zone in COVERAGE_ZONES:
            dist = _haversine_km(base["lat"], base["lon"], zone["lat"], zone["lon"])
            eta = 2.0 + dist / 55.0 * 60  # mobilisation + trajet
            matrix[base["id"]][zone["id"]] = round(eta, 1)
    return matrix

def _mexclp_coverage(available_units: dict[str, int], coverage_matrix: dict) -> dict[str, Any]:
    """
    Maximum Expected Coverage Location Problem (MEXCLP) simplifié.
    Maxwell et al. Interfaces (2010).
    Calcule le pourcentage de population couverte dans le délai cible.
    """
    total_pop = sum(z["population"] for z in COVERAGE_ZONES)
    covered_pop = 0
    zone_coverage = {}

    for zone in COVERAGE_ZONES:
        # Trouver les bases qui couvrent cette zone dans le délai cible
        covering_bases = []
        for base_id, units in available_units.items():
            if units > 0 and coverage_matrix[base_id][zone["id"]] <= MAX_RESPONSE_TIME_MIN:
                covering_bases.append({
                    "base_id": base_id,
                    "eta_min": coverage_matrix[base_id][zone["id"]],
                    "units": units,
                })

        if covering_bases:
            best_base = min(covering_bases, key=lambda x: x["eta_min"])
            covered_pop += zone["population"]
            zone_coverage[zone["id"]] = {
                "covered": True,
                "best_base": best_base["base_id"],
                "eta_min": best_base["eta_min"],
                "redundancy": len(covering_bases),
            }
        else:
            # Trouver la base la plus proche même hors délai
            all_bases = [(bid, coverage_matrix[bid][zone["id"]]) for bid in coverage_matrix]
            nearest = min(all_bases, key=lambda x: x[1])
            zone_coverage[zone["id"]] = {
                "covered": False,
                "best_base": nearest[0],
                "eta_min": nearest[1],
                "redundancy": 0,
            }

    coverage_pct = round(covered_pop / total_pop * 100, 1)
    return {"coverage_pct": coverage_pct, "covered_population": covered_pop, "total_population": total_pop, "zones": zone_coverage}

def optimize_deployment(
    available_units_per_base: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Optimise le déploiement des ambulances pour maximiser la couverture."""
    if available_units_per_base is None:
        available_units_per_base = {b["id"]: b["capacity"] for b in EMS_BASES}

    coverage_matrix = _coverage_matrix()
    current_coverage = _mexclp_coverage(available_units_per_base, coverage_matrix)

    # Identifier les zones non couvertes
    uncovered_zones = [zid for zid, zc in current_coverage["zones"].items() if not zc["covered"]]
    degraded_zones = [zid for zid, zc in current_coverage["zones"].items() if zc["covered"] and zc["eta_min"] > 6.0]

    # Recommandations de repositionnement
    recommendations = []
    for zid in uncovered_zones:
        zone = next(z for z in COVERAGE_ZONES if z["id"] == zid)
        best_base_id = current_coverage["zones"][zid]["best_base"]
        eta = current_coverage["zones"][zid]["eta_min"]
        recommendations.append({
            "type": "REPOSITIONNEMENT",
            "zone": zone["name"],
            "action": f"Positionner une unité à {zone['name']} — ETA actuel {eta:.0f} min (objectif {MAX_RESPONSE_TIME_MIN:.0f} min)",
            "priority": "HAUTE" if zone["demand_weight"] >= 1.0 else "NORMALE",
        })

    if current_coverage["coverage_pct"] < 80:
        recommendations.append({
            "type": "RENFORT",
            "zone": "Grand Genève",
            "action": f"Couverture insuffisante ({current_coverage['coverage_pct']}%). Activer les équipes de renfort.",
            "priority": "CRITIQUE",
        })

    return {
        "model": "AmbulanceDispatchOptimization v1.0 (MEXCLP)",
        "status": "live",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "coverage": {
            "coverage_pct": current_coverage["coverage_pct"],
            "covered_population": current_coverage["covered_population"],
            "total_population": current_coverage["total_population"],
            "uncovered_zones": len(uncovered_zones),
            "degraded_zones": len(degraded_zones),
        },
        "zone_details": [
            {
                "zone_id": zid,
                "zone_name": next(z["name"] for z in COVERAGE_ZONES if z["id"] == zid),
                **zdata
            }
            for zid, zdata in current_coverage["zones"].items()
        ],
        "recommendations": recommendations,
        "total_units_deployed": sum(available_units_per_base.values()),
        "evidence_base": [
            "MEXCLP — Maxwell MS et al. Interfaces (2010)",
            "Gendreau M et al. EJOR (2001) — Dynamic ambulance relocation",
            "Schmid V. EJOR (2012) — Ambulance deployment optimization",
        ]
    }


class AmbulanceDispatchModel:
    def predict_demo(self) -> dict[str, Any]:
        # Scénario de nuit avec moins d'unités disponibles
        units = {"HUG": 2, "SIS-GE": 1, "SIS-CAROUGE": 1, "SDIS74-ANNEMASSE": 1, "SDIS74-THONON": 1, "SAMU74-FERNEY": 0}
        return optimize_deployment(units)


ambulance_dispatch_model_singleton = AmbulanceDispatchModel()
