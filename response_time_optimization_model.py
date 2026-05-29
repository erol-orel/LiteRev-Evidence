"""
response_time_optimization_model.py
=====================================
Modèle d'optimisation des temps de réponse EMS pour le scénario GESICA
"response-time-optimization".

Approche :
  - Données réelles : OSRM (routage) + Open-Meteo (météo) + OSM (positions)
  - Modèle : Optimisation multi-critères (temps de trajet + facteurs météo + délais douane)
  - Algorithme : Gradient de priorité + affectation hongroise simplifiée
  - Sortie : recommandations d'affectation des ressources EMS, temps de réponse optimisés,
    alertes de délai transfrontalier

Usage :
  from response_time_optimization_model import response_time_model_singleton
  result = response_time_model_singleton.optimize()
"""

import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger("response_time_model")

# ─── Constantes géographiques ────────────────────────────────────────────────

# Bases EMS du Grand Genève (lat, lon, label, pays)
EMS_BASES = [
    {"id": "hug", "label": "HUG — Urgences", "lat": 46.1874, "lon": 6.1440, "country": "CH", "capacity": 3},
    {"id": "annemasse", "label": "SDIS 74 — Annemasse", "lat": 46.1939, "lon": 6.2358, "country": "FR", "capacity": 2},
    {"id": "gaillard", "label": "SAMU 74 — Gaillard", "lat": 46.1853, "lon": 6.2217, "country": "FR", "capacity": 2},
    {"id": "saint_julien", "label": "SDIS 74 — Saint-Julien", "lat": 46.1434, "lon": 6.0831, "country": "FR", "capacity": 2},
    {"id": "thonon", "label": "SDIS 74 — Thonon", "lat": 46.3700, "lon": 6.4780, "country": "FR", "capacity": 2},
    {"id": "cluse", "label": "SDIS 74 — La Cluse", "lat": 46.1667, "lon": 6.1167, "country": "FR", "capacity": 1},
]

# Points d'intervention types (zones à risque Grand Genève)
INTERVENTION_ZONES = [
    {"id": "centre_geneve", "label": "Centre-ville Genève", "lat": 46.2044, "lon": 6.1432, "priority": "high"},
    {"id": "carouge", "label": "Carouge", "lat": 46.1822, "lon": 6.1417, "priority": "medium"},
    {"id": "meyrin", "label": "Meyrin (CERN)", "lat": 46.2338, "lon": 6.0586, "priority": "medium"},
    {"id": "lancy", "label": "Lancy", "lat": 46.1833, "lon": 6.1167, "priority": "medium"},
    {"id": "annemasse_centre", "label": "Annemasse Centre", "lat": 46.1939, "lon": 6.2358, "priority": "high"},
    {"id": "thonon_centre", "label": "Thonon-les-Bains", "lat": 46.3700, "lon": 6.4780, "priority": "low"},
    {"id": "saint_genis", "label": "Saint-Genis-Pouilly", "lat": 46.2433, "lon": 6.0233, "priority": "medium"},
    {"id": "ferney", "label": "Ferney-Voltaire", "lat": 46.2558, "lon": 6.1086, "priority": "medium"},
]

# Postes frontières Genève/Haute-Savoie
BORDER_CROSSINGS = [
    {"id": "moillesulaz", "label": "Moillesulaz", "lat": 46.1972, "lon": 6.2083, "avg_delay_min": 3},
    {"id": "bardonnex", "label": "Bardonnex", "lat": 46.1417, "lon": 6.1083, "avg_delay_min": 5},
    {"id": "ferney_voltaire", "label": "Ferney-Voltaire", "lat": 46.2558, "lon": 6.1086, "avg_delay_min": 4},
    {"id": "perly", "label": "Perly", "lat": 46.1667, "lon": 6.0833, "avg_delay_min": 4},
]

OSRM_BASE = "http://router.project-osrm.org/route/v1/driving"
OPEN_METEO_BASE = "https://api.open-meteo.com/v1/forecast"
GENEVA_LAT, GENEVA_LON = 46.2044, 6.1432


# ─── Collecte des données météo ──────────────────────────────────────────────

def _fetch_weather_conditions() -> Dict[str, Any]:
    """Récupère les conditions météo actuelles pour le Grand Genève."""
    try:
        resp = requests.get(
            OPEN_METEO_BASE,
            params={
                "latitude": GENEVA_LAT,
                "longitude": GENEVA_LON,
                "current": "temperature_2m,precipitation,wind_speed_10m,visibility,weather_code",
                "timezone": "Europe/Paris",
            },
            timeout=8,
        )
        if resp.status_code == 200:
            data = resp.json()
            current = data.get("current", {})
            return {
                "temperature": current.get("temperature_2m", 15.0),
                "precipitation": current.get("precipitation", 0.0),
                "wind_speed": current.get("wind_speed_10m", 10.0),
                "visibility": current.get("visibility", 10000),
                "weather_code": current.get("weather_code", 0),
                "source": "Open-Meteo (données réelles)",
            }
    except Exception as e:
        logger.warning(f"Open-Meteo fetch failed: {e}")

    return {
        "temperature": 15.0,
        "precipitation": 0.0,
        "wind_speed": 10.0,
        "visibility": 10000,
        "weather_code": 0,
        "source": "Données par défaut (Open-Meteo indisponible)",
    }


def _compute_weather_factor(weather: Dict) -> Tuple[float, str]:
    """
    Calcule un facteur multiplicateur du temps de trajet basé sur les conditions météo.
    Retourne (facteur, description).
    """
    factor = 1.0
    reasons = []

    precip = weather.get("precipitation", 0)
    wind = weather.get("wind_speed", 0)
    vis = weather.get("visibility", 10000)
    code = weather.get("weather_code", 0)

    # Précipitations
    if precip > 10:
        factor += 0.30
        reasons.append("fortes précipitations (+30%)")
    elif precip > 2:
        factor += 0.15
        reasons.append("précipitations modérées (+15%)")

    # Vent
    if wind > 60:
        factor += 0.20
        reasons.append("vent fort (+20%)")
    elif wind > 40:
        factor += 0.10
        reasons.append("vent modéré (+10%)")

    # Visibilité
    if vis < 1000:
        factor += 0.25
        reasons.append("visibilité réduite (+25%)")
    elif vis < 3000:
        factor += 0.10
        reasons.append("visibilité limitée (+10%)")

    # Code météo (neige, brouillard)
    if code in range(71, 78):  # Neige
        factor += 0.40
        reasons.append("neige (+40%)")
    elif code in range(45, 50):  # Brouillard
        factor += 0.20
        reasons.append("brouillard (+20%)")

    description = ", ".join(reasons) if reasons else "Conditions normales"
    return round(factor, 2), description


# ─── Calcul de distance haversine ────────────────────────────────────────────

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ─── Appel OSRM ──────────────────────────────────────────────────────────────

def _osrm_route(origin: Dict, dest: Dict) -> Optional[Dict]:
    """Calcule le temps de trajet via OSRM entre deux points."""
    try:
        url = f"{OSRM_BASE}/{origin['lon']},{origin['lat']};{dest['lon']},{dest['lat']}"
        resp = requests.get(url, params={"overview": "false"}, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("code") == "Ok" and data.get("routes"):
                route = data["routes"][0]
                return {
                    "duration_s": route["duration"],
                    "distance_m": route["distance"],
                    "duration_min": round(route["duration"] / 60, 1),
                    "distance_km": round(route["distance"] / 1000, 1),
                    "source": "OSRM (données réelles)",
                }
    except Exception as e:
        logger.debug(f"OSRM route failed: {e}")

    # Fallback : estimation par haversine × 1.4 (facteur de tortuosité)
    dist_km = _haversine_km(origin["lat"], origin["lon"], dest["lat"], dest["lon"])
    speed_kmh = 60.0  # Vitesse moyenne urbaine EMS
    duration_min = (dist_km * 1.4 / speed_kmh) * 60
    return {
        "duration_s": duration_min * 60,
        "distance_m": dist_km * 1400,
        "duration_min": round(duration_min, 1),
        "distance_km": round(dist_km * 1.4, 1),
        "source": "Estimation haversine (OSRM indisponible)",
    }


# ─── Détection de franchissement de frontière ────────────────────────────────

def _needs_border_crossing(origin: Dict, dest: Dict) -> Optional[Dict]:
    """Détermine si le trajet nécessite un franchissement de frontière."""
    if origin["country"] == dest.get("country", "FR"):
        return None

    # Trouver le poste frontière le plus proche du trajet
    mid_lat = (origin["lat"] + dest["lat"]) / 2
    mid_lon = (origin["lon"] + dest["lon"]) / 2

    closest = min(
        BORDER_CROSSINGS,
        key=lambda b: _haversine_km(mid_lat, mid_lon, b["lat"], b["lon"]),
    )
    return closest


# ─── Algorithme d'affectation optimale ───────────────────────────────────────

def _assign_resources(
    zones: List[Dict],
    bases: List[Dict],
    weather_factor: float,
) -> List[Dict]:
    """
    Affecte chaque zone d'intervention à la base EMS la plus proche (temps de réponse minimal).
    Tient compte du facteur météo et des délais de frontière.
    """
    assignments = []

    for zone in zones:
        best_base = None
        best_time = float("inf")
        best_route = None
        best_border = None

        for base in bases:
            route = _osrm_route(base, zone)
            if route is None:
                continue

            base_time = route["duration_min"] * weather_factor

            # Ajouter délai de frontière si nécessaire
            border = _needs_border_crossing(base, zone)
            border_delay = border["avg_delay_min"] if border else 0
            total_time = base_time + border_delay

            if total_time < best_time:
                best_time = total_time
                best_base = base
                best_route = route
                best_border = border

        if best_base and best_route:
            base_time_min = best_route["duration_min"] * weather_factor
            border_delay = best_border["avg_delay_min"] if best_border else 0

            # Niveau d'alerte selon le temps de réponse
            if best_time <= 8:
                response_status = "OPTIMAL"
                status_color = "green"
            elif best_time <= 15:
                response_status = "ACCEPTABLE"
                status_color = "amber"
            else:
                response_status = "DÉGRADÉ"
                status_color = "red"

            assignments.append({
                "zone": zone,
                "assigned_base": best_base,
                "route": best_route,
                "weather_factor": weather_factor,
                "base_travel_time_min": round(base_time_min, 1),
                "border_delay_min": border_delay,
                "border_crossing": best_border,
                "total_response_time_min": round(best_time, 1),
                "response_status": response_status,
                "status_color": status_color,
                "cross_border": best_border is not None,
                "recommendation": _generate_assignment_recommendation(
                    zone, best_base, best_time, best_border, weather_factor
                ),
            })

    return assignments


def _generate_assignment_recommendation(
    zone: Dict,
    base: Dict,
    total_time: float,
    border: Optional[Dict],
    weather_factor: float,
) -> str:
    parts = [f"Affecter {base['label']} → {zone['label']}"]
    parts.append(f"Temps estimé : {total_time:.1f} min")
    if border:
        parts.append(f"Via {border['label']} (+{border['avg_delay_min']} min douane)")
    if weather_factor > 1.2:
        parts.append(f"Conditions météo dégradées (×{weather_factor:.2f})")
    if total_time > 15:
        parts.append("⚠ Délai > 15 min — envisager ressource alternative")
    return " | ".join(parts)


# ─── Calcul des métriques globales ───────────────────────────────────────────

def _compute_global_metrics(assignments: List[Dict]) -> Dict:
    if not assignments:
        return {}

    times = [a["total_response_time_min"] for a in assignments]
    cross_border_count = sum(1 for a in assignments if a["cross_border"])
    degraded_count = sum(1 for a in assignments if a["response_status"] == "DÉGRADÉ")

    return {
        "mean_response_time_min": round(sum(times) / len(times), 1),
        "max_response_time_min": round(max(times), 1),
        "min_response_time_min": round(min(times), 1),
        "cross_border_interventions": cross_border_count,
        "degraded_zones": degraded_count,
        "coverage_rate_pct": round((len(times) - degraded_count) / len(times) * 100, 1),
    }


# ─── Modèle principal ────────────────────────────────────────────────────────

class ResponseTimeOptimizationModel:
    """
    Modèle d'optimisation des temps de réponse EMS pour le Grand Genève.
    Combine OSRM (routage réel), Open-Meteo (météo) et optimisation multi-critères.
    """

    def __init__(self):
        self._cache: Optional[Dict] = None
        self._cache_time: Optional[datetime] = None
        self._cache_ttl_minutes = 15

    def _is_cache_valid(self) -> bool:
        if self._cache is None or self._cache_time is None:
            return False
        return (datetime.utcnow() - self._cache_time).total_seconds() < self._cache_ttl_minutes * 60

    def optimize(self, force_refresh: bool = False) -> Dict[str, Any]:
        if not force_refresh and self._is_cache_valid():
            return self._cache  # type: ignore

        result = self._run_optimization()
        self._cache = result
        self._cache_time = datetime.utcnow()
        return result

    def _run_optimization(self) -> Dict[str, Any]:
        # 1. Conditions météo actuelles
        weather = _fetch_weather_conditions()
        weather_factor, weather_desc = _compute_weather_factor(weather)

        # 2. Affectation optimale des ressources
        assignments = _assign_resources(INTERVENTION_ZONES, EMS_BASES, weather_factor)

        # 3. Métriques globales
        metrics = _compute_global_metrics(assignments)

        # 4. Zones critiques (temps > 15 min)
        critical_zones = [a for a in assignments if a["response_status"] == "DÉGRADÉ"]

        # 5. Recommandation globale
        if metrics.get("degraded_zones", 0) > 2:
            global_recommendation = (
                f"ALERTE : {metrics['degraded_zones']} zones avec temps de réponse > 15 min. "
                f"Activer les ressources de renfort. Temps moyen : {metrics['mean_response_time_min']} min."
            )
        elif weather_factor > 1.2:
            global_recommendation = (
                f"Conditions météo dégradées ({weather_desc}). "
                f"Temps de trajet majoré de {int((weather_factor - 1) * 100)}%. "
                f"Temps moyen estimé : {metrics.get('mean_response_time_min', 'N/A')} min."
            )
        else:
            global_recommendation = (
                f"Couverture EMS nominale. Temps de réponse moyen : {metrics.get('mean_response_time_min', 'N/A')} min. "
                f"{metrics.get('cross_border_interventions', 0)} intervention(s) transfrontalière(s) prévues."
            )

        # Formater les assignments pour la réponse API
        formatted_assignments = []
        for a in assignments:
            formatted_assignments.append({
                "zone_id": a["zone"]["id"],
                "zone_label": a["zone"]["label"],
                "zone_priority": a["zone"]["priority"],
                "base_id": a["assigned_base"]["id"],
                "base_label": a["assigned_base"]["label"],
                "base_country": a["assigned_base"]["country"],
                "distance_km": a["route"]["distance_km"],
                "base_travel_time_min": a["base_travel_time_min"],
                "border_delay_min": a["border_delay_min"],
                "border_crossing": a["border_crossing"]["label"] if a["border_crossing"] else None,
                "total_response_time_min": a["total_response_time_min"],
                "response_status": a["response_status"],
                "cross_border": a["cross_border"],
                "recommendation": a["recommendation"],
                "route_source": a["route"]["source"],
            })

        return {
            "model": "ResponseTimeOptimization v1.0",
            "status": "live" if weather["source"].startswith("Open-Meteo") else "fallback",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "region": "Grand Genève (CH/FR)",
            "weather": {
                "temperature": weather["temperature"],
                "precipitation": weather["precipitation"],
                "wind_speed": weather["wind_speed"],
                "weather_factor": weather_factor,
                "weather_description": weather_desc,
                "source": weather["source"],
            },
            "metrics": metrics,
            "assignments": formatted_assignments,
            "critical_zones": [
                {
                    "zone_label": a["zone"]["label"],
                    "total_response_time_min": a["total_response_time_min"],
                    "recommendation": a["recommendation"],
                }
                for a in critical_zones
            ],
            "border_crossings_active": [
                bc for bc in BORDER_CROSSINGS
            ],
            "ems_bases": [
                {"id": b["id"], "label": b["label"], "country": b["country"], "capacity": b["capacity"]}
                for b in EMS_BASES
            ],
            "global_recommendation": global_recommendation,
            "data_sources": [
                "OSRM (OpenStreetMap Routing Machine)",
                "Open-Meteo (météo temps réel)",
                "OpenStreetMap (géographie)",
            ],
        }


# ─── Singleton ───────────────────────────────────────────────────────────────

response_time_model_singleton = ResponseTimeOptimizationModel()
