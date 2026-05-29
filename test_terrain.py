#!/usr/bin/env python3
"""Test local pour simuler les endpoints de terrain (P5)."""
import requests
import json

def test_endpoints():
    print("Simulating MeteoSwiss/OSM/Sentinelles API endpoints...")
    
    # 1. Meteo Swiss simulation
    meteo_swiss = {
        "source": "MeteoSwiss",
        "station": "Geneva / Cointrin",
        "temperature": 28.4,
        "humidity": 45.0,
        "wind_speed": 12.5,
        "precipitation_probability": 10.0,
        "alert_level": "none",
        "alert_description": "No active weather alerts",
        "impact_on_ems": "Normal operations. High temperature may slightly increase cardiovascular calls (+5%)."
    }
    print("MeteoSwiss:", json.dumps(meteo_swiss, indent=2))
    
    # 2. OSM/OSRM simulation
    osm_osrm = {
        "source": "OpenStreetMap / OSRM",
        "origin": "Geneva, Switzerland",
        "destination": "Annemasse, France",
        "distance_km": 10.5,
        "estimated_duration_min": 15.2,
        "traffic_congestion_factor": 1.15,
        "cross_border_delay_min": 3.0,
        "total_response_time_min": 18.2,
        "routing_status": "optimal"
    }
    print("OSM/OSRM:", json.dumps(osm_osrm, indent=2))
    
    # 3. Sentinelles simulation
    sentinelles = {
        "source": "Sentinelles (France) / ECDC / Sentinella (Suisse)",
        "region": "Auvergne-Rhône-Alpes / Geneva / Vaud",
        "diseases": [
            {
                "name": "Influenza-like illness",
                "incidence_per_100k": 124.5,
                "epidemic_threshold": 150.0,
                "status": "under_threshold",
                "trend": "stable"
            },
            {
                "name": "COVID-19",
                "incidence_per_100k": 85.2,
                "epidemic_threshold": 100.0,
                "status": "under_threshold",
                "trend": "increasing"
            },
            {
                "name": "Gastroenteritis",
                "incidence_per_100k": 189.0,
                "epidemic_threshold": 170.0,
                "status": "epidemic",
                "trend": "decreasing"
            }
        ],
        "ems_impact_risk": "moderate"
    }
    print("Sentinelles:", json.dumps(sentinelles, indent=2))

if __name__ == "__main__":
    test_endpoints()
