#!/usr/bin/env python3
"""Stage 3 backfill: GESICA-specific scenario_type and geographic_scope enrichment.
Version 2.1 — 31 scénarios couvrant les 4 clusters de la revue systématique.
Opère uniquement sur literature_document (les chunks héritent via document_id).
"""
from __future__ import annotations
import os
from sqlalchemy import create_engine, text

DB_URL = os.getenv(
    "DB_URL",
    "postgresql+psycopg://literev:MyNewStrongPassword!@10.10.1.10:5432/literev",
)
engine = create_engine(DB_URL, pool_pre_ping=True)

# ─── Mise à jour des documents (literature_document) ─────────────────────────
STAGE3_DOC_SQL = """
UPDATE literature_document SET
  scenario_type = COALESCE(scenario_type, CASE
    -- Cluster 0: Patient-centered prehospital critical care
    WHEN title ILIKE '%cardiac arrest%' OR abstract ILIKE '%out-of-hospital cardiac arrest%'
      OR abstract ILIKE '%OHCA%' OR abstract ILIKE '%ventricular fibrillation%'
      THEN 'cardiac-arrest-prediction'
    WHEN title ILIKE '%stroke%' OR abstract ILIKE '%stroke%'
      OR abstract ILIKE '%large vessel occlusion%' OR abstract ILIKE '%thrombolysis%'
      OR abstract ILIKE '%transient ischemic%'
      THEN 'stroke-detection'
    WHEN title ILIKE '%trauma%' OR abstract ILIKE '%traumatic brain injury%'
      OR abstract ILIKE '%road traffic accident%' OR abstract ILIKE '%injury severity%'
      OR abstract ILIKE '%hemorrhage%'
      THEN 'trauma-severity-assessment'
    WHEN title ILIKE '%clinical deterioration%' OR abstract ILIKE '%in-transit%'
      OR abstract ILIKE '%deterioration%' OR abstract ILIKE '%SMUR%'
      OR abstract ILIKE '%mobile intensive care%'
      THEN 'clinical-deterioration-prediction'
    WHEN title ILIKE '%patient pathway%' OR abstract ILIKE '%hospital transfer%'
      OR abstract ILIKE '%appropriate hospital%' OR abstract ILIKE '%destination%'
      THEN 'patient-pathway-optimization'
    WHEN title ILIKE '%mass casualty%' OR abstract ILIKE '%mass casualty incident%'
      OR abstract ILIKE '%MCI%' OR abstract ILIKE '%victim estimation%'
      OR abstract ILIKE '%smart glasses%'
      THEN 'mci-victim-estimation'
    -- Cluster 1: Environmental & Disaster Risk Forecasting
    WHEN (title ILIKE '%environmental%' AND (title ILIKE '%forecast%' OR abstract ILIKE '%forecast%'))
      OR abstract ILIKE '%air quality%' OR abstract ILIKE '%heatwave%'
      OR abstract ILIKE '%heat stress%'
      THEN 'environmental-risk-forecasting'
    WHEN title ILIKE '%disaster%' OR abstract ILIKE '%natural disaster%'
      OR abstract ILIKE '%earthquake%' OR abstract ILIKE '%flood%'
      OR abstract ILIKE '%wildfire%' OR abstract ILIKE '%hurricane%'
      THEN 'disaster-risk-assessment'
    WHEN title ILIKE '%climate%' OR abstract ILIKE '%climate change%'
      OR (abstract ILIKE '%seasonal%' AND abstract ILIKE '%emergency%')
      THEN 'climate-impact-on-ems'
    -- Cluster 2: Prehospital Emergency Triage & Risk Stratification
    WHEN title ILIKE '%emergency call%' OR abstract ILIKE '%emergency call%'
      OR abstract ILIKE '%call qualification%' OR abstract ILIKE '%audio signal%'
      OR (abstract ILIKE '%acoustic%' AND abstract ILIKE '%EMS%')
      THEN 'emergency-call-qualification'
    WHEN title ILIKE '%call prioriti%' OR abstract ILIKE '%call prioriti%'
      OR abstract ILIKE '%dispatch prioriti%'
      THEN 'call-prioritization'
    WHEN (title ILIKE '%mass casualty%' OR abstract ILIKE '%mass casualty%')
      AND (abstract ILIKE '%START triage%' OR abstract ILIKE '%SALT triage%'
           OR abstract ILIKE '%on-site triage%')
      THEN 'mass-casualty-triage'
    WHEN title ILIKE '%undertriage%' OR abstract ILIKE '%undertriage%'
      OR abstract ILIKE '%under-triage%'
      THEN 'undertriage-detection'
    WHEN title ILIKE '%dispatch%' OR abstract ILIKE '%dispatch decision%'
      OR abstract ILIKE '%medical dispatcher%' OR abstract ILIKE '%EMD%'
      OR abstract ILIKE '%MRA%'
      THEN 'dispatch-decision-support'
    WHEN title ILIKE '%triage%' OR abstract ILIKE '%triage%'
      OR title ILIKE '%severity score%' OR abstract ILIKE '%patient acuity%'
      THEN 'triage-support'
    -- Cluster 3: Demand Forecasting, Response Time, Resource Management
    WHEN title ILIKE '%response time%' OR abstract ILIKE '%response time%'
      OR abstract ILIKE '%travel time%' OR abstract ILIKE '%ambulance time%'
      THEN 'response-time-optimization'
    WHEN title ILIKE '%ambulance dispatch%' OR abstract ILIKE '%ambulance dispatch%'
      OR abstract ILIKE '%fleet management%' OR abstract ILIKE '%coverage optimization%'
      OR abstract ILIKE '%station location%'
      THEN 'ambulance-dispatch-optimization'
    WHEN (title ILIKE '%staffing%' OR abstract ILIKE '%staffing%')
      AND abstract ILIKE '%EMS%'
      THEN 'staffing-level-prediction'
    WHEN title ILIKE '%hospital capacity%' OR abstract ILIKE '%hospital capacity%'
      OR abstract ILIKE '%bed management%' OR abstract ILIKE '%ED crowding%'
      THEN 'hospital-capacity-forecasting'
    WHEN title ILIKE '%demand forecast%' OR abstract ILIKE '%demand forecast%'
      OR title ILIKE '%call volume%' OR abstract ILIKE '%arrival rate%'
      THEN 'demand-forecasting'
    WHEN title ILIKE '%resource allocation%' OR abstract ILIKE '%resource allocation%'
      OR abstract ILIKE '%capacity planning%'
      THEN 'resource-allocation'
    -- Surveillance & Épidémie
    WHEN title ILIKE '%epidemic%' OR abstract ILIKE '%epidemic%'
      OR abstract ILIKE '%influenza%' OR abstract ILIKE '%COVID%'
      OR abstract ILIKE '%syndromic surveillance%'
      THEN 'epidemic-early-warning'
    WHEN title ILIKE '%surveillance%' OR abstract ILIKE '%surveillance%'
      OR title ILIKE '%early warning%' OR abstract ILIKE '%early warning%'
      OR abstract ILIKE '%outbreak detection%'
      THEN 'surveillance'
    WHEN title ILIKE '%surge%' OR abstract ILIKE '%surge%'
      OR abstract ILIKE '%mass gathering%'
      THEN 'surge-management'
    WHEN title ILIKE '%pandemic%' OR abstract ILIKE '%pandemic%'
      OR (abstract ILIKE '%preparedness%' AND abstract ILIKE '%emergency%')
      THEN 'pandemic-preparedness'
    WHEN title ILIKE '%cross-border%' OR abstract ILIKE '%cross-border%'
      OR title ILIKE '%transfrontalier%' OR abstract ILIKE '%transfrontalier%'
      THEN 'cross-border-coordination'
    WHEN title ILIKE '%situational awareness%' OR abstract ILIKE '%situational awareness%'
      THEN 'real-time-situational-awareness'
    ELSE NULL
  END),
  geographic_scope = COALESCE(geographic_scope, CASE
    WHEN title ILIKE '%switzerland%' OR abstract ILIKE '%switzerland%'
      OR title ILIKE '%suisse%' OR abstract ILIKE '%geneva%'
      OR abstract ILIKE '%hug%' OR abstract ILIKE '%chuv%'
      OR abstract ILIKE '%lausanne%' OR abstract ILIKE '%zurich%'
      THEN 'switzerland'
    WHEN title ILIKE '%france%' OR abstract ILIKE '%french%'
      OR abstract ILIKE '%samu%' OR abstract ILIKE '%smur%'
      OR abstract ILIKE '%paris%' OR abstract ILIKE '%marseille%'
      THEN 'france'
    WHEN title ILIKE '%europe%' OR abstract ILIKE '%european%'
      THEN 'europe'
    WHEN title ILIKE '%global%' OR abstract ILIKE '%worldwide%'
      OR title ILIKE '%international%' OR abstract ILIKE '%international%'
      THEN 'global'
    ELSE NULL
  END)
WHERE project_context = 'gesica';
"""

REPORT_SQL = """
SELECT
  COUNT(*) AS total,
  COUNT(scenario_type) AS with_scenario,
  ROUND(100.0 * COUNT(scenario_type) / NULLIF(COUNT(*), 0), 1) AS pct_scenario,
  COUNT(geographic_scope) AS with_geo,
  ROUND(100.0 * COUNT(geographic_scope) / NULLIF(COUNT(*), 0), 1) AS pct_geo
FROM literature_document
WHERE project_context = 'gesica';
"""

BREAKDOWN_SQL = """
SELECT scenario_type, COUNT(*) AS n FROM literature_document
WHERE project_context = 'gesica' AND scenario_type IS NOT NULL
GROUP BY scenario_type ORDER BY n DESC;
"""

GEO_SQL = """
SELECT geographic_scope, COUNT(*) AS n FROM literature_document
WHERE project_context = 'gesica' AND geographic_scope IS NOT NULL
GROUP BY geographic_scope ORDER BY n DESC;
"""

with engine.begin() as conn:
    r = conn.execute(text(STAGE3_DOC_SQL))
    print(f"Documents updated: {r.rowcount}")
    print("\n--- Stats globales ---")
    for row in conn.execute(text(REPORT_SQL)).fetchall():
        print(dict(row._mapping))
    print("\n--- Répartition par scénario ---")
    for row in conn.execute(text(BREAKDOWN_SQL)).fetchall():
        print(f"  {row[0]}: {row[1]}")
    print("\n--- Répartition géographique ---")
    for row in conn.execute(text(GEO_SQL)).fetchall():
        print(f"  {row[0]}: {row[1]}")
