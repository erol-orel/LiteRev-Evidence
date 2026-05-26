#!/usr/bin/env python3
"""Stage 3 backfill: GESICA-specific scenario_type and geographic_scope enrichment."""
from __future__ import annotations
import os
from sqlalchemy import create_engine, text

DB_URL = os.environ["DB_URL"]
engine = create_engine(DB_URL, pool_pre_ping=True)

STAGE3_SQL = """
UPDATE literature_document SET
  scenario_type = COALESCE(scenario_type, CASE
    WHEN title ILIKE '%demand forecast%' OR abstract ILIKE '%demand forecast%'
      OR title ILIKE '%call volume%' OR abstract ILIKE '%arrival rate%'
      THEN 'ems-demand-forecasting'
    WHEN title ILIKE '%resource allocation%' OR abstract ILIKE '%ambulance dispatch%'
      OR title ILIKE '%capacity planning%' OR abstract ILIKE '%bed management%'
      THEN 'resource-allocation'
    WHEN title ILIKE '%triage%' OR abstract ILIKE '%triage%'
      OR title ILIKE '%severity score%' OR abstract ILIKE '%patient acuity%'
      THEN 'triage-support'
    WHEN title ILIKE '%surge%' OR abstract ILIKE '%mass casualty%'
      OR title ILIKE '%disaster%' OR abstract ILIKE '%mass gathering%'
      THEN 'surge-management'
    WHEN title ILIKE '%surveillance%' OR abstract ILIKE '%syndromic surveillance%'
      OR title ILIKE '%early warning%' OR abstract ILIKE '%outbreak detection%'
      THEN 'surveillance'
    WHEN title ILIKE '%cross-border%' OR abstract ILIKE '%cross-border%'
      OR title ILIKE '%transfrontalier%' OR abstract ILIKE '%transfrontalier%'
      THEN 'cross-border-coordination'
    ELSE NULL
  END),
  geographic_scope = COALESCE(geographic_scope, CASE
    WHEN title ILIKE '%switzerland%' OR abstract ILIKE '%switzerland%'
      OR title ILIKE '%suisse%' OR abstract ILIKE '%geneva%'
      OR abstract ILIKE '%hug%' OR abstract ILIKE '%chuv%'
      THEN 'switzerland'
    WHEN title ILIKE '%france%' OR abstract ILIKE '%french%'
      OR abstract ILIKE '%samu%' OR abstract ILIKE '%smur%'
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
  ROUND(100.0 * COUNT(scenario_type) / COUNT(*), 1) AS pct_scenario,
  COUNT(geographic_scope) AS with_geo,
  ROUND(100.0 * COUNT(geographic_scope) / COUNT(*), 1) AS pct_geo
FROM literature_document
WHERE project_context = 'gesica';
"""

BREAKDOWN_SQL = """
SELECT scenario_type, COUNT(*) FROM literature_document
WHERE project_context = 'gesica' AND scenario_type IS NOT NULL
GROUP BY scenario_type ORDER BY COUNT(*) DESC;
"""

with engine.begin() as conn:
    r = conn.execute(text(STAGE3_SQL))
    print(f"Rows updated: {r.rowcount}")
    for row in conn.execute(text(REPORT_SQL)).fetchall():
        print(dict(row.mapping))
    print("\nScenario breakdown:")
    for row in conn.execute(text(BREAKDOWN_SQL)).fetchall():
        print(dict(row.mapping))
