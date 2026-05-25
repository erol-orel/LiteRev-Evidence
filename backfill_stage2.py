#!/usr/bin/env python3
"""Stage 2 backfill — extended NLP rules for disease_or_condition and geographic_scope."""
from __future__ import annotations
import os
from sqlalchemy import create_engine, text

DB_URL = os.environ["DB_URL"]
engine = create_engine(DB_URL, pool_pre_ping=True)

STAGE2_SQL = """
UPDATE literature_document
SET
  disease_or_condition = COALESCE(disease_or_condition, (
    CASE
      WHEN title ILIKE '%covid%' OR abstract ILIKE '%covid%'
        OR title ILIKE '%sars-cov%' OR abstract ILIKE '%sars-cov%'        THEN 'covid-19'
      WHEN title ILIKE '%influenza%' OR abstract ILIKE '%influenza%'
        OR title ILIKE '%flu%' OR abstract ILIKE '%flu%'                  THEN 'influenza'
      WHEN title ILIKE '%tuberculosis%' OR abstract ILIKE '%tuberculosis%'
        OR title ILIKE '%% tb %' OR abstract ILIKE '% tb %'              THEN 'tuberculosis'
      WHEN title ILIKE '%hiv%' OR abstract ILIKE '%hiv%'
        OR title ILIKE '%aids%' OR abstract ILIKE '%aids%'                THEN 'hiv'
      WHEN title ILIKE '%cholera%' OR abstract ILIKE '%cholera%'          THEN 'cholera'
      WHEN title ILIKE '%ebola%' OR abstract ILIKE '%ebola%'              THEN 'ebola'
      WHEN title ILIKE '%mpox%' OR abstract ILIKE '%mpox%'
        OR title ILIKE '%monkeypox%' OR abstract ILIKE '%monkeypox%'      THEN 'mpox'
      WHEN title ILIKE '%dengue%' OR abstract ILIKE '%dengue%'            THEN 'dengue'
      WHEN title ILIKE '%malaria%' OR abstract ILIKE '%malaria%'          THEN 'malaria'
      WHEN title ILIKE '%legionella%' OR abstract ILIKE '%legionella%'    THEN 'legionella'
      WHEN title ILIKE '%klebsiella%' OR abstract ILIKE '%klebsiella%'    THEN 'klebsiella'
      WHEN title ILIKE '%salmonella%' OR abstract ILIKE '%salmonella%'    THEN 'salmonella'
      WHEN title ILIKE '%norovirus%' OR abstract ILIKE '%norovirus%'      THEN 'norovirus'
      WHEN title ILIKE '%hepatitis%' OR abstract ILIKE '%hepatitis%'      THEN 'hepatitis'
      WHEN title ILIKE '%meningitis%' OR abstract ILIKE '%meningitis%'    THEN 'meningitis'
      WHEN title ILIKE '%infectious disease%'
        OR abstract ILIKE '%infectious disease%'                          THEN 'infectious_disease_general'
      ELSE NULL
    END
  )),
  geographic_scope = COALESCE(geographic_scope, (
    CASE
      WHEN title ILIKE '%switzerland%' OR abstract ILIKE '%switzerland%'
        OR title ILIKE '%suisse%' OR abstract ILIKE '%suisse%'            THEN 'switzerland'
      WHEN title ILIKE '%france%' OR abstract ILIKE '%france%'            THEN 'france'
      WHEN title ILIKE '%germany%' OR abstract ILIKE '%germany%'          THEN 'germany'
      WHEN title ILIKE '%united kingdom%' OR abstract ILIKE '%united kingdom%'
        OR title ILIKE '%% uk %' OR abstract ILIKE '% uk %'              THEN 'uk'
      WHEN title ILIKE '%europe%' OR abstract ILIKE '%europe%'
        OR title ILIKE '%european%' OR abstract ILIKE '%european%'        THEN 'europe'
      WHEN title ILIKE '%africa%' OR abstract ILIKE '%africa%'
        OR title ILIKE '%african%' OR abstract ILIKE '%african%'          THEN 'africa'
      WHEN title ILIKE '%bangladesh%' OR abstract ILIKE '%bangladesh%'    THEN 'asia'
      WHEN title ILIKE '%china%' OR abstract ILIKE '%china%'
        OR title ILIKE '%chinese%' OR abstract ILIKE '%chinese%'          THEN 'asia'
      WHEN title ILIKE '%india%' OR abstract ILIKE '%india%'              THEN 'asia'
      WHEN title ILIKE '%asia%' OR abstract ILIKE '%asia%'                THEN 'asia'
      WHEN title ILIKE '%united states%' OR abstract ILIKE '%united states%'
        OR title ILIKE '%% usa%' OR abstract ILIKE '% usa%'              THEN 'usa'
      WHEN title ILIKE '%canada%' OR abstract ILIKE '%canada%'            THEN 'north_america'
      WHEN title ILIKE '%lebanon%' OR abstract ILIKE '%lebanon%'          THEN 'middle_east'
      WHEN title ILIKE '%global%' OR abstract ILIKE '%global%'
        OR title ILIKE '%worldwide%' OR abstract ILIKE '%worldwide%'
        OR title ILIKE '%international%' OR abstract ILIKE '%international%' THEN 'global'
      ELSE NULL
    END
  ))
WHERE source = 'pubmed'
  AND (disease_or_condition IS NULL OR geographic_scope IS NULL);
"""

REPORT_SQL = """
SELECT
  COUNT(*) as total,
  COUNT(disease_or_condition) as with_disease,
  ROUND(100.0 * COUNT(disease_or_condition) / COUNT(*), 1) as pct_disease,
  COUNT(geographic_scope) as with_geo,
  ROUND(100.0 * COUNT(geographic_scope) / COUNT(*), 1) as pct_geo
FROM literature_document
WHERE source = 'pubmed';
"""

with engine.begin() as conn:
    result = conn.execute(text(STAGE2_SQL))
    print(f"Rows updated: {result.rowcount}")
    rows = conn.execute(text(REPORT_SQL)).fetchall()
    for row in rows:
        print(dict(row._mapping))
