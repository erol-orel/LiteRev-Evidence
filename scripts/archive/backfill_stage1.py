#!/usr/bin/env python3
from __future__ import annotations

import os
from sqlalchemy import create_engine, text

DB_URL = os.environ["DB_URL"]
engine = create_engine(DB_URL, pool_pre_ping=True)

STAGE1_SOURCES = ["pubmed", "pmc", "openalex", "crossref"]

NORMALIZE_SQL = """
UPDATE literature_document d
SET
  source_type = COALESCE(d.source_type, n.source_type),
  disease_or_condition = COALESCE(d.disease_or_condition, n.disease_or_condition),
  scenario_type = COALESCE(d.scenario_type, n.scenario_type),
  geographic_scope = COALESCE(d.geographic_scope, n.geographic_scope),
  evidence_category = COALESCE(d.evidence_category, n.evidence_category)
FROM (
  SELECT id,
         CASE
           WHEN source = 'pubmed' AND title ILIKE '%systematic review%' THEN 'systematic_review'
           WHEN source = 'pubmed' AND title ILIKE '%meta-analysis%' THEN 'meta_analysis'
           WHEN source = 'pubmed' AND title ILIKE '%review%' THEN 'review'
           WHEN source = 'openalex' AND source_type = 'review' THEN 'review'
           ELSE NULL
         END AS source_type,
         CASE
           WHEN title ILIKE '%covid%' OR abstract ILIKE '%covid%' THEN 'covid-19'
           WHEN title ILIKE '%influenza%' OR abstract ILIKE '%influenza%' THEN 'influenza'
           WHEN title ILIKE '%tuberculosis%' OR abstract ILIKE '%tuberculosis%' THEN 'tuberculosis'
           WHEN title ILIKE '%hiv%' OR abstract ILIKE '%hiv%' THEN 'hiv'
           ELSE NULL
         END AS disease_or_condition,
         CASE
           WHEN title ILIKE '%outbreak%' OR abstract ILIKE '%outbreak%' THEN 'outbreak_detection'
           WHEN title ILIKE '%surveillance%' OR abstract ILIKE '%surveillance%' THEN 'surveillance'
           WHEN title ILIKE '%forecast%' OR abstract ILIKE '%forecast%' THEN 'forecasting'
           WHEN title ILIKE '%wastewater%' OR abstract ILIKE '%wastewater%' THEN 'wastewater_surveillance'
           WHEN title ILIKE '%genomic%' OR abstract ILIKE '%genomic%' OR title ILIKE '%sequencing%' OR abstract ILIKE '%sequencing%' THEN 'genomic_epidemiology'
           ELSE NULL
         END AS scenario_type,
         CASE
           WHEN title ILIKE '%france%' OR abstract ILIKE '%france%' THEN 'france'
           WHEN title ILIKE '%europe%' OR abstract ILIKE '%europe%' THEN 'europe'
           WHEN title ILIKE '%africa%' OR abstract ILIKE '%africa%' THEN 'africa'
           WHEN title ILIKE '%global%' OR abstract ILIKE '%global%' THEN 'global'
           WHEN title ILIKE '%united states%' OR abstract ILIKE '%united states%' THEN 'usa'
           ELSE NULL
         END AS geographic_scope,
         CASE
           WHEN title ILIKE '%systematic review%' OR abstract ILIKE '%systematic review%' THEN 'systematic_review'
           WHEN title ILIKE '%meta-analysis%' OR abstract ILIKE '%meta-analysis%' THEN 'systematic_review'
           WHEN title ILIKE '%guideline%' OR abstract ILIKE '%guideline%' THEN 'guideline'
           WHEN title ILIKE '%cohort%' OR abstract ILIKE '%cohort%' OR title ILIKE '%cross-sectional%' OR abstract ILIKE '%cross-sectional%' THEN 'observational_study'
           WHEN title ILIKE '%trial%' OR abstract ILIKE '%trial%' THEN 'interventional_study'
           WHEN title ILIKE '%model%' OR abstract ILIKE '%model%' OR title ILIKE '%algorithm%' OR abstract ILIKE '%algorithm%' THEN 'methodological'
           WHEN title ILIKE '%surveillance%' OR abstract ILIKE '%surveillance%' THEN 'surveillance_report'
           ELSE NULL
         END AS evidence_category
  FROM literature_document
  WHERE source = ANY(:sources)
) n
WHERE d.id = n.id
  AND (
    d.source_type IS NULL OR
    d.disease_or_condition IS NULL OR
    d.scenario_type IS NULL OR
    d.geographic_scope IS NULL OR
    d.evidence_category IS NULL
  );
"""

REPORT_SQL = """
SELECT
  source,
  COUNT(*) AS total_docs,
  COUNT(source_type) AS with_source_type,
  COUNT(disease_or_condition) AS with_disease,
  COUNT(scenario_type) AS with_scenario,
  COUNT(geographic_scope) AS with_geo,
  COUNT(evidence_category) AS with_evidence
FROM literature_document
GROUP BY source
ORDER BY total_docs DESC, source ASC;
"""

if __name__ == "__main__":
    with engine.begin() as conn:
        conn.execute(text(NORMALIZE_SQL), {"sources": STAGE1_SOURCES})
        rows = conn.execute(text(REPORT_SQL)).mappings().all()

    print("BACKFILL_DONE")
    for row in rows:
        print(dict(row))
