#!/usr/bin/env python3
"""
LiteRev-Evidence NLP Normalization & Metadata Enrichment Pipeline (Stage 2)
Combines fast SQL heuristics and LLM-based (OpenAI GPT-4o-mini) extraction to backfill empty metadata.
"""
from __future__ import annotations
import os
import sys
import json
import logging
from sqlalchemy import create_engine, text
from openai import OpenAI

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("backfill-stage2")

# Database URL
DB_URL = os.getenv(
    "DB_URL",
    "postgresql+psycopg://literev:MyNewStrongPassword!@10.10.1.10:5432/literev",
)

engine = create_engine(DB_URL, pool_pre_ping=True)

# Fast SQL Heuristics
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

PROMPT_TEMPLATE = """
You are an expert academic metadata extractor specialized in emergency medicine, crisis management, and public health.
Analyze the following document (Title and Abstract) and extract structured metadata matching the schema below.

Title: {title}
Abstract: {abstract}

Schema Specifications:
1. project_context: Choose exactly one from ["gesica", "geoai4ei", "eva", "unassigned"].
   - "gesica": Emergency Medical Services (EMS), ambulance demand forecasting, hospital ER triage, pre-hospital care, resource allocation.
   - "geoai4ei": Climate change, infectious diseases, environmental health, global epidemic surveillance.
   - "eva": Validation methodologies, PRISMA, screening, research evaluation.
2. source_type: Choose exactly one from ["peer-reviewed-journal", "conference-proceedings", "preprint", "report", "other"].
3. disease_or_condition: If relevant, extract the disease or condition (e.g., "COVID-19", "Influenza", "Heatstroke", "Cardiovascular", "Trauma"). Use Title Case. Return null if not applicable.
4. scenario_type: Choose exactly one from ["demand-forecasting", "resource-allocation", "triage-support", "surge-management", "surveillance", "cross-border-coordination", "unassigned"].
5. geographic_scope: Extract the geographic scope (e.g., "France", "Switzerland", "Geneva", "Europe", "Global", "Cross-Border (France-Switzerland)"). Use Title Case. Return null if not applicable.
6. evidence_category: Choose exactly one from ["statistical-modeling", "machine-learning", "clinical-trial", "systematic-review", "expert-consensus", "operational-simulation"].

Return ONLY a valid JSON object matching this structure, with no markdown formatting or extra text:
{{
  "project_context": "...",
  "source_type": "...",
  "disease_or_condition": "..." or null,
  "scenario_type": "...",
  "geographic_scope": "..." or null,
  "evidence_category": "..."
}}
"""

def process_document(client: OpenAI, doc_id: int, title: str, abstract: str) -> dict | None:
    prompt = PROMPT_TEMPLATE.format(title=title, abstract=abstract or "No abstract available.")
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are a precise JSON-only metadata extractor."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        logger.error(f"Error calling OpenAI for doc {doc_id}: {e}")
        return None

def main():
    logger.info("Starting Stage 2 Metadata Backfill Pipeline...")
    
    # 1. Run fast SQL heuristics first
    logger.info("Running fast SQL heuristics...")
    with engine.begin() as conn:
        res = conn.execute(text(STAGE2_SQL))
        logger.info(f"Heuristics complete. Rows updated: {res.rowcount}")

    # 2. Setup OpenAI Client for LLM-based fallback
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set. Skipping LLM enrichment fallback.")
        return

    client = OpenAI(api_key=api_key)

    # Find documents where metadata is still missing or project_context is unassigned
    sql_find = text("""
        SELECT id, title, abstract 
        FROM literature_document 
        WHERE project_context IS NULL 
           OR project_context = 'unassigned'
           OR scenario_type IS NULL
           OR evidence_category IS NULL
        ORDER BY id ASC
    """)
    
    with engine.connect() as conn:
        docs = conn.execute(sql_find).mappings().all()
        
    if not docs:
        logger.info("All metadata is complete! No LLM fallback required.")
        return

    logger.info(f"Found {len(docs)} documents needing LLM metadata enrichment.")
    
    sql_update = text("""
        UPDATE literature_document
        SET 
            project_context = :project_context,
            source_type = :source_type,
            disease_or_condition = :disease_or_condition,
            scenario_type = :scenario_type,
            geographic_scope = :geographic_scope,
            evidence_category = :evidence_category
        WHERE id = :id
    """)
    
    success_count = 0
    for doc in docs:
        doc_id = doc["id"]
        title = doc["title"]
        abstract = doc["abstract"]
        
        logger.info(f"Processing doc {doc_id}: {title[:60]}...")
        metadata = process_document(client, doc_id, title, abstract)
        
        if metadata:
            logger.info(f"  Extracted: Context={metadata.get('project_context')}, Scenario={metadata.get('scenario_type')}, Category={metadata.get('evidence_category')}")
            try:
                with engine.begin() as conn:
                    conn.execute(sql_update, {
                        "id": doc_id,
                        "project_context": metadata.get("project_context"),
                        "source_type": metadata.get("source_type"),
                        "disease_or_condition": metadata.get("disease_or_condition"),
                        "scenario_type": metadata.get("scenario_type"),
                        "geographic_scope": metadata.get("geographic_scope"),
                        "evidence_category": metadata.get("evidence_category"),
                    })
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to update database for doc {doc_id}: {e}")
        else:
            logger.warning(f"  Skipping doc {doc_id} due to extraction failure.")

    logger.info(f"Pipeline finished. Successfully enriched {success_count}/{len(docs)} documents.")

if __name__ == "__main__":
    main()
