#!/usr/bin/env python3
"""GESICA-focused PubMed ingestion: EMS, hospital surge, cross-border crisis."""
from __future__ import annotations
import sys, os
sys.path.insert(0, '/opt/literev-api')
os.environ.setdefault('DB_URL', 'postgresql+psycopg://literev:MyNewStrongPassword!@10.10.1.10:5432/literev')
os.environ.setdefault('EMBED_MODEL_NAME', 'BAAI/bge-m3')
os.environ.setdefault('WRITE_API_KEY', 'LiteRev2026!')

QUERIES = {
    "gesica": [
        "emergency medical services demand forecasting AI machine learning",
        "ambulance dispatch optimization algorithm hospital",
        "hospital surge capacity planning pandemic crisis",
        "cross-border healthcare coordination emergency",
        "EMS triage prediction neural network",
        "sanitary crisis management decision support system",
        "emergency department overcrowding prediction model",
        "mass casualty incident resource allocation optimization",
        "prehospital emergency care AI outcome prediction",
        "disaster preparedness healthcare resource planning Switzerland France",
    ]
}

import importlib.util, subprocess
spec = importlib.util.spec_from_file_location("ingest_pubmed", "/opt/literev-api/ingest_pubmed.py")
mod = importlib.util.load_from_spec(spec)
spec.loader.exec_module(mod)

# Override QUERIES in ingest_pubmed module and run
mod.QUERIES = QUERIES
for ctx, queries in QUERIES.items():
    for q in queries:
        pmids = mod.esearch(q, retmax=25)
        articles = mod.efetch(pmids)
        for art in articles:
            if mod.already_exists(art.pmid):
                print(f"  SKIP {art.pmid}")
                continue
            ok = mod.ingest_article(art, ctx)
            print(f"  {'OK' if ok else 'SKIP'} {art.pmid}: {art.title[:60]}")
        import time; time.sleep(0.4)
