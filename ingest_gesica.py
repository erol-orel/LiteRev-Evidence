#!/usr/bin/env python3
"""GESICA-focused PubMed ingestion: EMS, hospital surge, cross-border crisis."""
from __future__ import annotations
import sys, os, time, importlib.util

sys.path.insert(0, '/opt/literev-api')
os.environ.setdefault('DB_URL', 'postgresql+psycopg://literev:MyNewStrongPassword!@10.10.1.10:5432/literev')
os.environ.setdefault('EMBED_MODEL_NAME', 'BAAI/bge-m3')
os.environ.setdefault('WRITE_API_KEY', 'LiteRev2026!')

GESICA_QUERIES = [
    "emergency medical services demand forecasting AI machine learning",
    "ambulance dispatch optimization algorithm hospital",
    "hospital surge capacity planning pandemic crisis",
    "cross-border healthcare coordination emergency",
    "EMS triage prediction neural network",
    "emergency department overcrowding prediction model",
    "mass casualty incident resource allocation optimization",
    "prehospital emergency care AI outcome prediction",
    "disaster preparedness healthcare resource planning Switzerland France",
    "sanitary crisis management decision support system",
]

spec = importlib.util.spec_from_file_location("ingest_pubmed", "/opt/literev-api/ingest_pubmed.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

total_ok = 0
total_skip = 0
for q in GESICA_QUERIES:
    print(f"\n==> Query: {q}")
    pmids = mod.esearch(q, retmax=25)
    articles = mod.efetch(pmids)
    for art in articles:
        if mod.already_exists(art.pmid):
            print(f"  SKIP (exists) {art.pmid}")
            total_skip += 1
            continue
        ok = mod.ingest_article(art, "gesica")
        status = "OK" if ok else "SKIP(ingest)"
        print(f"  {status} {art.pmid}: {art.title[:70]}")
        if ok:
            total_ok += 1
    time.sleep(0.4)

print(f"\n=== Done: {total_ok} ingested, {total_skip} already existed ===")
