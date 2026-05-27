#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
import time

os.environ.setdefault("DB_URL", "postgresql+psycopg://literev:MyNewStrongPassword!@10.10.1.10:5432/literev")
os.environ.setdefault("WRITE_API_KEY", "LiteRev2026!")
os.environ.setdefault("EMBED_MODEL_NAME", "BAAI/bge-m3")

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

for q in GESICA_QUERIES:
    print(f"\n==> Query: {q}")
    cmd = [sys.executable, "ingest_pubmed.py", "--project", "gesica", "--query", q]
    proc = subprocess.run(cmd, cwd="/opt/literev-api")
    if proc.returncode != 0:
        print(f"ERROR on query: {q}")
    time.sleep(0.5)
