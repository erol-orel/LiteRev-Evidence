#!/usr/bin/env python3
"""
LiteRev-Evidence Multi-Source Ingestion Pipeline
Sources: PubMed, PubMed Central (PMC), OpenAlex, CrossRef
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

import requests

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")
WRITE_API_KEY = os.getenv("WRITE_API_KEY", "LiteRev2026!")
HEADERS = {"X-Api-Key": WRITE_API_KEY}

# --- CONFIGURATION DES REQUÊTES PAR PROJET ---
QUERIES = {
    "gesica": [
        "emergency medical services forecasting machine learning",
        "ambulance demand prediction spatial temporal",
        "hospital emergency department overcrowding prediction",
        "syndromic surveillance early warning outbreak",
        "cross-border health emergency crisis management",
    ],
    "geoai4ei": [
        "genomic epidemiology pathogen outbreak tracking",
        "wastewater-based epidemiology infectious disease surveillance",
        "phylogenomics viral transmission mapping",
        "climate change vector-borne disease modeling",
    ],
    "eva": [
        "systematic review machine learning emergency medicine",
        "meta-analysis epidemic forecasting model evaluation",
    ]
}

@dataclass
class IngestArticle:
    source: str          # pubmed, openalex, crossref
    external_id: str     # PMID, OpenAlex ID, DOI
    title: str
    abstract: str | None
    year: int | None
    url: str | None
    project_context: str
    source_type: str = "article"

def already_exists(external_id: str) -> bool:
    """Vérifie si l'article existe déjà dans la base via l'endpoint de recherche."""
    try:
        r = requests.post(
            f"{API_BASE}/search",
            json={
                "query_text": external_id,
                "mode": "boolean",
                "limit": 3,
                "filters": {},
            },
            timeout=10,
        )
        if not r.ok:
            return False
        results = r.json().get("results", [])
        for res in results:
            if str(res.get("external_id") or "") == str(external_id):
                return True
        return False
    except Exception:
        return False

def ingest_article(art: IngestArticle) -> bool:
    """Envoie l'article et son chunk initial à l'API FastAPI."""
    content = f"{art.title}\n\n{art.abstract or ''}".strip()
    if len(content) < 50:
        print(f"  [Skip] Article trop court ({art.external_id})")
        return False

    try:
        # 1. Créer le document
        doc_r = requests.post(
            f"{API_BASE}/documents",
            headers=HEADERS,
            json={
                "source": art.source,
                "title": art.title,
                "abstract": art.abstract,
                "year": art.year,
                "url": art.url,
                "external_id": art.external_id,
                "project_context": art.project_context,
                "source_type": art.source_type,
                "disease_or_condition": None,
                "scenario_type": None,
                "geographic_scope": None,
                "evidence_category": None,
            },
            timeout=20,
        )
        doc_r.raise_for_status()
        doc_id = doc_r.json()["id"]

        # 2. Créer le chunk initial (titre + abstract)
        chunk_r = requests.post(
            f"{API_BASE}/chunks",
            headers=HEADERS,
            json={
                "document_id": doc_id,
                "chunk_index": 0,
                "content": content,
                "chunk_type": "title_abstract",
                "section_label": None,
                "char_start": None,
                "char_end": None,
                "token_count": len(content.split()),
                "chunk_weight": 1.0,
                "metadata_json": {},
            },
            timeout=30,
        )
        chunk_r.raise_for_status()
        return True
    except Exception as e:
        print(f"  [Error] Échec ingestion {art.external_id}: {e}")
        return False

# ==============================================================================
# 1. PIPELINE PUBMED
# ==============================================================================
def fetch_pubmed(query: str, limit: int, project: str) -> list[IngestArticle]:
    print(f"  -> Recherche PubMed : '{query}'")
    articles = []
    try:
        # Search
        r = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={
                "db": "pubmed",
                "term": query,
                "retmax": limit,
                "retmode": "json",
                "email": "literev@example.com",
            },
            timeout=20,
        )
        r.raise_for_status()
        pmids = r.json().get("esearchresult", {}).get("idlist", [])
        if not pmids:
            return []

        # Fetch
        rf = requests.post(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            data={
                "db": "pubmed",
                "id": ",".join(pmids),
                "rettype": "xml",
                "retmode": "xml",
                "email": "literev@example.com",
            },
            timeout=40,
        )
        rf.raise_for_status()
        root = ET.fromstring(rf.content)

        for article in root.findall(".//PubmedArticle"):
            pmid = article.findtext(".//PMID") or ""
            if not pmid:
                continue
            title = "".join(article.find(".//ArticleTitle").itertext()).strip() if article.find(".//ArticleTitle") is not None else ""
            abstract_parts = []
            for node in article.findall(".//Abstract/AbstractText"):
                txt = "".join(node.itertext()).strip()
                if txt:
                    abstract_parts.append(txt)
            abstract = " ".join(abstract_parts).strip()

            year = None
            year_text = article.findtext(".//PubDate/Year") or article.findtext(".//ArticleDate/Year") or ""
            if year_text[:4].isdigit():
                year = int(year_text[:4])

            articles.append(IngestArticle(
                source="pubmed",
                external_id=pmid,
                title=title,
                abstract=abstract or None,
                year=year,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                project_context=project
            ))
    except Exception as e:
        print(f"  [PubMed Error] {e}")
    return articles

# ==============================================================================
# 2. PIPELINE OPENALEX
# ==============================================================================
def fetch_openalex(query: str, limit: int, project: str) -> list[IngestArticle]:
    print(f"  -> Recherche OpenAlex : '{query}'")
    articles = []
    try:
        r = requests.get(
            "https://api.openalex.org/works",
            params={
                "search": query,
                "per_page": limit,
                "mailto": "literev@example.com",
            },
            timeout=20,
        )
        r.raise_for_status()
        results = r.json().get("results", [])

        for work in results:
            external_id = work.get("id", "").split("/")[-1] # e.g., W428543285
            if not external_id:
                continue
            title = work.get("title") or ""
            if not title:
                continue

            # OpenAlex abstract est stocké en index inversé
            abstract = None
            inv_index = work.get("abstract_inverted_index")
            if inv_index:
                try:
                    words = {}
                    for word, positions in inv_index.items():
                        for pos in positions:
                            words[pos] = word
                    abstract = " ".join([words[i] for i in sorted(words.keys())])
                except Exception:
                    pass

            year = work.get("publication_year")
            url = work.get("doi") or work.get("ids", {}).get("wikipedia") or f"https://openalex.org/{external_id}"

            articles.append(IngestArticle(
                source="openalex",
                external_id=external_id,
                title=title,
                abstract=abstract,
                year=year,
                url=url,
                project_context=project
            ))
    except Exception as e:
        print(f"  [OpenAlex Error] {e}")
    return articles

# ==============================================================================
# 3. PIPELINE CROSSREF
# ==============================================================================
def fetch_crossref(query: str, limit: int, project: str) -> list[IngestArticle]:
    print(f"  -> Recherche CrossRef : '{query}'")
    articles = []
    try:
        r = requests.get(
            "https://api.crossref.org/works",
            params={
                "query": query,
                "rows": limit,
                "mailto": "literev@example.com",
            },
            timeout=20,
        )
        r.raise_for_status()
        items = r.json().get("message", {}).get("items", [])

        for item in items:
            doi = item.get("DOI")
            if not doi:
                continue
            titles = item.get("title", [])
            title = titles[0] if titles else ""
            if not title:
                continue

            abstract = item.get("abstract")
            # Nettoyer les tags JATS XML de CrossRef si présents
            if abstract and abstract.startswith("<"):
                try:
                    abstract = "".join(ET.fromstring(abstract).itertext()).strip()
                except Exception:
                    pass

            year = None
            created = item.get("created", {}).get("date-parts", [])
            if created and created[0]:
                year = created[0][0]

            articles.append(IngestArticle(
                source="crossref",
                external_id=doi,
                title=title,
                abstract=abstract,
                year=year,
                url=f"https://doi.org/{doi}",
                project_context=project
            ))
    except Exception as e:
        print(f"  [CrossRef Error] {e}")
    return articles

# ==============================================================================
# ENTRYPOINT
# ==============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description="Pipeline d'ingestion multi-sources")
    parser.add_argument("--project", choices=["gesica", "geoai4ei", "eva"], help="Projet spécifique à ingérer")
    parser.add_argument("--limit", type=int, default=10, help="Nombre d'articles max par requête et par source")
    args = parser.parse_args()

    projects = [args.project] if args.project else list(QUERIES.keys())

    print("======================================================================")
    print("🚀 DÉMARRAGE DU PIPELINE D'INGESTION MULTI-SOURCES")
    print(f"   API Base: {API_BASE}")
    print(f"   Projets cibles: {', '.join(projects).upper()}")
    print(f"   Limite par source : {args.limit}")
    print("======================================================================")

    total_added = 0

    for project in projects:
        print(f"\n⚡ Traitement du projet : {project.upper()}")
        queries = QUERIES[project]

        for query in queries:
            print(f"\n🔍 Exécution de la requête : '{query}'")

            # 1. PubMed
            pm_articles = fetch_pubmed(query, args.limit, project)
            # 2. OpenAlex
            oa_articles = fetch_openalex(query, args.limit, project)
            # 3. CrossRef
            cr_articles = fetch_crossref(query, args.limit, project)

            all_articles = pm_articles + oa_articles + cr_articles
            print(f"   -> Récupérés : PubMed ({len(pm_articles)}), OpenAlex ({len(oa_articles)}), CrossRef ({len(cr_articles)})")

            for art in all_articles:
                if already_exists(art.external_id):
                    continue
                
                print(f"   [Ingest] ({art.source.upper()}) {art.external_id} : {art.title[:70]}...")
                success = ingest_article(art)
                if success:
                    total_added += 1
                
                # Rate limiting amical
                time.sleep(0.2)

    print("\n======================================================================")
    print(f"✅ PIPELINE TERMINÉ — {total_added} nouveaux articles ajoutés au corpus")
    print("======================================================================")
    return 0

if __name__ == "__main__":
    sys.exit(main())
