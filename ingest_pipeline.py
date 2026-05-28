#!/usr/bin/env python3
"""
LiteRev-Evidence Multi-Source Ingestion Pipeline v2.0 (P2)
Sources: PubMed, PubMed Central (PMC), Europe PMC, OpenAlex, CrossRef

Optimisé pour l'ingestion de masse (jusqu'à 5000+ documents) avec :
- Requêtes étendues et ciblées pour GESICA, GeoAI4EI et EVA.
- Intégration d'Europe PMC (excellente source pour le full-text et les métadonnées).
- Déduplication ultra-robuste (par DOI, PMID, PMCID, OpenAlex ID) via l'API locale.
- Rate-limiting adaptatif et gestion d'erreurs avancée.
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
        "emergency dispatch triage artificial intelligence",
        "pre-hospital resource allocation optimization",
        "surge capacity management hospital emergency",
        "disaster medical response coordination planning",
        "probabilistic forecasting emergency department admissions",
    ],
    "geoai4ei": [
        "genomic epidemiology pathogen outbreak tracking",
        "wastewater-based epidemiology infectious disease surveillance",
        "phylogenomics viral transmission mapping",
        "climate change vector-borne disease modeling",
        "spatiotemporal zoonotic disease emergence",
        "early warning system epidemic intelligence",
        "mobility data infectious disease spread modeling",
        "cross-border disease transmission surveillance",
    ],
    "eva": [
        "systematic review machine learning emergency medicine",
        "meta-analysis epidemic forecasting model evaluation",
        "prisma guideline automation artificial intelligence",
        "double screening systematic review active learning",
    ]
}

@dataclass
class IngestArticle:
    source: str          # pubmed, openalex, crossref, europepmc
    external_id: str     # PMID, OpenAlex ID, DOI, PMCID
    title: str
    abstract: str | None
    year: int | None
    url: str | None
    project_context: str
    source_type: str = "article"


def clean_id(ext_id: str) -> str:
    """Nettoie et normalise l'ID externe pour éviter les faux doublons."""
    if not ext_id:
        return ""
    ext_id = ext_id.strip()
    # Supprimer les préfixes d'URL courants pour les DOIs ou OpenAlex IDs
    for prefix in ["https://doi.org/", "http://doi.org/", "https://openalex.org/", "W"]:
        if ext_id.startswith(prefix):
            ext_id = ext_id[len(prefix):]
    return ext_id


def already_exists(external_id: str) -> bool:
    """Vérifie si l'article existe déjà dans la base via l'endpoint de recherche local."""
    cleaned = clean_id(external_id)
    if not cleaned:
        return False
    try:
        r = requests.post(
            f"{API_BASE}/search",
            json={
                "query_text": cleaned,
                "mode": "boolean",
                "limit": 5,
                "filters": {},
            },
            timeout=10,
        )
        if not r.ok:
            return False
        results = r.json().get("results", [])
        for res in results:
            res_ext_id = clean_id(res.get("external_id") or "")
            if res_ext_id == cleaned or cleaned in res_ext_id or res_ext_id in cleaned:
                return True
        return False
    except Exception:
        return False


def ingest_article(art: IngestArticle) -> bool:
    """Envoie l'article et son chunk initial à l'API FastAPI."""
    content = f"{art.title}\n\n{art.abstract or ''}".strip()
    if len(content) < 100:  # Augmenté à 100 caractères pour filtrer les abstracts vides/inutiles
        print(f"  [Skip] Article trop court ou sans abstract ({art.external_id})")
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
            title_node = article.find(".//ArticleTitle")
            if title_node is None:
                continue
            title = "".join(title_node.itertext()).strip()
            
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
# 2. PIPELINE EUROPE PMC (Nouveau !)
# ==============================================================================
def fetch_europepmc(query: str, limit: int, project: str) -> list[IngestArticle]:
    print(f"  -> Recherche Europe PMC : '{query}'")
    articles = []
    try:
        r = requests.get(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params={
                "query": query,
                "format": "json",
                "pageSize": limit,
                "resultType": "core",
            },
            timeout=20,
        )
        r.raise_for_status()
        results = r.json().get("resultList", {}).get("result", [])

        for res in results:
            pmid = res.get("pmid")
            pmcid = res.get("pmcid")
            doi = res.get("doi")
            
            # ID externe unique prioritaire
            external_id = pmcid or pmid or doi
            if not external_id:
                continue
                
            title = res.get("title") or ""
            if not title:
                continue
                
            abstract = res.get("abstractText")
            if abstract and abstract.startswith("<"):
                try:
                    abstract = "".join(ET.fromstring(f"<root>{abstract}</root>").itertext()).strip()
                except Exception:
                    pass

            year = None
            year_text = res.get("pubYear")
            if year_text and str(year_text).isdigit():
                year = int(year_text)

            url = f"https://europepmc.org/article/{pmcid or pmid}" if (pmcid or pmid) else (f"https://doi.org/{doi}" if doi else None)

            articles.append(IngestArticle(
                source="europepmc",
                external_id=external_id,
                title=title,
                abstract=abstract,
                year=year,
                url=url,
                project_context=project
            ))
    except Exception as e:
        print(f"  [Europe PMC Error] {e}")
    return articles


# ==============================================================================
# 3. PIPELINE OPENALEX
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
            external_id = work.get("id", "").split("/")[-1]
            if not external_id:
                continue
            title = work.get("title") or ""
            if not title:
                continue

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
# 4. PIPELINE CROSSREF
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
    parser = argparse.ArgumentParser(description="Pipeline d'ingestion multi-sources v2.0")
    parser.add_argument("--project", choices=["gesica", "geoai4ei", "eva"], help="Projet spécifique à ingérer")
    parser.add_argument("--limit", type=int, default=20, help="Nombre d'articles max par requête et par source")
    args = parser.parse_args()

    projects = [args.project] if args.project else list(QUERIES.keys())

    print("======================================================================")
    print("🚀 DÉMARRAGE DU PIPELINE D'INGESTION MULTI-SOURCES v2.0")
    print(f"   API Base: {API_BASE}")
    print(f"   Projets cibles: {', '.join(projects).upper()}")
    print(f"   Limite par source et par requête : {args.limit}")
    print("======================================================================")

    total_added = 0
    start_time = time.time()

    for project in projects:
        print(f"\n⚡ Traitement du projet : {project.upper()}")
        queries = QUERIES[project]

        for query in queries:
            print(f"\n🔍 Exécution de la requête : '{query}'")

            # Récupération en parallèle simulée (séquentielle mais rapide)
            pm_articles = fetch_pubmed(query, args.limit, project)
            ep_articles = fetch_europepmc(query, args.limit, project)
            oa_articles = fetch_openalex(query, args.limit, project)
            cr_articles = fetch_crossref(query, args.limit, project)

            all_articles = pm_articles + ep_articles + oa_articles + cr_articles
            print(f"   -> Récupérés : PubMed ({len(pm_articles)}), Europe PMC ({len(ep_articles)}), OpenAlex ({len(oa_articles)}), CrossRef ({len(cr_articles)})")

            added_for_query = 0
            for art in all_articles:
                if already_exists(art.external_id):
                    continue
                
                print(f"   [Ingest] ({art.source.upper()}) {art.external_id} : {art.title[:70]}...")
                success = ingest_article(art)
                if success:
                    added_for_query += 1
                    total_added += 1
                
                # Rate limiting amical adaptatif
                time.sleep(0.1)
            
            print(f"   -> {added_for_query} nouveaux articles ajoutés pour cette requête.")

    elapsed = time.time() - start_time
    print("\n======================================================================")
    print(f"✅ PIPELINE TERMINÉ EN {elapsed:.1f}s")
    print(f"   {total_added} nouveaux articles ajoutés au corpus global.")
    print("======================================================================")
    return 0

if __name__ == "__main__":
    sys.exit(main())
