#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import requests

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")
ENTREZ_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
EMAIL = os.getenv("PUBMED_EMAIL", "literev@example.com")
WRITE_API_KEY = os.getenv("WRITE_API_KEY", "")
HEADERS = {"X-Api-Key": WRITE_API_KEY}
MAX_PER_QUERY = 20

DEFAULT_QUERIES = {
    "geoai4ei": [
        "genomic epidemiology outbreak whole genome sequencing surveillance",
        "phylogenomics infectious disease outbreak detection",
        "SARS-CoV-2 genomic surveillance wastewater",
    ],
    "gesica": [
        "infectious disease surveillance system public health",
        "epidemic intelligence early warning system",
        "syndromic surveillance outbreak detection algorithm",
    ],
    "eva": [
        "systematic review infectious disease epidemiology",
        "meta-analysis epidemic modeling prediction",
        "evidence synthesis public health intervention",
    ],
}

@dataclass
class PubMedArticle:
    pmid: str
    title: str
    abstract: str
    year: int | None
    url: str

def esearch(query: str, retmax: int = MAX_PER_QUERY) -> list[str]:
    r = requests.get(
        f"{ENTREZ_BASE}/esearch.fcgi",
        params={
            "db": "pubmed",
            "term": query,
            "retmax": retmax,
            "retmode": "json",
            "email": EMAIL,
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["esearchresult"]["idlist"]

def efetch(pmids: list[str]) -> list[PubMedArticle]:
    if not pmids:
        return []

    r = requests.post(
        f"{ENTREZ_BASE}/efetch.fcgi",
        data={
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "xml",
            "retmode": "xml",
            "email": EMAIL,
        },
        timeout=60,
    )
    r.raise_for_status()
    root = ET.fromstring(r.content)

    articles: list[PubMedArticle] = []
    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID") or ""
        title = "".join(article.find(".//ArticleTitle").itertext()).strip() if article.find(".//ArticleTitle") is not None else ""
        abstract_parts = []
        for node in article.findall(".//Abstract/AbstractText"):
            txt = "".join(node.itertext()).strip()
            if txt:
                abstract_parts.append(txt)
        abstract = " ".join(abstract_parts).strip()

        year = None
        year_text = (
            article.findtext(".//PubDate/Year")
            or article.findtext(".//ArticleDate/Year")
            or ""
        )
        if year_text[:4].isdigit():
            year = int(year_text[:4])

        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        if pmid and title:
            articles.append(PubMedArticle(pmid=pmid, title=title, abstract=abstract, year=year, url=url))
    return articles

def already_exists(pmid: str) -> bool:
    r = requests.post(
        f"{API_BASE}/search",
        json={
            "query_text": pmid,
            "mode": "boolean",
            "limit": 3,
            "filters": {},
        },
        timeout=20,
    )
    if not r.ok:
        return False
    results = r.json().get("results", [])
    for res in results:
        if str(res.get("external_id") or "") == str(pmid):
            return True
    return False

def ingest_article(article: PubMedArticle, project_context: str) -> bool:
    content = f"{article.title}\n\n{article.abstract}".strip()
    if len(content) < 50:
        print(f"  skipped too short {article.pmid}")
        return False

    doc_r = requests.post(
        f"{API_BASE}/documents",
        headers=HEADERS,
        json={
            "source": "pubmed",
            "title": article.title,
            "abstract": article.abstract or None,
            "year": article.year,
            "url": article.url,
            "external_id": article.pmid,
            "project_context": project_context,
            "source_type": "article",
            "disease_or_condition": None,
            "scenario_type": None,
            "geographic_scope": None,
            "evidence_category": None,
        },
        timeout=30,
    )
    doc_r.raise_for_status()
    doc_id = doc_r.json()["id"]

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
        timeout=60,
    )
    chunk_r.raise_for_status()
    return True

def run_project(project: str, queries: list[str]) -> int:
    print(f"\n=== {project.upper()} ===")
    total_new = 0
    seen: set[str] = set()

    for query in queries:
        print(f"Query: {query}")
        try:
            pmids = esearch(query)
            articles = efetch(pmids)
        except Exception as e:
            print(f"  ERROR fetch: {e}")
            continue

        for art in articles:
            if art.pmid in seen:
                continue
            seen.add(art.pmid)

            try:
                if already_exists(art.pmid):
                    print(f"  SKIP exists {art.pmid}")
                    continue
                ok = ingest_article(art, project)
                if ok:
                    total_new += 1
                    print(f"  OK {art.pmid}: {art.title[:90]}")
            except Exception as e:
                print(f"  FAIL {art.pmid}: {e}")

            time.sleep(0.15)

        time.sleep(0.35)

    print(f"\n✅ Ingestion terminée — {total_new} nouveaux articles")
    return total_new

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", choices=["geoai4ei", "gesica", "eva"], help="Single project to ingest")
    parser.add_argument("--query", help="Single query to ingest for the selected project")
    args = parser.parse_args()

    if args.project and args.query:
        run_project(args.project, [args.query])
        return 0

    if args.project:
        run_project(args.project, DEFAULT_QUERIES[args.project])
        return 0

    for project, queries in DEFAULT_QUERIES.items():
        run_project(project, queries)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())