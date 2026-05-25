#!/usr/bin/env python3
"""Ingest PubMed articles via Entrez API → POST /documents + /chunks."""
from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
import requests

API_BASE = "http://127.0.0.1:8000"
ENTREZ_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
EMAIL = "literev@example.com"

# ── Requêtes par projet ──────────────────────────────────────────────────────
QUERIES = {
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

MAX_PER_QUERY = 20  # articles max par requête


@dataclass
class PubMedArticle:
    pmid: str
    title: str
    abstract: str
    year: int | None
    url: str


def esearch(query: str, retmax: int = MAX_PER_QUERY) -> list[str]:
    r = requests.get(f"{ENTREZ_BASE}/esearch.fcgi", params={
        "db": "pubmed", "term": query, "retmax": retmax,
        "retmode": "json", "email": EMAIL,
    }, timeout=30)
    r.raise_for_status()
    return r.json()["esearchresult"]["idlist"]


def efetch(pmids: list[str]) -> list[PubMedArticle]:
    if not pmids:
        return []
    r = requests.post(f"{ENTREZ_BASE}/efetch.fcgi", data={
        "db": "pubmed", "id": ",".join(pmids),
        "rettype": "xml", "retmode": "xml", "email": EMAIL,
    }, timeout=60)
    r.raise_for_status()
    root = ET.fromstring(r.content)
    articles = []
    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID") or ""
        title = article.findtext(".//ArticleTitle") or ""
        abstract = " ".join(
            t.text or "" for t in article.findall(".//AbstractText")
        ).strip()
        year_text = (
            article.findtext(".//PubDate/Year")
            or article.findtext(".//PubDate/MedlineDate", "")[:4]
        )
        try:
            year = int(year_text)
        except (ValueError, TypeError):
            year = None
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        articles.append(PubMedArticle(pmid=pmid, title=title, abstract=abstract, year=year, url=url))
    return articles


def already_exists(pmid: str) -> bool:
    r = requests.get(f"{API_BASE}/search", json={
        "query_text": pmid, "mode": "boolean", "limit": 1,
        "filters": {},
    }, timeout=10)
    if r.ok:
        results = r.json().get("results", [])
        return any(str(res.get("external_id")) == pmid for res in results)
    return False


def ingest_article(article: PubMedArticle, project_context: str) -> bool:
    content = f"{article.title}\n\n{article.abstract}".strip()
    if not content or len(content) < 50:
        return False

    # POST /documents
    doc_r = requests.post(f"{API_BASE}/documents", json={
        "source": "pubmed",
        "title": article.title,
        "abstract": article.abstract,
        "year": article.year,
        "url": article.url,
        "external_id": article.pmid,
        "project_context": project_context,
        "source_type": "article",
        "disease_or_condition": None,
        "scenario_type": None,
        "geographic_scope": None,
        "evidence_category": None,
    }, timeout=30)
    doc_r.raise_for_status()
    doc_id = doc_r.json()["id"]

    # POST /chunks
    chunk_r = requests.post(f"{API_BASE}/chunks", json={
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
    }, timeout=60)
    chunk_r.raise_for_status()
    return True


def main():
    total_new = 0
    for project, queries in QUERIES.items():
        print(f"\n── {project.upper()} ──")
        seen_pmids: set[str] = set()
        for query in queries:
            print(f"  Query: {query[:60]}...")
            try:
                pmids = esearch(query)
                articles = efetch(pmids)
                time.sleep(0.35)  # Entrez rate limit
            except Exception as e:
                print(f"  ERROR fetch: {e}")
                continue

            for art in articles:
                if art.pmid in seen_pmids:
                    continue
                seen_pmids.add(art.pmid)
                try:
                    ok = ingest_article(art, project)
                    if ok:
                        total_new += 1
                        print(f"  ✅ [{art.year}] {art.title[:70]}")
                    else:
                        print(f"  ⏭  skipped (too short): {art.pmid}")
                except Exception as e:
                    print(f"  ❌ {art.pmid}: {e}")
                time.sleep(0.1)

    print(f"\n✅ Ingestion terminée — {total_new} nouveaux articles")


if __name__ == "__main__":
    main()
