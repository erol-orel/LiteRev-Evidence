from __future__ import annotations

import os
import re
import time
import xml.etree.ElementTree as ET

import requests

API_BASE = os.getenv("LITEREV_API_BASE", "http://127.0.0.1:8000")
USER_EMAIL = os.getenv("LITEREV_EMAIL", "your-email@example.org")

PUBMED_QUERY = os.getenv(
    "LITEREV_QUERY",
    '(epidemic intelligence OR pandemic surveillance OR outbreak detection OR hospital epidemiology)'
)
MAX_RESULTS = int(os.getenv("LITEREV_MAX_RESULTS", "20"))

session = requests.Session()
session.headers.update({"User-Agent": f"LiteRevLevel1Importer/2.0 ({USER_EMAIL})"})


def pubmed_search(query: str, retmax: int = 20) -> list[str]:
    r = session.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        params={
            "db": "pubmed",
            "term": query,
            "retmax": retmax,
            "retmode": "json",
            "sort": "relevance",
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["esearchresult"].get("idlist", [])


def pubmed_fetch_details(pmids: list[str]) -> list[dict]:
    if not pmids:
        return []

    r = session.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        params={"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"},
        timeout=60,
    )
    r.raise_for_status()

    root = ET.fromstring(r.text)
    records = []

    for article in root.findall(".//PubmedArticle"):
        medline = article.find("MedlineCitation")
        article_node = medline.find("Article") if medline is not None else None
        pubmed_data = article.find("PubmedData")

        pmid = medline.findtext("PMID") if medline is not None else None
        title = "".join(article_node.find("ArticleTitle").itertext()).strip() if article_node is not None and article_node.find("ArticleTitle") is not None else None

        abstract_parts = []
        abstract_sections = []
        if article_node is not None:
            abstract = article_node.find("Abstract")
            if abstract is not None:
                for t in abstract.findall("AbstractText"):
                    label = (t.attrib.get("Label") or "").strip()
                    txt = "".join(t.itertext()).strip()
                    if txt:
                        abstract_parts.append(txt)
                        abstract_sections.append({"label": label or None, "text": txt})

        abstract_text = "\n".join(abstract_parts).strip() or None

        year = None
        if article_node is not None:
            year = article_node.findtext("./Journal/JournalIssue/PubDate/Year")
            if year is None:
                medline_date = article_node.findtext("./Journal/JournalIssue/PubDate/MedlineDate")
                if medline_date:
                    year = medline_date[:4]
        try:
            year = int(year) if year else None
        except Exception:
            year = None

        doi = None
        if article_node is not None:
            for eloc in article_node.findall("ELocationID"):
                if eloc.attrib.get("EIdType") == "doi" and eloc.text:
                    doi = eloc.text.strip()
                    break

        pmcid = None
        if pubmed_data is not None:
            for aid in pubmed_data.findall(".//ArticleId"):
                if aid.attrib.get("IdType") == "pmc" and aid.text:
                    pmcid = aid.text.strip()
                    break

        records.append(
            {
                "pmid": pmid,
                "pmcid": pmcid,
                "doi": doi,
                "title": title or f"PubMed article {pmid}",
                "abstract": abstract_text,
                "abstract_sections": abstract_sections,
                "year": year,
                "source": "pubmed",
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
            }
        )

    return records


def openalex_enrich(record: dict) -> dict:
    doi = record.get("doi")
    title = record.get("title")

    try:
        if doi:
            r = session.get(f"https://api.openalex.org/works/https://doi.org/{doi}", timeout=60)
            if r.status_code == 200:
                data = r.json()
                record["openalex_id"] = data.get("id")
                record["openalex_cited_by_count"] = data.get("cited_by_count")
                return record

        if title:
            r = session.get(
                "https://api.openalex.org/works",
                params={"search": title, "per-page": 1},
                timeout=60,
            )
            if r.status_code == 200:
                results = r.json().get("results", [])
                if results:
                    data = results[0]
                    record["openalex_id"] = data.get("id")
                    record["openalex_cited_by_count"] = data.get("cited_by_count")
    except Exception:
        pass

    return record


def crossref_enrich(record: dict) -> dict:
    doi = record.get("doi")
    if not doi:
        return record

    try:
        r = session.get(f"https://api.crossref.org/works/{doi}", timeout=60)
        if r.status_code == 200:
            msg = r.json().get("message", {})
            record["crossref_type"] = msg.get("type")
            record["crossref_publisher"] = msg.get("publisher")
    except Exception:
        pass

    return record


def normalize_section_label(label: str | None) -> str:
    value = (label or "").strip().lower()
    if value in {"background", "context"}:
        return "abstract_background"
    if value in {"methods", "method"}:
        return "abstract_methods"
    if value in {"results", "findings"}:
        return "abstract_results"
    if value in {"conclusion", "conclusions", "interpretation"}:
        return "abstract_conclusion"
    return "abstract_section"


def infer_document_tags(text: str) -> dict:
    lower = text.lower()
    return {
        "has_surveillance": any(k in lower for k in ["surveillance", "monitoring", "epidemic intelligence"]),
        "has_ai": any(k in lower for k in ["artificial intelligence", "machine learning", "deep learning", " ai "]),
        "has_spatial": any(k in lower for k in ["spatial", "geographic", "geospatial", "spatiotemporal", "cluster"]),
        "has_hospital": any(k in lower for k in ["hospital", "nosocomial", "healthcare setting"]),
        "has_genomics": any(k in lower for k in ["genomic", "genomics", "sequencing", "molecular surveillance"]),
        "has_climate": any(k in lower for k in ["climate", "weather", "temperature", "pollution", "air quality"]),
    }


def estimate_chunk_weight(chunk_type: str, content: str) -> float:
    base = {
        "title": 1.35,
        "abstract_background": 0.95,
        "abstract_methods": 0.90,
        "abstract_results": 1.20,
        "abstract_conclusion": 1.25,
        "abstract_section": 1.00,
        "abstract_full": 1.05,
    }.get(chunk_type, 1.0)

    lower = content.lower()
    bonus = 0.0
    if any(k in lower for k in ["surveillance", "outbreak", "epidemic intelligence", "public health"]):
        bonus += 0.10
    if any(k in lower for k in ["systematic review", "scoping review", "meta-analysis"]):
        bonus += 0.08
    if re.search(r"\b\d{4}\b", content):
        bonus += 0.03

    return round(base + bonus, 3)


def build_chunks(record: dict) -> list[dict]:
    chunks = []
    title = (record.get("title") or "").strip()
    abstract = (record.get("abstract") or "").strip()
    joined_text = "\n".join(x for x in [title, abstract] if x)
    doc_tags = infer_document_tags(joined_text)

    if title:
        chunks.append(
            {
                "chunk_type": "title",
                "section_label": "title",
                "content": title,
                "char_start": 0,
                "char_end": len(title),
                "token_count": len(title.split()),
                "metadata_json": {
                    "source_level": "document",
                    "pmid": record.get("pmid"),
                    "doi": record.get("doi"),
                    "pmcid": record.get("pmcid"),
                    "openalex_id": record.get("openalex_id"),
                    "openalex_cited_by_count": record.get("openalex_cited_by_count"),
                    "crossref_type": record.get("crossref_type"),
                    "crossref_publisher": record.get("crossref_publisher"),
                    **doc_tags,
                },
            }
        )

    sections = record.get("abstract_sections") or []
    if sections:
        cursor = 0
        for sec in sections:
            text = (sec.get("text") or "").strip()
            if not text:
                continue
            chunk_type = normalize_section_label(sec.get("label"))
            start = cursor
            end = cursor + len(text)
            cursor = end + 1
            chunks.append(
                {
                    "chunk_type": chunk_type,
                    "section_label": sec.get("label") or chunk_type,
                    "content": text,
                    "char_start": start,
                    "char_end": end,
                    "token_count": len(text.split()),
                    "metadata_json": {
                        "source_level": "abstract_section",
                        "pmid": record.get("pmid"),
                        "doi": record.get("doi"),
                        "pmcid": record.get("pmcid"),
                        "openalex_id": record.get("openalex_id"),
                        "openalex_cited_by_count": record.get("openalex_cited_by_count"),
                        "crossref_type": record.get("crossref_type"),
                        "crossref_publisher": record.get("crossref_publisher"),
                        **doc_tags,
                    },
                }
            )
    elif abstract:
        chunks.append(
            {
                "chunk_type": "abstract_full",
                "section_label": "abstract",
                "content": abstract,
                "char_start": 0,
                "char_end": len(abstract),
                "token_count": len(abstract.split()),
                "metadata_json": {
                    "source_level": "abstract_full",
                    "pmid": record.get("pmid"),
                    "doi": record.get("doi"),
                    "pmcid": record.get("pmcid"),
                    "openalex_id": record.get("openalex_id"),
                    "openalex_cited_by_count": record.get("openalex_cited_by_count"),
                    "crossref_type": record.get("crossref_type"),
                    "crossref_publisher": record.get("crossref_publisher"),
                    **doc_tags,
                },
            }
        )

    for chunk in chunks:
        chunk["chunk_weight"] = estimate_chunk_weight(chunk["chunk_type"], chunk["content"])

    return chunks


def create_document(record: dict) -> int:
    payload = {
        "source": record["source"],
        "title": record["title"],
        "abstract": record.get("abstract"),
        "year": record.get("year"),
        "url": record.get("url"),
        "external_id": record.get("pmid") or record.get("doi"),
        "project_context": "level1",
        "source_type": "article",
        "disease_or_condition": None,
        "scenario_type": None,
        "geographic_scope": None,
        "evidence_category": None,
    }
    r = session.post(f"{API_BASE}/documents", json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["id"]


def create_chunk(document_id: int, chunk_index: int, chunk: dict):
    payload = {
        "document_id": document_id,
        "chunk_index": chunk_index,
        "content": chunk["content"],
        "chunk_type": chunk["chunk_type"],
        "section_label": chunk["section_label"],
        "char_start": chunk["char_start"],
        "char_end": chunk["char_end"],
        "token_count": chunk["token_count"],
        "chunk_weight": chunk["chunk_weight"],
        "metadata_json": chunk["metadata_json"],
    }
    r = session.post(f"{API_BASE}/chunks", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def main():
    pmids = pubmed_search(PUBMED_QUERY, MAX_RESULTS)
    records = pubmed_fetch_details(pmids)

    inserted_docs = 0
    inserted_chunks = 0

    for rec in records:
        rec = openalex_enrich(rec)
        rec = crossref_enrich(rec)

        doc_id = create_document(rec)
        inserted_docs += 1

        chunks = build_chunks(rec)
        for idx, chunk in enumerate(chunks):
            create_chunk(doc_id, idx, chunk)
            inserted_chunks += 1
            time.sleep(0.05)

        print(f"OK doc={doc_id} pmid={rec.get('pmid')} title={rec.get('title')} chunks={len(chunks)} schema=v2")

    print(f"DONE documents={inserted_docs} chunks={inserted_chunks} query={PUBMED_QUERY!r}")


if __name__ == "__main__":
    main()
