#!/usr/bin/env python3
"""
Backfill des métadonnées bibliographiques complètes pour literature_document.
Enrichit les colonnes : authors, doi, journal, keywords, language, study_design,
sample_size, country, citation_count, open_access, pmid, openalex_id, volume,
issue, pages, publication_type, mesh_terms, affiliations.

Sources utilisées (gratuites) :
  1. PubMed E-utilities (pour les docs avec pmid ou source='pubmed')
  2. OpenAlex API (pour les docs avec openalex_id ou doi)
  3. CrossRef API (pour les docs avec doi)

Usage : .venv/bin/python3 backfill_bibliographic_metadata.py [--limit N] [--source pubmed|openalex|crossref|all]
"""

import os
import sys
import time
import json
import argparse
import logging
from typing import Optional

import requests
from sqlalchemy import create_engine, text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

DB_URL = os.getenv("DB_URL") or os.getenv("DATABASE_URL")
if not DB_URL:
    raise RuntimeError("DB_URL (or DATABASE_URL) environment variable is required")
engine = create_engine(DB_URL, pool_pre_ping=True)

PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
OPENALEX_BASE = "https://api.openalex.org"
CROSSREF_BASE = "https://api.crossref.org/works"
EMAIL = "gesica@literev.ai"  # Requis par NCBI et CrossRef pour la politesse


# ─── Helpers ──────────────────────────────────────────────────────────────────

def safe_get(url: str, params: dict = None, timeout: int = 15) -> Optional[dict]:
    """GET avec retry et gestion d'erreurs."""
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                log.warning(f"Rate limit sur {url}, attente 10s...")
                time.sleep(10)
            else:
                log.warning(f"HTTP {r.status_code} sur {url}")
                return None
        except Exception as e:
            log.warning(f"Erreur réseau ({attempt+1}/3) : {e}")
            time.sleep(2)
    return None


# ─── PubMed ───────────────────────────────────────────────────────────────────

import xml.etree.ElementTree as ET


def _xml_text(el, path: str, default: str = "") -> str:
    """Extrait le texte d'un élément XML via un chemin XPath simple."""
    node = el.find(path)
    return (node.text or "").strip() if node is not None else default


def fetch_pubmed_metadata(pmid: str) -> dict:
    """Récupère les métadonnées complètes d'un article PubMed via efetch XML.
    
    Note : efetch avec retmode=json ne fonctionne pas pour le type 'abstract'.
    On utilise retmode=xml et on parse le XML avec ElementTree.
    """
    for attempt in range(3):
        try:
            r = requests.get(
                f"{PUBMED_BASE}/efetch.fcgi",
                params={
                    "db": "pubmed", "id": pmid,
                    "retmode": "xml", "rettype": "abstract",
                    "tool": "literev", "email": EMAIL
                },
                timeout=20
            )
            if r.status_code == 200 and r.text.strip().startswith("<"):
                break
            elif r.status_code == 429:
                log.warning(f"Rate limit PubMed, attente 10s...")
                time.sleep(10)
            else:
                return {}
        except Exception as e:
            log.warning(f"Erreur réseau PubMed ({attempt+1}/3) : {e}")
            time.sleep(2)
    else:
        return {}

    try:
        root = ET.fromstring(r.text)
        art_el = root.find(".//Article")
        medline_el = root.find(".//MedlineCitation")
        if art_el is None or medline_el is None:
            return {}

        # Auteurs
        authors = []
        for a in art_el.findall(".//Author"):
            last = _xml_text(a, "LastName")
            fore = _xml_text(a, "ForeName")
            if last:
                authors.append(f"{last} {fore}".strip())

        # Journal
        journal = _xml_text(art_el, ".//Journal/Title") or _xml_text(art_el, ".//Journal/ISOAbbreviation")

        # Volume, issue, pages
        volume = _xml_text(art_el, ".//JournalIssue/Volume")
        issue = _xml_text(art_el, ".//JournalIssue/Issue")
        pages = _xml_text(art_el, ".//Pagination/MedlinePgn")

        # DOI
        doi = ""
        for loc in art_el.findall(".//ELocationID"):
            if loc.get("EIdType") == "doi":
                doi = (loc.text or "").strip()
                break

        # Keywords
        keywords = [kw.text.strip() for kw in medline_el.findall(".//Keyword") if kw.text]

        # MeSH terms
        mesh_terms = [dn.text.strip() for dn in medline_el.findall(".//MeshHeading/DescriptorName") if dn.text]

        # Language
        lang_el = art_el.find(".//Language")
        language = (lang_el.text or "en").strip()[:10] if lang_el is not None else "en"

        # Publication type
        pub_types = [pt.text.strip() for pt in art_el.findall(".//PublicationType") if pt.text]

        # Affiliations
        affiliations = list({
            aff.text.strip()
            for aff in art_el.findall(".//AffiliationInfo/Affiliation")
            if aff.text
        })

        return {
            "authors": ", ".join(authors) if authors else None,
            "doi": doi or None,
            "journal": journal or None,
            "keywords": ", ".join(keywords) if keywords else None,
            "language": language,
            "volume": volume or None,
            "issue": issue or None,
            "pages": pages or None,
            "mesh_terms": json.dumps(mesh_terms) if mesh_terms else None,
            "affiliations": " | ".join(affiliations)[:1000] if affiliations else None,
            "publication_type": ", ".join(pub_types[:3]) if pub_types else None,
        }
    except Exception as e:
        log.debug(f"Erreur parsing PubMed XML {pmid}: {e}")
        return {}


# ─── OpenAlex ─────────────────────────────────────────────────────────────────

def fetch_openalex_metadata(identifier: str, id_type: str = "doi") -> dict:
    """Récupère les métadonnées depuis OpenAlex par DOI ou OpenAlex ID."""
    if id_type == "doi":
        url = f"{OPENALEX_BASE}/works/https://doi.org/{identifier}"
    else:
        url = f"{OPENALEX_BASE}/works/{identifier}"
    
    data = safe_get(url, {"mailto": EMAIL})
    if not data or "id" not in data:
        return {}
    
    try:
        # Auteurs
        authors = []
        for a in data.get("authorships", []):
            name = a.get("author", {}).get("display_name", "")
            if name:
                authors.append(name)
        
        # Journal
        venue = data.get("primary_location", {}) or {}
        source = venue.get("source", {}) or {}
        journal = source.get("display_name", "")
        issn_list = source.get("issn_l", "")
        
        # Keywords / concepts
        keywords = [c.get("display_name", "") for c in data.get("concepts", [])[:10] if c.get("score", 0) > 0.3]
        
        # Citation count
        citation_count = data.get("cited_by_count", 0)
        
        # Open access
        oa_info = data.get("open_access", {})
        open_access = oa_info.get("is_oa", None)
        
        # Country (from first author's institution)
        country = None
        for a in data.get("authorships", []):
            for inst in a.get("institutions", []):
                country = inst.get("country_code", None)
                if country:
                    break
            if country:
                break
        
        # Language
        language = data.get("language", "en")
        
        # Publication type
        pub_type = data.get("type", "")
        
        # DOI
        doi = data.get("doi", "").replace("https://doi.org/", "") if data.get("doi") else None
        
        # OpenAlex ID
        openalex_id = data.get("id", "").replace("https://openalex.org/", "") if data.get("id") else None
        
        return {
            "authors": ", ".join(authors) if authors else None,
            "doi": doi,
            "journal": journal or None,
            "keywords": ", ".join(keywords) if keywords else None,
            "language": language[:10] if language else "en",
            "citation_count": citation_count,
            "open_access": open_access,
            "country": country,
            "issn": issn_list or None,
            "publication_type": pub_type or None,
            "openalex_id": openalex_id,
        }
    except Exception as e:
        log.debug(f"Erreur parsing OpenAlex {identifier}: {e}")
        return {}


# ─── CrossRef ─────────────────────────────────────────────────────────────────

def fetch_crossref_metadata(doi: str) -> dict:
    """Récupère les métadonnées depuis CrossRef par DOI."""
    data = safe_get(f"{CROSSREF_BASE}/{doi}", {"mailto": EMAIL})
    if not data or data.get("status") != "ok":
        return {}
    
    try:
        msg = data["message"]
        
        # Auteurs
        authors = []
        for a in msg.get("author", []):
            given = a.get("given", "")
            family = a.get("family", "")
            if family:
                authors.append(f"{family} {given}".strip())
        
        # Journal
        journal = ""
        container = msg.get("container-title", [])
        if container:
            journal = container[0]
        
        # Volume, issue, pages
        volume = msg.get("volume", "")
        issue = msg.get("issue", "")
        pages = msg.get("page", "")
        
        # ISSN
        issn_list = msg.get("ISSN", [])
        issn = issn_list[0] if issn_list else None
        
        # Citation count
        citation_count = msg.get("is-referenced-by-count", 0)
        
        # Open access
        license_list = msg.get("license", [])
        open_access = len(license_list) > 0  # Approximation
        
        # Publication type
        pub_type = msg.get("type", "")
        
        # Language
        language = msg.get("language", "en")
        
        # Funding
        funders = []
        for f in msg.get("funder", []):
            name = f.get("name", "")
            if name:
                funders.append(name)
        
        return {
            "authors": ", ".join(authors) if authors else None,
            "journal": journal or None,
            "volume": volume or None,
            "issue": issue or None,
            "pages": pages or None,
            "issn": issn,
            "citation_count": citation_count,
            "open_access": open_access,
            "language": language[:10] if language else "en",
            "publication_type": pub_type or None,
            "funding": ", ".join(funders[:5]) if funders else None,
        }
    except Exception as e:
        log.debug(f"Erreur parsing CrossRef {doi}: {e}")
        return {}


# ─── Merge et mise à jour ──────────────────────────────────────────────────────

def merge_metadata(*sources: dict) -> dict:
    """Fusionne les métadonnées de plusieurs sources, priorité à la première non-nulle."""
    merged = {}
    for source in sources:
        for key, value in source.items():
            if key not in merged or merged[key] is None:
                if value is not None and value != "" and value != []:
                    merged[key] = value
    return merged


def update_document(conn, doc_id: int, metadata: dict):
    """Met à jour un document avec les métadonnées enrichies."""
    if not metadata:
        return
    
    # Filtrer les clés valides (colonnes qui existent)
    valid_cols = {
        "authors", "doi", "journal", "keywords", "language", "study_design",
        "sample_size", "country", "citation_count", "open_access", "pmid",
        "openalex_id", "volume", "issue", "pages", "issn", "publication_type",
        "structured_abstract", "mesh_terms", "affiliations", "funding"
    }
    
    updates = {k: v for k, v in metadata.items() if k in valid_cols and v is not None}
    if not updates:
        return
    
    set_clauses = ", ".join(f"{col} = :{col}" for col in updates)
    updates["doc_id"] = doc_id
    
    conn.execute(text(f"UPDATE literature_document SET {set_clauses} WHERE id = :doc_id"), updates)


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Backfill métadonnées bibliographiques")
    parser.add_argument("--limit", type=int, default=0, help="Nombre max de docs à traiter (0 = tous)")
    parser.add_argument("--source", choices=["pubmed", "openalex", "crossref", "all"], default="all")
    parser.add_argument("--project", default="gesica", help="project_context à traiter")
    parser.add_argument("--only-missing", action="store_true", help="Traiter uniquement les docs sans authors")
    parser.add_argument("--workers", type=int, default=1,
                        help="Nombre de workers parallèles (défaut: 1, max recommandé: 3 pour respecter les rate limits)")
    args = parser.parse_args()
    
    with engine.connect() as conn:
        # Compter les documents à traiter
        where_clause = "WHERE project_context = :project"
        if args.only_missing:
            where_clause += " AND (authors IS NULL OR doi IS NULL)"
        
        count_sql = f"SELECT COUNT(*) FROM literature_document {where_clause}"
        total = conn.execute(text(count_sql), {"project": args.project}).scalar()
        log.info(f"Documents à traiter : {total} (project={args.project})")
        
        # Récupérer les documents
        limit_clause = f"LIMIT {args.limit}" if args.limit > 0 else ""
        docs_sql = f"""
            SELECT id, external_id, source, title, url
            FROM literature_document
            {where_clause}
            ORDER BY id
            {limit_clause}
        """
        docs = conn.execute(text(docs_sql), {"project": args.project}).mappings().all()
        
        updated = 0
        skipped = 0
        errors = 0
        
        for i, doc in enumerate(docs):
            doc_id = doc["id"]
            external_id = doc["external_id"] or ""
            source = (doc["source"] or "").lower()
            url = doc["url"] or ""
            
            log.info(f"[{i+1}/{len(docs)}] Doc {doc_id} — source={source}, external_id={external_id[:30]}")
            
            metadata_parts = []
            
            # Détecter le PMID
            pmid = None
            if source in ("pubmed", "pmc") or "pubmed" in url or "ncbi.nlm.nih.gov" in url:
                # Extraire le PMID depuis external_id ou url
                if external_id.isdigit():
                    pmid = external_id
                elif "/pubmed/" in url:
                    pmid = url.split("/pubmed/")[-1].strip("/").split("?")[0]
                elif "/articles/PMC" in url:
                    # PMC → chercher via eutils
                    pmcid = url.split("/articles/")[-1].strip("/")
                    search = safe_get(f"{PUBMED_BASE}/esearch.fcgi", {
                        "db": "pubmed", "term": f"{pmcid}[pmcid]",
                        "retmode": "json", "tool": "literev", "email": EMAIL
                    })
                    if search:
                        ids = search.get("esearchresult", {}).get("idlist", [])
                        if ids:
                            pmid = ids[0]
            
            # Détecter le DOI
            doi = None
            if "doi.org/" in url:
                doi = url.split("doi.org/")[-1].strip()
            elif external_id.startswith("10."):
                doi = external_id
            
            # Détecter l'OpenAlex ID
            openalex_id = None
            if "openalex.org/" in url:
                openalex_id = url.split("openalex.org/")[-1].strip("/")
            elif external_id.startswith("W") and external_id[1:].isdigit():
                openalex_id = external_id
            
            # Enrichissement PubMed
            if pmid and args.source in ("pubmed", "all"):
                pm_meta = fetch_pubmed_metadata(pmid)
                if pm_meta:
                    pm_meta["pmid"] = pmid
                    metadata_parts.append(pm_meta)
                time.sleep(0.35)  # Respecter la limite NCBI (3 req/s)
            
            # Enrichissement OpenAlex
            if args.source in ("openalex", "all"):
                oa_meta = {}
                if openalex_id:
                    oa_meta = fetch_openalex_metadata(openalex_id, "id")
                elif doi:
                    oa_meta = fetch_openalex_metadata(doi, "doi")
                if oa_meta:
                    metadata_parts.append(oa_meta)
                time.sleep(0.1)  # OpenAlex : 10 req/s
            
            # Enrichissement CrossRef
            if doi and args.source in ("crossref", "all"):
                cr_meta = fetch_crossref_metadata(doi)
                if cr_meta:
                    metadata_parts.append(cr_meta)
                time.sleep(0.1)
            
            # Fusionner et mettre à jour
            if metadata_parts:
                merged = merge_metadata(*metadata_parts)
                update_document(conn, doc_id, merged)
                conn.commit()
                updated += 1
                log.info(f"  → Mis à jour : {list(merged.keys())}")
            else:
                skipped += 1
                log.debug(f"  → Aucune métadonnée trouvée")
            
            # Commit toutes les 50 lignes
            if (i + 1) % 50 == 0:
                conn.commit()
                log.info(f"  Progression : {i+1}/{len(docs)} — mis à jour={updated}, ignorés={skipped}, erreurs={errors}")
        
        conn.commit()
    
    log.info(f"\n{'='*50}")
    log.info(f"TERMINÉ : {updated} mis à jour, {skipped} ignorés, {errors} erreurs")
    log.info(f"{'='*50}")


if __name__ == "__main__":
    main()
