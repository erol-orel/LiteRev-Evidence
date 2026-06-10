#!/usr/bin/env python3
"""
fetch_fulltext_bulk.py — Récupération massive de full-texts pour LiteRev-Evidence
==================================================================================
Stratégie multi-sources par ordre de priorité :
  1. PubMed Central (NCBI E-utilities) — via PMCID ou résolution PMID→PMCID
  2. Europe PMC REST API             — fallback PMC + articles non-NCBI
  3. Unpaywall API                   — DOI → PDF open-access (email requis)
  4. bioRxiv / medRxiv               — preprints (DOI 10.1101/...)
  5. Semantic Scholar Open Access    — fallback PDF open-access
  6. OpenAlex                        — open_access.oa_url si disponible

Pour chaque document sans full-text :
  - Résout l'identifiant (external_id, doi, pmid, openalex_id)
  - Essaie les sources dans l'ordre
  - Découpe le texte en chunks et les insère via l'API locale
  - Met à jour open_access=true si full-text récupéré

Usage :
  python3 fetch_fulltext_bulk.py [--dry-run] [--limit N] [--project gesica]
                                  [--source pmc|europepmc|unpaywall|biorxiv|semanticscholar|all]
                                  [--workers N] [--email your@email.com]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import requests
from sqlalchemy import create_engine, text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fetch-fulltext-bulk")

# ─── Configuration ─────────────────────────────────────────────────────────────

def _load_env_file(path: str) -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value

for _ep in [".env", str(Path(__file__).parent / ".env"), "/opt/literev-api/.env",
            "/opt/literev-api/secrets.env", "/etc/literev/secrets"]:
    _load_env_file(_ep)

DB_URL = os.getenv("DB_URL") or os.getenv("DATABASE_URL")
if not DB_URL:
    raise RuntimeError("DB_URL (or DATABASE_URL) environment variable is required")
API_BASE = os.getenv("LITEREV_API_BASE", "http://localhost:8000")
WRITE_API_KEY = os.getenv("WRITE_API_KEY", "")
UNPAYWALL_EMAIL = os.getenv("UNPAYWALL_EMAIL", "")  # requis pour Unpaywall

NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")  # optionnel, augmente le rate limit
NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
EUROPEPMC_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"
UNPAYWALL_BASE = "https://api.unpaywall.org/v2"
SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"
OPENALEX_BASE = "https://api.openalex.org"

# Taille des chunks (caractères)
CHUNK_SIZE = 4_000
CHUNK_OVERLAP = 400

# Délai entre requêtes par source (secondes)
RATE_LIMITS = {
    "pmc": 0.35,          # NCBI : 3 req/s sans clé, 10/s avec clé
    "europepmc": 0.5,
    "unpaywall": 1.0,
    "biorxiv": 0.5,
    "semanticscholar": 1.0,
    "openalex": 0.2,
}

# ─── Utilitaires ───────────────────────────────────────────────────────────────

def _get(url: str, params: dict = None, timeout: int = 20, headers: dict = None) -> Optional[requests.Response]:
    """GET avec retry 3x sur erreur réseau."""
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers=headers or {})
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 10))
                logger.warning(f"Rate limit — attente {wait}s")
                time.sleep(wait)
                continue
            return r
        except requests.RequestException as e:
            if attempt == 2:
                logger.debug(f"GET {url} échoué : {e}")
            time.sleep(1)
    return None


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Découpe un texte en chunks avec overlap."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # Couper sur un espace si possible
        if end < len(text):
            cut = text.rfind(" ", start, end)
            if cut > start:
                end = cut
        chunks.append(text[start:end].strip())
        start = end - overlap if end - overlap > start else end
    return [c for c in chunks if len(c) > 50]


def insert_fulltext_chunks(engine, doc_id: int, chunks: list[str], source_label: str, dry_run: bool) -> int:
    """Insère les chunks full-text dans document_chunk via SQLAlchemy direct.
    
    Utilise le pattern stable CAST(:metadata_json AS jsonb) avec json.dumps,
    identique à l'endpoint /chunks de main.py.
    chunk_type = 'fulltext_section' pour être visible dans les stats et la recherche.
    """
    if dry_run:
        return len(chunks)
    inserted = 0
    with engine.begin() as conn:
        # Supprimer les anciens chunks full-text pour ce document (les deux conventions)
        conn.execute(text(
            "DELETE FROM document_chunk WHERE document_id = :doc_id "
            "AND chunk_type IN ('fulltext_section', 'full_text')"
        ), {"doc_id": doc_id})
        for i, chunk in enumerate(chunks):
            meta = json.dumps({"source": source_label, "chunk_index": i})
            conn.execute(text("""
                INSERT INTO document_chunk
                    (document_id, content, chunk_index, chunk_type, chunk_weight, metadata_json)
                VALUES
                    (:doc_id, :content, :idx, 'fulltext_section', 1.0,
                     CAST(:metadata_json AS jsonb))
            """), {"doc_id": doc_id, "content": chunk, "idx": i, "metadata_json": meta})
            inserted += 1
        # Marquer open_access=true
        conn.execute(text(
            "UPDATE literature_document SET open_access = true WHERE id = :doc_id"
        ), {"doc_id": doc_id})
    return inserted


# ─── Source 1 : PubMed Central (NCBI) ─────────────────────────────────────────

def _clean_pmcid(raw: str) -> Optional[str]:
    """Normalise un PMCID brut vers le format 'PMCXXXXXXX'.
    
    Gère les formats :
      - 'PMC1234567'          → 'PMC1234567'
      - 'pmc-id: 1234567;'   → 'PMC1234567'
      - 'PMCpmc-id: 1234567;'→ 'PMC1234567'
      - '1234567'            → 'PMC1234567'
    """
    if not raw:
        return None
    raw = raw.strip()
    # Extraire les chiffres depuis n'importe quel format
    digits = re.search(r"(\d{5,10})", raw)
    if digits:
        return f"PMC{digits.group(1)}"
    return None


def resolve_pmcid(external_id: str, pmid: str = None, doi: str = None) -> Optional[str]:
    """Résout un PMCID depuis external_id, pmid ou doi."""
    # Cas 1 : external_id contient un PMCID (format propre ou brut)
    if external_id and "PMC" in external_id.upper():
        cleaned = _clean_pmcid(external_id)
        if cleaned:
            return cleaned
    # Cas 1b : external_id est déjà un PMCID pur
    if external_id and re.match(r"^PMC\d+$", external_id.upper()):
        return external_id.upper()

    # Cas 2 : external_id est un PMID numérique
    candidate_pmid = None
    if external_id and external_id.isdigit():
        candidate_pmid = external_id
    elif pmid:
        candidate_pmid = str(pmid).replace("PMID:", "").strip()
    elif external_id and external_id.upper().startswith("PMID:"):
        candidate_pmid = external_id[5:].strip()

    if candidate_pmid:
        params = {"db": "pubmed", "id": candidate_pmid, "retmode": "json"}
        if NCBI_API_KEY:
            params["api_key"] = NCBI_API_KEY
        r = _get(f"{NCBI_BASE}/esummary.fcgi", params=params)
        if r and r.status_code == 200:
            try:
                data = r.json()
                article_ids = data.get("result", {}).get(candidate_pmid, {}).get("articleids", [])
                for aid in article_ids:
                    if aid.get("idtype") == "pmcid":
                        pmcid = aid.get("value", "").replace("PMC", "")
                        if pmcid:
                            return f"PMC{pmcid}"
            except Exception:
                pass

    # Cas 3 : DOI → PMCID via NCBI ID converter
    if doi:
        r = _get(
            "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/",
            params={"ids": doi, "format": "json", "tool": "literev", "email": UNPAYWALL_EMAIL or "literev@example.com"},
        )
        if r and r.status_code == 200:
            try:
                records = r.json().get("records", [])
                if records and records[0].get("pmcid"):
                    return records[0]["pmcid"]
            except Exception:
                pass

    return None


def fetch_pmc_fulltext(pmcid: str) -> Optional[str]:
    """Récupère le full-text XML depuis NCBI PMC et extrait le texte."""
    # Essai 1 : Europe PMC XML (plus fiable)
    r = _get(f"{EUROPEPMC_BASE}/{pmcid}/fullTextXML", timeout=30)
    if r and r.status_code == 200 and r.text.strip().startswith("<"):
        text_content = _parse_pmc_xml(r.text)
        if text_content and len(text_content) > 500:
            return text_content

    # Essai 2 : NCBI efetch
    pmcid_num = pmcid.replace("PMC", "")
    params = {"db": "pmc", "id": pmcid_num, "rettype": "full", "retmode": "xml"}
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    r = _get(f"{NCBI_BASE}/efetch.fcgi", params=params, timeout=30)
    if r and r.status_code == 200 and r.text.strip().startswith("<"):
        text_content = _parse_pmc_xml(r.text)
        if text_content and len(text_content) > 500:
            return text_content

    return None


def _parse_pmc_xml(xml_str: str) -> Optional[str]:
    """Extrait le texte des balises body/abstract/sec d'un XML PMC."""
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError:
        # Essai avec nettoyage
        xml_str = re.sub(r"&(?!amp;|lt;|gt;|apos;|quot;)", "&amp;", xml_str)
        try:
            root = ET.fromstring(xml_str)
        except ET.ParseError:
            return None

    # Balises à ignorer complètement (références, remerciements, etc.)
    skip_tags = {"ref-list", "ack", "fn-group", "glossary", "app-group",
                 "notes", "bio", "author-notes"}

    parts = []

    def _walk(node):
        tag = node.tag.split("}")[-1] if "}" in node.tag else node.tag
        if tag in skip_tags:
            return
        # Collecter le texte direct du noeud
        if node.text and node.text.strip():
            parts.append(node.text.strip())
        # Traiter les enfants récursivement
        for child in node:
            _walk(child)
        # Collecter le texte tail (après la balise fermante)
        if node.tail and node.tail.strip():
            parts.append(node.tail.strip())

    _walk(root)
    text = " ".join(parts)
    # Nettoyage
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) > 50 else None


# ─── Source 2 : Europe PMC ─────────────────────────────────────────────────────

def fetch_europepmc_fulltext(pmcid: str = None, doi: str = None, ext_id: str = None) -> Optional[str]:
    """Récupère le full-text via Europe PMC."""
    # Résolution de l'ID Europe PMC
    epmc_id = None
    if pmcid:
        epmc_id = pmcid
    elif doi:
        r = _get(f"{EUROPEPMC_BASE}/search",
                 params={"query": f"DOI:{doi}", "format": "json", "resultType": "core", "pageSize": 1})
        if r and r.status_code == 200:
            try:
                results = r.json().get("resultList", {}).get("result", [])
                if results:
                    epmc_id = results[0].get("pmcid") or results[0].get("id")
            except Exception:
                pass
    elif ext_id:
        r = _get(f"{EUROPEPMC_BASE}/search",
                 params={"query": ext_id, "format": "json", "resultType": "core", "pageSize": 1})
        if r and r.status_code == 200:
            try:
                results = r.json().get("resultList", {}).get("result", [])
                if results:
                    epmc_id = results[0].get("pmcid") or results[0].get("id")
            except Exception:
                pass

    if not epmc_id:
        return None

    # Récupérer le XML full-text
    r = _get(f"{EUROPEPMC_BASE}/{epmc_id}/fullTextXML", timeout=30)
    if r and r.status_code == 200 and r.text.strip().startswith("<"):
        return _parse_pmc_xml(r.text)

    return None


# ─── Source 3 : Unpaywall ──────────────────────────────────────────────────────

def fetch_unpaywall_fulltext(doi: str, email: str) -> Optional[str]:
    """Récupère le PDF open-access via Unpaywall et en extrait le texte."""
    if not email or not doi:
        return None

    r = _get(f"{UNPAYWALL_BASE}/{doi}", params={"email": email})
    if not r or r.status_code != 200:
        return None

    try:
        data = r.json()
    except Exception:
        return None

    # Chercher une URL PDF open-access
    pdf_url = None
    best_oa = data.get("best_oa_location") or {}
    if best_oa.get("url_for_pdf"):
        pdf_url = best_oa["url_for_pdf"]
    elif best_oa.get("url"):
        pdf_url = best_oa["url"]

    if not pdf_url:
        for loc in data.get("oa_locations", []):
            if loc.get("url_for_pdf"):
                pdf_url = loc["url_for_pdf"]
                break

    if not pdf_url:
        return None

    # Télécharger et extraire le texte du PDF
    return _extract_pdf_text(pdf_url)


def _extract_pdf_text(pdf_url: str) -> Optional[str]:
    """Télécharge un PDF et en extrait le texte via pdftotext."""
    import tempfile
    import subprocess

    try:
        r = requests.get(pdf_url, timeout=30, stream=True,
                         headers={"User-Agent": "LiteRev-Evidence/1.0 (research tool)"})
        if r.status_code != 200:
            return None
        content_type = r.headers.get("content-type", "")
        if "pdf" not in content_type.lower() and not pdf_url.lower().endswith(".pdf"):
            # Peut être du HTML — essayer d'extraire le texte directement
            text = r.text
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text if len(text) > 500 else None

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
            tmp_path = f.name

        # Extraire avec pdftotext (poppler-utils, pré-installé)
        result = subprocess.run(
            ["pdftotext", "-layout", tmp_path, "-"],
            capture_output=True, text=True, timeout=30
        )
        os.unlink(tmp_path)

        if result.returncode == 0 and result.stdout.strip():
            text = re.sub(r"\s+", " ", result.stdout).strip()
            return text if len(text) > 500 else None

    except Exception as e:
        logger.debug(f"Extraction PDF échouée ({pdf_url}) : {e}")

    return None


# ─── Source 4 : bioRxiv / medRxiv ─────────────────────────────────────────────

def fetch_biorxiv_fulltext(doi: str) -> Optional[str]:
    """Récupère le full-text d'un preprint bioRxiv/medRxiv via leur API."""
    if not doi or not doi.startswith("10.1101/"):
        return None

    # API bioRxiv/medRxiv
    for server in ["biorxiv", "medrxiv"]:
        r = _get(f"https://api.biorxiv.org/details/{server}/{doi}/na/json")
        if r and r.status_code == 200:
            try:
                data = r.json()
                collection = data.get("collection", [])
                if collection:
                    item = collection[-1]  # version la plus récente
                    abstract = item.get("abstract", "")
                    # Essayer de récupérer le PDF
                    pdf_url = f"https://www.{server}.org/content/{doi}.full.pdf"
                    text = _extract_pdf_text(pdf_url)
                    if text and len(text) > 500:
                        return text
                    # Fallback : abstract enrichi
                    if abstract and len(abstract) > 200:
                        return abstract
            except Exception:
                pass

    return None


# ─── Source 5 : Semantic Scholar ──────────────────────────────────────────────

def fetch_semantic_scholar_fulltext(doi: str = None, external_id: str = None) -> Optional[str]:
    """Récupère le PDF open-access via Semantic Scholar."""
    paper_id = None
    if doi:
        paper_id = f"DOI:{doi}"
    elif external_id and external_id.isdigit():
        paper_id = f"PMID:{external_id}"
    elif external_id and external_id.upper().startswith("PMC"):
        paper_id = f"PMCID:{external_id}"

    if not paper_id:
        return None

    r = _get(
        f"{SEMANTIC_SCHOLAR_BASE}/paper/{paper_id}",
        params={"fields": "openAccessPdf,abstract"},
        headers={"User-Agent": "LiteRev-Evidence/1.0"},
    )
    if not r or r.status_code != 200:
        return None

    try:
        data = r.json()
        oa_pdf = data.get("openAccessPdf")
        if oa_pdf and oa_pdf.get("url"):
            text = _extract_pdf_text(oa_pdf["url"])
            if text and len(text) > 500:
                return text
        # Fallback : abstract
        abstract = data.get("abstract", "")
        if abstract and len(abstract) > 200:
            return abstract
    except Exception:
        pass

    return None


# ─── Source 6 : OpenAlex ──────────────────────────────────────────────────────

def fetch_openalex_fulltext(openalex_id: str = None, doi: str = None) -> Optional[str]:
    """Récupère le PDF open-access via OpenAlex."""
    if openalex_id:
        url = f"{OPENALEX_BASE}/works/{openalex_id}"
    elif doi:
        url = f"{OPENALEX_BASE}/works/doi:{doi}"
    else:
        return None

    r = _get(url, params={"select": "open_access,abstract_inverted_index"})
    if not r or r.status_code != 200:
        return None

    try:
        data = r.json()
        oa = data.get("open_access", {})
        oa_url = oa.get("oa_url")
        if oa_url:
            text = _extract_pdf_text(oa_url)
            if text and len(text) > 500:
                return text
    except Exception:
        pass

    return None


# ─── Pipeline principal ────────────────────────────────────────────────────────

def process_document(row: dict, args: argparse.Namespace, engine) -> dict:
    """Traite un document : essaie toutes les sources et insère le full-text."""
    doc_id = row["id"]
    external_id = row.get("external_id") or ""
    doi = row.get("doi") or ""
    pmid = row.get("pmid") or ""
    openalex_id = row.get("openalex_id") or ""
    source = row.get("source") or ""
    title = row.get("title") or ""

    result = {
        "doc_id": doc_id,
        "title": title[:80],
        "status": "skipped",
        "source_used": None,
        "chunks": 0,
    }

    # Normaliser le DOI
    if doi and not doi.startswith("10."):
        doi = ""

    # Déterminer les sources à essayer selon les identifiants disponibles
    sources_to_try = args.source if args.source != "all" else None

    fulltext = None
    source_used = None

    # ── Source 1 : PMC ──
    if not fulltext and (sources_to_try is None or "pmc" in sources_to_try):
        pmcid = resolve_pmcid(external_id, pmid, doi)
        if pmcid:
            time.sleep(RATE_LIMITS["pmc"])
            fulltext = fetch_pmc_fulltext(pmcid)
            if fulltext:
                source_used = f"pmc:{pmcid}"

    # ── Source 2 : Europe PMC ──
    if not fulltext and (sources_to_try is None or "europepmc" in sources_to_try):
        pmcid = resolve_pmcid(external_id, pmid, doi) if not source_used else None
        time.sleep(RATE_LIMITS["europepmc"])
        fulltext = fetch_europepmc_fulltext(pmcid, doi or None, external_id or None)
        if fulltext:
            source_used = "europepmc"

    # ── Source 3 : Unpaywall ──
    if not fulltext and doi and UNPAYWALL_EMAIL and (sources_to_try is None or "unpaywall" in sources_to_try):
        time.sleep(RATE_LIMITS["unpaywall"])
        fulltext = fetch_unpaywall_fulltext(doi, UNPAYWALL_EMAIL)
        if fulltext:
            source_used = "unpaywall"

    # ── Source 4 : bioRxiv/medRxiv ──
    if not fulltext and doi and doi.startswith("10.1101/") and (sources_to_try is None or "biorxiv" in sources_to_try):
        time.sleep(RATE_LIMITS["biorxiv"])
        fulltext = fetch_biorxiv_fulltext(doi)
        if fulltext:
            source_used = "biorxiv"

    # ── Source 5 : Semantic Scholar ──
    if not fulltext and (sources_to_try is None or "semanticscholar" in sources_to_try):
        time.sleep(RATE_LIMITS["semanticscholar"])
        fulltext = fetch_semantic_scholar_fulltext(doi or None, external_id or None)
        if fulltext:
            source_used = "semanticscholar"

    # ── Source 6 : OpenAlex ──
    if not fulltext and (openalex_id or doi) and (sources_to_try is None or "openalex" in sources_to_try):
        time.sleep(RATE_LIMITS["openalex"])
        fulltext = fetch_openalex_fulltext(openalex_id or None, doi or None)
        if fulltext:
            source_used = "openalex"

    if not fulltext:
        result["status"] = "not_found"
        return result

    # Découper en chunks et insérer
    chunks = chunk_text(fulltext)
    if not chunks:
        result["status"] = "empty"
        return result

    n_inserted = insert_fulltext_chunks(engine, doc_id, chunks, source_used, args.dry_run)
    result["status"] = "ok"
    result["source_used"] = source_used
    result["chunks"] = n_inserted
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Récupération massive de full-texts — LiteRev-Evidence"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulation : ne rien écrire en base")
    parser.add_argument("--limit", type=int, default=0,
                        help="Nombre max de documents à traiter (0 = tous)")
    parser.add_argument("--project", type=str, default="gesica",
                        help="Filtre sur project_context")
    parser.add_argument("--source", type=str, default="all",
                        choices=["all", "pmc", "europepmc", "unpaywall", "biorxiv", "semanticscholar", "openalex"],
                        help="Source à utiliser (défaut : all)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Nombre de workers parallèles (défaut : 4)")
    parser.add_argument("--email", type=str, default="",
                        help="Email pour Unpaywall (obligatoire pour cette source)")
    parser.add_argument("--reprocess", action="store_true",
                        help="Retraiter aussi les documents qui ont déjà un full-text")
    args = parser.parse_args()

    # Email Unpaywall
    if args.email:
        os.environ["UNPAYWALL_EMAIL"] = args.email
    global UNPAYWALL_EMAIL
    UNPAYWALL_EMAIL = os.getenv("UNPAYWALL_EMAIL", "")

    if not UNPAYWALL_EMAIL:
        logger.warning(
            "UNPAYWALL_EMAIL non défini — source Unpaywall désactivée. "
            "Ajoutez --email votre@email.com ou UNPAYWALL_EMAIL=... dans secrets.env"
        )

    engine = create_engine(DB_URL, pool_pre_ping=True)

    # ── Récupérer les documents sans full-text ──
    with engine.connect() as conn:
        if args.reprocess:
            # Tous les documents du projet
            where_clause = "WHERE d.project_context = :project"
        else:
            # Documents sans aucun chunk full_text
            where_clause = """
                WHERE d.project_context = :project
                  AND NOT EXISTS (
                      SELECT 1 FROM document_chunk c
                      WHERE c.document_id = d.id AND c.chunk_type = 'full_text'
                  )
            """

        query = text(f"""
            SELECT d.id, d.external_id, d.doi, d.pmid, d.openalex_id,
                   d.source, d.title, d.url, d.open_access
            FROM literature_document d
            {where_clause}
            ORDER BY d.id ASC
            {"LIMIT :limit" if args.limit > 0 else ""}
        """)
        params = {"project": args.project}
        if args.limit > 0:
            params["limit"] = args.limit

        rows = conn.execute(query, params).mappings().all()
        docs = [dict(r) for r in rows]

    total = len(docs)
    logger.info(f"{'[DRY-RUN] ' if args.dry_run else ''}Documents à traiter : {total}")
    if total == 0:
        logger.info("Aucun document à traiter — tous ont déjà un full-text.")
        return

    # ── Traitement parallèle ──
    stats = {"ok": 0, "not_found": 0, "empty": 0, "skipped": 0, "error": 0}
    sources_used: dict[str, int] = {}

    def _process(row):
        try:
            return process_document(row, args, engine)
        except Exception as e:
            logger.error(f"Erreur doc {row['id']} : {e}")
            return {"doc_id": row["id"], "status": "error", "source_used": None, "chunks": 0, "title": row.get("title", "")[:60]}

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_process, doc): doc for doc in docs}
        done = 0
        for future in as_completed(futures):
            done += 1
            res = future.result()
            status = res["status"]
            stats[status] = stats.get(status, 0) + 1
            if res["source_used"]:
                src = res["source_used"].split(":")[0]
                sources_used[src] = sources_used.get(src, 0) + 1

            if status == "ok":
                logger.info(
                    f"[{done}/{total}] ✅ doc {res['doc_id']} — {res['chunks']} chunks "
                    f"({res['source_used']}) — {res['title']}"
                )
            elif done % 50 == 0 or status == "error":
                logger.info(f"[{done}/{total}] {status.upper()} doc {res['doc_id']} — {res['title']}")

    # ── Rapport final ──
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"{'[DRY-RUN] ' if args.dry_run else ''}RAPPORT FINAL — {total} documents traités")
    logger.info(f"  ✅ Full-text récupéré : {stats.get('ok', 0)}")
    logger.info(f"  ❌ Non trouvé        : {stats.get('not_found', 0)}")
    logger.info(f"  ⚠️  Contenu vide      : {stats.get('empty', 0)}")
    logger.info(f"  💥 Erreurs           : {stats.get('error', 0)}")
    logger.info("")
    if sources_used:
        logger.info("  Sources utilisées :")
        for src, count in sorted(sources_used.items(), key=lambda x: -x[1]):
            logger.info(f"    {src:20s} : {count}")
    logger.info("=" * 60)

    if not args.dry_run and stats.get("ok", 0) > 0:
        logger.info("")
        logger.info("→ Relancez embed_corpus.py pour embedder les nouveaux chunks :")
        logger.info("  python3 embed_corpus.py --project gesica")


if __name__ == "__main__":
    main()
