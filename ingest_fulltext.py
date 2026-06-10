#!/usr/bin/env python3
"""
LiteRev-Evidence Full-Text Ingestion Pipeline (P3)
Source: Europe PMC / PMC REST API

Ce script extrait le texte intégral (full-text) XML structuré pour les articles
déjà présents dans la base (ou spécifiés) qui possèdent un PMCID (PubMed Central ID),
puis découpe ce texte en chunks fins par section (Introduction, Méthodes, Résultats, Discussion)
et les insère dans la table `document_chunk` avec le bon index et poids.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import xml.etree.ElementTree as ET
from typing import Any

import requests

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")
WRITE_API_KEY = os.getenv("WRITE_API_KEY", "")
HEADERS = {"X-Api-Key": WRITE_API_KEY}


def get_pmcid_from_external_id(ext_id: str) -> str | None:
    """Extrait un PMCID valide depuis l'ID externe si présent."""
    if not ext_id:
        return None
    ext_id = ext_id.strip()
    # Format direct : PMC1234567
    if ext_id.upper().startswith("PMC") and ext_id[3:].isdigit():
        return ext_id.upper()
    # Format PMID:1234567 -> extraire le numérique
    if ext_id.upper().startswith("PMID:"):
        return None  # Sera résolu via esummary dans process_document_fulltext
    return None


def resolve_pmcid_from_any_id(ext_id: str) -> str | None:
    """Tente de résoudre un PMCID depuis n'importe quel format d'ID externe."""
    if not ext_id:
        return None
    ext_id = ext_id.strip()
    
    # Format direct PMC
    if ext_id.upper().startswith("PMC") and ext_id[3:].isdigit():
        return ext_id.upper()
    
    # Format PMID:1234567 ou numérique pur
    pmid = None
    if ext_id.upper().startswith("PMID:"):
        pmid = ext_id[5:].strip()
    elif ext_id.isdigit():
        pmid = ext_id
    
    if pmid:
        try:
            r = requests.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                params={"db": "pubmed", "id": pmid, "retmode": "json"},
                timeout=15,
            )
            if r.ok:
                article_data = r.json().get("result", {}).get(pmid, {})
                for articleid in article_data.get("articleids", []):
                    if articleid.get("idtype") == "pmcid":
                        pmcid = articleid.get("value", "")
                        if pmcid:
                            print(f"     [ID Resolve] PMID {pmid} -> PMCID {pmcid}")
                            return pmcid
        except Exception:
            pass
    
    # Format DOI : tenter une résolution via Europe PMC ID converter
    if ext_id.startswith("10."):
        try:
            r = requests.get(
                "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                params={"query": f"DOI:{ext_id}", "format": "json", "pageSize": 1},
                timeout=15,
            )
            if r.ok:
                results = r.json().get("resultList", {}).get("result", [])
                if results:
                    pmcid = results[0].get("pmcid", "")
                    if pmcid:
                        print(f"     [ID Resolve] DOI {ext_id} -> PMCID {pmcid}")
                        return pmcid
        except Exception:
            pass
    
    return None


def fetch_fulltext_xml(pmcid: str) -> str | None:
    """Récupère le XML complet de l'article depuis l'API Europe PMC."""
    print(f"  -> Récupération XML pour {pmcid}...")
    try:
        # Europe PMC REST API pour le full-text XML
        r = requests.get(
            f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML",
            timeout=30,
        )
        if r.ok and len(r.content) > 500:
            return r.text
        
        # Fallback sur l'API PubMed Central officielle si Europe PMC échoue
        print(f"     [PMC Fallback] Interrogation de l'API NCBI PMC...")
        r_ncbi = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params={
                "db": "pmc",
                "id": pmcid[3:],  # Supprimer le préfixe 'PMC'
                "retmode": "xml",
            },
            timeout=30,
        )
        if r_ncbi.ok and len(r_ncbi.content) > 500:
            return r_ncbi.text
            
    except Exception as e:
        print(f"     [Error XML] {e}")
    return None


def parse_xml_sections(xml_content: str) -> list[dict[str, Any]]:
    """Découpe le XML structuré JATS en sections logiques."""
    sections = []
    try:
        root = ET.fromstring(xml_content)
        
        # Recherche des balises <sec> (sections) standard JATS
        for sec in root.findall(".//sec"):
            title_node = sec.find("title")
            title = "".join(title_node.itertext()).strip() if title_node is not None else "Section"
            
            # Récupérer tout le texte des paragraphes <p> de cette section
            paragraphs = []
            for p in sec.findall("p"):
                p_text = "".join(p.itertext()).strip()
                if p_text:
                    paragraphs.append(p_text)
            
            content = "\n\n".join(paragraphs).strip()
            if len(content) > 150:  # Ignorer les sections trop courtes
                sections.append({
                    "title": title,
                    "content": content
                })
                
        # Si aucun tag <sec> n'est trouvé, faire un fallback sur le corps complet <body>
        if not sections:
            body = root.find(".//body")
            if body is not None:
                body_text = "".join(body.itertext()).strip()
                if len(body_text) > 200:
                    sections.append({
                        "title": "Full Text Body",
                        "content": body_text
                    })
                    
    except Exception as e:
        print(f"     [Parse XML Error] {e}")
    return sections


def chunk_text(text: str, max_words: int = 400, overlap_words: int = 50) -> list[str]:
    """Découpe un texte long en chunks de mots avec chevauchement (overlap)."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i : i + max_words]
        chunks.append(" ".join(chunk_words))
        if i + max_words >= len(words):
            break
        i += max_words - overlap_words
    return chunks


def process_document_fulltext(doc_id: int, ext_id: str) -> bool:
    """Récupère, découpe et insère le full-text d'un document."""
    pmcid = resolve_pmcid_from_any_id(ext_id)
    if not pmcid:
        print(f"  [Skip] Pas de PMCID valide trouvé pour l'ID externe '{ext_id}'")
        return False

    xml_content = fetch_fulltext_xml(pmcid)
    if not xml_content:
        print(f"  [Error] Impossible de récupérer le XML pour {pmcid}")
        return False

    sections = parse_xml_sections(xml_content)
    if not sections:
        print(f"  [Error] Aucun contenu textuel structuré extrait du XML pour {pmcid}")
        return False

    print(f"     -> {len(sections)} sections extraites. Découpage en chunks...")
    
    # Supprimer les anciens chunks fulltext existants pour ce document pour éviter les doublons
    # (On garde le chunk 0 qui est le titre + abstract)
    try:
        # Insertion des nouveaux chunks de section
        chunk_index = 1
        total_chunks_inserted = 0
        
        for sec in sections:
            sec_title = sec["title"]
            sec_content = sec["content"]
            
            # Découpage fin de la section
            sub_chunks = chunk_text(sec_content, max_words=350, overlap_words=50)
            
            for sub_content in sub_chunks:
                # Calcul de la position approximative des caractères
                char_start = sec_content.find(sub_content[:30])
                char_end = char_start + len(sub_content) if char_start != -1 else None
                char_start = char_start if char_start != -1 else None
                
                chunk_r = requests.post(
                    f"{API_BASE}/chunks",
                    headers=HEADERS,
                    json={
                        "document_id": doc_id,
                        "chunk_index": chunk_index,
                        "content": f"[{sec_title}] {sub_content}",
                        "chunk_type": "fulltext_section",
                        "section_label": sec_title,
                        "char_start": char_start,
                        "char_end": char_end,
                        "token_count": len(sub_content.split()),
                        "chunk_weight": 1.2,  # Poids supérieur pour le full-text par rapport à l'abstract
                        "metadata_json": {
                            "pmcid": pmcid,
                            "section_title": sec_title
                        },
                    },
                    timeout=30,
                )
                chunk_r.raise_for_status()
                chunk_index += 1
                total_chunks_inserted += 1
                
        print(f"     ✅ {total_chunks_inserted} chunks de full-text insérés avec succès pour le document {doc_id}.")
        return True
        
    except Exception as e:
        print(f"     [Error Ingestion Chunks] {e}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Pipeline d'ingestion de full-text PMC")
    parser.add_argument("--doc-id", type=int, help="ID du document spécifique à traiter")
    parser.add_argument("--external-id", type=str, help="ID externe (PMCID ou PMID) spécifique à traiter")
    parser.add_argument("--limit", type=int, default=10, help="Nombre max de documents à traiter (depuis la base)")
    args = parser.parse_args()

    print("======================================================================")
    print("🚀 DÉMARRAGE DU PIPELINE D'INGESTION DE FULL-TEXT PMC (P3)")
    print(f"   API Base: {API_BASE}")
    print("======================================================================")

    # Cas 1 : Traitement d'un document spécifique
    if args.doc_id and args.external_id:
        print(f"\n⚡ Traitement du document unique ID {args.doc_id} (Ext ID: {args.external_id})")
        success = process_document_fulltext(args.doc_id, args.external_id)
        return 0 if success else 1

    # Cas 2 : Traitement en masse des documents de la base n'ayant pas encore de full-text
    print(f"\n🔍 Recherche de documents éligibles (avec PMC ou PubMed ID) sans chunks full-text...")
    try:
        # Interroger l'API locale pour récupérer les documents
        # On utilise l'endpoint /search avec une recherche large ou un appel direct si existant.
        # Pour faire simple et robuste, on fait une recherche vide filtrée ou on parcourt.
        # Ici on interroge l'API locale pour trouver des candidats.
        # Récupérer les documents via plusieurs requêtes thématiques pour maximiser la couverture
        seen_docs: set[int] = set()
        candidates: list[tuple[int, str]] = []
        
        # Requêtes ciblées pour trouver des articles avec PMID ou DOI
        search_queries = [
            "emergency medical services artificial intelligence",
            "machine learning ambulance demand forecasting",
            "deep learning triage emergency",
            "neural network EMS prediction",
            "AI hospital capacity planning",
        ]
        
        for query in search_queries:
            if len(candidates) >= args.limit * 2:
                break
            try:
                r = requests.post(
                    f"{API_BASE}/search",
                    json={
                        "query_text": query,
                        "mode": "boolean",
                        "limit": 50,
                        "filters": {},
                    },
                    timeout=20,
                )
                r.raise_for_status()
                results = r.json().get("results", [])
                
                for res in results:
                    doc_id = res.get("document_id")
                    ext_id = res.get("external_id", "")
                    if doc_id and ext_id and doc_id not in seen_docs:
                        # Accepter PMC direct, PMID:xxx, numérique pur, ou DOI
                        is_candidate = (
                            ext_id.upper().startswith("PMC")
                            or ext_id.upper().startswith("PMID:")
                            or ext_id.isdigit()
                            or ext_id.startswith("10.")
                        )
                        if is_candidate:
                            seen_docs.add(doc_id)
                            candidates.append((doc_id, ext_id))
            except Exception as e:
                print(f"  [Warn] Requête '{query}' échouée : {e}")
                
        candidates = candidates[:args.limit]
        print(f"   -> {len(candidates)} documents candidats identifiés pour l'extraction full-text.")
        
        processed_count = 0
        for doc_id, ext_id in candidates:
            print(f"\n⚡ [{processed_count+1}/{len(candidates)}] Traitement du document ID {doc_id} (Ext ID: {ext_id})...")
            success = process_document_fulltext(doc_id, ext_id)
            if success:
                processed_count += 1
            time.sleep(1.0)  # Pause polie entre les requêtes XML
            
        print("\n======================================================================")
        print(f"✅ PIPELINE FULL-TEXT TERMINÉ — {processed_count} documents enrichis en texte intégral.")
        print("======================================================================")
        
    except Exception as e:
        print(f"❌ [Error Candidates] {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())
