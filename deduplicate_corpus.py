#!/usr/bin/env python3
"""
deduplicate_corpus.py — Déduplication intelligente du corpus GESICA

Stratégie de déduplication (par ordre de priorité) :
  1. DOI exact (normalisé en minuscules, sans espaces)
  2. PMID exact
  3. Titre normalisé (minuscules, sans ponctuation, distance de Levenshtein < 5%)

Pour chaque groupe de doublons, le document maître (canonical) est choisi selon :
  - Priorité 1 : a du full-text (chunk de type 'fulltext_section')
  - Priorité 2 : citation_count le plus élevé
  - Priorité 3 : abstract le plus long
  - Priorité 4 : le plus ancien (id le plus petit)

Les doublons sont marqués avec is_duplicate=TRUE et canonical_id=<id_maître>.
Les métadonnées manquantes sont copiées du doublon vers le maître (merge).

Usage :
  python3 deduplicate_corpus.py --dry-run          # Affiche les doublons sans modifier la DB
  python3 deduplicate_corpus.py --execute           # Applique la déduplication
  python3 deduplicate_corpus.py --execute --delete  # Supprime les doublons (cascade chunks)

Dépendances : sqlalchemy, psycopg (déjà installés dans l'environnement LiteRev)
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import create_engine, text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("dedup")

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

for _ep in [".env", "/opt/literev-api/.env", "/opt/literev-api/secrets.env", "/etc/literev/secrets"]:
    _load_env_file(_ep)

DB_URL = os.environ.get(
    "DB_URL",
    os.environ.get(
        "DATABASE_URL",
        "postgresql+psycopg://literev:MyNewStrongPassword!@10.10.1.10:5432/literev"
    )
)

engine = create_engine(DB_URL, pool_pre_ping=True)


# ─── Fonctions utilitaires ──────────────────────────────────────────────────────

def normalize_doi(doi: Optional[str]) -> Optional[str]:
    """Normalise un DOI : minuscules, sans espaces, sans préfixe https://doi.org/."""
    if not doi:
        return None
    doi = doi.strip().lower()
    doi = re.sub(r'^https?://(dx\.)?doi\.org/', '', doi)
    doi = doi.strip('/')
    return doi if doi else None


def normalize_title(title: Optional[str]) -> Optional[str]:
    """Normalise un titre pour la comparaison : minuscules, sans accents, sans ponctuation."""
    if not title:
        return None
    title = unicodedata.normalize('NFKD', title)
    title = ''.join(c for c in title if not unicodedata.combining(c))
    title = title.lower()
    title = re.sub(r'[^\w\s]', ' ', title)
    title = re.sub(r'\s+', ' ', title).strip()
    return title if len(title) >= 10 else None


def title_hash(title: Optional[str]) -> Optional[str]:
    """Hash MD5 du titre normalisé pour indexation rapide."""
    norm = normalize_title(title)
    if not norm:
        return None
    return hashlib.md5(norm.encode()).hexdigest()


def levenshtein_ratio(s1: str, s2: str) -> float:
    """Calcule le ratio de similarité Levenshtein (0.0 à 1.0)."""
    if not s1 or not s2:
        return 0.0
    len1, len2 = len(s1), len(s2)
    if abs(len1 - len2) / max(len1, len2) > 0.2:
        return 0.0
    dp = list(range(len2 + 1))
    for i in range(1, len1 + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev[j - 1] + cost)
    dist = dp[len2]
    return 1.0 - dist / max(len1, len2)


def compute_quality_score(doc: dict, has_fulltext: bool) -> float:
    """Calcule un score de qualité (0-100) pour choisir le document maître."""
    score = 0.0
    if has_fulltext:
        score += 40.0
    if doc.get('abstract') and len(doc['abstract']) > 200:
        score += 20.0
    elif doc.get('abstract'):
        score += 10.0
    citations = doc.get('citation_count') or 0
    score += min(citations / 10.0, 20.0)
    if doc.get('doi'):
        score += 5.0
    if doc.get('authors'):
        score += 5.0
    if doc.get('journal'):
        score += 5.0
    if doc.get('year') and doc['year'] >= 2018:
        score += 5.0
    return min(score, 100.0)


# ─── Lecture des documents ──────────────────────────────────────────────────────

def fetch_all_documents() -> list[dict]:
    """Récupère tous les documents GESICA avec leurs métadonnées."""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT
                d.id, d.title, d.abstract, d.year, d.source, d.external_id,
                d.doi, d.pmid, d.authors, d.journal, d.citation_count,
                d.project_context, d.scenario_type, d.is_duplicate, d.canonical_id,
                EXISTS (
                    SELECT 1 FROM document_chunk c
                    WHERE c.document_id = d.id
                      AND c.chunk_type = 'fulltext_section'
                ) AS has_fulltext
            FROM literature_document d
            WHERE d.project_context = 'gesica'
              AND (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)
            ORDER BY d.id ASC
        """)).mappings().all()
    return [dict(r) for r in rows]


# ─── Détection des doublons ─────────────────────────────────────────────────────

def find_duplicate_groups(docs: list[dict]) -> list[list[dict]]:
    """Identifie les groupes de doublons. Retourne une liste de groupes (≥2 docs)."""
    groups: list[list[dict]] = []
    used_ids: set[int] = set()

    # Index par DOI normalisé
    doi_index: dict[str, list[dict]] = {}
    for doc in docs:
        ndoi = normalize_doi(doc.get('doi'))
        if ndoi:
            doi_index.setdefault(ndoi, []).append(doc)

    # Index par PMID
    pmid_index: dict[str, list[dict]] = {}
    for doc in docs:
        pmid = (doc.get('pmid') or '').strip()
        if pmid:
            pmid_index.setdefault(pmid, []).append(doc)

    # Groupes par DOI
    for doi, group in doi_index.items():
        if len(group) >= 2:
            ids = [d['id'] for d in group]
            if not any(i in used_ids for i in ids):
                groups.append(group)
                used_ids.update(ids)

    # Groupes par PMID
    for pmid, group in pmid_index.items():
        ungrouped = [d for d in group if d['id'] not in used_ids]
        if len(ungrouped) >= 2:
            groups.append(ungrouped)
            used_ids.update(d['id'] for d in ungrouped)

    # Groupes par hash de titre exact
    title_hash_index: dict[str, list[dict]] = {}
    for doc in docs:
        if doc['id'] in used_ids:
            continue
        th = title_hash(doc.get('title'))
        if th:
            title_hash_index.setdefault(th, []).append(doc)

    for th, group in title_hash_index.items():
        ungrouped = [d for d in group if d['id'] not in used_ids]
        if len(ungrouped) >= 2:
            groups.append(ungrouped)
            used_ids.update(d['id'] for d in ungrouped)

    # Groupes par similarité de titre (Levenshtein ≥ 0.92)
    # Optimisation par blocage : on ne compare que les titres qui partagent le même premier mot
    # et qui ont des longueurs similaires (différence < 15%)
    remaining = [d for d in docs if d['id'] not in used_ids]
    normalized_titles = []
    for d in remaining:
        nt = normalize_title(d.get('title'))
        if nt:
            first_word = nt.split()[0] if nt.split() else ""
            normalized_titles.append((d, nt, first_word, len(nt)))

    # Indexation par premier mot pour recherche rapide
    first_word_index: dict[str, list[tuple]] = {}
    for item in normalized_titles:
        first_word_index.setdefault(item[2], []).append(item)

    for first_word, items in first_word_index.items():
        if not first_word or len(items) < 2:
            continue
        # On compare les éléments du même groupe
        for i, (doc_i, title_i, _, len_i) in enumerate(items):
            if doc_i['id'] in used_ids:
                continue
            similar_group = [doc_i]
            for j, (doc_j, title_j, _, len_j) in enumerate(items):
                if i == j or doc_j['id'] in used_ids:
                    continue
                # Filtre de longueur rapide (différence max 15%)
                if abs(len_i - len_j) / max(len_i, len_j) > 0.15:
                    continue
                ratio = levenshtein_ratio(title_i, title_j)
                if ratio >= 0.92:
                    similar_group.append(doc_j)
            if len(similar_group) >= 2:
                groups.append(similar_group)
                used_ids.update(d['id'] for d in similar_group)

    return groups


def choose_canonical(group: list[dict]) -> dict:
    """Choisit le document maître dans un groupe de doublons."""
    scored = [(doc, compute_quality_score(doc, doc.get('has_fulltext', False))) for doc in group]
    scored.sort(key=lambda x: (-x[1], x[0]['id']))
    return scored[0][0]


def merge_metadata(canonical: dict, duplicates: list[dict]) -> dict:
    """Fusionne les métadonnées manquantes du canonique avec celles des doublons."""
    updates: dict[str, Any] = {}
    fields_to_merge = [
        'doi', 'pmid', 'authors', 'journal', 'citation_count',
        'abstract', 'year', 'keywords', 'language', 'study_design',
        'sample_size', 'country', 'open_access', 'affiliations',
        'mesh_terms', 'structured_abstract', 'publication_type',
    ]
    for field in fields_to_merge:
        if not canonical.get(field):
            for dup in duplicates:
                if dup.get(field):
                    updates[field] = dup[field]
                    break
        elif field == 'citation_count':
            max_citations = max(
                (d.get('citation_count') or 0) for d in [canonical] + duplicates
            )
            if max_citations > (canonical.get('citation_count') or 0):
                updates['citation_count'] = max_citations
    return updates


# ─── Application de la déduplication ───────────────────────────────────────────

def apply_deduplication(groups: list[list[dict]], delete_duplicates: bool = False) -> dict:
    """Applique la déduplication en base via SQLAlchemy."""
    stats = {
        'groups_processed': 0,
        'duplicates_marked': 0,
        'duplicates_deleted': 0,
        'metadata_merged': 0,
        'errors': 0,
    }

    with engine.begin() as conn:
        for group in groups:
            try:
                canonical = choose_canonical(group)
                duplicates = [d for d in group if d['id'] != canonical['id']]

                # Fusionner les métadonnées
                updates = merge_metadata(canonical, duplicates)
                if updates:
                    set_parts = ", ".join(f"{k} = :{k}" for k in updates)
                    updates['_id'] = canonical['id']
                    conn.execute(
                        text(f"UPDATE literature_document SET {set_parts} WHERE id = :_id"),
                        updates
                    )
                    stats['metadata_merged'] += 1

                # Mettre à jour quality_score et title_hash du canonique
                qs = compute_quality_score(canonical, bool(canonical.get('has_fulltext')))
                conn.execute(text("""
                    UPDATE literature_document
                    SET quality_score = :qs, title_hash = :th
                    WHERE id = :id
                """), {"qs": qs, "th": title_hash(canonical.get('title')), "id": canonical['id']})

                if delete_duplicates:
                    dup_ids = [d['id'] for d in duplicates]
                    conn.execute(
                        text("DELETE FROM literature_document WHERE id = ANY(:ids)"),
                        {"ids": dup_ids}
                    )
                    stats['duplicates_deleted'] += len(dup_ids)
                else:
                    for dup in duplicates:
                        dup_qs = compute_quality_score(dup, bool(dup.get('has_fulltext')))
                        conn.execute(text("""
                            UPDATE literature_document
                            SET is_duplicate = TRUE,
                                canonical_id  = :cid,
                                quality_score = :qs,
                                title_hash    = :th
                            WHERE id = :id
                        """), {
                            "cid": canonical['id'],
                            "qs": dup_qs,
                            "th": title_hash(dup.get('title')),
                            "id": dup['id'],
                        })
                        stats['duplicates_marked'] += 1

                stats['groups_processed'] += 1

            except Exception as e:
                logger.error(f"Erreur groupe {[d['id'] for d in group]}: {e}")
                stats['errors'] += 1
                continue

    return stats


# ─── Mise à jour des title_hash ─────────────────────────────────────────────────

def update_title_hashes() -> int:
    """Met à jour les title_hash pour tous les documents sans hash."""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT id, title FROM literature_document
            WHERE project_context = 'gesica'
              AND title_hash IS NULL
              AND (is_duplicate IS NULL OR is_duplicate = FALSE)
        """)).mappings().all()

    updated = 0
    with engine.begin() as conn:
        for row in rows:
            th = title_hash(row['title'])
            if th:
                conn.execute(
                    text("UPDATE literature_document SET title_hash = :th WHERE id = :id"),
                    {"th": th, "id": row['id']}
                )
                updated += 1
    return updated


# ─── Point d'entrée ─────────────────────────────────────────────────────────────

def run_dedup(dry_run: bool = True, delete_duplicates: bool = False) -> None:
    logger.info("=== Déduplication du corpus GESICA ===")

    # 1. Charger les documents
    logger.info("Chargement des documents...")
    docs = fetch_all_documents()
    logger.info(f"  {len(docs)} documents chargés")

    # 2. Trouver les groupes de doublons
    logger.info("Recherche des doublons...")
    groups = find_duplicate_groups(docs)
    total_dups = sum(len(g) - 1 for g in groups)
    logger.info(f"  {len(groups)} groupes de doublons trouvés ({total_dups} doublons)")

    if not groups:
        logger.info("Aucun doublon trouvé. Corpus propre.")
        return

    # 3. Afficher les groupes (max 20)
    for i, group in enumerate(groups[:20]):
        canonical = choose_canonical(group)
        dups = [d for d in group if d['id'] != canonical['id']]
        logger.info(f"\nGroupe {i+1} ({len(group)} docs) :")
        logger.info(f"  MAÎTRE [id={canonical['id']}] {(canonical['title'] or '')[:80]}")
        logger.info(f"    DOI={canonical.get('doi','')}, PMID={canonical.get('pmid','')}, "
                    f"fulltext={canonical.get('has_fulltext', False)}, "
                    f"citations={canonical.get('citation_count', 0)}")
        for dup in dups:
            logger.info(f"  DOUBLON [id={dup['id']}] {(dup['title'] or '')[:80]}")
            logger.info(f"    DOI={dup.get('doi','')}, PMID={dup.get('pmid','')}, "
                        f"fulltext={dup.get('has_fulltext', False)}, "
                        f"citations={dup.get('citation_count', 0)}")

    if len(groups) > 20:
        logger.info(f"  ... et {len(groups) - 20} autres groupes")

    if dry_run:
        logger.info("\n[DRY RUN] Aucune modification appliquée. Utilisez --execute pour appliquer.")
        return

    # 4. Appliquer la déduplication
    logger.info("\nApplication de la déduplication...")
    stats = apply_deduplication(groups, delete_duplicates=delete_duplicates)
    logger.info("\n=== Résultats ===")
    logger.info(f"  Groupes traités      : {stats['groups_processed']}")
    logger.info(f"  Doublons marqués     : {stats['duplicates_marked']}")
    logger.info(f"  Doublons supprimés   : {stats['duplicates_deleted']}")
    logger.info(f"  Métadonnées fusionnées: {stats['metadata_merged']}")
    logger.info(f"  Erreurs              : {stats['errors']}")

    # 5. Mettre à jour les title_hash
    logger.info("\nMise à jour des title_hash...")
    n = update_title_hashes()
    logger.info(f"  {n} title_hash mis à jour")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Déduplication du corpus GESICA")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Affiche les doublons sans modifier la DB (défaut)")
    parser.add_argument("--execute", action="store_true",
                        help="Applique la déduplication en base")
    parser.add_argument("--delete", action="store_true",
                        help="Supprime les doublons (au lieu de les marquer)")
    args = parser.parse_args()
    dry = not args.execute
    run_dedup(dry_run=dry, delete_duplicates=args.delete)
