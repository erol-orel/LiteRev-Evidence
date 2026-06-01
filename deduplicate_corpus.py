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
"""

import argparse
import hashlib
import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import Optional

import psycopg2
import psycopg2.extras

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("dedup")

# ─── Configuration ─────────────────────────────────────────────────────────────
DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://literev:MyNewStrongPassword!@10.10.1.10:5432/literev"
)


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
    # Normalisation Unicode (supprime les accents)
    title = unicodedata.normalize('NFKD', title)
    title = ''.join(c for c in title if not unicodedata.combining(c))
    # Minuscules
    title = title.lower()
    # Supprime la ponctuation et les espaces multiples
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
        return 0.0  # Optimisation : skip si longueurs trop différentes
    # Matrice DP
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
    score += min(citations / 10.0, 20.0)  # Max 20 points pour les citations
    if doc.get('doi'):
        score += 5.0
    if doc.get('authors'):
        score += 5.0
    if doc.get('journal'):
        score += 5.0
    if doc.get('year') and doc['year'] >= 2018:
        score += 5.0
    return min(score, 100.0)


def get_connection():
    """Crée une connexion PostgreSQL."""
    return psycopg2.connect(DB_URL)


def fetch_all_documents(conn) -> list[dict]:
    """Récupère tous les documents GESICA avec leurs métadonnées."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
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
        """)
        return [dict(r) for r in cur.fetchall()]


def find_duplicate_groups(docs: list[dict]) -> list[list[dict]]:
    """
    Identifie les groupes de doublons.
    Retourne une liste de groupes, chaque groupe contant ≥2 documents.
    """
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

    # Groupes par PMID (pour les non encore groupés)
    for pmid, group in pmid_index.items():
        ungrouped = [d for d in group if d['id'] not in used_ids]
        if len(ungrouped) >= 2:
            groups.append(ungrouped)
            used_ids.update(d['id'] for d in ungrouped)

    # Groupes par titre normalisé (hash exact d'abord)
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
    remaining = [d for d in docs if d['id'] not in used_ids]
    normalized_titles = [(d, normalize_title(d.get('title'))) for d in remaining]
    normalized_titles = [(d, nt) for d, nt in normalized_titles if nt]

    for i, (doc_i, title_i) in enumerate(normalized_titles):
        if doc_i['id'] in used_ids:
            continue
        similar_group = [doc_i]
        for j, (doc_j, title_j) in enumerate(normalized_titles):
            if i == j or doc_j['id'] in used_ids:
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
    # Trier par score de qualité décroissant
    scored = [(doc, compute_quality_score(doc, doc.get('has_fulltext', False))) for doc in group]
    scored.sort(key=lambda x: (-x[1], x[0]['id']))
    return scored[0][0]


def merge_metadata(canonical: dict, duplicates: list[dict]) -> dict:
    """
    Fusionne les métadonnées manquantes du canonique avec celles des doublons.
    Retourne un dict des champs à mettre à jour sur le canonique.
    """
    updates = {}
    fields_to_merge = ['doi', 'pmid', 'authors', 'journal', 'citation_count',
                       'abstract', 'year', 'keywords', 'language', 'study_design',
                       'sample_size', 'country', 'open_access', 'affiliations',
                       'mesh_terms', 'structured_abstract', 'publication_type']

    for field in fields_to_merge:
        if not canonical.get(field):
            for dup in duplicates:
                if dup.get(field):
                    updates[field] = dup[field]
                    break
        elif field == 'citation_count':
            # Prendre le maximum des citations
            max_citations = max(
                (d.get('citation_count') or 0) for d in [canonical] + duplicates
            )
            if max_citations > (canonical.get('citation_count') or 0):
                updates['citation_count'] = max_citations

    return updates


def apply_deduplication(conn, groups: list[list[dict]], delete_duplicates: bool = False) -> dict:
    """
    Applique la déduplication en base :
    - Marque les doublons avec is_duplicate=TRUE et canonical_id
    - Fusionne les métadonnées manquantes vers le canonique
    - Optionnel : supprime les doublons (cascade sur document_chunk)
    """
    stats = {
        'groups_processed': 0,
        'duplicates_marked': 0,
        'duplicates_deleted': 0,
        'metadata_merged': 0,
        'errors': 0,
    }

    with conn.cursor() as cur:
        for group in groups:
            try:
                canonical = choose_canonical(group)
                duplicates = [d for d in group if d['id'] != canonical['id']]

                # Fusionner les métadonnées
                updates = merge_metadata(canonical, duplicates)
                if updates:
                    set_clause = ', '.join(f"{k} = %({k})s" for k in updates)
                    updates['id'] = canonical['id']
                    cur.execute(
                        f"UPDATE literature_document SET {set_clause} WHERE id = %(id)s",
                        updates
                    )
                    stats['metadata_merged'] += 1

                # Mettre à jour le quality_score du canonique
                qs = compute_quality_score(canonical, canonical.get('has_fulltext', False))
                cur.execute(
                    "UPDATE literature_document SET quality_score = %s, title_hash = %s WHERE id = %s",
                    (qs, title_hash(canonical.get('title')), canonical['id'])
                )

                if delete_duplicates:
                    # Supprimer les doublons (cascade sur document_chunk via FK)
                    dup_ids = [d['id'] for d in duplicates]
                    cur.execute(
                        "DELETE FROM literature_document WHERE id = ANY(%s)",
                        (dup_ids,)
                    )
                    stats['duplicates_deleted'] += len(dup_ids)
                else:
                    # Marquer les doublons
                    for dup in duplicates:
                        dup_qs = compute_quality_score(dup, dup.get('has_fulltext', False))
                        cur.execute(
                            """UPDATE literature_document
                               SET is_duplicate = TRUE,
                                   canonical_id = %s,
                                   quality_score = %s,
                                   title_hash = %s
                               WHERE id = %s""",
                            (canonical['id'], dup_qs, title_hash(dup.get('title')), dup['id'])
                        )
                        stats['duplicates_marked'] += 1

                stats['groups_processed'] += 1

            except Exception as e:
                logger.error(f"Erreur groupe {[d['id'] for d in group]}: {e}")
                stats['errors'] += 1
                conn.rollback()
                continue

        conn.commit()

    return stats


def run_dedup(dry_run: bool = True, delete_duplicates: bool = False) -> None:
    """Point d'entrée principal."""
    logger.info("=== Déduplication du corpus GESICA ===")

    conn = get_connection()
    try:
        # 1. Récupérer tous les documents
        logger.info("Chargement des documents...")
        docs = fetch_all_documents(conn)
        logger.info(f"  {len(docs)} documents chargés")

        # 2. Trouver les groupes de doublons
        logger.info("Recherche des doublons...")
        groups = find_duplicate_groups(docs)
        total_dups = sum(len(g) - 1 for g in groups)
        logger.info(f"  {len(groups)} groupes de doublons trouvés ({total_dups} doublons)")

        if not groups:
            logger.info("Aucun doublon trouvé. Corpus propre.")
            return

        # 3. Afficher les groupes trouvés
        for i, group in enumerate(groups[:20]):  # Afficher max 20 groupes
            canonical = choose_canonical(group)
            dups = [d for d in group if d['id'] != canonical['id']]
            logger.info(f"\nGroupe {i+1} ({len(group)} docs) :")
            logger.info(f"  MAÎTRE [id={canonical['id']}] {canonical['title'][:80]}")
            logger.info(f"    DOI={canonical.get('doi','')}, PMID={canonical.get('pmid','')}, "
                       f"fulltext={canonical.get('has_fulltext',False)}, "
                       f"citations={canonical.get('citation_count',0)}")
            for dup in dups:
                logger.info(f"  DOUBLON [id={dup['id']}] {dup['title'][:80]}")
                logger.info(f"    DOI={dup.get('doi','')}, PMID={dup.get('pmid','')}, "
                           f"fulltext={dup.get('has_fulltext',False)}, "
                           f"citations={dup.get('citation_count',0)}")

        if len(groups) > 20:
            logger.info(f"  ... et {len(groups) - 20} autres groupes")

        if dry_run:
            logger.info("\n[DRY RUN] Aucune modification appliquée. Utilisez --execute pour appliquer.")
            return

        # 4. Appliquer la déduplication
        logger.info("\nApplication de la déduplication...")
        stats = apply_deduplication(conn, groups, delete_duplicates=delete_duplicates)
        logger.info(f"\n=== Résultats ===")
        logger.info(f"  Groupes traités : {stats['groups_processed']}")
        logger.info(f"  Doublons marqués : {stats['duplicates_marked']}")
        logger.info(f"  Doublons supprimés : {stats['duplicates_deleted']}")
        logger.info(f"  Métadonnées fusionnées : {stats['metadata_merged']}")
        logger.info(f"  Erreurs : {stats['errors']}")

        # 5. Mettre à jour les title_hash pour tous les documents non-doublons
        logger.info("\nMise à jour des title_hash...")
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT id, title FROM literature_document
                WHERE project_context = 'gesica'
                  AND title_hash IS NULL
                  AND (is_duplicate IS NULL OR is_duplicate = FALSE)
            """)
            rows = cur.fetchall()
        with conn.cursor() as cur:
            for row in rows:
                th = title_hash(row['title'])
                if th:
                    cur.execute(
                        "UPDATE literature_document SET title_hash = %s WHERE id = %s",
                        (th, row['id'])
                    )
            conn.commit()
        logger.info(f"  {len(rows)} title_hash mis à jour")

    finally:
        conn.close()


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
