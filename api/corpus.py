"""Corpus statistics, full-text stats, maintenance, deduplication status.

Extracted from main.py (LiteRev API); `main` re-exports everything for the scripts,
tools and tests.
"""
from __future__ import annotations

import os
from typing import Any

from fastapi import Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import text, bindparam

from .core import _openai_in_cooldown, _trip_openai_cooldown, app, engine, require_api_key
from .documents import _embed_chunks_resilient
from .search import _DUP_ANY_IDS_SQL, _DUP_FLAG_UPDATE_SQL, _count_corpus_duplicates

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 Endpoints: Stats and Scenarios
# ─────────────────────────────────────────────────────────────────────────────

# Regroupement des sources en libellés canoniques pour les tableaux de bord. La
# colonne `source` accumule des variantes héritées (casse, anciens tags, valeurs
# vides) qui gonflaient le compteur « Sources » et fragmentaient les barres. On
# mappe les variantes connues vers les sources fédérées + « Préprints » ; toute
# autre source garde son nom réel (les valeurs vides → « Non précisé »).
def _canonical_source(s: str | None) -> str:
    t = (s or "").strip().lower()
    if not t:
        return "Non précisé"
    # Europe PMC en premier : la chaîne 'europepmc' contient 'pmc'.
    if "europepmc" in t or "europe_pmc" in t or "europe pmc" in t or t == "epmc":
        return "Europe PMC"
    if any(k in t for k in ("pubmed", "pmc", "medline", "ncbi", "entrez", "pmid")):
        return "PubMed"
    if "openalex" in t:
        return "OpenAlex"
    if "crossref" in t or "cross_ref" in t or "cross-ref" in t:
        return "Crossref"
    if any(k in t for k in ("medrxiv", "biorxiv", "arxiv", "preprint", "ssrn",
                            "chemrxiv", "research square", "researchsquare", "osf", "psyarxiv")):
        return "Préprints"
    if "semantic" in t or t == "s2":
        return "Semantic Scholar"
    # Source non fédérée : on garde le nom réel (nettoyé) plutôt que de tout
    # masquer derrière « Autre » - l'utilisateur veut voir TOUTES les sources.
    return (s or "").strip()


@app.get("/corpus/stats")
def get_corpus_stats() -> dict[str, Any]:
    """Vue globale multi-projet du corpus."""
    sql_totals = text("""
        SELECT 
            COALESCE(project_context, 'unassigned') as project,
            COUNT(*) as count
        FROM literature_document
        GROUP BY project_context
    """)
    sql_sources = text("""
        SELECT 
            source,
            COUNT(*) as count
        FROM literature_document
        GROUP BY source
    """)
    sql_years = text("""
        SELECT
            year,
            COUNT(*) as count
        FROM literature_document
        WHERE year IS NOT NULL
          AND year BETWEEN 1800 AND EXTRACT(YEAR FROM CURRENT_DATE)::int  -- pas d'années futures/aberrantes
        GROUP BY year
        ORDER BY year DESC
    """)
    with engine.connect() as conn:
        totals = {r["project"]: r["count"] for r in conn.execute(sql_totals).mappings().all()}
        # Agrégation par source CANONIQUE (sinon les variantes héritées comptaient
        # chacune comme une « source » distincte → compteur et barres faussés).
        sources: dict[str, int] = {}
        for r in conn.execute(sql_sources).mappings().all():
            k = _canonical_source(r["source"])
            sources[k] = sources.get(k, 0) + int(r["count"] or 0)
        years = {r["year"]: r["count"] for r in conn.execute(sql_years).mappings().all()}
        
        total_docs = conn.execute(text("SELECT COUNT(*) FROM literature_document")).scalar() or 0
        total_chunks = conn.execute(text("SELECT COUNT(*) FROM document_chunk")).scalar() or 0

    return {
        "total_documents": total_docs,
        "total_chunks": total_chunks,
        "by_project": totals,
        "by_source": sources,
        "by_year": years,
    }

# ─── Endpoint : évolution temporelle et heatmap ─────────────────────────────
@app.get("/corpus/stats/by-year")
def get_corpus_stats_by_year() -> dict[str, Any]:
    """
    Distribution des articles par année (1800 → année courante), pour le graphique temporel.
    Retourne aussi la distribution par année ET par scénario pour la heatmap.
    """
    with engine.connect() as conn:
        # Articles par année (1800 → année courante)
        rows_year = conn.execute(text("""
            SELECT year, COUNT(*) as count
            FROM literature_document
            WHERE year >= 1800 AND year <= EXTRACT(YEAR FROM CURRENT_DATE)::int
            GROUP BY year
            ORDER BY year ASC
        """)).mappings().all()

        # Articles par année ET par scénario (1800 → année courante)
        rows_scenario_year = conn.execute(text("""
            SELECT d.year, ars.scenario_id, COUNT(*) as count
            FROM literature_document d
            JOIN article_scenarios ars ON ars.document_id = d.id
            WHERE d.year >= 1800 AND d.year <= EXTRACT(YEAR FROM CURRENT_DATE)::int
              AND (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)
            GROUP BY d.year, ars.scenario_id
            ORDER BY d.year ASC
        """)).mappings().all()

        # Articles par scénario ET par source (heatmap)
        rows_heatmap = conn.execute(text("""
            SELECT ars.scenario_id, d.source, COUNT(*) as count
            FROM literature_document d
            JOIN article_scenarios ars ON ars.document_id = d.id
            WHERE (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)
            GROUP BY ars.scenario_id, d.source
            ORDER BY ars.scenario_id, count DESC
        """)).mappings().all()

    by_year = {str(r["year"]): r["count"] for r in rows_year}

    # Construire la matrice scénario × année
    scenario_year: dict[str, dict[str, int]] = {}
    for r in rows_scenario_year:
        sid = r["scenario_id"]
        yr = str(r["year"])
        if sid not in scenario_year:
            scenario_year[sid] = {}
        scenario_year[sid][yr] = r["count"]

    # Construire la matrice scénario × source (même forme que l'endpoint /named :
    # clé = scenario_id, valeur = {name, sources: {src: {total, fulltext}}}).
    heatmap: dict[str, dict] = {}
    for r in rows_heatmap:
        sid = str(r["scenario_id"])
        src = _canonical_source(r["source"])
        entry = heatmap.setdefault(sid, {"name": sid, "sources": {}})
        cell = entry["sources"].setdefault(src, {"total": 0, "fulltext": 0})
        cell["total"] += int(r["count"] or 0)

    return {
        "by_year": by_year,
        "scenario_by_year": scenario_year,
        "heatmap_scenario_source": heatmap,
    }


# ─── Endpoint : statistiques full-text et mode hybrid ────────────────────────
@app.get("/corpus/fulltext-stats")
def get_fulltext_stats() -> dict[str, Any]:
    """
    Statistiques de couverture textuelle du corpus.
    Distingue les articles avec full-text (chunks 'fulltext_section')
    des articles avec seulement titre+abstract ('title_abstract').
    Expose aussi le statut du mode hybrid search.
    """
    with engine.connect() as conn:
        total_docs = conn.execute(
            text("SELECT COUNT(*) FROM literature_document")
        ).scalar() or 0

        docs_with_fulltext = conn.execute(text("""
            SELECT COUNT(DISTINCT document_id)
            FROM document_chunk
            WHERE chunk_type = 'fulltext_section'
        """)).scalar() or 0

        # Doublons RÉELS calculés à la lecture (même clé d'identité que la dédup) :
        # le badge reflète la vérité même si aucun script n'a jamais posé is_duplicate.
        # Tous projets confondus, comme total_docs ci-dessus.
        duplicates = _count_corpus_duplicates(conn)

        chunks_with_embedding = conn.execute(text("""
            SELECT COUNT(*) FROM document_chunk WHERE embedding IS NOT NULL
        """)).scalar() or 0

        total_chunks = conn.execute(
            text("SELECT COUNT(*) FROM document_chunk")
        ).scalar() or 0

        # « En attente d'indexation » HONNÊTE : exactement les chunks que le worker VA
        # embedder - types standard, contenu embeddable (> 20 car.), pas encore mis en
        # quarantaine (< 3 échecs). Exclut les types « Autres » (jamais embeddés) et les
        # chunks définitivement refusés par l'API, qui gonflaient artificiellement le
        # reliquat. (On embède désormais le title_abstract de TOUS les docs, full-text
        # compris - plus de « couverts par le texte intégral ».)
        chunks_pending = conn.execute(text("""
            SELECT COUNT(*) FROM document_chunk c
            WHERE c.embedding IS NULL
              AND c.chunk_type IN ('title_abstract', 'fulltext_section')
              AND LENGTH(c.content) > 20
              AND COALESCE(c.embedding_attempts, 0) < 3
        """)).scalar() or 0

        # Répartition des chunks par type : 'fulltext_section' (texte intégral)
        # vs 'title_abstract' (résumé, ~1 par document) vs le reste.
        chunks_by_type = conn.execute(text(
            "SELECT chunk_type, COUNT(*) AS n FROM document_chunk GROUP BY chunk_type"
        )).mappings().all()

        source_coverage = conn.execute(text("""
            SELECT
                d.source,
                COUNT(DISTINCT d.id) AS total,
                COUNT(DISTINCT CASE WHEN c.chunk_type = 'fulltext_section' THEN d.id END) AS with_fulltext
            FROM literature_document d
            LEFT JOIN document_chunk c ON c.document_id = d.id
            GROUP BY d.source
            ORDER BY total DESC
        """)).mappings().all()

        sample_fulltext = conn.execute(text("""
            SELECT DISTINCT d.id, d.title, d.source, d.year, d.url,
                   d.authors, d.doi
            FROM literature_document d
            JOIN document_chunk c ON c.document_id = d.id
            WHERE c.chunk_type = 'fulltext_section'
            ORDER BY d.year DESC NULLS LAST
            LIMIT 100000
        """)).mappings().all()

    openai_key = os.getenv("OPENAI_API_KEY")
    hybrid_active = bool(openai_key) and chunks_with_embedding > 0

    # Couverture full-text agrégée par source CANONIQUE (mêmes regroupements que
    # le tableau de bord global, sinon variantes héritées dupliquées).
    _ft_by_source: dict[str, dict] = {}
    for r in source_coverage:
        k = _canonical_source(r["source"])
        agg = _ft_by_source.setdefault(k, {"source": k, "total": 0, "with_fulltext": 0})
        agg["total"] += int(r["total"] or 0)
        agg["with_fulltext"] += int(r["with_fulltext"] or 0)
    ft_by_source = sorted(_ft_by_source.values(), key=lambda a: a["total"], reverse=True)
    for a in ft_by_source:
        a["abstract_only"] = a["total"] - a["with_fulltext"]
        a["fulltext_pct"] = round(a["with_fulltext"] / a["total"] * 100, 1) if a["total"] else 0

    _ct = {r["chunk_type"]: int(r["n"]) for r in chunks_by_type}
    fulltext_chunks = _ct.get("fulltext_section", 0)
    abstract_chunks = _ct.get("title_abstract", 0)
    other_chunks = max(0, total_chunks - fulltext_chunks - abstract_chunks)

    return {
        "corpus": {
            "total_documents": total_docs,
            "docs_with_fulltext": docs_with_fulltext,
            "docs_abstract_only": total_docs - docs_with_fulltext,
            "fulltext_coverage_pct": round(docs_with_fulltext / total_docs * 100, 1) if total_docs else 0,
            "duplicates": duplicates,
            "unique_documents": max(0, total_docs - duplicates),
        },
        "chunks": {
            "total": total_chunks,
            "fulltext": fulltext_chunks,
            "abstract": abstract_chunks,
            "other": other_chunks,
        },
        "embeddings": {
            "total_chunks": total_chunks,
            "chunks_with_embedding": chunks_with_embedding,
            "chunks_pending": int(chunks_pending),
            "embedding_coverage_pct": round(chunks_with_embedding / total_chunks * 100, 1) if total_chunks else 0,
        },
        "hybrid_search": {
            "active": hybrid_active,
            "openai_key_present": bool(openai_key),
            "embeddings_available": chunks_with_embedding > 0,
            "mode": "hybrid" if hybrid_active else ("lexical_only" if not openai_key else "no_embeddings"),
            "note": (
                "Mode hybride actif (pgvector cosine + BM25)" if hybrid_active
                else "Mode lexical uniquement : clé OpenAI absente ou embeddings non générés"
            ),
        },
        "by_source": ft_by_source,
        "sample_fulltext_docs": [
            {
                "id": r["id"],
                "title": r["title"],
                "source": r["source"],
                "year": r["year"],
                "url": r["url"],
                "authors": r["authors"],
                "doi": r["doi"],
            }
            for r in sample_fulltext
        ],
    }


# ─── Endpoint : maintenance du corpus (purge doublons + normalisation chunks) ─
class CorpusMaintenanceIn(BaseModel):
    dry_run: bool = True


@app.post("/admin/corpus-maintenance")
def corpus_maintenance(
    payload: CorpusMaintenanceIn | None = None,
    _: None = Depends(require_api_key),
) -> dict[str, Any]:
    """Maintenance idempotente et RÉVERSIBLE du corpus (protégée par WRITE_API_KEY).

    Deux opérations, appliquées uniquement si dry_run=False :

      1. Doublons : l'ensemble traité est l'UNION des documents déjà marqués
         `is_duplicate = TRUE` et de ceux que la clé de contenu (DOI › external_id
         normalisé › titre long) désigne comme doublons. Les seconds ne sont PAS
         encore marqués, donc ils comptent aujourd'hui dans les statistiques et les
         listes : les marquer puis les supprimer FERA BAISSER les compteurs affichés
         (c'est le but ; le dry_run donne le nombre exact avant d'agir). Les lignes
         `article_scenarios` correspondantes sont supprimées explicitement (cette
         table n'a pas de FK, sinon orphelins) ; les chunks partent en CASCADE
         (document_chunk.document_id ON DELETE CASCADE).

      2. Chunks « Autres » (type non standard, types hérités ou NULL, jamais embeddés
         par le worker). Ils sont partitionnés en trois ensembles disjoints et les
         TROIS sont traités, aucun n'est laissé tel quel :
           - junk (< 20 car., jamais embeddable) → SUPPRIMÉ ;
           - redundant (≥ 20 car. mais texte déjà contenu dans le chunk
             `title_abstract` du même document) → SUPPRIMÉ, sans perte : le texte
             reste indexé via `title_abstract` ;
           - unique (≥ 20 car., contenu réellement supplémentaire) → RECLASSÉ en
             `'fulltext_section'` pour que le worker l'indexe.
         Après application il ne reste donc plus aucun chunk « Autres ». Le dry_run
         donne les trois compteurs séparément (`junk_to_delete`,
         `redundant_to_delete`, `unique_to_reclassify`) avant toute écriture.

    Sécurité : tout tourne dans UNE transaction (atomique) ; avant chaque
    suppression, les lignes concernées sont copiées dans des tables `_maint_bak_*`
    (restaurables). dry_run=True (défaut) ne fait que COMPTER - aucune écriture.
    """
    dry_run = True if payload is None else bool(payload.dry_run)
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Prédicat SQL des chunks « non standard » (NULL inclus : `NOT IN` seul rate NULL).
    # Ces chunks « Autres » (types hérités : 'full_text', 'title', 'abstract_section'…,
    # ou NULL) ne sont JAMAIS embeddés par le worker (qui ne traite que les deux types
    # standard) → ils restent « en attente » indéfiniment. On les partitionne en trois
    # ensembles DISJOINTS, tous sûrs (aucune perte de contenu, sauvegardés avant action) :
    NONSTD = "(chunk_type IS NULL OR chunk_type NOT IN ('title_abstract','fulltext_section'))"
    # 1) junk : vide / inexploitable (< 20 car., jamais embeddable) → suppression.
    JUNK = f"{NONSTD} AND (content IS NULL OR length(btrim(content)) < 20)"
    # 2) redundant : fragment substantiel dont le TEXTE est déjà contenu dans le chunk
    #    title_abstract du même document (donc déjà indexé et cherchable) → suppression
    #    SÛRE (le contenu reste dans title_abstract, rien ne quitte l'index).
    _IN_TA = ("EXISTS (SELECT 1 FROM document_chunk ta "
              "WHERE ta.document_id = document_chunk.document_id "
              "AND ta.chunk_type = 'title_abstract' "
              "AND position(btrim(document_chunk.content) IN ta.content) > 0)")
    REDUNDANT = f"{NONSTD} AND content IS NOT NULL AND length(btrim(content)) >= 20 AND {_IN_TA}"
    # 3) unique : fragment substantiel dont le texte n'est PAS dans title_abstract (vrai
    #    contenu supplémentaire) → reclassé en 'fulltext_section' pour que le worker
    #    l'indexe (aucune perte). Après application : plus aucun chunk « Autres ».
    UNIQUE = f"{NONSTD} AND content IS NOT NULL AND length(btrim(content)) >= 20 AND NOT {_IN_TA}"

    report: dict[str, Any] = {
        "dry_run": dry_run,
        "duplicates": {},
        "legacy_chunks": {},
        "backups": [],
    }

    with engine.begin() as conn:
        has_ars = bool(conn.execute(
            text("SELECT to_regclass('public.article_scenarios')")
        ).scalar())

        # ── 1. DOUBLONS ────────────────────────────────────────────────────
        # Doublons = détectés à la lecture par la clé de contenu (DOI › external_id
        # normalisé › titre long) UNION ceux DÉJÀ marqués is_duplicate (script manuel /
        # historique) - on élargit le contrat existant, sans le remplacer. L'aperçu
        # (dry_run) COMPTE sans rien écrire ; à l'application on POSE d'abord is_duplicate
        # sur les doublons de contenu (idempotent), puis la sauvegarde + purge existantes
        # (WHERE is_duplicate IS TRUE) opèrent - le tout dans la même transaction atomique.
        dup_docs = conn.execute(text(
            f"SELECT COUNT(*) FROM ({_DUP_ANY_IDS_SQL}) x"
        )).scalar() or 0
        dup_chunks = conn.execute(text(
            f"SELECT COUNT(*) FROM document_chunk "
            f"WHERE document_id IN (SELECT id FROM ({_DUP_ANY_IDS_SQL}) x)"
        )).scalar() or 0
        dup_ars = 0
        if has_ars:
            dup_ars = conn.execute(text(
                f"SELECT COUNT(*) FROM article_scenarios "
                f"WHERE document_id IN (SELECT id FROM ({_DUP_ANY_IDS_SQL}) x)"
            )).scalar() or 0
        report["duplicates"] = {
            "documents": int(dup_docs),
            "chunks_cascade": int(dup_chunks),
            "article_scenarios": int(dup_ars),
        }
        if not dry_run and dup_docs:
            # Pose is_duplicate/canonical_id sur les doublons réels (idempotent) AVANT
            # la sauvegarde/purge, qui restent pilotées par is_duplicate IS TRUE.
            conn.execute(text(_DUP_FLAG_UPDATE_SQL))
            bak_docs, bak_chunks = f"_maint_bak_docs_{ts}", f"_maint_bak_chunks_{ts}"
            conn.execute(text(
                f'CREATE TABLE "{bak_docs}" AS '
                "SELECT * FROM literature_document WHERE is_duplicate IS TRUE"
            ))
            conn.execute(text(
                f'CREATE TABLE "{bak_chunks}" AS SELECT c.* FROM document_chunk c '
                "JOIN literature_document d ON d.id = c.document_id WHERE d.is_duplicate IS TRUE"
            ))
            report["backups"] += [bak_docs, bak_chunks]
            if has_ars:
                bak_ars = f"_maint_bak_ars_{ts}"
                conn.execute(text(
                    f'CREATE TABLE "{bak_ars}" AS SELECT a.* FROM article_scenarios a '
                    "JOIN literature_document d ON d.id = a.document_id WHERE d.is_duplicate IS TRUE"
                ))
                conn.execute(text(
                    "DELETE FROM article_scenarios WHERE document_id IN "
                    "(SELECT id FROM literature_document WHERE is_duplicate IS TRUE)"
                ))
                report["backups"].append(bak_ars)
            deleted = conn.execute(text(
                "DELETE FROM literature_document WHERE is_duplicate IS TRUE"
            )).rowcount  # chunks removed via ON DELETE CASCADE
            report["duplicates"]["deleted_documents"] = int(deleted)

        # ── 2. CHUNKS « AUTRES » (type non standard) ───────────────────────
        breakdown_rows = conn.execute(text(
            "SELECT COALESCE(chunk_type,'(null)') AS t, COUNT(*) AS n, "
            "COUNT(*) FILTER (WHERE embedding IS NOT NULL) AS embedded "
            f"FROM document_chunk WHERE {NONSTD} GROUP BY chunk_type ORDER BY n DESC"
        )).mappings().all()
        junk = conn.execute(text(f"SELECT COUNT(*) FROM document_chunk WHERE {JUNK}")).scalar() or 0
        redundant = conn.execute(text(f"SELECT COUNT(*) FROM document_chunk WHERE {REDUNDANT}")).scalar() or 0
        uniq = conn.execute(text(f"SELECT COUNT(*) FROM document_chunk WHERE {UNIQUE}")).scalar() or 0
        report["legacy_chunks"] = {
            "breakdown": [
                {"chunk_type": r["t"], "count": int(r["n"]), "embedded": int(r["embedded"])}
                for r in breakdown_rows
            ],
            "junk_to_delete": int(junk),
            "redundant_to_delete": int(redundant),      # already covered by title_abstract
            "unique_to_reclassify": int(uniq),          # real content → fulltext_section (indexed)
        }
        if not dry_run:
            # Delete junk + redundant (each backed up first); order is irrelevant - the
            # three sets are disjoint and none touches title_abstract chunks.
            for label, pred in (("junkchunks", JUNK), ("redundantchunks", REDUNDANT)):
                n = conn.execute(text(f"SELECT COUNT(*) FROM document_chunk WHERE {pred}")).scalar() or 0
                if n:
                    bak = f"_maint_bak_{label}_{ts}"
                    conn.execute(text(f'CREATE TABLE "{bak}" AS SELECT * FROM document_chunk WHERE {pred}'))
                    conn.execute(text(f"DELETE FROM document_chunk WHERE {pred}"))
                    report["backups"].append(bak)
            report["legacy_chunks"]["deleted_junk"] = int(junk)
            report["legacy_chunks"]["deleted_redundant"] = int(redundant)
            reclassified = conn.execute(text(
                f"UPDATE document_chunk SET chunk_type='fulltext_section' WHERE {UNIQUE}"
            )).rowcount
            report["legacy_chunks"]["reclassified"] = int(reclassified)

    return report


@app.post("/admin/embed-pending")
def embed_pending_chunks(limit: int = 200, _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Force l'indexation (embedding) des chunks « en attente » - à la demande, avec
    EXACTEMENT le sélecteur du worker d'arrière-plan. Vide immédiatement le petit
    reliquat sans attendre le cycle de 30 s. Traite au plus `limit` chunks (synchrone) ;
    renvoie le nombre embeddé et le reliquat restant."""
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise HTTPException(status_code=503, detail="Clé OpenAI non configurée")
    if _openai_in_cooldown():
        return {"embedded": 0, "remaining": None, "cooldown": True}

    # Sélecteur IDENTIQUE au worker (schema_boot) et au compteur « en attente » :
    # types standard, contenu embeddable (> 20 car.), quarantaine après 3 échecs.
    # Le title_abstract est embeddé pour TOUS les documents, texte intégral compris
    # (l'ancienne exception « couvert par ses sections » a été retirée).
    ELIGIBLE = (
        "c.embedding IS NULL "
        "AND c.chunk_type IN ('title_abstract','fulltext_section') "
        "AND LENGTH(c.content) > 20 "
        "AND COALESCE(c.embedding_attempts, 0) < 3"
    )

    from llm_usage import MeteredOpenAI as _OAI
    client = _OAI(api_key=openai_key, timeout=90.0)

    with engine.connect() as conn:
        rows = conn.execute(text(
            f"SELECT c.id, c.content FROM document_chunk c WHERE {ELIGIBLE} "
            "ORDER BY c.chunk_type DESC, c.id LIMIT :lim"
        ), {"lim": max(1, min(int(limit), 2000))}).mappings().fetchall()

    # Chemin résilient IDENTIQUE au worker : troncature par tokens + repli chunk par
    # chunk sur un lot empoisonné (ne se bloque plus sur un contenu que l'API refuse).
    embedded, failed, error = 0, [], None
    try:
        embedded, failed = _embed_chunks_resilient(client, list(rows))
    except Exception as e:                        # quota → cooldown, on ne réessaie pas ici
        _trip_openai_cooldown()
        error = str(e)
    if failed:
        with engine.begin() as cu:
            cu.execute(
                text("UPDATE document_chunk SET embedding_attempts = "
                     "COALESCE(embedding_attempts,0)+1 WHERE id IN :ids")
                .bindparams(bindparam("ids", expanding=True)),
                {"ids": failed},
            )

    with engine.connect() as conn:
        remaining = conn.execute(text(
            f"SELECT COUNT(*) FROM document_chunk c WHERE {ELIGIBLE}"
        )).scalar() or 0

    out: dict[str, Any] = {"embedded": int(embedded), "remaining": int(remaining)}
    if failed:
        out["quarantined"] = len(failed)
    if error:
        out["error"] = error
    return out


@app.get("/gesica/deduplication/status")
def get_deduplication_status() -> dict[str, Any]:
    """
    Retourne le statut de la déduplication du corpus.
    """
    with engine.connect() as conn:
        stats = conn.execute(text("""
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN title_hash IS NOT NULL THEN 1 ELSE 0 END) AS with_title_hash,
                SUM(CASE WHEN quality_score > 0 THEN 1 ELSE 0 END) AS with_quality_score
            FROM literature_document
            WHERE project_context = 'literev'
        """)).mappings().first()
        # Doublons RÉELS à la lecture (même clé que la dédup), pas le flag is_duplicate
        # qu'aucun runtime ne pose : le statut reflète la réalité en continu.
        duplicates = _count_corpus_duplicates(conn, "literev")
    total = int(stats["total"] or 0)
    canonical = max(0, total - duplicates)
    return {
        "total_documents": total,
        "canonical_documents": canonical,
        "duplicate_documents": duplicates,
        "with_title_hash": int(stats["with_title_hash"] or 0),
        "with_quality_score": int(stats["with_quality_score"] or 0),
        "deduplication_rate": round(duplicates / max(total, 1) * 100, 1),
        "instructions": {
            "dry_run": "python3 scripts/deduplicate_corpus.py --dry-run",
            "execute": "python3 scripts/deduplicate_corpus.py --execute",
            "execute_delete": "python3 scripts/deduplicate_corpus.py --execute --delete",
        },
    }
