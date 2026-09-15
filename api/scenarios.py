"""User scenarios: table, CRUD, folders, detail, corpus, jobs, counts, activity.

Extracted from main.py (LiteRev API); `main` re-exports everything for the scripts,
tools and tests.
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any

from fastapi import Depends, HTTPException, Query
from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy import text

from .core import _msg, app, engine, logger, require_api_key
from .documents import _strategy_is_degraded
from .scenario_store import _get_scenario_threshold, _get_user_scenario_or_404
from .schema_boot import _exec_ddl_isolated
from .search import (
    LIVE_MAX_PER_SOURCE,
    _combined_query_text,
    _facet_ops,
    _generate_search_strategy,
    _load_prisma_identification,
    _normalize_sub_queries,
)
from .alerts import _clean_email, _ensure_alert_subscription

# ─────────────────────────────────────────────────────────────────────────────
# USER SCENARIOS : Recherches sauvegardées persistées en base
# ─────────────────────────────────────────────────────────────────────────────
# Chaque recherche sauvegardée devient un vrai scénario utilisateur avec :
#   - son propre corpus (articles ingérés via PubMed)
#   - tous les onglets du ScenarioDetailPage (corpus, PICO, screening, RAG, etc.)
#   - un ID de la forme "usr-<uuid4_court>"
# ─────────────────────────────────────────────────────────────────────────────

# Schéma de `article_scenarios` — le lien N-N document ↔ scénario.
#
# Cette table est interrogée PARTOUT dans main.py (≈100 références) mais AUCUN fichier du
# dépôt ne la créait : ni schema.sql, ni les migrations Alembic (qui se contentent de lui
# AJOUTER des colonnes et se sautent elles-mêmes si la table est absente), ni le DDL de
# démarrage (qui l'ALTER directement). Elle n'existait qu'en production, posée
# historiquement par un script ad hoc. Conséquence : sur une base neuve, l'ALTER échouait
# et faisait annuler la création de `user_scenarios` (cf. _exec_ddl_isolated).
#
# Les colonnes sont reconstruites à partir de l'usage RÉEL :
#   • scenario_id / document_id / similarity_score : clé et score de rattachement
#     (confirmés par les deux fixtures de tests d'intégration, qui déclarent explicitement
#     refléter la production, et par les INSERT/UPDATE de main.py) ;
#   • cluster_id / cluster_label / rerank_score : ajoutées par le DDL de démarrage ;
#   • screening_* : migration c8d4e2f1a9b3 ; reviewer_*/kappa_* : migration e2f6a8b3c5d7.
# Les types reprennent ceux de ces migrations, à l'identique.
_ARTICLE_SCENARIOS_DDL = """
    CREATE TABLE IF NOT EXISTS article_scenarios (
        scenario_id        TEXT   NOT NULL,
        document_id        BIGINT NOT NULL,
        similarity_score   DOUBLE PRECISION,
        rerank_score       FLOAT,
        cluster_id         INTEGER,
        cluster_label      TEXT,
        screening_status   TEXT,
        screening_reason   TEXT,
        screening_notes    TEXT,
        screened_at        TIMESTAMP,
        reviewer_1_status  VARCHAR(20),
        reviewer_1_reason  TEXT,
        reviewer_2_status  VARCHAR(20),
        reviewer_2_reason  TEXT,
        kappa_resolved     BOOLEAN DEFAULT FALSE,
        kappa_final_status VARCHAR(20),
        assigned_at        TIMESTAMP,
        PRIMARY KEY (scenario_id, document_id)
    )
"""


def _ensure_user_scenarios_table() -> None:
    """Crée la table user_scenarios et user_scenario_folders si elles n'existent pas."""
    # `article_scenarios` d'ABORD, et hors du bloc transactionnel ci-dessous : les ALTER
    # plus bas la visent, et sur une base neuve son absence annulait tout le reste.
    _exec_ddl_isolated([_ARTICLE_SCENARIOS_DDL], "_ensure_user_scenarios_table")
    with engine.begin() as conn:
        # Table des dossiers
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS user_scenario_folders (
                id          VARCHAR(40)  PRIMARY KEY,
                name        VARCHAR(255) NOT NULL,
                color       VARCHAR(20)  DEFAULT '#6366f1',
                sort_order  INTEGER      DEFAULT 0,
                created_at  TIMESTAMP    DEFAULT NOW()
            )
        """))
        # Table des scénarios (avec folder_id optionnel)
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS user_scenarios (
                id          VARCHAR(40)  PRIMARY KEY,
                name        VARCHAR(255) NOT NULL,
                query       TEXT         NOT NULL,
                mode        VARCHAR(20)  NOT NULL DEFAULT 'hybrid',
                filters     JSONB        NOT NULL DEFAULT '{}',
                result_count INTEGER     DEFAULT 0,
                pinned      BOOLEAN      DEFAULT FALSE,
                folder_id   VARCHAR(40)  REFERENCES user_scenario_folders(id) ON DELETE SET NULL,
                created_at  TIMESTAMP    DEFAULT NOW(),
                updated_at  TIMESTAMP    DEFAULT NOW()
            )
        """))
        # Ajouter folder_id si la table existait déjà sans cette colonne
        conn.execute(text("""
            ALTER TABLE user_scenarios ADD COLUMN IF NOT EXISTS folder_id VARCHAR(40)
            REFERENCES user_scenario_folders(id) ON DELETE SET NULL
        """))
        # Colonnes du pipeline d'ingestion/populate (sur tables préexistantes).
        # NB : exécutées hors de cette transaction, une par une (cf. _exec_ddl_isolated),
        # pour qu'un ALTER en échec n'annule pas les CREATE TABLE ci-dessus.
        _pipeline_ddl = (
            "ALTER TABLE user_scenarios ADD COLUMN IF NOT EXISTS populate_status VARCHAR(20)",
            "ALTER TABLE user_scenarios ADD COLUMN IF NOT EXISTS pipeline_status VARCHAR(20)",
            "ALTER TABLE user_scenarios ADD COLUMN IF NOT EXISTS pipeline_step VARCHAR(80)",
            "ALTER TABLE user_scenarios ADD COLUMN IF NOT EXISTS pipeline_progress INTEGER DEFAULT 0",
            "ALTER TABLE user_scenarios ADD COLUMN IF NOT EXISTS pipeline_started_at TIMESTAMP",
            "ALTER TABLE user_scenarios ADD COLUMN IF NOT EXISTS article_count INTEGER DEFAULT 0",
            "ALTER TABLE user_scenarios ADD COLUMN IF NOT EXISTS search_strategy JSONB",
            # Chiffres PRISMA « identification » du dernier populate / rebuild (enregistrements
            # par source, doublons, uniques) — cf. _prisma_identification_figures.
            "ALTER TABLE user_scenarios ADD COLUMN IF NOT EXISTS prisma_identification JSONB",
            # Clustering persisté en base (sinon perdu au redémarrage du serveur)
            "ALTER TABLE article_scenarios ADD COLUMN IF NOT EXISTS cluster_id INTEGER",
            "ALTER TABLE article_scenarios ADD COLUMN IF NOT EXISTS cluster_label TEXT",
            # Score d'un cross-encoder (rerank Cohere) — précision supérieure au
            # cosinus pour ORDONNER le sous-ensemble pertinent (sélection = cosinus
            # >= seuil ; ordre = rerank_score quand présent).
            "ALTER TABLE article_scenarios ADD COLUMN IF NOT EXISTS rerank_score FLOAT",
            # Borne anti « lot empoisonné » : nb d'échecs d'embedding pour un chunk.
            # Au-delà de 3, il est exclu du worker / du compteur « en attente » / de
            # /admin/embed-pending — sinon un chunk que l'API refuse (contenu invalide)
            # resterait « en attente » à l'infini et serait ré-essayé chaque cycle.
            "ALTER TABLE document_chunk ADD COLUMN IF NOT EXISTS embedding_attempts INTEGER DEFAULT 0",
            # Recherche multi-sous-requêtes : liste [{"kind":"boolean"|"natural",
            # "text":...}] + combinateur ("union" = OU, "intersection" = ET) entre
            # leurs ensembles de résultats. NULL = recherche mono-requête classique
            # (colonnes query/mode) → comportement inchangé.
            "ALTER TABLE user_scenarios ADD COLUMN IF NOT EXISTS sub_queries JSONB",
            "ALTER TABLE user_scenarios ADD COLUMN IF NOT EXISTS combinator VARCHAR(12)",
            # Propriétaire (email) : living review + alertes email par utilisateur.
            "ALTER TABLE user_scenarios ADD COLUMN IF NOT EXISTS owner_email VARCHAR(255)",
            "CREATE INDEX IF NOT EXISTS ix_user_scenarios_owner ON user_scenarios (owner_email)",
            # Scénarios GESICA : _list_db_gesica_scenarios (main.py:3720-3727) filtre sur
            # is_system / hidden et trie sur title, SANS qualificatif de table. Ces
            # colonnes n'étaient créées nulle part : sur une base neuve
            # GET /gesica/scenarios renvoyait 500 et l'interface affichait
            # « Failed to load scenarios ».
            "ALTER TABLE user_scenarios ADD COLUMN IF NOT EXISTS is_system BOOLEAN DEFAULT FALSE",
            "ALTER TABLE user_scenarios ADD COLUMN IF NOT EXISTS hidden BOOLEAN DEFAULT FALSE",
            "ALTER TABLE user_scenarios ADD COLUMN IF NOT EXISTS title TEXT",
        )
        # Cache PERSISTANT des stratégies booléennes : même requête (clé normalisée) →
        # MÊME booléen, de façon déterministe, à travers les requêtes, les redémarrages
        # et les workers. Élimine la divergence « Main 57 vs sous-requête 56 » (deux
        # traductions LLM concurrentes de la même phrase). Voir _generate_search_strategy.
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS search_strategy_cache (
                query_key   TEXT       PRIMARY KEY,
                strategy    JSONB      NOT NULL,
                created_at  TIMESTAMP  DEFAULT NOW()
            )
        """))
    # Les ALTER/INDEX en dernier, isolés : ils portent sur des tables qui peuvent ne pas
    # exister encore sur une base neuve, et leur échec ne doit rien annuler.
    _failed = _exec_ddl_isolated(_pipeline_ddl, "_ensure_user_scenarios_table")
    if _failed:
        logger.warning(f"_ensure_user_scenarios_table: {len(_failed)} DDL ignorée(s) — "
                       f"la base est peut-être incomplète : {_failed[:3]}")
    logger.info("Tables user_scenarios et user_scenario_folders vérifiées/créées.")

try:
    _ensure_user_scenarios_table()
except Exception as _e:
    logger.warning(f"_ensure_user_scenarios_table: {_e}")


def _truncate_display_name(v: Any, limit: int = 255) -> Any:
    """Tronque un libellé d'affichage à `limit` caractères (colonne VARCHAR(255)).

    `name` est un LIBELLÉ ; la requête complète vit dans `query` (TEXT). Une requête
    booléenne un peu longue dépasse 255 caractères → sans ce garde-fou, Pydantic
    renvoie 422 (max_length) ou la colonne VARCHAR(255) déborde (500). On tronque
    proprement avec « … » pour que la recherche aboutisse quel que soit le client."""
    if isinstance(v, str):
        v = v.strip()
        if len(v) > limit:
            v = v[: limit - 1].rstrip() + "…"
    return v


class UserScenarioIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    query: str = Field(..., min_length=1)
    mode: str = Field(default="hybrid")
    filters: dict[str, Any] = Field(default_factory=dict)
    result_count: int = Field(default=0, ge=0)
    pinned: bool = Field(default=False)
    folder_id: str | None = None
    # Stratégie booléenne (générée par LLM) déjà calculée côté recherche. Si
    # fournie, on la persiste telle quelle pour que le corpus utilise EXACTEMENT
    # la même requête booléenne que celle affichée/comptée à la recherche.
    search_strategy: dict[str, Any] | None = None
    # Recherche MULTI-sous-requêtes : liste [{"kind":"boolean"|"natural","text":...}]
    # combinée par `combinator` ("union"=OU par défaut, "intersection"=ET explicite).
    # < 2 entrées valides → None (on retombe sur la recherche mono-requête query/mode).
    sub_queries: list[dict[str, Any]] | None = None
    combinator: str = Field(default="union")
    # Propriétaire (email) : relie le scénario à un utilisateur pour la living review
    # et les alertes email. Optionnel (les scénarios restent utilisables sans compte).
    owner_email: str | None = None

    @field_validator("name", mode="before")
    @classmethod
    def _cap_name(cls, v: Any) -> Any:
        return _truncate_display_name(v)

    @model_validator(mode="after")
    def _clean_sub_queries(self) -> "UserScenarioIn":
        cleaned = _normalize_sub_queries(self.sub_queries)
        self.sub_queries = cleaned if len(cleaned) >= 2 else None
        if self.combinator not in ("union", "intersection"):
            self.combinator = "union"
        self.owner_email = _clean_email(self.owner_email)
        return self


class UserScenarioPatch(BaseModel):
    name: str | None = None
    pinned: bool | None = None
    mode: str | None = None
    filters: dict[str, Any] | None = None
    folder_id: str | None = None  # Assigner à un dossier (None = hors dossier)

    @field_validator("name", mode="before")
    @classmethod
    def _cap_name(cls, v: Any) -> Any:
        return _truncate_display_name(v)


class FolderIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    color: str = Field(default='#6366f1')
    sort_order: int = Field(default=0)


def _user_scenario_to_gesica_format(
    row: dict[str, Any], counts_map: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Convertit une ligne user_scenarios au format GesicaScenario (liste).

    Si counts_map est fourni (chemin liste), on y lit les compteurs déjà agrégés
    au lieu d'exécuter la requête d'agrégation une fois PAR scénario (évite le N+1).
    Les appels unitaires (création / lecture d'un seul scénario) le laissent à None
    et interrogent la base normalement."""
    if counts_map is not None:
        counts = counts_map.get(str(row["id"]))
    else:
        with engine.connect() as conn:
            counts = conn.execute(text("""
                SELECT
                    COUNT(DISTINCT ars.document_id) AS article_count,
                    COUNT(DISTINCT ars.document_id) FILTER (
                        WHERE COALESCE(ars.screening_status, d.screening_status) = 'included'
                    ) AS included_count,
                    COUNT(DISTINCT ars.document_id) FILTER (
                        WHERE COALESCE(ars.screening_status, d.screening_status) = 'excluded'
                    ) AS excluded_count
                FROM article_scenarios ars
                JOIN literature_document d ON d.id = ars.document_id
                WHERE ars.scenario_id = :sid
                  AND (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)
            """), {"sid": row["id"]}).mappings().first()

    article_count = int(counts["article_count"] or 0) if counts else 0
    included = int(counts["included_count"] or 0) if counts else 0
    excluded = int(counts["excluded_count"] or 0) if counts else 0

    # Actions recommandées (cache) + résumé du modèle entraîné, pour la carte
    # généralisée du tableau de bord (mêmes blocs que les scénarios GESICA).
    actions: list = []
    model_summary: dict[str, Any] = {"has_model": False}
    try:
        with engine.connect() as _c2:
            _a = _c2.execute(text(
                "SELECT recommended_actions_json FROM scenario_settings WHERE scenario_id = :sid"
            ), {"sid": row["id"]}).scalar()
            if isinstance(_a, list):
                actions = _a
            _m = _c2.execute(text("""
                SELECT family, metric, metrics_json FROM scenario_model_run
                WHERE scenario_id = :sid AND is_active = TRUE
                ORDER BY created_at DESC LIMIT 1
            """), {"sid": row["id"]}).mappings().first()
            if _m:
                mj = _m["metrics_json"] or {}
                mv = mj.get(_m["metric"]) if _m["metric"] else (list(mj.values())[0] if mj else None)
                model_summary = {
                    "has_model": True, "family": _m["family"], "metric": _m["metric"],
                    "metric_value": (float(mv) if isinstance(mv, (int, float)) else None),
                }
    except Exception as _e_card:
        logger.warning(f"Card extras {row['id']}: {_e_card}")

    # Recherche multi-facettes : l'expression COMPLÈTE (« (A) AND (B) ») est ce que
    # la carte et l'indicateur d'activité doivent montrer ; `query` reste la facette
    # principale (identité du scénario, stratégie booléenne live).
    _sub_clean = _normalize_sub_queries(row.get("sub_queries"))
    _combined = _combined_query_text(row["query"], _sub_clean, row.get("combinator"))
    return {
        "id": row["id"],
        "name": row["name"],
        "title": row["name"],
        "description": f"Recherche sauvegardée : {_combined}",
        "cluster": "user",
        "article_count": article_count,
        "included_count": included,
        "excluded_count": excluded,
        "kappa_score": None,
        "hidden": False,
        "recommended_actions": actions,
        "model": model_summary,
        "relevant_articles": [],
        "living_evidence_note": (
            f"Living Evidence Review · {article_count} articles indexés. Mis à jour automatiquement à chaque ingestion."
            if article_count > 0
            else "Aucun article indexé. Lancez l'ingestion multi-sources pour construire le corpus."
        ),
        "pinned": bool(row.get("pinned", False)),
        "query": row["query"],
        "combined_query": _combined,
        "sub_queries": _sub_clean if len(_sub_clean) >= 2 else None,
        "combinator": (row.get("combinator") if len(_sub_clean) >= 2 else None),
        "mode": row["mode"],
        "filters": row.get("filters") or {},
        "result_count": row.get("result_count", 0),
        "folder_id": row.get("folder_id"),
        "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
        "updated_at": row["updated_at"].isoformat() if row.get("updated_at") else None,
        "is_user_scenario": True,
        "populate_status": row.get("populate_status", "idle"),
        "pipeline_status": row.get("pipeline_status", "idle"),
        "pipeline_step": row.get("pipeline_step"),
        "pipeline_progress": row.get("pipeline_progress", 0),
    }


# ── CRUD ──────────────────────────────────────────────────────────────────────

@app.get("/user-scenarios")
def list_user_scenarios() -> list[dict[str, Any]]:
    """Liste tous les scénarios utilisateur (recherches sauvegardées).
    Déduplique au passage les recherches récentes (non épinglées) par query+mode
    en ne conservant que la plus récente de chaque groupe."""
    with engine.begin() as conn:
        # Delete stale duplicates: for unpinned/unfoldered scenarios keep only
        # the most recent row per (query, mode) pair.
        # Identité COMPLÈTE d'une recherche = query + mode + sous-requêtes + combinateur :
        # « A » et « (A) AND (B) » partagent la même `query` (facette principale) mais
        # sont deux recherches distinctes — l'une ne doit pas purger l'autre.
        conn.execute(text("""
            DELETE FROM user_scenarios
            WHERE pinned = false AND folder_id IS NULL
              AND id NOT IN (
                SELECT DISTINCT ON (query, mode, sub_queries, combinator) id
                FROM user_scenarios
                WHERE pinned = false AND folder_id IS NULL
                ORDER BY query, mode, sub_queries, combinator, created_at DESC
              )
        """))
        # Un scénario SAUVEGARDÉ (épinglé) est unique : purge toute recherche récente
        # (non épinglée) qui DOUBLONNE un scénario épinglé de même query+mode. Sans ça,
        # relancer une recherche déjà sauvegardée laissait une 2e carte identique.
        conn.execute(text("""
            DELETE FROM user_scenarios u
            WHERE u.pinned = false AND u.folder_id IS NULL
              AND EXISTS (
                SELECT 1 FROM user_scenarios p
                WHERE p.pinned = true AND p.query = u.query AND p.mode = u.mode
                  AND p.sub_queries IS NOT DISTINCT FROM u.sub_queries
                  AND COALESCE(p.combinator, '') = COALESCE(u.combinator, '')
              )
        """))
        rows = conn.execute(text("""
            SELECT
                us.id, us.name, us.query, us.mode, us.filters,
                us.pinned, us.folder_id, us.created_at, us.updated_at,
                us.populate_status, us.pipeline_status, us.pipeline_step, us.pipeline_progress,
                COALESCE(us.result_count, 0) AS result_count,
                COALESCE(us.article_count, 0) AS article_count,
                us.is_system, us.sub_queries, us.combinator
            FROM user_scenarios us
            ORDER BY us.pinned DESC, us.created_at DESC
        """)).mappings().all()

        # Compteurs (articles / inclus / exclus) de TOUS les scénarios en UNE
        # requête, puis lookup par scénario → évite une requête d'agrégation par
        # ligne (N+1). Même forme que sql_counts dans /gesica/scenarios.
        counts_map: dict[str, Any] = {
            str(cr["scenario_id"]): dict(cr)
            for cr in conn.execute(text("""
                SELECT ars.scenario_id,
                       COUNT(DISTINCT ars.document_id) AS article_count,
                       COUNT(DISTINCT ars.document_id) FILTER (
                           WHERE COALESCE(ars.screening_status, d.screening_status) = 'included'
                       ) AS included_count,
                       COUNT(DISTINCT ars.document_id) FILTER (
                           WHERE COALESCE(ars.screening_status, d.screening_status) = 'excluded'
                       ) AS excluded_count
                FROM article_scenarios ars
                JOIN literature_document d ON d.id = ars.document_id
                WHERE (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)
                GROUP BY ars.scenario_id
            """)).mappings().all()
        }
    return [_user_scenario_to_gesica_format(dict(r), counts_map) for r in rows]


@app.post("/user-scenarios", status_code=201)
def create_user_scenario(payload: UserScenarioIn, _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Crée ou met à jour un scénario utilisateur depuis une recherche sauvegardée.
    Pour les recherches récentes (non épinglées, sans dossier), upsert par query+mode
    afin d'éviter l'accumulation de doublons lors des relances de recherche."""
    import uuid
    # For unpinned auto-saved searches: upsert by query+mode to avoid duplicates.
    # Skip the upsert for multi-sub-query searches: they share the synthesized
    # display `query` yet are distinct searches, so query+mode dedup would wrongly
    # merge them — always insert a fresh row instead.
    if not payload.pinned and not payload.folder_id and not payload.sub_queries:
        with engine.begin() as conn:
            # Un scénario SAUVEGARDÉ (épinglé) est UNIQUE : si CE query l'est déjà, une
            # relance de recherche ne doit PAS créer un doublon « récent ». On renvoie le
            # scénario épinglé existant tel quel (corpus + Variables/Modèle préservés) —
            # la recherche reste visible dans la vue de résultats, sans seconde carte.
            pinned_existing = conn.execute(text("""
                SELECT id FROM user_scenarios
                WHERE query = :query AND mode = :mode AND pinned = true AND folder_id IS NULL
                ORDER BY created_at DESC LIMIT 1
            """), {"query": payload.query, "mode": payload.mode}).scalar()
            if pinned_existing:
                return _user_scenario_to_gesica_format(_get_user_scenario_or_404(pinned_existing))
            existing = conn.execute(text("""
                SELECT id FROM user_scenarios
                WHERE query = :query AND mode = :mode AND pinned = false AND folder_id IS NULL
                ORDER BY created_at DESC LIMIT 1
            """), {"query": payload.query, "mode": payload.mode}).scalar()
            if existing:
                conn.execute(text("""
                    UPDATE user_scenarios
                    SET name = :name, filters = CAST(:filters AS jsonb),
                        result_count = :result_count, created_at = now(),
                        search_strategy = COALESCE(CAST(:strategy AS jsonb), search_strategy)
                    WHERE id = :id
                """), {
                    "id": existing,
                    "name": payload.name,
                    "filters": json.dumps(payload.filters),
                    "result_count": payload.result_count,
                    "strategy": json.dumps(payload.search_strategy) if payload.search_strategy else None,
                })
                row = _get_user_scenario_or_404(existing)
                return _user_scenario_to_gesica_format(row)
    # « Enregistrer comme scénario » (épinglage) : ne PAS créer un doublon vide qui
    # relancerait tout le populate (→ un scénario épinglé « 0 article, ingestion… »
    # EN PLUS de la recherche récente déjà peuplée). On PROMEUT plutôt la recherche
    # récente correspondante — même identité COMPLÈTE (query+mode+sous-requêtes+
    # combinateur) — en l'épinglant, ce qui conserve son corpus déjà construit.
    # Repli sur un INSERT si aucune récente ne correspond (scénario réellement neuf).
    if payload.pinned and not payload.folder_id:
        with engine.begin() as conn:
            existing = conn.execute(text("""
                SELECT id FROM user_scenarios
                WHERE query = :query AND mode = :mode AND pinned = false AND folder_id IS NULL
                  AND sub_queries IS NOT DISTINCT FROM CAST(:sub_queries AS jsonb)
                  AND COALESCE(combinator, '') = COALESCE(:combinator, '')
                ORDER BY created_at DESC LIMIT 1
            """), {"query": payload.query, "mode": payload.mode,
                   "sub_queries": json.dumps(payload.sub_queries) if payload.sub_queries else None,
                   "combinator": payload.combinator}).scalar()
            if existing:
                conn.execute(text("""
                    UPDATE user_scenarios
                    SET pinned = true, name = :name, filters = CAST(:filters AS jsonb),
                        result_count = :result_count, updated_at = now(),
                        search_strategy = COALESCE(CAST(:strategy AS jsonb), search_strategy)
                    WHERE id = :id
                """), {
                    "id": existing,
                    "name": payload.name,
                    "filters": json.dumps(payload.filters),
                    "result_count": payload.result_count,
                    "strategy": json.dumps(payload.search_strategy) if payload.search_strategy else None,
                })
                row = _get_user_scenario_or_404(existing)
                # Scénario SAUVEGARDÉ (épinglé) → tout se calcule côté serveur : on
                # déclenche le pipeline COMPLET d'enrichissement (best-effort, dédupliqué
                # par le verrou). Le front peut aussi l'appeler — le garde empêche le double.
                try:
                    _launch_full_pipeline(existing)
                except Exception as _e:
                    logger.warning(f"auto full-pipeline on pin {existing}: {_e}")
                return _user_scenario_to_gesica_format(row)
    new_id = "usr-" + str(uuid.uuid4()).replace("-", "")[:12]
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO user_scenarios (id, name, query, mode, filters, result_count, pinned, folder_id, search_strategy, sub_queries, combinator, owner_email)
            VALUES (:id, :name, :query, :mode, CAST(:filters AS jsonb), :result_count, :pinned, :folder_id, CAST(:strategy AS jsonb), CAST(:sub_queries AS jsonb), :combinator, :owner_email)
        """), {
            "id": new_id,
            "name": payload.name,
            "query": payload.query,
            "mode": payload.mode,
            "filters": json.dumps(payload.filters),
            "result_count": payload.result_count,
            "pinned": payload.pinned,
            "folder_id": payload.folder_id,
            "strategy": json.dumps(payload.search_strategy) if payload.search_strategy else None,
            "sub_queries": json.dumps(payload.sub_queries) if payload.sub_queries else None,
            "combinator": payload.combinator if payload.sub_queries else None,
            "owner_email": payload.owner_email,
        })
        # Propriétaire renseigné → l'abonner aux alertes de SON scénario (living review).
        if payload.owner_email:
            try:
                _ensure_alert_subscription(conn, payload.owner_email, new_id)
            except Exception as _e:
                logger.warning(f"auto-subscribe owner {new_id}: {_e}")
    # Génération de la stratégie de recherche en arrière-plan : l'appel OpenAI
    # ne doit JAMAIS bloquer (ni faire échouer) la création du scénario. On la
    # saute si le client a déjà fourni la stratégie (recherche booléenne).
    def _bg_strategy(sid: str, q: str) -> None:
        try:
            strategy = _generate_search_strategy(q)
            if _strategy_is_degraded(strategy, q):
                # Repli dégradé (panne LLM/quota) : ne PAS persister, sera
                # régénéré à la prochaine lecture une fois le quota rétabli.
                return
            with engine.begin() as conn2:
                conn2.execute(text("""
                    UPDATE user_scenarios SET search_strategy = CAST(:strategy AS jsonb) WHERE id = :id
                """), {"id": sid, "strategy": json.dumps(strategy)})
        except Exception as _se:
            logger.warning(f"search_strategy generation failed for {sid}: {_se}")
    if not payload.search_strategy:
        try:
            import threading as _threading
            _threading.Thread(target=_bg_strategy, args=(new_id, payload.query), daemon=True).start()
        except Exception as _te:
            logger.warning(f"could not start strategy thread for {new_id}: {_te}")
    # Nouveau scénario SAUVEGARDÉ (épinglé) → enrichissement complet côté serveur.
    if payload.pinned:
        try:
            _launch_full_pipeline(new_id)
        except Exception as _e:
            logger.warning(f"auto full-pipeline on new pin {new_id}: {_e}")
    row = _get_user_scenario_or_404(new_id)
    return _user_scenario_to_gesica_format(row)


class ScenarioOwnerIn(BaseModel):
    email: str = Field(..., max_length=255)
    frequency: str = Field(default="weekly")


@app.post("/user-scenarios/{scenario_id}/owner")
def set_user_scenario_owner(scenario_id: str, payload: ScenarioOwnerIn,
                            _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Attribue un propriétaire (email) à un scénario existant et l'abonne aux alertes
    de living review pour ce scénario. Permet à un utilisateur de « couvrir » ses
    propres scénarios (notification des nouveaux articles à son email)."""
    email = _clean_email(payload.email)
    if not email:
        raise HTTPException(status_code=400, detail="Adresse email invalide.")
    freq = payload.frequency if payload.frequency in ("daily", "weekly", "immediate") else "weekly"
    _get_user_scenario_or_404(scenario_id)
    with engine.begin() as conn:
        conn.execute(text("UPDATE user_scenarios SET owner_email = :e, updated_at = NOW() WHERE id = :id"),
                     {"e": email, "id": scenario_id})
        _ensure_alert_subscription(conn, email, scenario_id, freq)
    return {"status": "ok", "scenario_id": scenario_id, "owner_email": email, "frequency": freq,
            "subscribed": True}


@app.get("/user-scenarios/by-owner")
def list_user_scenarios_by_owner(email: str, _: None = Depends(require_api_key)) -> list[dict[str, Any]]:
    """Liste les scénarios appartenant à un email (« mes scénarios »).

    Protégé par clé API : à la différence du listing GLOBAL (public, sans email),
    by-owner permet d'ÉNUMÉRER les scénarios d'un email arbitraire — une surface de
    vie privée qu'on ne laisse pas ouverte. Aucun impact UI : le front n'appelle pas
    cet endpoint (les endpoints de qualité de réponse — RAG/recherche — restent, eux,
    publics)."""
    e = _clean_email(email)
    if not e:
        return []
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT id, name, query, created_at, article_count FROM user_scenarios "
            "WHERE owner_email = :e ORDER BY created_at DESC"
        ), {"e": e}).mappings().all()
    return [dict(r) for r in rows]


@app.delete("/user-scenarios/{scenario_id}", status_code=200)
def delete_user_scenario(scenario_id: str, _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Supprime un scénario (utilisateur OU GESICA) et ses associations."""
    _get_user_scenario_or_404(scenario_id)
    # Les scénarios GESICA (is_system) sont désormais des scénarios ordinaires :
    # supprimables comme les autres (généralisation). On nettoie aussi les tables
    # liées (datasets/runs de modèle) pour ne pas laisser d'orphelins.
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM article_scenarios WHERE scenario_id = :sid"), {"sid": scenario_id})
        conn.execute(text("DELETE FROM scenario_settings WHERE scenario_id = :sid"), {"sid": scenario_id})
        for _t in ("scenario_model_dataset", "scenario_model_run"):
            try:
                conn.execute(text(f"DELETE FROM {_t} WHERE scenario_id = :sid"), {"sid": scenario_id})
            except Exception:
                pass
        conn.execute(text("DELETE FROM user_scenarios WHERE id = :id"), {"id": scenario_id})
    return {"deleted": True, "id": scenario_id}


@app.patch("/user-scenarios/{scenario_id}")
def patch_user_scenario(scenario_id: str, payload: UserScenarioPatch, _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Met à jour le nom, le pin, le mode ou les filtres d'un scénario utilisateur."""
    _get_user_scenario_or_404(scenario_id)
    updates = []
    params: dict[str, Any] = {"id": scenario_id}
    if payload.name is not None:
        updates.append("name = :name")
        params["name"] = payload.name
    if payload.pinned is not None:
        updates.append("pinned = :pinned")
        params["pinned"] = payload.pinned
    if payload.mode is not None:
        updates.append("mode = :mode")
        params["mode"] = payload.mode
    if payload.filters is not None:
        updates.append("filters = CAST(:filters AS jsonb)")
        params["filters"] = json.dumps(payload.filters)
    if payload.folder_id is not None:
        # Permettre d'assigner ou de retirer d'un dossier ("" = retirer)
        updates.append("folder_id = :folder_id")
        params["folder_id"] = payload.folder_id if payload.folder_id != "" else None
    if not updates:
        row = _get_user_scenario_or_404(scenario_id)
        return _user_scenario_to_gesica_format(row)
    updates.append("updated_at = NOW()")
    with engine.begin() as conn:
        conn.execute(text(f"""
            UPDATE user_scenarios SET {', '.join(updates)} WHERE id = :id
        """), params)
    # Épinglage via PATCH → scénario SAUVEGARDÉ : enrichissement complet côté serveur.
    if payload.pinned is True:
        try:
            _launch_full_pipeline(scenario_id)
        except Exception as _e:
            logger.warning(f"auto full-pipeline on patch-pin {scenario_id}: {_e}")
    row = _get_user_scenario_or_404(scenario_id)
    return _user_scenario_to_gesica_format(row)


# ── Dossiers (folders) ────────────────────────────────────────────────────────

@app.get("/user-scenario-folders")
def list_folders() -> list[dict[str, Any]]:
    """Liste tous les dossiers de scénarios utilisateur."""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT f.id, f.name, f.color, f.sort_order, f.created_at,
                   COUNT(s.id) AS scenario_count
            FROM user_scenario_folders f
            LEFT JOIN user_scenarios s ON s.folder_id = f.id
            GROUP BY f.id, f.name, f.color, f.sort_order, f.created_at
            ORDER BY f.sort_order ASC, f.created_at DESC
        """)).mappings().all()
    return [
        {
            "id": r["id"],
            "name": r["name"],
            "color": r["color"],
            "sort_order": r["sort_order"],
            "scenario_count": r["scenario_count"],
            "created_at": r["created_at"].isoformat() if r.get("created_at") else None,
        }
        for r in rows
    ]


@app.post("/user-scenario-folders", status_code=201)
def create_folder(payload: FolderIn, _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Crée un nouveau dossier."""
    import uuid
    new_id = "fld-" + str(uuid.uuid4()).replace("-", "")[:12]
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO user_scenario_folders (id, name, color, sort_order)
            VALUES (:id, :name, :color, :sort_order)
        """), {"id": new_id, "name": payload.name, "color": payload.color, "sort_order": payload.sort_order})
    with engine.connect() as conn:
        row = conn.execute(text("SELECT id, name, color, sort_order, created_at FROM user_scenario_folders WHERE id = :id"), {"id": new_id}).mappings().first()
    # Réponse construite sur les valeurs connues (insérées) : robuste même si le
    # SELECT de relecture ne retrouve pas la ligne (race / connexion distincte).
    return {
        "id": new_id, "name": payload.name, "color": payload.color,
        "sort_order": payload.sort_order, "scenario_count": 0,
        "created_at": row["created_at"].isoformat() if row and row.get("created_at") else None,
    }


@app.patch("/user-scenario-folders/{folder_id}")
def patch_folder(folder_id: str, payload: FolderIn, _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Renomme ou recolore un dossier."""
    with engine.begin() as conn:
        result = conn.execute(text("""
            UPDATE user_scenario_folders
            SET name = :name, color = :color, sort_order = :sort_order
            WHERE id = :id
        """), {"id": folder_id, "name": payload.name, "color": payload.color, "sort_order": payload.sort_order})
        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Dossier non trouvé")
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT f.id, f.name, f.color, f.sort_order, f.created_at, COUNT(s.id) AS scenario_count
            FROM user_scenario_folders f
            LEFT JOIN user_scenarios s ON s.folder_id = f.id
            WHERE f.id = :id
            GROUP BY f.id, f.name, f.color, f.sort_order, f.created_at
        """), {"id": folder_id}).mappings().first()
    return {
        "id": row["id"], "name": row["name"], "color": row["color"],
        "sort_order": row["sort_order"], "scenario_count": row["scenario_count"],
        "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
    }


@app.delete("/user-scenario-folders/{folder_id}")
def delete_folder(folder_id: str, _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Supprime un dossier (les scénarios sont conservés, leur folder_id devient NULL)."""
    with engine.begin() as conn:
        # Désassocier les scénarios
        conn.execute(text("UPDATE user_scenarios SET folder_id = NULL WHERE folder_id = :id"), {"id": folder_id})
        result = conn.execute(text("DELETE FROM user_scenario_folders WHERE id = :id"), {"id": folder_id})
        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Dossier non trouvé")
    return {"deleted": True, "id": folder_id}


# ── Detail (compatible ScenarioDetail frontend) ───────────────────────────────

@app.get("/user-scenarios/{scenario_id}/detail")
def get_user_scenario_detail(scenario_id: str, lang: str | None = Query(None)) -> dict[str, Any]:
    """
    Retourne les informations enrichies d'un scénario utilisateur au format ScenarioDetail.
    Compatible avec ScenarioDetailPage (boolean_queries, nl_queries, corpus_stats, etc.)
    """
    row = _get_user_scenario_or_404(scenario_id)
    with engine.connect() as conn:
        stats = conn.execute(text("""
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN EXISTS (
                    SELECT 1 FROM document_chunk c
                    WHERE c.document_id = d.id AND c.chunk_type = 'fulltext_section'
                ) THEN 1 ELSE 0 END) AS with_fulltext,
                COUNT(DISTINCT d.year) AS years_covered,
                COUNT(DISTINCT d.journal) AS journals_count,
                MIN(d.year) FILTER (WHERE d.year BETWEEN 1800 AND EXTRACT(YEAR FROM CURRENT_DATE)::int) AS year_min,
                MAX(d.year) FILTER (WHERE d.year BETWEEN 1800 AND EXTRACT(YEAR FROM CURRENT_DATE)::int) AS year_max
            FROM literature_document d
            JOIN article_scenarios ars ON ars.document_id = d.id
            WHERE ars.scenario_id = :sid
              AND (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)
        """), {"sid": scenario_id}).mappings().first()

    # Recherche multi-sous-requêtes : on renvoie les vraies listes booléennes /
    # naturelles. Sinon (mono-requête), la requête sauvegardée est booléenne OU en
    # langage naturel selon le mode réellement utilisé — on ne l'affiche que dans la
    # catégorie employée (évite de montrer la même requête en booléen ET en naturel).
    query_text = row["query"]
    _sub = _normalize_sub_queries(row.get("sub_queries"))
    _combinator = row.get("combinator") if row.get("combinator") in ("union", "intersection") else "union"
    _combined = _combined_query_text(query_text, _sub, _combinator)
    if _sub:
        boolean_queries = [s["text"] for s in _sub if s["kind"] == "boolean"]
        nl_queries = [s["text"] for s in _sub if s["kind"] == "natural"]
        # Facettes DANS L'ORDRE avec l'opérateur effectif de chacune (None pour la
        # principale) : les listes booléen/naturel ci-dessus perdent l'ordre et les
        # opérateurs — c'est ce qui faisait « disparaître » le ET entre deux requêtes.
        _ops = _facet_ops(_sub, _combinator)
        facets = [{"kind": s["kind"], "text": s["text"],
                   "op": (None if i == 0 else _ops[i - 1])} for i, s in enumerate(_sub)]
    else:
        _mode = (row.get("mode") or "hybrid").lower()
        _saved = [query_text] if query_text else []
        boolean_queries = _saved if _mode == "boolean" else []
        nl_queries = [] if _mode == "boolean" else _saved
        facets = []

    return {
        "id": scenario_id,
        "name": row["name"],
        "title": row["name"],
        "description": _msg(lang, f"Scénario utilisateur basé sur la recherche : {_combined}",
                            f"User scenario built from the search: {_combined}"),
        "cluster": "user",
        "recommended_actions": [],
        "boolean_queries": boolean_queries,
        "nl_queries": nl_queries,
        "evidence_extraction_prompt": "",
        "model_info": {},
        "alert_thresholds": {
            "green": {"label": _msg(lang, "Normal", "Normal"), "threshold": 0},
            "orange": {"label": _msg(lang, "Vigilance", "Watch"), "threshold": 50},
            "red": {"label": _msg(lang, "Alerte", "Alert"), "threshold": 80},
        },
        "databases": ["PubMed"],
        "outcome_definition": "",
        "variables_detail": {},
        "keywords": [w for w in query_text.split() if len(w) > 3][:10],
        "clinical_rationale": "",
        "corpus_stats": {
            "total": int(stats["total"] or 0) if stats else 0,
            "with_fulltext": int(stats["with_fulltext"] or 0) if stats else 0,
            "years_covered": int(stats["years_covered"] or 0) if stats else 0,
            "journals_count": int(stats["journals_count"] or 0) if stats else 0,
            "year_min": stats["year_min"] if stats else None,
            "year_max": stats["year_max"] if stats else None,
        },
        "is_user_scenario": True,
        "query": query_text,
        "combined_query": _combined,
        "facets": facets,
        "combinator": _combinator if _sub else None,
        "mode": row["mode"],
        "filters": row.get("filters") or {},
        "pinned": bool(row.get("pinned", False)),
        "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
    }


# ── Corpus (compatible fetchScenarioCorpus frontend) ──────────────────────────

@app.get("/user-scenarios/{scenario_id}/corpus")
def get_user_scenario_corpus(
    scenario_id: str,
    limit: int = 100000,
    offset: int = 0,
    year_from: int | None = None,
    year_to: int | None = None,
    fulltext_only: bool = False,
    source: str | None = None,
    threshold: float | None = None,
    abstract_chars: int | None = Query(None, ge=0, le=20000),
) -> dict[str, Any]:
    """
    Retourne le corpus d'articles pour un scénario utilisateur.
    Compatible avec fetchScenarioCorpus (même format de réponse).

    `abstract_chars` tronque le résumé de chaque article à N caractères : la page de
    résultats de recherche n'affiche qu'un extrait (600 caractères) et lit le résumé
    complet via /documents/{id} au clic — envoyer 10 000 résumés entiers pesait des
    dizaines de Mo pour rien. Sans le paramètre, le résumé complet est renvoyé.
    """
    from .relevance import _RERANK_JOBS, _maybe_autorerank  # lazy: relevance is loaded after this module
    row = _get_user_scenario_or_404(scenario_id)
    # Endpoint ouvert : borne limit/offset (un ?limit=100000000 matérialiserait toute
    # la jointure en RAM/JSON). 100000 couvre largement le plus gros corpus.
    limit = max(1, min(int(limit), 100000))
    offset = max(0, int(offset))
    _abstract_sql = "d.abstract" if abstract_chars is None else "LEFT(d.abstract, :abstract_chars)"
    # Seuil effectif : paramètre explicite (curseur en direct) > seuil sauvegardé
    # dans scenario_settings > défaut 0.45. (Auparavant codé en dur à 0.45, donc
    # le compteur « auto-sélectionnés » ne suivait jamais le curseur.)
    eff_threshold = 0.45
    try:
        with engine.connect() as _tc:
            _ts = _tc.execute(text(
                "SELECT similarity_threshold FROM scenario_settings WHERE scenario_id = :sid"
            ), {"sid": scenario_id}).scalar()
        if _ts is not None:
            eff_threshold = float(_ts)
    except Exception:
        pass
    if threshold is not None:
        eff_threshold = float(threshold)
    # Conditions de filtre (article_scenarios géré par JOIN)
    conditions = [
        "(d.is_duplicate IS NULL OR d.is_duplicate = FALSE)",
    ]
    params: dict[str, Any] = {"sid": scenario_id, "limit": limit, "offset": offset}
    if year_from:
        conditions.append("d.year >= :year_from")
        params["year_from"] = year_from
    if year_to:
        conditions.append("d.year <= :year_to")
        params["year_to"] = year_to
    if source:
        conditions.append("d.source = :source")
        params["source"] = source
    if fulltext_only:
        conditions.append("""EXISTS (
            SELECT 1 FROM document_chunk c
            WHERE c.document_id = d.id AND c.chunk_type = 'fulltext_section'
        )""")
    where = " AND ".join(conditions)
    _screated = row.get("created_at")
    with engine.connect() as conn:
        # Single query for both total and above_threshold to avoid race condition
        counts_row = conn.execute(text(f"""
            SELECT
                COUNT(*) AS total,
                COUNT(*) FILTER (WHERE ars.similarity_score >= :threshold) AS above_threshold,
                COUNT(*) FILTER (WHERE ars.similarity_score IS NULL) AS unscored,
                COUNT(*) FILTER (WHERE :screated IS NOT NULL AND d.created_at >= :screated) AS newly_fetched,
                COUNT(*) FILTER (WHERE EXISTS (
                    SELECT 1 FROM document_chunk c
                    WHERE c.document_id = d.id AND c.chunk_type = 'fulltext_section'
                )) AS with_fulltext
            FROM literature_document d
            JOIN article_scenarios ars ON ars.document_id = d.id AND ars.scenario_id = :sid
            WHERE {where}
        """), {**{k: v for k, v in params.items() if k not in ('limit', 'offset')},
               'threshold': eff_threshold, 'screated': _screated}).mappings().first()
        total = int(counts_row["total"] or 0)
        above_threshold = int(counts_row["above_threshold"] or 0)
        unscored = int(counts_row["unscored"] or 0)
        newly_fetched = int(counts_row["newly_fetched"] or 0) if _screated else None
        from_local = (total - newly_fetched) if newly_fetched is not None else None
        with_fulltext = int(counts_row["with_fulltext"] or 0)
        articles = conn.execute(text(f"""
            SELECT
                d.id, d.title, {_abstract_sql} AS abstract, d.year, d.source, d.url,
                d.authors, d.doi, d.journal, d.keywords, d.language,
                d.study_design, d.sample_size, d.country, d.citation_count,
                d.open_access, d.pmid, d.publication_type, d.quality_score,
                COALESCE(ars.screening_status, d.screening_status) AS screening_status,
                COALESCE(ars.reviewer_1_status, d.reviewer_1_status) AS reviewer_1_status,
                COALESCE(ars.similarity_score, 0.0) AS similarity_score,
                ars.rerank_score AS rerank_score,
                (COALESCE(ars.similarity_score, 0.0) >= :threshold) AS above_threshold,
                -- is_new : ingéré pendant CE scénario (vs déjà présent en base).
                -- Donne un sens au badge "Nouveau" vs "Base locale" côté UI.
                (:screated IS NOT NULL AND d.created_at >= :screated) AS is_new,
                EXISTS (
                    SELECT 1 FROM document_chunk c
                    WHERE c.document_id = d.id AND c.chunk_type = 'fulltext_section'
                ) AS has_fulltext
            FROM literature_document d
            JOIN article_scenarios ars ON ars.document_id = d.id AND ars.scenario_id = :sid
            WHERE {where}
            ORDER BY
                CASE WHEN COALESCE(ars.similarity_score, 0.0) >= :threshold THEN 0 ELSE 1 END ASC,
                (ars.rerank_score IS NOT NULL) DESC,
                ars.rerank_score DESC NULLS LAST,
                ars.similarity_score DESC NULLS LAST,
                d.year DESC NULLS LAST,
                d.citation_count DESC NULLS LAST,
                d.title ASC
            LIMIT :limit OFFSET :offset
        """), {**params, 'threshold': eff_threshold, 'screated': _screated,
               **({'abstract_chars': int(abstract_chars)} if abstract_chars is not None else {})}).mappings().all()
        year_dist = conn.execute(text(f"""
            SELECT d.year, COUNT(*) AS cnt
            FROM literature_document d
            JOIN article_scenarios ars ON ars.document_id = d.id AND ars.scenario_id = :sid
            WHERE {where}
              AND d.year >= 1800 AND d.year <= EXTRACT(YEAR FROM CURRENT_DATE)::int
            GROUP BY d.year ORDER BY d.year DESC
        """), {k: v for k, v in params.items() if k not in ('limit', 'offset')}).mappings().all()
        # Répartition par source, en distinguant la base locale (docs déjà en base
        # avant ce scénario) des références ramenées en direct par les APIs pendant
        # la construction du corpus (docs créés après la création du scénario).
        # PAS de LIMIT SQL : un ancien « LIMIT 12 » coupait les sources classées #13+
        # (arXiv, bioRxiv, medRxiv, OpenAIRE…) → la SOMME des badges était inférieure au
        # compteur d'en-tête (p.ex. 5758 badges vs 6031 total). On récupère TOUTES les
        # sources et on regroupe la traîne dans « Autres » ci-dessous, si bien que la
        # somme des badges = le total du corpus.
        source_dist = conn.execute(text(f"""
            SELECT d.source,
                   COUNT(*) AS cnt,
                   COUNT(*) FILTER (WHERE :screated IS NULL OR d.created_at < :screated) AS local_cnt,
                   COUNT(*) FILTER (WHERE :screated IS NOT NULL AND d.created_at >= :screated) AS live_cnt
            FROM literature_document d
            JOIN article_scenarios ars ON ars.document_id = d.id AND ars.scenario_id = :sid
            WHERE {where}
            GROUP BY d.source ORDER BY cnt DESC
        """), {**{k: v for k, v in params.items() if k not in ('limit', 'offset')},
               'screated': _screated}).mappings().all()
        # Dict {source: n_local, "source (live)": n_live} pour le panneau de recherche.
        # On détaille les 12 plus grosses sources ; le reste est regroupé dans « Autres »
        # (local + live) → les badges totalisent EXACTEMENT le compteur du corpus.
        source_breakdown: dict[str, int] = {}
        _TOP_SOURCES = 12
        _tail_local = _tail_live = 0
        for _i, r in enumerate(source_dist):
            _src = r["source"] or "Autre"
            _lc, _vc = int(r["local_cnt"] or 0), int(r["live_cnt"] or 0)
            if _i < _TOP_SOURCES:
                if _lc > 0:
                    source_breakdown[_src] = _lc
                if _vc > 0:
                    source_breakdown[f"{_src} (live)"] = _vc
            else:
                _tail_local += _lc
                _tail_live += _vc
        if _tail_local > 0:
            source_breakdown["Autres"] = _tail_local
        if _tail_live > 0:
            source_breakdown["Autres (live)"] = _tail_live

    # Auto-score : si des articles ne sont pas encore scorés, on lance le rerank
    # en arrière-plan (une fois). Le seuil devient alors exploitable.
    rerank_running = _maybe_autorerank(scenario_id) if unscored > 0 else False

    return {
        "scenario_id": scenario_id,
        "scenario_title": row["name"],
        "total": total,
        "above_threshold": above_threshold,
        "below_threshold": max(0, total - above_threshold - unscored),
        "unscored": unscored,
        "from_local": from_local,
        "newly_fetched": newly_fetched,
        "docs_with_fulltext": with_fulltext,
        "docs_abstract_only": max(0, total - with_fulltext),
        "source_breakdown": source_breakdown,
        "rerank_running": rerank_running or (_RERANK_JOBS.get(scenario_id, {}).get("status") == "running"),
        "threshold": eff_threshold,
        "offset": offset,
        "limit": limit,
        "abstract_truncated": abstract_chars is not None,
        "articles": [dict(a) for a in articles],
        "year_distribution": [{"year": r["year"], "count": int(r["cnt"])} for r in year_dist],
        "source_distribution": [{"source": r["source"], "count": int(r["cnt"])} for r in source_dist],
        "is_user_scenario": True,
    }

# Verrous pour protéger l'accès concurrent aux états de jobs en mémoire (H-4)
_populate_jobs_lock = threading.Lock()
_pipeline_jobs_lock = threading.Lock()

_user_scenario_populate_jobs: dict[str, dict] = {}
_user_scenario_pipeline_jobs: dict[str, dict] = {}


def _launch_populate_job(scenario_id: str, query: str, filters: dict, max_results: int,
                         include_live: bool = True) -> str:
    """
    Démarre un job d'ingestion en arrière-plan pour un scénario, en garantissant
    qu'un seul job tourne à la fois (verrou partagé). Renvoie l'état : "started"
    ou "already_running". Utilisé par /populate ET /search/live afin qu'aucun des
    deux ne lance un populate concurrent sur le même scénario.
    """
    from .pipeline import _run_user_scenario_populate  # lazy: pipeline is loaded after this module
    import threading
    with _populate_jobs_lock:
        job = _user_scenario_populate_jobs.get(scenario_id)
        if job and job.get("status") == "running":
            return "already_running"
        _user_scenario_populate_jobs[scenario_id] = {
            "status": "running", "ingested": 0, "errors": 0, "total_found": 0,
            "sources": {"db_cache": 0, "pubmed": 0, "openalex": 0, "crossref": 0,
                        "europepmc": 0, "preprint": 0, "semantic_scholar": 0, "doaj": 0,
                        "clinicaltrials": 0, "core": 0, "arxiv": 0, "openaire": 0,
                        "biorxiv": 0, "medrxiv": 0},
        }
    # Persister l'état « en cours » : la liste des scénarios et l'indicateur global
    # (/activity) le lisent en base — une recherche lancée puis quittée (autre page,
    # rechargement) reste ainsi visible et retrouvable. Remis à done/error à la fin
    # du run ; les orphelins d'un redémarrage sont passés à 'error' au démarrage.
    try:
        with engine.begin() as _c:
            _c.execute(text("UPDATE user_scenarios SET populate_status = 'running', updated_at = NOW() "
                            "WHERE id = :sid"), {"sid": scenario_id})
    except Exception as _e:                              # noqa: BLE001 - jamais bloquant
        logger.warning(f"populate_status=running {scenario_id}: {_e}")
    def _guarded() -> None:
        # The job's own error handling starts inside the function; anything raised
        # before it (a missing dependency at its imports, say) killed the thread and
        # left the job "running" forever, with the search page polling it. Seen by
        # the browser smoke test on an API without `requests`.
        try:
            _run_user_scenario_populate(scenario_id, query, filters or {}, max_results, None, include_live)
        except BaseException as _e:                          # noqa: BLE001
            logger.error(f"Populate {scenario_id} crashed before its own error handling: {_e}", exc_info=True)
            _job = _user_scenario_populate_jobs.get(scenario_id)
            if _job is None or _job.get("status") == "running":
                _user_scenario_populate_jobs[scenario_id] = {
                    "status": "error", "error": f"{type(_e).__name__}: {_e}",
                    "ingested": (_job or {}).get("ingested", 0),
                }
            try:
                with engine.begin() as _c:
                    _c.execute(text("UPDATE user_scenarios SET populate_status = 'error', updated_at = NOW() "
                                    "WHERE id = :sid"), {"sid": scenario_id})
            except Exception as _e2:                          # noqa: BLE001 - jamais bloquant
                logger.warning(f"populate_status=error {scenario_id}: {_e2}")

    threading.Thread(target=_guarded, daemon=True).start()
    return "started"


@app.post("/user-scenarios/{scenario_id}/populate")
def populate_user_scenario(
    scenario_id: str,
    max_results: int = 100000,
    include_live: bool = True,
    _: None = Depends(require_api_key),
) -> dict[str, Any]:
    """
    Construit le corpus du scénario = requête booléenne sur (base locale ∪ live).
    Plafond LIVE_MAX_PER_SOURCE par source. include_live=False : base locale seule.
    """
    row = _get_user_scenario_or_404(scenario_id)
    query = row["query"]

    if _launch_populate_job(scenario_id, query, row.get("filters") or {}, max_results, include_live) == "already_running":
        job = _user_scenario_populate_jobs.get(scenario_id) or {}
        return {
            "scenario_id": scenario_id,
            "status": "already_running",
            "message": "Une ingestion est déjà en cours pour ce scénario.",
            "ingested": job.get("ingested", 0),
        }

    return {
        "scenario_id": scenario_id,
        "status": "started",
        "query": query,
        "max_results": max_results,
        "message": f"Ingération multi-sources lancée en arrière-plan pour '{row['name']}' "
                   "(DB Cache + PubMed + OpenAlex + Crossref + EuropePMC + Preprints + "
                   "Semantic Scholar + DOAJ + ClinicalTrials.gov + CORE + arXiv + OpenAIRE + "
                   "medRxiv + bioRxiv). "
                   "Utilisez /user-scenarios/{id}/populate/status pour suivre la progression.",
    }


@app.get("/user-scenarios/{scenario_id}/populate/status")
def get_user_scenario_populate_status(scenario_id: str) -> dict[str, Any]:
    """Retourne l'état de l'ingéstion multi-sources en cours pour un scénario utilisateur."""
    _get_user_scenario_or_404(scenario_id)
    job = _user_scenario_populate_jobs.get(scenario_id)
    if not job:
        return {
            "scenario_id": scenario_id,
            "status": "not_started",
            "message": "Aucune ingestion lancée. Appelez POST /user-scenarios/{id}/populate.",
        }
    return {"scenario_id": scenario_id, **job}


def _launch_full_pipeline(scenario_id: str, max_results: int = LIVE_MAX_PER_SOURCE) -> str:
    """Démarre le pipeline COMPLET d'enrichissement en arrière-plan (un seul à la fois
    par scénario). Renvoie 'started' | 'already_running' | 'no_query'. Partagé par
    l'endpoint POST /pipeline, l'auto-déclenchement à l'ÉPINGLAGE (« scénario sauvegardé
    → tout est calculé côté serveur ») et le bouton « tout recalculer ». Robuste : ne
    lève jamais (usage best-effort depuis les handlers de sauvegarde)."""
    from .pipeline import _run_user_scenario_full_pipeline  # lazy: pipeline is loaded after this module
    import threading
    try:
        row = _get_user_scenario_or_404(scenario_id)
    except Exception:
        return "no_query"
    query = row.get("query")
    if not query:
        return "no_query"
    with _pipeline_jobs_lock:
        job = _user_scenario_pipeline_jobs.get(scenario_id)
        if job and job.get("overall_status") in ("running", "starting"):
            return "already_running"
        _user_scenario_pipeline_jobs[scenario_id] = {
            "overall_status": "starting",
            "current_step": "ingest",
            "steps": {k: {"status": "pending"} for k in (
                "ingest", "fulltext", "embed", "rerank", "pico", "metadata",
                "clustering", "knowledge_graph", "evidence", "variables")},
        }
    threading.Thread(
        target=_run_user_scenario_full_pipeline,
        args=(scenario_id, query, row.get("filters") or {}, max_results),
        daemon=True,
    ).start()
    return "started"


@app.post("/user-scenarios/{scenario_id}/pipeline")
def start_user_scenario_pipeline(
    scenario_id: str,
    max_results: int = LIVE_MAX_PER_SOURCE,
    _: None = Depends(require_api_key),
) -> dict[str, Any]:
    """
    Déclenche le pipeline complet d'enrichissement en arrière-plan :
    ingest → fulltext → embed → rerank → PICO → métadonnées → clustering →
    knowledge graph → evidence brief → variables & modèle.
    Appelé dès qu'une recherche est sauvegardée en scénario, et par « tout recalculer ».
    """
    row = _get_user_scenario_or_404(scenario_id)
    status = _launch_full_pipeline(scenario_id, max_results)
    if status == "already_running":
        job = _user_scenario_pipeline_jobs.get(scenario_id) or {}
        return {
            "scenario_id": scenario_id,
            "status": "already_running",
            "message": "Un pipeline est déjà en cours pour ce scénario.",
            "current_step": job.get("current_step"),
        }

    return {
        "scenario_id": scenario_id,
        "status": "started",
        "query": row["query"],
        "max_results": max_results,
        "message": f"Pipeline complet lancé pour '{row['name']}' "
                   "(ingest 13 sources → fulltext → embeddings → rerank → PICO → métadonnées → "
                   "clustering → knowledge graph → evidence brief → variables & modèle). "
                   "Suivez la progression via GET /user-scenarios/{id}/pipeline/status.",
        "steps": ["ingest", "fulltext", "embed", "rerank", "pico", "metadata",
                  "clustering", "knowledge_graph", "evidence", "variables"],
    }


@app.get("/user-scenarios/{scenario_id}/pipeline/status")
def get_user_scenario_pipeline_status(scenario_id: str) -> dict[str, Any]:
    """Retourne l'état détaillé du pipeline d'enrichissement pour un scénario utilisateur."""
    _get_user_scenario_or_404(scenario_id)
    job = _user_scenario_pipeline_jobs.get(scenario_id)
    if not job:
        return {
            "scenario_id": scenario_id,
            "overall_status": "not_started",
            "message": "Aucun pipeline lancé. Appelez POST /user-scenarios/{id}/pipeline.",
            "steps": {},
        }
    return {"scenario_id": scenario_id, **job}


def _counts_consistency(counts: dict) -> tuple[bool, list[dict]]:
    """Compare entre eux les nombres d'articles que l'interface affiche pour UN scénario.

    Référence = `corpus_links`, les liens en base hors doublons (ce que l'onglet Corpus
    liste). La liste des scénarios affiche `article_count`, une COPIE stockée, mise à jour
    par étapes pendant une recherche (d'abord les correspondances locales, puis le corpus
    nettoyé) ; le PRISMA affiche `records_screened`, figé à la fin de la dernière
    recherche ; l'étape 2 compte au-dessus/en dessous du seuil en direct. Le temps d'un
    pipeline, ces trois lectures divergent légitimement ; à la fin, elles doivent
    coïncider — et c'est ce que vérifie cette fonction (pure, testée hors base)."""
    ref = int(counts.get("corpus_links") or 0)
    mismatches: list[dict] = []
    ac = counts.get("article_count")
    if ac is not None and int(ac) != ref:
        mismatches.append({"field": "article_count", "value": int(ac), "expected": ref})
    ps = counts.get("prisma_screened")
    if ps is not None and int(ps) != ref:
        mismatches.append({"field": "prisma_screened", "value": int(ps), "expected": ref})
    ab, be = counts.get("above_threshold"), counts.get("below_threshold")
    if ab is not None and be is not None and int(ab) + int(be) != ref:
        mismatches.append({"field": "above_plus_below", "value": int(ab) + int(be), "expected": ref})
    return (not mismatches), mismatches


def _scenario_counts(scenario_id: str, row: dict | None = None) -> dict[str, Any]:
    """Tous les compteurs d'un scénario utilisateur en une lecture, avec le verdict de
    _counts_consistency et l'état d'avancement (pipeline ou populate en cours)."""
    from datetime import datetime as _dt, timezone as _tz
    row = row or _get_user_scenario_or_404(scenario_id)
    thr = _get_scenario_threshold(scenario_id)
    with engine.connect() as conn:
        r = conn.execute(text("""
            SELECT COUNT(DISTINCT ars.document_id) AS corpus_links,
                   COUNT(DISTINCT ars.document_id) FILTER (WHERE COALESCE(ars.similarity_score, 0) >= :thr) AS above,
                   COUNT(DISTINCT ars.document_id) FILTER (WHERE COALESCE(ars.similarity_score, 0) < :thr) AS below,
                   COUNT(DISTINCT ars.document_id) FILTER (WHERE EXISTS (
                       SELECT 1 FROM document_chunk c
                       WHERE c.document_id = ars.document_id AND c.embedding IS NOT NULL)) AS embedded
            FROM article_scenarios ars
            JOIN literature_document d ON d.id = ars.document_id
            WHERE ars.scenario_id = :sid AND (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)
        """), {"sid": scenario_id, "thr": thr}).mappings().first()
    figures = _load_prisma_identification(scenario_id)
    job = _user_scenario_pipeline_jobs.get(scenario_id) or {}
    pjob = _user_scenario_populate_jobs.get(scenario_id) or {}
    in_progress = (
        row.get("pipeline_status") in ("running", "starting")
        or row.get("populate_status") == "running"
        or job.get("overall_status") in ("running", "starting")
        or pjob.get("status") in ("running", "starting")
    )
    counts = {
        "article_count": int(row.get("article_count") or 0),
        "corpus_links": int(r["corpus_links"] or 0),
        "prisma_screened": (int(figures.get("records_screened") or 0) if figures else None),
        "above_threshold": int(r["above"] or 0),
        "below_threshold": int(r["below"] or 0),
        "embedded": int(r["embedded"] or 0),
    }
    ok, mismatches = _counts_consistency(counts)
    return {
        "scenario_id": scenario_id,
        "in_progress": bool(in_progress),
        "pipeline_status": row.get("pipeline_status"),
        "populate_status": row.get("populate_status"),
        "current_step": job.get("current_step") or row.get("pipeline_step"),
        "threshold": thr,
        **counts,
        "consistent": ok,
        "mismatches": mismatches,
        "checked_at": _dt.now(_tz.utc).isoformat(),
    }


@app.get("/user-scenarios/{scenario_id}/counts")
def get_user_scenario_counts(scenario_id: str) -> dict[str, Any]:
    """Les nombres d'articles que l'interface affiche pour ce scénario (liste, en-tête,
    PRISMA, étape sémantique), comparés entre eux, et si un pipeline tourne encore.
    Sert la bannière de la page scénario : « pipeline en cours, compteurs provisoires »
    puis « terminé, N articles partout » — ou la liste des écarts."""
    return _scenario_counts(scenario_id)


@app.get("/activity")
def get_activity() -> dict[str, Any]:
    """Recherches et pipelines EN COURS, tous scénarios utilisateur confondus — pour
    l'indicateur global de l'interface, visible sur toutes les pages.

    Une recherche continue côté serveur quand on change de page ; sans indicateur elle
    « disparaissait » et, non épinglée, on ne savait plus où la retrouver. Source de
    vérité : les statuts persistés (populate_status 'running' posé au lancement,
    pipeline_status 'running'/'starting'), qui survivent au rechargement de la page ;
    l'état en mémoire des jobs n'apporte que l'étape courante. Les statuts orphelins
    d'un redémarrage de l'API sont remis à 'error' au démarrage (cf. startup_event)."""
    from datetime import datetime as _dt, timezone as _tz
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT id, name, query, pinned, populate_status, pipeline_status, pipeline_step,
                   COALESCE(article_count, 0) AS article_count, sub_queries, combinator
            FROM user_scenarios
            WHERE pipeline_status IN ('running', 'starting') OR populate_status = 'running'
            ORDER BY updated_at DESC
            LIMIT 30
        """)).mappings().all()
    items: list[dict[str, Any]] = []
    for r in rows:
        pipeline_running = r["pipeline_status"] in ("running", "starting")
        job = (_user_scenario_pipeline_jobs.get(r["id"]) if pipeline_running
               else _user_scenario_populate_jobs.get(r["id"])) or {}
        items.append({
            "scenario_id": r["id"],
            "name": r["name"],
            # Affichage : l'expression complète « (A) AND (B) » d'une recherche multi-facettes.
            "query": _combined_query_text(r["query"], r["sub_queries"], r["combinator"]),
            "pinned": bool(r["pinned"]),
            "kind": "pipeline" if pipeline_running else "search",
            "step": (job.get("current_step") or r["pipeline_step"]) if pipeline_running else job.get("phase"),
            "article_count": int(r["article_count"] or 0),
        })
    return {"running": items, "count": len(items), "checked_at": _dt.now(_tz.utc).isoformat()}


@app.get("/user-scenarios/{scenario_id}/embedding-status")
def get_user_scenario_embedding_status(scenario_id: str) -> dict[str, Any]:
    """
    Embedding status for a user scenario.
    Reports separately:
    - title+abstract docs pending (1 chunk each)
    - fulltext papers pending (N chunks each)
    """
    _get_user_scenario_or_404(scenario_id)
    with engine.connect() as conn:
        # Univers = corpus du scénario, hors doublons (MÊME filtre que /corpus),
        # pour réconcilier les compteurs. Inclut les docs SANS chunk (chunkless).
        corpus = conn.execute(text("""
            SELECT
                COUNT(*) AS corpus_total,
                COUNT(*) FILTER (
                    WHERE NOT EXISTS (SELECT 1 FROM document_chunk c WHERE c.document_id = ars.document_id)
                ) AS chunkless
            FROM article_scenarios ars
            JOIN literature_document ld ON ld.id = ars.document_id
            WHERE ars.scenario_id = :sid AND ld.is_duplicate IS NOT TRUE
        """), {"sid": scenario_id}).mappings().first()

        # Title+abstract: one chunk per doc. Un chunk title_abstract n'est "en
        # attente" QUE s'il sera réellement embeddé par le worker — qui IGNORE
        # (a) les chunks trop courts (length <= 20) et (b) le title_abstract d'un
        # doc qui possède aussi du plein texte (on embed alors le plein texte). Sans
        # ces deux filtres, le compteur restait bloqué à un petit nombre "en cours".
        ta = conn.execute(text("""
            SELECT
                COUNT(DISTINCT ars.document_id) AS total_docs,
                COUNT(DISTINCT CASE WHEN c.embedding IS NOT NULL THEN ars.document_id END) AS embedded_docs,
                COUNT(DISTINCT CASE
                    WHEN c.embedding IS NULL
                     AND length(c.content) > 20
                     AND NOT EXISTS (
                         SELECT 1 FROM document_chunk c2
                         WHERE c2.document_id = ars.document_id
                           AND c2.chunk_type = 'fulltext_section'
                     )
                    THEN ars.document_id END) AS pending_docs
            FROM article_scenarios ars
            JOIN document_chunk c ON c.document_id = ars.document_id
                AND c.chunk_type = 'title_abstract'
            JOIN literature_document ld ON ld.id = ars.document_id
            WHERE ars.scenario_id = :sid AND ld.is_duplicate IS NOT TRUE
        """), {"sid": scenario_id}).mappings().first()

        # Full-text: multiple chunks per doc
        ft = conn.execute(text("""
            SELECT
                COUNT(DISTINCT d.document_id) AS total_ft_docs,
                COUNT(DISTINCT CASE WHEN ft_emb.pending_chunks = 0 THEN d.document_id END) AS ft_docs_complete,
                COUNT(DISTINCT CASE WHEN ft_emb.pending_chunks > 0 THEN d.document_id END) AS ft_docs_pending,
                COALESCE(SUM(ft_emb.total_chunks), 0) AS total_ft_chunks,
                COALESCE(SUM(ft_emb.pending_chunks), 0) AS pending_ft_chunks,
                COALESCE(SUM(ft_emb.embedded_chunks), 0) AS embedded_ft_chunks
            FROM (
                SELECT DISTINCT ars.document_id
                FROM article_scenarios ars
                JOIN document_chunk c ON c.document_id = ars.document_id
                    AND c.chunk_type = 'fulltext_section'
                JOIN literature_document ld ON ld.id = ars.document_id
                WHERE ars.scenario_id = :sid AND ld.is_duplicate IS NOT TRUE
            ) d
            JOIN (
                SELECT
                    c.document_id,
                    COUNT(*) AS total_chunks,
                    COUNT(*) FILTER (WHERE c.embedding IS NULL) AS pending_chunks,
                    COUNT(*) FILTER (WHERE c.embedding IS NOT NULL) AS embedded_chunks
                FROM document_chunk c
                WHERE c.chunk_type = 'fulltext_section'
                GROUP BY c.document_id
            ) ft_emb ON ft_emb.document_id = d.document_id
        """), {"sid": scenario_id}).mappings().first()

        # Scores de pertinence (RANKING) — INDÉPENDANT de l'indexation RAG ci-dessus.
        # Le similarity_score affiché est calculé EN LIGNE pendant la phase "scoring"
        # (_run_semantic_rerank_inline : réutilise les embeddings stockés, ré-embedde
        # le reste à la volée), et le rerank Cohere écrit rerank_score sur le
        # sous-ensemble pertinent. Ce sont CES compteurs qui pilotent les voyants
        # "Sémantique" / "Cohere" — PAS le worker d'indexation RAG (chunks).
        ranking = conn.execute(text("""
            SELECT
                COUNT(*) AS total,
                COUNT(*) FILTER (WHERE ars.similarity_score IS NOT NULL) AS scored,
                COUNT(*) FILTER (WHERE ars.rerank_score IS NOT NULL) AS reranked
            FROM article_scenarios ars
            JOIN literature_document ld ON ld.id = ars.document_id
            WHERE ars.scenario_id = :sid AND ld.is_duplicate IS NOT TRUE
        """), {"sid": scenario_id}).mappings().first()

    ta_total = int(ta["total_docs"] or 0)
    ta_embedded = int(ta["embedded_docs"] or 0)
    ta_pending = int(ta["pending_docs"] or 0)

    corpus_total = int(corpus["corpus_total"] or 0)
    chunkless = int(corpus["chunkless"] or 0)

    ft_total = int(ft["total_ft_docs"] or 0)
    ft_pending_docs = int(ft["ft_docs_pending"] or 0)
    ft_total_chunks = int(ft["total_ft_chunks"] or 0)
    ft_pending_chunks = int(ft["pending_ft_chunks"] or 0)
    ft_embedded_chunks = int(ft["embedded_ft_chunks"] or 0)

    # Docs without any fulltext = abstract-only
    abstract_only_total = ta_total - ft_total
    # Total pending embedding work
    total_pending_chunks = ta_pending + ft_pending_chunks

    # ── Pertinence (ranking) : honnête, découplé de l'indexation RAG ────────────
    rank_total = int(ranking["total"] or 0)
    rank_scored = int(ranking["scored"] or 0)
    rank_reranked = int(ranking["reranked"] or 0)
    cohere_configured = bool(os.getenv("COHERE_API_KEY"))
    # Sémantique : prêt SEULEMENT quand TOUT le corpus est scoré (et non "au moins
    # un chunk vectorisé"). Tant que des articles restent non scorés, le classement
    # sémantique est incomplet → voyant non vert.
    semantic_ready = rank_total > 0 and rank_scored >= rank_total
    # Cohere : prêt SEULEMENT quand le rerank a RÉELLEMENT tourné (≥ 1 rerank_score),
    # pas simplement parce qu'une clé existe.
    cohere_ready = cohere_configured and rank_reranked > 0

    if chunkless > 0:
        status = "partial"
        status_label = f"{chunkless} document(s) pas encore découpé(s) (sans chunk) — invisibles à la recherche"
    elif total_pending_chunks == 0 and ta_total > 0:
        status = "complete"
        status_label = "All embeddings complete"
    elif ta_embedded == 0 and ft_embedded_chunks == 0:
        status = "none"
        status_label = "No embeddings yet — only lexical search available"
    else:
        status = "partial"
        status_label = f"{ta_pending} abstract-only docs + {ft_pending_chunks} fulltext chunks still to embed"

    return {
        "scenario_id": scenario_id,
        "status": status,
        "status_label": status_label,
        "corpus_total": corpus_total,
        "chunkless": chunkless,
        "abstract_only": {
            "total_docs": abstract_only_total,
            "embedded_docs": max(0, abstract_only_total - ta_pending),
            "pending_docs": ta_pending,
        },
        "title_abstract_chunks": {
            "total_docs": ta_total,
            "embedded_docs": ta_embedded,
            "pending_docs": ta_pending,
        },
        "fulltext": {
            "total_docs": ft_total,
            "docs_fully_embedded": int(ft["ft_docs_complete"] or 0),
            "docs_pending": ft_pending_docs,
            "total_chunks": ft_total_chunks,
            "embedded_chunks": ft_embedded_chunks,
            "pending_chunks": ft_pending_chunks,
        },
        "total_pending_chunks": total_pending_chunks,
        # ── Pertinence (ranking) : compteurs réels du classement affiché ─────────
        # Découplé de l'indexation RAG : `scored`/`reranked` portent sur les scores
        # effectivement présents sur article_scenarios (ce que l'utilisateur voit).
        "ranking": {
            "total": rank_total,
            "scored": rank_scored,
            "reranked": rank_reranked,
            "complete": semantic_ready,
        },
        # Disponibilité RÉELLE de chaque mode de pertinence (plus de voyant "lexical"
        # toujours vert, qui n'apportait aucune information) :
        # - sémantique : vert seulement quand TOUT le corpus est scoré.
        # - cohere     : vert seulement quand le rerank a réellement tourné.
        # - cohere_configured : distingue "pas de clé" de "clé OK, rerank pas encore".
        "score_availability": {
            "semantic": semantic_ready,
            "cohere": cohere_ready,
            "cohere_configured": cohere_configured,
        }
    }


# ─── Endpoint model-status pour user_scenarios ───────────────────────────────
@app.get("/user-scenarios/{scenario_id}/model-status")
def get_user_scenario_model_status(scenario_id: str, lang: str | None = Query(None)) -> dict[str, Any]:
    """
    Statut du modèle pour un scénario utilisateur.
    Retourne un statut neutre (pas de modèle prédictif pour les scénarios utilisateurs).
    """
    _get_user_scenario_or_404(scenario_id)
    from datetime import datetime, timezone
    # Compter les articles récents (30 derniers jours)
    with engine.connect() as conn:
        recent_count = conn.execute(text("""
            SELECT COUNT(*) AS cnt
            FROM literature_document d
            JOIN article_scenarios asn ON asn.document_id = d.id AND asn.scenario_id = :sid
            WHERE d.project_context = 'literev'
              AND d.created_at >= NOW() - INTERVAL '30 days'
        """), {"sid": scenario_id}).scalar()
    return {
        "scenario_id": scenario_id,
        "status_color": "blue",
        "status_label": _msg(lang, "Scénario personnalisé", "Custom scenario"),
        "model_info": {
            "name": "N/A",
            "description": "Les scénarios personnalisés ne disposent pas d'un modèle prédictif intégré. Utilisez l'onglet Variables & Données pour configurer votre propre modèle.",
            "type": "user_defined",
        },
        "alert_thresholds": {},
        "model_result": None,
        "model_error": None,
        "recent_articles_30d": int(recent_count or 0),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

@app.post("/user-scenarios/{scenario_id}/model-run")
def run_user_scenario_model(scenario_id: str, _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Re-run du modèle pour un scénario utilisateur (retourne le statut neutre)."""
    return get_user_scenario_model_status(scenario_id)
