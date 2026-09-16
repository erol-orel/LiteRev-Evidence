"""Startup DDL: tables, columns and indexes the app adds on top of schema.sql.

Extracted from main.py (LiteRev API); `main` re-exports everything for the scripts,
tools and tests.
"""
from __future__ import annotations

import json
import os
import re

from sqlalchemy import text, bindparam

import lexical_search as _lex
import llm_usage as _llm_usage

from .core import _is_openai_quota_error, _openai_in_cooldown, _trip_openai_cooldown, app, engine, logger
from .documents import _embed_chunks_resilient

# ─────────────────────────────────────────────────────────────────────────────
# Index de performance
# ─────────────────────────────────────────────────────────────────────────────
# Instructions de DDL de démarrage qui ont ÉCHOUÉ, accumulées au fil des _ensure_*.
# Le DDL de démarrage échoue OUVERT par conception (un serveur qui refuse de démarrer
# est pire) — mais jusqu'ici la seule trace était une ligne de log, et /health
# continuait d'annoncer « ok » sur une base incomplète. Cette liste rend l'état
# dégradé LISIBLE (cf. /health).
_SCHEMA_DDL_FAILURES: list[str] = []

# Tables sans lesquelles l'application ne peut pas servir ses écrans principaux.
_REQUIRED_TABLES = (
    "literature_document", "document_chunk", "article_scenarios",
    "user_scenarios", "user_scenario_folders", "scenario_settings",
)


def _exec_ddl_isolated(statements, label: str, record: bool = True) -> list[str]:
    """Exécute des instructions DDL CHACUNE DANS SA PROPRE TRANSACTION.

    Pourquoi : Postgres avorte la transaction ENTIÈRE à la première erreur. Regroupées
    dans un seul `with engine.begin()`, une instruction qui échoue (p. ex. un ALTER sur
    une table absente) annulait les CREATE TABLE réussis qui la précédaient dans le même
    bloc — la base ressortait SANS les tables que la fonction venait de créer, et le seul
    indice était un avertissement dans les logs. C'est exactement ce qui rendait toute
    base NEUVE inutilisable (voir tests/test_fresh_db_bootstrap.py).

    Renvoie la liste des instructions ayant échoué (vide = tout est passé).

    `record=False` : l'échec est journalisé et renvoyé, mais PAS inscrit dans
    _SCHEMA_DDL_FAILURES — donc sans effet sur `schema.ok`, qui est BLOQUANT au
    déploiement (scripts/check_health.py). À réserver aux objets dont l'absence
    dégrade (recherche plus lente) sans rien casser."""
    failed: list[str] = []
    for _sql in statements:
        try:
            with engine.begin() as _c:
                _c.execute(text(_sql))
        except Exception as _e:                     # noqa: BLE001 - best-effort par design
            _first = _sql.strip().split("\n")[0][:120]
            failed.append(_first)
            if record:
                _SCHEMA_DDL_FAILURES.append(f"{label}: {_first}")
            logger.warning(f"{label}: DDL ignorée ({_e.__class__.__name__}): {_first}")
    return failed


def _ensure_llm_usage_table() -> None:
    """Table de COMPTABILITÉ des appels OpenAI (cf. llm_usage.py).

    Créée AVANT le worker d'arrière-plan (lancé à l'import, plus bas) : sans elle, les
    premiers cycles d'embedding/PICO dépenseraient sans laisser de trace. `configure()`
    donne au module l'engine ; sans cet appel il n'enregistre rien (et ne casse rien)."""
    _llm_usage.configure(engine)
    _exec_ddl_isolated(_llm_usage.DDL, "_ensure_llm_usage_table")


try:
    _ensure_llm_usage_table()
except Exception as _e:                               # jamais bloquant : c'est de la mesure
    logger.warning(f"_ensure_llm_usage_table: {_e}")


def _ensure_document_search() -> None:
    """Table `document_search` + fonction + triggers de la recherche PLEIN TEXTE
    (cf. lexical_search.py) : un tsvector par document, tenu à jour par triggers, qui
    remplace les LIKE '%terme%' du match booléen (55 à 240 s par requête en prod).

    `record=False` : si ces objets ne peuvent pas être créés (droits sur les tables,
    par ex.), la recherche booléenne reste sur le chemin LIKE — plus lente, pas
    cassée. Ce n'est donc pas une dégradation du schéma au sens de `schema.ok`
    (bloquant au déploiement) ; l'état est exposé dans /health → lexical_search."""
    _lex.configure(engine)
    _lex.DDL_FAILURES[:] = _exec_ddl_isolated(_lex.DDL, "_ensure_document_search", record=False)


try:
    _ensure_document_search()
except Exception as _e:
    logger.warning(f"_ensure_document_search: {_e}")


def _ensure_performance_indexes() -> None:
    """Crée les index manquants sur les colonnes chaudes + un index ANN pgvector.

    Exécuté EN ARRIÈRE-PLAN (hors du chemin de démarrage) : la 1re création de
    l'index vectoriel peut durer plusieurs minutes sur un gros corpus et ne doit
    pas retarder le health check du déploiement. Tout est IF NOT EXISTS (idempotent)
    et chaque instruction est isolée : un échec est journalisé sans rien casser
    (au pire, la requête concernée reste en scan séquentiel, comme aujourd'hui)."""
    # 1) Index B-tree sur les colonnes de filtre / jointure les plus fréquentes.
    btree = [
        # Extension trigramme : permet aux index GIN gin_trgm_ops (créés plus bas)
        # d'assister les LIKE '%terme%' du match booléen. Tolérant : si le rôle DB ne
        # peut pas créer l'extension, les index GIN échouent et on reste en scan.
        "CREATE EXTENSION IF NOT EXISTS pg_trgm",
        "CREATE INDEX IF NOT EXISTS ix_article_scenarios_scenario ON article_scenarios (scenario_id)",
        "CREATE INDEX IF NOT EXISTS ix_article_scenarios_document ON article_scenarios (document_id)",
        "CREATE INDEX IF NOT EXISTS ix_article_scenarios_scen_sim ON article_scenarios (scenario_id, similarity_score)",
        "CREATE INDEX IF NOT EXISTS ix_litdoc_screening_status ON literature_document (screening_status)",
        "CREATE INDEX IF NOT EXISTS ix_litdoc_is_duplicate ON literature_document (is_duplicate)",
        "CREATE INDEX IF NOT EXISTS ix_litdoc_source ON literature_document (source)",
        "CREATE INDEX IF NOT EXISTS ix_litdoc_project_context ON literature_document (project_context)",
        "CREATE INDEX IF NOT EXISTS ix_doc_chunk_document ON document_chunk (document_id)",
        "CREATE INDEX IF NOT EXISTS ix_doc_chunk_type ON document_chunk (chunk_type)",
        # Dédup par titre : backfill title_norm des lignes existantes (idempotent —
        # WHERE title_norm IS NULL) puis index. Même normalisation que _normalize_title.
        "UPDATE literature_document SET title_norm = btrim(regexp_replace(lower(title), '[^a-z0-9]+', ' ', 'g')) "
        "WHERE title_norm IS NULL AND title IS NOT NULL",
        "CREATE INDEX IF NOT EXISTS ix_litdoc_title_norm ON literature_document (project_context, title_norm)",
        # Index UNIQUES rendant l'INSERT de _ingest_doc_direct atomique (ON CONFLICT
        # DO NOTHING) — ferment la course entre fetchers parallèles. best-effort : le
        # try/except par instruction ci-dessous journalise un échec sans rien casser.
        #  • DOI : présent en prod via un script ad-hoc mais NON versionné (cf.
        #    PIPELINE_AUDIT A1) — (re)créé ici pour garantir une cible à ON CONFLICT,
        #    y compris après une reconstruction depuis les seules migrations.
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_literature_document_doi "
        "ON literature_document (doi) WHERE doi IS NOT NULL",
        #  • Titre normalisé : même règle de dédup que _ingest_doc_direct (len≥20),
        #    limitée aux lignes CANONIQUES (is_duplicate NULL/FALSE) — ne rejette pas
        #    les doublons déjà marqués (en attente de purge). Si des collisions
        #    canoniques subsistent, la création échoue et est journalisée : la course
        #    reste alors ouverte jusqu'à ce que la maintenance nettoie les doublons.
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_litdoc_title_norm "
        "ON literature_document (project_context, title_norm) "
        "WHERE title_norm IS NOT NULL AND length(title_norm) >= 20 "
        "AND (is_duplicate IS NULL OR is_duplicate = FALSE)",
    ]
    for ddl in btree:
        try:
            with engine.begin() as conn:
                conn.execute(text(ddl))
        except Exception as e:
            logger.warning(f"_ensure_performance_indexes (btree) « {ddl[:60]}… » : {e}")

    # 2) Index ANN pgvector (HNSW, cosinus). CONCURRENTLY → hors transaction, ne
    # bloque pas les écritures. Accélère « ORDER BY embedding <=> q LIMIT k » qui
    # sinon scanne TOUS les embeddings à chaque requête.
    ann_ddl = ("CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_doc_chunk_embedding_hnsw "
               "ON document_chunk USING hnsw (embedding vector_cosine_ops)")
    try:
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.execute(text(ann_ddl))
        logger.info("Index ANN pgvector (HNSW) vérifié/créé.")
    except Exception as e:
        logger.warning(f"Index ANN pgvector indisponible ({e}); recherche vectorielle en scan séquentiel.")

    # 3) Index GIN TRIGRAMME (pg_trgm) sur titre / résumé / contenu : rendent les
    # `LOWER(COALESCE(x,'')) LIKE '%terme%'` du match booléen index-assistés. Le
    # wildcard EN TÊTE interdit tout index B-tree → sans ceci, chaque terme scanne
    # séquentiellement 207k docs (× termes × facettes) — cause des recherches à
    # plusieurs minutes. CONCURRENTLY + AUTOCOMMIT : build hors transaction, sans
    # bloquer les écritures ; chaque index isolé/tolérant. L'expression indexée est
    # IDENTIQUE à celle de la requête (sinon le planificateur n'utilise pas l'index).
    trgm_ddl = [
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_litdoc_title_trgm "
        "ON literature_document USING gin (LOWER(COALESCE(title,'')) gin_trgm_ops)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_litdoc_abstract_trgm "
        "ON literature_document USING gin (LOWER(COALESCE(abstract,'')) gin_trgm_ops)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_docchunk_content_trgm "
        "ON document_chunk USING gin (LOWER(COALESCE(content,'')) gin_trgm_ops)",
    ]
    for _tddl in trgm_ddl:
        try:
            with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
                conn.execute(text(_tddl))
            logger.info(f"Index trigramme vérifié/créé : « {_tddl[:55]}… ».")
        except Exception as e:
            logger.warning(f"Index trigramme indisponible « {_tddl[:55]}… » ({e}); LIKE en scan séquentiel.")


# ─────────────────────────────────────────────────────────────────────────────
# Startup
# ─────────────────────────────────────────────────────────────────────────────
def _warm_clustering_kernels() -> None:
    """Compile les noyaux numba d'UMAP et d'HDBSCAN sur un jeu minuscule, une fois par
    processus. Sans cela, la PREMIÈRE visualisation après un redémarrage payait ≈ 30 s de
    compilation sous les yeux de l'utilisateur (et le préchauffage figurait au runbook
    comme geste manuel). Best-effort : n'échoue jamais, ne bloque pas le démarrage."""
    import time as _t
    t0 = _t.time()
    try:
        import numpy as _np
        X = _np.random.RandomState(0).rand(60, 16).astype("float32")
        try:
            import umap as _umap
            _umap.UMAP(n_neighbors=8, n_components=2, min_dist=0.1, random_state=42).fit_transform(X)
        except Exception as _e:                              # noqa: BLE001
            logger.info(f"warm-up UMAP: {_e}")
        try:
            import hdbscan as _hdb
            _hdb.HDBSCAN(min_cluster_size=5).fit(X)
        except Exception as _e:                              # noqa: BLE001
            logger.info(f"warm-up HDBSCAN: {_e}")
        logger.info(f"Warm-up clustering kernels: {_t.time() - t0:.1f} s")
    except Exception as _e:                                  # noqa: BLE001
        logger.warning(f"warm-up: {_e}")


def _startup_flag(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() not in ("0", "false", "no", "off")


@app.on_event("startup")
def startup_event() -> None:
    from .scenarios import _pipeline_jobs_lock, _user_scenario_pipeline_jobs  # lazy: scenarios is loaded after this module
    from .pipeline import _run_user_scenario_full_pipeline  # lazy: pipeline is loaded after this module
    from .relevance import _backfill_title_abstract_chunks  # lazy: relevance is loaded after this module
    from .model_training import _seed_demo_scenarios  # lazy: model_training is loaded after this module
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    logger.info("Database connection OK")

    # Index de performance (B-tree + ANN pgvector) : en arrière-plan pour ne pas
    # retarder le démarrage / le health check ; idempotent et sans effet de bord
    # en cas d'échec (au pire, on reste sur le comportement actuel).
    try:
        import threading as _perf_threading
        _perf_threading.Thread(
            target=_ensure_performance_indexes, daemon=True, name="ensure-perf-indexes"
        ).start()
    except Exception as _e:
        logger.warning(f"spawn _ensure_performance_indexes: {_e}")

    # Préchauffage des noyaux UMAP/HDBSCAN (≈ 30 s de compilation numba, en arrière-plan),
    # pour que la première visualisation après ce redémarrage ne les paie pas.
    # WARM_ON_STARTUP=0 pour désactiver (tests, CI).
    if _startup_flag("WARM_ON_STARTUP"):
        try:
            import threading as _warm_threading
            _warm_threading.Thread(target=_warm_clustering_kernels, daemon=True, name="warm-umap").start()
        except Exception as _e:
            logger.warning(f"spawn warm-up: {_e}")

    # Recherche plein texte : remplissage initial puis rafraîchissement de
    # document_search en arrière-plan (cf. lexical_search.py). Tant que le remplissage
    # n'est pas COMPLET, la recherche booléenne reste sur le chemin LIKE.
    try:
        _lex.start_worker()
    except Exception as _e:
        logger.warning(f"spawn document_search worker: {_e}")

    # Scénario de démonstration intégré (dataset RÉEL grippe + modèle entraîné) : rend
    # l'essai « données réelles » visible dans la liste. Idempotent (id stable) et
    # best-effort — en arrière-plan pour ne jamais retarder/casser le démarrage.
    try:
        import threading as _seed_threading
        _seed_threading.Thread(
            target=_seed_demo_scenarios, daemon=True, name="seed-demo"
        ).start()
    except Exception as _e:
        logger.warning(f"spawn _seed_demo_scenarios: {_e}")

    # Pipelines orphelins : tout pipeline marqué 'running' ou 'starting' au
    # démarrage du serveur est forcément mort (le thread a été tué lors du
    # redémarrage précédent).
    # Stratégie : on les relance automatiquement en arrière-plan plutôt que
    # de les marquer 'failed' et forcer l'utilisateur à les relancer manuellement.
    try:
        with engine.connect() as _startup_conn:
            _orphan_rows = _startup_conn.execute(text("""
                SELECT id, query, filters, pipeline_lang
                FROM user_scenarios
                WHERE pipeline_status IN ('running', 'starting')
                  AND COALESCE(is_system, FALSE) = FALSE
            """)).mappings().fetchall()
            _pop_orphan_rows = _startup_conn.execute(text("""
                SELECT id, query, filters, pipeline_lang FROM user_scenarios
                WHERE populate_status = 'running'
                  AND COALESCE(is_system, FALSE) = FALSE
            """)).mappings().fetchall()

        # Recherches orphelines (populate 'running' au moment du redémarrage) : RELANCÉES,
        # même requête et même langue, plutôt que passées à 'error' — la page de recherche
        # qui les attendait retrouve un job en cours, et le pipeline complet enchaîne
        # ensuite comme après toute recherche. RESUME_ON_STARTUP=0 : marquées 'error'.
        _relaunched_pop: set[str] = set()
        if _pop_orphan_rows:
            if _startup_flag("RESUME_ON_STARTUP"):
                from .scenarios import LIVE_MAX_PER_SOURCE, _launch_populate_job  # lazy: scenarios is loaded after this module
                for _po in _pop_orphan_rows:
                    if not _po.get("query"):
                        continue
                    try:
                        _launch_populate_job(_po["id"], _po["query"], _po.get("filters") or {},
                                             LIVE_MAX_PER_SOURCE, True, _po.get("pipeline_lang"))
                        _relaunched_pop.add(_po["id"])
                        logger.warning(f"Startup: recherche {_po['id']} interrompue par le redémarrage → relancée.")
                    except Exception as _e_po:
                        logger.warning(f"Startup: relance populate {_po['id']}: {_e_po}")
            else:
                with engine.begin() as _c:
                    _c.execute(text("""
                        UPDATE user_scenarios
                        SET populate_status = 'error', updated_at = NOW()
                        WHERE populate_status = 'running'
                          AND COALESCE(is_system, FALSE) = FALSE
                    """))
                logger.warning(f"Startup: {len(_pop_orphan_rows)} populate(s) orphelin(s) reinitialisé(s) à 'error'.")

        # Relancer automatiquement les pipelines interrompus (sauf ceux dont la recherche
        # vient d'être relancée : le pipeline enchaînera à la fin de celle-ci).
        _orphan_rows = [r for r in _orphan_rows if r["id"] not in _relaunched_pop]
        if _orphan_rows:
            import threading as _startup_threading
            logger.warning(
                f"Startup: {len(_orphan_rows)} pipeline(s) interrompu(s) détecté(s) — "
                f"relance automatique en arrière-plan."
            )
            for _orphan in _orphan_rows:
                _oid = _orphan["id"]
                _oquery = _orphan["query"] or ""
                _ofilters = _orphan["filters"] or {}
                if not _oquery:
                    # Pas de requête → on ne peut pas relancer, marquer failed
                    with engine.begin() as _c:
                        _c.execute(text("""
                            UPDATE user_scenarios
                            SET pipeline_status = 'failed',
                                pipeline_step   = NULL,
                                updated_at      = NOW()
                            WHERE id = :sid
                        """), {"sid": _oid})
                    logger.warning(f"Startup: pipeline {_oid} sans requête → marqué 'failed'.")
                    continue
                # Initialiser le job en mémoire
                _olang = (_orphan.get("pipeline_lang") or "fr").lower()
                with _pipeline_jobs_lock:
                    _user_scenario_pipeline_jobs[_oid] = {
                        "overall_status": "starting",
                        "current_step": "ingest",
                        "auto_restarted": True,
                        "lang": _olang,
                        "steps": {k: {"status": "pending"} for k in (
                            "ingest", "fulltext", "embed", "rerank", "pico", "metadata",
                            "clustering", "knowledge_graph", "evidence", "variables", "actions")},
                    }
                _t = _startup_threading.Thread(
                    target=_run_user_scenario_full_pipeline,
                    args=(_oid, _oquery, _ofilters),
                    kwargs={"lang": _olang},              # même langue qu'avant l'interruption
                    daemon=True,
                )
                _t.start()
                logger.info(f"Startup: pipeline {_oid} relancé automatiquement (query={_oquery[:60]!r}).")
    except Exception as _se:
        logger.error(f"Startup cleanup/relance pipelines orphelins: {_se}")

    import threading
    # ── Worker d'enrichissement automatique (embedding + PICO) ──────────────
    # Tourne en permanence en arrière-plan. Chaque cycle :
    #   1. Embède tous les chunks title_abstract/fulltext_section sans embedding
    #   2. Extrait le PICO pour tous les articles avec abstract mais sans pico_json
    def _background_enrichment_worker():
        import time as _time
        from llm_usage import MeteredOpenAI as _OAI_bg
        from concurrent.futures import ThreadPoolExecutor as _TPE
        from datetime import datetime, timezone

        _EMBED_BATCH   = 100   # chunks par appel OpenAI embeddings
        _PICO_WORKERS  = 5     # threads parallèles pour extraction PICO
        _CYCLE_SLEEP   = 30    # secondes entre deux cycles
        _PICO_BATCH    = 50    # articles PICO par cycle
        _PICO_MAX_ATTEMPTS = 3 # essais max par article (borne les échecs déterministes)
        _ABS_BATCH     = 50    # notices sans résumé traitées par cycle (backfill)

        _system_pico = (
            "You are a systematic review expert. "
            "Extract PICO elements and return ONLY valid JSON:\n"
            '{"P":"Population","I":"Intervention","C":"Comparator or Not specified",'
            '"O":"Outcome(s)","study_design":"RCT|Cohort|Systematic review|etc",'
            '"pico_confidence":0.0-1.0,"pico_notes":""}\n'
            "Be concise (max 2 sentences per field). Return ONLY the JSON."
        )

        def _extract_pico_one(row, client):
            try:
                title    = row["title"] or ""
                abstract = row["abstract"] or ""
                # Évidence extraite du TEXTE INTÉGRAL si disponible, sinon du résumé.
                # Les chunks fulltext_section sont concaténés dans l'ordre ; on
                # marque la source (`pico_source`) pour pouvoir ré-extraire plus
                # tard les articles dont le PICO venait du résumé seul.
                body_text   = abstract[:3000]
                body_label  = "Abstract"
                pico_source = "abstract"
                if row.get("has_fulltext"):
                    with engine.connect() as _ftc:
                        _ft = _ftc.execute(text("""
                            SELECT string_agg(content, E'\n\n' ORDER BY chunk_index) AS ft
                            FROM document_chunk
                            WHERE document_id = :id AND chunk_type = 'fulltext_section'
                        """), {"id": row["id"]}).scalar()
                    if _ft and len(_ft) > len(abstract):
                        body_text   = _ft[:14000]
                        body_label  = "Full text"
                        pico_source = "fulltext"
                # Appel LLM isolé : une erreur d'API (quota/réseau) est TRANSITOIRE
                # → on ne compte PAS de tentative (le cooldown s'en charge) et on
                # réessaiera quand l'API sera saine.
                try:
                    resp = client.chat.completions.create(
                        model="gpt-4.1-mini",
                        messages=[
                            {"role": "system", "content": _system_pico},
                            {"role": "user",   "content": f"Title: {title}\n\n{body_label}: {body_text}"},
                        ],
                        temperature=0,
                        seed=42,
                        max_tokens=800,  # 400 tronquait le JSON des articles verbeux → json invalide
                        response_format={"type": "json_object"},
                    )
                except Exception as _api_e:
                    if _is_openai_quota_error(_api_e):
                        _trip_openai_cooldown()
                    logger.debug(f"BG PICO API doc {row['id']}: {_api_e}")
                    return None

                # On a une RÉPONSE → on COMPTE la tentative quoi qu'il arrive. Sinon,
                # un article dont la sortie LLM est malformée de façon déterministe
                # (clé manquante / JSON invalide, identique à chaque fois car
                # temperature=0/seed=42) serait ré-extrait à l'infini. pico_attempts
                # (plafonné par _PICO_MAX_ATTEMPTS dans le sélecteur) borne ces échecs.
                _pico = None
                try:
                    _pico = json.loads(resp.choices[0].message.content)
                except Exception:
                    _pico = None
                with engine.begin() as _c:
                    if isinstance(_pico, dict):
                        # TOLÉRANT : on complète les clés manquantes plutôt que de
                        # rejeter (et ré-essayer en boucle). Une extraction partielle
                        # à faible confiance vaut mieux qu'un article jamais traité —
                        # l'aval filtre déjà sur pico_confidence. Seul un JSON INVALIDE
                        # (rare avec response_format json_object + max_tokens relevé)
                        # est compté comme échec à borner.
                        for _k in ("P", "I", "C", "O"):
                            _pico.setdefault(_k, "")
                        _pico.setdefault("study_design", "non précisé")
                        try:
                            _pico["pico_confidence"] = float(_pico.get("pico_confidence", 0.3))
                        except (TypeError, ValueError):
                            _pico["pico_confidence"] = 0.3
                        _pico["pico_notes"]  = _pico.get("pico_notes", "")
                        _pico["pico_source"] = pico_source
                        _c.execute(text("""
                            UPDATE literature_document
                            SET pico_json = CAST(:pico AS jsonb),
                                pico_extracted_at = :ts,
                                pico_fulltext_attempted = CASE WHEN :hf THEN TRUE
                                                               ELSE pico_fulltext_attempted END,
                                pico_attempts = COALESCE(pico_attempts, 0) + 1
                            WHERE id = :doc_id
                        """), {
                            "pico":   json.dumps(_pico),
                            "ts":     datetime.now(timezone.utc),
                            "hf":     bool(row.get("has_fulltext")),
                            "doc_id": row["id"],
                        })
                        return row["id"]
                    # JSON invalide : on compte la tentative (+ fulltext_attempted)
                    # pour borner les ré-essais → fin de la boucle de tokens.
                    _c.execute(text("""
                        UPDATE literature_document
                        SET pico_attempts = COALESCE(pico_attempts, 0) + 1,
                            pico_fulltext_attempted = CASE WHEN :hf THEN TRUE
                                                           ELSE pico_fulltext_attempted END
                        WHERE id = :doc_id
                    """), {"hf": bool(row.get("has_fulltext")), "doc_id": row["id"]})
                    return None
            except Exception as _pe:
                logger.debug(f"BG PICO doc {row['id']}: {_pe}")
                return None

        def _europepmc_abstracts_by_doi(dois: list[str]) -> dict[str, str]:
            """Récupère le résumé via EuropePMC pour une liste de DOI (une requête
            OR groupée). EuropePMC agrège MEDLINE + PMC : meilleure couverture
            DOI→abstract que Crossref/OpenAlex. Renvoie {doi_minuscule: abstract}."""
            import requests as _rq
            out: dict[str, str] = {}
            _dois = [d for d in dois if d]
            if not _dois:
                return out
            q = " OR ".join(f'DOI:"{d}"' for d in _dois)
            try:
                _r = _rq.get(
                    "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                    params={"query": f"({q})", "resultType": "core",
                            "format": "json", "pageSize": len(_dois)},
                    timeout=30,
                )
                _r.raise_for_status()
                for _res in ((_r.json().get("resultList") or {}).get("result") or []):
                    _d = (_res.get("doi") or "").lower().strip()
                    _ab = _res.get("abstractText")
                    if _d and _ab:
                        _ab = re.sub(r"<[^>]+>", " ", _ab)      # retirer le JATS/HTML
                        _ab = re.sub(r"\s+", " ", _ab).strip()
                        if len(_ab) >= 30:
                            out[_d] = _ab
            except Exception as _ee:
                logger.debug(f"EuropePMC abstract batch: {_ee}")
            return out

        # Colonne de suivi : évite de re-tenter indéfiniment les notices dont
        # EuropePMC n'a pas de résumé (sinon le même lot bloquerait la file).
        try:
            with engine.begin() as _cc:
                _cc.execute(text(
                    "ALTER TABLE literature_document "
                    "ADD COLUMN IF NOT EXISTS abstract_backfill_attempted BOOLEAN DEFAULT FALSE"
                ))
                # Garde-fou anti-boucle : ne ré-extraire le PICO « texte intégral »
                # qu'UNE SEULE fois par article. Sans cela, tout article avec
                # has_fulltext=TRUE mais SANS chunk fulltext exploitable garde
                # pico_source='abstract' et re-matche le sélecteur PICO à CHAQUE
                # cycle (30 s) → ré-extraction gpt-4.1-mini infinie = fuite de tokens.
                _cc.execute(text(
                    "ALTER TABLE literature_document "
                    "ADD COLUMN IF NOT EXISTS pico_fulltext_attempted BOOLEAN DEFAULT FALSE"
                ))
                # Compteur de tentatives PICO. Un article dont l'extraction échoue
                # de façon DÉTERMINISTE (sortie LLM malformée : clé manquante ou JSON
                # invalide) revenait dans le sélecteur à CHAQUE cycle et était ré-envoyé
                # à gpt-4.1-mini indéfiniment (même échec, temperature=0/seed=42) →
                # 2e fuite de tokens. On borne à _PICO_MAX_ATTEMPTS essais par article.
                _cc.execute(text(
                    "ALTER TABLE literature_document "
                    "ADD COLUMN IF NOT EXISTS pico_attempts INTEGER DEFAULT 0"
                ))
        except Exception as _ce:
            logger.warning(f"ensure backfill/pico-attempt columns: {_ce}")

        logger.info("Background enrichment worker started (abstract backfill + embedding + PICO).")
        while True:
            try:
                openai_key = os.getenv("OPENAI_API_KEY")
                if not openai_key:
                    _time.sleep(_CYCLE_SLEEP)
                    continue

                _client = _OAI_bg(api_key=openai_key, timeout=90.0)

                # ── 0. BACKFILL DES RÉSUMÉS (notices sans abstract, via DOI) ──
                # Beaucoup de notices Crossref/OpenAlex arrivent sans résumé. On
                # tente de le récupérer via EuropePMC (par DOI) pour les rendre
                # exploitables (puis embedding + PICO par les étapes suivantes).
                try:
                    with engine.connect() as _conn:
                        _stub_rows = _conn.execute(text("""
                            SELECT id, doi FROM literature_document
                            WHERE project_context = 'literev'
                              AND doi IS NOT NULL
                              AND (abstract IS NULL OR length(trim(abstract)) < 30)
                              AND abstract_backfill_attempted IS NOT TRUE
                            ORDER BY id LIMIT :lim
                        """), {"lim": _ABS_BATCH}).mappings().fetchall()
                    if _stub_rows:
                        _doi_map: dict[str, list[int]] = {}
                        for _r in _stub_rows:
                            _doi_map.setdefault((_r["doi"] or "").lower().strip(), []).append(_r["id"])
                        _dois = [d for d in _doi_map if d]
                        _found: dict[str, str] = {}
                        for _k in range(0, len(_dois), 20):       # 20 DOI / requête
                            _found.update(_europepmc_abstracts_by_doi(_dois[_k:_k + 20]))
                        _filled = 0
                        with engine.begin() as _cu:
                            for _d, _ab in _found.items():
                                for _docid in _doi_map.get(_d, []):
                                    _cu.execute(text("""
                                        UPDATE literature_document SET abstract = :ab
                                        WHERE id = :id
                                          AND (abstract IS NULL OR length(trim(abstract)) < 30)
                                    """), {"ab": _ab, "id": _docid})
                                    _filled += 1
                            # Marquer TOUTES les notices tentées (trouvées ou non).
                            _cu.execute(
                                text("UPDATE literature_document SET abstract_backfill_attempted = TRUE "
                                     "WHERE id IN :ids").bindparams(bindparam("ids", expanding=True)),
                                {"ids": [_r["id"] for _r in _stub_rows]},
                            )
                        if _found:
                            # Créer les chunks title_abstract des docs nouvellement dotés
                            # d'un résumé (l'embedding ci-dessous les vectorisera).
                            _backfill_title_abstract_chunks()
                            logger.info(f"BG abstract backfill: {_filled} résumés récupérés (EuropePMC).")
                except Exception as _abe:
                    logger.warning(f"BG abstract backfill error: {_abe}")

                # ── 1. EMBEDDING ──────────────────────────────────────────────
                # On embède TOUS les chunks standard sans embedding — y compris le
                # title_abstract des docs à texte intégral : le résumé sert de vecteur
                # représentatif CONSTANT au clustering (même base pour chaque document)
                # pour un coût négligeable. (Avant, on le sautait pour les docs full-text
                # → incohérence : la plupart embeddés, ~153 non, et clustering par 1re
                # section au lieu du résumé.)
                with engine.connect() as _conn:
                    _chunks = _conn.execute(text("""
                        SELECT c.id, c.content
                        FROM document_chunk c
                        WHERE c.embedding IS NULL
                          AND c.chunk_type IN ('title_abstract', 'fulltext_section')
                          AND LENGTH(c.content) > 20
                          AND COALESCE(c.embedding_attempts, 0) < 3
                        ORDER BY c.chunk_type DESC, c.id
                        LIMIT 500
                    """)).mappings().fetchall()

                if _chunks and not _openai_in_cooldown():
                    try:
                        _emb_done, _failed = _embed_chunks_resilient(_client, list(_chunks))
                    except Exception:               # quota propagé → pause + réessai après cooldown
                        _trip_openai_cooldown()
                        _emb_done, _failed = 0, []
                    if _failed:
                        # Chunks refusés par l'API (hors quota) : incrémenter le compteur
                        # pour les sortir de la file après 3 essais (sinon « en attente »
                        # éternel + lot ré-échoué à chaque cycle).
                        with engine.begin() as _cu:
                            _cu.execute(
                                text("UPDATE document_chunk SET embedding_attempts = "
                                     "COALESCE(embedding_attempts,0)+1 WHERE id IN :ids")
                                .bindparams(bindparam("ids", expanding=True)),
                                {"ids": _failed},
                            )
                    if _emb_done:
                        logger.info(f"BG worker: {_emb_done} chunks embedded.")

                # ── 2. PICO ───────────────────────────────────────────────────
                with engine.connect() as _conn:
                    # On extrait le PICO des articles sans PICO, PUIS on ré-extrait
                    # ceux dont le PICO venait du résumé alors que le texte intégral
                    # est désormais disponible (pico_source != 'fulltext'). Les
                    # articles jamais traités passent en premier.
                    _pico_rows = _conn.execute(text("""
                        SELECT id, title, abstract, has_fulltext
                        FROM literature_document
                        WHERE project_context = 'literev'
                          AND abstract IS NOT NULL
                          AND LENGTH(abstract) > 50
                          AND COALESCE(pico_attempts, 0) < :max_attempts  -- borne les échecs déterministes
                          -- PICO UNIQUEMENT pour les articles appartenant à un scénario
                          -- réel (article_scenarios). Le PICO n'est affiché QUE par
                          -- scénario : extraire les ~milliers de documents orphelins /
                          -- hors-scénario du corpus était du pur gaspillage de tokens.
                          AND EXISTS (
                              SELECT 1 FROM article_scenarios ars
                              WHERE ars.document_id = literature_document.id
                          )
                          AND (
                            pico_json IS NULL
                            OR (
                                has_fulltext IS TRUE
                                AND (pico_json->>'pico_source') IS DISTINCT FROM 'fulltext'
                                AND pico_fulltext_attempted IS NOT TRUE  -- une seule tentative
                            )
                          )
                        ORDER BY (pico_json IS NULL) DESC, id
                        LIMIT :lim
                    """), {"lim": _PICO_BATCH, "max_attempts": _PICO_MAX_ATTEMPTS}).mappings().fetchall()

                # Coupe-circuit : PICO_AUTOEXTRACT_ENABLED=0 dans /etc/literev-api.env
                # (puis restart) met en pause l'extraction PICO automatique sans
                # toucher au code — utile pour stopper net la dépense OpenAI.
                _pico_enabled = os.getenv("PICO_AUTOEXTRACT_ENABLED", "1").strip().lower() not in ("0", "false", "no", "off")
                if _pico_rows and _pico_enabled and not _openai_in_cooldown():
                    _pico_done = 0
                    with _TPE(max_workers=_PICO_WORKERS) as _pool:
                        _futs = {_pool.submit(_extract_pico_one, r, _client): r["id"] for r in _pico_rows}
                        for _f in _futs:
                            if _f.result() is not None:
                                _pico_done += 1
                    if _pico_done:
                        logger.info(f"BG worker: {_pico_done} PICO extracted.")

            except Exception as _we:
                logger.error(f"BG enrichment worker error: {_we}")

            _time.sleep(_CYCLE_SLEEP)

    threading.Thread(target=_background_enrichment_worker, daemon=True, name="bg-enrichment").start()
    logger.info("Background enrichment worker launched.")


# ─── DOUBLE-AVEUGLE SCREENING + KAPPA DE COHEN ───────────────────────────────

def _ensure_double_blind_columns():
    """Crée les colonnes reviewer_1_status/reviewer_2_status si elles n'existent pas."""
    with engine.begin() as conn:
        conn.execute(text("""
            ALTER TABLE literature_document
            ADD COLUMN IF NOT EXISTS reviewer_1_status  VARCHAR(20) DEFAULT NULL,
            ADD COLUMN IF NOT EXISTS reviewer_1_reason  TEXT        DEFAULT NULL,
            ADD COLUMN IF NOT EXISTS reviewer_2_status  VARCHAR(20) DEFAULT NULL,
            ADD COLUMN IF NOT EXISTS reviewer_2_reason  TEXT        DEFAULT NULL,
            ADD COLUMN IF NOT EXISTS kappa_resolved     BOOLEAN     DEFAULT FALSE,
            ADD COLUMN IF NOT EXISTS kappa_final_status VARCHAR(20) DEFAULT NULL
        """))
        # Par SCÉNARIO (article_scenarios) : mêmes colonnes, car un document
        # appartenant à plusieurs scénarios doit porter une décision double-aveugle
        # DISTINCTE par scénario (sinon les kappa se contaminent entre scénarios).
        # Miroir de la migration Alembic e2f6a8b3c5d7 (belt-and-suspenders au boot).
        if conn.execute(text("SELECT to_regclass('public.article_scenarios')")).scalar():
            conn.execute(text("""
                ALTER TABLE article_scenarios
                ADD COLUMN IF NOT EXISTS reviewer_1_status  VARCHAR(20),
                ADD COLUMN IF NOT EXISTS reviewer_1_reason  TEXT,
                ADD COLUMN IF NOT EXISTS reviewer_2_status  VARCHAR(20),
                ADD COLUMN IF NOT EXISTS reviewer_2_reason  TEXT,
                ADD COLUMN IF NOT EXISTS kappa_resolved     BOOLEAN DEFAULT FALSE,
                ADD COLUMN IF NOT EXISTS kappa_final_status VARCHAR(20)
            """))
    logger.info("Colonnes double-aveugle (global + par scénario) vérifiées/créées.")

# Appel au démarrage
try:
    _ensure_double_blind_columns()
except Exception as _e:
    logger.warning(f"_ensure_double_blind_columns: {_e}")


def _ensure_dedup_columns():
    """Colonne title_norm (titre normalisé) pour la déduplication inter-sources par
    TITRE — en complément du DOI, afin de capter les doublons SANS DOI (préprints,
    essais) ou dont le DOI diffère d'une source à l'autre. L'ALTER est instantané ; le
    backfill des lignes existantes + l'index se font en arrière-plan
    (_ensure_performance_indexes)."""
    with engine.begin() as conn:
        conn.execute(text(
            "ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS title_norm TEXT"))
        # Colonnes de déduplication historiquement posées par des scripts ad-hoc
        # (scripts/archive/*.sql) — garanties ici pour que TOUTE base (prod comme base
        # reconstruite/CI) porte : `pmid` (renseignée à l'ingest → dédup PMID), et
        # `is_duplicate`/`canonical_id` (marquage + statut de dédup, filtrés par ~20
        # requêtes existantes). ADD COLUMN IF NOT EXISTS : no-op si déjà présentes.
        conn.execute(text(
            "ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS pmid TEXT"))
        conn.execute(text(
            "ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS is_duplicate BOOLEAN DEFAULT FALSE"))
        conn.execute(text(
            "ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS canonical_id BIGINT"))
    logger.info("Colonnes de déduplication (title_norm, pmid, is_duplicate, canonical_id) vérifiées/créées.")


try:
    _ensure_dedup_columns()
except Exception as _e:
    logger.warning(f"_ensure_dedup_columns: {_e}")


def _ensure_bibliographic_columns():
    """Colonnes bibliographiques / d'affichage historiquement posées par un script ad-hoc
    (scripts/archive/migrate_add_bibliographic_columns.sql) — donc ABSENTES d'une base
    reconstruite depuis schema.sql seul (CI incluse). On les garantit ici pour que le
    détail document enrichi (auteurs / revue / DOI / pays / devis / type d'article) ET les
    requêtes corpus existantes qui les lisent fonctionnent partout. ADD COLUMN IF NOT
    EXISTS : no-op si déjà présentes."""
    stmts = [
        "ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS authors TEXT",
        "ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS journal TEXT",
        "ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS doi TEXT",
        "ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS country TEXT",
        "ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS study_design TEXT",
        "ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS publication_type TEXT",
        # ── Colonnes de screening / qualité / enrichissement ────────────────────
        # Même situation que ci-dessus : lues massivement par le code (screening_status
        # ≈95 références, pico_json ≈24) mais absentes de schema.sql, donc introuvables
        # sur toute base neuve. `screening_reason`/`notes` sont, elles, lues par la
        # migration c8d4e2f1a9b3 qui recopie le verdict global vers article_scenarios.
        "ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS screening_status TEXT",
        "ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS screening_reason TEXT",
        "ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS screening_notes TEXT",
        "ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS screened_at TIMESTAMP",
        "ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS pico_json JSONB",
        "ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS metadata_json JSONB",
        # Concepts typés normalisés par le LLM (carte des concepts) — une fois par article.
        "ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS concepts_json JSONB",
        "ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS citation_count INTEGER",
        "ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS quality_score DOUBLE PRECISION",
        "ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS keywords TEXT",
        "ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS language TEXT",
        "ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS sample_size INTEGER",
        "ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS open_access BOOLEAN",
        # Lues par de vraies requêtes SQL (SELECT id, title, abstract, has_fulltext… ;
        # UPDATE … SET pico_extracted_at = :ts) et créées nulle part non plus.
        "ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS has_fulltext BOOLEAN DEFAULT FALSE",
        "ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS pico_extracted_at TIMESTAMP",
    ]
    # Une instruction par transaction : une seule qui échoue ne doit pas faire annuler
    # les précédentes (cf. _exec_ddl_isolated).
    _exec_ddl_isolated(stmts, "_ensure_bibliographic_columns")
    logger.info("Colonnes bibliographiques, de screening et de qualité vérifiées/créées.")


try:
    _ensure_bibliographic_columns()
except Exception as _e:
    logger.warning(f"_ensure_bibliographic_columns: {_e}")
