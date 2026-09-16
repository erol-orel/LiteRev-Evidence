"""Semantic scoring, cross-encoder rerank, threshold settings, relevant articles.

Extracted from main.py (LiteRev API); `main` re-exports everything for the scripts,
tools and tests.
"""
from __future__ import annotations

import os
from typing import Any

from fastapi import Depends, HTTPException
from sqlalchemy import text

from .core import (
    _is_openai_quota_error,
    _job_is_active,
    _openai_in_cooldown,
    _trip_openai_cooldown,
    app,
    engine,
    logger,
    require_api_key,
)
from .scenario_store import DEFAULT_SIMILARITY_THRESHOLD, _get_scenario_threshold, _get_user_scenario_or_404
from .search import (
    _boolean_corpus_ids,
    _dedup_scenario_links,
    _generate_search_strategy,
    _prisma_identification_figures,
    _set_scenario_corpus,
    _store_prisma_identification,
)
from .gesica import _gesica_title, _get_db_gesica_scenario_or_404

def _run_semantic_rerank_inline(scenario_id: str, query: str) -> int:
    """Score sémantique (cosinus requête↔article) du corpus, mis dans similarity_score.

    Optimisé : on RÉUTILISE les embeddings pgvector déjà stockés
    (document_chunk.embedding) et on calcule le cosinus EN BASE en UNE requête
    (au lieu de ré-embedder chaque résumé via OpenAI + cosinus Python + une
    transaction par article - ce qui rendait l'étape très lente). On ne ré-embedde
    via OpenAI QUE les articles fraîchement ingérés dont les chunks ne sont pas
    encore vectorisés (minorité)."""
    try:
        from llm_usage import MeteredOpenAI as _OAI
        _client = _OAI(timeout=90.0)
        q_emb = _client.embeddings.create(model="text-embedding-3-small", input=query[:2000]).data[0].embedding
        q_str = str(q_emb)

        # 1) Rapide : cosinus pgvector en base pour tous les docs déjà vectorisés.
        with engine.begin() as _c:
            n_fast = _c.execute(text("""
                UPDATE article_scenarios asn
                SET similarity_score = sub.sim
                FROM (
                    SELECT c.document_id,
                           MAX(1 - (c.embedding <=> CAST(:q AS vector))) AS sim
                    FROM document_chunk c
                    JOIN article_scenarios a
                      ON a.document_id = c.document_id AND a.scenario_id = :sid
                    WHERE c.embedding IS NOT NULL
                      AND c.chunk_type IN ('title_abstract', 'fulltext_section')
                    GROUP BY c.document_id
                ) sub
                WHERE asn.scenario_id = :sid AND asn.document_id = sub.document_id
            """), {"q": q_str, "sid": scenario_id}).rowcount or 0

        # 2) Repli OpenAI pour les articles encore non scorés (chunks pas encore
        #    vectorisés). On boucle par lots de 100 jusqu'à épuisement (PLUS de
        #    plafond à 1000) pour qu'AUCUN article pertinent ne reste non scoré ;
        #    chaque lot ré-interroge les NULL restants. Borne de sécurité pour
        #    éviter une boucle infinie si un lot ne parvenait pas à s'écrire.
        n_slow = 0
        if not _openai_in_cooldown():
            import numpy as _np
            _q = _np.asarray(q_emb, dtype=float)
            _qn = float(_np.linalg.norm(_q)) or 1.0
            _MAX_FALLBACK_BATCHES = 200  # 200 × 100 = 20 000 articles / scénario
            for _bi in range(_MAX_FALLBACK_BATCHES):
                if _openai_in_cooldown():
                    break
                with engine.connect() as _conn:
                    batch = _conn.execute(text("""
                        SELECT ld.id, ld.title, ld.abstract
                        FROM literature_document ld
                        JOIN article_scenarios asn ON asn.document_id = ld.id AND asn.scenario_id = :sid
                        WHERE asn.similarity_score IS NULL
                          AND ld.abstract IS NOT NULL AND length(ld.abstract) > 30
                        ORDER BY ld.id LIMIT 100
                    """), {"sid": scenario_id}).mappings().fetchall()
                if not batch:
                    break
                texts = [f"{r['title']}\n\n{(r['abstract'] or '')[:1500]}" for r in batch]
                try:
                    emb = _client.embeddings.create(model="text-embedding-3-small", input=texts).data
                    ups = []
                    for j, e in enumerate(emb):
                        _d = _np.asarray(e.embedding, dtype=float)
                        sim = float(_q @ _d) / (_qn * (float(_np.linalg.norm(_d)) or 1.0))
                        ups.append({"score": max(0.0, min(1.0, sim)), "doc_id": batch[j]["id"], "sid": scenario_id})
                    with engine.begin() as _c:
                        _c.execute(text("""
                            UPDATE article_scenarios SET similarity_score = :score
                            WHERE document_id = :doc_id AND scenario_id = :sid
                        """), ups)
                    n_slow += len(ups)
                except Exception as _e:
                    logger.warning(f"Rerank fallback batch {_bi}: {_e}")
                    if _is_openai_quota_error(_e):
                        _trip_openai_cooldown()
                    break  # toute erreur : on arrête (évite une boucle infinie)
            else:
                logger.warning(f"Rerank {scenario_id}: plafond de repli atteint "
                               f"({_MAX_FALLBACK_BATCHES * 100} articles) - certains peuvent rester non scorés.")
        logger.info(f"Rerank {scenario_id}: {n_fast} via pgvector + {n_slow} via OpenAI (fallback).")
        return n_fast + n_slow
    except Exception as _e:
        logger.error(f"Rerank inline {scenario_id} fatal: {_e}", exc_info=True)
        return 0


def _cohere_rerank(query: str, docs: list[str], model: str = "rerank-v3.5") -> list[float] | None:
    """Cross-encoder rerank via l'API Cohere. Renvoie un score de pertinence par
    document (aligné sur `docs`), ou None si pas de clé / échec. Pas de dépendance
    Python ajoutée : appel REST direct. Activé seulement si COHERE_API_KEY est défini.
    """
    key = os.getenv("COHERE_API_KEY")
    if not key or not docs:
        return None
    try:
        import requests as _rq
        resp = _rq.post(
            "https://api.cohere.com/v2/rerank",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": model, "query": query[:4000], "documents": docs, "top_n": len(docs)},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        scores: list[float | None] = [None] * len(docs)
        for item in data.get("results", []):
            idx = item.get("index")
            if idx is not None and 0 <= idx < len(docs):
                scores[idx] = float(item.get("relevance_score", 0.0))
        return scores  # type: ignore[return-value]
    except Exception as _e:
        logger.warning(f"Cohere rerank failed: {_e}")
        return None


def _run_cross_encoder_rerank(scenario_id: str, query: str, top_k: int = 1000) -> int:
    """Reranke le sous-ensemble PERTINENT (cosinus >= seuil) avec un cross-encoder
    (Cohere). La SÉLECTION reste pilotée par le cosinus + seuil ; on ne fait
    qu'AMÉLIORER l'ORDRE des articles pertinents (précision). No-op sans clé Cohere.
    """
    if not os.getenv("COHERE_API_KEY"):
        return 0
    try:
        eff_threshold = 0.45
        with engine.connect() as _tc:
            _ts = _tc.execute(text(
                "SELECT similarity_threshold FROM scenario_settings WHERE scenario_id = :sid"
            ), {"sid": scenario_id}).scalar()
            if _ts is not None:
                eff_threshold = float(_ts)
            rows = _tc.execute(text("""
                SELECT ld.id, ld.title, ld.abstract
                FROM literature_document ld
                JOIN article_scenarios asn ON asn.document_id = ld.id AND asn.scenario_id = :sid
                WHERE COALESCE(asn.similarity_score, 0.0) >= :thr
                  AND ld.abstract IS NOT NULL AND length(ld.abstract) > 30
                ORDER BY asn.similarity_score DESC NULLS LAST
                LIMIT :k
            """), {"sid": scenario_id, "thr": eff_threshold, "k": top_k}).mappings().all()
        if not rows:
            return 0
        docs = [f"{r['title']}\n\n{(r['abstract'] or '')[:1500]}" for r in rows]
        scores = _cohere_rerank(query, docs)
        if not scores:
            return 0
        updated = 0
        with engine.begin() as _c:
            for r, s in zip(rows, scores):
                if s is None:
                    continue
                _c.execute(text("""
                    UPDATE article_scenarios SET rerank_score = :s
                    WHERE document_id = :doc_id AND scenario_id = :sid
                """), {"s": s, "doc_id": r["id"], "sid": scenario_id})
                updated += 1
        logger.info(f"Cross-encoder rerank {scenario_id}: {updated} articles pertinents réordonnés.")
        return updated
    except Exception as _e:
        logger.warning(f"Cross-encoder rerank {scenario_id} failed: {_e}")
        return 0


# ═══════════════════════════════════════════════════════════════
# SCORING SÉMANTIQUE + EVIDENCE BRIEF LLM + VARIABLES + HEATMAP
# ═══════════════════════════════════════════════════════════════

# ─── SCORING SÉMANTIQUE POST-INGESTION ───────────────────────────────────────

_RERANK_JOBS: dict[str, dict] = {}


def _ensure_scenario_settings_table():
    """Table pour stocker les paramètres par scénario (seuil, etc.)."""
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS scenario_settings (
                scenario_id     VARCHAR(80) PRIMARY KEY,
                similarity_threshold FLOAT DEFAULT 0.45,
                evidence_brief_json  JSONB DEFAULT NULL,
                brief_generated_at   TIMESTAMP DEFAULT NULL,
                variables_json       JSONB DEFAULT NULL,
                variables_validated  BOOLEAN DEFAULT FALSE,
                variables_generated_at TIMESTAMP DEFAULT NULL,
                updated_at           TIMESTAMP DEFAULT NOW()
            )
        """))
    logger.info("Table scenario_settings vérifiée/créée.")

try:
    _ensure_scenario_settings_table()
except Exception as _e:
    logger.warning(f"_ensure_scenario_settings_table: {_e}")


def _embed_query_vector(query: str) -> str | None:
    """Embedding d'une requête → chaîne vecteur pgvector « [x,y,…] », ou None si
    indisponible (pas de clé, cooldown quota, ou erreur). Sert à classer les chunks
    de texte intégral par pertinence à la requête du scénario."""
    if not query or not query.strip():
        return None
    if not os.getenv("OPENAI_API_KEY") or _openai_in_cooldown():
        return None
    try:
        from llm_usage import MeteredOpenAI as _OAI2
        _emb = _OAI2(api_key=os.getenv("OPENAI_API_KEY"), timeout=30.0).embeddings.create(
            input=[query.replace("\n", " ").strip()[:2000]],
            model="text-embedding-3-small",
        ).data[0].embedding
        return "[" + ",".join(str(x) for x in _emb) + "]"
    except Exception as _e:
        if _is_openai_quota_error(_e):
            _trip_openai_cooldown()
        logger.warning(f"_embed_query_vector: {_e}")
        return None


def _fetch_fulltext_excerpts(top_ids: list, query_emb: str | None,
                             char_cap: int, chunks_per_doc: int) -> dict:
    """Pour chaque doc, concatène ses chunks `fulltext_section` les PLUS PERTINENTS
    à la requête (cosinus sur l'embedding déjà stocké du chunk), jusqu'à `char_cap`.
    Dépenser le budget de tokens sur les passages pertinents (méthodes/résultats)
    plutôt que sur les premiers caractères (souvent l'intro) : meilleure profondeur
    à coût égal. Repli sur l'ordre du document (par id) pour les docs dont les chunks
    ne sont pas (encore) embeddés, ou si aucun embedding de requête n'est disponible."""
    result: dict = {}
    if not top_ids:
        return result
    # 1) Sélection par pertinence (nécessite un embedding de requête + chunks embeddés)
    if query_emb is not None:
        try:
            with engine.connect() as conn:
                rows = conn.execute(text("""
                    SELECT document_id, string_agg(content, E'\n\n' ORDER BY rn) AS fulltext
                    FROM (
                        SELECT document_id, content,
                               ROW_NUMBER() OVER (PARTITION BY document_id
                                   ORDER BY embedding <=> CAST(:qemb AS vector)) AS rn
                        FROM document_chunk
                        WHERE document_id = ANY(CAST(:ids AS bigint[]))
                          AND chunk_type = 'fulltext_section'
                          AND embedding IS NOT NULL
                          AND content IS NOT NULL AND length(TRIM(content)) > 0
                    ) ranked
                    WHERE rn <= :cpd
                    GROUP BY document_id
                """), {"ids": top_ids, "qemb": query_emb, "cpd": chunks_per_doc}).mappings().fetchall()
            result = {r["document_id"]: (r["fulltext"] or "")[:char_cap] for r in rows}
        except Exception as _e:
            logger.warning(f"_fetch_fulltext_excerpts relevance: {_e}")
    # 2) Repli par ordre de document pour les docs sans résultat pertinent.
    _missing = [i for i in top_ids if i not in result]
    if _missing:
        try:
            with engine.connect() as conn:
                rows = conn.execute(text("""
                    SELECT document_id, string_agg(content, E'\n\n' ORDER BY id) AS fulltext
                    FROM document_chunk
                    WHERE document_id = ANY(CAST(:ids AS bigint[]))
                      AND chunk_type = 'fulltext_section'
                      AND content IS NOT NULL AND length(TRIM(content)) > 0
                    GROUP BY document_id
                """), {"ids": _missing}).mappings().fetchall()
            for r in rows:
                result[r["document_id"]] = (r["fulltext"] or "")[:char_cap]
        except Exception as _e:
            logger.warning(f"_fetch_fulltext_excerpts fallback: {_e}")
    return result


def _get_above_threshold_articles(scenario_id: str, threshold: float | None = None,
                                  include_fulltext: bool = False,
                                  fulltext_query: str | None = None,
                                  fulltext_top_docs: int = 25,
                                  fulltext_char_cap: int = 2500,
                                  fulltext_chunks_per_doc: int = 5,
                                  full_rows: int | None = None,
                                  require_pico: bool = False) -> list[dict]:
    """
    Retourne les articles au-dessus du seuil de similarité OU validés humainement.
    Priorité : included > similarity_score >= threshold > autres.

    include_fulltext=True attache à chaque article un champ `fulltext` : un extrait
    du TEXTE INTÉGRAL (chunks `fulltext_section`) pour les `fulltext_top_docs` plus
    pertinents. Quand `fulltext_query` est fourni, on choisit par doc les
    `fulltext_chunks_per_doc` chunks les PLUS PERTINENTS à cette requête (et non les
    premiers), plafonné à `fulltext_char_cap` caractères - le budget de tokens est
    ainsi dépensé sur les passages utiles. Les documents sans texte intégral gardent
    `fulltext=""` (title+abstract seuls).

    `full_rows=N` : TOUS les articles pertinents sont renvoyés (ids, année, statut,
    devis, `has_pico`… - de quoi compter et prendre l'empreinte du corpus) mais seuls
    les N premiers portent `abstract` et `pico_json`. Les générateurs LLM n'utilisent
    que 20 à 30 articles : charger 25 000 résumés + PICO (120 Mo) pour en lire 30 était
    inutile, et ces générateurs tournent en parallèle. `require_pico=True` restreint
    aux articles disposant d'un PICO extrait.
    """
    if threshold is None:
        threshold = _get_scenario_threshold(scenario_id)
    _fr = -1 if full_rows is None else max(0, int(full_rows))
    with engine.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT id, title, year, journal, authors, doi, study_design, citation_count,
                   screening_status, quality_score, similarity_score, has_pico,
                   CASE WHEN :fr < 0 OR rn <= :fr THEN abstract END AS abstract,
                   CASE WHEN :fr < 0 OR rn <= :fr THEN pico_json END AS pico_json
            FROM (
                SELECT ld.id, ld.title, ld.abstract, ld.year, ld.journal, ld.authors, ld.doi,
                       ld.study_design, ld.pico_json, ld.citation_count,
                       COALESCE(asn.screening_status, ld.screening_status) AS screening_status,
                       ld.quality_score, asn.similarity_score,
                       (ld.pico_json IS NOT NULL) AS has_pico,
                       ROW_NUMBER() OVER (ORDER BY
                           CASE WHEN COALESCE(asn.screening_status, ld.screening_status) = 'included' THEN 0 ELSE 1 END,
                           asn.similarity_score DESC NULLS LAST,
                           ld.citation_count DESC NULLS LAST, ld.id) AS rn
                FROM literature_document ld
                JOIN article_scenarios asn ON asn.document_id = ld.id AND asn.scenario_id = :sid
                WHERE ld.project_context = 'literev'
                  AND ld.is_duplicate IS NOT TRUE
                  -- Porte de screening (C1) : ne jamais alimenter le modèle avec un
                  -- article explicitement exclu (les autres statuts restent admis).
                  AND COALESCE(asn.screening_status, ld.screening_status) IS DISTINCT FROM 'excluded'
                  -- Décision produit : un article NON scoré (similarity_score NULL)
                  -- n'est PAS pertinent - même définition que tous les affichages
                  -- (COALESCE(score,0) >= seuil). On garde le rattrapage 'included'.
                  AND (
                      COALESCE(asn.screening_status, ld.screening_status) = 'included'
                      OR COALESCE(asn.similarity_score, 0) >= :threshold
                  )
                  {"AND ld.pico_json IS NOT NULL" if require_pico else ""}
            ) ranked
            ORDER BY rn
        """), {"sid": scenario_id, "threshold": threshold, "fr": _fr}).mappings().fetchall()
    articles = [dict(r) for r in rows]
    if include_fulltext and articles:
        _top_ids = [a["id"] for a in articles[:fulltext_top_docs]]
        _qemb = _embed_query_vector(fulltext_query) if fulltext_query else None
        _ft = _fetch_fulltext_excerpts(_top_ids, _qemb, fulltext_char_cap, fulltext_chunks_per_doc)
        for a in articles:
            a["fulltext"] = _ft.get(a["id"], "")
    return articles


def _evidence_fingerprint(doc_ids: list, threshold: float | None, lang: str | None, ctx: str) -> str:
    """Empreinte du corpus pertinent d'un scénario : hash de l'ensemble ORDONNÉ des
    IDs de documents + seuil + langue + version de contexte. Sert de clé de cache
    « ne pas régénérer si le corpus n'a pas changé » pour l'Evidence Brief et les
    Variables (seuil et langue inclus : un brief FR ne doit pas être resservi pour
    une requête EN, ni un corpus au seuil 0.45 pour un seuil 0.60)."""
    import hashlib
    _ids = "|".join(str(i) for i in sorted(doc_ids))
    _key = f"{ctx}|thr={threshold}|lang={(lang or 'fr').lower()[:2]}|{_ids}"
    return hashlib.sha256(_key.encode("utf-8")).hexdigest()


@app.post("/scenarios/{scenario_id}/rerank")
def trigger_rerank(scenario_id: str, _: None = Depends(require_api_key)) -> dict[str, Any]:
    """
    Déclenche le scoring sémantique post-ingestion pour un scénario.
    Fonctionne pour GESICA et user_scenarios.
    """
    import threading

    # Récupérer la requête du scénario
    if scenario_id.startswith("usr-"):
        row = _get_user_scenario_or_404(scenario_id)
        query = row["query"]
    else:
        meta = _get_db_gesica_scenario_or_404(scenario_id)
        nl_queries = meta.get("nl_queries") or []
        query = nl_queries[0] if nl_queries else _gesica_title(meta)

    import time
    if _job_is_active(_RERANK_JOBS.get(scenario_id)):
        return {"status": "already_running", "scenario_id": scenario_id}

    _RERANK_JOBS[scenario_id] = {"status": "running", "updated": 0, "started_at": time.time()}

    def _run():
        try:
            _backfill_title_abstract_chunks(scenario_id)  # docs sans chunk résumé -> searchable
            n = _run_semantic_rerank_inline(scenario_id, query)
            # Recalcul COMPLET : après le cosinus, relancer AUSSI le cross-encoder Cohere
            # sur le sous-ensemble pertinent - sinon « Recalculer scores » ne rafraîchissait
            # que le cosinus et les rerank_score restaient figés/partiels.
            try:
                _nce = _run_cross_encoder_rerank(scenario_id, query)
                logger.info(f"Rerank manuel {scenario_id}: {n} cosinus + {_nce} cross-encoder Cohere.")
            except Exception as _ece:
                logger.warning(f"cross-encoder (recalcul manuel) {scenario_id}: {_ece}")
            _RERANK_JOBS[scenario_id] = {"status": "done", "updated": n}
        except Exception as e:
            # Sans ce filet, une exception laisse le job en "running" pour toujours
            # (→ "already_running" + badge "recalcul en cours" figé jusqu'au restart).
            logger.error(f"Rerank job {scenario_id}: {e}", exc_info=True)
            _RERANK_JOBS[scenario_id] = {"status": "error", "error": str(e), "updated": 0}

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "started", "scenario_id": scenario_id, "query": query}


@app.get("/scenarios/{scenario_id}/rerank/status")
def get_rerank_status(scenario_id: str) -> dict[str, Any]:
    """Statut du job de reranking sémantique."""
    return _RERANK_JOBS.get(scenario_id, {"status": "idle"})


@app.post("/scenarios/{scenario_id}/rebuild-corpus")
def rebuild_corpus(scenario_id: str, _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Reconstruit l'appartenance au corpus (article_scenarios) d'un scénario à
    partir de SA requête booléenne sur la base LOCALE (aucune ré-ingestion live),
    PUIS recalcule les scores (cosinus + cross-encoder).

    Pourquoi : /rerank ne SCORE que les liens article_scenarios EXISTANTS ; il ne
    peut donc rien faire pour un scénario dont le corpus est vide (jamais peuplé,
    ou seulement via l'ancien champ scenario_type). C'est le cas des scénarios
    « ⚠ VIDÉ » repérés par scripts/migration1_scenario_type_diff.py. Cette route
    reconstruit leur appartenance puis les score.

    Coût OpenAI minime : ~1 embedding de requête par scénario (la sélection des
    documents se fait en base locale ; pas d'appel aux sources live). GESICA et
    user_scenarios partagent la table user_scenarios, donc le traitement est
    identique. Suivi via /scenarios/{id}/rerank/status."""
    import threading
    with engine.connect() as _conn:
        row = _conn.execute(
            text("SELECT * FROM user_scenarios WHERE id = :id"), {"id": scenario_id}
        ).mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail=f"Scénario '{scenario_id}' non trouvé")
    row = dict(row)

    # Requête en langage naturel (sert à l'embedding de scoring).
    _nl = row.get("nl_queries")
    query = (row.get("query")
             or (_nl[0] if isinstance(_nl, list) and _nl else None)
             or row.get("title") or row.get("name") or "")
    filters = row.get("filters") or {}

    # Requête BOOLÉENNE (définit l'appartenance, sur la base locale). Priorité :
    # stratégie stockée → boolean_queries (GESICA) → génération depuis la requête.
    _strat = row.get("search_strategy")
    _bq = row.get("boolean_queries")
    if isinstance(_strat, dict) and _strat.get("general"):
        boolean = _strat["general"]
    elif isinstance(_bq, list) and _bq:
        boolean = _bq[0]
    elif isinstance(_bq, str) and _bq.strip():
        boolean = _bq
    elif query:
        try:
            boolean = _generate_search_strategy(query).get("general") or query
        except Exception:
            boolean = query
    else:
        raise HTTPException(status_code=422,
                            detail="Scénario sans requête exploitable : reconstruction impossible.")

    if _RERANK_JOBS.get(scenario_id, {}).get("status") == "running":
        return {"status": "already_running", "scenario_id": scenario_id}
    _RERANK_JOBS[scenario_id] = {"status": "running", "updated": 0}

    def _run():
        try:
            ids = _boolean_corpus_ids(boolean, filters)        # base LOCALE uniquement
            n_corpus = _set_scenario_corpus(scenario_id, ids)  # fixe l'appartenance
            _n_dup_rows = _dedup_scenario_links(scenario_id)   # un seul lien / article distinct
            n_corpus -= _n_dup_rows
            # PRISMA : une reconstruction n'interroge que la base locale → une seule
            # source ; les seuls doublons sont les lignes fusionnées par la dédup.
            _store_prisma_identification(scenario_id, _prisma_identification_figures(
                {"db_cache": len(ids)}, len(set(ids)), _n_dup_rows, max(0, n_corpus), method="rebuild"))
            _backfill_title_abstract_chunks(scenario_id)       # chunks résumé manquants
            n = _run_semantic_rerank_inline(scenario_id, query or boolean)  # cosinus pgvector
            try:
                _run_cross_encoder_rerank(scenario_id, query or boolean)    # Cohere (si clé)
            except Exception as _ece:
                logger.warning(f"rebuild-corpus cross-encoder {scenario_id}: {_ece}")
            _RERANK_JOBS[scenario_id] = {"status": "done", "corpus": n_corpus, "updated": n}
            logger.info(f"Rebuild corpus {scenario_id}: {n_corpus} liens, {n} scorés.")
        except Exception as _e:
            _RERANK_JOBS[scenario_id] = {"status": "error", "error": str(_e)}
            logger.error(f"Rebuild corpus {scenario_id}: {_e}", exc_info=True)

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "started", "scenario_id": scenario_id, "boolean": str(boolean)[:200]}


def _backfill_title_abstract_chunks(scenario_id: str | None = None) -> int:
    """
    Crée un chunk `title_abstract` (embedding NULL) pour les documents qui ont un
    titre/résumé mais AUCUN chunk title_abstract - typiquement les docs liés depuis
    la base locale sans création de chunk. Le worker d'enrichissement les embed
    ensuite : recherche sémantique au niveau résumé + compteurs réconciliés.
    Idempotent (NOT EXISTS). Si scenario_id est None, traite tout le corpus literev.
    """
    scope = "JOIN article_scenarios ars ON ars.document_id = ld.id AND ars.scenario_id = :sid" if scenario_id else ""
    params = {"sid": scenario_id} if scenario_id else {}
    try:
        with engine.begin() as conn:
            n = conn.execute(text(f"""
                INSERT INTO document_chunk (document_id, chunk_index, content, chunk_type, created_at)
                SELECT DISTINCT ld.id,
                       (SELECT COALESCE(MAX(c2.chunk_index), -1) + 1 FROM document_chunk c2 WHERE c2.document_id = ld.id),
                       btrim(coalesce(ld.title, '') || E'\\n\\n' || coalesce(ld.abstract, '')),
                       'title_abstract', now()
                FROM literature_document ld
                {scope}
                WHERE ld.project_context = 'literev'
                  AND ld.is_duplicate IS NOT TRUE
                  AND length(btrim(coalesce(ld.title, '') || ' ' || coalesce(ld.abstract, ''))) >= 30
                  AND NOT EXISTS (
                      SELECT 1 FROM document_chunk c
                      WHERE c.document_id = ld.id AND c.chunk_type = 'title_abstract'
                  )
            """), params).rowcount
        if n:
            logger.info(f"Backfill title_abstract chunks ({scenario_id or 'global'}): {n} créés (embedding par le worker).")
        return n
    except Exception as e:
        logger.warning(f"Backfill title_abstract chunks {scenario_id}: {e}")
        return 0


def _maybe_autorerank(scenario_id: str) -> bool:
    """
    Lance le scoring sémantique en arrière-plan si jamais effectué (auto-score),
    et complète au passage les chunks title_abstract manquants (recherche + compteurs).
    Ne se déclenche qu'une fois par scénario (tant que le process vit) : si le
    job est déjà 'running' ou 'done', on ne relance pas. Renvoie True si lancé.
    """
    import threading

    st = _RERANK_JOBS.get(scenario_id, {}).get("status")
    if st in ("running", "done"):
        return False
    try:
        if scenario_id.startswith("usr-"):
            row = _get_user_scenario_or_404(scenario_id)
            query = row["query"]
        else:
            meta = _get_db_gesica_scenario_or_404(scenario_id)
            nl = meta.get("nl_queries") or []
            query = nl[0] if nl else _gesica_title(meta)
    except Exception:
        return False
    if not query:
        return False

    _RERANK_JOBS[scenario_id] = {"status": "running", "updated": 0}

    def _run():
        try:
            _backfill_title_abstract_chunks(scenario_id)  # docs sans chunk résumé -> searchable
            n = _run_semantic_rerank_inline(scenario_id, query)
            _RERANK_JOBS[scenario_id] = {"status": "done", "updated": n}
            logger.info(f"Auto-rerank {scenario_id}: {n} articles scorés.")
        except Exception as e:
            logger.warning(f"Auto-rerank {scenario_id}: {e}")
            _RERANK_JOBS[scenario_id] = {"status": "error", "error": str(e)}

    threading.Thread(target=_run, daemon=True).start()
    return True


_SETTINGS_BLOB_COLUMNS = ("evidence_brief_json", "variables_json", "clustering_json",
                          "knowledge_graph_json", "recommended_actions_json")


@app.get("/scenarios/{scenario_id}/settings")
def get_scenario_settings(scenario_id: str) -> dict[str, Any]:
    """Retourne les paramètres du scénario (seuil, dates de génération, variables validées).

    Les artefacts JSON en cache (brief, variables, clustering, graphe, actions) ne sont
    PAS renvoyés : chacun a son endpoint. Avec `SELECT *`, la page scénario téléchargeait
    à chaque ouverture, pour lire un seul seuil, le clustering complet et le graphe de
    connaissances (2,5 Mo pour 25 000 articles). Seule leur présence est indiquée."""
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT * FROM scenario_settings WHERE scenario_id = :sid
        """), {"sid": scenario_id}).mappings().first()
    if not row:
        return {
            "scenario_id": scenario_id,
            "similarity_threshold": DEFAULT_SIMILARITY_THRESHOLD,
            "brief_generated_at": None,
            "variables_validated": False,
            "variables_generated_at": None,
            "cached": {c[:-5]: False for c in _SETTINGS_BLOB_COLUMNS},
        }
    out = {k: v for k, v in dict(row).items() if k not in _SETTINGS_BLOB_COLUMNS}
    out["cached"] = {c[:-5]: bool(row.get(c)) for c in _SETTINGS_BLOB_COLUMNS if c in row}
    return out


@app.patch("/scenarios/{scenario_id}/settings")
def update_scenario_settings(scenario_id: str, payload: dict[str, Any], _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Met à jour les paramètres du scénario (seuil, variables validées, etc.)."""
    allowed = {"similarity_threshold", "variables_json", "variables_validated"}
    updates = {k: v for k, v in payload.items() if k in allowed}
    if not updates:
        raise HTTPException(status_code=422, detail="Aucun champ valide à mettre à jour")

    with engine.begin() as conn:
        # Upsert
        conn.execute(text("""
            INSERT INTO scenario_settings (scenario_id, updated_at)
            VALUES (:sid, NOW())
            ON CONFLICT (scenario_id) DO NOTHING
        """), {"sid": scenario_id})

        for key, val in updates.items():
            import json as _json
            if isinstance(val, (dict, list)):
                val = _json.dumps(val)
            conn.execute(text(f"""
                UPDATE scenario_settings SET {key} = :val, updated_at = NOW()
                WHERE scenario_id = :sid
            """), {"val": val, "sid": scenario_id})

    # Retourner l'objet settings complet mis à jour
    with engine.connect() as conn:
        updated_row = conn.execute(text("""
            SELECT scenario_id, similarity_threshold, brief_generated_at,
                   variables_validated, variables_generated_at, updated_at
            FROM scenario_settings WHERE scenario_id = :sid
        """), {"sid": scenario_id}).mappings().first()
    if updated_row:
        return {
            "status": "updated",
            "scenario_id": scenario_id,
            "updated": list(updates.keys()),
            "similarity_threshold": float(updated_row["similarity_threshold"]) if updated_row["similarity_threshold"] is not None else 0.45,
            "variables_validated": bool(updated_row["variables_validated"]),
            "updated_at": updated_row["updated_at"].isoformat() if updated_row["updated_at"] else None,
        }
    return {"status": "updated", "scenario_id": scenario_id, "updated": list(updates.keys())}
