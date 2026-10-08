"""Semantic scoring, cross-encoder rerank, threshold settings, relevant articles.

Extracted from main.py (LiteRev API); `main` re-exports everything for the scripts,
tools and tests.
"""
from __future__ import annotations

import math
import os
from typing import Any

from fastapi import Depends, HTTPException
from sqlalchemy import text

from .core import (
    _env_int,
    _is_openai_quota_error,
    _job_is_active,
    _openai_in_cooldown,
    _trip_openai_cooldown,
    app,
    engine,
    logger,
    require_api_key,
)
from .scenario_store import (
    CORPUS_DERIVED_CACHE_RESET,
    DEFAULT_RERANK_THRESHOLD,
    DEFAULT_SIMILARITY_THRESHOLD,
    _get_scenario_rerank_threshold,
    _get_scenario_threshold,
    _get_user_scenario_or_404,
)
from .search import (
    _boolean_corpus_ids,
    _dedup_scenario_links,
    _generate_search_strategy,
    _prisma_identification_figures,
    _set_scenario_corpus,
    _store_prisma_identification,
)
from .gesica import _gesica_title, _get_db_gesica_scenario_or_404
from llm_usage import model_for as _model

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
        q_emb = _client.embeddings.create(model=_model("embedding"), input=query[:2000]).data[0].embedding
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
                    emb = _client.embeddings.create(model=_model("embedding"), input=texts).data
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


#: Le rerank est limité en débit côté fournisseur, et le dépassement est la panne
#: NORMALE d'un balayage, pas l'exception : sur un rattrapage de 119 lots, 99 ont échoué
#: d'affilée après une vingtaine de succès. Sans reprise, un rattrapage ne finit jamais.
_RERANK_RETRIES = _env_int("RERANK_RETRIES", 5, 0)
_RERANK_BACKOFF_S = 4.0


def _cohere_rerank(query: str, docs: list[str], model: str = "rerank-v3.5") -> list[float] | None:
    """Cross-encoder rerank via Cohere, avec reprise sur limite de débit.

    Renvoie un score par document (aligné sur `docs`), ou None après épuisement des
    tentatives. Appel REST direct, aucune dépendance ajoutée.

    429 et 5xx sont RÉESSAYÉS avec une attente qui double, en respectant `Retry-After`
    quand le serveur le donne ; une erreur de requête (4xx autre que 429) ne l'est pas,
    puisque la réessayer donnera la même réponse. Avant, toute erreur rendait None du
    premier coup, et comme l'appelant jetait alors le lot entier, un simple dépassement de
    débit se présentait comme « ce scénario n'a pas de scores »."""
    key = os.getenv("COHERE_API_KEY")
    if not key or not docs:
        return None
    import time as _t

    import requests as _rq
    wait = _RERANK_BACKOFF_S
    for attempt in range(_RERANK_RETRIES + 1):
        try:
            resp = _rq.post(
                "https://api.cohere.com/v2/rerank",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": model, "query": query[:4000], "documents": docs, "top_n": len(docs)},
                timeout=60,
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt >= _RERANK_RETRIES:
                    logger.warning(f"Cohere rerank: {resp.status_code} after "
                                   f"{_RERANK_RETRIES} retries, giving up on this batch")
                    return None
                # Le serveur sait mieux que nous combien attendre quand il le dit.
                try:
                    hinted = float(resp.headers.get("Retry-After") or 0)
                except ValueError:
                    hinted = 0.0
                _t.sleep(max(wait, hinted))
                wait *= 2
                continue
            resp.raise_for_status()
            data = resp.json()
            scores: list[float | None] = [None] * len(docs)
            for item in data.get("results", []):
                idx = item.get("index")
                if idx is not None and 0 <= idx < len(docs):
                    scores[idx] = float(item.get("relevance_score", 0.0))
            return scores  # type: ignore[return-value]
        except _rq.exceptions.RequestException as _e:
            # Panne réseau ou délai dépassé : réessayable, contrairement à un 400.
            if attempt >= _RERANK_RETRIES:
                logger.warning(f"Cohere rerank failed after {_RERANK_RETRIES} retries: {_e}")
                return None
            _t.sleep(wait)
            wait *= 2
        except Exception as _e:
            logger.warning(f"Cohere rerank failed: {_e}")
            return None
    return None


# Aucun plafond par défaut : le rerank note TOUT le corpus du scénario, jamais un
# échantillon. Il y avait `top_k: int = 1000` en dur dans la signature, qu'aucun appelant
# ne surchargeait, et la conséquence se mesure : sur l'installation, 10 480 lignes
# pertinentes de 7 scénarios n'ont pas de score parce qu'elles tombaient au-delà du
# millième rang. RERANK_MAX_ARTICLES > 0 repose un plafond (secours d'exploitation),
# comme EPI_PARAM_MAX_ARTICLES pour l'extraction.
RERANK_MAX_ARTICLES = _env_int("RERANK_MAX_ARTICLES", 0, 0)
#: Documents par requête Cohere. Tout partait en UN appel, donc un scénario de 3 424
#: candidats tentait de les envoyer ensemble, et la moindre erreur perdait le lot entier.
_RERANK_BATCH = 100
#: Pause entre deux lots. Mieux vaut ne pas déclencher la limite que la rattraper.
_RERANK_PAUSE_S = float(os.getenv("RERANK_PAUSE_S", "1.0") or 1.0)


def _run_cross_encoder_rerank(scenario_id: str, query: str, top_k: int | None = None,
                              only_missing: bool = False) -> dict[str, int]:
    """Note le corpus du scénario avec le cross-encoder, par lots, en écrivant au fur.

    Trois choses ont changé, et chacune corrigeait une cause mesurée de trous :

    1. LA SÉLECTION NE LIT PLUS LE SEUIL. Elle le lisait au moment du lancement, si bien
       que tout ce qui passait sous le curseur n'était JAMAIS envoyé. Sur HPAI la coupure
       se voyait à six décimales : plus bas score noté 0.303309, plus haut non noté
       0.299115, rien entre les deux. Une variable continue ne se partitionne pas toute
       seule, seul un filtre le fait. Or c'est exactement l'article sous le seuil dont on
       a besoin du score pour décider de le faire entrer. On note donc tout le corpus.
    2. PLUS DE PLAFOND (voir RERANK_MAX_ARTICLES).
    3. PAR LOTS, ÉCRITS AU FUR ET À MESURE. Un lot qui échoue n'emporte plus les autres :
       avant, une seule requête portait tout et `return 0` jetait le travail entier.

    `only_missing` ne note que les lignes sans score : c'est le balayage de rattrapage,
    qui ne redépense rien pour ce qui est déjà noté.

    Renvoie le compte de ce qui s'est passé, au lieu d'un entier qui ne disait pas la
    différence entre « rien à faire » et « tout a échoué »."""
    out = {"scored": 0, "batches_ok": 0, "batches_failed": 0, "candidates": 0, "skipped_no_key": 0}
    if not os.getenv("COHERE_API_KEY"):
        out["skipped_no_key"] = 1
        return out
    cap = RERANK_MAX_ARTICLES if top_k is None else top_k
    try:
        with engine.connect() as _tc:
            rows = _tc.execute(text(f"""
                SELECT ld.id, ld.title, ld.abstract
                FROM literature_document ld
                JOIN article_scenarios asn ON asn.document_id = ld.id AND asn.scenario_id = :sid
                WHERE ld.is_duplicate IS NOT TRUE
                  AND ld.abstract IS NOT NULL AND length(ld.abstract) > 30
                  {"AND asn.rerank_score IS NULL" if only_missing else ""}
                ORDER BY asn.similarity_score DESC NULLS LAST, ld.id
                {"LIMIT :k" if cap > 0 else ""}
            """), ({"sid": scenario_id, "k": cap} if cap > 0 else {"sid": scenario_id})).mappings().all()
        out["candidates"] = len(rows)
        if not rows:
            return out
        import time as _t_batch
        for _b in range(0, len(rows), _RERANK_BATCH):
            if _b:
                _t_batch.sleep(_RERANK_PAUSE_S)
            chunk = rows[_b:_b + _RERANK_BATCH]
            docs = [f"{r['title']}\n\n{(r['abstract'] or '')[:1500]}" for r in chunk]
            scores = _cohere_rerank(query, docs)
            if not scores:
                # Le lot échoue, les suivants continuent : c'est tout l'intérêt des lots.
                out["batches_failed"] += 1
                continue
            ups = [{"s": s, "doc_id": r["id"], "sid": scenario_id}
                   for r, s in zip(chunk, scores) if s is not None]
            if not ups:
                out["batches_failed"] += 1
                continue
            with engine.begin() as _c:
                _c.execute(text("""
                    UPDATE article_scenarios SET rerank_score = :s
                    WHERE document_id = :doc_id AND scenario_id = :sid
                """), ups)
            out["scored"] += len(ups)
            out["batches_ok"] += 1
        logger.info(f"Cross-encoder rerank {scenario_id}: {out['scored']}/{out['candidates']} notés, "
                    f"{out['batches_ok']} lots OK, {out['batches_failed']} en échec.")
        return out
    except Exception as _e:
        logger.warning(f"Cross-encoder rerank {scenario_id} failed: {_e}")
        return out


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
                rerank_threshold     FLOAT DEFAULT 0,
                evidence_brief_json  JSONB DEFAULT NULL,
                brief_generated_at   TIMESTAMP DEFAULT NULL,
                variables_json       JSONB DEFAULT NULL,
                variables_validated  BOOLEAN DEFAULT FALSE,
                variables_generated_at TIMESTAMP DEFAULT NULL,
                updated_at           TIMESTAMP DEFAULT NOW()
            )
        """))
        # Ajoutée après coup : la table existe déjà en production, donc le CREATE ci-dessus
        # ne la crée pas et la colonne doit être posée séparément. NULL vaut 0, soit la
        # porte d'avant, pour tous les scénarios déjà là.
        conn.execute(text(
            "ALTER TABLE scenario_settings ADD COLUMN IF NOT EXISTS rerank_threshold FLOAT DEFAULT 0"))
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
            model=_model("embedding"),
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
                  AND (COALESCE(asn.screening_status, ld.screening_status) = 'included' OR (COALESCE(asn.similarity_score, 0) >= :threshold AND (asn.rerank_score IS NULL OR asn.rerank_score >= COALESCE((SELECT ss.rerank_threshold FROM scenario_settings ss WHERE ss.scenario_id = asn.scenario_id), 0.0))))
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
def trigger_rerank(scenario_id: str, missing_only: bool = False,
                   _: None = Depends(require_api_key)) -> dict[str, Any]:
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
            n = 0
            if not missing_only:
                _backfill_title_abstract_chunks(scenario_id)  # docs sans chunk -> searchable
                n = _run_semantic_rerank_inline(scenario_id, query)
            # Recalcul COMPLET : après le cosinus, relancer AUSSI le cross-encoder Cohere
            # sur le sous-ensemble pertinent - sinon « Recalculer scores » ne rafraîchissait
            # que le cosinus et les rerank_score restaient figés/partiels.
            _ce = {"scored": 0, "candidates": 0, "batches_failed": 0, "skipped_no_key": 0}
            try:
                _ce = _run_cross_encoder_rerank(scenario_id, query, only_missing=missing_only)
                logger.info(f"Rerank manuel {scenario_id}: {n} cosinus + {_ce['scored']} cross-encoder.")
            except Exception as _ece:
                logger.warning(f"cross-encoder (recalcul manuel) {scenario_id}: {_ece}")
            # Le job disait « done » avec le seul compte du cosinus, quoi qu'ait fait le
            # cross-encoder : un échec total ressortait identique à un succès. Il porte
            # maintenant les deux, et dit si des lots ont échoué.
            _RERANK_JOBS[scenario_id] = {
                "status": "done", "updated": n,
                "reranked": _ce.get("scored", 0),
                "rerank_candidates": _ce.get("candidates", 0),
                "rerank_batches_failed": _ce.get("batches_failed", 0),
                "rerank_skipped_no_key": bool(_ce.get("skipped_no_key")),
            }
        except Exception as e:
            # Sans ce filet, une exception laisse le job en "running" pour toujours
            # (→ "already_running" + badge "recalcul en cours" figé jusqu'au restart).
            logger.error(f"Rerank job {scenario_id}: {e}", exc_info=True)
            _RERANK_JOBS[scenario_id] = {"status": "error", "error": str(e), "updated": 0}

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "started", "scenario_id": scenario_id, "query": query}


@app.get("/scenarios/{scenario_id}/rerank/status")
def get_rerank_status(scenario_id: str) -> dict[str, Any]:
    """Statut du job de reranking, ET l'état réel de la couverture en base.

    Le job vit en mémoire, donc tout redémarrage le remet à « idle », et l'API redémarre
    à chaque déploiement. « idle » ne voulait donc pas dire « à jour », il voulait dire
    « je ne me souviens de rien ». Les deux compteurs ci-dessous viennent de la base, en
    UNE instruction, et disent ce qui est vrai maintenant : combien d'articles du corpus
    pourraient porter un score, et combien n'en ont pas. L'interface peut enfin écrire
    sur le bouton ce qu'il va faire."""
    out = dict(_RERANK_JOBS.get(scenario_id, {"status": "idle"}))
    try:
        with engine.connect() as _c:
            row = _c.execute(text("""
                SELECT COUNT(*) FILTER (WHERE ld.abstract IS NOT NULL
                                          AND length(ld.abstract) > 30)            AS scorable,
                       COUNT(*) FILTER (WHERE ld.abstract IS NOT NULL
                                          AND length(ld.abstract) > 30
                                          AND asn.rerank_score IS NULL)            AS missing,
                       COUNT(*) FILTER (WHERE ld.abstract IS NULL
                                           OR length(ld.abstract) <= 30)           AS unscorable
                FROM article_scenarios asn
                JOIN literature_document ld ON ld.id = asn.document_id
                WHERE asn.scenario_id = :sid AND ld.is_duplicate IS NOT TRUE
            """), {"sid": scenario_id}).mappings().first() or {}
        out["scorable"] = int(row.get("scorable") or 0)
        out["missing"] = int(row.get("missing") or 0)
        # Un article sans résumé utilisable ne peut PAS recevoir de score : le dire, plutôt
        # que de laisser croire qu'un lancement de plus le rattrapera.
        out["unscorable"] = int(row.get("unscorable") or 0)
    except Exception as _e:
        logger.warning(f"rerank coverage {scenario_id}: {_e}")
    return out


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
            "rerank_threshold": DEFAULT_RERANK_THRESHOLD,
            "brief_generated_at": None,
            "variables_validated": False,
            "variables_generated_at": None,
            "cached": {c[:-5]: False for c in _SETTINGS_BLOB_COLUMNS},
        }
    out = {k: v for k, v in dict(row).items() if k not in _SETTINGS_BLOB_COLUMNS}
    # NULL en base veut dire « jamais réglé », et la porte le lit comme 0. L'interface doit
    # lire la même chose, sans quoi le curseur s'afficherait vide sur un corpus non filtré.
    if out.get("rerank_threshold") is None:
        out["rerank_threshold"] = DEFAULT_RERANK_THRESHOLD
    out["cached"] = {c[:-5]: bool(row.get(c)) for c in _SETTINGS_BLOB_COLUMNS if c in row}
    return out


@app.patch("/scenarios/{scenario_id}/settings")
def update_scenario_settings(scenario_id: str, payload: dict[str, Any], _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Met à jour les paramètres du scénario (seuil, variables validées, etc.)."""
    allowed = {"similarity_threshold", "rerank_threshold", "variables_json", "variables_validated"}
    updates = {k: v for k, v in payload.items() if k in allowed}
    if not updates:
        raise HTTPException(status_code=422, detail="Aucun champ valide à mettre à jour")
    # Les deux seuils bornent un score, et un score hors de [0, 1] ne veut rien dire. Sans
    # cette garde, un 45 tapé pour 0.45 vidait le corpus sans un mot d'explication.
    for _k in ("similarity_threshold", "rerank_threshold"):
        if _k in updates:
            try:
                _v = float(updates[_k])
            except (TypeError, ValueError):
                raise HTTPException(status_code=422, detail=f"{_k} doit être un nombre")
            if not (0.0 <= _v <= 1.0):
                raise HTTPException(status_code=422, detail=f"{_k} doit être compris entre 0 et 1")
            updates[_k] = _v

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

        # Les artefacts mis en cache sont des FONCTIONS du seuil (clustering, graphe de
        # similarité, carte des concepts) ou du spec (projection SEIR par défaut). Ils
        # n'étaient invalidés par rien : déplacer le curseur de pertinence laissait servir,
        # jusqu'à 30 jours, des visualisations calculées sur un sous-ensemble qui n'existe
        # plus, et une édition du spec laissait une projection issue des anciens paramètres.
        # Le seuil de rerank découpe le corpus exactement comme celui de similarité, donc
        # il périme exactement les mêmes artefacts. L'oublier ici aurait servi un
        # clustering et des actions recommandées calculés sur le corpus d'avant.
        if "similarity_threshold" in updates or "rerank_threshold" in updates:
            # Les actions recommandées étaient les seules oubliées : elles sont générées
            # sur les articles pertinents ET sur le digest du corpus, mais leur cache
            # n'est indexé que par (scénario, langue). Déplacer le seuil de 0.45 à 0.60
            # continuait donc de servir, indéfiniment, des actions tirées du corpus
            # PRÉCÉDENT, présentées comme celles du corpus actuel. La liste est partagée
            # avec la living review (CORPUS_DERIVED_CACHE_RESET) pour qu'un artefact
            # ajouté demain ne soit pas oublié d'un côté.
            conn.execute(text(f"""
                UPDATE scenario_settings SET {CORPUS_DERIVED_CACHE_RESET}
                WHERE scenario_id = :sid
            """), {"sid": scenario_id})
        if "variables_json" in updates:
            conn.execute(text("""
                UPDATE scenario_settings
                SET seir_projection_json = NULL, seir_projection_generated_at = NULL,
                    variables_i18n = NULL
                WHERE scenario_id = :sid
            """), {"sid": scenario_id})

    # Retourner l'objet settings complet mis à jour
    with engine.connect() as conn:
        updated_row = conn.execute(text("""
            SELECT scenario_id, similarity_threshold, rerank_threshold, brief_generated_at,
                   variables_validated, variables_generated_at, updated_at
            FROM scenario_settings WHERE scenario_id = :sid
        """), {"sid": scenario_id}).mappings().first()
    if updated_row:
        return {
            "status": "updated",
            "scenario_id": scenario_id,
            "updated": list(updates.keys()),
            "similarity_threshold": float(updated_row["similarity_threshold"]) if updated_row["similarity_threshold"] is not None else 0.45,
            "rerank_threshold": float(updated_row["rerank_threshold"]) if updated_row["rerank_threshold"] is not None else 0.0,
            "variables_validated": bool(updated_row["variables_validated"]),
            "updated_at": updated_row["updated_at"].isoformat() if updated_row["updated_at"] else None,
        }
    return {"status": "updated", "scenario_id": scenario_id, "updated": list(updates.keys())}


# ─── Courbe du seuil : choisir par le NOMBRE d'articles, pas au jugé ─────────
# Le seuil se réglait à l'aveugle (« essayez 0.30, voyez ce que ça donne »). Or les
# scores sont déjà en base : « quel seuil garde 100 articles » est une question à réponse
# exacte, et « combien d'articles rapportant un paramètre ce seuil écarte-t-il » en fait
# un choix défendable dans une section Méthodes. C'est cette seconde colonne qui vaut le
# détour : un seuil justifié par « aucun article mesurant un paramètre n'est exclu »
# s'écrit dans un article, « nous avons pris 0.45 » non.
_CURVE_TARGETS = (25, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000)


def _floor4(x: float) -> float:
    """Tronque à 4 décimales VERS LE BAS. Le seuil renvoyé est donc toujours ≤ au score
    de l'article frontière : régler le curseur dessus garde bien ce qu'on a annoncé.
    Arrondir au plus proche pouvait remonter au-dessus du score et couper un article de
    plus que le compte affiché."""
    return math.floor(float(x) * 10000) / 10000


CURVE_SCORES = ("similarity", "rerank")


def _threshold_curve_inputs(scenario_id: str, score: str = "similarity") -> dict[str, Any]:
    """Lit d'un coup ce dont la courbe a besoin : un point par article CANDIDAT, plus les
    compteurs que la courbe ne peut pas déduire.

    Candidat = ni doublon, ni exclu, ni inclus à la main. Les inclus passent QUEL QUE SOIT
    le seuil (c'est la porte de pertinence commune à toute l'app) : ils s'ajoutent donc à
    chaque total sans jamais peser sur le choix du seuil, d'où leur comptage à part.

    `score` choisit la métrique, et les deux ne traitent PAS l'article non scoré pareil,
    parce que la porte ne le traite pas pareil :

      - `similarity` : un article sans score vaut 0, donc il sort dès que le seuil est non
        nul. Il est dans `rows`, à 0.
      - `rerank` : un article sans score n'est pas jugé, donc il reste quel que soit le
        seuil. Il n'est PAS dans `rows`, il rejoint le compte des articles qui passent
        d'office, sans quoi la courbe promettrait une coupe que la base ne ferait pas.

    `unscored_are_kept` dit laquelle des deux règles s'applique, pour que l'interface
    puisse l'écrire au lieu de laisser deviner."""
    from .variables import _param_regex                     # lazy: variables charge après
    if score not in CURVE_SCORES:
        raise HTTPException(status_code=422,
                            detail=f"score inconnu : '{score}' (attendu : {', '.join(CURVE_SCORES)})")
    rerank = score == "rerank"
    col = "ars.rerank_score" if rerank else "COALESCE(ars.similarity_score, 0)"
    sql_where = """
        FROM literature_document d
        JOIN article_scenarios ars ON ars.document_id = d.id
        WHERE ars.scenario_id = :sid
          AND d.is_duplicate IS NOT TRUE
          AND COALESCE(ars.screening_status, d.screening_status) IS DISTINCT FROM 'excluded'
    """
    # Le non scoré sort de la courbe pour le rerank : il passe d'office, donc il n'a pas
    # de place sur un axe de seuils.
    only_scored = " AND ars.rerank_score IS NOT NULL" if rerank else ""
    with engine.connect() as conn:
        rows = [(float(r["s"]), bool(r["p"])) for r in conn.execute(text(f"""
            SELECT {col} AS s,
                   ((COALESCE(d.title, '') || ' ' || COALESCE(d.abstract, '')) ~* :rx) AS p
            {sql_where}
              AND COALESCE(ars.screening_status, d.screening_status) IS DISTINCT FROM 'included'
              {only_scored}
            ORDER BY s DESC
        """), {"sid": scenario_id, "rx": _param_regex(boundary=r"\y")}).mappings()]
        head = conn.execute(text(f"""
            SELECT COUNT(*) FILTER (
                     WHERE COALESCE(ars.screening_status, d.screening_status) = 'included') AS included,
                   COUNT(*) FILTER (
                     WHERE COALESCE(ars.screening_status, d.screening_status) IS DISTINCT FROM 'included'
                       AND {'ars.rerank_score' if rerank else 'ars.similarity_score'} IS NULL) AS unscored
            {sql_where}
        """), {"sid": scenario_id}).mappings().first() or {}
    return {"rows": rows, "included": int(head.get("included") or 0),
            "unscored": int(head.get("unscored") or 0),
            "score": score, "unscored_are_kept": rerank}


def threshold_curve(rows: list[tuple[float, bool]], included: int = 0, targets=_CURVE_TARGETS,
                    current: float | None = None) -> list[dict]:
    """La courbe seuil ↔ nombre d'articles gardés. PURE, testable hors base.

    `rows` : (score, rapporte un paramètre) par candidat, DÉCROISSANT. `included` : les
    articles retenus à la main, qui passent quel que soit le seuil et s'ajoutent donc à
    chaque total.

    Les ÉGALITÉS empêchent d'atteindre un nombre rond : viser 100 sur un corpus où trente
    articles partagent le score frontière en garde 115, pas 100. Chaque point renvoie donc
    le nombre RÉELLEMENT gardé et `exact`, jamais le nombre demandé reformulé en réponse."""
    n = len(rows)
    if n == 0:
        return []
    total_param = sum(1 for _, p in rows if p)
    # Cumul des articles « à paramètre » : with_parameter_kept au rang i se lit en O(1).
    cum: list[int] = [0] * (n + 1)
    for i, (_, p) in enumerate(rows):
        cum[i + 1] = cum[i] + (1 if p else 0)
    scores = [s for s, _ in rows]

    here = _floor4(current) if current is not None else None

    def _point(thr: float, label: int | None) -> dict:
        # Compté AU SEUIL RENVOYÉ, pas au score brut : le chiffre affiché est celui que
        # donnera le curseur une fois posé là. Tous les ex aequo passent, comme en SQL.
        kept_scored = sum(1 for s in scores if s >= thr)
        kept_param = cum[kept_scored]
        return {
            "threshold": thr,
            "kept": kept_scored + included,
            "kept_scored": kept_scored,
            # « Où j'en suis » est une propriété du point, pas une ligne à part : quand une
            # cible du barème tombe exactement sur le seuil courant, le point n'était
            # ajouté qu'une fois, étiqueté par la cible, et l'interface n'avait plus rien
            # pour dire où était le curseur.
            "is_current": here is not None and thr == here,
            "requested": label,
            # None = « sans objet » (point du seuil courant, personne n'a demandé de
            # nombre) ; False = on a visé et les ex aequo ont fait rater la cible.
            "exact": None if label is None else (kept_scored + included) == label,
            "with_parameter_kept": kept_param,
            "with_parameter_cut": total_param - kept_param,
        }

    out: list[dict] = []
    seen: set[float] = set()
    # L'ORDRE de `targets` fait la priorité : deux cibles voisines peuvent tomber sur le
    # même seuil (un gros paquet d'ex aequo), et le point ne garde qu'une étiquette. La
    # cible explicitement demandée est donc passée en tête par l'appelant, sans quoi elle
    # se ferait absorber par une cible du barème et ressortirait comme « hors de portée ».
    for t in dict.fromkeys(int(x) for x in targets):
        want = t - included                     # les inclus manuels comptent déjà dedans
        if want < 1 or want > n:
            continue
        thr = _floor4(scores[want - 1])
        if thr in seen:
            continue                            # deux cibles tombent dans le même paquet
        seen.add(thr)
        out.append(_point(thr, t))
    if here is not None and here not in seen:
        seen.add(here)
        out.append(_point(here, None))
    out.sort(key=lambda p: -p["threshold"])
    return out


def _scope_note(scenario_id: str) -> dict[str, Any] | None:
    """Le découpage en vigueur, ou None. Lazy : `subsets` se charge après `relevance`.
    Silencieux en cas d'échec : la courbe reste utile sans cette mise en garde, elle
    n'a pas à tomber avec elle."""
    try:
        from .subsets import scope_state
        st = scope_state(scenario_id)
        return {k: st[k] for k in ("excluded_by_scope", "judged_above_threshold")} if st["narrowed"] else None
    except Exception as e:                                   # pragma: no cover - défensif
        logger.warning(f"threshold-curve scope note {scenario_id}: {e}")
        return None


@app.get("/scenarios/{scenario_id}/threshold-curve")
def get_threshold_curve(scenario_id: str, target: int | None = None,
                        score: str = "similarity") -> dict[str, Any]:
    """« Quel seuil garde N articles ? » - répondu sur les scores réellement en base.

    Aucun échantillonnage et aucun LLM : un point par article candidat, agrégé en
    mémoire. `curve` donne, pour une série de tailles de corpus, le seuil qui les
    approche et ce que ce seuil coûte en articles rapportant un paramètre
    épidémiologique (le seuil se justifie alors par « ce qu'il écarte », pas par
    l'habitude). `target` ajoute une cible libre et la renvoie dans `suggestion`, ou
    `null` si aucun seuil ne l'atteint : `reachable` dit alors ce que le seuil PEUT faire,
    plutôt que de rendre le point voisin en le faisant passer pour la réponse.

    Ce que la réponse ne cache pas : `unscored` (des articles sans score, comptés 0 comme
    partout ailleurs, donc gardés seulement à seuil 0) et `scoring_in_progress` (la courbe
    bouge encore). Aucun point n'est « exact » par construction : les ex aequo font qu'une
    cible ronde est rarement atteignable, et c'est `kept` qui fait foi."""
    _get_user_scenario_or_404(scenario_id)
    data = _threshold_curve_inputs(scenario_id, score)
    rows, included = data["rows"], data["included"]
    # Ce qui passe quel que soit le seuil : les inclus à la main, plus, pour le rerank,
    # les articles que le rerank n'a pas encore jugés. La courbe les ajoute à chaque
    # point, exactement comme la porte les laissera passer.
    free = included + (data["unscored"] if data["unscored_are_kept"] else 0)
    current = (_get_scenario_rerank_threshold(scenario_id) if score == "rerank"
               else _get_scenario_threshold(scenario_id))
    targets = list(_CURVE_TARGETS)
    if target is not None:
        t = int(target)
        if t < 1:
            raise HTTPException(status_code=422, detail="target doit être un entier positif")
        targets.insert(0, t)                    # en tête : elle l'emporte sur le barème
    curve = threshold_curve(rows, free, targets, current)
    out: dict[str, Any] = {
        "scenario_id": scenario_id,
        "score": score,
        "current_threshold": current,
        "candidates": len(rows),
        "included": included,
        "unscored": data["unscored"],
        # Le point sur lequel les deux courbes diffèrent, écrit plutôt que sous-entendu :
        # un article non scoré sort (similarité) ou reste (rerank).
        "unscored_are_kept": data["unscored_are_kept"],
        "always_kept": free,
        "corpus": len(rows) + free,
        # Ce que le seuil peut atteindre : en deçà de `min`, les articles qui passent
        # d'office passent quoi qu'on fasse ; au-delà de `max`, il n'y a plus d'articles.
        "reachable": {"min": free + 1, "max": len(rows) + free} if rows else None,
        "with_parameter_total": sum(1 for _, p in rows if p),
        "scoring_in_progress": _RERANK_JOBS.get(scenario_id, {}).get("status") == "running",
        # Un découpage par clusters ou par concepts ne juge que les articles pertinents AU
        # MOMENT où il est posé. Descendre le seuil sous cette frontière fait donc rentrer
        # des articles que la sélection n'a jamais vus, et la courbe les propose avec le
        # même aplomb que les autres. Elle doit dire à partir d'où elle ment par omission.
        "scope": _scope_note(scenario_id),
        "curve": curve,
    }
    if target is not None:
        # La cible peut être hors de portée (plus grande que le corpus, ou entièrement
        # couverte par les inclus manuels) : on le dit, plutôt que de renvoyer un point
        # voisin en le faisant passer pour la réponse.
        out["suggestion"] = next((p for p in curve if p["requested"] == int(target)), None)
    return out
