"""Semantic scoring, cross-encoder rerank, threshold settings, relevant articles.

Extracted from main.py (LiteRev API); `main` re-exports everything for the scripts,
tools and tests.
"""
from __future__ import annotations

import math
import os
import threading
from typing import Any

from fastapi import Depends, HTTPException
from sqlalchemy import text

from .core import (
    _OPENAI_QUOTA_COOLDOWN_UNTIL,
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
from .documents import _truncate_to_tokens
from .scenario_store import (
    DEFAULT_RERANK_THRESHOLD,
    _get_scenario_rerank_threshold,
    _get_scenario_threshold,
    _get_user_scenario_or_404,
    relevance_order_sql,
    relevant_gate_sql,
    reset_corpus_derived_caches,
    scenario_rerank_threshold_sql,
    scenario_threshold_info,
    scenario_threshold_sql,
    scenario_unscored_count,
    screening_status_sql,
    unscored_sql,
)
from .search import (
    _boolean_corpus_ids,
    _dedup_scenario_links,
    _generate_search_strategy,
    _load_prisma_identification,
    _prisma_identification_figures,
    _set_scenario_corpus,
    _store_prisma_identification,
)
from .gesica import _gesica_title, _get_db_gesica_scenario_or_404
from .study_design import raw_design_sql as _raw_design
from llm_usage import model_for as _model

#: Borne de sécurité sous les 8 191 jetons de text-embedding-3-small : 20 000 caractères
#: d'une requête booléenne anglaise font environ 5 000 jetons. L'ancienne borne, 2 000
#: caractères, coupait la requête du scénario HPAI (3 075 caractères) à 239 caractères
#: AVANT son bloc « ET (virus aviaires) » : les 201 articles « pertinents » de ce scénario
#: ont été classés par similarité avec un plongement qui ne contenait ni « influenza » ni
#: « H5N1 ». Tout ce qui en découle (extraction, estimations groupées, brief) reposait sur
#: un classement qui n'avait jamais vu la maladie.
EMBED_QUERY_MAX_CHARS = 20_000


def embedding_text_for_query(query: str) -> str:
    """Le texte RÉELLEMENT plongé pour une requête : la requête entière, sans les tags.

    Les tags de champ de PubMed (`[mh]`, `[tiab]`) n'apportent rien à un plongement et
    en gaspillent la place ; les retirer d'abord, c'est exactement ce que fait le reste du
    module pour les sources qui ne les parlent pas. Les blancs sont repliés, et la borne
    est celle du modèle, pas un 2 000 arbitraire. PUR/testable."""
    from .search import _strip_field_tags
    text_ = " ".join(_strip_field_tags(query or "").split())
    return text_[:EMBED_QUERY_MAX_CHARS]


#: Articles par appel d'embedding dans le repli OpenAI du scoring.
_FALLBACK_BATCH = 100
#: Lots d'affilée sans un seul article scoré avant d'arrêter le repli. Un lot qui échoue
#: est sauté et le suivant tente sa chance ; trois de suite, c'est une panne qui dure
#: (réseau, modèle refusé) et continuer ne ferait que la répéter. Le reste est dit, et
#: le scoring automatique repassera (cf. _autoscore_note).
_FALLBACK_MAX_STALLED = 3


def title_abstract_text(title: str | None, abstract: str | None) -> str:
    """Le texte d'un chunk `title_abstract` tel que l'ingestion l'écrit (api/sources.py,
    living_review_scheduler.py) : le titre, une ligne vide, le résumé. PUR."""
    return f"{title or ''}\n\n{abstract or ''}".strip()


def _scorable_text_sql(doc: str = "ld") -> str:
    """Un article a-t-il assez de texte pour porter un chunk `title_abstract`, donc un
    score ? La règle de `_backfill_title_abstract_chunks`, la même ici, pour qu'un article
    ne soit pas scorable par un chemin et pas par l'autre."""
    return f"length(btrim(coalesce({doc}.title, '') || ' ' || coalesce({doc}.abstract, ''))) >= 30"


def _fallback_text(row: dict) -> str:
    """Le texte que le repli plonge pour un article : EXACTEMENT celui que le worker plonge
    pour son chunk `title_abstract` (documents._embed_one_call : le contenu du chunk, coupé
    par jetons). Le chunk en base quand il existe, sinon le même texte reconstruit.

    Le repli plongeait `titre + résumé[:1500]` quand le worker plonge le résumé entier : un
    même article recevait deux cosinus selon le chemin pris, face à un seuil dur."""
    return _truncate_to_tokens(row.get("chunk_text") or title_abstract_text(row.get("title"),
                                                                           row.get("abstract")))


def _retryable_unscored(scenario_id: str) -> int:
    """Les articles sans score qu'un nouveau passage peut scorer : ceux qui ont du texte.
    Un article sans titre ni résumé exploitables n'en aura jamais, inutile de relancer pour
    lui."""
    with engine.connect() as conn:
        return int(conn.execute(text(f"""
            SELECT COUNT(*) FROM article_scenarios asn
            JOIN literature_document ld ON ld.id = asn.document_id
            WHERE asn.scenario_id = :sid AND ld.is_duplicate IS NOT TRUE
              AND {unscored_sql('ld', 'asn')} AND {_scorable_text_sql('ld')}
        """), {"sid": scenario_id}).scalar() or 0)


def _run_semantic_rerank_inline(scenario_id: str, query: str, report: dict | None = None,
                                missing_only: bool = False) -> int:
    """Score sémantique (cosinus requête↔article) du corpus, mis dans similarity_score.
    Renvoie le nombre de scores écrits ; `report`, s'il est fourni, reçoit le détail.

    LE SCORE EST CELUI DU TITRE ET DU RÉSUMÉ. Il était le MAXIMUM sur tous les chunks de
    l'article, résumé ET sections du texte intégral : un article en texte intégral tirait
    trente fois, un article réduit à son résumé une seule, si bien qu'à seuil égal le texte
    intégral passait la porte plus facilement, et que récupérer un texte intégral ne
    pouvait que monter un score. Sur HPAI_last, 869 des 1 043 pertinents ont un texte
    intégral, contre 4 176 des 15 660 du corpus. Les chunks du texte intégral servent
    toujours la recherche (assistant, extraits), pas la porte.

    Deux chemins, un seul texte :
      1. rapide : le cosinus pgvector EN BASE, en une requête, pour les articles dont le
         chunk `title_abstract` est déjà vectorisé ;
      2. repli OpenAI pour les autres, qui plonge le MÊME texte que le worker
         (`_fallback_text`), par lots, en parcourant les identifiants (`ld.id > dernier`) :
         plus de plafond (il était de 200 lots de 100, soit 20 000 articles par passage),
         un lot en échec est sauté et compté au lieu d'arrêter tout, et trois lots d'affilée
         sans progrès arrêtent le repli. Pendant la pause OpenAI (quota), le passage
         s'arrête « partiel » au lieu de ne rien dire, et le scoring automatique est
         réarmé pour la fin de la pause.

    `missing_only` : le repli ne traite que les articles sans score (le passage
    automatique). Sinon il reprend aussi ceux dont le score ne vient pas du chunk vectorisé
    (calculé par l'ancienne règle) : c'est un RE-scoring. Un article qui n'a ni chunk
    vectorisé ni texte exploitable perd alors son ancien score, qu'aucune règle actuelle ne
    peut plus produire (`cleared`).

    Ce que dit `report` : `fast` et `fallback` (scores écrits par chaque chemin),
    `newly_scored`, `changed` (scores qui ont bougé), `crossed_threshold` (articles passés
    d'un côté à l'autre du seuil), `cleared`, `fallback_failed` (articles des lots en
    échec), `partial` et `partial_reason` (`cooldown`, `failures`, `fast_path` ou
    `query_embedding`),
    `retry_after_s`, `unscored` (articles qui attendent un score pour entrer) et
    `retryable` (ceux qu'un nouveau passage peut scorer)."""
    rep = report if report is not None else {}
    rep.update({"fast": 0, "fallback": 0, "newly_scored": 0, "changed": 0,
                "crossed_threshold": 0, "cleared": 0, "fallback_failed": 0,
                "fallback_batches_failed": 0, "partial": False, "partial_reason": None,
                "retry_after_s": None, "unscored": None, "retryable": None, "error": None})
    try:
        from llm_usage import MeteredOpenAI as _OAI
        _client = _OAI(timeout=90.0)
        q_emb = _client.embeddings.create(model=_model("embedding"),
                                          input=embedding_text_for_query(query)).data[0].embedding
        q_str = str(q_emb)
        thr = _get_scenario_threshold(scenario_id)
    except Exception as _e:
        logger.error(f"Rerank inline {scenario_id} fatal: {_e}", exc_info=True)
        rep["error"] = f"{type(_e).__name__}: {_e}"
        # Rien n'a été scoré : le passage le dit, et le scoring automatique repassera.
        rep["partial"], rep["partial_reason"] = True, "query_embedding"
        if _is_openai_quota_error(_e):
            _trip_openai_cooldown()
        _close_scoring_report(scenario_id, rep)
        return 0

    # 1) Rapide : cosinus pgvector en base, sur le chunk title_abstract SEUL. L'ancien score
    #    est relu dans la même instruction : le passage dit combien de scores ont bougé.
    try:
        # `missing_only` (le scoring automatique) ne touche QUE les articles sans score : sans
        # cette borne, le passage rapide réécrivait tous les scores du scénario à la nouvelle
        # règle, et un scénario existant changeait de sous-ensemble pertinent parce qu'un
        # seul de ses articles attendait un score.
        _only_missing = " AND asn.similarity_score IS NULL" if missing_only else ""
        with engine.begin() as _c:
            fast = _c.execute(text("""
                WITH upd AS (
                    UPDATE article_scenarios asn
                    SET similarity_score = sub.sim
                    FROM (
                        SELECT c.document_id,
                               MAX(1 - (c.embedding <=> CAST(:q AS vector))) AS sim,
                               MAX(a.similarity_score) AS old
                        FROM document_chunk c
                        JOIN article_scenarios a
                          ON a.document_id = c.document_id AND a.scenario_id = :sid
                        WHERE c.embedding IS NOT NULL
                          AND c.chunk_type = 'title_abstract'
                        GROUP BY c.document_id
                    ) sub
                    WHERE asn.scenario_id = :sid AND asn.document_id = sub.document_id""" + _only_missing + """
                    RETURNING sub.old AS old, sub.sim AS new
                )
                SELECT COUNT(*) AS n,
                       COUNT(*) FILTER (WHERE old IS NULL) AS newly,
                       COUNT(*) FILTER (WHERE old IS NOT NULL AND abs(old - new) > 1e-6) AS changed,
                       COUNT(*) FILTER (WHERE (COALESCE(old, 0) >= :thr) <> (new >= :thr)) AS crossed
                FROM upd
            """), {"q": q_str, "sid": scenario_id, "thr": thr}).mappings().first() or {}
        rep["fast"] = int(fast.get("n") or 0)
        rep["newly_scored"] += int(fast.get("newly") or 0)
        rep["changed"] += int(fast.get("changed") or 0)
        rep["crossed_threshold"] += int(fast.get("crossed") or 0)
    except Exception as _e:
        # Le repli ne dépend pas de pgvector : il passe quand même, et le passage se dit
        # partiel au lieu de rendre 0 pour tout le corpus.
        logger.warning(f"Rerank {scenario_id}: cosinus pgvector en échec ({_e}), repli seul.")
        rep["partial"], rep["partial_reason"] = True, "fast_path"

    # 2) Un ancien score qu'aucune règle actuelle ne peut plus produire : ni chunk
    #    title_abstract vectorisé, ni texte pour le repli. Il venait du texte intégral.
    if not missing_only:
        try:
            with engine.begin() as _c:
                rep["cleared"] = int(_c.execute(text(f"""
                    WITH clr AS (
                        UPDATE article_scenarios asn SET similarity_score = NULL
                        FROM literature_document ld
                        WHERE asn.scenario_id = :sid AND ld.id = asn.document_id
                          AND asn.similarity_score IS NOT NULL
                          AND NOT ({_scorable_text_sql('ld')})
                          AND NOT EXISTS (
                              SELECT 1 FROM document_chunk c
                              WHERE c.document_id = ld.id AND c.chunk_type = 'title_abstract'
                                AND c.embedding IS NOT NULL)
                        RETURNING 1
                    )
                    SELECT COUNT(*) FROM clr
                """), {"sid": scenario_id}).scalar() or 0)
        except Exception as _e:
            logger.warning(f"Rerank {scenario_id}: anciens scores non effacés ({_e}).")

    # 3) Repli OpenAI : le même texte que le worker, par lots, sans plafond.
    _need = ("asn.similarity_score IS NULL" if missing_only else
             "(asn.similarity_score IS NULL OR NOT EXISTS ("
             " SELECT 1 FROM document_chunk c WHERE c.document_id = ld.id"
             " AND c.chunk_type = 'title_abstract' AND c.embedding IS NOT NULL))")
    import numpy as _np
    _q = _np.asarray(q_emb, dtype=float)
    _qn = float(_np.linalg.norm(_q)) or 1.0
    after, stalled = 0, 0
    while True:
        try:
            with engine.connect() as _conn:
                batch = _conn.execute(text(f"""
                    SELECT ld.id, ld.title, ld.abstract, asn.similarity_score AS old,
                           (SELECT c.content FROM document_chunk c
                             WHERE c.document_id = ld.id AND c.chunk_type = 'title_abstract'
                             ORDER BY c.id LIMIT 1) AS chunk_text
                    FROM literature_document ld
                    JOIN article_scenarios asn ON asn.document_id = ld.id AND asn.scenario_id = :sid
                    WHERE ld.id > :after
                      AND ld.is_duplicate IS NOT TRUE
                      AND {_need}
                      AND {_scorable_text_sql('ld')}
                    ORDER BY ld.id LIMIT :n
                """), {"sid": scenario_id, "after": after, "n": _FALLBACK_BATCH}).mappings().fetchall()
        except Exception as _e:
            logger.warning(f"Rerank fallback {scenario_id}: lecture du lot en échec ({_e}).")
            rep["partial"], rep["partial_reason"] = True, "failures"
            break
        if not batch:
            break
        if _openai_in_cooldown():
            # La pause OpenAI (quota) : le passage s'arrête PARTIEL, dit combien restent
            # sans score, et le scoring automatique est réarmé pour la fin de la pause.
            # Il sautait le repli sans un mot, et ces articles restaient à 0 pour la porte.
            import time as _t
            rep["partial"], rep["partial_reason"] = True, "cooldown"
            rep["retry_after_s"] = max(0, int(_OPENAI_QUOTA_COOLDOWN_UNTIL[0] - _t.time()) + 1)
            break
        after = int(batch[-1]["id"])
        try:
            data = _client.embeddings.create(model=_model("embedding"),
                                             input=[_fallback_text(dict(r)) for r in batch]).data
            by_index = {int(getattr(e, "index", j)): e.embedding for j, e in enumerate(data)}
            ups = []
            for j, r in enumerate(batch):
                if j not in by_index:
                    continue
                _d = _np.asarray(by_index[j], dtype=float)
                sim = max(0.0, min(1.0, float(_q @ _d) / (_qn * (float(_np.linalg.norm(_d)) or 1.0))))
                ups.append({"score": sim, "doc_id": r["id"], "sid": scenario_id})
                old = r["old"]
                if old is None:
                    rep["newly_scored"] += 1
                elif abs(float(old) - sim) > 1e-6:
                    rep["changed"] += 1
                if (float(old or 0) >= thr) != (sim >= thr):
                    rep["crossed_threshold"] += 1
            if ups:
                with engine.begin() as _c:
                    _c.execute(text("""
                        UPDATE article_scenarios SET similarity_score = :score
                        WHERE document_id = :doc_id AND scenario_id = :sid
                    """), ups)
            rep["fallback"] += len(ups)
            rep["fallback_failed"] += len(batch) - len(ups)
            stalled = 0 if ups else stalled + 1
        except Exception as _e:
            # Le lot est SAUTÉ (le curseur a déjà avancé) et compté ; les suivants tentent
            # leur chance. Il arrêtait tout le repli, si bien qu'une erreur passagère
            # laissait le reste du corpus sans score, à 0 pour la porte.
            logger.warning(f"Rerank fallback {scenario_id} (lot après l'id {after}): {_e}")
            rep["fallback_failed"] += len(batch)
            rep["fallback_batches_failed"] += 1
            stalled += 1
            if _is_openai_quota_error(_e):
                _trip_openai_cooldown()
        if stalled >= _FALLBACK_MAX_STALLED:
            rep["partial"], rep["partial_reason"] = True, "failures"
            break

    _close_scoring_report(scenario_id, rep)
    return rep["fast"] + rep["fallback"]


def _close_scoring_report(scenario_id: str, rep: dict) -> None:
    """La fin de tout passage de scoring cosinus, réussi ou non : ce qui reste sans score,
    une ligne de journal qui dit ce qui a changé, les cartes périmées quand le
    sous-ensemble pertinent a bougé, et la date à laquelle le scoring automatique pourra
    repasser. Jamais d'erreur : c'est un compte rendu."""
    try:
        rep["unscored"] = scenario_unscored_count(scenario_id)
        rep["retryable"] = _retryable_unscored(scenario_id)
    except Exception as _e:
        logger.warning(f"Rerank {scenario_id}: compte des non scorés impossible ({_e}).")
    if rep.get("fallback_failed") and rep.get("retryable"):
        # Des lots sautés ont laissé des articles qu'un nouvel essai peut scorer : le passage
        # est partiel même s'il est allé au bout, et le dit.
        rep["partial"] = True
        if rep.get("partial_reason") in (None, "fast_path"):
            rep["partial_reason"] = "failures"
    logger.info(f"Rerank {scenario_id}: {rep.get('fast', 0)} via pgvector + {rep.get('fallback', 0)} "
                f"via OpenAI (fallback) ; {rep.get('changed', 0)} scores modifiés, "
                f"{rep.get('newly_scored', 0)} nouveaux, {rep.get('cleared', 0)} effacés, "
                f"{rep.get('crossed_threshold', 0)} ont changé de côté du seuil ; "
                f"{rep.get('fallback_failed', 0)} en échec, {rep.get('unscored')} sans score"
                + (f" (partiel : {rep.get('partial_reason')})" if rep.get("partial") else "") + ".")
    if rep.get("newly_scored") or rep.get("changed") or rep.get("cleared"):
        try:
            with engine.begin() as _c:
                reset_corpus_derived_caches(_c, scenario_id)
        except Exception as _e:
            logger.warning(f"Rerank {scenario_id}: caches dérivés non invalidés ({_e}).")
    try:
        _autoscore_note(scenario_id, rep)
    except Exception as _e:                                  # noqa: BLE001
        logger.warning(f"Rerank {scenario_id}: réarmement du scoring automatique ({_e}).")


#: Le rerank est limité en débit côté fournisseur, et le dépassement est la panne
#: NORMALE d'un balayage, pas l'exception : sur un rattrapage de 119 lots, 99 ont échoué
#: d'affilée après une vingtaine de succès. Sans reprise, un rattrapage ne finit jamais.
_RERANK_RETRIES = _env_int("RERANK_RETRIES", 5, 0)
_RERANK_BACKOFF_S = 4.0

#: Caractères envoyés au cross-encoder par article (titre, ligne vide, résumé). Le résumé
#: était coupé à 1 500 caractères : un résumé structuré y perdait ses résultats et ses
#: conclusions, c'est-à-dire ce qui dit si l'article répond à la question, avant d'être
#: jugé. Cohere facture la requête et ses documents, et découpe au-delà de 500 jetons
#: (requête comprise) : le coût croît à peu près linéairement avec cette borne.
RERANK_DOC_MAX_CHARS = _env_int("RERANK_DOC_MAX_CHARS", 4000, 200)
#: La requête envoyée au cross-encoder, au plus. Bien au-dessus d'une question, et d'une
#: requête booléenne une fois ses tags de champ retirés ; si elle est coupée, le journal
#: le dit.
RERANK_QUERY_MAX_CHARS = 4000


def rerank_document_text(title: str | None, abstract: str | None) -> str:
    """Le texte d'un article tel que le cross-encoder le reçoit. PUR."""
    return title_abstract_text(title, abstract)[:RERANK_DOC_MAX_CHARS]


def rerank_query_text(query: str | None, scenario_id: str | None = None) -> str:
    """La requête telle que le cross-encoder la reçoit : sans les tags de champ PubMed,
    blancs repliés, bornée à RERANK_QUERY_MAX_CHARS.

    Elle partait brute, coupée à 4 000 caractères AVEC ses `[tiab]` et `[mh]` : une longue
    requête booléenne perdait sa fin, le défaut exact que décrit EMBED_QUERY_MAX_CHARS pour
    l'ancienne coupe du plongement, et cela se voyait dans l'ordre du rerank et dans sa
    porte. Retirer les tags d'abord rend la place ; ce qui reste coupé est journalisé
    quand `scenario_id` est donné (une fois par balayage, pas par lot)."""
    from .search import _strip_field_tags
    text_ = " ".join(_strip_field_tags(query or "").split())
    if len(text_) > RERANK_QUERY_MAX_CHARS:
        if scenario_id is not None:
            logger.warning(f"Cross-encoder rerank {scenario_id}: requête de {len(text_)} caractères "
                           f"(sans les tags de champ) coupée à {RERANK_QUERY_MAX_CHARS} : la fin "
                           f"n'est pas vue par le cross-encoder.")
        text_ = text_[:RERANK_QUERY_MAX_CHARS]
    return text_


def _cohere_error_text(resp) -> str:
    """Ce que Cohere dit de son refus : le champ `message` de son JSON, sinon le corps."""
    try:
        payload = resp.json()
        if isinstance(payload, dict) and payload.get("message"):
            return str(payload["message"])
    except Exception:
        pass
    return str(getattr(resp, "text", "") or "")


def _cohere_quota_exhausted(message: str) -> bool:
    """Un 429 qui ne passera pas en attendant une minute : le quota MENSUEL d'une clé
    d'essai (« limited to 1000 API calls / month »). Le plafond par minute, lui, se
    réessaie."""
    return "month" in (message or "").lower()


def _cohere_rerank(query: str, docs: list[str], model: str = "rerank-v3.5",
                   failure: dict | None = None) -> list[float] | None:
    """Cross-encoder rerank via Cohere, avec reprise sur limite de débit.

    Renvoie un score par document (aligné sur `docs`), ou None après épuisement des
    tentatives. Appel REST direct, aucune dépendance ajoutée. En cas d'échec, `failure`
    (si fourni) reçoit le code HTTP, le message du fournisseur et `permanent` : vrai quand
    réessayer ne changera rien, pour que l'appelant cesse d'envoyer des lots.

    429 et 5xx sont RÉESSAYÉS avec une attente qui double, en respectant `Retry-After`
    quand le serveur le donne. Une erreur de requête (4xx autre que 429) ne l'est pas,
    puisque la réessayer donnera la même réponse, ni un 429 de quota mensuel épuisé.

    Le code le promettait déjà, et réessayait pourtant les 400, 401 et 403 : il passait
    par `raise_for_status()`, dont l'exception est une `RequestException`, attrapée par la
    branche « panne réseau ». Une clé refusée coûtait ainsi deux minutes d'attentes par
    lot, et un balayage de 157 lots plus de cinq heures, sans un mot à l'écran."""
    key = os.getenv("COHERE_API_KEY")
    if not key or not docs:
        return None
    import time as _t

    import requests as _rq

    def _failed(status: int | None, message: str, permanent: bool) -> None:
        if failure is not None:
            failure.update({"status": status, "message": (message or "")[:300],
                            "permanent": permanent})
        return None

    wait = _RERANK_BACKOFF_S
    for attempt in range(_RERANK_RETRIES + 1):
        try:
            resp = _rq.post(
                "https://api.cohere.com/v2/rerank",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": model, "query": rerank_query_text(query), "documents": docs,
                      "top_n": len(docs)},
                timeout=60,
            )
        except _rq.exceptions.RequestException as _e:
            # Panne réseau ou délai dépassé : réessayable, contrairement à un 400.
            if attempt >= _RERANK_RETRIES:
                logger.warning(f"Cohere rerank failed after {_RERANK_RETRIES} retries: {_e}")
                return _failed(None, f"{type(_e).__name__}: {_e}", False)
            _t.sleep(wait)
            wait *= 2
            continue
        try:
            if resp.status_code == 429 or resp.status_code >= 500:
                detail = _cohere_error_text(resp)
                if resp.status_code == 429 and _cohere_quota_exhausted(detail):
                    logger.warning(f"Cohere rerank: monthly quota exhausted: {detail[:200]}")
                    return _failed(429, detail, True)
                if attempt >= _RERANK_RETRIES:
                    logger.warning(f"Cohere rerank: {resp.status_code} after "
                                   f"{_RERANK_RETRIES} retries, giving up on this batch")
                    return _failed(resp.status_code, detail, False)
                # Le serveur sait mieux que nous combien attendre quand il le dit.
                try:
                    hinted = float(resp.headers.get("Retry-After") or 0)
                except ValueError:
                    hinted = 0.0
                _t.sleep(max(wait, hinted))
                wait *= 2
                continue
            if resp.status_code >= 400:
                detail = _cohere_error_text(resp)
                logger.warning(f"Cohere rerank: {resp.status_code} {detail[:200]}")
                return _failed(resp.status_code, detail, True)
            data = resp.json()
            scores: list[float | None] = [None] * len(docs)
            for item in data.get("results", []):
                idx = item.get("index")
                if idx is not None and 0 <= idx < len(docs):
                    scores[idx] = float(item.get("relevance_score", 0.0))
            return scores  # type: ignore[return-value]
        except Exception as _e:
            logger.warning(f"Cohere rerank failed: {_e}")
            return _failed(getattr(resp, "status_code", None), f"{type(_e).__name__}: {_e}", False)
    return None


def _describe_rerank_failure(failure: dict) -> str:
    """Une ligne lisible pour l'écran : « Cohere 429 : <message> »."""
    status = failure.get("status")
    head = f"Cohere {status}" if status else "Cohere"
    msg = (failure.get("message") or "").strip()
    return f"{head}: {msg}" if msg else head


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
#: Lots refusés d'affilée, reprises épuisées, avant d'arrêter le balayage. Chaque lot a
#: déjà attendu deux minutes de reprises : trois refus de suite ne sont plus une limite
#: par minute, et continuer ne ferait que brûler le temps et le quota restant.
_RERANK_MAX_CONSECUTIVE_FAILURES = _env_int("RERANK_MAX_CONSECUTIVE_FAILURES", 3, 1)


def _run_cross_encoder_rerank(scenario_id: str, query: str, top_k: int | None = None,
                              only_missing: bool = False, progress=None) -> dict[str, Any]:
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
    différence entre « rien à faire » et « tout a échoué ». `last_error` dit le dernier
    refus du fournisseur ; `stopped` dit pourquoi le balayage s'est arrêté avant la fin
    (refus définitif, ou `_RERANK_MAX_CONSECUTIVE_FAILURES` lots refusés d'affilée) ; les
    articles non notés restent sans score, donc gardés par la porte, jamais comptés zéro.

    `progress`, s'il est fourni, reçoit ces compteurs après chaque lot : c'est ce que lit
    l'écran pendant le balayage, qui affichait « Lancement du scoring... » pendant des
    heures sans rien dire."""
    out: dict[str, Any] = {"scored": 0, "batches_ok": 0, "batches_failed": 0, "candidates": 0,
                           "skipped_no_key": 0, "last_error": None, "stopped": None}

    def _report() -> None:
        if progress is not None:
            try:
                progress(dict(out))
            except Exception:
                pass

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
        _report()
        if not rows:
            return out
        import time as _t_batch
        consecutive = 0
        # Préparée une fois : la coupe éventuelle est dite une fois, pas à chaque lot.
        _query = rerank_query_text(query, scenario_id)
        for _b in range(0, len(rows), _RERANK_BATCH):
            if _b:
                _t_batch.sleep(_RERANK_PAUSE_S)
            chunk = rows[_b:_b + _RERANK_BATCH]
            docs = [rerank_document_text(r["title"], r["abstract"]) for r in chunk]
            failure: dict[str, Any] = {}
            scores = _cohere_rerank(_query, docs, failure=failure)
            ups = [{"s": s, "doc_id": r["id"], "sid": scenario_id}
                   for r, s in zip(chunk, scores or []) if s is not None]
            if not ups:
                # Le lot échoue, les suivants continuent : c'est tout l'intérêt des lots.
                # Mais pas indéfiniment : un refus définitif (clé refusée, quota du mois
                # épuisé) ou trois refus d'affilée arrêtent le balayage, et le disent.
                out["batches_failed"] += 1
                consecutive += 1
                out["last_error"] = _describe_rerank_failure(failure) if failure else \
                    "Cohere: no score returned"
                if failure.get("permanent") or consecutive >= _RERANK_MAX_CONSECUTIVE_FAILURES:
                    out["stopped"] = out["last_error"]
                    logger.warning(f"Cross-encoder rerank {scenario_id} arrêté après "
                                   f"{out['batches_failed']} lot(s) refusé(s) : {out['stopped']}")
                    _report()
                    break
                _report()
                continue
            consecutive = 0
            with engine.begin() as _c:
                _c.execute(text("""
                    UPDATE article_scenarios SET rerank_score = :s
                    WHERE document_id = :doc_id AND scenario_id = :sid
                """), ups)
            out["scored"] += len(ups)
            out["batches_ok"] += 1
            _report()
        logger.info(f"Cross-encoder rerank {scenario_id}: {out['scored']}/{out['candidates']} notés, "
                    f"{out['batches_ok']} lots OK, {out['batches_failed']} en échec.")
        return out
    except Exception as _e:
        logger.warning(f"Cross-encoder rerank {scenario_id} failed: {_e}")
        out["last_error"] = out["stopped"] = f"{type(_e).__name__}: {_e}"
        return out


# ═══════════════════════════════════════════════════════════════
# SCORING SÉMANTIQUE + EVIDENCE BRIEF LLM + VARIABLES + HEATMAP
# ═══════════════════════════════════════════════════════════════

# ─── SCORING SÉMANTIQUE POST-INGESTION ───────────────────────────────────────

_RERANK_JOBS: dict[str, dict] = {}
#: Le verrou qui rend « lire puis écrire » indivisible. Deux appels simultanés
#: pouvaient tous deux voir « pas de job » et démarrer leur fil.
_rerank_jobs_lock = threading.Lock()


#: Silence au-delà duquel un scoring « en cours » est tenu pour mort. Lu sur le dernier
#: signe de vie (`heartbeat`, posé après chaque lot), et non sur l'heure de départ : avec
#: les 180 secondes communes comptées depuis le départ, un balayage de 157 lots était
#: « mort » au bout de trois minutes, et le clic suivant en lançait un SECOND, en
#: parallèle du premier, contre le même fournisseur. Un lot, reprises comprises, tient
#: sous dix minutes.
_RERANK_STALE_S = 900.0


def _rerank_job_active(scenario_id: str) -> bool:
    return _job_is_active(_RERANK_JOBS.get(scenario_id), stale_after=_RERANK_STALE_S)


def _claim_rerank_job(scenario_id: str) -> str | None:
    """Vérifie qu'aucun scoring ne tourne et pose l'entrée « en cours », SOUS LE VERROU.

    Renvoie l'identifiant du job, ou None si un autre tourne déjà. La vérification se
    faisait hors du verrou, avant la pose de l'entrée : deux clics simultanés pouvaient
    tous deux la passer. L'entrée porte `started_at` ET `heartbeat` sous une seule forme
    (une entrée sans horodatage était jugée vieille de 1970, donc morte), et l'identifiant
    sert à ce qu'un fil ancien ne réécrive pas l'entrée d'un job plus récent."""
    import time as _t
    import uuid as _uuid
    with _rerank_jobs_lock:
        if _job_is_active(_RERANK_JOBS.get(scenario_id), stale_after=_RERANK_STALE_S):
            return None
        job_id = _uuid.uuid4().hex[:12]
        now = _t.time()
        _RERANK_JOBS[scenario_id] = {"status": "running", "updated": 0, "job": job_id,
                                     "started_at": now, "heartbeat": now}
        return job_id


def _rerank_job_update(scenario_id: str, job_id: str, **fields) -> None:
    """Avancement d'un scoring en cours (phase, lots, dernier refus) et signe de vie.
    N'écrit que dans l'entrée de CE job."""
    import time as _t
    with _rerank_jobs_lock:
        job = _RERANK_JOBS.get(scenario_id)
        if job is not None and job.get("job") == job_id and job.get("status") == "running":
            job.update(fields)
            job["heartbeat"] = _t.time()


def _rerank_job_finish(scenario_id: str, job_id: str, entry: dict) -> None:
    """L'état final d'un scoring, sauf si un job plus récent a pris la place."""
    with _rerank_jobs_lock:
        job = _RERANK_JOBS.get(scenario_id)
        if job is None or job.get("job") in (None, job_id):
            _RERANK_JOBS[scenario_id] = {**entry, "job": job_id}


def _cosine_left_behind(cos: dict | None) -> bool:
    """Le passage cosinus a laissé sans score des articles qu'un nouvel essai PEUT scorer :
    c'est ce que « partial » veut dire dans un statut. Un passage dont seul le chemin
    pgvector a échoué, mais dont le repli a tout rattrapé, n'a rien laissé derrière lui."""
    return bool(cos and cos.get("partial") and int(cos.get("retryable") or 0) > 0)


def _cosine_entry(cos: dict | None) -> dict:
    """Ce que le passage cosinus laisse à lire dans le statut d'un job : combien d'articles
    restent sans score, combien ont échoué, et ce que le RE-scoring a changé. Vide sans
    compte rendu (le rattrapage du seul rerank ne fait pas de cosinus)."""
    if not cos:
        return {}
    out = {
        "unscored": cos.get("unscored"),
        "unscored_retryable": cos.get("retryable"),
        "cosine_failed": cos.get("fallback_failed", 0),
        "cosine_changed": cos.get("changed", 0),
        "cosine_newly_scored": cos.get("newly_scored", 0),
        "cosine_cleared": cos.get("cleared", 0),
        "cosine_crossed_threshold": cos.get("crossed_threshold", 0),
        "cosine_partial": bool(cos.get("partial")),
        "cosine_partial_reason": cos.get("partial_reason"),
        "retry_after_s": cos.get("retry_after_s"),
    }
    if _cosine_left_behind(cos):
        out["message"] = (f"{cos.get('unscored') or 0} article(s) sans score, "
                          f"nouvel essai automatique ({cos.get('partial_reason')}).")
    return out


def _rerank_final_entry(n_cosine: int, ce: dict, cos: dict | None = None) -> dict:
    """Ce qu'un scoring terminé laisse à lire : le cosinus, le cross-encoder, et, quand le
    fournisseur a refusé, pourquoi. Un balayage arrêté par un refus finit en « error » :
    finir en « done » avec 100 articles sur 15 660 se lisait comme un succès. Un cosinus
    PARTIEL (pause OpenAI, lots en échec) finit en « partial » : des articles restent hors
    de la porte faute de score, le statut le dit avec leur nombre, et un nouvel essai est
    prévu."""
    entry = {
        "status": ("error" if ce.get("stopped")
                   else "partial" if _cosine_left_behind(cos) else "done"),
        "updated": n_cosine,
        "reranked": ce.get("scored", 0),
        "rerank_candidates": ce.get("candidates", 0),
        "rerank_batches_ok": ce.get("batches_ok", 0),
        "rerank_batches_failed": ce.get("batches_failed", 0),
        "rerank_skipped_no_key": bool(ce.get("skipped_no_key")),
        "rerank_last_error": ce.get("last_error"),
        "rerank_stopped": ce.get("stopped"),
        **_cosine_entry(cos),
    }
    if ce.get("stopped"):
        entry["error"] = ce["stopped"]
    return entry


def _ensure_scenario_settings_table():
    """Table pour stocker les paramètres par scénario (seuil, etc.).

    `similarity_threshold` n'a PLUS de valeur par défaut : la DDL écrivait 0.45 dans toute
    ligne créée pour une autre raison (cache de visualisation, seuil de rerank, découpage),
    si bien que « jamais choisi » se lisait « choisi 0.45 », et qu'un changement du seuil
    de départ n'aurait jamais atteint ces scénarios. Rien n'est stocké tant qu'une personne
    n'a pas réglé le seuil, et `similarity_threshold_set_at` date ce réglage (cf.
    scenario_store.threshold_chosen_sql)."""
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS scenario_settings (
                scenario_id     VARCHAR(80) PRIMARY KEY,
                similarity_threshold FLOAT,
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
        # Idempotent : sans défaut, DROP DEFAULT ne fait rien.
        conn.execute(text(
            "ALTER TABLE scenario_settings ALTER COLUMN similarity_threshold DROP DEFAULT"))
        conn.execute(text(
            "ALTER TABLE scenario_settings ADD COLUMN IF NOT EXISTS similarity_threshold_set_at TIMESTAMP"))
        # La taille de l'échantillon d'audit du double aveugle, pris sous le seuil (0 = éteint).
        conn.execute(text(
            "ALTER TABLE scenario_settings ADD COLUMN IF NOT EXISTS double_blind_below_sample INTEGER DEFAULT 0"))
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
            input=[embedding_text_for_query(query)],
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
                -- `study_design` est le devis RÉSOLU, par l'expression partagée : deux
                -- passes indépendantes écrivent un devis depuis le même résumé (PICO et
                -- métadonnées), et chacune se tait parfois en écrivant un marqueur.
                -- Lire la seule colonne laissait 61 % du corpus HPAI sans devis pour la
                -- notation des affirmations, pendant que le profil de preuve, juste à
                -- côté, en classait 95 % : une étude transversale de 65 622 travailleurs
                -- exposés soutenait une affirmation marquée « Non évaluée », trois
                -- lignes sous un profil qui la classait en certitude faible.
                SELECT ld.id, ld.title, ld.abstract, ld.year, ld.journal, ld.authors, ld.doi,
                       NULLIF({_raw_design('ld')}, '') AS study_design,
                       ld.pico_json, ld.citation_count,
                       {screening_status_sql('ld', 'asn')} AS screening_status,
                       ld.quality_score, asn.similarity_score,
                       (ld.pico_json IS NOT NULL) AS has_pico,
                       -- MÊME ordre que l'onglet Corpus, par la fonction partagée : la
                       -- docstring de l'export promettait « le même ensemble et le même
                       -- ordre », et en livrait un troisième.
                       ROW_NUMBER() OVER (ORDER BY
                           {relevance_order_sql('ld', 'asn', ':threshold')}) AS rn
                FROM literature_document ld
                JOIN article_scenarios asn ON asn.document_id = ld.id AND asn.scenario_id = :sid
                -- La porte de pertinence, appelée et non recopiée : jamais un doublon,
                -- jamais un article exclu par un relecteur, sinon inclus à la main OU
                -- au-dessus des deux seuils. Un article non scoré compte pour 0.
                --
                -- Et RIEN d'autre : un filtre `project_context = 'literev'` vivait ici, et
                -- nulle part ailleurs. Les compteurs, la PRISMA, les graphes, les
                -- découpages et le double aveugle comptaient pertinent un article d'un
                -- autre contexte que ce chargeur, donc les exports et les extractions,
                -- ne rendaient jamais. « Pertinent » veut dire la même chose partout.
                WHERE {relevant_gate_sql('ld', 'asn', ':threshold')}
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

    job_id = _claim_rerank_job(scenario_id)
    if job_id is None:
        # L'écran affichait « Lancement du scoring... » et attendait un « done » qui
        # pouvait ne venir que des heures plus tard : il reçoit maintenant l'avancement du
        # scoring en cours, et peut le dire.
        return {"status": "already_running", "scenario_id": scenario_id,
                "job": {k: v for k, v in (_RERANK_JOBS.get(scenario_id) or {}).items()
                        if k != "job"}}

    def _run():
        try:
            n = 0
            _cos: dict[str, Any] | None = None
            if not missing_only:
                _rerank_job_update(scenario_id, job_id, phase="cosine")
                _backfill_title_abstract_chunks(scenario_id)  # docs sans chunk -> searchable
                _cos = {}
                n = _run_semantic_rerank_inline(scenario_id, query, report=_cos)
            # Recalcul COMPLET : après le cosinus, relancer AUSSI le cross-encoder Cohere
            # sur le sous-ensemble pertinent - sinon « Recalculer scores » ne rafraîchissait
            # que le cosinus et les rerank_score restaient figés/partiels.
            _rerank_job_update(scenario_id, job_id, phase="rerank", updated=n,
                               **_cosine_entry(_cos))
            _ce: dict[str, Any] = {"scored": 0, "candidates": 0, "batches_failed": 0,
                                   "skipped_no_key": 0}
            try:
                _ce = _run_cross_encoder_rerank(
                    scenario_id, query, only_missing=missing_only,
                    progress=lambda p: _rerank_job_update(
                        scenario_id, job_id, phase="rerank",
                        reranked=p.get("scored", 0), rerank_candidates=p.get("candidates", 0),
                        rerank_batches_ok=p.get("batches_ok", 0),
                        rerank_batches_failed=p.get("batches_failed", 0),
                        rerank_last_error=p.get("last_error")))
                logger.info(f"Rerank manuel {scenario_id}: {n} cosinus + {_ce['scored']} cross-encoder.")
            except Exception as _ece:
                logger.warning(f"cross-encoder (recalcul manuel) {scenario_id}: {_ece}")
                _ce["stopped"] = f"{type(_ece).__name__}: {_ece}"
            _maps_follow_rerank(scenario_id, _ce)
            # Le job disait « done » avec le seul compte du cosinus, quoi qu'ait fait le
            # cross-encoder : un échec total ressortait identique à un succès. Il porte
            # maintenant les deux, dit si des lots ont échoué, et pourquoi il s'est arrêté.
            _rerank_job_finish(scenario_id, job_id, _rerank_final_entry(n, _ce, _cos))
        except Exception as e:
            # Sans ce filet, une exception laisse le job en "running" pour toujours
            # (→ "already_running" + badge "recalcul en cours" figé jusqu'au restart).
            logger.error(f"Rerank job {scenario_id}: {e}", exc_info=True)
            _rerank_job_finish(scenario_id, job_id, {"status": "error", "error": str(e), "updated": 0})

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "started", "scenario_id": scenario_id, "query": query}


def _maps_follow_rerank(scenario_id: str, ce: dict | None) -> None:
    """Des scores de rerank viennent d'être écrits : ils déplacent la porte quand un seuil
    de rerank est posé, et l'ordre des cartes plafonnées dans tous les cas. Ce qui avait
    été calculé sur l'état d'avant est périmé (reset_corpus_derived_caches)."""
    if not (ce or {}).get("scored"):
        return
    try:
        with engine.begin() as _c:
            reset_corpus_derived_caches(_c, scenario_id)
    except Exception as _e:                                  # noqa: BLE001
        logger.warning(f"Rerank {scenario_id}: caches dérivés non invalidés ({_e}).")


@app.get("/scenarios/{scenario_id}/rerank/status")
def get_rerank_status(scenario_id: str) -> dict[str, Any]:
    """Statut du job de reranking, ET l'état réel de la couverture en base.

    Le job vit en mémoire, donc tout redémarrage le remet à « idle », et l'API redémarre
    à chaque déploiement. « idle » ne voulait donc pas dire « à jour », il voulait dire
    « je ne me souviens de rien ». Les deux compteurs ci-dessous viennent de la base, en
    UNE instruction, et disent ce qui est vrai maintenant : combien d'articles du corpus
    pourraient porter un score, et combien n'en ont pas. L'interface peut enfin écrire
    sur le bouton ce qu'il va faire.

    Un job « running » sans signe de vie depuis `_RERANK_STALE_S` se lit « stalled » :
    l'écran qui attendait sa fin cesse d'attendre, et un nouveau lancement est permis."""
    out = dict(_RERANK_JOBS.get(scenario_id, {"status": "idle"}))
    out.pop("job", None)
    if out.get("status") == "running" and not _rerank_job_active(scenario_id):
        out["status"] = "stalled"
    try:
        out.update(_rerank_coverage(scenario_id))
        # Et la similarité : les articles qui attendent un score pour pouvoir entrer.
        out["similarity_unscored"] = scenario_unscored_count(scenario_id)
    except Exception as _e:
        logger.warning(f"rerank coverage {scenario_id}: {_e}")
    return out


def _rerank_coverage(scenario_id: str, conn=None) -> dict[str, int]:
    """Combien d'articles du corpus peuvent porter un score de rerank (`scorable`),
    combien n'en ont pas encore (`missing`), et combien n'en auront jamais faute de
    résumé exploitable (`unscorable`). Une instruction, lue en base."""
    sql = text("""
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
    """)
    if conn is None:
        with engine.connect() as _c:
            row = _c.execute(sql, {"sid": scenario_id}).mappings().first() or {}
    else:
        row = conn.execute(sql, {"sid": scenario_id}).mappings().first() or {}
    # Un article sans résumé utilisable ne peut PAS recevoir de score : le dire, plutôt
    # que de laisser croire qu'un lancement de plus le rattrapera.
    return {"scorable": int(row.get("scorable") or 0), "missing": int(row.get("missing") or 0),
            "unscorable": int(row.get("unscorable") or 0)}


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

    job_id = _claim_rerank_job(scenario_id)
    if job_id is None:
        return {"status": "already_running", "scenario_id": scenario_id,
                "job": {k: v for k, v in (_RERANK_JOBS.get(scenario_id) or {}).items()
                        if k != "job"}}

    def _run():
        try:
            ids = _boolean_corpus_ids(boolean, filters)        # base LOCALE uniquement
            n_corpus = _set_scenario_corpus(scenario_id, ids)  # fixe l'appartenance
            _n_dup_rows = _dedup_scenario_links(scenario_id)   # un seul lien / article distinct
            n_corpus -= _n_dup_rows
            # L'appartenance vient de changer : les cartes décrivaient l'ancien corpus, et
            # le scoring qui suit ne voit pas les articles RETIRÉS. Elles restaient servies
            # jusqu'à 30 jours. (Le scoring périme de nouveau ce qu'il change lui-même.)
            with engine.begin() as _c:
                reset_corpus_derived_caches(_c, scenario_id)
            # PRISMA : une reconstruction n'interroge AUCUNE source. Elle rejoue la
            # requête booléenne sur la bibliothèque locale, et ses doublons sont les
            # lignes fusionnées par la dédup.
            #
            # Elle écrasait les chiffres de la dernière VRAIE recherche par une
            # identification à une seule source fabriquée (« db_cache »), et le panneau
            # continuait d'appeler le résultat une recherche : la trace de ce qui avait
            # été interrogé, et de ce qui avait échoué, disparaissait sans retour. On
            # garde donc ce que la dernière recherche avait établi, sous son propre nom.
            _figs = _prisma_identification_figures(
                {}, len(set(ids)), _n_dup_rows, max(0, n_corpus), method="rebuild",
                records_from_library=len(ids))
            _prev = _load_prisma_identification(scenario_id) or {}
            if _prev.get("method") == "populate":
                _figs["last_search"] = {
                    k: _prev.get(k) for k in (
                        "computed_at", "records_identified", "records_identified_databases",
                        "records_by_source", "source_outcomes", "sources_launched",
                        "sources_searched", "sources_failed", "sources_skipped",
                        "sources_cut_off", "per_source_cap", "federation_incomplete")
                    if _prev.get(k) is not None}
            _store_prisma_identification(scenario_id, _figs)
            _rerank_job_update(scenario_id, job_id, phase="cosine")
            _backfill_title_abstract_chunks(scenario_id)       # chunks résumé manquants
            _cos: dict[str, Any] = {}
            n = _run_semantic_rerank_inline(scenario_id, query or boolean, report=_cos)
            _rerank_job_update(scenario_id, job_id, phase="rerank", updated=n, **_cosine_entry(_cos))
            _ce: dict[str, Any] = {}
            try:
                _ce = _run_cross_encoder_rerank(                             # Cohere (si clé)
                    scenario_id, query or boolean,
                    progress=lambda p: _rerank_job_update(
                        scenario_id, job_id, reranked=p.get("scored", 0),
                        rerank_candidates=p.get("candidates", 0),
                        rerank_batches_failed=p.get("batches_failed", 0),
                        rerank_last_error=p.get("last_error")))
            except Exception as _ece:
                logger.warning(f"rebuild-corpus cross-encoder {scenario_id}: {_ece}")
                _ce = {"stopped": f"{type(_ece).__name__}: {_ece}"}
            _maps_follow_rerank(scenario_id, _ce)
            _rerank_job_finish(scenario_id, job_id,
                               {**_rerank_final_entry(n, _ce, _cos), "corpus": n_corpus})
            logger.info(f"Rebuild corpus {scenario_id}: {n_corpus} liens, {n} scorés.")
        except Exception as _e:
            _rerank_job_finish(scenario_id, job_id, {"status": "error", "error": str(_e)})
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


# ── Le scoring automatique, réarmé tant qu'il reste des articles sans score ──────
# Il ne passait qu'UNE fois par processus : un statut « done » le bloquait jusqu'au
# redémarrage suivant. Les articles liés ou vectorisés après coup (living review, worker)
# restaient sans score, donc à 0 pour la porte, jusqu'à un pipeline, un redémarrage ou un
# recalcul à la main. Il repasse maintenant tant qu'il reste des articles qu'un passage
# PEUT scorer (`_retryable_unscored`), borné dans le temps : au plus un passage par
# intervalle, et l'intervalle double à chaque passage qui n'avance pas (jusqu'à six
# heures). Jamais une boucle serrée.

#: Délai minimal entre deux passages automatiques sur un même scénario (secondes).
AUTOSCORE_RETRY_S = _env_int("AUTOSCORE_RETRY_S", 600, 60)
_AUTOSCORE_MAX_RETRY_S = 6 * 3600
#: Passages d'affilée sans un article scoré avant de cesser de se réarmer tout seul. La
#: page du corpus peut encore le relancer, à l'intervalle le plus long.
_AUTOSCORE_MAX_STALLS = 8
_AUTOSCORE: dict[str, dict] = {}
_autoscore_lock = threading.Lock()


def _schedule_autoscore(scenario_id: str, delay_s: float) -> threading.Timer:
    """Un passage automatique dans `delay_s` secondes (fil démon)."""
    timer = threading.Timer(delay_s, _maybe_autorerank, args=(scenario_id,))
    timer.daemon = True
    timer.start()
    return timer


def _autoscore_note(scenario_id: str, rep: dict) -> None:
    """Après chaque passage de scoring cosinus, automatique ou non : quand le prochain
    passage automatique est permis, et s'il faut le programmer.

    Il est programmé quand le passage a fini PARTIEL (pause OpenAI, lots en échec) et
    qu'il reste des articles scorables : « N sans score, nouvel essai à la fin de la
    pause ». Un passage qui a avancé remet l'intervalle au minimum ; un passage qui n'a
    rien pu faire de ce qu'il restait double le suivant, et au-delà de
    `_AUTOSCORE_MAX_STALLS` passages stériles d'affilée il ne se reprogramme plus."""
    import time as _t
    retryable = int(rep.get("retryable") or 0)
    progressed = int(rep.get("newly_scored") or 0) > 0
    with _autoscore_lock:
        st = _AUTOSCORE.setdefault(scenario_id, {"interval": AUTOSCORE_RETRY_S, "stalls": 0})
        if progressed or retryable == 0:
            st["interval"], st["stalls"] = AUTOSCORE_RETRY_S, 0
        # Pas avant la fin de la pause OpenAI, s'il y en a une.
        delay = max(float(st["interval"]), float(rep.get("retry_after_s") or 0) + 5.0)
        st["not_before"] = _t.time() + delay
        if not progressed and retryable > 0:
            st["interval"] = min(_AUTOSCORE_MAX_RETRY_S, st["interval"] * 2)
            st["stalls"] += 1
        timer = st.get("timer")
        if (rep.get("partial") and retryable > 0 and st["stalls"] <= _AUTOSCORE_MAX_STALLS
                and not (timer is not None and timer.is_alive()) and _openai_configured()):
            st["timer"] = _schedule_autoscore(scenario_id, delay + 1.0)


def _openai_configured() -> bool:
    """Ce processus peut-il appeler OpenAI ? Sans clé, ou avec OPENAI_ENABLED=0, un nouvel
    essai programmé ne pourrait pas réussir avant un redémarrage, qui relance de toute façon
    le scoring : rien n'est programmé (la page du corpus peut encore le relancer)."""
    if not os.getenv("OPENAI_API_KEY"):
        return False
    try:
        import llm_usage
        return bool(llm_usage.openai_enabled())
    except Exception:                                        # noqa: BLE001
        return True


def _maybe_autorerank(scenario_id: str) -> bool:
    """
    Lance le scoring sémantique en arrière-plan quand des articles du scénario attendent
    un score qu'un passage peut leur donner (auto-score), et complète au passage les chunks
    title_abstract manquants (recherche + compteurs). Pas plus d'un passage par intervalle
    (cf. _autoscore_note). Renvoie True si lancé.
    """
    import threading
    import time as _t

    # `_job_is_active`, comme les autres gardes : lire `status == "running"` nu
    # traitait une entrée laissée par un fil tué au redémarrage comme vivante (donc
    # blocage définitif), et une entrée POSÉE SANS HORODATAGE comme morte (donc un
    # second scoring concurrent du même scénario).
    if _rerank_job_active(scenario_id):
        return False
    if _t.time() < (_AUTOSCORE.get(scenario_id) or {}).get("not_before", 0.0):
        return False
    try:
        if scenario_id.startswith("usr-"):
            row = _get_user_scenario_or_404(scenario_id)
            query = row["query"]
        else:
            meta = _get_db_gesica_scenario_or_404(scenario_id)
            nl = meta.get("nl_queries") or []
            query = nl[0] if nl else _gesica_title(meta)
        if not query:
            return False
        if _retryable_unscored(scenario_id) == 0:
            # Rien qu'un passage puisse scorer : pas de job, et pas de nouvelle lecture
            # avant l'intervalle (la page du corpus appelle ceci à chaque affichage).
            with _autoscore_lock:
                _AUTOSCORE.setdefault(scenario_id, {"interval": AUTOSCORE_RETRY_S, "stalls": 0})[
                    "not_before"] = _t.time() + AUTOSCORE_RETRY_S
            return False
    except Exception:
        return False

    job_id = _claim_rerank_job(scenario_id)
    if job_id is None:
        return False

    def _run():
        try:
            _rerank_job_update(scenario_id, job_id, phase="cosine")
            _backfill_title_abstract_chunks(scenario_id)  # docs sans chunk résumé -> searchable
            _cos: dict[str, Any] = {}
            n = _run_semantic_rerank_inline(scenario_id, query, report=_cos, missing_only=True)
            _rerank_job_finish(scenario_id, job_id, {
                "status": "partial" if _cosine_left_behind(_cos) else "done", "updated": n,
                **_cosine_entry(_cos)})
            logger.info(f"Auto-rerank {scenario_id}: {n} articles scorés.")
        except Exception as e:
            logger.warning(f"Auto-rerank {scenario_id}: {e}")
            _rerank_job_finish(scenario_id, job_id, {"status": "error", "error": str(e)})

    threading.Thread(target=_run, daemon=True).start()
    return True


#: Les colonnes JSON volumineuses de `scenario_settings`, dont /settings n'indique que
#: la PRÉSENCE. La liste en oubliait cinq, et notamment les trois plus grosses : une
#: lecture de deux flottants rapatriait 208 776 octets sur le scénario HPAI, dont
#: 122 863 de projection SEIR, 63 672 de carte des concepts et 38 924 de traductions.
#: Toute colonne JSON de cette table y appartient : si une nouvelle est ajoutée sans
#: être listée, elle repart dans chaque ouverture de page (cf. test_scenario_payload_sizes).
_SETTINGS_BLOB_COLUMNS = ("evidence_brief_json", "variables_json", "variables_proposal_json",
                          "variables_i18n", "clustering_json", "knowledge_graph_json",
                          "concept_graph_json", "seir_projection_json", "seir_observed_json",
                          "codebook_json", "recommended_actions_json")


def _blob_key(column: str) -> str:
    """La clé servie pour une colonne JSON. Le suffixe `_json` était retiré par une
    coupe de cinq caractères, ce qui mutilerait `variables_i18n` en `variables_`."""
    return column[:-5] if column.endswith("_json") else column


@app.get("/scenarios/{scenario_id}/settings")
def get_scenario_settings(scenario_id: str) -> dict[str, Any]:
    """Retourne les paramètres du scénario (seuil, dates de génération, variables validées).

    Les artefacts JSON en cache (brief, variables, clustering, graphe, actions) ne sont
    PAS renvoyés : chacun a son endpoint. Avec `SELECT *`, la page scénario téléchargeait
    à chaque ouverture, pour lire un seul seuil, le clustering complet et le graphe de
    connaissances (2,5 Mo pour 25 000 articles). Seule leur présence est indiquée.

    `similarity_threshold` est le seuil EN VIGUEUR (jamais un NULL ni une valeur stockée
    d'office), et `threshold_chosen` dit si une personne l'a réglé : un seuil de départ
    ne se présente pas comme un choix."""
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT * FROM scenario_settings WHERE scenario_id = :sid
        """), {"sid": scenario_id}).mappings().first()
    info = scenario_threshold_info(scenario_id)
    if not row:
        return {
            "scenario_id": scenario_id,
            "similarity_threshold": info["threshold"],
            "threshold_chosen": False,
            "similarity_threshold_set_at": None,
            "rerank_threshold": DEFAULT_RERANK_THRESHOLD,
            "double_blind_below_sample": 0,
            "brief_generated_at": None,
            "variables_validated": False,
            "variables_generated_at": None,
            "cached": {_blob_key(c): False for c in _SETTINGS_BLOB_COLUMNS},
        }
    out = {k: v for k, v in dict(row).items() if k not in _SETTINGS_BLOB_COLUMNS}
    out["similarity_threshold"] = info["threshold"]
    out["threshold_chosen"] = info["chosen"]
    # NULL en base veut dire « jamais réglé », et la porte le lit comme 0. L'interface doit
    # lire la même chose, sans quoi le curseur s'afficherait vide sur un corpus non filtré.
    if out.get("rerank_threshold") is None:
        out["rerank_threshold"] = DEFAULT_RERANK_THRESHOLD
    out["double_blind_below_sample"] = int(out.get("double_blind_below_sample") or 0)
    out["cached"] = {_blob_key(c): bool(row.get(c)) for c in _SETTINGS_BLOB_COLUMNS if c in row}
    return out


@app.patch("/scenarios/{scenario_id}/settings")
def update_scenario_settings(scenario_id: str, payload: dict[str, Any], _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Met à jour les paramètres du scénario (seuil, variables validées, etc.).

    Régler `similarity_threshold` est un CHOIX, daté (`similarity_threshold_set_at`) : le
    seuil cesse alors d'être « par défaut, non choisi » dans les rapports et les exports.
    `double_blind_below_sample` règle l'échantillon d'audit du double aveugle (0 l'éteint)."""
    from .double_blind import AUDIT_SAMPLE_MAX              # lazy: même ordre que les autres
    allowed = {"similarity_threshold", "rerank_threshold", "variables_json", "variables_validated",
               "double_blind_below_sample"}
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
    if "double_blind_below_sample" in updates:
        try:
            _n = int(updates["double_blind_below_sample"])
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail="double_blind_below_sample doit être un entier")
        if not (0 <= _n <= AUDIT_SAMPLE_MAX):
            raise HTTPException(status_code=422, detail=(
                f"double_blind_below_sample doit être compris entre 0 et {AUDIT_SAMPLE_MAX}"))
        updates["double_blind_below_sample"] = _n

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
            # Le seuil de similarité réglé ici est choisi par une personne : on le date.
            _set_at = ", similarity_threshold_set_at = NOW()" if key == "similarity_threshold" else ""
            conn.execute(text(f"""
                UPDATE scenario_settings SET {key} = :val{_set_at}, updated_at = NOW()
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
            # ajouté demain ne soit pas oublié d'un côté ; reset_corpus_derived_caches
            # l'applique, et oublie aussi le clustering gardé en mémoire.
            reset_corpus_derived_caches(conn, scenario_id)
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
            SELECT scenario_id, rerank_threshold, brief_generated_at, double_blind_below_sample,
                   variables_validated, variables_generated_at, updated_at
            FROM scenario_settings WHERE scenario_id = :sid
        """), {"sid": scenario_id}).mappings().first()
    if updated_row:
        # Le seuil EN VIGUEUR et s'il a été choisi, par la même lecture que partout : plus
        # de repli sur un littéral 0.45 qui ne se distinguait pas d'un choix.
        info = scenario_threshold_info(scenario_id)
        return {
            "status": "updated",
            "scenario_id": scenario_id,
            "updated": list(updates.keys()),
            "similarity_threshold": info["threshold"],
            "threshold_chosen": info["chosen"],
            "similarity_threshold_set_at": info["set_at"],
            "rerank_threshold": (float(updated_row["rerank_threshold"])
                                 if updated_row["rerank_threshold"] is not None
                                 else DEFAULT_RERANK_THRESHOLD),
            "double_blind_below_sample": int(updated_row["double_blind_below_sample"] or 0),
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
    puisse l'écrire au lieu de laisser deviner.

    Le lot candidat passe par l'AUTRE moitié de la porte : une courbe de rerank ne porte
    que sur les articles qui passent déjà le seuil de similarité, et réciproquement. Sans
    cela, le panneau comptait le corpus entier et promettait des coupes que la base
    n'aurait jamais faites."""
    from .variables import _param_regex                     # lazy: variables charge après
    if score not in CURVE_SCORES:
        raise HTTPException(status_code=422,
                            detail=f"score inconnu : '{score}' (attendu : {', '.join(CURVE_SCORES)})")
    rerank = score == "rerank"
    col = "ars.rerank_score" if rerank else "COALESCE(ars.similarity_score, 0)"
    # ── L'AUTRE moitié de la porte est déjà appliquée ────────────────────────
    # Le lot candidat était « ni doublon, ni exclu, ni inclus à la main », et rien de
    # plus : le seuil de l'autre score était ignoré. Le panneau annonçait donc, pour le
    # rerank du scénario HPAI, « 640 articles gardés » et « 640 articles candidats » là
    # où toutes les autres surfaces de l'application en lisent 201, et proposait
    # « mettre 0,7144 pour en garder 25 ». Sur un scénario de 6 564 articles, il
    # affirmait qu'aucun seuil de rerank ne pourrait jamais en laisser moins de 6 323.
    #
    # Une courbe de seuil doit montrer ce que CE seuil fait, à partir de ce que l'autre
    # a déjà retenu : c'est exactement ce que la porte calcule.
    other_half = (f" AND COALESCE(ars.similarity_score, 0) >= {scenario_threshold_sql(':sid')}"
                  if rerank else
                  f" AND (ars.rerank_score IS NULL"
                  f" OR ars.rerank_score >= {scenario_rerank_threshold_sql(':sid')})")
    sql_where = f"""
        FROM literature_document d
        JOIN article_scenarios ars ON ars.document_id = d.id
        WHERE ars.scenario_id = :sid
          AND d.is_duplicate IS NOT TRUE
          AND {screening_status_sql('d', 'ars')} IS DISTINCT FROM 'excluded'
          {other_half}
    """
    # Le non scoré sort de la courbe pour le rerank : il passe d'office, donc il n'a pas
    # de place sur un axe de seuils.
    only_scored = " AND ars.rerank_score IS NOT NULL" if rerank else ""
    with engine.connect() as conn:
        rows = [(float(r["s"]), bool(r["p"])) for r in conn.execute(text(f"""
            SELECT {col} AS s,
                   ((COALESCE(d.title, '') || ' ' || COALESCE(d.abstract, '')) ~* :rx) AS p
            {sql_where}
              AND {screening_status_sql('d', 'ars')} IS DISTINCT FROM 'included'
              {only_scored}
            ORDER BY s DESC
        """), {"sid": scenario_id, "rx": _param_regex(boundary=r"\y")}).mappings()]
        head = conn.execute(text(f"""
            SELECT COUNT(*) FILTER (
                     WHERE {screening_status_sql('d', 'ars')} = 'included') AS included,
                   COUNT(*) FILTER (
                     WHERE {screening_status_sql('d', 'ars')} IS DISTINCT FROM 'included'
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
    if score == "rerank":
        current, current_chosen = _get_scenario_rerank_threshold(scenario_id), None
    else:
        # Le seuil de similarité peut n'être qu'un point de départ : la courbe le dit, pour
        # qu'un 0.45 jamais choisi ne se lise pas comme la décision de la revue.
        _info = scenario_threshold_info(scenario_id)
        current, current_chosen = _info["threshold"], _info["chosen"]
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
        "current_threshold_chosen": current_chosen,
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
        "scoring_in_progress": _rerank_job_active(scenario_id),
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
