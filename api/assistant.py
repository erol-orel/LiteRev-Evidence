"""AI assistant: question answering over the corpus, streaming, scenario RAG.

Extracted from main.py (LiteRev API); `main` re-exports everything for the scripts,
tools and tests.
"""
from __future__ import annotations

import os
import re
from typing import Any

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import text

from .core import RAG_MIN_SIMILARITY, app, engine, logger
from .documents import _llm_lang_directive
from .scenario_store import DEFAULT_SIMILARITY_THRESHOLD, _get_scenario_threshold, _get_user_scenario_or_404
from .search import _build_where

class AskIn(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)  # Limite d'entrée RAG (H-5)
    project_context: str | None = None
    filters: dict[str, Any] | None = None
    lang: str | None = None

# ─────────────────────────────────────────────────────────────────────────────
# RAG Assistant /ask
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/ask")
def ask_assistant(payload: AskIn) -> dict[str, Any]:
    # 1. Rechercher les chunks pertinents dans la DB
    # On réutilise la logique de recherche textuelle mais avec un filtre projet si spécifié
    filters = payload.filters or {}
    if payload.project_context:
        filters["project_context"] = payload.project_context
    
    where_sql, where_params = _build_where(filters)
    query_terms = [t.strip() for t in re.split(r"\s+", payload.question.lower()) if t.strip()]
    
    if not query_terms:
        raise HTTPException(status_code=422, detail="Empty question")
        
    like_clauses = []
    score_clauses = []
    params = {"limit": 6, "offset": 0, **where_params}
    
    for i, term in enumerate(query_terms):
        key = f"term_{i}"
        params[key] = f"%{term}%"
        like_clauses.append(
            f"(LOWER(COALESCE(d.title, '')) LIKE :{key} OR LOWER(COALESCE(d.abstract, '')) LIKE :{key} OR LOWER(COALESCE(c.content, '')) LIKE :{key})"
        )
        score_clauses.append(
            f"((CASE WHEN LOWER(COALESCE(d.title, '')) LIKE :{key} THEN 3 ELSE 0 END) + (CASE WHEN LOWER(COALESCE(d.abstract, '')) LIKE :{key} THEN 2 ELSE 0 END) + (CASE WHEN LOWER(COALESCE(c.content, '')) LIKE :{key} THEN 1 ELSE 0 END))"
        )
        
    any_match_sql = " OR ".join(like_clauses)
    score_sql = " + ".join(score_clauses)
    
    # On utilise la recherche sémantique pgvector si la clé OpenAI est présente
    openai_key = os.getenv("OPENAI_API_KEY")
    has_vector = False
    
    if openai_key:
        try:
            from llm_usage import MeteredOpenAI as OpenAI
            client = OpenAI(api_key=openai_key, timeout=90.0)
            # Générer l'embedding de la question
            response = client.embeddings.create(
                input=[payload.question.replace("\n", " ").strip()],
                model="text-embedding-3-small"
            )
            query_embedding = response.data[0].embedding
            has_vector = True
        except Exception as e:
            logger.error(f"Erreur lors de la génération de l'embedding pour /ask: {e}")
            
    if has_vector:
        # Recherche vectorielle pure pour le RAG. On exclut les doublons et les
        # articles écartés au screening, et on impose un plancher de similarité
        # pour ne pas répondre à partir de chunks hors-sujet (corpus mince).
        params = {"query_embedding": str(query_embedding), "limit": 6,
                  "max_dist": 1.0 - RAG_MIN_SIMILARITY, **where_params}
        sql = text(f"""
            SELECT
                d.id AS document_id,
                d.title,
                d.year,
                d.url,
                d.source,
                d.project_context,
                c.content,
                c.metadata_json,
                (1 - (c.embedding <=> CAST(:query_embedding AS vector))) AS score
            FROM document_chunk c
            JOIN literature_document d ON d.id = c.document_id
            WHERE c.embedding IS NOT NULL
              AND (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)
              AND d.screening_status IS DISTINCT FROM 'excluded'
              AND (c.embedding <=> CAST(:query_embedding AS vector)) <= :max_dist
            {where_sql}
            ORDER BY c.embedding <=> CAST(:query_embedding AS vector)
            LIMIT :limit
        """)
    else:
        # Fallback textuel classique
        params = {"limit": 6, "offset": 0, **where_params}
        for i, term in enumerate(query_terms):
            key = f"term_{i}"
            params[key] = f"%{term}%"
        sql = text(f"""
            SELECT 
                d.id AS document_id,
                d.title,
                d.year,
                d.url,
                d.source,
                d.project_context,
                c.content,
                c.metadata_json,
                ({score_sql}) AS score
            FROM document_chunk c
            JOIN literature_document d ON d.id = c.document_id
            WHERE ({any_match_sql})
              AND (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)
              AND d.screening_status IS DISTINCT FROM 'excluded'
            {where_sql}
            ORDER BY score DESC, d.year DESC NULLS LAST
            LIMIT :limit
        """)
    
    with engine.connect() as conn:
        rows = conn.execute(sql, params).mappings().all()
        
    if not rows:
        return {
            "answer": "Je n'ai pas trouvé d'articles ou d'évidences scientifiques dans le corpus actuel pour répondre à votre question. Veuillez élargir vos termes de recherche ou ingérer de nouveaux articles.",
            "sources": []
        }
        
    # 2. Construire le contexte pour l'API OpenAI
    context_blocks = []
    sources = []
    seen_docs = set()
    
    for i, r in enumerate(rows):
        doc_id = r["document_id"]
        # Récupérer la force des preuves si présente dans metadata_json
        meta = r["metadata_json"] or {}
        evidence_strength = meta.get("evidence_strength", "non spécifiée")
        
        context_blocks.append(
            f"--- SOURCE {i+1} ---\n"
            f"Titre: {r['title']}\n"
            f"Année: {r['year'] or 'Inconnue'}\n"
            f"Source: {r['source']}\n"
            f"Projet: {r['project_context']}\n"
            f"Force des preuves: {evidence_strength}\n"
            f"Contenu: {r['content']}\n"
        )
        
        if doc_id not in seen_docs:
            seen_docs.add(doc_id)
            sources.append({
                "document_id": doc_id,
                "title": r["title"],
                "year": r["year"],
                "url": r["url"],
                "source": r["source"],
                "project_context": r["project_context"],
                "evidence_strength": evidence_strength
            })
            
    context_str = "\n\n".join(context_blocks)
    
    # 3. Appeler l'API OpenAI (GPT-4o-mini)
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        # Fallback si pas de clé API configurée
        lines = []
        for s in sources:
            url_str = s['url'] if s['url'] else "Pas d'URL"
            year_str = str(s['year']) if s['year'] else "N/A"
            lines.append(f"- **{s['title']}** ({year_str}) - {url_str}")
        return {
            "answer": "[Mode dégradé - Clé OpenAI manquante]\n\nVoici les sources trouvées pour répondre à votre question :\n\n" + "\n".join(lines),
            "sources": sources
        }
        
    try:
        from llm_usage import MeteredOpenAI as OpenAI
        client = OpenAI(api_key=openai_key, timeout=90.0)
        
        system_prompt = (
            "Vous êtes l'assistant scientifique expert de LiteRev-Evidence, spécialisé dans la synthèse d'évidences "
            "scientifiques (recherche clinique et santé publique).\n\n"
            "Votre tâche est de répondre à la question de l'utilisateur en vous basant STRICTEMENT sur le contexte fourni. "
            "Ne faites pas d'affirmations qui ne sont pas étayées par les sources fournies.\n\n"
            "Règles de rédaction :\n"
            "1. Soyez précis, structuré et professionnel.\n"
            "2. Citez toujours vos sources dans le texte en utilisant le format [SOURCE 1], [SOURCE 2] etc. correspondant aux blocs du contexte.\n"
            "3. Mentionnez la force des preuves (forte, modérée, faible) quand elle est pertinente pour appuyer vos conclusions.\n"
            "4. Si le contexte ne contient pas assez d'informations pour répondre, dites-le honnêtement."
        ) + _llm_lang_directive(payload.lang)

        user_prompt = (
            f"CONTEXTE :\n{context_str}\n\n"
            f"QUESTION : {payload.question}"
        )
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=1000
        )
        
        answer = response.choices[0].message.content
        return {
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        logger.error(f"Erreur OpenAI API: {e}")
        return {
            "answer": f"Une erreur est survenue lors de la génération de la réponse via l'IA : {str(e)}\n\nNéanmoins, voici les sources scientifiques trouvées dans la base :",
            "sources": sources
        }

@app.post("/ask/stream")
async def ask_stream(payload: dict[str, Any]) -> StreamingResponse:
    """
    Version streaming (SSE) de l'endpoint /ask.
    Retourne les tokens au fur et à mesure via Server-Sent Events.
    """
    from llm_usage import MeteredAsyncOpenAI as AsyncOpenAI

    question = payload.get("question", "")
    project_context = payload.get("project_context", "literev")
    scenario_id = payload.get("scenario_id", None)
    # Public endpoint : borne top_k pour éviter un débordement de contexte (LIMIT
    # géant → prompt hors-limite) ou un LIMIT négatif (crash Postgres). Plafond
    # généreux (qualité préservée), plancher à 1.
    try:
        top_k = max(1, min(int(payload.get("top_k", 8)), 40))
    except (TypeError, ValueError):
        top_k = 8
    lang = payload.get("lang")

    if not question:
        raise HTTPException(status_code=422, detail="question est requis")

    # Récupérer le contexte RAG (chunks pertinents)
    try:
        from llm_usage import MeteredOpenAI as SyncOpenAI
        sync_client = SyncOpenAI(timeout=90.0)
        emb_resp = sync_client.embeddings.create(
            model="text-embedding-3-small",
            input=question[:2000],
        )
        q_emb = emb_resp.data[0].embedding
        emb_str = "[" + ",".join(str(x) for x in q_emb) + "]"
    except Exception as e:
        logger.error(f"Embedding error in /ask/stream: {e}")
        emb_str = None

    context_chunks = []
    sources = []
    if emb_str:
        where_extra = ""
        join_extra = ""
        # Migration 2 : quand un scénario est fourni, on JOINT article_scenarios
        # pour lire le screening PROPRE au scénario (COALESCE), comme
        # /ask/stream/filtered. Sans scénario, colonne globale.
        screen_expr = "d.screening_status"
        params_extra: dict[str, Any] = {
            "top_k": top_k, "emb": emb_str, "max_dist": 1.0 - RAG_MIN_SIMILARITY,
        }
        if project_context:
            where_extra += " AND d.project_context = :project_context"
            params_extra["project_context"] = project_context
        if scenario_id:
            join_extra = " JOIN article_scenarios ars ON ars.document_id = d.id AND ars.scenario_id = :scenario_id "
            screen_expr = "COALESCE(ars.screening_status, d.screening_status)"
            params_extra["scenario_id"] = scenario_id

        with engine.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT c.content, d.title, d.year, d.doi, d.authors, d.id AS doc_id,
                       1 - (c.embedding <=> CAST(:emb AS vector)) AS similarity
                FROM document_chunk c
                JOIN literature_document d ON d.id = c.document_id
                {join_extra}
                WHERE c.embedding IS NOT NULL
                  AND (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)
                  AND {screen_expr} IS DISTINCT FROM 'excluded'
                  AND (c.embedding <=> CAST(:emb AS vector)) <= :max_dist
                  {where_extra}
                ORDER BY c.embedding <=> CAST(:emb AS vector)
                LIMIT :top_k
            """), params_extra).mappings().all()

        for i, r in enumerate(rows):
            # Inclure titre + année DANS le contexte : le prompt demande de citer
            # les articles par leur titre, donc le modèle doit les voir.
            context_chunks.append(
                f"[{i + 1}] {r['title'] or 'Sans titre'} ({r['year'] or 'année inconnue'})\n{r['content']}"
            )
            # Champs canoniques (document_id/score/authors) attendus par le front.
            sources.append({
                "document_id": r["doc_id"],
                "title": r["title"],
                "year": r["year"],
                "doi": r["doi"],
                "authors": r["authors"],
                "score": round(float(r["similarity"]), 3),
            })

    context_text = "\n\n---\n\n".join(context_chunks[:top_k]) if context_chunks else "Aucun contexte disponible."

    system_prompt = """Tu es un assistant expert en sciences de la santé et en revue systématique de la littérature scientifique.
Tu réponds de manière précise, factuelle et synthétique.
Base-toi exclusivement sur le contexte fourni. Si l'information n'est pas dans le contexte, dis-le clairement.
Cite les articles pertinents par leur titre quand tu les mentionnes.""" + _llm_lang_directive(lang)

    user_prompt = f"""Contexte scientifique (extraits d'articles) :
{context_text}

Question : {question}

Réponds de manière structurée et cite les sources pertinentes du contexte."""

    async def event_generator():
        # D'abord envoyer les sources
        import json as _json
        sources_event = f"event: sources\ndata: {_json.dumps(sources)}\n\n"
        yield sources_event

        # Échec d'embedding (panne/quota OpenAI) → NE PAS prétendre que le corpus est
        # vide (réponse trompeuse). On signale une erreur transitoire honnête.
        if emb_str is None:
            err = ("Le service de recherche est momentanément indisponible "
                   "(erreur d'embedding). Merci de réessayer dans un instant.")
            yield f"data: {_json.dumps({'token': err})}\n\n"
            yield "event: error\ndata: {\"error\": \"embedding_unavailable\"}\n\n"
            return
        # Pas de contexte récupéré → ne PAS interroger le LLM (réponse non étayée).
        if not context_chunks:
            msg = ("Aucun passage pertinent n'a été trouvé dans le corpus pour cette "
                   "question. Reformulez la question ou élargissez le corpus.")
            yield f"data: {_json.dumps({'token': msg})}\n\n"
            yield "event: done\ndata: {}\n\n"
            return

        # Puis streamer la réponse LLM
        try:
            async_client = AsyncOpenAI(timeout=90.0)
            stream = await async_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=True,
                temperature=0.2,
                max_tokens=1200,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    token_event = f"data: {_json.dumps({'token': delta.content})}\n\n"
                    yield token_event
        except Exception as e:
            yield f"event: error\ndata: {_json.dumps({'error': str(e)})}\n\n"

        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/user-scenarios/{scenario_id}/rag")
def user_scenario_rag_assistant(scenario_id: str, payload: AskIn) -> dict[str, Any]:
    """Assistant RAG pour un scénario utilisateur (délègue au RAG générique filtré)."""
    row = _get_user_scenario_or_404(scenario_id)
    # L'Assistant n'interroge que le SOUS-ENSEMBLE PERTINENT (≥ seuil sémantique
    # ou inclus manuellement), pas tout le corpus du scénario.
    eff_thr = _get_scenario_threshold(scenario_id)
    payload.filters = payload.filters or {}
    payload.filters["project_context"] = "literev"

    openai_key = os.getenv("OPENAI_API_KEY")
    query_embedding = None
    if openai_key:
        try:
            from llm_usage import MeteredOpenAI as OpenAI
            client = OpenAI(api_key=openai_key, timeout=90.0)
            response = client.embeddings.create(
                input=[payload.question.replace("\n", " ").strip()],
                model="text-embedding-3-small"
            )
            query_embedding = response.data[0].embedding
        except Exception as e:
            logger.error(f"Erreur embedding RAG user_scenario {scenario_id}: {e}")

    with engine.connect() as conn:
        if query_embedding:
            rows = conn.execute(text("""
                SELECT d.id AS document_id, d.title, d.year, d.url, d.source,
                       d.authors, d.journal, d.doi,
                       c.content, c.metadata_json,
                       (1 - (c.embedding <=> CAST(:emb AS vector))) AS score
                FROM document_chunk c
                JOIN literature_document d ON d.id = c.document_id
                JOIN article_scenarios ars ON ars.document_id = d.id AND ars.scenario_id = :sid
                WHERE c.embedding IS NOT NULL
                  AND (COALESCE(ars.similarity_score, 0) >= :thr OR COALESCE(ars.screening_status, d.screening_status) = 'included')
                  AND (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)
                  AND COALESCE(ars.screening_status, d.screening_status) IS DISTINCT FROM 'excluded'  -- porte de screening (C1, per-scenario)
                ORDER BY c.embedding <=> CAST(:emb AS vector)
                LIMIT 8
            """), {"emb": str(query_embedding), "sid": scenario_id, "thr": eff_thr}).mappings().all()
        else:
            terms = [t.strip() for t in re.split(r"\s+", payload.question.lower()) if t.strip()]
            if not terms:
                return {"answer": "Question vide.", "sources": []}
            like_clauses = " OR ".join(
                f"(LOWER(COALESCE(d.title,'')) LIKE :t{i} OR LOWER(COALESCE(c.content,'')) LIKE :t{i})"
                for i in range(len(terms))
            )
            params: dict[str, Any] = {"sid": scenario_id, "thr": eff_thr}
            for i, t in enumerate(terms):
                params[f"t{i}"] = f"%{t}%"
            rows = conn.execute(text(f"""
                SELECT d.id AS document_id, d.title, d.year, d.url, d.source,
                       d.authors, d.journal, d.doi,
                       c.content, c.metadata_json, 1.0 AS score
                FROM document_chunk c
                JOIN literature_document d ON d.id = c.document_id
                JOIN article_scenarios ars ON ars.document_id = d.id AND ars.scenario_id = :sid
                WHERE ({like_clauses})
                  AND (COALESCE(ars.similarity_score, 0) >= :thr OR COALESCE(ars.screening_status, d.screening_status) = 'included')
                  AND COALESCE(ars.screening_status, d.screening_status) IS DISTINCT FROM 'excluded'  -- porte de screening (C1, per-scenario)
                ORDER BY d.year DESC NULLS LAST
                LIMIT 8
            """), params).mappings().all()

    if not rows:
        return {
            "answer": f"Aucun article trouvé dans le corpus du scénario '{row['name']}' pour cette question.",
            "sources": [], "scenario_id": scenario_id,
        }

    context_blocks = []
    sources = []
    seen: set = set()
    for i, r in enumerate(rows):
        doc_id = r["document_id"]
        context_blocks.append(
            f"--- SOURCE {i+1} ---\nTitre: {r['title']}\n"
            f"Auteurs: {r.get('authors','') or 'N/A'}\n"
            f"Journal: {r.get('journal','') or 'N/A'} ({r['year'] or 'N/A'})\n"
            f"DOI: {r.get('doi','') or 'N/A'}\nContenu: {r['content']}\n"
        )
        if doc_id not in seen:
            seen.add(doc_id)
            sources.append({
                "document_id": doc_id, "title": r["title"], "year": r["year"],
                "url": r["url"], "source": r["source"], "authors": r.get("authors"),
                "journal": r.get("journal"), "doi": r.get("doi"),
                "score": float(r.get("score", 0)),
            })

    context_str = "\n\n".join(context_blocks)
    if not openai_key:
        return {
            "answer": "[Mode dégradé]\n\nSources :\n" + "\n".join(f"- {s['title']} ({s['year']})" for s in sources),
            "sources": sources, "scenario_id": scenario_id,
        }

    try:
        from llm_usage import MeteredOpenAI as OpenAI
        client = OpenAI(api_key=openai_key, timeout=90.0)
        system_prompt = (
            f"Vous êtes l'assistant scientifique expert de LiteRev-Evidence pour la recherche : "
            f"**{row['name']}**.\n\n"
            "Règles de rédaction :\n"
            "1. Basez-vous STRICTEMENT sur les sources fournies dans le contexte.\n"
            "2. Citez vos sources avec [SOURCE 1], [SOURCE 2], etc.\n"
            "3. Mentionnez les niveaux de preuve (RCT, méta-analyse, étude observationnelle).\n"
            "4. Soyez précis et structuré. Si le contexte est insuffisant, dites-le.\n"
        ) + _llm_lang_directive(payload.lang)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"CONTEXTE :\n{context_str}\n\nQUESTION : {payload.question}"},
            ],
            temperature=0.2,
            max_tokens=1200,
        )
        return {
            "answer": response.choices[0].message.content,
            "sources": sources, "scenario_id": scenario_id, "model": "Assistant IA",
        }
    except Exception as e:
        logger.error(f"Erreur OpenAI RAG user_scenario {scenario_id}: {e}")
        return {"answer": f"Erreur : {str(e)}", "sources": sources, "scenario_id": scenario_id}


# ─── ASSISTANT IA FILTRÉ PAR SEUIL ───────────────────────────────────────────

@app.post("/ask/stream/filtered")
async def ask_stream_filtered(payload: dict[str, Any]):
    """
    Version de /ask/stream qui filtre les chunks par seuil de similarité
    et priorise les articles validés humainement.
    """
    from llm_usage import MeteredAsyncOpenAI as AsyncOpenAI, MeteredOpenAI as SyncOpenAI

    question = payload.get("question", "")
    scenario_id = payload.get("scenario_id", None)
    # Public endpoint : borne top_k (débordement de contexte / LIMIT négatif). Plafond généreux.
    try:
        top_k = max(1, min(int(payload.get("top_k", 12)), 40))
    except (TypeError, ValueError):
        top_k = 12
    project_context = payload.get("project_context", "literev")
    lang = payload.get("lang")

    if not question:
        raise HTTPException(status_code=422, detail="question est requis")

    threshold = _get_scenario_threshold(scenario_id) if scenario_id else DEFAULT_SIMILARITY_THRESHOLD

    # Compteurs « articles utilisés / avec texte intégral » à afficher sous la
    # réponse IA - MÊMES définitions que la vue Preuves (sous-ensemble PERTINENT du
    # scénario : ≥ seuil OU inclus). Volontairement PAS dérivés de `sources`, qui
    # sont des chunks plafonnés à top_k et ne refléteraient pas le nb d'articles.
    papers_used = 0
    papers_with_fulltext = 0
    if scenario_id:
        try:
            with engine.connect() as _cc:
                _cnt = _cc.execute(text("""
                    SELECT
                        COUNT(*) FILTER (WHERE d.is_duplicate IS NOT TRUE
                            AND COALESCE(ars.screening_status, d.screening_status) IS DISTINCT FROM 'excluded'
                            AND (COALESCE(ars.screening_status, d.screening_status) = 'included' OR COALESCE(ars.similarity_score, 0) >= :thr)) AS relevant,
                        COUNT(*) FILTER (WHERE d.is_duplicate IS NOT TRUE
                            AND COALESCE(ars.screening_status, d.screening_status) IS DISTINCT FROM 'excluded'
                            AND (COALESCE(ars.screening_status, d.screening_status) = 'included' OR COALESCE(ars.similarity_score, 0) >= :thr)
                            AND EXISTS (SELECT 1 FROM document_chunk c
                                WHERE c.document_id = d.id AND c.chunk_type = 'fulltext_section')) AS relevant_with_fulltext
                    FROM article_scenarios ars
                    JOIN literature_document d ON d.id = ars.document_id
                    WHERE ars.scenario_id = :sid
                """), {"sid": scenario_id, "thr": threshold}).mappings().fetchone()
            if _cnt:
                papers_used = int(_cnt["relevant"] or 0)
                papers_with_fulltext = int(_cnt["relevant_with_fulltext"] or 0)
        except Exception as _e_cnt:
            logger.warning(f"ask_stream_filtered counts {scenario_id}: {_e_cnt}")

    # Embedding de la question
    try:
        sync_client = SyncOpenAI(timeout=90.0)
        emb_resp = sync_client.embeddings.create(
            model="text-embedding-3-small",
            input=question[:2000],
        )
        q_emb = emb_resp.data[0].embedding
        emb_str = "[" + ",".join(str(x) for x in q_emb) + "]"
    except Exception as e:
        logger.error(f"Embedding error in /ask/stream/filtered: {e}")
        emb_str = None

    context_chunks = []
    sources = []

    if emb_str:
        # Construire le filtre scénario avec seuil
        where_extra = ""
        join_extra = ""
        # Migration 2 (screening par scénario) : quand un scénario est fourni, on
        # JOINT article_scenarios pour lire le statut PROPRE au scénario via
        # COALESCE (repli sur la colonne globale tant que la Phase 5 n'a pas
        # supprimé le fallback). Sans scénario, on reste sur la colonne globale.
        screen_expr = "d.screening_status"
        params_extra: dict[str, Any] = {"top_k": top_k, "emb": emb_str, "threshold": threshold}

        if project_context:
            where_extra += " AND d.project_context = :project_context"
            params_extra["project_context"] = project_context

        if scenario_id:
            # Filtrer par scénario ET par seuil de similarité (ou validé humainement).
            # JOIN (au lieu d'EXISTS) : (scenario_id, document_id) est unique dans
            # article_scenarios, donc la cardinalité est préservée et asn devient
            # lisible dans le SELECT/ORDER pour le screening par scénario.
            join_extra = " JOIN article_scenarios asn ON asn.document_id = d.id AND asn.scenario_id = :scenario_id "
            screen_expr = "COALESCE(asn.screening_status, d.screening_status)"
            where_extra += f"""
                AND (
                    asn.similarity_score >= :threshold
                    OR asn.similarity_score IS NULL
                    OR {screen_expr} = 'included'
                )
            """
            params_extra["scenario_id"] = scenario_id

        with engine.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT c.content, d.title, d.year, d.doi, d.authors, d.id AS doc_id,
                       {screen_expr} AS screening_status,
                       1 - (c.embedding <=> CAST(:emb AS vector)) AS similarity
                FROM document_chunk c
                JOIN literature_document d ON d.id = c.document_id
                {join_extra}
                WHERE c.embedding IS NOT NULL {where_extra}
                ORDER BY
                    CASE WHEN {screen_expr} = 'included' THEN 0 ELSE 1 END,
                    c.embedding <=> CAST(:emb AS vector)
                LIMIT :top_k
            """), params_extra).mappings().all()

        for i, r in enumerate(rows):
            # Inclure titre + année DANS le contexte (le prompt demande de citer
            # par titre, donc le modèle doit disposer des titres).
            context_chunks.append(
                f"[{i + 1}] {r['title'] or 'Sans titre'} ({r['year'] or 'année inconnue'})\n{r['content']}"
            )
            # Champs canoniques attendus par le front (ScenarioRagSource) : document_id
            # + score (et authors). Avant, on envoyait id/similarity et pas d'auteurs,
            # d'où « Pertinence: NaN% » et « undefined • » dans le panneau Sources.
            sources.append({
                "document_id": r["doc_id"],
                "title": r["title"],
                "year": r["year"],
                "doi": r["doi"],
                "authors": r["authors"],
                "score": round(float(r["similarity"]), 3),
                "validated": r["screening_status"] == "included",
            })

    context_text = "\n\n---\n\n".join(context_chunks[:top_k]) if context_chunks else "Aucun contexte disponible."

    system_prompt = """Tu es un assistant expert en sciences de la santé et en revue systématique de la littérature scientifique.
Tu réponds de manière précise, factuelle et structurée.
Base-toi exclusivement sur le contexte fourni. Si l'information n'est pas dans le contexte, dis-le clairement.
Cite les articles pertinents par leur titre quand tu les mentionnes.
Ne pas utiliser de tiret cadratin (em dash).""" + _llm_lang_directive(lang)

    user_prompt = f"""Contexte scientifique (extraits d'articles sélectionnés par pertinence sémantique) :
{context_text}

Question : {question}

Réponds de manière structurée et cite les sources pertinentes du contexte."""

    async def event_generator():
        import json as _json2
        # Méta d'abord : combien d'articles pertinents alimentent la réponse et
        # combien ont un texte intégral (affiché sous la réponse, comme la vue Preuves).
        meta_event = ("event: meta\ndata: "
                      + _json2.dumps({"papers_used": papers_used,
                                      "papers_with_fulltext": papers_with_fulltext,
                                      "threshold": round(float(threshold), 2)})
                      + "\n\n")
        yield meta_event
        sources_event = f"event: sources\ndata: {_json2.dumps(sources)}\n\n"
        yield sources_event

        # Échec d'embedding (panne/quota OpenAI) → erreur honnête, ne PAS prétendre
        # que rien n'est pertinent.
        if emb_str is None:
            err = ("Le service de recherche est momentanément indisponible "
                   "(erreur d'embedding). Merci de réessayer dans un instant.")
            yield f"data: {_json2.dumps({'token': err})}\n\n"
            yield "event: error\ndata: {\"error\": \"embedding_unavailable\"}\n\n"
            return
        # Pas de contexte pertinent → ne pas générer de réponse non étayée.
        if not context_chunks:
            msg = ("Aucun passage pertinent (au-dessus du seuil) n'a été trouvé pour "
                   "cette question dans ce scénario. Reformulez ou abaissez le seuil.")
            yield f"data: {_json2.dumps({'token': msg})}\n\n"
            yield "event: done\ndata: {}\n\n"
            return

        try:
            async_client = AsyncOpenAI(timeout=90.0)
            stream = await async_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=True,
                temperature=0.2,
                max_tokens=1500,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    token_event = f"data: {_json2.dumps({'token': delta.content})}\n\n"
                    yield token_event
        except Exception as e:
            yield f"event: error\ndata: {_json2.dumps({'error': str(e)})}\n\n"

        yield "event: done\ndata: {}\n\n"

    from fastapi.responses import StreamingResponse as _SR
    return _SR(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
