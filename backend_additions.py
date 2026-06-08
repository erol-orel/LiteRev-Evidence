"""
Bloc à ajouter à la fin de main.py — 6 fonctionnalités :
1. Scoring sémantique post-ingestion (rerank)
2. Evidence Brief LLM automatique (tous scénarios)
3. Variables & Modèle auto-rempli depuis PICO + notification
4. Heatmap avec vrais noms
5. Assistant IA filtré par seuil + articles validés
6. Endpoint seuil configurable par scénario
"""

# ─── SCORING SÉMANTIQUE POST-INGESTION ───────────────────────────────────────

_RERANK_JOBS: dict[str, dict] = {}

DEFAULT_SIMILARITY_THRESHOLD = 0.45


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


def _get_scenario_threshold(scenario_id: str) -> float:
    """Retourne le seuil de similarité configuré pour ce scénario."""
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT similarity_threshold FROM scenario_settings WHERE scenario_id = :sid
        """), {"sid": scenario_id}).mappings().first()
    return float(row["similarity_threshold"]) if row and row["similarity_threshold"] is not None else DEFAULT_SIMILARITY_THRESHOLD


def _get_scenario_name(scenario_id: str) -> str:
    """Retourne le nom lisible d'un scénario (user ou GESICA)."""
    if scenario_id.startswith("usr-"):
        with engine.connect() as conn:
            row = conn.execute(text(
                "SELECT name FROM user_scenarios WHERE id = :id"
            ), {"id": scenario_id}).mappings().first()
        return row["name"] if row else scenario_id
    # GESICA : utiliser les métadonnées
    meta = GESICA_SCENARIO_METADATA.get(scenario_id, {})
    return meta.get("title", scenario_id)


def _get_above_threshold_articles(scenario_id: str, threshold: float | None = None) -> list[dict]:
    """
    Retourne les articles au-dessus du seuil de similarité OU validés humainement.
    Priorité : included > similarity_score >= threshold > autres.
    """
    if threshold is None:
        threshold = _get_scenario_threshold(scenario_id)
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT ld.id, ld.title, ld.abstract, ld.year, ld.journal, ld.authors, ld.doi,
                   ld.study_design, ld.pico_json, ld.citation_count, ld.screening_status,
                   ld.quality_score, asn.similarity_score
            FROM literature_document ld
            JOIN article_scenarios asn ON asn.document_id = ld.id AND asn.scenario_id = :sid
            WHERE ld.project_context = 'literev'
              AND ld.is_duplicate IS NOT TRUE
              AND (
                  ld.screening_status = 'included'
                  OR asn.similarity_score >= :threshold
                  OR asn.similarity_score IS NULL
              )
            ORDER BY
                CASE WHEN ld.screening_status = 'included' THEN 0 ELSE 1 END,
                asn.similarity_score DESC NULLS LAST,
                ld.citation_count DESC NULLS LAST
        """), {"sid": scenario_id, "threshold": threshold}).mappings().fetchall()
    return [dict(r) for r in rows]


def _run_semantic_rerank(scenario_id: str, query: str) -> int:
    """
    Calcule le score de similarité sémantique entre la requête et chaque abstract,
    puis met à jour article_scenarios.similarity_score.
    Retourne le nombre d'articles rerankés.
    """
    import time as _time
    try:
        from openai import OpenAI as _OAI
        client = _OAI()

        # Embedding de la requête
        q_emb_resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=query[:2000],
        )
        q_emb = q_emb_resp.data[0].embedding

        # Récupérer tous les articles du scénario sans score ou avec score = 1.0 (défaut)
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT ld.id, ld.title, ld.abstract
                FROM literature_document ld
                JOIN article_scenarios asn ON asn.document_id = ld.id AND asn.scenario_id = :sid
                WHERE ld.project_context = 'literev'
                  AND ld.abstract IS NOT NULL
                  AND length(ld.abstract) > 30
                ORDER BY ld.id
            """), {"sid": scenario_id}).mappings().fetchall()

        updated = 0
        BATCH = 100  # OpenAI embeddings batch

        for i in range(0, len(rows), BATCH):
            batch = rows[i:i+BATCH]
            texts = [f"{r['title']}\n\n{(r['abstract'] or '')[:1500]}" for r in batch]
            try:
                emb_resp = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=texts,
                )
                for j, emb_data in enumerate(emb_resp.data):
                    doc_emb = emb_data.embedding
                    # Cosine similarity
                    dot = sum(a * b for a, b in zip(q_emb, doc_emb))
                    norm_q = sum(a * a for a in q_emb) ** 0.5
                    norm_d = sum(b * b for b in doc_emb) ** 0.5
                    sim = dot / (norm_q * norm_d) if norm_q and norm_d else 0.0
                    sim = max(0.0, min(1.0, sim))

                    with engine.begin() as conn:
                        conn.execute(text("""
                            UPDATE article_scenarios
                            SET similarity_score = :score
                            WHERE document_id = :doc_id AND scenario_id = :sid
                        """), {"score": sim, "doc_id": batch[j]["id"], "sid": scenario_id})
                    updated += 1
            except Exception as e:
                logger.warning(f"Rerank batch {i}: {e}")
            _time.sleep(0.2)

        logger.info(f"Rerank {scenario_id}: {updated} articles rerankés.")
        return updated

    except Exception as e:
        logger.error(f"Rerank {scenario_id} fatal: {e}", exc_info=True)
        return 0


@app.post("/scenarios/{scenario_id}/rerank")
def trigger_rerank(scenario_id: str) -> dict[str, Any]:
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
        meta = GESICA_SCENARIO_METADATA.get(scenario_id)
        if not meta:
            raise HTTPException(status_code=404, detail="Scénario non trouvé")
        query = meta.get("nl_queries", [meta.get("title", scenario_id)])[0] if meta.get("nl_queries") else meta.get("title", scenario_id)

    if _RERANK_JOBS.get(scenario_id, {}).get("status") == "running":
        return {"status": "already_running", "scenario_id": scenario_id}

    _RERANK_JOBS[scenario_id] = {"status": "running", "updated": 0}

    def _run():
        n = _run_semantic_rerank(scenario_id, query)
        _RERANK_JOBS[scenario_id] = {"status": "done", "updated": n}

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "started", "scenario_id": scenario_id, "query": query}


@app.get("/scenarios/{scenario_id}/rerank/status")
def get_rerank_status(scenario_id: str) -> dict[str, Any]:
    """Statut du job de reranking sémantique."""
    return _RERANK_JOBS.get(scenario_id, {"status": "idle"})


@app.get("/scenarios/{scenario_id}/settings")
def get_scenario_settings(scenario_id: str) -> dict[str, Any]:
    """Retourne les paramètres du scénario (seuil, état du brief LLM, variables)."""
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT * FROM scenario_settings WHERE scenario_id = :sid
        """), {"sid": scenario_id}).mappings().first()
    if not row:
        return {
            "scenario_id": scenario_id,
            "similarity_threshold": DEFAULT_SIMILARITY_THRESHOLD,
            "evidence_brief_json": None,
            "brief_generated_at": None,
            "variables_json": None,
            "variables_validated": False,
            "variables_generated_at": None,
        }
    return dict(row)


@app.patch("/scenarios/{scenario_id}/settings")
def update_scenario_settings(scenario_id: str, payload: dict[str, Any]) -> dict[str, Any]:
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

    return {"status": "updated", "scenario_id": scenario_id, "updated": list(updates.keys())}


# ─── EVIDENCE BRIEF LLM AUTOMATIQUE ──────────────────────────────────────────

_BRIEF_GENERATION_JOBS: dict[str, dict] = {}


def _generate_evidence_brief_llm(scenario_id: str, force: bool = False) -> dict[str, Any]:
    """
    Génère un Evidence Brief narratif complet via LLM à partir des articles
    au-dessus du seuil de similarité (ou validés humainement).
    Sauvegarde le résultat dans scenario_settings.evidence_brief_json.
    """
    import json as _json
    from datetime import datetime, timezone
    from openai import OpenAI as _OAI

    threshold = _get_scenario_threshold(scenario_id)
    articles = _get_above_threshold_articles(scenario_id, threshold)

    if not articles:
        return {"error": "Aucun article au-dessus du seuil pour générer le brief."}

    # Vérifier si un brief récent existe déjà (< 24h) et force=False
    if not force:
        with engine.connect() as conn:
            row = conn.execute(text("""
                SELECT evidence_brief_json, brief_generated_at
                FROM scenario_settings WHERE scenario_id = :sid
            """), {"sid": scenario_id}).mappings().first()
        if row and row["evidence_brief_json"] and row["brief_generated_at"]:
            age = (datetime.now(timezone.utc) - row["brief_generated_at"].replace(tzinfo=timezone.utc)).total_seconds()
            if age < 86400:  # 24h
                return dict(row["evidence_brief_json"])

    scenario_name = _get_scenario_name(scenario_id)

    # Préparer le contexte : top 30 articles avec PICO
    context_articles = []
    for a in articles[:30]:
        pj = a.get("pico_json") or {}
        context_articles.append({
            "title": a.get("title", ""),
            "year": a.get("year"),
            "journal": a.get("journal", ""),
            "citation_count": a.get("citation_count"),
            "study_design": a.get("study_design") or pj.get("study_design", ""),
            "screening_status": a.get("screening_status"),
            "P": pj.get("population", pj.get("P", "")),
            "I": pj.get("intervention", pj.get("I", "")),
            "C": pj.get("comparator", pj.get("C", "")),
            "O": pj.get("outcome", pj.get("O", "")),
            "key_finding": pj.get("key_finding", pj.get("conclusion", "")),
        })

    context_str = _json.dumps(context_articles, ensure_ascii=False, indent=2)

    # Stats corpus
    total = len(articles)
    included = sum(1 for a in articles if a.get("screening_status") == "included")
    with_pico = sum(1 for a in articles if a.get("pico_json"))
    years = [a["year"] for a in articles if a.get("year")]
    year_range = f"{min(years)}-{max(years)}" if years else "N/A"

    study_designs = {}
    for a in articles:
        pj = a.get("pico_json") or {}
        d = a.get("study_design") or pj.get("study_design", "Non classifié")
        study_designs[d] = study_designs.get(d, 0) + 1

    top_designs = sorted(study_designs.items(), key=lambda x: -x[1])[:5]

    system_prompt = """Tu es un expert en médecine d'urgence et en revue systématique de la littérature scientifique.
Tu génères des Evidence Briefs complets, rigoureux et structurés en français.
Tu dois produire un JSON structuré avec tous les champs demandés.
Sois précis, factuel, et base-toi exclusivement sur les articles fournis.
Ne pas utiliser de tiret em (—). Utiliser des tirets simples (-) si nécessaire."""

    user_prompt = f"""Génère un Evidence Brief complet pour le scénario de recherche : "{scenario_name}"

Corpus : {total} articles ({year_range}), {with_pico} avec PICO extrait, {included} validés humainement.
Designs d'étude principaux : {', '.join(f'{d} ({n})' for d, n in top_designs)}.

Articles (top 30 par pertinence) :
{context_str}

Génère un JSON avec EXACTEMENT ces champs :
{{
  "executive_summary": "Résumé exécutif en 3-4 phrases synthétisant les principales conclusions",
  "clinical_context": "Contexte clinique et importance du sujet (2-3 paragraphes)",
  "key_findings": ["Finding 1", "Finding 2", "Finding 3", "Finding 4", "Finding 5"],
  "recommended_actions": ["Action 1", "Action 2", "Action 3", "Action 4"],
  "evidence_synthesis": "Synthèse narrative détaillée des évidences (4-6 paragraphes)",
  "population_summary": "Résumé des populations étudiées",
  "intervention_summary": "Résumé des interventions/expositions étudiées",
  "outcome_summary": "Résumé des outcomes mesurés",
  "methodological_quality": "Évaluation de la qualité méthodologique globale",
  "limitations": ["Limite 1", "Limite 2", "Limite 3"],
  "research_gaps": ["Gap 1", "Gap 2", "Gap 3"],
  "clinical_implications": "Implications cliniques pratiques (2-3 paragraphes)",
  "implementation_recommendations": ["Recommandation 1", "Recommandation 2", "Recommandation 3"],
  "evidence_level": "Niveau de preuve global (Fort/Modéré/Faible/Insuffisant)",
  "grade_recommendation": "Grade de recommandation (A/B/C/D/GPP)",
  "future_research": "Directions pour la recherche future",
  "key_references": [
    {{"title": "...", "year": ..., "journal": "...", "key_contribution": "..."}}
  ]
}}
Retourne UNIQUEMENT le JSON valide."""

    try:
        client = _OAI()
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=3000,
            response_format={"type": "json_object"},
        )
        brief = _json.loads(response.choices[0].message.content)

        # Ajouter les métadonnées
        brief["_meta"] = {
            "scenario_id": scenario_id,
            "scenario_name": scenario_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "articles_used": total,
            "articles_above_threshold": total,
            "threshold": threshold,
            "human_validated": included,
            "year_range": year_range,
            "study_designs": dict(top_designs),
            "auto_generated": True,
            "model": "gpt-4.1",
        }

        # Sauvegarder en DB
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO scenario_settings (scenario_id, evidence_brief_json, brief_generated_at, updated_at)
                VALUES (:sid, CAST(:brief AS jsonb), NOW(), NOW())
                ON CONFLICT (scenario_id) DO UPDATE
                SET evidence_brief_json = CAST(:brief AS jsonb),
                    brief_generated_at = NOW(),
                    updated_at = NOW()
            """), {"sid": scenario_id, "brief": _json.dumps(brief)})

        logger.info(f"Evidence Brief LLM généré pour {scenario_id}: {len(context_articles)} articles.")
        return brief

    except Exception as e:
        logger.error(f"Evidence Brief LLM {scenario_id}: {e}", exc_info=True)
        return {"error": str(e)}


@app.post("/scenarios/{scenario_id}/evidence-brief/generate")
def generate_evidence_brief(scenario_id: str, force: bool = False) -> dict[str, Any]:
    """
    Déclenche la génération asynchrone de l'Evidence Brief LLM.
    Fonctionne pour GESICA et user_scenarios.
    """
    import threading

    if _BRIEF_GENERATION_JOBS.get(scenario_id, {}).get("status") == "running":
        return {"status": "already_running", "scenario_id": scenario_id}

    _BRIEF_GENERATION_JOBS[scenario_id] = {"status": "running"}

    def _run():
        result = _generate_evidence_brief_llm(scenario_id, force=force)
        if "error" in result:
            _BRIEF_GENERATION_JOBS[scenario_id] = {"status": "error", "error": result["error"]}
        else:
            _BRIEF_GENERATION_JOBS[scenario_id] = {"status": "done", "generated_at": result.get("_meta", {}).get("generated_at")}

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "started", "scenario_id": scenario_id}


@app.get("/scenarios/{scenario_id}/evidence-brief/generate/status")
def get_brief_generation_status(scenario_id: str) -> dict[str, Any]:
    """Statut du job de génération du brief LLM."""
    return _BRIEF_GENERATION_JOBS.get(scenario_id, {"status": "idle"})


@app.get("/scenarios/{scenario_id}/evidence-brief/llm")
def get_llm_evidence_brief(scenario_id: str) -> dict[str, Any]:
    """
    Retourne le brief LLM généré (depuis le cache DB).
    Si absent, déclenche la génération et retourne un statut pending.
    """
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT evidence_brief_json, brief_generated_at
            FROM scenario_settings WHERE scenario_id = :sid
        """), {"sid": scenario_id}).mappings().first()

    if row and row["evidence_brief_json"]:
        brief = dict(row["evidence_brief_json"])
        brief["_cached"] = True
        brief["_generated_at"] = row["brief_generated_at"].isoformat() if row["brief_generated_at"] else None
        return brief

    # Pas de brief en cache : déclencher la génération
    generate_evidence_brief(scenario_id)
    return {"status": "generating", "message": "Génération en cours, réessayez dans 30 secondes."}


# ─── VARIABLES & MODÈLE AUTO-REMPLI DEPUIS PICO ──────────────────────────────

_VARIABLES_GENERATION_JOBS: dict[str, dict] = {}


def _generate_variables_from_pico(scenario_id: str) -> dict[str, Any]:
    """
    Génère automatiquement les variables du modèle et l'outcome à partir des PICO extraits.
    Sauvegarde dans scenario_settings.variables_json.
    """
    import json as _json
    from datetime import datetime, timezone
    from openai import OpenAI as _OAI

    threshold = _get_scenario_threshold(scenario_id)
    articles = _get_above_threshold_articles(scenario_id, threshold)

    pico_articles = [a for a in articles if a.get("pico_json")]
    if not pico_articles:
        return {"error": "Aucun article avec PICO extrait pour générer les variables."}

    scenario_name = _get_scenario_name(scenario_id)

    # Construire le contexte PICO
    pico_context = []
    for a in pico_articles[:25]:
        pj = a.get("pico_json") or {}
        pico_context.append({
            "title": a.get("title", "")[:100],
            "year": a.get("year"),
            "study_design": a.get("study_design") or pj.get("study_design", ""),
            "P": pj.get("population", pj.get("P", "")),
            "I": pj.get("intervention", pj.get("I", "")),
            "C": pj.get("comparator", pj.get("C", "")),
            "O": pj.get("outcome", pj.get("O", "")),
            "key_finding": pj.get("key_finding", pj.get("conclusion", "")),
        })

    context_str = _json.dumps(pico_context, ensure_ascii=False, indent=2)

    system_prompt = """Tu es un expert en modélisation prédictive en médecine d'urgence.
A partir d'une revue systématique de la littérature, tu identifies les variables clés,
l'outcome principal, et le meilleur algorithme pour un modèle prédictif.
Tu génères un JSON structuré. Ne pas utiliser de tiret em (—)."""

    user_prompt = f"""Scénario : "{scenario_name}"
Basé sur {len(pico_articles)} articles avec extraction PICO :

{context_str}

Génère un JSON avec EXACTEMENT ces champs :
{{
  "primary_outcome": {{
    "name": "Nom de l'outcome principal",
    "definition": "Définition clinique précise",
    "measurement": "Comment le mesurer",
    "timeframe": "Horizon temporel"
  }},
  "secondary_outcomes": [
    {{"name": "...", "definition": "..."}}
  ],
  "predictor_variables": [
    {{
      "name": "Nom de la variable",
      "type": "continuous|binary|categorical|time_series",
      "definition": "Définition clinique",
      "data_source": "Source de données recommandée",
      "importance": "high|medium|low",
      "evidence_level": "Nombre d'études qui la mentionnent"
    }}
  ],
  "recommended_algorithm": {{
    "primary": "Algorithme principal recommandé",
    "alternatives": ["Alternative 1", "Alternative 2"],
    "rationale": "Justification basée sur la littérature",
    "validation_method": "Méthode de validation recommandée"
  }},
  "required_databases": ["Base 1", "Base 2"],
  "sample_size_recommendation": "Estimation de la taille d'échantillon nécessaire",
  "update_frequency": "Fréquence de mise à jour recommandée",
  "alert_thresholds": {{
    "green": {{"label": "Normal", "description": ""}},
    "orange": {{"label": "Vigilance", "description": ""}},
    "red": {{"label": "Alerte critique", "description": ""}}
  }},
  "implementation_notes": "Notes d'implémentation pratiques",
  "validation_status": "pending"
}}
Retourne UNIQUEMENT le JSON valide."""

    try:
        client = _OAI()
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.15,
            max_tokens=2000,
            response_format={"type": "json_object"},
        )
        variables = _json.loads(response.choices[0].message.content)
        variables["_meta"] = {
            "scenario_id": scenario_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "pico_articles_used": len(pico_articles),
            "auto_generated": True,
            "validation_status": "pending",
        }

        # Sauvegarder en DB
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO scenario_settings (scenario_id, variables_json, variables_validated, variables_generated_at, updated_at)
                VALUES (:sid, CAST(:vars AS jsonb), FALSE, NOW(), NOW())
                ON CONFLICT (scenario_id) DO UPDATE
                SET variables_json = CAST(:vars AS jsonb),
                    variables_validated = FALSE,
                    variables_generated_at = NOW(),
                    updated_at = NOW()
            """), {"sid": scenario_id, "vars": _json.dumps(variables)})

        logger.info(f"Variables & Modèle générés pour {scenario_id}: {len(pico_articles)} articles PICO.")
        return variables

    except Exception as e:
        logger.error(f"Variables generation {scenario_id}: {e}", exc_info=True)
        return {"error": str(e)}


@app.post("/scenarios/{scenario_id}/variables/generate")
def generate_scenario_variables(scenario_id: str) -> dict[str, Any]:
    """Déclenche la génération asynchrone des Variables & Modèle depuis les PICO."""
    import threading

    if _VARIABLES_GENERATION_JOBS.get(scenario_id, {}).get("status") == "running":
        return {"status": "already_running"}

    _VARIABLES_GENERATION_JOBS[scenario_id] = {"status": "running"}

    def _run():
        result = _generate_variables_from_pico(scenario_id)
        if "error" in result:
            _VARIABLES_GENERATION_JOBS[scenario_id] = {"status": "error", "error": result["error"]}
        else:
            _VARIABLES_GENERATION_JOBS[scenario_id] = {
                "status": "done",
                "generated_at": result.get("_meta", {}).get("generated_at"),
                "variables_count": len(result.get("predictor_variables", [])),
            }

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "started", "scenario_id": scenario_id}


@app.get("/scenarios/{scenario_id}/variables/generate/status")
def get_variables_generation_status(scenario_id: str) -> dict[str, Any]:
    """Statut du job de génération des variables."""
    return _VARIABLES_GENERATION_JOBS.get(scenario_id, {"status": "idle"})


@app.get("/scenarios/{scenario_id}/variables")
def get_scenario_variables(scenario_id: str) -> dict[str, Any]:
    """
    Retourne les variables & modèle générés.
    Si absent, déclenche la génération.
    """
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT variables_json, variables_validated, variables_generated_at
            FROM scenario_settings WHERE scenario_id = :sid
        """), {"sid": scenario_id}).mappings().first()

    if row and row["variables_json"]:
        result = dict(row["variables_json"])
        result["_validated"] = row["variables_validated"]
        result["_generated_at"] = row["variables_generated_at"].isoformat() if row["variables_generated_at"] else None
        return result

    # Déclencher la génération
    generate_scenario_variables(scenario_id)
    return {"status": "generating", "message": "Génération en cours, réessayez dans 30 secondes."}


@app.post("/scenarios/{scenario_id}/variables/validate")
def validate_scenario_variables(scenario_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """
    Valide (ou modifie) les variables & modèle générés par LLM.
    payload peut contenir les variables modifiées.
    """
    import json as _json
    from datetime import datetime, timezone

    variables_json = payload.get("variables_json")
    with engine.begin() as conn:
        if variables_json:
            conn.execute(text("""
                UPDATE scenario_settings
                SET variables_json = CAST(:vars AS jsonb),
                    variables_validated = TRUE,
                    updated_at = NOW()
                WHERE scenario_id = :sid
            """), {"sid": scenario_id, "vars": _json.dumps(variables_json)})
        else:
            conn.execute(text("""
                UPDATE scenario_settings
                SET variables_validated = TRUE, updated_at = NOW()
                WHERE scenario_id = :sid
            """), {"sid": scenario_id})

    return {"status": "validated", "scenario_id": scenario_id, "validated_at": datetime.now(timezone.utc).isoformat()}


# ─── HEATMAP AVEC VRAIS NOMS ─────────────────────────────────────────────────

@app.get("/corpus/stats/by-year/named")
def get_corpus_stats_by_year_named() -> dict[str, Any]:
    """
    Comme /corpus/stats/by-year mais avec les vrais noms des scénarios
    (GESICA et user_scenarios) dans la heatmap.
    """
    with engine.connect() as conn:
        # Articles par année (2000+)
        rows_year = conn.execute(text("""
            SELECT year, COUNT(*) as count
            FROM literature_document
            WHERE year >= 2000 AND year IS NOT NULL
            GROUP BY year ORDER BY year ASC
        """)).mappings().all()

        # Articles par scénario ET par source (heatmap)
        rows_heatmap = conn.execute(text("""
            SELECT ars.scenario_id, d.source, COUNT(*) as count
            FROM literature_document d
            JOIN article_scenarios ars ON ars.document_id = d.id
            GROUP BY ars.scenario_id, d.source
            ORDER BY ars.scenario_id, count DESC
        """)).mappings().all()

        # Articles par scénario ET par année
        rows_scenario_year = conn.execute(text("""
            SELECT d.year, ars.scenario_id, COUNT(*) as count
            FROM literature_document d
            JOIN article_scenarios ars ON ars.document_id = d.id
            WHERE d.year >= 2000 AND d.year IS NOT NULL
            GROUP BY d.year, ars.scenario_id ORDER BY d.year ASC
        """)).mappings().all()

        # Noms des user_scenarios
        user_names = conn.execute(text("""
            SELECT id, name FROM user_scenarios
        """)).mappings().all()

    user_name_map = {r["id"]: r["name"] for r in user_names}

    def _resolve_name(sid: str) -> str:
        if sid in user_name_map:
            return user_name_map[sid]
        meta = GESICA_SCENARIO_METADATA.get(sid, {})
        return meta.get("title", sid)

    by_year = {str(r["year"]): r["count"] for r in rows_year}

    heatmap: dict[str, dict[str, int]] = {}
    for r in rows_heatmap:
        name = _resolve_name(r["scenario_id"])
        src = r["source"] or "Autre"
        if name not in heatmap:
            heatmap[name] = {}
        heatmap[name][src] = heatmap[name].get(src, 0) + r["count"]

    scenario_year: dict[str, dict[str, int]] = {}
    for r in rows_scenario_year:
        name = _resolve_name(r["scenario_id"])
        yr = str(r["year"])
        if name not in scenario_year:
            scenario_year[name] = {}
        scenario_year[name][yr] = scenario_year[name].get(yr, 0) + r["count"]

    return {
        "by_year": by_year,
        "scenario_by_year": scenario_year,
        "heatmap_scenario_source": heatmap,
    }


# ─── ASSISTANT IA FILTRÉ PAR SEUIL ───────────────────────────────────────────

@app.post("/ask/stream/filtered")
async def ask_stream_filtered(payload: dict[str, Any]):
    """
    Version de /ask/stream qui filtre les chunks par seuil de similarité
    et priorise les articles validés humainement.
    """
    import asyncio
    import json as _json
    from openai import AsyncOpenAI, OpenAI as SyncOpenAI

    question = payload.get("question", "")
    scenario_id = payload.get("scenario_id", None)
    top_k = int(payload.get("top_k", 12))
    project_context = payload.get("project_context", "literev")

    if not question:
        raise HTTPException(status_code=422, detail="question est requis")

    threshold = _get_scenario_threshold(scenario_id) if scenario_id else DEFAULT_SIMILARITY_THRESHOLD

    # Embedding de la question
    try:
        sync_client = SyncOpenAI()
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
        params_extra: dict[str, Any] = {"top_k": top_k, "emb": emb_str, "threshold": threshold}

        if project_context:
            where_extra += " AND d.project_context = :project_context"
            params_extra["project_context"] = project_context

        if scenario_id:
            # Filtrer par scénario ET par seuil de similarité (ou validé humainement)
            where_extra += """
                AND EXISTS (
                    SELECT 1 FROM article_scenarios asn
                    WHERE asn.document_id = d.id
                      AND asn.scenario_id = :scenario_id
                      AND (
                          asn.similarity_score >= :threshold
                          OR asn.similarity_score IS NULL
                          OR d.screening_status = 'included'
                      )
                )
            """
            params_extra["scenario_id"] = scenario_id

        with engine.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT c.content, d.title, d.year, d.doi, d.id AS doc_id,
                       d.screening_status,
                       1 - (c.embedding <=> CAST(:emb AS vector)) AS similarity
                FROM document_chunk c
                JOIN literature_document d ON d.id = c.document_id
                WHERE c.embedding IS NOT NULL {where_extra}
                ORDER BY
                    CASE WHEN d.screening_status = 'included' THEN 0 ELSE 1 END,
                    c.embedding <=> CAST(:emb AS vector)
                LIMIT :top_k
            """), params_extra).mappings().all()

        for r in rows:
            context_chunks.append(r["content"])
            sources.append({
                "id": r["doc_id"],
                "title": r["title"],
                "year": r["year"],
                "doi": r["doi"],
                "similarity": round(float(r["similarity"]), 3),
                "validated": r["screening_status"] == "included",
            })

    context_text = "\n\n---\n\n".join(context_chunks[:top_k]) if context_chunks else "Aucun contexte disponible."

    system_prompt = """Tu es un assistant expert en médecine d'urgence et en revue systématique de la littérature scientifique.
Tu réponds en français de manière précise, factuelle et structurée.
Base-toi exclusivement sur le contexte fourni. Si l'information n'est pas dans le contexte, dis-le clairement.
Cite les articles pertinents par leur titre quand tu les mentionnes.
Ne pas utiliser de tiret em (—)."""

    user_prompt = f"""Contexte scientifique (extraits d'articles sélectionnés par pertinence sémantique) :
{context_text}

Question : {question}

Réponds de manière structurée et cite les sources pertinentes du contexte."""

    async def event_generator():
        import json as _json2
        sources_event = f"event: sources\ndata: {_json2.dumps(sources)}\n\n"
        yield sources_event

        try:
            async_client = AsyncOpenAI()
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


# ─── PIPELINE COMPLET AVEC BRIEF LLM ─────────────────────────────────────────

@app.post("/scenarios/{scenario_id}/full-pipeline")
def trigger_full_pipeline_with_brief(scenario_id: str) -> dict[str, Any]:
    """
    Déclenche le pipeline complet incluant :
    1. Reranking sémantique
    2. Génération Evidence Brief LLM
    3. Génération Variables & Modèle
    Fonctionne pour GESICA et user_scenarios.
    """
    import threading

    if scenario_id.startswith("usr-"):
        row = _get_user_scenario_or_404(scenario_id)
        query = row["query"]
    else:
        meta = GESICA_SCENARIO_METADATA.get(scenario_id)
        if not meta:
            raise HTTPException(status_code=404, detail="Scénario non trouvé")
        nl = meta.get("nl_queries", [])
        query = nl[0] if nl else meta.get("title", scenario_id)

    def _run():
        logger.info(f"Full pipeline with brief: {scenario_id}")
        # 1. Reranking
        _run_semantic_rerank(scenario_id, query)
        # 2. Evidence Brief LLM
        _generate_evidence_brief_llm(scenario_id, force=True)
        # 3. Variables & Modèle
        _generate_variables_from_pico(scenario_id)
        logger.info(f"Full pipeline with brief done: {scenario_id}")

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "started", "scenario_id": scenario_id, "steps": ["rerank", "evidence_brief", "variables"]}
