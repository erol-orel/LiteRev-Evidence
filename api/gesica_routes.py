"""GESICA-prefixed routes that delegate to the user-scenario implementations, plus aliases.

Extracted from main.py (LiteRev API); `main` re-exports everything for the scripts,
tools and tests.
"""
from __future__ import annotations

from typing import Any

from fastapi import Depends, Query
from fastapi import UploadFile, File

from .core import app, require_api_key
from .gesica import _get_db_gesica_scenario_or_404, get_scenario_detail
from .clustering import get_user_scenario_clustering, get_user_scenario_clustering_status
from .knowledge_graph import _compute_user_kg
from .double_blind import get_user_scenario_conflicts, get_user_scenario_kappa
from .review import (
    extract_user_scenario_article_pico,
    get_user_scenario_article_pico,
    get_user_scenario_pico_bulk,
    get_user_scenario_pico_stats,
    get_user_scenario_prisma,
    get_user_scenario_screening_progress,
    screen_user_scenario_article,
)
from .evidence import get_user_scenario_evidence_brief, get_user_scenario_evidence_brief_pdf
from .assistant import AskIn, user_scenario_rag_assistant
from .model_data import upload_model_dataset

@app.get("/gesica/scenarios/{scenario_id}/clustering")
def get_scenario_clustering(scenario_id: str, force_refresh: bool = False, lang: str | None = Query(None)) -> dict[str, Any]:
    """Delegue a l'implementation user-scenario unifiee (pipeline unique)."""
    return get_user_scenario_clustering(scenario_id, force_refresh, lang)


@app.get("/gesica/scenarios/{scenario_id}/clustering/status")
def get_clustering_status(scenario_id: str, lang: str | None = Query(None)) -> dict:
    """Delegue a l'implementation user-scenario unifiee (pipeline unique)."""
    return get_user_scenario_clustering_status(scenario_id, lang)


@app.post("/gesica/scenarios/{scenario_id}/rag")
def scenario_rag_assistant(scenario_id: str, payload: AskIn) -> dict[str, Any]:
    """Delegue a l'implementation user-scenario unifiee (pipeline unique)."""
    return user_scenario_rag_assistant(scenario_id, payload)


@app.get("/gesica/scenarios/{scenario_id}/prisma")
def get_scenario_prisma(scenario_id: str, threshold: float = Query(None)) -> dict[str, Any]:
    """Delegue a l'implementation user-scenario unifiee (PRISMA modernise)."""
    return get_user_scenario_prisma(scenario_id, threshold)

@app.post("/gesica/scenarios/{scenario_id}/upload-dataset")
async def upload_scenario_dataset(
    scenario_id: str,
    file: UploadFile = File(...),
    _: None = Depends(require_api_key),
) -> dict[str, Any]:
    """Upload d'un dataset (CSV/XLSX) pour l'entraînement du modèle d'un scénario.

    Délègue au pipeline unifié POST /scenarios/{id}/model/data : valide les en-têtes
    contre le data_template, stocke le dataset comme ACTIF (celui que lit réellement
    l'entraînement) et lance l'entraînement si les données suffisent. Corrige l'ancien
    comportement où le fichier était écrit sur un chemin JAMAIS lu par l'entraînement —
    « stocké » sans jamais alimenter le modèle, alors que la réponse le sous-entendait.
    Nécessite un model_spec validé (sinon 400 : définir d'abord les Variables & Modèle)."""
    _get_db_gesica_scenario_or_404(scenario_id)
    # auto_train=True : upload → validation → dataset ACTIF → entraînement automatique.
    return await upload_model_dataset(scenario_id, file, auto_train=True)


# ─── PICO Extraction Endpoints ───────────────────────────────────────────────

@app.get("/gesica/scenarios/{scenario_id}/articles/{article_id}/pico")
def get_article_pico(scenario_id: str, article_id: int):
    """Delegue a l'implementation user-scenario unifiee (pipeline unique)."""
    return get_user_scenario_article_pico(scenario_id, article_id)


@app.post("/gesica/scenarios/{scenario_id}/articles/{article_id}/pico/extract")
def extract_article_pico(scenario_id: str, article_id: int, _: None = Depends(require_api_key)):
    """Delegue a l'implementation user-scenario unifiee (pipeline unique)."""
    return extract_user_scenario_article_pico(scenario_id, article_id)


@app.get("/gesica/scenarios/{scenario_id}/pico-stats")
def get_scenario_pico_stats(scenario_id: str):
    """Statistiques PICO — delegue a l'implementation user-scenario unifiee."""
    return get_user_scenario_pico_stats(scenario_id)


# ─── Screening PRISMA par article dans un scénario ───────────────────────────

@app.post("/gesica/scenarios/{scenario_id}/articles/{article_id}/screen")
def screen_scenario_article(
    scenario_id: str,
    article_id: int,
    status: str,
    reason: str | None = None,
    notes: str | None = None,
    _: None = Depends(require_api_key),
):
    """Delegue a l'implementation user-scenario unifiee (pipeline unique)."""
    return screen_user_scenario_article(scenario_id, article_id, status, reason, notes)


@app.get("/gesica/scenarios/{scenario_id}/screening-progress")
def get_scenario_screening_progress(scenario_id: str) -> dict[str, Any]:
    """Progression du screening PRISMA — delegue a l'implementation user-scenario unifiee."""
    return get_user_scenario_screening_progress(scenario_id)

# ─── PICO Bulk : tous les articles d'un scénario avec PICO ────────────────────────────────────────────
@app.get("/gesica/scenarios/{scenario_id}/pico-bulk")
def get_scenario_pico_bulk(scenario_id: str, limit: int = 100000, offset: int = 0) -> dict[str, Any]:
    """Delegue a l'implementation user-scenario unifiee (pipeline unique)."""
    return get_user_scenario_pico_bulk(scenario_id, limit, offset)

# ─── Evidence Brief PDF ───────────────────────────────────────────────────────
@app.get("/gesica/scenarios/{scenario_id}/evidence-brief")
def get_evidence_brief(scenario_id: str) -> dict[str, Any]:
    """Delegue a l'implementation user-scenario unifiee (pipeline unique)."""
    return get_user_scenario_evidence_brief(scenario_id)


@app.get("/gesica/scenarios/{scenario_id}/double-blind/kappa")
def get_kappa_stats(scenario_id: str) -> dict[str, Any]:
    """Kappa de Cohen — delegue a l'implementation user-scenario unifiee."""
    return get_user_scenario_kappa(scenario_id)


@app.get("/gesica/scenarios/{scenario_id}/double-blind/conflicts")
def get_conflicts(scenario_id: str) -> list[dict[str, Any]]:
    """Conflits double-aveugle — delegue a l'implementation user-scenario unifiee."""
    return get_user_scenario_conflicts(scenario_id)


@app.get("/gesica/scenarios/{scenario_id}/knowledge-graph")
def get_knowledge_graph(
    scenario_id: str,
    max_nodes: int = 400,
    min_similarity: float = 0.35,
) -> dict[str, Any]:
    """Delegue a l'implementation user-scenario unifiee (pipeline unique)."""
    return _compute_user_kg(scenario_id, max_nodes, min_similarity)


# ─── PDF EVIDENCE BRIEF CÔTÉ SERVEUR ─────────────────────────────────────────

@app.get("/gesica/scenarios/{scenario_id}/evidence-brief/pdf")
def get_evidence_brief_pdf(scenario_id: str):
    """Delegue a l'implementation user-scenario unifiee (pipeline unique)."""
    return get_user_scenario_evidence_brief_pdf(scenario_id)

# ─── Alias GESICA : /gesica/scenarios/{id}/pico -> pico-bulk ─────────────────
@app.get("/gesica/scenarios/{scenario_id}/pico")
def get_gesica_scenario_pico_alias(
    scenario_id: str,
    limit: int = 100000,
) -> dict[str, Any]:
    """Alias vers pico-bulk pour compatibilité frontend."""
    return get_scenario_pico_bulk(scenario_id, limit=limit)

# ─── Alias GESICA : /gesica/scenarios/{id}/screening -> screening-progress ───
@app.get("/gesica/scenarios/{scenario_id}/screening")
def get_gesica_scenario_screening_alias(scenario_id: str) -> dict[str, Any]:
    """Alias vers screening-progress pour compatibilité frontend."""
    return get_scenario_screening_progress(scenario_id)

# ─── Alias user-scenarios : /user-scenarios/{id}/pico -> pico-bulk ───────────
@app.get("/user-scenarios/{scenario_id}/pico")
def get_user_scenario_pico_alias(
    scenario_id: str,
    limit: int = 100000,
) -> dict[str, Any]:
    """Alias vers pico-bulk pour compatibilité frontend."""
    return get_user_scenario_pico_bulk(scenario_id, limit=limit)

# ─── Alias user-scenarios : /user-scenarios/{id}/screening -> screening-progress
@app.get("/user-scenarios/{scenario_id}/screening")
def get_user_scenario_screening_alias(scenario_id: str) -> dict[str, Any]:
    """Alias vers screening-progress pour compatibilité frontend."""
    return get_user_scenario_screening_progress(scenario_id)

# ─── Alias GESICA : /gesica/scenarios/{id} → /gesica/scenarios/{id}/detail ───
@app.get("/gesica/scenarios/{scenario_id}")
def get_gesica_scenario_root_alias(scenario_id: str) -> dict[str, Any]:
    """Alias vers /detail pour compatibilité frontend (évite les 404 sur la route racine)."""
    return get_scenario_detail(scenario_id)
