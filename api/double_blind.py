"""Double-blind screening decisions, conflicts and Cohen's kappa.

Extracted from main.py (LiteRev API); `main` re-exports everything for the scripts,
tools and tests.
"""
from __future__ import annotations

from typing import Any

from fastapi import Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import text

from .core import app, engine, require_api_key
from .scenario_store import _get_user_scenario_or_404

class DoubleBlindDecisionIn(BaseModel):
    article_id: int
    reviewer: int  # 1 ou 2
    status: str    # 'included' | 'excluded' | 'pending'
    reason: str | None = None
    reviewer_code: str | None = None  # Code reviewer (ex: R-2847)


def _write_ars_screening(
    conn,
    scenario_id: str,
    document_id: int,
    status: str | None,
    reason: str | None = None,
    notes: str | None = None,
) -> None:
    """Migration 2 (per-scenario screening) dual-write: record the screening
    decision on the (scenario_id, document_id) article_scenarios row. Callers
    also keep the global literature_document.screening_status updated, which
    remains authoritative until the Phase 5 cutover. No-op for rows that are not
    members of the scenario (UPDATE matches nothing)."""
    conn.execute(text("""
        UPDATE article_scenarios
        SET screening_status = :status,
            screening_reason = :reason,
            screening_notes  = :notes,
            screened_at      = NOW()
        WHERE scenario_id = :sid AND document_id = :doc_id
    """), {"status": status, "reason": reason, "notes": notes,
           "sid": scenario_id, "doc_id": document_id})


@app.post("/gesica/scenarios/{scenario_id}/double-blind/decision")
def submit_double_blind_decision(
    scenario_id: str,
    payload: DoubleBlindDecisionIn,
    _: None = Depends(require_api_key),
) -> dict[str, Any]:
    """
    Soumet la décision d'un reviewer (1 ou 2) pour le screening double-aveugle.
    Si les deux reviewers ont statué, calcule automatiquement la concordance.
    """
    if payload.reviewer not in (1, 2):
        raise HTTPException(status_code=422, detail="reviewer doit être 1 ou 2")
    if payload.status not in ("included", "excluded", "pending"):
        raise HTTPException(status_code=422, detail="status invalide")

    col_status = f"reviewer_{payload.reviewer}_status"
    col_reason = f"reviewer_{payload.reviewer}_reason"

    with engine.begin() as conn:
        # Vérifier que l'article appartient bien au scénario (via article_scenarios)
        exists = conn.execute(text("""
            SELECT 1 FROM article_scenarios
            WHERE document_id = :article_id AND scenario_id = :scenario_id
        """), {"article_id": payload.article_id, "scenario_id": scenario_id}).first()
        
        if not exists:
            raise HTTPException(status_code=404, detail="Article non trouvé dans ce scénario")
        
        # Décision reviewer PAR SCÉNARIO (article_scenarios) — autoritative pour le
        # kappa : un document partagé entre scénarios porte des votes distincts. On
        # RÉCUPÈRE les deux statuts de CE scénario pour décider de la concordance.
        ars_row = conn.execute(text(f"""
            UPDATE article_scenarios
            SET {col_status} = :status,
                {col_reason} = :reason
            WHERE scenario_id = :sid AND document_id = :article_id
            RETURNING reviewer_1_status, reviewer_2_status
        """), {
            "status": payload.status,
            "reason": payload.reason,
            "sid": scenario_id,
            "article_id": payload.article_id,
        }).first()
        # Dual-write global (literature_document) — hérité, conservé pour le badge
        # du corpus et d'éventuels lecteurs legacy. best-effort (jamais bloquant).
        try:
            conn.execute(text(f"""
                UPDATE literature_document
                SET {col_status} = :status,
                    {col_reason} = :reason
                WHERE id = :article_id
            """), {
                "status": payload.status,
                "reason": payload.reason,
                "article_id": payload.article_id,
            })
        except Exception:
            pass

        if not ars_row:
            raise HTTPException(status_code=404, detail="Article non trouvé")

        r1 = ars_row["reviewer_1_status"]
        r2 = ars_row["reviewer_2_status"]
        agreement = None
        final_status = None

        # Si les deux reviewers ont statué DANS CE SCÉNARIO → concordance + résolution
        if r1 and r2:
            agreement = r1 == r2
            if agreement:
                final_status = r1
            else:
                # Désaccord → statut "conflict" (à résoudre manuellement)
                final_status = "conflict"

            conn.execute(text("""
                UPDATE article_scenarios
                SET kappa_resolved = :resolved,
                    kappa_final_status = :final
                WHERE scenario_id = :sid AND document_id = :article_id
            """), {
                "resolved": agreement,
                "final": final_status,
                "sid": scenario_id,
                "article_id": payload.article_id,
            })
            try:
                conn.execute(text("""
                    UPDATE literature_document
                    SET kappa_resolved = :resolved,
                        kappa_final_status = :final,
                        screening_status = :screening
                    WHERE id = :article_id
                """), {
                    "resolved": agreement,
                    "final": final_status,
                    "screening": final_status if agreement else "pending",
                    "article_id": payload.article_id,
                })
            except Exception:
                pass
            # Migration 2 dual-write: per-scenario screening final status
            _write_ars_screening(conn, scenario_id, payload.article_id,
                                 final_status if agreement else "pending")

    return {
        "id": payload.article_id,
        "reviewer": payload.reviewer,
        "status": payload.status,
        "reviewer_1_status": r1 if payload.reviewer == 2 else payload.status,
        "reviewer_2_status": r2 if payload.reviewer == 1 else payload.status,
        "agreement": agreement,
        "final_status": final_status,
    }


@app.post("/gesica/scenarios/{scenario_id}/double-blind/resolve")
def resolve_conflict(
    scenario_id: str,
    article_id: int,
    final_status: str,
    arbitrator_notes: str | None = None,
    _: None = Depends(require_api_key),
) -> dict[str, Any]:
    """Résout un conflit entre reviewers (arbitrage par un tiers)."""
    if final_status not in ("included", "excluded"):
        raise HTTPException(status_code=422, detail="final_status doit être 'included' ou 'excluded'")

    with engine.begin() as conn:
        # Résolution PAR SCÉNARIO (autoritative) — l'UPDATE scopé sert aussi de
        # contrôle d'appartenance (RETURNING vide ⇒ l'article n'est pas dans ce
        # scénario ⇒ 404).
        row = conn.execute(text("""
            UPDATE article_scenarios
            SET kappa_final_status = :final,
                kappa_resolved = TRUE
            WHERE scenario_id = :scenario_id AND document_id = :article_id
            RETURNING document_id
        """), {
            "final": final_status,
            "scenario_id": scenario_id,
            "article_id": article_id,
        }).first()
        if not row:
            raise HTTPException(status_code=404, detail="Article non trouvé")
        # Dual-write global (hérité, best-effort — badge du corpus / lecteurs legacy).
        try:
            conn.execute(text("""
                UPDATE literature_document
                SET kappa_final_status = :final,
                    kappa_resolved = TRUE,
                    screening_status = :final,
                    screening_notes = :notes
                WHERE id = :article_id AND project_context = 'literev'
            """), {
                "final": final_status,
                "notes": arbitrator_notes,
                "article_id": article_id,
            })
        except Exception:
            pass
        # Migration 2 dual-write: per-scenario screening for this scenario
        _write_ars_screening(conn, scenario_id, article_id, final_status, notes=arbitrator_notes)
    return {"id": article_id, "final_status": final_status, "resolved": True}


@app.get("/user-scenarios/{scenario_id}/double-blind/kappa")
def get_user_scenario_kappa(scenario_id: str) -> dict[str, Any]:
    """Kappa de Cohen pour un scénario utilisateur."""
    _get_user_scenario_or_404(scenario_id)
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT ars.reviewer_1_status, ars.reviewer_2_status
            FROM article_scenarios ars
            WHERE ars.scenario_id = :sid
              AND ars.reviewer_1_status IS NOT NULL
              AND ars.reviewer_2_status IS NOT NULL
        """), {"sid": scenario_id}).mappings().all()
    if not rows:
        return {
            "scenario_id": scenario_id, "n_evaluated": 0, "kappa": None,
            "interpretation": "Aucune évaluation double-aveugle disponible",
            "agreements": {}, "conflicts": 0,
        }
    n = len(rows)
    categories = ["included", "excluded", "pending"]
    matrix = {c1: {c2: 0 for c2 in categories} for c1 in categories}
    for r in rows:
        r1 = r["reviewer_1_status"] if r["reviewer_1_status"] in categories else "pending"
        r2 = r["reviewer_2_status"] if r["reviewer_2_status"] in categories else "pending"
        matrix[r1][r2] += 1
    po = sum(matrix[c][c] for c in categories) / n
    pe = sum((sum(matrix[c][c2] for c2 in categories) / n) * (sum(matrix[c1][c] for c1 in categories) / n) for c in categories)
    kappa = (po - pe) / (1 - pe) if pe < 1.0 else 1.0
    if kappa >= 0.81: interpretation = "Quasi-parfait (≥ 0.81)"
    elif kappa >= 0.61: interpretation = "Substantiel (0.61–0.80)"
    elif kappa >= 0.41: interpretation = "Modéré (0.41–0.60)"
    elif kappa >= 0.21: interpretation = "Faible (0.21–0.40)"
    else: interpretation = "Médiocre (< 0.21)"
    conflicts = sum(matrix[r1][r2] for r1 in categories for r2 in categories if r1 != r2)
    return {
        "scenario_id": scenario_id, "n_evaluated": n, "kappa": round(kappa, 4),
        "po_observed": round(po, 4), "pe_expected": round(pe, 4),
        "interpretation": interpretation, "conflicts": conflicts,
        "agreements": {c: matrix[c][c] for c in categories}, "matrix": matrix,
    }


@app.get("/user-scenarios/{scenario_id}/double-blind/conflicts")
def get_user_scenario_conflicts(scenario_id: str) -> list[dict[str, Any]]:
    """Conflits double-aveugle pour un scénario utilisateur."""
    _get_user_scenario_or_404(scenario_id)
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT d.id, d.title, d.abstract, d.year, d.journal,
                   ars.reviewer_1_status, ars.reviewer_1_reason,
                   ars.reviewer_2_status, ars.reviewer_2_reason, ars.kappa_final_status
            FROM article_scenarios ars
            JOIN literature_document d ON d.id = ars.document_id
            WHERE ars.scenario_id = :sid
              AND ars.reviewer_1_status IS NOT NULL
              AND ars.reviewer_2_status IS NOT NULL
              AND ars.reviewer_1_status != ars.reviewer_2_status
            ORDER BY d.id
        """), {"sid": scenario_id}).mappings().all()
    return [dict(r) for r in rows]


@app.post("/user-scenarios/{scenario_id}/double-blind/decision")
def submit_user_scenario_double_blind_decision(
    scenario_id: str,
    payload: DoubleBlindDecisionIn,
    _: None = Depends(require_api_key),
) -> dict[str, Any]:
    """Décision double-aveugle pour un article d'un scénario utilisateur."""
    _get_user_scenario_or_404(scenario_id)
    # Vérifier que l'article appartient au scénario
    with engine.connect() as conn:
        exists = conn.execute(text("""
            SELECT 1 FROM article_scenarios WHERE document_id = :doc_id AND scenario_id = :sid
        """), {"doc_id": payload.article_id, "sid": scenario_id}).first()
    if not exists:
        raise HTTPException(status_code=404, detail="Article non trouvé dans ce scénario utilisateur")
    # Déléguer à l'implémentation existante
    return submit_double_blind_decision(scenario_id, payload)
