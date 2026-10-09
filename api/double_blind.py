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
from .scenario_store import (_get_scenario_threshold, _get_user_scenario_or_404,
                             relevant_gate_sql)

class DoubleBlindDecisionIn(BaseModel):
    article_id: int
    # Conservé pour compatibilité ascendante, mais IGNORÉ : le rôle est déduit du code
    # du relecteur, côté serveur. Le laisser décider par le client était la cause du
    # « double aveugle à un seul relecteur » (cf. _reviewer_index_for_code).
    reviewer: int | None = None
    status: str    # 'included' | 'excluded' | 'pending'
    reason: str | None = None
    reviewer_code: str | None = None  # Code reviewer (ex: R-2847)


# ── Qui est relecteur 1, qui est relecteur 2 ─────────────────────────────────
# Le rôle était attribué PAR NAVIGATEUR : le client regardait un `sessionStorage`
# propre à son onglet et se donnait le rôle 1 s'il le trouvait libre. Deux relecteurs
# sur deux machines, ce qui est exactement la situation que le double aveugle décrit,
# recevaient donc tous les deux le rôle 1 : les décisions du second écrasaient celles
# du premier dans `reviewer_1_status`, `reviewer_2_status` restait NULL, et la requête
# du kappa (qui exige les deux colonnes non NULL) ne renvoyait aucune ligne. Le panneau
# annonçait « aucune évaluation double-aveugle disponible » après une journée entière de
# screening à deux. Le code envoyé par le client était par ailleurs ignoré : rien, nulle
# part, ne gardait trace de QUI avait voté.
#
# Le rôle est désormais attribué et conservé par le serveur, par scénario.
def _ensure_reviewer_registry() -> None:
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS scenario_reviewer (
                scenario_id    TEXT NOT NULL,
                reviewer_code  TEXT NOT NULL,
                reviewer_index SMALLINT NOT NULL CHECK (reviewer_index IN (1, 2)),
                registered_at  TIMESTAMP NOT NULL DEFAULT NOW(),
                PRIMARY KEY (scenario_id, reviewer_code)
            )
        """))
        conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS uq_scenario_reviewer_index "
                          "ON scenario_reviewer (scenario_id, reviewer_index)"))


try:
    _ensure_reviewer_registry()
except Exception as _e:                                   # noqa: BLE001 - jamais bloquant
    from .core import logger as _boot_logger
    _boot_logger.warning(f"_ensure_reviewer_registry: {_e}")


def _normalise_reviewer_code(code: str | None) -> str:
    c = (code or "").strip().upper()
    if not c:
        raise HTTPException(status_code=422,
                            detail="Code relecteur requis : le rôle (1 ou 2) en dépend.")
    if not c.startswith("R-"):
        c = f"R-{c}"
    return c[:32]


def _reviewer_index_for_code(conn, scenario_id: str, code: str) -> int:
    """Le rôle de ce code DANS CE SCÉNARIO, attribué une fois pour toutes.

    Premier code enregistré : relecteur 1. Deuxième : relecteur 2. Un TROISIÈME est
    refusé, parce qu'un double aveugle à trois n'existe pas et qu'accepter le troisième
    revenait à écraser le vote de l'un des deux."""
    existing = conn.execute(text("""
        SELECT reviewer_code, reviewer_index FROM scenario_reviewer
        WHERE scenario_id = :sid ORDER BY reviewer_index
    """), {"sid": scenario_id}).mappings().all()
    for row in existing:
        if row["reviewer_code"] == code:
            return int(row["reviewer_index"])
    if len(existing) >= 2:
        raise HTTPException(
            status_code=409,
            detail=("Deux relecteurs sont déjà enregistrés sur ce scénario "
                    f"({', '.join(r['reviewer_code'] for r in existing)}). "
                    "Un double aveugle en compte deux."))
    index = 2 if existing else 1
    conn.execute(text("""
        INSERT INTO scenario_reviewer (scenario_id, reviewer_code, reviewer_index)
        VALUES (:sid, :code, :idx)
        ON CONFLICT (scenario_id, reviewer_code) DO NOTHING
    """), {"sid": scenario_id, "code": code, "idx": index})
    return index


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
    if payload.status not in ("included", "excluded", "pending"):
        raise HTTPException(status_code=422, detail="status invalide")
    code = _normalise_reviewer_code(payload.reviewer_code)

    with engine.begin() as conn:
        # Vérifier que l'article appartient bien au scénario (via article_scenarios)
        exists = conn.execute(text("""
            SELECT 1 FROM article_scenarios
            WHERE document_id = :article_id AND scenario_id = :scenario_id
        """), {"article_id": payload.article_id, "scenario_id": scenario_id}).first()

        if not exists:
            raise HTTPException(status_code=404, detail="Article non trouvé dans ce scénario")

        # Le rôle vient du REGISTRE du serveur, jamais du client : `payload.reviewer`
        # est ignoré. Un client qui se donne son propre rôle finit par se le donner deux
        # fois, et le deuxième relecteur écrase le premier.
        reviewer = _reviewer_index_for_code(conn, scenario_id, code)
        col_status = f"reviewer_{reviewer}_status"
        col_reason = f"reviewer_{reviewer}_reason"

        # Décision reviewer PAR SCÉNARIO (article_scenarios) - autoritative pour le
        # kappa : un document partagé entre scénarios porte des votes distincts. On
        # RÉCUPÈRE les deux statuts de CE scénario pour décider de la concordance.
        #
        # ÉCRITURE UNIQUE par relecteur : le `WHERE ... IS NULL` empêche un arbitrage,
        # ou un second passage, de modifier un vote déjà rendu. Les boutons d'arbitrage
        # du panneau de conflits appelaient cet endpoint : l'arbitre réécrivait
        # `reviewer_1_status`, puis `r1 == r2` devenait vrai, `kappa_resolved` passait à
        # vrai, et le kappa comptait une concordance que personne n'avait exprimée.
        ars_row = conn.execute(text(f"""
            UPDATE article_scenarios
            SET {col_status} = :status,
                {col_reason} = :reason
            WHERE scenario_id = :sid AND document_id = :article_id
              AND {col_status} IS NULL
            RETURNING reviewer_1_status, reviewer_2_status
        """), {
            "status": payload.status,
            "reason": payload.reason,
            "sid": scenario_id,
            "article_id": payload.article_id,
        # `.mappings()` : la ligne était lue par son NOM sur un Row de SQLAlchemy 2, qui
        # ne s'indexe que par position. Le TypeError partait à l'intérieur de
        # `engine.begin()`, la transaction était annulée, le vote n'était jamais écrit et
        # l'API répondait 500. Aucun kappa ne pouvait donc exister, et la fonctionnalité
        # n'avait jamais marché depuis la découpe du paquet api.
        }).mappings().first()
        # Le dual-write global a été RETIRÉ. `literature_document` est partagé entre
        # tous les scénarios : écrire le vote d'un relecteur sur la ligne du document
        # faisait sortir l'article du sous-ensemble pertinent des AUTRES revues qui le
        # contiennent, de leur brief, de leurs extractions et de leurs exports, sans
        # qu'un seul écran de ces revues le dise. Une décision appartient à la revue où
        # elle est prise : `article_scenarios` la porte, et elle seule.

        if not ars_row:
            # L'article est bien dans le scénario (vérifié ci-dessus) : si rien n'a été
            # mis à jour, c'est que ce relecteur avait déjà voté.
            raise HTTPException(
                status_code=409,
                detail=(f"Le relecteur {reviewer} ({code}) a déjà statué sur cet article. "
                        "Un vote de double aveugle ne se réécrit pas ; un désaccord se "
                        "tranche par l'arbitrage."))

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
            # Même raison : le statut final d'un désaccord arbitré dans CETTE revue
            # n'a pas à décider du sort de l'article dans les autres.
            # Statut de screening final, par scénario
            _write_ars_screening(conn, scenario_id, payload.article_id,
                                 final_status if agreement else "pending")

    return {
        "id": payload.article_id,
        "reviewer": reviewer,
        "reviewer_code": code,
        "status": payload.status,
        "reviewer_1_status": r1,
        "reviewer_2_status": r2,
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
        # Résolution PAR SCÉNARIO (autoritative) - l'UPDATE scopé sert aussi de
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
        # Le dual-write global a été retiré ici aussi : un arbitrage rendu dans CETTE
        # revue décidait du sort de l'article dans toutes les autres.
        # Statut de screening de cette revue
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
                   ars.reviewer_2_status, ars.reviewer_2_reason, ars.kappa_final_status,
                   -- Qui a voté. Le panneau affichait déjà `reviewer_1_code` et
                   -- `reviewer_2_code`, qui n'existaient nulle part : la ligne était
                   -- donc toujours vide, et un conflit ne nommait personne.
                   (SELECT reviewer_code FROM scenario_reviewer sr
                     WHERE sr.scenario_id = ars.scenario_id AND sr.reviewer_index = 1)
                       AS reviewer_1_code,
                   (SELECT reviewer_code FROM scenario_reviewer sr
                     WHERE sr.scenario_id = ars.scenario_id AND sr.reviewer_index = 2)
                       AS reviewer_2_code
            FROM article_scenarios ars
            JOIN literature_document d ON d.id = ars.document_id
            WHERE ars.scenario_id = :sid
              AND ars.reviewer_1_status IS NOT NULL
              AND ars.reviewer_2_status IS NOT NULL
              AND ars.reviewer_1_status != ars.reviewer_2_status
            ORDER BY d.id
        """), {"sid": scenario_id}).mappings().all()
    return [dict(r) for r in rows]


@app.get("/user-scenarios/{scenario_id}/double-blind/queue")
def get_user_scenario_double_blind_queue(
    scenario_id: str,
    reviewer_code: str,
    limit: int = 25,
) -> dict[str, Any]:
    """Les articles sur lesquels CE relecteur n'a pas encore voté.

    Il n'existait aucun moyen de voter. Les deux seuls boutons du panneau de double
    aveugle étaient ceux d'arbitrage, et ils appelaient l'endpoint de décision : la liste
    des conflits ne peut se remplir que si les deux relecteurs ont voté, et rien dans
    l'interface ne permettait de le faire. La fonctionnalité affichait un kappa qui ne
    pouvait jamais exister.

    Le lot est le sous-ensemble pertinent, trié par pertinence : on ne demande pas à un
    relecteur de juger un article que la revue n'a pas retenu."""
    _get_user_scenario_or_404(scenario_id)
    code = _normalise_reviewer_code(reviewer_code)
    thr = _get_scenario_threshold(scenario_id)
    with engine.begin() as conn:
        reviewer = _reviewer_index_for_code(conn, scenario_id, code)
        col = f"reviewer_{reviewer}_status"
        rows = conn.execute(text(f"""
            SELECT d.id, d.title, d.abstract, d.year, d.journal, d.doi,
                   ars.similarity_score, ars.rerank_score,
                   ars.reviewer_1_status, ars.reviewer_2_status
            FROM article_scenarios ars
            JOIN literature_document d ON d.id = ars.document_id
            WHERE ars.scenario_id = :sid
              AND ars.{col} IS NULL
              AND {relevant_gate_sql('d', 'ars', ':thr')}
            ORDER BY COALESCE(ars.rerank_score, ars.similarity_score, 0) DESC NULLS LAST,
                     d.id
            LIMIT :lim
        """), {"sid": scenario_id, "thr": thr, "lim": max(1, min(int(limit), 200))}
        ).mappings().all()
        remaining = conn.execute(text(f"""
            SELECT COUNT(*) FROM article_scenarios ars
            JOIN literature_document d ON d.id = ars.document_id
            WHERE ars.scenario_id = :sid AND ars.{col} IS NULL
              AND {relevant_gate_sql('d', 'ars', ':thr')}
        """), {"sid": scenario_id, "thr": thr}).scalar() or 0
    return {
        "scenario_id": scenario_id, "reviewer": reviewer, "reviewer_code": code,
        "remaining": int(remaining), "returned": len(rows),
        "articles": [dict(r) for r in rows],
    }


@app.post("/user-scenarios/{scenario_id}/double-blind/register")
def register_user_scenario_reviewer(
    scenario_id: str,
    reviewer_code: str,
    _: None = Depends(require_api_key),
) -> dict[str, Any]:
    """Enregistre un code relecteur sur ce scénario et renvoie SON rôle.

    C'est le serveur qui décide, une fois pour toutes : le premier code enregistré est
    relecteur 1, le second relecteur 2, un troisième est refusé. Le client s'attribuait
    son rôle lui-même en lisant un `sessionStorage` propre à son onglet, donc deux
    relecteurs sur deux machines recevaient tous deux le rôle 1."""
    _get_user_scenario_or_404(scenario_id)
    code = _normalise_reviewer_code(reviewer_code)
    with engine.begin() as conn:
        index = _reviewer_index_for_code(conn, scenario_id, code)
        others = conn.execute(text("""
            SELECT reviewer_index, reviewer_code FROM scenario_reviewer
            WHERE scenario_id = :sid ORDER BY reviewer_index
        """), {"sid": scenario_id}).mappings().all()
    return {
        "scenario_id": scenario_id, "reviewer": index, "reviewer_code": code,
        "registered": {int(r["reviewer_index"]): r["reviewer_code"] for r in others},
    }


@app.post("/user-scenarios/{scenario_id}/double-blind/resolve")
def resolve_user_scenario_conflict(
    scenario_id: str,
    article_id: int,
    final_status: str,
    arbitrator_notes: str | None = None,
    _: None = Depends(require_api_key),
) -> dict[str, Any]:
    """Arbitrage d'un désaccord, pour un scénario utilisateur.

    La route n'existait que sous `/gesica/scenarios/...`, donc inatteignable pour tout
    scénario réel. Les boutons d'arbitrage du panneau appelaient faute de mieux
    l'endpoint de DÉCISION, qui réécrivait le vote d'un relecteur : `r1 == r2` devenait
    alors vrai, `kappa_resolved` passait à vrai, et le kappa comptait une concordance
    que personne n'avait exprimée."""
    _get_user_scenario_or_404(scenario_id)
    return resolve_conflict(scenario_id, article_id, final_status, arbitrator_notes)


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
