"""Living review status and triggers.

Extracted from main.py (LiteRev API); `main` re-exports everything for the scripts,
tools and tests.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import Depends, Query

from .core import _msg, app, engine, logger, require_api_key
from .gesica import (
    SCENARIO_LIVING_REVIEW_IDS,
    _gesica_title,
    _get_db_gesica_scenario_or_404,
    _list_db_gesica_scenarios,
)

# ─── Living Review Endpoints ──────────────────────────────────────────────────

@app.get("/living-review/status")
def living_review_status(lang: str | None = Query(None)):
    """Retourne le statut de la dernière exécution de la living review."""
    import json as _json
    report_path = Path("/opt/literev-api/living_review_last_run.json")
    if not report_path.exists():
        report_path = Path(__file__).parent / "living_review_last_run.json"
    if report_path.exists():
        try:
            return _json.loads(report_path.read_text())
        except Exception:
            pass
    return {
        "status": "no_run_yet",
        "message": _msg(lang, "Aucune living review n'a encore été exécutée.", "No living review has run yet."),
        "command": "python3 living_review_scheduler.py --all-scenarios",
        "scenarios_available": list(SCENARIO_LIVING_REVIEW_IDS),
    }


@app.post("/living-review/run")
def living_review_run(scenario_id: str = "all", days: int = 30, dry_run: bool = False, _: None = Depends(require_api_key)):
    """Lance la living review pour un scénario ou tous les scénarios (processus async)."""
    import subprocess as _subprocess
    import sys as _sys
    script = str(Path(__file__).parent / "living_review_scheduler.py")
    cmd = [_sys.executable, script, "--mode", "once", "--days", str(days)]
    if scenario_id == "all":
        cmd.append("--all-scenarios")
    else:
        cmd.extend(["--scenario", scenario_id])
    if dry_run:
        cmd.append("--dry-run")
    try:
        # stdout/stderr → DEVNULL (PAS de PIPE) : personne ne draine ces tuyaux, donc
        # dès que le buffer OS (~64 Ko) se remplissait, le scénario enfant se bloquait
        # sur write() pour toujours (living_review_last_run.json jamais écrit, statut
        # figé). Le scheduler journalise son résultat dans son propre fichier de statut.
        proc = _subprocess.Popen(cmd, stdout=_subprocess.DEVNULL, stderr=_subprocess.DEVNULL,
                                 stdin=_subprocess.DEVNULL, start_new_session=True)
        return {
            "status": "started",
            "pid": proc.pid,
            "scenario": scenario_id,
            "days": days,
            "dry_run": dry_run,
            "message": _msg(lang, "Living review lancée. Consultez /living-review/status pour le résultat.",
                            "Living review started. See /living-review/status for the result."),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ─── PIPELINE LIVING REVIEW AUTOMATISÉ ───────────────────────────────────────

@app.post("/gesica/living-review/trigger")
def trigger_living_review(
    scenario_id: str | None = None,
    dry_run: bool = True,
    lang: str | None = Query(None),
    _: None = Depends(require_api_key),
) -> dict[str, Any]:
    """
    Déclenche le pipeline Living Review :
    1. Interroge PubMed avec la requête booléenne du scénario
    2. Insère les nouveaux articles
    3. Génère les embeddings
    4. Invalide le cache clustering
    Retourne un rapport de ce qui a été fait (ou ce qui serait fait en dry_run).
    """
    import threading

    scenarios_to_update = []
    with engine.connect() as conn:
        if scenario_id:
            meta = _get_db_gesica_scenario_or_404(scenario_id, conn)
            scenarios_to_update = [(scenario_id, meta)]
        else:
            scenarios_to_update = [(row["id"], row) for row in _list_db_gesica_scenarios(conn)]

    report = {
        "dry_run": dry_run,
        "triggered_at": __import__("datetime").datetime.now().isoformat(),
        "scenarios": [],
        "status": "triggered" if not dry_run else "dry_run",
    }

    for sid, smeta in scenarios_to_update:
        scenario_report = {
            "scenario_id": sid,
            "title": _gesica_title(smeta),
            "query": ((smeta.get("boolean_queries") or ["N/A"])[0] if isinstance(smeta.get("boolean_queries"), list) else (smeta.get("boolean_queries") or smeta.get("query") or "N/A")),
            "action": "would_fetch" if dry_run else "fetching",
        }
        report["scenarios"].append(scenario_report)

    if not dry_run:
        def _run_living_review():
            try:
                import subprocess, sys as _sys
                _script = str(Path(__file__).parent / "living_review_scheduler.py")
                # Cible le scénario demandé : living_review_scheduler.py accepte
                # --scenario / --all-scenarios ; ingest_pubmed.py n'accepte que
                # --project/--query (--all-scenarios y était un flag INVALIDE →
                # argparse échouait → la « vraie » exécution ne faisait rien).
                _cmd = [_sys.executable, _script, "--mode", "once"]
                if scenario_id:
                    _cmd += ["--scenario", scenario_id]
                else:
                    _cmd.append("--all-scenarios")
                result = subprocess.run(
                    _cmd, capture_output=True, text=True, timeout=600,
                    cwd=str(Path(__file__).parent),
                )
                logger.info(f"Living Review pipeline: {result.stdout[:500]}")
                if result.returncode != 0:
                    logger.error(f"Living Review error: {result.stderr[:500]}")
            except Exception as e:
                logger.error(f"Living Review pipeline error: {e}")

        threading.Thread(target=_run_living_review, daemon=True).start()
        report["message"] = "Pipeline Living Review déclenché en arrière-plan. Vérifiez les logs dans 5-10 minutes."
    else:
        report["message"] = f"Dry run : {len(scenarios_to_update)} scénario(s) seraient mis à jour."

    return report
