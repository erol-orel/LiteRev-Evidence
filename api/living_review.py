"""Living review status and triggers.

Extracted from main.py (LiteRev API); `main` re-exports everything for the scripts,
tools and tests.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import Depends, Query
from sqlalchemy import text

from .core import _msg, app, engine, logger, require_api_key
from .scenario_store import CORPUS_DERIVED_CACHE_RESET
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
def living_review_run(scenario_id: str = "all", days: int = 30, dry_run: bool = False,
                      lang: str | None = Query(None), _: None = Depends(require_api_key)):
    """Lance la living review pour un scénario ou tous les scénarios (processus async).

    Deux défauts corrigés ici : le script était cherché dans `api/` (il est à la racine
    du dépôt, ce module ayant été extrait de main.py), donc l'enfant mourait aussitôt sur
    « can't open file » ; et `lang` n'était pas un paramètre, donc la construction de la
    réponse levait un NameError avalé par le `except` : l'endpoint renvoyait toujours
    `error`, jamais `started`, et rien n'était jamais récupéré."""
    import subprocess as _subprocess
    import sys as _sys
    script = str(Path(__file__).resolve().parent.parent / "living_review_scheduler.py")
    if not Path(script).exists():
        return {"status": "error",
                "error": f"Scheduler introuvable : {script}",
                "message": _msg(lang, "Le script de living review est introuvable sur le serveur.",
                                "The living review script is missing on the server.")}
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
    2. Insère les nouveaux articles et leurs chunks
    3. Invalide les caches de visualisation des scénarios rafraîchis (clustering, graphe
       de similarité, carte des concepts) : le corpus a changé, ils sont périmés
    Les EMBEDDINGS ne sont pas produits ici : le worker d'arrière-plan les calcule ensuite
    (cf. schema_boot), la docstring annonçait à tort une étape synchrone.
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
        _sids = [sid for sid, _ in scenarios_to_update]

        def _run_living_review():
            try:
                import subprocess, sys as _sys
                # Racine du dépôt, pas api/ : ce module a été extrait de main.py et le
                # chemin l'a suivi, si bien que l'enfant mourait sur « can't open file ».
                _script = str(Path(__file__).resolve().parent.parent / "living_review_scheduler.py")
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
                    cwd=str(Path(__file__).resolve().parent.parent),
                )
                logger.info(f"Living Review pipeline: {result.stdout[:500]}")
                if result.returncode != 0:
                    logger.error(f"Living Review error: {result.stderr[:500]}")
                elif _sids:
                    # Le corpus a gagné des articles : tout ce qui en est calculé ne le
                    # décrit plus. L'étape 4 annoncée par la docstring n'existait nulle
                    # part dans le scheduler ; elle est faite ici, où l'on sait quels
                    # scénarios ont été rafraîchis. La liste est celle du changement de
                    # seuil (CORPUS_DERIVED_CACHE_RESET) : les deux se sont déjà désaccordées
                    # une fois, sur les actions recommandées.
                    try:
                        with engine.begin() as _c:
                            _c.execute(text(f"""
                                UPDATE scenario_settings SET {CORPUS_DERIVED_CACHE_RESET}
                                WHERE scenario_id = ANY(:sids)
                            """), {"sids": _sids})
                        logger.info(f"Living Review: caches dérivés du corpus invalidés pour {_sids}")
                    except Exception as _ce:
                        logger.warning(f"Living Review cache invalidation: {_ce}")
            except Exception as e:
                logger.error(f"Living Review pipeline error: {e}")

        threading.Thread(target=_run_living_review, daemon=True).start()
        report["message"] = "Pipeline Living Review déclenché en arrière-plan. Vérifiez les logs dans 5-10 minutes."
    else:
        report["message"] = f"Dry run : {len(scenarios_to_update)} scénario(s) seraient mis à jour."

    return report
