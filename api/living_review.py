"""Living review status and triggers.

Extracted from main.py (LiteRev API); `main` re-exports everything for the scripts,
tools and tests.
"""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, HTTPException, Query

from .core import _msg, app, engine, logger, require_api_key
from .scenario_store import _get_user_scenario_or_404
from .gesica import (
    SCENARIO_LIVING_REVIEW_IDS,
    _gesica_title,
    _get_db_gesica_scenario_or_404,
    _list_db_gesica_scenarios,
)

# ─── Living Review Endpoints ──────────────────────────────────────────────────

#: Le chemin d'écriture d'avant LIVING_REVIEW_REPORT_DIR, relu s'il porte un rapport.
_LEGACY_REPORT = Path("/opt/literev-api/living_review_last_run.json")


def _scheduler():
    """Le module du planificateur, qui sait où il écrit son état et son rapport."""
    import living_review_scheduler as _lrs
    return _lrs


def _read_report() -> dict | None:
    """Le rapport de la dernière exécution, là où le planificateur l'écrit."""
    import json as _json
    paths = []
    try:
        lrs = _scheduler()
        paths.append(lrs.report_dir() / lrs.REPORT_FILE)
    except Exception as _e:                                  # noqa: BLE001 - jamais bloquant
        logger.warning(f"living review report dir: {_e}")
    paths.append(_LEGACY_REPORT)
    for _p in paths:
        try:
            if _p.exists():
                data = _json.loads(_p.read_text())
                if isinstance(data, dict):
                    return data
        except Exception:                                    # noqa: BLE001
            continue
    return None


# L'exécution lancée par CE processus : son Popen, pour savoir sans attendre qu'elle vit
# encore entre son lancement et le moment où elle écrit son propre état.
_RUN_LOCK = threading.Lock()
_LAUNCHED: dict[str, Any] = {}


def _current_run() -> dict | None:
    """L'exécution en cours, ou la dernière, telle que le planificateur l'a écrite.

    « running » dont le processus a disparu (redémarrage de l'API, manque de mémoire) se
    lit « interrupted » : le planificateur n'a pas pu écrire sa fin, et l'annoncer encore
    en cours serait faux."""
    try:
        lrs = _scheduler()
        state = lrs.read_run_state()
    except Exception as _e:                                  # noqa: BLE001
        logger.warning(f"living review run state: {_e}")
        return None
    proc = _LAUNCHED.get("proc")
    ours_alive = proc is not None and proc.poll() is None
    if ours_alive and (not state or state.get("pid") != proc.pid):
        # Lancée à l'instant : l'enfant n'a pas encore écrit son état.
        return {k: v for k, v in _LAUNCHED.items() if k != "proc"} | {
            "state": "starting", "pid": proc.pid}
    if state and state.get("state") in ("running", "starting"):
        alive = ((ours_alive and state.get("pid") == proc.pid)
                 or lrs.scheduler_alive(state.get("pid")))
        if not alive:
            state = {**state, "state": "interrupted"}
    if state and proc is not None and state.get("pid") == proc.pid and not state.get("log"):
        state = {**state, "log": _LAUNCHED.get("log")}      # le journal que l'API lui a ouvert
    return state


@app.get("/living-review/status")
def living_review_status(lang: str | None = Query(None)):
    """Retourne le statut de la dernière exécution de la living review, et sous `run`
    l'état de l'exécution en cours ou de la dernière (running, done, error, interrupted)."""
    run = _current_run()
    report = _read_report()
    if report is not None:
        return {**report, "run": run}
    running = bool(run and run.get("state") in ("starting", "running"))
    return {
        "status": "running" if running else "no_run_yet",
        "message": (_msg(lang, "Une living review est en cours ; son rapport s'écrira à la fin.",
                         "A living review is running; its report is written when it ends.")
                    if running else
                    _msg(lang, "Aucune living review n'a encore été exécutée.", "No living review has run yet.")),
        "command": "python3 living_review_scheduler.py --all-scenarios",
        # Ce que la veille peut RÉELLEMENT faire, pas une seconde liste tenue à la main.
        # `SCENARIO_LIVING_REVIEW_IDS` en annonçait 27 ; le planificateur n'a de requêtes
        # que pour 6 d'entre eux, et les 21 autres répondaient « scénario inconnu ».
        "scenarios_available": _scenarios_the_scheduler_can_run(),
        "scenarios_listed_but_unsupported": sorted(
            set(SCENARIO_LIVING_REVIEW_IDS) - set(_scenarios_the_scheduler_can_run())),
        "run": run,
    }


def _scenarios_the_scheduler_can_run() -> list[str]:
    """Les scénarios pour lesquels le PLANIFICATEUR a des requêtes.

    Le statut annonçait `SCENARIO_LIVING_REVIEW_IDS`, une liste tenue à la main dans
    api/gesica.py, qui comptait 27 identifiants là où `living_review_scheduler.
    SCENARIO_QUERIES` en couvre 6 : les 21 autres étaient proposés à l'interface et
    répondaient « scénario inconnu » au lancement. La capacité est lue à sa source."""
    try:
        import living_review_scheduler as _lrs
        return sorted(_lrs.SCENARIO_QUERIES)
    except Exception as _e:                                  # noqa: BLE001 - jamais bloquant
        logger.warning(f"living review capability: {_e}")
        return []


def _watch_run(proc, base: dict) -> None:
    """Attend la fin de l'exécution, SANS délai, et écrit sa fin si elle ne l'a pas fait.

    Le déclencheur attendait le planificateur 600 secondes puis le tuait : une veille plus
    longue (toutes les notices de la fenêtre, et non plus les 200 plus récentes) mourait
    sans rapport, et l'échec n'apparaissait que dans les journaux. L'exécution est
    détachée ; ce fil ne fait que la ramasser. Un enfant mort avant d'écrire son état
    (import en échec, tué) aurait laissé « running » : on écrit son code de sortie."""
    rc = proc.wait()
    try:
        lrs = _scheduler()
        state = lrs.read_run_state() or {}
        if state.get("pid") != proc.pid or state.get("state") in ("running", "starting"):
            lrs.write_run_state({**base, "pid": proc.pid,
                                 "state": "error" if rc else "done",
                                 "returncode": rc,
                                 "finished_at": datetime.now(timezone.utc).isoformat(),
                                 "error": (f"le planificateur s'est arrêté (code {rc}) sans écrire "
                                           f"son état ; voir {base.get('log')}" if rc else None)})
    except Exception as _e:                                  # noqa: BLE001
        logger.warning(f"living review watch: {_e}")
    (logger.error if rc else logger.info)(f"Living Review terminée, code {rc} ({base.get('scenario')})")


def _launch_scheduler(args: list[str], *, scenario: str, days: int | None, dry_run: bool,
                      lang: str | None) -> dict[str, Any]:
    """Lance le planificateur DÉTACHÉ et rend la main : son état se lit sur
    /living-review/status. Refuse une seconde exécution tant que la première vit, qu'elle
    vienne de l'API ou du cron : deux veilles en parallèle écriraient le même état.

    `days` n'est transmis que s'il est donné : sans lui, chaque préréglage garde sa propre
    fenêtre (30 ou 60 jours) ; avec lui, il s'applique à tous."""
    import subprocess as _subprocess
    import sys as _sys
    # Racine du dépôt, pas api/ : ce module a été extrait de main.py et le chemin l'a
    # suivi, si bien que l'enfant mourait sur « can't open file ».
    script = Path(__file__).resolve().parent.parent / "living_review_scheduler.py"
    if not script.exists():
        return {"status": "error",
                "error": f"Scheduler introuvable : {script}",
                "message": _msg(lang, "Le script de living review est introuvable sur le serveur.",
                                "The living review script is missing on the server.")}
    with _RUN_LOCK:
        current = _current_run()
        if current and current.get("state") in ("starting", "running"):
            return {"status": "already_running", "run": current,
                    "message": _msg(lang, "Une living review est déjà en cours : son état se lit sur "
                                          "/living-review/status.",
                                    "A living review is already running: see /living-review/status.")}
        cmd = [_sys.executable, str(script), "--mode", "once", *args]
        if days is not None:
            cmd += ["--days", str(int(days))]
        if dry_run:
            cmd.append("--dry-run")
        try:
            lrs = _scheduler()
            log_path = lrs.report_dir() / "living_review_last_run.log"
            # La sortie dans un FICHIER, pas dans un tuyau : personne ne draine un PIPE, et
            # dès que le tampon (~64 Ko) se remplissait l'enfant se bloquait sur write().
            # DEVNULL, choisi ensuite pour cette raison, ne laissait aucune trace d'un échec.
            # Le journal de la DERNIÈRE exécution, comme le rapport : il ne grossit pas.
            try:
                log = open(log_path, "wb")
            except OSError:
                log, log_path = _subprocess.DEVNULL, None
            try:
                proc = _subprocess.Popen(cmd, stdout=log, stderr=_subprocess.STDOUT,
                                         stdin=_subprocess.DEVNULL, start_new_session=True,
                                         cwd=str(script.parent))
            finally:
                if log is not _subprocess.DEVNULL:
                    log.close()
        except Exception as e:                               # noqa: BLE001
            return {"status": "error", "error": str(e)}
        base = {"scenario": scenario, "days": days, "dry_run": dry_run,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "log": str(log_path) if log_path else None}
        _LAUNCHED.clear()
        _LAUNCHED.update(base, proc=proc)
    threading.Thread(target=_watch_run, args=(proc, base), daemon=True,
                     name="living-review-watch").start()
    return {
        "status": "started",
        "pid": proc.pid,
        "scenario": scenario,
        "days": days,
        "dry_run": dry_run,
        "message": _msg(lang, "Living review lancée. Consultez /living-review/status pour son état et son résultat.",
                        "Living review started. See /living-review/status for its state and result."),
    }


@app.post("/living-review/run")
def living_review_run(scenario_id: str = "all", days: int | None = Query(None, ge=1, le=3650),
                      dry_run: bool = False,
                      lang: str | None = Query(None), _: None = Depends(require_api_key)):
    """Lance la living review pour un scénario (préréglé ou utilisateur) ou tous les
    scénarios préréglés, détachée : l'état se lit sur /living-review/status.

    `days` absent : chaque préréglage garde sa fenêtre. Le défaut était 30, transmis à un
    planificateur qui le remplaçait aussitôt par celle du préréglage : le paramètre
    n'avait aucun effet. Il s'applique maintenant à tous les scénarios lancés.

    Deux défauts corrigés avant : le script était cherché dans `api/` (il est à la racine
    du dépôt, ce module ayant été extrait de main.py), donc l'enfant mourait aussitôt sur
    « can't open file » ; et `lang` n'était pas un paramètre, donc la construction de la
    réponse levait un NameError avalé par le `except` : l'endpoint renvoyait toujours
    `error`, jamais `started`, et rien n'était jamais récupéré."""
    args = ["--all-scenarios"] if scenario_id == "all" else ["--scenario", scenario_id]
    return _launch_scheduler(args, scenario=scenario_id, days=days, dry_run=dry_run, lang=lang)


# ─── PIPELINE LIVING REVIEW AUTOMATISÉ ───────────────────────────────────────

@app.post("/gesica/living-review/trigger")
def trigger_living_review(
    scenario_id: str | None = None,
    dry_run: bool = True,
    days: int | None = Query(None, ge=1, le=3650),
    lang: str | None = Query(None),
    _: None = Depends(require_api_key),
) -> dict[str, Any]:
    """
    Déclenche le pipeline Living Review :
    1. Interroge PubMed (et, pour un préréglage, les préprints via Europe PMC) avec les
       requêtes du scénario, sur toute sa fenêtre
    2. Ingère les articles dans la bibliothèque et les lie au scénario
    3. Invalide les caches de visualisation des scénarios dont le corpus a gagné un lien
       (clustering, graphe de similarité, carte des concepts) : c'est le planificateur qui
       le fait, là où il crée les liens, pour qu'une veille lancée par cron le fasse aussi
    Les EMBEDDINGS ne sont pas produits ici : le worker d'arrière-plan les calcule ensuite
    (cf. schema_boot), la docstring annonçait à tort une étape synchrone.
    L'exécution est détachée : son état et son rapport se lisent sur /living-review/status.
    Retourne ce qui est lancé (ou ce qui le serait, en dry_run).
    """
    # ── Un scénario UTILISATEUR est un scénario ──────────────────────────────
    # La résolution passait par `_get_db_gesica_scenario_or_404`, qui exige
    # `is_system = TRUE` ET le dossier GESICA : tout scénario réel recevait donc 404, et
    # le bouton « Simuler (à blanc) » du panneau ne pouvait fonctionner pour aucun d'eux.
    scenarios_to_update = []
    with engine.connect() as conn:
        if scenario_id:
            try:
                meta = _get_db_gesica_scenario_or_404(scenario_id, conn)
            except HTTPException:
                meta = _get_user_scenario_or_404(scenario_id)
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
        # `--scenario` accepte aussi un scénario utilisateur, et ne traite que lui :
        # `--user-scenarios`, passé jusqu'ici, traitait TOUS les scénarios possédés, et
        # pas celui qu'on demandait s'il n'avait pas de propriétaire.
        launched = _launch_scheduler(
            ["--scenario", scenario_id] if scenario_id else ["--all-scenarios"],
            scenario=scenario_id or "all", days=days, dry_run=False, lang=lang)
        if launched["status"] != "started":
            report["status"] = launched["status"]
        report["run"] = launched.get("run") or {k: launched.get(k) for k in ("pid", "scenario", "days")}
        report["message"] = launched.get("message") or launched.get("error") or ""
    else:
        report["message"] = f"Dry run : {len(scenarios_to_update)} scénario(s) seraient mis à jour."

    return report
