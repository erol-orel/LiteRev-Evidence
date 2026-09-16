"""Model training, comparison, prediction, export and monitoring.

Extracted from main.py (LiteRev API); `main` re-exports everything for the scripts,
tools and tests.
"""
from __future__ import annotations

import json
import threading
from typing import Any

from fastapi import Depends, HTTPException, Query
from sqlalchemy import text

from .core import _msg, app, engine, logger, require_api_key
from .variables import _TS_ALGO_FAMILIES
from .model_data import (
    MODEL_DATA_DIR,
    _dataframe_dtype_kinds,
    _get_model_spec,
    _validate_dataset_against_template,
)

# ─── MODEL TRAINING (Phase 3) : sklearn + Optuna sur le dataset branché ───────
# Entraîne un vrai modèle à partir du model_spec (Phase 1) et du dataset
# uploadé (Phase 2) : préprocessing, HPO par validation croisée (Optuna),
# holdout, importances. Le pipeline entraîné est sérialisé (joblib) et sert les
# prédictions. Remplace les formules mock par un modèle réellement appris.

_MODEL_TRAIN_JOBS: dict[str, dict] = {}
_model_train_lock = threading.Lock()


def _claim_model_train(scenario_id: str) -> bool:
    """Réserve ATOMIQUEMENT le job d'entraînement d'un scénario. True si réservé (aucun
    entraînement en cours), False si un est déjà en cours. Ferme le TOCTOU où deux
    requêtes quasi simultanées lisaient toutes deux "non en cours" et lançaient deux
    entraînements concurrents (calcul redondant + un artefact écrasait l'autre)."""
    with _model_train_lock:
        if _MODEL_TRAIN_JOBS.get(scenario_id, {}).get("status") == "running":
            return False
        _MODEL_TRAIN_JOBS[scenario_id] = {"status": "running"}
        return True


def _ensure_model_run_table():
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS scenario_model_run (
                id               BIGSERIAL PRIMARY KEY,
                scenario_id      VARCHAR(80) NOT NULL,
                dataset_id       BIGINT,
                status           VARCHAR(20) DEFAULT 'running',
                family           TEXT,
                task_type        TEXT,
                metric           TEXT,
                metrics_json     JSONB,
                best_params_json JSONB,
                feature_importance_json JSONB,
                summary_json     JSONB,
                artifact_path    TEXT,
                error            TEXT,
                is_active        BOOLEAN DEFAULT FALSE,
                created_at       TIMESTAMP DEFAULT NOW()
            )
        """))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_model_run_scenario_active "
            "ON scenario_model_run (scenario_id, is_active)"
        ))
    logger.info("Table scenario_model_run vérifiée/créée.")


try:
    _ensure_model_run_table()
except Exception as _e:
    logger.warning(f"_ensure_model_run_table: {_e}")


def _seed_demo_scenarios() -> None:
    """Seed a built-in, model-ready DEMO scenario - a real influenza dataset + a
    freshly trained model - so the real-dataset trial is visible in the scenario
    list instead of living only as a script + committed files. Idempotent (guarded
    by a stable id) and strictly best-effort: any failure is logged and swallowed,
    the server boots regardless. Runs in a startup daemon thread.

    The four rows the dashboard/monitor need (user_scenarios + scenario_settings
    with a model_spec + an active dataset CSV + an active run with a joblib artifact)
    are written to mirror the app's own upload/train inserts. The scenario is pinned
    (so the recent-search dedup never purges it) and left non-system."""
    import os
    import joblib
    from datetime import datetime, timezone
    try:
        import demo_seed
    except Exception as _e:
        logger.warning(f"seed demo: import demo_seed failed: {_e}")
        return
    sid = demo_seed.DEMO_SCENARIO_ID
    try:
        with engine.connect() as conn:
            if conn.execute(text("SELECT 1 FROM user_scenarios WHERE id = :id"), {"id": sid}).first():
                return  # already seeded → idempotent no-op
        # Racine du DÉPÔT : ce module a été extrait de main.py, et le chemin est resté
        # relatif à api/. demo_seed cherchait scripts/trial_output/ sous api/, qui n'a pas
        # de sous-dossier scripts : le scénario de démonstration ne pouvait jamais être
        # semé, et c'est aussi lui qui produisait les seuls runs 'ready'.
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if not os.path.exists(demo_seed.dataset_path(repo_root)):
            logger.warning(f"seed demo: dataset missing at {demo_seed.dataset_path(repo_root)}")
            return
        # Train the real model FIRST (fresh, sklearn-version-matched); abort before any
        # DB write if it fails, so we never leave a modelless demo card.
        df, spec, result = demo_seed.train_demo(repo_root, n_trials=20)
        pipeline = result.pop("pipeline", None)
        if pipeline is None:
            logger.warning("seed demo: training produced no pipeline")
            return
        # Persist the dataset CSV + joblib artifact under MODEL_DATA_DIR (as upload/train do).
        ddir = MODEL_DATA_DIR / sid / "model"
        ddir.mkdir(parents=True, exist_ok=True)
        ts = int(datetime.now(timezone.utc).timestamp())
        stored_path = str(ddir / f"{ts}_influenza_ch.csv")
        df.to_csv(stored_path, index=False)
        artifact_path = str(ddir / f"artifact_{ts}.joblib")
        joblib.dump(pipeline, artifact_path)
        try:
            report = _validate_dataset_against_template(
                list(df.columns), spec["data_template"], _dataframe_dtype_kinds(df))
        except Exception:
            report = {"ok": True, "note": "demo seed"}
        summary = {k: v for k, v in result.items()
                   if k not in ("metrics", "best_params", "feature_importances")}
        with engine.begin() as conn:
            if conn.execute(text("SELECT 1 FROM user_scenarios WHERE id = :id"), {"id": sid}).first():
                return  # race: seeded by another worker between the checks
            conn.execute(text("""
                INSERT INTO user_scenarios
                    (id, name, query, mode, filters, result_count, pinned,
                     pipeline_status, pipeline_step, pipeline_progress)
                VALUES (:id, :name, :query, 'hybrid', CAST('{}' AS jsonb), 0, TRUE,
                     'done', 'done', 100)
                ON CONFLICT (id) DO NOTHING
            """), {"id": sid, "name": demo_seed.DEMO_SCENARIO_NAME, "query": demo_seed.DEMO_SCENARIO_QUERY})
            conn.execute(text("""
                INSERT INTO scenario_settings (scenario_id, variables_json, variables_validated, updated_at)
                VALUES (:sid, CAST(:vj AS jsonb), TRUE, NOW())
                ON CONFLICT (scenario_id) DO UPDATE
                    SET variables_json = CAST(:vj AS jsonb), variables_validated = TRUE, updated_at = NOW()
            """), {"sid": sid, "vj": json.dumps(demo_seed.demo_variables_json())})
            conn.execute(text(
                "UPDATE scenario_model_dataset SET is_active = FALSE WHERE scenario_id = :sid AND is_active = TRUE"
            ), {"sid": sid})
            did = conn.execute(text("""
                INSERT INTO scenario_model_dataset
                    (scenario_id, filename, stored_path, n_rows, n_cols, columns_json, validation_json, is_active, is_synthetic)
                VALUES (:sid, :fn, :sp, :nr, :nc, CAST(:cj AS jsonb), CAST(:vj AS jsonb), TRUE, FALSE)
                RETURNING id
            """), {"sid": sid, "fn": "influenza_ch_weekly.csv", "sp": stored_path,
                   "nr": int(len(df)), "nc": int(len(df.columns)),
                   "cj": json.dumps([str(c) for c in df.columns]),
                   "vj": json.dumps(report, default=str)}).scalar()
            conn.execute(text(
                "UPDATE scenario_model_run SET is_active = FALSE WHERE scenario_id = :sid AND is_active = TRUE"
            ), {"sid": sid})
            conn.execute(text("""
                INSERT INTO scenario_model_run
                    (scenario_id, dataset_id, status, family, task_type, metric,
                     metrics_json, best_params_json, feature_importance_json, summary_json,
                     artifact_path, is_active)
                VALUES (:sid, :did, 'ready', :fam, :tt, :met,
                     CAST(:mj AS jsonb), CAST(:bp AS jsonb), CAST(:fi AS jsonb), CAST(:sj AS jsonb),
                     :ap, TRUE)
            """), {"sid": sid, "did": did, "fam": result.get("family"), "tt": result.get("task_type"),
                   "met": result.get("metric"),
                   "mj": json.dumps(result.get("metrics", {}), default=str),
                   "bp": json.dumps(result.get("best_params", {}), default=str),
                   "fi": json.dumps(result.get("feature_importances", []), default=str),
                   "sj": json.dumps(summary, default=str), "ap": artifact_path})
        logger.info(f"seed demo scenario {sid} created (family={result.get('family')}, "
                    f"R2={(result.get('metrics') or {}).get('r2')})")
    except Exception as _e:
        logger.warning(f"seed demo scenario skipped: {_e}")


def _run_model_training(scenario_id: str, n_trials: int = 25, compare: bool = False) -> None:
    """Job d'entraînement (thread) : charge le dataset actif, entraîne, persiste.
    compare=True entraîne PLUSIEURS familles (lightgbm/xgboost/GB/RF/linéaire),
    persiste la MEILLEURE et joint le classement complet au résumé."""
    import json as _json
    import pandas as pd
    from datetime import datetime, timezone
    import model_trainer

    try:
        spec = _get_model_spec(scenario_id)
        if not spec:
            raise ValueError("Aucun model_spec (générez/validez les Variables & Modèle).")

        with engine.connect() as conn:
            ds = conn.execute(text("""
                SELECT id, stored_path, is_synthetic FROM scenario_model_dataset
                WHERE scenario_id = :sid AND is_active = TRUE
                ORDER BY created_at DESC LIMIT 1
            """), {"sid": scenario_id}).mappings().first()
        if not ds or not ds["stored_path"]:
            raise ValueError("Aucun dataset branché. Uploadez d'abord un CSV/XLSX.")

        df = pd.read_csv(ds["stored_path"])
        if compare:
            comparison = model_trainer.compare_models(df, spec, n_trials=n_trials)
            result = comparison.get("best")
            if not result:
                raise ValueError("Aucun modèle n'a pu être entraîné lors de la comparaison.")
            # Classement complet joint au résumé (affiché dans l'onglet Modèle).
            result["leaderboard"] = comparison["leaderboard"]
            result["families_compared"] = comparison["families_tried"]
            result["leaderboard_lower_is_better"] = comparison["lower_is_better"]
        else:
            result = model_trainer.train_model(df, spec, n_trials=n_trials)

        # Traçabilité : modèle entraîné sur des données SYNTHÉTIQUES (démo) vs réelles.
        result["is_synthetic"] = bool(ds.get("is_synthetic"))

        # Sérialiser le pipeline entraîné.
        pipeline = result.pop("pipeline")
        artifact_path = None
        try:
            import joblib
            adir = MODEL_DATA_DIR / scenario_id / "model"
            adir.mkdir(parents=True, exist_ok=True)
            artifact_path = str(adir / f"artifact_{int(datetime.now(timezone.utc).timestamp())}.joblib")
            joblib.dump(pipeline, artifact_path)
        except Exception as e:
            logger.error(f"Sérialisation artefact {scenario_id}: {e}", exc_info=True)
            artifact_path = None

        with engine.begin() as conn:
            conn.execute(text(
                "UPDATE scenario_model_run SET is_active = FALSE WHERE scenario_id = :sid AND is_active = TRUE"
            ), {"sid": scenario_id})
            run_id = conn.execute(text("""
                INSERT INTO scenario_model_run
                    (scenario_id, dataset_id, status, family, task_type, metric,
                     metrics_json, best_params_json, feature_importance_json, summary_json,
                     artifact_path, is_active)
                VALUES (:sid, :did, 'done', :fam, :tt, :met,
                     CAST(:mj AS jsonb), CAST(:bp AS jsonb), CAST(:fi AS jsonb), CAST(:sj AS jsonb),
                     :ap, TRUE)
                RETURNING id
            """), {
                "sid": scenario_id, "did": ds["id"], "fam": result["family"],
                "tt": result["task_type"], "met": result["metric"],
                "mj": _json.dumps(result["metrics"]),
                "bp": _json.dumps(result["best_params"]),
                "fi": _json.dumps(result["feature_importances"]),
                "sj": _json.dumps({k: v for k, v in result.items()
                                   if k not in ("metrics", "best_params", "feature_importances")}),
                "ap": artifact_path,
            }).scalar()

        _MODEL_TRAIN_JOBS[scenario_id] = {
            "status": "done", "run_id": run_id, "metrics": result["metrics"],
            "family": result["family"], "task_type": result["task_type"],
        }
        logger.info(f"Modèle entraîné {scenario_id}: run {run_id}, {result['metrics']}")
    except Exception as e:
        logger.error(f"Entraînement modèle {scenario_id}: {e}", exc_info=True)
        _MODEL_TRAIN_JOBS[scenario_id] = {"status": "error", "error": str(e)}
        try:
            with engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO scenario_model_run (scenario_id, status, error, is_active)
                    VALUES (:sid, 'error', :err, FALSE)
                """), {"sid": scenario_id, "err": str(e)[:2000]})
        except Exception:
            pass


@app.post("/scenarios/{scenario_id}/model/train")
def train_scenario_model(scenario_id: str, n_trials: int = 25,
                         _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Lance l'entraînement réel (async) du modèle sur le dataset branché."""
    import threading

    if _MODEL_TRAIN_JOBS.get(scenario_id, {}).get("status") == "running":
        return {"status": "already_running", "scenario_id": scenario_id}

    if not _get_model_spec(scenario_id):
        raise HTTPException(status_code=400, detail="Aucun model_spec. Générez puis validez les Variables & Modèle.")
    with engine.connect() as conn:
        ds = conn.execute(text(
            "SELECT id FROM scenario_model_dataset WHERE scenario_id = :sid AND is_active = TRUE LIMIT 1"
        ), {"sid": scenario_id}).first()
    if not ds:
        raise HTTPException(status_code=400, detail="Aucun dataset branché. Uploadez un CSV/XLSX d'abord.")

    n_trials = max(5, min(int(n_trials or 25), 100))
    if not _claim_model_train(scenario_id):
        return {"status": "already_running", "scenario_id": scenario_id}
    threading.Thread(target=_run_model_training, args=(scenario_id, n_trials), daemon=True).start()
    return {"status": "started", "scenario_id": scenario_id, "n_trials": n_trials}


@app.post("/scenarios/{scenario_id}/model/compare")
def compare_scenario_models(scenario_id: str, n_trials: int = 15,
                            _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Entraîne PLUSIEURS familles (lightgbm/xgboost/GB/RF/linéaire) sur le dataset
    branché, garde la meilleure comme modèle actif et joint le classement complet.
    Partage le registre de jobs et le endpoint de statut avec /model/train."""
    import threading

    if _MODEL_TRAIN_JOBS.get(scenario_id, {}).get("status") == "running":
        return {"status": "already_running", "scenario_id": scenario_id}

    if not _get_model_spec(scenario_id):
        raise HTTPException(status_code=400, detail="Aucun model_spec. Générez puis validez les Variables & Modèle.")
    with engine.connect() as conn:
        ds = conn.execute(text(
            "SELECT id FROM scenario_model_dataset WHERE scenario_id = :sid AND is_active = TRUE LIMIT 1"
        ), {"sid": scenario_id}).first()
    if not ds:
        raise HTTPException(status_code=400, detail="Aucun dataset branché. Uploadez un CSV/XLSX d'abord.")

    # La comparaison entraîne N familles → on limite le budget d'essais par famille
    # (défaut 15) pour garder un temps de calcul raisonnable.
    n_trials = max(5, min(int(n_trials or 15), 50))
    if not _claim_model_train(scenario_id):
        return {"status": "already_running", "scenario_id": scenario_id}
    threading.Thread(
        target=_run_model_training, args=(scenario_id, n_trials), kwargs={"compare": True}, daemon=True
    ).start()
    return {"status": "started", "scenario_id": scenario_id, "n_trials": n_trials, "mode": "compare"}


@app.get("/scenarios/{scenario_id}/model/train/status")
def get_model_train_status(scenario_id: str) -> dict[str, Any]:
    """Statut du job d'entraînement."""
    return _MODEL_TRAIN_JOBS.get(scenario_id, {"status": "idle"})


@app.get("/scenarios/{scenario_id}/model/run")
def get_model_run(scenario_id: str) -> dict[str, Any]:
    """Dernier modèle entraîné actif : métriques, hyperparamètres, importances."""
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT id, dataset_id, status, family, task_type, metric, metrics_json,
                   best_params_json, feature_importance_json, summary_json, error,
                   (artifact_path IS NOT NULL) AS has_artifact, created_at
            FROM scenario_model_run
            WHERE scenario_id = :sid AND is_active = TRUE
            ORDER BY created_at DESC LIMIT 1
        """), {"sid": scenario_id}).mappings().first()

    if not row:
        return {"status": "empty", "message": "Aucun modèle entraîné. Lancez l'entraînement après avoir branché des données."}

    return {
        # « ready » était renvoyé en dur, y compris pour un run dont la sérialisation du
        # modèle avait échoué (artifact_path NULL) : l'onglet Modèle annonçait un modèle
        # prêt pendant que la prédiction répondait 400 et le monitoring « unavailable ».
        "status": "ready" if row["has_artifact"] else "no_artifact",
        "usable": bool(row["has_artifact"]),
        "message": (None if row["has_artifact"] else
                    "Entraînement terminé mais le modèle n'a pas pu être enregistré : "
                    "prédiction et monitoring indisponibles. Relancez l'entraînement."),
        "run_id": row["id"],
        "dataset_id": row["dataset_id"],
        "family": row["family"],
        "task_type": row["task_type"],
        "metric": row["metric"],
        "metrics": row["metrics_json"],
        "best_params": row["best_params_json"],
        "feature_importances": row["feature_importance_json"],
        "summary": row["summary_json"],
        "has_artifact": row["has_artifact"],
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
    }


@app.post("/scenarios/{scenario_id}/model/predict")
def predict_scenario_model(scenario_id: str, payload: dict[str, Any],
                           _: None = Depends(require_api_key)) -> dict[str, Any]:
    """
    Prédit avec le modèle entraîné actif. Body: {"rows": [{feature: value, ...}, ...]}.
    Renvoie les prédictions (et probabilités si classification).
    """
    import pandas as pd

    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise HTTPException(status_code=400, detail="Body attendu: {\"rows\": [ {feature: value, ...} ]}")

    with engine.connect() as conn:
        run = conn.execute(text("""
            SELECT artifact_path, task_type, summary_json FROM scenario_model_run
            WHERE scenario_id = :sid AND is_active = TRUE AND artifact_path IS NOT NULL
            ORDER BY created_at DESC LIMIT 1
        """), {"sid": scenario_id}).mappings().first()
    if not run:
        raise HTTPException(status_code=400, detail="Aucun modèle entraîné disponible. Entraînez d'abord le modèle.")

    # Modèle de prévision : pas de prédiction ligne-à-ligne (la sortie est la
    # prévision temporelle, disponible dans l'onglet Modèle / le monitoring).
    if run["task_type"] == "forecast":
        raise HTTPException(status_code=400,
                            detail="Modèle de prévision : la prédiction se fait sur l'horizon temporel, pas ligne par ligne. Voir la prévision dans l'onglet Modèle.")

    try:
        import joblib
        pipeline = joblib.load(run["artifact_path"])
    except Exception as e:
        logger.error(f"Model load failed: {e}", exc_info=True)   # détail journalisé serveur, pas renvoyé
        raise HTTPException(status_code=500, detail="Chargement du modèle impossible.")

    df = pd.DataFrame(rows)
    classes = (run["summary_json"] or {}).get("classes")
    try:
        preds = pipeline.predict(df)
        # Classification : reconvertir les entiers encodés vers les labels d'origine.
        if run["task_type"] == "classification" and classes:
            predictions = [classes[int(p)] if 0 <= int(p) < len(classes) else _jsonable(p) for p in preds]
        else:
            predictions = [_jsonable(p) for p in preds]
        out: dict[str, Any] = {"status": "ok", "predictions": predictions}
        if run["task_type"] == "classification" and hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(df)
            out["classes"] = classes
            out["probabilities"] = [[float(x) for x in r] for r in proba]
        # Explication LOCALE par prédiction (contribution de chaque variable) - sans
        # dépendance externe, via ablation vers le fond. Plafonnée pour rester rapide.
        summ = run["summary_json"] or {}
        background = summ.get("explain_background")
        if background and len(df) <= 50:
            import model_trainer as _mt
            used = [{"machine_name": k} for k in background]
            explanations = []
            for i in range(len(df)):
                try:
                    explanations.append(_mt.explain_prediction(
                        pipeline, df.iloc[[i]], used, run["task_type"], background))
                except Exception:
                    explanations.append(None)
            out["explanations"] = explanations
        return out
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prédiction impossible (colonnes manquantes ?) : {e}")


def _jsonable(v):
    """Convertit les scalaires numpy en types Python natifs."""
    try:
        import numpy as np
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
    except Exception:
        pass
    return v


# ─── MODEL MONITORING (Phase 4) : statut live piloté par le modèle entraîné ───
# Score les données récentes via le modèle réel (Phase 3) et en déduit un niveau
# d'alerte green/orange/red, avec les libellés des alert_thresholds du spec.

_DEFAULT_ALERT_LABELS = {
    "green": "Normal", "orange": "Vigilance", "red": "Alerte critique",
    "unavailable": "Données insuffisantes",
}
_DEFAULT_ALERT_LABELS_EN = {
    "green": "Normal", "orange": "Watch", "red": "Critical alert",
    "unavailable": "Insufficient data",
}


def _alert_labels(lang) -> dict[str, str]:
    """Libellés par défaut des niveaux d'alerte du moniteur, dans la langue demandée."""
    return _DEFAULT_ALERT_LABELS_EN if _msg(lang, "fr", "en") == "en" else _DEFAULT_ALERT_LABELS


@app.get("/scenarios/{scenario_id}/model/export")
def export_model_bundle(scenario_id: str, include_data: bool = True,
                        _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Bundle de REPRODUCTIBILITÉ d'un modèle (« pas une boîte noire ») : le model_spec
    (cible, variables, algorithme, data_template, provenance vers les articles), TOUS les
    runs entraînés avec leurs HYPERPARAMÈTRES / métriques / importances, le JEU DE DONNÉES
    (schéma + lignes) et la PRÉDICTION courante. Tout ce qu'il faut pour ré-entraîner et
    reproduire la prédiction. Authentifié (expose le schéma/les données du scénario)."""
    from datetime import datetime, timezone
    with engine.connect() as conn:
        srow = conn.execute(text(
            "SELECT name, query, mode FROM user_scenarios WHERE id = :sid"
        ), {"sid": scenario_id}).mappings().first()
        runs = conn.execute(text("""
            SELECT id, dataset_id, family, task_type, metric, metrics_json,
                   best_params_json, feature_importance_json, summary_json, is_active, created_at
            FROM scenario_model_run
            -- 'done' est le statut qu'écrit l'entraînement réel ; 'ready' n'était produit
            -- que par le seeder de démonstration. Le bundle de reproductibilité filtrait
            -- sur 'ready' et ressortait donc TOUJOURS vide : aucun run, aucun
            -- hyperparamètre, aucune métrique, pour un modèle affiché comme entraîné.
            WHERE scenario_id = :sid AND status IN ('done', 'ready')
            ORDER BY created_at DESC
        """), {"sid": scenario_id}).mappings().all()
        ds = conn.execute(text("""
            SELECT id, filename, stored_path, n_rows, n_cols, columns_json, validation_json,
                   is_synthetic, created_at
            FROM scenario_model_dataset WHERE scenario_id = :sid AND is_active = TRUE
            ORDER BY created_at DESC LIMIT 1
        """), {"sid": scenario_id}).mappings().first()

    runs_out = [{
        "run_id": r["id"], "dataset_id": r["dataset_id"], "family": r["family"],
        "task_type": r["task_type"], "metric": r["metric"], "metrics": r["metrics_json"],
        "hyperparameters": r["best_params_json"], "feature_importances": r["feature_importance_json"],
        "summary": r["summary_json"], "is_active": bool(r["is_active"]),
        "trained_at": r["created_at"].isoformat() if r["created_at"] else None,
    } for r in runs]

    dataset_out = None
    if ds:
        dataset_out = {
            "dataset_id": ds["id"], "filename": ds["filename"], "n_rows": ds["n_rows"],
            "n_cols": ds["n_cols"], "columns": ds["columns_json"], "validation": ds["validation_json"],
            "is_synthetic": bool(ds["is_synthetic"]),
            "created_at": ds["created_at"].isoformat() if ds["created_at"] else None,
        }
        if include_data and ds["stored_path"]:
            try:
                import pandas as pd
                _df = pd.read_csv(ds["stored_path"])
                dataset_out["rows"] = json.loads(_df.to_json(orient="records"))
            except Exception as _e:
                dataset_out["rows_error"] = str(_e)

    prediction = None
    try:
        prediction = monitor_scenario_model(scenario_id)
    except Exception as _e:
        prediction = {"status": "error", "error": str(_e)}

    return {
        "schema": "literev_model_export/1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scenario": {
            "id": scenario_id,
            "name": srow["name"] if srow else None,
            "query": srow["query"] if srow else None,
            "mode": srow["mode"] if srow else None,
        },
        "model_spec": _get_model_spec(scenario_id) or {},
        "runs": runs_out,
        "active_run_id": next((r["run_id"] for r in runs_out if r["is_active"]), None),
        "dataset": dataset_out,
        "prediction": prediction,
        "reproducibility_note": (
            "Chaque run inclut sa famille d'algorithme, ses hyperparamètres (hyperparameters), "
            "ses métriques et ses importances. Le model_spec (cible, variables, provenance vers "
            "les articles) + le jeu de données (schéma + lignes) permettent de ré-entraîner le "
            "modèle et de reproduire la prédiction - rien n'est une boîte noire."
        ),
    }


@app.get("/scenarios/{scenario_id}/model/export.xlsx")
def export_model_xlsx(scenario_id: str, _: None = Depends(require_api_key)):
    """Même contenu que /model/export, mais en CLASSEUR EXCEL (.xlsx) : une feuille
    Variables (cible + chaque variable, avec provenance), une feuille Dataset (toutes
    les VALEURS des variables + l'issue, une ligne par observation) et une feuille
    Model runs (métriques + hyperparamètres). Bouton « Excel » du tableau de bord.
    Authentifié (expose le schéma/les données du scénario)."""
    from fastapi.responses import StreamingResponse
    import io
    import model_export
    bundle = export_model_bundle(scenario_id, include_data=True, _=None)
    try:
        data = model_export.build_model_xlsx(bundle)
    except ModuleNotFoundError:
        raise HTTPException(status_code=501, detail="Export Excel indisponible (openpyxl absent).")
    return StreamingResponse(
        io.BytesIO(data),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="model_{scenario_id}.xlsx"'},
    )


@app.get("/scenarios/{scenario_id}/model/monitor")
def monitor_scenario_model(scenario_id: str, window: int = 7, lang: str | None = Query(None)) -> dict[str, Any]:
    """
    Statut live du modèle entraîné : score les `window` dernières lignes du
    dataset branché et renvoie un niveau d'alerte + la valeur courante.
    """
    from datetime import datetime, timezone

    with engine.connect() as conn:
        run = conn.execute(text("""
            SELECT id, family, task_type, metric, metrics_json, summary_json, artifact_path
            FROM scenario_model_run
            WHERE scenario_id = :sid AND is_active = TRUE AND artifact_path IS NOT NULL
            ORDER BY created_at DESC LIMIT 1
        """), {"sid": scenario_id}).mappings().first()
    if not run:
        return {"status": "unavailable", "status_color": "unavailable",
                "status_label": _msg(lang, "Modèle non entraîné", "Model not trained"),
                "message": _msg(lang, "Entraînez le modèle après avoir branché des données.",
                                "Train the model once data is connected.")}
    # Imports lourds (pandas + pile d'entraînement, ~160 Mo de RSS au premier appel)
    # APRÈS le test « pas de modèle » : la page d'un scénario sans modèle les payait
    # à chaque ouverture de l'onglet Variables & Modèle.
    import pandas as pd
    import model_trainer

    # ── Modèle de PRÉVISION (Prophet/SARIMAX) : pas de scoring ligne-à-ligne. La
    # valeur « courante » est le PROCHAIN point prévu, lu dans le résumé. Les bandes
    # d'alerte numériques (seuils littéraires) s'appliquent si présentes. ──
    if run["family"] in _TS_ALGO_FAMILIES or run["task_type"] == "forecast":
        summ = run["summary_json"] or {}
        fc = summ.get("forecast") or {}
        preds = fc.get("predicted") or []
        spec = _get_model_spec(scenario_id) or {}
        outcome = spec.get("outcome") or {}
        if not preds:
            return {"status": "unavailable", "status_color": "unavailable",
                    "status_label": _alert_labels(lang)["unavailable"],
                    "message": _msg(lang, "Prévision indisponible ; ré-entraînez le modèle.",
                                    "Forecast unavailable; retrain the model.")}
        next_val = float(preds[0])
        with engine.connect() as conn:
            vj = conn.execute(text(
                "SELECT variables_json FROM scenario_settings WHERE scenario_id = :sid"
            ), {"sid": scenario_id}).scalar()
        alert_thresholds = (dict(vj).get("alert_thresholds") if vj else None) or {}
        orange, red = model_trainer._alert_bounds(alert_thresholds)
        if orange is not None and red is not None:
            level = model_trainer._level_from_value(next_val, orange, red)
            label = (alert_thresholds.get(level) or {}).get("label") or _alert_labels(lang).get(level, "-")
        else:
            # Sans bornes littérature, NE JAMAIS afficher "green/Normal" par défaut
            # (règle "jamais vert par défaut", cf. compute_monitoring / _level_from_value
            # qui renvoient 'unavailable'). Un point de prévision élevé ne doit pas
            # apparaître "Normal" faute de seuils.
            level = "unavailable"
            label = _DEFAULT_ALERT_LABELS.get("unavailable", "Indisponible")
        return {
            "status": "ready", "scenario_id": scenario_id,
            "status_color": level, "status_label": label,
            "value": next_val, "kind": "forecast",
            "unit": outcome.get("unit"), "outcome": outcome.get("name"),
            "bands": {"orange": orange, "red": red},
            "forecast": {"dates": (fc.get("dates") or [])[:12], "predicted": [float(v) for v in preds[:12]]},
            "horizon": summ.get("horizon"), "n_scored": len(preds),
            "model": {"run_id": run["id"], "family": run["family"], "task_type": run["task_type"],
                      "metric": run["metric"], "metrics": run["metrics_json"]},
            "alert_thresholds": alert_thresholds,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    with engine.connect() as conn:
        ds = conn.execute(text("""
            SELECT stored_path FROM scenario_model_dataset
            WHERE scenario_id = :sid AND is_active = TRUE
            ORDER BY created_at DESC LIMIT 1
        """), {"sid": scenario_id}).mappings().first()
    if not ds or not ds["stored_path"]:
        return {"status": "unavailable", "status_color": "unavailable",
                "status_label": _msg(lang, "Aucune donnée", "No data"),
                "message": _msg(lang, "Aucun dataset branché.", "No dataset connected.")}

    try:
        import joblib
        pipeline = joblib.load(run["artifact_path"])
        df = pd.read_csv(ds["stored_path"])
    except Exception as e:
        logger.error(f"Monitor load {scenario_id}: {e}", exc_info=True)
        return {"status": "error", "status_color": "unavailable",
                "status_label": _msg(lang, "Erreur de chargement", "Loading error"), "message": str(e)}

    summary = run["summary_json"] or {}
    task_type = run["task_type"]
    classes = summary.get("classes")
    target = summary.get("target")

    window = max(1, min(int(window or 7), 200))
    recent = df.tail(window)
    target_values = None
    if task_type in ("regression", "count") and target and target in df.columns:
        target_values = pd.to_numeric(df[target], errors="coerce").tolist()

    # Récupérer la classe positive + libellés d'alerte depuis le spec.
    spec = _get_model_spec(scenario_id) or {}
    positive_class = (spec.get("outcome") or {}).get("positive_class")
    with engine.connect() as conn:
        vj = conn.execute(text(
            "SELECT variables_json FROM scenario_settings WHERE scenario_id = :sid"
        ), {"sid": scenario_id}).scalar()
    alert_thresholds = (dict(vj).get("alert_thresholds") if vj else None) or {}

    try:
        mon = model_trainer.compute_monitoring(
            pipeline, recent, task_type, classes=classes,
            positive_class=positive_class, target_values=target_values,
            alert_thresholds=alert_thresholds)
    except Exception as e:
        logger.error(f"Monitor score {scenario_id}: {e}", exc_info=True)
        return {"status": "error", "status_color": "unavailable",
                "status_label": _msg(lang, "Erreur de scoring", "Scoring error"), "message": str(e)}

    level = mon["level"]
    # 'unavailable' (NaN / pas de données / seuils non interprétables) ne doit
    # jamais réutiliser le libellé « Normal ».
    label = (
        _alert_labels(lang)["unavailable"] if level == "unavailable"
        else ((alert_thresholds.get(level) or {}).get("label")
              or _alert_labels(lang).get(level, _msg(lang, "Indisponible", "Unavailable")))
    )
    outcome = (spec.get("outcome") or {})

    return {
        "status": "ready",
        "scenario_id": scenario_id,
        "status_color": level,
        "status_label": label,
        "value": _jsonable(mon["value"]),
        "kind": mon["kind"],
        "unit": outcome.get("unit"),
        "outcome": outcome.get("name"),
        "positive_class": mon.get("positive_class"),
        "bands": mon["bands"],
        "n_scored": mon["n_scored"],
        "window": window,
        "model": {"run_id": run["id"], "family": run["family"],
                  "task_type": task_type, "metric": run["metric"],
                  "metrics": run["metrics_json"]},
        "alert_thresholds": alert_thresholds,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
