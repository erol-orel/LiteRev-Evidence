"""Model spec: proposals, validation, outcome templates, edits.

Extracted from main.py (LiteRev API); `main` re-exports everything for the scripts,
tools and tests.
"""
from __future__ import annotations

from typing import Any

from fastapi import Depends, HTTPException, Query
from sqlalchemy import text

from .core import _job_is_active, app, engine, logger, require_api_key
from .variables import (
    MODEL_SPEC_SCHEMA,
    _DTYPES,
    _METRICS,
    _TASK_TYPES,
    _TS_ALGO_FAMILIES,
    _coerce_enum,
    _coerce_family_for_task,
    _derive_data_template,
    _generate_variables_from_pico,
    _get_localized_variables,
    _norm_col,
    _slug_identifier,
)

@app.get("/scenarios/{scenario_id}/model/spec")
def get_scenario_model_spec(scenario_id: str, lang: str | None = Query(None)) -> dict[str, Any]:
    """
    Vue 'machine' du modèle dérivée de variables_json.model_spec : outcome
    (task_type), features (machine_name/dtype/source), algorithme (famille, CV,
    métrique) et data_template — les noms de colonnes EXACTS à fournir pour
    l'upload de données. Chaque élément porte sa provenance (ids d'articles).
    Les libellés d'affichage (noms, justifications, seuils) suivent la langue de l'UI ;
    les identifiants machine et le data_template restent invariants.
    """
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT variables_json, variables_validated, variables_generated_at,
                   variables_lang, variables_i18n
            FROM scenario_settings WHERE scenario_id = :sid
        """), {"sid": scenario_id}).mappings().first()

    if not (row and row["variables_json"]):
        return {"status": "empty", "message": "Aucune spécification de modèle. Générez d'abord les Variables & Modèle."}

    # Variables localisées (affichage FR/EN) : le model_spec miroir porte les noms et
    # la justification traduits, mais les machine_name / data_template sont intacts.
    vj = dict(_get_localized_variables(
        scenario_id, lang, base_variables=dict(row["variables_json"]),
        variables_lang=row.get("variables_lang"), variables_i18n=row.get("variables_i18n")))
    spec = vj.get("model_spec")
    if not spec:
        # Spec générée avant la Phase 1 : pas encore de schéma machine.
        return {
            "status": "legacy",
            "message": "Spécification antérieure au schéma machine. Relancez la génération des Variables & Modèle pour obtenir le data_template.",
            "validated": row["variables_validated"],
        }

    # ── Résolution provenance : pour chaque outcome/variable/algorithme, on
    # remonte l'article SOURCE le plus pertinent (plus récent, puis plus cité). ──
    prov_ids: set[int] = set()
    for elem in (spec.get("outcome") or {}, spec.get("algorithm") or {}):
        prov_ids.update(int(i) for i in (elem.get("provenance") or []) if isinstance(i, (int, float)))
    for f in (spec.get("features") or []):
        prov_ids.update(int(i) for i in (f.get("provenance") or []) if isinstance(i, (int, float)))
    # Provenance des modalités d'alerte (seuils green/orange/red) : on les résout
    # aussi pour que chaque modalité affiche l'article source.
    _at_raw = vj.get("alert_thresholds") if isinstance(vj, dict) else None
    if isinstance(_at_raw, dict):
        for _band in _at_raw.values():
            if isinstance(_band, dict):
                prov_ids.update(int(i) for i in (_band.get("provenance") or []) if isinstance(i, (int, float)))

    resolved: dict[str, Any] = {}
    if prov_ids:
        with engine.connect() as conn:
            arts = conn.execute(text("""
                SELECT id, title, year, doi, citation_count
                FROM literature_document WHERE id = ANY(:ids)
            """), {"ids": list(prov_ids)}).mappings().all()
        for a in arts:
            resolved[str(a["id"])] = {
                "id": a["id"], "title": a["title"], "year": a["year"],
                "doi": a["doi"], "citation_count": a["citation_count"],
                "url": (f"https://doi.org/{a['doi']}" if a["doi"] else None),
            }

    def _best_article(ids):
        cand = [resolved[str(int(i))] for i in (ids or [])
                if isinstance(i, (int, float)) and str(int(i)) in resolved]
        if not cand:
            return None
        cand.sort(key=lambda a: ((a["year"] or 0), (a["citation_count"] or 0)), reverse=True)
        return cand[0]

    outcome = dict(spec.get("outcome") or {})
    outcome["best_article"] = _best_article(outcome.get("provenance"))
    algorithm = dict(spec.get("algorithm") or {})
    algorithm["best_article"] = _best_article(algorithm.get("provenance"))
    features = []
    for f in (spec.get("features") or []):
        f2 = dict(f)
        f2["best_article"] = _best_article(f.get("provenance"))
        features.append(f2)

    # Modalités d'alerte enrichies : chaque niveau porte son article source (le
    # plus pertinent de sa provenance) + la liste résolue, pour lier la littérature.
    alert_thresholds = {}
    if isinstance(_at_raw, dict):
        for _lvl, _band in _at_raw.items():
            if not isinstance(_band, dict):
                continue
            _b2 = dict(_band)
            _prov = _band.get("provenance") or []
            _b2["best_article"] = _best_article(_prov)
            _b2["provenance_articles"] = [
                resolved[str(int(i))] for i in _prov
                if isinstance(i, (int, float)) and str(int(i)) in resolved
            ]
            alert_thresholds[_lvl] = _b2

    return {
        "status": "ready",
        "scenario_id": scenario_id,
        "schema": spec.get("schema"),
        "version": spec.get("version"),
        "outcome": outcome,
        "features": features,
        "algorithm": algorithm,
        "alert_thresholds": alert_thresholds,
        "data_template": spec.get("data_template"),
        "provenance_index": resolved or vj.get("_provenance_index", {}),
        "validated": row["variables_validated"],
        "generated_at": row["variables_generated_at"].isoformat() if row["variables_generated_at"] else None,
    }


# ─── SPEC EVOLUTION (Phase 5) : nouvelle évidence -> proposition -> validation ─
# Quand de nouveaux articles apportent de l'évidence (nouvel outcome, nouvelle
# variable, meilleur algorithme), on RÉGÉNÈRE le spec dans un slot de staging,
# on le DIFFE contre le spec actif validé, l'utilisateur VALIDE, puis le nouveau
# spec devient actif (version +1) et le modèle est ré-entraîné.

_SPEC_PROPOSAL_JOBS: dict[str, dict] = {}


def _ensure_spec_proposal_columns():
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE scenario_settings ADD COLUMN IF NOT EXISTS variables_proposal_json JSONB"))
        conn.execute(text("ALTER TABLE scenario_settings ADD COLUMN IF NOT EXISTS proposal_generated_at TIMESTAMP"))
        conn.execute(text("ALTER TABLE scenario_settings ADD COLUMN IF NOT EXISTS recommended_actions_json JSONB"))
        conn.execute(text("ALTER TABLE scenario_settings ADD COLUMN IF NOT EXISTS actions_generated_at TIMESTAMP"))
        # Langue des actions en cache : on régénère quand l'UI change de langue.
        conn.execute(text("ALTER TABLE scenario_settings ADD COLUMN IF NOT EXISTS recommended_actions_lang VARCHAR(8)"))
        # Localisation NON destructive des Variables/Modèle : langue de génération +
        # cache des traductions d'affichage par langue (le spec fonctionnel est intact).
        conn.execute(text("ALTER TABLE scenario_settings ADD COLUMN IF NOT EXISTS variables_lang VARCHAR(8)"))
        conn.execute(text("ALTER TABLE scenario_settings ADD COLUMN IF NOT EXISTS variables_i18n JSONB"))
        # Caches de visualisation persistés en DB (durables, contrairement à /tmp).
        conn.execute(text("ALTER TABLE scenario_settings ADD COLUMN IF NOT EXISTS clustering_json JSONB"))
        conn.execute(text("ALTER TABLE scenario_settings ADD COLUMN IF NOT EXISTS clustering_generated_at TIMESTAMP"))
        conn.execute(text("ALTER TABLE scenario_settings ADD COLUMN IF NOT EXISTS knowledge_graph_json JSONB"))
        conn.execute(text("ALTER TABLE scenario_settings ADD COLUMN IF NOT EXISTS kg_generated_at TIMESTAMP"))
        # Série OBSERVÉE (réelle) attachée au modèle SEIR pour superposition + calibration.
        conn.execute(text("ALTER TABLE scenario_settings ADD COLUMN IF NOT EXISTS seir_observed_json JSONB"))
    logger.info("Colonnes de proposition de spec vérifiées/créées.")


try:
    _ensure_spec_proposal_columns()
except Exception as _e:
    logger.warning(f"_ensure_spec_proposal_columns: {_e}")



def _diff_model_spec(old: dict | None, new: dict | None) -> dict:
    """Diff structuré entre deux model_spec (pur, testable)."""
    old, new = old or {}, new or {}
    o_out, n_out = old.get("outcome") or {}, new.get("outcome") or {}
    outcome_fields = {}
    for f in ("name", "machine_name", "task_type", "unit"):
        if (o_out.get(f) or None) != (n_out.get(f) or None):
            outcome_fields[f] = {"old": o_out.get(f), "new": n_out.get(f)}

    o_feats = {f.get("machine_name"): f for f in (old.get("features") or [])}
    n_feats = {f.get("machine_name"): f for f in (new.get("features") or [])}
    added = [k for k in n_feats if k not in o_feats]
    removed = [k for k in o_feats if k not in n_feats]
    changed = []
    for k in n_feats:
        if k in o_feats:
            fc = {}
            for fld in ("dtype", "source", "importance"):
                if (o_feats[k].get(fld) or None) != (n_feats[k].get(fld) or None):
                    fc[fld] = {"old": o_feats[k].get(fld), "new": n_feats[k].get(fld)}
            if fc:
                changed.append({"machine_name": k, "fields": fc})

    o_alg, n_alg = old.get("algorithm") or {}, new.get("algorithm") or {}
    alg_fields = {}
    for f in ("family", "metric"):
        if (o_alg.get(f) or None) != (n_alg.get(f) or None):
            alg_fields[f] = {"old": o_alg.get(f), "new": n_alg.get(f)}

    has_changes = bool(outcome_fields or added or removed or changed or alg_fields)
    return {
        "has_changes": has_changes,
        "outcome_changed": bool(outcome_fields),
        "outcome_fields": outcome_fields,
        "features_added": added,
        "features_removed": removed,
        "features_changed": changed,
        "algorithm_changed": bool(alg_fields),
        "algorithm_fields": alg_fields,
        "summary": {
            "added": len(added), "removed": len(removed), "changed": len(changed),
            "outcome_changed": bool(outcome_fields), "algorithm_changed": bool(alg_fields),
        },
    }


@app.post("/scenarios/{scenario_id}/model/spec/propose")
def propose_scenario_spec(scenario_id: str, lang: str | None = Query(None), _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Régénère le spec depuis l'évidence courante dans un slot de proposition (async)."""
    import threading, time

    if _job_is_active(_SPEC_PROPOSAL_JOBS.get(scenario_id)):
        return {"status": "already_running", "scenario_id": scenario_id}

    _SPEC_PROPOSAL_JOBS[scenario_id] = {"status": "running", "started_at": time.time()}

    def _run():
        try:
            result = _generate_variables_from_pico(scenario_id, persist="proposal", lang=lang)
            if "error" in result:
                _SPEC_PROPOSAL_JOBS[scenario_id] = {"status": "error", "error": result["error"]}
            else:
                _SPEC_PROPOSAL_JOBS[scenario_id] = {"status": "done"}
        except Exception as e:
            logger.error(f"Spec proposal job {scenario_id}: {e}", exc_info=True)
            _SPEC_PROPOSAL_JOBS[scenario_id] = {"status": "error", "error": str(e)}

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "started", "scenario_id": scenario_id}


@app.get("/scenarios/{scenario_id}/model/spec/proposal")
def get_scenario_spec_proposal(scenario_id: str) -> dict[str, Any]:
    """Proposition de spec en attente + diff vs spec actif."""
    job = _SPEC_PROPOSAL_JOBS.get(scenario_id, {})
    if job.get("status") == "running":
        return {"status": "generating", "message": "Régénération en cours, réessayez bientôt."}

    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT variables_json, variables_proposal_json, proposal_generated_at
            FROM scenario_settings WHERE scenario_id = :sid
        """), {"sid": scenario_id}).mappings().first()

    if job.get("status") == "error":
        return {"status": "error", "error": job.get("error")}
    if not (row and row["variables_proposal_json"]):
        return {"status": "empty", "message": "Aucune proposition. Lancez /model/spec/propose."}

    proposal = dict(row["variables_proposal_json"])
    active = dict(row["variables_json"]) if row["variables_json"] else {}
    diff = _diff_model_spec(active.get("model_spec"), proposal.get("model_spec"))

    return {
        "status": "ready",
        "scenario_id": scenario_id,
        "diff": diff,
        "proposal_spec": proposal.get("model_spec"),
        "active_version": (active.get("model_spec") or {}).get("version"),
        "proposal_provenance": proposal.get("_provenance_index", {}),
        "generated_at": row["proposal_generated_at"].isoformat() if row["proposal_generated_at"] else None,
    }


@app.post("/scenarios/{scenario_id}/model/spec/proposal/validate")
def validate_scenario_spec_proposal(scenario_id: str, payload: dict[str, Any],
                                    _: None = Depends(require_api_key)) -> dict[str, Any]:
    """
    Valide ou rejette la proposition. Body: {"action": "accept"|"reject", "retrain": bool}.
    accept -> la proposition devient le spec actif (version +1), validé, et le
    modèle est ré-entraîné si un dataset est branché.
    """
    from .model_training import _MODEL_TRAIN_JOBS, _claim_model_train, _run_model_training  # lazy: model_training is loaded after this module
    import json as _json
    import threading
    from datetime import datetime, timezone

    action = (payload.get("action") or "").strip().lower()
    if action not in ("accept", "reject"):
        raise HTTPException(status_code=400, detail="action doit être 'accept' ou 'reject'.")

    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT variables_json, variables_proposal_json
            FROM scenario_settings WHERE scenario_id = :sid
        """), {"sid": scenario_id}).mappings().first()
    if not (row and row["variables_proposal_json"]):
        raise HTTPException(status_code=400, detail="Aucune proposition en attente.")

    if action == "reject":
        with engine.begin() as conn:
            conn.execute(text("""
                UPDATE scenario_settings
                SET variables_proposal_json = NULL, proposal_generated_at = NULL, updated_at = NOW()
                WHERE scenario_id = :sid
            """), {"sid": scenario_id})
        return {"status": "rejected", "scenario_id": scenario_id}

    # accept : promouvoir la proposition en spec actif, version +1.
    proposal = dict(row["variables_proposal_json"])
    active = dict(row["variables_json"]) if row["variables_json"] else {}
    old_ver = int((active.get("model_spec") or {}).get("version", 0) or 0)
    if proposal.get("model_spec"):
        proposal["model_spec"]["version"] = old_ver + 1

    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE scenario_settings
            SET variables_json = CAST(:vars AS jsonb),
                variables_validated = TRUE,
                variables_generated_at = NOW(),
                variables_lang = NULL,
                variables_i18n = NULL,
                variables_proposal_json = NULL,
                proposal_generated_at = NULL,
                updated_at = NOW()
            WHERE scenario_id = :sid
        """), {"sid": scenario_id, "vars": _json.dumps(proposal)})

    # Ré-entraînement automatique si un dataset est branché.
    retrain = payload.get("retrain", True)
    retrain_started = False
    if retrain:
        with engine.connect() as conn:
            ds = conn.execute(text(
                "SELECT id FROM scenario_model_dataset WHERE scenario_id = :sid AND is_active = TRUE LIMIT 1"
            ), {"sid": scenario_id}).first()
        if ds and _MODEL_TRAIN_JOBS.get(scenario_id, {}).get("status") != "running":
            if _claim_model_train(scenario_id):   # atomique : évite 2 entraînements concurrents
                threading.Thread(target=_run_model_training, args=(scenario_id, 25), daemon=True).start()
                retrain_started = True

    return {
        "status": "accepted",
        "scenario_id": scenario_id,
        "new_version": old_ver + 1,
        "retrain_started": retrain_started,
        "validated_at": datetime.now(timezone.utc).isoformat(),
    }


# Messages d'alerte « souple » (task_type ↔ contenu réel de la cible), bilingues.
_SPEC_WARN = {
    "target_not_numeric_for_regression": {
        "fr": "La cible « {t} » n'est pas numérique : une classification conviendrait sans doute mieux qu'une régression.",
        "en": "Target “{t}” is not numeric: a classification task would likely fit better than regression.",
    },
    "target_binary_for_regression": {
        "fr": "La cible « {t} » ne prend que {n} valeurs distinctes : une classification conviendrait sans doute mieux.",
        "en": "Target “{t}” has only {n} distinct values: a classification task would likely fit better.",
    },
    "target_high_cardinality_for_classification": {
        "fr": "La cible « {t} » est numérique avec {n} valeurs distinctes : une régression conviendrait sans doute mieux qu'une classification.",
        "en": "Target “{t}” is numeric with {n} distinct values: a regression task would likely fit better than classification.",
    },
}


def _task_target_sanity(scenario_id: str, outcome: dict) -> tuple[str, dict] | None:
    """Contrôle de cohérence (SOUPLE) entre le task_type choisi et le contenu réel
    de la colonne cible du dataset branché. Best-effort : renvoie None s'il n'y a pas
    de dataset ou si la lecture échoue. Retour : (code, params) → localisé par l'appelant."""
    import pandas as pd
    import pandas.api.types as pdt
    with engine.connect() as conn:
        ds = conn.execute(text(
            "SELECT stored_path FROM scenario_model_dataset WHERE scenario_id = :sid AND is_active = TRUE LIMIT 1"
        ), {"sid": scenario_id}).mappings().first()
    if not (ds and ds["stored_path"]):
        return None
    tt = (outcome.get("task_type") or "classification").strip().lower()
    target_norm = _norm_col(outcome.get("machine_name") or "")
    df = pd.read_csv(ds["stored_path"], nrows=5000)
    col = next((c for c in df.columns if _norm_col(c) == target_norm), None)
    if col is None:
        return None
    s = df[col].dropna()
    if s.empty:
        return None
    is_num = pdt.is_numeric_dtype(s)
    nuniq = int(s.nunique())
    if tt in ("regression", "count"):
        if not is_num:
            return ("target_not_numeric_for_regression", {"t": str(col)})
        if nuniq <= 2:
            return ("target_binary_for_regression", {"t": str(col), "n": nuniq})
    elif tt == "classification":
        if is_num and nuniq > 20:
            return ("target_high_cardinality_for_classification", {"t": str(col), "n": nuniq})
    return None


@app.get("/model/outcome-templates")
def list_outcome_templates() -> dict[str, Any]:
    """Catalogue des OUTCOMES prêts à l'emploi (GESICA) : surcharge des urgences, taux
    d'occupation des lits, volume d'appels, pic d'appels (forêt extrémale). Le frontend
    en applique un pour définir proprement la cible, puis on téléverse l'extract hospitalier."""
    import outcome_templates
    return {"templates": outcome_templates.as_list()}


@app.post("/scenarios/{scenario_id}/model/outcome-template")
def apply_outcome_template(scenario_id: str, payload: dict[str, Any],
                           _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Applique un OUTCOME prédéfini au model_spec du scénario : remplace la cible
    (nom/machine_name/task_type/unité/classe positive), pose l'algorithme recommandé
    (dont extremal_rf pour un pic), fusionne les variables suggérées (source=user, sans
    écraser les existantes) et les seuils d'alerte, reconstruit le data_template (les
    colonnes EXACTES à téléverser) et incrémente la version. Body : {"template_id": "..."}."""
    import json as _json
    import outcome_templates
    from datetime import datetime, timezone
    tpl = outcome_templates.get(payload.get("template_id"))
    if not tpl:
        raise HTTPException(status_code=404, detail="Modèle d'outcome inconnu.")

    with engine.connect() as conn:
        row = conn.execute(text(
            "SELECT variables_json FROM scenario_settings WHERE scenario_id = :sid"
        ), {"sid": scenario_id}).mappings().first()
    vj = dict(row["variables_json"]) if (row and row["variables_json"]) else {}
    spec = dict(vj.get("model_spec") or {})

    outcome = {
        "name": tpl["outcome"]["name"], "machine_name": tpl["outcome"]["machine_name"],
        "task_type": tpl["outcome"]["task_type"], "unit": tpl["outcome"].get("unit", ""),
        "positive_class": (tpl["outcome"].get("positive_class")
                           if tpl["outcome"]["task_type"] == "classification" else None),
        "description": tpl.get("description", ""), "source": "user", "provenance": [],
    }
    # Merge suggested features onto any existing ones (never overwrite; drop a feature that
    # collides with the new target machine_name so a column can't be both feature and outcome).
    features = [dict(f) for f in (spec.get("features") or [])
                if f.get("machine_name") != outcome["machine_name"]]
    seen = {f.get("machine_name") for f in features}
    for tf in tpl["features"]:
        mn = tf["machine_name"]
        if mn in seen or mn == outcome["machine_name"]:
            continue
        features.append({"name": tf["name"], "machine_name": mn, "dtype": tf.get("dtype", "float"),
                         "source": "user", "public_provider": None, "importance": "medium",
                         "provenance": []})
        seen.add(mn)

    algorithm = dict(tpl["algorithm"])
    spec.setdefault("schema", MODEL_SPEC_SCHEMA)
    spec["outcome"], spec["algorithm"], spec["features"] = outcome, algorithm, features
    spec["data_template"] = _derive_data_template(outcome, features)
    spec["version"] = int(spec.get("version", 0) or 0) + 1
    spec.setdefault("epidemic_parameters", {"applicable": False, "disease": None, "params": {}})
    vj["model_spec"] = spec
    if tpl.get("alert_thresholds"):
        vj["alert_thresholds"] = tpl["alert_thresholds"]
    vj["outcome_template_id"] = tpl["id"]

    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO scenario_settings (scenario_id, variables_json, variables_validated, variables_generated_at, updated_at)
            VALUES (:sid, CAST(:vj AS jsonb), TRUE, NOW(), NOW())
            ON CONFLICT (scenario_id) DO UPDATE
                SET variables_json = CAST(:vj AS jsonb), variables_validated = TRUE, updated_at = NOW()
        """), {"sid": scenario_id, "vj": _json.dumps(vj)})
    return {"status": "applied", "scenario_id": scenario_id, "template_id": tpl["id"],
            "model_spec": spec}


@app.post("/scenarios/{scenario_id}/model/spec/edit")
def edit_scenario_model_spec(scenario_id: str, payload: dict[str, Any],
                             lang: str | None = Query(None),
                             _: None = Depends(require_api_key)) -> dict[str, Any]:
    """
    Édite DIRECTEMENT le spec actif (sans régénérer depuis l'évidence) : choix de
    l'algorithme (parmi les candidats de l'évidence ou toute famille valide),
    task_type / positive_class de l'outcome, ajout/suppression de variables. Le
    data_template est reconstruit (colonnes attendues à l'upload), la version
    incrémentée, et le modèle ré-entraîné si demandé et si un dataset est branché.

    Body (tous les champs optionnels) :
      {"algorithm_family": "lightgbm", "metric": "rmse", "task_type": "regression",
       "positive_class": "oui", "remove_features": ["mn1"],
       "add_features": [{"name": "Âge", "dtype": "int", "importance": "medium"}],
       "retrain": true}
    """
    from .model_training import _MODEL_TRAIN_JOBS, _claim_model_train, _run_model_training  # lazy: model_training is loaded after this module
    import json as _json
    import threading
    from datetime import datetime, timezone

    with engine.connect() as conn:
        row = conn.execute(text(
            "SELECT variables_json FROM scenario_settings WHERE scenario_id = :sid"
        ), {"sid": scenario_id}).mappings().first()
    if not (row and row["variables_json"]):
        raise HTTPException(status_code=400, detail="Aucune spécification de modèle à éditer. Générez d'abord les Variables & Modèle.")
    vj = dict(row["variables_json"])
    spec = vj.get("model_spec")
    if not spec:
        raise HTTPException(status_code=400, detail="Spécification antérieure au schéma machine. Relancez la génération des Variables & Modèle.")

    spec = dict(spec)
    outcome = dict(spec.get("outcome") or {})
    algorithm = dict(spec.get("algorithm") or {})
    features = [dict(f) for f in (spec.get("features") or [])]
    changed = False

    # ── Outcome : task_type ──
    if payload.get("task_type"):
        new_tt = _coerce_enum(payload["task_type"], _TASK_TYPES, outcome.get("task_type") or "classification")
        if new_tt != (outcome.get("task_type") or ""):
            outcome["task_type"] = new_tt
            changed = True
            # Métrique par défaut si l'actuelle n'a plus de sens pour la nouvelle tâche.
            _reg_metrics, _clf_metrics = {"rmse", "mae", "r2"}, {"roc_auc", "average_precision"}
            _default_metric = {"classification": "roc_auc", "regression": "rmse",
                               "count": "rmse", "survival": "c_index"}[new_tt]
            cur_metric = (algorithm.get("metric") or "").strip().lower()
            ok = (cur_metric in _reg_metrics) if new_tt in ("regression", "count") \
                else (cur_metric in _clf_metrics) if new_tt == "classification" else True
            if not ok:
                algorithm["metric"] = _default_metric
            if new_tt != "classification":
                outcome["positive_class"] = None
            algorithm["family"] = _coerce_family_for_task(algorithm.get("family"), new_tt)

    # ── Outcome : positive_class (classification seulement) ──
    if "positive_class" in payload and outcome.get("task_type") == "classification":
        pc = payload.get("positive_class")
        pc = str(pc).strip() if pc not in (None, "") else None
        if pc != outcome.get("positive_class"):
            outcome["positive_class"] = pc
            changed = True

    # ── Algorithme : famille + métrique ──
    if payload.get("algorithm_family"):
        fam = _coerce_family_for_task(payload["algorithm_family"], outcome.get("task_type") or "classification")
        if fam != (algorithm.get("family") or ""):
            algorithm["family"] = fam
            changed = True
    if payload.get("metric"):
        met = _coerce_enum(payload["metric"], _METRICS, algorithm.get("metric") or "rmse")
        if met != (algorithm.get("metric") or ""):
            algorithm["metric"] = met
            changed = True

    # ── Features : suppression ──
    remove = {str(m).strip() for m in (payload.get("remove_features") or []) if str(m).strip()}
    if remove:
        kept = [f for f in features if f.get("machine_name") not in remove]
        if len(kept) != len(features):
            features, changed = kept, True

    # ── Features : ajout (variable manuelle, sans provenance évidence) ──
    used = {f.get("machine_name") for f in features if f.get("machine_name")}
    used.add(outcome.get("machine_name"))
    for add in (payload.get("add_features") or []):
        if not isinstance(add, dict):
            continue
        name = str(add.get("name") or "").strip()
        if not name:
            continue
        mn = _slug_identifier(add.get("machine_name") or name, used)
        features.append({
            "name": name, "machine_name": mn,
            "dtype": _coerce_enum(add.get("dtype"), _DTYPES, "float"),
            "source": "user", "public_provider": None,
            "importance": _coerce_enum(add.get("importance"), {"high", "medium", "low"}, "medium"),
            "provenance": [],
        })
        changed = True

    if not features:
        raise HTTPException(status_code=400, detail="Le modèle doit conserver au moins une variable explicative.")
    if not changed:
        return {"status": "unchanged", "scenario_id": scenario_id}

    # ── Cohérence prévision : une famille de série temporelle exige une cible
    # NUMÉRIQUE. Si l'outcome est en classification, on ramène à 'regression' (cible
    # float dans le data_template) et on nettoie la classe positive / la métrique. ──
    if (algorithm.get("family") in _TS_ALGO_FAMILIES) and outcome.get("task_type") == "classification":
        outcome["task_type"] = "regression"
        outcome["positive_class"] = None
        if (algorithm.get("metric") or "") in ("roc_auc", "average_precision"):
            algorithm["metric"] = "rmse"

    # ── Reconstruire le spec + le data_template (jamais désynchronisé des features) ──
    spec["outcome"], spec["algorithm"], spec["features"] = outcome, algorithm, features
    spec["data_template"] = _derive_data_template(outcome, features)
    spec["version"] = int(spec.get("version", 1) or 1) + 1
    vj["model_spec"] = spec

    # ── Contrôle souple task↔cible (vs dataset branché), localisé ──
    warnings: list[str] = []
    try:
        _sanity = _task_target_sanity(scenario_id, outcome)
        if _sanity:
            _code, _params = _sanity
            _lg = "en" if (lang or "fr").strip().lower().startswith("en") else "fr"
            warnings.append(_SPEC_WARN[_code][_lg].format(**_params))
    except Exception as _we:
        logger.warning(f"task_target_sanity {scenario_id}: {_we}")

    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE scenario_settings
            SET variables_json = CAST(:vars AS jsonb),
                variables_validated = TRUE,
                variables_generated_at = NOW(),
                variables_lang = NULL,
                variables_i18n = NULL,
                updated_at = NOW()
            WHERE scenario_id = :sid
        """), {"sid": scenario_id, "vars": _json.dumps(vj)})

    # ── Ré-entraînement si demandé + dataset branché ──
    retrain = payload.get("retrain", True)
    retrain_started = False
    if retrain:
        with engine.connect() as conn:
            ds = conn.execute(text(
                "SELECT id FROM scenario_model_dataset WHERE scenario_id = :sid AND is_active = TRUE LIMIT 1"
            ), {"sid": scenario_id}).first()
        if ds and _MODEL_TRAIN_JOBS.get(scenario_id, {}).get("status") != "running":
            if _claim_model_train(scenario_id):   # atomique : évite 2 entraînements concurrents
                threading.Thread(target=_run_model_training, args=(scenario_id, 25), daemon=True).start()
                retrain_started = True

    return {
        "status": "updated",
        "scenario_id": scenario_id,
        "new_version": spec["version"],
        "outcome": {"machine_name": outcome.get("machine_name"), "name": outcome.get("name"),
                    "task_type": outcome.get("task_type"), "positive_class": outcome.get("positive_class")},
        "algorithm": {"family": algorithm.get("family"), "metric": algorithm.get("metric")},
        "features": [{"name": f["name"], "machine_name": f["machine_name"],
                      "dtype": f["dtype"], "importance": f.get("importance")} for f in features],
        "data_template": spec["data_template"],
        "warnings": warnings,
        "retrain_started": retrain_started,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
