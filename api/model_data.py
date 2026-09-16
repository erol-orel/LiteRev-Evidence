"""Model datasets: upload, validation, public data connectors, auto-fetch, synthetic data.

Extracted from main.py (LiteRev API); `main` re-exports everything for the scripts,
tools and tests.
"""
from __future__ import annotations

import json
import os
import os as _os_mod
from pathlib import Path
from typing import Any

from fastapi import Depends, HTTPException
from fastapi import UploadFile, File
from pydantic import BaseModel, Field
from sqlalchemy import text

from .core import app, engine, logger, require_api_key
from .variables import _norm_col

# Défaut SOUS la racine de déploiement, comme annoncé par .env.example : le défaut réel
# était /home/ubuntu/uploads_datasets, donc hors du répertoire qu'un opérateur
# sauvegarde ou nettoie, et sans rapport avec la documentation.
MODEL_DATA_DIR = Path(_os_mod.environ.get("MODEL_DATA_DIR", "/opt/literev-api/uploads_datasets"))


def _ensure_model_dataset_table():
    """Suivi des datasets uploadés par scénario pour l'entraînement du modèle."""
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS scenario_model_dataset (
                id              BIGSERIAL PRIMARY KEY,
                scenario_id     VARCHAR(80) NOT NULL,
                filename        TEXT,
                stored_path     TEXT,
                n_rows          INTEGER,
                n_cols          INTEGER,
                columns_json    JSONB,
                validation_json JSONB,
                is_active       BOOLEAN DEFAULT TRUE,
                created_at      TIMESTAMP DEFAULT NOW()
            )
        """))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_model_dataset_scenario_active "
            "ON scenario_model_dataset (scenario_id, is_active)"
        ))
        # Données SYNTHÉTIQUES de démo (badge « non réel » dans l'UI) vs données réelles.
        conn.execute(text("ALTER TABLE scenario_model_dataset ADD COLUMN IF NOT EXISTS is_synthetic BOOLEAN DEFAULT FALSE"))
    logger.info("Table scenario_model_dataset vérifiée/créée.")


try:
    _ensure_model_dataset_table()
except Exception as _e:
    logger.warning(f"_ensure_model_dataset_table: {_e}")


def _get_model_spec(scenario_id: str) -> dict | None:
    """Retourne le model_spec stocké (ou None) pour un scénario."""
    with engine.connect() as conn:
        row = conn.execute(text(
            "SELECT variables_json FROM scenario_settings WHERE scenario_id = :sid"
        ), {"sid": scenario_id}).mappings().first()
    if not (row and row["variables_json"]):
        return None
    return dict(row["variables_json"]).get("model_spec")


def _validate_dataset_against_template(file_columns: list, data_template: dict,
                                       file_dtype_kinds: dict | None = None,
                                       n_rows: int | None = None) -> dict:
    """
    Compare les colonnes d'un fichier au data_template (pur, testable).
    Le matching est insensible à la casse/aux espaces ; on signale les colonnes
    cible/explicatives présentes, manquantes (user vs public_api), en trop, et
    les incompatibilités de type. Conclut sur la possibilité d'entraîner.
    """
    file_dtype_kinds = file_dtype_kinds or {}
    cols = list(file_columns or [])
    norm_to_file: dict[str, Any] = {}
    for c in cols:
        norm_to_file.setdefault(_norm_col(c), c)

    template_cols = data_template.get("columns") or []
    target_col = data_template.get("target_column")

    present_required, missing_required, present_optional = [], [], []
    missing_user, missing_public, missing_seir, matched_features = [], [], [], []
    renamed, dtype_warnings = [], []
    target_present = False
    n_features = n_features_present = 0

    def file_has(canonical):
        f = norm_to_file.get(_norm_col(canonical))
        if f is not None and f != canonical:
            renamed.append({"expected": canonical, "found": f})
        return f

    for col in template_cols:
        name, role = col.get("name"), col.get("role")
        required, source = col.get("required", False), col.get("source", "user")
        if role == "feature":
            n_features += 1
        found = file_has(name)
        if found is not None:
            if role == "outcome":
                target_present = True
            elif role == "feature":
                # UNIQUEMENT les vraies variables explicatives. En comptant ici toute
                # colonne non-outcome, la colonne de DATES était comptée comme une
                # variable explicative : n_features_present dépassait n_features_total
                # (ratio impossible affiché à l'utilisateur) et, plus grave, un fichier
                # ne contenant QUE la date et la cible - zéro variable explicative -
                # satisfaisait `n_features_present >= 1` et se voyait déclarer
                # « prêt à entraîner ».
                n_features_present += 1
                matched_features.append(name)
            (present_required if required else present_optional).append(name)
            kind, exp = file_dtype_kinds.get(_norm_col(found)), col.get("dtype")
            if exp in ("float", "int") and kind == "other":
                dtype_warnings.append({"column": name, "expected": exp, "found_kind": kind})
        else:
            if required:
                missing_required.append(name)
            if source == "public_api":
                missing_public.append(name)
            elif source == "seir":
                missing_seir.append(name)  # dérivée du sous-modèle SEIR → auto-remplie
            elif role != "outcome":
                missing_user.append(name)

    template_norms = {_norm_col(c.get("name")) for c in template_cols}
    extra_columns = [c for c in cols if _norm_col(c) not in template_norms]

    reasons = []
    if not target_present:
        reasons.append(f"Colonne cible '{target_col}' absente (obligatoire pour entraîner).")
    if n_features_present == 0:
        reasons.append("Aucune variable explicative présente dans le fichier.")
    # Assez de lignes ? L'entraînement exige un plancher (model_trainer: min 20).
    # Sans ce contrôle, un fichier de 5 lignes renvoyait « training_started: true »
    # puis échouait silencieusement en arrière-plan.
    _MIN_TRAIN_ROWS = 20
    if n_rows is not None and n_rows < _MIN_TRAIN_ROWS:
        reasons.append(f"Trop peu de lignes ({n_rows}) : minimum {_MIN_TRAIN_ROWS} pour entraîner.")
    # Colonnes numériques contenant des valeurs non numériques : bloquant (sinon
    # l'imputation/mise à l'échelle plante à l'entraînement en arrière-plan).
    if dtype_warnings:
        _bad = ", ".join(str(w.get("column")) for w in dtype_warnings)
        reasons.append(f"Colonne(s) numérique(s) avec des valeurs non numériques : {_bad}. Corrigez ou retirez ces valeurs.")
    can_train = (
        target_present and n_features_present >= 1
        and (n_rows is None or n_rows >= _MIN_TRAIN_ROWS)
        and not dtype_warnings
    )

    return {
        "target_column": target_col,
        "target_present": target_present,
        "n_features_total": n_features,
        "n_features_present": n_features_present,
        "matched_features": matched_features,
        "present_required": present_required,
        "missing_required": missing_required,
        "present_optional": present_optional,
        "missing_user": missing_user,
        "missing_public": missing_public,
        "missing_seir": missing_seir,
        "extra_columns": extra_columns,
        "renamed": renamed,
        "dtype_warnings": dtype_warnings,
        "readiness": {
            "can_train": can_train,
            "reasons": reasons,
            "auto_fetchable": missing_public + missing_seir,
        },
    }


def _dataframe_dtype_kinds(df) -> dict:
    """{col_normalisé: 'numeric'|'datetime'|'bool'|'other'} depuis un DataFrame pandas."""
    import pandas as pd
    kinds = {}
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_bool_dtype(s):
            k = "bool"
        elif pd.api.types.is_numeric_dtype(s):
            k = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(s):
            k = "datetime"
        else:
            k = "other"
        kinds[_norm_col(c)] = k
    return kinds


def _maybe_autotrain(scenario_id: str, report: dict) -> bool:
    """Démarre l'entraînement si les données branchées suffisent (can_train) et
    qu'aucun entraînement n'est déjà en cours. Renvoie True si lancé."""
    from .model_training import _MODEL_TRAIN_JOBS, _claim_model_train, _run_model_training  # lazy: model_training is loaded after this module
    import threading

    if not ((report or {}).get("readiness") or {}).get("can_train"):
        return False
    if _MODEL_TRAIN_JOBS.get(scenario_id, {}).get("status") == "running":
        return False
    if not _claim_model_train(scenario_id):
        return False   # un entraînement est déjà en cours pour ce scénario (TOCTOU fermé)
    threading.Thread(target=_run_model_training, args=(scenario_id, 25), daemon=True).start()
    logger.info(f"Auto-entraînement déclenché pour {scenario_id} (données suffisantes).")
    return True


# ─── PHASE 2 : Connecteurs de données publiques (auto-remplissage des variables) ─
# Chaque connecteur récupère une source publique RÉELLE et lisible par machine et
# renvoie une série temporelle quotidienne « tidy » joignable au dataset du modèle
# sur la clé date - au lieu d'un upload CSV manuel. Discovery + fetch ici ; le
# mapping variable→connecteur et l'assemblage du dataset arrivent ensuite.
class ConnectorFetchIn(BaseModel):
    region: str | None = None        # alias Romandie (geneva|lausanne|sion|…) ou lat+lon
    lat: float | None = None
    lon: float | None = None
    start_date: str = Field(..., min_length=8)   # YYYY-MM-DD
    end_date: str = Field(..., min_length=8)
    limit: int = Field(default=2000, ge=1, le=20000)


@app.get("/model/connectors")
def list_data_connectors(_: None = Depends(require_api_key)) -> dict[str, Any]:
    """Liste les connecteurs de données publiques disponibles (métadonnées : source,
    licence, granularité, variables fournies). Sert à mapper les colonnes `public_api`
    du data_template vers une source récupérable automatiquement."""
    import data_connectors
    return {"connectors": data_connectors.list_connectors()}


@app.post("/model/connectors/{connector_id}/fetch")
def fetch_data_connector(
    connector_id: str, payload: ConnectorFetchIn, _: None = Depends(require_api_key),
) -> dict[str, Any]:
    """Récupère la série quotidienne « tidy » d'un connecteur pour une zone + période.
    Renvoie les lignes (plafonnées à `limit`) + la provenance réelle (source, licence)."""
    import data_connectors
    if connector_id not in data_connectors.CONNECTORS:
        raise HTTPException(status_code=404, detail=f"Connecteur inconnu : {connector_id}")
    params = payload.model_dump(exclude_none=True)
    limit = params.pop("limit", 2000)
    try:
        rows = data_connectors.fetch_series(connector_id, params)
    except ValueError as _ve:
        raise HTTPException(status_code=422, detail=str(_ve))
    except Exception as _fe:
        logger.warning(f"Connector {connector_id} fetch failed: {_fe}")
        raise HTTPException(status_code=502,
                            detail=f"Échec de récupération depuis la source publique : {_fe}")
    meta = data_connectors.CONNECTORS[connector_id].metadata()
    return {
        "connector_id": connector_id, "provider": meta["provider"], "license": meta["license"],
        "commercial_ok": meta["commercial_ok"], "variables": meta["variables"],
        "row_count": len(rows), "rows": rows[:limit],
    }


def _agg_for_column(col: str) -> str:
    """Fonction d'agrégation au ré-échantillonnage, déduite du nom de colonne :
    précipitations → somme, charge virale (eaux usées) → dernière valeur du bucket,
    tout le reste (température, humidité, qualité de l'air, incidence) → moyenne."""
    c = (col or "").lower()
    if "precip" in c or "rain" in c or "rainfall" in c:
        return "sum"
    if "load" in c or "wastewater" in c or c.endswith("_ww") or "viral" in c:
        return "last"
    return "mean"


def _assemble_connector_frames(frames: dict, mappings: list[dict], freq: str,
                               datetime_col: str | None):
    """PUR : resample chaque série de connecteur à `freq` (moyenne), joint sur la clé
    'date' et renomme chaque `connector_variable` vers sa `template_column`. Renvoie
    (DataFrame assemblé | None, colonnes remplies). Aligne les fréquences hétérogènes
    (météo quotidienne, eaux usées / Sentinella hebdomadaires) sur une grille commune."""
    import pandas as pd
    resampled: dict[str, Any] = {}
    for cid, df in (frames or {}).items():
        try:
            if "date" not in getattr(df, "columns", []):
                continue
            d = df.copy()
            d["date"] = pd.to_datetime(d["date"], errors="coerce")
            d = d.dropna(subset=["date"])
            if d.empty:
                continue
            d = d.set_index("date")
            for _c in list(d.columns):
                d[_c] = pd.to_numeric(d[_c], errors="coerce")
            d = d.resample(freq).agg({_c: _agg_for_column(_c) for _c in d.columns}).reset_index()
            d["date"] = d["date"].dt.strftime("%Y-%m-%d")
            resampled[cid] = d
        except Exception:
            continue
    assembled = None
    filled: list[str] = []
    for m in mappings:
        d = resampled.get(m["connector_id"])
        var = m["connector_variable"]
        if d is None or var not in d.columns:
            continue
        part = d[["date", var]].rename(columns={var: m["template_column"]})
        part = part.dropna(subset=[m["template_column"]]).drop_duplicates(subset=["date"])
        assembled = part if assembled is None else assembled.merge(part, on="date", how="outer")
        filled.append(m["template_column"])
    if assembled is None or assembled.empty:
        return None, []
    if datetime_col and datetime_col not in assembled.columns:
        assembled = assembled.rename(columns={"date": datetime_col})
    assembled = assembled.sort_values(assembled.columns[0]).reset_index(drop=True)
    return assembled, filled


@app.post("/scenarios/{scenario_id}/model/data")
async def upload_model_dataset(
    scenario_id: str,
    file: UploadFile = File(...),
    auto_train: bool = True,
    _: None = Depends(require_api_key),
) -> dict[str, Any]:
    """
    Branche un jeu de données (CSV/XLSX) sur le model_spec d'un scénario.
    Valide les en-têtes contre le data_template, stocke le dataset (actif), et
    renvoie un rapport de validation + l'état de préparation à l'entraînement.
    Si les données suffisent (can_train) et auto_train, l'entraînement démarre
    automatiquement (upload -> entraînement -> modèle en ligne, sans étape manuelle).
    """
    import io
    import pandas as pd
    from datetime import datetime, timezone

    spec = _get_model_spec(scenario_id)
    if not spec:
        raise HTTPException(status_code=400,
                            detail="Aucune spécification de modèle. Générez puis validez les Variables & Modèle d'abord.")
    data_template = spec.get("data_template") or {}
    if not data_template.get("columns"):
        raise HTTPException(status_code=400, detail="data_template absent du model_spec. Relancez la génération des variables.")

    filename = file.filename or ""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ("csv", "xlsx", "xls"):
        raise HTTPException(status_code=400, detail="Seuls les fichiers CSV et Excel (.xlsx, .xls) sont acceptés.")

    # Plafond d'octets AVANT de tout charger en mémoire : lit au plus MAX+1 octets
    # (borne la RAM face à un upload démesuré / malveillant). 25 Mo couvrent largement
    # un CSV/XLSX de 500 000 lignes, qui est de toute façon rejeté plus bas.
    _MAX_UPLOAD_BYTES = 25 * 1024 * 1024
    content = await file.read(_MAX_UPLOAD_BYTES + 1)
    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413,
                            detail=f"Fichier trop volumineux (> {_MAX_UPLOAD_BYTES // (1024 * 1024)} Mo).")
    try:
        if ext in ("xlsx", "xls"):
            df = pd.read_excel(io.BytesIO(content))
        else:
            df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lecture du fichier impossible : {e}")

    if len(df) == 0:
        raise HTTPException(status_code=400, detail="Le fichier ne contient aucune ligne.")
    if len(df) > 500_000:
        raise HTTPException(status_code=400, detail="Fichier trop volumineux (> 500 000 lignes).")

    report = _validate_dataset_against_template(list(df.columns), data_template, _dataframe_dtype_kinds(df), n_rows=len(df))

    # Stockage (CSV canonique) + métadonnées ; le précédent dataset devient inactif.
    safe = Path(filename).name or "dataset.csv"
    stored_path = None
    try:
        ddir = MODEL_DATA_DIR / scenario_id / "model"
        ddir.mkdir(parents=True, exist_ok=True)
        stored_path = str(ddir / f"{int(datetime.now(timezone.utc).timestamp())}_{safe}.csv")
        df.to_csv(stored_path, index=False)
    except Exception as e:
        logger.error(f"Stockage dataset {scenario_id}: {e}", exc_info=True)

    # Échec de stockage → NE PAS activer un dataset au chemin NULL (il paraîtrait
    # « prêt » mais tout entraînement/monitoring échouerait faute de fichier).
    if stored_path is None:
        raise HTTPException(status_code=500,
                            detail="Échec du stockage du dataset (espace disque ?). Réessayez.")

    with engine.begin() as conn:
        conn.execute(text(
            "UPDATE scenario_model_dataset SET is_active = FALSE WHERE scenario_id = :sid AND is_active = TRUE"
        ), {"sid": scenario_id})
        new_id = conn.execute(text("""
            INSERT INTO scenario_model_dataset
                (scenario_id, filename, stored_path, n_rows, n_cols, columns_json, validation_json, is_active, is_synthetic)
            VALUES (:sid, :fn, :sp, :nr, :nc, CAST(:cj AS jsonb), CAST(:vj AS jsonb), TRUE, FALSE)
            RETURNING id
        """), {
            "sid": scenario_id, "fn": filename, "sp": stored_path,
            "nr": int(len(df)), "nc": int(len(df.columns)),
            "cj": json.dumps([str(c) for c in df.columns]),
            "vj": json.dumps(report),
        }).scalar()

    return {
        "status": "stored",
        "dataset_id": new_id,
        "scenario_id": scenario_id,
        "filename": filename,
        "n_rows": int(len(df)),
        "n_cols": int(len(df.columns)),
        "stored": stored_path is not None,
        "validation": report,
        "training_started": _maybe_autotrain(scenario_id, report) if auto_train else False,
    }


class AutoFetchMapping(BaseModel):
    template_column: str            # colonne du data_template à remplir
    connector_id: str               # ex. open-meteo-weather, eawag-wastewater
    connector_variable: str         # variable fournie par le connecteur (ex. temp_mean)


class AutoFetchIn(BaseModel):
    region: str | None = None       # alias Romandie OU lat+lon
    lat: float | None = None
    lon: float | None = None
    start_date: str = Field(..., min_length=8)
    end_date: str = Field(..., min_length=8)
    frequency: str = Field(default="W")     # W (hebdo, défaut épidémio) | D | MS
    # Pas de min_length : un scénario dont TOUTES les colonnes dérivent du sous-modèle
    # SEIR (source="seir") n'a aucun mapping à fournir - le code ci-dessous les ajoute
    # automatiquement. Exiger au moins un mapping rejetait à la porte le cas que la
    # fonction est précisément écrite pour traiter.
    mappings: list[AutoFetchMapping] = Field(default_factory=list)
    auto_train: bool = False


@app.post("/scenarios/{scenario_id}/model/data/auto-fetch")
def auto_fetch_model_dataset(
    scenario_id: str, payload: AutoFetchIn, _: None = Depends(require_api_key),
) -> dict[str, Any]:
    """Phase 2 - assemble le dataset du modèle depuis des connecteurs de données
    PUBLIQUES au lieu d'un upload. Chaque mapping relie une colonne du data_template
    à (connecteur, variable) ; on récupère chaque connecteur, on aligne sur une grille
    de dates commune (frequency) et on joint, on stocke le dataset actif, puis on
    renvoie la couverture (colonnes remplies vs encore requises) + un aperçu."""
    from .seir import _scenario_seed  # lazy: seir is loaded after this module
    import pandas as pd
    from datetime import datetime, timezone
    import data_connectors

    spec = _get_model_spec(scenario_id)
    if not spec:
        raise HTTPException(status_code=400,
                            detail="Aucune spécification de modèle. Générez puis validez les Variables & Modèle d'abord.")
    data_template = spec.get("data_template") or {}
    if not data_template.get("columns"):
        raise HTTPException(status_code=400, detail="data_template absent du model_spec. Relancez la génération des variables.")

    base_params = {"start_date": payload.start_date, "end_date": payload.end_date}
    if payload.region:
        base_params["region"] = payload.region
    if payload.lat is not None:
        base_params["lat"] = payload.lat
    if payload.lon is not None:
        base_params["lon"] = payload.lon

    # Mappings = ceux fournis par l'utilisateur + AUTO pour les colonnes dérivées du
    # sous-modèle SEIR (source="seir") : ces features sont remplies par le connecteur
    # SEIR sans que l'utilisateur ait à les mapper à la main.
    mapping_dicts = [m.model_dump() for m in payload.mappings]
    _mapped_cols = {m["template_column"] for m in mapping_dicts}
    for _c in (data_template.get("columns") or []):
        if _c.get("source") == "seir" and _c.get("seir_column") and _c["name"] not in _mapped_cols:
            mapping_dicts.append({
                "template_column": _c["name"],
                "connector_id": data_connectors.SEIR_CONNECTOR_ID,
                "connector_variable": _c["seir_column"],
            })
    if not mapping_dicts:
        raise HTTPException(status_code=422, detail=(
            "Rien à récupérer : aucun mapping fourni et aucune colonne dérivée du "
            "sous-modèle SEIR dans le data_template."))

    frames: dict[str, Any] = {}
    fetch_errors: dict[str, str] = {}
    for cid in {m["connector_id"] for m in mapping_dicts}:
        if cid not in data_connectors.CONNECTORS:
            fetch_errors[cid] = "connecteur inconnu"
            continue
        _cparams = dict(base_params)
        if cid == data_connectors.SEIR_CONNECTOR_ID:
            # Le connecteur SEIR n'est PAS géographique : on lui injecte les paramètres
            # épidémiologiques EXTRAITS du corpus (bloc model_spec) du scénario courant,
            # afin que la série d'incidence/prévalence qui alimente le modèle soit
            # paramétrée par la littérature de CE scénario. Population d'exposition + cas
            # initiaux DÉRIVÉS de la géographie du corpus (au lieu d'un 1e6 fixe).
            _cparams["epidemic_parameters"] = spec.get("epidemic_parameters") or {}
            _seir_pop, _seir_i0, _ = _scenario_seed(scenario_id)
            _cparams["population"] = _seir_pop
            _cparams["initial_infected"] = _seir_i0
        try:
            rows = data_connectors.fetch_series(cid, _cparams)
        except Exception as _e:
            fetch_errors[cid] = str(_e)
            continue
        if not rows:
            fetch_errors[cid] = "aucune donnée renvoyée pour cette zone/période"
            continue
        frames[cid] = pd.DataFrame(rows)

    _dt_col = next((c["name"] for c in data_template.get("columns", []) if c.get("dtype") == "datetime"), None)
    freq = (payload.frequency or "W").strip() or "W"
    assembled, filled = _assemble_connector_frames(
        frames, mapping_dicts, freq, _dt_col)
    if assembled is None or assembled.empty:
        raise HTTPException(status_code=422, detail={
            "message": "Aucune donnée assemblée depuis les connecteurs (mappings/fenêtre à vérifier).",
            "fetch_errors": fetch_errors,
        })

    # Colonnes DEMANDÉES mais absentes du résultat. `_assemble_connector_frames` saute
    # silencieusement un mapping dont la variable manque, si bien que la réponse
    # annonçait « stored » sans dire qu'une colonne - typiquement la colonne dérivée du
    # SEIR - n'avait pas été remplie ; le rapport de validation la rangeait alors dans
    # `missing_seir`, dont le sens est « sera auto-remplie », c'est-à-dire l'inverse de
    # ce qui venait de se produire. On nomme ici ce qui a été écarté ET pourquoi.
    _filled = set(filled)
    dropped_columns = [
        {"template_column": m["template_column"], "connector_id": m["connector_id"],
         "connector_variable": m["connector_variable"],
         "reason": fetch_errors.get(m["connector_id"])
                   or "variable absente des données renvoyées par le connecteur"}
        for m in mapping_dicts if m["template_column"] not in _filled
    ]

    # ── FUSION avec le dataset actif, au lieu de le REMPLACER ────────────────────
    # Un connecteur ne peut PAS fournir la variable à prédire : elle vient des données
    # propres de l'utilisateur (passages aux urgences, appels…). Or l'auto-récupération
    # désactivait le dataset uploadé et activait celui, purement covariables, qu'elle
    # venait d'assembler : le scénario perdait sa colonne d'outcome, et le rapport de
    # validation - calculé sur les seules colonnes assemblées - annonçait pourtant
    # `still_needed_user_columns: []`, c'est-à-dire « rien ne manque ». Silencieux et faux.
    # On fusionne donc sur la colonne de dates ; les colonnes effectivement récupérées
    # écrasent leurs homonymes (l'utilisateur vient de les demander), toutes les autres
    # colonnes du dataset précédent sont CONSERVÉES.
    merged_from, preserved_cols = None, []
    if _dt_col and _dt_col in assembled.columns:
        try:
            with engine.begin() as _c:
                _prev = _c.execute(text(
                    "SELECT id, stored_path FROM scenario_model_dataset "
                    "WHERE scenario_id = :sid AND is_active = TRUE LIMIT 1"),
                    {"sid": scenario_id}).mappings().first()
            if _prev and _prev["stored_path"] and os.path.exists(_prev["stored_path"]):
                _pdf = pd.read_csv(_prev["stored_path"])
                if _dt_col in _pdf.columns and not _pdf.empty:
                    _keep = [c for c in _pdf.columns
                             if c == _dt_col or c not in assembled.columns]
                    if len(_keep) > 1:            # au moins une colonne à préserver
                        _pdf = _pdf[_keep].copy()
                        # Clés de jointure comparables des deux côtés (dates ISO).
                        _pdf[_dt_col] = pd.to_datetime(_pdf[_dt_col], errors="coerce")
                        _adf = assembled.copy()
                        _adf[_dt_col] = pd.to_datetime(_adf[_dt_col], errors="coerce")
                        _pdf = _pdf.dropna(subset=[_dt_col])
                        _merged = _adf.merge(_pdf, on=_dt_col, how="outer").sort_values(_dt_col)
                        _merged[_dt_col] = _merged[_dt_col].dt.strftime("%Y-%m-%d")
                        assembled = _merged.reset_index(drop=True)
                        merged_from = int(_prev["id"])
                        preserved_cols = [c for c in _keep if c != _dt_col]
        except Exception as _e:                   # la fusion ne doit jamais casser le fetch
            logger.warning(f"auto-fetch {scenario_id}: fusion avec le dataset actif ignorée: {_e}")

    report = _validate_dataset_against_template(
        list(assembled.columns), data_template, _dataframe_dtype_kinds(assembled), n_rows=len(assembled))

    stored_path = None
    try:
        ddir = MODEL_DATA_DIR / scenario_id / "model"
        ddir.mkdir(parents=True, exist_ok=True)
        stored_path = str(ddir / f"{int(datetime.now(timezone.utc).timestamp())}_autofetch.csv")
        assembled.to_csv(stored_path, index=False)
    except Exception as _e:
        logger.error(f"Stockage auto-fetch {scenario_id}: {_e}", exc_info=True)
    if stored_path is None:
        raise HTTPException(status_code=500, detail="Échec du stockage du dataset auto-récupéré.")

    with engine.begin() as conn:
        conn.execute(text(
            "UPDATE scenario_model_dataset SET is_active = FALSE WHERE scenario_id = :sid AND is_active = TRUE"
        ), {"sid": scenario_id})
        new_id = conn.execute(text("""
            INSERT INTO scenario_model_dataset
                (scenario_id, filename, stored_path, n_rows, n_cols, columns_json, validation_json, is_active, is_synthetic)
            VALUES (:sid, :fn, :sp, :nr, :nc, CAST(:cj AS jsonb), CAST(:vj AS jsonb), TRUE, FALSE)
            RETURNING id
        """), {
            "sid": scenario_id, "fn": "auto_fetch.csv", "sp": stored_path,
            "nr": int(len(assembled)), "nc": int(len(assembled.columns)),
            "cj": json.dumps([str(c) for c in assembled.columns]),
            "vj": json.dumps(report),
        }).scalar()

    return {
        "status": "stored",
        "dataset_id": new_id,
        "scenario_id": scenario_id,
        "source": "auto_fetch",
        "frequency": freq,
        "n_rows": int(len(assembled)),
        "n_cols": int(len(assembled.columns)),
        "filled_columns": filled,
        # Traçabilité de la fusion : quel dataset a été repris et quelles colonnes
        # (dont l'outcome de l'utilisateur) ont été conservées.
        "merged_from_dataset_id": merged_from,
        "preserved_columns": preserved_cols,
        # Colonnes demandées et NON remplies, avec la cause réelle (cf. plus haut).
        "dropped_columns": dropped_columns,
        "still_needed_user_columns": report.get("missing_user", []),
        "fetch_errors": fetch_errors,
        "validation": report,
        "preview": json.loads(assembled.head(8).to_json(orient="records")),
        "training_started": _maybe_autotrain(scenario_id, report) if payload.auto_train else False,
    }


@app.get("/scenarios/{scenario_id}/model/data")
def get_model_dataset(scenario_id: str, _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Résumé du dataset actif d'un scénario + état de préparation à l'entraînement.
    Authentifié : le schéma des données uploadées par l'utilisateur ne doit pas être
    lisible publiquement par simple connaissance de l'ID de scénario."""
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT id, filename, n_rows, n_cols, columns_json, validation_json, is_synthetic, created_at
            FROM scenario_model_dataset
            WHERE scenario_id = :sid AND is_active = TRUE
            ORDER BY created_at DESC LIMIT 1
        """), {"sid": scenario_id}).mappings().first()

    if not row:
        return {"status": "empty", "message": "Aucun jeu de données branché. Uploadez un CSV/XLSX correspondant au data_template."}

    return {
        "status": "ready",
        "dataset_id": row["id"],
        "filename": row["filename"],
        "n_rows": row["n_rows"],
        "n_cols": row["n_cols"],
        "columns": row["columns_json"],
        "validation": row["validation_json"],
        "is_synthetic": bool(row["is_synthetic"]),
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
    }


@app.post("/scenarios/{scenario_id}/model/data/synthetic")
def generate_synthetic_model_dataset(scenario_id: str, n_rows: int = 400,
                                     auto_train: bool = True,
                                     _: None = Depends(require_api_key)) -> dict[str, Any]:
    """
    Génère un dataset SYNTHÉTIQUE cohérent avec le data_template du spec et le
    branche comme dataset actif. Permet de faire tourner un vrai modèle de
    démonstration (entraînable immédiatement) sans données réelles - utile pour
    transformer un scénario en démo « modèle en ligne ». Généralisable à tout scénario.
    """
    from datetime import datetime, timezone
    import model_trainer

    spec = _get_model_spec(scenario_id)
    if not spec:
        raise HTTPException(status_code=400,
                            detail="Aucune spécification de modèle. Générez puis validez les Variables & Modèle d'abord.")
    if not (spec.get("data_template") or {}).get("columns"):
        raise HTTPException(status_code=400, detail="data_template absent du model_spec. Relancez la génération des variables.")

    n_rows = max(50, min(int(n_rows or 400), 5000))
    try:
        df = model_trainer.generate_synthetic_dataset(spec, n_rows=n_rows)
    except Exception as e:
        logger.error(f"Synthetic gen {scenario_id}: {e}", exc_info=True)
        # Message générique côté client : l'exception (chemins/internes) est journalisée
        # serveur, pas renvoyée à l'appelant.
        raise HTTPException(status_code=500, detail="Génération synthétique impossible.")

    report = _validate_dataset_against_template(list(df.columns), spec["data_template"], _dataframe_dtype_kinds(df))

    stored_path = None
    try:
        ddir = MODEL_DATA_DIR / scenario_id / "model"
        ddir.mkdir(parents=True, exist_ok=True)
        stored_path = str(ddir / f"{int(datetime.now(timezone.utc).timestamp())}_synthetic.csv")
        df.to_csv(stored_path, index=False)
    except Exception as e:
        logger.error(f"Stockage dataset synthétique {scenario_id}: {e}", exc_info=True)

    with engine.begin() as conn:
        conn.execute(text(
            "UPDATE scenario_model_dataset SET is_active = FALSE WHERE scenario_id = :sid AND is_active = TRUE"
        ), {"sid": scenario_id})
        new_id = conn.execute(text("""
            INSERT INTO scenario_model_dataset
                (scenario_id, filename, stored_path, n_rows, n_cols, columns_json, validation_json, is_active, is_synthetic)
            VALUES (:sid, :fn, :sp, :nr, :nc, CAST(:cj AS jsonb), CAST(:vj AS jsonb), TRUE, TRUE)
            RETURNING id
        """), {
            "sid": scenario_id, "fn": f"synthetic_{n_rows}.csv", "sp": stored_path,
            "nr": int(len(df)), "nc": int(len(df.columns)),
            "cj": json.dumps([str(c) for c in df.columns]),
            "vj": json.dumps(report),
        }).scalar()

    return {
        "status": "stored",
        "synthetic": True,
        "dataset_id": new_id,
        "scenario_id": scenario_id,
        "n_rows": int(len(df)),
        "n_cols": int(len(df.columns)),
        "columns": [str(c) for c in df.columns],
        "stored": stored_path is not None,
        "validation": report,
        "training_started": _maybe_autotrain(scenario_id, report) if auto_train else False,
        "note": "Données synthétiques de démonstration - à remplacer par des données réelles pour un usage opérationnel.",
    }
