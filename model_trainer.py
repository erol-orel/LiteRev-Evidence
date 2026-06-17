"""Phase 3 — entraînement réel du modèle (scikit-learn + Optuna).

La littérature fournit le `model_spec` (outcome, variables explicatives,
algorithme) ; les données d'entraînement viennent de l'utilisateur (+ flux
publics). Ce module transforme ce contrat en un vrai modèle :

  préprocessing → recherche d'hyperparamètres par validation croisée (Optuna)
  → réentraînement → évaluation sur holdout → importances des variables.

Fonctions pures (DataFrame + spec → résultats), donc testables hors API.
Le pipeline scikit-learn entraîné est renvoyé pour être sérialisé (joblib) par
l'appelant.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

CLF_FAMILIES = {"gradient_boosting", "random_forest", "logistic_regression", "svm", "mlp", "knn"}
REG_FAMILIES = {"gradient_boosting", "random_forest", "linear_regression", "elasticnet", "svm", "mlp", "knn"}


def _norm(s: Any) -> str:
    return str(s if s is not None else "").strip().lower()


def _effective_family(family: str, task_type: str) -> str:
    """Ramène la famille demandée vers une famille valide pour la tâche."""
    family = (family or "").strip().lower()
    if task_type in ("regression", "count"):
        if family == "logistic_regression":
            return "linear_regression"
        return family if family in REG_FAMILIES else "gradient_boosting"
    # classification
    if family in ("linear_regression", "elasticnet"):
        return "logistic_regression"
    return family if family in CLF_FAMILIES else "gradient_boosting"


def resolve_columns(df, spec: dict):
    """
    Aligne le DataFrame sur le spec : trouve la cible et les variables présentes
    (matching insensible à la casse), renomme en machine_name. Les variables du
    spec absentes du fichier (ex: colonne publique non encore récupérée) sont
    simplement ignorées.
    """
    features = spec.get("features") or []
    outcome = spec.get("outcome") or {}
    target = outcome.get("machine_name")

    norm_map: dict[str, Any] = {}
    for c in df.columns:
        norm_map.setdefault(_norm(c), c)

    tcol = norm_map.get(_norm(target))
    if tcol is None:
        raise ValueError(f"Colonne cible '{target}' absente du dataset.")

    used, rename = [], {}
    for f in features:
        mn = f.get("machine_name")
        col = norm_map.get(_norm(mn))
        if col is None:
            continue
        used.append({"machine_name": mn, "dtype": f.get("dtype", "float"), "source_col": col})
        rename[col] = mn
    if not used:
        raise ValueError("Aucune variable explicative du spec n'est présente dans le dataset.")

    X = df[[u["source_col"] for u in used]].rename(columns=rename)
    y = df[tcol]
    return X, y, used, target


def build_preprocessor(used: list[dict]):
    """ColumnTransformer : numériques (impute médiane + scale), catégorielles (impute + one-hot)."""
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder

    numeric = [u["machine_name"] for u in used if u["dtype"] in ("float", "int", "bool")]
    categorical = [u["machine_name"] for u in used if u["dtype"] == "category"]
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))
    if not transformers:  # ni num ni cat reconnus -> tout en numérique
        transformers.append(("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
        ]), [u["machine_name"] for u in used]))
    return ColumnTransformer(transformers, remainder="drop")


def suggest_params(trial, family: str) -> dict:
    """Espace de recherche Optuna par famille."""
    if family == "gradient_boosting":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
        }
    if family == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        }
    if family == "logistic_regression":
        return {"C": trial.suggest_float("C", 1e-3, 1e2, log=True)}
    if family == "elasticnet":
        return {
            "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
        }
    if family == "svm":
        return {
            "C": trial.suggest_float("C", 1e-2, 1e2, log=True),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        }
    if family == "mlp":
        return {
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
            "hidden": trial.suggest_categorical("hidden", ["64", "128", "64_32"]),
        }
    if family == "knn":
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 25),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        }
    return {}  # linear_regression : rien à régler


def _has_tunable(family: str) -> bool:
    return family != "linear_regression"


def estimator_from_params(family: str, task_type: str, params: dict, random_state: int = 42):
    from sklearn.ensemble import (
        GradientBoostingClassifier, GradientBoostingRegressor,
        RandomForestClassifier, RandomForestRegressor,
    )
    from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet
    from sklearn.svm import SVC, SVR
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

    p = dict(params or {})
    clf = task_type == "classification"

    if family == "gradient_boosting":
        return (GradientBoostingClassifier if clf else GradientBoostingRegressor)(random_state=random_state, **p)
    if family == "random_forest":
        return (RandomForestClassifier if clf else RandomForestRegressor)(random_state=random_state, n_jobs=1, **p)
    if family == "logistic_regression":
        return LogisticRegression(max_iter=1000, **p)
    if family == "linear_regression":
        return LinearRegression(**p)
    if family == "elasticnet":
        return ElasticNet(random_state=random_state, **p)
    if family == "svm":
        return SVC(probability=True, random_state=random_state, **p) if clf else SVR(**p)
    if family == "mlp":
        hidden = {"64": (64,), "128": (128,), "64_32": (64, 32)}.get(p.pop("hidden", "64"), (64,))
        return (MLPClassifier if clf else MLPRegressor)(
            hidden_layer_sizes=hidden, max_iter=400, random_state=random_state, **p)
    if family == "knn":
        return (KNeighborsClassifier if clf else KNeighborsRegressor)(**p)
    return (GradientBoostingClassifier if clf else GradientBoostingRegressor)(random_state=random_state)


def _scoring(metric: str | None, task_type: str, n_classes: int | None) -> str:
    metric = (metric or "").strip().lower()
    if task_type in ("regression", "count"):
        return {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error",
                "r2": "r2"}.get(metric, "neg_root_mean_squared_error")
    # classification
    multi = (n_classes or 2) > 2
    if metric == "average_precision" and not multi:
        return "average_precision"
    if metric == "accuracy":
        return "accuracy"
    return "roc_auc_ovr" if multi else "roc_auc"


def _make_cv(strategy: str, folds: int, task_type: str, random_state: int = 42):
    from sklearn.model_selection import StratifiedKFold, KFold, TimeSeriesSplit
    if strategy == "timeseries":
        return TimeSeriesSplit(n_splits=folds)
    if task_type == "classification":
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    return KFold(n_splits=folds, shuffle=True, random_state=random_state)


def _evaluate(pipe, Xte, yte, task_type: str, classes: list | None) -> dict:
    import numpy as np
    from sklearn import metrics as M

    out: dict[str, Any] = {}
    if task_type in ("regression", "count"):
        pred = pipe.predict(Xte)
        out["rmse"] = float(np.sqrt(M.mean_squared_error(yte, pred)))
        out["mae"] = float(M.mean_absolute_error(yte, pred))
        out["r2"] = float(M.r2_score(yte, pred))
        return out

    pred = pipe.predict(Xte)
    out["accuracy"] = float(M.accuracy_score(yte, pred))
    multi = (len(classes) if classes else 2) > 2
    out["f1"] = float(M.f1_score(yte, pred, average="macro" if multi else "binary"))
    try:
        proba = pipe.predict_proba(Xte)
        if multi:
            out["roc_auc"] = float(M.roc_auc_score(yte, proba, multi_class="ovr", average="macro"))
        else:
            out["roc_auc"] = float(M.roc_auc_score(yte, proba[:, 1]))
            out["average_precision"] = float(M.average_precision_score(yte, proba[:, 1]))
    except Exception as e:  # proba indispo ou une seule classe dans le test
        logger.warning(f"AUC non calculable: {e}")
    return out


def _feature_importances(pipe, top: int = 20) -> list[dict]:
    import numpy as np
    try:
        pre, est = pipe.named_steps["pre"], pipe.named_steps["est"]
        names = list(pre.get_feature_names_out())
        if hasattr(est, "feature_importances_"):
            vals = np.asarray(est.feature_importances_, dtype=float)
        elif hasattr(est, "coef_"):
            coef = np.asarray(est.coef_, dtype=float)
            vals = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
        else:
            return []
        pairs = sorted(zip(names, vals), key=lambda kv: kv[1], reverse=True)[:top]
        return [{"feature": n, "importance": float(v)} for n, v in pairs]
    except Exception as e:
        logger.warning(f"Importances indisponibles: {e}")
        return []


def train_model(df, spec: dict, n_trials: int = 25, random_state: int = 42, test_size: float = 0.2) -> dict:
    """
    Entraîne un modèle réel à partir du dataset et du model_spec.
    Renvoie un dict de résultats incluant `pipeline` (Pipeline scikit-learn
    entraîné, à sérialiser par l'appelant). Lève ValueError si les données sont
    insuffisantes ou la tâche non supportée.
    """
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    outcome = spec.get("outcome") or {}
    task_type = (outcome.get("task_type") or "classification").strip().lower()
    if task_type == "survival":
        raise ValueError("task_type 'survival' non supporté en Phase 3 (nécessiterait lifelines).")

    algo = spec.get("algorithm") or {}
    family = _effective_family(algo.get("family", "gradient_boosting"), task_type)
    metric = algo.get("metric")
    cv_conf = algo.get("cv") or {}
    strategy = cv_conf.get("strategy", "kfold")
    folds = int(cv_conf.get("folds", 5) or 5)

    X, y, used, target = resolve_columns(df, spec)

    # Lignes avec cible manquante : inutilisables.
    mask = y.notna()
    X, y = X.loc[mask.values].reset_index(drop=True), y.loc[mask.values].reset_index(drop=True)

    classes = None
    if task_type == "classification":
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y.astype(str)))
        classes = [str(c) for c in le.classes_]
        if len(classes) < 2:
            raise ValueError("La cible n'a qu'une seule classe : impossible d'entraîner un classifieur.")
        min_class = int(pd.Series(y).value_counts().min())
        folds = max(2, min(folds, min_class))
    else:
        y = pd.to_numeric(y, errors="coerce")
        keep = y.notna()
        X, y = X.loc[keep.values].reset_index(drop=True), y.loc[keep.values].reset_index(drop=True)

    if len(X) < 20:
        raise ValueError(f"Pas assez de lignes exploitables ({len(X)}). Minimum 20.")

    scoring = _scoring(metric, task_type, n_classes=(len(classes) if classes else None))
    is_ts = strategy == "timeseries"
    stratify = y if (task_type == "classification" and not is_ts) else None
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        shuffle=not is_ts, stratify=stratify,
    )

    cv = _make_cv(strategy, folds, task_type, random_state)
    pre = build_preprocessor(used)

    def objective(trial):
        est = estimator_from_params(family, task_type, suggest_params(trial, family), random_state)
        pipe = Pipeline([("pre", pre), ("est", est)])
        scores = cross_val_score(pipe, Xtr, ytr, scoring=scoring, cv=cv, n_jobs=1, error_score="raise")
        return float(np.mean(scores))

    best_params: dict = {}
    if _has_tunable(family):
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=random_state))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_params, cv_best = dict(study.best_params), float(study.best_value)
    else:
        est0 = estimator_from_params(family, task_type, {}, random_state)
        cv_best = float(np.mean(cross_val_score(
            Pipeline([("pre", pre), ("est", est0)]), Xtr, ytr, scoring=scoring, cv=cv, n_jobs=1)))

    final = Pipeline([("pre", pre), ("est", estimator_from_params(family, task_type, best_params, random_state))])
    final.fit(Xtr, ytr)

    metrics = _evaluate(final, Xte, yte, task_type, classes)
    metrics[f"cv_{scoring}"] = cv_best

    return {
        "task_type": task_type,
        "family": family,
        "metric": metric,
        "scoring": scoring,
        "cv_strategy": strategy,
        "cv_folds": folds,
        "n_total": int(len(X)),
        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte)),
        "features_used": [u["machine_name"] for u in used],
        "target": target,
        "classes": classes,
        "best_params": best_params,
        "metrics": metrics,
        "feature_importances": _feature_importances(final),
        "pipeline": final,
    }
