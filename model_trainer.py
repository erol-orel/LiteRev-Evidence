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

CLF_FAMILIES = {"gradient_boosting", "lightgbm", "xgboost", "random_forest", "logistic_regression", "svm", "mlp", "knn"}
REG_FAMILIES = {"gradient_boosting", "lightgbm", "xgboost", "random_forest", "linear_regression", "elasticnet", "svm", "mlp", "knn"}

# Familles à gradient boosting « fortes » pour données tabulaires (préférées au
# GB scikit-learn quand le paquet est présent). Import paresseux : l'absence du
# paquet fait retomber sur "gradient_boosting" (cf. estimator_from_params).
_BOOSTED_FAMILIES = {"lightgbm", "xgboost"}

# Familles de PRÉVISION de série temporelle : hors du flux tabulaire sklearn
# (pas de ColumnTransformer→estimateur). train_model les route vers
# train_timeseries_model. Nécessitent une colonne datetime + une cible numérique.
_TS_FAMILIES = {"prophet", "sarimax"}


def _has_package(name: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(name) is not None


def _norm(s: Any) -> str:
    """Normalise un nom de COLONNE pour l'appariement fichier↔spec : repli des accents
    (NFKD), minuscules, et tout groupe de caractères non alphanumériques → « _ ».
    Un en-tête réel « Température max (J1) » s'apparie ainsi au machine_name
    « temperature_max_j1 » (mêmes règles que _slug_identifier à la génération), au
    lieu de l'ancien strip().lower() qui échouait sur accents/espaces/ponctuation."""
    import unicodedata as _ud
    import re as _re
    s = _ud.normalize("NFKD", str(s if s is not None else "")).encode("ascii", "ignore").decode()
    return _re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def generate_synthetic_dataset(spec: dict, n_rows: int = 400, seed: int = 42):
    """
    Génère un dataset synthétique cohérent avec le data_template du spec, pour
    faire tourner un VRAI modèle de démonstration (sans données réelles).
    La cible dépend des variables (signal réel) afin que les métriques soient
    significatives. Toutes les colonnes du template (user + public) sont créées
    pour un dataset auto-suffisant. Pur/testable.
    """
    import numpy as np
    import pandas as pd

    rng = np.random.RandomState(seed)
    dt = spec.get("data_template") or {}
    cols = dt.get("columns") or []
    outcome = spec.get("outcome") or {}
    task = (outcome.get("task_type") or "classification").strip().lower()
    target_name = dt.get("target_column") or outcome.get("machine_name") or "target"

    data: dict[str, Any] = {}
    signal: list = []  # composantes numériques pour construire une cible corrélée
    for c in cols:
        if c.get("role") != "feature":
            continue
        name, d = c.get("name"), c.get("dtype", "float")
        if d == "bool":
            v = rng.binomial(1, 0.4, n_rows)
            data[name] = v.astype(int); signal.append(v.astype(float))
        elif d == "int":
            v = rng.poisson(5, n_rows)
            data[name] = v; signal.append(v.astype(float))
        elif d == "category":
            levels = [f"{(name or 'cat')[:10]}_{k}" for k in range(3)]
            idx = rng.randint(0, len(levels), n_rows)
            data[name] = [levels[i] for i in idx]; signal.append(idx.astype(float))
        elif d == "datetime":
            base = pd.Timestamp("2024-01-01")
            data[name] = [(base + pd.Timedelta(days=int(i))).date().isoformat() for i in range(n_rows)]
        else:  # float
            v = rng.normal(0.0, 1.0, n_rows)
            data[name] = v; signal.append(v)

    if signal:
        M = np.vstack(signal).T
        w = rng.normal(0.0, 1.0, M.shape[1])
        lin = (M * w).sum(axis=1)
        lin = (lin - lin.mean()) / (lin.std() + 1e-9)
    else:
        lin = rng.normal(0.0, 1.0, n_rows)

    if task in ("regression", "count"):
        y = lin * 10.0 + rng.normal(0.0, 2.0, n_rows)
        if task == "count":
            y = np.clip(np.round(y - y.min()), 0, None).astype(int)
        data[target_name] = y
    else:
        # Échelle du logit > 1 pour des classes nettement séparables (démo lisible).
        prob = 1.0 / (1.0 + np.exp(-2.5 * lin))
        yb = (rng.rand(n_rows) < prob).astype(int)
        pos = outcome.get("positive_class") or "oui"
        neg = "non" if str(pos) != "non" else "autre"
        data[target_name] = [pos if v == 1 else neg for v in yb]

    return pd.DataFrame(data)


def _effective_family(family: str, task_type: str) -> str:
    """Ramène la famille demandée vers une famille valide pour la tâche."""
    family = (family or "").strip().lower()
    # Paquet de boosting absent → repli HONNÊTE sur le GB scikit-learn (sinon le run
    # rapporterait "lightgbm"/"xgboost" tout en entraînant un GradientBoosting).
    if family in _BOOSTED_FAMILIES and not _has_package(family):
        family = "gradient_boosting"
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
    if family in ("lightgbm", "xgboost"):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
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

    # Données médicales souvent déséquilibrées (événement rare) : on pondère les
    # classes par défaut pour les familles qui le supportent (RF, régression
    # logistique, SVM). GradientBoosting/MLP/KNN n'exposent pas class_weight et
    # restent inchangés. setdefault → un réglage explicite du spec a la priorité.
    if clf and family in ("random_forest", "logistic_regression", "svm"):
        p.setdefault("class_weight", "balanced")

    if family == "lightgbm":
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor
            if clf:
                p.setdefault("class_weight", "balanced")   # LGBM gère le déséquilibre
            return (LGBMClassifier if clf else LGBMRegressor)(
                random_state=random_state, n_jobs=1, verbosity=-1, **p)
        except Exception as _e:  # paquet absent → repli GB scikit-learn
            logger.warning(f"lightgbm indisponible ({_e}) → repli gradient_boosting")
            family = "gradient_boosting"
            p.pop("colsample_bytree", None)
    if family == "xgboost":
        try:
            from xgboost import XGBClassifier, XGBRegressor
            return (XGBClassifier if clf else XGBRegressor)(
                random_state=random_state, n_jobs=1, verbosity=0, **p)
        except Exception as _e:
            logger.warning(f"xgboost indisponible ({_e}) → repli gradient_boosting")
            family = "gradient_boosting"
            p.pop("colsample_bytree", None)
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


def _importances_by_variable(pipe, used: list[dict]) -> dict:
    """
    Agrège les importances par VARIABLE source (machine_name), en repliant les
    colonnes transformées (num__x, cat__x_niveau) sur leur variable d'origine.
    Normalisé (somme = 1). Permet d'afficher les mêmes noms que l'onglet
    Variables (pas les noms transformés du pipeline).
    """
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
            return {}
        # Plus long machine_name d'abord -> évite qu'"age" capte "age_group_*".
        mnames = sorted([u["machine_name"] for u in used], key=len, reverse=True)
        agg = {u["machine_name"]: 0.0 for u in used}
        for n, v in zip(names, vals):
            base = n.split("__", 1)[1] if "__" in n else n
            for m in mnames:
                if base == m or base.startswith(m + "_"):
                    agg[m] += float(v)
                    break
        total = sum(agg.values()) or 1.0
        return {m: agg[m] / total for m in agg}
    except Exception as e:
        logger.warning(f"importances_by_variable indisponible: {e}")
        return {}


def regression_diagnostics(pipe, X, y, used: list[dict]) -> dict:
    """Diagnostics des hypothèses de la régression (Gauss-Markov / OLS), calculés
    sur les résidus d'ENTRAÎNEMENT. Purement informatif — n'altère pas le modèle.

      - multicolinéarité   : VIF par variable numérique (>5 attention, >10 forte) ;
      - homoscédasticité   : test de Breusch-Pagan (p<0.05 = hétéroscédasticité) ;
      - autocorrélation    : Durbin-Watson (~2 = ok ; <1.5 ou >2.5 = autocorrélation) ;
      - normalité résidus  : Shapiro-Wilk (n≤5000) sinon Jarque-Bera (p<0.05 = non normal).

    Ces hypothèses concernent l'inférence des modèles LINÉAIRES (OLS). Pour les
    modèles à arbres/non linéaires elles ne s'appliquent pas au sens strict, mais les
    résidus restent informatifs — d'où le champ `applies` (True pour linéaire)."""
    import numpy as np
    from scipy import stats as _st

    out: dict[str, Any] = {"checks": [], "applies": None}
    try:
        pred = np.asarray(pipe.predict(X), dtype=float)
        yv = np.asarray(y, dtype=float)
        resid = yv - pred
        n = int(len(resid))
        if n < 8 or float(np.sum(resid ** 2)) < 1e-12:
            return {"checks": [], "applies": None, "note": "résidus insuffisants pour les diagnostics"}

        pre = pipe.named_steps["pre"]
        est = pipe.named_steps["est"]
        out["applies"] = est.__class__.__name__ in ("LinearRegression", "ElasticNet")
        Z = pre.transform(X)
        try:
            Z = Z.toarray()
        except Exception:
            Z = np.asarray(Z, dtype=float)
        names = list(pre.get_feature_names_out())

        # 1) Autocorrélation — Durbin-Watson
        dw = float(np.sum(np.diff(resid) ** 2) / (np.sum(resid ** 2) + 1e-12))
        out["checks"].append({
            "key": "autocorrelation", "name": "Autocorrélation (Durbin-Watson)",
            "statistic": round(dw, 3),
            "status": "ok" if 1.5 <= dw <= 2.5 else "warn",
            "detail": "≈2 = pas d'autocorrélation ; <1.5 ou >2.5 = résidus corrélés (souvent série temporelle mal ordonnée).",
        })

        # 2) Normalité des résidus — Shapiro-Wilk / Jarque-Bera
        try:
            if n <= 5000:
                _stat, p_norm = _st.shapiro(resid)
                tname = "Shapiro-Wilk"
            else:
                _stat, p_norm = _st.jarque_bera(resid)
                tname = "Jarque-Bera"
            out["checks"].append({
                "key": "normality", "name": f"Normalité des résidus ({tname})",
                "p_value": round(float(p_norm), 4),
                "status": "ok" if p_norm >= 0.05 else "warn",
                "detail": "p≥0.05 = résidus compatibles avec une loi normale (hypothèse OLS pour l'inférence).",
            })
        except Exception as _e:
            logger.warning(f"normality diag: {_e}")

        # 3) Homoscédasticité — test de White (LM = n·R² de resid² sur le design
        #    AUGMENTÉ des carrés des variables numériques). Le carré capte aussi
        #    l'hétéroscédasticité « en entonnoir » (variance ∝ x²) que le Breusch-Pagan
        #    strictement linéaire manque.
        try:
            e2 = resid ** 2
            num_cols = [i for i, nm in enumerate(names) if nm.startswith("num__")]
            aug = [np.ones(n), Z]
            if num_cols:
                aug.append(Z[:, num_cols] ** 2)  # carrés des seules numériques (évite le doublon one-hot²=one-hot)
            Zc = np.column_stack(aug)
            beta, *_ = np.linalg.lstsq(Zc, e2, rcond=None)
            fit = Zc @ beta
            ss_tot = float(np.sum((e2 - e2.mean()) ** 2)) + 1e-12
            r2_aux = 1.0 - float(np.sum((e2 - fit) ** 2)) / ss_tot
            lm = n * max(0.0, r2_aux)
            df_bp = max(1, Zc.shape[1] - 1)
            p_bp = float(1.0 - _st.chi2.cdf(lm, df_bp))
            out["checks"].append({
                "key": "homoscedasticity", "name": "Homoscédasticité (test de White)",
                "p_value": round(p_bp, 4),
                "status": "ok" if p_bp >= 0.05 else "warn",
                "detail": "p≥0.05 = variance des résidus constante ; p<0.05 = hétéroscédasticité.",
            })
        except Exception as _e:
            logger.warning(f"white/bp diag: {_e}")

        # 4) Multicolinéarité — VIF par variable numérique (régression de chaque
        #    colonne numérique sur les autres). Repli propre si colinéarité parfaite.
        try:
            num_idx = [i for i, nm in enumerate(names) if nm.startswith("num__")]
            if len(num_idx) >= 2:
                Zn = Z[:, num_idx]
                vifs = []
                for j in range(Zn.shape[1]):
                    other = np.delete(Zn, j, axis=1)
                    Ao = np.column_stack([np.ones(n), other])
                    coef, *_ = np.linalg.lstsq(Ao, Zn[:, j], rcond=None)
                    sst = float(np.sum((Zn[:, j] - Zn[:, j].mean()) ** 2)) + 1e-12
                    r2j = 1.0 - float(np.sum((Zn[:, j] - Ao @ coef) ** 2)) / sst
                    vif = 1.0 / max(1e-6, 1.0 - min(r2j, 1 - 1e-6))
                    vifs.append({"variable": names[num_idx[j]].split("__", 1)[1], "vif": round(float(vif), 2)})
                max_vif = max(v["vif"] for v in vifs)
                out["checks"].append({
                    "key": "multicollinearity", "name": "Multicolinéarité (VIF max)",
                    "statistic": max_vif, "per_variable": sorted(vifs, key=lambda d: -d["vif"])[:10],
                    "status": "ok" if max_vif < 5 else ("warn" if max_vif < 10 else "fail"),
                    "detail": "VIF<5 = ok ; 5–10 = modérée ; >10 = forte colinéarité (coefficients instables).",
                })
        except Exception as _e:
            logger.warning(f"vif diag: {_e}")
    except Exception as e:
        logger.warning(f"regression_diagnostics: {e}")
        return {"checks": [], "applies": None, "error": str(e)}
    return out


# ─── Garde-fous & explicabilité (Phase 4b) ───────────────────────────────────

def _bootstrap_metric_ci(pipe, Xte, yte, task_type: str, scoring: str,
                         classes: list | None, random_state: int = 42, n_boot: int = 300) -> dict | None:
    """IC 95 % de la métrique de test par bootstrap des LIGNES de test (on
    rééchantillonne les prédictions déjà calculées → aucun réentraînement). Donne une
    fourchette d'incertitude honnête sur la métrique phare. None si test trop petit."""
    import numpy as np
    from sklearn import metrics as M
    yv = np.asarray(yte)
    n = len(yv)
    if n < 20:
        return None
    try:
        proba = pipe.predict_proba(Xte) if (task_type == "classification" and hasattr(pipe, "predict_proba")) else None
        preds = pipe.predict(Xte) if task_type == "classification" else np.asarray(pipe.predict(Xte), dtype=float)
    except Exception:
        return None

    def _metric(idx):
        yt = yv[idx]
        try:
            if task_type == "classification":
                if scoring in ("roc_auc", "roc_auc_ovr") and proba is not None:
                    return (M.roc_auc_score(yt, proba[idx, 1]) if proba.shape[1] == 2
                            else M.roc_auc_score(yt, proba[idx], multi_class="ovr"))
                if scoring == "average_precision" and proba is not None and proba.shape[1] == 2:
                    return M.average_precision_score(yt, proba[idx, 1])
                return M.accuracy_score(yt, preds[idx])
            if scoring == "neg_mean_absolute_error":
                return M.mean_absolute_error(yt, preds[idx])
            if scoring == "r2":
                return M.r2_score(yt, preds[idx])
            return float(np.sqrt(M.mean_squared_error(yt, preds[idx])))
        except Exception:
            return None

    rng = np.random.RandomState(random_state)
    vals = []
    for _ in range(n_boot):
        v = _metric(rng.randint(0, n, n))
        if v is not None and np.isfinite(v):
            vals.append(float(v))
    if len(vals) < 30:
        return None
    lo, hi = np.percentile(vals, [2.5, 97.5])
    name = {"neg_root_mean_squared_error": "rmse", "neg_mean_absolute_error": "mae", "r2": "r2",
            "roc_auc": "roc_auc", "roc_auc_ovr": "roc_auc", "average_precision": "average_precision",
            "accuracy": "accuracy"}.get(scoring, scoring)
    return {"metric": name, "low": round(float(lo), 4), "high": round(float(hi), 4), "n_boot": len(vals)}


def model_guardrails(pipe, Xtr, ytr, Xte, yte, task_type: str, cv_scores,
                     scoring: str, classes: list | None = None, random_state: int = 42) -> dict:
    """Garde-fous POST-entraînement (informatif, n'altère pas le modèle) :
      - fuite de cible : variable à association quasi parfaite avec la cible ;
      - déséquilibre de classes (classification) ;
      - stabilité de la validation croisée (écart-type inter-folds) ;
      - IC bootstrap de la métrique de test.
    Chaque contrôle porte un statut ok/warn/fail pour l'affichage (comme les
    hypothèses de régression)."""
    import numpy as np
    import pandas as pd
    checks = []

    # 1) Fuite de cible : corrélation (régression) ou AUC univarié (classification
    # binaire) proche de la perfection → la variable « connaît » déjà l'issue.
    leaks = []
    try:
        yv = np.asarray(ytr, dtype=float)
        for col in Xtr.columns:
            xs = pd.to_numeric(Xtr[col], errors="coerce")
            m = xs.notna().values & np.isfinite(yv)
            if int(m.sum()) < 8 or xs[m].nunique() < 2:
                continue
            if task_type == "classification":
                yb = yv[m]
                if len(set(yb.tolist())) == 2:
                    try:
                        from sklearn.metrics import roc_auc_score
                        auc = roc_auc_score(yb, xs[m].values)
                        auc = max(auc, 1.0 - auc)
                    except Exception:
                        continue
                    if auc >= 0.999:
                        leaks.append({"feature": col, "assoc": round(float(auc), 4), "kind": "auc"})
            else:
                r = float(np.corrcoef(xs[m].values, yv[m])[0, 1])
                if np.isfinite(r) and abs(r) >= 0.999:
                    leaks.append({"feature": col, "assoc": round(abs(r), 4), "kind": "corr"})
    except Exception as e:
        logger.warning(f"guardrails leakage: {e}")
    checks.append({
        "key": "leakage", "name": "Fuite de cible",
        "status": "fail" if leaks else "ok",
        "detail": (("Association quasi parfaite avec la cible : " + ", ".join(l["feature"] for l in leaks)
                    + " — probable fuite (la variable encode l'issue). À retirer.") if leaks
                   else "Aucune variable ne prédit trivialement la cible."),
        "leaks": leaks,
    })

    # 2) Déséquilibre de classes (classification).
    if task_type == "classification" and classes:
        try:
            vc = pd.Series(np.asarray(ytr)).value_counts()
            total = int(vc.sum())
            dist = {(str(classes[int(i)]) if 0 <= int(i) < len(classes) else str(i)): int(c) for i, c in vc.items()}
            minfrac = float(vc.min() / total) if total else 0.0
            checks.append({
                "key": "class_balance", "name": "Équilibre des classes",
                "status": "warn" if minfrac < 0.10 else "ok",
                "statistic": round(minfrac, 3),
                "detail": (f"Classe minoritaire : {minfrac*100:.1f} %. "
                           + ("Fort déséquilibre (<10 %) : préférer average_precision / rappel à l'accuracy."
                              if minfrac < 0.10 else "Distribution acceptable.")),
                "distribution": dist,
            })
        except Exception as e:
            logger.warning(f"guardrails balance: {e}")

    # 3) Stabilité de la validation croisée (écart-type inter-folds du meilleur modèle).
    try:
        cvs = np.abs(np.asarray(list(cv_scores), dtype=float))   # neg_* → positif
        mean, std = float(np.mean(cvs)), float(np.std(cvs))
        rel = std / (abs(mean) + 1e-9)
        checks.append({
            "key": "cv_stability", "name": "Stabilité (validation croisée)",
            "status": "warn" if rel > 0.25 else "ok",
            "statistic": round(rel, 3), "mean": round(mean, 4), "std": round(std, 4),
            "detail": (f"Écart-type inter-folds {std:.3f} (moyenne {mean:.3f}). "
                       + ("Variance élevée : résultat sensible au découpage / peu de données."
                          if rel > 0.25 else "Résultats stables entre folds.")),
        })
    except Exception as e:
        logger.warning(f"guardrails cv: {e}")

    ci = _bootstrap_metric_ci(pipe, Xte, yte, task_type, scoring, classes, random_state)
    return {"checks": checks, "metric_ci": ci}


def explain_background(Xtr, used: list[dict]) -> dict:
    """Valeurs de « fond » par variable (médiane si numérique, mode sinon), pour
    l'explication locale par ablation. JSON-sérialisable (persisté dans le résumé)."""
    import pandas as pd
    bg = {}
    for u in used:
        mn = u["machine_name"]
        if mn not in Xtr.columns:
            continue
        col = Xtr[mn]
        num = pd.to_numeric(col, errors="coerce")
        if num.notna().mean() > 0.5:
            bg[mn] = float(num.median())
        else:
            m = col.mode()
            bg[mn] = (str(m.iloc[0]) if len(m) else None)
    return bg


def explain_prediction(pipe, X_row, used: list[dict], task_type: str, background: dict) -> dict:
    """Explication LOCALE d'une prédiction, SANS dépendance externe : contribution de
    chaque variable estimée par ABLATION vers la valeur de fond (on remplace la
    variable par son fond et on mesure la variation de sortie). Trié par |contribution|."""
    import numpy as np

    def _out(dfrow):
        if task_type == "classification" and hasattr(pipe, "predict_proba"):
            return float(pipe.predict_proba(dfrow)[0, -1])
        return float(np.asarray(pipe.predict(dfrow), dtype=float)[0])

    base = _out(X_row)
    contribs = []
    for u in used:
        mn = u["machine_name"]
        if mn not in X_row.columns or mn not in background:
            continue
        row2 = X_row.copy()
        row2[mn] = background.get(mn)
        try:
            contribs.append({"feature": mn, "contribution": round(float(base - _out(row2)), 5)})
        except Exception:
            continue
    contribs.sort(key=lambda c: abs(c["contribution"]), reverse=True)
    return {"base_output": round(base, 5), "contributions": contribs}


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

    # Prévision de série temporelle : hors du flux tabulaire (Prophet/SARIMAX ne
    # sont pas des estimateurs sklearn). On route AVANT _effective_family (qui
    # sinon rétrograderait "prophet"/"sarimax" vers gradient_boosting).
    _raw_family = ((spec.get("algorithm") or {}).get("family") or "").strip().lower()
    if _raw_family in _TS_FAMILIES:
        return train_timeseries_model(df, spec, family=_raw_family,
                                      random_state=random_state, test_size=test_size)

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
    cv_strategy_effective = strategy

    # Validation temporelle : TimeSeriesSplit et le train_test_split NON mélangé
    # supposent des lignes ORDONNÉES dans le temps. Le DataFrame n'est pas trié,
    # donc sans tri explicite le découpage suivrait l'ordre arbitraire des lignes →
    # holdout « temporel » dénué de sens. On trie sur la 1re colonne datetime du
    # spec ; à défaut (aucune colonne temporelle exploitable) on rétrograde vers un
    # découpage mélangé plutôt que de fabriquer un faux holdout chronologique.
    if is_ts:
        time_col = next(
            (u["machine_name"] for u in used
             if u.get("dtype") == "datetime" and u["machine_name"] in X.columns),
            None,
        )
        order = pd.to_datetime(X[time_col], errors="coerce") if time_col else None
        if order is not None and order.notna().any():
            sorted_idx = order.sort_values(kind="mergesort").index  # tri stable
            X = X.loc[sorted_idx].reset_index(drop=True)
            y = y.loc[sorted_idx].reset_index(drop=True)
        else:
            is_ts = False
            cv_strategy_effective = "stratified_kfold" if task_type == "classification" else "kfold"

    stratify = y if (task_type == "classification" and not is_ts) else None
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        shuffle=not is_ts, stratify=stratify,
    )

    cv = _make_cv(cv_strategy_effective, folds, task_type, random_state)
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
    # Scores par fold du MEILLEUR modèle : réutilisés pour la stabilité CV (garde-fous).
    try:
        cv_scores_arr = cross_val_score(
            Pipeline([("pre", pre), ("est", estimator_from_params(family, task_type, best_params, random_state))]),
            Xtr, ytr, scoring=scoring, cv=cv, n_jobs=1)
    except Exception:
        cv_scores_arr = [cv_best]
    final.fit(Xtr, ytr)

    metrics = _evaluate(final, Xte, yte, task_type, classes)
    # CV best rendu LISIBLE : les scorers sklearn sont "greater is better", donc rmse/mae
    # sont stockés en NÉGATIF (neg_*). On expose la valeur positive sous cv_<metric>.
    cv_pretty = -cv_best if scoring.startswith("neg_") else cv_best
    metrics[f"cv_{scoring.replace('neg_', '').replace('root_mean_squared_error', 'rmse').replace('mean_absolute_error', 'mae')}"] = float(cv_pretty)

    # Métrique AUTORITAIRE = celle réellement optimisée (déduite du scorer), et non
    # la métrique brute du spec (qui pouvait être incohérente avec la tâche, p.ex.
    # "rmse" sur une classification → carte UI affichant l'accuracy sous le libellé
    # "rmse"). `display_metric` existe toujours dans `metrics`, donc la carte affiche
    # la bonne valeur sous le bon nom. On conserve la demande d'origine séparément.
    _METRIC_FROM_SCORING = {
        "neg_root_mean_squared_error": "rmse", "neg_mean_absolute_error": "mae", "r2": "r2",
        "roc_auc": "roc_auc", "roc_auc_ovr": "roc_auc", "average_precision": "average_precision",
        "accuracy": "accuracy",
    }
    display_metric = _METRIC_FROM_SCORING.get(scoring, metric or scoring)

    result = {
        "task_type": task_type,
        "family": family,
        "metric": display_metric,
        "requested_metric": metric,
        "scoring": scoring,
        "cv_strategy": cv_strategy_effective,
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
        "importances_by_variable": _importances_by_variable(final, used),
        "pipeline": final,
    }
    if task_type in ("regression", "count"):
        # Diagnostics des hypothèses de régression (Gauss-Markov / OLS) — informatif.
        result["assumptions"] = regression_diagnostics(final, Xtr, ytr, used)
    # Garde-fous (fuite de cible, déséquilibre, stabilité CV, IC bootstrap) +
    # fond d'explication locale (pour /model/predict). Informatif, best-effort.
    try:
        result["guardrails"] = model_guardrails(final, Xtr, ytr, Xte, yte, task_type,
                                                cv_scores_arr, scoring, classes, random_state)
    except Exception as _ge:
        logger.warning(f"model_guardrails: {_ge}")
    try:
        result["explain_background"] = explain_background(Xtr, used)
    except Exception:
        pass
    return result


def _lower_is_better(metric: str) -> bool:
    return (metric or "").lower() in ("rmse", "mae")


def leaderboard_families(task_type: str) -> list[str]:
    """Ensemble CURÉ de familles à comparer (fortes pour données tabulaires + une
    base linéaire interprétable). Exclut lightgbm/xgboost si le paquet est absent —
    pas TOUTES les familles, pour garder la comparaison rapide."""
    tt = (task_type or "classification").strip().lower()
    lin = "linear_regression" if tt in ("regression", "count") else "logistic_regression"
    order = ["lightgbm", "xgboost", "gradient_boosting", "random_forest", lin]
    base = REG_FAMILIES if tt in ("regression", "count") else CLF_FAMILIES
    return [f for f in order if f in base and (f not in _BOOSTED_FAMILIES or _has_package(f))]


def compare_models(df, spec: dict, families: list[str] | None = None,
                   n_trials: int = 15, random_state: int = 42, test_size: float = 0.2) -> dict:
    """Entraîne PLUSIEURS familles sur le MÊME découpage (mêmes seeds → comparaison
    équitable) et les classe par la métrique de holdout. Renvoie un tableau
    comparatif + la meilleure famille (avec son pipeline entraîné). Pur/testable.

    Chaque entrée du classement : {family, metric, value, metrics, best_params}.
    Une famille qui échoue (données insuffisantes, paquet absent) apparaît avec
    `error` et n'est pas classée."""
    outcome = spec.get("outcome") or {}
    task_type = (outcome.get("task_type") or "classification").strip().lower()
    fams = families or leaderboard_families(task_type)

    entries: list[dict] = []
    best_result = None
    metric_key = None
    for fam in fams:
        _spec = {**spec, "algorithm": {**(spec.get("algorithm") or {}), "family": fam}}
        try:
            r = train_model(df, _spec, n_trials=n_trials, random_state=random_state, test_size=test_size)
        except Exception as e:
            entries.append({"family": fam, "error": str(e)[:300]})
            continue
        metric_key = r["metric"]
        val = r["metrics"].get(metric_key)
        entries.append({
            "family": r["family"], "metric": metric_key, "value": val,
            "metrics": r["metrics"], "best_params": r["best_params"],
            "n_train": r["n_train"], "n_test": r["n_test"],
        })
        if val is not None:
            if best_result is None:
                best_result = r
            else:
                bv = best_result["metrics"].get(best_result["metric"])
                better = (val < bv) if _lower_is_better(metric_key) else (val > bv)
                if bv is None or better:
                    best_result = r

    scored = [e for e in entries if e.get("value") is not None]
    scored.sort(key=lambda e: e["value"], reverse=not _lower_is_better(metric_key or ""))
    failed = [e for e in entries if e.get("value") is None]
    for rank, e in enumerate(scored, 1):
        e["rank"] = rank

    return {
        "task_type": task_type,
        "metric": metric_key,
        "lower_is_better": _lower_is_better(metric_key or ""),
        "leaderboard": scored + failed,
        "families_tried": fams,
        "best_family": (best_result["family"] if best_result else None),
        "best": best_result,   # inclut le pipeline entraîné (à sérialiser par l'appelant)
    }


# ─── Prévision de série temporelle (Phase 3b) : Prophet / SARIMAX ─────────────

def _infer_freq(dates) -> str:
    """Devine la fréquence (alias d'offset pandas) d'une suite de dates triée.
    Repli sur la médiane des écarts si pandas.infer_freq échoue (dates irrégulières)."""
    import numpy as np
    import pandas as pd
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    try:
        f = pd.infer_freq(idx)
        if f:
            return f
    except Exception:
        pass
    if len(idx) < 3:
        return "D"
    med = float(np.median(np.diff(idx.view("int64"))))   # ns entre points
    day = 86_400 * 1e9
    if med <= 1.5 * 3600e9:
        return "h"
    if med <= 1.5 * day:
        return "D"
    if med <= 10 * day:
        return "W"
    if med <= 45 * day:
        return "MS"
    if med <= 100 * day:
        return "QS"
    return "YS"


def _seasonal_period(freq: str) -> int:
    """Période saisonnière plausible selon la fréquence (0 = aucune) : horaire→24,
    journalier→7 (semaine), hebdo→52, mensuel→12, trimestriel→4."""
    f = (freq or "").upper()
    if f.startswith("H"):
        return 24
    if f.startswith("D"):
        return 7
    if f.startswith("W"):
        return 52
    if f.startswith("M"):
        return 12
    if f.startswith("Q"):
        return 4
    return 0


def _ts_metrics(actual, predicted) -> dict:
    """RMSE / MAE / MAPE sur le holdout (ignore les paires non finies)."""
    import numpy as np
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    m = np.isfinite(a) & np.isfinite(p)
    a, p = a[m], p[m]
    if a.size == 0:
        return {"rmse": None, "mae": None, "mape": None}
    err = a - p
    nz = np.abs(a) > 1e-9
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "mape": (float(np.mean(np.abs(err[nz] / a[nz])) * 100.0) if nz.any() else None),
    }


def train_timeseries_model(df, spec: dict, family: str = "prophet",
                           random_state: int = 42, test_size: float = 0.2,
                           horizon: int | None = None) -> dict:
    """Prévision de série temporelle univariée (Prophet ou SARIMAX).

    Exige une colonne datetime + une cible numérique. Découpe un holdout TEMPOREL
    (derniers points), évalue rmse/mae/mape sur ce holdout, puis réajuste sur toute
    la série pour produire la prévision future (H pas). Renvoie un dict compatible
    avec la persistance : le forecaster ajusté est sous `pipeline` (sérialisé par
    l'appelant), le reste alimente le résumé (holdout, forecast, historique)."""
    import numpy as np
    import pandas as pd

    outcome = spec.get("outcome") or {}
    target = outcome.get("machine_name")
    features = spec.get("features") or []

    norm_map: dict[str, Any] = {}
    for c in df.columns:
        norm_map.setdefault(_norm(c), c)
    tcol = norm_map.get(_norm(target))
    if tcol is None:
        raise ValueError(f"Colonne cible '{target}' absente du dataset.")

    # Colonne temporelle : 1re feature datetime du spec présente, sinon 1re colonne
    # réellement parsable en dates (>80 % de valeurs valides).
    time_mn = next((f.get("machine_name") for f in features if f.get("dtype") == "datetime"), None)
    time_col = norm_map.get(_norm(time_mn)) if time_mn else None
    if time_col is None:
        for c in df.columns:
            if c == tcol:
                continue
            if pd.to_datetime(df[c], errors="coerce").notna().mean() > 0.8:
                time_col = c
                break
    if time_col is None:
        raise ValueError("La prévision temporelle exige une colonne de date (aucune trouvée dans le dataset).")

    # Série propre : dates parsées + cible numérique, triée, doublons de date agrégés.
    s = pd.DataFrame({
        "ds": pd.to_datetime(df[time_col], errors="coerce"),
        "y": pd.to_numeric(df[tcol], errors="coerce"),
    }).dropna().sort_values("ds", kind="mergesort")
    s = s.groupby("ds", as_index=False)["y"].mean()
    n = len(s)
    if n < 16:
        raise ValueError(f"Série trop courte pour une prévision ({n} points exploitables). Minimum 16.")

    # Holdout temporel : derniers H points (≥2 ; ≤1/3 de la série).
    H = int(horizon) if horizon else max(3, min(int(round(n * test_size)), 24))
    H = max(2, min(H, n // 3))
    train, test = s.iloc[:n - H], s.iloc[n - H:]
    freq = _infer_freq(s["ds"])
    fam = (family or "prophet").strip().lower()

    def _iso(dts):
        return [pd.Timestamp(d).isoformat() for d in dts]

    best_params: dict[str, Any] = {}
    if fam == "sarimax":
        import warnings as _w
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        # Grille d'ordres ARIMA(p,d,q) × ordre SAISONNIER inféré de la fréquence.
        # La composante saisonnière (ex. cycle hebdo pour des données journalières)
        # est ajoutée uniquement si la série a la longueur pour l'estimer.
        m_season = _seasonal_period(freq)
        seasonal_orders = [(0, 0, 0, 0)]
        if m_season >= 2 and len(train) >= 2 * m_season + 8:
            seasonal_orders.append((1, 0, 1, m_season))
        candidates = [(o, so) for o in [(1, 1, 1), (2, 1, 1), (1, 1, 0), (0, 1, 1), (1, 0, 0)]
                      for so in seasonal_orders]
        best = None
        for order, sorder in candidates:
            try:
                with _w.catch_warnings():
                    _w.simplefilter("ignore")
                    res = SARIMAX(train["y"].to_numpy(), order=order, seasonal_order=sorder,
                                  enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                    pred = np.asarray(res.forecast(steps=H), dtype=float)
                mets = _ts_metrics(test["y"].to_numpy(), pred)
                if mets["rmse"] is not None and (best is None or mets["rmse"] < best[1]["rmse"]):
                    best = ((order, sorder), mets, pred)
            except Exception as e:
                logger.warning(f"SARIMAX {order}x{sorder}: {e}")
        if best is None:
            raise ValueError("SARIMAX n'a convergé pour aucun ordre candidat.")
        (order, sorder), holdout_metrics, holdout_pred = best
        best_params = {"order": list(order), "seasonal_order": list(sorder)}
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            full = SARIMAX(s["y"].to_numpy(), order=order, seasonal_order=sorder,
                           enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        fc = full.get_forecast(steps=H)
        fmean = np.asarray(fc.predicted_mean, dtype=float)
        ci = np.asarray(fc.conf_int(alpha=0.2), dtype=float)   # IC 80 %
        lower, upper = ci[:, 0], ci[:, 1]
        future_dates = pd.date_range(start=s["ds"].iloc[-1], periods=H + 1, freq=freq)[1:]
        fitted_model = full
    else:  # prophet
        import logging as _lg
        from prophet import Prophet
        _lg.getLogger("prophet").setLevel(_lg.ERROR)
        _lg.getLogger("cmdstanpy").setLevel(_lg.ERROR)
        m = Prophet(interval_width=0.8)
        m.fit(train[["ds", "y"]])
        holdout_pred = np.asarray(m.predict(pd.DataFrame({"ds": test["ds"]}))["yhat"], dtype=float)
        holdout_metrics = _ts_metrics(test["y"].to_numpy(), holdout_pred)
        mf = Prophet(interval_width=0.8)
        mf.fit(s[["ds", "y"]])
        fut = mf.predict(mf.make_future_dataframe(periods=H, freq=freq)).tail(H)
        fmean = np.asarray(fut["yhat"], dtype=float)
        lower = np.asarray(fut["yhat_lower"], dtype=float)
        upper = np.asarray(fut["yhat_upper"], dtype=float)
        future_dates = pd.to_datetime(fut["ds"])
        best_params = {"seasonality_mode": "additive", "interval_width": 0.8}
        fitted_model = mf

    tail_ctx = s.tail(min(60, n))
    return {
        "family": fam,
        "task_type": "forecast",
        "metric": "rmse",
        "metrics": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in holdout_metrics.items()},
        "best_params": best_params,
        "feature_importances": [],
        "importances_by_variable": {},
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_points": int(n),
        "target": target,
        "time_column": time_mn or _norm(time_col),
        "horizon": int(H),
        "series_freq": str(freq),
        "holdout": {
            "dates": _iso(test["ds"]),
            "actual": [float(v) for v in test["y"].to_numpy()],
            "predicted": [float(v) for v in holdout_pred],
        },
        "forecast": {
            "dates": _iso(future_dates),
            "predicted": [float(v) for v in fmean],
            "lower": [float(v) for v in lower],
            "upper": [float(v) for v in upper],
        },
        "history_tail": {
            "dates": _iso(tail_ctx["ds"]),
            "actual": [float(v) for v in tail_ctx["y"].to_numpy()],
        },
        "pipeline": fitted_model,   # forecaster ajusté (sérialisé par l'appelant)
    }


# ─── Monitoring (Phase 4) : score les données récentes -> niveau d'alerte ─────

# Jetons usuels d'« événement » (classe positive) pour les encodages médicaux
# courants, utilisés quand le spec ne fixe pas explicitement positive_class.
_POSITIVE_TOKENS = frozenset({
    "1", "true", "vrai", "oui", "yes", "positive", "positif", "pos",
    "event", "événement", "evenement", "case", "cas", "death", "décès", "deces",
    "deceased", "dead", "mort", "mortality", "mortalité", "malade", "disease",
    "present", "présent", "abnormal", "anormal",
})


def positive_index(classes: list | None, positive_class: str | None) -> int:
    """Index de la classe 'événement' (positive).

    Priorité : (1) positive_class explicite du spec ; (2) un libellé reconnu comme
    événement (oui/yes/1/positif/décès…) ; (3) à défaut seulement, la dernière
    classe. LabelEncoder trie les classes par ordre alphabétique, donc le repli
    « dernière classe » est arbitraire (ex. [décès, survie] → 'survie') : on
    l'évite dès qu'un libellé interprétable est présent."""
    if not classes:
        return 1
    if positive_class is not None and str(positive_class) in classes:
        return classes.index(str(positive_class))
    for i, c in enumerate(classes):
        if str(c).strip().lower() in _POSITIVE_TOKENS:
            return i
    return len(classes) - 1


def _level_from_value(v: float, orange: float | None, red: float | None) -> str:
    """Mappe une valeur croissante (risque) vers green/orange/red.

    Renvoie 'unavailable' (JAMAIS 'green' par défaut) si la valeur est NaN/None
    ou si les bandes sont inconnues : un seuil non interprétable ou une absence de
    donnée ne doit pas être présenté comme « Normal »."""
    import math
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "unavailable"
    if orange is None or red is None:
        return "unavailable"
    if v >= red:
        return "red"
    if v >= orange:
        return "orange"
    return "green"


def _alert_bounds(alert_thresholds):
    """(orange_lo, red_lo) déduits des SEUILS LITTÉRAIRES (alert_thresholds) via leurs
    champs 'range' ('< 5 /100k', '5–10', '> 10 cas pour 100 000', …). Renvoie
    (None, None) si introuvable. On retire le dénominateur d'unité ('/100k', 'pour
    100 000') pour ne garder que les bornes numériques."""
    if not alert_thresholds:
        return None, None
    import re as _re
    def _nums(band):
        raw = (alert_thresholds.get(band) or {}).get("range") or ""
        s = str(raw).split("/")[0]
        s = _re.split(r"(?i)\b(?:pour|per|par)\b", s)[0]
        return [float(x) for x in _re.findall(r"\d+(?:\.\d+)?", s)]
    g, o, rd = _nums("green"), _nums("orange"), _nums("red")
    orange_lo = min(o) if o else (max(g) if g else None)
    red_lo = max(o) if o else (min(rd) if rd else None)
    if orange_lo is not None and red_lo is not None and red_lo < orange_lo:
        orange_lo, red_lo = red_lo, orange_lo
    return orange_lo, red_lo


def compute_monitoring(pipeline, recent_df, task_type: str, classes: list | None = None,
                       positive_class: str | None = None, target_values=None,
                       alert_thresholds=None) -> dict:
    """
    Score les lignes récentes avec le modèle entraîné et en déduit un niveau
    d'alerte (green/orange/red). Pur/testable.

    - classification : risque = proba moyenne de la classe positive ; bandes
      fixes 0.33 / 0.66 (la proba est déjà normalisée).
    - régression : valeur = prédiction moyenne ; bandes = tertiles (p33/p66) de
      la cible d'entraînement (valeur haute = alerte, convention EMS).
    """
    import numpy as np

    n = int(len(recent_df))
    if n == 0:
        # Aucune donnée récente à scorer → état EXPLICITEMENT indisponible, jamais
        # « green/Normal » : un moniteur vide ne doit pas paraître sain.
        kind = "value" if task_type in ("regression", "count") else "probability"
        return {"kind": kind, "value": None, "level": "unavailable",
                "bands": {"orange": None, "red": None}, "n_scored": 0,
                "bands_source": None, "positive_class": None}
    if task_type in ("regression", "count"):
        preds = np.asarray(pipeline.predict(recent_df), dtype=float)
        value = float(np.mean(preds))
        # PRIORITÉ aux seuils LITTÉRAIRES (alert_thresholds). Les tertiles (p33/p66)
        # des données d'entraînement ne servent QUE de repli : sinon une valeur SOUS
        # le seuil « Normal » pouvait être classée « Tension » juste parce qu'elle
        # dépassait le 33ᵉ percentile des données (bug 2.892/100k → « Tension »
        # alors que Normal = < 5).
        lo, hi = _alert_bounds(alert_thresholds)
        bands_source = "literature" if (lo is not None and hi is not None) else None
        if (lo is None or hi is None) and target_values is not None:
            tv = np.asarray(target_values, dtype=float)
            tv = tv[~np.isnan(tv)]
            if len(tv) >= 5:
                if lo is None:
                    lo = float(np.quantile(tv, 0.33))
                if hi is None:
                    hi = float(np.quantile(tv, 0.66))
                bands_source = bands_source or "data_quantiles"
        return {"kind": "value", "value": value, "level": _level_from_value(value, lo, hi),
                "bands": {"orange": lo, "red": hi}, "n_scored": n,
                "bands_source": bands_source}

    idx = positive_index(classes, positive_class)
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(recent_df)
        col = min(idx, proba.shape[1] - 1)
        risk = float(np.mean(proba[:, col]))
    else:
        risk = float(np.mean(np.asarray(pipeline.predict(recent_df)) == idx))
    return {"kind": "probability", "value": risk, "level": _level_from_value(risk, 0.33, 0.66),
            "positive_class": (classes[idx] if classes else None),
            "bands": {"orange": 0.33, "red": 0.66}, "n_scored": n}
