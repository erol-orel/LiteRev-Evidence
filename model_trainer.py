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

    # Données médicales souvent déséquilibrées (événement rare) : on pondère les
    # classes par défaut pour les familles qui le supportent (RF, régression
    # logistique, SVM). GradientBoosting/MLP/KNN n'exposent pas class_weight et
    # restent inchangés. setdefault → un réglage explicite du spec a la priorité.
    if clf and family in ("random_forest", "logistic_regression", "svm"):
        p.setdefault("class_weight", "balanced")

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
    return result


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
