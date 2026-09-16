"""Variables and model spec generated from PICO, with localisation.

Extracted from main.py (LiteRev API); `main` re-exports everything for the scripts,
tools and tests.
"""
from __future__ import annotations

import re
from typing import Any

from fastapi import Depends, Query
from sqlalchemy import text

from .core import _env_int, _job_is_active, _msg, _norm_lang, app, engine, logger, require_api_key
from .documents import _llm_lang_directive
from .scenario_store import _get_scenario_threshold
from .gesica import _get_scenario_name
from .relevance import _evidence_fingerprint, _get_above_threshold_articles

# ─── VARIABLES & MODÈLE AUTO-REMPLI DEPUIS PICO ──────────────────────────────

_VARIABLES_GENERATION_JOBS: dict[str, dict] = {}


# ─── EXTRACTION CIBLÉE DES PARAMÈTRES ÉPIDÉMIOLOGIQUES (famille SEIR) ─────────
#
# Le spec est généré à partir des 25 articles LES PLUS PERTINENTS du corpus. Sur un
# corpus de plusieurs milliers d'articles, ceux qui RAPPORTENT un R0 ou une période
# d'incubation n'y figurent quasiment jamais : le bloc epidemic_parameters ressortait
# vide et l'onglet Modèle annonçait « aucun paramètre extrait de la littérature » pour
# un corpus chikungunya de 2 732 articles, où ces valeurs sont pourtant publiées.
#
# On CHERCHE donc, dans tout le sous-ensemble pertinent, les articles qui MESURENT un
# paramètre (termes de mesure dans le titre ou le résumé), et on n'extrait que sur
# ceux-là, une observation par étude. Le regroupement pondéré par la qualité
# (seir_model.pool_weighted, déjà en place) fait le reste : la valeur servie au modèle
# reste une moyenne d'études réelles, chacune cliquable.

# Termes de MESURE par paramètre (EN + FR). Une même source pour le pré-filtre SQL et
# le filtre Python : pas de divergence possible entre les deux.
_PARAM_PHRASES: dict[str, tuple[str, ...]] = {
    "r0": ("basic reproduction number", "reproduction number", "reproductive number",
           "reproduction ratio", "nombre de reproduction", "taux de reproduction"),
    "serial_interval_days": ("serial interval", "generation interval", "generation time",
                             "intervalle sériel", "intervalle de génération"),
    "incubation_period_days": ("incubation period", "période d'incubation", "periode d'incubation",
                               "latent period", "période de latence"),
    "infectious_period_days": ("infectious period", "infectiousness duration", "duration of infectiousness",
                               "shedding duration", "duration of viremia", "période infectieuse",
                               "durée d'infectiosité"),
    "cfr": ("case fatality", "case-fatality", "fatality rate", "fatality ratio",
            "létalité", "letalite", "taux de décès", "death rate among cases"),
    "immunity_duration_days": ("duration of immunity", "immunity duration", "waning immunity",
                               "duration of protection", "durée de l'immunité", "durée de protection",
                               "seroprotection duration"),
}
# Sigles courts : exigent des frontières de mot, sinon « R0 » matche « macro0 ».
_PARAM_TOKENS: dict[str, tuple[str, ...]] = {
    "r0": ("R0", "R-0", "R_0", "Rzero"),
}
# Paramètres qui PILOTENT la dynamique : un seul d'entre eux, mesuré dans le corpus,
# suffit à démontrer que le scénario est transmissible (cf. _merge_epidemic_observations).
_TRANSMISSION_PARAMS = ("r0", "serial_interval_days", "incubation_period_days")

# Aucun plafond par défaut : l'extraction lit TOUS les articles pertinents qui mesurent
# un paramètre, jamais un échantillon. EPI_PARAM_MAX_ARTICLES > 0 en pose un (secours
# d'exploitation si le budget LLM doit être tenu un jour donné).
EPI_PARAM_MAX_ARTICLES = _env_int("EPI_PARAM_MAX_ARTICLES", 0, 0)
_EPI_PARAM_BATCH = 10
_EPI_PARAM_WORKERS = 4


def _param_regex(params=None, boundary: str = r"\b") -> str:
    """Alternation regex des termes de mesure (paramètres demandés, ou tous). `boundary`
    vaut r'\\b' pour Python et r'\\y' pour Postgres. Pur."""
    names = tuple(params or _PARAM_PHRASES)
    parts: list[str] = []
    for name in names:
        parts.extend(re.escape(p) for p in _PARAM_PHRASES.get(name, ()))
        parts.extend(f"{boundary}{re.escape(t)}{boundary}" for t in _PARAM_TOKENS.get(name, ()))
    return "|".join(parts)


_PARAM_SCAN_RE = re.compile(_param_regex(), re.IGNORECASE)


def params_mentioned(text: str) -> list[str]:
    """Paramètres épidémiologiques dont un terme de MESURE apparaît dans ce texte
    (titre + résumé), dans l'ordre de _PARAM_PHRASES. Pur, testé hors ligne."""
    t = text or ""
    found = []
    for name in _PARAM_PHRASES:
        if re.search(_param_regex([name]), t, re.IGNORECASE):
            found.append(name)
    return found


def _parameter_candidate_articles(scenario_id: str, threshold: float | None = None,
                                  limit: int = 0) -> list[dict]:
    """TOUS les articles du sous-ensemble PERTINENT qui rapportent un paramètre (terme de
    mesure dans le titre ou le résumé), les meilleurs d'abord : synthèses en tête, puis
    qualité, citations et année. Même porte de screening que partout ailleurs.
    `limit <= 0` : aucun plafond (le cas par défaut)."""
    if threshold is None:
        threshold = _get_scenario_threshold(scenario_id)
    sql = text("""
        SELECT d.id, d.title, d.abstract, d.year, d.doi, d.study_design,
               d.quality_score, d.citation_count, d.source,
               COALESCE(ars.similarity_score, 0) AS similarity
        FROM literature_document d
        JOIN article_scenarios ars ON ars.document_id = d.id
        WHERE ars.scenario_id = :sid
          AND d.is_duplicate IS NOT TRUE
          AND d.abstract IS NOT NULL
          AND COALESCE(ars.screening_status, d.screening_status) IS DISTINCT FROM 'excluded'
          AND (COALESCE(ars.screening_status, d.screening_status) = 'included'
               OR COALESCE(ars.similarity_score, 0) >= :thr)
          AND (d.title || ' ' || d.abstract) ~* :rx
        ORDER BY
          CASE WHEN COALESCE(d.study_design, '') ~* 'systematic|meta-analy|méta-analy' THEN 0 ELSE 1 END,
          d.quality_score DESC NULLS LAST,
          d.citation_count DESC NULLS LAST,
          d.year DESC NULLS LAST,
          d.id
        LIMIT :cap
    """)
    # NULL en LIMIT = pas de limite en SQL : le défaut lit tout le corpus pertinent.
    _cap = int(limit) if int(limit or 0) > 0 else None
    with engine.connect() as conn:
        rows = [dict(r) for r in conn.execute(
            sql, {"sid": scenario_id, "thr": threshold, "rx": _param_regex(boundary=r"\y"),
                  "cap": _cap}).mappings().all()]
    for r in rows:
        r["params_mentioned"] = params_mentioned(f"{r.get('title') or ''} {r.get('abstract') or ''}")
    return [r for r in rows if r["params_mentioned"]]


_EPI_EXTRACT_SYSTEM = (
    "You extract epidemiological parameters from study abstracts, for a compartmental "
    "(SEIR family) model. Report ONLY values the abstract states for the disease under "
    "study: never infer, never carry a value over from another disease, never invent. "
    "Normalise units: r0 is a ratio; serial_interval_days, incubation_period_days, "
    "infectious_period_days and immunity_duration_days are in DAYS; cfr is a PROPORTION "
    "between 0 and 1 (a case fatality of 1.5 percent is 0.015). When a study gives a "
    "range or an interval, put its central value in value and the bounds in ci_low and "
    "ci_high. Return ONLY JSON: {\"articles\": [{\"id\": <id>, \"disease\": <name or null>, "
    "\"parameters\": [{\"name\": <one of r0, serial_interval_days, incubation_period_days, "
    "infectious_period_days, cfr, immunity_duration_days>, \"value\": <number>, "
    "\"ci_low\": <number or null>, \"ci_high\": <number or null>}]}]}. An article that "
    "reports no usable value gets an empty parameters list."
)


def _epi_llm_client():
    """Client LLM de l'extraction (une seule couture, que les tests remplacent)."""
    from llm_usage import MeteredOpenAI as _OAI
    return _OAI(timeout=120.0)


def _epi_extract_batch(client, batch: list[dict], disease_hint: str | None) -> list[dict]:
    """Un appel LLM pour un lot d'articles. Renvoie [{id, disease, parameters:[...]}].
    Robuste : un lot perdu renvoie []."""
    import json as _json
    items = [{
        "id": int(a["id"]),
        "title": (a.get("title") or "")[:300],
        "abstract": (a.get("abstract") or "")[:2000],
        "looks_like": a.get("params_mentioned") or [],
    } for a in batch]
    payload = {"disease_of_interest": disease_hint or None, "articles": items}
    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "system", "content": _EPI_EXTRACT_SYSTEM},
                      {"role": "user", "content": _json.dumps(payload, ensure_ascii=False)}],
            temperature=0, seed=42, max_tokens=3000,
            response_format={"type": "json_object"},
        )
        data = _json.loads(resp.choices[0].message.content)
    except Exception as e:                                   # noqa: BLE001
        logger.warning(f"epidemic parameter extraction batch: {e}")
        return []
    return [a for a in (data.get("articles") or []) if isinstance(a, dict)] if isinstance(data, dict) else []


def extract_epidemic_observations(scenario_id: str, disease_hint: str | None = None,
                                  threshold: float | None = None,
                                  max_articles: int = 0) -> dict[str, Any]:
    """Cherche dans TOUT le corpus pertinent les articles qui rapportent un paramètre
    épidémiologique et en extrait une observation par étude. Aucun échantillonnage :
    `max_articles <= 0` (le défaut) lit tous les articles qui en mesurent un.

    Renvoie ``{params: {nom: {observations: [{article_id, value, ci_low, ci_high}]}},
    disease, n_candidates, n_articles_used, n_with_values}``. Sans clé OpenAI :
    les candidats sont comptés, aucune observation n'est produite."""
    import os as _os
    from concurrent.futures import ThreadPoolExecutor

    candidates = _parameter_candidate_articles(scenario_id, threshold, max_articles or 0)
    # `articles` : de quoi TRACER la provenance. Ces articles sont hors des 25 plus
    # pertinents qui servent au reste du spec ; sans eux dans le pool de provenance,
    # _attach_model_spec filtrerait justement les observations qu'on vient de mesurer.
    out: dict[str, Any] = {
        "params": {}, "disease": None, "n_candidates": len(candidates),
        "n_articles_used": 0, "n_with_values": 0,
        "articles": [{"id": a["id"], "title": a.get("title"), "year": a.get("year"),
                      "doi": a.get("doi"), "quality_score": a.get("quality_score"),
                      "study_design": a.get("study_design"), "source": a.get("source"),
                      "params_mentioned": a.get("params_mentioned")} for a in candidates],
    }
    if not candidates or not _os.getenv("OPENAI_API_KEY"):
        return out
    try:
        client = _epi_llm_client()
    except Exception as _e:                                  # noqa: BLE001 - SDK absent ou clé illisible
        logger.warning(f"Extraction des paramètres {scenario_id}: client LLM indisponible ({_e})")
        return out                                           # les candidats restent comptés
    batches = [candidates[i:i + _EPI_PARAM_BATCH] for i in range(0, len(candidates), _EPI_PARAM_BATCH)]
    valid_ids = {int(a["id"]) for a in candidates}
    diseases: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=_EPI_PARAM_WORKERS) as ex:
        results = list(ex.map(lambda b: _epi_extract_batch(client, b, disease_hint), batches))
    import seir_model as _seir
    for res in results:
        for art in res:
            try:
                aid = int(art.get("id"))
            except (TypeError, ValueError):
                continue
            if aid not in valid_ids:
                continue                                     # id inventé : ignoré
            got = False
            for p in (art.get("parameters") or []):
                if not isinstance(p, dict):
                    continue
                name = str(p.get("name") or "").strip()
                val = _seir._num_or_none(p.get("value"))
                if name not in _PARAM_PHRASES or val is None:
                    continue
                obs = {"article_id": aid, "value": val}
                lo, hi = _seir._num_or_none(p.get("ci_low")), _seir._num_or_none(p.get("ci_high"))
                if lo is not None and hi is not None and lo < hi:
                    obs["ci_low"], obs["ci_high"] = lo, hi
                out["params"].setdefault(name, {"observations": []})["observations"].append(obs)
                got = True
            if got:
                out["n_with_values"] += 1
                _d = str(art.get("disease") or "").strip()
                if _d:
                    diseases[_d] = diseases.get(_d, 0) + 1
    out["n_articles_used"] = len(candidates)
    if diseases:
        out["disease"] = max(diseases.items(), key=lambda kv: kv[1])[0]
    logger.info(f"Paramètres épidémiologiques {scenario_id}: {len(candidates)} articles candidats, "
                f"{out['n_with_values']} avec une valeur, paramètres={sorted(out['params'])}")
    return out


def merge_epidemic_observations(block, targeted) -> dict:
    """Fusionne les observations CIBLÉES dans le bloc `epidemic_parameters` du LLM
    narratif (PUR, testé hors ligne).

    Les observations ciblées viennent d'articles qui MESURENT le paramètre : elles
    s'ajoutent à celles du bloc narratif, dédupliquées par (paramètre, article). Un
    paramètre absent du bloc narratif est créé. `applicable` passe à true lorsqu'un
    paramètre de TRANSMISSION est mesuré par au moins un article : le corpus le
    démontre, quoi qu'ait répondu le premier appel sur ses 25 articles."""
    merged = dict(block) if isinstance(block, dict) else {}
    tparams = (targeted or {}).get("params") or {}
    for name, blk in tparams.items():
        cur = dict(merged.get(name)) if isinstance(merged.get(name), dict) else {}
        obs = [o for o in (cur.get("observations") or []) if isinstance(o, dict)]
        seen = {(o.get("article_id"), o.get("value")) for o in obs}
        for o in blk.get("observations") or []:
            if (o.get("article_id"), o.get("value")) not in seen:
                obs.append(o)
                seen.add((o.get("article_id"), o.get("value")))
        cur["observations"] = obs
        # Provenance = les articles réellement mesurés (l'UI les rend cliquables).
        prov = list(dict.fromkeys([o["article_id"] for o in obs if o.get("article_id") is not None]))
        if prov:
            cur["provenance"] = prov
            cur["n_studies"] = len(prov)
        cur.setdefault("unit", "ratio" if name == "r0" else ("proportion" if name == "cfr" else "days"))
        merged[name] = cur
    if any(tparams.get(p, {}).get("observations") for p in _TRANSMISSION_PARAMS):
        merged["applicable"] = True
    if (targeted or {}).get("disease") and not merged.get("population_disease"):
        merged["population_disease"] = targeted["disease"]
    return merged


# ─── MODEL SPEC (Phase 1) : schéma machine + provenance ──────────────────────
# La littérature définit la SPÉCIFICATION du modèle (outcome, variables
# explicatives, algorithme). Les données d'entraînement viendront ensuite de
# l'utilisateur (CSV/XLSX) et de flux publics. Ces helpers normalisent la sortie
# LLM en un spec déterministe, exploitable par la machine, et tracé (provenance)
# vers les articles sources. Tout est ADDITIF : les clés existantes de
# variables_json restent intactes pour ne pas casser le frontend.

MODEL_SPEC_SCHEMA = "model_spec/1.0"

_TASK_TYPES = {"classification", "regression", "count", "survival"}
_DTYPES = {"float", "int", "bool", "category", "datetime"}
_FEATURE_SOURCES = {"user", "public_api", "seir"}
_ALGO_FAMILIES = {
    "gradient_boosting", "lightgbm", "xgboost", "random_forest", "logistic_regression",
    "linear_regression", "elasticnet", "svm", "mlp", "cox_ph", "knn",
    "prophet", "sarimax", "extremal_rf",
}
# Familles de PRÉVISION de série temporelle (routées hors du flux tabulaire par
# model_trainer). Une cible numérique s'impose → task_type ramené à 'regression'.
_TS_ALGO_FAMILIES = {"prophet", "sarimax"}
_METRICS = {"roc_auc", "average_precision", "rmse", "mae", "r2", "c_index"}
_CV_STRATEGIES = {"stratified_kfold", "kfold", "timeseries"}


def _slug_identifier(name: str, used: set[str]) -> str:
    """snake_case, identifiant valide et unique (pour colonnes CSV/DataFrame)."""
    import re as _re
    import unicodedata as _ud
    # Replier les accents (é -> e) avant de slugifier, sinon ils deviennent des "_".
    folded = _ud.normalize("NFKD", name or "").encode("ascii", "ignore").decode("ascii")
    base = _re.sub(r"[^a-z0-9]+", "_", folded.strip().lower()).strip("_")
    if not base or not _re.match(r"^[a-z_]", base):
        base = ("var_" + base).strip("_") if base else "var"
    candidate, i = base, 2
    while candidate in used:
        candidate = f"{base}_{i}"
        i += 1
    used.add(candidate)
    return candidate


def _coerce_enum(value: Any, allowed: set[str], default: str) -> str:
    v = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return v if v in allowed else default


def _infer_algo_family(text_blob: str) -> str:
    t = (text_blob or "").lower()
    table = [
        (("extremal", "quantile forest", "quantile regression forest", "forêt quantile", "foret quantile", "surge", "extreme quantile"), "extremal_rf"),
        (("xgboost", "lightgbm", "gradient boost", "gradient_boost", "boosting", "gbm"), "gradient_boosting"),
        (("random forest", "random_forest", "forêt aléatoire", "foret aleatoire"), "random_forest"),
        (("logistic", "logistique"), "logistic_regression"),
        (("cox", "proportional hazard", "survie", "survival"), "cox_ph"),
        (("ridge", "lasso", "elastic"), "elasticnet"),
        (("linear regression", "régression linéaire", "regression lineaire", "ols", "moindres carrés"), "linear_regression"),
        (("svm", "support vector"), "svm"),
        (("neural", "mlp", "deep", "réseau de neur", "reseau de neur"), "mlp"),
        (("knn", "nearest neighbor", "plus proches voisins"), "knn"),
    ]
    for keys, fam in table:
        if any(k in t for k in keys):
            return fam
    return "gradient_boosting"


def _dtype_for_var_type(var_type: Any) -> str:
    return {
        "continuous": "float", "binary": "bool", "categorical": "category",
        "time_series": "float", "count": "int", "integer": "int",
    }.get(str(var_type or "").strip().lower(), "float")


def _infer_feature_source(data_source: str, declared: str) -> tuple[str, str | None]:
    """Retourne (source, public_provider) - déclaré LLM sinon heuristique texte."""
    declared = (declared or "").strip().lower()
    if declared in _FEATURE_SOURCES:
        src = declared
    else:
        ds = (data_source or "").lower()
        public_hint = any(k in ds for k in (
            "météo", "meteo", "weather", "temperature", "température", "forecast",
            "prévision", "prevision", "open-meteo", "openmeteo", "insee", "open data",
            "opendata", "données publiques", "donnees publiques", "santé publique",
            "sentinel", "réseau sentinelles", "reseau sentinelles", "pollen", "air quality",
        ))
        src = "public_api" if public_hint else "user"
    provider = None
    if src == "public_api":
        ds = (data_source or "").lower()
        if any(k in ds for k in ("météo", "meteo", "weather", "temp", "forecast", "prévision", "prevision", "open-meteo")):
            provider = "open-meteo"
    return src, provider


def _filter_provenance(raw: Any, valid_ids: set) -> list:
    """Ne garde que les ids réellement présents dans le contexte fourni au LLM."""
    out: list = []
    for x in (raw or []):
        try:
            xi = int(x)
        except (TypeError, ValueError):
            continue
        if xi in valid_ids and xi not in out:
            out.append(xi)
    return out


_TARGET_DTYPE_FOR_TASK = {"classification": "category", "regression": "float", "count": "int", "survival": "float"}


def _derive_data_template(outcome: dict, features: list[dict]) -> dict:
    """Dérive le data_template (noms de colonnes EXACTS attendus à l'upload) depuis
    l'outcome + les features. Fonction PURE partagée par le générateur de spec et
    l'éditeur de spec, pour que le template ne DÉRIVE JAMAIS de la liste réelle des
    variables (sinon la validation d'upload attendrait des colonnes fantômes)."""
    task_type = (outcome.get("task_type") or "classification").strip().lower()
    outcome_mn = outcome.get("machine_name") or "outcome"
    target_dtype = _TARGET_DTYPE_FOR_TASK.get(task_type, "float")
    columns = [{
        "name": outcome_mn, "dtype": target_dtype, "role": "outcome",
        "required": True, "source": "user", "description": outcome.get("name", ""),
    }]
    for f in features:
        col = {
            "name": f["machine_name"], "dtype": f["dtype"], "role": "feature",
            "required": f.get("importance") == "high",
            "source": f.get("source", "user"), "public_provider": f.get("public_provider"),
            "description": f.get("name", ""),
        }
        if f.get("seir_column"):
            col["seir_column"] = f["seir_column"]  # dérivée du sous-modèle SEIR
        columns.append(col)
    return {
        "target_column": outcome_mn,
        "columns": columns,
        "formats": ["csv", "xlsx"],
        "user_columns": [c["name"] for c in columns if c["source"] == "user"],
        "public_columns": [c["name"] for c in columns if c["source"] == "public_api"],
        # Colonnes dérivées du sous-modèle SEIR (remplies automatiquement, pas d'upload).
        "seir_columns": [c["name"] for c in columns if c["source"] == "seir"],
        "notes": ("Les en-têtes du fichier doivent correspondre EXACTEMENT à ces noms. "
                  "Les colonnes 'public_api' pourront être récupérées automatiquement (Phase 2). "
                  "Les colonnes 'seir' sont DÉRIVÉES du sous-modèle épidémique (aucun upload)."),
    }


def _coerce_family_for_task(family: Any, task_type: str) -> str:
    """Ramène une famille d'algorithme choisie vers une famille COMPATIBLE avec la
    tâche (sans vérifier la présence du paquet - c'est le rôle du trainer). Ex.:
    logistic_regression sur une régression -> linear_regression, et inversement."""
    fam = _coerce_enum(family, _ALGO_FAMILIES, "gradient_boosting")
    tt = (task_type or "classification").strip().lower()
    if tt in ("regression", "count"):
        if fam in ("logistic_regression",):
            return "linear_regression"
    else:  # classification / survival
        if fam in ("linear_regression", "elasticnet"):
            return "logistic_regression"
    return fam


def _attach_model_spec(variables: dict, prov_articles: list[dict]) -> dict:
    """
    Construit un `model_spec` déterministe (outcome, features, algorithme,
    data_template) + un index de provenance, à partir de la sortie LLM.
    Robuste : si le LLM omet les champs machine, ils sont reconstruits depuis
    les champs humains existants. N'altère aucune clé existante (ajoute
    seulement `machine_name` en cross-link et les blocs `model_spec`/_provenance_index).
    """
    valid_ids = {a["id"] for a in prov_articles if a.get("id") is not None}
    prov_meta = {
        a["id"]: {"title": (a.get("title") or "")[:160], "year": a.get("year"), "doi": a.get("doi")}
        for a in prov_articles if a.get("id") is not None
    }
    # Poids qualité par article (quality_score déterministe du corpus) → regroupement
    # pondéré des paramètres épidémiologiques par étude (seir_model.pool_weighted).
    quality_by_id = {
        a["id"]: a.get("quality_score")
        for a in prov_articles if a.get("id") is not None
    }
    used: set[str] = set()
    cited: set = set()

    # ── Outcome ──
    po = variables.get("primary_outcome") or {}
    outcome_mn = _slug_identifier(po.get("machine_name") or po.get("name") or "outcome", used)
    task_type = _coerce_enum(po.get("task_type"), _TASK_TYPES, "classification")
    outcome_prov = _filter_provenance(po.get("provenance"), valid_ids)
    cited.update(outcome_prov)
    po["machine_name"] = outcome_mn  # cross-link additif
    po["task_type"] = task_type
    outcome = {
        "name": po.get("name", ""),
        "machine_name": outcome_mn,
        "task_type": task_type,
        "unit": po.get("unit") or po.get("measurement") or "",
        "positive_class": po.get("positive_class") if task_type == "classification" else None,
        "provenance": outcome_prov,
    }

    # ── Features ──
    features = []
    has_time_series = False
    for pv in (variables.get("predictor_variables") or []):
        mn = _slug_identifier(pv.get("machine_name") or pv.get("name") or "feature", used)
        dtype = _coerce_enum(pv.get("dtype"), _DTYPES, _dtype_for_var_type(pv.get("type")))
        source, provider = _infer_feature_source(pv.get("data_source", ""), pv.get("source", ""))
        prov = _filter_provenance(pv.get("provenance"), valid_ids)
        cited.update(prov)
        if str(pv.get("type", "")).strip().lower() == "time_series" or dtype == "datetime":
            has_time_series = True
        pv["machine_name"] = mn  # cross-link additif
        features.append({
            "name": pv.get("name", ""),
            "machine_name": mn,
            "dtype": dtype,
            "source": source,
            "public_provider": provider,
            "importance": _coerce_enum(pv.get("importance"), {"high", "medium", "low"}, "medium"),
            "provenance": prov,
        })

    # ── Algorithme ──
    ra = variables.get("recommended_algorithm") or {}
    family = _coerce_enum(
        ra.get("family"), _ALGO_FAMILIES,
        _infer_algo_family(f"{ra.get('primary', '')} {' '.join(ra.get('alternatives') or [])}"),
    )
    metric = _coerce_enum(
        ra.get("metric"), _METRICS,
        {"classification": "roc_auc", "regression": "rmse", "count": "rmse", "survival": "c_index"}[task_type],
    )
    default_cv = "timeseries" if has_time_series else ("stratified_kfold" if task_type == "classification" else "kfold")
    cv_strategy = _coerce_enum(ra.get("cv_strategy"), _CV_STRATEGIES, default_cv)
    try:
        cv_folds = int(ra.get("cv_folds") or 5)
    except (TypeError, ValueError):
        cv_folds = 5
    cv_folds = min(max(cv_folds, 3), 10)
    algo_prov = _filter_provenance(ra.get("provenance"), valid_ids)
    cited.update(algo_prov)
    candidates = [c for c in (_coerce_enum(x, _ALGO_FAMILIES, "") for x in (ra.get("alternatives") or [])) if c]
    algorithm = {
        "family": family,
        "candidates": candidates,
        "rationale": ra.get("rationale", ""),
        "cv": {"strategy": cv_strategy, "folds": cv_folds},
        "metric": metric,
        "provenance": algo_prov,
    }

    # ── Seuils d'alerte (modalités green/orange/red) : on filtre leur provenance
    # sur le POOL PERTINENT (mêmes valid_ids que le reste du spec) pour garantir
    # que chaque modalité est bien sourcée par un article réellement retenu, et on
    # réécrit la provenance nettoyée pour que l'UI puisse lier les articles. ──
    at = variables.get("alert_thresholds")
    if isinstance(at, dict):
        for _lvl in ("green", "orange", "red"):
            band = at.get(_lvl)
            if isinstance(band, dict):
                band_prov = _filter_provenance(band.get("provenance"), valid_ids)
                band["provenance"] = band_prov
                cited.update(band_prov)

    # ── Paramètres épidémiologiques (famille SEIR) ──────────────────────────────
    # Nettoyés (nombres coercés, provenance filtrée sur le pool pertinent) par le
    # module PUR seir_model (même logique testée hors-ligne). Bloc VIDE si le scénario
    # n'est pas une maladie transmissible ou si rien n'est rapporté → le modèle
    # compartimental (Phase 3) ne s'active tout simplement pas.
    import seir_model as _seir
    _epi = _seir.normalize_extracted_parameters(
        variables.get("epidemic_parameters"), valid_ids, quality_by_id
    )
    cited.update(_epi["cited"])

    # ── Le SEIR comme SOUS-MODÈLE : reclasser les variables qu'il DÉRIVE ─────────
    # Scénario épidémique (paramètres SEIR extraits) → une variable qui EST un paramètre
    # du modèle (R0, CFR, incubation…) n'est PAS une feature du prédicteur : on la retire
    # (elle vit dans epidemic_parameters). Une variable qui est une SORTIE du modèle
    # (incidence, prévalence, cumul, décès) RESTE une feature mais sa source devient
    # "seir" : remplie automatiquement par le connecteur SEIR (aucun upload / API externe).
    if _epi.get("applicable"):
        _pv_by_mn = {pv.get("machine_name"): pv
                     for pv in (variables.get("predictor_variables") or []) if isinstance(pv, dict)}
        _kept = []
        for f in features:
            _txt = f"{f.get('name', '')} {f.get('machine_name', '')}"
            _pv = _pv_by_mn.get(f["machine_name"])
            if _seir.is_seir_parameter(_txt):
                if _pv is not None:
                    _pv["_seir_role"] = "parameter"  # cross-link additif pour l'UI
                continue  # input de simulation → jamais une feature
            _col = _seir.seir_feature_column(_txt)
            if _col:
                f["source"], f["public_provider"], f["seir_column"] = "seir", None, _col
                if _pv is not None:
                    _pv["source"], _pv["_seir_role"], _pv["_seir_column"] = "seir", "derived", _col
            _kept.append(f)
        features = _kept

    # ── Data template (dérivé → garanti cohérent avec les machine_name ci-dessus) ──
    data_template = _derive_data_template(outcome, features)

    variables["model_spec"] = {
        "schema": MODEL_SPEC_SCHEMA,
        "version": 1,
        "outcome": outcome,
        "features": features,
        "algorithm": algorithm,
        "data_template": data_template,
        "epidemic_parameters": {
            "applicable": _epi["applicable"],
            "disease": _epi["disease"],
            "params": _epi["params"],
        },
    }
    variables["_provenance_index"] = {str(i): prov_meta[i] for i in sorted(cited) if i in prov_meta}
    return variables


def _loads_lenient(raw: str) -> dict:
    """`json.loads` TOLÉRANT aux sorties LLM tronquées (max_tokens atteint) : retire un
    éventuel fence ```json, puis, si le JSON est incomplet, ferme les chaînes / tableaux /
    objets restés ouverts en rognant la fin jusqu'à obtenir un objet valide. Best-effort :
    renvoie le meilleur dict récupérable (au pire {}) - les champs manquants sont
    reconstruits en aval (_attach_model_spec dérive les champs machine des champs humains).
    Évite qu'une réponse coupée d'un caractère fasse échouer TOUTE la génération."""
    import json as _j
    s = (raw or "").strip()
    if s.startswith("```"):                                   # fence markdown éventuel
        s = s.strip("`")
        s = s[4:] if s[:4].lower() == "json" else s
        s = s.strip()
    try:
        return _j.loads(s)
    except Exception:
        pass
    # Réparation : on tente de fermer les structures ouvertes à des points de coupe
    # proches de la fin (là où la troncature s'est produite). On borne les essais.
    cut_points = [i for i, ch in enumerate(s) if ch in ',}]"']
    for cut in reversed(cut_points[-3000:]):
        frag = s[:cut + 1].rstrip().rstrip(",")
        if frag.count('"') % 2 == 1:                          # chaîne ouverte → la fermer
            frag = frag[:-1] if frag.endswith('"') else frag + '"'
        opens = frag.count("{") - frag.count("}")
        obrk = frag.count("[") - frag.count("]")
        if opens < 0 or obrk < 0:
            continue
        cand = frag + ("]" * obrk) + ("}" * opens)
        try:
            out = _j.loads(cand)
            if isinstance(out, dict) and out:
                logger.warning("Variables/JSON LLM tronqué → réparé (best-effort, %d/%d caractères).",
                               cut + 1, len(s))
                return out
        except Exception:
            continue
    return {}


def _generate_variables_from_pico(scenario_id: str, persist: str = "active", lang: str | None = None) -> dict[str, Any]:
    """
    Génère automatiquement les variables du modèle et l'outcome à partir des articles
    pertinents (PICO structuré + titre/résumé + extrait de texte intégral quand
    disponible), et non des seuls PICO. Sauvegarde dans scenario_settings.variables_json.
    """
    import json as _json
    from datetime import datetime, timezone
    from llm_usage import MeteredOpenAI as _OAI

    threshold = _get_scenario_threshold(scenario_id)
    # Résumés/PICO/texte intégral pour les 25 articles du contexte seulement.
    articles = _get_above_threshold_articles(scenario_id, threshold, include_fulltext=True,
                                             fulltext_query=_get_scenario_name(scenario_id),
                                             fulltext_top_docs=25, fulltext_char_cap=2200,
                                             full_rows=25)

    # On n'EXIGE plus un PICO extrait : tout article pertinent contribue au choix des
    # variables/outcome/algorithme via son titre+abstract (+texte intégral si dispo).
    # Le PICO, quand présent, ajoute la structure P/I/C/O. Auparavant, un article
    # pertinent sans PICO était purement ignoré (perte de signal injustifiée).
    # `pico_articles` = désormais l'ensemble pertinent complet (nom conservé pour le
    # reste du flux ; le PICO reste optionnel par article).
    pico_articles = articles
    if not pico_articles:
        return {"error": _msg(lang, "Aucun article au-dessus du seuil pour générer les variables.",
                              "No article above the threshold to generate the variables.")}
    _n_with_pico = sum(1 for a in pico_articles if a.get("has_pico") or a.get("pico_json"))

    scenario_name = _get_scenario_name(scenario_id)

    # Contexte à trois niveaux (PICO structuré → abstract → extrait de TEXTE
    # INTÉGRAL quand disponible) : le LLM choisit les prédicteurs/métriques sur le
    # texte réel des articles, pas seulement sur la compression PICO.
    pico_context = []
    for a in pico_articles[:25]:
        pj = a.get("pico_json") or {}
        _entry = {
            "id": a.get("id"),
            "title": a.get("title", "")[:100],
            "year": a.get("year"),
            "study_design": a.get("study_design") or pj.get("study_design", ""),
            "P": pj.get("population", pj.get("P", "")),
            "I": pj.get("intervention", pj.get("I", "")),
            "C": pj.get("comparator", pj.get("C", "")),
            "O": pj.get("outcome", pj.get("O", "")),
            "key_finding": pj.get("key_finding", pj.get("conclusion", "")),
            "abstract": (a.get("abstract") or "")[:1200],
        }
        _ft = (a.get("fulltext") or "").strip()
        if _ft:
            _entry["fulltext_excerpt"] = _ft
        pico_context.append(_entry)

    context_str = _json.dumps(pico_context, ensure_ascii=False, indent=2)

    # Le spec engage le corpus ENTIER : le digest (agrégats sur TOUS les articles
    # pertinents, sans échantillonnage) précède les 25 articles reproduits, qui servent
    # à choisir des variables concrètes et à citer.
    from .digest import corpus_digest, digest_coverage_note, digest_to_prompt
    _digest = corpus_digest(scenario_id, threshold)
    _digest_block = digest_to_prompt(_digest)
    _coverage = digest_coverage_note(_digest, len(pico_context))

    system_prompt = """Tu es un expert en modélisation prédictive appliquée à la santé.
A partir d'une revue systématique de la littérature, tu identifies les variables clés,
l'outcome principal, et le meilleur algorithme pour un modèle prédictif.
Tu génères un JSON structuré. Ne pas utiliser de tiret cadratin (em dash).""" + _llm_lang_directive(lang)

    user_prompt = f"""Scénario : "{scenario_name}"

{_digest_block or f"Basé sur {len(pico_articles)} articles pertinents ({_n_with_pico} avec PICO extrait)."}

{_coverage}

Articles reproduits ({len(pico_context)} les mieux établis ; PICO, abstract et extrait de texte intégral quand disponibles) :

{context_str}

Génère un JSON avec EXACTEMENT ces champs :
{{
  "primary_outcome": {{
    "name": "Nom de l'outcome principal",
    "definition": "Définition clinique précise",
    "measurement": "Comment le mesurer",
    "timeframe": "Horizon temporel",
    "machine_name": "identifiant_snake_case_court",
    "task_type": "classification|regression|count|survival",
    "unit": "Unité de mesure (ex: bool, jours, /100k)",
    "positive_class": "Classe positive si classification, sinon null",
    "provenance": [ids d'articles de la liste ci-dessus soutenant cet outcome]
  }},
  "secondary_outcomes": [
    {{"name": "...", "definition": "..."}}
  ],
  "predictor_variables": [
    {{
      "name": "Nom de la variable",
      "type": "continuous|binary|categorical|time_series",
      "definition": "Définition clinique",
      "data_source": "Source de données recommandée",
      "importance": "high|medium|low",
      "evidence_level": "Nombre d'études qui la mentionnent",
      "machine_name": "identifiant_snake_case_court (ex: temp_max_j1)",
      "dtype": "float|int|bool|category|datetime",
      "source": "user (fournie par l'utilisateur) | public_api (récupérable: météo, open data...) | seir (DÉRIVÉE du sous-modèle épidémique: incidence/prévalence/cumul/décès)",
      "public_provider": "open-meteo si météo, sinon null",
      "provenance": [ids d'articles de la liste ci-dessus mentionnant cette variable]
    }}
  ],
  "recommended_algorithm": {{
    "primary": "Algorithme principal recommandé",
    "alternatives": ["Alternative 1", "Alternative 2"],
    "rationale": "Justification basée sur la littérature",
    "validation_method": "Méthode de validation recommandée",
    "family": "gradient_boosting|random_forest|logistic_regression|linear_regression|elasticnet|svm|mlp|cox_ph|knn",
    "metric": "roc_auc|average_precision|rmse|mae|c_index",
    "cv_strategy": "stratified_kfold|kfold|timeseries",
    "cv_folds": 5,
    "provenance": [ids d'articles de la liste ci-dessus justifiant l'algorithme]
  }},
  "required_databases": ["Base 1", "Base 2"],
  "sample_size_recommendation": "Estimation de la taille d'échantillon nécessaire",
  "update_frequency": "Fréquence de mise à jour recommandée",
  "alert_thresholds": {{
    "green":  {{"label": "Normal",  "range": "Plage de valeurs de l'OUTCOME considérée normale, NUMÉRIQUE et cohérente avec son unit/task_type (ex: '< 0.10' pour une probabilité, '< 5 /100k' pour un taux, '0-2' pour un compte)", "rationale": "Justification fondée sur les évidences (cite les seuils rapportés dans la littérature si disponibles)", "provenance": [ids d'articles]}},
    "orange": {{"label": "Tension", "range": "Plage intermédiaire (ex: '0.10-0.30')", "rationale": "...", "provenance": [ids d'articles]}},
    "red":    {{"label": "Alerte",  "range": "Plage critique (ex: '> 0.30')", "rationale": "...", "provenance": [ids d'articles]}}
  }},
  "epidemic_parameters": {{
    "applicable": true si le scénario porte sur une MALADIE TRANSMISSIBLE (épidémie / infection) modélisable par un compartimental SEIR, false sinon,
    "population_disease": "maladie / agent pathogène concerné si applicable, sinon null",
    "r0": {{"value": nombre|null, "ci_low": nombre|null, "ci_high": nombre|null, "unit": "ratio", "n_studies": entier, "provenance": [ids d'articles rapportant cette valeur], "observations": [{{"article_id": id, "value": nombre}}]}},
    "infectious_period_days": {{"value": nombre|null, "ci_low": nombre|null, "ci_high": nombre|null, "unit": "days", "n_studies": entier, "provenance": [ids], "observations": [{{"article_id": id, "value": nombre}}]}},
    "incubation_period_days": {{"value": nombre|null, "ci_low": nombre|null, "ci_high": nombre|null, "unit": "days", "n_studies": entier, "provenance": [ids], "observations": [{{"article_id": id, "value": nombre}}]}},
    "cfr": {{"value": nombre|null, "ci_low": nombre|null, "ci_high": nombre|null, "unit": "proportion", "n_studies": entier, "provenance": [ids], "observations": [{{"article_id": id, "value": nombre}}]}},
    "immunity_duration_days": {{"value": nombre|null, "ci_low": nombre|null, "ci_high": nombre|null, "unit": "days", "n_studies": entier, "provenance": [ids], "observations": [{{"article_id": id, "value": nombre}}]}},
    "serial_interval_days": {{"value": nombre|null, "ci_low": nombre|null, "ci_high": nombre|null, "unit": "days", "n_studies": entier, "provenance": [ids], "observations": [{{"article_id": id, "value": nombre}}]}}
  }},
  "implementation_notes": "Notes d'implémentation pratiques",
  "validation_status": "pending"
}}

IMPORTANT pour les champs "provenance" : ce sont des listes d'identifiants (champ "id")
des articles figurant dans la liste ci-dessus. N'invente AUCUN id : n'utilise que des id
réellement présents. Les "machine_name" doivent être de courts identifiants snake_case
(minuscules, chiffres, underscores) utilisables comme noms de colonnes.

IMPORTANT pour "alert_thresholds" : donne des plages de valeurs NUMÉRIQUES concrètes de
l'OUTCOME (pas des phrases vagues), cohérentes avec son "unit" et son "task_type", et
fondées sur les seuils rapportés dans les évidences quand ils existent (sinon des seuils
cliniquement plausibles). Les trois plages doivent être contiguës et couvrir tout le
domaine. Tu PEUX renommer les catégories si l'outcome s'y prête (ex. pour un compte :
"Faible"/"Modéré"/"Élevé"), mais garde 3 niveaux du plus sûr au plus critique.

IMPORTANT - le SEIR est un SOUS-MODÈLE, pas des features (scénarios de maladie
transmissible) : ne mets JAMAIS un PARAMÈTRE du modèle compartimental (R0, Rt, létalité/
CFR, période d'incubation, période infectieuse, durée d'immunité, intervalle sériel) dans
"predictor_variables" - ces valeurs vont dans "epidemic_parameters", pas dans les features.
En revanche, une SORTIE du sous-modèle (incidence, prévalence, cas actifs, infections
cumulées, décès) PEUT être une variable prédictive utile : liste-la normalement mais mets
son "source": "seir" - le serveur la remplira automatiquement depuis le sous-modèle
(aucune donnée externe requise). Les autres variables (météo, vaccination, mobilité, lits…)
gardent "source": "user" ou "public_api".

IMPORTANT pour "epidemic_parameters" : ne renseigne ces paramètres qu'à partir de
valeurs RÉELLEMENT rapportées dans les articles ci-dessus (avec leur provenance) ; mets
"value": null pour tout paramètre non rapporté - n'invente AUCUN chiffre. Donne une
estimation centrale + un intervalle (ci_low/ci_high, idéalement l'IC 95 %) reflétant la
dispersion entre études quand plusieurs la rapportent. Pour agréger, PRIVILÉGIE les
estimations des synthèses de meilleure qualité (revues systématiques / méta-analyses,
cf. "study_design") sur les études isolées. "cfr" en PROPORTION (0..1, pas en
pourcentage). Pour CHAQUE paramètre, renseigne aussi "observations" : la liste des
valeurs par étude RÉELLEMENT rapportées ({{"article_id": id, "value": nombre}}), une
entrée par article rapportant ce paramètre (mets [] si aucune). Ces observations
individuelles servent à un regroupement pondéré par la qualité côté serveur. Si le
scénario ne porte PAS sur une maladie transmissible, mets "applicable": false et tous
les "value" à null.

Retourne UNIQUEMENT le JSON valide."""

    try:
        # ── Déterminisme du recheck ──────────────────────────────────────────
        # Empreinte de l'évidence = ensemble des articles (au-dessus du seuil) qui
        # alimentent le modèle. Si elle est INCHANGÉE depuis le dernier spec validé,
        # on RÉUTILISE ce spec tel quel : « rechecker » sur la même évidence donne
        # exactement le même résultat (plus d'appel LLM non déterministe). On ne
        # régénère via le LLM QUE lorsque l'évidence change réellement.
        # Empreinte partagée avec l'Evidence Brief (_evidence_fingerprint) : ensemble
        # ORDONNÉ des IDs pertinents + seuil + langue + version de contexte. Le suffixe
        # de version invalide UNE FOIS les specs générés avec un contexte plus pauvre
        # (PICO seul, ou sans texte intégral) : régénérés puis réutilisés à évidence
        # constante. Seuil et langue inclus : un spec FR au seuil 0.45 ne doit pas être
        # resservi pour une requête EN ni pour un autre seuil.
        # v6 : les paramètres épidémiologiques viennent désormais d'une extraction CIBLÉE
        # sur les articles qui les mesurent, pas des seuls 25 plus pertinents. Le suffixe
        # invalide UNE FOIS les specs construits sans elle.
        _CTX_VERSION = "ctx-v7-full-corpus-digest"
        evidence_fingerprint = _evidence_fingerprint(
            [a.get("id") for a in pico_articles[:25] if a.get("id") is not None],
            threshold, lang, _CTX_VERSION)[:16]

        _reused = None
        with engine.connect() as _rc:
            _row = _rc.execute(text(
                "SELECT variables_json, variables_proposal_json FROM scenario_settings WHERE scenario_id = :sid"
            ), {"sid": scenario_id}).mappings().first()
        # On réutilise le spec validé (variables_json) en priorité, sinon la dernière
        # proposition (variables_proposal_json), si construit sur la MÊME évidence.
        for _slot in ("variables_json", "variables_proposal_json"):
            _cand = _row.get(_slot) if _row else None
            if (isinstance(_cand, dict)
                    and _cand.get("model_spec")
                    and _cand.get("_meta", {}).get("evidence_fingerprint") == evidence_fingerprint):
                _reused = _cand
                break

        if _reused is not None:
            variables = dict(_reused)
            logger.info(f"Variables {scenario_id}: évidence inchangée (fingerprint {evidence_fingerprint}) "
                        f"→ spec réutilisé (déterministe, sans appel LLM).")
        else:
            client = _OAI(timeout=90.0)
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                seed=42,
                max_tokens=8000,  # 3000 tronquait le spec (observations par étude + SEIR) → JSON invalide
                response_format={"type": "json_object"},
            )
            # Parse TOLÉRANT : une réponse coupée (rare à 8000 tokens) est réparée au lieu
            # de faire échouer toute la génération avec « Expecting ',' delimiter … ».
            variables = _loads_lenient(response.choices[0].message.content)
            if not variables:
                raise ValueError("Réponse LLM (variables) vide ou illisible même après réparation.")

            # Paramètres épidémiologiques : extraction CIBLÉE sur les articles du corpus
            # qui les MESURENT (les 25 plus pertinents n'en rapportent presque jamais).
            # Sans candidat, aucun appel LLM : un scénario non transmissible ne coûte rien.
            _epi_targeted: dict[str, Any] = {}
            try:
                _epi_targeted = extract_epidemic_observations(
                    scenario_id, disease_hint=scenario_name, threshold=threshold)
                if _epi_targeted.get("params"):
                    variables["epidemic_parameters"] = merge_epidemic_observations(
                        variables.get("epidemic_parameters"), _epi_targeted)
            except Exception as _epi_err:                    # jamais bloquant
                logger.warning(f"Extraction ciblée des paramètres {scenario_id}: {_epi_err}")

            # Phase 1 : normaliser en model_spec déterministe (machine_name, dtype,
            # algorithme/CV/métrique, data_template) + provenance tracée vers les articles.
            # Pool de provenance = les 25 articles du contexte UNION les articles qui ont
            # fourni une mesure (sinon leur provenance serait filtrée et le pooling vide).
            _prov_pool = list(pico_articles[:25])
            _prov_seen = {a.get("id") for a in _prov_pool}
            for _a in (_epi_targeted.get("articles") or []):
                if _a.get("id") not in _prov_seen:
                    _prov_pool.append(_a)
                    _prov_seen.add(_a.get("id"))
            try:
                variables = _attach_model_spec(variables, _prov_pool)
            except Exception as spec_err:  # le spec machine ne doit jamais bloquer la génération
                logger.error(f"model_spec build {scenario_id}: {spec_err}", exc_info=True)

        try:
            with engine.connect() as _cc:
                _corpus_total = _cc.execute(text("""
                    SELECT COUNT(*) FROM article_scenarios ars
                    JOIN literature_document d ON d.id = ars.document_id
                    WHERE ars.scenario_id = :sid AND d.is_duplicate IS NOT TRUE
                """), {"sid": scenario_id}).scalar() or 0
        except Exception:
            _corpus_total = len(articles)
        variables["_meta"] = {
            "scenario_id": scenario_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            # Outcome/variables/algorithme dérivés du sous-ensemble PERTINENT.
            "corpus_total": int(_corpus_total),            # corpus complet
            "relevant_total": len(articles),               # au-dessus du seuil
            "pico_articles_used": len(pico_articles),      # articles pertinents (entrée du modèle ; PICO optionnel par article)
            "auto_generated": True,
            "validation_status": "pending",
            "evidence_fingerprint": evidence_fingerprint,
            "reused": _reused is not None,
        }
        if _reused is None:
            # Combien d'articles du corpus MESURENT un paramètre, et combien en ont donné
            # une valeur : l'onglet Modèle peut dire « 0 sur 2 732 » au lieu de « aucun
            # paramètre extrait », qui laissait croire à une panne.
            variables["_meta"]["epidemic_parameter_candidates"] = int(_epi_targeted.get("n_candidates", 0))
            variables["_meta"]["epidemic_parameter_articles_with_values"] = int(_epi_targeted.get("n_with_values", 0))

        # Sauvegarder en DB. persist="proposal" (Phase 5) écrit dans un slot
        # de staging sans toucher le spec actif validé.
        with engine.begin() as conn:
            if persist == "proposal":
                conn.execute(text("""
                    INSERT INTO scenario_settings (scenario_id, variables_proposal_json, proposal_generated_at, updated_at)
                    VALUES (:sid, CAST(:vars AS jsonb), NOW(), NOW())
                    ON CONFLICT (scenario_id) DO UPDATE
                    SET variables_proposal_json = CAST(:vars AS jsonb),
                        proposal_generated_at = NOW(),
                        updated_at = NOW()
                """), {"sid": scenario_id, "vars": _json.dumps(variables)})
            else:
                # variables_lang = langue de génération (pour l'affichage localisé) ;
                # variables_i18n remis à NULL car les traductions en cache sont périmées.
                conn.execute(text("""
                    INSERT INTO scenario_settings (scenario_id, variables_json, variables_validated, variables_generated_at, variables_lang, variables_i18n, updated_at)
                    VALUES (:sid, CAST(:vars AS jsonb), FALSE, NOW(), :vlang, NULL, NOW())
                    ON CONFLICT (scenario_id) DO UPDATE
                    SET variables_json = CAST(:vars AS jsonb),
                        variables_validated = FALSE,
                        variables_generated_at = NOW(),
                        variables_lang = :vlang,
                        variables_i18n = NULL,
                        updated_at = NOW()
                """), {"sid": scenario_id, "vars": _json.dumps(variables), "vlang": (_norm_lang(lang) or "fr")})

        logger.info(f"Variables & Modèle générés ({persist}) pour {scenario_id}: {len(pico_articles)} articles PICO.")
        return variables

    except Exception as e:
        logger.error(f"Variables generation {scenario_id}: {e}", exc_info=True)
        return {"error": str(e)}


def _detect_variables_lang(variables: dict) -> str:
    """Heuristique FR/EN pour les lignes LEGACY dépourvues de variables_lang."""
    po = (variables or {}).get("primary_outcome") or {}
    sample = " ".join(str(po.get(k, "")) for k in ("name", "definition", "measurement")).lower()
    for pv in (variables.get("predictor_variables") or [])[:3]:
        sample += " " + str((pv or {}).get("definition", "")).lower()
    if not sample.strip():
        return "fr"
    fr = sum(sample.count(m) for m in (" le ", " la ", " les ", " des ", " une ", " du ", " et ",
                                       " pour ", " être", " taux ", "é", "è", "à", "ç"))
    en = sum(sample.count(m) for m in (" the ", " of ", " and ", " for ", " rate ", " number ",
                                       " is ", " to ", " in ", " with "))
    return "en" if en > fr else "fr"


def _llm_translate_strings(texts: list, target_lang: str) -> list:
    """Traduit une liste de chaînes vers 'en'/'fr' (ordre + longueur conservés).
    Renvoie les chaînes d'ORIGINE en cas d'échec - jamais de corruption."""
    import json as _json
    from llm_usage import MeteredOpenAI as _OAI
    if not texts:
        return texts
    lang_name = "English" if target_lang == "en" else "French"
    sys = (f"You are a professional medical translator. Translate each string in the input "
           f"array to {lang_name}. Preserve medical/technical terms, units (e.g. /100k, mg, %), "
           "acronyms (ARI, RSV, SARIMAX, ED), and all numbers. Do NOT add explanations or notes. "
           'Return ONLY a JSON object {"items": [...]} whose "items" is a JSON array of the '
           "translated strings, in the SAME order and with EXACTLY the same length as the input.")
    usr = _json.dumps({"items": list(texts)}, ensure_ascii=False)
    try:
        client = _OAI(timeout=90.0)
        resp = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": usr}],
            temperature=0, seed=42, max_tokens=4000,
            response_format={"type": "json_object"},
        )
        out = _json.loads(resp.choices[0].message.content)
        items = out.get("items") if isinstance(out, dict) else None
        if isinstance(items, list) and len(items) == len(texts) and all(isinstance(x, str) for x in items):
            return items
        logger.warning(f"translate_strings: shape mismatch ({type(items).__name__}, "
                       f"{len(items) if isinstance(items, list) else 'n/a'} vs {len(texts)})")
    except Exception as e:
        logger.warning(f"translate_strings failed: {e}")
    return texts


def _translate_variables_payload(variables: dict, target_lang: str) -> dict:
    """Copie TRADUITE de variables_json : seuls les champs d'AFFICHAGE sont traduits
    (noms, définitions, libellés, justifications, listes de bases/notes). machine_name,
    dtype, task_type, family, metric, cv, provenance, ranges numériques et structure du
    model_spec restent identiques. Les noms/justification traduits sont propagés dans le
    model_spec miroir (lu par /model/spec)."""
    import copy
    v = copy.deepcopy(variables or {})
    texts: list = []
    setters: list = []

    def _add(container, key):
        val = container.get(key)
        if isinstance(val, str) and val.strip():
            texts.append(val)
            setters.append((len(texts) - 1, lambda new, c=container, k=key: c.__setitem__(k, new)))

    def _add_list(lst):
        if isinstance(lst, list):
            for i, item in enumerate(lst):
                if isinstance(item, str) and item.strip():
                    texts.append(item)
                    setters.append((len(texts) - 1, lambda new, l=lst, idx=i: l.__setitem__(idx, new)))

    po = v.get("primary_outcome") or {}
    for k in ("name", "definition", "measurement", "timeframe"):
        _add(po, k)
    for so in (v.get("secondary_outcomes") or []):
        if isinstance(so, dict):
            for k in ("name", "definition"):
                _add(so, k)
    for pv in (v.get("predictor_variables") or []):
        if isinstance(pv, dict):
            for k in ("name", "definition", "data_source"):
                _add(pv, k)
    ra = v.get("recommended_algorithm") or {}
    for k in ("primary", "rationale", "validation_method"):
        _add(ra, k)
    _add_list(ra.get("alternatives"))
    at = v.get("alert_thresholds") or {}
    for lvl in ("green", "orange", "red"):
        band = at.get(lvl)
        if isinstance(band, dict):
            for k in ("label", "rationale", "description"):
                _add(band, k)
    _add_list(v.get("required_databases"))
    for k in ("sample_size_recommendation", "update_frequency", "implementation_notes"):
        _add(v, k)

    if not texts:
        return v
    translated = _llm_translate_strings(texts, target_lang)
    if len(translated) != len(texts):
        return v   # sécurité : ne jamais corrompre
    for idx, setter in setters:
        setter(translated[idx])

    # Propager les noms/justification traduits dans le model_spec miroir.
    ms = v.get("model_spec")
    if isinstance(ms, dict):
        if isinstance(ms.get("outcome"), dict) and po.get("name"):
            ms["outcome"]["name"] = po["name"]
        name_by_mn = {pv.get("machine_name"): pv.get("name")
                      for pv in (v.get("predictor_variables") or []) if isinstance(pv, dict)}
        for f in (ms.get("features") or []):
            if isinstance(f, dict) and name_by_mn.get(f.get("machine_name")):
                f["name"] = name_by_mn[f["machine_name"]]
        if isinstance(ms.get("algorithm"), dict) and ra.get("rationale"):
            ms["algorithm"]["rationale"] = ra["rationale"]
    return v


def _get_localized_variables(scenario_id: str, lang, base_variables: dict | None = None,
                             variables_lang=None, variables_i18n=None) -> dict | None:
    """variables_json rendu dans la langue demandée (traduction non destructive mise en
    cache par langue dans variables_i18n). Charge depuis la DB si base_variables absent."""
    import json as _json
    req = _norm_lang(lang)
    if base_variables is None:
        with engine.connect() as conn:
            r = conn.execute(text(
                "SELECT variables_json, variables_lang, variables_i18n "
                "FROM scenario_settings WHERE scenario_id = :sid"
            ), {"sid": scenario_id}).mappings().first()
        if not (r and r["variables_json"]):
            return None
        base_variables = dict(r["variables_json"])
        variables_lang, variables_i18n = r.get("variables_lang"), r.get("variables_i18n")
    if not req:
        return base_variables
    base = _norm_lang(variables_lang) or _detect_variables_lang(base_variables)
    if req == base:
        return base_variables
    cache = variables_i18n if isinstance(variables_i18n, dict) else {}
    if isinstance(cache.get(req), dict):
        return cache[req]
    translated = _translate_variables_payload(base_variables, req)
    try:  # cache best-effort ; l'affichage ne doit pas échouer si l'écriture échoue
        with engine.begin() as conn:
            conn.execute(text(
                "UPDATE scenario_settings SET variables_i18n = "
                "jsonb_set(COALESCE(variables_i18n, '{}'::jsonb), :path, CAST(:val AS jsonb), true) "
                "WHERE scenario_id = :sid"
            ), {"sid": scenario_id, "path": "{" + req + "}", "val": _json.dumps(translated)})
    except Exception as e:
        logger.warning(f"cache variables_i18n {scenario_id}/{req}: {e}")
    return translated


@app.post("/scenarios/{scenario_id}/variables/generate")
def generate_scenario_variables(scenario_id: str, lang: str | None = Query(None), _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Déclenche la génération asynchrone des Variables & Modèle depuis les articles pertinents."""
    import threading, time

    if _job_is_active(_VARIABLES_GENERATION_JOBS.get(scenario_id)):
        return {"status": "already_running"}

    _VARIABLES_GENERATION_JOBS[scenario_id] = {"status": "running", "started_at": time.time()}

    def _run():
        try:
            result = _generate_variables_from_pico(scenario_id, lang=lang)
            if "error" in result:
                _VARIABLES_GENERATION_JOBS[scenario_id] = {"status": "error", "error": result["error"]}
            else:
                _VARIABLES_GENERATION_JOBS[scenario_id] = {
                    "status": "done",
                    "generated_at": result.get("_meta", {}).get("generated_at"),
                    "variables_count": len(result.get("predictor_variables", [])),
                }
        except Exception as e:
            logger.error(f"Variables job {scenario_id}: {e}", exc_info=True)
            _VARIABLES_GENERATION_JOBS[scenario_id] = {"status": "error", "error": str(e)}

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "started", "scenario_id": scenario_id}


@app.get("/scenarios/{scenario_id}/variables/generate/status")
def get_variables_generation_status(scenario_id: str) -> dict[str, Any]:
    """Statut du job de génération des variables."""
    return _VARIABLES_GENERATION_JOBS.get(scenario_id, {"status": "idle"})


@app.get("/scenarios/{scenario_id}/epidemic-parameters/candidates")
def get_epidemic_parameter_candidates(scenario_id: str, limit: int = 0) -> dict[str, Any]:
    """Les articles du corpus pertinent qui RAPPORTENT un paramètre épidémiologique, avec
    le ou les paramètres que chacun mesure. Lecture seule, sans LLM : dit ce que la
    littérature du scénario contient avant toute extraction.

    `n_candidates` compte TOUT le corpus, comme le fait l'extraction : cet endroit
    répondait « 40 » pour n'importe quel corpus au-delà de 40, donc un chiffre différent
    de celui annoncé par l'extraction pour le même scénario. `limit > 0` ne tronque que la
    LISTE renvoyée (`articles`), jamais les compteurs."""
    arts = _parameter_candidate_articles(scenario_id, None, 0)      # jamais d'échantillon
    by_param: dict[str, int] = {}
    for a in arts:
        for p in a["params_mentioned"]:
            by_param[p] = by_param.get(p, 0) + 1
    _shown = arts if int(limit or 0) <= 0 else arts[:int(limit)]
    return {
        "scenario_id": scenario_id,
        "n_candidates": len(arts),                 # tout le corpus
        "by_parameter": by_param,                  # compté sur tout le corpus
        "articles_listed": len(_shown),
        "articles": [{"id": a["id"], "title": (a.get("title") or "")[:200], "year": a.get("year"),
                      "doi": a.get("doi"), "study_design": a.get("study_design"),
                      "quality_score": a.get("quality_score"),
                      "parameters": a["params_mentioned"]} for a in _shown],
    }


@app.post("/scenarios/{scenario_id}/epidemic-parameters/extract")
def extract_scenario_epidemic_parameters(scenario_id: str, max_articles: int = 0,
                                         _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Relance la seule extraction CIBLÉE des paramètres épidémiologiques et la fusionne
    dans le spec déjà stocké (le reste du spec est inchangé et n'est PAS régénéré).

    Utile quand le corpus a grandi, ou pour un scénario dont le spec a été construit
    avant cette extraction. Le pooling pondéré par la qualité est refait ensuite."""
    import json as _json

    with engine.connect() as conn:
        row = conn.execute(text(
            "SELECT variables_json FROM scenario_settings WHERE scenario_id = :sid"
        ), {"sid": scenario_id}).mappings().first()
    if not (row and row["variables_json"]):
        return {"status": "empty", "message": "Aucun spec stocké : générez d'abord les Variables & Modèle."}

    variables = dict(row["variables_json"])
    name = _get_scenario_name(scenario_id)
    targeted = extract_epidemic_observations(scenario_id, disease_hint=name, max_articles=max_articles)
    if not targeted.get("params"):
        return {"status": "no_parameters", "scenario_id": scenario_id,
                "n_candidates": targeted.get("n_candidates", 0),
                "message": ("Aucune valeur exploitable dans les articles qui mentionnent un paramètre."
                            if targeted.get("n_candidates")
                            else "Aucun article du corpus ne rapporte de paramètre épidémiologique.")}

    variables["epidemic_parameters"] = merge_epidemic_observations(
        variables.get("epidemic_parameters"), targeted)
    # Pool de provenance : les articles du contexte UNION ceux qui ont fourni une mesure.
    thr = _get_scenario_threshold(scenario_id)
    pool = _get_above_threshold_articles(scenario_id, thr, full_rows=25)[:25]
    seen = {a.get("id") for a in pool}
    for a in (targeted.get("articles") or []):
        if a.get("id") not in seen:
            pool.append(a)
            seen.add(a.get("id"))
    variables = _attach_model_spec(variables, pool)
    meta = dict(variables.get("_meta") or {})
    meta["epidemic_parameter_candidates"] = int(targeted.get("n_candidates", 0))
    meta["epidemic_parameter_articles_with_values"] = int(targeted.get("n_with_values", 0))
    variables["_meta"] = meta

    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE scenario_settings
            SET variables_json = CAST(:v AS jsonb), variables_i18n = NULL,
                seir_projection_json = NULL, seir_projection_generated_at = NULL, updated_at = NOW()
            WHERE scenario_id = :sid
        """), {"v": _json.dumps(variables, default=str), "sid": scenario_id})

    _epi = (variables.get("model_spec") or {}).get("epidemic_parameters") or {}
    return {
        "status": "ok",
        "scenario_id": scenario_id,
        "n_candidates": targeted.get("n_candidates", 0),
        "n_articles_with_values": targeted.get("n_with_values", 0),
        "disease": _epi.get("disease") or targeted.get("disease"),
        "applicable": bool(_epi.get("applicable")),
        "parameters": {k: {"value": v.get("value"), "ci_low": v.get("ci_low"), "ci_high": v.get("ci_high"),
                           "n_studies": v.get("n_studies"), "unit": v.get("unit")}
                       for k, v in (_epi.get("params") or {}).items()},
    }


@app.get("/scenarios/{scenario_id}/variables")
def get_scenario_variables(scenario_id: str, lang: str | None = Query(None)) -> dict[str, Any]:
    """
    Retourne les variables & modèle générés.
    Si absent, déclenche la génération.
    """
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT variables_json, variables_validated, variables_generated_at,
                   variables_lang, variables_i18n
            FROM scenario_settings WHERE scenario_id = :sid
        """), {"sid": scenario_id}).mappings().first()

    if row and row["variables_json"]:
        # Affichage localisé (traduction non destructive mise en cache par langue) :
        # le contenu fonctionnel (machine_name, model_spec) reste celui de variables_json.
        result = dict(_get_localized_variables(
            scenario_id, lang, base_variables=dict(row["variables_json"]),
            variables_lang=row.get("variables_lang"), variables_i18n=row.get("variables_i18n")))
        result["_validated"] = row["variables_validated"]
        result["_generated_at"] = row["variables_generated_at"].isoformat() if row["variables_generated_at"] else None
        return result

    # Vérifier qu'il y a des articles avant de déclencher la génération
    threshold = _get_scenario_threshold(scenario_id)
    articles = _get_above_threshold_articles(scenario_id, threshold, full_rows=0)   # existence seulement
    if not articles:
        return {"status": "empty", "message": _msg(lang,
                "Aucun article au-dessus du seuil. Ajoutez des articles ou abaissez le seuil de similarité.",
                "No article above the threshold. Add articles or lower the similarity threshold.")}

    # Job en échec : renvoyer l'erreur au lieu de relancer à chaque poll (évite
    # une boucle de regénération quand la génération échoue de façon persistante).
    _vj = _VARIABLES_GENERATION_JOBS.get(scenario_id, {})
    if _vj.get("status") == "error":
        return {"status": "error", "message": _vj.get("error", "Échec de la génération des variables.")}

    # Déclencher la génération. On passe lang EXPLICITEMENT : appeler l'endpoint
    # directement sans cet argument passerait l'objet Query(None) par défaut (et non
    # None) jusqu'à _llm_lang_directive → crash « 'Query' object has no attribute 'strip' ».
    generate_scenario_variables(scenario_id, lang=lang)
    return {"status": "generating", "message": _msg(lang, "Génération en cours, réessayez dans 30 secondes.",
                                                    "Generating, try again in 30 seconds.")}


@app.post("/scenarios/{scenario_id}/variables/validate")
def validate_scenario_variables(scenario_id: str, payload: dict[str, Any], _: None = Depends(require_api_key)) -> dict[str, Any]:
    """
    Valide (ou modifie) les variables & modèle générés par LLM.
    payload peut contenir les variables modifiées.
    """
    import json as _json
    from datetime import datetime, timezone

    variables_json = payload.get("variables_json")
    with engine.begin() as conn:
        if variables_json:
            # La projection SEIR en cache est calculée à partir de ce spec : un spec
            # remplacé par l'utilisateur la rend caduque. Sans cette invalidation, l'onglet
            # Modèle resservait, en la marquant « from_cache », une courbe issue des
            # paramètres que l'utilisateur venait justement de corriger.
            conn.execute(text("""
                UPDATE scenario_settings
                SET variables_json = CAST(:vars AS jsonb),
                    variables_validated = TRUE,
                    variables_lang = NULL,
                    variables_i18n = NULL,
                    seir_projection_json = NULL,
                    seir_projection_generated_at = NULL,
                    updated_at = NOW()
                WHERE scenario_id = :sid
            """), {"sid": scenario_id, "vars": _json.dumps(variables_json)})
        else:
            conn.execute(text("""
                UPDATE scenario_settings
                SET variables_validated = TRUE, updated_at = NOW()
                WHERE scenario_id = :sid
            """), {"sid": scenario_id})

    return {"status": "validated", "scenario_id": scenario_id, "validated_at": datetime.now(timezone.utc).isoformat()}


def _norm_col(s: Any) -> str:
    """Normalise un nom de colonne pour l'appariement fichier↔data_template. MÊMES
    règles que model_trainer._norm (repli NFKD des accents, minuscules, non-alnum →
    « _ ») pour que la VALIDATION et l'ENTRAÎNEMENT s'accordent : un en-tête réel
    « Température max (J1) » s'apparie au machine_name « temperature_max_j1 »."""
    import unicodedata as _ud
    _s = _ud.normalize("NFKD", str(s if s is not None else "")).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", _s.lower()).strip("_")
