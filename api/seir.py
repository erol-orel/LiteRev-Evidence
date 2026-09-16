"""SEIR projections, observed data and calibration.

Extracted from main.py (LiteRev API); `main` re-exports everything for the scripts,
tools and tests.
"""
from __future__ import annotations

import json
from typing import Any

from fastapi import Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import text

from .core import app, engine, logger, require_api_key
from .scenario_store import _get_scenario_threshold
from .model_data import _get_model_spec

def _scenario_seed(scenario_id: str) -> tuple[float, float, str | None]:
    """Dérive le contexte de projection d'un scénario - (population d'exposition, cas
    initiaux, libellé géographique) - à partir de la géographie MODALE de son corpus
    PERTINENT (champ `geographic_scope` des documents rattachés, filtrés par la même
    porte de screening/seuil que le modèle), au lieu d'un 1e6 fixe. On prend la première
    géographie (par fréquence décroissante) reconnue par `population_for_geography` ; les
    intitulés non mappables (« multi-country »…) sont ignorés. Cas initiaux ≈ 1 pour
    100 000 (min 1) - un simple amorçage, la dynamique SEIR y est peu sensible. Défaut
    (1e6, 10, None) si aucune géographie connue. Lecture seule ; robuste aux erreurs."""
    import seir_model
    pop: float = 1_000_000.0
    geo: str | None = None

    # 1) Géographie INTENTIONNELLE du scénario = sa requête / son titre. Prioritaire : le
    #    corpus d'un sujet épidémique est INTERNATIONAL (géographie modale ≈ « world »),
    #    mais l'utilisateur cible une zone précise (« … en suisse romande »). On prend la
    #    zone la plus LOCALE citée dans la requête.
    try:
        with engine.connect() as conn:
            _q = conn.execute(text(
                "SELECT COALESCE(NULLIF(query, ''), name, '') FROM user_scenarios WHERE id = :sid"
            ), {"sid": scenario_id}).scalar()
        _hit = seir_model.population_for_geography_in_text(_q or "")
        if _hit:
            pop, geo = _hit[0], _hit[1]
    except Exception as _e:
        logger.warning(f"_scenario_seed geo-from-query {scenario_id}: {_e}")

    # 2) Repli : géographie MODALE du corpus pertinent (si la requête ne cite aucune zone).
    if geo is None:
        try:
            threshold = _get_scenario_threshold(scenario_id)
            with engine.connect() as conn:
                rows = conn.execute(text("""
                    SELECT ld.geographic_scope AS geo, COUNT(*) AS n
                    FROM literature_document ld
                    JOIN article_scenarios asn ON asn.document_id = ld.id AND asn.scenario_id = :sid
                    WHERE ld.project_context = 'literev'
                      AND ld.is_duplicate IS NOT TRUE
                      AND ld.geographic_scope IS NOT NULL
                      AND COALESCE(asn.screening_status, ld.screening_status) IS DISTINCT FROM 'excluded'
                      AND (
                          COALESCE(asn.screening_status, ld.screening_status) = 'included'
                          OR COALESCE(asn.similarity_score, 0) >= :threshold
                      )
                    GROUP BY ld.geographic_scope
                    ORDER BY n DESC, ld.geographic_scope ASC
                """), {"sid": scenario_id, "threshold": threshold}).mappings().all()
            for r in rows:
                _p = seir_model.population_for_geography(r["geo"])
                if _p:
                    pop, geo = _p, str(r["geo"])
                    break
        except Exception as _e:
            logger.warning(f"_scenario_seed({scenario_id}): {_e}")

    i0 = max(1.0, pop * 1e-5)
    return pop, i0, geo


def _get_seir_observed(scenario_id: str) -> dict | None:
    """Série observée (réelle) attachée au SEIR d'un scénario, ou None."""
    with engine.connect() as conn:
        row = conn.execute(text(
            "SELECT seir_observed_json FROM scenario_settings WHERE scenario_id = :sid"
        ), {"sid": scenario_id}).mappings().first()
    if row and row["seir_observed_json"]:
        return dict(row["seir_observed_json"])
    return None


def _store_seir_observed(scenario_id: str, obj: dict | None) -> None:
    """Écrit (ou efface si obj=None) la série observée attachée au SEIR du scénario."""
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO scenario_settings (scenario_id, seir_observed_json, updated_at)
            VALUES (:sid, CAST(:obj AS jsonb), NOW())
            ON CONFLICT (scenario_id) DO UPDATE
                SET seir_observed_json = CAST(:obj AS jsonb), updated_at = NOW()
        """), {"sid": scenario_id, "obj": json.dumps(obj) if obj is not None else None})


def _seir_observed_overlay(scenario_id: str, eff_params: dict, pop: float, i0: float,
                           days: int) -> dict | None:
    """Prépare la superposition de la série observée sur la projection courante :
    ré-exprimée en UNITÉS MODÈLE (échelle par moindres carrés) + R² d'ajustement au
    modèle courant. Best-effort → None si rien d'attaché ou en cas d'erreur."""
    import seir_model
    obs = _get_seir_observed(scenario_id)
    if not obs or not obs.get("points"):
        return None
    base = {k: seir_model._num_or_none(v.get("value"))
            for k, v in (eff_params or {}).items() if isinstance(v, dict)}
    base = {k: val for k, val in base.items() if val is not None}
    base["population"], base["initial_infected"] = pop, i0
    al = seir_model.align_observed(obs["points"])
    col = obs.get("column") or "incidence"
    sc = seir_model.scale_observed_to_model(al["points"], base, col, days=days)
    return {
        "column": col, "label": obs.get("label"), "source": obs.get("source"),
        "n": al["n"], "start_date": al["start_date"],
        "scale": sc.get("scale"), "fit_r2": sc.get("r2"), "shift_days": sc.get("shift_days"),
        "points": sc.get("points", []),      # [{day, value}] en unités modèle, au jour aligné
    }


def _seir_projection_payload(
    scenario_id: str, days: int = 365, start_date: str | None = None,
    population: float | None = None, initial_infected: float | None = None,
    n_samples: int = 300, overrides: dict | None = None,
) -> dict[str, Any]:
    """Cœur PARTAGÉ de la projection SEIR (GET par défaut + POST avec overrides). Les
    `overrides` = {nom_param: {value, ci_low?, ci_high?}} remplacent/ajoutent la valeur
    (± IC) d'un paramètre saisie par l'utilisateur - pour explorer des variantes du
    modèle. Les paramètres SOURCE (littérature, avec provenance) restent renvoyés à part
    (`parameters`) ; les paramètres EFFECTIVEMENT simulés (source ⊕ overrides) le sont
    aussi (`effective_parameters`). Déterministe (seed fixe)."""
    import seir_model
    from datetime import date, timedelta
    spec = _get_model_spec(scenario_id) or {}
    epi = spec.get("epidemic_parameters") or {}
    src_params = dict(epi.get("params") or {})

    # Overrides utilisateur : coercés numériquement, marqués `overridden` pour l'UI.
    eff_params = {k: dict(v) for k, v in src_params.items() if isinstance(v, dict)}
    applied: dict[str, float] = {}
    for name, ov in (overrides or {}).items():
        if not isinstance(ov, dict):
            continue
        v = seir_model._num_or_none(ov.get("value"))
        if v is None:
            continue
        blk = dict(eff_params.get(name) or {})
        blk["value"] = v
        blk["ci_low"] = seir_model._num_or_none(ov.get("ci_low"))
        blk["ci_high"] = seir_model._num_or_none(ov.get("ci_high"))
        blk["overridden"] = True
        blk.setdefault("unit", (src_params.get(name) or {}).get("unit", ""))
        blk.setdefault("provenance", [])
        eff_params[name] = blk
        applied[name] = v

    dists = seir_model.params_to_distributions(eff_params)

    def _scan_counts() -> dict[str, int]:
        """Ce que la recherche de paramètres a trouvé dans le corpus (stocké par la
        génération). « Aucun paramètre extrait » laissait croire à une panne ; dire
        « 0 valeur sur 37 articles qui en parlent » est une information."""
        try:
            with engine.connect() as _c:
                _r = _c.execute(text(
                    "SELECT variables_json FROM scenario_settings WHERE scenario_id = :sid"
                ), {"sid": scenario_id}).mappings().first()
            _m = (dict(_r["variables_json"]).get("_meta") or {}) if _r and _r["variables_json"] else {}
            return {"articles_reporting_parameters": int(_m.get("epidemic_parameter_candidates") or 0),
                    "articles_with_values": int(_m.get("epidemic_parameter_articles_with_values") or 0)}
        except Exception:                                    # noqa: BLE001 - jamais bloquant
            return {"articles_reporting_parameters": 0, "articles_with_values": 0}
    # ── Trois portes AVANT de simuler ────────────────────────────────────────────
    # Un override explicite de l'utilisateur vaut décision consciente : il ouvre les
    # portes 1 et 2 (exploration « et si ? »), et la réponse est marquée `forced`.
    forced = bool(applied)
    # `reason_code` : identifiant STABLE de la porte fermée, que l'interface traduit
    # dans la langue choisie (le texte `reason` reste en français pour l'API / les logs).
    if not dists:
        _sc = _scan_counts()
        return {
            "applicable": False,
            "scenario_id": scenario_id,
            "reason_code": "no_parameters",
            "reason": ("Aucun paramètre épidémiologique extrait de la littérature "
                       "(scénario non transmissible, ou paramètres non rapportés). "
                       f"{_sc['articles_reporting_parameters']} article(s) du corpus mentionnent "
                       f"un paramètre, {_sc['articles_with_values']} en donnent une valeur."),
            **_sc,
        }
    # Porte 1 - le LLM a EXPLICITEMENT jugé le scénario non transmissible. Sans ce
    # test, un seul paramètre numérique rescapé (une létalité, p. ex.) suffisait à
    # servir une courbe épidémique complète pour un scénario d'oncologie.
    if not epi.get("applicable") and not forced:
        return {
            "applicable": False,
            "scenario_id": scenario_id,
            "reason_code": "not_transmissible",
            "reason": "Scénario marqué NON transmissible à l'extraction : pas de "
                      "modèle compartimental applicable.",
        }
    # Porte 2 - un paramètre extrait ne suffit pas, il faut un paramètre qui PILOTE
    # la dynamique. Sans r0 ni beta, `_rates` retombait sur un R0 = 2.5 codé en dur et
    # l'UI présentait la courbe qui en découle comme « issue de la littérature ».
    if not ({"r0", "beta"} & set(dists)) and not forced:
        _have = ", ".join(sorted(dists)) or "aucun"
        return {
            "applicable": False,
            "scenario_id": scenario_id,
            "reason_code": "no_transmission_parameter",
            "reason": ("Paramètre de transmission manquant : ni R₀ ni β n'a été extrait "
                       f"(disponibles : {_have}). Une projection reposerait sur une "
                       "valeur par défaut, pas sur la littérature - saisissez R₀ "
                       "manuellement pour explorer un scénario."),
            "missing": ["r0"],
            "available_parameters": sorted(dists),
            **_scan_counts(),
        }
    _pop_default, _i0_default, _geo_label = _scenario_seed(scenario_id)
    try:
        dists["population"] = float(population) if population is not None else _pop_default
    except (TypeError, ValueError):
        dists["population"] = _pop_default
    try:
        dists["initial_infected"] = float(initial_infected) if initial_infected is not None else _i0_default
    except (TypeError, ValueError):
        dists["initial_infected"] = _i0_default
    days = max(1, min(int(days or 365), 3650))
    n_samples = max(1, min(int(n_samples or 300), 1000))
    ens = seir_model.simulate_ensemble(dists, days=days, n_samples=n_samples)

    d0 = None
    if start_date:
        try:
            _y, _m, _d = (int(x) for x in str(start_date)[:10].split("-"))
            d0 = date(_y, _m, _d)
        except Exception:
            d0 = None
    dates = [((d0 + timedelta(days=int(_day))).isoformat() if d0 else int(_day))
             for _day in ens["days"]]

    try:
        observed = _seir_observed_overlay(scenario_id, eff_params, dists["population"],
                                          dists["initial_infected"], days)
    except Exception:
        observed = None

    return {
        "applicable": True,
        "scenario_id": scenario_id,
        "model": ens["model"],
        "disease": epi.get("disease"),
        # `forced` : projection obtenue grâce à des paramètres SAISIS, pas extraits -
        # l'UI doit le dire plutôt que de laisser croire à un résultat sourcé.
        "forced": forced,
        # "literature" (extrait) | "user" (saisi dans l'UI) | "assumed" (repli du moteur).
        # Un R0 tapé à la main ne doit pas être annoncé comme sourcé.
        "r0_source": ("user" if ({"r0", "beta"} & set(applied))
                      else ens["summary"].get("r0_source", "literature")),
        "n_samples": ens["n_samples"],
        "n_dropped": ens.get("n_dropped", 0),
        "population": dists["population"],
        "initial_infected": dists["initial_infected"],
        "geography": _geo_label,
        "dates": dates,
        "series": {k: ens[k] for k in ("incidence", "prevalence", "cumulative", "deaths", "r_eff",
                                        "vaccinated", "quarantined") if k in ens},
        "summary": ens["summary"],
        "parameters": src_params,             # littérature (avec provenance) → traçabilité
        "effective_parameters": eff_params,   # réellement simulés (source ⊕ overrides)
        "overrides_applied": applied,
        "observed": observed,                 # série réelle superposée (ou None)
    }


# ── Cache de la projection PAR DÉFAUT (365 j, 300 tirages, géographie du scénario) ──
# La projection est une simulation numérique (pas de LLM) mais ses 300 tirages prenaient
# quelques secondes à chaque ouverture de l'onglet Modèle. Le pipeline la calcule après
# les variables et la range dans scenario_settings ; l'onglet la lit telle quelle. Elle
# est périmée dès que le spec (variables_json) est plus récent qu'elle. La série observée
# (`observed`) n'est pas mise en cache : elle est relue à chaque fois (un jeu de données
# peut avoir été attaché entre-temps).

def _json_default(o):
    return o.item() if hasattr(o, "item") else str(o)


def _seir_cache_read(scenario_id: str) -> dict | None:
    with engine.connect() as conn:
        row = conn.execute(text(
            "SELECT seir_projection_json AS j, seir_projection_generated_at AS at, "
            "variables_generated_at AS vat FROM scenario_settings WHERE scenario_id = :sid"
        ), {"sid": scenario_id}).mappings().first()
    if not row or not row["j"]:
        return None
    if row["vat"] is not None and row["at"] is not None and row["at"] < row["vat"]:
        return None                                  # spec régénéré depuis : à recalculer
    return dict(row["j"])


def _seir_cache_write(scenario_id: str, payload: dict) -> None:
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO scenario_settings (scenario_id, seir_projection_json, seir_projection_generated_at, updated_at)
                VALUES (:sid, CAST(:p AS jsonb), NOW(), NOW())
                ON CONFLICT (scenario_id) DO UPDATE
                SET seir_projection_json = CAST(:p AS jsonb), seir_projection_generated_at = NOW(), updated_at = NOW()
            """), {"sid": scenario_id, "p": json.dumps(payload, default=_json_default)})
    except Exception as _e:
        logger.warning(f"seir cache write {scenario_id}: {_e}")


def _default_seir_projection(scenario_id: str, refresh: bool = False) -> dict[str, Any]:
    """Projection par défaut servie depuis le cache quand il est à jour, sinon calculée et
    mise en cache. `from_cache` dit d'où elle vient."""
    if not refresh:
        try:
            cached = _seir_cache_read(scenario_id)
        except Exception as _e:
            logger.warning(f"seir cache read {scenario_id}: {_e}")
            cached = None
        if cached is not None:
            cached["from_cache"] = True
            if cached.get("applicable"):
                try:
                    cached["observed"] = _seir_observed_overlay(
                        scenario_id, cached.get("effective_parameters") or {},
                        float(cached.get("population") or 1e6), float(cached.get("initial_infected") or 10), 365)
                except Exception:
                    cached["observed"] = None
            return cached
    payload = _seir_projection_payload(scenario_id, 365, None, None, None, 300, overrides=None)
    _seir_cache_write(scenario_id, {k: v for k, v in payload.items() if k != "observed"})
    payload["from_cache"] = False
    return payload


def _precompute_seir_projection(scenario_id: str) -> dict[str, Any]:
    """Appelé par le pipeline après la génération des variables : (re)calcule et range la
    projection par défaut."""
    return _default_seir_projection(scenario_id, refresh=True)


@app.get("/scenarios/{scenario_id}/seir/projection")
def get_seir_projection(
    scenario_id: str,
    days: int = 365,
    start_date: str | None = None,
    population: float | None = None,
    initial_infected: float | None = None,
    n_samples: int = 300,
    refresh: bool = False,
) -> dict[str, Any]:
    """Projection compartimentale (famille SEIR) d'un scénario, paramétrée par la
    littérature EXTRAITE (model_spec.epidemic_parameters). Séries incidence / prévalence
    / cumul / décès / R_eff AVEC bandes d'incertitude + résumé + paramètres source (avec
    provenance). Population + cas initiaux DÉRIVÉS de la géographie du scénario sauf
    override en query. `applicable=false` si non transmissible. Lecture seule.
    Aux paramètres par défaut, la projection vient du cache rempli par le pipeline
    (`from_cache`) ; `refresh=true` force le recalcul."""
    if (days == 365 and start_date is None and population is None
            and initial_infected is None and n_samples == 300):
        return _default_seir_projection(scenario_id, refresh=refresh)
    return _seir_projection_payload(scenario_id, days, start_date, population,
                                    initial_infected, n_samples, overrides=None)


class SeirProjectionIn(BaseModel):
    days: int = 365
    start_date: str | None = None
    population: float | None = None
    initial_infected: float | None = None
    n_samples: int = 300
    # {nom_param: {value, ci_low?, ci_high?}} - valeurs modifiées/ajoutées par l'utilisateur.
    overrides: dict[str, dict] | None = None


@app.post("/scenarios/{scenario_id}/seir/projection")
def post_seir_projection(scenario_id: str, payload: SeirProjectionIn) -> dict[str, Any]:
    """Même projection, mais AVEC des paramètres modifiés par l'utilisateur (onglet SEIR) :
    `overrides` remplace/ajoute value (± IC) par paramètre pour explorer des variantes du
    modèle, sans altérer les paramètres source extraits de la littérature."""
    return _seir_projection_payload(
        scenario_id, payload.days, payload.start_date, payload.population,
        payload.initial_infected, payload.n_samples, overrides=payload.overrides)


class SeirObservedIn(BaseModel):
    points: list[dict] | None = None          # [{date|day, value}] (upload direct)
    connector_id: str | None = None           # ou tirage d'un connecteur (ex. foph-sentinella-ili)
    connector_variable: str | None = None
    region: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    column: str = "incidence"                 # série modèle à comparer (incidence/prevalence/…)
    label: str | None = None


@app.post("/scenarios/{scenario_id}/seir/observed")
def post_seir_observed(scenario_id: str, payload: SeirObservedIn,
                       _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Attache une série OBSERVÉE (réelle) au SEIR - upload de points {date|jour, valeur}
    OU tirage d'un connecteur - pour la superposer au graphe et calibrer le modèle dessus.
    Stockée par scénario (scenario_settings.seir_observed_json). Au moins 3 points."""
    import seir_model, data_connectors
    pts: list[dict] = []
    source, label = "upload", payload.label
    if payload.connector_id and payload.connector_variable:
        prm = {k: v for k, v in {"region": payload.region, "start_date": payload.start_date,
                                 "end_date": payload.end_date}.items() if v is not None}
        try:
            rows = data_connectors.fetch_series(payload.connector_id, prm)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"connecteur: {e}")
        for r in rows:
            v = seir_model._num_or_none(r.get(payload.connector_variable))
            if v is not None and r.get("date"):
                pts.append({"date": r["date"], "value": v})
        source = payload.connector_id
        label = label or f"{payload.connector_id}:{payload.connector_variable}"
    else:
        for p in (payload.points or []):
            if not isinstance(p, dict):
                continue
            v = seir_model._num_or_none(p.get("value"))
            d = p.get("date", p.get("day"))
            if v is not None and d is not None:
                pts.append({"date": d, "value": v})
    if len(pts) < 3:
        raise HTTPException(status_code=400,
                            detail="Au moins 3 points observés (date/jour + valeur) sont requis.")
    col = payload.column if payload.column in ("incidence", "prevalence", "cumulative", "deaths") else "incidence"
    _store_seir_observed(scenario_id, {"points": pts, "column": col, "label": label, "source": source})
    al = seir_model.align_observed(pts)
    return {"ok": True, "n": len(pts), "column": col, "source": source,
            "label": label, "start_date": al["start_date"]}


@app.delete("/scenarios/{scenario_id}/seir/observed")
def delete_seir_observed(scenario_id: str,
                         _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Détache la série observée du SEIR du scénario."""
    _store_seir_observed(scenario_id, None)
    return {"ok": True}


class SeirCalibrateIn(BaseModel):
    column: str | None = None
    overrides: dict[str, dict] | None = None   # mêmes overrides que la projection (facultatif)
    days: int = 365


@app.post("/scenarios/{scenario_id}/seir/calibrate")
def post_seir_calibrate(scenario_id: str, payload: SeirCalibrateIn,
                        _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Calibre R0 (+ un facteur d'échelle d'amplitude) du SEIR sur la série observée
    attachée, par moindres carrés. Renvoie le R0 AJUSTÉ vs le R0 LITTÉRATURE + R²/RMSE.
    N'altère PAS le spec : l'UI applique le R0 ajusté comme override si l'utilisateur le
    souhaite (traçabilité littérature ↔ ajusté préservée)."""
    import seir_model
    obs = _get_seir_observed(scenario_id)
    if not obs or not obs.get("points"):
        raise HTTPException(status_code=400, detail="Aucune série observée attachée à ce scénario.")
    spec = _get_model_spec(scenario_id) or {}
    epi = spec.get("epidemic_parameters") or {}
    eff = {k: dict(v) for k, v in (epi.get("params") or {}).items() if isinstance(v, dict)}
    for name, ov in (payload.overrides or {}).items():
        if isinstance(ov, dict):
            v = seir_model._num_or_none(ov.get("value"))
            if v is not None:
                eff.setdefault(name, {})["value"] = v
    base = {k: seir_model._num_or_none(v.get("value")) for k, v in eff.items() if isinstance(v, dict)}
    base = {k: val for k, val in base.items() if val is not None}
    pop, i0, _ = _scenario_seed(scenario_id)
    base.setdefault("population", pop)
    base.setdefault("initial_infected", i0)
    col = payload.column or obs.get("column") or "incidence"
    al = seir_model.align_observed(obs["points"])
    fit = seir_model.calibrate_to_observed(al["points"], base, column=col, days=payload.days)
    if not fit.get("ok"):
        raise HTTPException(status_code=400, detail=fit.get("reason", "calibration impossible"))
    fit["literature_r0"] = base.get("r0")
    return fit
