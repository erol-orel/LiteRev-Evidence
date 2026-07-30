"""Modèles compartimentaux de la famille SEIR, paramétrés par la littérature.

Fonctions PURES — pas de base de données, pas de réseau, **bibliothèque standard
seulement** (`math`, `random`). Les numériques sont donc testables en CI sans
scipy/numpy. Intégrateur RK4 déterministe + un ensemble de propagation
d'incertitude qui échantillonne les distributions de paramètres issues des articles.

Sélection AUTOMATIQUE du modèle : la structure la plus riche que les paramètres
disponibles permettent — un seul membre de droite généralisé couvre toutes les
combinaisons (SIR, SEIR, SEIRD, SEIRS, SEIRDS…) :

  * compartiment latent E  ⇐ une période d'incubation est connue   → SEIR sinon SIR
  * compartiment décès D   ⇐ une létalité (CFR) est connue          → …D
  * immunité décroissante  ⇐ une durée d'immunité est connue (R→S)  → …S

`model_name()` déduit le libellé des indicateurs actifs.

Sorties (orientées santé publique), en séries journalières + résumé :
  incidence   nouvelles infections / jour   (β·S·I/N)
  prevalence  infectieux à l'instant t       (I)
  cumulative  infections cumulées            (∫ incidence)
  deaths      décès cumulés                  (D)            [si CFR]
  r_eff       nombre de reproduction effectif (R0·S/N)
  + pic (jour + hauteur) d'incidence et de prévalence, taux d'attaque, R0.

Convention : l'intégration se fait en FRACTIONS de population (numériquement
stable quel que soit N), les sorties sont remises à l'échelle en effectifs.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

# 6 compartiments d'état intégrés : S, E, I, R, D, C (C = infections cumulées,
# accumulateur monotone distinct — pas soumis à la conservation S+E+I+R+D = N).
_N_STATE = 6
_Z = 1.959963984540054  # quantile normal à 97.5 % (demi-largeur d'un IC 95 %)


@dataclass
class SeirParams:
    """Paramètres d'UNE trajectoire. Les champs à None/0 désactivent leur structure."""
    r0: float | None = None                       # nombre de reproduction de base
    beta: float | None = None                     # taux de transmission /j (sinon dérivé de r0)
    infectious_period_days: float = 7.0           # → gamma = 1/période
    incubation_period_days: float | None = None   # → sigma ; présent ⇒ compartiment E (SEIR)
    cfr: float = 0.0                              # létalité 0..1 ; > 0 ⇒ compartiment D
    immunity_duration_days: float | None = None   # → omega ; présent ⇒ R→S (immunité décroissante)
    population: float = 1_000_000.0
    initial_infected: float = 10.0
    initial_exposed: float = 0.0


def _rates(p: SeirParams) -> dict:
    """Convertit des paramètres « lisibles » (périodes, R0, CFR) en taux d'ODE + indicateurs."""
    gamma = 1.0 / max(p.infectious_period_days or 0.0, 1e-6)
    if p.beta is not None and p.beta > 0:
        beta = p.beta
    elif p.r0 is not None and p.r0 > 0:
        beta = p.r0 * gamma
    else:
        beta = 2.5 * gamma  # repli prudent (R0 = 2.5) si ni beta ni r0 fournis
    has_e = bool(p.incubation_period_days and p.incubation_period_days > 0)
    sigma = 1.0 / p.incubation_period_days if has_e else 0.0
    has_waning = bool(p.immunity_duration_days and p.immunity_duration_days > 0)
    omega = 1.0 / p.immunity_duration_days if has_waning else 0.0
    cfr = min(max(p.cfr or 0.0, 0.0), 1.0)
    return {
        "beta": beta, "gamma": gamma, "sigma": sigma, "omega": omega, "cfr": cfr,
        "has_e": has_e, "has_death": cfr > 0.0, "has_waning": has_waning,
        "r0": beta / gamma if gamma > 0 else float("nan"),
    }


def model_name(rates: dict) -> str:
    """Libellé du modèle effectivement simulé, d'après les indicateurs actifs."""
    base = "SEIR" if rates["has_e"] else "SIR"
    suffix = ("D" if rates["has_death"] else "") + ("S" if rates["has_waning"] else "")
    return base + suffix


def _deriv(y: list[float], r: dict) -> tuple[float, ...]:
    """Membre de droite généralisé (en fractions ; N = 1). Ordre : S, E, I, R, D, C.

    Conservation : dS+dE+dI+dR+dD = 0 (C est un accumulateur à part). Le flux de
    sortie de I (=gamma·I) se répartit CFR→décès, (1−CFR)→guérison ; sans période
    d'incubation, les nouvelles infections vont directement de S à I (pas de E)."""
    s, e, i, rec, _d, _c = y
    inf = r["beta"] * s * i          # force d'infection · S = nouvelles infections/j
    waning = r["omega"] * rec        # R → S (0 sans immunité décroissante)
    exit_i = r["gamma"] * i          # sortie totale de I
    to_death = r["cfr"] * exit_i
    to_recov = exit_i - to_death
    if r["has_e"]:
        ds = -inf + waning
        de = inf - r["sigma"] * e
        di = r["sigma"] * e - exit_i
    else:
        ds = -inf + waning
        de = 0.0
        di = inf - exit_i
    drec = to_recov - waning
    dd = to_death
    dc = inf
    return (ds, de, di, drec, dd, dc)


def _rk4_step(y: list[float], r: dict, dt: float) -> list[float]:
    """Un pas de Runge-Kutta d'ordre 4 (précis et stable pour des ODE SEIR non raides)."""
    k1 = _deriv(y, r)
    k2 = _deriv([y[j] + 0.5 * dt * k1[j] for j in range(_N_STATE)], r)
    k3 = _deriv([y[j] + 0.5 * dt * k2[j] for j in range(_N_STATE)], r)
    k4 = _deriv([y[j] + dt * k3[j] for j in range(_N_STATE)], r)
    return [y[j] + (dt / 6.0) * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j])
            for j in range(_N_STATE)]


def simulate(p: SeirParams, days: int = 365, dt: float = 0.25) -> dict:
    """Intègre UNE trajectoire déterministe et renvoie séries journalières + résumé.

    Séries (effectifs) : `days`, `incidence`, `prevalence`, `cumulative`, `deaths`,
    `r_eff`. Résumé : `model`, `r0`, pics d'incidence/prévalence (jour + hauteur),
    `attack_rate` (0..1) et `total_deaths`."""
    r = _rates(p)
    n = max(float(p.population or 0.0), 1.0)
    i0 = min(max(float(p.initial_infected or 0.0), 0.0) / n, 1.0)
    e0 = min(max(float(p.initial_exposed or 0.0), 0.0) / n, 1.0 - i0)
    s0 = max(1.0 - i0 - e0, 0.0)
    y = [s0, e0, i0, 0.0, 0.0, i0 + e0]  # C initial = déjà infectés (E+I)

    steps_per_day = max(int(round(1.0 / max(dt, 1e-6))), 1)
    step = 1.0 / steps_per_day
    r0v = r["r0"]

    days_out: list[int] = []
    incidence: list[float] = []
    prevalence: list[float] = []
    cumulative: list[float] = []
    deaths: list[float] = []
    r_eff: list[float] = []
    susceptible: list[float] = []
    exposed: list[float] = []
    recovered: list[float] = []

    def _record(t: int, yv: list[float]) -> None:
        s, _e, i, _rec, d, c = yv
        days_out.append(t)
        incidence.append(r["beta"] * s * i * n)  # nouvelles infections/j (effectifs)
        prevalence.append(i * n)
        cumulative.append(c * n)
        deaths.append(d * n)
        r_eff.append(r0v * s)
        susceptible.append(s * n)   # compartiments S / E / R exposés aussi (effectifs)
        exposed.append(_e * n)
        recovered.append(_rec * n)

    _record(0, y)
    for day in range(1, int(days) + 1):
        for _ in range(steps_per_day):
            y = _rk4_step(y, r, step)
            # borne les micro-négatifs dus à l'arithmétique flottante
            if y[0] < 0 or y[1] < 0 or y[2] < 0 or y[3] < 0 or y[4] < 0 or y[5] < 0:
                y = [v if v > 0.0 else 0.0 for v in y]
        _record(day, y)

    peak_inc_i = max(range(len(incidence)), key=lambda k: incidence[k])
    peak_prev_i = max(range(len(prevalence)), key=lambda k: prevalence[k])
    summary = {
        "model": model_name(r),
        "r0": round(r0v, 3),
        "peak_incidence": round(incidence[peak_inc_i], 2),
        "peak_incidence_day": days_out[peak_inc_i],
        "peak_prevalence": round(prevalence[peak_prev_i], 2),
        "peak_prevalence_day": days_out[peak_prev_i],
        "attack_rate": round(min(cumulative[-1] / n, 1.0), 4),
        "total_deaths": round(deaths[-1], 2),
    }
    return {
        "model": summary["model"],
        "days": days_out,
        "incidence": incidence,
        "prevalence": prevalence,
        "cumulative": cumulative,
        "deaths": deaths,
        "r_eff": r_eff,
        "susceptible": susceptible,
        "exposed": exposed,
        "recovered": recovered,
        "summary": summary,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble : propagation de l'incertitude des paramètres issus de la littérature
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ParamDist:
    """Distribution d'un paramètre : moyenne + IC 95 % (facultatif) issus du corpus.

    Échantillonnée comme une normale tronquée aux valeurs positives. Sans IC (ou IC
    dégénéré), la valeur est fixe (== moyenne) — l'incertitude ne vient alors que des
    autres paramètres."""
    mean: float
    ci_low: float | None = None
    ci_high: float | None = None

    def sd(self) -> float:
        if (self.ci_low is not None and self.ci_high is not None
                and self.ci_high > self.ci_low):
            return (self.ci_high - self.ci_low) / (2.0 * _Z)
        return 0.0

    def sample(self, rng: random.Random) -> float:
        sd = self.sd()
        if sd <= 0.0:
            return float(self.mean)
        return max(rng.gauss(self.mean, sd), 1e-9)  # garde le paramètre positif


# Paramètres dont la PRÉSENCE change la structure du modèle (E / D / R→S). Ils
# fixent la structure pour tout l'ensemble ; seules les VALEURS varient d'un tirage
# à l'autre (on ne veut pas que le modèle « clignote » entre tirages).
_STRUCTURAL = ("incubation_period_days", "cfr", "immunity_duration_days")
_PARAM_FIELDS = (
    "r0", "beta", "infectious_period_days", "incubation_period_days",
    "cfr", "immunity_duration_days", "population", "initial_infected",
    "initial_exposed",
)


def _as_dist(v) -> ParamDist | None:
    if v is None:
        return None
    if isinstance(v, ParamDist):
        return v
    return ParamDist(mean=float(v))


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Percentile (interpolation linéaire) d'une liste DÉJÀ triée. q en 0..100."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (q / 100.0) * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _band(series_per_sample: list[list[float]], lo_q: float, hi_q: float) -> dict:
    """Agrège N trajectoires (une liste par tirage) en médiane + IC par pas de temps."""
    if not series_per_sample:
        return {"median": [], "lower": [], "upper": []}
    horizon = len(series_per_sample[0])
    median, lower, upper = [], [], []
    for t in range(horizon):
        col = sorted(s[t] for s in series_per_sample)
        median.append(_percentile(col, 50.0))
        lower.append(_percentile(col, lo_q))
        upper.append(_percentile(col, hi_q))
    return {"median": median, "lower": lower, "upper": upper}


def simulate_ensemble(
    params: dict,
    days: int = 365,
    n_samples: int = 300,
    seed: int = 12345,
    ci: tuple[float, float] = (2.5, 97.5),
) -> dict:
    """Échantillonne les distributions de paramètres → N trajectoires → médiane + IC.

    `params` associe un nom (cf. `_PARAM_FIELDS`) à un float (fixe) ou à un `ParamDist`
    (moyenne + IC issus du corpus). La structure du modèle est déterminée UNE fois par
    la présence des clés structurelles (incubation/cfr/immunité). Renvoie, pour chaque
    sortie (incidence/prevalence/cumulative/deaths/r_eff), `{median, lower, upper}`,
    plus `days`, `model`, `n_samples` et un `summary` agrégé (pics, taux d'attaque, R0
    en médiane [IC]). Déterministe pour un `seed` donné."""
    rng = random.Random(seed)
    dists = {k: _as_dist(params.get(k)) for k in _PARAM_FIELDS}
    n_samples = max(int(n_samples), 1)
    lo_q, hi_q = ci

    runs: list[dict] = []
    for _ in range(n_samples):
        kwargs = {}
        for k in _PARAM_FIELDS:
            d = dists[k]
            if d is None:
                continue  # laisse le défaut du dataclass (et, pour les structurels, désactive)
            kwargs[k] = d.sample(rng)
        try:
            runs.append(simulate(SeirParams(**kwargs), days=days))
        except Exception:
            continue  # un tirage numériquement pathologique ne casse pas l'ensemble

    if not runs:
        raise ValueError("SEIR ensemble: aucune trajectoire valide (paramètres invalides ?)")

    out: dict = {
        "model": runs[0]["model"],
        "n_samples": len(runs),
        "days": runs[0]["days"],
    }
    for key in ("incidence", "prevalence", "cumulative", "deaths", "r_eff",
                "susceptible", "exposed", "recovered"):
        out[key] = _band([r[key] for r in runs], lo_q, hi_q)

    def _sum_band(field: str) -> dict:
        col = sorted(r["summary"][field] for r in runs)
        return {
            "median": round(_percentile(col, 50.0), 4),
            "lower": round(_percentile(col, lo_q), 4),
            "upper": round(_percentile(col, hi_q), 4),
        }

    out["summary"] = {
        "model": runs[0]["model"],
        "r0": _sum_band("r0"),
        "peak_incidence": _sum_band("peak_incidence"),
        "peak_incidence_day": _sum_band("peak_incidence_day"),
        "peak_prevalence": _sum_band("peak_prevalence"),
        "peak_prevalence_day": _sum_band("peak_prevalence_day"),
        "attack_rate": _sum_band("attack_rate"),
        "total_deaths": _sum_band("total_deaths"),
    }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Paramètres extraits de la littérature → entrées du modèle
# ─────────────────────────────────────────────────────────────────────────────
# Paramètres épidémiologiques qu'on tente d'extraire du corpus (via LLM, cf. main).
# `serial_interval_days` sert au recoupement de R0 et à l'affichage ; il n'est PAS
# consommé par simulate_ensemble (clé inconnue de _PARAM_FIELDS → simplement ignorée).
_EXTRACTED_FIELDS = (
    "r0", "infectious_period_days", "incubation_period_days",
    "cfr", "immunity_duration_days", "serial_interval_days",
)


def _num_or_none(v):
    """float fini, ou None (couvre None, '', texte, NaN/inf)."""
    try:
        if v is None or v == "":
            return None
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _clean_provenance(prov, valid_ids: set) -> list[int]:
    """Ne garde que des ids d'articles RÉELS (présents dans `valid_ids`), dédupliqués,
    dans l'ordre. Coerce en int ; ignore ce qui n'est pas un id valide."""
    out: list[int] = []
    for i in prov or []:
        try:
            iv = int(i)
        except (TypeError, ValueError):
            continue
        if iv in valid_ids and iv not in out:
            out.append(iv)
    return out


def normalize_extracted_parameters(raw, valid_ids=None, quality_by_id=None) -> dict:
    """Nettoie le bloc `epidemic_parameters` d'une extraction LLM en un bloc
    DÉTERMINISTE : nombres coercés, provenance filtrée sur `valid_ids` (le pool
    pertinent — si fourni), et SEULS les paramètres dont la valeur centrale est un
    nombre conservés. Renvoie
    ``{applicable, disease, params:{nom:{value,ci_low,ci_high,unit,n_studies,provenance}}, cited}``.
    Si `quality_by_id` est fourni ET qu'un paramètre porte des `observations` par étude,
    l'estimation « narrative » du LLM est REMPLACÉE par un POOL NUMÉRIQUE pondéré par la
    qualité (cf. pool_weighted) dès qu'au moins deux études pèsent — plus rigoureux et
    reproductible. Pur (aucune I/O) : testable sans base ni réseau."""
    valid = set(valid_ids) if valid_ids is not None else None
    if not isinstance(raw, dict):
        return {"applicable": False, "disease": None, "params": {}, "cited": []}
    params: dict = {}
    cited: list[int] = []
    for name in _EXTRACTED_FIELDS:
        blk = raw.get(name)
        if not isinstance(blk, dict):
            continue
        val = _num_or_none(blk.get("value"))         # estimation LLM (peut être None)
        lo = _num_or_none(blk.get("ci_low"))
        hi = _num_or_none(blk.get("ci_high"))
        if lo is not None and hi is not None and lo > hi:
            lo, hi = None, None  # IC incohérent → ignoré (pas de fausse incertitude)
        prov = _clean_provenance(blk.get("provenance"), valid) if valid is not None else []
        try:
            n_studies = int(blk.get("n_studies"))
        except (TypeError, ValueError):
            n_studies = 0

        # Pooling NUMÉRIQUE pondéré par la qualité (prioritaire sur l'estimation LLM)
        # dès qu'au moins deux études réelles fournissent une observation.
        if quality_by_id is not None and isinstance(blk.get("observations"), list):
            _obs = []
            for o in blk["observations"]:
                if not isinstance(o, dict):
                    continue
                try:
                    _aid = int(o.get("article_id"))
                except (TypeError, ValueError):
                    continue
                if valid is None or _aid in valid:
                    _obs.append(o)
            pooled = pool_weighted(_obs, quality_by_id)
            if pooled and pooled["n_studies"] >= 2:
                val, lo, hi = pooled["value"], pooled["ci_low"], pooled["ci_high"]
                if pooled["provenance"]:
                    prov = pooled["provenance"]
                n_studies = max(n_studies, pooled["n_studies"])

        if val is None:
            continue  # ni valeur LLM ni pool numérique → paramètre ignoré
        if name == "cfr":
            val = min(max(val, 0.0), 1.0)  # létalité = proportion 0..1
            lo = min(max(lo, 0.0), 1.0) if lo is not None else None
            hi = min(max(hi, 0.0), 1.0) if hi is not None else None
        for i in prov:
            if i not in cited:
                cited.append(i)
        params[name] = {
            "value": val, "ci_low": lo, "ci_high": hi,
            "unit": str(blk.get("unit") or "")[:32],
            "n_studies": max(n_studies, len(prov)),
            "provenance": prov,
        }
    applicable = bool(raw.get("applicable")) and bool(params)
    disease = None
    if raw.get("population_disease"):
        disease = (str(raw.get("population_disease")).strip()[:120]) or None
    return {"applicable": applicable, "disease": disease, "params": params, "cited": cited}


def params_to_distributions(params) -> dict:
    """Bloc `params` (cf. `normalize_extracted_parameters`) → dict d'entrée pour
    `simulate_ensemble` : `ParamDist(moyenne, IC)` quand un IC est disponible, sinon la
    valeur seule. Pur ; les paramètres non consommés par le modèle sont inoffensifs."""
    out: dict = {}
    for name, blk in (params or {}).items():
        if not isinstance(blk, dict):
            continue
        val = _num_or_none(blk.get("value"))
        if val is None:
            continue
        lo = _num_or_none(blk.get("ci_low"))
        hi = _num_or_none(blk.get("ci_high"))
        if lo is not None and hi is not None and hi > lo:
            out[name] = ParamDist(mean=val, ci_low=lo, ci_high=hi)
        else:
            out[name] = val
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Contexte : population par géographie + pooling numérique pondéré par la qualité
# ─────────────────────────────────────────────────────────────────────────────
# Estimations de population (ordre de grandeur, ~2024) par intitulé géographique
# libre, pour dériver un DÉFAUT de projection du scénario au lieu d'un 1e6 fixe. Les
# alias Romandie/Léman restent cohérents avec data_connectors._REGION_COORDS.
_GEO_POPULATION: dict[str, float] = {
    "world": 8_000_000_000, "global": 8_000_000_000, "worldwide": 8_000_000_000, "monde": 8_000_000_000,
    "europe": 745_000_000, "european union": 449_000_000, "eu": 449_000_000, "ue": 449_000_000,
    "africa": 1_400_000_000, "afrique": 1_400_000_000, "asia": 4_700_000_000, "asie": 4_700_000_000,
    "north america": 600_000_000, "south america": 435_000_000, "latin america": 660_000_000,
    "oceania": 45_000_000, "middle east": 380_000_000,
    "france": 68_000_000, "switzerland": 8_800_000, "suisse": 8_800_000,
    "united states": 335_000_000, "united states of america": 335_000_000, "usa": 335_000_000, "u.s.": 335_000_000,
    "united kingdom": 67_000_000, "uk": 67_000_000, "england": 56_000_000, "royaume-uni": 67_000_000,
    "germany": 84_000_000, "allemagne": 84_000_000, "italy": 59_000_000, "italie": 59_000_000,
    "spain": 48_000_000, "espagne": 48_000_000, "canada": 40_000_000, "china": 1_410_000_000, "chine": 1_410_000_000,
    "india": 1_430_000_000, "inde": 1_430_000_000, "brazil": 215_000_000, "brésil": 215_000_000,
    "japan": 124_000_000, "japon": 124_000_000, "australia": 26_000_000, "australie": 26_000_000,
    "netherlands": 17_700_000, "pays-bas": 17_700_000, "belgium": 11_700_000, "belgique": 11_700_000,
    "sweden": 10_500_000, "norway": 5_500_000, "denmark": 5_900_000, "austria": 9_100_000, "autriche": 9_100_000,
    "portugal": 10_300_000, "greece": 10_400_000, "poland": 37_700_000, "ireland": 5_100_000, "irlande": 5_100_000,
    "geneva": 500_000, "geneve": 500_000, "genève": 500_000, "grand geneve": 1_050_000, "grand genève": 1_050_000,
    "vaud": 815_000, "lausanne": 815_000, "valais": 350_000, "sion": 350_000,
    "neuchatel": 175_000, "neuchâtel": 175_000, "fribourg": 325_000, "jura": 73_000, "romandie": 2_000_000,
}


def population_for_geography(geo_label) -> float | None:
    """Estimation de population pour un intitulé géographique libre (pays/région/monde),
    ou None si inconnu. Insensible à la casse ; teste le libellé ENTIER puis chaque
    segment séparé par virgule/slash/point-virgule (ex. « France, Europe » → France).
    PAS de sous-chaîne (éviter « us » ⊂ « australia »). PUR."""
    if not geo_label:
        return None
    raw = str(geo_label).strip().lower()
    if not raw:
        return None
    segs = [raw]
    for sep in (",", "/", ";", "|"):
        raw = raw.replace(sep, ",")
    segs += [s.strip() for s in raw.split(",")]
    for c in segs:
        if c and c in _GEO_POPULATION:
            return float(_GEO_POPULATION[c])
    return None


def pool_weighted(observations, quality_by_id=None) -> dict | None:
    """Agrège des observations PAR ÉTUDE ``{"article_id", "value"}`` en une estimation
    PONDÉRÉE PAR LA QUALITÉ : moyenne pondérée + IC 95 % ≈ moyenne ± 1.96·écart-type
    pondéré (dispersion inter-études). Poids = ``quality_by_id[article_id]`` (> 0),
    défaut 1.0 si absent/nul. Renvoie ``{value, ci_low, ci_high, n_studies, provenance}``
    ou None si aucune observation numérique valide. PUR (aucune I/O)."""
    pts: list[tuple[float, float, int | None]] = []
    for o in observations or []:
        if not isinstance(o, dict):
            continue
        v = _num_or_none(o.get("value"))
        if v is None:
            continue
        try:
            aid: int | None = int(o.get("article_id"))
        except (TypeError, ValueError):
            aid = None
        w = 1.0
        if aid is not None and quality_by_id:
            try:
                qf = float(quality_by_id.get(aid))
                if qf > 0:
                    w = qf
            except (TypeError, ValueError):
                pass
        pts.append((v, w, aid))
    if not pts:
        return None
    sw = sum(w for _v, w, _a in pts)
    if sw <= 0:
        return None
    mean = sum(v * w for v, w, _a in pts) / sw
    if len(pts) >= 2:
        var = sum(w * (v - mean) ** 2 for v, w, _a in pts) / sw
        sd = math.sqrt(max(var, 0.0))
    else:
        sd = 0.0
    prov: list[int] = []
    for _v, _w, a in pts:
        if a is not None and a not in prov:
            prov.append(a)
    return {
        "value": mean,
        "ci_low": (mean - _Z * sd) if sd > 0 else None,
        "ci_high": (mean + _Z * sd) if sd > 0 else None,
        "n_studies": len(pts),
        "provenance": prov,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Le SEIR comme SOUS-MODÈLE : quelles variables prédictives il DÉRIVE
# ─────────────────────────────────────────────────────────────────────────────
# Deux classifications PURES d'une variable (par son nom / machine_name) :
#   • `seir_feature_column` : la variable est une SORTIE du sous-modèle (compartiments
#     S/E/I/R/D et flux : incidence, prévalence, cumul, décès, susceptibles, exposés,
#     rétablis) → renvoie la colonne du connecteur SEIR qui la remplit automatiquement.
#     La variable RESTE une feature, mais sa valeur vient du modèle.
#   • `is_seir_parameter` : la variable est un PARAMÈTRE du sous-modèle (R0/Rt, CFR/IFR,
#     incubation, période infectieuse, intervalle sériel, durée d'immunité, taux de
#     transmission/contact/guérison…) → un input de simulation, PAS une feature.
# Vocabulaire EN+FR couvrant la famille SEIR+. On ne teste que NOM + machine_name (courts,
# contrôlés), jamais la définition libre, pour éviter les faux positifs. La reclassification
# n'a lieu QUE sur un scénario épidémique (SEIR applicable), ce qui borne fortement le
# risque de capturer une covariable externe légitime.
SEIR_OUTPUT_COLUMNS = (
    "seir_incidence", "seir_prevalence", "seir_cumulative", "seir_deaths",
    "seir_susceptible", "seir_exposed", "seir_recovered",
)


def _norm_ascii(text) -> str:
    """Minuscule, sans accents, séparateurs → espaces (snake_case et texte libre
    convergent). Pur."""
    import re as _re
    import unicodedata as _ud
    s = _ud.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode("ascii").lower()
    return _re.sub(r"[^a-z0-9]+", " ", s).strip()


# Ordre = PRIORITÉ (première correspondance gagne) : décès → rétablis → exposés → cumul →
# incidence → prévalence. Ainsi « décès cumulés » vise seir_deaths, « incidence cumulée »
# et « infectés au total » visent seir_cumulative, « cas actifs » vise seir_prevalence.
# Vocabulaire EN+FR, forme ASCII espacée.
_SEIR_OUTPUT_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("seir_deaths",      ("death", "deaths", "deces", "mortalit", "fatalit", "fatal",
                          "cumulative death", "number of deaths")),
    ("seir_recovered",   ("recovered", "recovery", "removed", "immune", "immunis", "immunity",
                          "immunite", "gueri", "retabli", "convalescent", "seroconvert")),
    ("seir_exposed",     ("exposed", "expose", "latent", "latently", "incubating",
                          "presymptomatic", "presymptomatique", "pre symptomatic")),
    ("seir_susceptible", ("susceptible", "susceptibles", "at risk population", "population a risque",
                          "non immun", "non immunise")),
    ("seir_cumulative",  ("cumulative", "cumul", "cumule", "attack rate", "taux d attaque",
                          "total case", "total cases", "total infection", "total infections",
                          "cas totaux", "infections totales", "final size", "taille finale",
                          "ever infected", "deja infecte")),
    ("seir_incidence",   ("incidence", "incident", "new case", "new cases", "new infection",
                          "new infections", "nouveaux cas", "nouvelle infection",
                          "nouvelles infections", "daily case", "daily cases", "daily infection",
                          "cas quotidien", "cas journalier", "reported case", "confirmed case",
                          "notified case", "cas confirme", "cas notifie", "cas declare",
                          "cas rapporte", "infection rate", "taux d infection",
                          "taux d incidence", "notification rate")),
    ("seir_prevalence",  ("prevalence", "prevalent", "active case", "active cases", "cas actif",
                          "infectious", "infectieux", "currently infected", "actuellement infecte",
                          "infected individual", "infected population", "personnes infectees",
                          "nombre d infecte", "symptomatic case", "cas symptomatique",
                          "active infection", "en cours d infection")),
)


def seir_feature_column(text) -> str | None:
    """Colonne de sortie du sous-modèle SEIR qu'une variable prédictive DÉSIGNE
    (incidence/prévalence/cumul/décès), ou None si aucune. PUR."""
    t = _norm_ascii(text)
    if not t:
        return None
    for col, keys in _SEIR_OUTPUT_KEYWORDS:
        if any(k in t for k in keys):
            return col
    return None


# Paramètres du sous-modèle SEIR+. Tokens COURTS non ambigus en MOT ENTIER (r0, rt, reff,
# cfr, ifr) ; expressions distinctives en sous-chaîne (EN+FR). On NE capture PAS les taux
# gérés HORS du modèle (vaccination, quarantaine, mobilité…) : ce sont des covariables
# externes légitimes, pas des paramètres consommés par la simulation.
_SEIR_PARAM_WORDS: frozenset[str] = frozenset({"r0", "rt", "reff", "cfr", "ifr"})
_SEIR_PARAM_SUBSTR: tuple[str, ...] = (
    "reproduction", "reproductif", "reproductive",
    "transmission rate", "transmission coefficient", "taux de transmission",
    "contact rate", "taux de contact",
    "case fatality", "infection fatality", "fatality rate", "letalite",
    "recovery rate", "taux de guerison", "removal rate",
    "incubation",
    "latent period", "periode de latence",
    "infectious period", "periode infectieuse", "periode de contagiosite", "duree de contagiosite",
    "serial interval", "intervalle seriel",
    "generation time", "generation interval", "temps de generation", "intervalle de generation",
    "immunity duration", "duree d immunite", "duree de l immunite",
    "waning", "immunite decroissante",
)


def is_seir_parameter(text) -> bool:
    """True si la variable est un PARAMÈTRE du sous-modèle SEIR (R0/CFR/incubation/
    intervalle sériel…) — un input de simulation, jamais une feature du prédicteur. PUR."""
    t = _norm_ascii(text)
    if not t:
        return False
    if set(t.split()) & _SEIR_PARAM_WORDS:
        return True
    return any(k in t for k in _SEIR_PARAM_SUBSTR)
