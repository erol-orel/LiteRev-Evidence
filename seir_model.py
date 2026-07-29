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

    def _record(t: int, yv: list[float]) -> None:
        s, _e, i, _rec, d, c = yv
        days_out.append(t)
        incidence.append(r["beta"] * s * i * n)  # nouvelles infections/j (effectifs)
        prevalence.append(i * n)
        cumulative.append(c * n)
        deaths.append(d * n)
        r_eff.append(r0v * s)

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
    for key in ("incidence", "prevalence", "cumulative", "deaths", "r_eff"):
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
