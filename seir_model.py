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

# 8 compartiments d'état intégrés : S, E, I, R, D, C, V, Q (C = infections cumulées,
# accumulateur monotone distinct — hors conservation S+E+I+R+D+V+Q = N). V (vaccinés) et
# Q (isolés/quarantaine) sont DÉSACTIVÉS (restent 0) sans leur paramètre → modèles de base
# strictement inchangés (S+E+I+R+D = N, cf. relation de taille finale).
_N_STATE = 8
_Z = 1.959963984540054  # quantile normal à 97.5 % (demi-largeur d'un IC 95 %)

# Plancher PHYSIQUE d'une durée épidémiologique (6 h). Une « période » tirée sous ce
# seuil ne décrit plus une maladie : gamma = 1/période explose et la trajectoire part
# à l'infini. À ne PAS confondre avec un plancher de POSITIVITÉ (~1e-9), qui convient
# à un taux mais transforme une durée en maladie impossible (cf. ParamDist.sample).
_MIN_PERIOD_DAYS = 0.25
# Champs exprimés en JOURS (durées) : plancher = _MIN_PERIOD_DAYS.
_PERIOD_FIELDS = frozenset((
    "infectious_period_days", "incubation_period_days", "immunity_duration_days",
))
# Champs exprimés en PROPORTION 0..1 : bornés des deux côtés à l'échantillonnage.
_FRACTION_FIELDS = frozenset(("cfr", "vaccine_efficacy"))


@dataclass
class SeirParams:
    """Paramètres d'UNE trajectoire. Les champs à None/0 désactivent leur structure."""
    r0: float | None = None                       # nombre de reproduction de base
    beta: float | None = None                     # taux de transmission /j (sinon dérivé de r0)
    infectious_period_days: float = 7.0           # → gamma = 1/période
    incubation_period_days: float | None = None   # → sigma ; présent ⇒ compartiment E (SEIR)
    cfr: float = 0.0                              # létalité 0..1 ; > 0 ⇒ compartiment D
    immunity_duration_days: float | None = None   # → omega ; présent ⇒ R→S (immunité décroissante)
    vaccination_rate: float | None = None         # → nu (fraction de S vaccinée /j) ; présent ⇒ V
    vaccine_efficacy: float = 1.0                 # ε (0..1) : fraction protégée par la vaccination
    quarantine_rate: float | None = None          # → kappa (fraction de I isolée /j) ; présent ⇒ Q
    population: float = 1_000_000.0
    initial_infected: float = 10.0
    initial_exposed: float = 0.0


def _rates(p: SeirParams) -> dict:
    """Convertit des paramètres « lisibles » (périodes, R0, CFR) en taux d'ODE + indicateurs.

    Lève ValueError si la période infectieuse n'est pas physiquement plausible : une
    durée nulle/négative donnait gamma = 1e6/j, soit une « maladie » qui infecte puis
    guérit en une fraction de seconde — et la trajectoire absurde qui va avec. Mieux
    vaut échouer franchement (simulate_ensemble écarte le tirage) que rendre un chiffre
    faux avec assurance."""
    per = p.infectious_period_days
    if per is None or not math.isfinite(float(per)) or float(per) < _MIN_PERIOD_DAYS:
        raise ValueError(
            f"SEIR: période infectieuse implausible ({per!r} j ; minimum "
            f"{_MIN_PERIOD_DAYS} j). Paramètre manquant ou dans la mauvaise unité ?")
    gamma = 1.0 / float(per)
    # `r0_source` distingue une valeur ISSUE DE LA LITTÉRATURE d'un repli codé en dur.
    # Sans lui, l'UI présentait un R0 = 2.5 inventé comme s'il était sourcé (avec un IC
    # de largeur nulle, ce qui se lit comme une certitude).
    r0_source = "literature"
    if p.beta is not None and p.beta > 0:
        beta = p.beta
    elif p.r0 is not None and p.r0 > 0:
        beta = p.r0 * gamma
    else:
        beta = 2.5 * gamma  # repli prudent (R0 = 2.5) si ni beta ni r0 fournis
        r0_source = "assumed"
    has_e = bool(p.incubation_period_days and p.incubation_period_days > 0)
    sigma = 1.0 / p.incubation_period_days if has_e else 0.0
    has_waning = bool(p.immunity_duration_days and p.immunity_duration_days > 0)
    omega = 1.0 / p.immunity_duration_days if has_waning else 0.0
    cfr = min(max(p.cfr or 0.0, 0.0), 1.0)
    eff = min(max(p.vaccine_efficacy if p.vaccine_efficacy is not None else 1.0, 0.0), 1.0)
    nu = (p.vaccination_rate * eff) if (p.vaccination_rate and p.vaccination_rate > 0) else 0.0
    kappa = p.quarantine_rate if (p.quarantine_rate and p.quarantine_rate > 0) else 0.0
    # R0 = reproduction de BASE (sans intervention) = beta/gamma. Rc = reproduction
    # CONTRÔLÉE : l'isolement retire les infectieux au taux kappa, donc la durée
    # infectieuse effective tombe à 1/(gamma+kappa). Sans quarantaine, Rc == R0.
    # C'est Rc — pas R0 — qui doit piloter r_eff, sinon on affiche « R > 1 » sur une
    # épidémie que le modèle lui-même montre en train de s'éteindre.
    return {
        "beta": beta, "gamma": gamma, "sigma": sigma, "omega": omega, "cfr": cfr,
        "nu": nu, "kappa": kappa,
        "has_e": has_e, "has_death": cfr > 0.0, "has_waning": has_waning,
        "has_v": nu > 0.0, "has_q": kappa > 0.0,
        "r0": beta / gamma if gamma > 0 else float("nan"),
        "rc": beta / (gamma + kappa) if (gamma + kappa) > 0 else float("nan"),
        "r0_source": r0_source,
    }


def model_name(rates: dict) -> str:
    """Libellé du modèle effectivement simulé, assemblé compartiment par compartiment.

    Ordre canonique S [V] [E] I [Q] R [D] [S]. Rétro-compatible : sans vaccination ni
    quarantaine on retrouve exactement SIR/SEIR/SEIRD/SEIRS/SIRD… ; avec elles on obtient
    p. ex. SVEIR (vaccination) ou SEIQR (quarantaine)."""
    name = "S"
    if rates.get("has_v"):
        name += "V"
    if rates["has_e"]:
        name += "E"
    name += "I"
    if rates.get("has_q"):
        name += "Q"
    name += "R"
    if rates["has_death"]:
        name += "D"
    if rates["has_waning"]:
        name += "S"
    return name


def _deriv(y: list[float], r: dict) -> tuple[float, ...]:
    """Membre de droite généralisé (en fractions ; N = 1). Ordre : S, E, I, R, D, C, V, Q.

    Conservation : dS+dE+dI+dR+dD+dV+dQ = 0 (C est un accumulateur à part). Le flux de
    sortie infectieux (gamma·(I+Q)) se répartit CFR→décès, (1−CFR)→guérison ; sans période
    d'incubation, les nouvelles infections vont directement de S à I (pas de E). nu vaccine
    S→V (protection définitive dans ce modèle simple) ; kappa isole I→Q (Q reste infectieux
    au plan clinique mais ne transmet plus). Avec nu=kappa=0, se réduit exactement au SEIR."""
    s, e, i, rec, _d, _c, v, q = y
    inf = r["beta"] * s * i          # force d'infection · S = nouvelles infections/j
    waning = r["omega"] * rec        # R → S (0 sans immunité décroissante)
    vacc = r["nu"] * s               # S → V (0 sans vaccination)
    to_quar = r["kappa"] * i         # I → Q (0 sans quarantaine)
    exit_i = r["gamma"] * i          # sortie « naturelle » de I (guérison/décès)
    exit_q = r["gamma"] * q          # sortie de Q (même période infectieuse)
    to_death = r["cfr"] * (exit_i + exit_q)
    to_recov = (exit_i + exit_q) - to_death
    if r["has_e"]:
        ds = -inf - vacc + waning
        de = inf - r["sigma"] * e
        di = r["sigma"] * e - exit_i - to_quar
    else:
        ds = -inf - vacc + waning
        de = 0.0
        di = inf - exit_i - to_quar
    drec = to_recov - waning
    dd = to_death
    dc = inf
    dv = vacc
    dq = to_quar - exit_q
    return (ds, de, di, drec, dd, dc, dv, dq)


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
    # État : S, E, I, R, D, C, V, Q. C initial = déjà infectés (E+I) ; V/Q partent de 0.
    y = [s0, e0, i0, 0.0, 0.0, i0 + e0, 0.0, 0.0]

    steps_per_day = max(int(round(1.0 / max(dt, 1e-6))), 1)
    step = 1.0 / steps_per_day
    r0v = r["r0"]
    rcv = r["rc"]   # reproduction contrôlée (== r0 sans quarantaine) → pilote r_eff

    days_out: list[int] = []
    incidence: list[float] = []
    prevalence: list[float] = []
    cumulative: list[float] = []
    deaths: list[float] = []
    r_eff: list[float] = []
    susceptible: list[float] = []
    exposed: list[float] = []
    recovered: list[float] = []
    vaccinated: list[float] = []
    quarantined: list[float] = []

    def _record(t: int, yv: list[float]) -> None:
        s, _e, i, _rec, d, c, v, q = yv
        days_out.append(t)
        incidence.append(r["beta"] * s * i * n)  # nouvelles infections/j (effectifs)
        prevalence.append(i * n)
        cumulative.append(c * n)
        deaths.append(d * n)
        r_eff.append(rcv * s)
        susceptible.append(s * n)   # compartiments S / E / R / V / Q exposés aussi (effectifs)
        exposed.append(_e * n)
        recovered.append(_rec * n)
        vaccinated.append(v * n)
        quarantined.append(q * n)

    _record(0, y)
    for day in range(1, int(days) + 1):
        for _ in range(steps_per_day):
            y = _rk4_step(y, r, step)
            # Divergence numérique : NaN/inf. Le bornage ci-dessous ne peut PAS l'attraper
            # (`nan < 0.0` vaut False), donc sans ce test le NaN se propageait jusqu'au
            # résumé — attack_rate = nan, peak_prevalence = inf — sans la moindre erreur.
            if not all(math.isfinite(v) for v in y):
                raise ValueError(
                    f"SEIR: l'intégration a divergé au jour {day} "
                    f"(beta·dt trop grand ? paramètres hors domaine ?)")
            # borne les micro-négatifs dus à l'arithmétique flottante (tous compartiments)
            if any(v < 0.0 for v in y):
                y = [v if v > 0.0 else 0.0 for v in y]
        _record(day, y)

    peak_inc_i = max(range(len(incidence)), key=lambda k: incidence[k])
    peak_prev_i = max(range(len(prevalence)), key=lambda k: prevalence[k])
    summary = {
        "model": model_name(r),
        "r0": round(r0v, 3),
        "r_control": round(rcv, 3),      # == r0 sans quarantaine
        "r0_source": r["r0_source"],     # "literature" | "assumed" (repli codé en dur)
        "peak_incidence": round(incidence[peak_inc_i], 2),
        "peak_incidence_day": days_out[peak_inc_i],
        "peak_prevalence": round(prevalence[peak_prev_i], 2),
        "peak_prevalence_day": days_out[peak_prev_i],
        "attack_rate": round(min(cumulative[-1] / n, 1.0), 4),
        "total_deaths": round(deaths[-1], 2),
        "total_vaccinated": round(vaccinated[-1], 2),      # 0 sans vaccination
        "peak_quarantine": round(max(quarantined), 2),     # 0 sans quarantaine
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
        "vaccinated": vaccinated,
        "quarantined": quarantined,
        "summary": summary,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble : propagation de l'incertitude des paramètres issus de la littérature
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ParamDist:
    """Distribution d'un paramètre : moyenne + IC 95 % (facultatif) issus du corpus.

    Échantillonnée comme une normale tronquée au domaine PHYSIQUE du paramètre. Sans IC
    (ou IC dégénéré), la valeur est fixe (== moyenne) — l'incertitude ne vient alors que
    des autres paramètres.

    `kind` porte l'UNITÉ, et donc le domaine valide :
      • "rate"     (défaut) : > 0, plancher de positivité ;
      • "period"   : une durée en JOURS, plancher `_MIN_PERIOD_DAYS` ;
      • "fraction" : une proportion, bornée à [0, 1].
    Sans cette distinction, un plancher de 1e-9 — correct pour un taux — s'appliquait à
    une durée et fabriquait, dans ~2 % des tirages de CHAQUE ensemble, une maladie dont
    la période infectieuse valait un milliardième de jour. Ces trajectoires absurdes
    (pics à 10⁷–10¹¹ cas/j) remontaient telles quelles dans l'IC publié."""
    mean: float
    ci_low: float | None = None
    ci_high: float | None = None
    kind: str = "rate"

    def sd(self) -> float:
        if (self.ci_low is not None and self.ci_high is not None
                and self.ci_high > self.ci_low):
            return (self.ci_high - self.ci_low) / (2.0 * _Z)
        return 0.0

    def bounds(self) -> tuple[float, float]:
        """Domaine physique (min, max) du paramètre, selon son unité."""
        if self.kind == "period":
            return (_MIN_PERIOD_DAYS, float("inf"))
        if self.kind == "fraction":
            return (0.0, 1.0)
        return (1e-9, float("inf"))

    def sample(self, rng: random.Random) -> float:
        sd = self.sd()
        lo, hi = self.bounds()
        if sd <= 0.0:
            return min(max(float(self.mean), lo), hi)
        # Normale TRONQUÉE par rejet : on retire tant que le tirage sort du domaine,
        # plutôt que de l'écraser sur la borne (ce qui empilait des valeurs impossibles
        # exactement AU plancher et biaisait la queue basse de l'IC).
        for _ in range(64):
            v = rng.gauss(self.mean, sd)
            if lo <= v <= hi:
                return v
        return min(max(float(self.mean), lo), hi)  # IC pathologique → repli sur la moyenne


# Paramètres dont la PRÉSENCE change la structure du modèle (E / D / R→S). Ils
# fixent la structure pour tout l'ensemble ; seules les VALEURS varient d'un tirage
# à l'autre (on ne veut pas que le modèle « clignote » entre tirages).
_STRUCTURAL = ("incubation_period_days", "cfr", "immunity_duration_days",
               "vaccination_rate", "quarantine_rate")
_PARAM_FIELDS = (
    "r0", "beta", "infectious_period_days", "incubation_period_days",
    "cfr", "immunity_duration_days", "population", "initial_infected",
    "initial_exposed", "vaccination_rate", "vaccine_efficacy", "quarantine_rate",
)


def _kind_for(field: str) -> str:
    """Unité d'un champ de `_PARAM_FIELDS` → domaine d'échantillonnage (cf. ParamDist)."""
    if field in _PERIOD_FIELDS:
        return "period"
    if field in _FRACTION_FIELDS:
        return "fraction"
    return "rate"


def _as_dist(v, field: str | None = None) -> ParamDist | None:
    """Coerce en ParamDist en RENSEIGNANT l'unité déduite du nom du champ, faute de quoi
    une durée serait échantillonnée avec le plancher d'un taux (cf. ParamDist)."""
    if v is None:
        return None
    kind = _kind_for(field) if field else "rate"
    if isinstance(v, ParamDist):
        # Une distribution construite sans `kind` explicite hérite de celui du champ.
        if field and v.kind == "rate" and kind != "rate":
            return ParamDist(v.mean, v.ci_low, v.ci_high, kind)
        return v
    return ParamDist(mean=float(v), kind=kind)


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
    """Agrège N trajectoires (une liste par tirage) en médiane + IC par pas de temps.

    Les valeurs non finies sont ÉCARTÉES avant le tri : `sorted()` sur une colonne
    contenant NaN ne lève rien et rend une liste NON triée, ce qui cassait silencieusement
    la garantie q05 ≤ q50 ≤ q95 (bande « basse » tracée au-dessus de la médiane)."""
    if not series_per_sample:
        return {"median": [], "lower": [], "upper": []}
    horizon = len(series_per_sample[0])
    median, lower, upper = [], [], []
    for t in range(horizon):
        col = sorted(v for v in (s[t] for s in series_per_sample) if math.isfinite(v))
        median.append(_percentile(col, 50.0))
        lower.append(_percentile(col, lo_q))
        upper.append(_percentile(col, hi_q))
    return {"median": median, "lower": lower, "upper": upper}


def _run_is_finite(res: dict) -> bool:
    """Un tirage est retenu seulement si TOUTES ses séries et son résumé sont finis."""
    for key in ("incidence", "prevalence", "cumulative", "deaths", "r_eff",
                "susceptible", "exposed", "recovered", "vaccinated", "quarantined"):
        if any(not math.isfinite(v) for v in res.get(key, ())):
            return False
    return all(math.isfinite(v) for v in res["summary"].values()
               if isinstance(v, (int, float)) and not isinstance(v, bool))


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
    dists = {k: _as_dist(params.get(k), k) for k in _PARAM_FIELDS}
    n_samples = max(int(n_samples), 1)
    lo_q, hi_q = ci

    runs: list[dict] = []
    dropped = 0
    for _ in range(n_samples):
        kwargs = {}
        for k in _PARAM_FIELDS:
            d = dists[k]
            if d is None:
                continue  # laisse le défaut du dataclass (et, pour les structurels, désactive)
            kwargs[k] = d.sample(rng)
        try:
            res = simulate(SeirParams(**kwargs), days=days)
        except Exception:
            dropped += 1
            continue  # un tirage numériquement pathologique ne casse pas l'ensemble
        # Un tirage divergent doit être écarté de TOUTES les séries et du résumé, pas
        # neutralisé série par série : sinon les bandes reposent sur des sous-ensembles
        # de tirages différents et cessent d'être cohérentes entre elles.
        if not _run_is_finite(res):
            dropped += 1
            continue
        runs.append(res)

    if not runs:
        raise ValueError("SEIR ensemble: aucune trajectoire valide (paramètres invalides ?)")

    out: dict = {
        "model": runs[0]["model"],
        "n_samples": len(runs),
        "n_dropped": dropped,   # tirages écartés : diagnostic d'un IC d'entrée trop large
        "days": runs[0]["days"],
    }
    for key in ("incidence", "prevalence", "cumulative", "deaths", "r_eff",
                "susceptible", "exposed", "recovered", "vaccinated", "quarantined"):
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
        # La structure du modèle est fixée pour tout l'ensemble : la provenance du R0
        # (littérature vs repli codé en dur) l'est donc aussi.
        "r0_source": runs[0]["summary"]["r0_source"],
        "r0": _sum_band("r0"),
        "r_control": _sum_band("r_control"),
        "peak_incidence": _sum_band("peak_incidence"),
        "peak_incidence_day": _sum_band("peak_incidence_day"),
        "peak_prevalence": _sum_band("peak_prevalence"),
        "peak_prevalence_day": _sum_band("peak_prevalence_day"),
        "attack_rate": _sum_band("attack_rate"),
        "total_deaths": _sum_band("total_deaths"),
        "total_vaccinated": _sum_band("total_vaccinated"),
        "peak_quarantine": _sum_band("peak_quarantine"),
    }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Calibration : ajuste R0 (+ un facteur d'échelle) sur une série OBSERVÉE
# ─────────────────────────────────────────────────────────────────────────────
_SEIRPARAMS_FIELDS = frozenset((
    "r0", "beta", "infectious_period_days", "incubation_period_days", "cfr",
    "immunity_duration_days", "vaccination_rate", "vaccine_efficacy",
    "quarantine_rate", "population", "initial_infected", "initial_exposed",
))


def _interp_at(axis: list[int], series: list[float], x: float) -> float:
    """Interpolation linéaire de `series` (indexée par `axis` croissant) au point x."""
    if not axis:
        return 0.0
    if x <= axis[0]:
        return float(series[0])
    if x >= axis[-1]:
        return float(series[-1])
    lo, hi = 0, len(axis) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if axis[mid] <= x:
            lo = mid
        else:
            hi = mid
    x0, x1 = axis[lo], axis[hi]
    if x1 == x0:
        return float(series[lo])
    f = (x - x0) / (x1 - x0)
    return float(series[lo]) * (1.0 - f) + float(series[hi]) * f


def _fit_shift_scale(axis, series, xs, ys, max_shift: float, steps: int = 60):
    """Meilleur décalage temporel τ ≥ 0 (le modèle a démarré τ jours AVANT la 1re
    observation) alignant model[x+τ] sur l'observé, avec l'échelle d'amplitude k
    optimale (moindres carrés) pour chaque τ. Renvoie (τ, k, sse). Une série observée
    réelle est une FENÊTRE d'une épidémie en cours : sans ce décalage, la forme ne
    s'aligne pas sur la courbe du modèle (qui part de t=0)."""
    def _at(tau):
        m = [_interp_at(axis, series, x + tau) for x in xs]
        denom = sum(v * v for v in m)
        k = (sum(mi * yi for mi, yi in zip(m, ys)) / denom) if denom > 1e-12 else 0.0
        if k < 0:
            k = 0.0
        return k, sum((k * mi - yi) ** 2 for mi, yi in zip(m, ys))
    max_shift = max(0.0, float(max_shift))
    step = (max_shift / steps) if max_shift > 0 else 1.0
    best, tau = None, 0.0
    while tau <= max_shift + 1e-9:
        k, sse = _at(tau)
        if best is None or sse < best[2]:
            best = (tau, k, sse)
        tau += step
    if max_shift > 0:                                # raffinement local
        s = step
        for _ in range(20):
            improved = False
            for tau in (best[0] - s, best[0] + s):
                if 0.0 <= tau <= max_shift:
                    k, sse = _at(tau)
                    if sse < best[2]:
                        best, improved = (tau, k, sse), True
            if not improved:
                s *= 0.5
    return best


def calibrate_to_observed(observed, base_params: dict, column: str = "incidence",
                          days: int | None = None, r0_bounds: tuple = (0.4, 8.0),
                          grid: int = 48) -> dict:
    """Ajuste R0 à une série OBSERVÉE par moindres carrés. L'écart d'UNITÉ entre le
    modèle (effectifs) et l'observé (cas, %, /100k…) est absorbé par un facteur
    d'échelle multiplicatif k, optimal analytiquement pour chaque R0 (moindres carrés
    linéaires) — on ajuste ainsi la FORME/TIMING via R0 et l'AMPLITUDE via k.

    `observed` = liste de (jour_offset, valeur) ; `base_params` = valeurs (moyennes)
    des autres paramètres (période infectieuse, incubation, cfr, population…) ;
    `column` ∈ {incidence, prevalence, cumulative, deaths}. Renvoie `{ok, fitted_r0,
    scale, rmse, r2, n_points, column, horizon_days}`. PUR : n'appelle que `simulate`
    (pas de dépendance scientifique), donc testable sans réseau ni base."""
    pts = []
    for d, v in (observed or []):
        try:
            dd, vv = float(d), float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(dd) and math.isfinite(vv) and dd >= 0:
            pts.append((dd, vv))
    if len(pts) < 3:
        return {"ok": False, "reason": "au moins 3 points observés requis", "n_points": len(pts)}
    if column not in ("incidence", "prevalence", "cumulative", "deaths"):
        column = "incidence"
    xs = [d for d, _ in pts]
    ys = [v for _, v in pts]
    horizon = max(2, min(int(days or (max(xs) + 1)), 3650))
    ymean = sum(ys) / len(ys)
    sst = sum((y - ymean) ** 2 for y in ys) or 1e-9
    base = {k: v for k, v in (base_params or {}).items() if k in _SEIRPARAMS_FIELDS}

    max_shift = max(0.0, horizon - max(xs) - 1)

    def _sse(r0: float):
        p = dict(base)
        p["r0"] = r0
        p.pop("beta", None)                      # laisser R0 piloter beta
        try:
            res = simulate(SeirParams(**p), days=horizon)
        except Exception:
            return (float("inf"), 0.0, 0.0)
        tau, k, sse = _fit_shift_scale(res["days"], res[column], xs, ys, max_shift)
        return (sse, k, tau)

    lo, hi = r0_bounds
    best = None
    for j in range(grid + 1):                    # balayage grossier sur R0
        r0 = lo + (hi - lo) * j / grid
        sse, k, tau = _sse(r0)
        if best is None or sse < best[0]:
            best = (sse, r0, k, tau)
    step = (hi - lo) / grid                       # raffinement local (descente 1D)
    for _ in range(24):
        improved = False
        for r0 in (best[1] - step, best[1] + step):
            if lo <= r0 <= hi:
                sse, k, tau = _sse(r0)
                if sse < best[0]:
                    best, improved = (sse, r0, k, tau), True
        if not improved:
            step *= 0.5
    sse, r0, k, tau = best
    return {
        "ok": True, "fitted_r0": round(r0, 4), "scale": k, "shift_days": round(tau, 1),
        "rmse": (sse / len(ys)) ** 0.5, "r2": round(1.0 - sse / sst, 4),
        "n_points": len(ys), "column": column, "horizon_days": horizon,
    }


def align_observed(points) -> dict:
    """Normalise une série observée hétérogène `[{date|day, value}]` en offsets JOUR
    alignés sur l'axe du modèle. Les dates ISO deviennent des offsets depuis la PLUS
    ANCIENNE (jour 0) ; les jours numériques sont gardés tels quels. Renvoie
    `{points:[(day:float, value:float)], start_date:str|None, n:int}` trié par jour.
    PUR — ancre le t=0 du modèle sur la première observation."""
    import re as _re
    from datetime import date as _date
    parsed = []
    for p in (points or []):
        if isinstance(p, dict):
            raw = p.get("date", p.get("day"))
            val = p.get("value")
        else:
            raw = p[0] if p else None
            val = p[1] if p and len(p) > 1 else None
        fv = _num_or_none(val)
        if fv is None:
            continue
        s = str(raw).strip()
        if _re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
            try:
                _y, _m, _d = (int(x) for x in s.split("-"))
                parsed.append((_date(_y, _m, _d), None, fv))
            except ValueError:
                continue
        else:
            dn = _num_or_none(raw)
            if dn is not None:
                parsed.append((None, float(dn), fv))
    if not parsed:
        return {"points": [], "start_date": None, "n": 0}
    iso_dates = [d for d, _, _ in parsed if d is not None]
    start = min(iso_dates) if iso_dates else None
    out = []
    for d_iso, d_num, fv in parsed:
        if d_iso is not None and start is not None:
            out.append((float((d_iso - start).days), fv))
        elif d_num is not None:
            out.append((float(d_num), fv))
    out.sort(key=lambda t: t[0])
    return {"points": out, "start_date": start.isoformat() if start else None, "n": len(out)}


def scale_observed_to_model(observed, base_params: dict, column: str = "incidence",
                            days: int | None = None) -> dict:
    """Échelle d'amplitude (moindres carrés) mappant le modèle COURANT (params fixes)
    sur la série observée, + le R² obtenu. Renvoie `{scale, r2, points:[{day, value}]}`
    où `value = observé/scale` ré-exprime l'observé en UNITÉS MODÈLE, prêt à superposer
    à la courbe. PUR. `scale=None` si non calculable (garde l'observé brut)."""
    valid = column if column in ("incidence", "prevalence", "cumulative", "deaths") else "incidence"
    pts = [(float(d), float(v)) for d, v in (observed or []) if _num_or_none(v) is not None]
    if not pts:
        return {"scale": None, "r2": None, "points": []}
    xs = [d for d, _ in pts]
    ys = [v for _, v in pts]
    horizon = max(2, min(int(days or (max(xs) + 1)), 3650))
    base = {k: v for k, v in (base_params or {}).items() if k in _SEIRPARAMS_FIELDS}
    raw_pts = [{"day": d, "value": v} for d, v in pts]
    try:
        res = simulate(SeirParams(**base), days=horizon)
    except Exception:
        return {"scale": None, "r2": None, "shift_days": 0.0, "points": raw_pts}
    max_shift = max(0.0, horizon - max(xs) - 1)
    tau, k, sse = _fit_shift_scale(res["days"], res[valid], xs, ys, max_shift)
    if k <= 0:
        return {"scale": None, "r2": None, "shift_days": 0.0, "points": raw_pts}
    ymean = sum(ys) / len(ys)
    sst = sum((y - ymean) ** 2 for y in ys) or 1e-9
    # points placés au JOUR MODÈLE aligné (x + τ), valeur ré-exprimée en unités modèle
    return {"scale": k, "r2": round(1.0 - sse / sst, 4), "shift_days": round(tau, 1),
            "points": [{"day": d + tau, "value": v / k} for d, v in pts]}


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


_PERCENT_UNITS = frozenset(("%", "percent", "percentage", "pourcent", "pourcentage",
                            "per cent", "pct", "%%"))


def _is_percent_unit(unit) -> bool:
    """Vrai si l'unité déclarée dit « pour-cent » (et non « proportion » / vide)."""
    u = str(unit or "").strip().lower()
    return u in _PERCENT_UNITS


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
                # L'IC du pool est moyenne ± 1.96·écart-type : sur une dispersion
                # inter-études large il descend sous zéro, ce qui n'a de sens pour
                # AUCUNE de ces grandeurs (R0, durées, létalité sont ≥ 0).
                if lo is not None and lo < 0.0:
                    lo = 0.0
                if pooled["provenance"]:
                    prov = pooled["provenance"]
                n_studies = max(n_studies, pooled["n_studies"])

        if val is None:
            continue  # ni valeur LLM ni pool numérique → paramètre ignoré
        unit = str(blk.get("unit") or "")[:32]
        if name == "cfr":
            # Le champ `unit` était stocké mais jamais lu : une létalité extraite en
            # POUR-CENT (« 40 » pour 40 %) était écrasée à 1.0, c.-à-d. 100 % de décès.
            # On convertit quand l'unité le dit, avant de borner.
            if _is_percent_unit(unit):
                val, lo, hi = (v / 100.0 if v is not None else None for v in (val, lo, hi))
            val = min(max(val, 0.0), 1.0)  # létalité = proportion 0..1
            lo = min(max(lo, 0.0), 1.0) if lo is not None else None
            hi = min(max(hi, 0.0), 1.0) if hi is not None else None
        for i in prov:
            if i not in cited:
                cited.append(i)
        params[name] = {
            "value": val, "ci_low": lo, "ci_high": hi,
            "unit": unit,
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
    # NB: pas de clé « eu » — c'est le participe passé du verbe avoir, qui apparaît
    # dans n'importe quel titre français (« ont eu ») et ancrait la projection sur
    # 449 millions d'habitants. Les formes non ambiguës suffisent.
    "europe": 745_000_000, "european union": 449_000_000, "union européenne": 449_000_000,
    "ue": 449_000_000,
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
    "neuchatel": 175_000, "neuchâtel": 175_000, "fribourg": 325_000, "jura": 73_000,
    "romandie": 2_000_000, "suisse romande": 2_000_000, "swiss romande": 2_000_000,
    "french-speaking switzerland": 2_000_000, "lake geneva region": 1_100_000,
    "leman": 1_100_000, "léman": 1_100_000,
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


def population_for_geography_in_text(text) -> tuple[float, str] | None:
    """Cherche DANS un texte libre (titre / requête d'un scénario) le meilleur intitulé
    géographique connu et renvoie (population, libellé), ou None. On préfère la zone la
    plus LOCALE citée (plus petite population) : « … en suisse romande » → Romandie (2M)
    plutôt que « suisse » (8.8M) ou « monde ». Correspondance par MOTS (frontières de
    mots après normalisation accents/séparateurs → espaces) pour éviter les collisions
    de sous-chaîne. PUR — sert à ancrer la projection SEIR sur la géographie du scénario,
    car le corpus d'un sujet épidémique est international (géographie modale ≈ « world »)."""
    t = _norm_ascii(text)
    if not t:
        return None
    padded = f" {t} "
    best: tuple[float, str] | None = None
    for label, pop in _GEO_POPULATION.items():
        lab = _norm_ascii(label)
        if lab and f" {lab} " in padded:
            if best is None or pop < best[0]:
                best = (float(pop), label)
    return best


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
