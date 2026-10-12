"""Pooled estimates from the extraction: what the studies say TOGETHER.

The extraction gives, per study, the number of cases in a group and the size of that group
(`n_cases`, `pop_risk`). This pools them across the whole relevant corpus:

  - a proportion per codebook label (how often "male" or "unvaccinated" or "full PPE" is a
    case), by random-effects meta-analysis on the logit scale, with a 95% interval, a 95%
    prediction interval (where a NEW study would fall), and the heterogeneity (Q, I2, tau2);
  - an odds ratio between two labels of one group reported in the same paper (male against
    female, vaccinated against unvaccinated), pooled the same way over the papers that report both.

A pooled number is only as good as the question "are these studies measuring the same
thing?", and this code does not pretend to answer it. So it narrows what it will combine and it
shows its working:

  - rows are pooled only within one codebook label, one sheet and (by default) one disease. The
    disease is recognised whatever the spelling (`api/diseases.py`): "HPAI H5N1", "influenza
    A(H5N1)" and "H5N1" are one disease, A(H5N1), and A(H9N2) is another. Each pool lists the
    spellings it merged, so the merge can be checked;
  - fewer than `MIN_STUDIES` studies get no pooled estimate, only the studies themselves;
  - a row counts only if both counts are whole numbers with 0 <= cases <= population, and (by
    default) the quote it rests on was found in the text; rejected rows and rows two reviewers
    disagree on never count; a reviewer's correction replaces the model's value;
  - one study contributes one row per label (the largest population), and the ones dropped are
    counted;
  - every result carries its studies, their weights, and I2, so a wide disagreement is
    visible beside the number and not hidden behind it.

The statistics are the standard ones: DerSimonian-Laird for the between-study variance, the
Hartung-Knapp-Sidik-Jonkman interval (with the safeguard that it is never narrower than the
DerSimonian-Laird one) because the intervals of the plain method are too narrow with few
studies, and the Wilson interval for a single study. They are checked in the tests against
`statsmodels` and `scipy`, which this module does not import.
"""
from __future__ import annotations

import math
from itertools import combinations
from typing import Any

from fastapi import Query
from sqlalchemy import text

from .codebook import Index, get_index
from .core import app, engine, logger
from .diseases import disease_key, disease_spellings, fold_disease, spellings_text
from .extraction_review import load_reviews, overlay_annotated
from .scenario_store import (_get_scenario_threshold, _get_user_scenario_or_404,
                             relevant_gate_sql)

MIN_STUDIES = 3
_MAX_GROUPS = 120
_MAX_PAIRS_PER_GROUP = 15     # LISTED per group in the served page; every pair is computed and counted
_MAX_PAIRS = 40

# ─────────────────────────────────────────────────────────────────────────────
# Distribution helpers (no scipy at runtime)
# ─────────────────────────────────────────────────────────────────────────────
#: Two-sided 95% critical values of Student's t. Exact to 4 decimals up to 30 degrees of
#: freedom; beyond that interpolated in 1/df between the usual table points (error < 0.002).
_T975 = {1: 12.7062, 2: 4.3027, 3: 3.1824, 4: 2.7764, 5: 2.5706, 6: 2.4469, 7: 2.3646, 8: 2.3060,
         9: 2.2622, 10: 2.2281, 11: 2.2010, 12: 2.1788, 13: 2.1604, 14: 2.1448, 15: 2.1314,
         16: 2.1199, 17: 2.1098, 18: 2.1009, 19: 2.0930, 20: 2.0860, 21: 2.0796, 22: 2.0739,
         23: 2.0687, 24: 2.0639, 25: 2.0595, 26: 2.0555, 27: 2.0518, 28: 2.0484, 29: 2.0452,
         30: 2.0423}
_T_TAIL = [(30, 2.0423), (40, 2.0211), (60, 2.0003), (120, 1.9799), (10 ** 9, 1.9600)]


def t975(df: int) -> float:
    if df < 1:
        raise ValueError("degrees of freedom must be at least 1")
    if df in _T975:
        return _T975[df]
    for (d0, t0), (d1, t1) in zip(_T_TAIL, _T_TAIL[1:]):
        if d0 <= df <= d1:
            f = (1 / df - 1 / d0) / (1 / d1 - 1 / d0)
            return t0 + f * (t1 - t0)
    return 1.96


def _gammainc_lower_series(a: float, x: float) -> float:
    term = total = 1.0 / a
    n = a
    for _ in range(500):
        n += 1
        term *= x / n
        total += term
        if abs(term) < abs(total) * 1e-14:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gammainc_upper_cf(a: float, x: float) -> float:
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def chi2_sf(x: float, df: int) -> float:
    """P(chi-square with `df` degrees of freedom > x): the p-value of the heterogeneity test."""
    if df < 1 or x <= 0:
        return 1.0
    a, half = df / 2.0, x / 2.0
    if half < a + 1:
        return max(0.0, min(1.0, 1.0 - _gammainc_lower_series(a, half)))
    return max(0.0, min(1.0, _gammainc_upper_cf(a, half)))


# ─────────────────────────────────────────────────────────────────────────────
# One study
# ─────────────────────────────────────────────────────────────────────────────
def wilson(x: float, n: float, z: float = 1.959964) -> tuple[float, float]:
    """The Wilson score interval of a proportion, for a single study."""
    p = x / n
    z2 = z * z
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    # The bounds are exactly 0 with no cases and exactly 1 with all cases, not 0.9999999999999999.
    return (0.0 if x == 0 else max(0.0, centre - half)), (1.0 if x == n else min(1.0, centre + half))


def logit_proportion(x: float, n: float) -> tuple[float, float]:
    """(logit, variance). A study with no cases, or all cases, gets 0.5 added to each cell,
    which is what keeps its logit finite, and only that study."""
    a, b = x, n - x
    if a == 0 or b == 0:
        a, b = a + 0.5, b + 0.5
    return math.log(a / b), 1.0 / a + 1.0 / b


def log_odds_ratio(x1: float, n1: float, x2: float, n2: float) -> tuple[float, float]:
    """(log odds ratio of group 1 against group 2, variance), 0.5 added to every cell of a
    study that has an empty one."""
    a, b, c, d = x1, n1 - x1, x2, n2 - x2
    if 0 in (a, b, c, d):
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    return math.log((a * d) / (b * c)), 1 / a + 1 / b + 1 / c + 1 / d


# ─────────────────────────────────────────────────────────────────────────────
# Pooling
# ─────────────────────────────────────────────────────────────────────────────
def pool_effects(ys: list[float], vs: list[float]) -> dict[str, Any]:
    """Random-effects pooling of effects `ys` with variances `vs`.

    tau2 by DerSimonian-Laird. The interval is Hartung-Knapp-Sidik-Jonkman, on k-1 degrees of
    freedom, never narrower than the DerSimonian-Laird interval. The prediction interval (k >= 3)
    uses k-2 degrees of freedom. Returns the weights (percent) too."""
    k = len(ys)
    if k < 2 or len(vs) != k or any(v <= 0 for v in vs):
        raise ValueError("pooling needs at least two effects with positive variances")
    w = [1.0 / v for v in vs]
    sw = sum(w)
    y_fe = sum(wi * yi for wi, yi in zip(w, ys)) / sw
    q = sum(wi * (yi - y_fe) ** 2 for wi, yi in zip(w, ys))
    df = k - 1
    c = sw - sum(wi * wi for wi in w) / sw
    tau2 = max(0.0, (q - df) / c) if c > 0 else 0.0
    ws = [1.0 / (v + tau2) for v in vs]
    sws = sum(ws)
    mu = sum(wi * yi for wi, yi in zip(ws, ys)) / sws
    se_dl = math.sqrt(1.0 / sws)
    q_hk = sum(wi * (yi - mu) ** 2 for wi, yi in zip(ws, ys)) / df
    se = math.sqrt(max(q_hk, 1.0) / sws)
    tc = t975(df)
    out: dict[str, Any] = {
        "k": k, "mu": mu, "se": se, "se_dl": se_dl, "ci": (mu - tc * se, mu + tc * se),
        "tau2": tau2, "Q": q, "df": df, "p_Q": chi2_sf(q, df),
        "I2": max(0.0, (q - df) / q * 100.0) if q > 0 else 0.0,
        "weights_pct": [wi / sws * 100.0 for wi in ws], "pi": None,
    }
    if k >= 3:
        half = t975(k - 2) * math.sqrt(tau2 + se * se)
        out["pi"] = (mu - half, mu + half)
    return out


def _expit(y: float) -> float:
    return 1.0 / (1.0 + math.exp(-y))


def heterogeneity_band(i2: float) -> str:
    """I2 in words (Higgins), so the number is not read without its meaning."""
    return "low" if i2 < 25 else "moderate" if i2 < 50 else "substantial" if i2 < 75 else "considerable"


def _het(p: dict[str, Any]) -> dict[str, Any]:
    return {"Q": round(p["Q"], 3), "df": p["df"], "p": round(p["p_Q"], 4), "I2": round(p["I2"], 1),
            "tau2": round(p["tau2"], 4), "band": heterogeneity_band(p["I2"])}


def pool_proportions(studies: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool studies carrying `x` (cases) and `n` (group size). Returns the pooled proportion with
    its interval and prediction interval, the heterogeneity, and each study with its own Wilson
    interval and weight."""
    ys, vs = zip(*(logit_proportion(s["x"], s["n"]) for s in studies))
    p = pool_effects(list(ys), list(vs))
    lo, hi = p["ci"]
    pooled = {"p": _expit(p["mu"]), "ci_low": _expit(lo), "ci_high": _expit(hi),
              "pi_low": _expit(p["pi"][0]) if p["pi"] else None,
              "pi_high": _expit(p["pi"][1]) if p["pi"] else None}
    rows = []
    for s, wt in zip(studies, p["weights_pct"]):
        l, h = wilson(s["x"], s["n"])
        rows.append({**s, "p": s["x"] / s["n"], "ci_low": l, "ci_high": h, "weight_pct": wt})
    return {"pooled": pooled, "heterogeneity": _het(p), "studies": rows}


def pool_odds_ratios(studies: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool studies carrying (`x1`, `n1`) and (`x2`, `n2`): the odds ratio of group 1 against 2."""
    ys, vs = zip(*(log_odds_ratio(s["x1"], s["n1"], s["x2"], s["n2"]) for s in studies))
    p = pool_effects(list(ys), list(vs))
    lo, hi = p["ci"]
    pooled = {"or": math.exp(p["mu"]), "ci_low": math.exp(lo), "ci_high": math.exp(hi),
              "pi_low": math.exp(p["pi"][0]) if p["pi"] else None,
              "pi_high": math.exp(p["pi"][1]) if p["pi"] else None}
    rows = []
    for s, y, v, wt in zip(studies, ys, vs, p["weights_pct"]):
        se = math.sqrt(v)
        rows.append({**s, "or": math.exp(y), "ci_low": math.exp(y - 1.96 * se), "ci_high": math.exp(y + 1.96 * se),
                     "weight_pct": wt})
    return {"pooled": pooled, "heterogeneity": _het(p), "studies": rows}


# ─────────────────────────────────────────────────────────────────────────────
# From extracted rows to groups of studies
# ─────────────────────────────────────────────────────────────────────────────
def _whole(v: Any) -> int | None:
    if isinstance(v, bool) or v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return int(f) if f == f and f.is_integer() and 0 <= f < 1e12 else None


def row_counts(o: dict[str, Any]) -> tuple[int, int] | str:
    """(cases, population) of an observation, or the reason it cannot be used: `missing`
    (a count is absent) or `invalid` (not whole numbers, or cases above the population)."""
    x_raw, n_raw = o.get("n_cases"), o.get("pop_risk")
    if x_raw is None or n_raw is None:
        return "missing"
    x, n = _whole(x_raw), _whole(n_raw)
    if x is None or n is None or n < 1 or x > n:
        return "invalid"
    return x, n


def group_key(o: dict[str, Any], split_disease: bool) -> tuple:
    return (o.get("sheet") or "", o.get("group_key") or "", o.get("covariate_key") or "",
            disease_key(o.get("disease")) if split_disease else "")


def _spelt(rows: list[dict[str, Any]]) -> list[Any]:
    """The disease as one study's rows wrote it, each spelling once (a study's two rows of a
    comparison may write it alike or not)."""
    seen: dict[str, Any] = {}
    for r in rows:
        seen.setdefault(fold_disease(r["obs"].get("disease")), r["obs"].get("disease"))
    return list(seen.values())


def build_groups(rows: list[dict[str, Any]], split_disease: bool = True) -> tuple[dict[tuple, list[dict]], int]:
    """{(sheet, group, label, disease key): [one study per article]}, and how many rows were
    dropped because the same article gave the same label twice for one disease, however spelt
    (the largest population is kept)."""
    best: dict[tuple, dict[int, dict]] = {}
    dropped = 0
    for r in rows:
        k = group_key(r["obs"], split_disease)
        cur = best.setdefault(k, {}).get(r["article_id"])
        if cur is None:
            best[k][r["article_id"]] = r
        else:
            dropped += 1
            if r["n"] > cur["n"]:
                best[k][r["article_id"]] = r
    return {k: list(v.values()) for k, v in best.items()}, dropped


def _study(r: dict[str, Any]) -> dict[str, Any]:
    return {"article_id": r["article_id"], "title": r["title"], "year": r["year"], "first_author": r["first_author"],
            "x": r["x"], "n": r["n"], "review_status": r["review_status"]}


def pooled_groups(rows: list[dict[str, Any]], min_studies: int = MIN_STUDIES, split_disease: bool = True) -> list[dict[str, Any]]:
    groups, _ = build_groups(rows, split_disease)
    out = []
    for (sheet, g, label, disease), items in groups.items():
        studies = [_study(r) for r in items]
        entry: dict[str, Any] = {
            "sheet": sheet, "group": g, "label": label, "disease": disease or None,
            "disease_spellings": disease_spellings(r["obs"].get("disease") for r in items),
            "label_path": items[0]["obs"].get("label_path"), "mapped": bool(items[0]["obs"].get("matched")),
            "k": len(studies), "n_total": sum(s["n"] for s in studies), "events_total": sum(s["x"] for s in studies),
            "pooled": None, "heterogeneity": None, "studies": studies, "reason": None,
        }
        if len(studies) >= max(min_studies, 2):
            entry.update(pool_proportions(studies))
        else:
            entry["reason"] = "too_few_studies"
            for s in studies:
                s["p"], (s["ci_low"], s["ci_high"]) = s["x"] / s["n"], wilson(s["x"], s["n"])
        out.append(entry)
    out.sort(key=lambda e: (-e["k"], -e["n_total"], e["sheet"], e["label"]))
    # AUCUNE coupe ici : la fonction est pure et rend tout ce qu'elle a calculé. La
    # coupe appartient à la réponse servie, qui doit la DIRE (cf. pooled_estimates) : le
    # panneau affichait « Libellés avec moins de 3 études (120) » alors qu'il y en avait
    # 306, parce qu'il comptait la liste déjà tronquée, et le CSV téléchargé ne portait
    # que les 120 premiers sans un mot.
    return out


def pooled_comparisons(rows: list[dict[str, Any]], order: dict[tuple, int], min_studies: int = MIN_STUDIES,
                       split_disease: bool = True) -> list[dict[str, Any]]:
    """Odds ratios between two labels of one group, over the papers that report both. The pair is
    ordered as the codebook orders its labels (male before female)."""
    groups, _ = build_groups(rows, split_disease)
    by_cat: dict[tuple, dict[str, dict[int, dict]]] = {}
    for (sheet, g, label, disease), items in groups.items():
        for r in items:
            by_cat.setdefault((sheet, g, disease), {}).setdefault(label, {})[r["article_id"]] = r
    out = []
    for (sheet, g, disease), labels in by_cat.items():
        pairs = []
        for a, b in combinations(sorted(labels), 2):
            common = labels[a].keys() & labels[b].keys()
            if len(common) >= max(min_studies, 2):
                pairs.append((len(common), a, b))
        for k, a, b in sorted(pairs, reverse=True):
            first, second = sorted((a, b), key=lambda lab: (order.get((sheet, g, lab), 10 ** 6), lab))
            studies, written = [], []
            for aid in sorted(labels[a].keys() & labels[b].keys()):
                r1, r2 = labels[first][aid], labels[second][aid]
                studies.append({**_study(r1), "x1": r1["x"], "n1": r1["n"], "x2": r2["x"], "n2": r2["n"]})
                written += _spelt([r1, r2])
            out.append({"sheet": sheet, "group": g, "disease": disease or None,
                        "disease_spellings": disease_spellings(written), "a": first, "b": second,
                        "k": len(studies), **pool_odds_ratios(studies)})
    out.sort(key=lambda e: (-e["k"], e["sheet"], e["group"], e["a"]))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# From the database
# ─────────────────────────────────────────────────────────────────────────────


def _pool_first_author(authors: str | None) -> str:
    first = (authors or "").replace("\n", ";").split(";")[0].split(",")[0].strip()
    return first.split()[0] if first else ""


def collect_rows(scenario_id: str, reviewed_only: bool = False, verified_only: bool = True) -> tuple[list[dict], dict[str, int], Index]:
    """The rows that may be pooled, from EVERY relevant extracted article, and why the others
    were left out."""
    thr = _get_scenario_threshold(scenario_id)
    gate = relevant_gate_sql("d", "ars", ":thr")
    index = get_index(scenario_id)
    with engine.connect() as conn:
        arts = conn.execute(text(f"""
            SELECT d.id, d.title, d.year, d.authors, d.extraction_json
            FROM literature_document d JOIN article_scenarios ars ON ars.document_id = d.id
            WHERE ars.scenario_id = :sid AND {gate} AND jsonb_typeof(d.extraction_json->'observations') = 'array'
        """), {"sid": scenario_id, "thr": thr}).mappings().all()
    reviews = load_reviews([int(a["id"]) for a in arts])
    excl = {"rejected": 0, "conflict": 0, "not_reviewed": 0, "quote_not_found": 0, "missing_counts": 0, "invalid_counts": 0}
    rows: list[dict] = []
    for a in arts:
        stored = [o for o in a["extraction_json"].get("observations", []) if isinstance(o, dict)]
        obs, _stale = overlay_annotated(int(a["id"]), stored, index, reviews)
        for o in obs:
            st = o["review_status"]
            if st in ("rejected", "conflict"):
                excl[st] += 1
                continue
            if reviewed_only and st == "unreviewed":
                excl["not_reviewed"] += 1
                continue
            eff = o["effective"]
            if verified_only and eff.get("quote_verified") is not True:
                excl["quote_not_found"] += 1
                continue
            c = row_counts(eff)
            if isinstance(c, str):
                excl[f"{c}_counts"] += 1
                continue
            rows.append({"article_id": int(a["id"]), "title": a["title"], "year": a["year"],
                         "first_author": _pool_first_author(a["authors"]), "x": c[0], "n": c[1],
                         "review_status": st, "obs": eff})
    return rows, excl, index


def pooled_estimates(scenario_id: str, reviewed_only: bool = False, verified_only: bool = True,
                     split_disease: bool = True, min_studies: int = MIN_STUDIES, full: bool = False) -> dict[str, Any]:
    rows, excl, index = collect_rows(scenario_id, reviewed_only, verified_only)
    order = {(n["sheet"], n["l1"], n["l2"]): i for i, n in enumerate(index.nodes) if n.get("l2")}
    _groups, dropped = build_groups(rows, split_disease)
    _all_pooled = pooled_groups(rows, min_studies, split_disease)
    _all_comps = pooled_comparisons(rows, order, min_studies, split_disease)
    pooled, pooled_total = (_all_pooled if full else _all_pooled[:_MAX_GROUPS]), len(_all_pooled)
    (comps, over_group), comps_total = ((_all_comps, 0) if full else _listed_comparisons(_all_comps)), len(_all_comps)
    # Compté sur TOUT ce qui existe, pas sur la page servie : c'est la ligne que le
    # panneau affiche, et elle disait 120 pour 306.
    _too_few = sum(1 for g in _all_pooled if g.get("reason") == "too_few_studies")
    return {
        "scenario_id": scenario_id,
        "filters": {"reviewed_only": reviewed_only, "verified_only": verified_only,
                    "split_disease": split_disease, "min_studies": min_studies},
        "n_rows_used": len(rows), "n_duplicate_rows_dropped": dropped, "excluded": excl,
        "pooled": pooled,
        "comparisons": comps,
        # Ce qui EXISTE, à côté de ce qui est servi. Le panneau comptait ses libellés
        # « trop peu nombreux » sur la liste tronquée, et le CSV téléchargé ne portait
        # que les 120 premiers sans le dire.
        "groups_total": pooled_total,
        "groups_returned": len(pooled),
        "groups_truncated": pooled_total > len(pooled),
        "groups_too_few_studies": _too_few,
        "comparisons_total": comps_total,
        "comparisons_returned": len(comps),
        "comparisons_truncated": comps_total > len(comps),
        "comparisons_over_group_cap": over_group, "full": bool(full),
        "caps": {"groups": _MAX_GROUPS, "comparisons": _MAX_PAIRS,
                 "pairs_per_group": _MAX_PAIRS_PER_GROUP},
    }


@app.get("/user-scenarios/{scenario_id}/extraction/pooled")
def get_pooled_estimates(scenario_id: str, reviewed_only: bool = Query(False), verified_only: bool = Query(True),
                         split_disease: bool = Query(True), min_studies: int = Query(MIN_STUDIES, ge=2, le=20),
                         full: bool = Query(False)) -> dict[str, Any]:
    """Pooled proportions and odds ratios over the whole relevant corpus. See the module note
    for what is pooled, what is left out, and how to read it."""
    _get_user_scenario_or_404(scenario_id)
    try:
        return pooled_estimates(scenario_id, reviewed_only, verified_only, split_disease, min_studies, full)
    except Exception as e:                                       # noqa: BLE001
        logger.warning(f"pooled_estimates {scenario_id}: {e}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# What is listed, and everything as tables (the CSV download, the Excel export)
# ─────────────────────────────────────────────────────────────────────────────
def _listed_comparisons(comps: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """The comparisons a served page lists: the `_MAX_PAIRS_PER_GROUP` with the most papers in
    each group (one group with many labels, occupation has 55 possible pairs, does not fill the
    page), then the first `_MAX_PAIRS`; and how many were passed over for their group's cap.
    Every pair is computed and counted all the same: the cap was 6, applied before anything was
    counted, so the seventh odds ratio of a group was never computed."""
    per_group: dict[tuple, int] = {}
    listed, over = [], 0
    for c in comps:                                   # already ordered by papers, most first
        key = (c["sheet"], c["group"], c["disease"])
        if per_group.get(key, 0) >= _MAX_PAIRS_PER_GROUP:
            over += 1
            continue
        per_group[key] = per_group.get(key, 0) + 1
        listed.append(c)
    return listed[:_MAX_PAIRS], over


POOLED_LABEL_HEADER = ["Sheet", "Group", "Label", "Codebook label", "Disease", "Studies", "Cases", "People",
                       "Pooled proportion", "95% CI low", "95% CI high", "95% PI low", "95% PI high",
                       "I2 (%)", "tau2", "Disagreement", "Pooled", "Studies (first author year: cases/people)",
                       "Disease as written (studies)"]
COMPARISON_HEADER = ["Sheet", "Group", "Disease", "Label A", "Label B", "Papers", "Odds ratio (A against B)",
                     "95% CI low", "95% CI high", "95% PI low", "95% PI high", "I2 (%)", "tau2", "Disagreement",
                     "Studies (first author year: A cases/people vs B cases/people)", "Disease as written (studies)"]


def _who(s: dict[str, Any]) -> str:
    return f"{s.get('first_author') or '?'} {s.get('year') or ''}".strip()


def pooled_label_rows(groups: list[dict[str, Any]]) -> list[list[Any]]:
    """One row per label, pooled or not (`Pooled` = N: fewer studies than asked)."""
    out = []
    for g in groups:
        p, h = g.get("pooled") or {}, g.get("heterogeneity") or {}
        out.append([g["sheet"], g["group"], g["label"], g.get("label_path"), g.get("disease"), g["k"],
                    g["events_total"], g["n_total"], p.get("p"), p.get("ci_low"), p.get("ci_high"),
                    p.get("pi_low"), p.get("pi_high"), h.get("I2"), h.get("tau2"), h.get("band"),
                    "Y" if g.get("pooled") else "N",
                    "; ".join(f"{_who(s)}: {s['x']}/{s['n']}" for s in g.get("studies") or []),
                    spellings_text(g.get("disease_spellings") or [])])
    return out


def comparison_rows(comps: list[dict[str, Any]]) -> list[list[Any]]:
    """One row per comparison between two labels of a group."""
    out = []
    for c in comps:
        p, h = c.get("pooled") or {}, c.get("heterogeneity") or {}
        out.append([c["sheet"], c["group"], c.get("disease"), c["a"], c["b"], c["k"], p.get("or"),
                    p.get("ci_low"), p.get("ci_high"), p.get("pi_low"), p.get("pi_high"), h.get("I2"),
                    h.get("tau2"), h.get("band"),
                    "; ".join(f"{_who(s)}: {s.get('x1')}/{s.get('n1')} vs {s.get('x2')}/{s.get('n2')}"
                              for s in c.get("studies") or []),
                    spellings_text(c.get("disease_spellings") or [])])
    return out


def pooled_csv(result: dict[str, Any]) -> str:
    """Every label and every comparison of a `full` result, one long table: `kind` is
    `proportion` (label B and the odds ratio empty) or `odds_ratio` (cases and people empty)."""
    import csv
    import io
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["kind", "sheet", "group", "label", "label_b", "codebook_label", "disease", "studies", "cases",
                "people", "estimate", "ci_low", "ci_high", "pi_low", "pi_high", "I2", "tau2", "disagreement",
                "pooled", "study_list", "disease_as_written"])
    for r in pooled_label_rows(result["pooled"]):
        w.writerow(["proportion", r[0], r[1], r[2], None, r[3], r[4], *r[5:]])
    for r in comparison_rows(result["comparisons"]):
        w.writerow(["odds_ratio", r[0], r[1], r[3], r[4], None, r[2], r[5], None, None, *r[6:14], "Y", r[14], r[15]])
    return buf.getvalue()


@app.get("/user-scenarios/{scenario_id}/extraction/pooled/export")
def export_pooled_estimates(scenario_id: str, reviewed_only: bool = Query(False), verified_only: bool = Query(True),
                            split_disease: bool = Query(True), min_studies: int = Query(MIN_STUDIES, ge=2, le=20)):
    """EVERY label (pooled or with too few studies) and EVERY comparison as one CSV, with the
    same filters as the panel. The panel lists a capped page; this is the whole of it."""
    import re

    from fastapi.responses import Response
    _get_user_scenario_or_404(scenario_id)
    res = pooled_estimates(scenario_id, reviewed_only, verified_only, split_disease, min_studies, full=True)
    return Response(content=pooled_csv(res).encode("utf-8-sig"), media_type="text/csv; charset=utf-8", headers={
        "Content-Disposition": f'attachment; filename="pooled_{re.sub(r"[^A-Za-z0-9_-]", "_", scenario_id)}.csv"',
        "X-Labels": str(len(res["pooled"])), "X-Comparisons": str(len(res["comparisons"]))})
