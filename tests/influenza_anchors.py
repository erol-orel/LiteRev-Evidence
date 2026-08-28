"""Published influenza parameter values — the yardstick for judging a real LLM extraction.

Nothing downstream of the LLM has ever been checked against reality. `epidemic_parameters`
comes out of a generation step no test can run (it needs an API key), and every SEIR curve
the app draws rests on it. This module pins what the literature actually says, so that an
extraction can be JUDGED rather than trusted.

The bands are deliberately WIDE. Their job is to catch the errors that make a projection
meaningless — a case fatality ratio given in percent (100x out), a period given in hours,
a hallucinated magnitude, a provenance id naming no article in the corpus — not to referee
between studies. A value inside a band is plausible, not verified; a value outside one is
very probably wrong.

Sources, all systematic reviews, which is what the extraction prompt tells the model to
prefer (main.py, "PRIVILÉGIE les estimations des synthèses de meilleure qualité"):

  R0 ................. Biggerstaff et al., BMC Infect Dis 2014;14:480. Seasonal median
                       1.28 (IQR 1.19-1.37); 2009 pandemic 1.46 (1.30-1.70); 1918 ~1.80.
  Incubation ......... Lessler et al., Lancet Infect Dis 2009;9:291. Influenza A median
                       1.4 d (95% CI 1.3-1.5); influenza B ~0.6 d.
  Serial interval .... Vink et al., Am J Epidemiol 2014;180:865. Influenza around 2.8 d.
  Infectious period .. Carrat et al., Am J Epidemiol 2008;167:775. Volunteer challenge
                       studies; viral shedding averaging roughly 4.8 d.
  CFR ................ 2009 H1N1 symptomatic case fatality is of the order of 0.02 %
                       (Wong et al., Lancet Infect Dis 2013). This band spans several
                       orders of magnitude ON PURPOSE: the anchor exists to detect the
                       percent/proportion confusion, not to settle a value.

The point estimates below are reference marks recorded from those reviews; only the BANDS
are asserted anywhere. Treat a `typical` as "what a faithful extraction tends to land near",
never as ground truth to reproduce.

To judge a REAL extraction, from the repository root:

    curl -sS localhost:8000/scenarios/<id>/variables > variables.json
    python -c "
    import json, sys; sys.path.insert(0, 'tests')
    from influenza_anchors import check_extraction
    spec = json.load(open('variables.json'))
    ids = {int(k) for k in (spec.get('_provenance_index') or {})}
    for c in check_extraction(spec['epidemic_parameters'], corpus_ids=ids) or ['looks plausible']:
        print(c)"

Note WHICH block to hand it. `variables['epidemic_parameters']` is the raw model output and
still carries `observations`, so every check applies. `variables['model_spec']
['epidemic_parameters']` has already been through `normalize_extracted_parameters`, which
consumes the observations into a pooled estimate — the checker detects that shape and says
which checks it could not run.
"""
from __future__ import annotations

from dataclasses import dataclass

# The checker must agree with the code on what "percent" means, rather than re-guessing
# it: `_is_percent_unit` is the function that actually decides whether a CFR gets divided
# by 100, so a second opinion here would only diverge.
import seir_model as sm

# ── the anchors ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Anchor:
    """A plausibility band for one extracted parameter, with the review behind it."""
    low: float
    high: float
    typical: float
    unit: str
    source: str
    #: multipliers that would bring a WRONG unit back into the band, and their names.
    #: Used to turn "1.8 is out of band" into "1.8 looks like a percentage".
    confusions: tuple[tuple[float, str], ...] = ()

    def holds(self, value) -> bool:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return False
        return self.low <= v <= self.high


_PERCENT = ((0.01, "a percentage (value/100 would be in range)"),)
_TIME = ((1 / 24.0, "hours (value/24 would be in range)"),
         (7.0, "weeks (value*7 would be in range)"))

ANCHORS: dict[str, Anchor] = {
    "r0": Anchor(
        low=1.0, high=3.0, typical=1.4, unit="ratio",
        source="Biggerstaff 2014: seasonal 1.28 (IQR 1.19-1.37), 2009 pandemic 1.46, 1918 ~1.80",
    ),
    "incubation_period_days": Anchor(
        low=0.5, high=4.0, typical=1.4, unit="days",
        source="Lessler 2009: influenza A median 1.4 d (95% CI 1.3-1.5)",
        confusions=_TIME,
    ),
    "infectious_period_days": Anchor(
        low=2.0, high=8.0, typical=4.8, unit="days",
        source="Carrat 2008: viral shedding averaging ~4.8 d in challenge studies",
        confusions=_TIME,
    ),
    "serial_interval_days": Anchor(
        low=1.5, high=5.0, typical=2.8, unit="days",
        source="Vink 2014: influenza serial interval around 2.8 d",
        confusions=_TIME,
    ),
    "cfr": Anchor(
        # Upper bound set at 3 % so that 1918-scale estimates stay inside it; anything
        # above that is not influenza, it is a unit error.
        low=1e-5, high=3e-2, typical=2e-4, unit="proportion",
        source="2009 H1N1 symptomatic case fatality of the order of 0.02 % (Wong 2013); "
               "1918-scale estimates reach a few percent",
        confusions=_PERCENT,
    ),
    "immunity_duration_days": Anchor(
        low=180.0, high=3650.0, typical=365.0, unit="days",
        source="strain-specific immunity eroded by antigenic drift within a season or few",
        confusions=((365.0, "years (value*365 would be in range)"),),
    ),
}

#: R0 or beta is the ONLY parameter without which a projection cannot be served —
#: `_seir_projection_payload` gate 2 (main.py) refuses rather than fall back to 2.5.
REQUIRED_FOR_PROJECTION = "r0"

#: Extracted, normalised and displayed, but absent from `seir_model._PARAM_FIELDS`, so it
#: never reaches `simulate`. Recorded here so nobody reads the serial interval off the SEIR
#: tab believing it drove the curve.
NOT_SIMULATED = ("serial_interval_days",)


# ── a realistic extraction, shaped exactly as the prompt asks the model to return it ──

#: Stand-ins for the six papers the influenza query is expected to surface.
CORPUS_IDS = frozenset({101, 102, 103, 104, 105, 106})

#: `_compute_quality_score` values of the same order these designs earn: systematic reviews
#: near the top of the evidence pyramid, a single-season cohort mid-table, a small outbreak
#: investigation low. These ARE the pooling weights (`seir_model.pool_weighted`).
QUALITY_BY_ID = {
    101: 0.88,   # Biggerstaff 2014 — systematic review, heavily cited
    102: 0.86,   # Lessler 2009     — systematic review
    103: 0.84,   # Vink 2014        — systematic review
    104: 0.80,   # Carrat 2008      — review of volunteer challenge studies
    105: 0.55,   # single-season household cohort
    106: 0.42,   # small outbreak investigation
}

#: What a faithful extraction of that corpus looks like: per-study `observations` (which is
#: what `pool_weighted` re-pools by quality), real provenance ids, CFR as a proportion.
#: `immunity_duration_days` deliberately carries ONE observation, so it exercises the other
#: path — under two studies the pool is refused and the model's own estimate is kept.
INFLUENZA_EXTRACTION: dict = {
    "applicable": True,
    "population_disease": "Influenza (seasonal and pandemic)",
    "r0": {
        "value": 1.46, "ci_low": 1.19, "ci_high": 2.27, "unit": "ratio", "n_studies": 3,
        "provenance": [101, 105, 106],
        "observations": [
            {"article_id": 101, "value": 1.28},   # seasonal, same review …
            {"article_id": 101, "value": 1.46},   # … and its 2009 pandemic estimate
            {"article_id": 105, "value": 1.60},
            {"article_id": 106, "value": 2.10},
        ],
    },
    "incubation_period_days": {
        "value": 1.4, "ci_low": 1.3, "ci_high": 1.5, "unit": "days", "n_studies": 2,
        "provenance": [102, 105],
        "observations": [{"article_id": 102, "value": 1.4},
                         {"article_id": 105, "value": 1.9}],
    },
    "infectious_period_days": {
        "value": 4.8, "ci_low": 3.2, "ci_high": 6.1, "unit": "days", "n_studies": 2,
        "provenance": [104, 105],
        "observations": [{"article_id": 104, "value": 4.8},
                         {"article_id": 105, "value": 3.8}],
    },
    "serial_interval_days": {
        "value": 2.8, "ci_low": 2.2, "ci_high": 3.6, "unit": "days", "n_studies": 2,
        "provenance": [103, 105],
        "observations": [{"article_id": 103, "value": 2.8},
                         {"article_id": 105, "value": 3.6}],
    },
    "cfr": {
        "value": 0.0002, "ci_low": 0.0001, "ci_high": 0.0009, "unit": "proportion",
        "n_studies": 2, "provenance": [105, 106],
        "observations": [{"article_id": 105, "value": 0.0002},
                         {"article_id": 106, "value": 0.0009}],
    },
    "immunity_duration_days": {
        "value": 365.0, "ci_low": 180.0, "ci_high": 730.0, "unit": "days", "n_studies": 1,
        "provenance": [105],
        "observations": [{"article_id": 105, "value": 365.0}],
    },
}


# ── the checker ──────────────────────────────────────────────────────────────

def _num(v):
    """float fini, ou None — même tolérance que `seir_model._num_or_none`."""
    try:
        if v is None or v == "":
            return None
        f = float(v)
        return f if f == f and f not in (float("inf"), float("-inf")) else None
    except (TypeError, ValueError):
        return None


def _diagnose(anchor: Anchor, value: float) -> str:
    """Name the likely unit confusion behind an out-of-band value, when one fits."""
    for factor, label in anchor.confusions:
        if anchor.low <= value * factor <= anchor.high:
            return f" — looks like {label}"
    return ""


def check_extraction(epi, corpus_ids=None) -> list[str]:
    """Judge an `epidemic_parameters` block against the anchors. Returns complaints.

    An empty list means "nothing implausible found" — which is not the same as correct.
    Accepts either shape: the RAW model output (flat, with `observations`) or the block
    already through `normalize_extracted_parameters` (`{applicable, disease, params}`).
    On the normalised shape the observation-level checks cannot run, and the return value
    says so rather than passing silently.

    `corpus_ids` — the article ids actually in the scenario's corpus. Provenance is only
    checked for fabrication when this is supplied; without it a made-up id is invisible
    here, exactly as it is to `_clean_provenance`, which drops it without complaint.
    """
    out: list[str] = []
    if not isinstance(epi, dict):
        return [f"epidemic_parameters is {type(epi).__name__}, not an object"]

    normalised = isinstance(epi.get("params"), dict)
    blocks = epi["params"] if normalised else epi
    if normalised:
        out.append("NOTE: normalised block — per-study `observations` were already pooled "
                   "away, so they cannot be checked here. Re-run on "
                   "variables['epidemic_parameters'] for the full check.")

    if not epi.get("applicable"):
        out.append("applicable is false: influenza IS transmissible, so either the model "
                   "misjudged the scenario or no parameter survived normalisation — "
                   "either way the SEIR tab is gated off (main.py gate 1)")

    ids = set(corpus_ids) if corpus_ids is not None else None

    for name, anchor in ANCHORS.items():
        blk = blocks.get(name)
        if not isinstance(blk, dict):
            if name == REQUIRED_FOR_PROJECTION:
                out.append(f"{name}: absent — without it no projection is served at all "
                           "(gate 2 refuses rather than assume 2.5)")
            else:
                out.append(f"{name}: absent (acceptable if no article reports it)")
            continue

        value = _num(blk.get("value"))
        if value is None:
            out.append(f"{name}: value is null or not a number "
                       f"({blk.get('value')!r}) — correct if unreported, since the prompt "
                       "asks for null over invention")
            continue

        # A CFR whose unit SAYS percent is converted by `normalize_extracted_parameters`,
        # so judge the converted value — otherwise a correct extraction reads as out of
        # band. The dangerous case is the opposite: percent MAGNITUDE, proportion LABEL.
        unit = str(blk.get("unit") or "")
        declared_percent = name == "cfr" and sm._is_percent_unit(unit)
        banded = value / 100.0 if declared_percent else value
        if (name == "cfr" and not declared_percent and value > anchor.high
                and anchor.low <= value / 100.0 <= anchor.high):
            out.append(f"cfr = {value} declared as {unit!r}: this is a PERCENTAGE wearing "
                       "the label of a proportion. `_is_percent_unit` reads the unit, not "
                       "the magnitude, so nothing converts it — deaths come out 100x high")
        elif not anchor.holds(banded):
            out.append(f"{name} = {value} is outside the plausible band "
                       f"[{anchor.low}, {anchor.high}] {anchor.unit}"
                       f"{_diagnose(anchor, banded)}. Literature: {anchor.source}")

        lo, hi = _num(blk.get("ci_low")), _num(blk.get("ci_high"))
        if lo is not None and hi is not None:
            if lo > hi:
                out.append(f"{name}: ci_low {lo} > ci_high {hi} — normalisation drops both "
                           "rather than invent an interval, so the parameter loses its "
                           "uncertainty band silently")
            elif not (lo <= value <= hi):
                out.append(f"{name}: value {value} sits outside its own interval [{lo}, {hi}]")

        prov = blk.get("provenance")
        if not isinstance(prov, list) or not prov:
            out.append(f"{name}: no provenance — the value is unattributable, and the UI "
                       "will show a study count with nothing to click")
        elif ids is not None:
            bogus = [p for p in prov if _num(p) is None or int(_num(p)) not in ids]
            if bogus:
                out.append(f"{name}: provenance {bogus} names no article in the corpus "
                           "(fabricated). `_clean_provenance` drops these, so the symptom "
                           "is a parameter quietly losing its citations, not an error")

        if not normalised:
            obs = blk.get("observations")
            if not isinstance(obs, list) or not obs:
                out.append(f"{name}: `observations` is empty — quality-weighted pooling has "
                           "nothing to re-pool, so the model's own averaging is what you get")
            else:
                for o in obs:
                    if not isinstance(o, dict) or _num(o.get("value")) is None:
                        out.append(f"{name}: malformed observation {o!r}")
                    elif ids is not None and (_num(o.get("article_id")) is None
                                              or int(_num(o["article_id"])) not in ids):
                        out.append(f"{name}: observation attributed to article "
                                   f"{o.get('article_id')!r}, which is not in the corpus")

    return out
