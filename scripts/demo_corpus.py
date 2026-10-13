#!/usr/bin/env python3
"""The demo corpus: a synthetic, deterministic literature corpus for the demo video.

The demo video (scripts/demo_video.py) records the real interface on a throwaway
database. Real articles can neither ship with the repository nor be fetched offline, so
this module writes a SYNTHETIC corpus into the built-in demo scenario
(`demo_seed.DEMO_SCENARIO_ID`, whose model is trained at startup on the real EAWAG
wastewater series): about 200 records on influenza, wastewater and weather signals, each
carrying what a finished search and map step leave on an article row (relevance score,
study design, PICO, main finding, typed concepts), the identification figures of the
search that would have found them, and a few screening decisions made through the API.

What the app writes with an LLM is never faked here: no evidence brief, no assistant
answer, no cluster summary, no Boolean translation. The records are fiction and say so:
their journal is "Synthetic demo record", they carry no authors, no DOI and no PMID, and
the video marks every frame as demo data.

Usage, against a running API on a database whose data does not matter:

  python3 scripts/demo_corpus.py --db-url postgresql+psycopg://... \\
      --api http://127.0.0.1:8000 --api-key "$WRITE_API_KEY"

`build_corpus` is pure (no database, no network), so the tests pin it directly.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: The journal every record carries: the corpus says what it is wherever a reader looks.
JOURNAL = "Synthetic demo record"
#: Prefix of `external_id`, and the key by which `remove` finds the records again.
EXTERNAL_PREFIX = "literev-demo-"
#: Default size: big enough for the charts and the clusters to mean something, small
#: enough to seed in a few seconds.
DEFAULT_ARTICLES = 200
DEFAULT_SEED = 2026
#: The two reviewer codes of the double-blind screening.
REVIEWERS = ("R-1024", "R-2048")
#: The last publication year of the corpus (only partly published: it weighs less).
LAST_YEAR = 2026
#: The highest relevance an off-topic record gets: under the default threshold (0.45).
OFF_TOPIC_MAX_SCORE = 0.42


# ─────────────────────────────────────────────────────────────────────────────
# Vocabulary
# ─────────────────────────────────────────────────────────────────────────────
# (place as written in a title, ISO 3166-1 alpha-2 code, weight). Switzerland first: the
# demo scenario is about Switzerland, and a search on it finds mostly Swiss studies with
# their European neighbours.
PLACES = (
    ("Switzerland", "CH", 10), ("Geneva", "CH", 6), ("the canton of Vaud", "CH", 4),
    ("Zurich", "CH", 5), ("Ticino", "CH", 2), ("Bern", "CH", 3), ("Basel", "CH", 3),
    ("western Switzerland", "CH", 3), ("Lausanne", "CH", 2), ("Lucerne", "CH", 1),
    ("France", "FR", 2), ("Lyon", "FR", 1), ("northern Italy", "IT", 2), ("Lombardy", "IT", 1),
    ("Austria", "AT", 2), ("Bavaria", "DE", 1), ("Germany", "DE", 2),
    ("the Netherlands", "NL", 2), ("Denmark", "DK", 1), ("Catalonia", "ES", 1),
    ("Ontario", "CA", 1), ("New York City", "US", 1), ("Japan", "JP", 1),
    ("Victoria, Australia", "AU", 1),
)

# (as written in a sentence, concept {en, fr}).
OUTCOMES = (
    ("influenza-like illness consultations", ("influenza-like illness", "syndrome grippal")),
    ("laboratory-confirmed influenza cases", ("laboratory-confirmed influenza", "grippe confirmée en laboratoire")),
    ("influenza hospital admissions", ("hospital admission", "hospitalisation")),
    ("emergency department visits for influenza-like illness", ("emergency department visit", "passage aux urgences")),
    ("sentinel consultations for influenza-like illness", ("influenza-like illness", "syndrome grippal")),
    ("intensive care admissions for severe influenza", ("intensive care admission", "admission en soins intensifs")),
)

PATHOGENS = {
    "flu_a": ("influenza A virus", "virus de la grippe A"),
    "flu_b": ("influenza B virus", "virus de la grippe B"),
    "rsv": ("respiratory syncytial virus", "virus respiratoire syncytial"),
    "sars2": ("SARS-CoV-2", "SARS-CoV-2"),
}

#: Northern-hemisphere influenza seasons. 2020/21 is missing on purpose: the measures
#: against COVID-19 stopped influenza for a season.
SEASONS = tuple(f"{y}/{(y + 1) % 100:02d}" for y in range(2008, 2026) if y != 2020)

# (source, weight, first year it can carry): the mix of a federated search.
SOURCES = (
    ("pubmed", 34, 0), ("openalex", 26, 0), ("europepmc", 18, 0), ("crossref", 8, 0),
    ("semantic_scholar", 5, 0), ("preprint", 6, 2019), ("doaj", 3, 0), ("openaire", 2, 0),
    ("core", 2, 0),
)

# Exclusion reasons, as the screening panel offers them.
EXCLUSION_REASONS = ("wrong-population", "wrong-intervention", "wrong-outcome")


@dataclass
class Record:
    """One synthetic article, with the facts the map step would have cached on its row."""
    key: int
    theme: str
    source: str
    year: int
    title: str
    abstract: str
    design: str
    pico: dict
    concepts: list = field(default_factory=list)
    country: str | None = None
    sample_size: int | None = None
    keywords: str = ""
    score: float = 0.0
    has_fulltext: bool = False
    open_access: bool = False
    citation_count: int = 0
    found_in: tuple = ()

    @property
    def external_id(self) -> str:
        return f"{EXTERNAL_PREFIX}{self.key:04d}"


class _Draw:
    """`random.Random` with the few draws the templates need. Deterministic: the same
    seed gives the same corpus on every machine and every Python 3 version."""

    def __init__(self, seed: int):
        self.r = random.Random(seed)

    def pick(self, items):
        return items[self.r.randrange(len(items))]

    def weighted(self, pairs):
        total = sum(w for _, w in pairs)
        x = self.r.uniform(0, total)
        for item, w in pairs:
            x -= w
            if x <= 0:
                return item
        return pairs[-1][0]

    def place(self):
        label, iso, _ = self.weighted([((p[0], p[1], 0), p[2]) for p in PLACES])
        return label, iso

    def between(self, lo: float, hi: float, digits: int = 2) -> float:
        return round(self.r.uniform(lo, hi), digits)

    def period(self, published: int, span=(2, 6)):
        """The study period of an article published in `published`: it ends one or two
        years before publication."""
        y2 = published - self.r.randint(1, 2)
        return y2 - self.r.randint(*span), y2

    def year(self, first: int) -> int:
        """A publication year, weighted towards recent years like any live corpus. The
        current year is only partly published, so it weighs less than the one before."""
        years = list(range(first, LAST_YEAR + 1))
        return self.weighted([(y, (1 + (y - first) * 0.7) * (0.55 if y == LAST_YEAR else 1))
                              for y in years])


def _season_before(d: _Draw, year: int) -> str:
    """A season that ended one to four years before the article was published."""
    recent = [s for s in SEASONS if year - 5 <= int(s[:4]) <= year - 2] or [SEASONS[0]]
    return d.pick(tuple(recent))


def _c(t: str, label: tuple) -> dict:
    return {"t": t, "en": label[0], "fr": label[1]}


def _pico(p, i, c, o, design, finding, direction) -> dict:
    """A PICO as the extraction writes it (`pico_confidence` is drawn by build_corpus)."""
    return {"P": p, "I": i, "C": c, "O": o, "study_design": design,
            "key_finding": finding, "finding_direction": direction,
            "pico_notes": "", "pico_source": "abstract"}


def _abstract(background, methods, results, conclusions) -> str:
    return (f"Background: {background} Methods: {methods} "
            f"Results: {results} Conclusions: {conclusions}")


# ─────────────────────────────────────────────────────────────────────────────
# Themes: each returns the article fields for one draw
# ─────────────────────────────────────────────────────────────────────────────
def _wastewater(d: _Draw, year: int) -> dict:
    place, iso = d.place()
    oc, oc_c = d.pick(OUTCOMES)
    n = d.r.randint(3, 28)
    pop = d.between(0.15, 2.4)
    y1, y2 = d.period(year)
    y1 = max(y1, 2012)
    rho = d.between(0.61, 0.92)
    lead = d.r.randint(4, 17)
    m = min(d.r.randint(3, 7), y2 - y1 + 1)
    k = m - d.pick((0, 0, 1))
    design = d.weighted([("Ecological time-series study", 5), ("Surveillance study", 4),
                         ("Retrospective cohort study", 1), ("Modelling study", 2),
                         ("Systematic review of observational studies", 1),
                         ("Narrative review", 1)])
    if year < 2022 and "review" in design.lower():      # reviews of the field came after COVID-19
        design = "Ecological time-series study"
    concepts = [_c("pathogen", PATHOGENS["flu_a"]),
                _c("method", ("wastewater-based epidemiology", "épidémiologie des eaux usées")),
                _c("exposure", ("viral RNA in wastewater", "ARN viral dans les eaux usées")),
                _c("outcome", oc_c), _c("population", ("general population", "population générale"))]
    if design.startswith("Systematic"):
        k_s, k_l = d.r.randint(14, 41), d.r.randint(6, 13)
        return dict(
            title=d.pick((f"Wastewater-based surveillance of influenza: a systematic review of {k_s} studies",
                          "Does sewage anticipate influenza epidemics? A systematic review of lead times")),
            abstract=_abstract(
                "Wastewater surveillance spread during the COVID-19 pandemic and is now applied to influenza.",
                f"We searched MEDLINE, Embase and Web of Science to {year - 1} for studies comparing influenza RNA "
                "in wastewater with clinical surveillance, and extracted correlations and lead times.",
                f"{k_s} observational studies from {d.r.randint(6, 15)} countries were included; {k_l} reported "
                f"a lead of wastewater over clinical data, with a median of {lead} days. Sampling and "
                "normalisation methods varied widely.",
                "Wastewater signals usually precede clinical influenza data, but methods need harmonising "
                "before they can trigger alerts."),
            design=design, country=None, sample_size=k_s, concepts=concepts[:4],
            pico=_pico("Studies of community wastewater surveillance of influenza",
                       "Influenza RNA concentration in wastewater", "Clinical surveillance data",
                       "Correlation with and lead time over clinical incidence", design,
                       f"{k_l} of {k_s} studies found wastewater signals ahead of clinical data, by a median of {lead} days.",
                       "increase"))
    if design == "Narrative review":
        return dict(
            title=d.pick(("Sewage as a sentinel for respiratory viruses: a narrative review",
                          "From SARS-CoV-2 to influenza: what wastewater surveillance has taught us")),
            abstract=_abstract(
                "Wastewater surveillance moved from polio eradication to SARS-CoV-2 and now to influenza.",
                "We review sampling strategies, laboratory methods, normalisation and the integration of "
                "wastewater data into public health decisions.",
                "The main obstacles are shedding variability, dilution by rainfall and the absence of agreed "
                "alert thresholds.",
                "Wastewater data are most useful combined with clinical surveillance, not instead of it."),
            design=design, country=None, sample_size=None, concepts=concepts[:3],
            pico=_pico("Community wastewater surveillance programmes", "Wastewater monitoring of respiratory viruses",
                       "Not specified", "Usefulness for public health decisions", design, "", "none"))
    if design == "Modelling study":
        err = d.r.randint(8, 24)
        return dict(
            title=d.pick((f"A shedding model linking influenza incidence to wastewater loads in {place}",
                          f"Back-calculating influenza incidence from wastewater in {place}: a modelling study")),
            abstract=_abstract(
                "Wastewater loads reflect infections, but converting them into incidence requires a shedding model.",
                f"We built a model of faecal shedding, in-sewer decay and dilution, calibrated on {n} treatment "
                f"plants in {place} ({y1}-{y2}) against {oc}.",
                f"Back-calculated incidence reproduced the timing of {k} of {m} epidemic peaks, with a "
                f"{err}% error on peak height.",
                "A calibrated shedding model turns wastewater loads into incidence estimates usable for planning."),
            design=design, country=iso, sample_size=n, concepts=concepts,
            pico=_pico(f"Catchments of {n} wastewater treatment plants in {place}",
                       "Influenza RNA loads with a shedding and dilution model", f"Observed {oc}",
                       "Accuracy of back-calculated incidence", design,
                       f"The model reproduced {k} of {m} epidemic peaks with a {err}% error on peak height.",
                       "descriptive"))
    titles = [f"Influenza A RNA in wastewater as an early warning signal for {oc} in {place}",
              f"Wastewater monitoring of influenza A and B at {n} treatment plants in {place}, {y1}-{y2}",
              f"How far ahead does sewage see influenza? Lead times over sentinel surveillance in {place}",
              f"Normalising influenza A loads in wastewater by faecal markers: evidence from {place}",
              f"Detecting the onset of the {_season_before(d, year)} influenza season from wastewater in {place}"]
    if y1 >= 2020 and m >= 2:
        titles.append(f"Respiratory viruses in the wastewater of {place}: influenza, RSV and SARS-CoV-2 over {m} seasons")
    return dict(
        title=d.pick(tuple(titles)),
        abstract=_abstract(
            "Clinical surveillance of influenza depends on care seeking and testing, and reports epidemic "
            "activity with a delay.",
            f"We quantified influenza A and B RNA by RT-qPCR in composite influent samples from {n} wastewater "
            f"treatment plants serving about {pop} million residents of {place} between {y1} and {y2}, and "
            f"compared loads normalised by flow and population with {oc}.",
            f"Normalised influenza A loads correlated with {oc} (Spearman rho {rho}) and crossed the epidemic "
            f"threshold {lead} days earlier on average, in {k} of {m} seasons.",
            "Wastewater monitoring gave an earlier, care-independent signal of influenza activity and can feed "
            "an early warning system."),
        design=design, country=iso, sample_size=int(pop * 1_000_000), concepts=concepts,
        pico=_pico(f"Residents served by {n} wastewater treatment plants in {place} (about {pop} million people)",
                   "Influenza A and B RNA loads in influent wastewater", f"Clinical surveillance ({oc})",
                   f"Correlation with and lead time over {oc}", design,
                   f"Wastewater influenza A loads correlated with {oc} (rho {rho}) and led them by {lead} days on average.",
                   "increase"))


_WEATHER = (
    ("absolute humidity", ("absolute humidity", "humidité absolue"), "per 1 g/m3 decrease", "decrease"),
    ("ambient temperature", ("ambient temperature", "température ambiante"), "per 5 degree C drop", "decrease"),
    ("ultraviolet radiation", ("ultraviolet radiation", "rayonnement ultraviolet"), "per interquartile drop", "decrease"),
    ("temperature variability", ("temperature variability", "variabilité thermique"), "per 1 degree C of daily range", "increase"),
    ("cold spells", ("cold spell", "vague de froid"), "during cold spells", "increase"),
    ("school holidays", ("school holiday", "vacances scolaires"), "during school holidays", "decrease"),
)


def _weather(d: _Draw, year: int) -> dict:
    place, iso = d.place()
    oc, oc_c = d.pick(OUTCOMES[:5])
    exp, exp_c, unit, effect = d.pick(_WEATHER)
    y1, y2 = d.period(year, span=(5, 12))
    rr = d.between(1.06, 1.38)
    lag = d.r.randint(3, 14)
    cases = d.r.randint(1800, 64000)
    design = d.weighted([("Ecological time-series study", 6),
                         ("Case-control study (time-stratified case-crossover design)", 2),
                         ("Modelling study", 2), ("Systematic review and meta-analysis of observational studies", 1),
                         ("Retrospective cohort study", 1), ("Not stated", 1)])
    rise = "rose" if effect == "increase" else "fell"
    concepts = [_c("pathogen", PATHOGENS["flu_a"]), _c("exposure", exp_c), _c("outcome", oc_c),
                _c("method", ("time-series analysis", "analyse de séries temporelles"))]
    if design.startswith("Systematic"):
        k = d.r.randint(12, 38)
        return dict(
            title=f"{exp.capitalize()} and influenza transmission in temperate climates: a systematic review and meta-analysis",
            abstract=_abstract(
                f"Laboratory and epidemiological studies disagree on the role of {exp} in influenza transmission.",
                f"We searched four databases to {year - 1} for observational studies relating {exp} to influenza "
                "incidence in temperate regions, and pooled relative risks with random effects.",
                f"{k} studies were included. The pooled relative risk was {rr} {unit} (I2 {d.r.randint(41, 88)}%), "
                f"with effects strongest at lags of {lag} days.",
                f"{exp.capitalize()} is a consistent but modest driver of influenza activity."),
            design=design, country=None, sample_size=k, concepts=concepts[:3],
            pico=_pico("Populations of temperate regions", exp.capitalize(), "Reference exposure levels",
                       "Influenza incidence", design,
                       f"Pooled relative risk of influenza {rr} {unit} across {k} studies.", "increase"))
    if design == "Modelling study":
        return dict(
            title=d.pick((f"Does {exp} explain the timing of influenza epidemics in {place}? A transmission model",
                          f"A climate-forced transmission model of seasonal influenza in {place}")),
            abstract=_abstract(
                f"The onset of influenza epidemics varies from year to year, and {exp} is a candidate driver.",
                f"We fitted an SIRS model forced by {exp} to weekly {oc} in {place} from {y1} to {y2}.",
                f"Forcing by {exp} improved the fit over a model with seasonality alone and predicted the onset week "
                f"within one week in {d.r.randint(5, 9)} of 10 seasons.",
                f"{exp.capitalize()} helps anticipate when the epidemic starts."),
            design=design, country=iso, sample_size=cases, concepts=concepts,
            pico=_pico(f"Population of {place}", f"Transmission forced by {exp}", "Seasonality alone",
                       "Timing of epidemic onset", design,
                       f"Forcing by {exp} predicted the onset week within one week in most seasons.", "descriptive"))
    return dict(
        title=d.pick((f"{exp.capitalize()} and the timing of seasonal influenza epidemics in {place}",
                      f"Weather and {oc}: a time-series analysis in {place}, {y1}-{y2}",
                      f"Short-term effects of {exp} on {oc} in {place}",
                      f"{exp.capitalize()}, indoor crowding and {oc} in {place}")),
        abstract=_abstract(
            f"Influenza transmission is seasonal in temperate climates, and {exp} has been proposed as a driver.",
            f"We related daily {exp} to {cases:,} {oc} recorded in {place} between {y1} and {y2}, with "
            "distributed lag non-linear models adjusted for holidays and long-term trends.",
            f"The risk of influenza {rise} {unit} (relative risk {rr}), with the strongest effect at a lag of "
            f"{lag} days.",
            f"Short-term changes in {exp} anticipate influenza activity by one to two weeks and can inform "
            "early warning."),
        design=design, country=iso, sample_size=cases, concepts=concepts,
        pico=_pico(f"{cases:,} {oc} in {place}", exp.capitalize(), "Reference exposure levels",
                   oc.capitalize(), design,
                   f"The risk of influenza {rise} {unit} (relative risk {rr}), peaking at a lag of {lag} days.",
                   effect))


def _cocirculation(d: _Draw, year: int) -> dict:
    place, iso = d.place()
    y1, y2 = d.period(year)
    after_covid = y2 >= 2021                            # SARS-CoV-2 exists in the data
    other = d.pick(("rsv", "sars2", "flu_b") if after_covid else ("rsv", "flu_b"))
    other_label = {"flu_b": "influenza B"}.get(other, PATHOGENS[other][0])
    tests = d.r.randint(4000, 90000)
    shift = d.r.randint(3, 11)
    designs = [("Surveillance study", 4), ("Retrospective cohort study", 2), ("Modelling study", 2),
               ("Cross-sectional study", 1)]
    if after_covid:
        designs.append(("Interrupted time series analysis", 2))
    design = d.weighted(designs)
    concepts = [_c("pathogen", PATHOGENS["flu_a"]), _c("pathogen", PATHOGENS[other]),
                _c("outcome", ("seasonal timing", "calendrier saisonnier")),
                _c("method", ("multiplex PCR surveillance", "surveillance par PCR multiplex"))]
    if design == "Interrupted time series analysis":
        y1 = min(y1, 2017)
        return dict(
            title=f"Influenza seasonality before and after the COVID-19 non-pharmaceutical interventions in {place}",
            abstract=_abstract(
                "Non-pharmaceutical interventions against COVID-19 interrupted the circulation of influenza.",
                f"We analysed {tests:,} respiratory samples tested in {place} from {y1} to {y2} as an interrupted "
                "time series around March 2020.",
                f"Influenza disappeared for a season, then returned {shift} weeks earlier than the pre-pandemic "
                "average, with a steeper rise.",
                "Seasonal patterns learnt before 2020 should not be assumed to hold since."),
            design=design, country=iso, sample_size=tests, concepts=concepts[:1] + concepts[2:],
            pico=_pico(f"{tests:,} respiratory samples tested in {place}",
                       "Non-pharmaceutical interventions against COVID-19", "Pre-pandemic seasons",
                       "Timing and shape of influenza seasons", design,
                       f"Influenza returned {shift} weeks earlier than before the pandemic, with a steeper rise.",
                       "mixed"))
    if design == "Modelling study":
        return dict(
            title=f"Viral interference between influenza A and {other_label}: a two-pathogen model fitted to data from {place}",
            abstract=_abstract(
                f"Influenza A and {other_label} rarely peak together, which suggests interference between them.",
                f"We fitted a two-pathogen transmission model to weekly detections in {place} from {y1} to {y2}.",
                f"A short period of cross-protection after infection reproduced the observed offset of {shift} weeks "
                "between the two peaks.",
                f"Forecasts of influenza should account for the circulation of {other_label}."),
            design=design, country=iso, sample_size=tests, concepts=concepts,
            pico=_pico(f"Population of {place}", f"Co-circulation of {other_label}", "Independent circulation",
                       "Offset between epidemic peaks", design,
                       f"Cross-protection reproduced the {shift}-week offset between the influenza A and {other_label} peaks.",
                       "descriptive"))
    titles = [f"Do influenza A and {other_label} compete? Evidence from surveillance in {place}, {y1}-{y2}",
              f"Respiratory virus co-detection among {tests:,} tested patients in {place}"]
    if after_covid and year >= 2023:
        titles.append(f"Co-circulation of influenza, RSV and SARS-CoV-2 in {place} after the COVID-19 pandemic")
    return dict(
        title=d.pick(tuple(titles)),
        abstract=_abstract(
            f"Influenza A and {other_label} circulate in the same months, and their overlap varies by season.",
            f"We analysed {tests:,} multiplex PCR results from sentinel and hospital laboratories in {place} "
            f"between {y1} and {y2}.",
            f"Influenza peaks were shifted by {shift} weeks when {other_label} circulated early, and co-detections "
            f"were rarer than expected (odds ratio {d.between(0.31, 0.78)}).",
            f"The circulation of {other_label} is an informative covariate for influenza forecasting."),
        design=design, country=iso, sample_size=tests, concepts=concepts,
        pico=_pico(f"{tests:,} patients tested for respiratory viruses in {place}",
                   f"Early circulation of {other_label}", "Seasons without early circulation",
                   "Timing of the influenza peak", design,
                   f"Influenza peaks shifted by {shift} weeks when {other_label} circulated early.", "mixed"))


_METHODS = (
    ("random forests", ("random forest", "forêt aléatoire")),
    ("gradient boosting", ("gradient boosting", "gradient boosting")),
    ("SARIMA models", ("SARIMA model", "modèle SARIMA")),
    ("a Bayesian hierarchical model", ("Bayesian hierarchical model", "modèle hiérarchique bayésien")),
    ("an ensemble of mechanistic and statistical models", ("ensemble forecasting", "prévision d'ensemble")),
    ("long short-term memory networks", ("LSTM network", "réseau LSTM")),
)


def _forecasting(d: _Draw, year: int) -> dict:
    place, iso = d.place()
    oc, oc_c = d.pick(OUTCOMES)
    meth, meth_c = d.pick(_METHODS)
    y1, y2 = d.period(year, span=(4, 9))
    gain = d.r.randint(9, 37)
    auc = d.between(0.78, 0.94)
    horizon = d.pick((1, 2, 3, 4))
    streams = d.pick(("wastewater and weather data", "wastewater, weather and sentinel data",
                      "emergency calls and wastewater data", "sentinel and weather data")
                     if year >= 2021 else ("internet search and weather data", "sentinel and weather data",
                                           "emergency calls and weather data"))
    design = d.weighted([("Modelling study", 6), ("Modelling study with retrospective validation", 2),
                         ("Systematic review of forecasting models", 1), ("Prospective observational study", 1)])
    concepts = [_c("pathogen", PATHOGENS["flu_a"]), _c("method", meth_c),
                _c("method", ("forecasting", "prévision")), _c("outcome", oc_c)]
    if design.startswith("Systematic"):
        k = d.r.randint(18, 64)
        return dict(
            title=f"Forecasting seasonal influenza: a systematic review of {k} models and their evaluation",
            abstract=_abstract(
                "Influenza forecasting has grown fast, with heterogeneous methods and metrics.",
                f"We searched five databases to {year - 1} for studies forecasting influenza activity and assessed how "
                "they were evaluated.",
                f"{k} models were included; {d.r.randint(30, 60)}% were evaluated out of sample and few compared "
                "against a naive baseline. Ensembles performed best when compared.",
                "Forecasts should be evaluated prospectively and against simple baselines."),
            design=design, country=None, sample_size=k, concepts=concepts[2:],
            pico=_pico("Studies forecasting influenza activity", "Statistical, mechanistic and machine learning models",
                       "Naive baselines", "Forecast accuracy", design,
                       f"Ensembles outperformed single models; only part of the {k} models were evaluated out of sample.",
                       "mixed"))
    return dict(
        title=d.pick((f"Forecasting {oc} in {place} with {meth}: a retrospective evaluation",
                      f"{meth[0].upper() + meth[1:]} combining {streams} to forecast {oc}",
                      f"Nowcasting {oc} during the {_season_before(d, year)} season in {place}",
                      f"An early warning score for influenza epidemics based on {streams}: development and validation in {place}")),
        abstract=_abstract(
            "Hospitals and public health services need forecasts of influenza activity weeks ahead.",
            f"We trained {meth} on {streams} to forecast {oc} {horizon} weeks ahead in {place}, using the seasons "
            f"{y1} to {y2 - 1} for training and {y2} for validation.",
            f"The model reduced the mean absolute error by {gain}% over a seasonal baseline and detected the epidemic "
            f"onset with an area under the curve of {auc}.",
            f"Adding early signals such as wastewater improves short-term forecasts of {oc}."),
        design=design, country=iso, sample_size=d.r.randint(150, 900), concepts=concepts,
        pico=_pico(f"Weekly {oc} in {place}, {y1}-{y2}", f"Forecasts from {meth} using {streams}",
                   "Seasonal baseline forecast", f"Accuracy {horizon} weeks ahead", design,
                   f"Mean absolute error reduced by {gain}% over the baseline; onset detected with AUC {auc}.",
                   "decrease"))


def _syndromic(d: _Draw, year: int) -> dict:
    place, iso = d.place()
    y1, y2 = d.period(year, span=(3, 9))
    stream, stream_c = d.pick((
        ("emergency department visits", ("emergency department visit", "passage aux urgences")),
        ("ambulance calls for respiratory complaints", ("ambulance call", "appel d'ambulance")),
        ("sentinel general practitioner consultations", ("sentinel consultation", "consultation sentinelle")),
        ("telephone triage calls", ("telephone triage", "triage téléphonique")),
    ))
    n = d.r.randint(12000, 480000)
    rho = d.between(0.58, 0.9)
    lead = d.r.randint(2, 9)
    design = d.weighted([("Surveillance study", 4), ("Retrospective cohort study", 2),
                         ("Cross-sectional study", 2), ("Qualitative study based on interviews", 1),
                         ("Not stated", 1)])
    concepts = [_c("pathogen", PATHOGENS["flu_a"]), _c("method", ("syndromic surveillance", "surveillance syndromique")),
                _c("outcome", stream_c), _c("population", ("general population", "population générale"))]
    if design.startswith("Qualitative"):
        k = d.r.randint(12, 31)
        return dict(
            title=f"How public health officers in {place} use early warning signals for influenza: an interview study",
            abstract=_abstract(
                "Early warning signals are only useful if they change decisions.",
                f"We interviewed {k} public health officers and hospital managers in {place} about the signals they "
                "use before and during influenza epidemics.",
                "Participants trusted syndromic and laboratory data, wanted wastewater data presented with its "
                "uncertainty, and asked for alerts tied to concrete actions.",
                "Early warning tools should be designed with their users and their decisions."),
            design=design, country=iso, sample_size=k, concepts=concepts[1:2],
            pico=_pico(f"{k} public health officers and hospital managers in {place}", "Early warning signals",
                       "Not specified", "Perceived usefulness for decisions", design,
                       "Officers wanted wastewater signals with their uncertainty and alerts tied to actions.",
                       "descriptive"))
    return dict(
        title=d.pick((f"Syndromic surveillance of influenza-like illness from {stream} in {place}",
                      f"{stream[0].upper() + stream[1:]} as an early indicator of influenza activity in {place}, {y1}-{y2}",
                      f"Agreement between syndromic and laboratory surveillance of influenza in {place}")),
        abstract=_abstract(
            "Syndromic data are available in near real time, before laboratory confirmation.",
            f"We analysed {n:,} {stream} coded for influenza-like symptoms in {place} between {y1} and {y2}, "
            "and compared them with laboratory-confirmed influenza.",
            f"Syndromic counts correlated with laboratory data (rho {rho}) and peaked {lead} days earlier.",
            f"{stream[0].upper() + stream[1:]} provide a timely, low-cost signal of influenza activity."),
        design=design, country=iso, sample_size=n, concepts=concepts,
        pico=_pico(f"{n:,} {stream} in {place}", "Syndromic coding of influenza-like illness",
                   "Laboratory-confirmed influenza", "Correlation and timeliness", design,
                   f"Syndromic counts correlated with laboratory data (rho {rho}) and peaked {lead} days earlier.",
                   "increase"))


def _burden(d: _Draw, year: int) -> dict:
    place, iso = d.place()
    season = _season_before(d, year)
    group, group_c = d.pick((("adults aged 65 and over", ("older adults", "personnes âgées")),
                             ("healthcare workers", ("healthcare worker", "soignant")),
                             ("pregnant women", ("pregnant women", "femmes enceintes")),
                             ("children under five", ("children", "enfants"))))
    ve = d.r.randint(21, 63)
    n = d.r.randint(400, 9000)
    design = d.weighted([("observational", 7), ("Cluster randomised controlled trial", 2),
                         ("Practice guideline", 1), ("Case series", 1)])
    concepts = [_c("pathogen", PATHOGENS["flu_a"]), _c("population", group_c),
                _c("intervention", ("influenza vaccination", "vaccination antigrippale"))]
    if design.startswith("Cluster randomised"):
        arm = d.pick(("reminder letters", "on-site vaccination days", "text message reminders"))
        return dict(
            title=f"{arm.capitalize()} to increase influenza vaccine uptake among {group}: a cluster randomised trial",
            abstract=_abstract(
                f"Influenza vaccine uptake among {group} remains below targets.",
                f"We randomised {d.r.randint(20, 60)} practices in {place} to {arm} or usual care during the "
                f"{season} season ({n:,} participants).",
                f"Uptake was {d.r.randint(4, 12)} percentage points higher with {arm}.",
                f"{arm.capitalize()} are a cheap way to raise uptake."),
            design=design, country=iso, sample_size=n, concepts=concepts[1:],
            pico=_pico(f"{n:,} {group} in {place}", arm.capitalize(), "Usual care", "Vaccine uptake", design,
                       f"Uptake rose with {arm}.", "increase"))
    if design == "Practice guideline":
        return dict(
            title=f"Recommendations for influenza vaccination of {group}: {place}, {year}/{(year + 1) % 100:02d}",
            abstract=_abstract(
                "National recommendations are updated each season.",
                "An expert panel reviewed the evidence on effectiveness and safety.",
                f"Annual vaccination is recommended for {group}, ideally before the start of the season.",
                "The recommendation is maintained for the coming season."),
            design=design, country=iso, sample_size=None, concepts=concepts[1:],
            pico=_pico(group.capitalize(), "Seasonal influenza vaccination", "No vaccination",
                       "Recommendation", design, "", "none"))
    if design == "Case series":
        k = d.r.randint(5, 19)
        return dict(
            title=f"Influenza outbreaks in {k} nursing homes in {place} during the {season} season: a case series",
            abstract=_abstract(
                "Nursing homes are vulnerable to influenza outbreaks.",
                f"We describe {k} outbreaks notified in {place} during the {season} season.",
                f"Attack rates ranged from {d.r.randint(8, 20)}% to {d.r.randint(25, 48)}%; outbreaks were "
                "shorter where antivirals were started early.",
                "Early detection and antiviral prophylaxis limit outbreaks."),
            design=design, country=iso, sample_size=k, concepts=concepts[:2],
            pico=_pico(f"Residents of {k} nursing homes in {place}", "Antiviral prophylaxis", "Not specified",
                       "Attack rate and outbreak duration", design,
                       "Outbreaks were shorter where antivirals were started early.", "decrease"))
    variant = d.pick(("effectiveness", "intensive care", "uptake"))
    if variant == "effectiveness":
        design = "Test-negative case-control study"
        return dict(
            title=f"Influenza vaccine effectiveness against hospitalisation among {group} in {place}, {season} season",
            abstract=_abstract(
                f"Influenza causes a large burden of hospital care among {group}.",
                f"We compared vaccination among {n:,} {group} hospitalised with a respiratory infection in {place} "
                f"during the {season} season, influenza-positive cases against test-negative controls.",
                f"Vaccine effectiveness against hospitalisation was {ve}% (95% CI {max(1, ve - 18)} to {ve + 12}).",
                "Vaccination remains the main lever against severe influenza."),
            design=design, country=iso, sample_size=n, concepts=concepts,
            pico=_pico(f"{n:,} {group} hospitalised with a respiratory infection in {place}",
                       "Seasonal influenza vaccination", "Unvaccinated", "Influenza hospitalisation", design,
                       f"Vaccine effectiveness against hospitalisation was {ve}%.", "decrease"))
    if variant == "intensive care":
        design = "Retrospective cohort study"
        icu = d.r.randint(9, 24)
        return dict(
            title=f"Burden of laboratory-confirmed influenza on intensive care units in {place}, {season} season",
            abstract=_abstract(
                "Severe influenza strains intensive care capacity during epidemic peaks.",
                f"We followed {n:,} patients admitted with laboratory-confirmed influenza in {place} during the "
                f"{season} season.",
                f"{icu}% needed intensive care, and intensive care occupancy by influenza peaked two weeks after "
                "the epidemic peak.",
                "Forecasts of the epidemic peak can help plan intensive care capacity."),
            design=design, country=iso, sample_size=n,
            concepts=concepts[:1] + [_c("outcome", ("intensive care admission", "admission en soins intensifs"))],
            pico=_pico(f"{n:,} patients admitted with influenza in {place}", "Laboratory-confirmed influenza",
                       "Not specified", "Intensive care admission", design,
                       f"{icu}% of patients admitted with influenza needed intensive care.", "descriptive"))
    design = "Cross-sectional survey"
    uptake = d.r.randint(18, 71)
    return dict(
        title=f"Uptake of seasonal influenza vaccination among {group} in {place}",
        abstract=_abstract(
            f"Influenza vaccination is recommended for {group}, but uptake varies.",
            f"We surveyed {n:,} {group} in {place} after the {season} season.",
            f"{uptake}% reported being vaccinated; a recommendation by a health professional was the strongest "
            "predictor of uptake.",
            "Health professionals are the key to higher uptake."),
        design=design, country=iso, sample_size=n, concepts=concepts,
        pico=_pico(f"{n:,} {group} in {place}", "Recommendation by a health professional", "No recommendation",
                   "Vaccine uptake", design, f"{uptake}% of {group} reported being vaccinated.", "descriptive"))


# Records a broad search also returns, about something else: they sit below the threshold.
# (title, background, methods, results, conclusions, population, exposure, outcome,
#  typed concept, design)
_OFF_TOPIC = (
    ("Antimicrobial prescribing for respiratory tract infections in primary care in {place}",
     "Antibiotics are often prescribed for viral respiratory infections.",
     "We analysed prescriptions in {n:,} primary care consultations for acute respiratory infection.",
     "{pct}% of these consultations led to an antibiotic prescription.",
     "Stewardship should target respiratory consultations.",
     "{n:,} primary care consultations in {place}", "Acute respiratory infection", "Antibiotic prescription",
     ("outcome", "antibiotic prescribing", "prescription d'antibiotiques"), "Cross-sectional study"),
    ("Air pollution and paediatric asthma admissions in {place}",
     "Fine particulate matter is a known trigger of asthma.",
     "We related daily PM2.5 to {n:,} paediatric asthma admissions with a case-crossover design.",
     "Each 10 microgram/m3 increase in PM2.5 raised admissions by {pct}%.",
     "Air quality alerts could protect children with asthma.",
     "{n:,} paediatric asthma admissions in {place}", "Daily PM2.5 concentration", "Asthma admission",
     ("exposure", "fine particulate matter", "particules fines"), "Case-control study (case-crossover design)"),
    ("Tick-borne encephalitis incidence and climate in {place}",
     "Tick-borne encephalitis is spreading to higher altitudes.",
     "We modelled {n:,} notified cases against temperature and vegetation indices.",
     "Incidence rose by {pct}% per decade, faster at higher altitudes.",
     "Vaccination recommendations should follow the spread of ticks.",
     "{n:,} notified cases in {place}", "Temperature and vegetation indices", "Tick-borne encephalitis incidence",
     ("pathogen", "tick-borne encephalitis virus", "virus de l'encéphalite à tiques"), "Ecological time-series study"),
    ("Norovirus outbreaks in long-term care facilities in {place}",
     "Norovirus causes most gastroenteritis outbreaks in care homes.",
     "We reviewed {n:,} outbreak notifications.",
     "{pct}% of outbreaks occurred between December and March.",
     "Winter infection control should cover norovirus as well as influenza.",
     "Residents of long-term care facilities in {place}", "Season", "Gastroenteritis outbreak",
     ("pathogen", "norovirus", "norovirus"), "Surveillance study"),
    ("Antibiotic resistance genes in hospital wastewater in {place}",
     "Hospital effluents may spread antimicrobial resistance.",
     "We sequenced {n:,} samples of hospital and municipal wastewater.",
     "Carbapenemase genes were {pct} times more abundant in hospital effluents.",
     "Hospital wastewater should be monitored for resistance genes.",
     "{n:,} wastewater samples in {place}", "Hospital effluent", "Abundance of resistance genes",
     ("outcome", "antimicrobial resistance", "résistance aux antimicrobiens"), "Cross-sectional study"),
    ("Heat waves and emergency medical service demand in {place}",
     "Heat waves increase demand on emergency services.",
     "We analysed {n:,} ambulance dispatches during summer months.",
     "Dispatches rose by {pct}% on heat-wave days.",
     "Heat-health action plans should include ambulance capacity.",
     "{n:,} ambulance dispatches in {place}", "Heat wave", "Ambulance dispatches",
     ("exposure", "heat wave", "canicule"), "Ecological time-series study"),
    ("Legionella in hot water systems: environmental surveillance in {place}",
     "Legionella grows in poorly maintained hot water systems.",
     "We tested {n:,} water samples from public buildings.",
     "{pct}% of samples exceeded the action threshold.",
     "Regular temperature checks prevent colonisation.",
     "{n:,} water samples from public buildings in {place}", "Hot water temperature", "Legionella colonisation",
     ("pathogen", "Legionella pneumophila", "Legionella pneumophila"), "Surveillance study"),
    ("Measles immunity gaps among adults in {place}",
     "Measles outbreaks increasingly involve adults.",
     "We measured measles antibodies in {n:,} blood donors.",
     "{pct}% of adults born between 1970 and 1985 lacked protective antibodies.",
     "Catch-up vaccination of adults should be offered.",
     "{n:,} blood donors in {place}", "Birth cohort", "Measles seroprotection",
     ("outcome", "measles immunity", "immunité contre la rougeole"), "Cross-sectional survey"),
)


def _off_topic(d: _Draw, year: int) -> dict:
    place, iso = d.place()
    title, bg, me, re_, co, pop, exposure, outcome, (ctype, en, fr), design = d.pick(_OFF_TOPIC)
    fill = {"place": place, "n": d.r.randint(300, 40000), "pct": d.r.randint(4, 48)}
    return dict(
        title=title.format(**fill),
        abstract=_abstract(bg, me.format(**fill), re_.format(**fill), co),
        design=design, country=iso, sample_size=fill["n"], concepts=[_c(ctype, (en, fr))],
        pico=_pico(pop.format(**fill), exposure, "Not specified", outcome, design,
                   re_.format(**fill), "descriptive"))


#: Author keywords by theme. Distinct from the typed concepts on purpose: the concept map
#: turns keywords into Topic nodes, and a keyword repeating a concept would link it to itself.
THEME_KEYWORDS = {
    "wastewater": ("sewage", "early warning", "RT-qPCR", "public health surveillance", "catchment"),
    "forecasting": ("early warning", "prediction", "decision support", "epidemic intelligence"),
    "weather": ("climate", "seasonality", "environmental epidemiology", "lag effects"),
    "cocirculation": ("viral interference", "respiratory viruses", "COVID-19", "seasonality"),
    "syndromic": ("early warning", "emergency medical services", "public health surveillance", "timeliness"),
    "burden": ("vaccine uptake", "hospital capacity", "immunisation programmes", "prevention"),
    "off_topic": ("environmental health", "primary care", "infection control"),
}

# (name, maker, share of the corpus, mean relevance score, first year)
THEMES = (
    ("wastewater", _wastewater, 0.17, 0.63, 2019),
    ("forecasting", _forecasting, 0.14, 0.59, 2012),
    ("weather", _weather, 0.15, 0.56, 2008),
    ("cocirculation", _cocirculation, 0.11, 0.51, 2016),
    ("syndromic", _syndromic, 0.12, 0.47, 2008),
    ("burden", _burden, 0.12, 0.38, 2011),
    ("off_topic", _off_topic, 0.19, 0.27, 2008),
)


def build_corpus(n: int = DEFAULT_ARTICLES, seed: int = DEFAULT_SEED) -> list[Record]:
    """`n` synthetic records, the same for the same (n, seed). PURE.

    Titles are unique (a draw that repeats one is redrawn), every theme gets its share,
    and the relevance score of a record is drawn around the mean of its theme, so the
    default threshold (0.45) keeps the on-topic themes and drops most of the rest."""
    d = _Draw(seed)
    counts = [max(1, round(n * share)) for _, _, share, _, _ in THEMES]
    counts[-1] += n - sum(counts)                      # the remainder goes to the last theme
    plan = [i for i, c in enumerate(counts) for _ in range(c)]
    d.r.shuffle(plan)
    out: list[Record] = []
    seen: set[str] = set()
    sources = [(s, w) for s, w, _ in SOURCES]
    since = {s: y for s, _, y in SOURCES}
    for key, ti in enumerate(plan, start=1):
        name, maker, _, mean, first = THEMES[ti]
        source = d.weighted(sources)
        year = d.year(max(first, since[source]))
        for _ in range(50):
            f = maker(d, year)
            if f["title"].lower() not in seen:
                break
        seen.add(f["title"].lower())
        also = sorted({d.weighted(sources) for _ in range(d.r.randint(0, 2))} - {source})
        score = min(0.91, max(0.09, round(d.r.gauss(mean, 0.075), 3)))
        if name == "off_topic":                        # a broad search's noise stays below the threshold
            score = min(score, OFF_TOPIC_MAX_SCORE)
        has_fulltext = d.r.random() < (0.75 if source in ("europepmc", "pubmed", "preprint") else 0.45)
        pool = list(THEME_KEYWORDS[name])
        d.r.shuffle(pool)
        keywords = "; ".join(pool[:d.r.randint(2, 3)])
        pico = dict(f["pico"], pico_confidence=round(d.r.uniform(0.62, 0.95), 2))
        out.append(Record(
            key=key, theme=name, source=source, year=year, title=f["title"], abstract=f["abstract"],
            design=f["design"], pico=pico, concepts=f["concepts"], country=f.get("country"),
            sample_size=f.get("sample_size"), keywords=keywords, score=score,
            has_fulltext=has_fulltext, open_access=has_fulltext or d.r.random() < 0.3,
            citation_count=max(0, int(d.r.expovariate(1 / max(1, (LAST_YEAR - year) * 4)))),
            found_in=tuple([source] + also)))
    return out


def concepts_json(rec: Record, version: int = 2) -> dict:
    """The row's `concepts_json`, in the shape the concept pass writes: typed concepts
    plus the country as an ISO2 `place`."""
    concepts = list(rec.concepts)
    if rec.country:
        concepts.append({"t": "place", "en": rec.country, "fr": rec.country})
    return {"v": version, "concepts": concepts[:8]}


def fulltext_sections(rec: Record) -> list[tuple[str, str]]:
    """The full text of a record that has one: its abstract, laid out in the sections a
    full text has, so the counters and the full-text readers find what they look for."""
    if not rec.has_fulltext:
        return []
    parts = {}
    rest = rec.abstract
    for label in ("Conclusions", "Results", "Methods", "Background"):
        head, sep, tail = rest.rpartition(f"{label}: ")
        if sep:
            parts[label], rest = tail.strip(), head.strip()
    return [(section, parts[key]) for section, key in (("Introduction", "Background"), ("Methods", "Methods"),
                                                       ("Results", "Results"), ("Discussion", "Conclusions"))
            if parts.get(key)]


def records_by_source(records: list[Record]) -> dict[str, int]:
    """Records each source returned, duplicates across sources included (PRISMA
    identification counts a record once per source that returned it)."""
    out: dict[str, int] = {}
    for r in records:
        for s in r.found_in:
            out[s] = out.get(s, 0) + 1
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Seeding: SQL for the rows, the API for the reviewers' decisions
# ─────────────────────────────────────────────────────────────────────────────
def _post(api: str, path: str, api_key: str, params: dict | None = None, body: dict | None = None) -> dict:
    url = api.rstrip("/") + path + ("?" + urllib.parse.urlencode(params) if params else "")
    data = json.dumps(body).encode() if body is not None else b""
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"X-API-Key": api_key, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read() or b"{}")


def wait_for_scenario(engine, sid: str, timeout: float = 240) -> None:
    """The built-in demo scenario is created by a startup thread that first trains its
    model: wait until its row exists."""
    from sqlalchemy import text
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with engine.connect() as c:
            if c.execute(text("SELECT 1 FROM user_scenarios WHERE id = :sid"), {"sid": sid}).first():
                return
        time.sleep(1)
    raise RuntimeError(f"scenario {sid} did not appear within {timeout:.0f} s (startup seed failed?)")


def insert(engine, sid: str, records: list[Record]) -> dict[int, int]:
    """Write the records and link them to the scenario. Returns {record key: document id}.

    Idempotent: the records of an earlier run (found by their `external_id` prefix) are
    removed first, with their links."""
    from sqlalchemy import text
    from api.pico_map import normalise_pico
    from api.knowledge_graph import CONCEPTS_VERSION
    remove(engine, sid)
    ids: dict[int, int] = {}
    with engine.begin() as c:
        _sync_id_sequence(c)
        for r in records:
            doc_id = c.execute(text("""
                INSERT INTO literature_document
                    (external_id, source, title, abstract, year, journal, project_context,
                     source_type, study_design, publication_type, pico_json, pico_extracted_at,
                     concepts_json, country, sample_size, keywords, language, has_fulltext,
                     open_access, citation_count)
                VALUES (:ext, :source, :title, :abstract, :year, :journal, 'literev',
                        'article', :design, :ptype, CAST(:pico AS jsonb), NOW(),
                        CAST(:concepts AS jsonb), :country, :n, :kw, 'en', :ft, :oa, :cites)
                RETURNING id
            """), {"ext": r.external_id, "source": r.source, "title": r.title, "abstract": r.abstract,
                   "year": r.year, "journal": JOURNAL, "design": r.design,
                   "ptype": "Preprint" if r.source == "preprint" else "Journal Article",
                   "pico": json.dumps(normalise_pico(r.pico, source="abstract")),
                   "concepts": json.dumps(concepts_json(r, CONCEPTS_VERSION)),
                   "country": r.country, "n": r.sample_size, "kw": r.keywords,
                   "ft": r.has_fulltext, "oa": r.open_access, "cites": r.citation_count}).scalar()
            ids[r.key] = int(doc_id)
            c.execute(text("""
                INSERT INTO article_scenarios (scenario_id, document_id, similarity_score)
                VALUES (:sid, :did, :score) ON CONFLICT DO NOTHING
            """), {"sid": sid, "did": doc_id, "score": r.score})
            c.execute(text("""
                INSERT INTO document_chunk (document_id, chunk_index, content, chunk_type)
                VALUES (:did, 0, :content, 'title_abstract')
            """), {"did": doc_id, "content": (r.title + "\n\n" + r.abstract)[:2000]})
            for i, (label, body) in enumerate(fulltext_sections(r), start=1):
                c.execute(text("""
                    INSERT INTO document_chunk (document_id, chunk_index, content, chunk_type, section_label)
                    VALUES (:did, :i, :content, 'fulltext_section', :label)
                """), {"did": doc_id, "i": i, "content": f"{label}\n\n{body}", "label": label})
    return ids


def _sync_id_sequence(conn) -> None:
    """Bring the id sequence of `literature_document` up to its highest id.

    A seeder that inserts explicit ids (scripts/bench_scenario.py) leaves the sequence
    behind when it is not OWNED BY the column, as in a database bootstrapped from
    schema.sql, and the next ordinary insert collides with id 1."""
    from sqlalchemy import text
    default = conn.execute(text(
        "SELECT column_default FROM information_schema.columns "
        "WHERE table_name = 'literature_document' AND column_name = 'id'")).scalar() or ""
    m = re.search(r"nextval\('([^']+)'", default)
    if m:
        conn.execute(text("SELECT setval(CAST(:seq AS regclass), "
                          "GREATEST((SELECT COALESCE(MAX(id), 0) FROM literature_document), 1))"),
                     {"seq": m.group(1)})


def remove(engine, sid: str) -> None:
    """Remove the records of an earlier run and their links (never any other article)."""
    from sqlalchemy import text
    with engine.begin() as c:
        c.execute(text("""
            DELETE FROM article_scenarios WHERE scenario_id = :sid AND document_id IN (
                SELECT id FROM literature_document WHERE external_id LIKE :pfx)
        """), {"sid": sid, "pfx": EXTERNAL_PREFIX + "%"})
        c.execute(text("DELETE FROM literature_document WHERE external_id LIKE :pfx"),
                  {"pfx": EXTERNAL_PREFIX + "%"})


def finish_search(engine, sid: str, records: list[Record]) -> dict:
    """What a completed search leaves on the scenario row: its counts, its status, and the
    PRISMA identification figures, computed by the app's own function from the records
    each source returned."""
    from sqlalchemy import text
    from api.search import _prisma_identification_figures
    by_source = records_by_source(records)
    outcomes = {s: ("ok" if by_source.get(s) else "empty")
                for s in ("pubmed", "europepmc", "openalex", "crossref", "semantic_scholar", "doaj",
                          "core", "openaire", "arxiv", "clinicaltrials", "preprints")}
    by_source["preprint"] = by_source.pop("preprint", 0)
    figures = _prisma_identification_figures(
        by_source, unique_records=len(records), duplicate_rows_removed=0, corpus_total=len(records),
        method="populate", source_outcomes=outcomes, per_source_cap=20000, search_mode="exhaustive",
        source_caps={"crossref": 2000})
    with engine.begin() as c:
        c.execute(text("""
            UPDATE user_scenarios
            SET article_count = :n, result_count = :n, populate_status = 'done',
                pipeline_status = 'done', pipeline_step = 'done', pipeline_progress = 100,
                search_mode = 'exhaustive', prisma_identification = CAST(:f AS jsonb), updated_at = NOW()
            WHERE id = :sid
        """), {"n": len(records), "f": json.dumps(figures), "sid": sid})
    return figures


def screen(api: str, api_key: str, sid: str, records: list[Record], ids: dict[int, int],
           threshold: float = 0.45, seed: int = DEFAULT_SEED) -> dict:
    """A reviewer's first pass, through the API: some relevant articles included, a few
    excluded with a reason (vetoed), two below the threshold rescued; and a double-blind
    pass by two reviewers who disagree on a few articles. Returns the counts."""
    d = _Draw(seed + 1)
    above = [r for r in records if r.score >= threshold and r.theme != "off_topic"]
    below = [r for r in records if r.score < threshold and r.theme in ("syndromic", "burden")]
    d.r.shuffle(above)
    d.r.shuffle(below)
    included, vetoed, rescued = above[:22], above[22:27], below[:2]
    base = f"/user-scenarios/{sid}/articles"
    for r in included + rescued:
        _post(api, f"{base}/{ids[r.key]}/screen", api_key, {"status": "included"})
    for r in vetoed:
        _post(api, f"{base}/{ids[r.key]}/screen", api_key,
              {"status": "excluded", "reason": d.pick(EXCLUSION_REASONS)})
    # A double-blind pass on 24 more: the reviewers agree to include 16 and to exclude 5,
    # and disagree on 3 (the second excludes), which stay open as conflicts to arbitrate.
    # An agreed exclusion carries no reason of its own, so the reason is then recorded the
    # way a reviewer would, through the screening endpoint.
    blind = above[27:51]
    for code in REVIEWERS:
        _post(api, f"/user-scenarios/{sid}/double-blind/register", api_key, {"reviewer_code": code})
    votes = ["included"] * 16 + ["excluded"] * 5 + ["conflict"] * 3
    for r, vote in zip(blind, votes):
        first, second = ("included", "excluded") if vote == "conflict" else (vote, vote)
        reason = d.pick(EXCLUSION_REASONS)
        for code, status in zip(REVIEWERS, (first, second)):
            body = {"article_id": ids[r.key], "status": status, "reviewer_code": code}
            if status == "excluded":
                body["reason"] = reason
            _post(api, f"/user-scenarios/{sid}/double-blind/decision", api_key, body=body)
        if vote == "excluded":
            _post(api, f"{base}/{ids[r.key]}/screen", api_key, {"status": "excluded", "reason": reason})
    disagree = [r for r, vote in zip(blind, votes) if vote == "conflict"]
    return {"included": len(included), "vetoed": len(vetoed), "rescued": len(rescued),
            "double_blind": len(blind), "disagreements": len(disagree)}


#: A library this big is somebody's real one: the fiction stays out of it unless forced.
REAL_LIBRARY = 500


def check_library(engine) -> int:
    """The articles of the library that are not demo records. The demo records are
    fiction, and a library is shared by every scenario: seeded into a real one, they would
    turn up in real searches."""
    from sqlalchemy import text
    with engine.connect() as c:
        return int(c.execute(text(
            "SELECT COUNT(*) FROM literature_document "
            "WHERE external_id IS NULL OR external_id NOT LIKE :pfx"), {"pfx": EXTERNAL_PREFIX + "%"}).scalar() or 0)


def seed(db_url: str, api: str, api_key: str, sid: str | None = None,
         n: int = DEFAULT_ARTICLES, seed_value: int = DEFAULT_SEED, force: bool = False) -> dict:
    """Seed the demo corpus into `sid` (default: the built-in demo scenario)."""
    import sqlalchemy as sa
    sys.path.insert(0, ROOT)
    import demo_seed
    sid = sid or demo_seed.DEMO_SCENARIO_ID
    engine = sa.create_engine(db_url)
    try:
        real = check_library(engine)
        if real > REAL_LIBRARY and not force:
            raise SystemExit(f"this database already holds {real} real articles: the demo corpus is fiction "
                             "and would mix with them. Seed a throwaway database (scripts/demo_video.py "
                             "makes one), or pass --force.")
        wait_for_scenario(engine, sid)
        records = build_corpus(n, seed_value)
        ids = insert(engine, sid, records)
        figures = finish_search(engine, sid, records)
        decisions = screen(api, api_key, sid, records, ids, seed=seed_value)
    finally:
        engine.dispose()
    above = sum(1 for r in records if r.score >= 0.45)
    return {"scenario": sid, "articles": len(records), "above_threshold": above,
            "identified": figures.get("records_identified"), **decisions}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db-url", default=os.environ.get("DB_URL"), help="database of the running API")
    ap.add_argument("--api", default="http://127.0.0.1:8000", help="base URL of the running API")
    ap.add_argument("--api-key", default=os.environ.get("WRITE_API_KEY"), help="its WRITE_API_KEY")
    ap.add_argument("--scenario", default=None, help="scenario id (default: the built-in demo scenario)")
    ap.add_argument("--articles", type=int, default=DEFAULT_ARTICLES)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--force", action="store_true", help="seed even into a database that holds a real library")
    args = ap.parse_args()
    if not args.db_url or not args.api_key:
        ap.error("--db-url (or DB_URL) and --api-key (or WRITE_API_KEY) are required")
    # The app's helpers read their configuration at import: point them at the same database.
    os.environ["DB_URL"] = args.db_url
    os.environ.setdefault("WRITE_API_KEY", args.api_key)
    summary = seed(args.db_url, args.api, args.api_key, args.scenario, args.articles, args.seed, args.force)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
