"""The label codebook: a hierarchy of labels (up to three levels) per extraction sheet.

The extraction writes the labels as the paper words them ("males", "Men", "male workers").
To count, compare or pool anything, the same thing must carry the same label: that is what
the codebook is for. Level 1 is a group of covariates (sex or gender, age, occupation...),
level 2 a value in it (male, female...), level 3 an optional detail.

The labels are NORMALISED AT READ TIME and never rewritten in the stored extraction, so a
change to the codebook (a synonym added, the reviewers' Annex 2 loaded) corrects every
extraction already made, with no new model call.

A scenario uses the default codebook until it stores its own (PUT, or a CSV import). The
default is a starting vocabulary drawn from the sheets of the review template; the reviewers'
own hierarchy replaces it.

Matching is deterministic and conservative: case, accents, punctuation and plurals are
folded, a synonym matches whole words only ("male" never matches inside "female"), and a
label that matches nothing stays as it was written and is listed as unmapped so that a
reviewer can map it.
"""
from __future__ import annotations

import csv
import io
import json
import re
import unicodedata
from typing import Any

from fastapi import Depends, HTTPException, Query, Request
from fastapi.responses import Response
from sqlalchemy import text

from .core import app, engine, logger, require_api_key
from .scenario_store import (_get_scenario_threshold, _get_user_scenario_or_404,
                             relevant_gate_sql)
from .schema_boot import _exec_ddl_isolated

CODEBOOK_SHEETS = ("human_susc", "human_exp", "env", "animal", "vector")
#: The template's own sheet names, accepted in an imported file.
_SHEET_ALIASES = {
    "human_cov_susc": "human_susc", "human_cov_exp": "human_exp", "env_cov": "env",
    "animalorreservoir_cov": "animal", "animal_or_reservoir": "animal", "animal": "animal",
    "vector_cov": "vector",
}
_MAX_NODES = 5000
_MAX_SYNONYMS = 40


def _ensure_codebook_column() -> None:
    _exec_ddl_isolated(
        ["ALTER TABLE scenario_settings ADD COLUMN IF NOT EXISTS codebook_json JSONB"],
        "_ensure_codebook_column")


try:
    _ensure_codebook_column()
except Exception as _e:                                       # noqa: BLE001 - never blocks startup
    logger.warning(f"_ensure_codebook_column: {_e}")


# ─────────────────────────────────────────────────────────────────────────────
# Folding a label
# ─────────────────────────────────────────────────────────────────────────────
_IRREGULAR = {"men": "man", "women": "woman", "children": "child", "people": "person",
              "persons": "person", "geese": "goose", "mice": "mouse", "lice": "louse"}
_KEEP_S = {"ss", "us", "is", "as", "os"}


def _singular(w: str) -> str:
    if w in _IRREGULAR:
        return _IRREGULAR[w]
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith("sses"):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and w[-2:] not in _KEEP_S:
        return w[:-1]
    return w


def clean_label(s: Any) -> str:
    """A label folded for comparison: ASCII, lower case, punctuation as spaces, singular.
    Characters that carry meaning in a covariate (+ < > %) are kept."""
    if s is None:
        return ""
    t = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    t = re.sub(r"[^a-z0-9+<>%]+", " ", t).strip()
    return " ".join(_singular(w) for w in t.split())


def key_slug(s: Any) -> str:
    """A codebook KEY: lower-case ASCII with underscores. Not singularised: "species" stays
    "species" (the folding above is for comparing labels, not for naming a node)."""
    t = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "_", t).strip("_")


# ─────────────────────────────────────────────────────────────────────────────
# The default codebook
# ─────────────────────────────────────────────────────────────────────────────
def _n(sheet: str, l1: str, l2: str | None = None, syn: tuple[str, ...] = (),
       en: str | None = None, fr: str | None = None) -> dict[str, Any]:
    return {"sheet": sheet, "l1": l1, "l2": l2, "l3": None, "synonyms": list(syn),
            "label_en": en, "label_fr": fr}


def _default_nodes() -> list[dict[str, Any]]:
    S, E, V, N, A = "human_susc", "human_exp", "env", "animal", "vector"
    return [
        # ── human susceptibility ─────────────────────────────────────────────
        _n(S, "sex_gender", None, ("sex", "gender", "sex gender", "sex/gender", "sexe", "genre"), "Sex or gender", "Sexe ou genre"),
        _n(S, "sex_gender", "male", ("man", "boy", "masculine", "homme", "masculin"), "Male", "Homme"),
        _n(S, "sex_gender", "female", ("woman", "girl", "feminine", "femme", "feminin"), "Female", "Femme"),
        _n(S, "sex_gender", "non_binary", ("non binary", "nonbinary", "transgender", "other gender"), "Non-binary", "Non binaire"),
        _n(S, "age", None, ("age group", "age category", "age band", "age range", "age", "tranche d age"), "Age", "Âge"),
        _n(S, "age", "infant", ("infant", "newborn", "neonate", "under 1", "<1", "nourrisson"), "Infants", "Nourrissons"),
        _n(S, "age", "child", ("child", "pediatric", "paediatric", "minor", "school age", "adolescent", "toddler", "under 5", "<5", "<18", "under 18", "enfant"), "Children", "Enfants"),
        _n(S, "age", "adult", ("adult", "working age", "18 64", "adulte"), "Adults", "Adultes"),
        _n(S, "age", "elderly", ("elderly", "older adult", "older people", "aged 65", "65+", ">65", "senior", "aine"), "Elderly", "Personnes âgées"),
        _n(S, "comorbidity", None, ("comorbidity", "underlying condition", "chronic disease", "chronic condition", "comorbidite"), "Comorbidity", "Comorbidité"),
        _n(S, "comorbidity", "immunocompromised", ("immunocompromised", "immunosuppressed", "immunodeficiency"), "Immunocompromised", "Immunodéprimés"),
        _n(S, "comorbidity", "diabetes", ("diabetes", "diabetic", "diabete"), "Diabetes", "Diabète"),
        _n(S, "comorbidity", "obesity", ("obesity", "obese", "obesite"), "Obesity", "Obésité"),
        _n(S, "pregnancy", None, ("pregnancy", "pregnant", "gestation", "grossesse"), "Pregnancy", "Grossesse"),
        _n(S, "immune_status", None, ("immune status", "immunity", "serostatus", "vaccination status", "vaccination", "immunite"), "Immune status", "Statut immunitaire"),
        _n(S, "immune_status", "vaccinated", ("vaccinated", "immunized", "immunised", "received vaccine", "vaccine recipient", "vaccin"), "Vaccinated", "Vaccinés"),
        _n(S, "immune_status", "unvaccinated", ("unvaccinated", "non vaccinated", "not vaccinated", "non vaccine", "no vaccine", "vaccine naive"), "Unvaccinated", "Non vaccinés"),
        _n(S, "immune_status", "seropositive", ("seropositive", "seroprevalence", "antibody positive", "seroconversion", "antibody"), "Seropositive", "Séropositifs"),
        _n(S, "ethnicity", None, ("ethnicity", "race", "ethnic group", "origine"), "Ethnicity", "Origine ethnique"),
        _n(S, "socioeconomic", None, ("socioeconomic", "socio economic", "income", "education", "deprivation", "socio demographic", "sociodemographic"), "Socioeconomic", "Socio-économique"),
        # ── human exposure ───────────────────────────────────────────────────
        _n(E, "occupation", None, ("occupation", "profession", "job", "work", "occupational group", "workplace", "metier"), "Occupation", "Profession"),
        _n(E, "occupation", "poultry_worker", ("poultry worker", "poultry farm worker", "poultry farmer", "poultry handler", "poultry keeper", "poultry holder", "poultry owner", "farm worker", "avicultural"), "Poultry worker", "Travailleur de la volaille"),
        _n(E, "occupation", "farmer", ("farmer", "farm", "agricultural worker", "livestock worker", "agriculteur"), "Farmer", "Agriculteur"),
        _n(E, "occupation", "veterinarian", ("veterinarian", "veterinary", "vet", "veterinary staff", "veterinary authority staff", "veterinaire"), "Veterinarian", "Vétérinaire"),
        _n(E, "occupation", "health_worker", ("health worker", "healthcare worker", "health care worker", "hcw", "nurse", "physician", "doctor", "clinician", "medical staff", "soignant"), "Health worker", "Soignant"),
        _n(E, "occupation", "slaughterhouse_worker", ("slaughterhouse worker", "slaughterhouse", "abattoir worker", "butcher", "meat processor", "slaughter worker"), "Slaughterhouse worker", "Travailleur d'abattoir"),
        _n(E, "occupation", "culler", ("culler", "culling worker", "culling team", "cull", "depopulation worker", "responder", "first responder", "police officer", "police"), "Culling or response staff", "Personnel d'abattage ou d'intervention"),
        _n(E, "occupation", "market_vendor", ("market vendor", "market worker", "live bird market", "seller", "trader", "vendor"), "Market vendor", "Vendeur de marché"),
        _n(E, "occupation", "laboratory_worker", ("laboratory worker", "lab worker", "laboratory staff", "researcher", "technician"), "Laboratory worker", "Personnel de laboratoire"),
        _n(E, "occupation", "shelter_staff", ("shelter staff", "animal shelter staff", "animal shelter", "animal caretaker", "zoo", "zoo keeper"), "Animal shelter staff", "Personnel de refuge"),
        _n(E, "occupation", "hunter", ("hunter", "hunting", "gamekeeper", "chasseur"), "Hunter", "Chasseur"),
        _n(E, "occupation", "general_public", ("general public", "general population", "community", "resident", "population generale"), "General public", "Grand public"),
        _n(E, "contact_type", None, ("contact", "contact type", "type of contact", "exposure type", "exposure", "type d exposition"), "Type of contact", "Type de contact"),
        _n(E, "contact_type", "direct_contact", ("direct contact", "close contact", "contact with infected bird", "contact with sick bird", "contact with poultry", "handling bird", "handling poultry", "contact with animal"), "Direct contact", "Contact direct"),
        _n(E, "contact_type", "indirect_contact", ("indirect contact", "contaminated environment", "contaminated material", "fomite", "contaminated surface"), "Indirect contact", "Contact indirect"),
        _n(E, "contact_type", "slaughtering", ("slaughtering", "slaughter", "defeathering", "plucking", "butchering", "processing"), "Slaughtering or processing", "Abattage ou transformation"),
        _n(E, "contact_type", "culling", ("culling", "depopulation", "euthanasia", "euthanised", "euthanized", "disposal"), "Culling", "Abattage sanitaire"),
        _n(E, "contact_type", "consumption", ("consumption", "eating", "undercooked", "raw egg", "raw meat", "food"), "Consumption", "Consommation"),
        _n(E, "behaviour_kap", None, ("kap", "knowledge attitude practice", "knowledge attitudes and practices", "knowledge", "attitude", "practice", "behaviour", "behavior", "risk perception", "perceived risk", "awareness", "cap", "connaissance", "comportement"), "Knowledge, attitudes, practices", "Connaissances, attitudes, pratiques"),
        _n(E, "behaviour_kap", "knowledge", ("knowledge", "knowledge score", "knowledge level", "awareness", "aware", "knows", "good knowledge", "poor knowledge"), "Knowledge", "Connaissances"),
        _n(E, "behaviour_kap", "attitude", ("attitude", "attitude score", "belief", "concern", "worry", "worried", "fear", "trust"), "Attitude", "Attitudes"),
        _n(E, "behaviour_kap", "practice", ("practice", "practice score", "hygiene", "hand washing", "handwashing", "hand hygiene", "biosecurity practice", "reporting", "good practice", "poor practice"), "Practice", "Pratiques"),
        _n(E, "behaviour_kap", "risk_perception", ("risk perception", "perceived risk", "perceived susceptibility", "perceived severity", "risk awareness", "perceived threat"), "Risk perception", "Perception du risque"),
        _n(E, "ppe", None, ("ppe", "personal protective equipment", "protective equipment", "protection", "equipement de protection", "epi"), "Protective equipment", "Équipements de protection"),
        _n(E, "ppe", "ppe_full", ("full ppe", "complete ppe", "appropriate ppe", "adequate ppe", "ffp2", "ffp3", "n95", "respirator", "full protection"), "Full PPE", "EPI complet"),
        _n(E, "ppe", "ppe_partial", ("partial ppe", "partial protection", "gloves", "glove", "mask", "surgical mask", "gown", "goggles", "boot", "shoe cover", "inadequate ppe"), "Partial PPE", "EPI partiel"),
        _n(E, "ppe", "ppe_none", ("no ppe", "none", "no protection", "without ppe", "unprotected", "no protective equipment"), "No PPE", "Sans EPI"),
        # ── environment ──────────────────────────────────────────────────────
        _n(V, "biosecurity", None, ("biosecurity", "biosafety", "biosecurity level", "biosecurity measure", "biosecurite"), "Biosecurity", "Biosécurité"),
        _n(V, "biosecurity", "biosecurity_low", ("low biosecurity", "poor biosecurity", "no biosecurity", "weak biosecurity", "inadequate biosecurity", "lack of biosecurity"), "Low biosecurity", "Biosécurité faible"),
        _n(V, "biosecurity", "biosecurity_high", ("high biosecurity", "good biosecurity", "strict biosecurity", "adequate biosecurity"), "High biosecurity", "Biosécurité élevée"),
        _n(V, "setting", None, ("setting", "location type", "farm type", "production system", "housing", "rearing", "contexte"), "Setting", "Contexte"),
        _n(V, "setting", "backyard", ("backyard", "small holding", "smallholder", "hobby farm", "free range", "free ranging", "free roaming", "outdoor", "village"), "Backyard or free-range", "Basse-cour ou plein air"),
        _n(V, "setting", "commercial_farm", ("commercial farm", "commercial poultry", "industrial farm", "intensive", "indoor", "poultry house", "broiler", "layer"), "Commercial farm", "Élevage commercial"),
        _n(V, "setting", "live_bird_market", ("live bird market", "wet market", "live poultry market", "market"), "Live bird market", "Marché d'oiseaux vivants"),
        _n(V, "setting", "wild_bird_contact", ("wild bird", "wild bird contact", "migratory", "waterfowl", "wetland", "lake", "contact with wild bird"), "Wild-bird contact", "Contact avec l'avifaune sauvage"),
        _n(V, "climate", None, ("climate", "weather", "season", "seasonality", "temperature", "humidity", "rainfall", "precipitation", "wind", "climat"), "Climate or season", "Climat ou saison"),
        _n(V, "persistence", None, ("persistence", "survival", "inactivation", "stability", "decay", "half life", "environmental persistence", "persistance"), "Persistence", "Persistance"),
        # ── animals and reservoirs ───────────────────────────────────────────
        _n(N, "species", None, ("species", "host", "animal species", "host species", "animal", "espece"), "Species", "Espèce"),
        _n(N, "species", "chicken", ("chicken", "gallus gallus", "gallus gallus domesticus", "hen", "broiler", "layer hen", "poulet"), "Chicken", "Poulet"),
        _n(N, "species", "duck", ("duck", "anas platyrhynchos", "mallard", "canard"), "Duck", "Canard"),
        _n(N, "species", "turkey", ("turkey", "meleagris gallopavo", "dinde"), "Turkey", "Dinde"),
        _n(N, "species", "goose", ("goose", "anser", "oie"), "Goose", "Oie"),
        _n(N, "species", "wild_bird", ("wild bird", "wild waterfowl", "migratory bird", "gull", "swan", "wild duck", "oiseau sauvage"), "Wild bird", "Oiseau sauvage"),
        _n(N, "species", "cat", ("cat", "felis catus", "domestic cat", "feline", "chat"), "Cat", "Chat"),
        _n(N, "species", "dog", ("dog", "canis lupus familiaris", "canine", "chien"), "Dog", "Chien"),
        _n(N, "species", "swine", ("swine", "pig", "sus scrofa", "porcine", "porc"), "Swine", "Porc"),
        _n(N, "species", "cattle", ("cattle", "cow", "dairy cow", "bos taurus", "bovine", "dairy cattle", "bovin"), "Cattle", "Bovin"),
        _n(N, "species", "wild_mammal", ("wild mammal", "fox", "mink", "seal", "marine mammal", "otter", "mustela vison", "neovison vison", "vulpes vulpes"), "Wild mammal", "Mammifère sauvage"),
        _n(N, "test", None, ("test", "diagnostic test", "assay", "laboratory result", "laboratory test", "pcr", "serology", "elisa", "hi assay", "hemagglutination inhibition", "test result", "dosage"), "Test", "Test"),
        _n(N, "test", "pcr_positive", ("pcr positive", "rt pcr positive", "rt qpcr positive", "virus detection", "viral rna", "pcr result"), "PCR positive", "PCR positive"),
        _n(N, "test", "seropositive", ("seropositive", "antibody positive", "seroprevalence", "elisa positive", "serological", "antibody"), "Seropositive", "Séropositif"),
        # ── vectors ──────────────────────────────────────────────────────────
        _n("vector", "species", None, ("species", "vector", "vector species", "mosquito", "tick", "espece"), "Vector species", "Espèce de vecteur"),
        _n("vector", "density", None, ("density", "abundance", "biting rate", "infection rate", "vector index", "densite"), "Density or infection rate", "Densité ou taux d'infection"),
    ]


DEFAULT_NODES: list[dict[str, Any]] = _default_nodes()


# ─────────────────────────────────────────────────────────────────────────────
# Validation, CSV, the index
# ─────────────────────────────────────────────────────────────────────────────
def _clean_node(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("a codebook node must be an object")
    sheet = str(raw.get("sheet") or "").strip().lower()
    sheet = _SHEET_ALIASES.get(sheet, sheet)
    if sheet not in CODEBOOK_SHEETS:
        raise ValueError(f"unknown sheet {raw.get('sheet')!r}; use one of {', '.join(CODEBOOK_SHEETS)}")
    l1, l2, l3 = (key_slug(raw.get(k)) or None for k in ("l1", "l2", "l3"))
    if not l1:
        raise ValueError("level 1 is required")
    if l3 and not l2:
        raise ValueError("level 3 needs level 2")
    syn_in = raw.get("synonyms") or []
    if isinstance(syn_in, str):
        syn_in = re.split(r"[|;]", syn_in)
    syn = []
    for s_ in syn_in:
        c = clean_label(s_)
        if c and c not in syn and len(c) <= 80:
            syn.append(c)
    return {"sheet": sheet, "l1": l1, "l2": l2, "l3": l3, "synonyms": syn[:_MAX_SYNONYMS],
            "label_en": (str(raw.get("label_en") or "").strip()[:120] or None),
            "label_fr": (str(raw.get("label_fr") or "").strip()[:120] or None)}


def validate_nodes(nodes: Any) -> list[dict[str, Any]]:
    """The nodes, cleaned. Raises ValueError with a message a person can act on."""
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("the codebook needs at least one node")
    if len(nodes) > _MAX_NODES:
        raise ValueError(f"too many nodes (maximum {_MAX_NODES})")
    out, seen = [], set()
    for i, raw in enumerate(nodes, 1):
        try:
            n = _clean_node(raw)
        except ValueError as e:
            raise ValueError(f"node {i}: {e}") from None
        key = (n["sheet"], n["l1"], n["l2"], n["l3"])
        if key in seen:
            raise ValueError(f"node {i}: duplicate of {n['sheet']} > {n['l1']}"
                             + (f" > {n['l2']}" if n["l2"] else ""))
        seen.add(key)
        out.append(n)
    return out


def parse_codebook_csv(content: str) -> list[dict[str, Any]]:
    """The reviewers' hierarchy from a CSV with the columns sheet, level1, level2, level3,
    synonyms (separated by | or ;), label_en, label_fr. Headers are matched loosely."""
    reader = csv.DictReader(io.StringIO(content.lstrip("﻿")))
    if not reader.fieldnames:
        raise ValueError("the file has no header row")
    cols = {re.sub(r"[^a-z0-9]", "", f.lower()): f for f in reader.fieldnames}

    def pick(row: dict, *names: str) -> str:
        for n in names:
            if n in cols and row.get(cols[n]) is not None:
                return str(row[cols[n]])
        return ""

    if "sheet" not in cols or not ({"level1", "l1"} & set(cols)):
        raise ValueError("the file needs at least the columns: sheet, level1")
    nodes = []
    for row in reader:
        if not any((v or "").strip() for v in row.values()):
            continue
        nodes.append({"sheet": pick(row, "sheet"), "l1": pick(row, "level1", "l1"),
                      "l2": pick(row, "level2", "l2"), "l3": pick(row, "level3", "l3"),
                      "synonyms": pick(row, "synonyms", "synonym"),
                      "label_en": pick(row, "labelen", "label"), "label_fr": pick(row, "labelfr")})
    return validate_nodes(nodes)


def codebook_to_csv(nodes: list[dict[str, Any]]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["sheet", "level1", "level2", "level3", "synonyms", "label_en", "label_fr"])
    for n in nodes:
        w.writerow([n["sheet"], n["l1"], n.get("l2") or "", n.get("l3") or "",
                    "|".join(n.get("synonyms") or []), n.get("label_en") or "", n.get("label_fr") or ""])
    return buf.getvalue()


class Index:
    """The codebook prepared for matching: per sheet, the level-1 synonyms and the nodes
    with their folded synonyms (the key itself is always a synonym of its node)."""

    def __init__(self, nodes: list[dict[str, Any]]):
        self.nodes = nodes
        self.by_sheet: dict[str, list[dict[str, Any]]] = {s: [] for s in CODEBOOK_SHEETS}
        self.l1_syn: dict[str, dict[str, str]] = {s: {} for s in CODEBOOK_SHEETS}
        for n in nodes:
            syns = {clean_label(n["l1"].replace("_", " "))}
            if n.get("l2"):
                syns = {clean_label(n["l2"].replace("_", " "))}
            if n.get("l3"):
                syns = {clean_label(n["l3"].replace("_", " "))}
            syns |= {clean_label(s) for s in (n.get("synonyms") or [])}
            for lab in (n.get("label_en"), n.get("label_fr")):
                if lab:
                    syns.add(clean_label(lab))
            syns.discard("")
            entry = {**n, "_syn": syns, "_tok": {s: tuple(s.split()) for s in syns}}
            self.by_sheet[n["sheet"]].append(entry)
            self.l1_syn[n["sheet"]].setdefault(clean_label(n["l1"].replace("_", " ")), n["l1"])
            if not n.get("l2"):
                for s in syns:
                    self.l1_syn[n["sheet"]].setdefault(s, n["l1"])


def _contains(tokens: tuple[str, ...], sub: tuple[str, ...]) -> bool:
    if not sub or len(sub) > len(tokens):
        return False
    return any(tokens[i:i + len(sub)] == sub for i in range(len(tokens) - len(sub) + 1))


def normalise(index: Index, sheet: str, group: Any, covariate: Any) -> dict[str, Any]:
    """The codebook position of a (group, covariate) as written in a paper.

    Returns l1, l2, l3 (None where nothing matched), `matched` (a position was found) and
    `group_key` / `covariate_key`: the codebook keys when matched, else the folded text, so
    that unmatched labels still group by their own spelling and never merge by accident."""
    g, c = clean_label(group), clean_label(covariate)
    nodes = index.by_sheet.get(sheet, [])
    l1 = index.l1_syn.get(sheet, {}).get(g) if g else None
    cand = [n for n in nodes if n["l2"] and (l1 is None or n["l1"] == l1)]
    best, best_len = None, 0
    if c:
        for n in cand:                                           # exact synonym first
            if c in n["_syn"]:
                best, best_len = n, 99
                break
        if best is None:                                         # then whole-word containment
            ctok = tuple(c.split())
            for n in cand:
                for syn, tok in n["_tok"].items():
                    if len(tok) > best_len and _contains(ctok, tok):
                        best, best_len = n, len(tok)
    if best is not None:
        return {"l1": best["l1"], "l2": best["l2"], "l3": best.get("l3"), "matched": True,
                "group_key": best["l1"], "covariate_key": best["l2"]}
    if l1 is not None:                                           # the group is known, the value is not
        return {"l1": l1, "l2": None, "l3": None, "matched": False,
                "group_key": l1, "covariate_key": c or None}
    return {"l1": None, "l2": None, "l3": None, "matched": False,
            "group_key": g or None, "covariate_key": c or None}


def label_path(n: dict[str, Any]) -> str | None:
    parts = [n.get("l1"), n.get("l2"), n.get("l3")]
    path = " > ".join(p for p in parts if p)
    return path if n.get("matched") and path else None


def annotate(index: Index, obs: dict[str, Any]) -> dict[str, Any]:
    """The observation with its codebook position added (the stored labels are untouched)."""
    n = normalise(index, obs.get("sheet") or "", obs.get("group"), obs.get("covariate"))
    return {**obs, "l1": n["l1"], "l2": n["l2"], "l3": n["l3"], "matched": n["matched"],
            "group_key": n["group_key"], "covariate_key": n["covariate_key"], "label_path": label_path(n)}


def vocabulary_prompt(nodes: list[dict[str, Any]]) -> str:
    """The level-1 and level-2 names, for the extraction prompt, so the model words its
    groups the way the codebook does. A paper's own wording is still kept in the label
    fields: this only steers the group name."""
    by: dict[str, dict[str, list[str]]] = {}
    for n in nodes:
        by.setdefault(n["sheet"], {}).setdefault(n["l1"], [])
        if n.get("l2") and n["l2"] not in by[n["sheet"]][n["l1"]]:
            by[n["sheet"]][n["l1"]].append(n["l2"])
    lines = []
    for sheet in CODEBOOK_SHEETS:
        if sheet in by:
            items = "; ".join(f"{l1}" + (f" ({', '.join(v[:8])})" if v else "") for l1, v in by[sheet].items())
            lines.append(f"{sheet}: {items}")
    return ("\n\nPREFERRED GROUP NAMES. For 'group', use one of these exact names when it fits, "
            "and keep the paper's own wording in 'covariate'. Otherwise write your own group.\n"
            + "\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# A scenario's codebook
# ─────────────────────────────────────────────────────────────────────────────
def get_codebook(scenario_id: str | None) -> dict[str, Any]:
    """{source: 'default' | 'custom', nodes}. The default when the scenario stores none, or
    when the stored one cannot be read."""
    if scenario_id:
        try:
            with engine.connect() as conn:
                raw = conn.execute(text("SELECT codebook_json FROM scenario_settings WHERE scenario_id = :sid"),
                                   {"sid": scenario_id}).scalar()
            if raw:
                data = raw if isinstance(raw, dict) else json.loads(raw)
                return {"source": "custom", "nodes": validate_nodes(data.get("nodes"))}
        except Exception as e:                                   # noqa: BLE001 - fall back, never block
            logger.warning(f"get_codebook {scenario_id}: {e}")
    return {"source": "default", "nodes": [dict(n) for n in DEFAULT_NODES]}


def get_index(scenario_id: str | None) -> Index:
    return Index(get_codebook(scenario_id)["nodes"])


def _store(scenario_id: str, nodes: list[dict[str, Any]] | None) -> None:
    payload = json.dumps({"nodes": nodes}, ensure_ascii=False) if nodes else None
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO scenario_settings (scenario_id, codebook_json)
            VALUES (:sid, CAST(:j AS jsonb))
            ON CONFLICT (scenario_id) DO UPDATE SET codebook_json = CAST(:j AS jsonb)
        """), {"sid": scenario_id, "j": payload})


def unmapped_labels(scenario_id: str, top: int = 50) -> dict[str, Any]:
    """Labels in the relevant corpus that match nothing in the codebook, most frequent first,
    so a reviewer can map them. Counted over ALL the relevant articles."""
    thr = _get_scenario_threshold(scenario_id)
    gate = relevant_gate_sql("d", "ars", ":thr")
    index = get_index(scenario_id)
    with engine.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT o->>'sheet' AS sheet, o->>'group' AS grp, o->>'covariate' AS cov,
                   COUNT(*) AS n_rows, COUNT(DISTINCT d.id) AS n_articles
            FROM literature_document d
            JOIN article_scenarios ars ON ars.document_id = d.id
            CROSS JOIN LATERAL jsonb_array_elements(d.extraction_json->'observations') AS o
            WHERE ars.scenario_id = :sid AND {gate}
              AND jsonb_typeof(d.extraction_json->'observations') = 'array'
            GROUP BY 1, 2, 3
        """), {"sid": scenario_id, "thr": thr}).mappings().all()
    mapped = unmapped = 0
    miss = []
    for r in rows:
        n = normalise(index, r["sheet"] or "", r["grp"], r["cov"])
        if n["matched"]:
            mapped += int(r["n_rows"])
        else:
            unmapped += int(r["n_rows"])
            miss.append({"sheet": r["sheet"], "group": r["grp"], "covariate": r["cov"],
                         "n_rows": int(r["n_rows"]), "n_articles": int(r["n_articles"]),
                         "l1": n["l1"]})
    miss.sort(key=lambda m: (-m["n_rows"], str(m["covariate"])))
    return {"scenario_id": scenario_id, "rows_mapped": mapped, "rows_unmapped": unmapped,
            "n_distinct_unmapped": len(miss), "unmapped": miss[:top]}


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/user-scenarios/{scenario_id}/codebook")
def read_codebook(scenario_id: str) -> dict[str, Any]:
    _get_user_scenario_or_404(scenario_id)
    cb = get_codebook(scenario_id)
    return {"scenario_id": scenario_id, "source": cb["source"], "n_nodes": len(cb["nodes"]),
            "nodes": cb["nodes"]}


@app.get("/user-scenarios/{scenario_id}/codebook/export")
def export_codebook(scenario_id: str) -> Response:
    _get_user_scenario_or_404(scenario_id)
    body = codebook_to_csv(get_codebook(scenario_id)["nodes"]).encode("utf-8-sig")
    return Response(content=body, media_type="text/csv; charset=utf-8", headers={
        "Content-Disposition": f'attachment; filename="codebook_{re.sub(r"[^A-Za-z0-9_-]", "_", scenario_id)}.csv"'})


@app.put("/user-scenarios/{scenario_id}/codebook")
def write_codebook(scenario_id: str, payload: dict[str, Any], _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Replace the scenario's codebook with `{"nodes": [...]}`."""
    _get_user_scenario_or_404(scenario_id)
    try:
        nodes = validate_nodes(payload.get("nodes"))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from None
    _store(scenario_id, nodes)
    return {"scenario_id": scenario_id, "source": "custom", "n_nodes": len(nodes)}


@app.post("/user-scenarios/{scenario_id}/codebook/import")
async def import_codebook(scenario_id: str, request: Request, _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Replace the codebook with a CSV (sheet, level1, level2, level3, synonyms, label_en,
    label_fr) sent as the request body: the reviewers' hierarchy."""
    _get_user_scenario_or_404(scenario_id)
    body = (await request.body()).decode("utf-8", "replace")
    try:
        nodes = parse_codebook_csv(body)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from None
    _store(scenario_id, nodes)
    return {"scenario_id": scenario_id, "source": "custom", "n_nodes": len(nodes)}


@app.delete("/user-scenarios/{scenario_id}/codebook")
def reset_codebook(scenario_id: str, _: None = Depends(require_api_key)) -> dict[str, Any]:
    _get_user_scenario_or_404(scenario_id)
    _store(scenario_id, None)
    return {"scenario_id": scenario_id, "source": "default", "n_nodes": len(DEFAULT_NODES)}


@app.get("/user-scenarios/{scenario_id}/codebook/unmapped")
def read_unmapped(scenario_id: str, top: int = Query(50, ge=1, le=500)) -> dict[str, Any]:
    _get_user_scenario_or_404(scenario_id)
    return unmapped_labels(scenario_id, top)


@app.post("/user-scenarios/{scenario_id}/codebook/synonym")
def add_synonym(scenario_id: str, payload: dict[str, Any], _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Map a label as a paper wrote it to a codebook node: adds it as a synonym of
    (sheet, l1, l2). The default codebook is copied into the scenario first."""
    _get_user_scenario_or_404(scenario_id)
    sheet = _SHEET_ALIASES.get(str(payload.get("sheet") or "").lower(), str(payload.get("sheet") or "").lower())
    l1, l2 = key_slug(payload.get("l1")) or None, key_slug(payload.get("l2")) or None
    label = clean_label(payload.get("label"))
    if sheet not in CODEBOOK_SHEETS or not l1 or not label:
        raise HTTPException(status_code=422, detail="sheet, l1 and label are required")
    cb = get_codebook(scenario_id)
    target = next((n for n in cb["nodes"] if n["sheet"] == sheet and n["l1"] == l1 and n.get("l2") == l2), None)
    if target is None:
        raise HTTPException(status_code=404, detail="no such node in the codebook: add it first")
    if label not in target["synonyms"]:
        target["synonyms"] = (target["synonyms"] + [label])[:_MAX_SYNONYMS]
    _store(scenario_id, cb["nodes"])
    return {"scenario_id": scenario_id, "source": "custom", "node": {"sheet": sheet, "l1": l1, "l2": l2},
            "synonyms": target["synonyms"]}
