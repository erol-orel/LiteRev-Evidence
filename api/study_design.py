"""One controlled vocabulary for study designs, and one GRADE scale, for the whole app.

Three things were wrong, and all three were visible in the interface.

1. THE VOCABULARY WAS NEVER CONTROLLED. The PICO extraction prompt asked for
   `"study_design":"RCT|Cohort|Systematic review|etc"`. That `etc` let the model write
   whatever it liked, so the field was open text, and every consumer pattern-matched it
   after the fact. On a real corpus that left 266 of about 600 articles in a bucket called
   "Autre": not a classification, a residue. The prompt now asks for one of the terms
   below, and this module still parses the legacy free text so existing rows keep working.

2. A SYNTHESIS WAS GRADED BY ITS FORM, NOT BY ITS CONTENT. The old rule read
   `systematic review OR meta-analysis -> 'Forte'`, unconditionally. On the same corpus
   that produced 105 articles of "Forte" evidence out of 2 randomised trials: the strong
   evidence was 108 systematic reviews, most of them reviewing observational studies.
   GRADE does not let a review upgrade what it includes.

3. TYPE AND CERTAINTY WERE ONE ANSWER. They are two questions. "Systematic review" is a
   type; whether it supports high certainty depends on what it reviews. So `classify` and
   `grade_level` are separate functions over the same raw string, which is also why the
   interface can show one bar per type and a different distribution of levels.

WHERE THE TERMS COME FROM. The vocabulary is drawn from the NLM MeSH Publication Types,
tree V03 "Study Characteristics" (Randomized Controlled Trial, Controlled Clinical Trial,
Clinical Trial, Observational Study, Meta-Analysis, Systematic Review, Case Reports,
Guideline, Review, Editorial, Validation Study), extended with MeSH Epidemiologic Study
Characteristics (E05.318: Cohort Studies, Case-Control Studies, Cross-Sectional Studies)
because an epidemiology tool has to tell those three apart and PubMed tags them all as one
"Observational Study". Two terms have no MeSH publication type and are kept because this
corpus is full of them: modelling studies and qualitative research.

WHERE THE LEVELS COME FROM. GRADE's four certainties (high, moderate, low, very low), with
randomised trials starting high and observational studies starting low. Two labels are NOT
GRADE levels and say so: "Non applicable" for things that are not evidence of an effect (a
guideline is a recommendation, a model is an extrapolation, qualitative research is graded
by CERQual and not by GRADE) and "Non évaluée" for a design nobody stated. Marking those
"very low" would put them on a scale they do not belong on.

WHAT THIS IS NOT. Not a GRADE assessment. GRADE weighs risk of bias, inconsistency,
indirectness, imprecision and publication bias, none of which follows from a design label.
This is the CEILING a design allows before any of that is considered. `GRADE_NOTE` says so
in the interface, next to the table.
"""
from __future__ import annotations

from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# The levels
# ─────────────────────────────────────────────────────────────────────────────
#: GRADE's four, strongest first, then the two honest non-answers.
LEVEL_HIGH = "Élevée"
LEVEL_MODERATE = "Modérée"
LEVEL_LOW = "Faible"
LEVEL_VERY_LOW = "Très faible"
LEVEL_NA = "Non applicable"
LEVEL_UNKNOWN = "Non évaluée"

#: Ranking for "which of these is strongest". The two non-levels sit at the end: they are
#: not weaker evidence, they are not on the scale, and nothing may be promoted past them.
LEVEL_ORDER = (LEVEL_HIGH, LEVEL_MODERATE, LEVEL_LOW, LEVEL_VERY_LOW,
               LEVEL_NA, LEVEL_UNKNOWN)

GRADE_NOTE = (
    "Ces niveaux sont le PLAFOND que permet le devis d'étude, avant toute appréciation du "
    "risque de biais, de la cohérence, du caractère direct, de la précision et du biais de "
    "publication. Une évaluation GRADE complète pèse ces cinq domaines et peut abaisser "
    "n'importe quelle ligne de ce tableau ; elle ne peut pas la relever, sauf justification "
    "explicite (effet de grande taille, gradient dose-réponse)."
)


# ─────────────────────────────────────────────────────────────────────────────
# The rules
# ─────────────────────────────────────────────────────────────────────────────
# Ordered: the first rule whose keywords match wins, and `none_of` is checked first. The
# order IS the content, because the hard cases are all precedence:
#   "Non-randomized controlled trial"  contains "randomi"
#   "meta-analysis of cohort studies"  is a synthesis AND observational
#   "clinical trial"                   says nothing about allocation
# A keyword table without `none_of` got the first of these exactly backwards and graded a
# non-randomised trial as the strongest evidence there is.
_RULES: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    # key,                  any_of,                                           none_of
    ("synthesis", ("meta-analys", "méta-analys", "metaanalys", "systematic review",
                   "revue systématique", "revue systematique", "pooled analysis",
                   "umbrella review", "scoping review", "revue de portée"), ()),
    ("nonrandomised_trial", ("non-randomi", "non randomi", "nonrandomi", "non-randomisé",
                             "non randomisé", "quasi-exper", "quasi exper",
                             "quasi-expérim", "quasi expérim", "interrupted time series",
                             "controlled before", "avant-après", "before-after",
                             "single-arm", "bras unique"), ()),
    ("rct", ("randomi", "rct", "randomisé", "essai aléatoire"), ()),
    ("clinical_trial", ("controlled trial", "clinical trial", "essai contrôlé",
                        "essai clinique", "intervention study", "étude d'intervention",
                        "trial"), ()),
    ("case_report", ("case report", "case series", "cas clinique", "série de cas",
                     "serie de cas", "case-report"), ()),
    ("cohort", ("cohort", "cohorte", "longitudinal", "follow-up study"), ()),
    ("case_control", ("case-control", "case control", "cas-témoins", "cas temoins"), ()),
    ("cross_sectional", ("cross-sectional", "cross sectional", "transversal",
                         "prevalence survey", "enquête de prévalence"), ()),
    ("surveillance", ("surveillance", "registry", "registre", "ecological",
                      "écologique", "routine data", "administrative data",
                      "database study", "record linkage"), ()),
    # AVANT « modelling » : « murine model », « animal model » et « mouse model »
    # contiennent « model » et sont des études précliniques, pas des simulations.
    ("preclinical", ("in vitro", "in vivo", "animal", "murine", " mice", "laboratory",
                     "préclinique", "preclinical", "experimental infection"), ()),
    ("modelling", ("model", "modélis", "modelis", "simulation", "forecast", "projection",
                   "in silico", "machine learning", "apprentissage automatique",
                   "predictive", "prédictif"), ()),
    ("qualitative", ("qualitative", "interview", "entretien", "focus group",
                     "ethnograph", "grounded theory"), ()),
    ("guideline", ("guideline", "recommandation", "recommendation", "practice guideline",
                   "consensus statement", "consensus", "position statement"), ()),
    ("narrative_review", ("narrative review", "literature review", "revue narrative",
                          "revue de la littérature", "editorial", "éditorial",
                          "commentary", "commentaire", "opinion", "letter", "lettre",
                          "perspective", "viewpoint", "review", "revue"), ()),
    # Observational sans plus de précision : APRÈS les sous-types, jamais avant.
    ("observational", ("observational", "observationnel", "prospective", "retrospective",
                       "rétrospectiv", "survey", "enquête"), ()),
)

#: key -> label, GRADE ceiling, MeSH provenance, and why. `grade` is None where the level
#: depends on what the article contains rather than on its type alone (a synthesis).
STUDY_TYPES: dict[str, dict[str, Any]] = {
    "synthesis": {
        "fr": "Revue systématique / méta-analyse", "en": "Systematic review / meta-analysis",
        "mesh": "Systematic Review, Meta-Analysis (MeSH V03)", "grade": None,
        "why_fr": "Hérite du devis des études incluses : élevée si elle synthétise des "
                  "essais randomisés, faible si elle synthétise de l'observationnel. Une "
                  "revue ne relève pas ce qu'elle inclut.",
    },
    "rct": {
        "fr": "Essai contrôlé randomisé", "en": "Randomized controlled trial",
        "mesh": "Randomized Controlled Trial (MeSH V03)", "grade": LEVEL_HIGH,
        "why_fr": "GRADE fait partir les essais randomisés de la certitude la plus haute.",
    },
    "clinical_trial": {
        "fr": "Essai clinique (allocation non précisée)", "en": "Clinical trial (allocation unstated)",
        "mesh": "Clinical Trial (MeSH V03)", "grade": LEVEL_MODERATE,
        "why_fr": "Étude d'intervention dont l'étiquette ne dit pas s'il y a eu "
                  "randomisation. L'inconnu est noté au-dessus de la faiblesse déclarée.",
    },
    "nonrandomised_trial": {
        "fr": "Essai non randomisé / quasi-expérimental", "en": "Non-randomised / quasi-experimental",
        "mesh": "Controlled Clinical Trial (MeSH V03)", "grade": LEVEL_LOW,
        "why_fr": "GRADE fait partir une étude d'intervention non randomisée au même "
                  "niveau qu'une étude observationnelle. Noté SOUS l'essai clinique non "
                  "précisé : une faiblesse déclarée pèse plus qu'une inconnue.",
    },
    "cohort": {
        "fr": "Cohorte", "en": "Cohort study",
        "mesh": "Cohort Studies (MeSH E05.318)", "grade": LEVEL_LOW,
        "why_fr": "Observationnel : départ en certitude faible.",
    },
    "case_control": {
        "fr": "Cas-témoins", "en": "Case-control study",
        "mesh": "Case-Control Studies (MeSH E05.318)", "grade": LEVEL_LOW,
        "why_fr": "Observationnel : départ en certitude faible.",
    },
    "cross_sectional": {
        "fr": "Transversale", "en": "Cross-sectional study",
        "mesh": "Cross-Sectional Studies (MeSH E05.318)", "grade": LEVEL_LOW,
        "why_fr": "Observationnel : départ en certitude faible. Ne peut pas établir "
                  "d'antériorité temporelle.",
    },
    "surveillance": {
        "fr": "Surveillance / registre / écologique", "en": "Surveillance / registry / ecological",
        "mesh": "Observational Study (MeSH V03)", "grade": LEVEL_LOW,
        "why_fr": "Observationnel. Les devis écologiques portent en plus un risque "
                  "d'erreur écologique, que GRADE traiterait comme un caractère indirect.",
    },
    "observational": {
        "fr": "Observationnelle (sous-type non précisé)", "en": "Observational (subtype unstated)",
        "mesh": "Observational Study (MeSH V03)", "grade": LEVEL_LOW,
        "why_fr": "Observationnel : départ en certitude faible.",
    },
    "case_report": {
        "fr": "Cas clinique / série de cas", "en": "Case report / case series",
        "mesh": "Case Reports (MeSH V03)", "grade": LEVEL_VERY_LOW,
        "why_fr": "Sans groupe de comparaison : ne soutient pas une estimation d'effet.",
    },
    "narrative_review": {
        "fr": "Revue narrative / éditorial / avis", "en": "Narrative review / editorial / opinion",
        "mesh": "Review, Editorial, Letter (MeSH V03)", "grade": LEVEL_VERY_LOW,
        "why_fr": "Sans méthode de recherche ni de sélection reproductible.",
    },
    "guideline": {
        "fr": "Recommandation / guide de pratique", "en": "Guideline / practice guideline",
        "mesh": "Guideline, Practice Guideline (MeSH V03)", "grade": LEVEL_NA,
        "why_fr": "Une recommandation n'est pas une preuve : elle est DÉRIVÉE de preuves. "
                  "Lui attribuer une certitude compterait deux fois les études qu'elle cite.",
    },
    "modelling": {
        "fr": "Modélisation / simulation", "en": "Modelling / simulation",
        "mesh": "hors MeSH V03", "grade": LEVEL_NA,
        "why_fr": "La certitude d'un modèle est celle de ses paramètres d'entrée, pas de "
                  "son devis. À juger sur ses sources, que LiteRev extrait séparément.",
    },
    "qualitative": {
        "fr": "Qualitative", "en": "Qualitative research",
        "mesh": "Qualitative Research (MeSH, hors V03)", "grade": LEVEL_NA,
        "why_fr": "Évaluée par CERQual, pas par GRADE : la question n'est pas la taille "
                  "d'un effet. L'absence de niveau ici n'est pas une faiblesse.",
    },
    "preclinical": {
        "fr": "Expérimentale / préclinique", "en": "Experimental / preclinical",
        "mesh": "hors MeSH V03", "grade": LEVEL_NA,
        "why_fr": "In vitro ou animal : le caractère indirect vis-à-vis d'une population "
                  "humaine est tel que GRADE ne part pas d'un niveau de devis.",
    },
    "not_stated": {
        "fr": "Devis non précisé", "en": "Design not stated",
        "mesh": "-", "grade": LEVEL_UNKNOWN,
        "why_fr": "Aucun devis identifiable dans la notice. Explicitement NON évalué "
                  "plutôt que rangé au plus bas : l'inconnu n'est pas une faiblesse "
                  "mesurée, et le compter comme telle fausserait la distribution.",
    },
}

#: Ce qu'une synthèse synthétise, quand l'étiquette le dit. Testé APRÈS avoir reconnu une
#: synthèse, et seulement pour le niveau : le type reste « revue systématique ».
_SYNTHESIS_OF_TRIALS = ("randomi", "rct", "trial", "essai")
_SYNTHESIS_OF_OBSERVATIONAL = ("cohort", "cohorte", "observational", "observationnel",
                               "case-control", "cas-témoins", "cross-sectional",
                               "transversal", "prevalence", "prévalence")


def _norm(raw: str | None) -> str:
    return (raw or "").strip().lower()


def classify(raw: str | None) -> str:
    """The vocabulary key for a raw design string. Never raises, never returns None."""
    blob = _norm(raw)
    if not blob:
        return "not_stated"
    for key, any_of, none_of in _RULES:
        if none_of and any(k in blob for k in none_of):
            continue
        if any(k in blob for k in any_of):
            return key
    return "not_stated"


def label(key: str, lang: str = "fr") -> str:
    entry = STUDY_TYPES.get(key) or STUDY_TYPES["not_stated"]
    return entry.get("en" if str(lang).lower().startswith("en") else "fr", key)


def grade_level(raw: str | None) -> str:
    """The certainty ceiling a design allows, as a GRADE label or an honest non-answer.

    Separate from `classify` because type and certainty are different questions: a
    systematic review is one type, and whether it supports high certainty depends on what
    it reviews. That dependency is the whole reason the old rule inflated a corpus."""
    key = classify(raw)
    if key != "synthesis":
        return STUDY_TYPES[key]["grade"]
    blob = _norm(raw)
    # Une synthèse hérite de ce qu'elle inclut. L'observationnel est testé D'ABORD : une
    # « revue systématique d'essais et d'études de cohorte » contient les deux, et la
    # borne basse est la seule qu'on puisse défendre.
    if any(k in blob for k in _SYNTHESIS_OF_OBSERVATIONAL):
        return LEVEL_LOW
    if any(k in blob for k in _SYNTHESIS_OF_TRIALS):
        return LEVEL_HIGH
    # Ce qu'elle synthétise n'est pas dit : on ne peut pas affirmer « élevée ». C'est
    # exactement le cas qui faisait passer 108 revues pour des preuves fortes.
    return LEVEL_LOW


def strongest(levels) -> str:
    """The strongest level in an iterable, or LEVEL_UNKNOWN if there is none."""
    best = LEVEL_UNKNOWN
    for level in levels:
        if level in LEVEL_ORDER and LEVEL_ORDER.index(level) < LEVEL_ORDER.index(best):
            best = level
    return best


def weaken(level: str, steps: int = 1) -> str:
    """One step down the GRADE scale. The two non-levels do not move: they are not on it."""
    if level not in (LEVEL_HIGH, LEVEL_MODERATE, LEVEL_LOW, LEVEL_VERY_LOW):
        return level
    index = min(LEVEL_ORDER.index(level) + steps, LEVEL_ORDER.index(LEVEL_VERY_LOW))
    return LEVEL_ORDER[index]


def vocabulary(lang: str = "fr") -> list[dict[str, Any]]:
    """The table, for the interface and for the report: which design is which level, and
    why. This IS the plain-language explanation; there is no second copy to drift."""
    out = []
    for key, entry in STUDY_TYPES.items():
        out.append({
            "key": key,
            "label": label(key, lang),
            "mesh": entry["mesh"],
            "grade": entry["grade"] or "hérité des études incluses",
            "why": entry["why_fr"],
        })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# The same rules, as SQL
# ─────────────────────────────────────────────────────────────────────────────
# GENERATED from `_RULES`, not written a second time. The charts aggregate in SQL over the
# whole corpus and the claim grading runs in Python over a handful of articles; when those
# were two hand-written tables they disagreed, which is how the interface came to show a
# distribution that the grading did not believe. `tests/test_study_design.py` runs both
# over the same strings and asserts they agree.
def _sql_literal(value: str) -> str:
    """A single-quoted SQL literal, apostrophes doubled.

    Not theoretical: the keyword `étude d'intervention` and the label
    `Essai clinique (allocation non précisée)` both pass through here, and an unescaped
    apostrophe would close the literal early and leave the rest of the CASE as syntax. The
    test that counts quotes caught it. Nothing user-supplied reaches this function, but a
    generator that produces invalid SQL for its own vocabulary is a generator that will
    produce it for the next word someone adds."""
    return "'" + str(value).replace("'", "''") + "'"


def _like(column: str, keywords) -> str:
    return " OR ".join(f"{column} LIKE {_sql_literal('%' + k + '%')}" for k in keywords)


def design_case(column: str = "d") -> str:
    """CASE mapping a lowercased raw design to its French vocabulary label."""
    unknown = _sql_literal(STUDY_TYPES["not_stated"]["fr"])
    parts = [f"CASE WHEN {column} = '' THEN {unknown}"]
    for key, any_of, none_of in _RULES:
        guard = _like(column, any_of)
        if none_of:
            guard = f"({guard}) AND NOT ({_like(column, none_of)})"
        parts.append(f"WHEN {guard} THEN {_sql_literal(STUDY_TYPES[key]['fr'])}")
    parts.append(f"ELSE {unknown} END")
    return "\n        ".join(parts)


def grade_case(column: str = "d") -> str:
    """CASE mapping a lowercased raw design to its GRADE ceiling, synthesis included."""
    parts = [f"CASE WHEN {column} = '' THEN {_sql_literal(LEVEL_UNKNOWN)}"]
    synthesis_any, synthesis_none = next(
        (a, n) for k, a, n in _RULES if k == "synthesis")
    # La synthèse d'abord, et à l'intérieur l'observationnel avant les essais : même ordre
    # que `grade_level`, pour la même raison.
    parts.append(
        f"WHEN ({_like(column, synthesis_any)}) AND ({_like(column, _SYNTHESIS_OF_OBSERVATIONAL)})"
        f" THEN {_sql_literal(LEVEL_LOW)}")
    parts.append(
        f"WHEN ({_like(column, synthesis_any)}) AND ({_like(column, _SYNTHESIS_OF_TRIALS)})"
        f" THEN {_sql_literal(LEVEL_HIGH)}")
    parts.append(f"WHEN {_like(column, synthesis_any)} THEN {_sql_literal(LEVEL_LOW)}")
    for key, any_of, none_of in _RULES:
        if key == "synthesis":
            continue
        guard = _like(column, any_of)
        if none_of:
            guard = f"({guard}) AND NOT ({_like(column, none_of)})"
        parts.append(f"WHEN {guard} THEN {_sql_literal(STUDY_TYPES[key]['grade'])}")
    parts.append(f"ELSE {_sql_literal(LEVEL_UNKNOWN)} END")
    return "\n        ".join(parts)
