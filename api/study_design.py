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

#: Les niveaux ci-dessus sont les VALEURS canoniques : elles voyagent entre le serveur
#: et l'interface, servent de clé de sélection et sont comparées. On ne les traduit donc
#: pas à la source ; on expose un libellé à côté, et l'interface affiche celui-là.
LEVEL_EN = {
    LEVEL_HIGH: "High",
    LEVEL_MODERATE: "Moderate",
    LEVEL_LOW: "Low",
    LEVEL_VERY_LOW: "Very low",
    LEVEL_NA: "Not applicable",
    LEVEL_UNKNOWN: "Not assessed",
}


def level_label(level: str, lang: str = "fr") -> str:
    """Le libellé affichable d'un niveau. La valeur reste celle du serveur."""
    return LEVEL_EN.get(level, level) if (lang or "fr").lower().startswith("en") else level


GRADE_NOTE = (
    "Ces niveaux sont le PLAFOND que permet le devis d'étude, avant toute appréciation du "
    "risque de biais, de la cohérence, du caractère direct, de la précision et du biais de "
    "publication. Une évaluation GRADE complète pèse ces cinq domaines et peut abaisser "
    "n'importe quelle ligne de ce tableau ; elle ne peut pas la relever, sauf justification "
    "explicite (effet de grande taille, gradient dose-réponse)."
)
GRADE_NOTE_EN = (
    "These levels are the CEILING a study design allows, before any appraisal of risk of "
    "bias, inconsistency, indirectness, imprecision and publication bias. A full GRADE "
    "assessment weighs those five domains and can lower any line of this table; it cannot "
    "raise one, save on explicit grounds (a large effect, a dose-response gradient)."
)


def grade_note(lang: str = "fr") -> str:
    return GRADE_NOTE_EN if (lang or "fr").lower().startswith("en") else GRADE_NOTE


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


#: Index inverse du libellé français vers l'anglais, pour afficher une distribution
#: dont les valeurs restent celles du serveur.
DESIGN_EN = {v["fr"]: v["en"] for v in STUDY_TYPES.values()}


def design_label(fr_label: str, lang: str = "fr") -> str:
    """Le libellé affichable d'un devis. La valeur reste celle du serveur."""
    return DESIGN_EN.get(fr_label, fr_label) if (lang or "fr").lower().startswith("en") else fr_label


#: Pourquoi un NIVEAU vaut ce qu'il vaut. Une explication par niveau, et non une par
#: devis : seize paragraphes disaient six choses, et les quinze premiers répétaient
#: « observationnel : départ en certitude faible ». Le lecteur a besoin de la règle,
#: pas de sa récitation ligne à ligne.
LEVEL_WHY = {
    LEVEL_HIGH: (
        "GRADE fait partir les essais randomisés de la certitude la plus haute.",
        "GRADE starts randomised evidence at the highest certainty."),
    LEVEL_MODERATE: (
        "Étude d'intervention dont l'étiquette ne dit pas s'il y a eu randomisation. "
        "L'inconnu est noté au-dessus de la faiblesse déclarée.",
        "An intervention study whose label does not say whether it randomised. The "
        "unknown is placed above the stated weakness."),
    LEVEL_LOW: (
        "Observationnel : GRADE part en certitude faible. Un essai qui déclare ne pas "
        "avoir randomisé est traité de même, une faiblesse déclarée pesant plus qu'une "
        "inconnue.",
        "Observational: GRADE starts at low certainty. A trial that states it did not "
        "randomise is treated the same, a stated weakness weighing more than an "
        "unknown one."),
    LEVEL_VERY_LOW: (
        "Sans groupe de comparaison, ni méthode de recherche et de sélection "
        "reproductible : ne soutient pas une estimation d'effet.",
        "No comparison group, and no reproducible search and selection method: does "
        "not support an effect estimate."),
    LEVEL_NA: (
        "Pas une preuve primaire. Une recommandation DÉRIVE d'études, et lui donner une "
        "certitude compterait deux fois celles qu'elle cite ; un modèle vaut ses "
        "paramètres d'entrée, pas son devis ; le qualitatif s'évalue par CERQual, pas "
        "par GRADE ; le préclinique ne porte pas sur une population humaine.",
        "Not primary evidence. A guideline DERIVES from studies, and giving it a "
        "certainty would count those studies twice; a model is worth its inputs, not "
        "its design; qualitative work is appraised with CERQual, not GRADE; preclinical "
        "work is not about a human population."),
    LEVEL_UNKNOWN: (
        "Aucun devis identifiable dans la notice. Explicitement NON évalué plutôt que "
        "rangé au plus bas : l'inconnu n'est pas une faiblesse mesurée.",
        "No design identifiable in the record. Explicitly NOT assessed rather than "
        "filed at the bottom: an unknown is not a measured weakness."),
}

#: Le cas de la synthèse, qui n'a pas de niveau propre.
INHERITED = ("hérité des études incluses", "inherited from the studies it includes")
INHERITED_WHY = (
    "Élevée si elle synthétise des essais randomisés, faible si elle synthétise de "
    "l'observationnel. Une revue ne relève pas ce qu'elle inclut.",
    "High if it synthesises randomised trials, low if it synthesises observational "
    "studies. A review does not upgrade what it includes.")


def _pick(pair: tuple[str, str], lang: str) -> str:
    return pair[1] if (lang or "fr").lower().startswith("en") else pair[0]


def vocabulary(lang: str = "fr") -> list[dict[str, Any]]:
    """Which design is which level, GROUPED BY LEVEL.

    This IS the plain-language explanation; there is no second copy to drift. Grouped
    rather than listed per design, because the reader needs the rule and six groups
    state it, where sixteen rows restated one of them fifteen times."""
    en = (lang or "fr").lower().startswith("en")
    groups: dict[str, dict[str, Any]] = {}
    for key, entry in STUDY_TYPES.items():
        grade = entry["grade"]
        bucket = grade or "inherited"
        g = groups.setdefault(bucket, {
            "level": grade,                       # la VALEUR, ou None pour la synthèse
            "label": (_pick(INHERITED, lang) if grade is None
                      else (LEVEL_EN.get(grade, grade) if en else grade)),
            "why": (_pick(INHERITED_WHY, lang) if grade is None
                    else _pick(LEVEL_WHY.get(grade, (entry["why_fr"], entry["why_fr"])), lang)),
            "designs": [],
        })
        g["designs"].append({"key": key, "label": label(key, lang), "mesh": entry["mesh"]})
    order = ["inherited"] + [lv for lv in LEVEL_ORDER if lv in groups]
    return [groups[k] for k in order if k in groups]


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


#: Ce que les deux extractions écrivent quand elles n'ont rien trouvé. Traité comme
#: une absence des DEUX côtés : sans cela, un « non précisé » venant d'une extraction
#: l'emportait sur un devis réel trouvé par l'autre.
_DESIGN_PLACEHOLDERS = (
    "", "non précisé", "non precise", "non précisée", "non spécifié", "non spécifiée",
    "not specified", "unspecified", "not stated", "unknown", "inconnu", "n/a", "na",
    "none", "null", "autre", "other",
)


def raw_design_sql(doc: str = "d") -> str:
    """L'expression SQL du devis BRUT d'un article, écrite UNE fois.

    Deux extractions indépendantes écrivent un devis depuis le même résumé : la passe
    PICO (`pico_json.study_design`) et la passe métadonnées (`study_design`). Aucune
    n'est meilleure que l'autre, mais chacune se tait parfois, et chacune écrit alors
    un marqueur plutôt que rien.

    Six endroits du code combinaient ces deux champs, et pas dans le même ordre : les
    graphiques du profil de preuve lisaient la colonne d'abord, le sélecteur de corpus
    lisait le PICO d'abord. Un article dont les deux passes divergent tombait donc dans
    un niveau sur le graphique et dans un autre dans la sélection, et les deux nombres
    affichés côte à côte ne s'additionnaient pas. Une seule expression, un seul devis.
    """
    def _meaningful(expr: str) -> str:
        placeholders = ", ".join(_sql_literal(p) for p in _DESIGN_PLACEHOLDERS)
        return (f"CASE WHEN lower(trim(coalesce({expr}, ''))) IN ({placeholders})"
                f" THEN NULL ELSE trim({expr}) END")
    pico = _meaningful(f"{doc}.pico_json->>'study_design'")
    column = _meaningful(f"{doc}.study_design")
    return f"COALESCE({pico}, {column}, '')"


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
