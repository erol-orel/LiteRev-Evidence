"""One vocabulary, one GRADE scale, and the SQL cannot disagree with the Python.

Three faults this replaces, all of them visible in the interface on a real corpus:

  - the PICO prompt asked for `"study_design":"RCT|Cohort|Systematic review|etc"`, so the
    field was open text and 266 of about 600 articles landed in a bucket called "Autre";
  - `systematic review OR meta-analysis -> 'Forte'` was unconditional, so a corpus holding
    2 randomised trials displayed 105 articles of strong evidence, which were 108
    systematic reviews of observational studies;
  - the chart's SQL and the claim grading's Python were two hand-written tables, so the
    distribution shown and the grading applied could disagree with each other.

The last of those is why `test_sql_and_python_agree` exists and runs the generated CASE
against a real Postgres: a rule written twice is a rule that drifts.
"""
import pytest

from api.study_design import (GRADE_NOTE, LEVEL_HIGH, LEVEL_LOW, LEVEL_MODERATE,
                              LEVEL_NA, LEVEL_UNKNOWN, LEVEL_VERY_LOW, STUDY_TYPES,
                              classify, design_case, grade_case, grade_level, label,
                              strongest, vocabulary, weaken)


# ── the vocabulary ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("raw,key", [
    ("Randomized controlled trial", "rct"),
    ("essai contrôlé randomisé", "rct"),
    ("RCT", "rct"),
    ("Non-randomized controlled trial", "nonrandomised_trial"),
    ("quasi-experimental study", "nonrandomised_trial"),
    ("interrupted time series", "nonrandomised_trial"),
    ("Clinical trial", "clinical_trial"),
    ("controlled trial", "clinical_trial"),
    ("Systematic review and meta-analysis", "synthesis"),
    ("meta-analysis of cohort studies", "synthesis"),
    ("scoping review", "synthesis"),
    ("Prospective cohort study", "cohort"),
    ("étude de cohorte", "cohort"),
    ("case-control study", "case_control"),
    ("cross-sectional survey", "cross_sectional"),
    ("étude transversale", "cross_sectional"),
    ("national surveillance data", "surveillance"),
    ("registry study", "surveillance"),
    ("ecological study", "surveillance"),
    ("case report", "case_report"),
    ("série de cas", "case_report"),
    ("mathematical modelling study", "modelling"),
    ("SEIR simulation", "modelling"),
    ("qualitative interviews", "qualitative"),
    ("WHO guideline", "guideline"),
    ("consensus statement", "guideline"),
    ("narrative review", "narrative_review"),
    ("editorial", "narrative_review"),
    ("in vitro study", "preclinical"),
    ("murine model of infection", "preclinical"),
    ("observational study", "observational"),
    ("", "not_stated"),
    (None, "not_stated"),
    ("   ", "not_stated"),
    ("blah", "not_stated"),
])
def test_a_raw_design_lands_on_one_vocabulary_term(raw, key):
    assert classify(raw) == key


def test_the_vocabulary_is_exhaustive_but_short():
    """A list nobody can hold in their head is a list nobody uses, and a list with a
    catch-all is a list that hides its failures. Fifteen terms, no "Autre"."""
    assert 12 <= len(STUDY_TYPES) <= 18
    assert "autre" not in {k.lower() for k in STUDY_TYPES}
    assert all({"fr", "en", "mesh", "grade", "why_fr"} <= set(v) for v in STUDY_TYPES.values())


def test_every_term_says_where_it_comes_from():
    """The user asked for official provenance. Each term names its MeSH tree, or says
    plainly that it has none."""
    for key, entry in STUDY_TYPES.items():
        assert entry["mesh"], key
        if key not in ("modelling", "preclinical", "not_stated", "qualitative"):
            assert "MeSH" in entry["mesh"], key


def test_a_classified_design_always_has_a_label_in_both_languages():
    for key in STUDY_TYPES:
        assert label(key, "fr") and label(key, "en")
        assert label(key, "fr") != key
    assert label("no-such-key") == STUDY_TYPES["not_stated"]["fr"]


# ── the precedences that were wrong ──────────────────────────────────────────
def test_a_non_randomised_trial_is_not_read_as_randomised():
    """It contains "randomi". A keyword table graded it the strongest evidence there is."""
    for raw in ("Non-randomized controlled trial", "essai non randomisé",
                "nonrandomised intervention study"):
        assert classify(raw) == "nonrandomised_trial", raw
        assert grade_level(raw) == LEVEL_LOW, raw


def test_a_synthesis_of_observational_studies_is_not_strong_evidence():
    """THE correction. 108 systematic reviews were displayed as strong evidence in a
    corpus holding 2 randomised trials."""
    assert grade_level("systematic review of cohort studies") == LEVEL_LOW
    assert grade_level("meta-analysis of case-control studies") == LEVEL_LOW
    assert grade_level("revue systématique d'études transversales") == LEVEL_LOW


def test_a_synthesis_of_trials_is_strong_evidence():
    assert grade_level("systematic review of randomised trials") == LEVEL_HIGH
    assert grade_level("meta-analysis of RCTs") == LEVEL_HIGH


def test_a_synthesis_that_does_not_say_what_it_reviews_cannot_claim_strength():
    """The common case, and the one that did the damage: a bare "systematic review"."""
    assert grade_level("systematic review") == LEVEL_LOW
    assert grade_level("meta-analysis") == LEVEL_LOW
    assert classify("systematic review") == "synthesis", "the TYPE is still a synthesis"


def test_a_mixed_synthesis_takes_the_lower_bound():
    """"A review of trials and cohort studies" contains both, and only the low bound is
    defensible."""
    assert grade_level("systematic review of trials and cohort studies") == LEVEL_LOW


def test_a_stated_weakness_ranks_below_an_unknown():
    """Deliberate: "clinical trial" says nothing about allocation and scores Modérée,
    while "non-randomised controlled trial" says it was not randomised and scores Faible.
    The direction that cannot flatter a corpus."""
    assert grade_level("clinical trial") == LEVEL_MODERATE
    assert grade_level("non-randomised clinical trial") == LEVEL_LOW


# ── the levels ───────────────────────────────────────────────────────────────
def test_randomised_trials_start_high_and_observational_starts_low():
    """GRADE's starting points, which is the whole basis of the scale."""
    assert grade_level("randomized controlled trial") == LEVEL_HIGH
    for raw in ("cohort study", "case-control study", "cross-sectional survey",
                "surveillance data", "observational study"):
        assert grade_level(raw) == LEVEL_LOW, raw


def test_things_that_are_not_evidence_of_an_effect_are_not_given_a_level():
    """A guideline is derived FROM evidence, a model's certainty is its inputs', and
    qualitative research is graded by CERQual. Calling any of them "very low" would put
    them on a scale they are not on."""
    for raw in ("WHO guideline", "SEIR modelling study", "qualitative interviews",
                "in vitro study"):
        assert grade_level(raw) == LEVEL_NA, raw


def test_an_unstated_design_is_not_evaluated_rather_than_rated_worst():
    """Counting the unknown as the lowest would skew the distribution with articles
    nobody assessed."""
    assert grade_level("") == LEVEL_UNKNOWN
    assert grade_level("blah") == LEVEL_UNKNOWN
    assert grade_level(None) == LEVEL_UNKNOWN


def test_case_reports_and_editorials_are_very_low_not_unknown():
    """These ARE on the scale, at the bottom: they are evidence, just weak."""
    assert grade_level("case series") == LEVEL_VERY_LOW
    assert grade_level("editorial") == LEVEL_VERY_LOW


def test_the_strongest_of_several_levels():
    assert strongest([LEVEL_LOW, LEVEL_HIGH, LEVEL_VERY_LOW]) == LEVEL_HIGH
    assert strongest([]) == LEVEL_UNKNOWN
    assert strongest([LEVEL_NA, LEVEL_UNKNOWN]) == LEVEL_NA


def test_weakening_moves_down_the_scale_but_not_off_it():
    assert weaken(LEVEL_HIGH) == LEVEL_MODERATE
    assert weaken(LEVEL_LOW) == LEVEL_VERY_LOW
    assert weaken(LEVEL_VERY_LOW) == LEVEL_VERY_LOW, "the bottom is the bottom"
    # The two non-levels are not on the scale, so they cannot be stepped along it.
    assert weaken(LEVEL_NA) == LEVEL_NA
    assert weaken(LEVEL_UNKNOWN) == LEVEL_UNKNOWN


# ── the explanation ──────────────────────────────────────────────────────────
def test_the_table_explains_itself():
    """The user asked for a plain explanation of which design is which level. The table
    IS it, so there is no second copy to drift from the rules.

    Grouped by LEVEL rather than listed per design: sixteen rows restated one rule
    nine times, and a reader needs the rule, not its recitation. Every design still
    appears, exactly once, under the level it maps to."""
    groups = vocabulary("fr")
    assert all(g["why"] and g["label"] for g in groups)
    seen = [d["key"] for g in groups for d in g["designs"]]
    assert sorted(seen) == sorted(STUDY_TYPES), "every design belongs to exactly one group"
    assert all(d["mesh"] for g in groups for d in g["designs"])
    synthesis = next(g for g in groups if any(d["key"] == "synthesis" for d in g["designs"]))
    assert synthesis["level"] is None and "hérité" in synthesis["label"], \
        "a synthesis has no level of its own"


def test_the_grouping_is_shorter_than_the_list_it_replaces():
    groups = vocabulary("fr")
    assert len(groups) < len(STUDY_TYPES) / 2


def test_the_table_says_it_is_not_a_full_grade_assessment():
    assert "risque de biais" in GRADE_NOTE and "cohérence" in GRADE_NOTE
    assert "plafond" in GRADE_NOTE.lower()


# ── the SQL and the Python are the same rules ────────────────────────────────
_SAMPLES = [
    "Randomized controlled trial", "Non-randomized controlled trial", "clinical trial",
    "systematic review", "systematic review of cohort studies",
    "meta-analysis of randomised trials", "prospective cohort study", "case-control study",
    "cross-sectional survey", "surveillance data", "case report", "narrative review",
    "WHO guideline", "SEIR modelling study", "qualitative interviews", "in vitro study",
    "observational study", "", "blah", "quasi-experimental", "registry study",
    "revue systématique d'études de cohorte", "essai non randomisé", "étude transversale",
]


def test_sql_and_python_agree(db_conn):
    """A rule written twice is a rule that drifts, and this pair drifted in production:
    the chart showed a distribution the claim grading did not believe. The CASE is now
    GENERATED from the same table, and this runs both over the same strings."""
    from api.study_design import _sql_literal

    with db_conn.cursor() as cur:
        for raw in _SAMPLES:
            # Aucun paramètre : les motifs LIKE sont pleins de `%`, et psycopg les lit
            # comme des marqueurs dès qu'on lui passe des paramètres. L'échantillon passe
            # donc par le même échappement que le reste du CASE, ce qui exerce en prime
            # `_sql_literal` sur des chaînes à apostrophes.
            cur.execute(f"SELECT ({design_case(_sql_literal(raw.lower()))}), "
                        f"({grade_case(_sql_literal(raw.lower()))})")
            sql_design, sql_grade = cur.fetchone()
            assert sql_design == label(classify(raw)), f"design disagrees on {raw!r}"
            assert sql_grade == grade_level(raw), f"grade disagrees on {raw!r}"


def test_the_generated_sql_is_valid_and_quotes_nothing_dangerous():
    """The CASE is interpolated into queries, so it must contain no quote that could end
    the literal early. Every keyword is written here, not supplied by a user."""
    for sql in (design_case("d"), grade_case("d")):
        assert sql.count("'") % 2 == 0
        assert ";" not in sql and "--" not in sql
