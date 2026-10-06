"""A claim asserts only what its cited studies can support.

The brief used to carry ONE `evidence_level` for everything it said, so a sentence resting
on three randomised trials and one resting on a single cross-sectional survey read with
identical authority. Commercial reports grade per claim but have the model state the grade
in its own prose ("Strong - repeated across reviews"), which asks the thing being appraised
to do the appraising.

So the strength is computed, from the designs of the articles the model cited and the
ceiling the whole corpus allows, by rules a reader can check. These tests are those rules.
They are pure: no LLM, no database.

What is deliberately NOT claimed anywhere here: that this is GRADE. GRADE weighs risk of
bias, inconsistency, indirectness, imprecision and publication bias, none of which follows
from a design label.
"""
import pytest

from api.evidence import (_STRENGTH_ORDER, attach_claim_strength, claim_strength,
                          corpus_ceiling, design_level)


# ── what one design can support ──────────────────────────────────────────────
@pytest.mark.parametrize("design,expected", [
    ("Randomized controlled trial", "Fort"),
    ("essai randomisé contrôlé", "Fort"),
    ("Meta-analysis", "Fort"),
    ("Systematic review and meta-analysis of RCTs", "Fort"),
    ("revue systématique", "Fort"),
    ("Non-randomized controlled trial", "Faible"),
    ("quasi-experimental study", "Faible"),
    ("Prospective cohort study", "Faible"),
    ("étude de cohorte rétrospective", "Faible"),
    ("case-control study", "Faible"),
    ("cross-sectional survey", "Faible"),
    ("étude transversale", "Faible"),
    ("surveillance data", "Faible"),
    ("ecological study", "Faible"),
])
def test_a_design_supports_what_its_kind_supports(design, expected):
    assert design_level(design) == expected


@pytest.mark.parametrize("design", [
    None, "", "   ", "Case report", "série de cas", "Expert opinion", "editorial",
    "modelling study", "Non classifié", "unknown",
])
def test_anything_unrecognised_supports_nothing(design):
    """Most of a corpus has no usable design label. Reading those as adequate is how an
    observational corpus comes to carry a strong recommendation."""
    assert design_level(design) == "Insuffisant"


def test_a_non_randomised_trial_is_not_read_as_a_randomised_one():
    """The inversion a keyword table produced and this test caught: "Non-randomized
    controlled trial" contains "randomi", so it graded as the strongest evidence there is.
    The negation has to be tested before the thing it negates.

    They land on "Faible", the same as a quasi-experimental study, because they are the
    same thing described twice and GRADE starts a non-randomised study of an intervention
    at low certainty."""
    for design in ("Non-randomized controlled trial", "non randomised trial",
                   "essai non randomisé", "nonrandomized intervention study",
                   "quasi-experimental study"):
        assert design_level(design) == "Faible", design


def test_a_stated_absence_of_randomisation_scores_below_an_unstated_one():
    """Deliberate, and worth pinning: "clinical trial" says nothing about allocation and
    scores Modéré, while "non-randomised controlled trial" says it was not randomised and
    scores Faible. Rating the known weakness below the unknown is the direction that
    cannot flatter a corpus."""
    assert design_level("clinical trial") == "Modéré"
    assert design_level("non-randomised clinical trial") == "Faible"


def test_a_synthesis_inherits_from_what_it_includes():
    """GRADE does not let a review upgrade its inputs. A meta-analysis of cohort studies
    is observational evidence that has been pooled, not trial evidence, and reading it as
    "Fort" is how a corpus of observational reviews comes to carry a strong
    recommendation."""
    assert design_level("systematic review and meta-analysis of randomised trials") == "Fort"
    assert design_level("meta-analysis of cohort studies") == "Faible"
    assert design_level("systematic review of observational studies") == "Faible"
    assert design_level("revue systématique d'études de cohorte") == "Faible"


def test_a_trial_that_does_not_say_randomised_is_only_moderate():
    """"Controlled trial" is an intervention study; nothing in the label says anyone was
    randomised, and assuming it would be generous in the one direction that matters."""
    assert design_level("controlled trial") == "Modéré"
    assert design_level("clinical trial") == "Modéré"
    assert design_level("randomized controlled trial") == "Fort"


# ── the corpus ceiling ───────────────────────────────────────────────────────
def test_the_ceiling_is_the_best_design_in_the_corpus():
    label, sentence = corpus_ceiling(["cohort study", "case-control", "RCT"])
    assert label == "Fort" and "essais randomisés" in sentence


def test_an_observational_corpus_caps_at_faible():
    label, sentence = corpus_ceiling(["cohort study", "cross-sectional", "case report"])
    assert label == "Faible" and "observationnel" in sentence


def test_a_corpus_with_no_designs_at_all_says_so():
    label, sentence = corpus_ceiling([])
    assert label == "Insuffisant" and "Insuffisante" in sentence
    assert corpus_ceiling(["Non classifié", ""])[0] == "Insuffisant"


# ── grading one claim ────────────────────────────────────────────────────────
def test_several_trials_support_a_strong_claim():
    got = claim_strength(["RCT", "RCT", "meta-analysis"])
    assert got["strength"] == "Fort"
    assert got["basis"]["n_articles"] == 3
    assert got["basis"]["downgraded_single_study"] is False


def test_a_single_study_drops_one_level():
    """One study is a result, not a body of evidence. A lone randomised trial must not
    read like a settled question."""
    got = claim_strength(["Randomized controlled trial"])
    assert got["strength"] == "Modéré"
    assert got["basis"]["from_designs"] == "Fort"
    assert got["basis"]["downgraded_single_study"] is True


def test_a_single_observational_study_drops_too():
    assert claim_strength(["cohort study"])["strength"] == "Insuffisant"


def test_the_strongest_cited_design_sets_the_level():
    got = claim_strength(["case report", "cohort study", "RCT"])
    assert got["basis"]["from_designs"] == "Fort" and got["strength"] == "Fort"


def test_the_corpus_ceiling_caps_the_claim(caplog):
    """THE rule that matters: a claim cannot be more certain than the corpus it is drawn
    from, whatever the model cited. If an observational corpus produces a claim citing
    something the model believes is a trial, the ceiling still holds."""
    got = claim_strength(["RCT", "RCT"], ceiling="Faible")
    assert got["strength"] == "Faible"
    assert got["basis"]["capped_by_corpus"] is True
    assert got["basis"]["from_designs"] == "Fort", "the input is still reported"


def test_a_claim_citing_nothing_asserts_nothing():
    got = claim_strength([])
    assert got["strength"] == "Insuffisant"
    assert got["basis"]["n_articles"] == 0
    assert "aucun article" in got["basis"]["note"]


def test_the_basis_counts_the_designs_it_saw():
    got = claim_strength(["Cohort study", "cohort study", "RCT"])
    assert got["basis"]["designs"] == {"cohort study": 2, "rct": 1}


def test_a_blank_design_is_counted_rather_than_dropped():
    """Two cited articles with no design label are two articles, and the claim is graded
    on two. Dropping them would quietly turn a thin claim into no claim."""
    got = claim_strength([None, ""])
    assert got["basis"]["n_articles"] == 2
    assert got["basis"]["designs"] == {"non renseigné": 2}
    assert got["strength"] == "Insuffisant"


def test_every_strength_is_one_of_the_four_labels():
    for designs in ([], ["RCT"], ["cohort"], ["x"], ["RCT", "cohort"]):
        assert claim_strength(designs)["strength"] in _STRENGTH_ORDER


# ── verifying the citations ──────────────────────────────────────────────────
_CORPUS = {
    11: {"id": 11, "title": "A trial", "year": 2020, "study_design": "RCT"},
    22: {"id": 22, "title": "A cohort", "year": 2021, "study_design": "cohort study"},
}


def test_a_claim_keeps_only_citations_the_corpus_can_confirm():
    got = attach_claim_strength(
        [{"claim": "Vaccination reduces incidence", "article_ids": [11, 22, 999],
          "reasoning": "two designs"}], _CORPUS)
    assert got[0]["article_ids"] == [11, 22]
    assert got[0]["unverified_ids"] == [999], "an invented id is reported, not swallowed"
    assert got[0]["strength"] == "Fort"


def test_a_claim_citing_only_invented_articles_asserts_nothing():
    """The failure mode worth catching: a confident sentence whose every citation is
    fabricated would otherwise read as confidently as a real one."""
    got = attach_claim_strength(
        [{"claim": "Something sweeping", "article_ids": [404, 405]}], _CORPUS)
    assert got[0]["strength"] == "Insuffisant"
    assert got[0]["article_ids"] == [] and got[0]["unverified_ids"] == [404, 405]


def test_ids_that_are_not_numbers_are_unverified_not_crashes():
    got = attach_claim_strength(
        [{"claim": "x", "article_ids": ["Smith et al. 2020", None, 11]}], _CORPUS)
    assert got[0]["article_ids"] == [11]
    assert got[0]["unverified_ids"] == ["Smith et al. 2020", None]


def test_a_string_id_the_model_quoted_as_text_still_resolves():
    """Models return `"11"` as often as `11`, and a citation that resolves must resolve."""
    got = attach_claim_strength([{"claim": "x", "article_ids": ["11"]}], _CORPUS)
    assert got[0]["article_ids"] == [11] and "unverified_ids" not in got[0]


def test_each_claim_carries_the_articles_a_reader_can_open():
    got = attach_claim_strength([{"claim": "x", "article_ids": [11]}], _CORPUS)
    assert got[0]["articles"] == [
        {"id": 11, "title": "A trial", "year": 2020, "study_design": "RCT"}]


def test_a_claim_with_no_text_is_not_a_claim():
    assert attach_claim_strength([{"article_ids": [11]}, {"claim": "   "}], _CORPUS) == []


def test_junk_in_the_claims_array_does_not_break_the_brief():
    """The whole brief is one `except Exception` away from being lost, so a malformed
    claims array must cost the claims and nothing else."""
    assert attach_claim_strength(None, _CORPUS) == []
    assert attach_claim_strength(["a string", 42, None], _CORPUS) == []
    assert attach_claim_strength([{"claim": "ok", "article_ids": "not a list"}],
                                 _CORPUS)[0]["strength"] == "Insuffisant"


def test_the_ceiling_reaches_every_claim():
    got = attach_claim_strength(
        [{"claim": "a", "article_ids": [11, 11]}, {"claim": "b", "article_ids": [22]}],
        _CORPUS, ceiling="Faible")
    assert [c["strength"] for c in got] == ["Faible", "Insuffisant"]


# ── the wiring the pure tests cannot see ─────────────────────────────────────
def test_the_generator_grades_the_claims_itself():
    """The pure rules are worth nothing if the generated brief keeps the model's own
    claims array. The grading has to happen after the model and outside its reach."""
    import inspect

    from api import evidence

    src = inspect.getsource(evidence._generate_evidence_brief_llm)
    assert 'brief["claims"] = attach_claim_strength(' in src
    assert "corpus_ceiling(" in src, "the ceiling must come from the whole corpus"


def test_the_prompt_asks_for_citable_ids_and_forbids_self_grading():
    """Two instructions carry the whole design: cite ids that exist, and do not state a
    strength. Without the first there is nothing to verify; without the second the model
    grades its own work."""
    import inspect

    from api import evidence

    src = inspect.getsource(evidence._generate_evidence_brief_llm)
    assert '"article_ids"' in src
    assert "N'INDIQUE PAS de niveau de preuve" in src
    # And the reproduced articles must carry the id the prompt asks it to cite.
    assert '"id": a.get("id")' in src
