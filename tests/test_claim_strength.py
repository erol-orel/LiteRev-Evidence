"""A claim asserts only what its cited studies can support.

The brief used to carry ONE `evidence_level` for everything it said, so a sentence resting
on three randomised trials and one resting on a single cross-sectional survey read with
identical authority. Commercial reports grade per claim but have the model state the grade
in its own prose, which asks the thing being appraised to do the appraising.

So the strength is computed: from the designs of the articles the model cited, and capped
by the ceiling the whole corpus allows.

The DESIGN rules themselves (which design is which level, and why) live in
`api/study_design.py` and are tested in `tests/test_study_design.py`, including against
the generated SQL. This file tests only what is specific to a CLAIM: the three rules that
turn a set of cited designs into one label, and the verification of the citations.
"""
import pytest

from api.evidence import attach_claim_strength, claim_strength, corpus_ceiling
from api.study_design import (LEVEL_HIGH, LEVEL_LOW, LEVEL_MODERATE, LEVEL_NA,
                              LEVEL_ORDER, LEVEL_UNKNOWN, LEVEL_VERY_LOW)


# ── the corpus ceiling ───────────────────────────────────────────────────────
def test_the_ceiling_is_the_best_design_in_the_corpus():
    label, sentence = corpus_ceiling(["cohort study", "case-control", "RCT"])
    assert label == LEVEL_HIGH and "essais randomisés" in sentence


def test_an_observational_corpus_caps_at_low():
    label, sentence = corpus_ceiling(["cohort study", "cross-sectional", "case report"])
    assert label == LEVEL_LOW and "observationnel" in sentence


def test_a_corpus_of_reviews_of_observational_studies_does_not_cap_high():
    """The production case: 108 systematic reviews in a corpus holding 2 trials used to
    make the whole corpus look capable of strong evidence."""
    label, _ = corpus_ceiling(["systematic review of cohort studies", "meta-analysis"])
    assert label == LEVEL_LOW


def test_a_corpus_with_no_designs_at_all_says_so():
    label, sentence = corpus_ceiling([])
    assert label == LEVEL_UNKNOWN and "Non évaluable" in sentence


# ── the three claim rules ────────────────────────────────────────────────────
def test_several_trials_support_a_strong_claim():
    got = claim_strength(["RCT", "RCT", "meta-analysis of randomised trials"])
    assert got["strength"] == LEVEL_HIGH
    assert got["basis"]["n_articles"] == 3
    assert got["basis"]["downgraded_single_study"] is False


def test_a_single_study_drops_one_level():
    """One study is a result, not a body of evidence. A lone randomised trial must not
    read like a settled question."""
    got = claim_strength(["Randomized controlled trial"])
    assert got["strength"] == LEVEL_MODERATE
    assert got["basis"]["from_designs"] == LEVEL_HIGH
    assert got["basis"]["downgraded_single_study"] is True


def test_a_single_observational_study_drops_too():
    assert claim_strength(["cohort study"])["strength"] == LEVEL_VERY_LOW


def test_a_single_study_off_the_scale_is_not_stepped_down():
    """A guideline or a model has no rung to fall from, and pretending otherwise would
    quietly turn "not applicable" into "very low"."""
    for design in ("WHO guideline", "SEIR modelling study", "qualitative interviews"):
        got = claim_strength([design])
        assert got["strength"] == LEVEL_NA, design
        assert got["basis"]["downgraded_single_study"] is False, design


def test_the_strongest_cited_design_sets_the_level():
    got = claim_strength(["case report", "cohort study", "RCT"])
    assert got["basis"]["from_designs"] == LEVEL_HIGH and got["strength"] == LEVEL_HIGH


def test_the_corpus_ceiling_caps_the_claim():
    """A claim cannot be more certain than the corpus it is drawn from, whatever the model
    cited."""
    got = claim_strength(["RCT", "RCT"], ceiling=LEVEL_LOW)
    assert got["strength"] == LEVEL_LOW
    assert got["basis"]["capped_by_corpus"] is True
    assert got["basis"]["from_designs"] == LEVEL_HIGH, "the input is still reported"


def test_a_ceiling_never_raises_a_claim():
    """It is a ceiling, not a floor: a weak claim in a strong corpus stays weak."""
    got = claim_strength(["cohort study", "cohort study"], ceiling=LEVEL_HIGH)
    assert got["strength"] == LEVEL_LOW and got["basis"]["capped_by_corpus"] is False


def test_a_claim_citing_nothing_asserts_nothing():
    got = claim_strength([])
    assert got["strength"] == LEVEL_UNKNOWN
    assert got["basis"]["n_articles"] == 0
    assert "aucun article" in got["basis"]["note"]


def test_the_basis_counts_the_designs_it_saw():
    got = claim_strength(["Cohort study", "cohort study", "RCT"])
    assert got["basis"]["designs"] == {"cohort study": 2, "rct": 1}


def test_a_blank_design_is_counted_rather_than_dropped():
    """Two cited articles with no design label are two articles. Dropping them would
    quietly turn a thin claim into no claim."""
    got = claim_strength([None, ""])
    assert got["basis"]["n_articles"] == 2
    assert got["basis"]["designs"] == {"non renseigné": 2}
    assert got["strength"] == LEVEL_UNKNOWN


def test_every_strength_is_on_the_published_scale():
    for designs in ([], ["RCT"], ["cohort"], ["x"], ["RCT", "cohort"], ["guideline"]):
        assert claim_strength(designs)["strength"] in LEVEL_ORDER


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
    assert got[0]["strength"] == LEVEL_HIGH


def test_a_claim_citing_only_invented_articles_asserts_nothing():
    """A confident sentence whose every citation is fabricated would otherwise read as
    confidently as a real one."""
    got = attach_claim_strength(
        [{"claim": "Something sweeping", "article_ids": [404, 405]}], _CORPUS)
    assert got[0]["strength"] == LEVEL_UNKNOWN
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
                                 _CORPUS)[0]["strength"] == LEVEL_UNKNOWN


def test_the_ceiling_reaches_every_claim():
    got = attach_claim_strength(
        [{"claim": "a", "article_ids": [11, 11]}, {"claim": "b", "article_ids": [22]}],
        _CORPUS, ceiling=LEVEL_LOW)
    assert [c["strength"] for c in got] == [LEVEL_LOW, LEVEL_VERY_LOW]


# ── the wiring the pure tests cannot see ─────────────────────────────────────
def test_the_generator_grades_the_claims_itself():
    import inspect

    from api import evidence

    src = inspect.getsource(evidence._generate_evidence_brief_llm)
    assert 'brief["claims"] = attach_claim_strength(' in src
    assert "corpus_ceiling(" in src


def test_the_prompt_asks_for_citable_ids_and_forbids_self_grading():
    import inspect

    from api import evidence

    src = inspect.getsource(evidence._generate_evidence_brief_llm)
    assert '"article_ids"' in src
    assert "N'INDIQUE PAS de niveau de preuve" in src
    assert '"id": a.get("id")' in src


def test_the_module_no_longer_carries_its_own_design_table():
    """It did, and `api/documents.py` carried a second one in SQL. They disagreed."""
    import inspect

    from api import evidence

    src = inspect.getsource(evidence)
    assert "from .study_design import" in src
    assert "_NOT_RANDOMISED = (" not in src and "_DESIGN_LEVELS = (" not in src
