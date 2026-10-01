"""The gold standard's arithmetic, which will produce numbers printed in a paper.

Everything here is pure: the sampling and the workbooks need a database and a person, but
the confusion matrix and the agreement statistic do not, and those are the two things a
reviewer will recompute. A precision that is wrong by one cell is a wrong claim about how
well the extraction works.

The cell this file exists for is `mention`. An abstract that NAMES a parameter without
giving a value is the keyword screen's characteristic false positive, and an extraction
that returns a number there has invented one. Counting mention as "reported" would turn
every invention into a true positive and hide the one failure mode worth measuring.
"""
import importlib.util
import pathlib
import sys

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "gold_standard", pathlib.Path(__file__).resolve().parent.parent / "scripts" / "gold_standard.py")
gs = importlib.util.module_from_spec(_SPEC)
sys.modules["gold_standard"] = gs
_SPEC.loader.exec_module(gs)


def _gold(**per_id):
    """{1: ("value", 2.0)} as the scorer wants it. The keys arrive as strings through
    **kwargs, and the article ids are integers."""
    return {int(i): {"values": {"r0": v}} for i, v in per_id.items()}


def test_a_value_both_sides_agree_on_is_a_true_positive():
    r = gs.score_parameter(_gold(**{"1": ("value", 2.0)}), {1: {"r0": 2.05}}, [1], "r0")
    assert (r["tp"], r["fp"], r["fn"]) == (1, 0, 0)
    assert r["precision"] == 1.0 and r["recall"] == 1.0
    assert r["within_tolerance"] == 1.0            # 2.5% off, inside the 10% tolerance


def test_an_invented_value_on_a_mention_is_a_false_positive():
    """THE case. The abstract says "we discuss the reproduction number" and gives no
    number; the extraction returns 2.4. That is a fabrication, and it must land in fp."""
    r = gs.score_parameter(_gold(**{"1": ("mention", None)}), {1: {"r0": 2.4}}, [1], "r0")
    assert (r["tp"], r["fp"], r["fn"]) == (0, 1, 0)
    assert r["precision"] == 0.0


def test_an_invented_value_where_the_abstract_is_silent_is_also_a_false_positive():
    r = gs.score_parameter(_gold(**{"1": ("none", None)}), {1: {"r0": 2.4}}, [1], "r0")
    assert (r["tp"], r["fp"], r["fn"]) == (0, 1, 0)


def test_a_value_the_extraction_did_not_find_is_a_false_negative():
    r = gs.score_parameter(_gold(**{"1": ("value", 2.0)}), {}, [1], "r0")
    assert (r["tp"], r["fp"], r["fn"]) == (0, 0, 1)
    assert r["recall"] == 0.0 and r["precision"] is None     # nothing was predicted


def test_correctly_staying_silent_counts_nowhere():
    """A true negative is not in the table, which is correct: precision and recall do not
    use it, and counting it would inflate an accuracy nobody should report here."""
    r = gs.score_parameter(_gold(**{"1": ("none", None), "2": ("mention", None)}), {}, [1, 2], "r0")
    assert (r["tp"], r["fp"], r["fn"]) == (0, 0, 0)
    assert r["precision"] is None and r["recall"] is None and r["f1"] is None


def test_a_disputed_cell_is_excluded_rather_than_guessed():
    """When the two annotators disagreed the cell is not evidence. Scoring it either way
    would turn a disagreement between people into a verdict on the model."""
    gold = {1: {"values": {"r0": ("disputed", None)}}, 2: {"values": {"r0": ("value", 3.0)}}}
    r = gs.score_parameter(gold, {1: {"r0": 9.9}, 2: {"r0": 3.0}}, [1, 2], "r0")
    assert (r["tp"], r["fp"], r["fn"]) == (1, 0, 0)
    assert r["n_values_compared"] == 1


def test_a_value_outside_the_tolerance_is_still_a_true_positive_but_counted_wrong():
    """Found-the-right-parameter and got-the-right-number are two questions. An
    extraction that correctly spots an R0 and reads 6.0 for 2.0 has not failed at
    detection, and reporting it as a false positive would hide a detector that works
    behind an arithmetic error that does not."""
    gold = {1: {"values": {"r0": ("value", 2.0)}}, 2: {"values": {"r0": ("value", 2.0)}}}
    r = gs.score_parameter(gold, {1: {"r0": 2.0}, 2: {"r0": 6.0}}, [1, 2], "r0")
    assert (r["tp"], r["fp"], r["fn"]) == (2, 0, 0)
    assert r["recall"] == 1.0
    assert r["within_tolerance"] == 0.5            # ... and half the values are wrong
    assert r["median_relative_error"] == 2.0


def test_precision_recall_and_f1_on_a_mixed_sample():
    gold = {
        1: {"values": {"r0": ("value", 2.0)}},     # found      -> tp
        2: {"values": {"r0": ("value", 3.0)}},     # found      -> tp
        3: {"values": {"r0": ("value", 4.0)}},     # missed     -> fn
        4: {"values": {"r0": ("mention", None)}},  # invented   -> fp
        5: {"values": {"r0": ("none", None)}},     # silent, correctly
    }
    pred = {1: {"r0": 2.0}, 2: {"r0": 3.0}, 4: {"r0": 1.5}}
    r = gs.score_parameter(gold, pred, [1, 2, 3, 4, 5], "r0")
    assert (r["tp"], r["fp"], r["fn"]) == (2, 1, 1)
    assert r["precision"] == round(2 / 3, 3)
    assert r["recall"] == round(2 / 3, 3)
    assert r["f1"] == round(2 / 3, 3)


def test_kappa_is_not_percent_agreement():
    """Two annotators who both say "not reported" 98 times out of 100 agree 98% of the
    time while agreeing on almost nothing. Reporting the raw agreement for a rare
    parameter would make the annotation look far better than it is."""
    a = [True] * 2 + [False] * 98
    b = [True] * 1 + [False] * 1 + [False] * 98        # they agree on one of the two
    assert gs._kappa(a, b) == pytest.approx(0.656, abs=0.01)
    assert sum(1 for x, y in zip(a, b) if x == y) / 100 == 0.99


def test_kappa_is_undefined_rather_than_perfect_when_nobody_ever_said_yes():
    """A parameter no article reports gives 100% agreement and no information. Returning
    1.0 would publish a perfect agreement that was never tested."""
    assert gs._kappa([False] * 50, [False] * 50) is None
    assert gs._kappa([], []) is None


def test_kappa_is_one_on_perfect_agreement_and_zero_at_chance():
    assert gs._kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert gs._kappa([True, True, False, False], [True, False, True, False]) == 0.0


def test_the_extraction_output_is_inverted_to_one_row_per_article():
    out = gs._extraction_by_article(
        {"params": {"r0": {"observations": [{"article_id": 7, "value": 2.4},
                                            {"article_id": 8, "value": 1.1}]},
                    "cfr": {"observations": [{"article_id": 7, "value": 0.015}]},
                    "not_a_parameter": {"observations": [{"article_id": 7, "value": 1}]}}},
        ["r0", "cfr"])
    assert out == {7: {"r0": 2.4, "cfr": 0.015}, 8: {"r0": 1.1}}


def test_a_malformed_observation_is_dropped_rather_than_crashing_the_score():
    out = gs._extraction_by_article(
        {"params": {"r0": {"observations": [{"value": 2.4},                  # no id
                                            {"article_id": "x", "value": 1}, # bad id
                                            {"article_id": 9, "value": 3.0}]}}},
        ["r0"])
    assert out == {9: {"r0": 3.0}}
