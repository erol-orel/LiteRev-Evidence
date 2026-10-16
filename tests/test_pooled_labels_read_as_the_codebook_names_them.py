"""A pooled row is named the way the codebook names it, not by its key.

The panel showed "sex gender > male", "immune status > seropositive", "test > pcr
positive": the grouping KEYS with their underscores turned into spaces. The codebook
carries a name for every one of those nodes, in English and French ("Sex or gender",
"Male", "PCR positive"), and nothing was reading it, so a reviewer met an internal
identifier where their own vocabulary should have been.

The key still groups; only the reading changes.
"""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from api.codebook import DEFAULT_NODES, Index  # noqa: E402
from api.pooling import display_names, display_path  # noqa: E402


@pytest.fixture()
def names():
    return display_names(Index([dict(n) for n in DEFAULT_NODES]), "en")


def test_a_group_and_its_value_read_as_words(names):
    assert display_path(names, "human_susc", "sex_gender", "male") == "Sex or gender > Male"
    assert display_path(names, "human_susc", "immune_status", "seropositive") == "Immune status > Seropositive"
    assert display_path(names, "animal", "test", "pcr_positive") == "Test > PCR positive"
    assert display_path(names, "human_exp", "occupation", "poultry_worker") == "Occupation > Poultry worker"
    assert display_path(names, "env", "setting", "live_bird_market") == "Setting > Live bird market"


def test_a_group_alone_is_the_group(names):
    assert display_path(names, "human_exp", "occupation", None) == "Occupation"


def test_french_is_the_french_name():
    fr = display_names(Index([dict(n) for n in DEFAULT_NODES]), "fr")
    assert display_path(fr, "human_susc", "sex_gender", "female", "fr") == "Sexe ou genre > Femme"


def test_a_label_the_codebook_does_not_know_stays_readable(names):
    """An unmapped label is the paper's own words: show them, never an empty cell."""
    assert display_path(names, "human_susc", "sex_gender", "third_gender") == "Sex or gender > third gender"
    assert display_path(names, "human_susc", "made_up_group", "made_up_value") == "made up group > made up value"


def test_the_pooled_rows_carry_the_name_beside_the_key():
    import inspect

    from api import pooling

    src = inspect.getsource(pooling.pooled_estimates)
    assert 'g["label_display"] = display_path(' in src
    assert 'c["a_display"]' in src and 'c["b_display"]' in src
    # The key is untouched: nothing downstream regroups on the display name.
    assert 'g["label"] =' not in src and 'c["a"] =' not in src


def test_the_panel_prefers_the_name_and_falls_back_to_the_key():
    from pathlib import Path

    panel = (Path(__file__).resolve().parents[1] / "frontend" / "src" / "components"
             / "PooledPanel.tsx").read_text(encoding="utf-8")
    assert "g.label_display ?? (g.label_path ?? g.label)" in panel
    assert "c.a_display ?? c.a" in panel and "c.b_display ?? c.b" in panel
