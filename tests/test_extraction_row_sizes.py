"""How much of a paper the extraction reads, and how much of each row it keeps.

A row's quote holds up to 600 characters and its description 800: at 300 and 400, a serology
table row or an adjusted odds ratio with its interval and covariates did not fit, and the cut
quote no longer held the figures it was there to prove. A paper's text is read up to 300,000
characters: at 150,000, 24 papers of the three HPAI scenarios were still cut, at the end, where
the tables sit, and a paper cut under the old limit is read again. Pure: no database, no model."""
import pytest

pytest.importorskip("fastapi")

import main  # noqa: E402,F401
from api import extraction as E  # noqa: E402


def _raw(quote: str, descr: str) -> dict:
    return {"ref": {}, "observations": [{"sheet": "human_susc", "group": "sex", "covariate": "male",
                                         "quote": quote, "descr": descr}]}


def test_a_quote_keeps_600_characters_and_a_description_800():
    quote, descr = "a" * 700, "b" * 900
    obs = E.parse_extraction(_raw(quote, descr), quote)["observations"][0]
    assert len(obs["quote"]) == 600 and len(obs["descr"]) == 800


def test_the_prompt_asks_for_the_quote_size_the_parser_keeps():
    assert f"at most {E._MAX_QUOTE_CHARS} characters" in E._EXTRACTION_SYSTEM


def test_a_paper_is_read_to_300000_characters_and_one_cut_at_150000_is_read_again():
    assert E.EXTRACTION_MAX_CHARS == 300000
    row = {"extraction_attempts": 0, "has_fulltext_chunk": True, "abstract": "x" * 200,
           "extraction_json": {"v": E.EXTRACTION_VERSION, "source": "fulltext", "truncated": True, "max_chars": 150000}}
    assert E._needs_extraction(row)
    # Read whole under today's limit: not read again.
    row["extraction_json"] = {**row["extraction_json"], "truncated": False}
    assert not E._needs_extraction(row)
