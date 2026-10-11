"""The digest and extraction blocks reach the model whole, and say how the quoted articles
were chosen.

Both blocks were cut from the end at a fixed number of characters (2,600 for the digest,
1,800 when the actions passed it, 2,200 for the extraction), in the middle of a line. A
corpus with nine concept types lost its last two or three, and the extraction block lost
its crude counts first and then its last line: the very instruction to quote the
denominator. They are now about 6,000 characters by default and cut by whole sections,
instructions first, with a "(+N more)" line naming what was left out.

And the prompts called the reproduced articles "the best established (quality,
citations)" while they are taken in the corpus's relevance order."""
import inspect

import main
from api import digest as digest_mod
from api import extraction

_TYPES = ("pathogen", "vector", "host", "population", "exposure", "intervention", "outcome",
          "method", "place")


def _big_digest() -> dict:
    """Nine concept types of eight long labels each: the case the audit measured at about
    3,300 characters, which the old 2,600-character cut lost the end of."""
    return {
        "n_articles": 1200, "complete": True, "threshold": 0.45, "n_included": 3,
        "n_with_pico": 1150, "n_with_fulltext": 300, "year_min": 2001, "year_max": 2026,
        "mean_quality": 0.52, "n_with_quality": 900, "total_citations": 5000, "n_with_citations": 800,
        "by_design": [{"value": f"design {i}", "n": 50 - i} for i in range(12)],
        "by_country": [{"value": c, "n": 10} for c in ("IT", "FR", "DE", "ES")],
        "by_journal": [{"value": f"Journal of things {i}", "n": 9} for i in range(6)],
        "by_year": [{"value": 2026 - i, "n": 40} for i in range(8)],
        "concepts": {t: [{"label": f"{t} concept {i}", "n": 30 - i} for i in range(8)] for t in _TYPES},
        "concepts_total": {t: 20 for t in _TYPES},
        "findings": {"n_read": 1100, "n_with_finding": 1000, "outcome_basis": "concepts",
                     "by_direction": {"increase": 400, "decrease": 200, "no_difference": 150,
                                      "mixed": 50, "descriptive": 200, "none": 100},
                     "by_outcome": [{"label": f"outcome {i}", "n": 100, "n_read": 90,
                                     "by_direction": {"increase": 40, "decrease": 20, "no_difference": 10,
                                                      "mixed": 5, "descriptive": 10, "none": 5}}
                                    for i in range(8)],
                     "n_outcome_labels": 30},
    }


def test_the_default_digest_block_keeps_every_concept_type_and_the_totals():
    block = main.digest_to_prompt(_big_digest())
    assert len(block) <= digest_mod.PROMPT_BLOCK_CHARS
    for t in _TYPES:
        assert f"Concepts [{t}]:" in block, t
    assert "(+12 more)" in block                      # 20 labels per type, 8 shown
    assert "(+22 more)" in block                      # 30 outcomes, 8 shown
    assert "RESULTATS sur les 1200 articles pertinents" in block


def test_a_short_digest_block_drops_whole_sections_and_says_which():
    full = main.digest_to_prompt(_big_digest(), max_chars=100000)
    small = main.digest_to_prompt(_big_digest(), max_chars=1500)
    assert len(small) <= 1500
    lines = small.split("\n")
    assert lines[0].startswith("CORPUS COMPLET: 1200 articles pertinents")
    # Every line but the marker is a whole line of the full block: nothing cut mid-line.
    assert all(line in full.split("\n") for line in lines[:-1])
    assert lines[-1].startswith("(+") and "more, left out for length" in lines[-1]
    named = [n for n in lines[-1].split(": ", 1)[1].rstrip(")").split(", ") if n.startswith("Concepts [")]
    assert named, lines[-1]
    for label in named:                               # what the marker names is really absent
        assert f"{label}:" not in "\n".join(lines[:-1])
    # What matters most comes first: the findings survive a cut the concepts do not.
    assert any(line.startswith("RESULTATS sur les 1200") for line in lines)


def test_the_header_and_the_instructions_are_never_dropped():
    out = digest_mod.fit_prompt_sections([("A", "x" * 50), ("B", "y" * 50), ("C", "z" * 50)],
                                         max_chars=60, mandatory=2)
    assert out.split("\n")[:2] == ["x" * 50, "y" * 50]
    assert out.split("\n")[2] == "(+1 more, left out for length: C)"
    assert digest_mod.fit_prompt_sections([("A", "a"), ("B", ""), ("C", "c")], 100) == "a\nc"


def _extraction_digest() -> dict:
    return {"complete": True, "n_extracted": 12, "n_relevant": 40, "n_fulltext": 5, "n_abstract": 7,
            "n_unread": 28,
            "coverage": {k: 3 for k in extraction.COVERAGE_KEYS},
            "coverage_fulltext": {k: 1 for k in extraction.COVERAGE_KEYS},
            "by_sheet": [{"sheet": "human_susc", "n_rows": 30, "n_articles": 9, "n_quote_found": 25}],
            "top_groups": [{"sheet": "human_susc", "value": f"group {i}", "n": 5} for i in range(12)],
            "crude_counts": [{"sheet": "human_exp", "covariate": f"a long covariate label {i}", "n_studies": 3,
                              "n_cases": 10.0, "pop_risk": 200.0} for i in range(12)]}


def test_the_extraction_block_puts_its_instructions_first_and_cuts_whole_sections():
    full = extraction.extraction_to_prompt(_extraction_digest())
    assert "Totaux bruts" in full and "(+" not in full          # 6,000 by default: all of it
    small = extraction.extraction_to_prompt(_extraction_digest(), max_chars=700)
    lines = small.split("\n")
    assert lines[0].startswith("EXTRACTION STRUCTUREE: 12 des 40 articles pertinents")
    assert "denominateur (12 articles extraits sur 40)" in lines[1]
    assert lines[2].startswith("28 articles pertinents ne sont pas encore extraits")
    assert lines[-1].startswith("(+") and "Totaux bruts" in lines[-1]
    assert all(line in full.split("\n") for line in lines[:-1])


def test_the_reproduced_articles_are_described_as_what_they_are():
    note = main.digest_coverage_note({"n_articles": 900, "complete": True}, 30)
    assert "TOTALITE des 900 articles" in note and "30 articles reproduits" in note
    assert "les plus proches de la question du scenario" in note
    assert "mieux" not in note
    rag = main.digest_coverage_note({"n_articles": 900, "complete": True}, 24,
                                    chosen_by=digest_mod.VERBATIM_BY_QUESTION)
    assert "les plus proches de la question posee" in rag

    from api import actions, assistant, evidence, variables
    for fn in (evidence._generate_evidence_brief_llm, actions._generate_recommended_actions,
               variables._generate_variables_from_pico):
        src = inspect.getsource(fn)
        assert "mieux établis" not in src, fn.__name__
        assert "les plus proches de la question" in src, fn.__name__
    for fn in (assistant.ask_stream_filtered, assistant.user_scenario_rag_assistant):
        assert "VERBATIM_BY_QUESTION" in inspect.getsource(fn), fn.__name__
