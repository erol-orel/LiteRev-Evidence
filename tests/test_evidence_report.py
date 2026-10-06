"""The brief as a citable document (api/report.py).

The commercial report we compared against is a ten-page PDF with numbered figures and a
full reference list, and it reads like a paper. Our brief held the same substance in
seventeen JSON fields and a reference list the model wrote from memory. The difference is
not cosmetic: a claim you cannot trace to a row is a claim a reviewer cannot check.

So the properties under test are the ones that make the document checkable rather than
merely handsome: a citation that resolves resolves to the right number, a citation that
does not is REPORTED and left visible, the bibliography comes from database rows, and the
report never generates a brief of its own.
"""
import json

import main
from conftest import ensure_document_columns  # noqa: E402

from api.report import build_report, cited_ids, format_reference, renumber

SID = "usr-report-test"

_ARTICLES = {
    11: {"id": 11, "title": "A trial of vaccination.", "year": 2020, "study_design": "RCT",
         "journal": "Lancet", "doi": "10.1/abc",
         "authors": "Smith J; Doe A; Roe B"},
    22: {"id": 22, "title": "A cohort study", "year": 2021, "study_design": "cohort study",
         "journal": "BMJ", "pmid": "123456", "authors": "Dupont M"},
}


# ── finding the citations ────────────────────────────────────────────────────
def test_citations_are_collected_from_prose_and_from_claims():
    """A reference list built from only one source leaves numbers in the text pointing at
    nothing."""
    brief = {
        "executive_summary": "Vaccination works [22].",
        "key_findings": ["Coverage rose [11, 22].", "No effect on mortality."],
        "claims": [{"claim": "x", "article_ids": [33]}],
    }
    assert cited_ids(brief) == [22, 11, 33]


def test_the_same_article_is_collected_once_in_order_of_appearance():
    brief = {"executive_summary": "a [11] b [11] c [22]",
             "claims": [{"claim": "x", "article_ids": [11]}]}
    assert cited_ids(brief) == [11, 22]


def test_nothing_cited_is_an_empty_list_not_an_error():
    assert cited_ids({}) == []
    assert cited_ids({"executive_summary": None, "key_findings": "not a list"}) == []
    assert cited_ids({"claims": ["junk", None, {"article_ids": ["x"]}]}) == []


# ── renumbering ──────────────────────────────────────────────────────────────
def test_a_marker_becomes_its_reference_number():
    text, missing = renumber("Vaccination works [22] and so does this [11].",
                             {22: 1, 11: 2})
    assert text == "Vaccination works [1] and so does this [2]."
    assert missing == []


def test_a_grouped_marker_is_renumbered_as_a_group():
    text, _ = renumber("Both agree [11, 22].", {11: 1, 22: 2})
    assert text == "Both agree [1, 2]."


def test_an_unresolvable_citation_is_left_visible_and_reported():
    """THE property. Deleting it would make the text read better than it is while
    destroying the evidence of why, and a reader would never know a number had been
    invented."""
    text, missing = renumber("As shown [999] and confirmed [11].", {11: 1})
    assert text == "As shown [999?] and confirmed [1]."
    assert missing == [999]


def test_an_unresolvable_citation_is_reported_once():
    _, missing = renumber("[999] again [999] and [998]", {})
    assert missing == [999, 998]


def test_text_without_citations_is_untouched():
    text, missing = renumber("No citations here at all.", {11: 1})
    assert text == "No citations here at all." and missing == []


def test_something_that_only_looks_like_a_citation_is_left_alone():
    """Brackets appear in prose. `[sic]` and `[...]` must survive."""
    for prose in ("A quote [sic] here", "An ellipsis [...] there", "A range [a-b]"):
        assert renumber(prose, {11: 1})[0] == prose


def test_renumbering_an_empty_field_does_not_crash():
    assert renumber("", {}) == ("", [])
    assert renumber(None, {}) == ("", [])


# ── the bibliography ─────────────────────────────────────────────────────────
def test_a_reference_is_built_from_the_row_not_from_the_model():
    got = format_reference(1, _ARTICLES[11])
    assert got.startswith("1. Smith J, Doe A, Roe B.")
    assert "A trial of vaccination." in got and "Lancet." in got and "2020." in got
    assert "doi:10.1/abc" in got
    assert "[#11]" in got, "the internal id is how a reviewer finds the row"


def test_a_reference_falls_back_to_the_pmid_when_there_is_no_doi():
    got = format_reference(2, _ARTICLES[22])
    assert "PMID:123456" in got and "doi:" not in got


def test_a_long_author_list_is_truncated_the_way_journals_do():
    article = {"id": 1, "title": "T", "authors": "; ".join(f"A{i} B" for i in range(10))}
    assert format_reference(1, article).count(",") >= 6
    assert "et al." in format_reference(1, article)


def test_missing_fields_are_named_rather_than_left_blank():
    """A reference with a silent hole reads as complete. "[titre non renseigné]" does
    not."""
    got = format_reference(3, {"id": 9})
    assert "[auteurs non renseignés]" in got and "[titre non renseigné]" in got


# ── the whole document ───────────────────────────────────────────────────────
def _brief():
    return {
        "executive_summary": "Vaccination reduces severe disease [11].",
        "key_findings": ["Coverage rose [22]."],
        "research_gaps": ["Nothing on children."],
        "evidence_level": "Modéré",
        "grade_recommendation": "B",
        "claims": [{
            "claim": "Vaccination reduces severe dengue",
            "reasoning": "one trial, one cohort",
            "strength": "Modéré", "article_ids": [11, 22],
            "basis": {"n_articles": 2, "designs": {"rct": 1, "cohort study": 1},
                      "from_designs": "Fort", "downgraded_single_study": False,
                      "capped_by_corpus": True},
            "unverified_ids": [404],
        }],
        "_meta": {"model": "gpt-5.6-luna", "reasoning_effort": "none",
                  "grade_ceiling": "Modéré", "generated_at": "2026-10-06T09:00:00"},
    }


def _digest():
    return {"n_articles": 1049, "n_included": 12, "n_with_pico": 900,
            "n_with_fulltext": 140, "year_min": 2004, "year_max": 2026, "complete": True}


def _matrix():
    return {"row_type": "intervention", "col_type": "outcome",
            "rows": [{"label": "vaccination", "n": 223}],
            "cols": [{"label": "severe dengue", "n": 244}, {"label": "case fatality", "n": 221}],
            "rows_total": 4, "cols_total": 4,
            "cells": [{"row": "vaccination", "col": "severe dengue", "n": 223},
                      {"row": "vaccination", "col": "case fatality", "n": 0}],
            "coverage": {"relevant": 1049, "with_concepts": 928}}


def test_the_report_states_what_the_synthesis_covers():
    """The claim that distinguishes this from a report written over a sample, so it has to
    be in the methods section in words and in figures."""
    out = build_report(_brief(), _digest(), _matrix(), None, _ARTICLES,
                       "Dengue Geneva", "dengue AND vaccine", 0.45)
    md = out["markdown"]
    assert "## 1. Méthodes" in md
    assert "totalité des 1049 articles pertinents" in md
    assert "sans échantillonnage" in md
    assert "`dengue AND vaccine`" in md and "0.45" in md
    assert "2004 à 2026" in md


def test_the_report_names_the_model_and_says_the_strengths_are_computed():
    out = build_report(_brief(), _digest(), None, None, _ARTICLES, "S", None, 0.45)
    assert "gpt-5.6-luna" in out["markdown"]
    assert "ne sont pas écrites par le modèle" in out["markdown"]


def test_prose_citations_are_renumbered_and_the_bibliography_matches():
    out = build_report(_brief(), _digest(), None, None, _ARTICLES, "S", None, 0.45)
    md = out["markdown"]
    # 11 is cited first, so it is reference 1.
    assert "Vaccination reduces severe disease [1]." in md
    assert "Coverage rose [2]." in md
    assert "1. Smith J, Doe A, Roe B." in md and "2. Dupont M." in md
    assert out["references"] == 2


def test_an_unresolvable_citation_reaches_the_reader():
    brief = _brief()
    brief["executive_summary"] = "A claim about nothing [777]."
    out = build_report(brief, _digest(), None, None, _ARTICLES, "S", None, 0.45)
    assert "[777?]" in out["markdown"]
    assert "Appels de citation non résolus" in out["markdown"]
    assert out["citations_unresolved"] == [777]


def test_the_claim_table_carries_its_basis_and_its_unverified_citations():
    out = build_report(_brief(), _digest(), None, None, _ARTICLES, "S", None, 0.45)
    md = out["markdown"]
    assert "Figure 1. Affirmations et force des preuves" in md
    assert "Vaccination reduces severe dengue" in md
    assert "plafonné par le corpus" in md
    assert "pas une évaluation GRADE complète" in md
    assert "Citations non vérifiables" in md and "404" in md


def test_the_gap_matrix_is_a_numbered_figure_with_its_denominator():
    out = build_report(_brief(), _digest(), _matrix(), None, _ARTICLES, "S", None, 0.45)
    md = out["markdown"]
    assert "Figure 2. Matrice de lacunes (intervention x outcome)" in md
    assert "| vaccination | 223 | - |" in md
    assert "aucun article de ce corpus" in md
    assert "928 des 1049" in md, "a gap figure must show what it could not read"
    assert "Axes tronqués" in md, "4 rows reported, 1 shown"
    assert out["figures"] == 2


def test_a_pipe_in_a_title_cannot_break_the_table():
    brief = _brief()
    brief["claims"][0]["claim"] = "A claim | with a pipe"
    out = build_report(brief, _digest(), None, None, _ARTICLES, "S", None, 0.45)
    assert "A claim \\| with a pipe" in out["markdown"]


def test_a_failed_digest_is_declared_rather_than_passed_off():
    digest = _digest()
    digest.update(complete=False, n_articles=0)
    out = build_report(_brief(), digest, None, None, _ARTICLES, "S", None, 0.45)
    assert "Avertissement" in out["markdown"]


def test_a_brief_with_no_citations_still_produces_a_document():
    out = build_report({"executive_summary": "No citations."}, _digest(), None, None,
                       {}, "S", None, 0.45)
    assert "Aucune citation résolue" in out["markdown"]
    assert out["references"] == 0 and out["citations_unresolved"] == []


# ── against a real database ──────────────────────────────────────────────────
def _seed(db_conn, with_brief=True):
    ensure_document_columns(db_conn.cursor())
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM literature_document WHERE id BETWEEN 8400 AND 8499")
        cur.execute("DELETE FROM scenario_settings WHERE scenario_id = %s", (SID,))
        cur.execute("INSERT INTO user_scenarios (id, name, query, created_at, updated_at) "
                    "VALUES (%s,'Dengue report','dengue AND vaccine',NOW(),NOW()) "
                    "ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name", (SID,))
        cur.execute("INSERT INTO literature_document (id, title, abstract, year, journal, "
                    "authors, doi, study_design, source, project_context, quality_score) VALUES "
                    "(8400,'Seeded trial','a',2022,'Lancet','Smith J','10.1/x','RCT','pubmed','literev',0.9)")
        cur.execute("INSERT INTO article_scenarios (document_id, scenario_id, similarity_score) "
                    "VALUES (8400,%s,0.9)", (SID,))
        if with_brief:
            brief = json.dumps({
                "executive_summary": "A seeded finding [8400].",
                "claims": [{"claim": "c", "strength": "Modéré", "article_ids": [8400],
                            "basis": {"n_articles": 1, "designs": {"rct": 1},
                                      "from_designs": "Fort",
                                      "downgraded_single_study": True,
                                      "capped_by_corpus": False}}],
                "_meta": {"model": "gpt-5.6-luna"},
            })
            cur.execute("INSERT INTO scenario_settings (scenario_id, evidence_brief_json, "
                        "brief_generated_at, updated_at) VALUES (%s, CAST(%s AS jsonb), NOW(), NOW()) "
                        "ON CONFLICT (scenario_id) DO UPDATE SET evidence_brief_json = EXCLUDED.evidence_brief_json",
                        (SID, brief))


def _cleanup(db_conn):
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM literature_document WHERE id BETWEEN 8400 AND 8499")
        cur.execute("DELETE FROM scenario_settings WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))


def test_the_endpoint_assembles_the_cached_brief(db_conn):
    from fastapi.testclient import TestClient

    _seed(db_conn)
    try:
        got = TestClient(main.app).get(f"/user-scenarios/{SID}/evidence-report").json()
        md = got["markdown"]
        assert "# Dengue report" in md
        assert "`dengue AND vaccine`" in md
        assert "A seeded finding [1]." in md, "the citation resolved against the database"
        assert "1. Smith J. Seeded trial. Lancet. 2022. doi:10.1/x [#8400]" in md
        assert got["references"] == 1 and got["citations_unresolved"] == []
    finally:
        _cleanup(db_conn)


def test_the_report_refuses_to_generate_a_brief_it_does_not_have(db_conn):
    """A GET must not quietly spend a minute of model time and a few thousand tokens."""
    from fastapi.testclient import TestClient

    _seed(db_conn, with_brief=False)
    try:
        got = TestClient(main.app).get(f"/user-scenarios/{SID}/evidence-report").json()
        assert got["status"] == "no_brief"
        assert "il n'en produit pas" in got["message"]
    finally:
        _cleanup(db_conn)


def test_the_download_is_a_markdown_attachment(db_conn):
    from fastapi.testclient import TestClient

    _seed(db_conn)
    try:
        r = TestClient(main.app).get(f"/user-scenarios/{SID}/evidence-report?download=true")
        assert r.status_code == 200
        assert "text/markdown" in r.headers["content-type"]
        assert f"evidence-report-{SID}.md" in r.headers["content-disposition"]
        assert r.text.startswith("# Dengue report")
    finally:
        _cleanup(db_conn)
