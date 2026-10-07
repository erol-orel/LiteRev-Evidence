"""Questions asked of a scenario are kept, exportable, and can change the scenario
(api/questions.py).

An answer from the assistant was a chat turn that scrolled away. It is in fact a small
piece of research: it has a scope (which scenario, which threshold, which narrowing), a
date, and a set of sources. These tests pin the three properties that follow from taking
that seriously: the scope travels with the answer, an export says the same thing in every
format, and a value an answer proposes only becomes the scenario's when a reviewer says so
and lands where the projection will actually read it.
"""
import json
import zipfile

import main
from conftest import ensure_document_columns  # noqa: E402

from api.questions import (
    describe_scope,
    diff_against_current,
    extract_parameter_claims,
    question_markdown,
)

SID = "usr-questions-test"


# ── la portée, en une phrase ─────────────────────────────────────────────────
def test_an_answer_without_a_narrowing_says_so():
    """Left blank, a reader assumes the widest reading; the phrase has to state it."""
    assert describe_scope(None, None) == "the whole relevant subset"
    assert describe_scope({}, None) == "the whole relevant subset"


def test_the_phrase_carries_the_threshold_and_every_narrowing():
    got = describe_scope({"clusters": ["wastewater"], "designs": ["Cohorte", "RCT"],
                          "relevant_only": True}, 0.62)
    assert "threshold 0.62" in got
    assert "cluster: wastewater" in got
    assert "study designs: Cohorte, RCT" in got
    assert "relevant subset only" in got


def test_a_blank_narrowing_is_not_a_narrowing():
    assert describe_scope({"clusters": ["", "  "]}, None) == "the whole relevant subset"


# ── ce qu'une réponse affirme ────────────────────────────────────────────────
def test_a_named_quantity_with_a_number_is_read():
    claims = extract_parameter_claims(
        "The basic reproduction number was estimated at 2.4 in this setting.")
    assert claims == [{"key": "r0", "unit": "", "low": None, "high": None,
                       "value": 2.4, "quote": claims[0]["quote"]}]


def test_an_interval_keeps_both_ends():
    claims = extract_parameter_claims("The incubation period ranged from 3 to 7 days.")
    assert claims[0]["key"] == "incubation_period"
    assert (claims[0]["low"], claims[0]["high"]) == (3.0, 7.0)
    assert claims[0]["value"] == 5.0


def test_quantities_do_not_leak_between_sentences():
    """A number in one sentence must not attach to a quantity named in the next."""
    claims = extract_parameter_claims(
        "Fourteen studies were included. The serial interval was 4.2 days.")
    by_key = {c["key"]: c for c in claims}
    assert by_key["serial_interval"]["value"] == 4.2


def test_a_quantity_named_without_a_number_proposes_nothing():
    assert extract_parameter_claims("The case fatality rate was not reported.") == []


def test_prose_with_no_named_quantity_proposes_nothing():
    """The most important negative: an answer is mostly text, and text is not a
    parameter. Inventing one would be worse than missing one."""
    assert extract_parameter_claims(
        "Wastewater surveillance detected signals about 5 days before clinical "
        "reporting in several European cities.") == []
    assert extract_parameter_claims("") == []


def test_the_same_quantity_cited_twice_is_proposed_once():
    claims = extract_parameter_claims(
        "R0 was 2.1. Later work put the basic reproduction number at 2.9.")
    assert [c["key"] for c in claims] == ["r0"]
    assert claims[0]["value"] == 2.1


# ── ce qui diffère de ce que le scénario tient déjà ──────────────────────────
def test_a_value_the_scenario_already_holds_is_not_a_change():
    """Showing a confirmation as a change to accept would spend the reviewer's
    attention on nothing, and they would stop reading the list."""
    claims = [{"key": "r0", "value": 2.40, "unit": "", "low": None, "high": None}]
    assert diff_against_current(claims, {"r0": {"value": 2.41}}) == []


def test_a_different_value_is_an_update_and_carries_the_old_one():
    claims = [{"key": "r0", "value": 3.6, "unit": "", "low": None, "high": None}]
    got = diff_against_current(claims, {"r0": {"value": 2.1}})
    assert got[0]["kind"] == "update" and got[0]["current"] == 2.1


def test_a_quantity_the_scenario_does_not_hold_is_new():
    claims = [{"key": "serial_interval", "value": 4.2, "unit": "days",
               "low": None, "high": None}]
    got = diff_against_current(claims, {})
    assert got[0]["kind"] == "new" and got[0]["current"] is None


# ── l'export ─────────────────────────────────────────────────────────────────
def _question(**over):
    base = {
        "question": "Which indicators lead clinical reporting?",
        "answer": "Wastewater signals lead by several days.",
        "created_at": "2026-10-07T09:30:00",
        "threshold": 0.45,
        "scope": {"clusters": ["wastewater"]},
        "scope_label": describe_scope({"clusters": ["wastewater"]}, 0.45),
        "papers_used": 1049, "papers_quoted": 24,
        "sources": [{"title": "Wastewater monitoring in Geneva", "authors": "Roe B",
                     "year": 2025, "doi": "10.1000/abc"}],
        "proposals": [{"key": "r0", "value": 2.4, "unit": "", "current": 2.1,
                       "kind": "update", "low": None, "high": None}],
    }
    base.update(over)
    return base


def test_the_export_states_the_scope_the_date_and_the_denominator():
    """Without these three, an exported answer is an unattributable paragraph."""
    md = question_markdown(_question(), scenario_name="Respiratory indicators")
    assert md.startswith("# Which indicators lead clinical reporting?")
    assert "Respiratory indicators" in md and "2026-10-07 09:30:00" in md
    assert "threshold 0.45" in md and "cluster: wastewater" in md
    assert "Answered over 1049 relevant articles, quoting 24." in md


def test_the_export_lists_its_sources():
    md = question_markdown(_question())
    assert "## Sources" in md
    assert "1. Roe B. Wastewater monitoring in Geneva. 2025. https://doi.org/10.1000/abc" in md


def test_the_export_shows_what_the_answer_proposes_and_what_is_held_today():
    md = question_markdown(_question())
    assert "## Values this answer proposes" in md
    assert "**r0**: 2.4, currently 2.1" in md


def test_a_rejected_proposal_does_not_appear_in_the_export():
    q = _question(proposals=[{"key": "r0", "value": 2.4, "current": 2.1,
                              "decision": "rejected"}])
    assert "Values this answer proposes" not in question_markdown(q)


def test_an_answer_with_no_sources_still_exports():
    md = question_markdown(_question(sources=[], proposals=[]))
    assert "## Sources" not in md and md.strip().endswith("several days.")


def test_the_word_document_is_a_real_docx():
    from api.questions import _docx_bytes
    data = _docx_bytes(question_markdown(_question()), "A question")
    assert data[:2] == b"PK"
    with zipfile.ZipFile(__import__("io").BytesIO(data)) as z:
        names = set(z.namelist())
        assert {"[Content_Types].xml", "_rels/.rels", "word/document.xml"} <= names
        doc = z.read("word/document.xml").decode("utf-8")
    assert "Which indicators lead clinical reporting?" in doc
    assert "Wastewater signals lead by several days." in doc


def test_the_word_document_escapes_what_would_break_its_xml():
    from api.questions import _docx_bytes
    data = _docx_bytes("# A & B <c>\n\nBody & more", "t")
    with zipfile.ZipFile(__import__("io").BytesIO(data)) as z:
        doc = z.read("word/document.xml").decode("utf-8")
    assert "&amp;" in doc and "&lt;c&gt;" in doc


def test_the_pdf_is_a_real_pdf():
    from api.questions import _pdf_bytes
    data = _pdf_bytes(question_markdown(_question()), "A question")
    assert data[:5] == b"%PDF-"


# ── contre une vraie base ────────────────────────────────────────────────────
def _seed(db_conn):
    ensure_document_columns(db_conn.cursor())
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))
        cur.execute("INSERT INTO user_scenarios (id, name, query, created_at, updated_at) "
                    "VALUES (%s, 'Respiratory indicators', 'wastewater', NOW(), NOW())", (SID,))
        cur.execute("DELETE FROM scenario_settings WHERE scenario_id = %s", (SID,))


def _cleanup(db_conn):
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM scenario_question WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM scenario_settings WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))


def test_a_question_is_kept_with_its_scope_and_can_be_read_back(db_conn):
    _seed(db_conn)
    try:
        saved = main.save_scenario_question(SID, main.QuestionIn(
            question="Which indicators lead clinical reporting?",
            answer="Wastewater signals lead by several days.",
            threshold=0.62, scope={"clusters": ["wastewater"]},
            sources=[{"document_id": 1, "title": "A paper", "year": 2025}],
            papers_used=1049, papers_quoted=24, digest_complete=True))
        assert saved["threshold"] == 0.62
        assert "cluster: wastewater" in saved["scope_label"]

        listed = main.list_scenario_questions(SID)
        assert listed["total"] == 1
        assert listed["items"][0]["question"].startswith("Which indicators")

        one = main.get_scenario_question(SID, saved["id"])
        assert one["answer"] == "Wastewater signals lead by several days."
        assert one["sources"][0]["title"] == "A paper"
    finally:
        _cleanup(db_conn)


def test_the_same_question_asked_twice_keeps_both_answers(db_conn):
    """The point of a history: a question re-asked after the corpus grew must sit
    beside the earlier answer rather than replace it."""
    _seed(db_conn)
    try:
        for answer in ("First reading.", "Second reading, after more articles."):
            main.save_scenario_question(SID, main.QuestionIn(
                question="Which indicators lead?", answer=answer))
        listed = main.list_scenario_questions(SID)
        assert listed["total"] == 2
        assert listed["items"][0]["answer"] == "Second reading, after more articles."
        assert all(i["asked_times_in_page"] == 2 for i in listed["items"])
    finally:
        _cleanup(db_conn)


def test_a_proposal_only_becomes_the_scenarios_when_a_reviewer_accepts_it(db_conn):
    """THE safety property: an answer may not silently rewrite a projection input."""
    _seed(db_conn)
    try:
        with db_conn.cursor() as cur:
            cur.execute("""INSERT INTO scenario_settings (scenario_id, variables_json)
                           VALUES (%s, %s)""",
                        (SID, json.dumps({"model_spec": {
                            "epidemic_parameters": {"applicable": True, "disease": "flu",
                                                    "params": {"r0": {"value": 2.1}}}}})))
        saved = main.save_scenario_question(SID, main.QuestionIn(
            question="What is R0 here?",
            answer="The basic reproduction number was estimated at 3.6."))
        proposals = saved["proposals"]
        assert [p["key"] for p in proposals] == ["r0"]
        assert proposals[0]["current"] == 2.1 and proposals[0]["value"] == 3.6
        # Rien n'a bougé tant que personne n'a décidé.
        assert main._current_parameters(SID)["r0"]["value"] == 2.1

        main.decide_proposal(SID, saved["id"],
                             main.ProposalDecision(key="r0", decision="accepted"))
        now = main._current_parameters(SID)["r0"]
        assert now["value"] == 3.6
        # La provenance accompagne la valeur, sinon ce n'est qu'un nombre.
        assert now["source"] == "assistant_answer" and now["question_id"] == saved["id"]
    finally:
        _cleanup(db_conn)


def test_rejecting_a_proposal_leaves_the_scenario_alone(db_conn):
    _seed(db_conn)
    try:
        with db_conn.cursor() as cur:
            cur.execute("""INSERT INTO scenario_settings (scenario_id, variables_json)
                           VALUES (%s, %s)""",
                        (SID, json.dumps({"model_spec": {"epidemic_parameters": {
                            "applicable": True, "params": {"r0": {"value": 2.1}}}}})))
        saved = main.save_scenario_question(SID, main.QuestionIn(
            question="R0?", answer="R0 was estimated at 3.6."))
        main.decide_proposal(SID, saved["id"],
                             main.ProposalDecision(key="r0", decision="rejected"))
        assert main._current_parameters(SID)["r0"]["value"] == 2.1
        again = main.get_scenario_question(SID, saved["id"])
        assert again["proposals"][0]["decision"] == "rejected"
    finally:
        _cleanup(db_conn)


def test_adopting_a_parameter_needs_a_model_spec_to_adopt_it_into(db_conn):
    """Writing it anywhere else would leave the projection reading the old value."""
    _seed(db_conn)
    try:
        saved = main.save_scenario_question(SID, main.QuestionIn(
            question="R0?", answer="R0 was estimated at 3.6."))
        import pytest
        with pytest.raises(Exception) as err:
            main.decide_proposal(SID, saved["id"],
                                 main.ProposalDecision(key="r0", decision="accepted"))
        assert "409" in str(getattr(err.value, "status_code", "")) or \
               getattr(err.value, "status_code", None) == 409
    finally:
        _cleanup(db_conn)


def test_a_question_can_be_deleted(db_conn):
    _seed(db_conn)
    try:
        saved = main.save_scenario_question(SID, main.QuestionIn(
            question="Anything?", answer="Yes."))
        main.delete_scenario_question(SID, saved["id"])
        assert main.list_scenario_questions(SID)["total"] == 0
    finally:
        _cleanup(db_conn)


def test_every_export_format_says_the_same_thing(db_conn):
    _seed(db_conn)
    try:
        saved = main.save_scenario_question(SID, main.QuestionIn(
            question="Which indicators lead?", answer="Wastewater leads by days.",
            threshold=0.45))
        md = main.export_scenario_question(SID, saved["id"], format="md")
        docx = main.export_scenario_question(SID, saved["id"], format="docx")
        pdf = main.export_scenario_question(SID, saved["id"], format="pdf")
        assert b"Which indicators lead?" in md.body
        assert docx.body[:2] == b"PK" and pdf.body[:5] == b"%PDF-"
        for resp, ext in ((md, "md"), (docx, "docx"), (pdf, "pdf")):
            assert resp.headers["content-disposition"].endswith(f'.{ext}"')
            assert "respiratory-indicators" in resp.headers["content-disposition"]
    finally:
        _cleanup(db_conn)
