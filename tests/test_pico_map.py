"""The per-article PICO map (api/pico_map) and the pass that completes it.

Five writers fill `pico_json`: the pipeline's PICO step, the background worker, the
per-article endpoint, the batch endpoint and scripts/extract_pico_batch.py. None of them
asked for a result, so the `key_finding` that the brief, the actions and the model spec
read was always empty, and whatever they said about findings rested on the 20 to 30
articles they reproduce. They now share one prompt that asks for the main finding and its
direction, one text (the abstract, plus the methods and results of the full text, where
the full text used to REPLACE the abstract and only its first 14,000 characters were
read), one normaliser, and one JSON call that asks again, shorter, when an answer is cut.

Pure tests first; the writers and the completing pass run against the test database with
the LLM client stubbed (skipped when no database is reachable)."""
import inspect
import json
import pathlib
import types

import pytest

import llm_usage
import main
from api import pico_map
from conftest import ensure_document_columns, patch_app  # noqa: E402

SID = "usr-findings-test"
IDS = (8601, 8602, 8603, 8604, 8605, 8606)
HDR = {"X-API-Key": "test-write-key"}


def _completion(content: str, finish: str = "stop"):
    msg = types.SimpleNamespace(content=content)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg, finish_reason=finish)])


# ── What each article is asked ───────────────────────────────────────────────
def test_every_writer_asks_for_the_finding_and_its_direction():
    prompt = pico_map.pico_system_prompt()
    assert '"key_finding"' in prompt and '"finding_direction"' in prompt
    for direction in ("increase", "decrease", "no_difference", "mixed", "descriptive", "none"):
        assert direction in prompt
    assert "em dash" in prompt                       # the house rule, asked of the model
    assert chr(0x2014) not in prompt
    # The interface language reaches the prose, never the two identifiers the code counts.
    localised = pico_map.pico_system_prompt("\n\nRESPOND ENTIRELY IN ENGLISH.")
    assert 'keep "study_design" and "finding_direction" in English' in localised
    assert "em dash" in pico_map.FINDING_SYSTEM_PROMPT


def test_a_finding_is_kept_with_a_direction_from_the_closed_list():
    nf = pico_map.normalise_finding
    assert nf("Seroprevalence was higher in exposed workers (4.2% vs 1.1%).", "Increase") == (
        "Seroprevalence was higher in exposed workers (4.2% vs 1.1%).", "increase")
    assert nf("No difference in mortality.", "no difference")[1] == "no_difference"
    assert nf("Vaccination was protective.", "protective")[1] == "decrease"
    # No result means "none", whatever the model said about its direction.
    assert nf("", "increase") == ("", "none")
    assert nf("Not reported.", "decrease") == ("", "none")
    assert nf(None, None) == ("", "none")
    # A result is never "none", and an unknown direction asserts no sense at all.
    assert nf("A prevalence of 4%.", "none") == ("A prevalence of 4%.", "descriptive")
    assert nf("Something happened.", "sideways")[1] == "descriptive"
    assert chr(0x2014) not in nf("Risk rose " + chr(0x2014) + " sharply.", "increase")[0]


def test_the_pico_is_completed_rather_than_rejected():
    pico = pico_map.normalise_pico({"P": "poultry workers", "pico_confidence": "high",
                                    "key_finding": "Seropositivity was 2%.",
                                    "finding_direction": "descriptive"}, "fulltext")
    assert pico["I"] == "" and pico["O"] == ""
    assert pico["study_design"] == "not specified"           # read as an absence downstream
    assert pico["pico_confidence"] == 0.3 and pico["pico_source"] == "fulltext"
    assert (pico["key_finding"], pico["finding_direction"]) == ("Seropositivity was 2%.", "descriptive")
    assert pico_map.normalise_pico(["not", "an", "object"]) is None


# ── What the model reads ─────────────────────────────────────────────────────
def _chunks(text: str, size: int = 4000, overlap: int = 400) -> list[str]:
    """The pipeline's chunker: sentences packed to ~4,000 characters, the last ~400
    characters of whole sentences repeated at the head of the next chunk."""
    import re
    sentences = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
    chunks, cur, cur_len = [], [], 0
    for s in sentences:
        if cur and cur_len + len(s) + 1 > size:
            chunks.append(" ".join(cur))
            keep, kept = [], 0
            for p in reversed(cur):
                if kept + len(p) > overlap:
                    break
                keep.insert(0, p)
                kept += len(p)
            cur, cur_len = keep, sum(len(p) + 1 for p in keep)
        cur.append(s)
        cur_len += len(s) + 1
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def _paper(methods_heading="2. Materials and methods", results_heading="3. Results",
           discussion_heading="4. Discussion", n_results=60) -> str:
    front = ("Avian influenza in poultry workers. J Smith, A Doe. Abstract Background H5N1 "
             "circulates. Methods We sampled workers. Results Two percent were positive. "
             "Conclusions The risk is low. ")
    intro = "Introduction " + "".join(f"Intro sentence {i} about zoonoses. " for i in range(60))
    methods = (f"{methods_heading} Study area The study took place in Vietnam. "
               + "".join(f"We recruited workers from farm {i}. " for i in range(60)))
    results = (f"{results_heading} Of the 1,234 participants, 25 were seropositive (2.0%). "
               + "".join(f"Exposure {i} to sick birds was associated with seropositivity. "
                         for i in range(n_results)))
    discussion = f"{discussion_heading} " + "".join(f"Our finding {i} agrees with earlier work. "
                                                    for i in range(60))
    references = "References " + "".join(f"{i}. Author A. A paper on birds. Journal {i}. " for i in range(20))
    return front + intro + methods + results + discussion + references


def test_the_abstract_is_always_read_and_the_full_text_adds_its_methods_and_results():
    text, source = pico_map.pico_source_text("Avian influenza", "The ABSTRACT says little.",
                                             _chunks(_paper()))
    assert source == "fulltext"
    assert "Abstract: The ABSTRACT says little." in text
    assert "25 were seropositive" in text                     # the result
    assert "We recruited workers from farm 3." in text        # the methods
    assert "Intro sentence 5 about zoonoses" not in text      # not the introduction
    assert "Our finding 7 agrees" not in text                 # nor the discussion
    assert "A paper on birds" not in text                     # nor the references


def test_a_structured_abstract_copied_at_the_top_is_not_taken_for_the_paper():
    excerpt = pico_map.fulltext_excerpt(_chunks(_paper()))
    assert excerpt.startswith("Materials and methods")
    assert "Exposure 42 to sick birds" in excerpt


def test_french_headings_are_recognised():
    excerpt = pico_map.fulltext_excerpt(_chunks(_paper(
        methods_heading="2. Matériel et méthodes", results_heading="3. Résultats",
        discussion_heading="4. Discussion")))
    assert excerpt.startswith("Matériel et méthodes")
    assert "25 were seropositive" in excerpt and "Our finding 3 agrees" not in excerpt


def test_the_reading_is_bounded_and_the_abstract_too():
    excerpt = pico_map.fulltext_excerpt(_chunks(_paper(n_results=600)))
    assert len(excerpt) <= pico_map.PICO_FULLTEXT_CHARS + 20
    assert "We recruited workers from farm 3." in excerpt     # methods keep their share
    text, _ = pico_map.pico_source_text("T", "z" * 5000, [])
    assert text.count("z") == pico_map.PICO_ABSTRACT_CHARS


def test_without_headings_the_reading_starts_after_the_introduction():
    chunks = ["Site banner, menu and cookies. " * 120,
              "".join(f"Body sentence {i}. " for i in range(200)),
              "References 1. A. 2. B. 3. C. " * 30]
    excerpt = pico_map.fulltext_excerpt(chunks)
    assert "Body sentence 150." in excerpt
    assert "Site banner" not in excerpt and "References" not in excerpt


def test_without_a_full_text_the_abstract_alone_is_read():
    text, source = pico_map.pico_source_text("T", "An abstract long enough.", [])
    assert source == "abstract" and "Full text" not in text
    assert pico_map.fulltext_excerpt(["Too short."]) == ""


def test_the_overlap_between_chunks_is_read_once():
    paper = " ".join(f"Sentence number {i} of the paper." for i in range(600))
    assert pico_map._join_chunks(_chunks(paper)) == paper


# ── A cut answer is asked again, shorter ─────────────────────────────────────
def test_a_cut_answer_is_asked_again_with_an_instruction_to_be_brief():
    calls = []

    def call(messages):
        calls.append(messages)
        return _completion('{"a": [1, 2', "length") if len(calls) == 1 else _completion('{"a": 1}')

    messages = [{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": "USER"}]
    assert json.loads(pico_map.json_completion(call, messages, "test")) == {"a": 1}
    assert len(calls) == 2
    assert calls[0][0]["content"] == "SYSTEM"                 # the caller's messages, untouched
    assert calls[1][0]["content"].startswith("SYSTEM") and "CUT OFF" in calls[1][0]["content"]
    assert calls[1][1] == messages[1]


def test_a_complete_answer_is_asked_once():
    calls = []
    out = pico_map.json_completion(lambda m: calls.append(m) or _completion('{"b": 2}'),
                                   [{"role": "user", "content": "U"}])
    assert json.loads(out) == {"b": 2} and len(calls) == 1


def test_an_answer_cut_twice_is_no_answer_and_carries_its_text():
    with pytest.raises(pico_map.CutAnswer) as raised:
        pico_map.json_completion(lambda m: _completion('{"a": [1, 2', "length"),
                                 [{"role": "user", "content": "U"}], "test")
    assert raised.value.content == '{"a": [1, 2'
    assert isinstance(raised.value, llm_usage.LLMTruncated)


def test_extract_pico_counts_an_unusable_answer_and_lets_an_outage_through():
    class _Client:
        def __init__(self, reply):
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=reply))

    with pytest.raises(pico_map.PicoUnreadable):
        pico_map.extract_pico(_Client(lambda **kw: _completion("not json")), "T", "An abstract.")
    with pytest.raises(pico_map.PicoUnreadable):
        pico_map.extract_pico(_Client(lambda **kw: _completion('{"P": "x', "length")), "T", "A.")

    def _down(**kw):
        raise RuntimeError("503 from the API")
    with pytest.raises(RuntimeError):
        pico_map.extract_pico(_Client(_down), "T", "An abstract.")


# ── Every writer goes through the map ────────────────────────────────────────
def test_every_writer_goes_through_the_shared_map():
    from api import enrichment, pipeline, review, schema_boot
    for name, fn in (("batch", enrichment.extract_pico_batch),
                     ("pipeline", pipeline._run_user_scenario_full_pipeline),
                     ("worker", schema_boot.startup_event),
                     ("per-article", review.extract_user_scenario_article_pico)):
        src = inspect.getsource(fn)
        assert "extract_pico(" in src, f"the {name} writer has its own PICO call again"
        assert "[:14000]" not in src, f"the {name} writer reads the head of the full text again"
    # The pipeline completes the older PICO right after its PICO step.
    assert "run_findings_pass(" in inspect.getsource(pipeline._run_user_scenario_full_pipeline)
    script = (pathlib.Path(__file__).resolve().parent.parent / "scripts" / "extract_pico_batch.py"
              ).read_text(encoding="utf-8")
    assert "extract_pico(" in script
    assert 'model="gpt' not in script and "max_tokens=400" not in script


def test_the_metadata_writers_ask_again_when_cut():
    from api import enrichment, pipeline
    assert "json_completion(" in inspect.getsource(enrichment.extract_metadata_batch)
    assert "json_completion(" in inspect.getsource(pipeline._run_user_scenario_full_pipeline)


# ── Integration: the writers store the fields, the pass completes the rest ───
def _engine_ok() -> bool:
    try:
        with main.engine.connect():
            return True
    except Exception:
        return False


@pytest.fixture()
def seeded(db_conn):
    """8601 above the threshold, 8602 without an abstract but with a full text (relevant only
    because a reviewer included it: a record without an abstract never enters on its score),
    8603 included by hand below it, 8604 below it, 8605 excluded, 8606 above it and already
    carrying its finding. 8601-8605 hold a PICO written before the finding existed."""
    if not _engine_ok():
        pytest.skip("main.engine cannot reach the database")
    with db_conn.cursor() as cur:
        cur.execute("SELECT to_regclass('user_scenarios') IS NULL")
        if cur.fetchone()[0]:
            main._ensure_user_scenarios_table()
        cur.execute("CREATE TABLE IF NOT EXISTS literature_document (id BIGINT PRIMARY KEY)")
        cur.execute("CREATE TABLE IF NOT EXISTS article_scenarios ("
                    "scenario_id TEXT, document_id BIGINT, PRIMARY KEY (scenario_id, document_id))")
        ensure_document_columns(cur)
        for col, typ in (("title", "TEXT"), ("abstract", "TEXT"), ("is_duplicate", "BOOLEAN DEFAULT FALSE"),
                         ("pico_attempts", "INTEGER DEFAULT 0"), ("pico_extracted_at", "TIMESTAMP"),
                         ("pico_fulltext_attempted", "BOOLEAN DEFAULT FALSE"), ("authors", "TEXT"),
                         ("doi", "TEXT"), ("journal", "TEXT")):
            cur.execute(f"ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS {col} {typ}")
        cur.execute("ALTER TABLE article_scenarios ADD COLUMN IF NOT EXISTS similarity_score DOUBLE PRECISION")
        cur.execute("SELECT to_regclass('scenario_settings') IS NULL")
        if cur.fetchone()[0]:
            main._ensure_scenario_settings_table()
        cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM scenario_settings WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))
        cur.execute("DELETE FROM document_chunk WHERE document_id = ANY(%s)", (list(IDS),))
        cur.execute("DELETE FROM literature_document WHERE id = ANY(%s)", (list(IDS),))
        cur.execute("INSERT INTO user_scenarios (id, name, query, mode, filters) "
                    "VALUES (%s, 'Avian influenza', 'h5n1', 'boolean', '{}')", (SID,))
        old_pico = json.dumps({"P": "workers", "I": "exposure", "C": "", "O": "seropositivity",
                               "study_design": "Cross-sectional study", "pico_confidence": 0.9})
        for i, abstract in ((8601, "Workers exposed to sick birds were tested for antibodies."),
                            (8602, None),
                            (8603, "A cohort of veterinarians followed for two seasons."),
                            (8604, "Below the threshold, so never read for its finding."),
                            (8605, "Set aside by a reviewer after reading the abstract."),
                            (8606, "Already carries its finding from an earlier pass.")):
            cur.execute("INSERT INTO literature_document (id, title, abstract, year, source, "
                        "project_context, is_duplicate, pico_json) VALUES "
                        "(%s, %s, %s, 2025, 'pubmed', 'literev', false, %s)",
                        (i, f"Article {i}", abstract, old_pico))
        cur.execute("UPDATE literature_document SET pico_json = pico_json || %s::jsonb WHERE id = 8606",
                    (json.dumps({"key_finding": "Known.", "finding_direction": "mixed"}),))
        cur.execute("INSERT INTO article_scenarios (scenario_id, document_id, similarity_score, screening_status) VALUES "
                    "(%s, 8601, 0.90, NULL), (%s, 8602, 0.80, 'included'), (%s, 8603, 0.10, 'included'),"
                    "(%s, 8604, 0.10, NULL), (%s, 8605, 0.95, 'excluded'), (%s, 8606, 0.70, NULL)", (SID,) * 6)
        cur.execute("INSERT INTO scenario_settings (scenario_id, similarity_threshold) VALUES (%s, 0.45) "
                    "ON CONFLICT (scenario_id) DO UPDATE SET similarity_threshold = 0.45", (SID,))
        full_text = _paper()
        for k, chunk in enumerate(_chunks(full_text)):
            cur.execute("INSERT INTO document_chunk (document_id, chunk_index, content, chunk_type) "
                        "VALUES (8602, %s, %s, 'fulltext_section')", (k, chunk))
    yield db_conn
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM scenario_settings WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))
        cur.execute("DELETE FROM document_chunk WHERE document_id = ANY(%s)", (list(IDS),))
        cur.execute("DELETE FROM literature_document WHERE id = ANY(%s)", (list(IDS),))


class _FakeLLM:
    """Stands in for llm_usage.MeteredOpenAI: answers each chat call with `reply(kw)` and
    keeps what it was sent."""

    def __init__(self, reply):
        self.calls: list[dict] = []
        outer = self

        def _create(**kw):
            outer.calls.append(kw)
            return reply(kw)
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=_create))

    def __call__(self, *a, **kw):                     # the factory call: MeteredOpenAI(...)
        return self


def _pico_of(db_conn, doc_id):
    with db_conn.cursor() as cur:
        cur.execute("SELECT pico_json, COALESCE(pico_attempts, 0) FROM literature_document WHERE id = %s", (doc_id,))
        return cur.fetchone()


def _user(kw) -> str:
    return kw["messages"][-1]["content"]


def test_the_batch_writer_stores_the_finding_and_reads_the_full_text(seeded, monkeypatch):
    with seeded.cursor() as cur:
        cur.execute("UPDATE literature_document SET pico_json = NULL, abstract = COALESCE(abstract, "
                    "'The abstract of a paper that also has its full text stored in chunks.') "
                    "WHERE id IN (8601, 8602)")
    answer = {"P": "poultry workers", "I": "exposure to sick birds", "C": "unexposed", "O": "seropositivity",
              "study_design": "Cross-sectional study", "key_finding": "2.0% of workers were seropositive.",
              "finding_direction": "Increased", "pico_confidence": 0.8}
    fake = _FakeLLM(lambda kw: _completion(json.dumps(answer)))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(llm_usage, "MeteredOpenAI", fake)
    out = main.extract_pico_batch(scenario_id=SID, scope="relevant")
    assert out["extracted"] == 2, out
    sent = {int(_user(kw).split("Article ")[1].split()[0]): kw for kw in fake.calls}
    assert '"key_finding"' in sent[8601]["messages"][0]["content"]
    assert "Full text, methods and results" in _user(sent[8602])
    assert "25 were seropositive" in _user(sent[8602]) and "Intro sentence 5" not in _user(sent[8602])
    for doc_id, source in ((8601, "abstract"), (8602, "fulltext")):
        pico, _ = _pico_of(seeded, doc_id)
        assert pico["key_finding"] == "2.0% of workers were seropositive."
        assert pico["finding_direction"] == "increase" and pico["pico_source"] == source


def test_a_writer_asks_again_when_the_answer_is_cut(seeded, monkeypatch):
    with seeded.cursor() as cur:
        cur.execute("UPDATE literature_document SET pico_json = NULL WHERE id = 8601")
    good = json.dumps({"P": "x", "key_finding": "Short.", "finding_direction": "decrease"})
    fake = _FakeLLM(lambda kw: _completion('{"P": "a very long', "length")
                    if "CUT OFF" not in kw["messages"][0]["content"] else _completion(good))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(llm_usage, "MeteredOpenAI", fake)
    assert main.extract_pico_batch(scenario_id=SID, scope="relevant")["extracted"] == 1
    pico, attempts = _pico_of(seeded, 8601)
    assert pico["finding_direction"] == "decrease" and attempts == 1
    assert len(fake.calls) == 2


def test_the_per_article_endpoint_stores_the_finding(seeded, monkeypatch):
    from fastapi.testclient import TestClient
    answer = {"P": "veterinarians", "O": "infection", "key_finding": "", "finding_direction": "increase"}
    fake = _FakeLLM(lambda kw: _completion(json.dumps(answer)))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(llm_usage, "MeteredOpenAI", fake)
    body = TestClient(main.app).post(f"/user-scenarios/{SID}/articles/8603/pico/extract", headers=HDR).json()
    assert body["extracted"] is True
    # No result reported: the direction says so, whatever the model claimed.
    assert (body["pico"]["key_finding"], body["pico"]["finding_direction"]) == ("", "none")
    assert _pico_of(seeded, 8603)[0]["finding_direction"] == "none"


def test_the_pass_completes_every_relevant_pico_and_nothing_else(seeded, monkeypatch):
    fake = _FakeLLM(lambda kw: _completion(json.dumps(
        {"key_finding": "Seropositivity was 2%.", "finding_direction": "descriptive"})))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    patch_app(monkeypatch, "_findings_client", lambda: fake)
    before = main.findings_status(SID)
    assert (before["n_relevant"], before["n_with_finding"], before["n_pending"]) == (4, 1, 3)

    out = main.run_findings_pass(SID)
    assert out == {"status": "done", "total": 3, "done": 3, "failed": 0}
    for doc_id in (8601, 8602, 8603):
        pico, _ = _pico_of(seeded, doc_id)
        assert pico["finding_direction"] == "descriptive" and pico["P"] == "workers"   # merged, not replaced
    # Below the threshold, excluded, or already done: never sent.
    for doc_id in (8604, 8605):
        assert "finding_direction" not in _pico_of(seeded, doc_id)[0]
    assert _pico_of(seeded, 8606)[0]["key_finding"] == "Known."
    # Without an abstract, the methods and results of the full text are read instead.
    by_title = {_user(kw).split("\n")[0]: kw for kw in fake.calls}
    assert "Full text, methods and results" in _user(by_title["Title: Article 8602"])
    assert _pico_of(seeded, 8602)[0]["finding_source"] == "fulltext"
    after = main.findings_status(SID)
    assert (after["n_with_finding"], after["n_pending"]) == (4, 0)
    assert main.run_findings_pass(SID)["total"] == 0                      # cached on the row


def test_an_unusable_answer_counts_an_attempt_and_an_outage_does_not(seeded, monkeypatch):
    def reply(kw):
        if "Article 8603" in _user(kw):
            raise RuntimeError("timeout")
        return _completion("this is not JSON")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    patch_app(monkeypatch, "_findings_client", lambda: _FakeLLM(reply))
    for _ in range(3):
        assert main.run_findings_pass(SID)["failed"] >= 1
    status = main.findings_status(SID)
    assert status["n_given_up"] == 2                       # 8601 and 8602, three unusable answers
    assert status["n_pending"] == 1                        # 8603: an outage costs no attempt
    assert "findings_attempts" not in _pico_of(seeded, 8603)[0]


def test_the_cap_takes_the_most_relevant_first(seeded, monkeypatch):
    fake = _FakeLLM(lambda kw: _completion(json.dumps({"key_finding": "x.", "finding_direction": "mixed"})))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    patch_app(monkeypatch, "_findings_client", lambda: fake)
    assert main.run_findings_pass(SID, max_articles=1)["total"] == 1
    # The app's relevance order puts hand inclusions first: 8602 (included, 0.80) before
    # 8601 (0.90 by score alone).
    assert _pico_of(seeded, 8602)[0]["finding_direction"] == "mixed"
    assert "finding_direction" not in _pico_of(seeded, 8601)[0]


def test_the_endpoints_need_the_write_key_to_spend(seeded, monkeypatch):
    from fastapi.testclient import TestClient
    client = TestClient(main.app)
    monkeypatch.setenv("OPENAI_API_KEY", "")
    assert client.post(f"/user-scenarios/{SID}/findings/run").status_code in (401, 403)
    run = client.post(f"/user-scenarios/{SID}/findings/run", headers=HDR).json()
    assert run["status"] == "no_llm" and (run["n_relevant"], run["n_pending"]) == (4, 3)
    status = client.get(f"/user-scenarios/{SID}/findings/status").json()
    assert status["n_with_finding"] == 1 and status["running"] is False
    assert client.get("/user-scenarios/usr-nope/findings/status").status_code == 404
