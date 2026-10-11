"""The structured extraction reads the whole paper, and an outage costs no article its turn.

On HPAI_last, 5 rich papers were given up after three answers cut at 8,000 tokens, 149 were
read from a text cut at 60,000 characters (minus the chunk overlaps read twice, and with the
reference list read instead of the tables at the end), an outage counted as one of an article's
three attempts, 1,462 values had a quote "not found" that a ligature or a word split at a line
break explained, and rows whose sheet was spelled "Human_Susc" vanished without a count. These
tests pin each of those, with a fake client and the extraction test database."""
import io
import json
import types

import pytest

pytest.importorskip("fastapi")

import llm_usage  # noqa: E402
import main  # noqa: E402
from api import extraction  # noqa: E402
from conftest import patch_app  # noqa: E402
from test_extraction import HDR, PAPER, SID, _fake_client, _raw, _run, _stored, extracted, seeded  # noqa: E402,F401

LIG_FI = chr(0xFB01)
SOFT = chr(0x00AD)
EN_DASH, EM_DASH = chr(0x2013), chr(0x2014)
LQ, RQ = chr(0x201C), chr(0x201D)


# ── C14: the quote check undoes what a PDF does to text ─────────────────────
def test_a_ligature_a_soft_hyphen_a_word_split_at_a_line_break_and_curly_quotes_still_match():
    text_ = (f"The {LIG_FI}rst cases were con{SOFT}firmed in trans-\nmission studies; the {LQ}culling"
             f"{RQ} team (n = 12{EN_DASH}15) was VACCINATED in Göttingen.")
    assert extraction.quote_in_text("The first cases were confirmed in transmission studies", text_)
    assert extraction.quote_in_text('the "culling" team (n = 12-15) was vaccinated', text_)
    assert extraction.quote_in_text("vaccinated in Gottingen", text_)               # accents folded on both sides
    # A real compound at a line break still reads as written.
    assert extraction.quote_in_text("COVID-19 cases", "Most COVID-\n19 cases were mild.")
    assert extraction.quote_in_text("pre- and post-exposure", "pre- and post-exposure prophylaxis")
    # The check still refuses other words or other figures, and stays a boolean.
    assert extraction.quote_in_text("The first cases were refuted", text_) is False
    assert extraction.quote_in_text("n = 12-16", text_) is False
    assert extraction.quote_in_text(None, text_) is False and extraction.quote_in_text("", text_) is False


def test_the_text_is_folded_once_per_article_not_once_per_row():
    extraction._haystack.cache_clear()
    big = PAPER * 50
    for q in ("7 of 7 were not vaccinated", "Seventeen exposed persons", "No antibodies against H5N1"):
        assert extraction.quote_in_text(q, big)
    # Found as written: the text was folded once, and the hyphen-joined reading never needed.
    assert extraction._haystack.cache_info().misses == 1
    assert not extraction.quote_in_text("an invented sentence", big)
    assert not extraction.quote_in_text("another invented sentence", big)
    assert extraction._haystack.cache_info().misses == 2


# ── C15: a sheet spelled another way is read, and what is dropped is counted ─
def test_sheets_written_another_way_are_read_and_the_rows_still_dropped_are_counted():
    raw = _raw()
    raw["observations"] = [
        {**raw["observations"][0], "sheet": "Human_Exp"},
        {**raw["observations"][0], "sheet": "HUMAN COV SUSC", "covariate": "male"},
        {**raw["observations"][0], "sheet": "Environment", "covariate": "free ranging"},
        {**raw["observations"][0], "sheet": "ANIMALorRESERVOIR_COV", "covariate": "Felis catus"},
        {**raw["observations"][0], "sheet": "vectors", "covariate": "Culex pipiens"},
        {"sheet": "lab results", "covariate": "x"}, {"sheet": "env", "covariate": ""}, "not an object",
    ]
    out = extraction.parse_extraction(raw, PAPER)
    assert [o["sheet"] for o in out["observations"]] == ["human_exp", "human_susc", "env", "animal", "vector"]
    assert out["n_rows_dropped"] == 3
    assert out["rows_dropped"] == {"unknown_sheet": 1, "no_covariate": 1, "not_an_object": 1}


def test_rows_beyond_the_limit_are_counted_too():
    many = _raw(observations=[{"sheet": "env", "covariate": f"c{i}"} for i in range(130)])
    out = extraction.parse_extraction(many, "")
    assert len(out["observations"]) == extraction._MAX_OBSERVATIONS
    assert out["rows_dropped"] == {"over_limit": 130 - extraction._MAX_OBSERVATIONS}


# ── C1: the answer ceiling ──────────────────────────────────────────────────
def test_the_ceiling_fits_the_answer_the_prompt_asks_for_and_the_client_waits_for_it(monkeypatch):
    # 120 observations with 300-character quotes need about 18,000 to 21,000 tokens.
    assert extraction._EXTRACTION_MAX_TOKENS >= 20000
    assert extraction._FALLBACK_MAX_TOKENS == 8000
    seen = {}
    monkeypatch.setattr(llm_usage, "MeteredOpenAI", lambda **kw: seen.update(kw) or object())
    extraction._llm_client()
    assert seen["timeout"] >= 600


class _BadRequest(Exception):
    status_code = 400


def _ceiling_client(seen, refuse_above=16384, message=None):
    def create(**kw):
        seen.append(kw["max_tokens"])
        if kw["max_tokens"] > refuse_above:
            raise _BadRequest(message or (f"Error code: 400 - max_tokens is too large: {kw['max_tokens']}. This model "
                                          f"supports at most {refuse_above} completion tokens"))
        msg = types.SimpleNamespace(content=json.dumps(_raw()))
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg, finish_reason="stop")])
    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))


def test_a_ceiling_the_api_refuses_is_asked_again_at_8000_and_remembered(monkeypatch):
    monkeypatch.setattr(extraction, "_CEILING_REFUSED", {})
    seen: list = []
    out = extraction.extract_article(_ceiling_client(seen), {"id": 1}, PAPER, "fulltext", False)
    assert seen == [extraction._EXTRACTION_MAX_TOKENS, 8000] and out["max_tokens"] == 8000
    # Learned once: the next article is asked 8,000 straight away, not refused again.
    seen.clear()
    extraction.extract_article(_ceiling_client(seen), {"id": 2}, PAPER, "fulltext", False)
    assert seen == [8000]


def test_another_bad_request_is_not_mistaken_for_the_ceiling(monkeypatch):
    monkeypatch.setattr(extraction, "_CEILING_REFUSED", {})
    seen: list = []
    with pytest.raises(_BadRequest):
        extraction.extract_article(_ceiling_client(seen, refuse_above=0, message="Error code: 400 - invalid schema"),
                                   {"id": 1}, PAPER, "fulltext", False)
    assert seen == [extraction._EXTRACTION_MAX_TOKENS]
    # The rename of max_tokens is llm_usage's to repair, not a ceiling to lower.
    assert not extraction._refuses_ceiling(_BadRequest("Unsupported parameter: 'max_tokens' is not supported "
                                                       "with this model. Use 'max_completion_tokens' instead."))


def test_the_condensed_answer_keeps_the_counts_of_every_sheet_first():
    rule = extraction._CONDENSED_RULE
    assert "at most 50 observations" in rule and EM_DASH not in rule
    first_counts, then_human = rule.index("counts"), rule.index("sex, gender, age")
    assert first_counts < then_human
    for sheet in ("human", "environment", "animal", "vector"):
        assert sheet in rule[:then_human]


# ── C2: the text read per paper ─────────────────────────────────────────────
S1 = "Seventeen persons were exposed to infected poultry during the culling operations in March. "
S2 = "Among the veterinary authority staff, seven of seven were not vaccinated against seasonal influenza. "
S3 = "No antibodies against H5N1 were detected in the eleven sera collected three weeks later. "
S4 = "The holding had no biosecurity measures and comprised about twenty-one chickens and nine cats."


def test_the_chunk_overlaps_are_read_once():
    # As the full-text chunker writes them: the next chunk starts with the last sentences of the previous.
    chunks = [(S1 + S2).strip(), (S2 + S3).strip(), (S3 + S4).strip()]
    joined = extraction._join_fulltext_chunks(chunks)
    assert joined.count("seven of seven") == 1 and joined.count("eleven sera") == 1
    assert " ".join(joined.split()) == " ".join((S1 + S2 + S3 + S4).split())
    # No repeat, nothing removed; a short coincidence is not a repeat.
    assert extraction._join_fulltext_chunks(["First part ends here.", "Second part."]) == "First part ends here.\n\nSecond part."
    assert extraction._chunk_overlap("A sentence that ends with in.", "in a second chunk") == 0


def _paper_with_references(tail: str = "") -> str:
    body = (S1 + S2 + S3 + S4 + " ") * 30
    refs = " ".join(f"{i}. Author{i} A, Other B. A title about avian influenza {i}. J Infect. {2000 + i};{i}:1-9."
                    for i in range(1, 40))
    return body + "Conflict of interest. None declared. References " + refs + tail


def test_the_reference_list_is_dropped_and_what_follows_it_is_kept():
    text_ = _paper_with_references(" Table 3. Serology results of the exposed persons: 0 of 11 positive.")
    out, dropped = extraction._drop_references(text_)
    assert dropped > 1500 and "Author12" not in out and "J Infect" not in out
    assert out.startswith(S1.strip()) and "Table 3. Serology results" in out and "None declared." in out
    # Nothing that looks like a reference list, nothing dropped.
    early = "References to earlier outbreaks are discussed below. " + (S1 + S2) * 40
    assert extraction._drop_references(early) == (early, 0)
    inline = (S1 + S2) * 40 + "We followed the references cited by the WHO guidance in 2020."
    assert extraction._drop_references(inline)[1] == 0


@pytest.fixture()
def long_paper(seeded):
    """9702 gets a long full text, in overlapping chunks, ending with a reference list."""
    text_ = _paper_with_references()
    step, chunks, start = 4000, [], 0
    while start < len(text_):
        piece = text_[max(0, start - 300):start + step] if start else text_[:step]
        chunks.append(piece)
        start += step
    with seeded.cursor() as cur:
        cur.execute("DELETE FROM document_chunk WHERE document_id = 9702")
        for i, c in enumerate(chunks):
            cur.execute("INSERT INTO document_chunk (document_id, chunk_index, content, chunk_type) "
                        "VALUES (9702, %s, %s, 'fulltext_section')", (i, c))
    return text_


def test_the_article_text_says_how_much_of_the_paper_was_read(long_paper, monkeypatch):
    with main.engine.connect() as conn:
        info = extraction._article_text_info(conn, {"id": 9702, "title": "T", "abstract": "a"})
    assert info["source"] == "fulltext" and info["truncated"] is False
    assert info["max_chars"] == extraction.EXTRACTION_MAX_CHARS and info["references_dropped"] > 1500
    assert info["chars_read"] == info["chars_total"] == len(info["text"]) - len("T\n\n")
    assert info["text"].count("Author12") == 0
    monkeypatch.setattr(extraction, "EXTRACTION_MAX_CHARS", 5000)
    with main.engine.connect() as conn:
        cut = extraction._article_text_info(conn, {"id": 9702, "title": "T", "abstract": "a"})
    assert cut["truncated"] is True and cut["chars_read"] == 5000 and cut["max_chars"] == 5000
    assert cut["chars_total"] == info["chars_total"]


def test_a_paper_cut_under_a_smaller_limit_is_read_again_and_only_then(monkeypatch):
    monkeypatch.setattr(extraction, "EXTRACTION_MAX_CHARS", 150000)
    base = {"abstract": "x" * 200, "has_fulltext_chunk": True}
    ex = {"v": extraction.EXTRACTION_VERSION, "source": "fulltext"}
    # Cut at 60,000 (said, or not said: an extraction made before the limit was recorded).
    assert extraction._needs_extraction({**base, "extraction_json": {**ex, "truncated": True, "max_chars": 60000}})
    assert extraction._needs_extraction({**base, "extraction_json": {**ex, "truncated": True}})
    # Cut at today's limit, or not cut: left alone.
    assert not extraction._needs_extraction({**base, "extraction_json": {**ex, "truncated": True, "max_chars": 150000}})
    assert not extraction._needs_extraction({**base, "extraction_json": {**ex, "truncated": False}})
    # Its text is gone: a cut full text is never replaced by the abstract.
    assert not extraction._needs_extraction({**base, "has_fulltext_chunk": False,
                                             "extraction_json": {**ex, "truncated": True}})


def test_a_run_records_what_it_read_and_the_status_counts_the_cut_papers(seeded, long_paper, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(extraction, "EXTRACTION_MAX_CHARS", 5000)
    patch_app(monkeypatch, "_llm_client", lambda: _fake_client())
    _run(None)
    ex = _stored(seeded, 9702)[0]
    assert ex["truncated"] is True and ex["max_chars"] == 5000 and ex["chars_read"] == 5000
    assert ex["chars_total"] > 5000 and ex["references_dropped"] > 0 and ex["max_tokens"] == extraction._EXTRACTION_MAX_TOKENS
    st = extraction.extraction_status(SID)
    assert st["n_truncated"] == 1 and st["max_chars"] == 5000 and st["n_to_reread"] == 0
    assert extraction.extraction_digest(SID)["n_truncated"] == 1
    # The limit is raised: the cut paper is pending again, said as a re-read.
    monkeypatch.setattr(extraction, "EXTRACTION_MAX_CHARS", 150000)
    st = extraction.extraction_status(SID)
    assert st["n_to_reread"] == 1 and st["n_pending"] == 1
    assert _run(None)["total"] == 1
    assert extraction.extraction_status(SID)["n_truncated"] == 0


# ── C3: an outage costs no article an attempt ───────────────────────────────
class RateLimitError(Exception):
    """Named like the SDK's: the classification goes by name, so the SDK is never imported."""
    status_code = 429


def test_transient_and_deterministic_failures_are_told_apart():
    T = llm_usage.is_transient_llm_error
    assert T(llm_usage.LLMCallBlocked("off")) and T(RateLimitError("insufficient_quota")) and T(TimeoutError())
    assert T(ConnectionResetError()) and T(type("InternalServerError", (Exception,), {})("502"))
    assert T(type("X", (Exception,), {"status_code": 503})("bad gateway"))
    assert not T(llm_usage.LLMTruncated("cut")) and not T(ValueError("Expecting value: line 1"))
    assert not T(RuntimeError("boom")) and not T(_BadRequest("max_tokens is too large"))


def test_the_watch_stops_a_job_on_a_refusal_after_a_streak_or_while_the_breaker_is_open():
    w = llm_usage.ProviderWatch(limit=3)
    for _ in range(2):
        w.failure(TimeoutError("t"))
    w.success()                                                      # an answer between: the streak starts again
    w.failure(TimeoutError("t"))
    w.failure(TimeoutError("t"))
    assert not w.stopped()
    w.failure(TimeoutError("t"))
    assert w.stopped() and "3 failures in a row" in w.reason and w.n_transient == 5
    blocked = llm_usage.ProviderWatch()
    blocked.failure(llm_usage.LLMCallBlocked("daily LLM budget reached"))
    assert blocked.stopped() and "budget" in blocked.reason
    breaker = llm_usage.ProviderWatch(cooldown=lambda: True)
    assert breaker.stopped() and "circuit breaker" in breaker.reason


def _failing_client(exc_for):
    """A client that raises `exc_for(text)` (or answers when it returns None)."""
    good = _fake_client()

    def create(**kw):
        e = exc_for(json.loads(kw["messages"][1]["content"])["text"])
        if e is not None:
            raise e
        return good.chat.completions.create(**kw)
    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))


def test_a_transient_failure_charges_no_attempt(seeded, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    trips = []
    patch_app(monkeypatch, "_trip_openai_cooldown", lambda *a: trips.append(1))
    patch_app(monkeypatch, "_llm_client",
              lambda: _failing_client(lambda t: RateLimitError("insufficient_quota") if t.startswith("Fails") else None))
    res = _run(None)
    assert res == {"total": 4, "done": 3, "failed": 0}
    assert _stored(seeded, 9706) == (None, 0), "an outage is not the article's failure"
    job = extraction.extraction_status(SID)["job"]
    assert job["transient"] == 1 and job["paused"] is None and trips == [1]     # the quota trips the breaker


def test_when_the_provider_is_down_the_run_pauses_instead_of_failing_its_queue(seeded, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(extraction, "_EXTRACTION_WORKERS", 1)
    patch_app(monkeypatch, "_llm_client", lambda: _failing_client(lambda t: TimeoutError("Request timed out.")))
    monkeypatch.setattr(extraction, "ProviderWatch",
                        lambda **kw: llm_usage.ProviderWatch(limit=2, **kw))
    res = _run(None)
    assert res == {"total": 4, "done": 0, "failed": 0}
    st = extraction.extraction_status(SID)
    assert st["job"]["transient"] == 2 and st["job"]["skipped"] == 2 and "2 failures in a row" in st["job"]["paused"]
    with seeded.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM literature_document WHERE id IN (9701, 9702, 9703, 9706) "
                    "AND COALESCE(extraction_attempts, 0) > 0")
        assert cur.fetchone()[0] == 0
    assert st["n_given_up"] == 0 and st["n_pending"] == 4


def test_nothing_starts_while_calls_are_refused(seeded, monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_ENABLED", "0")
    body = TestClient(main.app).post(f"/user-scenarios/{SID}/extraction/run", headers=HDR).json()
    assert body["status"] == "paused" and "OPENAI_ENABLED=0" in body["reason"]
    monkeypatch.setenv("OPENAI_ENABLED", "1")
    patch_app(monkeypatch, "_openai_in_cooldown", lambda: True)
    body = TestClient(main.app).post(f"/user-scenarios/{SID}/extraction/run", headers=HDR).json()
    assert body["status"] == "paused" and "circuit breaker" in body["reason"]


def test_a_cut_answer_after_the_condensed_retry_still_charges(seeded, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def create(**kw):
        msg = types.SimpleNamespace(content="{}")
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg, finish_reason="length")])
    patch_app(monkeypatch, "_llm_client",
              lambda: types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))))
    assert _run(None)["failed"] == 4
    assert _stored(seeded, 9701)[1] == 1


# ── C14: the stored quotes checked again, no model call ─────────────────────
def test_the_quotes_are_checked_again_against_the_whole_text(extracted, monkeypatch):
    from fastapi.testclient import TestClient
    # The paper says "identified"; the model copied it from a PDF with the "fi" ligature.
    obs = [{"sheet": "human_susc", "group": "sex", "covariate": "male", "quote": "No antibodies against H5N1 were detected",
            "quote_verified": False},
           {"sheet": "human_susc", "group": "sex", "covariate": "female", "quote": "an invented sentence", "quote_verified": True},
           {"sheet": "human_susc", "group": "age", "covariate": "adult",
            "quote": f"Seventeen exposed persons were identi{LIG_FI}ed", "quote_verified": False}]
    with extracted.cursor() as cur:
        cur.execute("UPDATE literature_document SET extraction_json = %s::jsonb WHERE id = 9701",
                    (json.dumps({"v": 1, "source": "abstract", "extracted_at": "2026-10-01T00:00:00+00:00",
                                 "coverage": {}, "observations": obs}),))
        cur.execute("UPDATE literature_document SET extraction_json = jsonb_set(extraction_json, '{observations}', "
                    "'[]'::jsonb) WHERE id = 9702")
    c = TestClient(main.app)
    assert c.post(f"/user-scenarios/{SID}/extraction/reverify").status_code in (401, 403)
    # Not while a run is writing the extractions.
    monkeypatch.setitem(extraction._jobs, SID, {"running": True})
    assert c.post(f"/user-scenarios/{SID}/extraction/reverify", headers=HDR).status_code == 409
    monkeypatch.setitem(extraction._jobs, SID, {"running": False})
    r = c.post(f"/user-scenarios/{SID}/extraction/reverify", headers=HDR)
    assert r.status_code == 200, r.text
    body = r.json()
    assert (body["n_observations"], body["verified_before"], body["verified_after"]) == (3, 1, 2)
    assert body["newly_found"] == 2 and body["no_longer_found"] == 1 and body["n_articles_changed"] == 1
    stored = _stored(extracted, 9701)[0]
    assert [o["quote_verified"] for o in stored["observations"]] == [True, False, True]
    assert stored["quotes_rechecked_at"] and stored["observations"][0]["quote"] == obs[0]["quote"]
    # Nothing left to change: a second pass changes nothing.
    again = c.post(f"/user-scenarios/{SID}/extraction/reverify", headers=HDR).json()
    assert again["newly_found"] == again["no_longer_found"] == again["n_articles_changed"] == 0
    assert c.post("/user-scenarios/nope/extraction/reverify", headers=HDR).status_code == 404


# ── The exports say which papers were read cut or condensed ─────────────────
def test_the_exports_flag_cut_and_condensed_papers(extracted):
    openpyxl = pytest.importorskip("openpyxl")
    from fastapi.testclient import TestClient
    with extracted.cursor() as cur:
        cur.execute("UPDATE literature_document SET extraction_json = extraction_json || "
                    "'{\"condensed\": true}'::jsonb WHERE id = 9701")
        cur.execute("UPDATE literature_document SET extraction_json = extraction_json || "
                    "'{\"truncated\": true, \"max_chars\": 60000, \"chars_read\": 60000}'::jsonb WHERE id = 9702")
    c = TestClient(main.app)
    r = c.get(f"/user-scenarios/{SID}/extraction/export")
    assert "1 read from a text cut at the size limit; 1 condensed" in r.headers["x-coverage"]
    wb = openpyxl.load_workbook(io.BytesIO(r.content))
    readme = " ".join(str(x[0].value) for x in wb["README"].iter_rows())
    assert "1 papers were read from a text cut at the size limit" in readme
    assert "1 papers held more data than one answer" in readme
    head = [x.value for x in wb["REF"][1]]
    rows = {x[0].value: dict(zip(head, (y.value for y in x))) for x in wb["REF"].iter_rows(min_row=2)}
    assert rows["LR9701"]["Condensed answer (Y/N)"] == "Y" and rows["LR9701"]["Text cut at the size limit (Y/N)"] == "N"
    assert rows["LR9702"]["Text cut at the size limit (Y/N)"] == "Y" and rows["LR9702"]["Characters read"] == 60000
    import csv
    csv_rows = list(csv.DictReader(io.StringIO(c.get(f"/user-scenarios/{SID}/extraction/export?format=csv").text.lstrip(chr(0xFEFF)))))
    by_id = {}
    for row in csv_rows:
        by_id.setdefault(row["id"], row)
    assert by_id["LR9701"]["condensed"] == "True" and by_id["LR9702"]["text_truncated"] == "True"
    lines = [json.loads(x) for x in c.get(f"/user-scenarios/{SID}/extraction/dataset?include_unreviewed=true").text.splitlines()]
    flags = {x["article"]["id"]: x["extraction"] for x in lines}
    assert flags[9701]["condensed"] is True and flags[9701]["text_truncated"] is False
    assert flags[9702]["text_truncated"] is True and flags[9702]["max_chars"] == 60000
