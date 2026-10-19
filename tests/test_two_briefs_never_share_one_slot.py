"""The Evidence tab holds two briefs, and opening it used to destroy one of them.

Two different things are called "the brief":

  - the PORTRAIT, six aggregates over the whole corpus, pure SQL, rebuilt on demand and
    cached under a `_stamp` that any corpus change invalidates;
  - the REPORT, written by a model from the papers, costing a call, cached under a
    `_corpus_fingerprint` of the exact relevant set plus the threshold plus the language.

Both wrote `scenario_settings.evidence_brief_json`, and each reader judged the other's
payload invalid. So opening the Evidence tab called the portrait endpoint, which found a
report where it expected a portrait, rebuilt the portrait and OVERWROTE the report. The
report endpoint then found a portrait, refused it, and refused to spend a model call on a
GET, which is right: so the panel stayed empty for good. The reviewers' words for this
were "it is never generated automatically" and "it gets lost when I switch tabs and come
back", and both were the same bug, with the twist that the tab meant to show the report
was what deleted it.

What these tests pin, in the order the failure happened:

  1. the two briefs live in two columns, and writing either leaves the other intact;
  2. a report whose corpus has moved is served DATED, never dropped, with what it was
     written from and what the corpus holds now;
  3. the same for a report written in another language;
  4. a GET still never spends;
  5. a corpus change keeps the report (it costs money) and drops the portrait (it does
     not);
  6. the citable report reads the REPORT and not whichever brief won the race.
"""
import json

import pytest

pytest.importorskip("fastapi")

import main  # noqa: E402
from api.evidence import (  # noqa: E402
    cached_evidence_brief,
    is_llm_brief,
    stored_llm_brief,
    store_evidence_brief,
)
from api.scenario_store import CORPUS_DERIVED_CACHE_RESET  # noqa: E402

SID = "usr-two-briefs"
IDS = (9901, 9902, 9903)

#: Ce qu'écrit le générateur narratif : le cachet `_corpus_fingerprint` le distingue.
REPORT = {
    "executive_summary": "Exposure to infected poultry dominates the reported cases.",
    "claims": [{"text": "Cullers are the most exposed group.", "article_ids": [9901]}],
    "evidence_level": "moderate",
    "lang": "fr",
    "_corpus_fingerprint": "fp-of-the-corpus-when-it-was-written",
    "_corpus_n": 2,
    "_threshold": 0.45,
}


def test_the_two_payloads_are_told_apart_by_their_shape():
    """The discriminator, pure. A report written before the fingerprint existed is still a
    report, so shape and not only stamp; and the portrait's `evidence_level_distribution`
    must never be read as the report's `evidence_level`."""
    assert is_llm_brief(REPORT)
    assert is_llm_brief({"executive_summary": "written before fingerprints existed"})
    assert is_llm_brief({"claims": [{"text": "a claim"}]})
    assert not is_llm_brief({"n_relevant": 818, "_stamp": "abc"})
    assert not is_llm_brief({"evidence_level_distribution": [{"level": "moderate"}],
                             "study_design_distribution": []})
    # A stamped portrait stays a portrait even if it ever grew a narrative-looking key.
    assert not is_llm_brief({"_stamp": "abc", "executive_summary": "x"})
    for empty in ({}, None, [], "executive_summary"):
        assert not is_llm_brief(empty)


def _engine_ok() -> bool:
    try:
        with main.engine.connect():
            return True
    except Exception:                                        # noqa: BLE001
        return False


@pytest.fixture()
def scenario(db_conn):
    if not _engine_ok():
        pytest.skip("main.engine cannot reach the database")
    with db_conn.cursor() as cur:
        cur.execute("SELECT to_regclass('user_scenarios') IS NULL")
        if cur.fetchone()[0]:
            main._ensure_user_scenarios_table()
        cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM scenario_settings WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))
        cur.execute("INSERT INTO user_scenarios (id, name, query, mode, filters, pinned) "
                    "VALUES (%s, 'Two briefs', 'avian influenza', 'boolean', '{}', TRUE)",
                    (SID,))
        cur.execute("INSERT INTO scenario_settings (scenario_id, similarity_threshold) "
                    "VALUES (%s, 0.45) ON CONFLICT (scenario_id) DO UPDATE "
                    "SET similarity_threshold = 0.45", (SID,))
    yield db_conn
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM scenario_settings WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))


def _write_report(cur, payload: dict | None = None) -> None:
    cur.execute("UPDATE scenario_settings SET evidence_brief_llm_json = %s::jsonb, "
                "brief_generated_at = NOW() WHERE scenario_id = %s",
                (json.dumps(payload if payload is not None else REPORT), SID))


def test_the_report_survives_the_portrait_being_written(scenario):
    """The exact sequence a reviewer performed: a report exists, they open the Evidence
    tab, the portrait is rebuilt and stored. The report must still be there afterwards."""
    with scenario.cursor() as cur:
        _write_report(cur)
    assert stored_llm_brief(SID)["executive_summary"].startswith("Exposure")

    store_evidence_brief(SID, {"n_relevant": 2, "study_design_distribution": []})

    kept = stored_llm_brief(SID)
    assert kept is not None, "the portrait overwrote the report, which is the whole bug"
    assert kept["claims"][0]["article_ids"] == [9901]
    assert kept["_corpus_fingerprint"] == "fp-of-the-corpus-when-it-was-written"


def test_the_portrait_never_reads_a_report_as_its_own_cache(scenario):
    """The other direction. A report sitting in the legacy column must not be served as a
    portrait: the portrait reader would hand `claims` to a page expecting distributions,
    and then store a portrait over it."""
    with scenario.cursor() as cur:
        cur.execute("UPDATE scenario_settings SET evidence_brief_json = %s::jsonb "
                    "WHERE scenario_id = %s", (json.dumps(REPORT), SID))
    assert cached_evidence_brief(SID) is None


def test_a_report_written_before_the_split_is_not_lost(scenario):
    """The migration case: production holds reports in the old shared column. They are
    found by their fingerprint, so a deploy does not throw away work already paid for."""
    with scenario.cursor() as cur:
        cur.execute("UPDATE scenario_settings SET evidence_brief_json = %s::jsonb, "
                    "evidence_brief_llm_json = NULL WHERE scenario_id = %s",
                    (json.dumps(REPORT), SID))
    found = stored_llm_brief(SID)
    assert found is not None and found["evidence_level"] == "moderate"


def test_a_portrait_in_the_old_column_is_never_mistaken_for_a_report(scenario):
    """The same fallback must not pick up a portrait: it has no fingerprint, so there is
    no report, and the page says so rather than rendering a portrait as a report."""
    with scenario.cursor() as cur:
        cur.execute("UPDATE scenario_settings SET evidence_brief_json = %s::jsonb, "
                    "evidence_brief_llm_json = NULL WHERE scenario_id = %s",
                    (json.dumps({"n_relevant": 2, "_stamp": "whatever"}), SID))
    assert stored_llm_brief(SID) is None


def test_a_corpus_change_keeps_the_report_and_drops_the_portrait():
    """What invalidation is for. The portrait is SQL and rebuilds itself, so dropping it
    costs nothing. The report costs a model call, and dropping it left the page blank with
    nothing allowed to refill it."""
    assert "evidence_brief_json = NULL" in CORPUS_DERIVED_CACHE_RESET
    assert "evidence_brief_llm_json" not in CORPUS_DERIVED_CACHE_RESET


def test_the_settings_endpoint_does_not_ship_the_report_on_every_page(scenario):
    """Every JSON column of this table has to be in the blob list, or it travels with
    each scenario page open. A report is a few thousand words."""
    from api.relevance import _SETTINGS_BLOB_COLUMNS

    assert "evidence_brief_llm_json" in _SETTINGS_BLOB_COLUMNS


# ── Through the endpoint ────────────────────────────────────────────────────
def _client():
    from fastapi.testclient import TestClient
    return TestClient(main.app)


@pytest.fixture()
def seeded(scenario):
    """Two relevant papers, so the report endpoint has a corpus to compare against."""
    from conftest import ensure_document_columns

    with scenario.cursor() as cur:
        cur.execute("CREATE TABLE IF NOT EXISTS literature_document (id BIGINT PRIMARY KEY)")
        cur.execute("CREATE TABLE IF NOT EXISTS article_scenarios ("
                    "scenario_id TEXT, document_id BIGINT, PRIMARY KEY (scenario_id, document_id))")
        created = ensure_document_columns(cur)
        for col, typ in (("title", "TEXT"), ("abstract", "TEXT"),
                         ("is_duplicate", "BOOLEAN DEFAULT FALSE")):
            cur.execute(f"ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS {col} {typ}")
        cur.execute("ALTER TABLE article_scenarios ADD COLUMN IF NOT EXISTS "
                    "similarity_score DOUBLE PRECISION")
        cur.execute("DELETE FROM literature_document WHERE id = ANY(%s)", (list(IDS),))
        cur.execute("""
            INSERT INTO literature_document (id, title, abstract, source, is_duplicate,
                                             project_context)
            VALUES (9901, 'Cullers', 'exposure among cullers', 'pubmed', FALSE, 'literev'),
                   (9902, 'Markets', 'exposure at markets', 'pubmed', FALSE, 'literev'),
                   (9903, 'Off topic', 'something else', 'pubmed', FALSE, 'literev')
        """)
        cur.execute("INSERT INTO article_scenarios (scenario_id, document_id, similarity_score) "
                    "VALUES (%s, 9901, 0.9), (%s, 9902, 0.8), (%s, 9903, 0.1)", (SID,) * 3)
    yield scenario
    with scenario.cursor() as cur:
        cur.execute("DELETE FROM literature_document WHERE id = ANY(%s)", (list(IDS),))
        if created:
            cur.execute("DROP TABLE document_chunk")


def test_a_report_whose_corpus_moved_is_served_dated_not_dropped(seeded):
    """The fix for the complaint. The stored report was written from 2 papers under a
    fingerprint that no longer matches; the page gets the report, marked stale, with both
    counts, instead of an empty panel it can never refill."""
    with seeded.cursor() as cur:
        _write_report(cur)
    r = _client().get(f"/scenarios/{SID}/evidence-brief/llm", params={"lang": "fr"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("executive_summary"), "the report is served"
    assert body["stale"] is True and body["_cached"] is True
    assert body["_corpus_n"] == 2, "what it was written from"
    assert "_corpus_now" in body, "and what the corpus holds today, so the two compare"
    assert body.get("_generated_at"), "and it says when it was written"
    # This fixture's papers carry no embedding, so the relevant subset reads as empty:
    # which is exactly the case that used to answer "no article above the threshold" and
    # hide an existing report. A momentarily empty corpus is a fact beside the report,
    # never a reason to withhold it.
    assert body["_corpus_now"] == 0


def test_a_report_written_in_the_other_language_is_served_and_says_so(seeded):
    """The trap that made the panel permanently empty for English readers: the language is
    part of the cache key, the pipeline writes one language, and the tab asks in the
    reader's. Now the report arrives with both languages named."""
    with seeded.cursor() as cur:
        _write_report(cur)
    body = _client().get(f"/scenarios/{SID}/evidence-brief/llm",
                         params={"lang": "en"}).json()
    assert body.get("executive_summary")
    assert body["_lang_written"] == "fr" and body["_lang_asked"] == "en"
    assert body["_lang_mismatch"] is True


def test_reading_the_report_never_spends_a_model_call(seeded, monkeypatch):
    """A GET must not generate: an unauthenticated read used to launch a full generation
    and overwrite the cache. Pinned here because the fix above makes the read richer, and
    a richer read is exactly where a generation creeps back in."""
    import llm_usage

    def _explode(*a, **kw):                                  # noqa: ANN002, ANN003
        raise AssertionError("a GET spent a model call")

    monkeypatch.setattr(llm_usage, "MeteredOpenAI", _explode)
    with seeded.cursor() as cur:
        _write_report(cur)
    assert _client().get(f"/scenarios/{SID}/evidence-brief/llm").status_code == 200
    # And with no report at all: still no spend, and a status the page can act on.
    with seeded.cursor() as cur:
        cur.execute("UPDATE scenario_settings SET evidence_brief_llm_json = NULL, "
                    "evidence_brief_json = NULL WHERE scenario_id = %s", (SID,))
    out = _client().get(f"/scenarios/{SID}/evidence-brief/llm").json()
    assert out.get("status") in ("generating", "not_generated", "error", "empty"), out
