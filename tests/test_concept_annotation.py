"""The concept annotation pass (api/knowledge_graph.py): every relevant article gets its turn.

Batches of 15 under a 4,000-token ceiling sat at the ceiling; a cut answer lost the whole
batch and charged an attempt to all fifteen articles, whatever the cause (an outage too); after
three an article was never annotated again, with no way back; the map's GET relaunched the paid
pass on every visit, checking only for a key; and articles with no abstract left the map and
its denominator. These tests pin the new rules with a fake client."""
import json
import types

import pytest

pytest.importorskip("fastapi")

import llm_usage  # noqa: E402
import main  # noqa: E402
from api import knowledge_graph as KG  # noqa: E402
from conftest import patch_app  # noqa: E402
from test_extraction import HDR, SID, seeded  # noqa: E402,F401

RELEVANT = (9701, 9702, 9703, 9706)
H5N1 = {"t": "pathogen", "en": "H5N1 virus", "fr": "virus H5N1"}


def _answer(content, finish="stop"):
    msg = types.SimpleNamespace(content=content if isinstance(content, str) else json.dumps(content))
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg, finish_reason=finish)])


def _client(respond, seen=None):
    """A client whose answer to a batch is `respond(ids)`: a dict, a raw string, ("cut", ids) or an exception."""
    def create(**kw):
        items = json.loads(kw["messages"][1]["content"])["articles"]
        if seen is not None:
            seen.append(items)
        out = respond([it["id"] for it in items])
        if isinstance(out, BaseException):
            raise out
        if isinstance(out, tuple) and out[0] == "cut":
            return _answer('{"articles": [', finish="length")
        return _answer(out)
    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))


def _all(ids, extra=()):
    return {"articles": [{"id": i, "concepts": [H5N1]} for i in [*ids, *extra]]}


def _rows(ids, abstract="An outbreak of avian influenza in poultry holdings. " * 100):
    return [{"id": i, "title": f"Paper {i}", "abstract": abstract, "pico_json": {"P": "poultry workers", "O": "infection"}}
            for i in ids]


# ── One batch ───────────────────────────────────────────────────────────────
def test_a_batch_sends_the_pico_and_the_whole_abstract_and_keeps_only_its_own_ids():
    seen: list = []
    out = KG._llm_concepts_for_batch(_client(lambda ids: _all(ids, extra=(424242,)), seen), _rows([1, 2]))
    assert set(out) == {1, 2}, "an id the batch did not hold is never written"
    sent = seen[0][0]
    assert sent["P"] == "poultry workers" and len(sent["abstract"]) == KG._CONCEPT_ABSTRACT_CHARS == 3000
    assert KG._CONCEPT_BATCH == 8


def test_a_cut_answer_is_named_instead_of_read_as_an_empty_batch():
    with pytest.raises(llm_usage.LLMTruncated):
        KG._llm_concepts_for_batch(_client(lambda ids: ("cut", ids)), _rows([1, 2]))
    with pytest.raises(ValueError):
        KG._llm_concepts_for_batch(_client(lambda ids: "not json"), _rows([1, 2]))


def test_an_attempt_is_charged_only_to_an_article_a_valid_answer_left_out():
    watch = llm_usage.ProviderWatch()
    got, charge = KG._concepts_ask(_client(lambda ids: _all(ids[:-1])), _rows([1, 2, 3]), watch)
    assert set(got) == {1, 2} and charge == [3]


def test_a_cut_batch_is_split_in_two_and_each_half_tried_once():
    calls: list = []

    def respond(ids):
        calls.append(list(ids))
        if len(ids) == 8:
            return ("cut", ids)
        return ("cut", ids) if ids[0] == 5 else _all(ids)
    got, charge = KG._concepts_ask(_client(respond), _rows(range(1, 9)), llm_usage.ProviderWatch())
    assert calls == [list(range(1, 9)), [1, 2, 3, 4], [5, 6, 7, 8]]
    assert set(got) == {1, 2, 3, 4} and charge == [5, 6, 7, 8], "a half still cut is charged, and not split again"


def test_an_outage_costs_no_attempt_and_an_unreadable_answer_does():
    watch = llm_usage.ProviderWatch()
    got, charge = KG._concepts_ask(_client(lambda ids: TimeoutError("Request timed out.")), _rows([1, 2]), watch)
    assert (got, charge) == ({}, []) and watch.n_transient == 1
    got, charge = KG._concepts_ask(_client(lambda ids: "{not json"), _rows([1, 2]), watch)
    assert (got, charge) == ({}, [1, 2])
    blocked = llm_usage.ProviderWatch()
    KG._concepts_ask(_client(lambda ids: llm_usage.LLMCallBlocked("budget")), _rows([1]), blocked)
    assert blocked.stopped()
    assert KG._concepts_ask(_client(lambda ids: _all(ids)), _rows([1]), blocked) == ({}, []), "a stopped pass calls nothing"


# ── The pass over a scenario ────────────────────────────────────────────────
@pytest.fixture()
def concept_corpus(seeded):
    with seeded.cursor() as cur:
        cur.execute("UPDATE literature_document SET concepts_json = NULL, concepts_attempts = 0, pico_json = NULL "
                    "WHERE id BETWEEN 9701 AND 9718")
        cur.execute("DELETE FROM article_scenarios WHERE document_id = 9718")
        cur.execute("DELETE FROM literature_document WHERE id = 9718")
    KG._concept_pass_state.pop(SID, None)
    yield seeded
    with seeded.cursor() as cur:
        cur.execute("DELETE FROM article_scenarios WHERE document_id = 9718")
        cur.execute("DELETE FROM literature_document WHERE id = 9718")
    KG._concept_pass_state.pop(SID, None)


def _state(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT id, concepts_json IS NOT NULL, COALESCE(concepts_attempts, 0) FROM literature_document "
                    "WHERE id BETWEEN 9701 AND 9706 ORDER BY id")
        return {i: (has, n) for i, has, n in cur.fetchall()}


def _use(monkeypatch, client):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(llm_usage, "MeteredOpenAI", lambda **kw: client)


def test_every_relevant_article_is_annotated_and_nothing_else_is_written(concept_corpus, monkeypatch):
    seen: list = []
    _use(monkeypatch, _client(lambda ids: _all([i for i in ids if i != 9706], extra=(9704,)), seen))
    assert KG._extract_concepts_for_scenario(SID) == 3
    st = _state(concept_corpus)
    assert {i for i, (has, _n) in st.items() if has} == {9701, 9702, 9703}
    assert st[9706] == (False, 1), "left out of a valid answer: one attempt"
    assert st[9704][0] is False, "9704 was not asked: what the model said about it is not written"
    assert sorted(it["id"] for batch in seen for it in batch) == list(RELEVANT)
    assert KG._concept_pass_state[SID]["status"] == "done" and KG._concept_pass_state[SID]["charged"] == 1


def test_a_provider_that_is_down_stops_the_pass_and_charges_nothing(concept_corpus, monkeypatch):
    monkeypatch.setattr(KG, "_CONCEPT_BATCH", 1)
    monkeypatch.setattr(KG, "_CONCEPT_WORKERS", 1)
    monkeypatch.setattr(KG, "ProviderWatch", lambda **kw: llm_usage.ProviderWatch(limit=2, **kw))
    calls: list = []
    _use(monkeypatch, _client(lambda ids: calls.append(ids) or TimeoutError("Request timed out.")))
    assert KG._extract_concepts_for_scenario(SID) == 0
    assert len(calls) == 2, "two failures in a row: the other batches are not tried"
    assert all(n == 0 for _has, n in _state(concept_corpus).values())
    state = KG._concept_pass_state[SID]
    assert state["status"] == "paused" and "2 failures in a row" in state["reason"] and state["pending"] == 4


def test_no_call_is_made_while_calls_are_refused(concept_corpus, monkeypatch):
    calls: list = []
    _use(monkeypatch, _client(lambda ids: calls.append(ids) or _all(ids)))
    monkeypatch.setenv("OPENAI_ENABLED", "0")
    assert KG._extract_concepts_for_scenario(SID) == 0 and calls == []
    assert KG._concept_pass_state[SID]["status"] == "paused"
    monkeypatch.setenv("OPENAI_ENABLED", "1")
    patch_app(monkeypatch, "_openai_in_cooldown", lambda: True)
    assert KG._extract_concepts_for_scenario(SID) == 0 and calls == []


def test_an_article_without_any_text_counts_as_unannotatable(concept_corpus):
    with concept_corpus.cursor() as cur:
        cur.execute("INSERT INTO literature_document (id, title, abstract, source, is_duplicate, project_context) "
                    "VALUES (9718, 'A title and nothing else', NULL, 'pubmed', false, 'literev')")
        cur.execute("INSERT INTO article_scenarios (scenario_id, document_id, similarity_score) VALUES (%s, 9718, 0.9)", (SID,))
    g = KG._compute_concept_graph(SID)
    assert g["n_without_text"] == 1 and g["n_unannotatable"] >= 1
    assert g["n_total"] == g["n_articles"] + 1, "the denominator holds the article the map cannot read"


# ── The endpoints ───────────────────────────────────────────────────────────
def test_reading_the_map_never_starts_the_paid_pass(monkeypatch):
    from fastapi.testclient import TestClient
    cached = {"kind": "concepts", "version": KG.CONCEPTS_VERSION, "n_articles": 10, "n_missing_concepts": 5,
              "n_unannotatable": 2, "nodes": [], "edges": []}
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    patch_app(monkeypatch, "_get_user_scenario_or_404", lambda sid: {"id": sid})
    patch_app(monkeypatch, "_load_viz_cache", lambda sid, col, ttl=None: dict(cached))
    patch_app(monkeypatch, "_start_concept_enrichment",
              lambda sid: (_ for _ in ()).throw(AssertionError("a GET must not start the pass")))
    body = TestClient(main.app).get("/user-scenarios/usr-x/concept-graph").json()
    assert body["enriching"] is False and body["n_missing_concepts"] == 5 and body["n_unannotatable"] == 2


def test_the_pass_is_started_on_request_with_the_key_and_only_when_calls_can_be_made(concept_corpus, monkeypatch):
    from fastapi.testclient import TestClient
    c = TestClient(main.app)
    url = f"/user-scenarios/{SID}/concept-graph/annotate"
    started: list = []
    patch_app(monkeypatch, "_start_concept_enrichment", lambda sid: started.append(sid) or True)
    assert c.post(url).status_code in (401, 403)
    monkeypatch.setenv("OPENAI_API_KEY", "")
    assert c.post(url, headers=HDR).json()["status"] == "no_llm"
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_ENABLED", "0")
    body = c.post(url, headers=HDR).json()
    assert body["status"] == "paused" and "OPENAI_ENABLED=0" in body["reason"] and started == []
    monkeypatch.setenv("OPENAI_ENABLED", "1")
    assert c.post(url, headers=HDR).json()["status"] == "started" and started == [SID]
    assert c.post("/user-scenarios/nope/concept-graph/annotate", headers=HDR).status_code == 404


def test_articles_given_up_get_their_attempts_back_in_their_scenario_only(concept_corpus, monkeypatch):
    from fastapi.testclient import TestClient
    with concept_corpus.cursor() as cur:
        cur.execute("UPDATE literature_document SET concepts_attempts = 3 WHERE id IN (9701, 9704)")
    assert KG._compute_concept_graph(SID)["n_concepts_given_up"] == 1
    patch_app(monkeypatch, "_start_concept_enrichment", lambda sid: True)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    body = TestClient(main.app).post(f"/user-scenarios/{SID}/concept-graph/annotate?retry_given_up=true",
                                     headers=HDR).json()
    assert body["n_reset"] == 1 and body["status"] == "started"
    st = _state(concept_corpus)
    assert st[9701] == (False, 0) and st[9704] == (False, 3), "9704 is not relevant to this scenario"
