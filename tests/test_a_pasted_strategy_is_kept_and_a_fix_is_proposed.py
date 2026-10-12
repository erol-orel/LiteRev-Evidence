"""A pasted search strategy is kept as entered, checked, and a fix is PROPOSED, never applied.

A systematic-review PubMed strategy is often 3,000 to 8,000 characters. The strategy writer
refused anything over 2,000 (422), answered within 1,500 tokens, and its prompt said both
"preserve the boolean" and "keep queries practical, 2-4 concept groups max": a long
professional strategy came back refused, cut, or simplified, without a word.

Decision of the product owner: "Raise the limits and go to the current path anyway, so that
if the query is not correct or doesn't make sense, we can ask the user for a proposed change
(he can accept the change or leave it as it was entered)." So:

- up to 8,000 characters are accepted, the model may answer in 6,000 tokens, and a model
  that refuses that ceiling is asked again at the one it names;
- a Boolean strategy is used EXACTLY as entered (`general` and `pubmed`), whatever the model
  writes back;
- the model checks it, and a pure-Python check of parentheses and quotes (with the search's
  own tokenizer) flags an unbalanced query even when the model misses it;
- a problem gives `proposed_query` and `proposal_reason` (in the interface language), the
  query used stays the one entered until the user accepts, and a scenario never stores a
  pending proposal.

The OpenAI client is faked: nothing leaves the machine.
"""
from __future__ import annotations

import json
import types

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402
from api import search as S  # noqa: E402

KEY = {"X-API-Key": "test-write-key"}


def _long_strategy(target: int = 5000) -> str:
    """A PubMed strategy in the shape reviewers write: three concept groups, each a long OR
    of MeSH headings and title/abstract synonyms, grown to `target` characters."""
    birds = ['"Influenza in Birds"[Mesh]', '"Influenza A Virus, H5N1 Subtype"[Mesh]',
             '"avian influenza"[tiab]', '"bird flu"[tiab]', '"highly pathogenic avian influenza"[tiab]',
             'HPAI[tiab]', 'H5N1[tiab]', 'H7N9[tiab]', 'H5N6[tiab]', 'H9N2[tiab]', '"clade 2.3.4.4b"[tiab]']
    people = ['"Occupational Exposure"[Mesh]', '"Farmers"[Mesh]', '"poultry worker*"[tiab]',
              '"farm worker*"[tiab]', 'veterinarian*[tiab]', 'cull*[tiab]', 'slaughterhouse*[tiab]',
              '"live bird market*"[tiab]', 'backyard[tiab]']
    outcomes = ['"Seroepidemiologic Studies"[Mesh]', 'seroprevalence[tiab]', 'serolog*[tiab]',
                'antibod*[tiab]', '"hemagglutination inhibition"[tiab]', 'microneutralization[tiab]']
    i = 0
    while True:
        q = " AND ".join("(" + " OR ".join(g) + ")" for g in (birds, people, outcomes))
        if len(q) > target - 150:                     # each round adds about 110 characters
            return q
        i += 1
        birds.append(f'"avian influenza variant {i}"[tiab]')
        people.append(f'"poultry handler group {i}"[tiab]')
        outcomes.append(f'"serosurvey design {i}"[tiab]')


class _Completions:
    def __init__(self, answer, seen: list, refuse_above: int | None = None):
        self.answer, self.seen, self.refuse_above = answer, seen, refuse_above

    def create(self, **kw):
        self.seen.append(kw)
        ceiling = kw.get("max_tokens") or kw.get("max_completion_tokens") or 0
        if self.refuse_above and ceiling > self.refuse_above:
            raise RuntimeError(
                "Error code: 400 - {'error': {'message': 'max_tokens is too large: "
                f"{ceiling}. This model supports at most {self.refuse_above} completion tokens, "
                f"whereas you provided {ceiling}.', 'type': 'invalid_request_error'}}}}")
        content = self.answer(kw) if callable(self.answer) else self.answer
        msg = types.SimpleNamespace(content=content if isinstance(content, str) else json.dumps(content))
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg, finish_reason="stop")])


class _NoDb:
    """Every connection fails: the persistent cache is skipped (best-effort both ways)."""

    def connect(self):
        raise RuntimeError("no database in this test")

    def begin(self):
        raise RuntimeError("no database in this test")


@pytest.fixture
def model(monkeypatch):
    """Install a fake model answering `answer` (a dict, a string, or a function of the
    request); returns the list of requests it received."""
    import llm_usage

    seen: list = []

    def install(answer, refuse_above: int | None = None):
        class _Client:
            def __init__(self, *a, **k):
                self.chat = types.SimpleNamespace(completions=_Completions(answer, seen, refuse_above))
        monkeypatch.setattr(llm_usage, "MeteredOpenAI", _Client)
        return seen

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(S, "_STRATEGY_CACHE", {})
    monkeypatch.setattr(S, "engine", _NoDb())
    return install


def _kept(proposed: str = "", reason: dict | None = None) -> dict:
    """What the model answers for a Boolean strategy: kept, not repeated."""
    return {"input_kind": "boolean", "general": "", "pubmed": "",
            "explanation": "The strategy was kept as entered.", "synonyms": [],
            "proposed_query": proposed, "proposal_reason": reason or {"en": "", "fr": ""}}


def _post(query: str, lang: str | None = None):
    body = {"query": query} | ({"lang": lang} if lang else {})
    return TestClient(main.app).post("/search-strategy", json=body)


# ── the limits ───────────────────────────────────────────────────────────────

def test_a_5000_character_strategy_is_accepted_and_preserved_as_entered(model):
    strategy = _long_strategy(5000)
    assert 4850 <= len(strategy) <= 5000
    # The model "helpfully" sends back a shortened strategy: it must not be used.
    seen = model(_kept() | {"general": '("avian influenza") AND (poultry)', "pubmed": "h5n1[tiab]"})
    r = _post(strategy, "en")
    assert r.status_code == 200, r.text[:300]
    out = r.json()
    assert out["general"] == strategy and out["pubmed"] == strategy
    assert out["entered_query"] == strategy and out["input_kind"] == "boolean"
    assert out["proposed_query"] is None and not out.get("degraded")
    # The whole strategy reached the model, with room for a whole corrected copy.
    assert strategy in seen[0]["messages"][1]["content"]
    assert seen[0]["max_tokens"] == S.SEARCH_STRATEGY_MAX_TOKENS == 6000


def test_up_to_8000_characters_and_no_further():
    assert S.SEARCH_STRATEGY_MAX_CHARS == 8000
    assert _post("x" * 8001).status_code == 422


def test_the_prompt_no_longer_asks_to_shorten_a_strategy():
    prompt = S._STRATEGY_SYSTEM_PROMPT
    boolean_part = prompt[prompt.index("If it is a BOOLEAN strategy"):prompt.index("If it is NATURAL language")]
    assert "Do NOT rewrite, shorten, simplify" in boolean_part
    assert "2 to 4 concept groups" not in boolean_part
    # The practical bound belongs to a translation only.
    assert "2 to 4 concept groups" in prompt[prompt.index("If it is NATURAL language"):]
    assert "em dash" in prompt
    assert chr(0x2014) not in prompt


def test_a_model_that_refuses_the_ceiling_is_asked_again_at_its_own(model):
    seen = model(_kept(), refuse_above=4096)
    out = S._generate_search_strategy("(h5n1 OR \"avian influenza\") AND poultry")
    assert [kw["max_tokens"] for kw in seen] == [6000, 4096]
    assert not out.get("degraded") and out["general"] == "(h5n1 OR \"avian influenza\") AND poultry"


def test_another_refusal_is_not_mistaken_for_the_ceiling():
    assert S._ceiling_refused(RuntimeError("Error code: 429 - rate limit")) is None
    assert S._ceiling_refused(RuntimeError(
        "max_completion_tokens is too large: 6000. This model supports at most 2048")) == 2048
    assert S._ceiling_refused(RuntimeError("max_tokens is too large")) == 4096


# ── the check, and the proposal ──────────────────────────────────────────────

UNBALANCED = '(poultry worker* OR veterinarian*) AND (h5n1 OR "avian influenza" AND seroprevalence'


def test_an_unbalanced_query_yields_a_proposal_and_the_query_used_is_still_the_original(model):
    model(_kept())                                       # the model misses it
    out = _post(UNBALANCED, "en").json()
    assert out["general"] == UNBALANCED and out["pubmed"] == UNBALANCED
    assert out["proposed_query"] == UNBALANCED + ")"
    assert "never closed" in out["proposal_reason"]
    assert out["proposal_issues"] == ["unclosed_parenthesis"]
    # In the interface language.
    out_fr = _post(UNBALANCED, "fr").json()
    assert out_fr["proposal_reason"].startswith("Parenthèses ouvrantes jamais fermées")
    assert chr(0x2014) not in out["proposal_reason"] + out_fr["proposal_reason"]


def test_a_correct_query_yields_no_proposal(model):
    model(_kept())
    q = '(poultry worker* OR veterinarian*) AND (h5n1 OR "avian influenza")'
    out = _post(q, "en").json()
    assert out["general"] == q and out["proposed_query"] is None and out["proposal_reason"] is None
    assert out["proposal_issues"] == []


def test_the_models_proposal_is_offered_not_applied(model):
    entered = '(poultry OR farm*) ANDD (h5n1 OR "avian influenza")'
    fixed = '(poultry OR farm*) AND (h5n1 OR "avian influenza")'
    model(_kept(fixed, {"en": "ANDD is not an operator: it was read as a word. AND is proposed.",
                        "fr": "ANDD n'est pas un opérateur : il était lu comme un mot. AND est proposé."}))
    out = _post(entered, "en").json()
    assert out["general"] == entered and out["proposed_query"] == fixed
    assert out["proposal_reason"].startswith("ANDD is not an operator")
    assert "model" in out["proposal_issues"]
    assert _post(entered, "fr").json()["proposal_reason"].startswith("ANDD n'est pas")


def test_a_model_fix_that_leaves_a_parenthesis_open_is_completed(model):
    entered = '(poultry OR farm*) ANDD (h5n1 OR "avian influenza"'
    model(_kept('(poultry OR farm*) AND (h5n1 OR "avian influenza"', {"en": "ANDD is not an operator."}))
    out = S._strategy_in_lang(S._generate_search_strategy(entered), "en")
    assert out["proposed_query"] == '(poultry OR farm*) AND (h5n1 OR "avian influenza")'
    assert out["proposal_reason"].startswith("ANDD is not an operator.") and "never closed" in out["proposal_reason"]


def test_a_proposal_that_drops_much_of_a_long_strategy_is_not_offered(model):
    strategy = _long_strategy(3000)
    model(_kept('("avian influenza") AND (poultry)', {"en": "Simplified.", "fr": "Simplifiée."}))
    out = _post(strategy, "en").json()
    assert out["general"] == strategy and out["proposed_query"] is None


def test_a_natural_query_is_translated_and_needs_no_proposal(model):
    model({"input_kind": "natural", "general": '("poultry worker" OR farmer) AND (h5n1 OR "avian influenza")',
           "pubmed": '"Influenza in Birds"[Mesh] AND "Occupational Exposure"[Mesh]',
           "explanation": "Two concept groups.", "synonyms": [], "proposed_query": "", "proposal_reason": ""})
    q = "occupational exposure of poultry workers to avian influenza"
    out = _post(q, "en").json()
    assert out["input_kind"] == "natural" and out["entered_query"] == q
    assert out["general"].startswith('("poultry worker" OR farmer)') and out["proposed_query"] is None


def test_without_the_model_an_unbalanced_strategy_is_still_flagged(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setattr(S, "_STRATEGY_CACHE", {})
    monkeypatch.setattr(S, "engine", _NoDb())
    out = S._strategy_in_lang(S._generate_search_strategy(UNBALANCED), "en")
    assert out["general"] == UNBALANCED and out["degraded"] is True
    assert out["proposed_query"] == UNBALANCED + ")"


@pytest.mark.parametrize("query,kinds,fixed", [
    ('(a OR b)) AND c', ["unmatched_closing_parenthesis"], "(a OR b) AND c"),
    ('((a OR b) AND (c', ["unclosed_parenthesis"], "((a OR b) AND (c))"),
    ('"avian influenza OR h5n1', ["unpaired_quote"], "avian influenza OR h5n1"),
    ('“avian influenza” AND poultry', ["typographic_quotes"], '"avian influenza" AND poultry'),
    ('« avian influenza » AND poultry', ["typographic_quotes"], '"avian influenza" AND poultry'),
    ('"influenza A(H5N1" AND poultry', [], None),        # a parenthesis inside a phrase is a character
    ('(h5n1[tiab] OR "bird flu"[tiab]) AND poultry[Mesh]', [], None),
])
def test_the_check_without_a_model(query, kinds, fixed):
    out = S._boolean_syntax_check(query)
    assert [i["kind"] for i in out["issues"]] == kinds
    assert out["fixed"] == fixed


def test_the_check_reads_the_parentheses_as_the_search_does():
    """An extra closing parenthesis stops the search's parser: what follows is IGNORED. That
    is why it must be flagged even when the model misses it."""
    ast = S._parse_boolean_ast(S._tokenize_boolean("(a OR b)) AND c"))
    assert ast == ("or", [("term", "a"), ("term", "b")])       # c is gone
    fixed = S._boolean_syntax_check("(a OR b)) AND c")["fixed"]
    assert S._parse_boolean_ast(S._tokenize_boolean(fixed)) == ("and", [("or", [("term", "a"), ("term", "b")]),
                                                                        ("term", "c")])


def test_a_strategy_is_cached_under_its_exact_text_and_a_long_one_under_its_digest():
    assert S._strategy_cache_key("a AND b", True) != S._strategy_cache_key("A AND B", True)
    assert S._strategy_cache_key("flu  Forecasting", False) == "flu forecasting"
    long_key = S._strategy_cache_key(_long_strategy(6000), True)
    assert long_key.startswith("b:sha256:") and len(long_key) < 100


def test_a_scenario_never_stores_a_pending_proposal(db_conn):
    main._ensure_user_scenarios_table()
    strategy = {"general": "(a OR b) AND c", "pubmed": "(a OR b) AND c", "entered_query": "(a OR b) AND c",
                "input_kind": "boolean", "proposed_query": "(a OR b) AND (c)",
                "proposal_reason": "x", "proposal_reasons": {"en": "x", "fr": "x"},
                "proposal_issues": ["model"]}
    client = TestClient(main.app)
    r = client.post("/user-scenarios", headers=KEY, json={
        "name": "pending proposal", "query": "(a OR b) AND c zq7k", "mode": "boolean",
        "search_strategy": strategy})
    assert r.status_code == 201, r.text[:300]
    sid = r.json()["id"]
    try:
        with db_conn.cursor() as cur:
            cur.execute("SELECT search_strategy FROM user_scenarios WHERE id = %s", (sid,))
            stored = cur.fetchone()[0]
        assert stored["general"] == "(a OR b) AND c" and stored["entered_query"] == "(a OR b) AND c"
        assert not any(k in stored for k in ("proposed_query", "proposal_reason", "proposal_reasons",
                                             "proposal_issues")), stored
    finally:
        with db_conn.cursor() as cur:
            cur.execute("DELETE FROM user_scenarios WHERE id = %s", (sid,))
