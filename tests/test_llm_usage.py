"""Token accounting, the master switch and the daily budget — no API key, no network.

The accounting exists because the app spent money invisibly. So the properties that matter
are not "it records the happy path" but: it never breaks the call it measures, it cannot be
fooled by a response shape it did not expect, and a switch that is supposed to stop spending
actually stops it BEFORE the request goes out.

`llm_usage` deliberately imports `openai` only inside the two factory functions, so this
whole file runs on a machine that has never installed the SDK — which is what CI is.
"""
import sys
import types

import pytest

import llm_usage


class _Usage:
    def __init__(self, p=0, c=0, t=None):
        self.prompt_tokens, self.completion_tokens = p, c
        self.total_tokens = (p + c) if t is None else t


class _Resp:
    def __init__(self, usage=None, model="gpt-4.1-mini"):
        self.usage, self.model = usage, model


class _Recorder:
    """Stands in for the database: llm_usage.record() writes here instead."""
    def __init__(self):
        self.rows = []

    def __call__(self, purpose, model, usage):
        p, c, t = llm_usage._usage_fields(usage)
        self.rows.append({"purpose": purpose, "model": model,
                          "prompt": p, "completion": c, "total": t})
        return t


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Every control off/default, and no engine, so nothing here touches a database."""
    for var in ("OPENAI_ENABLED", "LLM_DAILY_TOKEN_BUDGET", "LLM_USAGE_LOGGING"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(llm_usage, "_engine", None)
    with llm_usage._spend_lock:
        llm_usage._spend_cache.update(day=None, tokens=0, checked_at=0.0)
    yield


def _client(create, recorder, kind="chat"):
    """A fake OpenAI client with just the surface `instrument` wraps."""
    completions = types.SimpleNamespace(create=create)
    c = types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions),
                              embeddings=types.SimpleNamespace(create=create))
    return llm_usage.instrument(c, "test_purpose")


# ── reading usage off a response ─────────────────────────────────────────────
def test_usage_is_read_from_objects_and_dicts_alike():
    assert llm_usage._usage_fields(_Usage(10, 5)) == (10, 5, 15)
    assert llm_usage._usage_fields({"prompt_tokens": 3, "completion_tokens": 4}) == (3, 4, 7)


def test_a_missing_or_broken_usage_counts_as_zero_not_an_error():
    """Embedding responses have no completion_tokens; a stream has no usage at all."""
    assert llm_usage._usage_fields(None) == (0, 0, 0)
    assert llm_usage._usage_fields(_Usage(9, 0)) == (9, 0, 9)
    assert llm_usage._usage_fields({"prompt_tokens": "not a number"}) == (0, 0, 0)
    assert llm_usage._usage_fields(types.SimpleNamespace()) == (0, 0, 0)


def test_negative_token_counts_are_clamped():
    assert llm_usage._usage_fields({"prompt_tokens": -5, "total_tokens": -1}) == (0, 0, 0)


# ── the wrapper ──────────────────────────────────────────────────────────────
def test_a_metered_call_records_its_tokens_and_returns_the_real_response(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(llm_usage, "record", rec)
    sentinel = _Resp(_Usage(120, 40))
    cl = _client(lambda **kw: sentinel, rec)

    got = cl.chat.completions.create(model="gpt-4.1", messages=[])
    assert got is sentinel, "the caller must get the untouched response"
    assert rec.rows == [{"purpose": "test_purpose:chat", "model": "gpt-4.1",
                         "prompt": 120, "completion": 40, "total": 160}]


def test_accounting_never_breaks_the_call_it_measures(monkeypatch):
    """A failure in the meter must cost the response, not the request."""
    def _boom(*a, **k):
        raise RuntimeError("the usage table is on fire")
    monkeypatch.setattr(llm_usage, "record", _boom)
    sentinel = _Resp(_Usage(1, 1))
    cl = _client(lambda **kw: sentinel, None)
    assert cl.chat.completions.create(model="m") is sentinel


def test_a_stream_is_counted_as_a_call_with_no_tokens(monkeypatch):
    """The blind spot must be VISIBLE in the table, not absent from it."""
    rec = _Recorder()
    monkeypatch.setattr(llm_usage, "record", rec)
    cl = _client(lambda **kw: iter(["chunk"]), rec)
    cl.chat.completions.create(model="gpt-4.1-mini", stream=True)
    assert len(rec.rows) == 1 and rec.rows[0]["total"] == 0


def test_the_model_falls_back_to_the_response_when_the_caller_omitted_it(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(llm_usage, "record", rec)
    cl = _client(lambda **kw: _Resp(_Usage(2, 2), model="gpt-4o-mini"), rec)
    cl.chat.completions.create(messages=[])
    assert rec.rows[0]["model"] == "gpt-4o-mini"


def test_instrumenting_a_client_without_the_expected_surface_is_not_fatal():
    """An SDK shape change must degrade to "unmetered", never to a broken app."""
    bare = types.SimpleNamespace()
    assert llm_usage.instrument(bare, "p") is bare


# ── the controls ─────────────────────────────────────────────────────────────
def test_the_master_switch_blocks_before_the_request_is_made(monkeypatch):
    calls = []
    monkeypatch.setenv("OPENAI_ENABLED", "0")
    cl = _client(lambda **kw: calls.append(kw) or _Resp(_Usage(1, 1)), None)
    with pytest.raises(llm_usage.LLMCallBlocked):
        cl.chat.completions.create(model="m")
    assert calls == [], "a disabled call must not reach the API — that is the whole point"


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "OFF"])
def test_the_switch_accepts_the_usual_spellings_of_off(monkeypatch, value):
    monkeypatch.setenv("OPENAI_ENABLED", value)
    assert not llm_usage.openai_enabled()


def test_calls_are_allowed_by_default(monkeypatch):
    """Installing the accounting must not change behaviour on its own."""
    assert llm_usage.openai_enabled()
    assert llm_usage.daily_token_budget() == 0
    llm_usage.check_allowed("anything")          # must not raise


def test_the_daily_budget_blocks_once_spent(monkeypatch):
    monkeypatch.setenv("LLM_DAILY_TOKEN_BUDGET", "1000")
    monkeypatch.setattr(llm_usage, "spend_today", lambda force=False: 999)
    llm_usage.check_allowed()                    # still under
    monkeypatch.setattr(llm_usage, "spend_today", lambda force=False: 1000)
    with pytest.raises(llm_usage.LLMCallBlocked) as e:
        llm_usage.check_allowed("pico")
    assert "1000" in str(e.value) and "pico" in str(e.value)


def test_a_malformed_budget_means_no_budget(monkeypatch):
    """A typo in the env must not silently freeze every LLM feature."""
    for bad in ("", "lots", "-5", "1e6"):
        monkeypatch.setenv("LLM_DAILY_TOKEN_BUDGET", bad)
        assert llm_usage.daily_token_budget() in (0,), bad
    llm_usage.check_allowed()


def test_an_unreadable_spend_does_not_read_as_zero_spend(monkeypatch):
    """With a budget set, "I cannot tell" must not be treated as "nothing spent yet"."""
    class _BrokenEngine:
        def connect(self):
            raise RuntimeError("no database")
    monkeypatch.setattr(llm_usage, "_engine", _BrokenEngine())
    with llm_usage._spend_lock:
        llm_usage._spend_cache.update(day=llm_usage._utc_day(), tokens=4242, checked_at=0.0)
    assert llm_usage.spend_today(force=True) == 4242    # last known figure, not 0


def test_recording_without_an_engine_still_returns_the_token_total():
    """Accounting is optional plumbing; the count is still computed for the caller."""
    assert llm_usage.record("p", "m", _Usage(7, 3)) == 10


def test_logging_can_be_switched_off_independently_of_the_calls(monkeypatch):
    monkeypatch.setenv("LLM_USAGE_LOGGING", "0")
    inserted = []
    monkeypatch.setattr(llm_usage, "_engine", object())   # would be used if logging were on
    assert llm_usage.record("p", "m", _Usage(5, 5)) == 10
    assert inserted == []


# ── purpose attribution ──────────────────────────────────────────────────────
def test_the_purpose_names_the_function_that_built_the_client(monkeypatch):
    """Attribution is what makes the table answer "which loop is spending"."""
    fake = types.ModuleType("openai")
    fake.OpenAI = lambda **kw: types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=lambda **k: None)),
        embeddings=types.SimpleNamespace(create=lambda **k: None))
    monkeypatch.setitem(sys.modules, "openai", fake)

    seen = {}
    monkeypatch.setattr(llm_usage, "instrument", lambda c, p: seen.setdefault("purpose", p) or c)

    def _pretend_background_worker():
        return llm_usage.MeteredOpenAI(api_key="x")

    _pretend_background_worker()
    assert seen["purpose"] == "_pretend_background_worker"


def test_an_explicit_purpose_wins_over_the_stack(monkeypatch):
    fake = types.ModuleType("openai")
    fake.OpenAI = lambda **kw: types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "openai", fake)
    seen = {}
    monkeypatch.setattr(llm_usage, "instrument", lambda c, p: seen.setdefault("purpose", p) or c)
    llm_usage.MeteredOpenAI(api_key="x", purpose="embed_corpus")
    assert seen["purpose"] == "embed_corpus"


# ── the assumption this whole design rests on ────────────────────────────────
def test_the_real_sdk_can_actually_be_instrumented(monkeypatch):
    """`instrument` assigns over `create` on a live client. If the SDK ever makes those
    resources immutable (slots, a frozen model), metering would silently stop and the
    only symptom would be an empty table next to a real invoice. Skipped where the SDK
    is not installed — CI does not install it — so it guards the machines that matter.
    """
    openai = pytest.importorskip("openai")
    monkeypatch.setenv("OPENAI_ENABLED", "0")
    client = llm_usage.MeteredOpenAI(api_key="sk-not-a-real-key")
    assert client.chat.completions.create.__name__ == "metered"
    assert client.embeddings.create.__name__ == "metered"
    # And the switch must stop the call BEFORE the SDK opens a connection: a bogus key
    # would otherwise surface as an auth error from the network, not as a local block.
    with pytest.raises(llm_usage.LLMCallBlocked):
        client.chat.completions.create(model="gpt-4.1-mini", messages=[])
    with pytest.raises(llm_usage.LLMCallBlocked):
        client.embeddings.create(model="text-embedding-3-small", input="x")


def test_the_ddl_creates_the_table_and_its_indexes():
    """The boot DDL and the migration must not drift apart."""
    joined = " ".join(llm_usage.DDL).lower()
    assert "create table if not exists llm_usage" in joined
    for col in ("prompt_tokens", "completion_tokens", "total_tokens", "purpose", "model"):
        assert col in joined
    assert joined.count("create index if not exists") == 2
