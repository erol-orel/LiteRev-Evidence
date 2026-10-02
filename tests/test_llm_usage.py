"""Token accounting, the master switch and the daily budget - no API key, no network.

The accounting exists because the app spent money invisibly. So the properties that matter
are not "it records the happy path" but: it never breaks the call it measures, it cannot be
fooled by a response shape it did not expect, and a switch that is supposed to stop spending
actually stops it BEFORE the request goes out.

`llm_usage` deliberately imports `openai` only inside the two factory functions, so this
whole file runs on a machine that has never installed the SDK - which is what CI is.
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
    for var in ("OPENAI_ENABLED", "LLM_DAILY_TOKEN_BUDGET", "LLM_USAGE_LOGGING",
                "LLM_REASONING_EFFORT", "LLM_MODEL_BULK", "LLM_MODEL_WRITE",
                "LLM_MODEL_CHAT", "EMBEDDING_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(llm_usage, "_engine", None)
    with llm_usage._spend_lock:
        llm_usage._spend_cache.update(day=None, tokens=0, checked_at=0.0)
    # Repairs are learned per process and would otherwise leak from one test into the
    # next, which is exactly the kind of order dependence that makes a suite lie.
    with llm_usage._REPAIRS_LOCK:
        llm_usage._REPAIRS.clear()
    yield
    with llm_usage._REPAIRS_LOCK:
        llm_usage._REPAIRS.clear()


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
    assert calls == [], "a disabled call must not reach the API - that is the whole point"


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
    is not installed - CI does not install it - so it guards the machines that matter.
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


# ── the shape of the request ─────────────────────────────────────────────────
# The failure these guard against is not an exception a user would see. Every LLM call
# site degrades to "this feature is unavailable", so a request shape the API refuses
# produces no error page at all: PICO stops filling, briefs come back empty, the
# assistant apologises, /health stays green.
def test_a_token_ceiling_is_translated_for_the_generation_that_renamed_it():
    """`max_tokens` is a 400 on the gpt-5 generation. Twenty call sites pass it."""
    sent = llm_usage.shape_chat_kwargs({"model": "gpt-5.6-luna", "max_tokens": 800})
    assert sent == {"model": "gpt-5.6-luna", "max_completion_tokens": 800,
                    "reasoning_effort": "none"}


def test_a_model_of_the_older_generation_is_left_exactly_as_the_call_site_wrote_it():
    """The translation must not break a rollback: LLM_MODEL_BULK=gpt-4.1-mini has to keep
    working, and gpt-4.1 refuses `max_completion_tokens` and `reasoning_effort` alike."""
    asked = {"model": "gpt-4.1-mini", "max_tokens": 800, "temperature": 0, "seed": 42}
    assert llm_usage.shape_chat_kwargs(asked) == asked


def test_an_explicit_ceiling_in_the_new_name_is_not_overwritten():
    sent = llm_usage.shape_chat_kwargs(
        {"model": "gpt-5.6-luna", "max_tokens": 800, "max_completion_tokens": 50})
    assert sent["max_completion_tokens"] == 50 and "max_tokens" not in sent


def test_the_default_effort_is_none_so_the_temperature_and_seed_survive():
    """What the extractions depend on: temperature 0 and seed 42 are only accepted while
    the model is not reasoning, and they are what makes a measured precision repeatable."""
    sent = llm_usage.shape_chat_kwargs(
        {"model": "gpt-5.6-luna", "temperature": 0, "seed": 42, "max_tokens": 300})
    assert sent["reasoning_effort"] == "none"
    assert sent["temperature"] == 0 and sent["seed"] == 42


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_asking_the_model_to_reason_drops_the_sampling_parameters_it_would_refuse(
        monkeypatch, effort):
    monkeypatch.setenv("LLM_REASONING_EFFORT", effort)
    sent = llm_usage.shape_chat_kwargs(
        {"model": "gpt-5.6-luna", "temperature": 0.2, "top_p": 0.9,
         "presence_penalty": 0.1, "frequency_penalty": 0.1})
    assert sent == {"model": "gpt-5.6-luna", "reasoning_effort": effort}


def test_the_api_default_can_be_asked_for_explicitly_and_sends_nothing(monkeypatch):
    monkeypatch.setenv("LLM_REASONING_EFFORT", "default")
    sent = llm_usage.shape_chat_kwargs({"model": "gpt-5.6-luna", "temperature": 0.2})
    assert "reasoning_effort" not in sent
    # The API's own default is to reason, so the temperature still cannot be sent.
    assert "temperature" not in sent


def test_a_typo_in_the_effort_does_not_silently_buy_the_expensive_default(monkeypatch):
    """Unreadable means "none", not "whatever the API charges for by default"."""
    monkeypatch.setenv("LLM_REASONING_EFFORT", "hihg")
    assert llm_usage.reasoning_effort() == "none"


def test_a_call_site_may_ask_for_its_own_effort(monkeypatch):
    monkeypatch.setenv("LLM_REASONING_EFFORT", "none")
    sent = llm_usage.shape_chat_kwargs(
        {"model": "gpt-5.6-luna", "reasoning_effort": "high", "temperature": 0.2})
    assert sent["reasoning_effort"] == "high" and "temperature" not in sent


@pytest.mark.parametrize("model,legacy", [
    ("gpt-4.1", True), ("gpt-4.1-mini", True), ("gpt-4o-mini", True),
    ("gpt-3.5-turbo", True), ("ft:gpt-4.1-mini:unige::a7f3k2", True),
    ("gpt-5.6-luna", False), ("gpt-5.4-nano", False), ("openai/gpt-5.6-luna", False),
    ("gpt-6-luna", False), ("o3-mini", False), ("some-model-released-next-year", False),
])
def test_which_generation_a_name_belongs_to(model, legacy):
    """An unknown name is assumed to take the NEW shape: that is where the API went, and
    a wrong guess costs one refused request and a repair learned from it."""
    assert llm_usage.legacy_request_shape(model) is legacy


def test_every_model_this_repository_ships_is_shaped_so_a_ceiling_survives():
    """A behaviour, not a name: whatever the defaults become, a call site that asks for a
    ceiling must end up with one the named model accepts."""
    for role in ("bulk", "write", "chat"):
        sent = llm_usage.shape_chat_kwargs({"model": llm_usage.model_for(role),
                                            "max_tokens": 800})
        assert "max_tokens" not in sent or llm_usage.legacy_request_shape(sent["model"])
        assert sent.get("max_completion_tokens", sent.get("max_tokens")) == 800


def test_an_embedding_request_is_not_shaped(monkeypatch):
    """Embeddings take neither a ceiling nor an effort; adding one would be a 400."""
    rec = _Recorder()
    monkeypatch.setattr(llm_usage, "record", rec)
    seen = {}
    cl = _client(lambda **kw: seen.update(kw) or _Resp(_Usage(5, 0)), rec)
    cl.embeddings.create(model="text-embedding-3-small", input="x")
    assert seen == {"model": "text-embedding-3-small", "input": "x"}


# ── learning the shape from the API's own refusals ───────────────────────────
class _Refuses:
    """A fake `create` that refuses the named parameters the way the API does, once each,
    and records every attempt it was given."""
    def __init__(self, *refusals: str):
        self.refusals = list(refusals)
        self.attempts: list[dict] = []

    def __call__(self, **kwargs):
        self.attempts.append(dict(kwargs))
        for param in list(self.refusals):
            if param in kwargs:
                self.refusals.remove(param)
                if param == "max_tokens":
                    raise RuntimeError(
                        "Error code: 400 - Unsupported parameter: 'max_tokens' is not "
                        "supported with this model. Use 'max_completion_tokens' instead.")
                raise RuntimeError(
                    f"Error code: 400 - Unsupported value: '{param}' does not support 0.2 "
                    "with this model. Only the default (1) value is supported.")
        return _Resp(_Usage(3, 1))


def test_a_refused_parameter_is_renamed_as_the_api_says_and_the_call_retried(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(llm_usage, "record", rec)
    create = _Refuses("max_tokens")
    # A legacy NAME with a model that behaves like the new generation: exactly the case
    # the prefix table gets wrong, and the one that must still work.
    cl = _client(create, rec)
    cl.chat.completions.create(model="gpt-4.1-impostor", max_tokens=800, messages=[])
    assert len(create.attempts) == 2
    assert create.attempts[0]["max_tokens"] == 800
    assert create.attempts[1]["max_completion_tokens"] == 800
    assert "max_tokens" not in create.attempts[1]
    assert len(rec.rows) == 1, "the retry is one call, recorded once"


def test_a_refusal_is_remembered_so_the_next_call_pays_for_it_only_once(monkeypatch):
    monkeypatch.setattr(llm_usage, "record", _Recorder())
    create = _Refuses("temperature")
    cl = _client(create, None)
    cl.chat.completions.create(model="gpt-5.6-luna", temperature=0.2, messages=[])
    cl.chat.completions.create(model="gpt-5.6-luna", temperature=0.2, messages=[])
    assert len(create.attempts) == 3, "two refused once, then shaped before it is sent"
    assert "temperature" not in create.attempts[2]
    assert llm_usage.request_policy()["learned_repairs"] == {
        "gpt-5.6-luna": {"temperature": "dropped"}}


def test_an_sdk_too_old_to_know_the_parameter_degrades_instead_of_dying(monkeypatch):
    """`create` takes named parameters and no **kwargs, so an installation older than
    `reasoning_effort` raises TypeError before any request - on every call in the
    application. requirements.txt rules it out; this makes it survivable anyway, at the
    API's own default effort."""
    monkeypatch.setattr(llm_usage, "record", _Recorder())
    attempts = []

    def _create(**kwargs):
        attempts.append(dict(kwargs))
        if "reasoning_effort" in kwargs:
            raise TypeError("create() got an unexpected keyword argument 'reasoning_effort'")
        return _Resp(_Usage(2, 1))
    cl = _client(_create, None)
    cl.chat.completions.create(model="gpt-5.6-luna", max_tokens=16, messages=[])
    assert len(attempts) == 2 and "reasoning_effort" not in attempts[1]
    assert attempts[1]["max_completion_tokens"] == 16


def test_a_refusal_naming_something_we_did_not_send_is_the_callers_to_see(monkeypatch):
    """A field inside `messages` is not a parameter we can drop; retrying would be a
    loop that spends money without ever changing the request."""
    monkeypatch.setattr(llm_usage, "record", _Recorder())
    attempts = []

    def _create(**kwargs):
        attempts.append(kwargs)
        raise RuntimeError("Error code: 400 - Unsupported value: 'messages[0].role' does "
                           "not support 'tool' with this model.")
    cl = _client(_create, None)
    with pytest.raises(RuntimeError):
        cl.chat.completions.create(model="gpt-5.6-luna", messages=[])
    assert len(attempts) == 1


@pytest.mark.parametrize("message", [
    "Error code: 429 - Rate limit reached for gpt-5.6-luna",
    "Error code: 400 - This model's maximum context length is 1048576 tokens",
    "Connection error.",
])
def test_an_error_that_names_no_parameter_is_raised_at_once(monkeypatch, message):
    monkeypatch.setattr(llm_usage, "record", _Recorder())
    attempts = []

    def _create(**kwargs):
        attempts.append(kwargs)
        raise RuntimeError(message)
    cl = _client(_create, None)
    with pytest.raises(RuntimeError):
        cl.chat.completions.create(model="gpt-5.6-luna", temperature=0.2, messages=[])
    assert len(attempts) == 1, "retrying these is how a bounded loop becomes a bill"


def test_the_repair_loop_cannot_run_forever(monkeypatch):
    """Every round removes a parameter, so a model that keeps refusing the same one runs
    out of request to repair and the error reaches the caller."""
    monkeypatch.setattr(llm_usage, "record", _Recorder())
    attempts = []

    def _create(**kwargs):
        attempts.append(kwargs)
        raise RuntimeError("400 - Unsupported value: 'temperature' does not support 0.2 "
                           "with this model. Only the default (1) value is supported.")
    cl = _client(_create, None)
    with pytest.raises(RuntimeError):
        cl.chat.completions.create(model="gpt-5.6-luna", temperature=0.2, messages=[])
    assert len(attempts) == 2, "sent, refused, sent without it, refused for no reason"


def test_a_blocked_call_is_never_retried(monkeypatch):
    """The switch and the budget are local refusals; they must not look repairable."""
    monkeypatch.setenv("OPENAI_ENABLED", "0")
    create = _Refuses()
    cl = _client(create, None)
    with pytest.raises(llm_usage.LLMCallBlocked):
        cl.chat.completions.create(model="gpt-5.6-luna", max_tokens=10)
    assert create.attempts == []


def test_the_policy_says_what_every_request_will_look_like():
    """/health reports this. A deployment whose model table is wrong says so here."""
    policy = llm_usage.request_policy()
    assert policy["reasoning_effort"] == "none"
    assert policy["sampling_params"] == "sent"
    assert policy["learned_repairs"] == {}
    assert set(policy["token_ceiling_param"]) == {"bulk", "write", "chat"}


def test_the_ddl_creates_the_table_and_its_indexes():
    """The boot DDL and the migration must not drift apart."""
    joined = " ".join(llm_usage.DDL).lower()
    assert "create table if not exists llm_usage" in joined
    for col in ("prompt_tokens", "completion_tokens", "total_tokens", "purpose", "model"):
        assert col in joined
    assert joined.count("create index if not exists") == 2
