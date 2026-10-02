"""Token accounting, a master switch and a daily budget for every OpenAI call.

The app made LLM calls from about thirty places and recorded NOTHING: no call site
read `response.usage`, so the only signal that something was spending was the invoice.
Two token leaks had already been found and fixed by reading code (see the comments in
`main.py` around the PICO worker) because there was no other way to find them.

Every call site constructs its client with a local `from openai import OpenAI as X`.
That single shared line is the seam: point those imports here instead and each call is
metered, without touching the calls themselves.

    from llm_usage import MeteredOpenAI as OpenAI      # was: from openai import OpenAI

`MeteredOpenAI` returns a real client whose `chat.completions.create` and
`embeddings.create` record `response.usage` into the `llm_usage` table, tagged with the
name of the function that built the client. Everything else on the client is untouched.

That seam carries two more things, both of which exist because the call sites are many
and the model is one setting: the model NAME each job uses (`model_for`, so that a switch
is an environment variable rather than an edit in twenty files) and the request SHAPE the
named model accepts (`shape_chat_kwargs`, because the gpt-5 generation refuses the
`max_tokens` and the `temperature` that the gpt-4 generation required). A chat request is
therefore rewritten on its way out; an embedding request is not.

Three controls, all off by default so that installing this changes no behaviour:

  OPENAI_ENABLED=0             refuse every call, immediately, without a request.
  LLM_DAILY_TOKEN_BUDGET=N     refuse once N tokens have been recorded since midnight
                               UTC (0 = no budget, the default).
  LLM_USAGE_LOGGING=0          stop recording (the accounting itself, not the calls).

A refusal raises `LLMCallBlocked`. Call sites already wrap their LLM calls in
`try/except Exception`, so a blocked call degrades to "this feature is unavailable"
rather than a 500 - the same shape as an API outage, which is what a spend freeze is.

NOT metered: the two streaming chat calls (`stream=True`). The API only reports usage
for a stream when asked via `stream_options`, which appends a final chunk with empty
`choices` - enough to break a consumer that assumes every chunk has one. Adding that
needs a test against the live API, which is exactly what this environment cannot do.
They are counted as calls with zero tokens, so they show up in the table as a known
blind spot rather than silently missing. The master switch and budget DO apply to them.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from datetime import datetime, timezone

logger = logging.getLogger("llm-usage")


# ── Which model does which job ───────────────────────────────────────────────
# The model name was written out at 27 call sites, which is why two of them were still
# on gpt-4o-mini long after everything else had moved on: upgrading meant finding and
# editing all 27, so in practice nobody did. Here they are named by JOB, so that a job
# whose economics differ can be moved on its own.
#
# The three jobs happen to share one model today. gpt-5.6-luna is cheaper per token than
# the gpt-4.1-mini it replaces for `bulk` and `chat` ($0.20/$1.20 per million against
# $0.40/$1.60) and an order of magnitude cheaper than the gpt-4.1 it replaces for
# `write` ($2.00/$8.00), so there is no longer a cost argument for a smaller model on the
# bulk path. What still separates the jobs is how hard the model is asked to think, and
# that is `reasoning_effort` below, not the name.
#
# Changing a model INVALIDATES any measurement made against the old one. The gold
# standard records which model it validated (scripts/gold_standard.py), and the figures
# it produced do not carry over to a different one.
_MODEL_ROLES = {
    # Per-article work over the whole corpus: PICO, concepts, epidemiological parameters,
    # cluster summaries, query translation. Thousands of calls, so cost dominates and the
    # output is consumed by code rather than read.
    "bulk": ("LLM_MODEL_BULK", "gpt-5.6-luna"),
    # Prose a person reads and may quote: the evidence brief, the recommended actions,
    # the variables and model spec. Few calls, and the quality is the product.
    "write": ("LLM_MODEL_WRITE", "gpt-5.6-luna"),
    # The assistant's answers. Interactive, so latency counts as much as quality. These
    # two call sites were the ones left on gpt-4o-mini.
    "chat": ("LLM_MODEL_CHAT", "gpt-5.6-luna"),
    # CAREFUL. Every vector already in document_chunk.embedding was produced by this
    # model at 1536 dimensions, and vectors from two models are not comparable. Changing
    # it means re-embedding the entire corpus. A model of a different dimension fails
    # loudly on insert (the column is vector(1536)), which is the good case; one of the
    # SAME dimension would silently degrade every similarity in the database.
    "embedding": ("EMBEDDING_MODEL", "text-embedding-3-small"),
}


def model_for(role: str) -> str:
    """The model to use for a job, overridable per role by environment variable.

    Read at CALL time, not at import: a deployment can change a model and restart the
    API without a code change, which is the point of having this at all."""
    try:
        var, default = _MODEL_ROLES[role]
    except KeyError:
        raise ValueError(f"unknown model role {role!r}; known: {sorted(_MODEL_ROLES)}") from None
    return os.getenv(var) or default


def models_in_use() -> dict[str, str]:
    """What every role resolves to right now. Reported by /health so a deployment can be
    asked which models it is actually running, and recorded by any measurement that
    depends on them."""
    return {role: model_for(role) for role in _MODEL_ROLES}


# ── The request shape each generation of model accepts ───────────────────────
# gpt-4.1 took `max_tokens` and any `temperature`. The gpt-5 generation takes neither:
# `max_tokens` is refused outright (400 "Unsupported parameter ... use
# `max_completion_tokens` instead"), and a temperature other than the default is refused
# while the model is reasoning - which it does by default, at effort "medium".
#
# Twenty call sites pass `max_tokens` and most pass a temperature. Pointing LLM_MODEL_*
# at a gpt-5 model without translating the request would therefore 400 EVERY LLM call in
# the application, and each one would disappear into its own `except Exception` branch:
# no PICO, no brief, no assistant, no error page, just features that quietly return
# nothing. So the translation lives here, in the one place every call already passes
# through, for the same reason the model names do: a call site expresses an intent (a
# ceiling on the answer, a low temperature) that does not change when the generation
# answering it changes, and a switch that needs a twenty-file edit does not get made.

#: Families that still take the old shape. A closed set: everything released since
#: (gpt-5, gpt-6, the o-series) takes the new one, so an UNKNOWN name is assumed new.
#: That is the direction the API moved, and a wrong guess either way costs one refused
#: request and a repair learned from it - see `_create_with_repair`.
_LEGACY_REQUEST_SHAPE = ("gpt-4", "gpt-3.5", "chatgpt-4o", "text-davinci", "davinci",
                         "babbage", "curie", "ada")

#: Parameters the gpt-5 generation only accepts while it is NOT reasoning.
_SAMPLING_PARAMS = ("temperature", "top_p", "presence_penalty", "frequency_penalty")

#: Efforts the gpt-5 generation accepts, cheapest first.
_REASONING_EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")


def _model_family(model: str) -> str:
    """A model name reduced to what the prefixes below can be matched against: no
    provider prefix (`openai/gpt-5.6-luna`), no fine-tuning wrapper (`ft:gpt-4.1:org`)."""
    m = (model or "").strip().lower().rsplit("/", 1)[-1]
    return m[3:] if m.startswith("ft:") else m


def legacy_request_shape(model: str) -> bool:
    """True if this model wants `max_tokens` and accepts a free temperature."""
    return _model_family(model).startswith(_LEGACY_REQUEST_SHAPE)


def reasoning_effort() -> str:
    """How hard every chat call asks the model to think, or "" to send nothing and let
    the API choose (which means "medium").

    "none" by default, and that default is a decision with three consequences - each of
    them the reason for it:

      - a temperature is only accepted while the effort is "none", and this application
        asks for temperature 0 and seed 42 wherever the answer is parsed as JSON, so the
        reproducibility of every extraction hangs on it;
      - reasoning tokens are billed as output and are absent from the answer, so effort
        "medium" over a corpus of several thousand abstracts spends an unknown multiple
        of what the extraction appears to cost;
      - reasoning tokens come out of the SAME ceiling as the answer, so the 300- and
        800-token ceilings the extraction call sites pass could be consumed entirely by
        thinking and return an empty string - a silent extraction failure, the worst
        shape a failure can take here.

    Raise it (LLM_REASONING_EFFORT=low|medium|high|xhigh|max) for prose, knowing that it
    strips the temperature from every call and needs larger ceilings. The value `default`
    sends nothing at all and accepts the API's own choice."""
    raw = (os.getenv("LLM_REASONING_EFFORT") or "none").strip().lower()
    if raw in _REASONING_EFFORTS:
        return raw
    if raw in ("default", "api", "api-default", ""):
        return ""
    logger.warning(f"LLM_REASONING_EFFORT={raw!r} is not one of {_REASONING_EFFORTS}; "
                   "using 'none'")
    return "none"


def shape_chat_kwargs(kwargs: dict) -> dict:
    """A chat request as the model named in it will actually accept it.

    Pure, and a copy: the caller's dict is never modified."""
    out = dict(kwargs)
    model = out.get("model") or ""
    if legacy_request_shape(model):
        return out
    if "max_tokens" in out:
        # Both present means the caller was explicit; the new name wins and the old one
        # must still go, because leaving it in is the 400 this whole function exists for.
        out.setdefault("max_completion_tokens", out["max_tokens"])
        del out["max_tokens"]
    effort = out.get("reasoning_effort") or reasoning_effort()
    if effort:
        out["reasoning_effort"] = effort
    if effort != "none":
        # Reasoning on: these are refused. Dropping them is not a free choice, it is the
        # only one - but it means the call is no longer reproducible, which is why
        # `reasoning_effort` is recorded next to every measurement.
        for param in _SAMPLING_PARAMS:
            out.pop(param, None)
    return _apply_repairs(model, out)


# ── What the API itself says it will not accept ───────────────────────────────
#: Repairs learned from the API's own refusals: {model family: {parameter: new name, or
#: None to drop it}}. The API is the authority on what it accepts and it names the
#: offending parameter in the error, so the process learns the shape of a model it has
#: never seen instead of relying on a hand-written table that goes stale with the next
#: release. Learned once per process, then applied before the request rather than after a
#: refusal, so the cost of meeting a new model is a handful of 400s, not one per call.
_REPAIRS: dict[str, dict[str, str | None]] = {}
_REPAIRS_LOCK = threading.Lock()

_RE_UNSUPPORTED_PARAM = re.compile(r"unsupported parameter:\s*'([^']+)'", re.I)
_RE_UNSUPPORTED_VALUE = re.compile(r"unsupported value:\s*'([^']+)'", re.I)
_RE_UNRECOGNIZED = re.compile(
    r"unrecognized request argument supplied:?\s*'?([A-Za-z0-9_]+)", re.I)
#: Not the API refusing but the SDK: `create` takes named parameters and no **kwargs, so
#: an installation that predates `reasoning_effort` raises TypeError before any request.
#: requirements.txt sets a floor that rules this out, and pip has been known to leave an
#: old satisfied requirement in place, so the recovery stays: a call without the effort
#: still works, at the API's own default.
_RE_UNEXPECTED_KWARG = re.compile(r"unexpected keyword argument '([^']+)'", re.I)
_RE_USE_INSTEAD = re.compile(r"use '([^']+)' instead", re.I)

#: A refusal may name one parameter at a time, so a request with several problems needs
#: several rounds. Bounded because each round must END one: the loop cannot run forever.
_MAX_REPAIRS_PER_CALL = 4


def _repair_from_error(exc: Exception, kwargs: dict) -> tuple[str, str | None] | None:
    """(parameter, replacement name or None to drop) if this error names a parameter we
    actually sent, else None - in which case the error is the caller's to see.

    Deliberately narrow. A refusal that names something we did not send (a field inside
    `messages`), or no parameter at all (a rate limit, an oversized context, an outage),
    must propagate untouched: retrying those is how a bounded loop becomes an unbounded
    bill."""
    msg = str(exc)
    for pattern in (_RE_UNSUPPORTED_PARAM, _RE_UNSUPPORTED_VALUE, _RE_UNRECOGNIZED,
                    _RE_UNEXPECTED_KWARG):
        found = pattern.search(msg)
        if not found:
            continue
        param = found.group(1)
        if param not in kwargs:
            continue
        instead = _RE_USE_INSTEAD.search(msg)
        replacement = instead.group(1) if instead else None
        if replacement in (param, *kwargs):
            # Renaming onto a parameter already in the request would overwrite it; the
            # request is then simply wrong, so drop the refused one.
            replacement = None
        return param, replacement
    return None


def _apply_repair(kwargs: dict, param: str, replacement: str | None) -> dict:
    out = dict(kwargs)
    value = out.pop(param, None)
    if replacement:
        out[replacement] = value
    return out


def _remember_repair(model: str, param: str, replacement: str | None) -> None:
    family = _model_family(model) or "unknown"
    with _REPAIRS_LOCK:
        _REPAIRS.setdefault(family, {})[param] = replacement
    logger.warning(
        f"{family} refused {param!r}; "
        f"{'renaming to ' + replacement if replacement else 'dropping it'} "
        "from now on in this process. Check LLM_MODEL_* and llm_usage._LEGACY_REQUEST_SHAPE.")


def _apply_repairs(model: str, kwargs: dict) -> dict:
    with _REPAIRS_LOCK:
        learned = dict(_REPAIRS.get(_model_family(model), {}))
    for param, replacement in learned.items():
        if param in kwargs:
            kwargs = _apply_repair(kwargs, param, replacement)
    return kwargs


def request_policy() -> dict:
    """How chat requests are shaped before they leave, and what the API has taught this
    process about the models it is pointed at.

    Reported by /health: a deployment repairing every call is a deployment whose model
    configuration is wrong, and that is otherwise visible only in the logs."""
    effort = reasoning_effort()
    with _REPAIRS_LOCK:
        learned = {model: {p: (r or "dropped") for p, r in params.items()}
                   for model, params in _REPAIRS.items()}
    return {
        "reasoning_effort": effort or "api-default",
        "sampling_params": "sent" if effort == "none" else "stripped",
        "token_ceiling_param": {
            role: ("max_tokens" if legacy_request_shape(model) else "max_completion_tokens")
            for role, model in models_in_use().items() if role != "embedding"},
        "learned_repairs": learned,
    }


#: Set by `configure()` from main.py. Without it, recording is skipped (never fatal).
_engine = None

#: Cached total of today's recorded tokens, so the budget costs one query a minute
#: instead of one per call. Refreshed from the table so several workers converge.
_spend_lock = threading.Lock()
_spend_cache = {"day": None, "tokens": 0, "checked_at": 0.0}
_SPEND_TTL_SECONDS = 60.0

DDL = (
    """CREATE TABLE IF NOT EXISTS llm_usage (
        id                BIGSERIAL PRIMARY KEY,
        ts                TIMESTAMPTZ NOT NULL DEFAULT now(),
        purpose           TEXT        NOT NULL,
        model             TEXT        NOT NULL,
        prompt_tokens     INTEGER     NOT NULL DEFAULT 0,
        completion_tokens INTEGER     NOT NULL DEFAULT 0,
        total_tokens      INTEGER     NOT NULL DEFAULT 0
    )""",
    "CREATE INDEX IF NOT EXISTS idx_llm_usage_ts ON llm_usage (ts DESC)",
    "CREATE INDEX IF NOT EXISTS idx_llm_usage_purpose_ts ON llm_usage (purpose, ts DESC)",
)


class LLMCallBlocked(RuntimeError):
    """An OpenAI call was refused locally - by the master switch or the daily budget."""


# ── configuration ────────────────────────────────────────────────────────────

def configure(engine) -> None:
    """Give the module the SQLAlchemy engine to record into. Idempotent."""
    global _engine
    _engine = engine


def _flag(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() not in ("0", "false", "no", "off")


def openai_enabled() -> bool:
    return _flag("OPENAI_ENABLED")


def daily_token_budget() -> int:
    """Tokens allowed per UTC day, or 0 for unlimited (the default)."""
    try:
        return max(0, int(os.getenv("LLM_DAILY_TOKEN_BUDGET", "0")))
    except (TypeError, ValueError):
        return 0


# ── recording ────────────────────────────────────────────────────────────────

def _usage_fields(usage) -> tuple[int, int, int]:
    """(prompt, completion, total) from a response's `usage`, whatever shape it has.

    Embedding responses carry no `completion_tokens`; a streamed response may carry no
    usage at all. Missing means zero, never an exception - accounting must not be able
    to break the call it is accounting for.
    """
    def _int(v):
        try:
            return max(0, int(v))
        except (TypeError, ValueError):
            return 0
    if usage is None:
        return 0, 0, 0
    get = usage.get if isinstance(usage, dict) else lambda k, d=None: getattr(usage, k, d)
    prompt = _int(get("prompt_tokens", 0))
    completion = _int(get("completion_tokens", 0))
    total = _int(get("total_tokens", 0)) or (prompt + completion)
    return prompt, completion, total


def record(purpose: str, model: str, usage) -> int:
    """Persist one call's usage. Returns the total tokens recorded (0 if not recorded).

    Best effort by construction: a failure here is logged at debug and swallowed. An
    accounting table that can take the app down is worse than no accounting table.
    """
    prompt, completion, total = _usage_fields(usage)
    if not _flag("LLM_USAGE_LOGGING") or _engine is None:
        return total
    try:
        from sqlalchemy import text
        with _engine.begin() as c:
            c.execute(text(
                "INSERT INTO llm_usage (purpose, model, prompt_tokens, completion_tokens,"
                " total_tokens) VALUES (:p, :m, :pt, :ct, :tt)"),
                {"p": (purpose or "unknown")[:120], "m": (model or "unknown")[:80],
                 "pt": prompt, "ct": completion, "tt": total})
    except Exception as e:
        logger.debug(f"llm_usage record failed ({purpose}/{model}): {e}")
        return total
    with _spend_lock:
        if _spend_cache["day"] == _utc_day():
            _spend_cache["tokens"] += total
    return total


def _utc_day() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def spend_today(force: bool = False) -> int:
    """Tokens recorded since midnight UTC, cached for `_SPEND_TTL_SECONDS`."""
    day = _utc_day()
    with _spend_lock:
        fresh = (not force and _spend_cache["day"] == day
                 and (time.time() - _spend_cache["checked_at"]) < _SPEND_TTL_SECONDS)
        if fresh:
            return int(_spend_cache["tokens"])
    total = 0
    if _engine is not None:
        try:
            from sqlalchemy import text
            with _engine.connect() as c:
                total = int(c.execute(text(
                    "SELECT COALESCE(SUM(total_tokens), 0) FROM llm_usage "
                    "WHERE ts >= date_trunc('day', now() AT TIME ZONE 'UTC')")).scalar() or 0)
        except Exception as e:
            logger.debug(f"llm_usage spend_today failed: {e}")
            # Unknown spend must not act like zero spend under a budget, nor block every
            # call when the table is simply missing. Reuse the last known figure.
            with _spend_lock:
                return int(_spend_cache["tokens"] if _spend_cache["day"] == day else 0)
    with _spend_lock:
        _spend_cache.update(day=day, tokens=total, checked_at=time.time())
    return total


def check_allowed(purpose: str = "") -> None:
    """Raise `LLMCallBlocked` if the master switch is off or the day's budget is spent."""
    if not openai_enabled():
        raise LLMCallBlocked(
            f"OpenAI calls are disabled (OPENAI_ENABLED=0){f' [{purpose}]' if purpose else ''}")
    budget = daily_token_budget()
    if budget and spend_today() >= budget:
        raise LLMCallBlocked(
            f"daily LLM budget reached ({spend_today()}/{budget} tokens today)"
            f"{f' [{purpose}]' if purpose else ''}. Raise LLM_DAILY_TOKEN_BUDGET or wait "
            "for the UTC day to roll over.")


# ── the metered client ───────────────────────────────────────────────────────

def _caller_name(depth: int = 2) -> str:
    """Name of the function that asked for a client - used as the usage `purpose`."""
    try:
        import inspect
        f = inspect.currentframe()
        for _ in range(depth):
            f = f.f_back if f is not None else None
        return f.f_code.co_name if f is not None else "unknown"
    except Exception:
        return "unknown"


def _create_with_repair(create, args: tuple, kwargs: dict):
    """Make one chat call in the shape the model accepts, and if the API refuses a
    parameter anyway, do what the refusal says and try again.

    The retry is bounded and strictly shrinking: every round removes one parameter from
    the request, so at most `_MAX_REPAIRS_PER_CALL` rounds can happen and the last
    attempt always raises. A refused parameter costs nothing (the request is rejected
    before any tokens are read), which is what makes retrying it safe to do silently."""
    kwargs = shape_chat_kwargs(kwargs)
    for attempt in range(_MAX_REPAIRS_PER_CALL + 1):
        try:
            return create(*args, **kwargs)
        except Exception as exc:
            repair = (None if attempt == _MAX_REPAIRS_PER_CALL
                      else _repair_from_error(exc, kwargs))
            if repair is None:
                raise
            param, replacement = repair
            _remember_repair(kwargs.get("model") or "", param, replacement)
            kwargs = _apply_repair(kwargs, param, replacement)
    raise AssertionError("unreachable: the last attempt re-raises")  # pragma: no cover


def _wrap(create, purpose: str, kind: str):
    def metered(*args, **kwargs):
        check_allowed(purpose)
        resp = (_create_with_repair(create, args, kwargs) if kind == "chat"
                else create(*args, **kwargs))
        try:
            model = kwargs.get("model") or getattr(resp, "model", "") or "unknown"
            # A stream is an iterator, not a response object: it has no usage to read
            # (see the module docstring). Record the call at zero tokens so the blind
            # spot is visible in the table instead of being invisible.
            usage = None if kwargs.get("stream") else getattr(resp, "usage", None)
            record(f"{purpose}:{kind}", model, usage)
        except Exception as e:
            logger.debug(f"llm_usage metering failed ({purpose}/{kind}): {e}")
        return resp
    return metered


def instrument(client, purpose: str):
    """Wrap a live OpenAI client's two spending surfaces in place, then return it."""
    for attr, kind in (("chat", "chat"), ("embeddings", "embeddings")):
        try:
            target = client.chat.completions if attr == "chat" else client.embeddings
            target.create = _wrap(target.create, purpose, kind)
        except Exception as e:                       # SDK shape changed - never fatal
            logger.warning(f"llm_usage could not instrument {attr}: {e}")
    return client


def MeteredOpenAI(*args, purpose: str | None = None, **kwargs):
    """Drop-in for `openai.OpenAI` that records what every call spends.

    Named like a class because it replaces one at the import site; it is a factory, so
    the real client (and any future SDK surface) passes through unchanged.
    """
    from openai import OpenAI
    return instrument(OpenAI(*args, **kwargs), purpose or _caller_name())


def MeteredAsyncOpenAI(*args, purpose: str | None = None, **kwargs):
    """Async counterpart. Both current async call sites stream, so see the docstring:
    the switch and the budget apply, the token counts do not."""
    from openai import AsyncOpenAI
    client = AsyncOpenAI(*args, **kwargs)
    name = purpose or _caller_name()

    async def _acreate_with_repair(create, a: tuple, kw: dict):
        """Async twin of `_create_with_repair`; same bound, same reasoning."""
        kw = shape_chat_kwargs(kw)
        for attempt in range(_MAX_REPAIRS_PER_CALL + 1):
            try:
                return await create(*a, **kw)
            except Exception as exc:
                repair = (None if attempt == _MAX_REPAIRS_PER_CALL
                          else _repair_from_error(exc, kw))
                if repair is None:
                    raise
                param, replacement = repair
                _remember_repair(kw.get("model") or "", param, replacement)
                kw = _apply_repair(kw, param, replacement)
        raise AssertionError("unreachable: the last attempt re-raises")  # pragma: no cover

    def _wrap_async(create, kind: str):
        async def metered(*a, **kw):
            check_allowed(name)
            resp = (await _acreate_with_repair(create, a, kw) if kind == "chat"
                    else await create(*a, **kw))
            try:
                model = kw.get("model") or getattr(resp, "model", "") or "unknown"
                record(f"{name}:{kind}", model,
                       None if kw.get("stream") else getattr(resp, "usage", None))
            except Exception as e:
                logger.debug(f"llm_usage async metering failed ({name}/{kind}): {e}")
            return resp
        return metered

    for attr, kind in (("chat", "chat"), ("embeddings", "embeddings")):
        try:
            target = client.chat.completions if attr == "chat" else client.embeddings
            target.create = _wrap_async(target.create, kind)
        except Exception as e:
            logger.warning(f"llm_usage could not instrument async {attr}: {e}")
    return client


# ── reporting ────────────────────────────────────────────────────────────────

def summary(hours: int = 24) -> dict:
    """Usage over the last `hours`, broken down by purpose and model.

    This is the query the app could not answer before: which loop is spending, and how
    much. Sorted by tokens so the top row is the thing to look at first.
    """
    out: dict = {"hours": hours, "enabled": openai_enabled(),
                 "daily_token_budget": daily_token_budget(),
                 "tokens_today": 0, "total_tokens": 0, "total_calls": 0, "by_purpose": []}
    if _engine is None:
        out["error"] = "no database configured"
        return out
    try:
        from sqlalchemy import text
        with _engine.connect() as c:
            rows = c.execute(text("""
                SELECT purpose, model, COUNT(*) AS calls,
                       SUM(prompt_tokens) AS prompt_tokens,
                       SUM(completion_tokens) AS completion_tokens,
                       SUM(total_tokens) AS total_tokens,
                       MAX(ts) AS last_call
                FROM llm_usage WHERE ts >= now() - make_interval(hours => :h)
                GROUP BY purpose, model ORDER BY SUM(total_tokens) DESC NULLS LAST
            """), {"h": max(1, int(hours))}).mappings().all()
        out["by_purpose"] = [{
            "purpose": r["purpose"], "model": r["model"], "calls": int(r["calls"] or 0),
            "prompt_tokens": int(r["prompt_tokens"] or 0),
            "completion_tokens": int(r["completion_tokens"] or 0),
            "total_tokens": int(r["total_tokens"] or 0),
            "last_call": r["last_call"].isoformat() if r["last_call"] else None,
        } for r in rows]
        out["total_tokens"] = sum(r["total_tokens"] for r in out["by_purpose"])
        out["total_calls"] = sum(r["calls"] for r in out["by_purpose"])
        out["tokens_today"] = spend_today(force=True)
    except Exception as e:
        out["error"] = str(e)
    return out
