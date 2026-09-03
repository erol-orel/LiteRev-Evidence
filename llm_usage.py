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

Three controls, all off by default so that installing this changes no behaviour:

  OPENAI_ENABLED=0             refuse every call, immediately, without a request.
  LLM_DAILY_TOKEN_BUDGET=N     refuse once N tokens have been recorded since midnight
                               UTC (0 = no budget, the default).
  LLM_USAGE_LOGGING=0          stop recording (the accounting itself, not the calls).

A refusal raises `LLMCallBlocked`. Call sites already wrap their LLM calls in
`try/except Exception`, so a blocked call degrades to "this feature is unavailable"
rather than a 500 — the same shape as an API outage, which is what a spend freeze is.

NOT metered: the two streaming chat calls (`stream=True`). The API only reports usage
for a stream when asked via `stream_options`, which appends a final chunk with empty
`choices` — enough to break a consumer that assumes every chunk has one. Adding that
needs a test against the live API, which is exactly what this environment cannot do.
They are counted as calls with zero tokens, so they show up in the table as a known
blind spot rather than silently missing. The master switch and budget DO apply to them.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timezone

logger = logging.getLogger("llm-usage")

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
    """An OpenAI call was refused locally — by the master switch or the daily budget."""


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
    usage at all. Missing means zero, never an exception — accounting must not be able
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
    """Name of the function that asked for a client — used as the usage `purpose`."""
    try:
        import inspect
        f = inspect.currentframe()
        for _ in range(depth):
            f = f.f_back if f is not None else None
        return f.f_code.co_name if f is not None else "unknown"
    except Exception:
        return "unknown"


def _wrap(create, purpose: str, kind: str):
    def metered(*args, **kwargs):
        check_allowed(purpose)
        resp = create(*args, **kwargs)
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
        except Exception as e:                       # SDK shape changed — never fatal
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

    def _wrap_async(create, kind: str):
        async def metered(*a, **kw):
            check_allowed(name)
            resp = await create(*a, **kw)
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
