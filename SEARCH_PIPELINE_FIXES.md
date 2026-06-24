# Search page + pipeline — audit findings & fixes

Date: 2026-06-24. Context: after the OpenAI quota outage, three symptoms were reported —
boolean-query translation echoing the raw text, the live search "taking ages" (~77s),
and every result card showing "Base locale" despite "+N nouvelles références".

## Key architectural finding (reframes everything)

**The search page does NOT use `POST /search` / `_federated_live_search`.** Its flow is:
`POST /search-strategy` → `createUserScenario` → `POST /user-scenarios/{id}/populate` →
poll `GET /user-scenarios/{id}/corpus`. So `_federated_live_search` (and the perf work in
PR #91 that targeted it) is on a **parallel path the page doesn't exercise** — which is why
the earlier optimizations didn't change what the user saw. The real path is
`_run_user_scenario_populate` + `get_user_scenario_corpus`.

All three symptoms were verified (via `git`) to **predate** PRs #88/#90/#91 — not regressions
introduced by those changes. But the fixes now target the **correct** paths.

## Issue 1 — Boolean query echoes the raw text

Root cause: `_generate_search_strategy` (`main.py`) translates NL→boolean via GPT-4.1-mini, and
on **any** OpenAI error (incl. `429 insufficient_quota`) its `except` returns
`{"general": query, ...}` — the raw text. Worse, that degraded result was **cached** in
`user_scenarios.search_strategy` and never refreshed (no error sentinel), so a scenario created
during the outage stays poisoned even after quota returns.

Fix:
- Degraded fallbacks now carry `"degraded": True`.
- New `_strategy_is_degraded()` (degraded flag / empty / no boolean operators / equals raw query).
- The three caching callers (`get_search_strategy`, `create_user_scenario._bg_strategy`, the populate
  path) now **regenerate** a degraded/poisoned strategy and **never persist** a degraded one.

## Issue 2 — Live search ~77s, "stuck on bioRxiv"

Root cause: the populate federation (`_run_user_scenario_populate`) waited on
`as_completed(futures)` with **no timeout**, so the slowest of 7 sources (PubMed efetch
`timeout=90`, Cochrane 3×-retry-with-backoff) gated the whole run. During the outage, the inline
rerank/embed step also retry-stormed OpenAI (3× per failed call). The "· bioRxiv" label is a
**frontend cosmetic timer** (`liveSources[searchElapsed % …]`), not real telemetry.

Fix:
- Added a wall-clock budget `POPULATE_FEDERATION_BUDGET` (env-tunable, default 55s):
  `as_completed(futures, timeout=…)` + `except TimeoutError` → proceed with the partial corpus;
  slow sources keep ingesting in the background.
- Quota cooldown (Issue 4) removes the outage-time retry-storm in the inline rerank/embed.

## Issue 3 — "+N nouvelles références" but every card says "Base locale"

Root cause: `renderCorpus` built each card **without** setting any provenance flag, so the badge's
`result.isLive` was always falsy → **always "Base locale"**. Meanwhile "+N nouvelles références"
came from `corpus.newly_fetched` (a `created_at >= scenario.created_at` timestamp count). Different
fields → a paper could be counted "new" yet badged "Base locale". The badge was indeed
uninformative (and, as the user noted, every paper becomes "local" once ingested).

Fix:
- `get_user_scenario_corpus` now returns a per-article **`is_new`** (`created_at >= scenario
  created_at`).
- Frontend maps it to `SearchResult.isNew`; the badge is now three-way: **API live** / **Nouveau**
  (fetched during this search) / **Base locale** (pre-existing). So a "+N nouvelles références"
  paper now correctly shows **Nouveau**.

## Issue 4 — OpenAI quota flood mitigation (the actual outage cause)

The account hit `429 insufficient_quota`; every chat/embedding call failed and the SDK retried
3×, and the background worker ground through every batch → a sustained flood (1656 429s / 3000 log
lines). The cure is billing (done by the user). Code mitigation so a future outage is non-fatal:
- `_is_openai_quota_error()` + a global cooldown (`_openai_in_cooldown` / `_trip_openai_cooldown`).
- Background loops (worker embed, worker PICO, inline rerank) **break on the first quota error** and
  **skip OpenAI work while in cooldown** (5 min) instead of re-flooding every cycle.

Also done on the server (not code): de-duplicated the duplicated midnight `living_review_scheduler`
cron line (was running twice concurrently).

## Recommended (not in this PR)
- Lower the page's `maxResults: 2000` / `LIVE_MAX_PER_SOURCE` if faster-but-smaller corpora are OK.
- Set `NCBI_API_KEY` (10 req/s vs 3) — already used by the code.
- Long term: converge the page onto a single federation path so `/search` and `/populate` share
  timeouts, provenance, and instrumentation.
