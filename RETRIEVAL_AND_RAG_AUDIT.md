# Retrieval & RAG Audit — LiteRev-Evidence

Date: 2026-06-22 · Scope: (1) external "live API sources" retrieval (why it's
*not working / very slow*) and (2) the RAG answer pipeline (what information feeds
the answers). Method: full static read of `main.py` (+ `ingest_pipeline.py`,
`ingest_pubmed.py`, `living_review_scheduler.py`). The sandbox blocks outbound
HTTP (uniform `403` to every host), so external calls were audited from code, not
reproduced live.

This document records the audit **and** the fixes shipped in this PR.

---

## Part A — Live external-source retrieval

### A.0 How it works
`POST /user-scenarios/{id}/search/live` (`main.py:2310`) → `_federated_live_search`
(`main.py:2152`) fans out to **8 sources** (PubMed, OpenAlex, Crossref, EuropePMC,
medRxiv, bioRxiv, PROSPERO, Cochrane) in a thread pool, dedups, marks
`in_local_db`, **scores every result**, and returns sorted by hybrid score. A
separate background job (`_launch_populate_job`) then ingests new papers. Note the
deploy smoke test only hits `/search` (local pgvector), so it **never** exercises
this external path — "smoke test green" said nothing about live sources.

### A.1 Why it can be SLOW — ranked (worst-case for one request)
| # | Cause | Location | Worst case |
|---|---|---|---|
| 1 | **OpenAI scoring of *every* deduped result, synchronously, on the request path** | `_federated_live_search` (scoring block) | up to ~400 embeds, client timeout 20 s × up to 3 calls ≈ **+40 s** |
| 2 | **NCBI global lock** serializes 6 eutils calls (PubMed+PROSPERO+Cochrane each do esearch+esummary) at 0.4 s spacing | `_ncbi_get`, `_NCBI_LOCK`, `_NCBI_MIN_INTERVAL` | **~2.4 s** spacing + RTT, more on 429 retries |
| 3 | **Preprint over-scan**: medRxiv & bioRxiv each scan up to 300 records (3 pages × 10 s) with a near-zero client-side match rate | `_live_fetch_preprint_server` | up to **30 s each** for ~0 results |
| 4 | **30 s blanket federation cap** — slowest source gates the whole response; anything not done is silently dropped | `_federated_live_search` (`as_completed timeout=30`) | up to **30 s** |

### A.2 Why it can show NOTHING ("not working")
- **All source errors are swallowed** → a source that's 100% down looks like "queried, 0 results" (`_live_fetch_* except: logger.warning; return []`).
- **Slow sources silently dropped at 30 s** — and the serialized NCBI trio + over-scanning preprints are the most likely to miss the cut, so the *best* sources go missing first.
- **`require_api_key` on the endpoint** → a missing/stale `X-API-Key` (e.g. after a `WRITE_API_KEY` rotation) returns **401** before any source is queried. Very common "stopped working" cause.
- **Preprint keyword filter too strict** (`has_primary` AND ≥3 hits within 180 days) → medRxiv/bioRxiv almost always empty.
- **PROSPERO/Cochrane** wrap a natural-language `general_query` in extra boolean `[Publication Type]`/`[Journal]` filters → frequently match nothing.
- **OpenAI scoring failure** zeroes all `semantic_score` → ranking collapses to lexical (looks like "junk results").
- **Config**: no `NCBI_API_KEY` (3 req/s cap); no `OPENAI_API_KEY` (no semantic rank + degraded strategy generation).

### A.3 Fixes shipped in this PR
- **New `GET /sources/health`** (`main.py`, near the live-search endpoint): probes each upstream in parallel with a minimal query and returns per-source `{ok, http, latency_ms, count, error}` + a `config` block (`ncbi_api_key`/`openai_api_key` presence). **This is the fastest way to see, from production, exactly which source is down or slow.** Read-only, no auth.
- **OpenAI scoring bounded** (`_federated_live_search`): retrieve-then-rerank — embed only the **top 60 by lexical overlap** (not all ~400), client **timeout 20 s → 8 s**, and a **per-batch guard** so one slow/failed call no longer hangs or zeroes the whole ranking. Cuts the dominant latency term from ~40 s to a few seconds.
- **Preprint scan tightened**: `max_scan 300 → 120`, page `timeout 10 s → 6 s`, + a `User-Agent`. Reclaims most of the wasted federation budget.
- **EuropePMC** now sends a `User-Agent` (was missing → throttling risk).
- **NCBI spacing** drops to `0.11 s` when `NCBI_API_KEY` is set (10 req/s tier), reducing the lock serialization tax across the PubMed-backed trio.

### A.4 Recommended next (config / larger, not in this PR)
1. **Set `NCBI_API_KEY`** on the server (env) — biggest free win for the PubMed trio; the code already uses it and now spaces at 10 req/s.
2. **Confirm egress + `WRITE_API_KEY`** in prod: hit `/sources/health` (egress) and verify the frontend's stored `X-API-Key` matches the server (401).
3. **Surface per-source status** in the `/search/live` response (ok/timeout/error) instead of silently dropping — removes the core "is it down or just empty?" ambiguity.
4. **Fix `in_local_db` matching** (`external_id` is `pmid:`/`prospero:`/`cochrane:`/OpenAlex-URL, not a bare DOI) so new/duplicate accounting is correct.
5. **PROSPERO/Cochrane**: pass the MeSH `pubmed_query` and fall back to a bare query when the filtered one returns 0.

---

## Part B — RAG answer pipeline (what feeds the answers)

All RAG endpoints embed the question with `text-embedding-3-small`, retrieve nearest
`document_chunk`s (cosine `<=>`), and answer with `gpt-4o-mini` (non-stream) or
`gpt-4.1-mini` (stream). **Answers come strictly from the retrieved chunks of your
ingested corpus** — grounding is enforced only by the prompt ("STRICTEMENT /
exclusivement sur le contexte"); there is no programmatic citation check.

### B.1 Endpoints (after this PR's fixes)
| Endpoint | Scenario filter | Screening/dup gate | Min similarity | top_k | Model | Empty retrieval |
|---|---|---|---|---|---|---|
| `POST /ask` | `_build_where` | **now: excl. dup + 'excluded'** | **now: ≥ RAG_MIN_SIMILARITY** | 6 | gpt-4o-mini | "no articles" |
| `POST /ask/stream` | `d.scenario_type` | **now: excl. dup + 'excluded'** | **now: floor** | 8 | gpt-4.1-mini | **now: short-circuits (no ungrounded LLM call)** |
| `POST /ask/stream/filtered` | `article_scenarios` join | included-prioritized | scenario threshold (0.45) | 12 | gpt-4.1-mini | **now: short-circuits** |
| `gesica/scenarios/{id}/rag` | join | dup excl. + not 'excluded' | — | 8 | gpt-4o-mini | "no articles" |
| `user-scenarios/{id}/rag` | join | dup excl. + not 'excluded' | — | 8 | gpt-4o-mini | "no articles" |

`RAG_MIN_SIMILARITY` is a new, env-tunable floor (default **0.18**) — intentionally
low because `text-embedding-3-small` produces modest question→passage cosine
similarities; it removes only clear noise, not good matches.

### B.2 Fixes shipped in this PR
- **Screening + duplicate gate added to `/ask` and `/ask/stream`** — they no longer surface chunks from `excluded` or duplicate articles (now consistent with the `/rag` endpoints).
- **Similarity floor added to `/ask` and `/ask/stream`** — no longer answers from arbitrarily-distant chunks on a thin corpus.
- **Titles + years now placed *in* the LLM context for both streaming endpoints** — the prompt tells the model to "cite by title", but previously only raw chunk text was provided (titles went only to the client). The model now actually sees them.
- **Streaming endpoints short-circuit on empty retrieval** — instead of calling the LLM with "Aucun contexte disponible" (which invites ungrounded answers), they emit a clear "no relevant passages" message and stop.
- **Removed dead `enriched_question`** in the GESICA RAG handler (was built, never used).

### B.3 Known remaining inconsistencies (documented, not changed)
- `/ask/stream` filters scenarios via the `d.scenario_type` column while the others use the `article_scenarios` join — can yield different corpora for the "same" scenario.
- No programmatic citation/faithfulness enforcement anywhere (prompt-only grounding).
- No token budget; only `top_k` bounds context size (long chunks passed in full).

---

## Verify
```bash
python -m py_compile main.py
ruff check main.py --select F821,F811   # clean
# in prod, to pinpoint live-source failures:
curl -s http://localhost:8000/sources/health | jq
```
