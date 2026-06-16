# LiteRev-Evidence — Production-Grade Audit

Date: 2026-06-16 · Auditor: automated deep audit (repo + live server + live DB)
Commit audited: `745be1a` (server == origin/main, clean) · DB: PostgreSQL 14.23, pgvector 0.8.2

Evidence sources: full repo read (5 parallel deep-dives), live runtime snapshot of
`literev-app-01` (systemd/nginx/cron/processes), and a read-only schema+integrity
sweep of the production DB (`literev-db-01`) via the audited `server-command` workflow.

---

## 1. Executive summary

The application **is up and serving** (`/health` 200, embeddings 99.99% populated, FKs
enforced with zero orphans, SQL fully parameterized, timing-safe auth where applied).
The engineering of the request/SQL layer is, in isolation, fairly solid.

However the audit found **five Critical issues**, several of which are live in production
right now (not just latent):

1. **Secret exposure** — the OpenAI API key is stored in **plaintext** in the systemd
   override and was printed into a **public** GitHub Actions log during this audit
   (log since deleted). The key must be rotated.
2. **No ANN/vector index** on `document_chunk.embedding` (323,868 rows × 1536-dim, **5 GB**
   table). Every semantic/hybrid search is a full sequential scan — the dominant latency
   risk and it worsens linearly with corpus growth.
3. **Large-scale data duplication**: **8,957** duplicate-DOI groups, **1,033** duplicate-URL,
   **696** duplicate-`external_id`, **647** duplicate-PMID in `literature_document` (81,209
   rows). There is **no unique constraint** on any natural key, the dedup job is largely
   inert (`canonical_id` is 99.6% NULL), and scenario article counts do **not** filter
   duplicates → inflated counts.
4. **No TLS** — nginx listens on `:80` only. The `WRITE_API_KEY` travels as a cleartext
   `X-API-Key` header; all traffic is interceptable.
5. **The schema lives only in the live DB.** Alembic has a single **no-op** revision;
   every real object (`article_scenarios`, `screening_status`, all bibliographic/PICO/dedup
   columns) was hand-applied. A fresh deploy / disaster-recovery rebuild produces a
   **broken** schema. `alembic upgrade head` is theater.

Important nuance from comparing code vs live state: static analysis flagged
`article_scenarios` and `screening_status` as "missing → app broken." **The live DB proves
they exist** with correct PK/FK/unique constraints. So the *running* app is fine; the real
exposure is **disaster recovery / environment reproducibility**, not a current outage. This
distinction only emerged by querying production.

A second nuance: many concurrency findings are **mitigated today by a single uvicorn worker**
(confirmed: one process, no `--workers`). They are latent — but nothing *enforces* the single
worker, and one cross-process race (the **duplicated midnight cron line**) is live regardless.

---

## 2. System architecture & data flow map

```
        Browser (React 19 SPA, state-nav, no router)
              │  HTTPS? NO — plain HTTP :80
              ▼
        nginx (literev-app-01:80)
          ├─ /            → static /var/www/literev-frontend (SPA)
          ├─ /api/*       → 127.0.0.1:8000/*   (strips /api ✓)
          └─ =/search =/documents =/chunks =/health =/embed-info → :8000 (direct)
              ▼
        uvicorn main:app  (SINGLE worker, port 8000, systemd literev-api)
          ├─ in-process daemon threads:
          │     • background enrichment worker (embed + PICO every 30s)
          │     • demand-model training thread
          │     • per-request clustering/UMAP threads
          ├─ in-memory state: rate limiter, _clustering_jobs, _pipeline_jobs (per-process)
          └─ SQLAlchemy engine (pool_pre_ping/recycle) ─┐
                                                         ▼
        PostgreSQL 14.23 + pgvector 0.8.2 (literev-db-01, private 10.10.1.10)
          literature_document(81,209) ─1:N→ document_chunk(323,868, 5GB, embedding vector(1536))
                  ▲                    └─1:N→ article_scenarios(25,301)  [join → scenarios]
                  └ self-FK canonical_id (dedup)        user_scenarios(38) ─→ user_scenario_folders(7)
                                                        scenario_settings(10), alert_subscriptions(5)

   Three UNCOORDINATED write paths to the same tables:
     (A) HTTP ingest scripts → POST /documents,/chunks   (dedup via fuzzy /search)
     (B) direct-DB jobs: living_review_scheduler, embed_corpus, extract_pico, deduplicate, fetch_fulltext
     (C) in-process orchestrator (_run_user_scenario_full_pipeline) + BG worker
   Cron (daily 00:00, TWO identical lines): scheduler --all-scenarios --days 7 && embed_corpus
```

Pipeline order (orchestrator, correct): `ingest → fulltext → embed → rerank → pico → metadata → clustering`.
Cron path runs only `scheduler + embed_corpus` — **fulltext and PICO are never run by cron**;
they rely on the in-process BG worker.

---

## 3. Findings by severity

Legend for each: **What / Why / Evidence / Location / Fix / Verify.**

### CRITICAL

**C1 — OpenAI API key in plaintext + leaked to public CI log**
- What: `/etc/systemd/system/literev-api.service.d/override.conf` sets
  `Environment="OPENAI_API_KEY=sk-proj-…"` in clear; the key is also in
  `/opt/literev-api/secrets.env`. During this audit a `systemctl cat` printed it into a
  GitHub Actions log on a **public** repo.
- Why: Anyone who saw the log (or the repo's Actions history) has a working, billable key.
- Evidence: runtime snapshot, override.conf `Environment=` line; repo is `private:false`.
- Location: server `override.conf`; `/opt/literev-api/secrets.env`.
- Fix: **Rotate the OpenAI key now.** Keep it only in the 0600 `secrets.env` (or a secret
  manager), remove the `Environment=` literal from override.conf. (Audit already deleted the
  exposed run log.)
- Verify: new key works via `/health`-adjacent embed call; `grep -r OPENAI override.conf` empty.

**C2 — No vector ANN index on `document_chunk.embedding`**
- What: 323,868 rows, 1536-dim, 5,088 MB; the only indexes are btree/gin (search_vector,
  metadata_json, chunk_type, chunk_weight). No ivfflat/hnsw (it's commented out in `schema.sql:98-99`).
- Why: Every `ORDER BY embedding <=> :q` (12+ sites: `main.py:730,4556,6160,9940`, threshold
  filters `main.py:1044,1176,1242…`) is a full scan + sort over 5 GB → high latency, gets worse with growth.
- Evidence: live `== INDEXES ==` (no vector index) + `== VECTOR COLUMNS == dim=1536 nonnull=323866/323868`.
- Location: `document_chunk.embedding`; `schema.sql:98-99`.
- Fix: `CREATE INDEX CONCURRENTLY document_chunk_embedding_hnsw ON document_chunk USING hnsw (embedding vector_cosine_ops);`
  (pgvector 0.8.2 supports HNSW; `vector_cosine_ops` matches `<=>`). Then `ANALYZE`.
- Verify: `EXPLAIN (ANALYZE) SELECT … ORDER BY embedding <=> :q LIMIT 10;` shows Index Scan, not Seq Scan.

**C3 — Large-scale duplicate documents; no DB-level dedup anchor**
- What: dup groups — `doi=8957`, `url=1033`, `external_id=696`, `pmid=647`. No UNIQUE on any
  natural key. Dedup is app-only and inert: `canonical_id` 99.6% NULL.
- Why: `/documents` (`main.py:613`) inserts with no `ON CONFLICT`/pre-check; ingest dedup is a
  fuzzy `/search` that **fails open** (returns False on error → re-insert,
  `ingest_pipeline.py:108`, `living_review_scheduler.py:184`). Scenario counts use
  `COUNT(DISTINCT document_id)` **without** `is_duplicate` filter (`main.py:2713-2717`) → inflated.
- Evidence: live `== DUP CHECK ==`; `canonical_id null_frac=0.996`.
- Location: `literature_document`; `main.py:613,2713-2717`; ingest dedup helpers.
- Fix: dedupe existing rows; add partial UNIQUE (`WHERE doi IS NOT NULL`) on doi and pmid (and
  `(source, external_id)`); `ON CONFLICT DO NOTHING` in `/documents`; make dedup checks
  **fail-closed**; add `AND NOT d.is_duplicate` to count/article queries.
- Verify: the dup-group SQL returns 0 after cleanup; re-POST a doc twice → row count unchanged.

**C4 — No TLS (plain HTTP)**
- What: nginx `server { listen 80; … }` only; no 443/cert/redirect.
- Why: `WRITE_API_KEY` is sent as an `X-API-Key` header in cleartext; all
  request/response data interceptable on the wire.
- Evidence: runtime nginx site config (`listen 80;` only).
- Location: `/etc/nginx/sites-enabled/literev-frontend`.
- Fix: add TLS (Let's Encrypt/Caddy), 80→443 redirect, HSTS.
- Verify: `curl -I http://host` 301→https; `https://host/health` 200 with valid cert.

**C5 — Schema exists only in the live DB; Alembic is a no-op (DR/reproducibility)**
- What: single revision `eb6b9e396ffc` has `upgrade()/downgrade()` = `pass`; `alembic_version`
  is stamped to it. Every real object was hand-applied via `_ensure_*()` boot DDL +
  `scripts/archive/*.sql` run by hand.
- Why: `alembic upgrade head` (deploy.sh:64, CI `migrate`) applies nothing → a fresh DB /
  DR rebuild lacks `article_scenarios`, `screening_status`, bibliographic/PICO/dedup columns
  and the app 500s broadly. False confidence.
- Evidence: live `alembic_version=eb6b9e396ffc`; revision file body; live tables present but
  absent from any repo migration.
- Location: `alembic/versions/eb6b9e396ffc_*.py`; `alembic/env.py:27` (`target_metadata=None`).
- Fix: author a real baseline migration reproducing the **current live schema** (dump → initial
  revision), set `target_metadata`, fold `_ensure_*`/archive SQL into versioned migrations.
- Verify: `alembic upgrade head` on an empty DB then boot app + smoke test passes with no `_ensure_*` warnings.

### HIGH

**H1 — Unauthenticated mutating endpoints (IDOR + cost-DoS)**
- `/alerts/subscribe` (POST, `main.py:6514`) and `/alerts/unsubscribe` (DELETE, `:6550`) have
  no `require_api_key`; unsubscribe deletes by attacker-supplied `email`+`scenario_id` → **IDOR**.
- `/ask/stream` (`:6110`), `/ask/stream/filtered` (`:11081`), `/gesica/scenarios/{id}/rag`
  (`:4511`), `/user-scenarios/{id}/rag` (`:9907`) are unauthenticated paths to **paid OpenAI**
  calls → cost-amplification DoS, guarded only by an in-memory rate limiter.
- Fix: add `Depends(require_api_key)` to `/alerts/*`; scope unsubscribe to caller; gate RAG/stream
  behind auth or a read-key. Verify: anonymous DELETE → 401; unsubscribe can't touch others' rows.

**H2 — Frontend has no way to provide the API key → all write features 401 in prod**
- `authHeaders()` reads `sessionStorage/localStorage["api_key"]` but **nothing writes them**
  (no settings UI/prompt); `VITE_API_KEY` unset in prod. So create/delete/screen/rerank/
  pipeline/brief/alerts all send no key → 401. `App.tsx:3276` even auto-creates a scenario on
  **every search** as fire-and-forget, swallowing the 401 in console.
- Evidence: `frontend/src/lib/api.ts:15-22`; zero setters (grep); `main.py:6844` write routes require key.
- Fix: add an API-key entry UI (→ sessionStorage), or move write-auth to a server session;
  stop the per-search auto-save or make it explicit/idempotent.
- Verify: with no key, a create action surfaces an auth error (not silent); with key, it persists.

**H3 — Cross-process pipeline races → duplicate paid OpenAI work & rows**
- The crontab has **two identical** `0 0 * * *` lines running `scheduler … && embed_corpus`
  → two concurrent midnight runs. The in-process BG worker also embeds/PICOs the same
  `embedding IS NULL` rows with **no lock** (`main.py:441-457,484-493`) concurrently with
  cron `embed_corpus.py` and the orchestrator (`main.py:8619`).
- Why: duplicate (billable) embedding/PICO calls; duplicate inserts where `ON CONFLICT` is absent.
  Single uvicorn worker prevents *intra*-process dup but not these *cross*-process ones.
- Evidence: runtime cron dump (two lines); no `FOR UPDATE`/advisory lock in claim queries.
- Fix: remove the duplicate cron line; claim work with `SELECT … FOR UPDATE SKIP LOCKED` or
  `pg_try_advisory_lock`; `flock` the scheduler; atomic DB pipeline-claim
  (`UPDATE … WHERE pipeline_status<>'running' RETURNING id`).
- Verify: two concurrent embed loops on a fixture → OpenAI calls == #chunks, not 2×.

**H4 — No OpenAI retry/backoff; transient failures silently drop work and advance the pipeline**
- Every embed/PICO call is a single attempt (`embed_corpus.py:74`, `main.py:464,8639,8372`,
  `extract_pico_batch.py:131`). On 429/timeout the batch is dropped (`main.py:8654`
  `_emb_errors += len(_batch)`), then the pipeline **proceeds to rerank/pico on partially
  embedded data**.
- Fix: wrap calls in exponential backoff honoring `Retry-After`; don't advance a step while
  `errors > threshold`. Verify: inject a 429 → batch retried, step not marked done.

**H5 — Pervasive silent failure (swallowed excepts + HTTP 200 on error)**
- 13+ `except Exception: pass` (`main.py:2742,3988,4458,7146,8035,8285,8312,8406…`), several
  wrapping DB writes/cleanup before `populate_status='done'` is committed (`main.py:8021`).
  Many endpoints return `{"status":"error"}`/`{"error":…}` with **200** (`main.py:3772,10574,
  10708`; `/alerts/subscriptions` returns `[]` on any DB error `:6574`).
- Why: failures invisible; 5xx monitoring blind; dedup checks that fail-open cause C3.
- Fix: `logger.exception` in handlers; raise `HTTPException` with real status codes; fail-closed on dedup.

**H6 — Deploy: failed migration is non-fatal; no backend rollback**
- `alembic upgrade head || (warn; continue)` (`deploy.sh:62-68`) — a failed migration doesn't
  abort. On health-check failure the script restores **only the frontend** from `.prev`
  (`deploy.sh:120-129`); the new (possibly broken) backend code stays → 502 (the very thing
  `fix_502.sh`/`diagnose_502.sh` firefight).
- Fix: capture `PREV_COMMIT` pre-pull; on health failure `git reset --hard $PREV_COMMIT && restart`;
  make migration failure fatal once Alembic is real (C5).

**H7 — Single-worker constraint is required but unenforced**
- The app starts in-process daemon threads + holds in-memory rate-limit/job state
  (`main.py:90-91,510,4485,7245`). It works **only** because the unit runs one uvicorn worker
  (confirmed). Adding `--workers N` would N× the rate limit, run N enrichment/training threads,
  and split job state → duplicate work and "job not found" polls.
- Fix: commit the systemd unit with a pinned single worker **and** a comment; or externalize
  background work (separate timer/worker) + shared state (Redis/DB) before scaling.

### MEDIUM

- **M1 — N+1 + unbounded fetch on `GET /gesica/scenarios`** (`main.py:2746-2768`): one full
  per-scenario article query (no LIMIT, abstracts included) for ~27 scenarios per page load.
  Fix: counts-only list; lazy/paginated detail.
- **M2 — `article_count` cache incoherent**: written non-atomically across separate txns
  (`main.py:7434` interim wrong value, `8015`, `8058`) and two endpoints disagree —
  `/gesica/scenarios` computes live, `/user-scenarios` reads the stored column
  (`main.py:6774`). Fix: always derive live, or maintain via trigger on `article_scenarios`.
- **M3 — Clustering/kappa caches never invalidated on corpus change** (`main.py:4214-4227`,
  `/tmp/literev_clustering_cache`); `force_refresh` is the only buster. `scenario_kappa_cache`
  is dead (`if False`, `main.py:2733`).
- **M4 — Mixed `timestamp` vs `timestamptz`**: newer tables (`literature_document`,
  `document_chunk`, `article_scenarios`) use `timestamptz`; `user_scenarios`,
  `scenario_settings`, `user_scenario_folders`, `alert_subscriptions` use naive `timestamp`.
  Off-by-tz risk for the scheduler/alerts. Fix: standardize on `timestamptz`.
- **M5 — Env/secret path fragmentation**: systemd `EnvironmentFile=/etc/literev-api.env`,
  but `secrets.env` lives at `/opt/literev-api/secrets.env` (loaded separately by `main.py`);
  `OPENAI_API_KEY` is duplicated in override.conf **and** secrets.env; `deploy.sh:82` reads
  `WRITE_API_KEY` from `/etc/literev-api.env` for `VITE_API_KEY` injection. Drift here silently
  yields write-401s. Fix: one canonical env file across unit/main.py/alembic/deploy.
- **M6 — Raw `dict` bodies bypass Pydantic / unchecked casts**: `ask_stream(payload: dict)`
  (`main.py:6111`), settings/variables endpoints; `int(payload.get("top_k"))` (`:6122`) → 500
  on bad input (no exception handler). Fix: Pydantic models with bounds.
- **M7 — State-mutating work on a GET + check-then-set race**: `GET /gesica/scenarios/{id}/clustering`
  (`main.py:4467`) spawns a DB-mutating thread and writes `_clustering_jobs` unlocked; two GETs
  start two threads. Fix: POST + reuse `_pipeline_jobs_lock` pattern.
- **M8 — CI has weak gates**: `deploy.yml` runs only `compileall` + `pip --dry-run` + `npm build`
  (tsc). No pytest (none exist), no `npm run lint` (script exists, never called), no alembic
  check. Deploy runs on **every** push to main; the only gate is the `production` environment's
  (unverified) reviewers. Fix: add lint + tests + migration check; require reviewers/tags.
- **M9 — Frontend hardening**: TS not `strict` (`tsconfig.app.json`), heavy `any`;
  bleeding-edge/likely-invalid dep pins (`react@^19.2.6`, `vite@^8`, `typescript@~6.0`,
  `eslint@^10`) risk non-reproducible builds; PDF export revokes the blob URL before the
  deferred `print()` (`ScenarioDetailPage.tsx:3667-3670`) → blank tab; no URL router
  (back/refresh/deep-link broken). Fixes per item.
- **M10 — Unrestricted root SSH + arbitrary `custom` command channel**: the single
  `DEPLOY_SSH_KEY` is unrestricted root (`INFRASTRUCTURE.md:27,37`) and `server-command.yml`
  exposes a `custom` arbitrary-bash-as-root path. Audit-logged but not prevented. Fix:
  non-root deploy user, forced-command whitelist, drop `custom`, rotate key, `from=` allowlist.

### LOW

- **L1 — Dead code/features (confirmed by data):** double-blind/screening columns are ~100%
  NULL (`reviewer_1/2_status`, `screening_reason/notes`, `kappa_final_status`,
  `structured_abstract`, `study_design`, `sample_size`) → those features are effectively
  unused; `scenario_kappa_cache` dead (`if False`); `getEvidenceBriefPdfUrl` exported-unused;
  several backend endpoints unreferenced by the frontend. Decide: implement or remove.
- **L2 — `document_chunk` metadata-scoring columns mostly empty**: `char_start/char_end/
  section_label` 100% NULL, `token_count` 75% NULL → the chunk-scoring feature is unpopulated.
- **L3 — `generate_schema.py:99` emits invalid `ALTER TABLE … ADD CONSTRAINT IF NOT EXISTS`**
  (Postgres rejects `IF NOT EXISTS` on ADD CONSTRAINT) → regenerated schema won't replay.
- **L4 — Bad defaults / docs:** `quality_score DEFAULT 0.0` (NULL would distinguish "unscored");
  `INFRASTRUCTURE.md` claims PostgreSQL 15 but it's **14.23**; `updated_at` columns have
  `DEFAULT now()` but no auto-update trigger; README is the default Vite template.

---

## 4. Database findings (consolidated)

Live DB: PostgreSQL **14.23**, pgvector **0.8.2**. 8 tables, all with PKs.

| Table | Rows | Size | Notes |
|---|---|---|---|
| document_chunk | 323,868 | 5,088 MB | embedding vector(1536) 99.99% populated; **no ANN index** (C2); good FTS gin + btree |
| literature_document | 81,209 | 215 MB | **heavy duplication** (C3); ~40 cols, many ~100% NULL (L1) |
| article_scenarios | 25,301 | 59 MB | PK+UNIQUE(document_id,scenario_id)+FK+2 indexes — **healthy** (exists despite not being in repo) |
| scenario_settings | 10 | — | naive timestamps (M4) |
| user_scenarios | 38 | — | naive timestamps; pipeline_* cols |
| alert_subscriptions | 5 | — | UNIQUE(email,scenario_id) ✓ |
| user_scenario_folders | 7 | — | — |
| alembic_version | 1 | — | stamped `eb6b9e396ffc` (no-op) (C5) |

- **Constraints/keys present & correct:** all FKs (`article_scenarios.document_id`,
  `document_chunk.document_id`, `literature_document.canonical_id` self-FK,
  `user_scenarios.folder_id`) exist; **FK orphan check = 0** across all. Good.
- **Missing:** ANN vector index (C2); UNIQUE on doi/pmid/external_id (C3).
- **Null hotspots (>50% NULL):** ~40 columns; the double-blind/screening/bibliographic
  enrichment columns dominate (features unused, L1/L2).
- **Migration drift:** live schema ⟂ repo migrations (C5). Alembic stamped but empty.
- **Type hazards:** mixed timestamp/timestamptz (M4).

Reproducible checks (read-only) used: table sizes/counts, information_schema
columns/constraints/FKs, `pg_indexes`, `pg_stats.null_frac`, FK left-join orphan counts,
and `GROUP BY … HAVING count(*)>1` duplicate detection. (See `scripts/_audit_db.py`.)

---

## 5. Pipeline / deployment findings (consolidated)

- **Cron:** two identical `0 0 * * *` lines (H3 double-run); **fulltext & PICO not in cron**
  (rely on BG worker); embed runs but without the BG worker's "skip title_abstract when
  fulltext exists" optimization → wasted embeddings.
- **Order:** orchestrator order is correct (fulltext→embed→…); standalone scripts have no
  enforced order; PICO never re-runs after fulltext arrives (quality capped at abstract).
- **Idempotency:** `/documents` and chunk inserts non-idempotent (no `ON CONFLICT`); fulltext
  insert is idempotent (delete-then-insert) — good; `article_scenarios` link uses `ON CONFLICT`.
- **Single-instance:** scheduler has no lockfile/timer guard (H3/H7).
- **Deploy:** good — `set -euo pipefail`, atomic frontend swap + `.prev` rollback, blocking
  DB-touching `/health` gate, real post-deploy smoke test, serial `concurrency` group. Bad —
  non-fatal migration, frontend-only rollback (H6), half-deploy window (frontend swapped before
  backend validated).
- **CI:** no tests/lint/migration check; deploy on every push (M8).
- **Secrets/infra:** plaintext OpenAI key (C1), no TLS (C4), unrestricted root key + `custom`
  RCE channel (M10), env-path fragmentation (M5). No live secrets committed to git (good).
- **Runtime health:** disk 19%, mem fine, single uvicorn worker, `Restart=always`,
  NRestarts=0, only benign `prophet/plotly` warnings in logs.

---

## 6. What is healthy / well-designed

- **SQL injection: none.** Heavy dynamic SQL but columns are whitelisted and values bound
  throughout (`_build_where`, `_build_boolean_match_sql`); user terms additionally regex-sanitized.
- **Auth where applied:** timing-safe `compare_digest`, mandatory `WRITE_API_KEY` at boot
  (fail-fast), CORS is an explicit allowlist (not `*`).
- **Data integrity primitives:** all FKs enforced, **zero orphans**, embeddings 99.99%
  populated and dimension-consistent (1536 everywhere, matches `vector_cosine_ops`), `document_chunk`
  well-indexed for FTS + lookups, `alert_subscriptions`/`article_scenarios` have the right UNIQUE keys.
- **Atomic critical writes** (`/documents`, `/chunks`, multi-table scenario DELETE use one `engine.begin()`).
- **Connection pool tuned** (`pool_pre_ping`, `pool_recycle`); **startup orphan-recovery** for interrupted pipelines.
- **Frontend** has a clean centralized snake↔camel mapper layer, consistent `!ok`→throw,
  loading/error/empty states, `AbortController` on SSE streams, client-side export.
- **Deploy/CI** atomic frontend swap + rollback, blocking health gate, real smoke test, serial concurrency group.
- **Pipeline** orchestrator ordering, fulltext-before-embed, DB-as-source-of-truth counts, HTTP retry+`Retry-After` on fulltext fetchers.

---

## 7. Prioritized action plan (in order)

**Now / today (containment & safety):**
1. **Rotate the OpenAI API key** (C1); remove the `Environment=` literal from override.conf.
2. **Add TLS + HTTP→HTTPS redirect** (C4) — `WRITE_API_KEY` is currently on the wire in clear.
3. **Add auth to `/alerts/*` and gate `/ask/stream*` + `/*/rag`** (H1) — IDOR + billable DoS.

**This week (correctness & cost):**
4. **Create the HNSW vector index** (C2) — biggest single perf win; build `CONCURRENTLY`.
5. **De-duplicate `literature_document` + add UNIQUE(doi)/UNIQUE(pmid) partial indexes; make
   `/documents` `ON CONFLICT`; dedup checks fail-closed; filter `is_duplicate` in counts** (C3, H5).
6. **Remove the duplicate cron line; add work-claiming locks (`FOR UPDATE SKIP LOCKED`) and a
   scheduler `flock`** (H3); add OpenAI retry/backoff and don't advance the pipeline on partial
   embedding (H4).
7. **Add an API-key entry UI (or server session)** so write features work (H2).

**This month (resilience & hygiene):**
8. **Make Alembic real**: baseline migration reproducing the live schema; set `target_metadata`;
   fold `_ensure_*`/archive SQL into versioned migrations; make migration failure fatal in deploy (C5, H6).
9. **Commit the systemd unit pinned to one worker** (or externalize background work) (H7);
   **standardize the env-file path** (M5).
10. **Add a backend rollback to deploy** on health failure (H6); **add lint + a starter pytest
    suite + alembic-on-empty-DB check to CI** (M8). Then write regression tests for each fix above.
11. **Replace silent excepts with logging; return real HTTP status codes** (H5).
12. Address Mediums: de-N+1 `/gesica/scenarios` (M1), coherent `article_count` (M2), cache
    invalidation (M3), timestamptz (M4), Pydantic bodies (M6), GET→POST clustering (M7),
    frontend strict/deps/PDF/routing (M9), SSH hardening (M10).
13. Lows: remove dead code/columns or implement the features behind them (L1–L4).

### Test-coverage note
There is **zero** automated test coverage today (no pytest, no frontend test runner). Per the
request to cover every confirmed bug: each fix above has a concrete verification listed, but
they should be encoded as regression tests once a `tests/` harness + CI job exist (step 10) —
starting with auth (`require_api_key`, `/alerts/*`), dedup idempotency (`/documents` double-POST),
count coherence, and the vector-index EXPLAIN check.
