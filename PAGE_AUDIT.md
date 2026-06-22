# Page-by-Page Error Audit — LiteRev-Evidence

Date: 2026-06-22 · Branch: `claude/inspiring-gates-0u6dce` · PR #88

Trigger: runtime errors reported across many pages — **HTTP 429** on the
EVIDENCES tab and **HTTP 500** on the PRISMA tab (screenshots) — with the note
"errors on many pages, full audit of every page".

Method: static analysis of the backend (`main.py`, ~13.3k lines, ~150 endpoints)
and the React SPA (`frontend/src/App.tsx`, `ScenarioDetailPage.tsx`, `lib/api.ts`).
Undefined-name bugs were surfaced with `ruff --select F821,F811`; a structured
page→endpoint inventory and a seven-pattern 500-risk sweep of every page-serving
handler were run in parallel and each candidate read in context to rule out guards.

The frontend prints the backend status code verbatim (`throw new Error(\`HTTP ${status}\`)`
in 60+ API functions), so **every error box on screen maps 1:1 to a real backend
response**. Fix the backend → the boxes disappear.

---

## 1. Root causes found & fixed

| # | Symptom (UI) | Endpoint | Root cause | Fix |
|---|---|---|---|---|
| 1 | **HTTP 500** on PRISMA (every system scenario) | `GET /gesica/scenarios/{id}/prisma` | Handler used `meta["title"]` but never assigned `meta` — it discarded the result of `_get_db_gesica_scenario_or_404(...)`. Unbound name → `NameError` → 500 on every call. | Capture the row: `meta = _get_db_gesica_scenario_or_404(scenario_id)`; use null-safe `_gesica_title(meta)`. |
| 2 | **HTTP 429** on EVIDENCES and most scenario tabs | rate-limit middleware | The "expensive endpoint" matcher keyed on the path prefix *before* the first `{`. So `/user-scenarios/{id}/rag` matched **all** of `/user-scenarios/*`, `/gesica/scenarios/*`, and `/scenarios/*` — the entire scenario detail page (corpus, prisma, evidence-brief, clustering, model, variables…) was throttled at **30 req/min** instead of 600. One page load + 5 s polling exhausts it → spurious 429s app-wide. | Precise per-segment regex matching (`{param}` → exactly one path segment). Only `rag` / `full-pipeline` / `search` / `ask` stay on the 30/min bucket. Added `Retry-After` header. Unit-tested. |
| 3 | Silent — full text never retrieved (no error shown) | user-scenario pipeline (full-text step) | `_requests` used but never imported; every fetch raised `NameError`, swallowed by a broad `except` → full text silently never fetched, degrading evidence quality. | Added `import requests as _requests`. |
| 4 | Latent (theoretical) — 500 on folder create | `POST /user-scenario-folders` | Re-read `row` after INSERT (separate connection) was dereferenced with no `if row:` guard. Would only fire on a race/external delete, but the whole response block was exposed. | Build the response from the known inserted values; guard `created_at` with `row and …`. |

All verified: `python -m py_compile main.py` ✓ · `ruff --select F821,F811` clean ✓ ·
rate-limiter matcher unit-tested across the full path matrix ✓ · CI run #226 green ✓.

### Why these explain "errors on many pages"
Fix #2 is the systemic one: because the over-greedy matcher swept three whole
URL namespaces into the 30/min bucket, **almost every tab of the scenario detail
page** (the app's main workspace) could randomly return 429 under normal use.
That single bug accounts for the "many pages" breadth; #1 accounts for the
specific PRISMA 500.

---

## 2. Per-page audit

Legend: ✅ no code-level error found · 🔧 fixed in this PR · ⚠️ pre-existing UX note.

### Top-level app (`App.tsx`)

| Page / tab | Key endpoints | Status |
|---|---|---|
| **Recherche** (search) | `POST /search`, `GET /filters-options`, `GET /documents/{id}`, `GET /evidence-summary/{id}` | ✅ (`/search` is expensive-limited by design) |
| **Scénarios** (list) | `GET /gesica/scenarios`, `GET /gesica/model/demand-forecasting`, `GET /scenarios/{id}/recommended-actions` | ✅ |
| **Statistiques** | `GET /corpus/stats`, `/corpus/fulltext-stats`, `/corpus/stats/by-year[/named]`, `/gesica/stats`, `/gesica/scenarios` | ✅ |
| **Terrain** | `GET /terrain/{meteo,geo,epidemic,demographics,pharmacies,informal-signals,climate}` | ✅ (errors already soft-handled with a friendly message) |

### Scenario detail (`ScenarioDetailPage.tsx`) — 8 sections

| Section → sub-tab | Key endpoints | Status |
|---|---|---|
| Corpus & Revue → **Corpus** | `GET {base}/{id}/corpus`, `/detail`, `/embedding-status`, `PATCH /scenarios/{id}/settings` | 🔧 429 (fix #2) |
| Corpus & Revue → **PRISMA** | `GET {base}/{id}/prisma` | 🔧 **500 (fix #1)** + 429 (fix #2) |
| Corpus & Revue → **Double-Aveugle** | `GET {base}/{id}/double-blind/{kappa,conflicts}`, `POST …/decision` | 🔧 429 (fix #2) |
| PICO & Evidence → **PICO** | `GET {base}/{id}/pico-bulk` | 🔧 429 (fix #2) |
| PICO & Evidence → **EVIDENCES** (brief) | `GET {base}/{id}/evidence-brief`, `GET /scenarios/{id}/evidence-brief/llm`, `POST …/generate`, `GET …/generate/status` (polls 5 s) | 🔧 **429 (fix #2)** |
| **Assistant IA** | `POST /ask/stream` (SSE) | ✅ (expensive-limited by design) |
| Visualisation → **Knowledge graph** | `GET {base}/{id}/knowledge-graph` | 🔧 429 (fix #2) |
| Visualisation → **Clustering** | `GET {base}/{id}/clustering` | 🔧 429 (fix #2) |
| Variables & Modèle → **Variables** | `GET/POST /scenarios/{id}/variables[/generate|/validate]` (polls) | 🔧 429 (fix #2) |
| Variables & Modèle → **Monitor/Model** | `GET /scenarios/{id}/model/{run,data,monitor,spec,train/status}`, several POST | 🔧 429 (fix #2) |
| **Stratégie** | `GET /user-scenarios/{id}/search-strategy`, `POST /search-strategy` | ✅ |
| **Enrichissement** | `POST /{pico,metadata}/extract`, `/fulltext/fetch`, `GET /enrichment/status` | 🔧 full-text path benefits from fix #3 |
| **Alertes** | `POST /alerts/subscribe` | ✅ |

`{base}` = `/user-scenarios` (ids starting `usr-`) or `/gesica/scenarios` (system scenarios).

---

## 3. Backend 500-risk sweep — result

A seven-pattern sweep (ZeroDivisionError, None-indexing of DB rows, `json.loads`
on None, `int/float/strftime` on None, `KeyError`, unbound locals) over every
page-serving handler found **no remaining high-confidence 500 bugs** beyond the
fixes above. The codebase is unusually defensive:

- Every division is guarded (`or 1`, `if x > 0 else 0`, `max(.., 1)`, or early return).
- Every `int(col)`/`float(col)` on a DB value is wrapped (`or 0`, `or 0.0`, `if col else …`).
- Ungrouped SQL aggregates (`COUNT/SUM/MIN/MAX … WHERE …` with no `GROUP BY`)
  always return exactly one row, so `.mappings().first()` is never `None` there
  even for an empty scenario — invalidating the "empty corpus → None row → crash" theory.
- By-id lookups are guarded by `if not row:` / `_get_*_or_404()`.
- JSONB columns are auto-parsed by the driver (used as dicts; no `loads`-on-dict bug).

~16 candidates were examined and cleared (e.g. `main.py:4192, 5119, 5839, 6552,
7404, 7519, 9818, 10088, 10227, 10797`).

---

## 4. Systemic notes & recommendations (not yet changed)

1. **⚠️ Frontend surfaces raw `HTTP <status>`** in 60+ API helpers. Even after
   the backend fixes, any transient hiccup (a genuine 429 under real load, a 502
   during deploy) shows a scary red box. Recommend a shared GET helper that
   retries once or twice on 429/503 honoring `Retry-After`, and maps statuses to
   human messages (404 → "introuvable", 5xx → "réessayez"). Deferred here because
   it touches ~40 call sites and the frontend can't be type-checked in this
   sandbox (no `node_modules`); best done as its own reviewed change.

2. **Rate-limiter is per-process, in-memory, keyed by `X-Forwarded-For` last hop.**
   This is correct behind a single nginx that sets the header, but: (a) with
   multiple uvicorn workers each worker has its own counters (limits multiply);
   (b) if nginx does *not* forward the header, every client collapses to
   `127.0.0.1` and shares one bucket. Worth confirming the nginx `proxy_set_header
   X-Forwarded-For` config on the deployed box (not in this repo).

3. **If 500s persist in production after this PR**, the remaining likely causes
   are *outside* application logic: a DB schema/migration drift (a column the SQL
   selects that's missing in the deployed DB — consistent with the existing
   `AUDIT_REPORT.md` finding that the schema lives only in the live DB), an
   external-call failure (OpenAI/HTTP) on generate paths, or auth. These need a
   live log line (the exact 500 traceback) to pin down.

---

## 5. How to verify

```bash
python -m py_compile main.py            # syntax
ruff check main.py --select F821,F811   # undefined names / redefinitions → clean
```

Functional check once deployed:
- Open a system scenario → **PRISMA** tab → should render the flow (was 500).
- Open **EVIDENCES** → brief loads or shows the friendly "génération en cours"
  (no raw HTTP 429); rapid tab-switching across the detail page no longer 429s.
