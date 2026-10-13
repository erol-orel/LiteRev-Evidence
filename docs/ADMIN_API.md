# LiteRev - Admin / Operations API

Operational reference for driving LiteRev directly (build corpora, run maintenance,
enrichment, scoring, digests). Frontend-only read endpoints are omitted - see the route
table in `main.py` for the full 136-route surface.

## Base URL

- **On the server:** `http://localhost:8000`
- **Public:** behind nginx on `literev-app-01` (`62.238.39.50`) - use your API host.

Below, `$BASE` = the API base and `$KEY` = the admin key.

## Authentication

Mutating endpoints require the **`X-API-Key`** header, compared (constant-time) against the
running service's `WRITE_API_KEY`. `GET` stats/status endpoints are open.

Read the key from the **running process** - this is authoritative. systemd may inject
`WRITE_API_KEY` via a drop-in `Environment=` **or a separate `EnvironmentFile=`** (find it with
`systemctl show literev-api -p Environment -p EnvironmentFiles`), which can differ from
`secrets.env`, and `main.py` does not override an env var that is already set. (Reading
`secrets.env` directly can therefore give a stale key → `Invalid API key`.)

```bash
BASE=http://localhost:8000
PID=$(systemctl show -p MainPID --value literev-api)
KEY=$(tr '\0' '\n' < /proc/$PID/environ | sed -n 's/^WRITE_API_KEY=//p')
[ -n "$KEY" ] && echo "key loaded (${#KEY} chars)"   # never echo the value itself into logs
```

- `401 Invalid API key` - missing/wrong/stale key. `503 Server not configured…` - `WRITE_API_KEY` unset.

> **Drift check.** If `grep '^WRITE_API_KEY=' /opt/literev-api/secrets.env` differs from the process
> value above, `secrets.env` is stale. Pick one source of truth - either put the real key in
> `secrets.env` and remove the systemd `Environment=` line (`systemctl edit literev-api`), or update
> `secrets.env` to match - then `systemctl restart literev-api`. Rotate the key if it has leaked.

---

## Health & diagnostics

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| GET | `/health` | open | liveness (`{"status":"ok"}`) |
| GET | `/sources/health?query=cardiac%20arrest&timeout=12` | open | per-source live probe (status, latency, count) - diagnose slow/broken sources |
| GET | `/search/limits` | open | the per-source caps this server applies (standard, exhaustive, relevance-ranked passes), as the search screen states them |
| GET | `/corpus/stats` | open | global corpus counts |
| GET | `/corpus/fulltext-stats` | open | full-text vs abstract-only breakdown |
| GET | `/enrichment/status` | open | PICO / metadata / full-text backfill progress |

```bash
curl -s "$BASE/sources/health" | jq .
```

---

## Build a corpus

The corpus of a scenario = the boolean query over (local library ∪ live sources). Every search
is exhaustive: each database returns every record it matches, up to the larger of
`EXHAUSTIVE_MAX_PER_SOURCE` (20000) and `LIVE_MAX_PER_SOURCE` per database (Crossref, which only
ranks keywords, keeps its `RANKED_PASS_MAX` most relevant), within the larger of
`EXHAUSTIVE_FEDERATION_BUDGET` (2700 s) and `POPULATE_FEDERATION_BUDGET`. `GET /search/limits`
returns these caps and the budget. This is the full federation (12 fetchers / 13 sources).

| Method | Path | Auth | Key params |
|--------|------|------|-----------|
| POST | `/user-scenarios/{id}/populate` | ✅ | `max_results` (default `LIVE_MAX_PER_SOURCE`), `include_live` (default true), `exclude_preprints`. The search is always exhaustive; an `exhaustive` parameter from an older caller is ignored |
| GET | `/user-scenarios/{id}/populate/status` | open | `{status, phase, ingested, sources, rerank_status}` - poll until `status=done` |
| POST | `/scenarios/{id}/rebuild-corpus` | ✅ | re-run boolean membership on the existing library (no live fetch) |
| POST | `/user-scenarios/{id}/pipeline` | ✅ | `max_results` (default `LIVE_MAX_PER_SOURCE`) - full pipeline (populate → embed → scores → full text → PICO → metadata → views). Full text, PICO and metadata cover the relevant articles (`PIPELINE_ENRICH_SCOPE`, default `relevant`; `all` for the whole scenario) and wait until a person has chosen the threshold: the status then carries `waiting_threshold: true` |
| GET | `/user-scenarios/{id}/pipeline/status` | open | pipeline phase/progress |

```bash
# Build the full corpus, then watch it fill live:
curl -s "${auth[@]}" -X POST "$BASE/user-scenarios/usr-XXXX/populate?max_results=2000&include_live=true"
watch -n3 "curl -s $BASE/user-scenarios/usr-XXXX/populate/status | jq '{status,phase,ingested,sources}'"
```

---

## Corpus maintenance

| Method | Path | Auth | Key params |
|--------|------|------|-----------|
| POST | `/admin/corpus-maintenance` | ✅ | optional JSON body; idempotent + reversible: purge duplicates, normalize legacy chunks |
| POST | `/admin/embed-pending?limit=200` | ✅ | embed chunks still missing vectors (`limit` per call) |
| POST | `/admin/recompute-quality-scores?limit=5000&only_missing=true` | ✅ | recompute per-doc quality scores; `&scenario_id=<id>` holds it to one scenario's articles |

```bash
curl -s "${auth[@]}" -X POST "$BASE/admin/corpus-maintenance"
curl -s "${auth[@]}" -X POST "$BASE/admin/embed-pending?limit=500"
```

---

## Enrichment (batch)

Each accepts `?scenario_id=<id>` (omit = whole library) and `?limit=<n>`.

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| POST | `/fulltext/fetch?scenario_id=…&scope=…&limit=…` (scope `relevant` or `all`) | ✅ | download the open-access full text (PMC, Europe PMC, Unpaywall, bioRxiv/medRxiv, Semantic Scholar, OpenAlex) of the articles that have none, with the pipeline's own code; an article that gains a text is read again by the next extraction |
| POST | `/pico/extract?scenario_id=…&limit=…` | ✅ | LLM PICO extraction |
| POST | `/metadata/extract?scenario_id=…&limit=…` | ✅ | LLM metadata extraction; fills the design, sample size and quality score columns, as the pipeline step does |

```bash
curl -s "${auth[@]}" -X POST "$BASE/fulltext/fetch?scenario_id=usr-XXXX&limit=1000"
```

---

## Scoring

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| POST | `/scenarios/{id}/rerank` | ✅ | Cohere cross-encoder rerank of the relevant subset |
| GET | `/scenarios/{id}/rerank/status` | ✅ | rerank progress |

---

## Living review & alerts

| Method | Path | Auth | Key params |
|--------|------|------|-----------|
| GET | `/living-review/status` | open | last report, and `run`: running · done · error · interrupted |
| POST | `/living-review/run?scenario_id=all&dry_run=false` | ✅ | run living review (detached; `days` optional, applies to every scenario run, otherwise each preset keeps its own window) |
| POST | `/alerts/run-digests?dry_run=false` | ✅ | send all due email digests |
| POST | `/alerts/send-digest` | ✅ | send one subscription's digest |
| GET | `/alerts/subscriptions` | open | list subscriptions |

---

## Scenario management

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| GET | `/user-scenarios` · `/user-scenarios/by-owner?owner=<email>` | open | list |
| POST | `/user-scenarios` | ✅ | create (body: `{name, query, mode, filters, search_strategy, sub_queries, combinator}`) |
| PATCH | `/user-scenarios/{id}` | ✅ | rename / pin / move folder |
| DELETE | `/user-scenarios/{id}` | ✅ | delete |
| GET | `/user-scenarios/{id}/detail` · `/user-scenarios/{id}/corpus?limit=…&offset=…` | open | detail / paginated corpus |

---

## Environment knobs (`/opt/literev-api/secrets.env`, then `systemctl restart literev-api`)

| Var | Default | Effect |
|-----|---------|--------|
| `WRITE_API_KEY` | - | admin key (required) |
| `LIVE_MAX_PER_SOURCE` | 2000 | floor of the per-source cap of a search (the larger of it and `EXHAUSTIVE_MAX_PER_SOURCE`), and ceiling of the ranked passes |
| `EXHAUSTIVE_MAX_PER_SOURCE` | 20000 | the per-source guard rail of a search |
| `POPULATE_FEDERATION_BUDGET` / `EXHAUSTIVE_FEDERATION_BUDGET` | 600 / 2700 | seconds the live federation runs (the larger of the two) before slow sources are cut off |
| `NCBI_API_KEY` / `SEMANTIC_SCHOLAR_API_KEY` / `CORE_API_KEY` | - | raise per-source rate limits / enable CORE |
| `OPENAI_API_KEY` | - | embeddings + LLM extraction |
| `RAG_MIN_SIMILARITY` | 0.18 | RAG passage-match floor |

> The **Server command (manual)** GitHub Action (`Actions → Server command`) runs `diagnose`,
> `migrate`, `restart`, `logs`, `deploy`, or a custom command over SSH without shell access.
