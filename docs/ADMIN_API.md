# LiteRev — Admin / Operations API

Operational reference for driving LiteRev directly (build corpora, run maintenance,
enrichment, scoring, digests). Frontend-only read endpoints are omitted — see the route
table in `main.py` for the full 136-route surface.

## Base URL

- **On the server:** `http://localhost:8000`
- **Public:** behind nginx on `literev-app-01` (`62.238.39.50`) — use your API host.

Below, `$BASE` = the API base and `$KEY` = the admin key.

## Authentication

Mutating endpoints require the **`X-API-Key`** header, compared (constant-time) against the
`WRITE_API_KEY` env var (in `/opt/literev-api/secrets.env`). `GET` stats/status endpoints are open.

```bash
BASE=http://localhost:8000
KEY=$(grep -E '^WRITE_API_KEY=' /opt/literev-api/secrets.env | cut -d= -f2-)
auth=(-H "X-API-Key: $KEY")
```

- `401 Invalid API key` — missing/wrong key. `503 Server not configured…` — `WRITE_API_KEY` unset.

---

## Health & diagnostics

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| GET | `/health` | open | liveness (`{"status":"ok"}`) |
| GET | `/sources/health?query=cardiac%20arrest&timeout=12` | open | per-source live probe (status, latency, count) — diagnose slow/broken sources |
| GET | `/corpus/stats` | open | global corpus counts |
| GET | `/corpus/fulltext-stats` | open | full-text vs abstract-only breakdown |
| GET | `/enrichment/status` | open | PICO / metadata / full-text backfill progress |

```bash
curl -s "$BASE/sources/health" | jq .
```

---

## Build a corpus

The corpus of a scenario = the boolean query over (local library ∪ live sources), capped at
`LIVE_MAX_PER_SOURCE` (default 2000) per source, within the `POPULATE_FEDERATION_BUDGET` time
budget. This is the full federation (12 fetchers / 13 sources).

| Method | Path | Auth | Key params |
|--------|------|------|-----------|
| POST | `/user-scenarios/{id}/populate` | ✅ | `max_results` (default 100000 → clamped to `LIVE_MAX_PER_SOURCE`), `include_live` (default true) |
| GET | `/user-scenarios/{id}/populate/status` | open | `{status, phase, ingested, sources, rerank_status}` — poll until `status=done` |
| POST | `/scenarios/{id}/rebuild-corpus` | ✅ | re-run boolean membership on the existing library (no live fetch) |
| POST | `/user-scenarios/{id}/pipeline` | ✅ | `max_results` (default 500) — full pipeline (populate → fulltext → embed) |
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
| POST | `/admin/recompute-quality-scores?limit=5000&only_missing=true` | ✅ | recompute per-doc quality scores |

```bash
curl -s "${auth[@]}" -X POST "$BASE/admin/corpus-maintenance"
curl -s "${auth[@]}" -X POST "$BASE/admin/embed-pending?limit=500"
```

---

## Enrichment (batch)

Each accepts `?scenario_id=<id>` (omit = whole library) and `?limit=<n>`.

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| POST | `/fulltext/fetch?scenario_id=…&limit=…` | ✅ | fetch full text (Unpaywall / PMC) |
| POST | `/pico/extract?scenario_id=…&limit=…` | ✅ | LLM PICO extraction |
| POST | `/metadata/extract?scenario_id=…&limit=…` | ✅ | LLM metadata (design, quality) extraction |

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
| GET | `/living-review/status` | open | scheduler state |
| POST | `/living-review/run?scenario_id=all&days=30&dry_run=false` | ✅ | run living review (async) |
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
| `WRITE_API_KEY` | — | admin key (required) |
| `LIVE_MAX_PER_SOURCE` | 2000 | max docs fetched **per source** during populate |
| `POPULATE_FEDERATION_BUDGET` | 180 | seconds the live federation runs before slow sources are cut off |
| `NCBI_API_KEY` / `SEMANTIC_SCHOLAR_API_KEY` / `CORE_API_KEY` | — | raise per-source rate limits / enable CORE |
| `OPENAI_API_KEY` | — | embeddings + LLM extraction |
| `RAG_MIN_SIMILARITY` | 0.18 | RAG passage-match floor |

> The **Server command (manual)** GitHub Action (`Actions → Server command`) runs `diagnose`,
> `migrate`, `restart`, `logs`, `deploy`, or a custom command over SSH without shell access.
