# Operations Runbook — LiteRev-Evidence

Steps that run on the **server** (`literev-app-01`) or in external **dashboards**,
which the agent can't execute remotely. Run as root/sudo on the server unless noted.

Conventions:
- App env file: `/etc/literev-api.env` (holds `DB_URL`, `OPENAI_API_KEY`, `WRITE_API_KEY`).
- App runs on `localhost:8000` (nginx adds the public `/api` prefix).
- Service: `systemctl restart literev-api`. Health: `curl -fsS http://localhost:8000/health`.
- **Always** back up the env file before editing: `sudo cp /etc/literev-api.env /etc/literev-api.env.bak`.

---

## 1. HTTPS for `literev-scenario.com`

**Prerequisites**
- DNS **A record**: `literev-scenario.com` → server's public IP. Verify before proceeding:
  ```bash
  dig +short literev-scenario.com        # must return the server IP
  ```
  (Let's Encrypt will NOT issue a cert for a bare IP — the domain must resolve first.)
- Ports **80 and 443** open in the firewall / cloud security group.
- nginx is the front proxy (it already terminates the public site and proxies `/api`).

**Issue + install the certificate (certbot auto-configures nginx)**
```bash
sudo apt-get update && sudo apt-get install -y certbot python3-certbot-nginx
sudo certbot --nginx -d literev-scenario.com          # add: -d www.literev-scenario.com  if you use www
#   prompts: email, agree to TOS, and choose "Redirect" (HTTP→HTTPS) when asked.
```
certbot edits the existing nginx server block to listen on 443 with the cert, adds the
80→443 redirect, and installs a renewal systemd timer.

**Verify + confirm auto-renewal**
```bash
curl -fsSI https://literev-scenario.com | head -1     # expect HTTP/2 200
sudo certbot renew --dry-run                           # must succeed
systemctl list-timers | grep certbot                  # renewal timer present
```

**Frontend check after switching to HTTPS**
- The frontend calls the API through nginx with relative paths, so it should "just work".
- If a build-time `VITE_API_BASE`/base URL is pinned to `http://<ip>` anywhere, change it to the
  HTTPS domain (or a relative path) and redeploy, otherwise the browser blocks mixed content.
- (If you want a hand-written nginx server block instead of the certbot plugin, paste your
  current nginx site config and I'll produce the exact 443 block.)

---

## 2. Rotate `WRITE_API_KEY`

```bash
sudo cp /etc/literev-api.env /etc/literev-api.env.bak
NEW=$(openssl rand -hex 32)
sudo sed -i "s|^WRITE_API_KEY=.*|WRITE_API_KEY=${NEW}|" /etc/literev-api.env
sudo systemctl restart literev-api
curl -fsS http://localhost:8000/health                 # expect {"status":"ok"}
printf 'NEW WRITE_API_KEY: %s\n' "$NEW"                # copy to your password manager, then clear scrollback
```
Then update the **only** client that holds it: in the web app, open the admin key control
(lock/key button in the header), clear the old key, paste the new one. (The frontend stopped
shipping a build-time key in #119, so nothing else needs changing.)

---

## 3. Rotate the OpenAI API key

1. In the OpenAI dashboard (platform.openai.com) → **API keys** → create a new secret key.
2. On the server:
   ```bash
   sudo cp /etc/literev-api.env /etc/literev-api.env.bak
   sudo sed -i "s|^OPENAI_API_KEY=.*|OPENAI_API_KEY=sk-...NEW...|" /etc/literev-api.env
   sudo systemctl restart literev-api
   curl -fsS http://localhost:8000/health
   ```
3. Back in the dashboard, **revoke the old key**.

---

## 4. OpenAI budget cap (stop any future unbounded bleed)

In platform.openai.com:
- **Settings → Limits** (org and/or project): set a **hard monthly usage limit** and a lower
  **email alert threshold**. The hard limit makes the API start returning 429s instead of
  spending past the cap.
- Recommended: create a dedicated **Project** for LiteRev with its own key + budget, so a bug
  can never exceed that project's cap.
- Belt-and-suspenders already in code: set `PICO_AUTOEXTRACT_ENABLED=0` in
  `/etc/literev-api.env` (+ restart) to hard-stop the background PICO worker instantly.

---

## 5. Fake / empty scenario records — list them (read-only) before any deletion

Deleting records is destructive, so first produce the list and review it together. This query
only SELECTs. It flags scenarios with **no** scored members AND **no** ingestion docs:

```bash
cd /opt/literev-api && set -a; . /etc/literev-api.env; set +a
.venv/bin/python3 - <<'PY'
import os
from sqlalchemy import create_engine, text
e = create_engine(os.environ["DB_URL"])
with e.connect() as c:
    rows = c.execute(text("""
        SELECT s.id, s.is_system,
               COALESCE(s.name, s.title) AS label,
               (SELECT count(*) FROM article_scenarios ars WHERE ars.scenario_id = s.id) AS members,
               (SELECT count(*) FROM literature_document d
                  WHERE d.scenario_type = s.id AND d.project_context = 'literev') AS ingest_docs
        FROM user_scenarios s
        ORDER BY members ASC, ingest_docs ASC, s.id
    """)).mappings().all()
    print(f"{'id':28} sys   members  ingest  label")
    for r in rows:
        flag = "  <== EMPTY" if (r['members'] == 0 and r['ingest_docs'] == 0) else ""
        print(f"{r['id']:28} {str(r['is_system'])[:1]:3} {r['members']:8} {r['ingest_docs']:7}  {r['label']}{flag}")
PY
```
Paste the output here. I'll mark which `<== EMPTY` rows are safe to delete (the known fake
GESICA stubs vs any legitimately-new-but-unpopulated user scenario), then hand you a guarded
`DELETE` wrapped in a transaction with a row-count assertion so it can't over-delete.

---

## 6. Verify a deploy from the server (optional)

```bash
curl -fsS http://localhost:8000/health
curl -fsS -X POST http://localhost:8000/search -H 'Content-Type: application/json' \
  -d '{"query_text":"test","mode":"hybrid","limit":1}' | head -c 200; echo
journalctl -u literev-api -n 50 --no-pager        # recent service logs
```

---

## 7. Monitoring & alerting

Motivation: a deploy failed **silently for hours** (the `deploy.sh` SIGPIPE bug)
before it was noticed. These catch that class of problem early. Nothing here is
required for the app to run — they're guardrails.

### 7a. Backend error visibility (in code, already shipped)
- An HTTP middleware logs every **unhandled** exception with `method path from IP`
  at `ERROR`, so real 500s are greppable:
  ```bash
  journalctl -u literev-api --since "-1h" | grep -iE "Unhandled error|Traceback"
  ```
- **Optional Sentry** (off by default). To get email/Slack alerts on backend
  exceptions, install the SDK and set the DSN, then restart:
  ```bash
  /opt/literev-api/.venv/bin/pip install sentry-sdk
  sudo sed -i '/^SENTRY_DSN=/d' /etc/literev-api.env
  echo 'SENTRY_DSN=https://<your-dsn>@sentry.io/<project>' | sudo tee -a /etc/literev-api.env
  sudo systemctl restart literev-api
  ```
  With no `SENTRY_DSN` (or no `sentry-sdk` installed) it's a no-op — errors still
  hit journalctl via the middleware above.

### 7a′. Slow requests and process memory (in code, already shipped)
- Every request slower than `SLOW_REQUEST_MS` (default 2000) is logged at
  `WARNING` with its route, duration, status and size — the first thing to read
  when the interface shows "Failed to fetch" or a tab spins:
  ```bash
  journalctl -u literev-api --since "-1h" | grep "slow request"
  ```
- `/health` carries a `process` block: `rss_mb`, `rss_peak_mb`, `threads`,
  `uptime_s` and the DB pool line. A short `uptime_s` right after a "Failed to
  fetch" means the API restarted (deploy or crash); a `rss_peak_mb` close to the
  machine's RAM means the process is being killed for memory.
- `scripts/bench_scenario.py` times every read endpoint of the scenario page for
  one scenario. On the server, read-only, against the running API:
  ```bash
  cd /opt/literev-api && .venv/bin/python3 scripts/bench_scenario.py \
      --base http://127.0.0.1:8000 --scenario usr-xxxxxxxxxxxx --read-only
  ```
  It prints one line per endpoint (ms, KB, status), sorted by time, and flags
  anything slower than 2 s or larger than 2 MB. Locally, `--seed 25000` builds a
  synthetic 25,000-article scenario first (that run found the 27 MB search-page
  corpus fetch, the 2.5 MB settings call and the 2.5 MB clustering payload fixed
  in September 2026). `--compute` (local only) also times the computations —
  scoring, cross-encoder, brief context, clustering, knowledge graph, PRISMA,
  counts, the LLM generators — with OpenAI and Cohere stubbed and random
  embeddings seeded; it is what showed clustering all 25,000 articles peaking at
  3 GB of RAM, hence `CLUSTER_MAX_DOCS`.
- `scripts/audit_scenario.py` cross-checks every number the interface shows for
  one scenario (card count, header total, PRISMA arithmetic and screened count,
  embedding status, clustering sizes and language, threshold, cached artefacts)
  and prints OK / WARN / FAIL per check. Read-only, GET endpoints only:
  ```bash
  cd /opt/literev-api && .venv/bin/python3 scripts/audit_scenario.py \
      --base http://127.0.0.1:8000 --scenario usr-xxxxxxxxxxxx --lang en
  ```
  While a search or pipeline is running for that scenario, disagreements are
  reported as WARN (expected) rather than FAIL; rerun when it ends.

### 7b. Uptime check on `/health` (external)
Point any uptime monitor (UptimeRobot, Better Stack, Hetzner, a cron+curl) at
**`https://literev-scenario.com/api/health`** (through nginx) — expect HTTP 200
`{"status":"ok","database":"ok"}`. Alert if non-200 or the body's `database` isn't
`ok`. A 1–5 min interval is plenty. `/health` is exempt from rate limiting.

### 7c. Deploy-failure alert (GitHub Actions)
The "Deploy to production" job can fail without anyone noticing. Add a failure
notification to `.github/workflows/deploy.yml` (a final step with
`if: failure()`), e.g. a Slack/Discord webhook or an email action:
```yaml
      - name: Notify on failure
        if: failure()
        run: |
          curl -fsS -X POST "$DEPLOY_ALERT_WEBHOOK" \
            -H 'Content-Type: application/json' \
            -d "{\"text\":\"❌ LiteRev deploy failed on ${{ github.sha }} — ${{ github.event.head_commit.message }}\"}"
        env:
          DEPLOY_ALERT_WEBHOOK: ${{ secrets.DEPLOY_ALERT_WEBHOOK }}
```
(Add the `DEPLOY_ALERT_WEBHOOK` repo secret first. Tell me the channel and I'll
wire the exact step.)

## 8. Tests (backend, frontend, browser)

Three layers, all run by CI on every pull request (`.github/workflows/deploy.yml`,
job "Build & checks") before anything reaches production:

| Layer | Command | What it covers |
|---|---|---|
| Backend | `pytest -q` (repo root) | pure logic + integration tests on the Postgres named by `DB_URL` (skipped without one). **The integration tests truncate and rewrite the tables of that database: point `DB_URL` at a scratch database, never at production or at a database whose data you want to keep** |
| Frontend unit | `npm test` (in `frontend/`) | vitest + Testing Library: search-text helpers, API client (retries, URLs, admin key, error messages), locale files (same keys and placeholders in French and English), language provider, error boundary |
| Browser smoke | `python3 scripts/smoke_e2e.py` (repo root) | Playwright drives the **built** interface against a **real API on a throwaway database**: scenario list and language toggle, scenario page (header, corpus, PRISMA, clustering), a two-facet local search that creates a scenario named with its AND |

### Running the browser smoke test locally
```bash
pip install uvicorn                                   # once, in the API virtualenv
cd frontend && npm ci && npx playwright install --with-deps chromium && cd ..
DB_URL=postgresql+psycopg://postgres:postgres@127.0.0.1:5432/postgres \
    python3 scripts/smoke_e2e.py                      # builds the frontend, ~2 min
python3 scripts/smoke_e2e.py --no-build -- --headed -g "two-facet"   # one test, visible browser
```
`DB_URL` can name any database of the server: the script creates
`literev_smoke_<random>` next to it, applies `schema.sql`, boots the API on port 8765
(its startup DDL completes the schema), seeds a 40-article scenario with the seeder
of `scripts/bench_scenario.py`, serves `frontend/dist` with `vite preview` (port 4173,
`/api` proxied to the API) and drops the database at the end (`--keep-db` keeps it).
No OpenAI or Cohere key is passed, so the run is deterministic and offline. On
failure the API log is in `frontend/e2e-api.log`, Playwright's trace and screenshot
in `frontend/test-results/` (CI uploads both as the `playwright-report` artifact).

**Never point it at production**: it creates and drops databases on the server it
is given.

## 9. Before a demo or a presentation

The day before, and again an hour before, run the preflight against production (from
the server, or from a laptop with `--base https://literev-scenario.com/api`) on the
scenarios you will show, in the language you will use:
```bash
WRITE_API_KEY=… python3 scripts/preflight_demo.py --scenario usr-aaa --scenario usr-bbb --lang en
```
One line per check, `OK` / `WARN` / `FAIL`, exit code 1 on a FAIL:
- health: database, schema, full-text engine, memory and uptime, rate limits;
- OpenAI: one cheap real call (a search-strategy translation) proves the key works and
  has quota — without it the briefs, variables and actions do not generate;
- per scenario: the header numbers agree (the audit script runs inside), and every
  artefact a tab shows — clustering, knowledge graph, evidence brief, LLM brief,
  variables, recommended actions, model spec — is cached in the requested language.
  With the write key the missing ones are generated now and awaited (`--wait`,
  default 300 s); without it they are reported so you can open the tab once. Every
  read endpoint is timed (above `--slow-ms`, default 2000, is a WARN).

Then:
- **deploy freeze**: every merge to `main` restarts the API and cuts any search or
  pipeline in flight, so nothing merges from the morning of the session until it ends;
- **build in the language you will present in**: everything the search, the pin and
  the rebuild cache (cluster summaries, brief, variables and model, actions) is produced
  in the language of the toggle at that moment; a tab opened under the other toggle
  regenerates its text on the spot (LLM, 10-40 s). Switch the toggle first, then run the
  searches or the preflight with `--lang`;
- **warm the clustering** once after any restart (its first computation compiles
  UMAP, about 30 s) — the preflight warns when the API restarted recently;
- **prefer pre-built scenarios** in the session; a live search can take up to the
  federation budget (3 min) when a source is slow — keep a pre-built one as fallback;
- **a room sharing one public IP** (audience on the venue Wi-Fi) hits the per-IP
  limits: raise `RATE_LIMIT_GENERAL_PER_MIN` (600) and `RATE_LIMIT_EXPENSIVE_PER_MIN`
  (30: `/ask*`, scenario RAG, full pipeline) in `/etc/literev-api.env` for the day and
  restart the service; `/health` shows the values in force;
- hard-reload the browser once before presenting when a deploy happened since the
  last visit, and record a short screen capture of the flows as a network fallback.

