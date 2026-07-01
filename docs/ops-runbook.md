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
