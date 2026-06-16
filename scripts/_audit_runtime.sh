set +e
# NOTE: never `systemctl cat` this unit — its drop-in override.conf stores secrets in
# Environment=. Use `systemctl show` and select only non-secret properties, and redact
# any Environment= values defensively.
echo "== UNIT STATE (no secrets) =="
systemctl show literev-api -p ExecStart -p Restart -p RestartSec -p User \
    -p EnvironmentFiles -p ActiveState -p NRestarts -p FragmentPath -p DropInPaths 2>&1 \
    | grep -ivE 'OPENAI|API_KEY|PASS|SECRET|TOKEN|DB_URL'
echo "== UVICORN WORKERS (cmdline) =="
ps -eo pid,cmd | grep -E 'uvicorn|gunicorn' | grep -v grep
echo "== SECRETS.ENV KEYS (names only) =="
if [ -f /opt/literev-api/secrets.env ]; then awk -F= '/^[A-Za-z]/{print "  "$1}' /opt/literev-api/secrets.env; ls -l /opt/literev-api/secrets.env; else echo "  (missing)"; fi
echo "== TIMERS / CRON =="
systemctl list-timers --all --no-pager 2>&1 | grep -iE 'literev|review|ingest|embed' || echo "  no matching systemd timers"
echo "  --- /etc/cron* ---"
grep -rIl . /etc/cron.d /etc/crontab /var/spool/cron 2>/dev/null | xargs -r grep -iE 'literev|review|ingest|embed|python' 2>/dev/null || echo "  no matching cron entries"
echo "== NGINX SITES =="
ls -l /etc/nginx/sites-enabled/ 2>&1
for f in /etc/nginx/sites-enabled/*; do echo "--- $f ---"; sed -n '1,120p' "$f"; done 2>&1
echo "== NGINX -t =="
nginx -t 2>&1
echo "== HEALTH ENDPOINT =="
curl -s -m 8 -o /dev/null -w "  /health -> %{http_code}\n" http://localhost:8000/health
curl -s -m 8 http://localhost:8000/health | head -c 300; echo
echo "== GIT STATE =="
cd /opt/literev-api && git rev-parse HEAD && git status --porcelain=v1 -b | head -20 && echo "  behind/ahead:" && git rev-list --left-right --count origin/main...HEAD 2>/dev/null
echo "== PIP vs requirements (mismatches only) =="
cd /opt/literev-api && .venv/bin/pip freeze 2>/dev/null > /tmp/_frozen.txt
while IFS= read -r line; do
  case "$line" in ''|\#*) continue;; esac
  pkg=$(echo "$line" | sed -E 's/[><=!~ ].*//' | tr 'A-Z' 'a-z')
  grep -iq "^${pkg}==" /tmp/_frozen.txt || echo "  MISSING/!=req: $line"
done < requirements.txt
echo "== DISK / MEM =="
df -h / /opt 2>/dev/null | sed -n '1,6p'; free -m | sed -n '1,3p'
echo "== RECENT ERRORS in journal (last 1500 lines, filtered) =="
journalctl -u literev-api --no-pager -n 1500 2>&1 | grep -iE 'error|traceback|exception|critical|failed|500 |502 |rollback|deadlock|timeout' | tail -40 || echo "  (none)"
echo "== DONE =="
