#!/usr/bin/env python3
"""Run the browser smoke test (frontend/e2e, Playwright) end to end, locally or in CI.

The test drives the real interface against a real API. This script:

  1. creates a throwaway database on the Postgres server of --db-url and applies
     schema.sql the way a new deployment does (scripts/db_bootstrap.py);
  2. boots the API on it (uvicorn on --api-port); its startup DDL completes the schema;
  3. seeds one synthetic scenario (--articles articles with abstracts, scored links and
     a cached clustering) with the seeder of scripts/bench_scenario.py;
  4. builds the frontend (unless --no-build) and runs `npx playwright test` in
     frontend/, whose configuration serves dist/ with `vite preview` and proxies /api
     to the API of step 2;
  5. stops the API and drops the database (unless --keep-db).

No OpenAI or Cohere key is passed to the API, so the LLM generators stay off and the
search runs on the local corpus only: the run is deterministic and works offline.

Usage:
  DB_URL=postgresql+psycopg://postgres:postgres@127.0.0.1:5432/postgres \\
      python3 scripts/smoke_e2e.py [--no-build] [--keep-db] [-- <playwright args>]

  # one test, headed, on an API already built:
  python3 scripts/smoke_e2e.py --no-build -- --headed -g "two-facet"

Needs: the backend dependencies plus uvicorn, Node with the frontend dev dependencies
(npm ci in frontend/) and a Playwright Chromium (npx playwright install --with-deps
chromium in frontend/). The exit code is Playwright's (0 = every test passed). The
API log of the run is written to frontend/e2e-api.log.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.request
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND = os.path.join(ROOT, "frontend")
API_LOG = os.path.join(FRONTEND, "e2e-api.log")
SCENARIO_NAME = "Smoke corpus"
SCENARIO_QUERY = "influenza AND surveillance"        # the query bench_scenario.seed stores
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def _log(msg: str) -> None:
    print(f"[smoke] {msg}", file=sys.stderr, flush=True)


def _wait_for_health(base: str, timeout: float, proc: subprocess.Popen) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"the API exited with code {proc.returncode} before answering /health")
        try:
            with urllib.request.urlopen(f"{base}/health", timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError(f"the API did not answer /health within {timeout:.0f} s")


def _tail(path: str, n: int = 60) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db-url", default=os.environ.get("DB_URL"),
                    help="Postgres URL (any database on the server; the throwaway one is created next to it)")
    ap.add_argument("--db-name", default=None, help="name of the throwaway database (default: literev_smoke_<random>)")
    ap.add_argument("--api-port", type=int, default=8765)
    ap.add_argument("--web-port", type=int, default=4173)
    ap.add_argument("--scenario", default="usr-smoke", help="id of the seeded scenario")
    ap.add_argument("--articles", type=int, default=40)
    ap.add_argument("--api-key", default="smoke-key", help="WRITE_API_KEY of the API; the browser test sets it too")
    ap.add_argument("--no-build", action="store_true", help="serve the existing frontend/dist")
    ap.add_argument("--keep-db", action="store_true", help="keep the database at the end (to inspect it)")
    ap.add_argument("playwright_args", nargs=argparse.REMAINDER, help="arguments after -- go to `playwright test`")
    args = ap.parse_args()
    if not args.db_url:
        ap.error("--db-url or DB_URL is required")

    import sqlalchemy as sa
    from sqlalchemy.engine import make_url
    import db_bootstrap

    url = make_url(args.db_url)
    db_name = args.db_name or f"literev_smoke_{uuid.uuid4().hex[:8]}"
    smoke_url = url.set(database=db_name)
    # render_as_string(hide_password=False): str(URL) masks the password as "***".
    smoke_url_str = smoke_url.render_as_string(hide_password=False)
    admin = sa.create_engine(url.set(database="postgres"), isolation_level="AUTOCOMMIT", pool_pre_ping=True)

    def drop_db() -> None:
        with admin.connect() as c:
            c.execute(sa.text("SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                              "WHERE datname = :n AND pid <> pg_backend_pid()"), {"n": db_name})
            c.execute(sa.text(f'DROP DATABASE IF EXISTS "{db_name}"'))

    _log(f"creating database {db_name}")
    drop_db()
    with admin.connect() as c:
        c.execute(sa.text(f'CREATE DATABASE "{db_name}"'))

    api: subprocess.Popen | None = None
    api_log = None
    rc = 1
    try:
        has_vector = db_bootstrap.apply_schema(smoke_url)
        _log(f"schema.sql applied ({'with' if has_vector else 'without'} pgvector)")

        env = {**os.environ, "DB_URL": smoke_url_str, "WRITE_API_KEY": args.api_key,
               "ADMIN_API_KEY": args.api_key, "OPENAI_API_KEY": "", "COHERE_API_KEY": "",
               "PYTHONUNBUFFERED": "1"}
        api_log = open(API_LOG, "w", encoding="utf-8")
        api = subprocess.Popen([sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1",
                                "--port", str(args.api_port), "--log-level", "info"],
                               cwd=ROOT, env=env, stdout=api_log, stderr=subprocess.STDOUT)
        base = f"http://127.0.0.1:{args.api_port}"
        _wait_for_health(base, 180, api)
        _log(f"API up on {base} (log: {API_LOG})")

        import bench_scenario
        eng = sa.create_engine(smoke_url)
        bench_scenario.seed(eng, args.articles, args.scenario, name=SCENARIO_NAME)
        eng.dispose()
        _log(f"seeded {args.articles} articles into {args.scenario}")

        if not args.no_build:
            _log("building the frontend")
            subprocess.run(["npm", "run", "build"], cwd=FRONTEND, check=True)

        pw_env = {**os.environ,
                  "E2E_SCENARIO_ID": args.scenario, "E2E_SCENARIO_NAME": SCENARIO_NAME,
                  "E2E_SCENARIO_QUERY": SCENARIO_QUERY, "E2E_ARTICLES": str(args.articles),
                  "E2E_API_KEY": args.api_key, "E2E_WEB_PORT": str(args.web_port),
                  "VITE_API_PROXY_TARGET": base}
        extra = [a for a in args.playwright_args if a != "--"]
        _log("running playwright test " + " ".join(extra))
        rc = subprocess.run(["npx", "playwright", "test", *extra], cwd=FRONTEND, env=pw_env).returncode
    except Exception as e:                                  # noqa: BLE001
        _log(f"failed: {e}")
        rc = 1
    finally:
        if api is not None:
            api.terminate()
            try:
                api.wait(15)
            except subprocess.TimeoutExpired:
                api.kill()
        if api_log is not None:
            api_log.close()
        if rc != 0:
            _log(f"last lines of the API log ({API_LOG}):\n" + _tail(API_LOG))
        if args.keep_db:
            _log(f"database kept: {smoke_url_str}")
        else:
            drop_db()
            _log(f"database {db_name} dropped")
        admin.dispose()
    return rc


if __name__ == "__main__":
    sys.exit(main())
