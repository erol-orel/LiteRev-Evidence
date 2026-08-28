"""A brand-new database must yield a WORKING app — not just a booting one.

This is the regression guard for a failure that reached production unnoticed: on a
genuinely fresh database, `_ensure_user_scenarios_table` ran every statement inside ONE
transaction, and its `ALTER TABLE article_scenarios ...` raised because NO file in the
repository ever created `article_scenarios` (not schema.sql, not the Alembic migrations —
which only ADD columns and skip themselves when the table is missing — and not the boot
DDL, which altered it directly). Postgres then aborted the whole transaction and rolled
back the `user_scenarios` tables created moments earlier in the same block.

The result was an app that answered `/health` with 200 while `/user-scenarios`,
`/gesica/scenarios` and `/corpus/fulltext-stats` all returned 500 — and the last of those
is the endpoint the production deploy smoke-tests, so a deploy onto a fresh database would
have failed its own smoke test while `/health` looked fine.

The test builds a database the way a new deployment does (schema.sql, then the app's own
startup DDL) and asserts the endpoints actually answer. It skips cleanly when no Postgres
is reachable, so CI without a database service still passes the pure tests.
"""
import os
import subprocess
import uuid

import pytest

pytest.importorskip("sqlalchemy")

# Endpoints a fresh deployment must serve. /corpus/fulltext-stats is in the production
# deploy smoke test (.github/workflows/deploy.yml), so it is the load-bearing one.
REQUIRED_OK = [
    "/health",
    "/user-scenarios",
    "/gesica/scenarios",
    "/corpus/stats",
    "/corpus/fulltext-stats",
    "/filters-options",
]

# Tables the application genuinely requires. article_scenarios is the one that was missing.
REQUIRED_TABLES = {
    "literature_document", "document_chunk", "article_scenarios",
    "user_scenarios", "user_scenario_folders", "scenario_settings",
}


def _admin_url():
    """A libpq URL for creating/dropping throwaway databases, or None to skip."""
    raw = os.getenv("DB_URL") or ""
    if not raw:
        return None
    from sqlalchemy.engine import make_url
    try:
        u = make_url(raw)
    except Exception:
        return None
    return u


@pytest.fixture()
def fresh_db():
    """Create a throwaway database, yield its SQLAlchemy URL, drop it afterwards."""
    import sqlalchemy as sa

    u = _admin_url()
    if u is None:
        pytest.skip("DB_URL not set — fresh-database bootstrap test needs Postgres")
    name = f"freshboot_{uuid.uuid4().hex[:10]}"
    admin = u.set(database="postgres")
    try:
        eng = sa.create_engine(admin, isolation_level="AUTOCOMMIT", pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(sa.text(f'CREATE DATABASE "{name}"'))
    except Exception as e:                                    # unreachable / no rights
        pytest.skip(f"cannot create a throwaway database: {e}")
    try:
        yield u.set(database=name)
    finally:
        try:
            with eng.connect() as c:
                c.execute(sa.text(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                    "WHERE datname = :n AND pid <> pg_backend_pid()"), {"n": name})
                c.execute(sa.text(f'DROP DATABASE IF EXISTS "{name}"'))
        except Exception:
            pass


def _apply_schema(url):
    """Apply schema.sql the way a new deployment would.

    pgvector may be absent in CI, and schema.sql declares `vector(1536)`; the app's own
    SEIR/corpus paths never touch embeddings, so the column type is rewritten to text
    when the extension cannot be created. That mirrors what a deploy without pgvector
    gets, which is the case under test.
    """
    import sqlalchemy as sa

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sql = open(os.path.join(root, "schema.sql"), encoding="utf-8").read()
    eng = sa.create_engine(url, isolation_level="AUTOCOMMIT")
    with eng.connect() as c:
        try:
            c.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))
        except Exception:
            sql = sql.replace("vector(1536)", "text")
        sql = "\n".join(l for l in sql.splitlines()
                        if "CREATE EXTENSION" not in l.upper() or "vector" not in l)
        for stmt in _split_statements(sql):
            try:
                c.execute(sa.text(stmt))
            except Exception:
                pass          # schema.sql is partly historical; the boot DDL completes it
    eng.dispose()


def _split_statements(sql: str) -> list:
    """Split on semicolons that are NOT inside a dollar-quoted body.

    schema.sql defines a PL/pgSQL trigger function whose body contains semicolons between
    `$$ ... $$`; a naive split on ";" would shred it into invalid fragments and quietly
    skip the function, so the test would be exercising a schema subtly unlike the one a
    real deployment gets.
    """
    out, buf, tag, i = [], [], None, 0
    while i < len(sql):
        if tag is None and sql[i] == "$":
            end = sql.find("$", i + 1)
            candidate = sql[i:end + 1] if end != -1 else None
            if candidate and (candidate[1:-1] == "" or candidate[1:-1].isidentifier()):
                tag = candidate
                buf.append(candidate)
                i = end + 1
                continue
        elif tag is not None and sql.startswith(tag, i):
            buf.append(tag)
            i += len(tag)
            tag = None
            continue
        if tag is None and sql[i] == ";":
            stmt = "".join(buf).strip()
            if stmt:
                out.append(stmt)
            buf = []
        else:
            buf.append(sql[i])
        i += 1
    tail = "".join(buf).strip()
    if tail:
        out.append(tail)
    return out


def test_fresh_database_serves_every_required_endpoint(fresh_db):
    """The whole point: build from scratch, boot the real app, hit the real routes."""
    import sqlalchemy as sa
    from fastapi.testclient import TestClient

    _apply_schema(fresh_db)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # main.py runs its startup DDL at IMPORT time, so it must be imported in a
    # subprocess bound to the fresh database — importing it here would bind to the
    # session's own DB_URL and prove nothing.
    script = r"""
import json, os, sys
sys.path.insert(0, os.environ["REPO_ROOT"])
import main
from fastapi.testclient import TestClient
# raise_server_exceptions=False so a 500 is RECORDED rather than re-raised: the point of
# this test is to report WHICH endpoints a fresh database cannot serve, not to die on the
# first one and hide the rest behind a traceback.
cl = TestClient(main.app, raise_server_exceptions=False)
out = {}
for p in json.loads(os.environ["PATHS"]):
    try:
        out[p] = cl.get(p).status_code
    except Exception as e:
        out[p] = f"raised {type(e).__name__}"
with main.engine.begin() as c:
    from sqlalchemy import text
    out["_tables"] = sorted(r[0] for r in c.execute(text(
        "SELECT tablename FROM pg_tables WHERE schemaname='public'")))
print("RESULT " + json.dumps(out))
"""
    env = {**os.environ,
           "REPO_ROOT": root,
           "PATHS": __import__("json").dumps(REQUIRED_OK),
           # render_as_string(hide_password=False): str(URL) masks the password as
           # "***", so the subprocess would authenticate with a literal "***". Invisible
           # on a trust-auth local cluster, fatal on CI's password-authenticated one.
           "DB_URL": fresh_db.render_as_string(hide_password=False),
           "WRITE_API_KEY": "test-write-key",
           "ADMIN_API_KEY": "test-admin-key",
           "OPENAI_API_KEY": ""}
    proc = subprocess.run([os.sys.executable, "-c", script], env=env,
                          capture_output=True, text=True, timeout=300)
    line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT ")), None)
    assert line, f"app failed to boot on a fresh database:\nSTDOUT{proc.stdout[-2000:]}\nSTDERR{proc.stderr[-3000:]}"
    got = __import__("json").loads(line[len("RESULT "):])

    missing = REQUIRED_TABLES - set(got.pop("_tables"))
    assert not missing, f"a fresh database is missing required tables: {sorted(missing)}"

    bad = {p: c for p, c in got.items() if c != 200}
    assert not bad, (
        f"a fresh database does not serve these endpoints: {bad}. "
        "This is the failure mode where /health returns 200 while the app is unusable — "
        "and /corpus/fulltext-stats is what the production deploy smoke-tests.")


def test_ddl_helper_isolates_failures(fresh_db):
    """One failing statement must not roll back the ones that succeeded before it."""
    import sqlalchemy as sa

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = r"""
import os, sys, json
sys.path.insert(0, os.environ["REPO_ROOT"])
import main
from sqlalchemy import text
failed = main._exec_ddl_isolated([
    "CREATE TABLE ddl_probe_a (id int)",
    "ALTER TABLE table_that_does_not_exist ADD COLUMN x int",   # must not undo the CREATE
    "CREATE TABLE ddl_probe_b (id int)",
], "probe")
with main.engine.begin() as c:
    tables = sorted(r[0] for r in c.execute(text(
        "SELECT tablename FROM pg_tables WHERE schemaname='public' "
        "AND tablename LIKE 'ddl_probe%'")))
print("RESULT " + json.dumps({"failed": len(failed), "tables": tables}))
"""
    env = {**os.environ, "REPO_ROOT": root,
           "DB_URL": fresh_db.render_as_string(hide_password=False),
           "WRITE_API_KEY": "k", "ADMIN_API_KEY": "k", "OPENAI_API_KEY": ""}
    proc = subprocess.run([os.sys.executable, "-c", script], env=env,
                          capture_output=True, text=True, timeout=300)
    line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT ")), None)
    assert line, f"probe failed:\n{proc.stdout[-1500:]}\n{proc.stderr[-2000:]}"
    got = __import__("json").loads(line[len("RESULT "):])
    assert got["failed"] == 1                       # exactly the bad statement
    assert got["tables"] == ["ddl_probe_a", "ddl_probe_b"]   # both survived
