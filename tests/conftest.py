"""Shared pytest setup for the LiteRev backend tests.

`main.py` reads DB_URL and WRITE_API_KEY at import time (and raises if missing),
but `create_engine(DB_URL)` is lazy — so importing `main` needs the env vars set
but NOT a reachable database (its startup DDL is wrapped in try/except). Pure
tests therefore run with a dummy DB_URL; integration tests use the `db_conn`
fixture (a raw psycopg connection), which skips cleanly when no Postgres is
reachable, so CI without a DB service still passes on the pure-logic tests.
"""
import os

# Must be set BEFORE `import main` anywhere in the session.
os.environ.setdefault("WRITE_API_KEY", "test-write-key")
os.environ.setdefault("DB_URL", "postgresql+psycopg://u:p@127.0.0.1:1/nodb")  # unreachable dummy
os.environ.setdefault("OPENAI_API_KEY", "")

import pytest


@pytest.fixture()
def db_conn():
    """Raw psycopg connection to DB_URL, or skip the integration tests.

    Uses psycopg directly (not SQLAlchemy) so the test DB layer is independent
    of driver/ORM version quirks; `main`'s own SQL under test is still exercised
    via the functions being tested.
    """
    import psycopg
    from sqlalchemy.engine import make_url  # pure URL parse, no connection

    url = make_url(os.environ["DB_URL"])
    kwargs = {
        "host": url.host or "localhost",
        "port": url.port or 5432,
        "dbname": url.database,
        "user": url.username,
        "password": url.password,
        "connect_timeout": 5,
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    try:
        conn = psycopg.connect(**kwargs)
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"No reachable Postgres at DB_URL ({e}); skipping integration tests.")
    conn.autocommit = True
    yield conn
    conn.close()
