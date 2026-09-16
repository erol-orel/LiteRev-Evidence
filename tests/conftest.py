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
# No background side effects in the tests: no UMAP warm-up at startup, no automatic
# full pipeline after a corpus build (the tests stub or drive the workers themselves).
os.environ.setdefault("WARM_ON_STARTUP", "0")
os.environ.setdefault("AUTO_PIPELINE_AFTER_SEARCH", "0")

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


def ensure_document_columns(cur) -> bool:
    """Give a suite-bootstrapped database the document/link columns production has.

    On CI pgvector is unavailable, so schema.sql is not applied and the documents
    table only carries the base columns; the boot DDL that adds the bibliographic,
    screening and dedup columns ran before the table existed. Endpoints that read
    those columns (corpus, relevant articles…) need them. Returns True when a minimal
    `document_chunk` table had to be created (the caller drops it at teardown)."""
    import main

    cur.execute("SELECT to_regclass('document_chunk') IS NULL")
    created_chunk_table = bool(cur.fetchone()[0])
    if created_chunk_table:
        cur.execute("""
            CREATE TABLE document_chunk (
                id BIGSERIAL PRIMARY KEY, document_id BIGINT NOT NULL,
                chunk_index INTEGER NOT NULL DEFAULT 0, content TEXT NOT NULL DEFAULT '',
                chunk_type TEXT, created_at TIMESTAMP DEFAULT now())""")
    for ensure_name in ("_ensure_bibliographic_columns", "_ensure_double_blind_columns", "_ensure_dedup_columns"):
        ensure = getattr(main, ensure_name, None)
        if ensure:
            ensure()
    for col, typ in (("created_at", "TIMESTAMP DEFAULT now()"), ("url", "TEXT"), ("pmid", "TEXT"),
                     ("year", "INTEGER"), ("source", "TEXT"), ("keywords", "TEXT"), ("language", "TEXT"),
                     ("open_access", "BOOLEAN"), ("sample_size", "INTEGER"), ("pico_json", "JSONB"),
                     ("metadata_json", "JSONB"), ("concepts_json", "JSONB"), ("country", "TEXT"),
                     ("study_design", "TEXT"), ("quality_score", "FLOAT"),
                     ("project_context", "VARCHAR(32) DEFAULT 'literev'")):
        cur.execute(f"ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS {col} {typ}")
    for col, typ in (("rerank_score", "FLOAT"), ("screening_status", "TEXT"), ("reviewer_1_status", "VARCHAR(20)")):
        cur.execute(f"ALTER TABLE article_scenarios ADD COLUMN IF NOT EXISTS {col} {typ}")
    return created_chunk_table


def patch_app(monkeypatch, name: str, value) -> None:
    """Patch `name` on `main` and on every module of the `api` package that carries it.

    The API used to be one module, where patching `main.X` reached every caller. The
    domain modules each bind the names they import (`from .search import X`), so a
    test has to patch the binding the code under test actually reads; patching every
    module that has the name restores the old single-namespace behaviour (the lazy
    imports inside functions read the defining module at call time, so they follow)."""
    import importlib

    import api
    import main

    targets = [main] + [importlib.import_module(f"api.{m}") for m in api.MODULES]
    found = False
    for mod in targets:
        if hasattr(mod, name):
            monkeypatch.setattr(mod, name, value)
            found = True
    if not found:
        raise AttributeError(f"{name} is not defined in the API")
