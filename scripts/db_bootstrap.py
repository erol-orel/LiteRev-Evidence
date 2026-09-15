"""Build a database the way a new deployment does: schema.sql, then the app's own
startup DDL (which runs when `main` is imported or the API boots).

Shared by the fresh-database test (tests/test_fresh_db_bootstrap.py) and the browser
smoke test runner (scripts/smoke_e2e.py). Pure helpers: no side effect at import.
"""
from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def split_statements(sql: str) -> list[str]:
    """Split on semicolons that are NOT inside a dollar-quoted body.

    schema.sql defines a PL/pgSQL trigger function whose body contains semicolons between
    `$$ ... $$`; a naive split on ";" would shred it into invalid fragments and quietly
    skip the function, so a test would be exercising a schema subtly unlike the one a
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


def apply_schema(url) -> bool:
    """Apply schema.sql to the database at `url` (a SQLAlchemy URL or string).

    pgvector may be absent (CI's plain Postgres image), and schema.sql declares
    `vector(1536)`; nothing in the corpus, scenario or SEIR paths needs the embeddings
    column type, so it is rewritten to text when the extension cannot be created. That
    mirrors what a deploy without pgvector gets. Statements schema.sql cannot apply on
    an empty database (it is partly historical) are skipped: the boot DDL completes the
    schema. Returns True when pgvector is available.
    """
    import sqlalchemy as sa

    sql = open(os.path.join(ROOT, "schema.sql"), encoding="utf-8").read()
    eng = sa.create_engine(url, isolation_level="AUTOCOMMIT")
    has_vector = True
    with eng.connect() as c:
        try:
            c.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))
        except Exception:
            has_vector = False
            sql = sql.replace("vector(1536)", "text")
        sql = "\n".join(l for l in sql.splitlines()
                        if "CREATE EXTENSION" not in l.upper() or "vector" not in l)
        for stmt in split_statements(sql):
            try:
                c.execute(sa.text(stmt))
            except Exception:
                pass
    eng.dispose()
    return has_vector
