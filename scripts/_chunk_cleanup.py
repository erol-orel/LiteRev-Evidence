"""Chunk-integrity cleanup (audit A3 + B3). Guarded, idempotent.

Dry-run by default — pass `--execute` to apply. Destructive deletes are backed
up to an in-DB table first (same convention as `_phase2_execute.py`).

Actions when `--execute`:
  1. [B3] De-dupe `title_abstract` chunks: per document keep ONE (prefer the row
     that already has an embedding, else lowest id), back up + delete the rest.
  2. Add partial unique index `uq_document_chunk_title_abstract`
     (document_id WHERE chunk_type='title_abstract') so the bug can't recur at
     the DB level — complements the app-level fix in PR #44.
  3. [A3a] Recover chunkless docs that have text: insert a `title_abstract`
     chunk (content = title + abstract, embedding left NULL). The enrichment
     worker then embeds it (main.py: "embède les chunks title_abstract sans
     embedding"). Idempotent via NOT EXISTS.
  4. [A3b] Truly-empty chunkless docs are only REPORTED, never deleted here.

Reads DB_URL / DATABASE_URL from the environment. Run via the audited
server-command workflow.

    python scripts/_chunk_cleanup.py            # dry-run (no writes)
    python scripts/_chunk_cleanup.py --execute  # apply
"""
import os
import sys
from datetime import datetime, timezone
from sqlalchemy import create_engine, text

url = os.environ.get("DB_URL") or os.environ.get("DATABASE_URL")
if not url:
    print("NO DB_URL"); sys.exit(1)

EXECUTE = "--execute" in sys.argv
eng = create_engine(url)

SCOPE = "ld.project_context = 'literev' AND ld.is_duplicate IS NOT TRUE"
BAK = f"_chunk_dedup_bak_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

print(f"== CHUNK CLEANUP ({'EXECUTE' if EXECUTE else 'DRY-RUN'}) ==")


def scalar(conn, sql, **p):
    return conn.execute(text(sql), p).scalar()


with eng.begin() as c:
    # ── 1. B3 : identifier les chunks title_abstract en trop (rn>1) ─────────
    to_delete = scalar(c, """
        SELECT count(*) FROM (
            SELECT id, row_number() OVER (
                PARTITION BY document_id
                ORDER BY (embedding IS NOT NULL) DESC, id ASC) AS rn
            FROM document_chunk WHERE chunk_type = 'title_abstract'
        ) t WHERE t.rn > 1
    """)
    print(f"[B3] extra title_abstract chunks to delete: {to_delete}")

    if EXECUTE and to_delete:
        c.execute(text(f"""
            CREATE TABLE {BAK} AS
            SELECT dc.* FROM document_chunk dc JOIN (
                SELECT id FROM (
                    SELECT id, row_number() OVER (
                        PARTITION BY document_id
                        ORDER BY (embedding IS NOT NULL) DESC, id ASC) AS rn
                    FROM document_chunk WHERE chunk_type = 'title_abstract'
                ) t WHERE t.rn > 1
            ) d ON dc.id = d.id
        """))
        deleted = c.execute(text(f"""
            DELETE FROM document_chunk dc USING (
                SELECT id FROM (
                    SELECT id, row_number() OVER (
                        PARTITION BY document_id
                        ORDER BY (embedding IS NOT NULL) DESC, id ASC) AS rn
                    FROM document_chunk WHERE chunk_type = 'title_abstract'
                ) t WHERE t.rn > 1
            ) d WHERE dc.id = d.id
        """)).rowcount
        print(f"     backed up to {BAK}, deleted {deleted} rows")

    # ── 3. A3a : recover chunkless docs ayant du texte ──────────────────────
    recoverable = scalar(c, f"""
        SELECT count(*) FROM literature_document ld
        WHERE {SCOPE}
          AND NOT EXISTS (SELECT 1 FROM document_chunk dc WHERE dc.document_id = ld.id)
          AND btrim(coalesce(ld.title, '') || ' ' || coalesce(ld.abstract, '')) <> ''
    """)
    print(f"[A3a] chunkless docs with text to recover: {recoverable}")

    if EXECUTE and recoverable:
        inserted = c.execute(text(f"""
            INSERT INTO document_chunk (document_id, chunk_index, content, chunk_type, created_at)
            SELECT ld.id, 0,
                   btrim(coalesce(ld.title, '') || E'\\n\\n' || coalesce(ld.abstract, '')),
                   'title_abstract', now()
            FROM literature_document ld
            WHERE {SCOPE}
              AND NOT EXISTS (SELECT 1 FROM document_chunk dc WHERE dc.document_id = ld.id)
              AND btrim(coalesce(ld.title, '') || ' ' || coalesce(ld.abstract, '')) <> ''
        """)).rowcount
        print(f"      inserted {inserted} recovery chunks (embedding NULL -> worker embeds)")

    # ── 4. A3b : empties — report only ──────────────────────────────────────
    empty = scalar(c, f"""
        SELECT count(*) FROM literature_document ld
        WHERE {SCOPE}
          AND NOT EXISTS (SELECT 1 FROM document_chunk dc WHERE dc.document_id = ld.id)
          AND btrim(coalesce(ld.title, '') || ' ' || coalesce(ld.abstract, '')) = ''
    """)
    print(f"[A3b] truly-empty chunkless docs (reported, NOT deleted): {empty}")

# ── 2. Index unique partiel (hors transaction ; après dé-dup) ──────────────
if EXECUTE:
    with eng.connect().execution_options(isolation_level="AUTOCOMMIT") as c:
        c.execute(text("""
            CREATE UNIQUE INDEX IF NOT EXISTS uq_document_chunk_title_abstract
            ON document_chunk (document_id) WHERE chunk_type = 'title_abstract'
        """))
        print("[B3] partial unique index uq_document_chunk_title_abstract ensured")
else:
    print("[B3] (dry-run) would ensure partial unique index uq_document_chunk_title_abstract")

print("== DONE ==" + ("" if EXECUTE else "  (dry-run — no writes)"))
