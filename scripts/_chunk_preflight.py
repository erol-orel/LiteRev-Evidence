"""Read-only preflight for the chunk-integrity cleanup (audit A3 + B3).

Reports, without changing anything:
  - B3: documents with >1 `title_abstract` chunk (duplicate chunks);
  - A3a: chunkless documents that DO have title/abstract text (recoverable);
  - A3b: chunkless documents with NO usable text (truly empty — decision needed).

Reads DB_URL / DATABASE_URL from the environment. Run via the audited
server-command workflow, same as the other scripts in this folder.

    python scripts/_chunk_preflight.py
"""
import os
import sys
from sqlalchemy import create_engine, text

url = os.environ.get("DB_URL") or os.environ.get("DATABASE_URL")
if not url:
    print("NO DB_URL"); sys.exit(1)
eng = create_engine(url)

# Périmètre : corpus literev, hors doublons DOI déjà neutralisés.
SCOPE = "ld.project_context = 'literev' AND ld.is_duplicate IS NOT TRUE"


def one(conn, sql, **p):
    return conn.execute(text(sql), p).scalar()


with eng.connect() as c:
    print("== CHUNK INTEGRITY PREFLIGHT ==")

    total_docs = one(c, f"SELECT count(*) FROM literature_document ld WHERE {SCOPE}")
    print(f"  in-scope documents (literev, non-dup): {total_docs}")

    # ── B3 : doublons de chunk title_abstract ───────────────────────────────
    dup_docs = one(c, """
        SELECT count(*) FROM (
            SELECT document_id FROM document_chunk
            WHERE chunk_type = 'title_abstract'
            GROUP BY document_id HAVING count(*) > 1
        ) x
    """)
    dup_extra = one(c, """
        SELECT coalesce(sum(n - 1), 0) FROM (
            SELECT document_id, count(*) AS n FROM document_chunk
            WHERE chunk_type = 'title_abstract'
            GROUP BY document_id HAVING count(*) > 1
        ) x
    """)
    print(f"  [B3] docs with >1 title_abstract chunk: {dup_docs}  (extra chunks to delete: {dup_extra})")

    # ── A3 : documents sans aucun chunk ─────────────────────────────────────
    chunkless = one(c, f"""
        SELECT count(*) FROM literature_document ld
        WHERE {SCOPE}
          AND NOT EXISTS (SELECT 1 FROM document_chunk dc WHERE dc.document_id = ld.id)
    """)
    chunkless_text = one(c, f"""
        SELECT count(*) FROM literature_document ld
        WHERE {SCOPE}
          AND NOT EXISTS (SELECT 1 FROM document_chunk dc WHERE dc.document_id = ld.id)
          AND btrim(coalesce(ld.title, '') || ' ' || coalesce(ld.abstract, '')) <> ''
    """)
    chunkless_empty = chunkless - chunkless_text
    print(f"  [A3] chunkless docs (no chunk at all): {chunkless}")
    print(f"       - recoverable (have title/abstract text): {chunkless_text}")
    print(f"       - truly empty (no text -> decision needed): {chunkless_empty}")

    # Échantillon d'ids vides pour inspection humaine.
    if chunkless_empty:
        rows = c.execute(text(f"""
            SELECT ld.id, ld.doi, ld.source FROM literature_document ld
            WHERE {SCOPE}
              AND NOT EXISTS (SELECT 1 FROM document_chunk dc WHERE dc.document_id = ld.id)
              AND btrim(coalesce(ld.title, '') || ' ' || coalesce(ld.abstract, '')) = ''
            ORDER BY ld.id LIMIT 15
        """)).fetchall()
        print("       sample empty ids:", [r[0] for r in rows])

print("== DONE (read-only) ==")
