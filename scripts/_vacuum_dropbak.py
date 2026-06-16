import os, time
from sqlalchemy import create_engine, text
e = create_engine(os.environ["DB_URL"])
TABLES = ("literature_document", "document_chunk", "article_scenarios")
with e.connect() as c:
    c = c.execution_options(isolation_level="AUTOCOMMIT")
    print("START", time.strftime("%H:%M:%S"), flush=True)
    print("before sizes:", flush=True)
    for t in TABLES:
        print("  ", t, c.execute(text(f"select pg_size_pretty(pg_total_relation_size('{t}'))")).scalar(), flush=True)
    # 3. drop Phase 2 backup tables (irreversible after this)
    for b in ("_dedup_bak_documents", "_dedup_bak_chunks", "_dedup_bak_links"):
        c.execute(text(f"DROP TABLE IF EXISTS {b}"))
        print("dropped", b, flush=True)
    # 2. VACUUM ANALYZE (reclaim dead tuples to reusable + refresh planner stats)
    for t in TABLES:
        t0 = time.time()
        c.execute(text(f"VACUUM (ANALYZE) {t}"))
        print(f"vacuumed {t} in {round(time.time()-t0,1)}s", flush=True)
    print("after sizes:", flush=True)
    for t in TABLES:
        print("  ", t, c.execute(text(f"select pg_size_pretty(pg_total_relation_size('{t}'))")).scalar(), flush=True)
    print("backups remaining (should be 0):",
          c.execute(text("select count(*) from information_schema.tables where table_name like '_dedup_bak%'")).scalar(), flush=True)
print("DONE", time.strftime("%H:%M:%S"), flush=True)
