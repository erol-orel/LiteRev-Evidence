import os, time
from sqlalchemy import create_engine, text

url = os.environ["DB_URL"]
eng = create_engine(url)
with eng.connect() as c:
    c = c.execution_options(isolation_level="AUTOCOMMIT")
    c.execute(text("SET maintenance_work_mem='2GB'"))
    c.execute(text("SET statement_timeout=0"))
    print("START", time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    t0 = time.time()
    # CONCURRENTLY: no table lock on writes; cannot run in a txn (hence AUTOCOMMIT).
    c.execute(text(
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS document_chunk_embedding_hnsw "
        "ON document_chunk USING hnsw (embedding vector_cosine_ops)"
    ))
    print("CREATED in", round(time.time() - t0, 1), "s", flush=True)
    r = c.execute(text(
        "select indisvalid, pg_size_pretty(pg_relation_size('document_chunk_embedding_hnsw')) "
        "from pg_index where indexrelid='document_chunk_embedding_hnsw'::regclass"
    )).fetchall()
    print("VERIFY indisvalid,size:", r, flush=True)
    print("END", time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
