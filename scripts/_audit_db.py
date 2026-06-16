import os, sys, json
from sqlalchemy import create_engine, text

url = os.environ.get("DB_URL") or os.environ.get("DATABASE_URL")
if not url:
    print("NO DB_URL"); sys.exit(1)
eng = create_engine(url)

def q(conn, sql, params=None):
    return conn.execute(text(sql), params or {}).fetchall()

with eng.connect() as c:
    print("== VERSION =="); print(q(c,"select version()")[0][0])
    try:
        print("== EXTENSIONS ==");
        for r in q(c,"select extname, extversion from pg_extension order by 1"): print(" ",r[0],r[1])
    except Exception as e: print("ext err",e)

    print("== ALEMBIC_VERSION ==")
    try:
        print("  ", q(c,"select version_num from alembic_version"))
    except Exception as e: print("  none:",e)

    print("== TABLES (size, est rows) ==")
    rows = q(c,"""
      select c.relname,
             pg_size_pretty(pg_total_relation_size(c.oid)) sz,
             c.reltuples::bigint est
      from pg_class c join pg_namespace n on n.oid=c.relnamespace
      where n.nspname='public' and c.relkind='r' order by pg_total_relation_size(c.oid) desc""")
    tables=[r[0] for r in rows]
    for r in rows: print(f"  {r[0]:40s} {r[1]:>12s} est={r[2]}")

    print("== EXACT COUNTS ==")
    counts={}
    for t in tables:
        try:
            n=q(c,f'select count(*) from "{t}"')[0][0]; counts[t]=n; print(f"  {t:40s} {n}")
        except Exception as e: print(f"  {t} count err {e}")

    print("== COLUMNS ==")
    cols = q(c,"""
      select table_name, column_name, data_type, is_nullable, column_default, udt_name,
             character_maximum_length
      from information_schema.columns where table_schema='public'
      order by table_name, ordinal_position""")
    cur=None
    for r in cols:
        if r[0]!=cur: cur=r[0]; print(f"  --- {cur} ---")
        dflt=(str(r[4])[:40] if r[4] else "")
        print(f"     {r[1]:34s} {r[5]:14s} null={r[3]:3s} def={dflt}")

    print("== PRIMARY KEYS / UNIQUE / CHECK ==")
    for r in q(c,"""
      select tc.table_name, tc.constraint_type, tc.constraint_name,
             string_agg(kcu.column_name, ',' order by kcu.ordinal_position)
      from information_schema.table_constraints tc
      left join information_schema.key_column_usage kcu
        on tc.constraint_name=kcu.constraint_name and tc.table_schema=kcu.table_schema
      where tc.table_schema='public' and tc.constraint_type in ('PRIMARY KEY','UNIQUE','CHECK')
      group by 1,2,3 order by 1,2"""):
        print(f"  {r[0]:34s} {r[1]:12s} {r[3] or ''}  ({r[2]})")

    print("== FOREIGN KEYS ==")
    fks = q(c,"""
      select tc.table_name, kcu.column_name, ccu.table_name, ccu.column_name, tc.constraint_name
      from information_schema.table_constraints tc
      join information_schema.key_column_usage kcu on tc.constraint_name=kcu.constraint_name
      join information_schema.constraint_column_usage ccu on tc.constraint_name=ccu.constraint_name
      where tc.constraint_type='FOREIGN KEY' and tc.table_schema='public' order by 1""")
    if not fks: print("  (NONE)")
    for r in fks: print(f"  {r[0]}.{r[1]} -> {r[2]}.{r[3]} ({r[4]})")

    print("== INDEXES ==")
    for r in q(c,"select tablename, indexname, indexdef from pg_indexes where schemaname='public' order by tablename, indexname"):
        print(f"  {r[0]:30s} {r[1]:40s} {r[2][:120]}")

    print("== VECTOR COLUMNS ==")
    vc = q(c,"""select table_name, column_name from information_schema.columns
              where table_schema='public' and udt_name='vector' order by 1""")
    for r in vc:
        t,col=r[0],r[1]
        try:
            dim=q(c,f'select vector_dims("{col}") from "{t}" where "{col}" is not null limit 1')
            nn=q(c,f'select count(*) from "{t}" where "{col}" is not null')[0][0]
            tot=counts.get(t,"?")
            print(f"  {t}.{col} dim={dim[0][0] if dim else 'NA'} nonnull={nn}/{tot}")
        except Exception as e: print(f"  {t}.{col} err {e}")

    print("== NULL HOTSPOTS (pg_stats null_frac>0.5) ==")
    for r in q(c,"""select tablename, attname, round(null_frac::numeric,3), n_distinct
                   from pg_stats where schemaname='public' and null_frac>0.5
                   order by null_frac desc limit 60"""):
        print(f"  {r[0]}.{r[1]} null_frac={r[2]} n_distinct={r[3]}")

    print("== FK ORPHAN CHECK ==")
    for r in fks:
        ct,cc,pt,pc=r[0],r[1],r[2],r[3]
        try:
            o=q(c,f'select count(*) from "{ct}" ch left join "{pt}" pa on ch."{cc}"=pa."{pc}" where ch."{cc}" is not null and pa."{pc}" is null')[0][0]
            if o: print(f"  ORPHANS {ct}.{cc}->{pt}.{pc}: {o}")
        except Exception as e: print(f"  orphan check err {ct}.{cc}: {e}")
    print("  (done)")

    print("== DUP CHECK on candidate natural-key columns ==")
    cand = q(c,"""select table_name, column_name from information_schema.columns
       where table_schema='public' and (column_name in ('doi','pmid','pmcid','external_id','source_id','url')
         or column_name like '%_doi' or column_name like '%_pmid') order by 1""")
    for r in cand:
        t,col=r[0],r[1]
        try:
            d=q(c,f'select count(*) from (select "{col}" from "{t}" where "{col}" is not null group by "{col}" having count(*)>1) x')[0][0]
            print(f"  {t}.{col} dup_groups={d}")
        except Exception as e: print(f"  dup err {t}.{col}: {e}")
print("== DONE ==")
