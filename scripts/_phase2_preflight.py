import os
from sqlalchemy import create_engine, text
e = create_engine(os.environ["DB_URL"])
def q(c, s): return c.execute(text(s)).fetchall()

DEL_CTE = """
WITH grp AS (
  SELECT id, doi, MIN(id) OVER (PARTITION BY doi) AS keep_id
  FROM literature_document
  WHERE doi IS NOT NULL AND doi IN (
    SELECT doi FROM literature_document WHERE doi IS NOT NULL GROUP BY doi HAVING count(*)>1)
),
del AS (SELECT id, keep_id FROM grp WHERE id <> keep_id)
"""

with e.connect() as c:
    print("== FK delete behavior (confdeltype: a=noaction r=restrict c=cascade n=setnull) ==")
    for r in q(c, """select conname, conrelid::regclass::text rel, confdeltype
                     from pg_constraint where contype='f'
                       and conrelid in ('article_scenarios'::regclass,'document_chunk'::regclass,'literature_document'::regclass)
                     order by rel"""):
        print(f"  {r[1]:22s} {r[0]:42s} confdeltype={r[2]}")

    print("== DELETE SET ==")
    print("  del rows:", q(c, DEL_CTE + "select count(*) from del")[0][0])
    print("  chunks on del:", q(c, DEL_CTE +
        "select count(*) from document_chunk dc where dc.document_id in (select id from del)")[0][0])
    print("  links on del:", q(c, DEL_CTE +
        "select count(*) from article_scenarios a where a.document_id in (select id from del)")[0][0])
    print("  links to REPOINT (keep,scenario not already present):", q(c, DEL_CTE + """
        select count(*) from article_scenarios a
        join del on del.id = a.document_id
        where not exists (select 1 from article_scenarios a2
                          where a2.document_id = del.keep_id and a2.scenario_id = a.scenario_id)""")[0][0])

    print("== POST-DELETE UNIQUENESS FEASIBILITY ==")
    print("  surviving DOIs with >1 row (must be 0 for UNIQUE doi):", q(c, DEL_CTE + """
        select count(*) from (
          select doi from literature_document
          where doi is not null and id not in (select id from del)
          group by doi having count(*)>1) x""")[0][0])
    print("  surviving PMIDs with >1 row (0 needed for UNIQUE pmid):", q(c, DEL_CTE + """
        select count(*) from (
          select pmid from literature_document
          where pmid is not null and id not in (select id from del)
          group by pmid having count(*)>1) x""")[0][0])

    print("== DANGLING REFERENCES AFTER DELETE ==")
    print("  survivors whose canonical_id points INTO del (would dangle self-FK):", q(c, DEL_CTE + """
        select count(*) from literature_document d
        where d.canonical_id in (select id from del) and d.id not in (select id from del)""")[0][0])
    print("== DONE ==")
