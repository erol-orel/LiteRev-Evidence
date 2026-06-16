import os
from sqlalchemy import create_engine, text
e = create_engine(os.environ["DB_URL"])
def q(c,s): return c.execute(text(s)).fetchall()
with e.connect() as c:
    print("total docs:", q(c,"select count(*) from literature_document")[0][0])
    print("with doi:", q(c,"select count(*) from literature_document where doi is not null")[0][0])
    print("with pmid:", q(c,"select count(*) from literature_document where pmid is not null")[0][0])
    print("already is_duplicate=true:", q(c,"select count(*) from literature_document where is_duplicate")[0][0])
    print("with canonical_id set:", q(c,"select count(*) from literature_document where canonical_id is not null")[0][0])
    # extra rows = rows beyond the first per dup group
    for col in ("doi","pmid"):
        r=q(c,f"""select coalesce(sum(cnt-1),0), count(*) from
                  (select {col}, count(*) cnt from literature_document
                   where {col} is not null group by {col} having count(*)>1) x""")
        print(f"{col}: dup_groups={r[0][1]} extra_rows(removable)={r[0][0]}")
    # canonical = min(id) per doi group; how many NON-canonical dup rows carry data?
    print("--- impact if canonical = MIN(id) per doi group ---")
    base="""
      with g as (select id, doi, min(id) over (partition by doi) keep_id
                 from literature_document
                 where doi is not null and doi in
                   (select doi from literature_document where doi is not null
                    group by doi having count(*)>1)),
      noncanon as (select id from g where id<>keep_id)"""
    print("noncanon doi rows:", q(c,base+" select count(*) from noncanon")[0][0])
    print("  of which have chunks:", q(c,base+
        " select count(distinct n.id) from noncanon n join document_chunk dc on dc.document_id=n.id")[0][0])
    print("  chunks attached to them:", q(c,base+
        " select count(*) from noncanon n join document_chunk dc on dc.document_id=n.id")[0][0])
    print("  with article_scenarios links:", q(c,base+
        " select count(distinct n.id) from noncanon n join article_scenarios a on a.document_id=n.id")[0][0])
    print("  article_scenarios links on them:", q(c,base+
        " select count(*) from noncanon n join article_scenarios a on a.document_id=n.id")[0][0])
    # how many of those scenario links would collide with canonical (already linked)?
    print("  links whose (canonical,scenario) already exists (would merge):", q(c,base+"""
        select count(*) from noncanon n
        join article_scenarios a on a.document_id=n.id
        join g gg on gg.id=n.id
        join article_scenarios a2 on a2.document_id=gg.keep_id and a2.scenario_id=a.scenario_id""")[0][0])
    print("DONE")
