import os
from sqlalchemy import create_engine, text
e = create_engine(os.environ["DB_URL"])

DOI_SQL = """
WITH g AS (
  SELECT id, MIN(id) OVER (PARTITION BY doi) AS keep_id
  FROM literature_document
  WHERE doi IS NOT NULL
    AND doi IN (SELECT doi FROM literature_document
                WHERE doi IS NOT NULL GROUP BY doi HAVING count(*)>1)
)
UPDATE literature_document d
SET is_duplicate = TRUE, canonical_id = g.keep_id
FROM g
WHERE d.id = g.id AND g.id <> g.keep_id
"""

# PMID pass: only rows not already flagged, canonical = min id among not-yet-dup rows
PMID_SQL = """
WITH g AS (
  SELECT id, MIN(id) OVER (PARTITION BY pmid) AS keep_id
  FROM literature_document
  WHERE pmid IS NOT NULL AND is_duplicate IS NOT TRUE
    AND pmid IN (SELECT pmid FROM literature_document
                 WHERE pmid IS NOT NULL AND is_duplicate IS NOT TRUE
                 GROUP BY pmid HAVING count(*)>1)
)
UPDATE literature_document d
SET is_duplicate = TRUE, canonical_id = g.keep_id
FROM g
WHERE d.id = g.id AND g.id <> g.keep_id
"""

def scal(c, s): return c.execute(text(s)).scalar()

with e.connect() as c:
    print("BEFORE is_duplicate:", scal(c, "select count(*) from literature_document where is_duplicate"))
    print("BEFORE canonical_id:", scal(c, "select count(*) from literature_document where canonical_id is not null"))

with e.begin() as c:
    r1 = c.execute(text(DOI_SQL)).rowcount
    r2 = c.execute(text(PMID_SQL)).rowcount
    print("DOI rows flagged:", r1)
    print("PMID rows flagged:", r2)

with e.connect() as c:
    print("AFTER is_duplicate:", scal(c, "select count(*) from literature_document where is_duplicate"))
    print("AFTER canonical_id:", scal(c, "select count(*) from literature_document where canonical_id is not null"))
    # sanity: no canonical pointing at a duplicate, no self-canonical
    print("self-canonical (should be 0):",
          scal(c, "select count(*) from literature_document where canonical_id = id"))
    print("canonical-points-to-duplicate (informational):",
          scal(c, """select count(*) from literature_document d
                     join literature_document p on p.id=d.canonical_id
                     where d.canonical_id is not null and p.is_duplicate"""))
    print("remaining NON-dup DOI dup-groups (should be 0):",
          scal(c, """select count(*) from (select doi from literature_document
                     where doi is not null and is_duplicate is not true
                     group by doi having count(*)>1) x"""))
print("DONE")
