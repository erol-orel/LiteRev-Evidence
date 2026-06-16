import os
from sqlalchemy import create_engine, text
e = create_engine(os.environ["DB_URL"])

DEL_CTE = """
WITH grp AS (
  SELECT id, doi, MIN(id) OVER (PARTITION BY doi) AS keep_id
  FROM literature_document
  WHERE doi IS NOT NULL AND doi IN (
    SELECT doi FROM literature_document WHERE doi IS NOT NULL GROUP BY doi HAVING count(*)>1)
),
del AS (SELECT id, keep_id FROM grp WHERE id <> keep_id)
"""

with e.begin() as c:
    # Materialize delete set for reuse within this transaction
    c.execute(text("DROP TABLE IF EXISTS _del"))
    c.execute(text("CREATE TEMP TABLE _del AS " + DEL_CTE + " SELECT id, keep_id FROM del"))
    c.execute(text("CREATE INDEX ON _del(id)"))
    n_del = c.execute(text("SELECT count(*) FROM _del")).scalar()
    print("del set:", n_del)

    # 1. Backups (restorable safety net) — persistent tables
    for bak, src, col in [
        ("_dedup_bak_documents", "literature_document", "id"),
        ("_dedup_bak_chunks", "document_chunk", "document_id"),
        ("_dedup_bak_links", "article_scenarios", "document_id"),
    ]:
        c.execute(text(f"DROP TABLE IF EXISTS {bak}"))
        c.execute(text(f"CREATE TABLE {bak} AS SELECT * FROM {src} WHERE {col} IN (SELECT id FROM _del)"))
        print(f"backup {bak}:", c.execute(text(f"SELECT count(*) FROM {bak}")).scalar())

    # 2. Repoint the unique scenario memberships onto the canonical doc
    rep = c.execute(text("""
        INSERT INTO article_scenarios
            (document_id, scenario_id, similarity_score, assignment_method, assigned_at, is_primary, cluster_id, cluster_label)
        SELECT d.keep_id, a.scenario_id, a.similarity_score, a.assignment_method, a.assigned_at,
               a.is_primary, a.cluster_id, a.cluster_label
        FROM article_scenarios a JOIN _del d ON d.id = a.document_id
        ON CONFLICT (document_id, scenario_id) DO NOTHING
    """)).rowcount
    print("links repointed:", rep)

    # 3. Delete duplicates (CASCADE removes chunks+old links; SET NULL fixes canonical refs)
    dd = c.execute(text("DELETE FROM literature_document WHERE id IN (SELECT id FROM _del)")).rowcount
    print("docs deleted:", dd)

# 4. Add the DOI unique constraint (data now supports it). Skip PMID (residual dups).
with e.connect() as c:
    c = c.execution_options(isolation_level="AUTOCOMMIT")
    c.execute(text(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_literature_document_doi "
        "ON literature_document (doi) WHERE doi IS NOT NULL"))
    print("UNIQUE(doi) index created")
    print("--- verification ---")
    print("remaining dup-doi groups:", c.execute(text(
        "select count(*) from (select doi from literature_document where doi is not null group by doi having count(*)>1) x")).scalar())
    print("total docs:", c.execute(text("select count(*) from literature_document")).scalar())
    print("total chunks:", c.execute(text("select count(*) from document_chunk")).scalar())
    print("total links:", c.execute(text("select count(*) from article_scenarios")).scalar())
    print("is_duplicate remaining:", c.execute(text("select count(*) from literature_document where is_duplicate")).scalar())
    print("DONE")
