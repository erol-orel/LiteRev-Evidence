"""migration 3: version-control the ingest-dedup UNIQUE indexes

Revision ID: d1e5f7a2b4c6
Revises: c8d4e2f1a9b3
Create Date: 2026-07-27

Makes ``_ingest_doc_direct``'s ``ON CONFLICT DO NOTHING`` race-safe by
version-controlling the two partial UNIQUE indexes it relies on:

  * ``uq_literature_document_doi`` -- already present in production, but created
    only by an ad-hoc script (``scripts/_phase2_execute.py``), never a migration
    (PIPELINE_AUDIT "A1 [High]"). Re-declaring it here (``IF NOT EXISTS`` -> a
    no-op on prod) guarantees ``ON CONFLICT (doi)`` keeps a target after any
    rebuild-from-migrations.
  * ``uq_litdoc_title_norm`` -- a NEW partial unique index on
    ``(project_context, title_norm)`` restricted to CANONICAL rows with a
    normalized title of length >= 20, matching ``_ingest_doc_direct``'s own
    dedup rule. This closes the parallel-fetcher race for DOI-less papers
    (preprints, trials) fetched from two sources at once.

Safety: the title_norm index is created inside a SAVEPOINT. On an existing
corpus that still holds *canonical* title_norm collisions the CREATE fails; we
roll back the savepoint, log, and continue rather than abort the migration /
deploy. A controlled, backed-up dedup (``scripts/_phase2_execute.py``-style)
clears the collisions, after which a later deploy's idempotent re-run creates it.
The startup DDL in ``main.py`` (``_ensure_performance_indexes``) mirrors both
creations belt-and-suspenders.

Idempotent (``CREATE UNIQUE INDEX IF NOT EXISTS`` + table guard). Reversible
(downgrade drops ``uq_litdoc_title_norm``; ``uq_literature_document_doi`` is left
in place because production depends on it for ``ON CONFLICT (doi)``).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "d1e5f7a2b4c6"
down_revision: Union[str, Sequence[str], None] = "c8d4e2f1a9b3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


DOI_INDEX_SQL = """
CREATE UNIQUE INDEX IF NOT EXISTS uq_literature_document_doi
  ON literature_document (doi) WHERE doi IS NOT NULL
"""

TITLE_NORM_INDEX_SQL = """
CREATE UNIQUE INDEX IF NOT EXISTS uq_litdoc_title_norm
  ON literature_document (project_context, title_norm)
  WHERE title_norm IS NOT NULL AND length(title_norm) >= 20
    AND (is_duplicate IS NULL OR is_duplicate = FALSE)
"""


def upgrade() -> None:
    bind = op.get_bind()
    tables = set(sa.inspect(bind).get_table_names())
    if "literature_document" not in tables:
        print("[migration d1e5f7a2b4c6] literature_document absent -- skipping "
              "(startup DDL creates the indexes once the table exists)")
        return
    # DOI: no-op where it already exists; creates it on a fresh/rebuilt DB so the
    # ingest path's ON CONFLICT always has a matching unique index.
    op.execute(DOI_INDEX_SQL)
    # title_norm: tolerant -- an existing corpus may still hold canonical
    # collisions that block a UNIQUE index. Contain the failure in a SAVEPOINT so
    # the migration (and the deploy) proceed; a later idempotent re-run succeeds
    # once maintenance has purged the duplicates.
    try:
        with bind.begin_nested():
            op.execute(TITLE_NORM_INDEX_SQL)
        print("[migration d1e5f7a2b4c6] uq_litdoc_title_norm ensured")
    except Exception as e:  # best-effort; logged, never fatal to the deploy
        print(f"[migration d1e5f7a2b4c6] uq_litdoc_title_norm skipped "
              f"(canonical title_norm collisions still present?): {e}")


def downgrade() -> None:
    # Leave uq_literature_document_doi in place: production ON CONFLICT (doi)
    # depends on it, and it predates this migration.
    op.execute("DROP INDEX IF EXISTS uq_litdoc_title_norm")
