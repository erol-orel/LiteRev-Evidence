"""migration 2 phase 1: per-scenario screening columns + backfill

Revision ID: c8d4e2f1a9b3
Revises: a7c3e1b9d2f4
Create Date: 2026-06-30

Migration 2 (docs/screening-status-per-scenario-migration.md) moves PRISMA
screening decisions from global per-document to per-(scenario, document).

Phase 1 is additive and behaviour-preserving: it adds the screening columns to
``article_scenarios`` and backfills each link from the document's current global
``screening_status`` (option A: copy the global decision onto all of a doc's
scenario links). Nothing reads these columns yet -- the global
``literature_document.screening_status`` stays authoritative -- so there is zero
behaviour change. Fully reversible (downgrade drops the columns).

Idempotent: ``ADD COLUMN IF NOT EXISTS`` + ``IS DISTINCT FROM`` backfill +
``CREATE INDEX IF NOT EXISTS``. Validated end-to-end on local Postgres 16.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "c8d4e2f1a9b3"
down_revision: Union[str, Sequence[str], None] = "a7c3e1b9d2f4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


ADD_COLUMNS_SQL = """
ALTER TABLE article_scenarios
  ADD COLUMN IF NOT EXISTS screening_status TEXT,
  ADD COLUMN IF NOT EXISTS screening_reason TEXT,
  ADD COLUMN IF NOT EXISTS screening_notes  TEXT,
  ADD COLUMN IF NOT EXISTS screened_at      TIMESTAMP
"""

BACKFILL_SQL = """
UPDATE article_scenarios ars
SET screening_status = d.screening_status,
    screening_reason = d.screening_reason,
    screening_notes  = d.screening_notes
FROM literature_document d
WHERE d.id = ars.document_id
  AND ars.screening_status IS DISTINCT FROM d.screening_status
"""

INDEX_SQL = """
CREATE INDEX IF NOT EXISTS ix_article_scenarios_scen_screen
  ON article_scenarios (scenario_id, screening_status)
"""


def upgrade() -> None:
    bind = op.get_bind()
    tables = set(sa.inspect(bind).get_table_names())
    if not {"article_scenarios", "literature_document"} <= tables:
        print("[migration c8d4e2f1a9b3] article_scenarios/literature_document "
              "absent -- skipping (will apply once tables exist)")
        return
    op.execute(ADD_COLUMNS_SQL)
    result = bind.execute(sa.text(BACKFILL_SQL))
    op.execute(INDEX_SQL)
    print(f"[migration c8d4e2f1a9b3] added per-scenario screening columns; "
          f"backfilled {result.rowcount} link(s) from the global column")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_article_scenarios_scen_screen")
    op.execute(
        "ALTER TABLE article_scenarios "
        "DROP COLUMN IF EXISTS screening_status, "
        "DROP COLUMN IF EXISTS screening_reason, "
        "DROP COLUMN IF EXISTS screening_notes, "
        "DROP COLUMN IF EXISTS screened_at"
    )
