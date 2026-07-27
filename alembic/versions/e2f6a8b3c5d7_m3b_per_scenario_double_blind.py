"""migration 3b: per-scenario double-blind reviewer columns + backfill

Revision ID: e2f6a8b3c5d7
Revises: d1e5f7a2b4c6
Create Date: 2026-07-27

Cohen's kappa / double-blind screening decisions (``reviewer_1_status``,
``reviewer_1_reason``, ``reviewer_2_status``, ``reviewer_2_reason``,
``kappa_resolved``, ``kappa_final_status``) were stored as GLOBAL columns on
``literature_document``, but a document can belong to multiple scenarios
(``article_scenarios`` is many-to-many). Two scenarios screening the same
document therefore contaminated each other's kappa / conflict counts.

This mirrors Migration 2 (``c8d4e2f1a9b3``), which already moved the single /
final ``screening_status`` onto ``article_scenarios``; here we do the same for
the six per-reviewer double-blind columns.

Phase 1 (this migration) is additive + behaviour-preserving: add the columns to
``article_scenarios`` and backfill each scenario link from the document's current
global decision (copy the global verdict onto all of a doc's links, so every
scenario keeps showing exactly what it showed before). The reader endpoints
switch to the per-scenario columns in the same release; the global columns stay
as a legacy / corpus-badge fallback (still dual-written) until a later cutover.

Idempotent (``ADD COLUMN IF NOT EXISTS`` + ``IS DISTINCT FROM`` backfill + table
guard). Reversible (downgrade drops the six columns).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "e2f6a8b3c5d7"
down_revision: Union[str, Sequence[str], None] = "d1e5f7a2b4c6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


ADD_COLUMNS_SQL = """
ALTER TABLE article_scenarios
  ADD COLUMN IF NOT EXISTS reviewer_1_status  VARCHAR(20),
  ADD COLUMN IF NOT EXISTS reviewer_1_reason  TEXT,
  ADD COLUMN IF NOT EXISTS reviewer_2_status  VARCHAR(20),
  ADD COLUMN IF NOT EXISTS reviewer_2_reason  TEXT,
  ADD COLUMN IF NOT EXISTS kappa_resolved     BOOLEAN DEFAULT FALSE,
  ADD COLUMN IF NOT EXISTS kappa_final_status VARCHAR(20)
"""

# Copy the current global verdict onto every one of a doc's scenario links, so
# each scenario keeps showing what it showed pre-migration; decisions diverge
# per-scenario only going forward. IS DISTINCT FROM keeps re-runs idempotent.
BACKFILL_SQL = """
UPDATE article_scenarios ars
SET reviewer_1_status  = d.reviewer_1_status,
    reviewer_1_reason  = d.reviewer_1_reason,
    reviewer_2_status  = d.reviewer_2_status,
    reviewer_2_reason  = d.reviewer_2_reason,
    kappa_resolved     = d.kappa_resolved,
    kappa_final_status = d.kappa_final_status
FROM literature_document d
WHERE d.id = ars.document_id
  AND (d.reviewer_1_status IS NOT NULL OR d.reviewer_2_status IS NOT NULL
       OR d.kappa_final_status IS NOT NULL)
  AND ars.reviewer_1_status IS DISTINCT FROM d.reviewer_1_status
"""

INDEX_SQL = """
CREATE INDEX IF NOT EXISTS ix_article_scenarios_scen_kappa
  ON article_scenarios (scenario_id, kappa_final_status)
"""


def upgrade() -> None:
    bind = op.get_bind()
    tables = set(sa.inspect(bind).get_table_names())
    if not {"article_scenarios", "literature_document"} <= tables:
        print("[migration e2f6a8b3c5d7] article_scenarios/literature_document "
              "absent -- skipping (will apply once tables exist)")
        return
    op.execute(ADD_COLUMNS_SQL)
    result = bind.execute(sa.text(BACKFILL_SQL))
    op.execute(INDEX_SQL)
    print(f"[migration e2f6a8b3c5d7] added per-scenario double-blind columns; "
          f"backfilled {result.rowcount} link(s) from the global columns")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_article_scenarios_scen_kappa")
    op.execute(
        "ALTER TABLE article_scenarios "
        "DROP COLUMN IF EXISTS reviewer_1_status, "
        "DROP COLUMN IF EXISTS reviewer_1_reason, "
        "DROP COLUMN IF EXISTS reviewer_2_status, "
        "DROP COLUMN IF EXISTS reviewer_2_reason, "
        "DROP COLUMN IF EXISTS kappa_resolved, "
        "DROP COLUMN IF EXISTS kappa_final_status"
    )
