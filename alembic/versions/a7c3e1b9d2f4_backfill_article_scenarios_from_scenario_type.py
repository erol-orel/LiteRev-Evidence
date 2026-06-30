"""backfill article_scenarios from scenario_type (Migration 1, Step 2)

Revision ID: a7c3e1b9d2f4
Revises: eb6b9e396ffc
Create Date: 2026-06-30

Migration 1 (docs/scenario-type-migration.md) moves scenario scoping from
the legacy ``literature_document.scenario_type`` ("Way A" / ingestion
membership) to ``article_scenarios`` ("Way B" / scored membership). The
production preview found 15 scenarios that exist via ``scenario_type`` but
have ZERO ``article_scenarios`` rows (the "VIDÉ" scenarios) -- switching the
reads to Way B would empty them.

This is the documented Step 2 backfill: for every literev document that has a
``scenario_type`` but no matching ``article_scenarios`` row, create that row
with ``similarity_score = NULL`` ("member but unscored"). It guarantees Way B
membership is a superset of Way A, so flipping the scoping reads can no longer
empty any scenario. NULL-score rows are NOT counted as relevant by the
canonical predicate (``screening_status = 'included' OR
COALESCE(similarity_score, 0) >= threshold``), so relevance counts are
unchanged; only membership/corpus counts converge toward Way A union Way B.

Idempotent (NOT EXISTS guard). Validated on a local Postgres 16 fixture.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a7c3e1b9d2f4"
down_revision: Union[str, Sequence[str], None] = "eb6b9e396ffc"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


BACKFILL_SQL = """
INSERT INTO article_scenarios (scenario_id, document_id, similarity_score)
SELECT d.scenario_type, d.id, NULL
FROM literature_document d
WHERE d.project_context = 'literev'
  AND d.scenario_type IS NOT NULL
  AND (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)
  AND NOT EXISTS (
      SELECT 1 FROM article_scenarios ars
      WHERE ars.scenario_id = d.scenario_type
        AND ars.document_id = d.id
  )
"""


def upgrade() -> None:
    bind = op.get_bind()
    # These tables are created at application startup, not by an earlier
    # migration, so guard against a fresh DB where they don't exist yet.
    tables = set(sa.inspect(bind).get_table_names())
    if not {"article_scenarios", "literature_document"} <= tables:
        print("[migration a7c3e1b9d2f4] article_scenarios/literature_document "
              "absent -- skipping backfill (will apply once tables exist)")
        return
    result = bind.execute(sa.text(BACKFILL_SQL))
    print(f"[migration a7c3e1b9d2f4] backfilled {result.rowcount} "
          f"article_scenarios membership row(s) from scenario_type")


def downgrade() -> None:
    # Additive, NULL-score membership rows; they are indistinguishable from
    # genuinely-unscored rows and are harmless (never counted as relevant),
    # so the downgrade is intentionally a no-op.
    pass
