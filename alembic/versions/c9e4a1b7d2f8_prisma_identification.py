"""PRISMA identification figures per search run

The PRISMA box reported "duplicates removed: 0" for every scenario, by construction:
it counted corpus documents flagged `is_duplicate`, a flag no runtime sets. The real
de-duplication happens at ingestion (unique indexes on DOI and normalised title: a paper
returned by OpenAlex and PubMed becomes one row, the second arrival absorbed without a
trace) and at linking (_dedup_scenario_links, whose count was only logged).

`user_scenarios.prisma_identification` stores, for the last populate or rebuild, what
each source returned (the local database included), the distinct documents behind those
records, the duplicates (overlaps between sources plus rows merged by the link dedup),
the records removed for other reasons (no abstract, no local match to the boolean) and
the records that reached screening — see main._prisma_identification_figures. The
PRISMA endpoint reads it when present and falls back to the corpus otherwise, saying so.

Revision ID: c9e4a1b7d2f8
Revises: b8d3f0a6c1e7
"""
from typing import Sequence, Union

from alembic import op

revision: str = "c9e4a1b7d2f8"
down_revision: Union[str, Sequence[str], None] = "b8d3f0a6c1e7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # user_scenarios is created by the application's boot DDL, not by an earlier
    # migration; skip cleanly on a database that has not booted the app yet.
    op.execute("""
        DO $$
        BEGIN
            IF to_regclass('user_scenarios') IS NOT NULL THEN
                ALTER TABLE user_scenarios ADD COLUMN IF NOT EXISTS prisma_identification JSONB;
            END IF;
        END
        $$
    """)


def downgrade() -> None:
    op.execute("ALTER TABLE IF EXISTS user_scenarios DROP COLUMN IF EXISTS prisma_identification")
