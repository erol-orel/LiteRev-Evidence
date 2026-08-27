"""ReliefWeb situation reports — a SEPARATE evidence stream

Deliberately NOT stored in `literature_document`. Four blockers, each verified against a
real database rather than assumed:

1. `uq_litdoc_title_norm` is UNIQUE (project_context, title_norm) WHERE length >= 20, and
   ReliefWeb's normal pattern is a RECURRING title — "Ukraine: Humanitarian Impact
   Situation Report No. 12" normalises to 49 characters. Inserting the next issue raises
   UniqueViolation. Changing project_context does not help: the index covers it.
2. `_ingest_doc_direct` drops the second report before the index even fires (its
   pre-SELECT ORs external_id / title_norm / doi) and returns is_new=False, so the loss is
   indistinguishable from a legitimate cross-source duplicate and invisible in the logs.
3. There is no project isolation on corpus membership: `_build_where({})` returns
   ('', {}), so any row in literature_document is reachable from an ordinary paper
   scenario on a lexical match — exactly the contamination this feature must avoid.
4. `year` is an INTEGER. ReliefWeb publishes daily; a whole outbreak collapses into one
   bucket, breaking the year filters, the corpus year histogram and the recency term of
   the quality score.

Keeping a separate table also means PRISMA counts, /corpus/stats and the 12-source
federation are structurally untouched by this feature.

Revision ID: f3a9c1d4e8b2
Revises: e2f6a8b3c5d7
"""
from typing import Sequence, Union

from alembic import op

revision: str = "f3a9c1d4e8b2"
down_revision: Union[str, Sequence[str], None] = "e2f6a8b3c5d7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS situation_report (
            id              SERIAL PRIMARY KEY,
            rw_id           TEXT NOT NULL,
            rw_kind         TEXT NOT NULL DEFAULT 'report',
            title           TEXT NOT NULL,
            body            TEXT,
            url             TEXT,
            -- date.original: the SOURCE's publication date, not our ingest time. A
            -- timestamp, not a year: ReliefWeb publishes daily and the cadence matters.
            published_at    TIMESTAMPTZ,
            changed_at      TIMESTAMPTZ,
            fetched_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            sources         JSONB NOT NULL DEFAULT '[]'::jsonb,
            format          TEXT,
            primary_country TEXT,
            primary_iso3    TEXT,
            countries       JSONB NOT NULL DEFAULT '[]'::jsonb,
            glide           TEXT,
            disaster_names  JSONB NOT NULL DEFAULT '[]'::jsonb,
            disaster_types  JSONB NOT NULL DEFAULT '[]'::jsonb,
            themes          JSONB NOT NULL DEFAULT '[]'::jsonb,
            language        TEXT,
            status          TEXT,
            -- Grey-literature credibility, capped at 0.45 by reliefweb_source so a
            -- situation report can never weigh like a peer-reviewed study.
            credibility     DOUBLE PRECISION NOT NULL DEFAULT 0.2,
            title_norm      TEXT,
            -- Collapses a recurring numbered series ("... No. 12" / "... No. 13") so
            -- publication cadence is never mistaken for an epidemic signal.
            series_key      TEXT
        )
    """)
    # ReliefWeb's own id is the natural key and is stable across revisions of a report,
    # so re-ingesting an edited report UPDATES rather than duplicating.
    op.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_sitrep_rw "
               "ON situation_report (rw_kind, rw_id)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_sitrep_published ON situation_report (published_at DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_sitrep_iso3 ON situation_report (primary_iso3)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_sitrep_glide ON situation_report (glide)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_sitrep_series ON situation_report (series_key)")

    op.execute("""
        CREATE TABLE IF NOT EXISTS situation_report_scenarios (
            id          SERIAL PRIMARY KEY,
            report_id   INTEGER NOT NULL REFERENCES situation_report(id) ON DELETE CASCADE,
            scenario_id TEXT NOT NULL,
            relevance   DOUBLE PRECISION,
            linked_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_sitrep_scenario "
               "ON situation_report_scenarios (report_id, scenario_id)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_sitrep_scenario_sid "
               "ON situation_report_scenarios (scenario_id)")

    # Daily call counter. The API documents 1000 calls/day but exposes NO rate-limit
    # header, no reset time and no error code for exhaustion, so the budget has to be
    # tracked locally or it cannot be respected at all.
    op.execute("""
        CREATE TABLE IF NOT EXISTS reliefweb_quota (
            day         DATE PRIMARY KEY,
            calls_used  INTEGER NOT NULL DEFAULT 0
        )
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS situation_report_scenarios")
    op.execute("DROP TABLE IF EXISTS situation_report")
    op.execute("DROP TABLE IF EXISTS reliefweb_quota")
