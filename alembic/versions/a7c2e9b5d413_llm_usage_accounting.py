"""Token accounting for every OpenAI call

The application called the OpenAI API from about thirty places and recorded nothing: no
call site read `response.usage`, so the only evidence that something was spending was the
invoice. Two token leaks had already been found and fixed by reading code, because there
was no other way to find them — see the comments around the PICO worker in main.py.

One row per call, tagged with the calling function, makes "which loop is spending" a
one-query question (`GET /llm-usage`). The rows the answer usually turns on:

  _background_enrichment_worker:chat        automatic PICO extraction — 50 articles every
                                            30 s, gpt-4.1-mini over up to 14 000 chars of
                                            full text. The largest potential consumer.
  _background_enrichment_worker:embeddings  500 chunks every 30 s. Cheap per unit; the
                                            volume follows whatever ingestion produces.
  _generate_variables_from_pico:chat        gpt-4.1 (not mini), max_tokens 8000, ~20k
                                            input tokens. Spiky: one per scenario whose
                                            evidence fingerprint changes.

Deliberately minimal: no cost column. Prices change and are per-account, so storing a
computed cost would bake today's rate into history. Tokens are the durable fact; multiply
by the current rate at read time.

Revision ID: a7c2e9b5d413
Revises: f3a9c1d4e8b2
"""
from typing import Sequence, Union

from alembic import op

revision: str = "a7c2e9b5d413"
down_revision: Union[str, Sequence[str], None] = "f3a9c1d4e8b2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS llm_usage (
            id                BIGSERIAL PRIMARY KEY,
            ts                TIMESTAMPTZ NOT NULL DEFAULT now(),
            -- "<calling function>:<surface>", e.g. "_extract_pico_one:chat". Derived
            -- from the stack frame that built the client, so a new call site is
            -- accounted for without anyone remembering to label it.
            purpose           TEXT        NOT NULL,
            model             TEXT        NOT NULL,
            prompt_tokens     INTEGER     NOT NULL DEFAULT 0,
            completion_tokens INTEGER     NOT NULL DEFAULT 0,
            -- Zero for streamed chat calls: the API only reports usage for a stream when
            -- asked via stream_options, which changes the chunk sequence. Those rows
            -- still count the CALL, so the blind spot is visible rather than missing.
            total_tokens      INTEGER     NOT NULL DEFAULT 0
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_llm_usage_ts ON llm_usage (ts DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_llm_usage_purpose_ts "
               "ON llm_usage (purpose, ts DESC)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_llm_usage_purpose_ts")
    op.execute("DROP INDEX IF EXISTS idx_llm_usage_ts")
    op.execute("DROP TABLE IF EXISTS llm_usage")
