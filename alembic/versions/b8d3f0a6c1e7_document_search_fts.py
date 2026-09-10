"""Full-text search table for lexical corpus membership

Corpus membership — which documents a scenario's boolean query matches — compiled each
term to `LIKE '%term%'` on title, abstract and every chunk, trigram-assisted. On the
production corpus (346 152 documents, 1 245 182 chunks) one query took 55 to 240 s, and
substrings matched (`%ai%` inside "chain"), which is how one corpus reached 238 438
documents.

`document_search` holds ONE tsvector per document (title + abstract + full-text chunks),
GIN-indexed; the whole boolean becomes one tsquery evaluated in the index, with the same
per-document semantics. Triggers on literature_document and document_chunk keep it
current; the application's background worker backfills existing documents and
recomputes rows the chunk triggers mark stale. The DDL is imported from
lexical_search.py so that a migrated database and a booted one get the identical
objects (alembic.ini sets prepend_sys_path = ., so the module resolves).

Revision ID: b8d3f0a6c1e7
Revises: a7c2e9b5d413
"""
from typing import Sequence, Union

from alembic import op

from lexical_search import DDL, TRIGGER_NAMES

revision: str = "b8d3f0a6c1e7"
down_revision: Union[str, Sequence[str], None] = "a7c2e9b5d413"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    for statement in DDL:
        op.execute(statement)


def downgrade() -> None:
    for name in TRIGGER_NAMES:
        table = "literature_document" if "_doc_" in name else "document_chunk"
        op.execute(f"DROP TRIGGER IF EXISTS {name} ON {table}")
    op.execute("DROP FUNCTION IF EXISTS literev_document_search_chunk_trg()")
    op.execute("DROP FUNCTION IF EXISTS literev_document_search_doc_trg()")
    op.execute("DROP FUNCTION IF EXISTS literev_document_tsv(BIGINT)")
    op.execute("DROP TABLE IF EXISTS document_search")
