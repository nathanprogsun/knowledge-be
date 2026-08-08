"""Create the `knowledge_processing_spans` table.

Per-(knowledge, attempt) span tree for the document parsing pipeline,
mirroring Langfuse's trace / span / generation vocabulary:

- one ``root`` span per (knowledge_id, attempt) acting as the trace;
- ``stage`` spans (docreader / chunking / embedding / multimodal /
  postprocess) as children of root;
- free-form ``subspan`` / ``generation`` rows hanging off any parent.

The table records per-stage progress so the frontend can render a
five-segment timeline, failures carry a stable ``error_code``, reparse
history is preserved across attempts (``?attempt=N`` post-mortem), and
cascade-cancel rules can flip dependent spans to ``cancelled`` when an
upstream span fails.

``name`` is VARCHAR(255): wiki ingestion builds names like
``postprocess.wiki.page[<slug>]`` that can exceed 64 characters, and the
tracker truncates at the DB limit with a hash suffix.

Indexes mirror the query shapes: the (knowledge_id, attempt) range scan
for tree assembly, the ``status + started_at`` diagnostic sweep for
spans stuck in ``running``, and parent-span lineage walks for the
cascade-cancel rules. The unique constraint backs the tracker's upsert.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0023_knowledge_processing_spans"
down_revision: str | None = "0022_temporary_documents"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    op.create_table(
        "knowledge_processing_spans",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("knowledge_id", sa.String(length=64), nullable=False),
        sa.Column(
            "attempt",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("1"),
        ),
        sa.Column("span_id", sa.String(length=64), nullable=False),
        sa.Column("parent_span_id", sa.String(length=64), nullable=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("input", postgresql.JSONB(), nullable=True),
        sa.Column("output", postgresql.JSONB(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), nullable=True),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("error_detail", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_ms", sa.BigInteger(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.UniqueConstraint(
            "knowledge_id",
            "attempt",
            "span_id",
            name="uq_kpspan_attempt_span",
        ),
    )
    # Primary read path: every span for a (knowledge, attempt) tuple in
    # one indexed range scan, then the tree is built in memory.
    op.create_index(
        "idx_kpspan_knowledge_attempt",
        "knowledge_processing_spans",
        ["knowledge_id", "attempt"],
    )
    # Diagnostic / housekeeping sweep: "find spans stuck in running".
    op.create_index(
        "idx_kpspan_status_started",
        "knowledge_processing_spans",
        ["status", "started_at"],
    )
    # Lineage walks for cascade-cancel: find every child by parent id.
    op.create_index(
        "idx_kpspan_parent",
        "knowledge_processing_spans",
        ["parent_span_id"],
        postgresql_where=sa.text("parent_span_id IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("idx_kpspan_parent", table_name="knowledge_processing_spans")
    op.drop_index("idx_kpspan_status_started", table_name="knowledge_processing_spans")
    op.drop_index("idx_kpspan_knowledge_attempt", table_name="knowledge_processing_spans")
    op.drop_table("knowledge_processing_spans")
