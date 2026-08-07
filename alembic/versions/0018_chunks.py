"""Create the `chunks` table.

Mirrors the upstream chunk schema: the basic retrieval unit split out of
a document, carrying its positional relationship with the source text
(``start_at`` / ``end_at`` / ``chunk_index``) plus indexing bookkeeping.

Column notes
------------

- ``seq_id`` is a DB-assigned auto-increment (sequence-backed) value used
  for FAQ-style external references; it carries a unique index.
- ``relation_chunks`` / ``indirect_relation_chunks`` / ``metadata`` are
  JSONB. ``image_info`` is a plain ``text`` column whose JSON payload is
  parsed at the service layer on read.
- ``chunk_type`` defaults to ``text``; the other chunk kinds (parent,
  image OCR/caption, FAQ, entity, ...) are written by their owning
  pipelines.
- Rows are soft-deleted (``deleted_at``); reads filter live rows.

Indexes mirror the upstream query shapes: parent lookups, tenant+knowledge
scans, tag/content-hash lookups, KB-scoped aggregates, and the unique
``seq_id`` reference.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0018"
down_revision: str | None = "0017_faq"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None

_SEQ_ID_SEQUENCE = "chunks_seq_id_seq"


def upgrade() -> None:
    op.execute(sa.text(f"CREATE SEQUENCE IF NOT EXISTS {_SEQ_ID_SEQUENCE} START WITH 100000000"))
    op.create_table(
        "chunks",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("tenant_id", sa.Integer(), nullable=False),
        sa.Column("knowledge_base_id", sa.String(length=36), nullable=False),
        sa.Column("knowledge_id", sa.String(length=36), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column(
            "is_enabled",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("true"),
        ),
        sa.Column("start_at", sa.Integer(), nullable=False),
        sa.Column("end_at", sa.Integer(), nullable=False),
        sa.Column("pre_chunk_id", sa.String(length=36), nullable=True),
        sa.Column("next_chunk_id", sa.String(length=36), nullable=True),
        sa.Column(
            "chunk_type",
            sa.String(length=20),
            nullable=False,
            server_default="text",
        ),
        sa.Column("parent_chunk_id", sa.String(length=36), nullable=True),
        sa.Column("image_info", sa.Text(), nullable=True),
        sa.Column("relation_chunks", postgresql.JSONB(), nullable=True),
        sa.Column("indirect_relation_chunks", postgresql.JSONB(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), nullable=True),
        sa.Column("tag_id", sa.String(length=36), nullable=True),
        sa.Column("status", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("content_hash", sa.String(length=64), nullable=True),
        sa.Column("flags", sa.Integer(), nullable=False, server_default=sa.text("1")),
        sa.Column(
            "seq_id",
            sa.BigInteger(),
            nullable=False,
            server_default=sa.text(f"nextval('{_SEQ_ID_SEQUENCE}')"),
        ),
        sa.Column("source_content", sa.Text(), nullable=False, server_default=""),
        sa.Column(
            "content_revision",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "index_status",
            sa.String(length=16),
            nullable=False,
            server_default="ready",
        ),
        sa.Column(
            "last_editor_id",
            sa.String(length=64),
            nullable=False,
            server_default="",
        ),
        sa.Column("context_header", sa.Text(), nullable=False, server_default=""),
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
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("idx_chunks_tenant_kg", "chunks", ["tenant_id", "knowledge_id"])
    op.create_index("idx_chunks_parent_id", "chunks", ["parent_chunk_id"])
    op.create_index("idx_chunks_chunk_type", "chunks", ["chunk_type"])
    op.create_index("idx_chunks_tag", "chunks", ["tag_id"])
    op.create_index("idx_chunks_content_hash", "chunks", ["content_hash"])
    op.create_index("idx_chunks_seq_id", "chunks", ["seq_id"], unique=True)
    op.create_index(
        "idx_chunks_kb_tenant",
        "chunks",
        ["knowledge_base_id", "tenant_id"],
    )
    op.create_index(
        "idx_chunks_knowledge_enabled",
        "chunks",
        ["knowledge_id", "is_enabled", "deleted_at"],
    )


def downgrade() -> None:
    op.drop_index("idx_chunks_knowledge_enabled", table_name="chunks")
    op.drop_index("idx_chunks_kb_tenant", table_name="chunks")
    op.drop_index("idx_chunks_seq_id", table_name="chunks")
    op.drop_index("idx_chunks_content_hash", table_name="chunks")
    op.drop_index("idx_chunks_tag", table_name="chunks")
    op.drop_index("idx_chunks_chunk_type", table_name="chunks")
    op.drop_index("idx_chunks_parent_id", table_name="chunks")
    op.drop_index("idx_chunks_tenant_kg", table_name="chunks")
    op.drop_table("chunks")
    op.execute(sa.text(f"DROP SEQUENCE IF EXISTS {_SEQ_ID_SEQUENCE}"))
