"""Create the `documents` table.

Mirrors the upstream persistence contract for knowledge entries: document
metadata, ingestion channel, parse / summary status, and the physical
file reference when one exists.

``metadata`` / ``custom_metadata`` / ``last_faq_import_result`` are
JSONB. ``custom_metadata`` is user-authored descriptive metadata and is
deliberately kept separate from ``metadata`` (internal ingestion state).
Documents are soft-deleted (``deleted_at``).
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0016"
down_revision: str | None = "0015_knowledge_bases"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("tenant_id", sa.BigInteger(), nullable=False),
        sa.Column("knowledge_base_id", sa.String(length=36), nullable=False),
        sa.Column("type", sa.String(length=50), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("source", sa.String(length=2048), nullable=False),
        sa.Column("channel", sa.String(length=50), nullable=False, server_default="web"),
        sa.Column(
            "parse_status",
            sa.String(length=50),
            nullable=False,
            server_default="unprocessed",
        ),
        sa.Column(
            "pending_subtasks_count",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "summary_status",
            sa.String(length=32),
            nullable=True,
            server_default="none",
        ),
        sa.Column(
            "enable_status",
            sa.String(length=50),
            nullable=False,
            server_default="enabled",
        ),
        sa.Column("embedding_model_id", sa.String(length=64), nullable=True),
        sa.Column("file_name", sa.String(length=255), nullable=True),
        sa.Column("file_type", sa.String(length=50), nullable=True),
        sa.Column("file_size", sa.BigInteger(), nullable=True),
        sa.Column("file_hash", sa.String(length=64), nullable=True),
        sa.Column("file_path", sa.Text(), nullable=True),
        sa.Column(
            "storage_size",
            sa.BigInteger(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column("metadata", postgresql.JSONB(), nullable=True),
        sa.Column(
            "custom_metadata",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("last_faq_import_result", postgresql.JSONB(), nullable=True),
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
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("idx_documents_tenant_id", "documents", ["tenant_id"])
    op.create_index(
        "idx_documents_knowledge_base_id",
        "documents",
        ["knowledge_base_id"],
    )
    op.create_index("idx_documents_parse_status", "documents", ["parse_status"])
    op.create_index("idx_documents_enable_status", "documents", ["enable_status"])
    op.create_index("idx_documents_summary_status", "documents", ["summary_status"])
    op.create_index("idx_documents_deleted_at", "documents", ["deleted_at"])


def downgrade() -> None:
    op.drop_index("idx_documents_deleted_at", table_name="documents")
    op.drop_index("idx_documents_summary_status", table_name="documents")
    op.drop_index("idx_documents_enable_status", table_name="documents")
    op.drop_index("idx_documents_parse_status", table_name="documents")
    op.drop_index("idx_documents_knowledge_base_id", table_name="documents")
    op.drop_index("idx_documents_tenant_id", table_name="documents")
    op.drop_table("documents")
