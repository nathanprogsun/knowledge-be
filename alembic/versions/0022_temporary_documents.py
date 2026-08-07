"""Create the `temporary_documents` table.

Session-scoped, expiring documents uploaded for chat turns. Parsed
artifacts (``content`` / ``chunks`` / ``image_refs``) are retained
separately from the source file so a later turn can select only the
useful parts without re-parsing the upload.

Columns mirror the upstream entity: the four JSONB payload columns carry
server defaults (``[]`` / ``[]`` / ``{}`` / ``{}``) that the application
also applies on create, and ``deleted_at`` is the soft-delete marker
used by the expiry sweep. ``tenant_id``, ``session_id``, ``status`` and
``expires_at`` are indexed to back the scoped lookups and the sweep.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0022_temporary_documents"
down_revision: str | None = "0021_wiki_pages"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    op.create_table(
        "temporary_documents",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("tenant_id", sa.BigInteger(), nullable=False),
        sa.Column("session_id", sa.String(length=36), nullable=False),
        sa.Column("resource_ref", sa.Text(), nullable=False),
        sa.Column("file_name", sa.String(length=1024), nullable=False),
        sa.Column("file_type", sa.String(length=32), nullable=False),
        sa.Column(
            "mime_type",
            sa.String(length=255),
            nullable=False,
            server_default="",
        ),
        sa.Column("file_size", sa.BigInteger(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("content", sa.Text(), nullable=True),
        sa.Column(
            "chunks",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "image_refs",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "metadata",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "processing_options",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "token_count",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "chunk_count",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ready_at", sa.DateTime(timezone=True), nullable=True),
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
    op.create_index(
        "idx_temporary_documents_tenant_id",
        "temporary_documents",
        ["tenant_id"],
    )
    op.create_index(
        "idx_temporary_documents_session_id",
        "temporary_documents",
        ["session_id"],
    )
    op.create_index(
        "idx_temporary_documents_status",
        "temporary_documents",
        ["status"],
    )
    op.create_index(
        "idx_temporary_documents_expires_at",
        "temporary_documents",
        ["expires_at"],
    )
    op.create_index(
        "idx_temporary_documents_deleted_at",
        "temporary_documents",
        ["deleted_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "idx_temporary_documents_deleted_at",
        table_name="temporary_documents",
    )
    op.drop_index(
        "idx_temporary_documents_expires_at",
        table_name="temporary_documents",
    )
    op.drop_index(
        "idx_temporary_documents_status",
        table_name="temporary_documents",
    )
    op.drop_index(
        "idx_temporary_documents_session_id",
        table_name="temporary_documents",
    )
    op.drop_index(
        "idx_temporary_documents_tenant_id",
        table_name="temporary_documents",
    )
    op.drop_table("temporary_documents")
