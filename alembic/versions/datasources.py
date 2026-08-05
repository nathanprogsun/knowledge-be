"""Create the `data_sources` and `sync_logs` tables.

Mirrors ``migrations/versioned/000029_datasource_tables.up.sql``:
external data-source configurations plus the per-run sync history.

``config`` / ``last_sync_cursor`` / ``last_sync_result`` / ``result`` are
JSONB. ``config`` carries the AES-encrypted connector credential blob;
the service redacts it on the way out, the column itself is opaque.

Data sources are soft-deleted (``deleted_at``); sync logs are not (they
are the audit trail and cascade-delete with their source).

Revision id is a placeholder — the checkpoint PR assigns the final
number and ``down_revision`` link.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "datasources"
down_revision: str | None = "0008_tenant_kv"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    op.create_table(
        "data_sources",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("tenant_id", sa.BigInteger(), nullable=False),
        sa.Column("knowledge_base_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("type", sa.String(length=50), nullable=False),
        sa.Column("config", postgresql.JSONB(), nullable=True),
        sa.Column("sync_schedule", sa.String(length=100), nullable=False, server_default=""),
        sa.Column(
            "sync_mode",
            sa.String(length=20),
            nullable=False,
            server_default="incremental",
        ),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="active"),
        sa.Column(
            "conflict_strategy",
            sa.String(length=32),
            nullable=False,
            server_default="overwrite",
        ),
        sa.Column(
            "sync_deletions",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("true"),
        ),
        sa.Column("last_sync_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_sync_cursor", postgresql.JSONB(), nullable=True),
        sa.Column("last_sync_result", postgresql.JSONB(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=False, server_default=""),
        sa.Column(
            "sync_log_retention_days",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("30"),
        ),
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
    op.create_index("idx_data_sources_tenant_id", "data_sources", ["tenant_id"])
    op.create_index(
        "idx_data_sources_knowledge_base_id",
        "data_sources",
        ["knowledge_base_id"],
    )
    op.create_index("idx_data_sources_type", "data_sources", ["type"])
    op.create_index("idx_data_sources_status", "data_sources", ["status"])
    op.create_index("idx_data_sources_deleted_at", "data_sources", ["deleted_at"])

    op.create_table(
        "sync_logs",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("data_source_id", sa.String(length=36), nullable=False),
        sa.Column("tenant_id", sa.BigInteger(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("items_total", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("items_created", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("items_updated", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("items_deleted", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("items_skipped", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("items_failed", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("error_message", sa.Text(), nullable=False, server_default=""),
        sa.Column("result", postgresql.JSONB(), nullable=True),
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
        sa.ForeignKeyConstraint(
            ["data_source_id"],
            ["data_sources.id"],
            name="fk_sync_logs_data_source",
            ondelete="CASCADE",
        ),
    )
    op.create_index("idx_sync_logs_data_source_id", "sync_logs", ["data_source_id"])
    op.create_index("idx_sync_logs_tenant_id", "sync_logs", ["tenant_id"])
    op.create_index("idx_sync_logs_status", "sync_logs", ["status"])
    op.create_index("idx_sync_logs_started_at", "sync_logs", ["started_at"])


def downgrade() -> None:
    op.drop_index("idx_sync_logs_started_at", table_name="sync_logs")
    op.drop_index("idx_sync_logs_status", table_name="sync_logs")
    op.drop_index("idx_sync_logs_tenant_id", table_name="sync_logs")
    op.drop_index("idx_sync_logs_data_source_id", table_name="sync_logs")
    op.drop_table("sync_logs")
    op.drop_index("idx_data_sources_deleted_at", table_name="data_sources")
    op.drop_index("idx_data_sources_status", table_name="data_sources")
    op.drop_index("idx_data_sources_type", table_name="data_sources")
    op.drop_index("idx_data_sources_knowledge_base_id", table_name="data_sources")
    op.drop_index("idx_data_sources_tenant_id", table_name="data_sources")
    op.drop_table("data_sources")
