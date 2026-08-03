"""Create the `tenants` table.

The legacy per-tenant `api_key` column is intentionally absent —
revocable machine credentials live in `tenant_api_keys` instead.
Storage quota defaults to 10 GiB.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003_tenants"
down_revision: str | None = "0002_auth_tokens"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None

_DEFAULT_STORAGE_QUOTA_BYTES = "10737418240"


def upgrade() -> None:
    op.create_table(
        "tenants",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "retriever_engines",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{\"engines\": []}'::jsonb"),
        ),
        sa.Column(
            "status",
            sa.String(length=50),
            nullable=False,
            server_default=sa.text("'active'"),
        ),
        sa.Column(
            "business",
            sa.String(length=255),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "storage_quota",
            sa.BigInteger(),
            nullable=False,
            server_default=sa.text(_DEFAULT_STORAGE_QUOTA_BYTES),
        ),
        sa.Column(
            "storage_used",
            sa.BigInteger(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column("agent_config", postgresql.JSONB(), nullable=True),
        sa.Column("context_config", postgresql.JSONB(), nullable=True),
        sa.Column("conversation_config", postgresql.JSONB(), nullable=True),
        sa.Column("web_search_config", postgresql.JSONB(), nullable=True),
        sa.Column("parser_engine_config", postgresql.JSONB(), nullable=True),
        sa.Column("storage_engine_config", postgresql.JSONB(), nullable=True),
        sa.Column("default_storage_backend_id", sa.String(length=36), nullable=True),
        sa.Column("credentials", postgresql.JSONB(), nullable=True),
        sa.Column("chat_history_config", postgresql.JSONB(), nullable=True),
        sa.Column("retrieval_config", postgresql.JSONB(), nullable=True),
        sa.Column("api_principal_config", postgresql.JSONB(), nullable=True),
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
    op.create_index("idx_tenants_status", "tenants", ["status"])
    op.create_index("ix_tenants_deleted_at", "tenants", ["deleted_at"])


def downgrade() -> None:
    op.drop_index("ix_tenants_deleted_at", table_name="tenants")
    op.drop_index("idx_tenants_status", table_name="tenants")
    op.drop_table("tenants")
