"""Create the `custom_agents` table.

GPTs-style agent configuration rows: a soft-deletable entity whose
``config`` JSONB blob carries the full agent policy (model bindings,
tool allow-list, knowledge-base selection, suggestion settings). The
primary key is composite ``(id, tenant_id)`` so the same built-in id
can hold per-tenant customised configs while custom agents keep
caller-minted UUID ids.

``config`` is NOT NULL with a server default (``{}``) that the
application also applies on create. ``created_by`` records the owning
user for the workspace RBAC guard, or stays empty for tenant-owned rows
created through the API-key path. ``tenant_id`` is INTEGER to match the
upstream storage shape.

Indexes mirror the query patterns: tenant-scoped listings, the built-in
flag filter, and the soft-delete sweep.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0023_custom_agents"
down_revision: str | None = "0022_temporary_documents"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    op.create_table(
        "custom_agents",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("avatar", sa.String(length=64), nullable=True),
        sa.Column(
            "is_builtin",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column("tenant_id", sa.Integer(), primary_key=True),
        sa.Column("created_by", sa.String(length=36), nullable=True),
        sa.Column(
            "config",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
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
    op.create_index(
        "idx_custom_agents_tenant_id",
        "custom_agents",
        ["tenant_id"],
    )
    op.create_index(
        "idx_custom_agents_is_builtin",
        "custom_agents",
        ["is_builtin"],
    )
    op.create_index(
        "idx_custom_agents_deleted_at",
        "custom_agents",
        ["deleted_at"],
    )


def downgrade() -> None:
    op.drop_index("idx_custom_agents_deleted_at", table_name="custom_agents")
    op.drop_index("idx_custom_agents_is_builtin", table_name="custom_agents")
    op.drop_index("idx_custom_agents_tenant_id", table_name="custom_agents")
    op.drop_table("custom_agents")
