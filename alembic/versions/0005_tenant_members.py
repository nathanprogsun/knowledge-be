"""Create the `tenant_members` table.

The `(user_id, tenant_id)` unique index is partial on
`deleted_at IS NULL`, so a removed member can be re-added.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0005_tenant_members"
down_revision: str | None = "0004_tenant_api_keys"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None

_LIVE_ROW = sa.text("deleted_at IS NULL")


def upgrade() -> None:
    op.create_table(
        "tenant_members",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("tenant_id", sa.BigInteger(), nullable=False),
        sa.Column(
            "role",
            sa.String(length=20),
            nullable=False,
            server_default=sa.text("'contributor'"),
        ),
        sa.Column(
            "status",
            sa.String(length=20),
            nullable=False,
            server_default=sa.text("'active'"),
        ),
        sa.Column("invited_by", sa.String(length=36), nullable=True),
        sa.Column(
            "joined_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
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
        "idx_tenant_members_user_tenant_unique",
        "tenant_members",
        ["user_id", "tenant_id"],
        unique=True,
        postgresql_where=_LIVE_ROW,
    )
    op.create_index(
        "idx_tenant_members_tenant_role",
        "tenant_members",
        ["tenant_id", "role"],
        postgresql_where=_LIVE_ROW,
    )
    op.create_index(
        "idx_tenant_members_user",
        "tenant_members",
        ["user_id"],
        postgresql_where=_LIVE_ROW,
    )


def downgrade() -> None:
    op.drop_index("idx_tenant_members_user", table_name="tenant_members")
    op.drop_index("idx_tenant_members_tenant_role", table_name="tenant_members")
    op.drop_index("idx_tenant_members_user_tenant_unique", table_name="tenant_members")
    op.drop_table("tenant_members")
