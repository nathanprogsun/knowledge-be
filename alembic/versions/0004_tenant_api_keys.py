"""Create the `tenant_api_keys` table.

There is no `deleted_at`: a key is retired by stamping `revoked_at`,
which every read filters on.

The CHECK constraint enforces `(scope_type, tenant_id, full_access)`
consistency: tenant keys carry a tenant id, platform keys carry none
and are not full-access.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004_tenant_api_keys"
down_revision: str | None = "0003_tenants"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    op.create_table(
        "tenant_api_keys",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("tenant_id", sa.BigInteger(), nullable=True),
        sa.Column(
            "scope_type",
            sa.String(length=16),
            nullable=False,
            server_default=sa.text("'tenant'"),
        ),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("key_hash", sa.String(length=64), nullable=False, unique=True),
        sa.Column("api_key", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column(
            "full_access",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "knowledge_base_ids",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "capabilities",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
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
            ["tenant_id"],
            ["tenants.id"],
            name="fk_tenant_api_keys_tenant",
            ondelete="CASCADE",
        ),
        sa.CheckConstraint(
            "(scope_type = 'tenant' AND tenant_id IS NOT NULL)"
            " OR (scope_type = 'platform' AND tenant_id IS NULL AND full_access = FALSE)",
            name="chk_tenant_api_keys_scope",
        ),
    )
    op.create_index("idx_tenant_api_keys_tenant", "tenant_api_keys", ["tenant_id"])
    op.create_index("idx_tenant_api_keys_revoked_at", "tenant_api_keys", ["revoked_at"])
    op.create_index("idx_tenant_api_keys_scope_type", "tenant_api_keys", ["scope_type"])


def downgrade() -> None:
    op.drop_index("idx_tenant_api_keys_scope_type", table_name="tenant_api_keys")
    op.drop_index("idx_tenant_api_keys_revoked_at", table_name="tenant_api_keys")
    op.drop_index("idx_tenant_api_keys_tenant", table_name="tenant_api_keys")
    op.drop_table("tenant_api_keys")
