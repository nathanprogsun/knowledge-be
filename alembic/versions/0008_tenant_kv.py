"""Create the `tenant_kv` table.

Generic key-value store scoped to a workspace. Each row binds a JSON value
to a (tenant, key) pair; the pair is unique among live rows. Values are
JSONB and carry no column-level schema (the wire contract types shape the
decoded value).

Rows are soft-deleted so a removed key is re-addable under the same
partial unique index (matching the tenant_members pattern).
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0008_tenant_kv"
down_revision: str | None = "0007_audit_settings"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    op.create_table(
        "tenant_kv",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("tenant_id", sa.BigInteger(), nullable=False),
        sa.Column("key", sa.String(length=128), nullable=False),
        sa.Column("value", postgresql.JSONB(), nullable=False),
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
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.id"],
            name="fk_tenant_kv_tenant",
            ondelete="CASCADE",
        ),
    )
    op.create_index(
        "uq_tenant_kv_tenant_key_live",
        "tenant_kv",
        ["tenant_id", "key"],
        unique=True,
        postgresql_where=sa.text("deleted_at IS NULL"),
    )
    op.create_index("idx_tenant_kv_tenant", "tenant_kv", ["tenant_id"])


def downgrade() -> None:
    op.drop_index("idx_tenant_kv_tenant", table_name="tenant_kv")
    op.drop_index("uq_tenant_kv_tenant_key_live", table_name="tenant_kv")
    op.drop_table("tenant_kv")
