"""Create the `storage_backends` table.

One row per concrete file/object storage instance. A workspace may
register several instances of the same provider and bind each knowledge
base to a different one; ``tenants.default_storage_backend_id`` (added in
0003_tenants) points at the workspace default.

Mirrors ``internal/types/storagebackend.go::StorageBackend``. ``id`` is a
service-assigned UUID string (Go's ``BeforeCreate`` hook), ``config`` is
the JSONB normalized union of provider settings, and rows are
soft-deleted so a legacy alias still referenced by old file paths is
never physically removed.

The revision id is a placeholder — the numeric ordering is assigned when
the infra migrations are sequenced together.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0013_storage_backends"
down_revision: str | None = "0012_web_search_providers"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    op.create_table(
        "storage_backends",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("tenant_id", sa.BigInteger(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column(
            "config",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "source",
            sa.String(length=16),
            nullable=False,
            server_default=sa.text("'user'"),
        ),
        sa.Column(
            "status",
            sa.String(length=16),
            nullable=False,
            server_default=sa.text("'active'"),
        ),
        sa.Column(
            "legacy_alias",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
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
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.id"],
            name="fk_storage_backends_tenant",
            ondelete="CASCADE",
        ),
    )
    # A name is unique per workspace among live rows; the partial index
    # makes a soft-deleted name re-addable (matches tenant_kv).
    op.create_index(
        "uq_storage_backends_tenant_name_live",
        "storage_backends",
        ["tenant_id", "name"],
        unique=True,
        postgresql_where=sa.text("deleted_at IS NULL"),
    )
    op.create_index("idx_storage_backends_tenant", "storage_backends", ["tenant_id"])
    op.create_index("idx_storage_backends_provider", "storage_backends", ["provider"])
    op.create_index("idx_storage_backends_deleted_at", "storage_backends", ["deleted_at"])


def downgrade() -> None:
    op.drop_index("idx_storage_backends_deleted_at", table_name="storage_backends")
    op.drop_index("idx_storage_backends_provider", table_name="storage_backends")
    op.drop_index("idx_storage_backends_tenant", table_name="storage_backends")
    op.drop_index("uq_storage_backends_tenant_name_live", table_name="storage_backends")
    op.drop_table("storage_backends")
