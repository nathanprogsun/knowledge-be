"""Create the `user_resource_favorites` table.

One row records that a user starred a single resource of a specific
type (``kb`` / ``agent``) in the current workspace. The composite
primary key ``(user_id, tenant_id, resource_type, resource_id)`` makes
inserts idempotent — a second star of the same resource collapses to
no-op — and tenant-scoped reads are naturally enforced at the key
level.

No soft-delete column is carried: unstarring deletes the row, matching
the upstream insert/delete-only schema. Indexes mirror the query
shapes: the per-user favorites list (newest first) and the tenant-wide
sweep.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0034_favorites"
down_revision: str | None = "0033_evaluation"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    op.create_table(
        "user_resource_favorites",
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("tenant_id", sa.BigInteger(), nullable=False),
        sa.Column("resource_type", sa.String(length=16), nullable=False),
        sa.Column("resource_id", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.PrimaryKeyConstraint(
            "user_id",
            "tenant_id",
            "resource_type",
            "resource_id",
            name="pk_user_resource_favorites",
        ),
    )
    op.create_index(
        "idx_user_resource_favorites_user_tenant_type_created_at",
        "user_resource_favorites",
        ["user_id", "tenant_id", "resource_type", "created_at"],
    )
    op.create_index(
        "idx_user_resource_favorites_tenant_id",
        "user_resource_favorites",
        ["tenant_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "idx_user_resource_favorites_tenant_id",
        table_name="user_resource_favorites",
    )
    op.drop_index(
        "idx_user_resource_favorites_user_tenant_type_created_at",
        table_name="user_resource_favorites",
    )
    op.drop_table("user_resource_favorites")
