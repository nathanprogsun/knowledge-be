"""Create the `tenant_invitations` table.

All three indexes are partial, and the predicates carry the semantics:

- `unique_pending` allows exactly one pending invitation per
  (workspace, invitee) while letting terminal rows accumulate as history
  — and skips share-link rows, which have no invitee and may coexist.
- `token` is unique only among rows that actually have one.
- the tenant/invitee read indexes ignore soft-deleted rows.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0006_tenant_invitations"
down_revision: str | None = "0005_tenant_members"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None

_LIVE_ROW = sa.text("deleted_at IS NULL")
_LIVE_PENDING_USER_ROW = sa.text(
    "status = 'pending' AND deleted_at IS NULL AND invitee_user_id <> ''"
)
_LIVE_TOKEN_ROW = sa.text("token <> '' AND deleted_at IS NULL")


def upgrade() -> None:
    op.create_table(
        "tenant_invitations",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("tenant_id", sa.BigInteger(), nullable=False),
        sa.Column(
            "invitee_user_id",
            sa.String(length=36),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "token",
            sa.String(length=64),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column("invited_by", sa.String(length=36), nullable=True),
        sa.Column("role", sa.String(length=20), nullable=False),
        sa.Column(
            "status",
            sa.String(length=20),
            nullable=False,
            server_default=sa.text("'pending'"),
        ),
        sa.Column("message", sa.String(length=500), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("responded_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "accepted_count",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
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
        "idx_tenant_invitations_unique_pending",
        "tenant_invitations",
        ["tenant_id", "invitee_user_id"],
        unique=True,
        postgresql_where=_LIVE_PENDING_USER_ROW,
    )
    op.create_index(
        "idx_tenant_invitations_token",
        "tenant_invitations",
        ["token"],
        unique=True,
        postgresql_where=_LIVE_TOKEN_ROW,
    )
    op.create_index(
        "idx_tenant_invitations_tenant",
        "tenant_invitations",
        ["tenant_id"],
        postgresql_where=_LIVE_ROW,
    )
    op.create_index(
        "idx_tenant_invitations_invitee",
        "tenant_invitations",
        ["invitee_user_id"],
        postgresql_where=_LIVE_ROW,
    )


def downgrade() -> None:
    op.drop_index("idx_tenant_invitations_invitee", table_name="tenant_invitations")
    op.drop_index("idx_tenant_invitations_tenant", table_name="tenant_invitations")
    op.drop_index("idx_tenant_invitations_token", table_name="tenant_invitations")
    op.drop_index("idx_tenant_invitations_unique_pending", table_name="tenant_invitations")
    op.drop_table("tenant_invitations")
