"""Create the `sessions` table.

One row records one chat session, scoped by ``tenant_id``. ``id`` is an
application-assigned UUID (the service mints it on create, mirroring the
upstream entity). ``user_id`` is the owner scope: Web-console users, API
external-user principals, and embed visitor principals all use this
column, while legacy / API-key rows keep it empty and stay visible at
the tenant level.

``is_pinned`` / ``pinned_at`` back the pin toggle: a pinned row carries
a non-null ``pinned_at`` timestamp, and unpinning clears it. ``tenant_id``
is INTEGER to match the upstream storage shape.

Indexes mirror the query patterns: the tenant-scoped listing, the
pin-aware per-user list (partial on live rows), and the soft-delete
sweep.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0025_sessions"
down_revision: str | None = "0024_knowledge_processing_spans"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None

_LIVE_ROW = sa.text("deleted_at is null")


def upgrade() -> None:
    op.create_table(
        "sessions",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("tenant_id", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("user_id", sa.String(length=36), nullable=True),
        sa.Column(
            "is_pinned",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column("pinned_at", sa.DateTime(timezone=True), nullable=True),
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
        "idx_sessions_tenant_id",
        "sessions",
        ["tenant_id"],
    )
    op.create_index(
        "idx_sessions_tenant_user_pin",
        "sessions",
        ["tenant_id", "user_id", "is_pinned", "pinned_at", "updated_at"],
        postgresql_where=_LIVE_ROW,
    )
    op.create_index(
        "idx_sessions_deleted_at",
        "sessions",
        ["deleted_at"],
    )


def downgrade() -> None:
    op.drop_index("idx_sessions_deleted_at", table_name="sessions")
    op.drop_index("idx_sessions_tenant_user_pin", table_name="sessions")
    op.drop_index("idx_sessions_tenant_id", table_name="sessions")
    op.drop_table("sessions")
