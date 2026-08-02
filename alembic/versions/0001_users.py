"""Create the `users` table.

Consolidates the three upstream migrations that built up `users`:

  - `migrations/versioned/000001_agent.up.sql` — initial CREATE TABLE
  - `migrations/versioned/000049_user_preferences.up.sql` — `preferences`
    JSONB column
  - `migrations/versioned/000053_system_admin_and_settings.up.sql` —
    `is_system_admin` column

The Python rewrite starts from a clean schema, so all three land in a
single migration rather than three sequential ones.

`password_hash` carries the bcrypt digest produced by passlib. `tenant_id`
is intentionally nullable so user records that pre-date tenant
provisioning (typical of self-registration) round-trip cleanly without
violating the FK. `preferences` uses `JSONB` on Postgres (the project's
deployment target) via `JSON().with_variant(JSONB, "postgresql")` so the
migration is dialect-portable.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_users"
down_revision: str | None = "0000_init"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None

preferences_json = sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("username", sa.String(length=100), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("avatar", sa.String(length=500), nullable=True),
        sa.Column("tenant_id", sa.Integer(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column(
            "can_access_all_tenants", sa.Boolean(), nullable=False, server_default=sa.text("false")
        ),
        sa.Column("is_system_admin", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column(
            "preferences",
            preferences_json,
            nullable=False,
            server_default=sa.text("'{}'"),
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
        sa.UniqueConstraint("username", name="uq_users_username"),
        sa.UniqueConstraint("email", name="uq_users_email"),
    )
    op.create_index("ix_users_username", "users", ["username"])
    op.create_index("ix_users_email", "users", ["email"])
    op.create_index("ix_users_tenant_id", "users", ["tenant_id"])
    op.create_index("ix_users_deleted_at", "users", ["deleted_at"])
    op.create_index("ix_users_is_system_admin", "users", ["is_system_admin"])


def downgrade() -> None:
    op.drop_index("ix_users_is_system_admin", table_name="users")
    op.drop_index("ix_users_deleted_at", table_name="users")
    op.drop_index("ix_users_tenant_id", table_name="users")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_index("ix_users_username", table_name="users")
    op.drop_table("users")
