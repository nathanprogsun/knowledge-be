"""Create the `auth_tokens` table.

Mirrors the upstream schema in
`migrations/versioned/000001_agent.up.sql` (Postgres) and
`migrations/sqlite/000000_init.up.sql` (SQLite). Tokens are stored
verbatim in the `token` column (no hashing at rest) — the upstream
choice — so that `ValidateToken` can resolve them in O(1) by exact
match.

`is_revoked = TRUE` rows are kept (not soft-deleted) so audit logs
and replay-attack checks remain possible.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0002_auth_tokens"
down_revision: str | None = "0001_users"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    op.create_table(
        "auth_tokens",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("token", sa.Text(), nullable=False),
        sa.Column("token_type", sa.String(length=50), nullable=False),
        sa.Column(
            "expires_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column(
            "is_revoked",
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
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_auth_tokens_user",
            ondelete="CASCADE",
        ),
    )
    op.create_index("ix_auth_tokens_user_id", "auth_tokens", ["user_id"])
    op.create_index("ix_auth_tokens_token", "auth_tokens", ["token"], unique=True)
    op.create_index("ix_auth_tokens_token_type", "auth_tokens", ["token_type"])
    op.create_index("ix_auth_tokens_expires_at", "auth_tokens", ["expires_at"])


def downgrade() -> None:
    op.drop_index("ix_auth_tokens_expires_at", table_name="auth_tokens")
    op.drop_index("ix_auth_tokens_token_type", table_name="auth_tokens")
    op.drop_index("ix_auth_tokens_token", table_name="auth_tokens")
    op.drop_index("ix_auth_tokens_user_id", table_name="auth_tokens")
    op.drop_table("auth_tokens")