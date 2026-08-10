"""Create the `message_suggestion_sets` table.

One row records the durable generation / cache record for one assistant
message and one effective agent configuration. ``id`` is a caller-assigned
UUID (the application mints it before insert, matching the upstream
storage shape); ``tenant_id`` scopes the row to its workspace.

The unique constraint on the cache key
(``tenant_id`` + ``assistant_message_id`` + ``placement`` +
``config_hash`` + ``locale``) backs the generation acquisition: a
concurrent duplicate request is suppressed at the database instead of
racing in application code. The remaining indexes mirror the session
listing and the status / lease sweep for stale-generation recovery.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0026_message_suggestions"
down_revision: str | None = "0025_messages"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    op.create_table(
        "message_suggestion_sets",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("tenant_id", sa.Integer(), nullable=False),
        sa.Column("session_id", sa.String(length=36), nullable=False),
        sa.Column("assistant_message_id", sa.String(length=36), nullable=False),
        sa.Column(
            "agent_id",
            sa.String(length=36),
            nullable=False,
            server_default="",
        ),
        sa.Column(
            "agent_tenant_id",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column("placement", sa.String(length=32), nullable=False),
        sa.Column("config_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "locale",
            sa.String(length=16),
            nullable=False,
            server_default="",
        ),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column(
            "allow_regenerate",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "suppression_reason",
            sa.String(length=64),
            nullable=False,
            server_default="",
        ),
        sa.Column(
            "questions",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "model_id",
            sa.String(length=64),
            nullable=False,
            server_default="",
        ),
        sa.Column(
            "prompt_tokens",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "completion_tokens",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "latency_ms",
            sa.BigInteger(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "error_code",
            sa.String(length=64),
            nullable=False,
            server_default="",
        ),
        sa.Column("lease_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("generated_at", sa.DateTime(timezone=True), nullable=True),
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
            name="fk_message_suggestion_sets_tenant",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "assistant_message_id",
            "placement",
            "config_hash",
            "locale",
            name="uq_message_suggestion_sets_cache_key",
        ),
    )
    # Session listing: every suggestion set of a session, newest first.
    op.create_index(
        "idx_message_suggestion_sets_session",
        "message_suggestion_sets",
        ["tenant_id", "session_id", "created_at"],
    )
    # Stale-generation sweep: find sets stuck in ``generating``.
    op.create_index(
        "idx_message_suggestion_sets_status",
        "message_suggestion_sets",
        ["status", "lease_until"],
    )


def downgrade() -> None:
    op.drop_index("idx_message_suggestion_sets_status", table_name="message_suggestion_sets")
    op.drop_index("idx_message_suggestion_sets_session", table_name="message_suggestion_sets")
    op.drop_table("message_suggestion_sets")
