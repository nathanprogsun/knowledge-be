"""Create the `messages` table.

One row records one chat message inside a session. ``id`` is a
caller-assigned UUID (the application mints it before insert, matching
the upstream storage shape); ``session_id`` scopes the row to its
conversation. The JSONB columns carry the retrieval references, the
agent execution trace, and the attachment / image payloads; the
``execution_context`` blob snapshots the non-secret per-turn request
state used by derived experiences such as follow-up suggestions.

Indexes mirror the query shapes: the session feed (``session_id`` +
``created_at``), the request-id resolution, the knowledge-link lookup,
and the soft-delete sweep.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0025_messages"
down_revision: str | None = "0024_knowledge_processing_spans"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    op.create_table(
        "messages",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("request_id", sa.String(length=36), nullable=False, server_default=""),
        sa.Column("session_id", sa.String(length=36), nullable=False),
        sa.Column("role", sa.String(length=50), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "knowledge_references",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("agent_steps", postgresql.JSONB(), nullable=True),
        sa.Column(
            "is_completed",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "is_fallback",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "agent_duration_ms",
            sa.BigInteger(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "rendered_content",
            sa.Text(),
            nullable=False,
            server_default="",
        ),
        sa.Column(
            "channel",
            sa.String(length=50),
            nullable=False,
            server_default="",
        ),
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
        sa.Column(
            "model_id",
            sa.String(length=64),
            nullable=False,
            server_default="",
        ),
        sa.Column(
            "execution_context",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "knowledge_id",
            sa.String(length=36),
            nullable=False,
            server_default="",
        ),
        sa.Column(
            "mentioned_items",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "images",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "attachments",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
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
    # Session feed: every message of a session in one indexed range scan.
    op.create_index(
        "idx_messages_session_created",
        "messages",
        ["session_id", "created_at"],
    )
    # Request-id resolution within a session (Q&A pair lookup).
    op.create_index(
        "idx_messages_session_request",
        "messages",
        ["session_id", "request_id"],
    )
    # Knowledge-link lookup for the chat-history indexing path.
    op.create_index(
        "idx_messages_knowledge_id",
        "messages",
        ["knowledge_id"],
    )
    # Soft-delete sweep.
    op.create_index(
        "idx_messages_deleted_at",
        "messages",
        ["deleted_at"],
    )


def downgrade() -> None:
    op.drop_index("idx_messages_deleted_at", table_name="messages")
    op.drop_index("idx_messages_knowledge_id", table_name="messages")
    op.drop_index("idx_messages_session_request", table_name="messages")
    op.drop_index("idx_messages_session_created", table_name="messages")
    op.drop_table("messages")
