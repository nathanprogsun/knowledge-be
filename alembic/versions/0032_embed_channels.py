"""Create the `embed_channels` table.

Each row is a tenant-scoped embed channel configuration that publishes
an agent chat surface for external websites. The primary key is a UUID
string; callers (the service layer) generate it client-side before
INSERT so the row carries the id from the start.

`allowed_origins` is JSONB so the schema can flex per channel without
ALTER TABLE churn; it holds the array of origin patterns the embed
client checks before loading the widget.

`publish_token` and `webhook_secret` are secret-bearing columns: the
service layer controls what crosses the wire, and the projection
excludes them.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0032_embed_channels"
down_revision: str | None = "0031_im_channels"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None

_allowed_origins_json = sa.JSON().with_variant(
    postgresql.JSONB(astext_type=sa.Text()),
    "postgresql",
)


def upgrade() -> None:
    op.create_table(
        "embed_channels",
        sa.Column("id", sa.String(length=36), primary_key=True, nullable=False),
        sa.Column("tenant_id", sa.BigInteger(), nullable=False),
        sa.Column(
            "agent_id",
            sa.String(length=36),
            nullable=False,
            server_default=sa.text("'builtin-quick-answer'"),
        ),
        sa.Column(
            "name",
            sa.String(length=255),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "enabled",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("true"),
        ),
        sa.Column(
            "publish_token",
            sa.String(length=64),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "allowed_origins",
            _allowed_origins_json,
            nullable=False,
            server_default=sa.text("'[]'"),
        ),
        sa.Column(
            "welcome_message",
            sa.Text(),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "rate_limit_per_minute",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("30"),
        ),
        sa.Column(
            "rate_limit_per_day",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("10000"),
        ),
        sa.Column(
            "primary_color",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "page_title",
            sa.String(length=255),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "header_title_mode",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("'channel'"),
        ),
        sa.Column(
            "show_suggested_questions",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("true"),
        ),
        sa.Column(
            "widget_position",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("'bottom-right'"),
        ),
        sa.Column(
            "allow_web_search",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "allow_file_upload",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "default_locale",
            sa.String(length=16),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "webhook_url",
            sa.String(length=512),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "webhook_secret",
            sa.String(length=128),
            nullable=False,
            server_default=sa.text("''"),
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
            name="fk_embed_channels_tenant",
            ondelete="CASCADE",
        ),
    )
    op.create_index("idx_embed_channels_tenant_id", "embed_channels", ["tenant_id"])
    op.create_index("idx_embed_channels_agent_id", "embed_channels", ["agent_id"])
    op.create_index("idx_embed_channels_deleted_at", "embed_channels", ["deleted_at"])


def downgrade() -> None:
    op.drop_index("idx_embed_channels_deleted_at", table_name="embed_channels")
    op.drop_index("idx_embed_channels_agent_id", table_name="embed_channels")
    op.drop_index("idx_embed_channels_tenant_id", table_name="embed_channels")
    op.drop_table("embed_channels")
