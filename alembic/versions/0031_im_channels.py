"""Create the `im_channels` table.

Each row is a tenant-scoped IM channel configuration that binds an
agent to a platform-specific bot. The primary key is a UUID string;
callers (the service layer) generate it client-side before INSERT
so the row carries the id from the start.

`credentials` is JSONB so the schema can flex per platform
(app_id, bot_token, corp_id, ...) without ALTER TABLE churn.

`bot_identity` is a derived unique key (platform + mode + credential
fields) computed by the service layer before save. The partial unique
index enforces one live channel per bot identity; the service's
duplicate-bot guard is the first line of defence and this index the
safety net.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0031_im_channels"
down_revision: str | None = "0030_agent_shares"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None

_credentials_json = sa.JSON().with_variant(
    postgresql.JSONB(astext_type=sa.Text()),
    "postgresql",
)


def upgrade() -> None:
    op.create_table(
        "im_channels",
        sa.Column("id", sa.String(length=36), primary_key=True, nullable=False),
        sa.Column("tenant_id", sa.BigInteger(), nullable=False),
        sa.Column("agent_id", sa.String(length=36), nullable=False),
        sa.Column("platform", sa.String(length=20), nullable=False),
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
            "mode",
            sa.String(length=20),
            nullable=False,
            server_default=sa.text("'websocket'"),
        ),
        sa.Column(
            "output_mode",
            sa.String(length=20),
            nullable=False,
            server_default=sa.text("'stream'"),
        ),
        sa.Column(
            "knowledge_base_id",
            sa.String(length=36),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "bot_identity",
            sa.String(length=255),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "session_mode",
            sa.String(length=20),
            nullable=False,
            server_default=sa.text("'user'"),
        ),
        sa.Column(
            "credentials",
            _credentials_json,
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
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.id"],
            name="fk_im_channels_tenant",
            ondelete="CASCADE",
        ),
    )
    op.create_index("idx_im_channels_tenant_id", "im_channels", ["tenant_id"])
    op.create_index("idx_im_channels_agent_id", "im_channels", ["agent_id"])
    op.create_index("idx_im_channels_deleted_at", "im_channels", ["deleted_at"])
    op.create_index(
        "idx_im_channels_bot_identity",
        "im_channels",
        ["bot_identity"],
        unique=True,
        postgresql_where=sa.text("deleted_at IS NULL AND bot_identity != ''"),
    )


def downgrade() -> None:
    op.drop_index("idx_im_channels_bot_identity", table_name="im_channels")
    op.drop_index("idx_im_channels_deleted_at", table_name="im_channels")
    op.drop_index("idx_im_channels_agent_id", table_name="im_channels")
    op.drop_index("idx_im_channels_tenant_id", table_name="im_channels")
    op.drop_table("im_channels")
