"""Create the MCP service domain tables.

Tables created:

  - ``mcp_services``         — MCP server configurations
  - ``mcp_tool_approvals``   — per-tool approval overrides
  - ``mcp_oauth_clients``    — per-tenant registered OAuth client metadata
  - ``mcp_oauth_tokens``     — per-user OAuth token storage

Storage-only columns (``api_key`` / ``token`` inside ``auth_config``)
live inside the JSON blob and are never a column on the SQL side.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0010_mcp_services"
down_revision: str | None = "0009_models"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None

_headers_json = sa.JSON().with_variant(
    postgresql.JSONB(astext_type=sa.Text()),
    "postgresql",
)
_auth_config_json = sa.JSON().with_variant(
    postgresql.JSONB(astext_type=sa.Text()),
    "postgresql",
)
_advanced_config_json = sa.JSON().with_variant(
    postgresql.JSONB(astext_type=sa.Text()),
    "postgresql",
)
_stdio_config_json = sa.JSON().with_variant(
    postgresql.JSONB(astext_type=sa.Text()),
    "postgresql",
)
_env_vars_json = sa.JSON().with_variant(
    postgresql.JSONB(astext_type=sa.Text()),
    "postgresql",
)


def upgrade() -> None:
    op.create_table(
        "mcp_services",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("tenant_id", sa.BigInteger(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("transport_type", sa.String(length=50), nullable=False),
        sa.Column("url", sa.String(length=512), nullable=True),
        sa.Column("headers", _headers_json, nullable=True),
        sa.Column("auth_config", _auth_config_json, nullable=True),
        sa.Column("advanced_config", _advanced_config_json, nullable=True),
        sa.Column("stdio_config", _stdio_config_json, nullable=True),
        sa.Column("env_vars", _env_vars_json, nullable=True),
        sa.Column("is_builtin", sa.Boolean(), nullable=False, server_default=sa.text("false")),
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
        sa.UniqueConstraint(
            "tenant_id",
            "name",
            name="idx_mcp_services_tenant_name",
        ),
    )
    op.create_index("idx_mcp_services_tenant_id", "mcp_services", ["tenant_id"])
    op.create_index("idx_mcp_services_enabled", "mcp_services", ["enabled"])
    op.create_index("idx_mcp_services_is_builtin", "mcp_services", ["is_builtin"])
    op.create_index("idx_mcp_services_deleted_at", "mcp_services", ["deleted_at"])

    op.create_table(
        "mcp_tool_approvals",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("tenant_id", sa.BigInteger(), nullable=False),
        sa.Column("service_id", sa.String(length=36), nullable=False),
        sa.Column("tool_name", sa.String(length=512), nullable=False),
        sa.Column(
            "require_approval",
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
        sa.UniqueConstraint(
            "tenant_id",
            "service_id",
            "tool_name",
            name="idx_mcp_tool_approvals_tenant_svc_tool",
        ),
        sa.ForeignKeyConstraint(
            ["service_id"],
            ["mcp_services.id"],
            ondelete="CASCADE",
        ),
    )
    op.create_index(
        "idx_mcp_tool_approvals_service_id",
        "mcp_tool_approvals",
        ["service_id"],
    )

    op.create_table(
        "mcp_oauth_clients",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("tenant_id", sa.BigInteger(), nullable=False),
        sa.Column("service_id", sa.String(length=36), nullable=False),
        sa.Column("client_id", sa.String(length=512), nullable=False),
        sa.Column("client_secret", sa.Text(), nullable=True),
        sa.Column("redirect_uri", sa.String(length=1024), nullable=True),
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
        sa.UniqueConstraint(
            "tenant_id",
            "service_id",
            name="idx_mcp_oauth_clients_tenant_svc",
        ),
        sa.ForeignKeyConstraint(
            ["service_id"],
            ["mcp_services.id"],
            ondelete="CASCADE",
        ),
    )
    op.create_index(
        "idx_mcp_oauth_clients_service_id",
        "mcp_oauth_clients",
        ["service_id"],
    )

    op.create_table(
        "mcp_oauth_tokens",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("tenant_id", sa.BigInteger(), nullable=False),
        sa.Column("user_id", sa.String(length=64), nullable=False),
        sa.Column("service_id", sa.String(length=36), nullable=False),
        sa.Column("access_token", sa.Text(), nullable=True),
        sa.Column("refresh_token", sa.Text(), nullable=True),
        sa.Column("token_type", sa.String(length=32), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("refresh_lease_id", sa.String(length=36), nullable=True),
        sa.Column("refresh_lease_until", sa.DateTime(timezone=True), nullable=True),
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
        sa.UniqueConstraint(
            "tenant_id",
            "user_id",
            "service_id",
            name="idx_mcp_oauth_tokens_tenant_user_svc",
        ),
        sa.ForeignKeyConstraint(
            ["service_id"],
            ["mcp_services.id"],
            ondelete="CASCADE",
        ),
    )
    op.create_index(
        "idx_mcp_oauth_tokens_service_id",
        "mcp_oauth_tokens",
        ["service_id"],
    )
    op.create_index(
        "idx_mcp_oauth_tokens_user_id",
        "mcp_oauth_tokens",
        ["user_id"],
    )


def downgrade() -> None:
    op.drop_index("idx_mcp_oauth_tokens_user_id", table_name="mcp_oauth_tokens")
    op.drop_index("idx_mcp_oauth_tokens_service_id", table_name="mcp_oauth_tokens")
    op.drop_table("mcp_oauth_tokens")
    op.drop_index("idx_mcp_oauth_clients_service_id", table_name="mcp_oauth_clients")
    op.drop_table("mcp_oauth_clients")
    op.drop_index(
        "idx_mcp_tool_approvals_service_id",
        table_name="mcp_tool_approvals",
    )
    op.drop_table("mcp_tool_approvals")
    op.drop_index("idx_mcp_services_deleted_at", table_name="mcp_services")
    op.drop_index("idx_mcp_services_is_builtin", table_name="mcp_services")
    op.drop_index("idx_mcp_services_enabled", table_name="mcp_services")
    op.drop_index("idx_mcp_services_tenant_id", table_name="mcp_services")
    op.drop_table("mcp_services")
