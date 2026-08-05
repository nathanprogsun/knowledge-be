"""Create the `models` table.

Mirrors the WeKnora DDL assembled from
``migrations/versioned/000000_init.up.sql`` (initial CREATE TABLE),
``000001_agent.up.sql`` (ADD COLUMN is_builtin + index),
``000052_models_managed_by.up.sql`` (ADD COLUMN managed_by + partial
index), and ``000057_models_display_name.up.sql``
(ADD COLUMN display_name).

`id` is application-assigned (UUID); the Go ``BeforeCreate`` hook
fills it when empty, and the Python service stamps one explicitly.
`tenant_id` is non-nullable. `parameters` is JSONB and stores the
full ``ModelParameters`` blob (including ``api_key`` / ``app_secret``,
encrypted at rest on the Go side). `is_default`, `is_builtin` and
`managed_by` carry the platform-side metadata that the YAML loader
needs to reconcile on startup.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "models"
down_revision: str | None = "0008_tenant_kv"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None

parameters_json = sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")


def upgrade() -> None:
    op.create_table(
        "models",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("tenant_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column(
            "display_name",
            sa.String(length=255),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column("type", sa.String(length=50), nullable=False),
        sa.Column("source", sa.String(length=50), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "parameters",
            parameters_json,
            nullable=False,
        ),
        sa.Column(
            "is_default",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "is_builtin",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "managed_by",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "status",
            sa.String(length=50),
            nullable=False,
            server_default=sa.text("'active'"),
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
    op.create_index("ix_models_tenant_id", "models", ["tenant_id"])
    op.create_index("ix_models_type", "models", ["type"])
    op.create_index("ix_models_source", "models", ["source"])
    op.create_index("ix_models_is_builtin", "models", ["is_builtin"])
    op.create_index("ix_models_deleted_at", "models", ["deleted_at"])
    # Partial index on the YAML-managed slice: keeps the startup
    # sweep cheap even when the table grows large.
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_models_managed_by_yaml "
        "ON models (managed_by) WHERE managed_by <> ''"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_models_managed_by_yaml")
    op.drop_index("ix_models_deleted_at", table_name="models")
    op.drop_index("ix_models_is_builtin", table_name="models")
    op.drop_index("ix_models_source", table_name="models")
    op.drop_index("ix_models_type", table_name="models")
    op.drop_index("ix_models_tenant_id", table_name="models")
    op.drop_table("models")
