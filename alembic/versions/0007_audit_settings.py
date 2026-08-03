"""Create the `audit_logs` and `system_settings` tables.

Consolidates the upstream migrations:

  - `migrations/versioned/000044_audit_log.up.sql` — initial
    `audit_logs` table (generic per-tenant audit feed).
  - `migrations/versioned/000073_kb_activity_scope.up.sql` — adds
    `scope_type` / `scope_id` columns to `audit_logs`.
  - `migrations/versioned/000053_system_admin_and_settings.up.sql` —
    the `system_settings` table (platform-wide tunables gated by
    SystemAdmin).

The Python rewrite starts from a clean schema, so all three land in a
single migration rather than three sequential ones.

`audit_logs` is append-only (no `updated_at`, no soft-delete column).
The monotonic `id` (BIGSERIAL) doubles as the cursor for the
newest-first paginated feed. `tenant_id = 0` is the system-scope
convention used by `system.setting_changed`, admin promote/revoke, and
the apply-default-storage-quota bulk write — those rows live outside
any tenant's feed and surface only through the
`GET /system/admin/audit-log` endpoint.

`system_settings.value` is JSONB so the same column can hold
ints / strings / booleans / arrays; `value_type` tells the service how
to decode the raw bytes. Rows are intentionally NOT seeded here — for
migrated deployments a DB row has higher precedence than ENV, so
inserting built-in defaults would silently override existing operator
configuration. The service exposes registry-backed virtual rows until
an admin explicitly saves a value.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0007_audit_settings"
down_revision: str | None = "0006_tenant_invitations"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None

_details_json = sa.JSON().with_variant(
    postgresql.JSONB(astext_type=sa.Text()),
    "postgresql",
)
_value_json = sa.JSON().with_variant(
    postgresql.JSONB(astext_type=sa.Text()),
    "postgresql",
)


def upgrade() -> None:
    # ── audit_logs ────────────────────────────────────────────────────
    op.create_table(
        "audit_logs",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("tenant_id", sa.BigInteger(), nullable=False),
        sa.Column("actor_user_id", sa.String(length=36), nullable=False, server_default=""),
        sa.Column("actor_role", sa.String(length=32), nullable=False, server_default=""),
        sa.Column("action", sa.String(length=64), nullable=False),
        sa.Column("scope_type", sa.String(length=32), nullable=False, server_default=""),
        sa.Column("scope_id", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("target_type", sa.String(length=32), nullable=False, server_default=""),
        sa.Column("target_id", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("target_user_id", sa.String(length=36), nullable=False, server_default=""),
        sa.Column("request_path", sa.String(length=512), nullable=False, server_default=""),
        sa.Column("request_method", sa.String(length=16), nullable=False, server_default=""),
        sa.Column("outcome", sa.String(length=16), nullable=False, server_default="success"),
        sa.Column("details", _details_json, nullable=False, server_default=sa.text("'{}'")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
    )
    op.create_index(
        "idx_audit_logs_tenant_id_desc",
        "audit_logs",
        ["tenant_id", sa.text("id DESC")],
    )
    op.create_index("idx_audit_logs_actor", "audit_logs", ["actor_user_id"])
    op.create_index(
        "idx_audit_logs_tenant_action",
        "audit_logs",
        ["tenant_id", "action"],
    )
    op.create_index("idx_audit_logs_created_at", "audit_logs", ["created_at"])
    op.create_index(
        "idx_audit_logs_tenant_scope_desc",
        "audit_logs",
        ["tenant_id", "scope_type", "scope_id", sa.text("id DESC")],
    )

    # ── system_settings ───────────────────────────────────────────────
    op.create_table(
        "system_settings",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("key", sa.String(length=128), nullable=False, unique=True),
        sa.Column("value", _value_json, nullable=False),
        sa.Column("value_type", sa.String(length=16), nullable=False),
        sa.Column("category", sa.String(length=32), nullable=False),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("is_secret", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column(
            "requires_restart",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "last_modified_by",
            sa.String(length=36),
            nullable=False,
            server_default="",
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
    )
    op.create_index("idx_system_settings_category", "system_settings", ["category"])


def downgrade() -> None:
    op.drop_index("idx_system_settings_category", table_name="system_settings")
    op.drop_table("system_settings")
    op.drop_index("idx_audit_logs_tenant_scope_desc", table_name="audit_logs")
    op.drop_index("idx_audit_logs_created_at", table_name="audit_logs")
    op.drop_index("idx_audit_logs_tenant_action", table_name="audit_logs")
    op.drop_index("idx_audit_logs_actor", table_name="audit_logs")
    op.drop_index("idx_audit_logs_tenant_id_desc", table_name="audit_logs")
    op.drop_table("audit_logs")
