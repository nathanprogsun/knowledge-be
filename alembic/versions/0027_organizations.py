"""Create the organization collaboration tables.

Three tables back the cross-tenant sharing domain:

- ``organizations`` — the collaboration space itself. ``owner_tenant_id``
  is pinned at creation time (never updated) so the owning workspace can
  never be orphaned. ``invite_code`` is unique only among live rows that
  actually carry one.
- ``organization_tenant_members`` — tenant-scoped membership. The
  (org, tenant) tuple is unique; ``representative_user_id`` is
  display/audit only.
- ``organization_join_requests`` — join / role-upgrade requests. The
  partial unique index allows exactly one pending request per
  (org, tenant, request_type) while letting approved / rejected rows
  accumulate as history.

Indexes mirror the query shapes: the tenant's org list (join on
membership), the discovery list (``searchable``), the review inbox
(org + status), and the pending-request dedup.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0027_organizations"
down_revision: str | None = "0024_knowledge_processing_spans"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None

_LIVE_ROW = sa.text("deleted_at IS NULL")
_LIVE_INVITE_CODE_ROW = sa.text("invite_code IS NOT NULL AND deleted_at IS NULL")
_PENDING_REQUEST_ROW = sa.text("status = 'pending'")


def upgrade() -> None:
    op.create_table(
        "organizations",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("owner_id", sa.String(length=36), nullable=False),
        sa.Column("owner_tenant_id", sa.BigInteger(), nullable=False),
        sa.Column("invite_code", sa.String(length=32), nullable=True),
        sa.Column("invite_code_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "invite_code_validity_days",
            sa.SmallInteger(),
            nullable=False,
            server_default=sa.text("7"),
        ),
        sa.Column(
            "avatar",
            sa.String(length=512),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "require_approval",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "searchable",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "member_limit",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("50"),
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
    op.create_index(
        "idx_organizations_invite_code",
        "organizations",
        ["invite_code"],
        unique=True,
        postgresql_where=_LIVE_INVITE_CODE_ROW,
    )
    op.create_index("idx_organizations_owner_id", "organizations", ["owner_id"])
    op.create_index("idx_organizations_owner_tenant", "organizations", ["owner_tenant_id"])
    op.create_index(
        "idx_organizations_deleted_at",
        "organizations",
        ["deleted_at"],
        postgresql_where=_LIVE_ROW,
    )

    op.create_table(
        "organization_tenant_members",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("organization_id", sa.String(length=36), nullable=False),
        sa.Column("tenant_id", sa.BigInteger(), nullable=False),
        sa.Column(
            "role",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("'viewer'"),
        ),
        sa.Column(
            "representative_user_id",
            sa.String(length=36),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column("joined_at", sa.DateTime(timezone=True), nullable=True),
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
    op.create_index(
        "idx_org_tenant_members_unique",
        "organization_tenant_members",
        ["organization_id", "tenant_id"],
        unique=True,
    )
    op.create_index(
        "idx_org_tenant_members_by_tenant",
        "organization_tenant_members",
        ["tenant_id"],
    )
    op.create_index(
        "idx_org_tenant_members_role",
        "organization_tenant_members",
        ["organization_id", "role"],
    )

    op.create_table(
        "organization_join_requests",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("organization_id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("tenant_id", sa.BigInteger(), nullable=False),
        sa.Column(
            "status",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("'pending'"),
        ),
        sa.Column(
            "requested_role",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("'viewer'"),
        ),
        sa.Column(
            "request_type",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("'join'"),
        ),
        sa.Column("prev_role", sa.String(length=32), nullable=True),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column("reviewed_by", sa.String(length=36), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("review_message", sa.Text(), nullable=True),
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
    op.create_index(
        "uq_org_join_requests_pending_per_tenant",
        "organization_join_requests",
        ["organization_id", "tenant_id", "request_type"],
        unique=True,
        postgresql_where=_PENDING_REQUEST_ROW,
    )
    op.create_index(
        "idx_org_join_requests_org_id",
        "organization_join_requests",
        ["organization_id"],
    )
    op.create_index(
        "idx_org_join_requests_user_id",
        "organization_join_requests",
        ["user_id"],
    )
    op.create_index(
        "idx_org_join_requests_status",
        "organization_join_requests",
        ["status"],
    )
    op.create_index(
        "idx_org_join_requests_type",
        "organization_join_requests",
        ["request_type"],
    )


def downgrade() -> None:
    op.drop_index("idx_org_join_requests_type", table_name="organization_join_requests")
    op.drop_index("idx_org_join_requests_status", table_name="organization_join_requests")
    op.drop_index("idx_org_join_requests_user_id", table_name="organization_join_requests")
    op.drop_index("idx_org_join_requests_org_id", table_name="organization_join_requests")
    op.drop_index(
        "uq_org_join_requests_pending_per_tenant",
        table_name="organization_join_requests",
    )
    op.drop_table("organization_join_requests")
    op.drop_index("idx_org_tenant_members_role", table_name="organization_tenant_members")
    op.drop_index("idx_org_tenant_members_by_tenant", table_name="organization_tenant_members")
    op.drop_index("idx_org_tenant_members_unique", table_name="organization_tenant_members")
    op.drop_table("organization_tenant_members")
    op.drop_index("idx_organizations_deleted_at", table_name="organizations")
    op.drop_index("idx_organizations_owner_tenant", table_name="organizations")
    op.drop_index("idx_organizations_owner_id", table_name="organizations")
    op.drop_index("idx_organizations_invite_code", table_name="organizations")
    op.drop_table("organizations")
