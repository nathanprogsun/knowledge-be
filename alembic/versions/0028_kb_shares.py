"""Create the `kb_shares` table.

One row records that a knowledge base was shared into an organization
for cross-tenant collaboration. ``source_tenant_id`` is the tenant that
owns the knowledge base; ``permission`` is the org-level grant
(admin / editor / viewer).

The partial unique index allows exactly one live share per
(knowledge base, organization) pair while letting soft-deleted rows
accumulate as history. Indexes mirror the query shapes: the per-KB
share list, the per-org share list, and the source-tenant sweep.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0028_kb_shares"
down_revision: str | None = "0027_organizations"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None

_LIVE_ROW = sa.text("deleted_at IS NULL")


def upgrade() -> None:
    op.create_table(
        "kb_shares",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("knowledge_base_id", sa.String(length=36), nullable=False),
        sa.Column("organization_id", sa.String(length=36), nullable=False),
        sa.Column("shared_by_user_id", sa.String(length=36), nullable=False),
        sa.Column("source_tenant_id", sa.BigInteger(), nullable=False),
        sa.Column(
            "permission",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("'viewer'"),
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
        "idx_kb_shares_kb_org",
        "kb_shares",
        ["knowledge_base_id", "organization_id"],
        unique=True,
        postgresql_where=_LIVE_ROW,
    )
    op.create_index("idx_kb_shares_kb_id", "kb_shares", ["knowledge_base_id"])
    op.create_index("idx_kb_shares_org_id", "kb_shares", ["organization_id"])
    op.create_index("idx_kb_shares_source_tenant", "kb_shares", ["source_tenant_id"])
    op.create_index(
        "idx_kb_shares_deleted_at",
        "kb_shares",
        ["deleted_at"],
        postgresql_where=_LIVE_ROW,
    )


def downgrade() -> None:
    op.drop_index("idx_kb_shares_deleted_at", table_name="kb_shares")
    op.drop_index("idx_kb_shares_source_tenant", table_name="kb_shares")
    op.drop_index("idx_kb_shares_org_id", table_name="kb_shares")
    op.drop_index("idx_kb_shares_kb_id", table_name="kb_shares")
    op.drop_index("idx_kb_shares_kb_org", table_name="kb_shares")
    op.drop_table("kb_shares")
