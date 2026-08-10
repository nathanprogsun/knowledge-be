"""Create the `agent_shares` table.

One row records that a custom agent was shared into an organization for
cross-tenant collaboration. ``source_tenant_id`` is the tenant that owns
the agent; ``permission`` is the org-level grant (admin / editor /
viewer).

The partial unique index allows exactly one live share per
(agent, source tenant, organization) tuple while letting soft-deleted
rows accumulate as history. Indexes mirror the query shapes: the
per-agent share list, the per-org share list, and the source-tenant
sweep.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0030_agent_shares"
down_revision: str | None = "0029_kb_shares"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None

_LIVE_ROW = sa.text("deleted_at IS NULL")


def upgrade() -> None:
    op.create_table(
        "agent_shares",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("agent_id", sa.String(length=36), nullable=False),
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
        "idx_agent_shares_agent_org",
        "agent_shares",
        ["agent_id", "source_tenant_id", "organization_id"],
        unique=True,
        postgresql_where=_LIVE_ROW,
    )
    op.create_index("idx_agent_shares_agent_id", "agent_shares", ["agent_id"])
    op.create_index("idx_agent_shares_org_id", "agent_shares", ["organization_id"])
    op.create_index("idx_agent_shares_source_tenant", "agent_shares", ["source_tenant_id"])
    op.create_index(
        "idx_agent_shares_deleted_at",
        "agent_shares",
        ["deleted_at"],
        postgresql_where=_LIVE_ROW,
    )


def downgrade() -> None:
    op.drop_index("idx_agent_shares_deleted_at", table_name="agent_shares")
    op.drop_index("idx_agent_shares_source_tenant", table_name="agent_shares")
    op.drop_index("idx_agent_shares_org_id", table_name="agent_shares")
    op.drop_index("idx_agent_shares_agent_id", table_name="agent_shares")
    op.drop_index("idx_agent_shares_agent_org", table_name="agent_shares")
    op.drop_table("agent_shares")
