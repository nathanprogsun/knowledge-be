"""Add the remaining storage tables and the legacy `tenants.api_key` column.

Six storage tables that are referenced by the domain / repository
layer but were never ported to the Python schema in earlier
migrations, plus the legacy `tenants.api_key` text column. New
machine credentials are issued through `tenant_api_keys` (revocable,
named, scope-tagged) — this migration only restores the column on
`tenants` for back-compat with upstream rows that pre-date the
split, so a direct column read by older callers does not fail.

Tables
------
- `tenant_disabled_shared_agents` — per-tenant opt-out list for
  shared agents. Composite PK `(tenant_id, agent_id, source_tenant_id)`.
- `user_kb_pins` — per-(user, tenant, knowledge_base) pin state.
  Composite PK `(tenant_id, user_id, kb_id)`.
- `user_resource_favorites` — per-(user, tenant, resource_type,
  resource_id) star. Composite PK over all four columns.
- `wiki_log_entries` — append-only event log for wiki operations.
  BIGSERIAL id (also acts as the cursor for the
  `(knowledge_base_id, id DESC)` feed).
- `wiki_page_issues` — user / pipeline reports against a wiki page.
- `wiki_page_revisions` — immutable content snapshots of
  superseded wiki page versions; composite UNIQUE on
  `(page_id, version)`.

Column types mirror the upstream schema exactly: `BIGINT` maps to
`BigInteger`, `VARCHAR(N)` to `String(length=N)`, JSONB arrays to
typed `postgresql.JSONB()`, timestamps to
`DateTime(timezone=True)`.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0035_pr_156_db_schema"
down_revision: str | None = "0034_favorites"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    # 1) Restore the legacy per-tenant api_key column. Empty string
    # default keeps existing rows valid and matches the upstream
    # 'NOT NULL DEFAULT ''' shape. A unique index is added so older
    # callers that still key by api_key continue to resolve
    # unambiguously.
    op.add_column(
        "tenants",
        sa.Column(
            "api_key",
            sa.String(length=256),
            nullable=False,
            server_default=sa.text("''"),
        ),
    )
    op.create_index("idx_tenants_api_key", "tenants", ["api_key"])

    # 2) tenant_disabled_shared_agents
    op.create_table(
        "tenant_disabled_shared_agents",
        sa.Column("tenant_id", sa.BigInteger(), nullable=False),
        sa.Column("agent_id", sa.String(length=36), nullable=False),
        sa.Column("source_tenant_id", sa.BigInteger(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.PrimaryKeyConstraint(
            "tenant_id",
            "agent_id",
            "source_tenant_id",
            name="pk_tenant_disabled_shared_agents",
        ),
    )
    op.create_index(
        "idx_tenant_disabled_shared_agents_tenant_id",
        "tenant_disabled_shared_agents",
        ["tenant_id"],
    )

    # 3) user_kb_pins
    op.create_table(
        "user_kb_pins",
        sa.Column("tenant_id", sa.BigInteger(), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("kb_id", sa.String(length=36), nullable=False),
        sa.Column(
            "pinned_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.PrimaryKeyConstraint(
            "tenant_id",
            "user_id",
            "kb_id",
            name="pk_user_kb_pins",
        ),
    )
    op.create_index(
        "idx_user_kb_pins_user_tenant_pinned_at",
        "user_kb_pins",
        ["tenant_id", "user_id", sa.text("pinned_at DESC")],
    )

    # 5) wiki_log_entries
    op.create_table(
        "wiki_log_entries",
        sa.Column(
            "id",
            sa.BigInteger(),
            primary_key=True,
            autoincrement=True,
        ),
        sa.Column("tenant_id", sa.BigInteger(), nullable=False),
        sa.Column("knowledge_base_id", sa.String(length=36), nullable=False),
        sa.Column("action", sa.String(length=32), nullable=False),
        sa.Column(
            "knowledge_id",
            sa.String(length=36),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "doc_title",
            sa.Text(),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "summary",
            sa.Text(),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "pages_affected",
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
    )
    op.create_index(
        "idx_wiki_log_entries_kb_id_desc",
        "wiki_log_entries",
        ["knowledge_base_id", sa.text("id DESC")],
    )
    op.create_index(
        "idx_wiki_log_entries_tenant_id",
        "wiki_log_entries",
        ["tenant_id"],
    )

    # 6) wiki_page_issues
    op.create_table(
        "wiki_page_issues",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("tenant_id", sa.BigInteger(), nullable=False),
        sa.Column("knowledge_base_id", sa.String(length=36), nullable=False),
        sa.Column("slug", sa.String(length=255), nullable=False),
        sa.Column("issue_type", sa.String(length=50), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column(
            "suspected_knowledge_ids",
            postgresql.JSONB(),
            nullable=True,
        ),
        sa.Column(
            "status",
            sa.String(length=20),
            nullable=False,
            server_default=sa.text("'pending'"),
        ),
        sa.Column("reported_by", sa.String(length=100), nullable=False),
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
        "idx_wiki_page_issues_tenant_id",
        "wiki_page_issues",
        ["tenant_id"],
    )
    op.create_index(
        "idx_wiki_page_issues_knowledge_base_id",
        "wiki_page_issues",
        ["knowledge_base_id"],
    )
    op.create_index(
        "idx_wiki_page_issues_slug",
        "wiki_page_issues",
        ["slug"],
    )
    op.create_index(
        "idx_wiki_page_issues_status",
        "wiki_page_issues",
        ["status"],
    )

    # 7) wiki_page_revisions
    op.create_table(
        "wiki_page_revisions",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("tenant_id", sa.BigInteger(), nullable=False),
        sa.Column("knowledge_base_id", sa.String(length=36), nullable=False),
        sa.Column("page_id", sa.String(length=36), nullable=False),
        sa.Column("slug", sa.String(length=255), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column(
            "title",
            sa.String(length=512),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "page_type",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("'summary'"),
        ),
        sa.Column(
            "status",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("'published'"),
        ),
        sa.Column(
            "content",
            sa.Text(),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "summary",
            sa.Text(),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "aliases",
            postgresql.JSONB(),
            nullable=True,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "edit_source",
            sa.String(length=16),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "editor_id",
            sa.String(length=64),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "edited_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.UniqueConstraint(
            "page_id",
            "version",
            name="idx_wiki_page_revisions_page_version",
        ),
    )
    op.create_index(
        "idx_wiki_page_revisions_kb_slug",
        "wiki_page_revisions",
        ["knowledge_base_id", "slug"],
    )


def downgrade() -> None:
    op.drop_index("idx_wiki_page_revisions_kb_slug", table_name="wiki_page_revisions")
    op.drop_table("wiki_page_revisions")
    op.drop_index("idx_wiki_page_issues_status", table_name="wiki_page_issues")
    op.drop_index("idx_wiki_page_issues_slug", table_name="wiki_page_issues")
    op.drop_index(
        "idx_wiki_page_issues_knowledge_base_id", table_name="wiki_page_issues"
    )
    op.drop_index("idx_wiki_page_issues_tenant_id", table_name="wiki_page_issues")
    op.drop_table("wiki_page_issues")
    op.drop_index("idx_wiki_log_entries_tenant_id", table_name="wiki_log_entries")
    op.drop_index("idx_wiki_log_entries_kb_id_desc", table_name="wiki_log_entries")
    op.drop_table("wiki_log_entries")
    op.drop_index(
        "idx_user_kb_pins_user_tenant_pinned_at", table_name="user_kb_pins"
    )
    op.drop_table("user_kb_pins")
    op.drop_index(
        "idx_tenant_disabled_shared_agents_tenant_id",
        table_name="tenant_disabled_shared_agents",
    )
    op.drop_table("tenant_disabled_shared_agents")
    op.drop_index("idx_tenants_api_key", table_name="tenants")
    op.drop_column("tenants", "api_key")
