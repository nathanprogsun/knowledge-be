"""Create six new tables and add ``tenants.api_key``.

This migration closes the gap between the Go upstream schema and the
Python port: six tables previously defined only in the upstream SQL
migrations are added here, alongside the legacy per-tenant ``api_key``
column on ``tenants``.

The six tables are:

- ``tenant_disabled_shared_agents`` (per-tenant "disabled by me" toggle
  for shared agents received via the organization share graph);
- ``user_kb_pins`` (per-user pin state for the knowledge-base list,
  replacing the legacy tenant-wide ``knowledge_bases.is_pinned``);
- ``user_resource_favorites`` (per-user starred resources);
- ``wiki_log_entries`` (append-only event log for wiki ingest / retract
  ops);
- ``wiki_page_issues`` (curator-flagged / LLM-flagged page issues);
- ``wiki_page_revisions`` (per-version page snapshots for diff /
  rollback).

The ``tenants.api_key`` column is the legacy machine credential field;
revocable per-principal keys live in ``tenant_api_keys``. The column is
nullable with an empty-string default so existing rows pass validation
without backfill.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0025_pr_156_db_schema"
down_revision: str | None = "0024_knowledge_processing_spans"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    # ── tenants.api_key ──────────────────────────────────────────────
    op.add_column(
        "tenants",
        sa.Column(
            "api_key",
            sa.String(length=255),
            nullable=False,
            server_default="",
        ),
    )

    # ── tenant_disabled_shared_agents ────────────────────────────────
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

    # ── user_kb_pins ────────────────────────────────────────────────
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

    # ── user_resource_favorites ─────────────────────────────────────
    op.create_table(
        "user_resource_favorites",
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("tenant_id", sa.BigInteger(), nullable=False),
        sa.Column("resource_type", sa.String(length=16), nullable=False),
        sa.Column("resource_id", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.PrimaryKeyConstraint(
            "user_id",
            "tenant_id",
            "resource_type",
            "resource_id",
            name="pk_user_resource_favorites",
        ),
    )
    op.create_index(
        "idx_user_resource_favorites_user_tenant_type_created_at",
        "user_resource_favorites",
        [
            "user_id",
            "tenant_id",
            "resource_type",
            sa.text("created_at DESC"),
        ],
    )
    op.create_index(
        "idx_user_resource_favorites_tenant_id",
        "user_resource_favorites",
        ["tenant_id"],
    )

    # ── wiki_log_entries ────────────────────────────────────────────
    op.create_table(
        "wiki_log_entries",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("tenant_id", sa.BigInteger(), nullable=False),
        sa.Column("knowledge_base_id", sa.String(length=36), nullable=False),
        sa.Column("action", sa.String(length=32), nullable=False),
        sa.Column(
            "knowledge_id",
            sa.String(length=36),
            nullable=False,
            server_default="",
        ),
        sa.Column("doc_title", sa.Text(), nullable=False, server_default=""),
        sa.Column("summary", sa.Text(), nullable=False, server_default=""),
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
            server_default=sa.text("NOW()"),
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

    # ── wiki_page_issues ────────────────────────────────────────────
    op.create_table(
        "wiki_page_issues",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("tenant_id", sa.BigInteger(), nullable=False),
        sa.Column("knowledge_base_id", sa.String(length=36), nullable=False),
        sa.Column("slug", sa.String(length=255), nullable=False),
        sa.Column("issue_type", sa.String(length=50), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("suspected_knowledge_ids", postgresql.JSONB(), nullable=True),
        sa.Column(
            "status",
            sa.String(length=20),
            nullable=False,
            server_default="pending",
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
    op.create_index("idx_wiki_page_issues_slug", "wiki_page_issues", ["slug"])
    op.create_index("idx_wiki_page_issues_status", "wiki_page_issues", ["status"])

    # ── wiki_page_revisions ──────────────────────────────────────────
    op.create_table(
        "wiki_page_revisions",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("tenant_id", sa.BigInteger(), nullable=False),
        sa.Column("knowledge_base_id", sa.String(length=36), nullable=False),
        sa.Column("page_id", sa.String(length=36), nullable=False),
        sa.Column("slug", sa.String(length=255), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(length=512), nullable=False, server_default=""),
        sa.Column(
            "page_type",
            sa.String(length=32),
            nullable=False,
            server_default="summary",
        ),
        sa.Column(
            "status",
            sa.String(length=32),
            nullable=False,
            server_default="published",
        ),
        sa.Column("content", sa.Text(), nullable=False, server_default=""),
        sa.Column("summary", sa.Text(), nullable=False, server_default=""),
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
            server_default="",
        ),
        sa.Column(
            "editor_id",
            sa.String(length=64),
            nullable=False,
            server_default="",
        ),
        sa.Column(
            "edited_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.UniqueConstraint("page_id", "version", name="uq_wiki_page_revisions_page_version"),
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
        "idx_wiki_page_issues_knowledge_base_id",
        table_name="wiki_page_issues",
    )
    op.drop_index("idx_wiki_page_issues_tenant_id", table_name="wiki_page_issues")
    op.drop_table("wiki_page_issues")

    op.drop_index("idx_wiki_log_entries_tenant_id", table_name="wiki_log_entries")
    op.drop_index("idx_wiki_log_entries_kb_id_desc", table_name="wiki_log_entries")
    op.drop_table("wiki_log_entries")

    op.drop_index(
        "idx_user_resource_favorites_tenant_id",
        table_name="user_resource_favorites",
    )
    op.drop_index(
        "idx_user_resource_favorites_user_tenant_type_created_at",
        table_name="user_resource_favorites",
    )
    op.drop_table("user_resource_favorites")

    op.drop_index(
        "idx_user_kb_pins_user_tenant_pinned_at",
        table_name="user_kb_pins",
    )
    op.drop_table("user_kb_pins")

    op.drop_index(
        "idx_tenant_disabled_shared_agents_tenant_id",
        table_name="tenant_disabled_shared_agents",
    )
    op.drop_table("tenant_disabled_shared_agents")

    op.drop_column("tenants", "api_key")