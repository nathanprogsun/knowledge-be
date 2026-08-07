"""Create the `wiki_pages` and `wiki_folders` tables.

Wiki pages are LLM-generated, interlinked markdown documents that form a
persistent wiki for a knowledge base. ``wiki_pages`` mirrors the wiki
page contract: the content and its versioning bookkeeping (``version``,
``last_edit_source``, ``last_editor_id``), the link / source-reference
JSON arrays, and the denormalised directory cache (``folder_id`` /
``category_path`` / ``wiki_path`` / ``depth``).

``folder_id`` is the single source of truth for a page's placement in
the directory tree (empty string = wiki root). ``wiki_folders`` is the
adjacency-list directory tree (``parent_id``, empty string = root);
``path`` is the materialized "/"-joined name chain kept for cheap
display / sort.

Pages are soft-deleted (``deleted_at``); the slug is unique among live
pages within a knowledge base (partial unique index), and a folder name
is unique among its live siblings under the same parent.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0021"
down_revision: str | None = "0020_tags"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    op.create_table(
        "wiki_pages",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("tenant_id", sa.BigInteger(), nullable=False),
        sa.Column("knowledge_base_id", sa.String(length=36), nullable=False),
        sa.Column("slug", sa.String(length=255), nullable=False),
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
        sa.Column("parent_slug", sa.String(length=255), nullable=False, server_default=""),
        sa.Column("folder_id", sa.String(length=36), nullable=False, server_default=""),
        sa.Column(
            "category_path",
            postgresql.JSONB(),
            nullable=True,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("wiki_path", sa.String(length=1024), nullable=False, server_default=""),
        sa.Column("depth", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("sort_order", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column(
            "source_refs",
            postgresql.JSONB(),
            nullable=True,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "chunk_refs",
            postgresql.JSONB(),
            nullable=True,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "in_links",
            postgresql.JSONB(),
            nullable=True,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "out_links",
            postgresql.JSONB(),
            nullable=True,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "page_metadata",
            postgresql.JSONB(),
            nullable=True,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "aliases",
            postgresql.JSONB(),
            nullable=True,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("version", sa.Integer(), nullable=False, server_default=sa.text("1")),
        sa.Column(
            "last_edit_source",
            sa.String(length=16),
            nullable=False,
            server_default="",
        ),
        sa.Column(
            "last_editor_id",
            sa.String(length=64),
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
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )
    # slug unique among live pages within a knowledge base.
    op.create_index(
        "idx_wiki_pages_kb_slug",
        "wiki_pages",
        ["knowledge_base_id", "slug"],
        unique=True,
        postgresql_where=sa.text("deleted_at IS NULL"),
    )
    op.create_index("idx_wiki_pages_kb_id", "wiki_pages", ["knowledge_base_id"])
    op.create_index("idx_wiki_pages_page_type", "wiki_pages", ["knowledge_base_id", "page_type"])
    op.create_index(
        "idx_wiki_pages_parent_slug",
        "wiki_pages",
        ["knowledge_base_id", "parent_slug"],
    )
    op.create_index(
        "idx_wiki_pages_tree",
        "wiki_pages",
        ["knowledge_base_id", "page_type", "wiki_path", "sort_order", "title"],
    )
    op.create_index("idx_wiki_pages_folder", "wiki_pages", ["knowledge_base_id", "folder_id"])
    op.create_index("idx_wiki_pages_tenant_id", "wiki_pages", ["tenant_id"])
    op.create_index("idx_wiki_pages_deleted_at", "wiki_pages", ["deleted_at"])
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_wiki_pages_fulltext ON wiki_pages "
        "USING GIN (to_tsvector('simple', "
        "coalesce(title, '') || ' ' || coalesce(content, '')))"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_wiki_pages_source_refs "
        "ON wiki_pages USING GIN (source_refs jsonb_path_ops)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_wiki_pages_source_refs_text "
        "ON wiki_pages USING GIN (to_tsvector('simple', source_refs::text))"
    )
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_wiki_pages_title_trgm "
        "ON wiki_pages USING GIN (lower(title) gin_trgm_ops)"
    )

    op.create_table(
        "wiki_folders",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("tenant_id", sa.BigInteger(), nullable=False, server_default=sa.text("0")),
        sa.Column("knowledge_base_id", sa.String(length=36), nullable=False),
        sa.Column("parent_id", sa.String(length=36), nullable=False, server_default=""),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("path", sa.String(length=1024), nullable=False, server_default=""),
        sa.Column("depth", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("sort_order", sa.Integer(), nullable=False, server_default=sa.text("0")),
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
    # Folder name unique among live siblings under the same parent.
    op.create_index(
        "idx_wiki_folders_parent_name",
        "wiki_folders",
        ["knowledge_base_id", "parent_id", "name"],
        unique=True,
        postgresql_where=sa.text("deleted_at IS NULL"),
    )
    op.create_index(
        "idx_wiki_folders_parent",
        "wiki_folders",
        ["knowledge_base_id", "parent_id"],
    )
    op.create_index("idx_wiki_folders_deleted_at", "wiki_folders", ["deleted_at"])


def downgrade() -> None:
    op.drop_index("idx_wiki_folders_deleted_at", table_name="wiki_folders")
    op.drop_index("idx_wiki_folders_parent", table_name="wiki_folders")
    op.drop_index("idx_wiki_folders_parent_name", table_name="wiki_folders")
    op.drop_table("wiki_folders")
    op.execute("DROP INDEX IF EXISTS idx_wiki_pages_title_trgm")
    op.execute("DROP INDEX IF EXISTS idx_wiki_pages_source_refs_text")
    op.execute("DROP INDEX IF EXISTS idx_wiki_pages_source_refs")
    op.execute("DROP INDEX IF EXISTS idx_wiki_pages_fulltext")
    op.drop_index("idx_wiki_pages_deleted_at", table_name="wiki_pages")
    op.drop_index("idx_wiki_pages_tenant_id", table_name="wiki_pages")
    op.drop_index("idx_wiki_pages_folder", table_name="wiki_pages")
    op.drop_index("idx_wiki_pages_tree", table_name="wiki_pages")
    op.drop_index("idx_wiki_pages_parent_slug", table_name="wiki_pages")
    op.drop_index("idx_wiki_pages_page_type", table_name="wiki_pages")
    op.drop_index("idx_wiki_pages_kb_id", table_name="wiki_pages")
    op.drop_index("idx_wiki_pages_kb_slug", table_name="wiki_pages")
    op.drop_table("wiki_pages")
