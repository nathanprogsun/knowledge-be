"""Create the `knowledge_bases` table.

Captures the storage shape of the knowledge-base entity: tenant-scoped
rows with JSONB configuration blobs and a soft-delete marker.

``chunking_config`` / ``image_processing_config`` / ``cos_config`` /
``vlm_config`` are NOT NULL with legacy server defaults (the app always
writes its own values on insert); the remaining config blobs are
nullable. ``vector_store_id`` and ``storage_backend_id`` are nullable
bindings to engine instances; ``creator_id`` records who created the
row for the workspace-level RBAC guard.

Indexes mirror the query patterns: tenant-scoped listings and the
composite lookups on (tenant_id, vector_store_id) / (tenant_id,
creator_id) / (tenant_id, storage_backend_id), plus the soft-delete
filter.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0015_knowledge_bases"
down_revision: str | None = "0014_datasources"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    op.create_table(
        "knowledge_bases",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("type", sa.String(length=32), nullable=False, server_default="document"),
        sa.Column(
            "is_temporary",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("tenant_id", sa.BigInteger(), nullable=False),
        sa.Column("creator_id", sa.String(length=36), nullable=True),
        sa.Column(
            "chunking_config",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text(
                r"""'{"chunk_size": 512, "chunk_overlap": 50, "split_markers": ["\n\n", "\n", "。"], "keep_separator": true}'::jsonb"""
            ),
        ),
        sa.Column(
            "image_processing_config",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text(r"""'{"enable_multimodal": false, "model_id": ""}'::jsonb"""),
        ),
        sa.Column("embedding_model_id", sa.String(length=64), nullable=False),
        sa.Column("summary_model_id", sa.String(length=64), nullable=False),
        sa.Column(
            "vlm_config",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("asr_config", postgresql.JSONB(), nullable=True),
        sa.Column("storage_provider_config", postgresql.JSONB(), nullable=True),
        sa.Column("storage_backend_id", sa.String(length=36), nullable=True),
        sa.Column(
            "cos_config",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("vector_store_id", sa.String(length=36), nullable=True),
        sa.Column("extract_config", postgresql.JSONB(), nullable=True),
        sa.Column("faq_config", postgresql.JSONB(), nullable=True),
        sa.Column("question_generation_config", postgresql.JSONB(), nullable=True),
        sa.Column("wiki_config", postgresql.JSONB(), nullable=True),
        sa.Column("indexing_strategy", postgresql.JSONB(), nullable=True),
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
    op.create_index("idx_knowledge_bases_tenant_id", "knowledge_bases", ["tenant_id"])
    op.create_index(
        "idx_knowledge_bases_tenant_vector_store",
        "knowledge_bases",
        ["tenant_id", "vector_store_id"],
    )
    op.create_index(
        "idx_knowledge_bases_tenant_creator",
        "knowledge_bases",
        ["tenant_id", "creator_id"],
    )
    op.create_index(
        "idx_knowledge_bases_storage_backend",
        "knowledge_bases",
        ["tenant_id", "storage_backend_id"],
    )
    op.create_index("idx_knowledge_bases_deleted_at", "knowledge_bases", ["deleted_at"])


def downgrade() -> None:
    op.drop_index("idx_knowledge_bases_deleted_at", table_name="knowledge_bases")
    op.drop_index(
        "idx_knowledge_bases_storage_backend",
        table_name="knowledge_bases",
    )
    op.drop_index(
        "idx_knowledge_bases_tenant_creator",
        table_name="knowledge_bases",
    )
    op.drop_index(
        "idx_knowledge_bases_tenant_vector_store",
        table_name="knowledge_bases",
    )
    op.drop_index("idx_knowledge_bases_tenant_id", table_name="knowledge_bases")
    op.drop_table("knowledge_bases")
