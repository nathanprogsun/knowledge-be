"""Create the `faq` table.

One row is one FAQ entry of a knowledge base. The column set mirrors the
FAQ entry shape of the upstream contract: the standard question, the
similar / negative question aliases and answers (JSONB lists), the
answer strategy, the entry-level flags, and the scope columns. The
search-only result fields are computed at query time and are not
columns.

``id`` is a database-assigned identity (the entry sequence id). Rows are
tenant-scoped and belong to exactly one knowledge base; ``chunk_id`` is
unique so the backing chunk reference can be resolved back to its entry.
Entries are hard-deleted, so there is no ``deleted_at`` column.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0017_faq"
down_revision: str | None = "0016_documents"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    op.create_table(
        "faq",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("tenant_id", sa.BigInteger(), nullable=False),
        sa.Column("chunk_id", sa.String(length=36), nullable=False),
        sa.Column("knowledge_id", sa.String(length=36), nullable=False),
        sa.Column("knowledge_base_id", sa.String(length=36), nullable=False),
        sa.Column("tag_id", sa.BigInteger(), nullable=True),
        sa.Column("tag_name", sa.String(length=255), nullable=True),
        sa.Column(
            "is_enabled",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("true"),
        ),
        sa.Column(
            "is_recommended",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column("standard_question", sa.Text(), nullable=False),
        sa.Column(
            "similar_questions",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "negative_questions",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "answers",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "answer_strategy",
            sa.String(length=16),
            nullable=False,
            server_default="all",
        ),
        sa.Column("index_mode", sa.String(length=32), nullable=True),
        sa.Column(
            "chunk_type",
            sa.String(length=32),
            nullable=False,
            server_default="faq",
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
    op.create_index("idx_faq_tenant_id", "faq", ["tenant_id"])
    op.create_index("idx_faq_knowledge_base_id", "faq", ["knowledge_base_id"])
    op.create_index("idx_faq_knowledge_id", "faq", ["knowledge_id"])
    op.create_index("uq_faq_chunk_id", "faq", ["chunk_id"], unique=True)


def downgrade() -> None:
    op.drop_index("uq_faq_chunk_id", table_name="faq")
    op.drop_index("idx_faq_knowledge_id", table_name="faq")
    op.drop_index("idx_faq_knowledge_base_id", table_name="faq")
    op.drop_index("idx_faq_tenant_id", table_name="faq")
    op.drop_table("faq")
