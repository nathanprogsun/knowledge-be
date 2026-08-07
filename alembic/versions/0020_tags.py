"""Create the `tags` and `document_tags` tables.

A tag belongs to exactly one knowledge base and is scoped by tenant.
``seq_id`` is a DB-sequence integer id exposed to external APIs; the
sequence starts high so freshly minted ids never collide with legacy
FAQ data. Tag names are unique within a knowledge base (the unique
index on ``(tenant_id, knowledge_base_id, name)``).

``document_tags`` is the many-to-many association between a document
knowledge entry and a tag. Its composite primary key makes re-binding
idempotent; the ``knowledge_id`` / ``tag_id`` indexes back the two
join directions.

Tags and their associations are hard-deleted (no ``deleted_at``
column); unbinding removes the association row outright.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0020_tags"
down_revision: str | None = "0019_chunk_revisions"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None

_TAG_SEQ_NAME = "tags_seq_id_seq"
_TAG_SEQ_START = "10000000"


def upgrade() -> None:
    op.execute(f"CREATE SEQUENCE {_TAG_SEQ_NAME} START WITH {_TAG_SEQ_START}")

    op.create_table(
        "tags",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "seq_id",
            sa.BigInteger(),
            nullable=False,
            server_default=sa.text(f"nextval('{_TAG_SEQ_NAME}')"),
        ),
        sa.Column("tenant_id", sa.BigInteger(), nullable=False),
        sa.Column("knowledge_base_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("color", sa.String(length=32), nullable=True),
        sa.Column(
            "sort_order",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
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
    op.create_index("idx_tags_seq_id", "tags", ["seq_id"], unique=True)
    op.create_index(
        "idx_tags_tenant_kb_name",
        "tags",
        ["tenant_id", "knowledge_base_id", "name"],
        unique=True,
    )
    op.create_index("idx_tags_tenant_kb", "tags", ["tenant_id", "knowledge_base_id"])

    op.create_table(
        "document_tags",
        sa.Column("knowledge_id", sa.String(length=36), nullable=False),
        sa.Column("tag_id", sa.String(length=36), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.PrimaryKeyConstraint("knowledge_id", "tag_id", name="pk_document_tags"),
    )
    op.create_index("idx_document_tags_knowledge_id", "document_tags", ["knowledge_id"])
    op.create_index("idx_document_tags_tag_id", "document_tags", ["tag_id"])


def downgrade() -> None:
    op.drop_index("idx_document_tags_tag_id", table_name="document_tags")
    op.drop_index("idx_document_tags_knowledge_id", table_name="document_tags")
    op.drop_table("document_tags")

    op.drop_index("idx_tags_tenant_kb", table_name="tags")
    op.drop_index("idx_tags_tenant_kb_name", table_name="tags")
    op.drop_index("idx_tags_seq_id", table_name="tags")
    op.drop_table("tags")

    op.execute(f"DROP SEQUENCE {_TAG_SEQ_NAME}")
