"""Create the `chunk_revisions` table.

Immutable snapshots of superseded chunk revisions: each user edit or
rollback appends one row, while the current content stays on the
`chunks` row. The unique index ``(chunk_id, revision)`` guarantees at
most one snapshot per revision number; ``(tenant_id, chunk_id)`` backs
the newest-first history listing.

The ``revision`` id and ``down_revision`` link are assigned by the PR
that sequences this migration after the `chunks` table migration.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0019_chunk_revisions"
down_revision: str | None = "0018_chunks"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    op.create_table(
        "chunk_revisions",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("tenant_id", sa.BigInteger(), nullable=False),
        sa.Column("knowledge_base_id", sa.String(length=36), nullable=False),
        sa.Column("knowledge_id", sa.String(length=36), nullable=False),
        sa.Column("chunk_id", sa.String(length=36), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False, server_default=""),
        sa.Column(
            "is_enabled",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("true"),
        ),
        sa.Column("editor_id", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("edit_source", sa.String(length=16), nullable=False, server_default="user"),
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
    )
    op.create_index(
        "idx_chunk_revisions_chunk_revision",
        "chunk_revisions",
        ["chunk_id", "revision"],
        unique=True,
    )
    op.create_index(
        "idx_chunk_revisions_tenant_chunk",
        "chunk_revisions",
        ["tenant_id", "chunk_id"],
    )


def downgrade() -> None:
    op.drop_index("idx_chunk_revisions_tenant_chunk", table_name="chunk_revisions")
    op.drop_index("idx_chunk_revisions_chunk_revision", table_name="chunk_revisions")
    op.drop_table("chunk_revisions")
