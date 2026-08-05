"""Create the `vector_stores` table.

Mirrors ``internal/types/vectorstore.go::VectorStore`` (Go GORM tag
analysis fetches the same column set). Each row is a tenant-scoped
configuration of a vector database (Elasticsearch, Qdrant, Milvus,
Tencent VectorDB, Weaviate, Doris, OpenSearch). Agents reference
vector stores by UUID ``id``.

Connection parameters and index settings are stored as JSONB blobs
(`connection_config` and `index_config`). The wire layer masks
sensitive fields (`password`, `api_key`) before the row crosses the
service boundary.

`source` is the classifier from the Go contract: ``"user"`` for
DB-managed rows; the ``"env"`` value is set by the service when it
synthesises virtual entries from ``RETRIEVE_DRIVER`` and never
persisted. `readonly` mirrors the wire contract for parity.

`deleted_at` is the soft-delete marker. Mirrors the Go entity's
``gorm.DeletedAt``; a partial unique index on `(tenant_id, name)` keeps
live names unique per workspace.

The migration places itself at the head of the chain (down_revision
points to the latest existing migration) so the checkpoint-2 re-number
step only has to renumber the file, not its content.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0011_vector_stores"
down_revision: str | None = "0010_mcp_services"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None

_connection_config_json = sa.JSON().with_variant(
    postgresql.JSONB(astext_type=sa.Text()),
    "postgresql",
)
_index_config_json = sa.JSON().with_variant(
    postgresql.JSONB(astext_type=sa.Text()),
    "postgresql",
)

_LIVE_ROW = sa.text("deleted_at IS NULL")


def upgrade() -> None:
    op.create_table(
        "vector_stores",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("tenant_id", sa.BigInteger(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("engine_type", sa.String(length=50), nullable=False),
        sa.Column(
            "connection_config",
            _connection_config_json,
            nullable=True,
        ),
        sa.Column(
            "index_config",
            _index_config_json,
            nullable=True,
        ),
        sa.Column(
            "source",
            sa.String(length=16),
            nullable=False,
            server_default=sa.text("'user'"),
        ),
        sa.Column(
            "readonly",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
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
        "idx_vector_stores_tenant_id",
        "vector_stores",
        ["tenant_id"],
    )
    op.create_index(
        "idx_vector_stores_tenant_engine",
        "vector_stores",
        ["tenant_id", "engine_type"],
        postgresql_where=_LIVE_ROW,
    )
    op.create_index(
        "uq_vector_stores_tenant_name_live",
        "vector_stores",
        ["tenant_id", "name"],
        unique=True,
        postgresql_where=_LIVE_ROW,
    )


def downgrade() -> None:
    op.drop_index("uq_vector_stores_tenant_name_live", table_name="vector_stores")
    op.drop_index("idx_vector_stores_tenant_engine", table_name="vector_stores")
    op.drop_index("idx_vector_stores_tenant_id", table_name="vector_stores")
    op.drop_table("vector_stores")
