"""Create the `web_search_providers` table.

Mirrors ``migrations/versioned/000030_web_search_providers.up.sql``.

Each row is a tenant-scoped configuration of an upstream web-search
provider (Bing, Google CSE, DuckDuckGo, Tavily, Ollama, Baidu, SearXNG,
Keenable, Zhipu). The primary key is a UUID string (Go uses
``uuid.New().String()``); callers (the service layer) generate it
client-side before INSERT so the row carries the id from the start.

`parameters` is JSONB so the schema can flex per provider type
(api_key, engine_id, base_url, proxy_url, extra_config) without ALTER
TABLE churn.

`is_default` is a workspace-level flag — at most one live row per
tenant may hold it; the service flips it atomically via a dedicated
``clear_default`` SQL helper, so a unique partial index is intentionally
NOT created here (that would couple the guarantee to a DB constraint
without giving us anything we don't already enforce).

Placeholder migration number — checkpoint-2 promotes it to a final
revision id.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "web_search_providers"
down_revision: str | None | None = "0008_tenant_kv"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None

_parameters_json = sa.JSON().with_variant(
    postgresql.JSONB(astext_type=sa.Text()),
    "postgresql",
)


def upgrade() -> None:
    op.create_table(
        "web_search_providers",
        sa.Column("id", sa.String(length=36), primary_key=True, nullable=False),
        sa.Column("tenant_id", sa.BigInteger(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("provider", sa.String(length=50), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("parameters", _parameters_json, nullable=True),
        sa.Column(
            "is_default",
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
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.id"],
            name="fk_web_search_providers_tenant",
            ondelete="CASCADE",
        ),
    )
    op.create_index(
        "idx_web_search_providers_tenant_id",
        "web_search_providers",
        ["tenant_id"],
    )
    op.create_index(
        "idx_web_search_providers_provider",
        "web_search_providers",
        ["provider"],
    )
    op.create_index(
        "idx_web_search_providers_deleted_at",
        "web_search_providers",
        ["deleted_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "idx_web_search_providers_deleted_at",
        table_name="web_search_providers",
    )
    op.drop_index(
        "idx_web_search_providers_provider",
        table_name="web_search_providers",
    )
    op.drop_index(
        "idx_web_search_providers_tenant_id",
        table_name="web_search_providers",
    )
    op.drop_table("web_search_providers")
