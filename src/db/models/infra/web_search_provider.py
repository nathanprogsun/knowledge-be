"""Storage row for the `web_search_providers` table.

Mirrors ``internal/types/web_search_provider.go::WebSearchProviderEntity``.
Each row is a tenant-scoped configuration of an upstream web-search
provider (Bing, Google CSE, DuckDuckGo, Tavily, Ollama, Baidu, SearXNG,
Keenable, Zhipu). Agents reference rows by UUID `id`.

`parameters` is a JSONB blob carrying provider-specific credentials and
options (api_key, engine_id, base_url, proxy_url, extra_config). The
service layer controls what crosses the wire; the row carries everything
that was persisted.

`is_default` is a workspace-level flag — at most one row per tenant may
hold it; the service flips it atomically (clearing prior defaults before
inserting a new one).

`deleted_at` is the soft-delete marker. Mirrors the Go entity's
`gorm.DeletedAt`.
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from src.common.json import JsonObject
from src.common.table_model import TableModel


class WebSearchProvider(TableModel):
    """One row of the `web_search_providers` table."""

    table: ClassVar[str] = "web_search_providers"
    primary_keys: ClassVar[tuple[str, ...]] = ("id",)
    json_columns: ClassVar[tuple[str, ...]] = ("parameters",)
    db_generated_columns: ClassVar[tuple[str, ...]] = ()  # id is caller-assigned (UUID).

    id: str
    tenant_id: int
    name: str
    provider: str
    description: str | None = None
    parameters: JsonObject | None = None
    is_default: bool = False
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None


__all__ = ["WebSearchProvider"]
