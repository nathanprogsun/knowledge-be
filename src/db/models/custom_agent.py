"""Storage row for the `custom_agents` table.

The column shape mirrors the upstream entity captured during the
initial migration; column names match the storage layer exactly. The
``config`` JSONB blob carries the full agent configuration (model
bindings, tool allow-list, knowledge-base selection, suggestion
policy) — typed parsing happens in the agent domain layer.

``id`` is application-assigned: custom agents get a caller-minted UUID,
built-in agents use a fixed preset id. The primary key is composite
``(id, tenant_id)`` so the same built-in id can hold per-tenant
customised configs.

``is_builtin`` is fixed at creation time; the service rejects edits
that would change built-in rows. ``created_by`` records the owning
user id, or stays empty for tenant-owned rows created through the
API-key path.
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from pydantic import Field

from src.common.json import JsonObject
from src.common.table_model import TableModel


class CustomAgent(TableModel):
    """One row of the `custom_agents` table."""

    table: ClassVar[str] = "custom_agents"
    primary_keys: ClassVar[tuple[str, ...]] = ("id", "tenant_id")
    json_columns: ClassVar[tuple[str, ...]] = ("config",)
    # ``id`` is application-assigned (UUID / preset id), so it takes
    # part in the INSERT column list.
    db_generated_columns: ClassVar[tuple[str, ...]] = ()

    id: str
    name: str
    description: str | None = None
    avatar: str | None = None
    is_builtin: bool = False
    tenant_id: int
    created_by: str | None = None
    config: JsonObject = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None


__all__ = ["CustomAgent"]
