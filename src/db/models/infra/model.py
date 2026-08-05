"""Storage row for the `models` table.

Mirrors ``internal/types/model.go`` (``Model`` struct) on the Go side.

Column notes
------------

- ``id`` is application-assigned (string), so it participates in INSERT.
  The Go side defaults a UUID via a ``BeforeCreate`` hook when empty;
  we keep the contract here by letting the service stamp a UUID, so
  the schema is consistent across both code paths.
- ``tenant_id`` is non-nullable; ``is_builtin=true`` rows live in
  tenant 10000 (the Go constant ``DefaultBuiltinModelTenantID``),
  visible to every tenant via the repository's ``(tenant_id = X OR
  is_builtin = true)`` predicate; the cross-tenant visibility helper
  arrives with the built-in loader.
- ``parameters`` is JSONB and carries a free-form ``ModelParameters``
  blob; on the wire it is a structured Pydantic model, but the row
  stores the JSON dump.
- ``deleted_at`` is soft-delete (see ``000000_init.up.sql``).
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from src.common.json import JsonObject
from src.common.table_model import TableModel


class Model(TableModel):
    """One row of the `models` table."""

    table: ClassVar[str] = "models"
    primary_keys: ClassVar[tuple[str, ...]] = ("id",)
    json_columns: ClassVar[tuple[str, ...]] = ("parameters",)
    # ``id`` is application-assigned (string UUID), so it participates in INSERT.
    db_generated_columns: ClassVar[tuple[str, ...]] = ()

    id: str
    tenant_id: int
    name: str
    display_name: str | None = None
    type: str
    source: str
    description: str | None = None
    parameters: JsonObject
    is_default: bool = False
    is_builtin: bool = False
    managed_by: str = ""
    status: str | None = "active"
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None


__all__ = ["Model"]
