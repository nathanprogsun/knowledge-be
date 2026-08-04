"""Storage row for the `storage_backends` table.

One row is a concrete file/object storage instance. A workspace may
register several instances of the same provider and bind each knowledge
base to a different one.

Mirrors ``internal/types/storagebackend.go::StorageBackend`` — column
names match the Go ``json``/``gorm`` tags one-for-one. ``config`` is the
JSON-backed normalized union of provider settings; the typed view lives
on ``src.core.infra.storage_backends.types.StorageBackendConfigInfo``.

Column notes
------------

- ``id`` is a UUID string assigned by the service (Go's ``BeforeCreate``
  hook), so it participates in INSERT — ``db_generated_columns`` is
  emptied to keep it in the column list.
- Rows are soft-deleted (``deleted_at``): a legacy alias backend may
  still be referenced by old file paths, so hard deletes are avoided.
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar, Final

from pydantic import Field

from src.common.json import JsonObject
from src.common.table_model import TableModel

# ``source`` values — a row is either user-managed or an env snapshot.
STORAGE_BACKEND_SOURCE_USER: Final = "user"
STORAGE_BACKEND_SOURCE_ENV: Final = "env"

# ``status`` values.
STORAGE_BACKEND_STATUS_ACTIVE: Final = "active"
STORAGE_BACKEND_STATUS_DISABLED: Final = "disabled"


class StorageBackend(TableModel):
    """One row of the `storage_backends` table."""

    table: ClassVar[str] = "storage_backends"
    primary_keys: ClassVar[tuple[str, ...]] = ("id",)
    json_columns: ClassVar[tuple[str, ...]] = ("config",)
    # ``id`` is a service-assigned UUID, not a DB sequence.
    db_generated_columns: ClassVar[tuple[str, ...]] = ()

    id: str
    tenant_id: int
    name: str
    provider: str
    config: JsonObject = Field(default_factory=dict)
    source: str = STORAGE_BACKEND_SOURCE_USER
    status: str = STORAGE_BACKEND_STATUS_ACTIVE
    legacy_alias: bool = False
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None


__all__ = [
    "STORAGE_BACKEND_SOURCE_ENV",
    "STORAGE_BACKEND_SOURCE_USER",
    "STORAGE_BACKEND_STATUS_ACTIVE",
    "STORAGE_BACKEND_STATUS_DISABLED",
    "StorageBackend",
]
