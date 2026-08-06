"""Storage row for the `system_settings` table.

The table holds platform-wide tunables (NOT tenant-scoped), gated
by SystemAdmin.

``value`` is JSONB so the same column can hold ints / strings /
booleans / arrays; ``value_type`` tells the service how to decode the
raw bytes. ``Enum`` and ``LastModifiedByName`` are NOT persisted —
they are derived per request by the service / handler.

Rows are intentionally NOT seeded by the migration — for migrated
deployments a DB row has higher precedence than ENV, so inserting
built-in defaults would silently override existing operator
configuration. The service exposes registry-backed virtual rows until
an admin explicitly saves a value.
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from src.common.json import JsonObject, JsonValue
from src.common.table_model import TableModel


class SystemSetting(TableModel):
    """One row of the ``system_settings`` table."""

    table: ClassVar[str] = "system_settings"
    primary_keys: ClassVar[tuple[str, ...]] = ("id",)
    json_columns: ClassVar[tuple[str, ...]] = ("value",)

    id: int
    key: str
    value: JsonObject | list[JsonValue] | str | int | bool
    value_type: str
    category: str
    description: str = ""
    is_secret: bool = False
    requires_restart: bool = False
    last_modified_by: str = ""
    created_at: datetime
    updated_at: datetime


__all__ = ["SystemSetting"]
