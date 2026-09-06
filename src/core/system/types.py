"""Internal DTOs for the system domain.

These are service-output projections, not the HTTP wire shape. The
HTTP wire shape is in ``src/core/contracts/system.py``.

- ``AuditLogInfo`` mirrors ``internal/types/audit_log.go::AuditLog``
  projected to the wire (no storage-only columns — there are none
  beyond ``id``; the projection exists for future stripping of
  sensitive ``details`` payloads).
- ``SystemSettingInfo`` mirrors ``internal/types/system_setting.go::SystemSetting``
  with the two non-persisted display fields (``enum``,
  ``last_modified_by_name``) populated by the service from the
  in-code registry and the user service respectively.

Both expose ``map_from_db`` per AGENTS.md §9.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Self, cast

from pydantic import BaseModel, ConfigDict, Field

from src.common.json import JsonObject, JsonValue
from src.db.models.system.audit_log import AuditLog
from src.db.models.system.system_setting import SystemSetting
from src.db.models.user_resource_favorite import UserResourceFavorite


class AuditLogInfo(BaseModel):
    """Wire-side projection of an ``audit_logs`` row."""

    model_config = ConfigDict(frozen=True)

    id: int
    tenant_id: int
    actor_user_id: str = ""
    actor_role: str = ""
    action: str
    scope_type: str = ""
    scope_id: str = ""
    target_type: str = ""
    target_id: str = ""
    target_user_id: str = ""
    request_path: str = ""
    request_method: str = ""
    outcome: str = "success"
    details: JsonObject = Field(default_factory=dict)
    created_at: datetime

    @classmethod
    def from_json(cls, raw: JsonObject | str | None) -> JsonObject:
        """Decode the ``details`` JSON column.

        Accepts both a parsed ``dict`` and a raw JSON string (SQLite
        persists some JSON columns as text). ``None`` / empty /
        unparseable input yields an empty object.
        """
        if raw is None or raw == "":
            return {}
        if isinstance(raw, str):
            try:
                decoded = json.loads(raw)
            except json.JSONDecodeError:
                return {}
            return decoded if isinstance(decoded, dict) else {}
        return raw

    @classmethod
    def map_from_db(cls, db: AuditLog) -> Self:
        record = db.model_dump()
        record["details"] = AuditLogInfo.from_json(record.get("details"))
        return cls.model_validate(record)


class SystemSettingInfo(BaseModel):
    """Wire-side projection of a ``system_settings`` row.

    ``enum`` and ``last_modified_by_name`` are NOT persisted — they are
    derived per request by the service (registry + user lookup).
    """

    model_config = ConfigDict(frozen=True)

    id: int
    key: str
    value: JsonObject | list[str] | list[JsonValue] | str | int | bool
    value_type: str
    category: str
    description: str = ""
    is_secret: bool = False
    requires_restart: bool = False
    last_modified_by: str = ""
    created_at: datetime
    updated_at: datetime
    enum: list[str] = Field(default_factory=list)
    last_modified_by_name: str = ""

    @classmethod
    def from_json(
        cls,
        raw: JsonObject | list[str] | list[JsonValue] | str | int | bool | None,
    ) -> JsonObject | list[str] | list[JsonValue] | str | int | bool:
        """Decode the ``value`` column, mirroring the SQLite round-trip.

        Persisted scalars (int / bool / string) round-trip as strings
        when the ``value_type`` is non-JSON; only strings that look like
        a JSON document (``{`` / ``[``) are parsed. ``None`` yields an
        empty object.
        """
        if raw is None:
            return {}
        if isinstance(raw, str):
            stripped = raw.strip()
            if stripped.startswith(("{", "[")):
                try:
                    return cast(
                        "JsonObject | list[str] | list[JsonValue] | str | int | bool",
                        json.loads(stripped),
                    )
                except json.JSONDecodeError:
                    return raw
            return raw
        return raw

    @classmethod
    def map_from_db(
        cls,
        db: SystemSetting,
        *,
        enum: list[str] | None = None,
        last_modified_by_name: str = "",
    ) -> Self:
        record = db.model_dump()
        record["value"] = SystemSettingInfo.from_json(record.get("value"))
        record["enum"] = enum or []
        record["last_modified_by_name"] = last_modified_by_name
        return cls.model_validate(record)


class FavoriteInfo(BaseModel):
    """Service-side projection of a user-resource-favorite storage row."""

    model_config = ConfigDict(frozen=True)

    user_id: str
    tenant_id: int
    resource_type: str
    resource_id: str
    created_at: datetime

    @classmethod
    def map_from_db(cls, db: UserResourceFavorite) -> Self:
        """Project one storage row onto the service DTO."""
        return cls.model_validate(db.model_dump())


__all__ = ["AuditLogInfo", "FavoriteInfo", "SystemSettingInfo"]
