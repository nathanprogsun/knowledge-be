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
from typing import Self

from pydantic import BaseModel, ConfigDict, Field

from src.db.models.system.audit_log import AuditLog
from src.db.models.system.system_setting import SystemSetting


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
    details: dict[str, object] = Field(default_factory=dict)
    created_at: datetime

    @classmethod
    def map_from_db(cls, db: AuditLog) -> Self:
        record = db.model_dump()
        details = record.get("details")
        if isinstance(details, str):
            details = json.loads(details)
        if details is None:
            details = {}
        record["details"] = details
        return cls.model_validate(record)


class SystemSettingInfo(BaseModel):
    """Wire-side projection of a ``system_settings`` row.

    ``enum`` and ``last_modified_by_name`` are NOT persisted — they are
    derived per request by the service (registry + user lookup).
    """

    model_config = ConfigDict(frozen=True)

    id: int
    key: str
    value: dict[str, object] | list[str] | list[object] | str | int | bool
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
    def map_from_db(
        cls,
        db: SystemSetting,
        *,
        enum: list[str] | None = None,
        last_modified_by_name: str = "",
    ) -> Self:
        record = db.model_dump()
        value = record.get("value")
        if isinstance(value, str):
            # Persisted scalars (int / bool / string) round-trip as
            # strings when the value_type is non-JSON; only try to
            # parse when the string looks like a JSON document.
            stripped = value.strip()
            if stripped.startswith(("{", "[")):
                value = json.loads(stripped)
        if value is None:
            value = {}
        record["value"] = value
        record["enum"] = enum or []
        record["last_modified_by_name"] = last_modified_by_name
        return cls.model_validate(record)


__all__ = ["AuditLogInfo", "SystemSettingInfo"]
