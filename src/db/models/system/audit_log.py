"""Storage row for the `audit_logs` table.

Mirrors ``internal/types/audit_log.go::AuditLog``. The table is
append-only — no ``updated_at``, no ``deleted_at``. The monotonic
``id`` (BIGSERIAL) doubles as the pagination cursor (newest-first is
``WHERE id < after_id ORDER BY id DESC``).

``tenant_id = 0`` is the system-scope convention used by
``system.setting_changed``, admin promote/revoke, and the
apply-default-storage-quota bulk write — those rows live outside any
tenant's feed and surface only through the
``GET /system/admin/audit-log`` endpoint.

``AuditAction`` is a dot-namespaced string literal
(e.g. ``"rbac.member_added"``) — kept as a plain ``str`` so future PRs
can plug in new action classes without redefining an enum here. The
full constant set lives in ``src/core/system/audit_actions.py``.
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from pydantic import Field

from src.common.json import JsonObject
from src.common.table_model import TableModel


class AuditLog(TableModel):
    """One immutable row of the ``audit_logs`` table."""

    table: ClassVar[str] = "audit_logs"
    primary_keys: ClassVar[tuple[str, ...]] = ("id",)
    json_columns: ClassVar[tuple[str, ...]] = ("details",)

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


__all__ = ["AuditLog"]
