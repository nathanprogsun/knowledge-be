"""Storage rows for the system domain (audit log + settings)."""

from __future__ import annotations

from src.db.models.system.audit_log import AuditLog
from src.db.models.system.system_setting import SystemSetting

__all__ = ["AuditLog", "SystemSetting"]
