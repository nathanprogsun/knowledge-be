"""System-domain request-scoped service factories.

See ``src.core.auth.factory`` for the pattern: repos are built per
request on the shared ``AsyncSession``; ``web`` never imports ``db``.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.system.audit_service import AuditLogService
from src.core.system.system_setting_service import SystemSettingService
from src.db.dao.audit_log_repository import AuditLogRepository
from src.db.dao.system_setting_repository import SystemSettingRepository


def build_system_setting_service(session: AsyncSession) -> SystemSettingService:
    """Per-request ``SystemSettingService`` with fresh repos.

    Both repositories share the session so an ``update`` (setting row +
    ``system.setting_changed`` audit row) lands in one transaction.
    """
    return SystemSettingService(
        settings_repo=SystemSettingRepository(session),
        audit_repo=AuditLogRepository(session),
    )


def build_audit_log_service(session: AsyncSession) -> AuditLogService:
    """Per-request ``AuditLogService`` with a fresh repository."""
    return AuditLogService(audit_repo=AuditLogRepository(session))


__all__ = ["build_audit_log_service", "build_system_setting_service"]
