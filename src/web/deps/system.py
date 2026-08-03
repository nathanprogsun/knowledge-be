"""System-domain FastAPI dependency factories.

Builds per-request ``SystemSettingService`` and ``AuditLogService`` from
fresh repositories sharing the request-scoped ``AsyncSession``. Mirrors
the per-domain split introduced for auth/tenants: the repositories are
constructed per request so the services' reads and writes join the same
transactional unit of work.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from src.core.system.audit_service import AuditLogService
from src.core.system.system_setting_service import SystemSettingService
from src.db.dao.audit_log_repository import AuditLogRepository
from src.db.dao.system_setting_repository import SystemSettingRepository
from src.web.deps.session import SessionDep


def get_system_setting_service(session: SessionDep) -> SystemSettingService:
    """Build a per-request ``SystemSettingService`` with fresh repos.

    Both the settings repository and the audit repository share the
    same session so an ``update`` (which writes both a setting row and
    a ``system.setting_changed`` audit row) lands in one transaction.
    """
    settings_repo = SystemSettingRepository(session)
    audit_repo = AuditLogRepository(session)
    return SystemSettingService(
        settings_repo=settings_repo,
        audit_repo=audit_repo,
    )


SystemSettingServiceDep = Annotated[SystemSettingService, Depends(get_system_setting_service)]


def get_audit_log_service(session: SessionDep) -> AuditLogService:
    """Build a per-request ``AuditLogService`` with a fresh repository."""
    audit_repo = AuditLogRepository(session)
    return AuditLogService(audit_repo=audit_repo)


AuditLogServiceDep = Annotated[AuditLogService, Depends(get_audit_log_service)]


__all__ = [
    "AuditLogServiceDep",
    "SystemSettingServiceDep",
    "get_audit_log_service",
    "get_system_setting_service",
]
