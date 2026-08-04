"""System-domain FastAPI dependency factories.

One-line forwarders to ``src.core.system.factory``: repositories are
assembled in ``core`` on the request-scoped ``AsyncSession`` so the
request's reads and writes share one transactional unit of work.
``web`` never imports ``db``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from src.core.system.audit_service import AuditLogService
from src.core.system.factory import build_audit_log_service, build_system_setting_service
from src.core.system.system_setting_service import SystemSettingService
from src.web.deps.session import SessionDep


def get_system_setting_service(session: SessionDep) -> SystemSettingService:
    """Build a per-request ``SystemSettingService`` on the shared session."""
    return build_system_setting_service(session)


SystemSettingServiceDep = Annotated[SystemSettingService, Depends(get_system_setting_service)]


def get_audit_log_service(session: SessionDep) -> AuditLogService:
    """Build a per-request ``AuditLogService`` on the shared session."""
    return build_audit_log_service(session)


AuditLogServiceDep = Annotated[AuditLogService, Depends(get_audit_log_service)]


__all__ = [
    "AuditLogServiceDep",
    "SystemSettingServiceDep",
    "get_audit_log_service",
    "get_system_setting_service",
]
