"""System-domain FastAPI dependency factories.

One-line forwarders to ``src.core.system.factory``: repositories are
assembled in ``core`` on the request-scoped ``AsyncSession`` so the
request's reads and writes share one transactional unit of work.
``web`` never imports ``db``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, cast

from fastapi import Depends, Request

from src.core.system.audit_service import AuditLogService
from src.core.system.factory import (
    build_audit_log_service,
    build_favorite_service,
    build_system_setting_service,
)
from src.core.system.favorite_service import UserResourceFavoriteService
from src.core.system.info_service import SystemInfoService
from src.core.system.system_setting_service import SystemSettingService
from src.web.deps.session import SessionDep


def get_system_setting_service(session: SessionDep) -> SystemSettingService:
    """Build a per-request ``SystemSettingService`` on the shared session."""
    return build_system_setting_service(session)


SystemSettingServiceDep = Annotated[SystemSettingService, Depends(get_system_setting_service)]


def get_audit_log_service(session: SessionDep) -> AuditLogService:
    """Build a per-request ``AuditLogService`` with a fresh repository."""
    return build_audit_log_service(session)


AuditLogServiceDep = Annotated[AuditLogService, Depends(get_audit_log_service)]


def get_favorite_service(session: SessionDep) -> UserResourceFavoriteService:
    """Build a per-request ``UserResourceFavoriteService`` on the shared session."""
    return build_favorite_service(session)


FavoriteServiceDep = Annotated[UserResourceFavoriteService, Depends(get_favorite_service)]


def get_system_info_service(
    request: Request,
    session: SessionDep,
) -> SystemInfoService:
    """Build a per-request ``SystemInfoService`` with the lifespan ``started_at``.

    The boot instant is recorded on ``app.state.started_at`` during the
    FastAPI lifespan startup. When the lifespan was bypassed (tests)
    the attribute is ``None`` and the service falls back to ``now`` so
    the uptime reads zero rather than raising.
    """
    started_at = cast("datetime | None", getattr(request.app.state, "started_at", None))
    return SystemInfoService(session=session, started_at=started_at)


SystemInfoServiceDep = Annotated[SystemInfoService, Depends(get_system_info_service)]


__all__ = [
    "AuditLogServiceDep",
    "FavoriteServiceDep",
    "SystemInfoServiceDep",
    "SystemSettingServiceDep",
    "get_audit_log_service",
    "get_favorite_service",
    "get_system_info_service",
    "get_system_setting_service",
]
