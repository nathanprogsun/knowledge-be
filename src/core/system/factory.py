"""System-domain request-scoped service factories.

See ``src.core.auth.factory`` for the pattern: repos are built per
request on the shared ``AsyncSession``; ``web`` never imports ``db``.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.system.admin_service import SystemAdminService
from src.core.system.audit_service import AuditLogService
from src.core.system.favorite_service import UserResourceFavoriteService
from src.core.system.system_setting_service import SystemSettingService
from src.db.dao.audit_log_repository import AuditLogRepository
from src.db.dao.auth_tokens_repository import AuthTokenRepository
from src.db.dao.system_setting_repository import SystemSettingRepository
from src.db.dao.user_resource_favorite_repository import UserResourceFavoriteRepository
from src.db.dao.users_repository import UserRepository


def build_system_admin_service(session: AsyncSession) -> SystemAdminService:
    """Per-request ``SystemAdminService`` with fresh repos.

    The three repositories share the request session so a promote /
    revoke / password reset and its audit row land in one transaction.
    """
    return SystemAdminService(
        users_repo=UserRepository(session),
        tokens_repo=AuthTokenRepository(session),
        audit_repo=AuditLogRepository(session),
    )


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


def build_favorite_service(session: AsyncSession) -> UserResourceFavoriteService:
    """Per-request ``UserResourceFavoriteService`` with a fresh repo.

    The repository is built on the shared request session so the
    composite-key INSERT / DELETE participates in the same unit of work
    as any caller audit row.
    """
    return UserResourceFavoriteService(
        repo=UserResourceFavoriteRepository(session),
    )


__all__ = [
    "build_audit_log_service",
    "build_favorite_service",
    "build_system_admin_service",
    "build_system_setting_service",
]
