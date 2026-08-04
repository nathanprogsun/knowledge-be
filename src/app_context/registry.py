"""Singleton service registry + DI accessors.

``LifeSpanService`` is the dataclass that holds every domain service as a
singleton. It is populated during FastAPI lifespan startup and attached to
``app.state.lifespan_service``. Web routers obtain services via the
``get_xxx_from_lifespan`` factories here.

It lives in its own module (rather than ``lifespan.py``) so ``web.deps``
can import the accessors without creating an import cycle with the app
factory (which mounts routers that import ``web.deps``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from fastapi import FastAPI

from src.core.auth.service import AuthService
from src.core.system.audit_service import AuditLogService
from src.core.system.system_setting_service import SystemSettingService
from src.core.tenants.api_key_service import TenantAPIKeyService
from src.core.tenants.invitation_service import TenantInvitationService
from src.core.tenants.kv_service import TenantKVService
from src.core.tenants.member_service import TenantMemberService
from src.core.tenants.service import TenantService
from src.db.base import DatabaseEngine


@dataclass
class LifeSpanService:
    """Registry of singleton services.

    Populated during `lifespan` startup. Access via `get_xxx_from_lifespan`
    factories; never import this class directly from `web/` modules.
    """

    db_engine: DatabaseEngine | None = None
    auth_service: AuthService | None = None
    tenant_service: TenantService | None = None
    tenant_api_key_service: TenantAPIKeyService | None = None
    tenant_kv_service: TenantKVService | None = None
    tenant_member_service: TenantMemberService | None = None
    tenant_invitation_service: TenantInvitationService | None = None
    audit_log_service: AuditLogService | None = None
    system_setting_service: SystemSettingService | None = None


def get_lifespan_service(app: FastAPI) -> LifeSpanService:
    """Return the lifespan service attached to the FastAPI app."""
    if not hasattr(app.state, "lifespan_service"):
        raise RuntimeError("LifeSpanService is not initialized — was the lifespan started?")
    return cast(LifeSpanService, app.state.lifespan_service)


def get_db_engine_from_lifespan(app: FastAPI) -> DatabaseEngine:
    """DI factory for the database engine."""
    service = get_lifespan_service(app)
    if service.db_engine is None:
        raise RuntimeError("DatabaseEngine is not initialized.")
    return service.db_engine


def get_auth_service_from_lifespan(app: FastAPI) -> AuthService:
    """DI factory for ``AuthService``."""
    service = get_lifespan_service(app)
    if service.auth_service is None:
        raise RuntimeError("AuthService is not initialized.")
    return service.auth_service


def get_tenant_service_from_lifespan(app: FastAPI) -> TenantService:
    """DI factory for ``TenantService``."""
    service = get_lifespan_service(app)
    if service.tenant_service is None:
        raise RuntimeError("TenantService is not initialized.")
    return service.tenant_service


def get_tenant_api_key_service_from_lifespan(app: FastAPI) -> TenantAPIKeyService:
    """DI factory for ``TenantAPIKeyService``."""
    service = get_lifespan_service(app)
    if service.tenant_api_key_service is None:
        raise RuntimeError("TenantAPIKeyService is not initialized.")
    return service.tenant_api_key_service


def get_tenant_kv_service_from_lifespan(app: FastAPI) -> TenantKVService:
    """DI factory for ``TenantKVService``."""
    service = get_lifespan_service(app)
    if service.tenant_kv_service is None:
        raise RuntimeError("TenantKVService is not initialized.")
    return service.tenant_kv_service


def get_tenant_member_service_from_lifespan(app: FastAPI) -> TenantMemberService:
    """DI factory for ``TenantMemberService``."""
    service = get_lifespan_service(app)
    if service.tenant_member_service is None:
        raise RuntimeError("TenantMemberService is not initialized.")
    return service.tenant_member_service


def get_tenant_invitation_service_from_lifespan(app: FastAPI) -> TenantInvitationService:
    """DI factory for ``TenantInvitationService``."""
    service = get_lifespan_service(app)
    if service.tenant_invitation_service is None:
        raise RuntimeError("TenantInvitationService is not initialized.")
    return service.tenant_invitation_service


def get_audit_log_service_from_lifespan(app: FastAPI) -> AuditLogService:
    """DI factory for ``AuditLogService``."""
    service = get_lifespan_service(app)
    if service.audit_log_service is None:
        raise RuntimeError("AuditLogService is not initialized.")
    return service.audit_log_service


def get_system_setting_service_from_lifespan(app: FastAPI) -> SystemSettingService:
    """DI factory for ``SystemSettingService``."""
    service = get_lifespan_service(app)
    if service.system_setting_service is None:
        raise RuntimeError("SystemSettingService is not initialized.")
    return service.system_setting_service


__all__ = [
    "LifeSpanService",
    "get_audit_log_service_from_lifespan",
    "get_auth_service_from_lifespan",
    "get_db_engine_from_lifespan",
    "get_lifespan_service",
    "get_system_setting_service_from_lifespan",
    "get_tenant_api_key_service_from_lifespan",
    "get_tenant_invitation_service_from_lifespan",
    "get_tenant_kv_service_from_lifespan",
    "get_tenant_member_service_from_lifespan",
    "get_tenant_service_from_lifespan",
]
