"""Web-layer FastAPI dependency factories.

One module per domain: ``session`` owns the request-scoped ``AsyncSession``;
every other module builds that domain's repositories + service on top of
it. This package re-exports the public names so callers keep importing
from ``src.web.deps`` regardless of which module a dependency lives in.
"""

from __future__ import annotations

from src.web.deps.auth import (
    AuthServiceDep,
    CurrentUserContextDep,
    OidcServiceDep,
    get_auth_service,
    get_oidc_service,
)
from src.web.deps.rbac import (
    CrossTenantDep,
    PathTenantMatchDep,
    RoleAdminDep,
    RoleContributorDep,
    RoleOwnerDep,
    RoleViewerDep,
    SystemAdminDep,
)
from src.web.deps.session import SessionDep, get_async_session
from src.web.deps.system import (
    AuditLogServiceDep,
    SystemSettingServiceDep,
    get_audit_log_service,
    get_system_setting_service,
)
from src.web.deps.tenants import (
    TenantAPIKeyServiceDep,
    TenantKVServiceDep,
    TenantMemberServiceDep,
    TenantServiceDep,
    get_tenant_api_key_service,
    get_tenant_kv_service,
    get_tenant_member_service,
    get_tenant_service,
)
from src.web.middleware.auth import AuthDep

__all__ = [
    "AuditLogServiceDep",
    "AuthDep",
    "AuthServiceDep",
    "CrossTenantDep",
    "CurrentUserContextDep",
    "OidcServiceDep",
    "PathTenantMatchDep",
    "RoleAdminDep",
    "RoleContributorDep",
    "RoleOwnerDep",
    "RoleViewerDep",
    "SessionDep",
    "SystemAdminDep",
    "SystemSettingServiceDep",
    "TenantAPIKeyServiceDep",
    "TenantKVServiceDep",
    "TenantMemberServiceDep",
    "TenantServiceDep",
    "get_async_session",
    "get_audit_log_service",
    "get_auth_service",
    "get_oidc_service",
    "get_system_setting_service",
    "get_tenant_api_key_service",
    "get_tenant_kv_service",
    "get_tenant_member_service",
    "get_tenant_service",
]
