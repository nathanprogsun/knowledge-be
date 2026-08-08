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
from src.web.deps.chunks import (
    ChunkRevisionServiceDep,
    ChunkServiceDep,
    get_chunk_revision_service,
    get_chunk_service,
)
from src.web.deps.infra_datasources import (
    DataSourceServiceDep,
    get_datasource_service,
)
from src.web.deps.infra_initialization import (
    InitializationServiceDep,
    get_initialization_service,
)
from src.web.deps.infra_mcp import (
    MCPServiceDep,
    RequestTenantIdDep,
    RequestUserIdDep,
    RequireTenantIdDep,
    RequireUserIdDep,
    get_mcp_service,
    get_request_tenant_id,
    get_request_user_id,
    require_tenant_id,
    require_user_id,
)
from src.web.deps.infra_models import (
    ModelServiceDep,
    get_model_service,
)
from src.web.deps.infra_storage_backends import (
    StorageBackendServiceDep,
    get_storage_backend_service,
)
from src.web.deps.infra_vector_stores import (
    VectorStoreServiceDep,
    get_vector_store_service,
)
from src.web.deps.infra_web_search import (
    WebSearchProviderServiceDep,
    get_web_search_provider_service,
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
    "ChunkRevisionServiceDep",
    "ChunkServiceDep",
    "CrossTenantDep",
    "CurrentUserContextDep",
    "DataSourceServiceDep",
    "InitializationServiceDep",
    "MCPServiceDep",
    "ModelServiceDep",
    "OidcServiceDep",
    "PathTenantMatchDep",
    "RequestTenantIdDep",
    "RequestUserIdDep",
    "RequireTenantIdDep",
    "RequireUserIdDep",
    "RoleAdminDep",
    "RoleContributorDep",
    "RoleOwnerDep",
    "RoleViewerDep",
    "SessionDep",
    "StorageBackendServiceDep",
    "SystemAdminDep",
    "SystemSettingServiceDep",
    "TenantAPIKeyServiceDep",
    "TenantKVServiceDep",
    "TenantMemberServiceDep",
    "TenantServiceDep",
    "VectorStoreServiceDep",
    "WebSearchProviderServiceDep",
    "get_async_session",
    "get_audit_log_service",
    "get_auth_service",
    "get_chunk_revision_service",
    "get_chunk_service",
    "get_datasource_service",
    "get_initialization_service",
    "get_mcp_service",
    "get_model_service",
    "get_oidc_service",
    "get_request_tenant_id",
    "get_request_user_id",
    "get_storage_backend_service",
    "get_system_setting_service",
    "get_tenant_api_key_service",
    "get_tenant_kv_service",
    "get_tenant_member_service",
    "get_tenant_service",
    "get_vector_store_service",
    "get_web_search_provider_service",
    "require_tenant_id",
    "require_user_id",
]
