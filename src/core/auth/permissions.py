"""Permission types and role hierarchy for tenant-scoped RBAC.

These types are the shared vocabulary the middleware
(``src/web/middleware/``) and service layer use to gate endpoints.

Role hierarchy (numeric level, higher = more privileged):

    owner       40
    admin       30
    contributor 20
    viewer      10

``HasPermission`` checks ``level >= required.level``. Unknown roles
default to level 0 (strictly less than any defined role), so a missing
role token fails closed.

API-key capabilities are an additive any-of allow-list on top of the
role ladder. An API-key principal carries a ``TenantAPIKeyScope`` (not
a ``TenantRole``); the API Key Gate middleware is the single place that
authorizes machine principals.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Final


class TenantRoleLevel(IntEnum):
    """Numeric privilege level for ``TenantRole``.

    Spaced by 10 so new roles can be inserted between existing ones.
    """

    VIEWER = 10
    CONTRIBUTOR = 20
    ADMIN = 30
    OWNER = 40


class TenantRole:
    """Tenant-scoped role constants (dot-free string literals).

    Mirrors the ``tenant_members.role`` column values.
    """

    OWNER: Final[str] = "owner"
    ADMIN: Final[str] = "admin"
    CONTRIBUTOR: Final[str] = "contributor"
    VIEWER: Final[str] = "viewer"

    _LEVELS: Final[dict[str, int]] = {
        OWNER: TenantRoleLevel.OWNER,
        ADMIN: TenantRoleLevel.ADMIN,
        CONTRIBUTOR: TenantRoleLevel.CONTRIBUTOR,
        VIEWER: TenantRoleLevel.VIEWER,
    }

    @staticmethod
    def level(role: str) -> int:
        """Return the numeric privilege level. Unknown → 0 (fail closed)."""
        return TenantRole._LEVELS.get(role, 0)

    @staticmethod
    def has_permission(role: str, required: str) -> bool:
        """Check if ``role`` is at least as privileged as ``required``."""
        return TenantRole.level(role) >= TenantRole.level(required)

    @staticmethod
    def is_valid(role: str) -> bool:
        """Check if ``role`` is one of the four defined tenant roles."""
        return role in TenantRole._LEVELS


# ── API Key capabilities ───────────────────────────────────────────


class APIKeyCapability:
    """Additive capability grants for scoped API keys.

    A capability never widens which knowledge bases a key may touch;
    KB scoping is enforced downstream by KB-access guards.
    """

    RETRIEVE: Final[str] = "retrieve"
    CHAT: Final[str] = "chat"
    READ_AGENTS: Final[str] = "read_agents"
    INGEST: Final[str] = "ingest"
    MANAGE_KNOWLEDGE_BASES: Final[str] = "manage_kbs"
    MANAGE_AGENTS: Final[str] = "manage_agents"
    MESSAGE_HISTORY: Final[str] = "message_history"
    MANAGE_MODELS: Final[str] = "manage_models"
    MANAGE_MCP_SERVICES: Final[str] = "manage_mcp_services"
    MANAGE_DATASOURCES: Final[str] = "manage_datasources"
    MANAGE_CHANNELS: Final[str] = "manage_channels"
    MANAGE_VECTOR_STORES: Final[str] = "manage_vector_stores"
    MANAGE_STORAGE_BACKENDS: Final[str] = "manage_storage_backends"
    MANAGE_WEB_SEARCH: Final[str] = "manage_web_search"
    RUN_EVALUATIONS: Final[str] = "run_evaluations"
    MANAGE_MEMBERS: Final[str] = "manage_members"
    MANAGE_SPACES: Final[str] = "manage_spaces"
    MANAGE_TENANT_SETTINGS: Final[str] = "manage_tenant_settings"
    SYSTEM_TENANTS_READ: Final[str] = "system_tenants_read"
    SYSTEM_TENANTS_MANAGE: Final[str] = "system_tenants_manage"
    SYSTEM_SETTINGS_READ: Final[str] = "system_settings_read"
    SYSTEM_SETTINGS_MANAGE: Final[str] = "system_settings_manage"
    SYSTEM_RUNTIME_READ: Final[str] = "system_runtime_read"
    SYSTEM_RUNTIME_MANAGE: Final[str] = "system_runtime_manage"


# ── API Key scope types ────────────────────────────────────────────


class APIKeyScopeType:
    """Scope type for an API key: ``tenant`` (workspace-bound) or ``platform``."""

    TENANT: Final[str] = "tenant"
    PLATFORM: Final[str] = "platform"


class TenantAPIKeyScope:
    """Resolved API-key scope at request time.

    Carries the key id, scope type, full-access flag, KB-id allow-list,
    and capabilities. The API Key Gate middleware constructs this from
    the ``tenant_api_keys`` row and stashes it on the request context;
    downstream guards read it to decide route admission.

    This is a plain class (not a Pydantic model) because it is a
    request-scoped value object, not a serialised DTO.
    """

    __slots__ = (
        "capabilities",
        "full_access",
        "key_id",
        "knowledge_base_ids",
        "scope_type",
    )

    def __init__(
        self,
        *,
        key_id: int,
        scope_type: str = APIKeyScopeType.TENANT,
        full_access: bool = False,
        knowledge_base_ids: list[str] | None = None,
        capabilities: list[str] | None = None,
    ) -> None:
        self.key_id = key_id
        self.scope_type = scope_type
        self.full_access = full_access
        self.knowledge_base_ids = knowledge_base_ids or []
        self.capabilities = capabilities or []

    def is_platform(self) -> bool:
        """True when this is a platform-scope key (not workspace-bound)."""
        return self.scope_type == APIKeyScopeType.PLATFORM

    def has_capability(self, capability: str) -> bool:
        """Check if the scope carries the given additive grant."""
        if not capability:
            return False
        return capability in self.capabilities

    def is_knowledge_base_restricted(self) -> bool:
        """True when the key is scoped to specific KB ids."""
        return len(self.knowledge_base_ids) > 0


# ── API Key route policy ───────────────────────────────────────────


class APIKeyRoutePolicy:
    """Per-route API-key admission policy.

    Routes that declare no policy are denied for API keys by default
    (fail-closed). ``PlatformOnly`` rejects workspace-bound keys even
    when they are full-access. ``RequireFullAccess`` admits only
    full-access keys unless a capability matches. ``Capabilities`` is
    an any-of allow-list for scoped keys.
    """

    __slots__ = ("capabilities", "platform_only", "require_full_access")

    def __init__(
        self,
        *,
        platform_only: bool = False,
        require_full_access: bool = False,
        capabilities: list[str] | None = None,
    ) -> None:
        self.platform_only = platform_only
        self.require_full_access = require_full_access
        self.capabilities = capabilities or []

    def with_capability(self, capability: str) -> APIKeyRoutePolicy:
        """Return a copy that additionally admits keys carrying ``capability``."""
        if capability in self.capabilities:
            return APIKeyRoutePolicy(
                platform_only=self.platform_only,
                require_full_access=self.require_full_access,
                capabilities=list(self.capabilities),
            )
        new_caps = [*list(self.capabilities), capability]
        return APIKeyRoutePolicy(
            platform_only=self.platform_only,
            require_full_access=self.require_full_access,
            capabilities=new_caps,
        )


__all__ = [
    "APIKeyCapability",
    "APIKeyRoutePolicy",
    "APIKeyScopeType",
    "TenantAPIKeyScope",
    "TenantRole",
    "TenantRoleLevel",
]
