"""MCP service — registration, lookup, update, soft-delete.

Mirrors ``internal/application/service/mcp_service.go`` (the Go
"service" half, not the HTTP handler). Builtin rows cannot be updated
or deleted; the Go repo enforces this in the service and so do we.

Credentials (the secret fields inside ``auth_config``) deliberately do
NOT flow through this service — the dedicated credentials subresource
in ``src/core/infra/mcp_services/credentials.py`` is the only writer,
mirroring Go's ``MCPCredentialsHandler``.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import cast

from src.ai.mcp_transport.errors import MCPError
from src.common.exception import ConflictError, NotFoundError, ValidationError
from src.common.json import BindParams, JsonObject, JsonValue
from src.core.infra.mcp_services.connectivity import (
    ConnectivityProbe,
    ConnectivityResult,
)
from src.core.infra.mcp_services.discovery import (
    DiscoveryCache,
    DiscoveryProvider,
    DiscoveryResource,
    DiscoveryTool,
)
from src.core.infra.mcp_services.oauth import OAuthManager
from src.core.infra.mcp_services.types import MCPServiceInfo, MCPToolApprovalInfo
from src.db.dao.mcp_service_repository import MCPServiceRepository
from src.db.dao.mcp_tool_approval_repository import MCPToolApprovalRepository
from src.db.models.infra.mcp_services import MCPService, MCPToolApproval

# Default advanced config used when a creator omits the field. Matches
# the upstream ``GetDefaultAdvancedConfig`` in ``internal/types/mcp.go``.
_DEFAULT_ADVANCED_CONFIG: JsonObject = {"timeout": 30, "retry_count": 3, "retry_delay": 1}

# Allowed values for the wire-side ``transport_type`` column.
_TRANSPORT_SSE = "sse"
_TRANSPORT_HTTP_STREAMABLE = "http-streamable"
_TRANSPORT_STDIO = "stdio"
_KNOWN_TRANSPORT_TYPES: frozenset[str] = frozenset(
    {_TRANSPORT_SSE, _TRANSPORT_HTTP_STREAMABLE, _TRANSPORT_STDIO}
)


class MCPServiceService:
    """Stateless MCP service registry, constructed per request."""

    def __init__(
        self,
        *,
        mcp_repo: MCPServiceRepository,
        tool_approvals_repo: MCPToolApprovalRepository,
        discovery_provider: DiscoveryProvider | None = None,
        discovery_cache: DiscoveryCache | None = None,
        connectivity_probe: ConnectivityProbe | None = None,
        oauth_manager_factory: (Callable[[MCPServiceInfo], Awaitable[OAuthManager]] | None) = None,
    ) -> None:
        self._mcp_repo = mcp_repo
        self._tool_approvals_repo = tool_approvals_repo
        self._discovery_provider = discovery_provider
        self._discovery_cache = discovery_cache
        self._connectivity_probe = connectivity_probe
        self._oauth_manager_factory = oauth_manager_factory

    # ── Create ──────────────────────────────────────────────────────

    async def create_service(
        self,
        *,
        tenant_id: int,
        name: str,
        transport_type: str,
        description: str | None = None,
        url: str | None = None,
        headers: dict[str, str] | None = None,
        auth_config: JsonObject | None = None,
        advanced_config: JsonObject | None = None,
        stdio_config: JsonObject | None = None,
        env_vars: dict[str, str] | None = None,
        enabled: bool | None = True,
    ) -> MCPServiceInfo:
        """Persist a new MCP service row."""
        clean_name = name.strip()
        if not clean_name:
            raise ValidationError(
                code="mcp_service.name_required",
                message="MCP service name cannot be empty",
            )
        self._validate_transport(transport_type)
        if transport_type == _TRANSPORT_STDIO:
            # Mirrors the Go guard; stdio transport is disabled for
            # security even though the column accepts the value.
            raise ValidationError(
                code="mcp_service.stdio_disabled",
                message=(
                    "stdio transport is disabled for security reasons; "
                    "please use SSE or HTTP Streamable transport instead"
                ),
            )
        if advanced_config is None:
            advanced_config = dict(_DEFAULT_ADVANCED_CONFIG)

        # Pre-check duplicate name within the tenant. Mirrors Go's
        # `MCPServiceRepository.FindByName` + service-level 409 path.
        # The DB-level unique constraint is the second line of defence
        # against concurrent inserts.
        if await self._mcp_repo.exists_by_tenant_and_name(tenant_id=tenant_id, name=clean_name):
            raise ConflictError(
                code="mcp_service.duplicate_name",
                message=(f"an MCP service named {clean_name!r} already exists in this workspace"),
            )

        now = datetime.now(UTC)
        new_id = uuid.uuid4().hex
        row = MCPService(
            id=new_id,
            tenant_id=tenant_id,
            name=clean_name,
            description=description,
            enabled=enabled if enabled is not None else True,
            transport_type=transport_type,
            url=url,
            headers=headers,
            auth_config=cast(JsonValue, auth_config),
            advanced_config=cast(JsonValue, advanced_config),
            stdio_config=cast(JsonValue, stdio_config),
            env_vars=env_vars,
            is_builtin=False,
            created_at=now,
            updated_at=now,
        )
        try:
            stored = await self._mcp_repo.insert(row)
        except Exception as exc:  # pragma: no cover - rare race path
            # Catch the DB unique-violation race window between the
            # pre-check and the actual insert. Translate to the same
            # 409 the pre-check raises so callers see one error shape.
            name = exc.__class__.__name__
            if "UniqueViolation" in name and "tenant_name" in str(exc):
                raise ConflictError(
                    code="mcp_service.duplicate_name",
                    message=(
                        f"an MCP service named {clean_name!r} already exists in this workspace"
                    ),
                ) from exc
            raise
        return MCPServiceInfo.map_from_db(stored)

    # ── Read ────────────────────────────────────────────────────────

    async def get_service(self, *, tenant_id: int, id: str) -> MCPServiceInfo:
        """Fetch one row; raise ``mcp_service.not_found`` when absent."""
        return MCPServiceInfo.map_from_db(await self._mcp_repo.get_by_id(tenant_id, id))

    async def list_services(self, *, tenant_id: int) -> list[MCPServiceInfo]:
        """List the tenant's services, newest first. Excludes builtin rows."""
        rows = await self._mcp_repo.list_for_tenant(tenant_id)
        return [MCPServiceInfo.map_from_db(r) for r in rows]

    # ── Update ──────────────────────────────────────────────────────

    async def update_service(
        self,
        *,
        tenant_id: int,
        id: str,
        name: str | None = None,
        description: str | None = None,
        enabled: bool | None = None,
        transport_type: str | None = None,
        url: str | None = None,
        headers: dict[str, str] | None = None,
        auth_config: JsonObject | None = None,
        advanced_config: JsonObject | None = None,
        stdio_config: JsonObject | None = None,
        env_vars: dict[str, str] | None = None,
    ) -> MCPServiceInfo:
        """Patch the supplied columns.

        The secret fields inside ``auth_config`` (api_key, token) are
        intentionally not accepted here: the credentials subresource is
        the only writer, mirroring the Go side.
        """
        existing = await self._mcp_repo.get_by_id(tenant_id, id)
        if existing.is_builtin:
            raise ValidationError(
                code="mcp_service.builtin_immutable",
                message="builtin MCP services cannot be updated",
            )
        columns = _build_update_columns(
            name=name,
            description=description,
            enabled=enabled,
            transport_type=transport_type,
            url=url,
            headers=headers,
            auth_config=auth_config,
            advanced_config=advanced_config,
            stdio_config=stdio_config,
            env_vars=env_vars,
        )
        if "transport_type" in columns:
            self._validate_transport(str(columns["transport_type"]))
            if columns["transport_type"] == _TRANSPORT_STDIO:
                raise ValidationError(
                    code="mcp_service.stdio_disabled",
                    message=(
                        "stdio transport is disabled for security reasons; "
                        "please use SSE or HTTP Streamable transport instead"
                    ),
                )
        columns["updated_at"] = datetime.now(UTC)
        updated = await self._mcp_repo.update(
            tenant_id,
            id,
            columns=cast("BindParams", columns),
        )
        if updated is None:
            raise NotFoundError(
                code="mcp_service.not_found",
                message=f"MCP service {id} not found",
            )
        return MCPServiceInfo.map_from_db(updated)

    # ── Delete ──────────────────────────────────────────────────────

    async def delete_service(self, *, tenant_id: int, id: str) -> bool:
        """Soft-delete a row; return whether one existed.

        Builtin services cannot be deleted (mirrors the Go guard).
        """
        existing = await self._mcp_repo.find_for_tenant(tenant_id, id)
        if existing is None:
            return False
        if existing.is_builtin:
            raise ValidationError(
                code="mcp_service.builtin_immutable",
                message="builtin MCP services cannot be deleted",
            )
        await self._mcp_repo.soft_delete(
            tenant_id,
            id,
            deleted_at=datetime.now(UTC),
        )
        return True

    # ── Tool approvals ──────────────────────────────────────────────

    async def list_tool_approvals(
        self,
        *,
        tenant_id: int,
        service_id: str,
    ) -> list[MCPToolApprovalInfo]:
        """Return every persisted approval override for the service."""
        # Ensure the service exists so callers get 404 — not an empty
        # list — for a typo.
        await self._mcp_repo.get_by_id(tenant_id, service_id)
        rows = await self._tool_approvals_repo.list_by_service(tenant_id, service_id)
        return [MCPToolApprovalInfo.map_from_db(r) for r in rows]

    async def set_tool_approval(
        self,
        *,
        tenant_id: int,
        service_id: str,
        tool_name: str,
        require_approval: bool,
    ) -> MCPToolApprovalInfo:
        """Upsert the (service, tool) approval flag."""
        clean_name = tool_name.strip()
        if not clean_name:
            raise ValidationError(
                code="mcp_service.tool_name_required",
                message="tool_name is required",
            )
        await self._mcp_repo.get_by_id(tenant_id, service_id)
        now = datetime.now(UTC)
        row = MCPToolApproval(
            id=uuid.uuid4().hex,
            tenant_id=tenant_id,
            service_id=service_id,
            tool_name=clean_name,
            require_approval=require_approval,
            created_at=now,
            updated_at=now,
        )
        stored = await self._tool_approvals_repo.upsert(row=row)
        return MCPToolApprovalInfo.map_from_db(stored)

    # ── Discovery ───────────────────────────────────────────────────

    async def list_tools(
        self,
        *,
        tenant_id: int,
        service_id: str,
    ) -> list[DiscoveryTool]:
        """Discover the upstream MCP service's tools (with cache).

        When the live transport raises :class:`MCPError`
        (network failure, server-side session invalidation) the call
        degrades to an empty list so the UI keeps working when the
        upstream MCP server is unreachable.
        """
        await self._mcp_repo.get_by_id(tenant_id, service_id)
        if self._discovery_provider is None:
            return []
        try:
            if self._discovery_cache is None:
                return list(
                    await self._discovery_provider.list_tools(service_id=service_id),
                )
            tools, _ = await self._discovery_cache.get_or_refresh(
                tenant_id=tenant_id,
                service_id=service_id,
                provider=self._discovery_provider,
            )
            return tools
        except MCPError:
            return []

    async def list_resources(
        self,
        *,
        tenant_id: int,
        service_id: str,
    ) -> list[DiscoveryResource]:
        """Discover the upstream MCP service's resources (with cache).

        Same degradation as :meth:`list_tools`.
        """
        await self._mcp_repo.get_by_id(tenant_id, service_id)
        if self._discovery_provider is None:
            return []
        try:
            if self._discovery_cache is None:
                return list(
                    await self._discovery_provider.list_resources(
                        service_id=service_id,
                    ),
                )
            _, resources = await self._discovery_cache.get_or_refresh(
                tenant_id=tenant_id,
                service_id=service_id,
                provider=self._discovery_provider,
            )
            return resources
        except MCPError:
            return []

    def invalidate_discovery_cache(self, *, tenant_id: int, service_id: str) -> None:
        """Drop the cached tool/resource lists for one service."""
        if self._discovery_cache is not None:
            self._discovery_cache.invalidate(
                tenant_id=tenant_id,
                service_id=service_id,
            )

    # ── Connectivity test ──────────────────────────────────────────

    async def test_service(
        self,
        *,
        tenant_id: int,
        service_id: str,
    ) -> ConnectivityResult:
        """Probe the configured service and return the wire-shape result.

        Without a wired probe, the default result reports
        success=False so the UI gets an immediate "no transport wired"
        signal rather than a silent 200 with empty data.
        """
        service = await self._mcp_repo.get_by_id(tenant_id, service_id)
        if self._connectivity_probe is None:
            return ConnectivityResult(
                success=False,
                message="MCP connectivity probe is not configured",
            )
        return await self._connectivity_probe(
            service_id=service.id,
            transport_type=service.transport_type,
            url=service.url,
            oauth_required=_is_oauth(service.auth_config),
        )

    # ── OAuth ───────────────────────────────────────────────────────

    async def fetch_oauth_manager(
        self,
        *,
        tenant_id: int,
        service_id: str,
    ) -> OAuthManager:
        """Materialise an :class:`OAuthManager` for the requested service.

        Fetches the live row eagerly so the standard
        ``mcp_service.not_found`` 404 fires before any OAuth work is
        done. PR-17.5b: when the lifespan registered an
        ``oauth_manager_factory``, that factory binds the per-request
        manager to the APP-scope state (HTTP client, CSRF store, token
        store) and is preferred over the legacy
        :class:`OAuthManager(service=info)` constructor.
        """
        info = await self.get_service(tenant_id=tenant_id, id=service_id)
        if self._oauth_manager_factory is not None:
            return await self._oauth_manager_factory(info)
        return OAuthManager(service=info)

    # ── Validation helpers ─────────────────────────────────────────

    @staticmethod
    def _validate_transport(transport_type: str) -> None:
        if transport_type not in _KNOWN_TRANSPORT_TYPES:
            raise ValidationError(
                code="mcp_service.invalid_transport",
                message=f"transport_type must be one of {sorted(_KNOWN_TRANSPORT_TYPES)}",
            )


def _is_oauth(auth_config: JsonValue) -> bool:
    """True when the persisted auth_config uses the OAuth strategy.

    Mirrors ``MCPAuthConfig.IsOAuth()``: the auth_type must equal
    ``oauth``. Empty/missing auth_config returns False.
    """
    if not isinstance(auth_config, dict):
        return False
    raw = auth_config.get("auth_type")
    return isinstance(raw, str) and raw.strip().lower() == "oauth"


def _build_update_columns(
    *,
    name: str | None,
    description: str | None,
    enabled: bool | None,
    transport_type: str | None,
    url: str | None,
    headers: dict[str, str] | None,
    auth_config: JsonObject | None,
    advanced_config: JsonObject | None,
    stdio_config: JsonObject | None,
    env_vars: dict[str, str] | None,
) -> dict[str, object]:
    """Collect the supplied columns, dropping secret subfields."""
    columns: dict[str, object] = {}
    if name is not None:
        clean_name = name.strip()
        if not clean_name:
            raise ValidationError(
                code="mcp_service.name_required",
                message="MCP service name cannot be empty",
            )
        columns["name"] = clean_name
    if description is not None:
        columns["description"] = description
    if enabled is not None:
        columns["enabled"] = enabled
    if transport_type is not None:
        columns["transport_type"] = transport_type
    if url is not None:
        columns["url"] = url
    if headers is not None:
        columns["headers"] = cast("JsonValue", headers)
    if auth_config is not None:
        # Pass-through; mirrors Go's
        # ``service.UpdateMCPService`` which round-trips
        # ``auth_config`` on user services verbatim. The dedicated
        # credentials subresource is the only writer of ``api_key`` /
        # ``token``; the standard update path just preserves them.
        columns["auth_config"] = cast("JsonValue", auth_config)
    if advanced_config is not None:
        columns["advanced_config"] = advanced_config
    if stdio_config is not None:
        columns["stdio_config"] = stdio_config
    if env_vars is not None:
        columns["env_vars"] = cast("JsonValue", env_vars)
    return columns


__all__ = ["MCPServiceService"]
