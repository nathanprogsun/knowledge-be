"""Request-scoped factory for the MCP service domain.

Mirrors ``src.core.tenants.factory`` / ``src.core.auth.factory`` —
repositories are constructed per request on the shared
``AsyncSession``; ``web`` never imports ``db``.

PR-17.5b: the factory forwards the APP-scope singletons the
lifespan registered (discovery provider, connectivity probe, OAuth
factory). All three are optional: tests pass static fakes and the
legacy dependency-overrides flow skips them.

PR-30.6c C7: :func:`build_mcp_resolvers` moved here from
:mod:`src.web.deps.infra_mcp` so ``web`` no longer imports
``db.dao.mcp_service_repository``. The factory now owns the
``(service_resolver, connectivity_resolver)`` construction; the web
layer just forwards the live ``connection_manager`` and calls
:func:`build_mcp_service`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import cast

from sqlalchemy.ext.asyncio import AsyncSession

from src.common.json import JsonObject
from src.core.infra.mcp_services.connectivity import (
    ConnectivityProbe,
    ConnectivityResolver,
    HTTPMCPConnectivityProbe,
)
from src.core.infra.mcp_services.connectivity import (
    _ConnectionManagerLike as _ConnectionManagerLike,
)
from src.core.infra.mcp_services.discovery import (
    DiscoveryCache,
    DiscoveryProvider,
    HTTPMCPDiscoveryProvider,
    ServiceResolver,
)
from src.core.infra.mcp_services.oauth import OAuthManager
from src.core.infra.mcp_services.service import MCPServiceService
from src.core.infra.mcp_services.types import MCPServiceInfo
from src.db.dao.mcp_service_repository import MCPServiceRepository
from src.db.dao.mcp_tool_approval_repository import MCPToolApprovalRepository

# Async because the lifespan-side factory looks up the live
# ``MCPServiceInfo`` in the repository before binding the manager.
OAuthManagerFactoryLike = Callable[[MCPServiceInfo], Awaitable[OAuthManager]]


def build_mcp_resolvers(
    session: AsyncSession,
    tenant_id: int,
) -> tuple[ServiceResolver, ConnectivityResolver]:
    """Return ``(discovery_resolver, connectivity_resolver)`` for one request.

    PR-17.5c C2: takes the active ``tenant_id`` so the resolver
    lookup is tenant-scoped (the previous ``tenant_id=0`` hard-coding
    would have leaked cross-tenant OAuth tokens / URLs).

    PR-30.6c C7: moved here from ``src.web.deps.infra_mcp`` so ``web``
    no longer reaches into ``db.dao.mcp_service_repository``. The
    resolver returns the projected :class:`MCPServiceInfo` (or an
    empty dict for a missing row) directly.
    """
    repo = MCPServiceRepository(session)

    async def _discovery_resolver(
        resolved_tenant_id: int,
        service_id: str,
    ) -> MCPServiceInfo | JsonObject:
        row = await repo.find_for_tenant(resolved_tenant_id, service_id)
        if row is None:
            return cast(JsonObject, {})
        return MCPServiceInfo.map_from_db(row)

    async def _connectivity_resolver(
        resolved_tenant_id: int,
        service_id: str,
    ) -> MCPServiceInfo | JsonObject:
        row = await repo.find_for_tenant(resolved_tenant_id, service_id)
        if row is None:
            return cast(JsonObject, {})
        return MCPServiceInfo.map_from_db(row)

    del tenant_id  # captured by closure above; ``resolved_tenant_id`` per call
    return (
        cast(ServiceResolver, _discovery_resolver),
        cast(ConnectivityResolver, _connectivity_resolver),
    )


def build_live_discovery_provider(
    *,
    session: AsyncSession,
    tenant_id: int,
    connection_manager: _ConnectionManagerLike,
) -> HTTPMCPDiscoveryProvider:
    """Construct an :class:`HTTPMCPDiscoveryProvider` for one request.

    PR-30.6c C7: lives here in core so the ``MCPServiceRepository``
    import stays out of the ``web`` layer's responsibility surface.
    """
    service_resolver, _ = build_mcp_resolvers(
        session=session,
        tenant_id=tenant_id,
    )
    return HTTPMCPDiscoveryProvider(
        connection_manager=connection_manager,
        service_resolver=service_resolver,
    )


def build_live_connectivity_probe(
    *,
    session: AsyncSession,
    tenant_id: int,
    connection_manager: _ConnectionManagerLike,
) -> HTTPMCPConnectivityProbe:
    """Construct an :class:`HTTPMCPConnectivityProbe` for one request.

    PR-30.6c C7: lives here in core so the ``MCPServiceRepository``
    import stays out of the ``web`` layer's responsibility surface.
    """
    _, connectivity_resolver = build_mcp_resolvers(
        session=session,
        tenant_id=tenant_id,
    )
    return HTTPMCPConnectivityProbe(
        connection_manager=connection_manager,
        resolver=connectivity_resolver,
    )


def build_mcp_service(
    session: AsyncSession,
    *,
    discovery_provider: DiscoveryProvider | None = None,
    discovery_cache: DiscoveryCache | None = None,
    connectivity_probe: ConnectivityProbe | None = None,
    oauth_manager_factory: OAuthManagerFactoryLike | None = None,
) -> MCPServiceService:
    """Per-request ``MCPServiceService`` with fresh repositories + APP-scope deps."""
    return MCPServiceService(
        mcp_repo=MCPServiceRepository(session),
        tool_approvals_repo=MCPToolApprovalRepository(session),
        discovery_provider=discovery_provider,
        discovery_cache=discovery_cache,
        connectivity_probe=connectivity_probe,
        oauth_manager_factory=oauth_manager_factory,
    )


__all__ = [
    "OAuthManagerFactoryLike",
    "build_live_connectivity_probe",
    "build_live_discovery_provider",
    "build_mcp_resolvers",
    "build_mcp_service",
]
