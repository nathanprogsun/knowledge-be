"""Request-scoped factory for the MCP service domain.

Mirrors ``src.core.tenants.factory`` / ``src.core.auth.factory`` —
repositories are constructed per request on the shared
``AsyncSession``; ``web`` never imports ``db``.

PR-17.5b: the factory also forwards the APP-scope singletons the
lifespan registered (discovery provider, connectivity probe, OAuth
factory). All three are optional: tests pass static fakes and the
legacy dependency-overrides flow skips them.

The factory also exposes :func:`build_live_resolvers`, the single
spot where the live discovery / connectivity probes bridge the
shared DB session into the request-scoped resolver callable.
``web/deps/infra_mcp.py`` calls this helper so the web layer never
imports the repository directly.
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
    _ConnectionManagerLike as _ConnManagerLikeProbe,
)
from src.core.infra.mcp_services.discovery import (
    DiscoveryCache,
    DiscoveryProvider,
    HTTPMCPDiscoveryProvider,
    ServiceResolver,
)
from src.core.infra.mcp_services.discovery import (
    _ConnectionManagerLike as _ConnManagerLikeDisc,
)
from src.core.infra.mcp_services.oauth import OAuthManager
from src.core.infra.mcp_services.service import MCPServiceService
from src.core.infra.mcp_services.types import MCPServiceInfo
from src.db.dao.mcp_service_repository import MCPServiceRepository
from src.db.dao.mcp_tool_approval_repository import MCPToolApprovalRepository

# Async because the lifespan-side factory looks up the live
# ``MCPServiceInfo`` in the repository before binding the manager.
OAuthManagerFactoryLike = Callable[[MCPServiceInfo], Awaitable[OAuthManager]]


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


def build_live_resolvers(
    *,
    session: AsyncSession,
) -> tuple[ServiceResolver, ConnectivityResolver]:
    """Return ``(discovery_resolver, connectivity_resolver)`` for one request.

    Lives here (and not in ``web/deps``) so the ``MCPServiceRepository``
    import stays inside ``core``. The resolvers consult the shared
    ``AsyncSession`` to fetch the requested service row.
    """
    repo = MCPServiceRepository(session)

    async def _discovery_resolver(
        service_id: str,
    ) -> MCPServiceInfo | JsonObject:
        row = await repo.find_for_tenant(0, service_id)
        if row is None:
            return {}
        return MCPServiceInfo.map_from_db(row)

    async def _connectivity_resolver(
        service_id: str,
    ) -> MCPServiceInfo | JsonObject:
        row = await repo.find_for_tenant(0, service_id)
        if row is None:
            return {}
        return MCPServiceInfo.map_from_db(row)

    return _discovery_resolver, _connectivity_resolver


# Type aliases so the public surface stays descriptive.

ConnectionManagerLike = object


def build_live_discovery_provider(
    *,
    session: AsyncSession,
    connection_manager: ConnectionManagerLike,
) -> HTTPMCPDiscoveryProvider:
    """Construct an :class:`HTTPMCPDiscoveryProvider` for one request.

    Lives in ``core`` so the web layer never imports the AI transport
    package directly.
    """
    service_resolver, _ = build_live_resolvers(session=session)
    return HTTPMCPDiscoveryProvider(
        connection_manager=cast(_ConnManagerLikeDisc, connection_manager),
        service_resolver=service_resolver,
    )


def build_live_connectivity_probe(
    *,
    session: AsyncSession,
    connection_manager: ConnectionManagerLike,
) -> HTTPMCPConnectivityProbe:
    """Construct an :class:`HTTPMCPConnectivityProbe` for one request."""
    _, connectivity_resolver = build_live_resolvers(session=session)
    return HTTPMCPConnectivityProbe(
        connection_manager=cast(_ConnManagerLikeProbe, connection_manager),
        resolver=connectivity_resolver,
    )


__all__ = [
    "OAuthManagerFactoryLike",
    "build_live_connectivity_probe",
    "build_live_discovery_provider",
    "build_live_resolvers",
    "build_mcp_service",
]
