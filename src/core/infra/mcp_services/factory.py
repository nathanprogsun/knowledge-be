"""Request-scoped factory for the MCP service domain.

Mirrors ``src.core.tenants.factory`` / ``src.core.auth.factory`` —
repositories are constructed per request on the shared
``AsyncSession``; ``web`` never imports ``db``.

PR-17.5b: the factory forwards the APP-scope singletons the
lifespan registered (discovery provider, connectivity probe, OAuth
factory). All three are optional: tests pass static fakes and the
legacy dependency-overrides flow skips them.

PR-17.5c C4: the per-request live ``HTTPMCPDiscoveryProvider`` /
``HTTPMCPConnectivityProbe`` builders were moved to
:mod:`src.web.deps.infra_mcp` so the ``web`` layer can import the
``MCPServiceRepository`` constructor directly. ``core`` now only
exposes :func:`build_mcp_service` (which accepts pre-constructed
providers) and :func:`build_oauth_manager_factory` (the lifespan-
side helper that wires the per-service ``OAuthManager``).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from sqlalchemy.ext.asyncio import AsyncSession

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
    discovery_provider: object | None = None,
    discovery_cache: object | None = None,
    connectivity_probe: object | None = None,
    oauth_manager_factory: OAuthManagerFactoryLike | None = None,
) -> MCPServiceService:
    """Per-request ``MCPServiceService`` with fresh repositories + APP-scope deps."""
    return MCPServiceService(
        mcp_repo=MCPServiceRepository(session),
        tool_approvals_repo=MCPToolApprovalRepository(session),
        discovery_provider=discovery_provider,  # type: ignore[arg-type]
        discovery_cache=discovery_cache,  # type: ignore[arg-type]
        connectivity_probe=connectivity_probe,  # type: ignore[arg-type]
        oauth_manager_factory=oauth_manager_factory,
    )


__all__ = [
    "OAuthManagerFactoryLike",
    "build_mcp_service",
]
