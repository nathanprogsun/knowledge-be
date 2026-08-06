"""MCP-domain FastAPI dependency factories.

Per-domain forwarder for the MCP service module: the per-request
``MCPServiceService`` is built in :func:`src.core.infra.mcp_services.factory.build_mcp_service`
so the request's reads and writes share one transactional unit of work
on the shared ``AsyncSession``.

PR-17.5b also reads the live MCP singletons the lifespan registered
(connection pool, OAuth state + secret stores, OAuth factory) and
forwards them to :func:`src.core.infra.mcp_services.factory.build_mcp_service`
so the per-request service can drive the live transport layer when
the lifespan was started, while keeping the dependency-overrides path
green for tests that bypass lifespan.

PR-30.6c C7: ``build_live_resolvers`` / ``build_live_discovery_provider``
/ ``build_live_connectivity_probe`` moved into
:mod:`src.core.infra.mcp_services.factory`. The web layer no longer
imports ``db.dao.mcp_service_repository`` directly — the core factory
owns repository construction on the request session. The web layer
keeps these names available as re-exports so the lifespan-side
forwarder (and any tests that patched the symbol) keep working.

Adds two FastAPI dependency factories for ``tenant_id`` and ``user_id``
so the router can read them from the per-request auth state without
reaching into the context layer directly.
"""

from __future__ import annotations

from typing import Annotated, cast

from fastapi import Depends, Request

from src.app_context import request_context
from src.app_context.registry import LifeSpanService
from src.common.exception import ValidationError
from src.core.infra.mcp_services.connectivity import (
    ConnectionManagerLike,
    ConnectivityResolver,
    HTTPMCPConnectivityProbe,
)
from src.core.infra.mcp_services.discovery import (
    HTTPMCPDiscoveryProvider,
    ServiceResolver,
)
from src.core.infra.mcp_services.factory import (
    build_live_connectivity_probe,
    build_live_discovery_provider,
    build_mcp_resolvers,
    build_mcp_service,
)
from src.core.infra.mcp_services.service import MCPServiceService
from src.web.deps.session import SessionDep
from src.web.middleware.context import get_tenant_id as _gtid
from src.web.middleware.context import get_user_info as _gui


def _resolve_lifespan_service(request: Request) -> LifeSpanService | None:
    """Return the lifespan service if the lifespan was started, else ``None``."""
    return cast("LifeSpanService | None", getattr(request.app.state, "lifespan_service", None))


# Live discovery/connectivity provider construction has been moved to
# ``core.infra.mcp_services.factory.build_mcp_resolvers`` /
# ``build_live_discovery_provider`` / ``build_live_connectivity_probe``
# so the ``MCPServiceRepository`` import lives in the core layer where
# repositories are allowed. The web layer keeps these names available
# as re-exports so the lifespan-side forwarder (and any tests that
# patched the symbol) keep working.
build_live_resolvers = build_mcp_resolvers
build_live_discovery_provider = build_live_discovery_provider
build_live_connectivity_probe = build_live_connectivity_probe


def get_mcp_service(
    request: Request,
    session: SessionDep,
    tenant_id: RequireTenantIdDep,
) -> MCPServiceService:
    """Build a per-request ``MCPServiceService`` on the shared session.

    PR-17.5b: when the lifespan registered a live MCP connection pool
    we forward it (plus the OAuth factory the lifespan owns) to the
    factory so the service can drive live discovery, connectivity, and
    OAuth flows. When the lifespan was bypassed (the test-app path)
    we hand the factory ``None`` so the static fakes take over.

    PR-17.5c C2: the live providers are constructed with the active
    ``tenant_id`` (not a hard-coded zero) so cross-tenant lookups
    cannot leak via the resolver.

    PR-30.6c C7: the live discovery / connectivity builders were
    moved into ``src.core.infra.mcp_services.factory`` so the web
    layer no longer reaches into ``db.dao``. The factory constructs
    the resolvers internally on the request ``AsyncSession``.
    """
    lifespan_service = _resolve_lifespan_service(request)
    discovery_provider = (
        build_live_discovery_provider(
            session=session,
            tenant_id=tenant_id,
            connection_manager=lifespan_service.mcp_connection_manager,
        )
        if lifespan_service is not None and lifespan_service.mcp_connection_manager is not None
        else None
    )
    connectivity_probe = (
        build_live_connectivity_probe(
            session=session,
            tenant_id=tenant_id,
            connection_manager=lifespan_service.mcp_connection_manager,
        )
        if lifespan_service is not None and lifespan_service.mcp_connection_manager is not None
        else None
    )
    oauth_factory = (
        getattr(lifespan_service, "mcp_oauth_manager_factory", None)
        if lifespan_service is not None
        else None
    )
    return build_mcp_service(
        session,
        discovery_provider=discovery_provider,
        connectivity_probe=connectivity_probe,
        oauth_manager_factory=oauth_factory,
    )


def get_request_tenant_id(request: Request) -> int:
    """Return the active tenant id, or 0 when unset.

    The auth middleware populates ``request.state.tenant_id``; the
    contextvar mirror is read for endpoints that don't go through the
    middleware path (e.g. test-only routes).
    """
    state_value = _gtid(request)
    if state_value:
        return state_value
    raw = request_context.get_tenant_id()
    if raw is None or raw == "":
        return 0
    try:
        return int(raw)
    except ValueError:
        return 0


def get_request_user_id(request: Request) -> str:
    """Return the authenticated user id, or ``""`` when unset.

    Empty string is the fail-closed sentinel — endpoints that actually
    require a user (oauth management, ... ) check it explicitly.
    """
    info = _gui(request)
    if info is not None and isinstance(info, dict):
        user_id = info.get("id")
        if isinstance(user_id, str):
            return user_id
    raw = request_context.get_user_id()
    return raw or ""


MCPServiceDep = Annotated[MCPServiceService, Depends(get_mcp_service)]
RequestTenantIdDep = Annotated[int, Depends(get_request_tenant_id)]
RequestUserIdDep = Annotated[str, Depends(get_request_user_id)]


def require_tenant_id(request: Request) -> int:
    """Same as ``RequestTenantIdDep`` but raises when no tenant is set."""
    value = get_request_tenant_id(request)
    if value <= 0:
        raise ValidationError(
            code="tenant.context_missing",
            message="No active workspace in request context",
        )
    return value


def require_user_id(request: Request) -> str:
    """Same as ``RequestUserIdDep`` but raises when no user is set."""
    value = get_request_user_id(request)
    if not value:
        raise ValidationError(
            code="auth.user_missing",
            message="Authenticated user is required",
        )
    return value


RequireTenantIdDep = Annotated[int, Depends(require_tenant_id)]
RequireUserIdDep = Annotated[str, Depends(require_user_id)]


__all__ = [
    "ConnectionManagerLike",
    "ConnectivityResolver",
    "HTTPMCPConnectivityProbe",
    "HTTPMCPDiscoveryProvider",
    "MCPServiceDep",
    "RequestTenantIdDep",
    "RequestUserIdDep",
    "RequireTenantIdDep",
    "RequireUserIdDep",
    "ServiceResolver",
    "build_live_connectivity_probe",
    "build_live_discovery_provider",
    "build_mcp_resolvers",
    "get_mcp_service",
    "get_request_tenant_id",
    "get_request_user_id",
    "require_tenant_id",
    "require_user_id",
]
