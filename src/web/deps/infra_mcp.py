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

PR-17.5c C4: the per-request live ``HTTPMCPDiscoveryProvider`` /
``HTTPMCPConnectivityProbe`` builders (:func:`build_live_*`) live here
on the web side because they need to construct the
``MCPServiceRepository`` on the request ``AsyncSession``. The
previous implementation had them in ``core.infra.mcp_services.factory``,
which made ``web`` indirectly import ``db.dao.mcp_service_repository``
through the factory — the layer-violation hard rule requires ``web``
to import ``db`` directly when it needs to.

Adds two FastAPI dependency factories for ``tenant_id`` and ``user_id``
so the router can read them from the per-request auth state without
reaching into the context layer directly.
"""

from __future__ import annotations

from typing import Annotated, cast

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from src.app_context import request_context
from src.app_context.registry import LifeSpanService
from src.common.exception import ValidationError
from src.common.json import JsonObject
from src.core.infra.mcp_services.connectivity import (
    ConnectivityResolver,
    HTTPMCPConnectivityProbe,
)
from src.core.infra.mcp_services.connectivity import (
    _ConnectionManagerLike as _ConnManagerLikeProbe,
)
from src.core.infra.mcp_services.discovery import (
    HTTPMCPDiscoveryProvider,
    ServiceResolver,
)
from src.core.infra.mcp_services.discovery import (
    _ConnectionManagerLike as _ConnManagerLikeDisc,
)
from src.core.infra.mcp_services.factory import build_mcp_service
from src.core.infra.mcp_services.service import MCPServiceService
from src.core.infra.mcp_services.types import MCPServiceInfo
from src.db.dao.mcp_service_repository import MCPServiceRepository
from src.web.deps.session import SessionDep
from src.web.middleware.context import get_tenant_id as _gtid
from src.web.middleware.context import get_user_info as _gui


def _resolve_lifespan_service(request: Request) -> LifeSpanService | None:
    """Return the lifespan service if the lifespan was started, else ``None``."""
    return cast("LifeSpanService | None", getattr(request.app.state, "lifespan_service", None))


# ``ConnectionManagerLike`` is the static-type face of the AI-layer
# ``MCPConnectionManager``. The web layer cannot import that class
# directly (layer directionality — web must not reach into ai), so we
# re-export the Protocol defined in ``core.infra.mcp_services.discovery``
# under its public name. The two core protocols (``_ConnManagerLikeDisc``
# and ``_ConnManagerLikeProbe``) are the same shape; aliasing keeps the
# public ``ConnectionManagerLike`` as the canonical type for downstream
# annotations.
ConnectionManagerLike = _ConnManagerLikeDisc


def build_live_resolvers(
    *,
    session: AsyncSession,
    tenant_id: int,
) -> tuple[ServiceResolver, ConnectivityResolver]:
    """Return ``(discovery_resolver, connectivity_resolver)`` for one request.

    PR-17.5c C2: takes the active ``tenant_id`` so the resolver lookup
    is tenant-scoped (the previous ``tenant_id=0`` hard-coding would
    have leaked cross-tenant OAuth tokens / URLs).
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
    connection_manager: ConnectionManagerLike,
) -> HTTPMCPDiscoveryProvider:
    """Construct an :class:`HTTPMCPDiscoveryProvider` for one request.

    PR-17.5c C4: lives on the ``web`` side so the ``MCPServiceRepository``
    import stays local to the web layer's responsibility for
    constructing repositories from the request ``AsyncSession``.
    """
    service_resolver, _ = build_live_resolvers(
        session=session,
        tenant_id=tenant_id,
    )
    return HTTPMCPDiscoveryProvider(
        connection_manager=cast(_ConnManagerLikeDisc, connection_manager),
        service_resolver=service_resolver,
    )


def build_live_connectivity_probe(
    *,
    session: AsyncSession,
    tenant_id: int,
    connection_manager: ConnectionManagerLike,
) -> HTTPMCPConnectivityProbe:
    """Construct an :class:`HTTPMCPConnectivityProbe` for one request."""
    _, connectivity_resolver = build_live_resolvers(
        session=session,
        tenant_id=tenant_id,
    )
    return HTTPMCPConnectivityProbe(
        connection_manager=cast(_ConnManagerLikeProbe, connection_manager),
        resolver=connectivity_resolver,
    )


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
    "MCPServiceDep",
    "RequestTenantIdDep",
    "RequestUserIdDep",
    "RequireTenantIdDep",
    "RequireUserIdDep",
    "build_live_connectivity_probe",
    "build_live_discovery_provider",
    "build_live_resolvers",
    "get_mcp_service",
    "get_request_tenant_id",
    "get_request_user_id",
    "require_tenant_id",
    "require_user_id",
]
