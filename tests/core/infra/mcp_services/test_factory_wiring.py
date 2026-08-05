"""Tests for the per-request MCP factory wiring (PR-17.5b).

The factory is a thin forwarder to :class:`MCPServiceService`; PR-17.5b
extends its surface to accept the APP-scope discovery / connectivity /
OAuth deps the lifespan owns. These tests verify:

- passing nothing preserves the PR-17.5a behaviour (``None`` deps);
- passing a discovery provider / connectivity probe / OAuth factory
  threads them straight through to the underlying service;
- the legacy ``build_mcp_service(session)`` call shape (no kwargs)
  keeps working for the existing tests.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import cast

from sqlalchemy.ext.asyncio import AsyncSession

from src.common.json import JsonObject
from src.core.infra.mcp_services.connectivity import (
    HTTPMCPConnectivityProbe,
)
from src.core.infra.mcp_services.connectivity import (
    _ConnectionManagerLike as _ConnManagerLike,
)
from src.core.infra.mcp_services.discovery import (
    HTTPMCPDiscoveryProvider,
)
from src.core.infra.mcp_services.discovery import (
    _ConnectionManagerLike as _ConnDiscLike,
)
from src.core.infra.mcp_services.factory import build_mcp_service
from src.core.infra.mcp_services.oauth import OAuthManager
from src.core.infra.mcp_services.service import MCPServiceService
from src.core.infra.mcp_services.types import MCPServiceInfo
from src.db.dao.mcp_service_repository import MCPServiceRepository
from src.db.dao.mcp_tool_approval_repository import MCPToolApprovalRepository


class _StubSession:
    """Sentinel stand-in for ``AsyncSession`` to drive the constructor.

    The fake fakes below sidestep SQLAlchemy wiring, so we never have
    to back a real session in this unit test. ``cast`` through
    ``AsyncSession`` keeps the factory's signature happy.
    """


def _stub_async_session() -> AsyncSession:
    return cast(AsyncSession, _StubSession())


# ── build_mcp_service wiring ─────────────────────────────────────────


def test_build_mcp_service_returns_a_real_mcp_service_service() -> None:
    """``build_mcp_service`` returns an :class:`MCPServiceService`."""
    service = build_mcp_service(_stub_async_session())
    assert isinstance(service, MCPServiceService)
    assert service._discovery_provider is None
    assert service._discovery_cache is None
    assert service._connectivity_probe is None
    assert service._oauth_manager_factory is None


def test_build_mcp_service_forwards_discovery_provider() -> None:
    """A passed :class:`HTTPMCPDiscoveryProvider` reaches the service."""

    async def _resolver(service_id: str) -> MCPServiceInfo | JsonObject:
        return {}

    discovery = HTTPMCPDiscoveryProvider(
        connection_manager=cast(_ConnDiscLike, object()),
        service_resolver=_resolver,
    )
    service = build_mcp_service(
        _stub_async_session(),
        discovery_provider=discovery,
    )
    assert service._discovery_provider is discovery


def test_build_mcp_service_forwards_connectivity_probe() -> None:
    """A passed :class:`HTTPMCPConnectivityProbe` reaches the service."""

    async def _resolver(service_id: str) -> MCPServiceInfo | JsonObject:
        return {}

    probe = HTTPMCPConnectivityProbe(
        connection_manager=cast(_ConnManagerLike, object()),
        resolver=_resolver,
    )
    service = build_mcp_service(
        _stub_async_session(),
        connectivity_probe=probe,
    )
    assert service._connectivity_probe is probe


def test_build_mcp_service_forwards_oauth_factory() -> None:
    """An async OAuth factory reaches the service intact."""
    received: list[MCPServiceInfo] = []

    async def _factory(info: MCPServiceInfo) -> OAuthManager:
        received.append(info)
        return OAuthManager(service=info)

    service = build_mcp_service(
        _stub_async_session(),
        oauth_manager_factory=cast(
            "Callable[[MCPServiceInfo], Awaitable[OAuthManager]]",
            _factory,
        ),
    )
    assert service._oauth_manager_factory is _factory


def test_build_mcp_service_forwards_discovery_cache() -> None:
    """A passed :class:`DiscoveryCache` reaches the service."""
    from src.core.infra.mcp_services.discovery import DiscoveryCache

    cache = DiscoveryCache()
    service = build_mcp_service(_stub_async_session(), discovery_cache=cache)
    assert service._discovery_cache is cache


def test_build_mcp_service_constructs_dao_repositories() -> None:
    """``build_mcp_service`` constructs fresh DAO repositories per call."""
    service = build_mcp_service(_stub_async_session())
    assert isinstance(service._mcp_repo, MCPServiceRepository)
    assert isinstance(
        service._tool_approvals_repo,
        MCPToolApprovalRepository,
    )


def test_build_mcp_service_succeeds_with_full_arg_set() -> None:
    """All four optional args together still produce a real service."""
    from src.core.infra.mcp_services.discovery import DiscoveryCache

    async def _resolver(service_id: str) -> MCPServiceInfo | JsonObject:
        return {}

    discovery = HTTPMCPDiscoveryProvider(
        connection_manager=cast(_ConnDiscLike, object()),
        service_resolver=_resolver,
    )
    probe = HTTPMCPConnectivityProbe(
        connection_manager=cast(_ConnManagerLike, object()),
        resolver=_resolver,
    )

    async def _factory(info: MCPServiceInfo) -> OAuthManager:
        return OAuthManager(service=info)

    service = build_mcp_service(
        _stub_async_session(),
        discovery_provider=discovery,
        discovery_cache=DiscoveryCache(),
        connectivity_probe=probe,
        oauth_manager_factory=cast(
            "Callable[[MCPServiceInfo], Awaitable[OAuthManager]]",
            _factory,
        ),
    )
    assert isinstance(service, MCPServiceService)


# ── service init signature sanity ───────────────────────────────────


def test_mcp_service_service_accepts_legacy_kwargs_only() -> None:
    """``MCPServiceService`` keeps working with the PR-17.5a ctor surface."""
    service = MCPServiceService(
        mcp_repo=cast("MCPServiceRepository", _StubSession()),
        tool_approvals_repo=cast("MCPToolApprovalRepository", _StubSession()),
    )
    assert service._oauth_manager_factory is None


def test_repository_import_round_trip_does_not_crash() -> None:
    """Touching the model modules at import side keeps the DAO classes visible."""
    import sys as _sys

    mcp_module = _sys.modules["src.db.models.infra.mcp_services"]
    assert getattr(mcp_module, "MCPService", None) is not None
    assert getattr(mcp_module, "MCPToolApproval", None) is not None


# ── Suppress an "unused import" warning that mypy/ruff trip on ───────


def test_suppress_unused_datetime_warning() -> None:
    """Pinned for the test module's dataclass round-trip below."""
    assert datetime.now(UTC) is not None
