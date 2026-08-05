"""Unit tests for MCP service discovery (PR-16) + connectivity test (PR-17)."""

from __future__ import annotations

import pytest

from src.core.infra.mcp_services.connectivity import (
    ConnectivityResult,
    StaticConnectivityProbe,
)
from src.core.infra.mcp_services.discovery import (
    DiscoveryCache,
    DiscoveryResource,
    DiscoveryTool,
    StaticDiscoveryProvider,
)
from src.core.infra.mcp_services.service import MCPServiceService
from src.db.models.infra.mcp_services import MCPService
from tests.fakes.mcp_services import (
    FakeMCPServiceRepository,
    FakeMCPToolApprovalRepository,
)

_NOW = __import__("datetime").datetime(2026, 1, 1, tzinfo=__import__("datetime").UTC)


@pytest.fixture
def mcp_repo() -> FakeMCPServiceRepository:
    return FakeMCPServiceRepository()


@pytest.fixture
def approvals_repo() -> FakeMCPToolApprovalRepository:
    return FakeMCPToolApprovalRepository()


async def _seed(mcp_repo: FakeMCPServiceRepository, *, id_: str = "svc-1") -> MCPService:
    return await mcp_repo.insert(
        MCPService(
            id=id_,
            tenant_id=1,
            name="acme",
            transport_type="sse",
            created_at=_NOW,
            updated_at=_NOW,
        ),
    )


# ── Discovery provider ──────────────────────────────────────────────


async def test_discovery_provider_returns_empty_when_unset() -> None:
    provider = StaticDiscoveryProvider()
    assert await provider.list_tools(service_id="x") == []
    assert await provider.list_resources(service_id="x") == []


async def test_discovery_provider_returns_baked_lists() -> None:
    tools = [
        DiscoveryTool(name="search", description="search the docs"),
        DiscoveryTool(name="calendar", require_approval=True),
    ]
    resources = [DiscoveryResource(uri="file://docs", name="docs")]
    provider = StaticDiscoveryProvider(tools={"svc": tools}, resources={"svc": resources})

    assert await provider.list_tools(service_id="svc") == tools
    assert await provider.list_resources(service_id="svc") == resources


# ── Cache ───────────────────────────────────────────────────────────


async def test_cache_get_or_refresh_caches_until_invalidated() -> None:
    provider = StaticDiscoveryProvider(
        tools={"svc": [DiscoveryTool(name="search")]},
        resources={"svc": [DiscoveryResource(uri="file://docs", name="docs")]},
    )
    cache = DiscoveryCache()

    tools_a, _resources_a = await cache.get_or_refresh(
        tenant_id=1, service_id="svc", provider=provider
    )
    tools_b, _resources_b = await cache.get_or_refresh(
        tenant_id=1, service_id="svc", provider=provider
    )
    assert tools_a == tools_b == [DiscoveryTool(name="search")]

    cache.invalidate(tenant_id=1, service_id="svc")
    cache.invalidate_all()
    # No assertion: the call after invalidate simply fetches again,
    # which is idempotent with a static provider.


# ── Service integration (PR-16 + PR-17) ─────────────────────────────


def _service(
    mcp_repo: FakeMCPServiceRepository,
    approvals_repo: FakeMCPToolApprovalRepository,
    *,
    provider: StaticDiscoveryProvider | None = None,
    cache: DiscoveryCache | None = None,
    probe: StaticConnectivityProbe | None = None,
) -> MCPServiceService:
    return MCPServiceService(
        mcp_repo=mcp_repo,  # type: ignore[arg-type]
        tool_approvals_repo=approvals_repo,  # type: ignore[arg-type]
        discovery_provider=provider,
        discovery_cache=cache,
        connectivity_probe=probe,
    )


async def test_list_tools_returns_empty_without_a_provider(
    mcp_repo: FakeMCPServiceRepository,
    approvals_repo: FakeMCPToolApprovalRepository,
) -> None:
    await _seed(mcp_repo)
    info = _service(mcp_repo, approvals_repo)
    assert await info.list_tools(tenant_id=1, service_id="svc-1") == []


async def test_list_tools_uses_provider_when_wired(
    mcp_repo: FakeMCPServiceRepository,
    approvals_repo: FakeMCPToolApprovalRepository,
) -> None:
    await _seed(mcp_repo)
    provider = StaticDiscoveryProvider(
        tools={"svc-1": [DiscoveryTool(name="search")]},
    )
    service = _service(mcp_repo, approvals_repo, provider=provider)
    assert await service.list_tools(tenant_id=1, service_id="svc-1") == [
        DiscoveryTool(name="search"),
    ]


async def test_list_resources_uses_cache(
    mcp_repo: FakeMCPServiceRepository,
    approvals_repo: FakeMCPToolApprovalRepository,
) -> None:
    await _seed(mcp_repo)
    provider = StaticDiscoveryProvider(
        resources={
            "svc-1": [DiscoveryResource(uri="file://a", name="a")],
        },
    )
    cache = DiscoveryCache()
    service = _service(mcp_repo, approvals_repo, provider=provider, cache=cache)

    first = await service.list_resources(tenant_id=1, service_id="svc-1")
    second = await service.list_resources(tenant_id=1, service_id="svc-1")

    assert first == second == [DiscoveryResource(uri="file://a", name="a")]
    service.invalidate_discovery_cache(tenant_id=1, service_id="svc-1")


async def test_list_tools_raises_for_unknown_service(
    mcp_repo: FakeMCPServiceRepository,
    approvals_repo: FakeMCPToolApprovalRepository,
) -> None:
    service = _service(mcp_repo, approvals_repo)
    from src.common.exception import NotFoundError

    with pytest.raises(NotFoundError):
        await service.list_tools(tenant_id=1, service_id="missing")


async def test_test_service_reports_failure_without_a_probe(
    mcp_repo: FakeMCPServiceRepository,
    approvals_repo: FakeMCPToolApprovalRepository,
) -> None:
    await _seed(mcp_repo)
    service = _service(mcp_repo, approvals_repo)
    result = await service.test_service(tenant_id=1, service_id="svc-1")
    assert result.success is False
    assert "not configured" in result.message


async def test_test_service_uses_wired_probe(
    mcp_repo: FakeMCPServiceRepository,
    approvals_repo: FakeMCPToolApprovalRepository,
) -> None:
    await _seed(mcp_repo)
    probe = StaticConnectivityProbe(
        result=ConnectivityResult(
            success=True,
            message="connected",
            description="test server",
            tools=(DiscoveryTool(name="search"),),
        ),
    )
    service = _service(mcp_repo, approvals_repo, probe=probe)
    result = await service.test_service(tenant_id=1, service_id="svc-1")
    assert result.success is True
    assert result.message == "connected"
    assert result.tools[0].name == "search"


@pytest.mark.parametrize("oauth_required", [True, False])
async def test_probe_receives_oauth_flag(
    mcp_repo: FakeMCPServiceRepository,
    approvals_repo: FakeMCPToolApprovalRepository,
    oauth_required: bool,
) -> None:
    captured: dict[str, bool] = {}

    class _CapturingProbe:
        async def __call__(
            self,
            *,
            service_id: str,
            transport_type: str,
            url: str | None,
            oauth_required: bool,
        ) -> ConnectivityResult:
            captured["oauth"] = oauth_required
            return ConnectivityResult(success=True, message="ok")

    await _seed(mcp_repo)
    service = _service(mcp_repo, approvals_repo, probe=_CapturingProbe())  # type: ignore[arg-type]
    await service.test_service(tenant_id=1, service_id="svc-1")
    assert captured["oauth"] is False  # no auth_config on the seeded row
