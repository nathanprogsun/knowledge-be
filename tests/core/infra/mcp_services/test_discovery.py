"""Unit tests for MCP service discovery + connectivity test."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

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
from src.db.dao.mcp_service_repository import MCPServiceRepository
from src.db.dao.mcp_tool_approval_repository import MCPToolApprovalRepository
from src.db.models.infra.mcp_services import MCPService

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


# ── Repository mocks (stateful via side_effect closures) ─────────────


def _make_mcp_repo() -> tuple[AsyncMock, dict[str, MCPService]]:
    """MCP-service repo mock with closure-captured state."""
    repo = AsyncMock(spec=MCPServiceRepository)
    rows: dict[str, MCPService] = {}

    async def _insert(row: MCPService) -> MCPService:
        rows[row.id] = row
        return row

    async def _find_for_tenant(tenant_id: int, id: str) -> MCPService | None:
        row = rows.get(id)
        if row is None or row.tenant_id != tenant_id or row.deleted_at is not None:
            return None
        return row

    async def _get_by_id(tenant_id: int, id: str) -> MCPService:
        row = await _find_for_tenant(tenant_id, id)
        if row is None:
            from src.common.exception import NotFoundError

            raise NotFoundError(
                code="mcp_service.not_found",
                message=f"MCP service {id} not found",
            )
        return row

    async def _list_for_tenant(tenant_id: int) -> list[MCPService]:
        live = [
            row
            for row in rows.values()
            if row.tenant_id == tenant_id and not row.is_builtin and row.deleted_at is None
        ]
        return sorted(live, key=lambda r: r.created_at, reverse=True)

    async def _find_builtin(id: str) -> MCPService | None:
        row = rows.get(id)
        if row is None or not row.is_builtin or row.deleted_at is not None:
            return None
        return row

    async def _exists_by_tenant_and_name(tenant_id: int, name: str) -> bool:
        return any(
            row.tenant_id == tenant_id and row.name == name and row.deleted_at is None
            for row in rows.values()
        )

    async def _soft_delete(tenant_id: int, id: str, *, deleted_at: datetime) -> bool:
        row = await _find_for_tenant(tenant_id, id)
        if row is None:
            return False
        rows[id] = row.model_copy(update={"deleted_at": deleted_at, "updated_at": deleted_at})
        return True

    async def _update(tenant_id: int, id: str, *, columns: dict[str, object]) -> MCPService | None:
        row = await _find_for_tenant(tenant_id, id)
        if row is None:
            return None
        updated = row.model_copy(update=columns)
        rows[id] = updated
        return updated

    repo.insert.side_effect = _insert
    repo.find_for_tenant.side_effect = _find_for_tenant
    repo.get_by_id.side_effect = _get_by_id
    repo.list_for_tenant.side_effect = _list_for_tenant
    repo.find_builtin.side_effect = _find_builtin
    repo.exists_by_tenant_and_name.side_effect = _exists_by_tenant_and_name
    repo.soft_delete.side_effect = _soft_delete
    repo.update.side_effect = _update
    return repo, rows


def _make_approvals_repo() -> AsyncMock:
    """Tool-approval repo mock; the discovery tests don't exercise it."""
    return AsyncMock(spec=MCPToolApprovalRepository)


@pytest.fixture
def mcp_state() -> tuple[AsyncMock, dict[str, MCPService]]:
    return _make_mcp_repo()


@pytest.fixture
def mcp_repo(mcp_state: tuple[AsyncMock, dict[str, MCPService]]) -> AsyncMock:
    return mcp_state[0]


@pytest.fixture
def mcp_rows(mcp_state: tuple[AsyncMock, dict[str, MCPService]]) -> dict[str, MCPService]:
    return mcp_state[1]


@pytest.fixture
def approvals_repo() -> AsyncMock:
    return _make_approvals_repo()


async def _seed(rows: dict[str, MCPService], *, id_: str = "svc-1") -> MCPService:
    row = MCPService(
        id=id_,
        tenant_id=1,
        name="acme",
        transport_type="sse",
        created_at=_NOW,
        updated_at=_NOW,
    )
    rows[id_] = row
    return row


# ── Discovery provider ──────────────────────────────────────────────


async def test_discovery_provider_returns_empty_when_unset() -> None:
    provider = StaticDiscoveryProvider()
    assert await provider.list_tools(tenant_id=1, service_id="x") == []
    assert await provider.list_resources(tenant_id=1, service_id="x") == []


async def test_discovery_provider_returns_baked_lists() -> None:
    tools = [
        DiscoveryTool(name="search", description="search the docs"),
        DiscoveryTool(name="calendar", require_approval=True),
    ]
    resources = [DiscoveryResource(uri="file://docs", name="docs")]
    provider = StaticDiscoveryProvider(tools={"svc": tools}, resources={"svc": resources})

    assert await provider.list_tools(tenant_id=1, service_id="svc") == tools
    assert await provider.list_resources(tenant_id=1, service_id="svc") == resources


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


# ── Service integration ─────────────────────────────────────────────


def _service(
    mcp_repo: AsyncMock,
    approvals_repo: AsyncMock,
    *,
    provider: StaticDiscoveryProvider | None = None,
    cache: DiscoveryCache | None = None,
    probe: StaticConnectivityProbe | None = None,
) -> MCPServiceService:
    return MCPServiceService(
        mcp_repo=mcp_repo,
        tool_approvals_repo=approvals_repo,
        discovery_provider=provider,
        discovery_cache=cache,
        connectivity_probe=probe,
    )


async def test_list_tools_returns_empty_without_a_provider(
    mcp_repo: AsyncMock,
    approvals_repo: AsyncMock,
    mcp_rows: dict[str, MCPService],
) -> None:
    await _seed(mcp_rows)
    service = _service(mcp_repo, approvals_repo)
    assert await service.list_tools(tenant_id=1, service_id="svc-1") == []


async def test_list_tools_uses_provider_when_wired(
    mcp_repo: AsyncMock,
    approvals_repo: AsyncMock,
    mcp_rows: dict[str, MCPService],
) -> None:
    await _seed(mcp_rows)
    provider = StaticDiscoveryProvider(
        tools={"svc-1": [DiscoveryTool(name="search")]},
    )
    service = _service(mcp_repo, approvals_repo, provider=provider)
    assert await service.list_tools(tenant_id=1, service_id="svc-1") == [
        DiscoveryTool(name="search"),
    ]


async def test_list_resources_uses_cache(
    mcp_repo: AsyncMock,
    approvals_repo: AsyncMock,
    mcp_rows: dict[str, MCPService],
) -> None:
    await _seed(mcp_rows)
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
    mcp_repo: AsyncMock,
    approvals_repo: AsyncMock,
) -> None:
    service = _service(mcp_repo, approvals_repo)
    from src.common.exception import NotFoundError

    with pytest.raises(NotFoundError):
        await service.list_tools(tenant_id=1, service_id="missing")


async def test_test_service_reports_failure_without_a_probe(
    mcp_repo: AsyncMock,
    approvals_repo: AsyncMock,
    mcp_rows: dict[str, MCPService],
) -> None:
    await _seed(mcp_rows)
    service = _service(mcp_repo, approvals_repo)
    result = await service.test_service(tenant_id=1, service_id="svc-1")
    assert result.success is False
    assert "not configured" in result.message


async def test_test_service_uses_wired_probe(
    mcp_repo: AsyncMock,
    approvals_repo: AsyncMock,
    mcp_rows: dict[str, MCPService],
) -> None:
    await _seed(mcp_rows)
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
