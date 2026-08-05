"""Unit tests for the MCP service CRUD layer (PR-15)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.common.exception import NotFoundError, ValidationError
from src.core.infra.mcp_services.service import MCPServiceService
from src.db.models.infra.mcp_services import MCPService
from tests.fakes.mcp_services import (
    FakeMCPServiceRepository,
    FakeMCPToolApprovalRepository,
)

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture
def mcp_repo() -> FakeMCPServiceRepository:
    return FakeMCPServiceRepository()


@pytest.fixture
def approvals_repo() -> FakeMCPToolApprovalRepository:
    return FakeMCPToolApprovalRepository()


@pytest.fixture
def service(
    mcp_repo: FakeMCPServiceRepository,
    approvals_repo: FakeMCPToolApprovalRepository,
) -> MCPServiceService:
    return MCPServiceService(
        mcp_repo=mcp_repo,  # type: ignore[arg-type]
        tool_approvals_repo=approvals_repo,  # type: ignore[arg-type]
    )


async def _seed(
    mcp_repo: FakeMCPServiceRepository,
    *,
    name: str = "acme",
    tenant_id: int = 1,
    transport_type: str = "sse",
    is_builtin: bool = False,
    created_at: datetime = _NOW,
) -> MCPService:
    row = MCPService(
        id="seed-" + name,
        tenant_id=tenant_id,
        name=name,
        description=None,
        enabled=True,
        transport_type=transport_type,
        url="https://example.com/mcp",
        created_at=created_at,
        updated_at=created_at,
        is_builtin=is_builtin,
    )
    return await mcp_repo.insert(row)


# ── Create ──────────────────────────────────────────────────────────


async def test_create_service_assigns_a_new_id(service: MCPServiceService) -> None:
    info = await service.create_service(
        tenant_id=1,
        name="acme",
        transport_type="sse",
    )
    assert info.id, "service id should be a non-empty uuid"
    assert info.name == "acme"
    assert info.enabled is True
    assert info.advanced_config == {"timeout": 30, "retry_count": 3, "retry_delay": 1}


async def test_create_service_trims_blank_name(service: MCPServiceService) -> None:
    with pytest.raises(ValidationError) as excinfo:
        await service.create_service(tenant_id=1, name="   ", transport_type="sse")
    assert excinfo.value.code == "mcp_service.name_required"


async def test_create_service_rejects_unknown_transport(service: MCPServiceService) -> None:
    with pytest.raises(ValidationError) as excinfo:
        await service.create_service(
            tenant_id=1,
            name="acme",
            transport_type="bogus",
        )
    assert excinfo.value.code == "mcp_service.invalid_transport"


async def test_create_service_rejects_stdio(service: MCPServiceService) -> None:
    with pytest.raises(ValidationError) as excinfo:
        await service.create_service(
            tenant_id=1,
            name="acme",
            transport_type="stdio",
        )
    assert excinfo.value.code == "mcp_service.stdio_disabled"


async def test_create_service_strips_secret_auth_fields(
    service: MCPServiceService,
) -> None:
    info = await service.create_service(
        tenant_id=1,
        name="acme",
        transport_type="sse",
        auth_config={
            "auth_type": "api_key",
            "api_key": "should-not-persist",
            "token": "should-not-persist",
            "scopes": ["read"],
        },
    )
    assert info.auth_config is not None
    assert "api_key" not in info.auth_config
    assert "token" not in info.auth_config
    assert info.auth_config["auth_type"] == "api_key"


# ── Read ────────────────────────────────────────────────────────────


async def test_get_service_returns_one(
    service: MCPServiceService,
    mcp_repo: FakeMCPServiceRepository,
) -> None:
    await _seed(mcp_repo, name="alpha")

    info = await service.get_service(tenant_id=1, id="seed-alpha")

    assert info.name == "alpha"


async def test_get_service_raises_for_missing(service: MCPServiceService) -> None:
    with pytest.raises(NotFoundError) as excinfo:
        await service.get_service(tenant_id=1, id="missing")
    assert excinfo.value.code == "mcp_service.not_found"


async def test_get_service_scopes_by_tenant(
    service: MCPServiceService,
    mcp_repo: FakeMCPServiceRepository,
) -> None:
    await _seed(mcp_repo, name="other", tenant_id=2)
    with pytest.raises(NotFoundError):
        await service.get_service(tenant_id=1, id="seed-other")


async def test_list_services_excludes_builtin(
    service: MCPServiceService,
    mcp_repo: FakeMCPServiceRepository,
) -> None:
    await _seed(mcp_repo, name="user")
    await _seed(mcp_repo, name="built", is_builtin=True, tenant_id=0)

    listed = await service.list_services(tenant_id=1)

    assert [info.name for info in listed] == ["user"]


# ── Update ──────────────────────────────────────────────────────────


async def test_update_service_patches_columns(
    service: MCPServiceService,
    mcp_repo: FakeMCPServiceRepository,
) -> None:
    await _seed(mcp_repo, name="alpha")

    info = await service.update_service(
        tenant_id=1,
        id="seed-alpha",
        description="new description",
        enabled=False,
    )

    assert info.description == "new description"
    assert info.enabled is False
    assert info.updated_at > _NOW


async def test_update_service_rejects_blank_name(
    service: MCPServiceService,
    mcp_repo: FakeMCPServiceRepository,
) -> None:
    await _seed(mcp_repo, name="alpha")
    with pytest.raises(ValidationError) as excinfo:
        await service.update_service(tenant_id=1, id="seed-alpha", name="  ")
    assert excinfo.value.code == "mcp_service.name_required"


async def test_update_service_rejects_builtin(
    service: MCPServiceService,
    mcp_repo: FakeMCPServiceRepository,
) -> None:
    await _seed(mcp_repo, name="built", is_builtin=True, tenant_id=1)
    with pytest.raises(ValidationError) as excinfo:
        await service.update_service(tenant_id=1, id="seed-built", description="new")
    assert excinfo.value.code == "mcp_service.builtin_immutable"


async def test_update_service_strips_secret_auth_fields(
    service: MCPServiceService,
    mcp_repo: FakeMCPServiceRepository,
) -> None:
    await _seed(mcp_repo, name="alpha")
    info = await service.update_service(
        tenant_id=1,
        id="seed-alpha",
        auth_config={
            "auth_type": "oauth",
            "api_key": "should-be-dropped",
            "scopes": ["read"],
        },
    )
    assert info.auth_config is not None
    assert info.auth_config.get("auth_type") == "oauth"
    assert info.auth_config.get("scopes") == ["read"]
    assert "api_key" not in info.auth_config


async def test_update_service_rejects_stdio_transition(
    service: MCPServiceService,
    mcp_repo: FakeMCPServiceRepository,
) -> None:
    await _seed(mcp_repo, name="alpha", transport_type="sse")
    with pytest.raises(ValidationError) as excinfo:
        await service.update_service(
            tenant_id=1,
            id="seed-alpha",
            transport_type="stdio",
        )
    assert excinfo.value.code == "mcp_service.stdio_disabled"


# ── Delete ──────────────────────────────────────────────────────────


async def test_delete_service_returns_true_when_present(
    service: MCPServiceService,
    mcp_repo: FakeMCPServiceRepository,
) -> None:
    await _seed(mcp_repo, name="alpha")
    assert await service.delete_service(tenant_id=1, id="seed-alpha") is True
    assert await service.list_services(tenant_id=1) == []


async def test_delete_service_returns_false_when_missing(
    service: MCPServiceService,
) -> None:
    assert await service.delete_service(tenant_id=1, id="missing") is False


async def test_delete_service_rejects_builtin(
    service: MCPServiceService,
    mcp_repo: FakeMCPServiceRepository,
) -> None:
    await _seed(mcp_repo, name="built", is_builtin=True)
    with pytest.raises(ValidationError) as excinfo:
        await service.delete_service(tenant_id=1, id="seed-built")
    assert excinfo.value.code == "mcp_service.builtin_immutable"


# ── Tool approvals ──────────────────────────────────────────────────


async def test_list_tool_approvals_rejects_unknown_service(
    service: MCPServiceService,
) -> None:
    with pytest.raises(NotFoundError):
        await service.list_tool_approvals(tenant_id=1, service_id="missing")


async def test_set_tool_approval_upserts(
    service: MCPServiceService,
    mcp_repo: FakeMCPServiceRepository,
) -> None:
    await _seed(mcp_repo, name="alpha")

    first = await service.set_tool_approval(
        tenant_id=1, service_id="seed-alpha", tool_name="search", require_approval=True
    )
    assert first.require_approval is True

    second = await service.set_tool_approval(
        tenant_id=1, service_id="seed-alpha", tool_name="search", require_approval=False
    )
    assert second.require_approval is False

    approvals = await service.list_tool_approvals(tenant_id=1, service_id="seed-alpha")
    assert len(approvals) == 1
    assert approvals[0].require_approval is False


async def test_set_tool_approval_rejects_blank_tool_name(
    service: MCPServiceService,
    mcp_repo: FakeMCPServiceRepository,
) -> None:
    await _seed(mcp_repo, name="alpha")
    with pytest.raises(ValidationError) as excinfo:
        await service.set_tool_approval(
            tenant_id=1, service_id="seed-alpha", tool_name="", require_approval=True
        )
    assert excinfo.value.code == "mcp_service.tool_name_required"
