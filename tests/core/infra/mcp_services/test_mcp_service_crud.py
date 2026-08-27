"""Unit tests for the MCP service CRUD layer."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from src.common.exception import ConflictError, NotFoundError, ValidationError
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
    """Tool-approval repo mock; the CRUD tests don't exercise it."""
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


@pytest.fixture
def service(
    mcp_repo: AsyncMock,
    approvals_repo: AsyncMock,
) -> MCPServiceService:
    return MCPServiceService(
        mcp_repo=mcp_repo,
        tool_approvals_repo=approvals_repo,
    )


async def _seed(
    rows: dict[str, MCPService],
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
    rows[row.id] = row
    return row


# ── Create ──────────────────────────────────────────────────────────


async def test_create_service_assigns_a_new_id(service: MCPServiceService) -> None:
    info = await service.create_service(
        tenant_id=1,
        name="acme",
        transport_type="sse",
        url="https://example.com/mcp",
    )
    assert info.id, "service id should be a non-empty uuid"
    assert info.name == "acme"
    assert info.enabled is True
    assert info.advanced_config == {"timeout": 30, "retry_count": 3, "retry_delay": 1}


async def test_create_service_trims_blank_name(service: MCPServiceService) -> None:
    with pytest.raises(ValidationError) as excinfo:
        await service.create_service(
            tenant_id=1, name="   ", transport_type="sse", url="https://example.com/mcp"
        )
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


async def test_create_service_duplicate_name_raises_conflict(
    service: MCPServiceService,
    mcp_repo: AsyncMock,
) -> None:
    """Creating two services with the same (tenant, name) raises 409.

    Mirrors Go's ``MCPServiceService.CreateMCPService`` → 1005
    ``ErrConflict`` path; the Python side uses the explicit
    ``mcp_service.duplicate_name`` code so the web layer can render a
    targeted error message.
    """
    await service.create_service(
        tenant_id=1, name="dup", transport_type="sse", url="https://example.com/mcp"
    )
    with pytest.raises(ConflictError) as excinfo:
        await service.create_service(
            tenant_id=1, name="dup", transport_type="sse", url="https://example.com/mcp"
        )
    assert excinfo.value.code == "mcp_service.duplicate_name"


async def test_create_service_same_name_different_tenant_is_allowed(
    service: MCPServiceService,
) -> None:
    """Name uniqueness is scoped to (tenant_id, name), not global."""
    await service.create_service(
        tenant_id=1, name="shared", transport_type="sse", url="https://example.com/mcp"
    )
    # No conflict on a different tenant.
    info = await service.create_service(
        tenant_id=2, name="shared", transport_type="sse", url="https://example.com/mcp"
    )
    assert info.tenant_id == 2


async def test_create_service_preserves_auth_config(
    service: MCPServiceService,
) -> None:
    info = await service.create_service(
        tenant_id=1,
        name="acme",
        transport_type="sse",
        url="https://example.com/mcp",
        auth_config={
            "auth_type": "api_key",
            "api_key": "should-be-preserved",
            "token": "should-be-preserved",
            "scopes": ["read"],
        },
    )
    assert info.auth_config is not None
    # Mirrors Go: credentials are kept on user services verbatim.
    assert info.auth_config["auth_type"] == "api_key"
    assert info.auth_config["api_key"] == "should-be-preserved"
    assert info.auth_config["token"] == "should-be-preserved"
    assert info.auth_config["scopes"] == ["read"]


# ── Read ────────────────────────────────────────────────────────────


async def test_get_service_returns_one(
    service: MCPServiceService,
    mcp_rows: dict[str, MCPService],
) -> None:
    await _seed(mcp_rows, name="alpha")

    info = await service.get_service(tenant_id=1, id="seed-alpha")

    assert info.name == "alpha"


async def test_get_service_raises_for_missing(service: MCPServiceService) -> None:
    with pytest.raises(NotFoundError) as excinfo:
        await service.get_service(tenant_id=1, id="missing")
    assert excinfo.value.code == "mcp_service.not_found"


async def test_get_service_scopes_by_tenant(
    service: MCPServiceService,
    mcp_rows: dict[str, MCPService],
) -> None:
    await _seed(mcp_rows, name="other", tenant_id=2)
    with pytest.raises(NotFoundError):
        await service.get_service(tenant_id=1, id="seed-other")


async def test_list_services_excludes_builtin(
    service: MCPServiceService,
    mcp_rows: dict[str, MCPService],
) -> None:
    await _seed(mcp_rows, name="user")
    await _seed(mcp_rows, name="built", is_builtin=True, tenant_id=0)

    listed = await service.list_services(tenant_id=1)

    assert [info.name for info in listed] == ["user"]


# ── Update ──────────────────────────────────────────────────────────


async def test_update_service_patches_columns(
    service: MCPServiceService,
    mcp_rows: dict[str, MCPService],
) -> None:
    await _seed(mcp_rows, name="alpha")

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
    mcp_rows: dict[str, MCPService],
) -> None:
    await _seed(mcp_rows, name="alpha")
    with pytest.raises(ValidationError) as excinfo:
        await service.update_service(tenant_id=1, id="seed-alpha", name="  ")
    assert excinfo.value.code == "mcp_service.name_required"


async def test_update_service_rejects_builtin(
    service: MCPServiceService,
    mcp_rows: dict[str, MCPService],
) -> None:
    await _seed(mcp_rows, name="built", is_builtin=True, tenant_id=1)
    with pytest.raises(ValidationError) as excinfo:
        await service.update_service(tenant_id=1, id="seed-built", description="new")
    assert excinfo.value.code == "mcp_service.builtin_immutable"


async def test_update_service_preserves_auth_config(
    service: MCPServiceService,
    mcp_rows: dict[str, MCPService],
) -> None:
    await _seed(mcp_rows, name="alpha")
    info = await service.update_service(
        tenant_id=1,
        id="seed-alpha",
        auth_config={
            "auth_type": "oauth",
            "api_key": "should-be-preserved",
            "scopes": ["read"],
        },
    )
    assert info.auth_config is not None
    assert info.auth_config.get("auth_type") == "oauth"
    assert info.auth_config.get("scopes") == ["read"]
    assert info.auth_config.get("api_key") == "should-be-preserved"


async def test_update_service_rejects_stdio_transition(
    service: MCPServiceService,
    mcp_rows: dict[str, MCPService],
) -> None:
    await _seed(mcp_rows, name="alpha", transport_type="sse")
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
    mcp_rows: dict[str, MCPService],
) -> None:
    await _seed(mcp_rows, name="alpha")
    assert await service.delete_service(tenant_id=1, id="seed-alpha") is True
    assert await service.list_services(tenant_id=1) == []


async def test_delete_service_returns_false_when_missing(
    service: MCPServiceService,
) -> None:
    assert await service.delete_service(tenant_id=1, id="missing") is False
