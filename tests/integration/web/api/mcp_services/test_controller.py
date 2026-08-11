"""Web-layer tests for the MCP service router.

Exercises the 13 endpoints over HTTP via ``TestClient`` with
``get_mcp_service`` overridden to a real ``MCPServiceService`` backed
by ``AsyncMock(spec=...)`` repositories configured with stateful
closures. The discovery + connectivity probes are also injected so the
routes reach a known state.

Uses the shared ``web_app`` fixture (header-based auth) and applies
the service dep override on it; the real ``require_auth`` dep resolves
the principal via the ``X-User-Id/X-Tenant-ID/X-Roles`` header trio.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.common.exception import NotFoundError
from src.core.infra.mcp_services.connectivity import (
    ConnectivityResult,
    StaticConnectivityProbe,
)
from src.core.infra.mcp_services.discovery import (
    DiscoveryResource,
    DiscoveryTool,
    StaticDiscoveryProvider,
)
from src.core.infra.mcp_services.service import MCPServiceService
from src.db.dao.mcp_service_repository import MCPServiceRepository
from src.db.dao.mcp_tool_approval_repository import MCPToolApprovalRepository
from src.db.models.infra.mcp_services import MCPToolApproval
from src.web.deps.infra_mcp import get_mcp_service


@pytest.fixture
def mcp_repo() -> AsyncMock:
    """``AsyncMock(spec=MCPServiceRepository)`` with stateful closures."""
    repo = AsyncMock(spec=MCPServiceRepository)
    rows: dict[str, MCPServiceStub] = {}

    async def _find_for_tenant(tenant_id: int, id: str):
        row = rows.get(id)
        if row is None or row.tenant_id != tenant_id or row.deleted_at is not None:
            return None
        return row

    async def _get_by_id(tenant_id: int, id: str):
        row = await _find_for_tenant(tenant_id, id)
        if row is None:
            raise NotFoundError(
                code="mcp_service.not_found",
                message=f"MCP service {id} not found",
            )
        return row

    async def _list_for_tenant(tenant_id: int) -> list:
        out = [
            r
            for r in rows.values()
            if r.tenant_id == tenant_id and not r.is_builtin and r.deleted_at is None
        ]
        return sorted(out, key=lambda r: r.created_at, reverse=True)

    async def _find_builtin(id: str):
        row = rows.get(id)
        if row is None or not row.is_builtin or row.deleted_at is not None:
            return None
        return row

    async def _exists_by_tenant_and_name(tenant_id: int, name: str) -> bool:
        return any(
            row.tenant_id == tenant_id and row.name == name and row.deleted_at is None
            for row in rows.values()
        )

    async def _insert(row) -> object:  # type: ignore[no-untyped-def]
        rows[row.id] = row
        return row

    async def _soft_delete(tenant_id: int, id: str, *, deleted_at: datetime) -> bool:
        row = await _find_for_tenant(tenant_id, id)
        if row is None:
            return False
        rows[id] = row.model_copy(
            update={"deleted_at": deleted_at, "updated_at": deleted_at},
        )
        return True

    async def _update(tenant_id: int, id: str, *, columns: dict) -> object:
        row = await _find_for_tenant(tenant_id, id)
        if row is None:
            return None
        updated = row.model_copy(update=columns)
        rows[id] = updated
        return updated

    repo.find_for_tenant.side_effect = _find_for_tenant
    repo.get_by_id.side_effect = _get_by_id
    repo.list_for_tenant.side_effect = _list_for_tenant
    repo.find_builtin.side_effect = _find_builtin
    repo.exists_by_tenant_and_name.side_effect = _exists_by_tenant_and_name
    repo.insert.side_effect = _insert
    repo.soft_delete.side_effect = _soft_delete
    repo.update.side_effect = _update
    repo._rows = rows  # type: ignore[attr-defined]
    return repo


@pytest.fixture
def approvals_repo() -> AsyncMock:
    """``AsyncMock(spec=MCPToolApprovalRepository)`` with stateful upsert."""
    repo = AsyncMock(spec=MCPToolApprovalRepository)
    rows: dict[str, MCPToolApproval] = {}

    async def _list_by_service(tenant_id: int, service_id: str) -> list[MCPToolApproval]:
        live = [r for r in rows.values() if r.tenant_id == tenant_id and r.service_id == service_id]
        return sorted(live, key=lambda r: r.tool_name)

    async def _upsert(*, row: MCPToolApproval) -> MCPToolApproval:
        existing = next(
            (
                r
                for r in rows.values()
                if (r.tenant_id, r.service_id, r.tool_name)
                == (row.tenant_id, row.service_id, row.tool_name)
            ),
            None,
        )
        stored_id = existing.id if existing is not None else row.id
        merged = row.model_copy(update={"id": stored_id, "updated_at": row.created_at})
        rows[stored_id] = merged
        return merged

    repo.list_by_service.side_effect = _list_by_service
    repo.upsert.side_effect = _upsert
    repo._rows = rows  # type: ignore[attr-defined]
    return repo


@pytest.fixture(autouse=True)
def _override_services(
    web_app: FastAPI,
    mcp_repo: AsyncMock,
    approvals_repo: AsyncMock,
) -> FastAPI:
    """Override ``get_mcp_service`` on the shared web app (autouse)."""
    web_app.dependency_overrides[get_mcp_service] = lambda: MCPServiceService(
        mcp_repo=mcp_repo,
        tool_approvals_repo=approvals_repo,
        discovery_provider=StaticDiscoveryProvider(
            tools={"seed-alpha": [DiscoveryTool(name="search")]},
            resources={
                "seed-alpha": [
                    DiscoveryResource(uri="file://docs", name="docs"),
                ],
            },
        ),
        connectivity_probe=StaticConnectivityProbe(
            result=ConnectivityResult(
                success=True,
                message="connected",
                description="test server",
                tools=(DiscoveryTool(name="search"),),
            ),
        ),
    )
    return web_app


async def _seed(
    mcp_repo: AsyncMock,
    *,
    name: str = "alpha",
    tenant_id: int | None = None,
) -> str:
    from src.db.models.infra.mcp_services import MCPService

    now = datetime.now(UTC)
    row = MCPService(
        id="seed-" + name,
        tenant_id=tenant_id if tenant_id is not None else _MCP_SEED_TENANT_ID,
        name=name,
        transport_type="sse",
        url="https://example.com/mcp",
        created_at=now,
        updated_at=now,
    )
    await mcp_repo.insert(row)
    return row.id


# Module-level placeholder; the ``_bind_tenant_id_to_admin`` autouse
# fixture below rebinds it to the per-test minted admin tenant so the
# seed rows match the principal the authed client presents.
_MCP_SEED_TENANT_ID = 1


@pytest.fixture(autouse=True)
def _bind_tenant_id_to_admin(
    admin_user: tuple[int, int],
) -> None:
    """Pin ``_MCP_SEED_TENANT_ID`` to the minted admin tenant per test."""
    global _MCP_SEED_TENANT_ID
    _MCP_SEED_TENANT_ID = admin_user[1]


# Sentinel type alias to keep the find_for_tenant type narrow without
# importing the real model class at module level (the tests use it
# through the insert call).
MCPServiceStub = object


# ── POST /mcp-services ──────────────────────────────────────────────


async def test_create_service_returns_201_envelope(web_authed_client: TestClient) -> None:
    resp = web_authed_client.post(
        "/mcp-services",
        json={"name": "alpha", "transport_type": "sse", "url": "https://example.com/mcp"},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["name"] == "alpha"
    assert body["data"]["transport_type"] == "sse"
    assert body["data"]["id"]


async def test_create_service_duplicate_name_returns_409(web_authed_client: TestClient) -> None:
    """Two creates with the same (tenant, name) → second returns 409.

    Mirrors Go's ErrConflict (code 1005); the Python wire code is
    ``mcp_service.duplicate_name`` so the UI can render a targeted
    message.
    """
    payload = {"name": "alpha", "transport_type": "sse", "url": "https://example.com/mcp"}
    first = web_authed_client.post("/mcp-services", json=payload)
    assert first.status_code == 201

    second = web_authed_client.post("/mcp-services", json=payload)
    assert second.status_code == 409
    body = second.json()
    assert body["success"] is False
    assert body["error"]["code"] == "mcp_service.duplicate_name"


async def test_create_service_rejects_blank_name(web_authed_client: TestClient) -> None:
    resp = web_authed_client.post("/mcp-services", json={"name": "  ", "transport_type": "sse"})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "mcp_service.name_required"


async def test_create_service_rejects_stdio(web_authed_client: TestClient) -> None:
    resp = web_authed_client.post(
        "/mcp-services", json={"name": "alpha", "transport_type": "stdio"}
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "mcp_service.stdio_disabled"


async def test_create_service_never_echoes_secrets(web_authed_client: TestClient) -> None:
    """``api_key`` / ``token`` are never echoed on the wire.

    Mirrors Go's ``dto.MCPAuthConfigResponse`` — the response DTO has no
    APIKey/Token fields; presence is signalled via the ``credentials``
    metadata map.
    """
    resp = web_authed_client.post(
        "/mcp-services",
        json={
            "name": "alpha",
            "transport_type": "sse",
            "url": "https://example.com/mcp",
            "auth_config": {
                "auth_type": "api_key",
                "api_key": "should-be-preserved",
                "token": "should-be-preserved",
                "scopes": ["read"],
            },
        },
    )
    assert resp.status_code == 201
    data = resp.json()["data"]
    auth = data["auth_config"]
    assert auth["auth_type"] == "api_key"
    assert "api_key" not in auth
    assert "token" not in auth
    assert auth["scopes"] == ["read"]
    assert data["credentials"]["api_key"]["configured"] is True
    assert data["credentials"]["token"]["configured"] is True


# ── GET /mcp-services ───────────────────────────────────────────────


async def test_list_services_returns_envelope(
    web_authed_client: TestClient, mcp_repo: AsyncMock
) -> None:
    await _seed(mcp_repo, name="alpha")
    resp = web_authed_client.get("/mcp-services")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert len(data) == 1
    assert data[0]["name"] == "alpha"


# ── GET /mcp-services/{id} ───────────────────────────────────────────


async def test_get_service_returns_envelope(
    web_authed_client: TestClient, mcp_repo: AsyncMock
) -> None:
    await _seed(mcp_repo, name="alpha")
    resp = web_authed_client.get("/mcp-services/seed-alpha")
    assert resp.status_code == 200
    assert resp.json()["data"]["id"] == "seed-alpha"


async def test_get_service_missing_returns_404(web_authed_client: TestClient) -> None:
    resp = web_authed_client.get("/mcp-services/does-not-exist")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "mcp_service.not_found"


# ── PUT /mcp-services/{id} ──────────────────────────────────────────


async def test_update_service_patches_columns(
    web_authed_client: TestClient, mcp_repo: AsyncMock
) -> None:
    await _seed(mcp_repo, name="alpha")
    resp = web_authed_client.put("/mcp-services/seed-alpha", json={"description": "new"})
    assert resp.status_code == 200
    assert resp.json()["data"]["description"] == "new"


async def test_update_service_preserves_secret_auth(
    web_authed_client: TestClient, mcp_repo: AsyncMock
) -> None:
    await _seed(mcp_repo, name="alpha")
    resp = web_authed_client.put(
        "/mcp-services/seed-alpha",
        json={
            "auth_config": {
                "auth_type": "oauth",
                "api_key": "should-be-preserved",
            },
        },
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    auth = data["auth_config"]
    assert auth.get("auth_type") == "oauth"
    assert "api_key" not in auth
    assert data["credentials"]["api_key"]["configured"] is True


# ── DELETE /mcp-services/{id} ────────────────────────────────────────


async def test_delete_service_returns_ack(
    web_authed_client: TestClient, mcp_repo: AsyncMock
) -> None:
    await _seed(mcp_repo, name="alpha")
    resp = web_authed_client.delete("/mcp-services/seed-alpha")
    assert resp.status_code == 200
    assert resp.json() == {
        "success": True,
        "message": "MCP service deleted successfully",
    }


async def test_delete_missing_service_returns_404(web_authed_client: TestClient) -> None:
    resp = web_authed_client.delete("/mcp-services/missing")
    assert resp.status_code == 404


async def test_delete_rejects_builtin(web_authed_client: TestClient, mcp_repo: AsyncMock) -> None:
    from src.db.models.infra.mcp_services import MCPService

    await mcp_repo.insert(
        MCPService(
            id="seed-built",
            tenant_id=_MCP_SEED_TENANT_ID,
            name="built",
            transport_type="sse",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            is_builtin=True,
        ),
    )
    resp = web_authed_client.delete("/mcp-services/seed-built")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "mcp_service.builtin_immutable"


# ── POST /mcp-services/{id}/test ────────────────────────────────────


async def test_test_runs_connectivity_probe(
    web_authed_client: TestClient, mcp_repo: AsyncMock
) -> None:
    await _seed(mcp_repo, name="alpha")
    resp = web_authed_client.post("/mcp-services/seed-alpha/test")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["success"] is True
    assert body["data"]["message"] == "connected"


# ── GET /mcp-services/{id}/tools ────────────────────────────────────


async def test_list_tools_returns_baked_results(
    web_authed_client: TestClient, mcp_repo: AsyncMock
) -> None:
    await _seed(mcp_repo, name="alpha")
    resp = web_authed_client.get("/mcp-services/seed-alpha/tools")
    assert resp.status_code == 200
    tools = resp.json()["data"]
    assert len(tools) == 1
    assert tools[0]["name"] == "search"


# ── GET /mcp-services/{id}/resources ────────────────────────────────


async def test_list_resources_returns_baked_results(
    web_authed_client: TestClient, mcp_repo: AsyncMock
) -> None:
    await _seed(mcp_repo, name="alpha")
    resp = web_authed_client.get("/mcp-services/seed-alpha/resources")
    assert resp.status_code == 200
    resources = resp.json()["data"]
    assert resources[0]["uri"] == "file://docs"


# ── GET / PUT tool-approvals ────────────────────────────────────────


async def test_tool_approval_round_trip(web_authed_client: TestClient, mcp_repo: AsyncMock) -> None:
    await _seed(mcp_repo, name="alpha")
    put = web_authed_client.put(
        "/mcp-services/seed-alpha/tool-approvals/search",
        json={"require_approval": True},
    )
    assert put.status_code == 200
    assert put.json() == {"success": True}

    listed = web_authed_client.get("/mcp-services/seed-alpha/tool-approvals")
    assert listed.status_code == 200
    rows = listed.json()["data"]
    assert len(rows) == 1
    assert rows[0]["tool_name"] == "search"
    assert rows[0]["require_approval"] is True


# ── OAuth endpoints ─────────────────────────────────────────────────


async def test_oauth_authorize_url(web_authed_client: TestClient, mcp_repo: AsyncMock) -> None:
    await _seed(mcp_repo, name="alpha")
    resp = web_authed_client.post(
        "/mcp-services/seed-alpha/oauth/authorize-url",
        json={
            "redirect_uri": "https://example.com/oauth/callback",
            "frontend_redirect": "/",
        },
    )
    # Auth bypass feeds a non-OAuth-configured service; we still get
    # 422 ``mcp_service.oauth_not_configured`` from the manager.
    assert resp.status_code == 422


async def test_oauth_authorize_url_requires_redirect(
    web_authed_client: TestClient, mcp_repo: AsyncMock
) -> None:
    await _seed(mcp_repo, name="alpha")
    resp = web_authed_client.post("/mcp-services/seed-alpha/oauth/authorize-url", json={})
    assert resp.status_code == 422


async def test_oauth_status_for_seeded_service(
    web_authed_client: TestClient, mcp_repo: AsyncMock
) -> None:
    await _seed(mcp_repo, name="alpha")
    resp = web_authed_client.get("/mcp-services/seed-alpha/oauth/status")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["authorized"] is False
    assert data["state"] == "pending"


async def test_oauth_revoke(web_authed_client: TestClient, mcp_repo: AsyncMock) -> None:
    await _seed(mcp_repo, name="alpha")
    resp = web_authed_client.delete("/mcp-services/seed-alpha/oauth/token")
    assert resp.status_code == 204


__all__ = []
