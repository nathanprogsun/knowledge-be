"""Web-layer tests for the MCP service router.

Exercises the 13 endpoints over HTTP via ``httpx.AsyncClient`` with
``get_mcp_service`` overridden to a real ``MCPServiceService`` backed
by the shared in-memory fake repository. The discovery + connectivity
fakes are also injected so the routes reach a known state.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.app_context.lifespan import create_app
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
from src.web.api.infra.mcp_services.router import router as mcp_router
from src.web.deps.infra_mcp import get_mcp_service
from tests.fakes.auth_gates import override_auth_gates
from tests.fakes.mcp_services import (
    FakeMCPServiceRepository,
    FakeMCPToolApprovalRepository,
)


@pytest.fixture
def mcp_repo() -> FakeMCPServiceRepository:
    return FakeMCPServiceRepository()


@pytest.fixture
def approvals_repo() -> FakeMCPToolApprovalRepository:
    return FakeMCPToolApprovalRepository()


@pytest.fixture
def app(
    mcp_repo: FakeMCPServiceRepository,
    approvals_repo: FakeMCPToolApprovalRepository,
) -> FastAPI:
    application = create_app()
    application.include_router(mcp_router)

    def _override_service() -> MCPServiceService:
        return MCPServiceService(
            mcp_repo=mcp_repo,  # type: ignore[arg-type]
            tool_approvals_repo=approvals_repo,  # type: ignore[arg-type]
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

    application.dependency_overrides[get_mcp_service] = _override_service
    override_auth_gates(application)
    return application


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _seed(mcp_repo: FakeMCPServiceRepository, *, name: str = "alpha") -> str:
    from datetime import UTC, datetime

    from src.db.models.infra.mcp_services import MCPService

    now = datetime.now(UTC)
    row = MCPService(
        id="seed-" + name,
        tenant_id=1,
        name=name,
        transport_type="sse",
        url="https://example.com/mcp",
        created_at=now,
        updated_at=now,
    )
    await mcp_repo.insert(row)
    return row.id


# ── POST /mcp-services ──────────────────────────────────────────────


async def test_create_service_returns_201_envelope(client: AsyncClient) -> None:
    resp = await client.post(
        "/mcp-services",
        json={"name": "alpha", "transport_type": "sse", "url": "https://example.com/mcp"},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["name"] == "alpha"
    assert body["data"]["transport_type"] == "sse"
    assert body["data"]["id"]


async def test_create_service_duplicate_name_returns_409(client: AsyncClient) -> None:
    """Two creates with the same (tenant, name) → second returns 409.

    Mirrors Go's ErrConflict (code 1005); the Python wire code is
    ``mcp_service.duplicate_name`` so the UI can render a targeted
    message.
    """
    payload = {"name": "alpha", "transport_type": "sse", "url": "https://example.com/mcp"}
    first = await client.post("/mcp-services", json=payload)
    assert first.status_code == 201

    second = await client.post("/mcp-services", json=payload)
    assert second.status_code == 409
    body = second.json()
    assert body["success"] is False
    assert body["error"]["code"] == "mcp_service.duplicate_name"


async def test_create_service_rejects_blank_name(client: AsyncClient) -> None:
    resp = await client.post("/mcp-services", json={"name": "  ", "transport_type": "sse"})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "mcp_service.name_required"


async def test_create_service_rejects_stdio(client: AsyncClient) -> None:
    resp = await client.post("/mcp-services", json={"name": "alpha", "transport_type": "stdio"})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "mcp_service.stdio_disabled"


async def test_create_service_preserves_secrets(client: AsyncClient) -> None:
    """``api_key`` / ``token`` are kept verbatim on user services.

    Mirrors Go's ``dto.NewMCPServiceResponse`` — credentials are not
    stripped on user-created rows; only built-in services (not exposed
    via the public API) are masked.
    """
    resp = await client.post(
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
    auth = resp.json()["data"]["auth_config"]
    assert auth["auth_type"] == "api_key"
    assert auth["api_key"] == "should-be-preserved"
    assert auth["token"] == "should-be-preserved"
    assert auth["scopes"] == ["read"]


# ── GET /mcp-services ───────────────────────────────────────────────


async def test_list_services_returns_envelope(
    client: AsyncClient, mcp_repo: FakeMCPServiceRepository
) -> None:
    await _seed(mcp_repo, name="alpha")
    resp = await client.get("/mcp-services")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert len(data) == 1
    assert data[0]["name"] == "alpha"


# ── GET /mcp-services/{id} ───────────────────────────────────────────


async def test_get_service_returns_envelope(
    client: AsyncClient, mcp_repo: FakeMCPServiceRepository
) -> None:
    await _seed(mcp_repo, name="alpha")
    resp = await client.get("/mcp-services/seed-alpha")
    assert resp.status_code == 200
    assert resp.json()["data"]["id"] == "seed-alpha"


async def test_get_service_missing_returns_404(client: AsyncClient) -> None:
    resp = await client.get("/mcp-services/does-not-exist")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "mcp_service.not_found"


# ── PUT /mcp-services/{id} ──────────────────────────────────────────


async def test_update_service_patches_columns(
    client: AsyncClient, mcp_repo: FakeMCPServiceRepository
) -> None:
    await _seed(mcp_repo, name="alpha")
    resp = await client.put("/mcp-services/seed-alpha", json={"description": "new"})
    assert resp.status_code == 200
    assert resp.json()["data"]["description"] == "new"


async def test_update_service_preserves_secret_auth(
    client: AsyncClient, mcp_repo: FakeMCPServiceRepository
) -> None:
    await _seed(mcp_repo, name="alpha")
    resp = await client.put(
        "/mcp-services/seed-alpha",
        json={
            "auth_config": {
                "auth_type": "oauth",
                "api_key": "should-be-preserved",
            },
        },
    )
    assert resp.status_code == 200
    auth = resp.json()["data"]["auth_config"]
    assert auth.get("auth_type") == "oauth"
    assert auth.get("api_key") == "should-be-preserved"


# ── DELETE /mcp-services/{id} ────────────────────────────────────────


async def test_delete_service_returns_ack(
    client: AsyncClient, mcp_repo: FakeMCPServiceRepository
) -> None:
    await _seed(mcp_repo, name="alpha")
    resp = await client.delete("/mcp-services/seed-alpha")
    assert resp.status_code == 200
    assert resp.json() == {
        "success": True,
        "message": "MCP service deleted successfully",
    }


async def test_delete_missing_service_returns_404(client: AsyncClient) -> None:
    resp = await client.delete("/mcp-services/missing")
    assert resp.status_code == 404


async def test_delete_rejects_builtin(
    client: AsyncClient, mcp_repo: FakeMCPServiceRepository
) -> None:
    from datetime import UTC, datetime

    from src.db.models.infra.mcp_services import MCPService

    await mcp_repo.insert(
        MCPService(
            id="seed-built",
            tenant_id=1,
            name="built",
            transport_type="sse",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            is_builtin=True,
        ),
    )
    resp = await client.delete("/mcp-services/seed-built")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "mcp_service.builtin_immutable"


# ── POST /mcp-services/{id}/test ────────────────────────────────────


async def test_test_runs_connectivity_probe(
    client: AsyncClient, mcp_repo: FakeMCPServiceRepository
) -> None:
    await _seed(mcp_repo, name="alpha")
    resp = await client.post("/mcp-services/seed-alpha/test")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["success"] is True
    assert body["data"]["message"] == "connected"


# ── GET /mcp-services/{id}/tools ────────────────────────────────────


async def test_list_tools_returns_baked_results(
    client: AsyncClient, mcp_repo: FakeMCPServiceRepository
) -> None:
    await _seed(mcp_repo, name="alpha")
    resp = await client.get("/mcp-services/seed-alpha/tools")
    assert resp.status_code == 200
    tools = resp.json()["data"]
    assert len(tools) == 1
    assert tools[0]["name"] == "search"


# ── GET /mcp-services/{id}/resources ────────────────────────────────


async def test_list_resources_returns_baked_results(
    client: AsyncClient, mcp_repo: FakeMCPServiceRepository
) -> None:
    await _seed(mcp_repo, name="alpha")
    resp = await client.get("/mcp-services/seed-alpha/resources")
    assert resp.status_code == 200
    resources = resp.json()["data"]
    assert resources[0]["uri"] == "file://docs"


# ── GET / PUT tool-approvals ────────────────────────────────────────


async def test_tool_approval_round_trip(
    client: AsyncClient, mcp_repo: FakeMCPServiceRepository
) -> None:
    await _seed(mcp_repo, name="alpha")
    put = await client.put(
        "/mcp-services/seed-alpha/tool-approvals/search",
        json={"require_approval": True},
    )
    assert put.status_code == 200
    assert put.json() == {"success": True}

    listed = await client.get("/mcp-services/seed-alpha/tool-approvals")
    assert listed.status_code == 200
    rows = listed.json()["data"]
    assert len(rows) == 1
    assert rows[0]["tool_name"] == "search"
    assert rows[0]["require_approval"] is True


# ── OAuth endpoints ─────────────────────────────────────────────────


async def test_oauth_authorize_url(client: AsyncClient, mcp_repo: FakeMCPServiceRepository) -> None:
    await _seed(mcp_repo, name="alpha")
    resp = await client.post(
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
    client: AsyncClient, mcp_repo: FakeMCPServiceRepository
) -> None:
    await _seed(mcp_repo, name="alpha")
    resp = await client.post("/mcp-services/seed-alpha/oauth/authorize-url", json={})
    assert resp.status_code == 422


async def test_oauth_status_for_seeded_service(
    client: AsyncClient, mcp_repo: FakeMCPServiceRepository
) -> None:
    await _seed(mcp_repo, name="alpha")
    resp = await client.get("/mcp-services/seed-alpha/oauth/status")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["authorized"] is False
    assert data["state"] == "pending"


async def test_oauth_revoke(client: AsyncClient, mcp_repo: FakeMCPServiceRepository) -> None:
    await _seed(mcp_repo, name="alpha")
    resp = await client.delete("/mcp-services/seed-alpha/oauth/token")
    assert resp.status_code == 204
