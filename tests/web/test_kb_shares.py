"""HTTP tests for knowledge-base share routes."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from src.app_context import request_context
from src.common.exception import NotFoundError
from src.core.knowledge.knowledge_bases.types import KnowledgeBaseInfo
from src.core.organizations.service.organization_service import OrganizationService
from src.core.organizations.service.shared_resource_service import SharedResourceService
from src.core.organizations.types import OrganizationInfo, SharedKnowledgeBaseInfo
from src.core.sharing.kb_share_service import KBShareServiceImpl
from src.core.sharing.types import KnowledgeBaseShareInfo
from src.web.api.knowledge_bases.shares_router import router as kb_shares_router
from src.web.api.organizations.shared_router import router as shared_router
from src.web.deps.organizations import get_organization_service, get_shared_resource_service
from src.web.deps.sharing import get_kb_share_service
from src.web.exception_handler import register_exception_handlers
from src.web.middleware.auth import require_auth

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_TENANT = 42
_USER = "user-1"
_holder: dict[str, object] = {}


def _share_info(*, permission: str = "editor") -> KnowledgeBaseShareInfo:
    return KnowledgeBaseShareInfo(
        id="share-1",
        knowledge_base_id="kb-1",
        organization_id="org-1",
        shared_by_user_id=_USER,
        source_tenant_id=_TENANT,
        permission=permission,
        created_at=_NOW,
        updated_at=_NOW,
        my_role_in_org="editor",
        my_permission="editor",
    )


def _org_info() -> OrganizationInfo:
    return OrganizationInfo(
        id="org-1",
        name="Acme Org",
        owner_id=_USER,
        owner_tenant_id=_TENANT,
        created_at=_NOW,
        updated_at=_NOW,
    )


async def _fake_auth(request: Request) -> None:
    role = str(_holder.get("role", "admin"))
    tenant_id = str(_holder.get("tenant_id", _TENANT))
    user_id = str(_holder.get("user_id", _USER))
    request.state.tenant_id = tenant_id
    request.state.tenant_role = role
    request.state.user_info = {
        "id": user_id,
        "username": "alice",
        "email": "alice@example.com",
        "is_active": "1",
        "can_access_all_tenants": "0",
        "is_system_admin": "0",
    }
    request.state.is_system_admin = False
    request.state.api_key_scope = None
    request_context.set_tenant_id(tenant_id)
    request_context.set_user_id(user_id)


def _build_app() -> FastAPI:
    application = FastAPI()
    register_exception_handlers(application)
    application.include_router(kb_shares_router, prefix="/api/v1")
    application.include_router(shared_router, prefix="/api/v1")
    application.dependency_overrides[require_auth] = _fake_auth
    application.dependency_overrides[get_kb_share_service] = lambda: _holder["share"]
    application.dependency_overrides[get_organization_service] = lambda: _holder["org"]
    application.dependency_overrides[get_shared_resource_service] = lambda: _holder["shared"]
    return application


@pytest.fixture
def client() -> TestClient:
    _holder["role"] = "admin"
    _holder["tenant_id"] = _TENANT
    _holder["user_id"] = _USER
    share = AsyncMock(spec=KBShareServiceImpl)
    share.share_knowledge_base.return_value = _share_info()
    share.list_shares_by_knowledge_base.return_value = [_share_info()]
    share.list_shares_by_organization.return_value = [_share_info()]
    share.get_share.return_value = _share_info()
    share.update_share_permission.return_value = None
    share.remove_share.return_value = None
    org = AsyncMock(spec=OrganizationService)
    org.get_organization.return_value = _org_info()
    shared = AsyncMock(spec=SharedResourceService)
    shared.list_organization_shared_knowledge_bases.return_value = []
    _holder["share"] = share
    _holder["org"] = org
    _holder["shared"] = shared
    return TestClient(_build_app())


def _share_svc() -> AsyncMock:
    return _holder["share"]  # type: ignore[return-value]


def test_create_share(client: TestClient) -> None:
    response = client.post(
        "/api/v1/knowledge-bases/kb-1/shares",
        json={"organization_id": "org-1", "permission": "editor"},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["success"] is True
    assert body["data"]["permission"] == "editor"
    assert body["data"]["organization_name"] == "Acme Org"
    _share_svc().share_knowledge_base.assert_awaited_once()


def test_list_shares(client: TestClient) -> None:
    response = client.get("/api/v1/knowledge-bases/kb-1/shares")
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["total"] == 1
    assert data["shares"][0]["id"] == "share-1"


def test_update_share(client: TestClient) -> None:
    response = client.put(
        "/api/v1/knowledge-bases/kb-1/shares/share-1",
        json={"permission": "viewer"},
    )
    assert response.status_code == 200
    _share_svc().update_share_permission.assert_awaited_once()


def test_delete_share(client: TestClient) -> None:
    response = client.delete("/api/v1/knowledge-bases/kb-1/shares/share-1")
    assert response.status_code == 200
    _share_svc().remove_share.assert_awaited_once()


def test_other_tenant_is_not_found(client: TestClient) -> None:
    _share_svc().share_knowledge_base.side_effect = NotFoundError(
        code="knowledge_base.not_found",
        message="knowledge base kb-1 not found",
    )
    response = client.post(
        "/api/v1/knowledge-bases/kb-1/shares",
        json={"organization_id": "org-1", "permission": "viewer"},
    )
    assert response.status_code == 404


def test_viewer_cannot_create_share(client: TestClient) -> None:
    _holder["role"] = "viewer"
    response = client.post(
        "/api/v1/knowledge-bases/kb-1/shares",
        json={"organization_id": "org-1", "permission": "viewer"},
    )
    assert response.status_code == 403
    _share_svc().share_knowledge_base.assert_not_awaited()


def test_dest_org_admin_cannot_mutate(client: TestClient) -> None:
    _holder["tenant_id"] = 99
    _holder["user_id"] = "usr-dest-admin"
    _share_svc().remove_share.side_effect = NotFoundError(
        code="knowledge_base.not_found",
        message="knowledge base kb-1 not found",
    )
    response = client.delete("/api/v1/knowledge-bases/kb-1/shares/share-1")
    assert response.status_code == 404


def test_duplicate_upgrades_permission(client: TestClient) -> None:
    _share_svc().share_knowledge_base.return_value = _share_info(permission="editor")
    response = client.post(
        "/api/v1/knowledge-bases/kb-1/shares",
        json={"organization_id": "org-1", "permission": "editor"},
    )
    assert response.status_code == 201
    assert response.json()["data"]["permission"] == "editor"


def test_list_organization_shares(client: TestClient) -> None:
    response = client.get("/api/v1/organizations/org-1/shares")
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["total"] == 1
    assert data["shares"][0]["organization_id"] == "org-1"


def test_list_organization_shared_knowledge_bases(client: TestClient) -> None:
    _holder["shared"].list_organization_shared_knowledge_bases.return_value = [  # type: ignore[union-attr]
        SharedKnowledgeBaseInfo(
            knowledge_base=KnowledgeBaseInfo(
                id="kb-1",
                name="KB",
                tenant_id=_TENANT,
                created_at=_NOW,
                updated_at=_NOW,
            ),
            share_id="share-1",
            organization_id="org-1",
            org_name="Acme Org",
            permission="editor",
            source_tenant_id=_TENANT,
            shared_at=_NOW,
            is_mine=True,
        )
    ]
    response = client.get("/api/v1/organizations/org-1/shared-knowledge-bases")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["data"][0]["is_mine"] is True
    assert "source_from_agent" not in body["data"][0]
