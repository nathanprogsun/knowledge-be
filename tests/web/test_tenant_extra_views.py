"""Web-layer tests for the added tenant endpoints (api-keys, KV, principal).

Uses in-memory fakes for the api-key repository; KV and principal-config
endpoints are exercised against fakes of the underlying tenant repos.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.app_context import request_context
from src.app_context.lifespan import create_app
from src.core.tenants.api_key_service import TenantAPIKeyService
from src.core.tenants.kv_service import TenantKVService
from src.db.models.tenants.tenant_api_keys import TenantAPIKey
from src.db.models.tenants.tenant_kv import TenantKV
from src.web.deps import (
    get_current_user_context,
    get_tenant_api_key_service,
    get_tenant_kv_service,
)
from tests.fakes.tenant_api_keys import FakeTenantAPIKeyRepository

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


class _FakeKVRepo:
    """Minimal in-memory tenant_kv repository (upsert + find + delete)."""

    def __init__(self) -> None:
        self._store: dict[tuple[int, str], TenantKV] = {}
        self._next_id = 1

    async def find_value(self, *, tenant_id: int, key: str) -> TenantKV | None:
        return self._store.get((tenant_id, key))

    async def upsert(self, *, tenant_id: int, key: str, value: object) -> TenantKV:
        row = TenantKV(
            id=self._next_id,
            tenant_id=tenant_id,
            key=key,
            value=value,  # type: ignore[arg-type]
            created_at=_NOW,
            updated_at=_NOW,
            deleted_at=None,
        )
        self._next_id += 1
        self._store[(tenant_id, key)] = row
        return row

    async def delete(self, *, tenant_id: int, key: str) -> bool:
        return self._store.pop((tenant_id, key), None) is not None


@pytest.fixture
def api_key_repo() -> FakeTenantAPIKeyRepository:
    return FakeTenantAPIKeyRepository()


@pytest.fixture
def kv_repo() -> _FakeKVRepo:
    return _FakeKVRepo()


@pytest.fixture
def app(
    api_key_repo: FakeTenantAPIKeyRepository,
    kv_repo: _FakeKVRepo,
) -> FastAPI:
    application = create_app()
    application.dependency_overrides[get_current_user_context] = lambda: None
    application.dependency_overrides[get_tenant_api_key_service] = lambda: TenantAPIKeyService(
        api_keys_repo=api_key_repo,  # type: ignore[arg-type]
    )
    application.dependency_overrides[get_tenant_kv_service] = lambda: TenantKVService(
        kv_repo=kv_repo,  # type: ignore[arg-type]
    )
    return application


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=app)
    token_tenant = request_context.set_tenant_id("7")
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    request_context._tenant_id.reset(token_tenant)


async def _seed_key(
    repo: FakeTenantAPIKeyRepository,
    *,
    tenant_id: int = 7,
    name: str = "deploy",
) -> TenantAPIKey:
    return await repo.insert(
        TenantAPIKey(
            id=0,
            tenant_id=tenant_id,
            scope_type="tenant",
            name=name,
            key_hash="abc123",
            full_access=True,
            created_at=_NOW,
            updated_at=_NOW,
        )
    )


# ── GET /tenants/{id}/api-keys ────────────────────────────────────────


async def test_list_api_keys(client: AsyncClient, api_key_repo: FakeTenantAPIKeyRepository) -> None:
    await _seed_key(api_key_repo)
    resp = await client.get("/tenants/7/api-keys")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    assert len(body) == 1
    assert body[0]["name"] == "deploy"
    assert "api_key" not in body[0]
    assert "key_hash" not in body[0]


# ── POST /tenants/{id}/api-keys ───────────────────────────────────────


async def test_create_api_key(client: AsyncClient) -> None:
    resp = await client.post(
        "/tenants/7/api-keys",
        json={"name": "ci", "full_access": True},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["name"] == "ci"
    assert body["scope_type"] == "tenant"


# ── DELETE /tenants/{id}/api-keys/{key_id} ───────────────────────────


async def test_revoke_api_key(
    client: AsyncClient, api_key_repo: FakeTenantAPIKeyRepository
) -> None:
    key = await _seed_key(api_key_repo)
    resp = await client.delete(f"/tenants/7/api-keys/{key.id}")
    assert resp.status_code == 200
    assert resp.json()["success"] is True
    assert await api_key_repo.list_for_tenant(7) == []


# ── GET/PUT /tenants/kv/{key} ─────────────────────────────────────────


async def test_put_and_get_kv(client: AsyncClient, kv_repo: _FakeKVRepo) -> None:
    put = await client.put("/tenants/kv/web-search-config", json={"max_results": 20})
    assert put.status_code == 200
    assert put.json() == {"max_results": 20}

    get = await client.get("/tenants/kv/web-search-config")
    assert get.status_code == 200
    assert get.json() == {"max_results": 20}


async def test_get_kv_missing(client: AsyncClient) -> None:
    resp = await client.get("/tenants/kv/nonexistent")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "tenant_kv.not_found"


__all__ = []
