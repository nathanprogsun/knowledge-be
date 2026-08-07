"""Web-layer tests for the added tenant endpoints (api-keys, KV, principal).

Uses ``AsyncMock(spec=...)`` repositories configured with stateful
closures for the api-key and KV repos. Principal-config endpoints are
exercised against the same tenant mocks.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.core.tenants.api_key_service import TenantAPIKeyService
from src.core.tenants.kv_service import TenantKVService
from src.db.dao.tenant_api_keys_repository import TenantAPIKeyRepository
from src.db.dao.tenant_kv_repository import TenantKVRepository
from src.db.models.tenants.tenant_api_keys import TenantAPIKey
from src.db.models.tenants.tenant_kv import TenantKV
from src.web.deps import (
    get_tenant_api_key_service,
    get_tenant_kv_service,
)

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture
def api_key_repo() -> AsyncMock:
    repo = AsyncMock(spec=TenantAPIKeyRepository)
    rows: dict[int, TenantAPIKey] = {}
    counter = [0]

    def _live() -> dict[int, TenantAPIKey]:
        return {kid: r for kid, r in rows.items() if r.revoked_at is None}

    async def _insert(row: TenantAPIKey) -> TenantAPIKey:
        counter[0] += 1
        stored = row.model_copy(update={"id": counter[0]})
        rows[stored.id] = stored
        return stored

    async def _list_for_tenant(tenant_id: int) -> list[TenantAPIKey]:
        live = [r for r in _live().values() if r.tenant_id == tenant_id]
        return sorted(live, key=lambda r: r.created_at, reverse=True)

    async def _revoke(key_id: int, *, tenant_id: int, revoked_at: datetime) -> None:
        row = _live().get(key_id)
        if row is None or row.tenant_id != tenant_id:
            from src.common.exception import NotFoundError

            raise NotFoundError(code="tenant_api_key.not_found", message="Tenant API key not found")
        rows[key_id] = row.model_copy(update={"revoked_at": revoked_at})

    repo.insert.side_effect = _insert
    repo.list_for_tenant.side_effect = _list_for_tenant
    repo.revoke.side_effect = _revoke
    repo._rows = rows  # type: ignore[attr-defined]
    return repo


@pytest.fixture
def kv_repo() -> AsyncMock:
    repo = AsyncMock(spec=TenantKVRepository)
    store: dict[tuple[int, str], TenantKV] = {}
    counter = [0]

    async def _find_value(*, tenant_id: int, key: str) -> TenantKV | None:
        return store.get((tenant_id, key))

    async def _upsert(*, tenant_id: int, key: str, value: object) -> TenantKV:
        counter[0] += 1
        row = TenantKV(
            id=counter[0],
            tenant_id=tenant_id,
            key=key,
            value=value,  # type: ignore[arg-type]
            created_at=_NOW,
            updated_at=_NOW,
            deleted_at=None,
        )
        store[(tenant_id, key)] = row
        return row

    async def _delete(*, tenant_id: int, key: str) -> bool:
        return store.pop((tenant_id, key), None) is not None

    repo.find_value.side_effect = _find_value
    repo.upsert.side_effect = _upsert
    repo.delete.side_effect = _delete
    return repo


@pytest.fixture(autouse=True)
def _override_services(
    web_app: FastAPI,
    api_key_repo: AsyncMock,
    kv_repo: AsyncMock,
) -> FastAPI:
    """Override tenant service deps on the shared web app (autouse)."""
    web_app.dependency_overrides[get_tenant_api_key_service] = lambda: TenantAPIKeyService(
        api_keys_repo=api_key_repo,
    )
    web_app.dependency_overrides[get_tenant_kv_service] = lambda: TenantKVService(
        kv_repo=kv_repo,
    )
    return web_app


async def _seed_key(
    api_key_repo: AsyncMock,
    *,
    tenant_id: int = 7,
    name: str = "deploy",
) -> TenantAPIKey:
    return await api_key_repo.insert(
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


async def test_list_api_keys(
    web_authed_client: TestClient,
    api_key_repo: AsyncMock,
) -> None:
    await _seed_key(api_key_repo)
    resp = web_authed_client.get("/tenants/7/api-keys")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    assert len(body) == 1
    assert body[0]["name"] == "deploy"
    assert "api_key" not in body[0]
    assert "key_hash" not in body[0]


# ── POST /tenants/{id}/api-keys ───────────────────────────────────────


async def test_create_api_key(web_authed_client: TestClient) -> None:
    resp = web_authed_client.post(
        "/tenants/7/api-keys",
        json={"name": "ci", "full_access": True},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["name"] == "ci"
    assert body["data"]["scope_type"] == "tenant"
    # The plaintext token is embedded once, on create only.
    assert body["data"]["api_key"]


# ── DELETE /tenants/{id}/api-keys/{key_id} ───────────────────────────


async def test_revoke_api_key(
    web_authed_client: TestClient,
    api_key_repo: AsyncMock,
) -> None:
    key = await _seed_key(api_key_repo)
    resp = web_authed_client.delete(f"/tenants/7/api-keys/{key.id}")
    assert resp.status_code == 200
    assert resp.json()["success"] is True
    assert await api_key_repo.list_for_tenant(7) == []


# ── GET/PUT /tenants/kv/{key} ─────────────────────────────────────────


async def test_put_and_get_kv(
    web_authed_client: TestClient,
    kv_repo: AsyncMock,
) -> None:
    put = web_authed_client.put("/tenants/kv/web-search-config", json={"max_results": 20})
    assert put.status_code == 200
    assert put.json() == {"max_results": 20}

    get = web_authed_client.get("/tenants/kv/web-search-config")
    assert get.status_code == 200
    assert get.json() == {"max_results": 20}


async def test_get_kv_unsupported_key(web_authed_client: TestClient) -> None:
    resp = web_authed_client.get("/tenants/kv/nonexistent")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "tenant_kv.unsupported_key"


__all__ = []
