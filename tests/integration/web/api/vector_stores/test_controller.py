"""Web-layer tests for the vector-store router (CRUD + types + test).

Exercises the router over HTTP via ``TestClient`` against the
app. The service dependency is overridden with an
``AsyncMock(spec=VectorStoreService)`` configured with stateful
closures, so the tests exercise the full HTTP path (routing,
serialization, exception handling) without touching a real database.

The env-store synthesis lives inside the router; the tests assert
on its presence by setting ``RETRIEVE_DRIVER`` in the env map.

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

from src.common.exception import ValidationError
from src.core.contracts.infra import (
    CreateVectorStoreRequest,
    UpdateVectorStoreRequest,
)
from src.core.contracts.infra import (
    TestVectorStoreResponse as _TestResponse,
)
from src.core.infra.vector_stores.service.vector_store_service import VectorStoreService
from src.core.infra.vector_stores.types import VectorStoreInfo
from src.db.models.infra.vector_store import VectorStore
from src.web.deps.infra_vector_stores import get_vector_store_service


@pytest.fixture
def fake_service() -> AsyncMock:
    """``AsyncMock(spec=VectorStoreService)`` with stateful closures."""
    repo = AsyncMock(spec=VectorStoreService)
    rows: dict[str, VectorStore] = {}
    counter = [0]

    async def _list_stores(tenant_id: int) -> list[VectorStoreInfo]:
        out = [r for r in rows.values() if r.tenant_id == tenant_id and r.deleted_at is None]
        out.sort(key=lambda r: r.created_at, reverse=True)
        return [VectorStoreInfo.map_from_db(r) for r in out]

    async def _get_store(tenant_id: int, store_id: str) -> VectorStoreInfo:
        for r in rows.values():
            if r.id == store_id and r.tenant_id == tenant_id and r.deleted_at is None:
                return VectorStoreInfo.map_from_db(r)
        raise ValidationError(
            code="vector_store.not_found",
            message=f"vector store {store_id} not found",
        )

    async def _create_store(*, tenant_id: int, body: CreateVectorStoreRequest) -> VectorStoreInfo:
        now = datetime.now(UTC)
        counter[0] += 1
        row = VectorStore(
            id=f"vs-{counter[0]}",
            tenant_id=tenant_id,
            name=body.name,
            engine_type=body.engine_type,
            connection_config=dict(body.connection_config),
            index_config=dict(body.index_config) if body.index_config else None,
            source="user",
            readonly=False,
            created_at=now,
            updated_at=now,
        )
        rows[row.id] = row
        return VectorStoreInfo.map_from_db(row)

    async def _update_store(
        *, tenant_id: int, store_id: str, body: UpdateVectorStoreRequest
    ) -> VectorStoreInfo:
        for r in rows.values():
            if r.id == store_id and r.tenant_id == tenant_id:
                updated = r.model_copy(update={"name": body.name, "updated_at": datetime.now(UTC)})
                rows[store_id] = updated
                return VectorStoreInfo.map_from_db(updated)
        raise ValidationError(
            code="vector_store.not_found",
            message=f"vector store {store_id} not found",
        )

    async def _delete_store(tenant_id: int, store_id: str) -> bool:
        for r in rows.values():
            if r.id == store_id and r.tenant_id == tenant_id:
                updated = r.model_copy(update={"deleted_at": datetime.now(UTC)})
                rows[store_id] = updated
                return True
        return False

    async def _test_by_id(tenant_id: int, store_id: str) -> _TestResponse:
        for r in rows.values():
            if r.id == store_id and r.tenant_id == tenant_id:
                return _TestResponse(success=True, version="", error=None)
        return _TestResponse(
            success=False,
            version=None,
            error="not found",
        )

    async def _test_raw(engine_type: str, connection_config: dict[str, object]) -> _TestResponse:
        return _TestResponse(success=True, version="", error=None)

    repo.list_stores.side_effect = _list_stores
    repo.get_store.side_effect = _get_store
    repo.create_store.side_effect = _create_store
    repo.update_store.side_effect = _update_store
    repo.delete_store.side_effect = _delete_store
    repo.test_by_id.side_effect = _test_by_id
    repo.test_raw.side_effect = _test_raw
    repo._rows = rows  # type: ignore[attr-defined]
    return repo


@pytest.fixture
def app(
    request: pytest.FixtureRequest,
    web_app: FastAPI,
    fake_service: AsyncMock,
) -> FastAPI:
    """Override ``get_vector_store_service`` on the shared web app."""
    web_app.dependency_overrides[get_vector_store_service] = lambda: fake_service
    return web_app


@pytest.fixture
def client(app: FastAPI, web_authed_client: TestClient) -> TestClient:
    """Alias ``web_authed_client``; depending on ``app`` forces the
    dep-override fixture to run before the test executes."""
    return web_authed_client


# ── GET /vector-stores/types ─────────────────────────────────────────


async def test_list_types_returns_seven_engines(client: TestClient) -> None:
    """The types endpoint returns the seven supported engine types."""
    resp = client.get("/api/v1/vector-stores/types")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    types = {entry["type"] for entry in body["data"]}
    assert types == {
        "elasticsearch",
        "qdrant",
        "milvus",
        "tencent_vectordb",
        "weaviate",
        "doris",
        "opensearch",
    }


# ── POST /vector-stores/test (raw) ──────────────────────────────────


async def test_test_raw_returns_success(client: TestClient) -> None:
    """A valid raw config yields a success response with empty version."""
    resp = client.post(
        "/api/v1/vector-stores/test",
        json={"engine_type": "elasticsearch", "connection_config": {"addr": "http://es:9200"}},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["version"] == ""


# ── POST /vector-stores ──────────────────────────────────────────────


async def test_create_vector_store_returns_envelope(
    client: TestClient,
    fake_service: AsyncMock,
) -> None:
    """A create call returns the wrapped envelope with masked credentials."""
    resp = client.post(
        "/api/v1/vector-stores",
        json={
            "name": "es-hot",
            "engine_type": "elasticsearch",
            "connection_config": {"addr": "http://es:9200", "password": "secret"},
        },
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["name"] == "es-hot"
    assert body["data"]["source"] == "user"
    assert body["data"]["readonly"] is False
    # Sensitive fields are masked.
    assert body["data"]["connection_config"]["password"] == "***"
    rows = fake_service._rows  # type: ignore[attr-defined]
    assert len(rows) == 1


# ── GET /vector-stores ──────────────────────────────────────────────


async def test_list_stores_returns_db_rows(
    client: TestClient,
    fake_service: AsyncMock,
) -> None:
    """The list endpoint returns the DB-managed rows."""
    client.post(
        "/api/v1/vector-stores",
        json={
            "name": "es-a",
            "engine_type": "elasticsearch",
            "connection_config": {"addr": "http://es-a:9200"},
        },
    )
    resp = client.get("/api/v1/vector-stores")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    # Env stores are absent in this test (no RETRIEVE_DRIVER set).
    data = body["data"]
    assert len(data) == 1
    assert data[0]["source"] == "user"
    assert data[0]["readonly"] is False


async def test_list_stores_synthesises_env_entries(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Setting ``RETRIEVE_DRIVER`` surfaces an env-store virtual entry."""
    monkeypatch.setenv("RETRIEVE_DRIVER", "postgres")
    resp = client.get("/api/v1/vector-stores")
    assert resp.status_code == 200
    body = resp.json()
    data = body["data"]
    # The env entry comes first; check its identifying marks.
    assert data[0]["id"] == "__env_postgres__"
    assert data[0]["source"] == "env"
    assert data[0]["readonly"] is True


# ── GET /vector-stores/{id} ─────────────────────────────────────────


async def test_get_store_returns_envelope(client: TestClient) -> None:
    """The get endpoint returns the wrapped store after a create."""
    create_resp = client.post(
        "/api/v1/vector-stores",
        json={
            "name": "es-get",
            "engine_type": "elasticsearch",
            "connection_config": {"addr": "http://es:9200"},
        },
    )
    store_id = create_resp.json()["data"]["id"]
    resp = client.get(f"/api/v1/vector-stores/{store_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["data"]["id"] == store_id


async def test_get_unknown_store_returns_404(
    client: TestClient,
) -> None:
    """An unknown id is rejected with a 404-style status code."""
    resp = client.get("/api/v1/vector-stores/missing")
    # ValidationError is mapped to 422 by the exception handler.
    assert resp.status_code in (404, 422)


async def test_get_env_store_returns_envelope(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An env-store id resolves without touching the service."""
    monkeypatch.setenv("RETRIEVE_DRIVER", "qdrant")
    resp = client.get("/api/v1/vector-stores/__env_qdrant__")
    assert resp.status_code == 200
    body = resp.json()
    assert body["data"]["id"] == "__env_qdrant__"
    assert body["data"]["source"] == "env"
    assert body["data"]["readonly"] is True


# ── PUT /vector-stores/{id} ─────────────────────────────────────────


async def test_update_store_renames(
    client: TestClient,
) -> None:
    """The put endpoint only mutates the ``name`` field."""
    create_resp = client.post(
        "/api/v1/vector-stores",
        json={
            "name": "es-old",
            "engine_type": "elasticsearch",
            "connection_config": {"addr": "http://es:9200"},
        },
    )
    store_id = create_resp.json()["data"]["id"]
    resp = client.put(
        f"/api/v1/vector-stores/{store_id}",
        json={"name": "es-new"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["data"]["name"] == "es-new"


async def test_update_env_store_rejected(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Env-store ids cannot be updated."""
    monkeypatch.setenv("RETRIEVE_DRIVER", "qdrant")
    resp = client.put(
        "/api/v1/vector-stores/__env_qdrant__",
        json={"name": "renamed"},
    )
    # ValidationError is mapped to 422.
    assert resp.status_code in (400, 422)


# ── DELETE /vector-stores/{id} ──────────────────────────────────────


async def test_delete_store_soft_deletes(
    client: TestClient,
) -> None:
    """A successful delete returns the success envelope."""
    create_resp = client.post(
        "/api/v1/vector-stores",
        json={
            "name": "es-del",
            "engine_type": "elasticsearch",
            "connection_config": {"addr": "http://es:9200"},
        },
    )
    store_id = create_resp.json()["data"]["id"]
    resp = client.delete(f"/api/v1/vector-stores/{store_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    # The follow-up get returns a 422 / 404 — the row is invisible to reads.
    follow = client.get(f"/api/v1/vector-stores/{store_id}")
    assert follow.status_code in (404, 422)


async def test_delete_env_store_rejected(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Env-store ids cannot be deleted."""
    monkeypatch.setenv("RETRIEVE_DRIVER", "qdrant")
    resp = client.delete("/api/v1/vector-stores/__env_qdrant__")
    assert resp.status_code in (400, 422)


# ── POST /vector-stores/{id}/test ──────────────────────────────────


async def test_test_by_id_runs_probe(
    client: TestClient,
) -> None:
    """The by-id test endpoint surfaces the probe's success response."""
    create_resp = client.post(
        "/api/v1/vector-stores",
        json={
            "name": "es-probe",
            "engine_type": "elasticsearch",
            "connection_config": {"addr": "http://es:9200"},
        },
    )
    store_id = create_resp.json()["data"]["id"]
    resp = client.post(f"/api/v1/vector-stores/{store_id}/test")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True


async def test_test_env_store_runs_probe(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The by-id test endpoint probes env-store entries directly."""
    monkeypatch.setenv("RETRIEVE_DRIVER", "qdrant")
    resp = client.post("/api/v1/vector-stores/__env_qdrant__/test")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True


__all__ = []
