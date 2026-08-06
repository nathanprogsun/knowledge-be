"""Web-layer tests for the vector-store router (CRUD + types + test).

Per AGENTS.md §9, web routers are tested via ``httpx.AsyncClient``
against the app. The service dependency is overridden with a
service-backed fake so the tests exercise the full HTTP path (routing,
serialization, exception handling) without touching a real database.

The env-store synthesis lives inside the router; the tests assert
on its presence by setting ``RETRIEVE_DRIVER`` in the env map.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.app_context.lifespan import create_app
from src.core.contracts.infra import (
    CreateVectorStoreRequest,
    UpdateVectorStoreRequest,
)
from src.core.contracts.infra import (
    TestVectorStoreResponse as _TestResponse,
)
from src.core.infra.vector_stores.types import VectorStoreInfo
from src.db.models.infra.vector_store import VectorStore
from src.web.deps.infra_vector_stores import get_vector_store_service
from tests.unit.fakes.auth_gates import override_auth_gates

# ── In-memory fake service ──────────────────────────────────────────


class _FakeService:
    """In-memory ``VectorStoreService`` replacement.

    Mirrors the service surface used by the router: ``list_stores``,
    ``get_store``, ``create_store``, ``update_store``, ``delete_store``,
    ``test_by_id``, ``test_raw``. Soft-delete semantics match the real
    repository: a deleted row is invisible to reads.
    """

    def __init__(self) -> None:
        self.rows: dict[str, VectorStore] = {}
        self._next_call: int = 0

    async def list_stores(self, tenant_id: int) -> list[VectorStoreInfo]:
        out = [r for r in self.rows.values() if r.tenant_id == tenant_id and r.deleted_at is None]
        out.sort(key=lambda r: r.created_at, reverse=True)
        return [VectorStoreInfo.map_from_db(r) for r in out]

    async def get_store(self, tenant_id: int, store_id: str) -> VectorStoreInfo:
        for r in self.rows.values():
            if r.id == store_id and r.tenant_id == tenant_id and r.deleted_at is None:
                return VectorStoreInfo.map_from_db(r)
        from src.common.exception import ValidationError

        raise ValidationError(
            code="vector_store.not_found",
            message=f"vector store {store_id} not found",
        )

    async def create_store(
        self,
        *,
        tenant_id: int,
        body: CreateVectorStoreRequest,
    ) -> VectorStoreInfo:
        now = datetime.now(UTC)
        self._next_call += 1
        row = VectorStore(
            id=f"vs-{self._next_call}",
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
        self.rows[row.id] = row
        return VectorStoreInfo.map_from_db(row)

    async def update_store(
        self,
        *,
        tenant_id: int,
        store_id: str,
        body: UpdateVectorStoreRequest,
    ) -> VectorStoreInfo:
        for r in self.rows.values():
            if r.id == store_id and r.tenant_id == tenant_id:
                updated = r.model_copy(update={"name": body.name, "updated_at": datetime.now(UTC)})
                self.rows[store_id] = updated
                return VectorStoreInfo.map_from_db(updated)
        from src.common.exception import ValidationError

        raise ValidationError(
            code="vector_store.not_found",
            message=f"vector store {store_id} not found",
        )

    async def delete_store(self, tenant_id: int, store_id: str) -> bool:
        for r in self.rows.values():
            if r.id == store_id and r.tenant_id == tenant_id:
                updated = r.model_copy(update={"deleted_at": datetime.now(UTC)})
                self.rows[store_id] = updated
                return True
        return False

    async def test_by_id(
        self,
        tenant_id: int,
        store_id: str,
    ) -> _TestResponse:
        for r in self.rows.values():
            if r.id == store_id and r.tenant_id == tenant_id:
                return _TestResponse(success=True, version="", error=None)
        return _TestResponse(
            success=False,
            version=None,
            error="not found",
        )

    async def test_raw(
        self,
        engine_type: str,
        connection_config: dict[str, object],
    ) -> _TestResponse:
        return _TestResponse(success=True, version="", error=None)


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def fake_service() -> _FakeService:
    return _FakeService()


@pytest.fixture
def app(fake_service: _FakeService) -> FastAPI:
    application = create_app()
    override_auth_gates(application)
    application.dependency_overrides[get_vector_store_service] = lambda: fake_service
    # Mount the vector-store router in this isolated app instance so the
    # test exercises the full HTTP path without registering it in the
    # global lifespan.
    from src.web.api.infra.vector_stores.router import router as vs_router

    application.include_router(vs_router)
    return application


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ── GET /vector-stores/types ─────────────────────────────────────────


async def test_list_types_returns_seven_engines(client: AsyncClient) -> None:
    """The types endpoint returns the seven supported engine types."""
    resp = await client.get("/vector-stores/types")
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


async def test_test_raw_returns_success(client: AsyncClient) -> None:
    """A valid raw config yields a success response with empty version."""
    resp = await client.post(
        "/vector-stores/test",
        json={"engine_type": "elasticsearch", "connection_config": {"addr": "http://es:9200"}},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["version"] == ""


# ── POST /vector-stores ──────────────────────────────────────────────


async def test_create_vector_store_returns_envelope(
    client: AsyncClient,
    fake_service: _FakeService,
) -> None:
    """A create call returns the wrapped envelope with masked credentials."""
    resp = await client.post(
        "/vector-stores",
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
    assert len(fake_service.rows) == 1


# ── GET /vector-stores ──────────────────────────────────────────────


async def test_list_stores_returns_db_rows(
    client: AsyncClient,
    fake_service: _FakeService,
) -> None:
    """The list endpoint returns the DB-managed rows."""
    await client.post(
        "/vector-stores",
        json={
            "name": "es-a",
            "engine_type": "elasticsearch",
            "connection_config": {"addr": "http://es-a:9200"},
        },
    )
    resp = await client.get("/vector-stores")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    # Env stores are absent in this test (no RETRIEVE_DRIVER set).
    data = body["data"]
    assert len(data) == 1
    assert data[0]["source"] == "user"
    assert data[0]["readonly"] is False


async def test_list_stores_synthesises_env_entries(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Setting ``RETRIEVE_DRIVER`` surfaces an env-store virtual entry."""
    monkeypatch.setenv("RETRIEVE_DRIVER", "postgres")
    resp = await client.get("/vector-stores")
    assert resp.status_code == 200
    body = resp.json()
    data = body["data"]
    # The env entry comes first; check its identifying marks.
    assert data[0]["id"] == "__env_postgres__"
    assert data[0]["source"] == "env"
    assert data[0]["readonly"] is True


# ── GET /vector-stores/{id} ─────────────────────────────────────────


async def test_get_store_returns_envelope(client: AsyncClient) -> None:
    """The get endpoint returns the wrapped store after a create."""
    create_resp = await client.post(
        "/vector-stores",
        json={
            "name": "es-get",
            "engine_type": "elasticsearch",
            "connection_config": {"addr": "http://es:9200"},
        },
    )
    store_id = create_resp.json()["data"]["id"]
    resp = await client.get(f"/vector-stores/{store_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["data"]["id"] == store_id


async def test_get_unknown_store_returns_404(
    client: AsyncClient,
) -> None:
    """An unknown id is rejected with a 404-style status code."""
    resp = await client.get("/vector-stores/missing")
    # ValidationError is mapped to 422 by the exception handler.
    assert resp.status_code in (404, 422)


async def test_get_env_store_returns_envelope(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An env-store id resolves without touching the service."""
    monkeypatch.setenv("RETRIEVE_DRIVER", "qdrant")
    resp = await client.get("/vector-stores/__env_qdrant__")
    assert resp.status_code == 200
    body = resp.json()
    assert body["data"]["id"] == "__env_qdrant__"
    assert body["data"]["source"] == "env"
    assert body["data"]["readonly"] is True


# ── PUT /vector-stores/{id} ─────────────────────────────────────────


async def test_update_store_renames(
    client: AsyncClient,
) -> None:
    """The put endpoint only mutates the ``name`` field."""
    create_resp = await client.post(
        "/vector-stores",
        json={
            "name": "es-old",
            "engine_type": "elasticsearch",
            "connection_config": {"addr": "http://es:9200"},
        },
    )
    store_id = create_resp.json()["data"]["id"]
    resp = await client.put(
        f"/vector-stores/{store_id}",
        json={"name": "es-new"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["data"]["name"] == "es-new"


async def test_update_env_store_rejected(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Env-store ids cannot be updated."""
    monkeypatch.setenv("RETRIEVE_DRIVER", "qdrant")
    resp = await client.put(
        "/vector-stores/__env_qdrant__",
        json={"name": "renamed"},
    )
    # ValidationError is mapped to 422.
    assert resp.status_code in (400, 422)


# ── DELETE /vector-stores/{id} ──────────────────────────────────────


async def test_delete_store_soft_deletes(
    client: AsyncClient,
) -> None:
    """A successful delete returns the success envelope."""
    create_resp = await client.post(
        "/vector-stores",
        json={
            "name": "es-del",
            "engine_type": "elasticsearch",
            "connection_config": {"addr": "http://es:9200"},
        },
    )
    store_id = create_resp.json()["data"]["id"]
    resp = await client.delete(f"/vector-stores/{store_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    # The follow-up get returns a 422 / 404 — the row is invisible to reads.
    follow = await client.get(f"/vector-stores/{store_id}")
    assert follow.status_code in (404, 422)


async def test_delete_env_store_rejected(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Env-store ids cannot be deleted."""
    monkeypatch.setenv("RETRIEVE_DRIVER", "qdrant")
    resp = await client.delete("/vector-stores/__env_qdrant__")
    assert resp.status_code in (400, 422)


# ── POST /vector-stores/{id}/test ──────────────────────────────────


async def test_test_by_id_runs_probe(
    client: AsyncClient,
) -> None:
    """The by-id test endpoint surfaces the probe's success response."""
    create_resp = await client.post(
        "/vector-stores",
        json={
            "name": "es-probe",
            "engine_type": "elasticsearch",
            "connection_config": {"addr": "http://es:9200"},
        },
    )
    store_id = create_resp.json()["data"]["id"]
    resp = await client.post(f"/vector-stores/{store_id}/test")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True


async def test_test_env_store_runs_probe(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The by-id test endpoint probes env-store entries directly."""
    monkeypatch.setenv("RETRIEVE_DRIVER", "qdrant")
    resp = await client.post("/vector-stores/__env_qdrant__/test")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True


__all__ = []
