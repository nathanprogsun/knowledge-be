"""Live e2e tests for the vector-stores infra router.

Exercises the full HTTP path over ``TestClient`` against the real app:
routing, serialization, role gates, exception handling. The service
dependency is overridden with a stateful closure-backed fake, so the
tests run without a database or any live engine.

Endpoint coverage:

| Method | Path                          |
| ------ | ----------------------------- |
| GET    | /vector-stores/types          |
| POST   | /vector-stores/test           |
| POST   | /vector-stores                |
| GET    | /vector-stores                |
| GET    | /vector-stores/{id}           |
| PUT    | /vector-stores/{id}           |
| DELETE | /vector-stores/{id}           |
| POST   | /vector-stores/{id}/test      |

Auth: the authed client carries the ``X-User-Id/X-Tenant-ID/X-Roles`` header trio;
the unauthorised tests build a bare ``TestClient`` and assert the
401 raised by the global ``require_auth`` dependency.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

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
from src.core.infra.vector_stores.types import VectorStoreInfo
from src.db.models.infra.vector_store import VectorStore
from src.web.deps.infra_vector_stores import get_vector_store_service

# ── Fake service backing the dep override ────────────────────────────


class _FakeVectorStoreService:
    """Stateful in-memory replacement for ``VectorStoreService``.

    Each method records the latest arguments for assertion and returns
    a value shaped to match what the real service hands back, so the
    view / router layers run unmodified.
    """

    def __init__(self) -> None:
        self.rows: dict[str, VectorStore] = {}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._id_seq = itertools.count(1)

    def _next_id(self) -> str:
        return f"vs-test-{next(self._id_seq)}"

    def _record(self, method: str, **kwargs: Any) -> None:
        self.calls.append((method, kwargs))

    async def list_stores(self, tenant_id: int) -> list[VectorStoreInfo]:
        self._record("list_stores", tenant_id=tenant_id)
        out = [r for r in self.rows.values() if r.tenant_id == tenant_id and r.deleted_at is None]
        out.sort(key=lambda r: r.created_at, reverse=True)
        return [VectorStoreInfo.map_from_db(r) for r in out]

    async def get_store(self, tenant_id: int, store_id: str) -> VectorStoreInfo:
        self._record("get_store", tenant_id=tenant_id, store_id=store_id)
        for r in self.rows.values():
            if r.id == store_id and r.tenant_id == tenant_id and r.deleted_at is None:
                return VectorStoreInfo.map_from_db(r)
        raise ValidationError(
            code="vector_store.not_found",
            message=f"vector store {store_id} not found",
        )

    async def create_store(
        self, *, tenant_id: int, body: CreateVectorStoreRequest
    ) -> VectorStoreInfo:
        self._record("create_store", tenant_id=tenant_id, name=body.name, engine=body.engine_type)
        if not body.name.strip():
            raise ValidationError(
                code="vector_store.name_required",
                message="name is required",
            )
        now = datetime.now(UTC)
        row = VectorStore(
            id=self._next_id(),
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
        self, *, tenant_id: int, store_id: str, body: UpdateVectorStoreRequest
    ) -> VectorStoreInfo:
        self._record("update_store", tenant_id=tenant_id, store_id=store_id, name=body.name)
        existing = self.rows.get(store_id)
        if existing is None or existing.tenant_id != tenant_id or existing.deleted_at is not None:
            raise ValidationError(
                code="vector_store.not_found",
                message=f"vector store {store_id} not found",
            )
        updated = existing.model_copy(update={"name": body.name, "updated_at": datetime.now(UTC)})
        self.rows[store_id] = updated
        return VectorStoreInfo.map_from_db(updated)

    async def delete_store(self, tenant_id: int, store_id: str) -> bool:
        self._record("delete_store", tenant_id=tenant_id, store_id=store_id)
        existing = self.rows.get(store_id)
        if existing is None or existing.tenant_id != tenant_id:
            return False
        self.rows[store_id] = existing.model_copy(update={"deleted_at": datetime.now(UTC)})
        return True

    async def test_by_id(self, tenant_id: int, store_id: str) -> _TestResponse:
        self._record("test_by_id", tenant_id=tenant_id, store_id=store_id)
        for r in self.rows.values():
            if r.id == store_id and r.tenant_id == tenant_id and r.deleted_at is None:
                return _TestResponse(success=True, version="v8.0.0", error=None)
        raise ValidationError(
            code="vector_store.not_found",
            message=f"vector store {store_id} not found",
        )

    async def test_raw(self, engine_type: str, connection_config: dict[str, Any]) -> _TestResponse:
        self._record(
            "test_raw",
            engine_type=engine_type,
            connection_config=connection_config,
        )
        return _TestResponse(success=True, version="v8.0.0", error=None)


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def fake_service() -> _FakeVectorStoreService:
    return _FakeVectorStoreService()


@pytest.fixture
def app(
    web_app: FastAPI,
    fake_service: _FakeVectorStoreService,
) -> FastAPI:
    """Override the per-request vector-store service factory."""
    web_app.dependency_overrides[get_vector_store_service] = lambda: fake_service
    return web_app


@pytest.fixture
def client(app: FastAPI, web_authed_client: TestClient) -> TestClient:
    return web_authed_client


@pytest.fixture
def anon_client(app: FastAPI) -> Iterator[TestClient]:
    """A ``TestClient`` without the auth header trio — 401 surface."""
    with TestClient(app=app) as c:
        yield c


def _create_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "name": "es-primary",
        "engine_type": "elasticsearch",
        "connection_config": {"addr": "http://es:9200"},
    }
    body.update(overrides)
    return body


# ── GET /vector-stores/types ──────────────────────────────────────────


async def test_list_types_returns_registered_engines(client: TestClient) -> None:
    """The /types endpoint surfaces the static engine-type registry."""
    resp = client.get("/api/v1/vector-stores/types")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    types = {entry["type"] for entry in body["data"]}
    # The registry exposes exactly these engines.
    assert types == {
        "elasticsearch",
        "qdrant",
        "milvus",
        "tencent_vectordb",
        "weaviate",
        "doris",
        "opensearch",
    }


# ── POST /vector-stores/test ─────────────────────────────────────────


async def test_test_raw_returns_success_envelope(client: TestClient) -> None:
    """A valid raw config yields 200 with ``success=true`` + a version."""
    resp = client.post(
        "/api/v1/vector-stores/test",
        json={"engine_type": "elasticsearch", "connection_config": {"addr": "http://es:9200"}},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["version"] == "v8.0.0"
    assert body["error"] is None


async def test_test_raw_reports_failure_within_envelope(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing probe still returns 200 with ``success=false``."""

    async def _fail(self: Any, **kwargs: Any) -> _TestResponse:
        return _TestResponse(
            success=False,
            version=None,
            error="connection refused",
        )

    monkeypatch.setattr(_FakeVectorStoreService, "test_raw", _fail)
    resp = client.post(
        "/api/v1/vector-stores/test",
        json={"engine_type": "elasticsearch", "connection_config": {"addr": "http://es:9200"}},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is False
    assert body["error"] == "connection refused"


async def test_test_raw_rejects_missing_required_field_with_422(client: TestClient) -> None:
    """A request body missing ``engine_type`` is rejected by Pydantic."""
    resp = client.post(
        "/api/v1/vector-stores/test",
        json={"connection_config": {"addr": "http://es:9200"}},
    )
    assert resp.status_code == 422


# ── POST /vector-stores ──────────────────────────────────────────────


async def test_create_returns_201_with_id_and_masked_credentials(
    client: TestClient,
    fake_service: _FakeVectorStoreService,
) -> None:
    """The create endpoint returns the wrapped envelope; credentials masked."""
    resp = client.post(
        "/api/v1/vector-stores",
        json=_create_body(
            name="es-hot",
            connection_config={"addr": "http://es:9200", "password": "secret"},
        ),
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["name"] == "es-hot"
    assert body["data"]["source"] == "user"
    assert body["data"]["readonly"] is False
    assert body["data"]["connection_config"]["password"] == "***"
    # The id is server-generated and persisted.
    assert body["data"]["id"] in fake_service.rows


async def test_create_rejects_blank_name_with_422(client: TestClient) -> None:
    """An empty name is rejected at the validation boundary."""
    resp = client.post(
        "/api/v1/vector-stores",
        json=_create_body(name="   "),
    )
    assert resp.status_code == 422


# ── GET /vector-stores ───────────────────────────────────────────────


async def test_list_returns_tenant_scoped_rows_in_envelope(
    client: TestClient,
) -> None:
    """The list endpoint returns the tenant's vector stores in the envelope."""
    client.post("/api/v1/vector-stores", json=_create_body(name="es-a"))
    client.post("/api/v1/vector-stores", json=_create_body(name="es-b"))

    resp = client.get("/api/v1/vector-stores")

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert len(body["data"]) == 2
    names = {row["name"] for row in body["data"]}
    assert names == {"es-a", "es-b"}


# ── GET /vector-stores/{id} ──────────────────────────────────────────


async def test_get_returns_envelope_for_existing_store(client: TestClient) -> None:
    """The get endpoint returns the requested store."""
    created = client.post("/api/v1/vector-stores", json=_create_body(name="es-get"))
    store_id = created.json()["data"]["id"]

    resp = client.get(f"/api/v1/vector-stores/{store_id}")

    assert resp.status_code == 200
    body = resp.json()
    assert body["data"]["id"] == store_id
    assert body["data"]["name"] == "es-get"


async def test_get_unknown_store_returns_error_status(client: TestClient) -> None:
    """An unknown id is reported as a validation error (422) or 404."""
    resp = client.get("/api/v1/vector-stores/does-not-exist")
    # The router raises ValidationError; the global handler maps to 422
    # unless a dedicated mapping for ``not_found`` codes exists. Accept
    # either rendering.
    assert resp.status_code in (404, 422)


# ── PUT /vector-stores/{id} ──────────────────────────────────────────


async def test_update_renames_the_store(client: TestClient) -> None:
    """The put endpoint mutates only the mutable ``name`` field."""
    created = client.post("/api/v1/vector-stores", json=_create_body(name="es-old"))
    store_id = created.json()["data"]["id"]

    resp = client.put(f"/api/v1/vector-stores/{store_id}", json={"name": "es-new"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["data"]["name"] == "es-new"


async def test_update_unknown_store_returns_error_status(client: TestClient) -> None:
    """Updating an unknown id yields 422/404, not a silent success."""
    resp = client.put("/api/v1/vector-stores/missing", json={"name": "renamed"})
    assert resp.status_code in (404, 422)


# ── DELETE /vector-stores/{id} ───────────────────────────────────────


async def test_delete_returns_success_and_marks_row_invisible(
    client: TestClient,
) -> None:
    """A successful delete returns the success envelope; follow-up get 4xxs."""
    created = client.post("/api/v1/vector-stores", json=_create_body(name="es-del"))
    store_id = created.json()["data"]["id"]

    resp = client.delete(f"/api/v1/vector-stores/{store_id}")
    assert resp.status_code == 200
    assert resp.json() == {"success": True}

    # Follow-up get returns 422/404 because the row is soft-deleted.
    follow = client.get(f"/api/v1/vector-stores/{store_id}")
    assert follow.status_code in (404, 422)


async def test_delete_unknown_store_returns_error_status(client: TestClient) -> None:
    """Deleting an unknown id yields a 4xx (the router translates to 404)."""
    resp = client.delete("/api/v1/vector-stores/missing")
    assert resp.status_code in (404, 422)


# ── POST /vector-stores/{id}/test ────────────────────────────────────


async def test_test_by_id_returns_probe_response(client: TestClient) -> None:
    """The by-id test endpoint surfaces the probe's outcome."""
    created = client.post("/api/v1/vector-stores", json=_create_body(name="es-probe"))
    store_id = created.json()["data"]["id"]

    resp = client.post(f"/api/v1/vector-stores/{store_id}/test")

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["version"] == "v8.0.0"


async def test_test_by_id_unknown_returns_error_in_envelope(client: TestClient) -> None:
    """Probing an unknown id surfaces the error inside the success envelope.

    The router maps a service-level ``ValidationError`` to a 200 with
    ``success=false`` (so the UI renders the error inline rather than
    treating it as a transport failure). The test pins that shape.
    """
    resp = client.post("/api/v1/vector-stores/missing/test")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is False
    assert body["error"]


# ── Auth gate ────────────────────────────────────────────────────────


async def test_unauthed_request_returns_401(anon_client: TestClient) -> None:
    """Requests without the header trio are rejected by the auth gate."""
    resp = anon_client.get("/api/v1/vector-stores/types")
    assert resp.status_code == 401


async def test_unauthed_post_returns_401(anon_client: TestClient) -> None:
    """Writes also require the header trio; no header means 401."""
    resp = anon_client.post(
        "/api/v1/vector-stores",
        json=_create_body(name="es-401"),
    )
    assert resp.status_code == 401


__all__ = [
    "_FakeVectorStoreService",
    "anon_client",
    "app",
    "client",
    "fake_service",
]
