"""Live e2e tests for the storage-backends infra router.

Exercises the full HTTP path over ``TestClient`` against the real app.
The service dependency is overridden with a real ``StorageBackendService``
backed by an in-memory repository, so the tests run without a database
or any live storage adapter.

Endpoint coverage:

| Method | Path                              |
| ------ | --------------------------------- |
| GET    | /storage-backends/types           |
| POST   | /storage-backends/test            |
| POST   | /storage-backends                 |
| GET    | /storage-backends                 |
| GET    | /storage-backends/{id}            |
| PUT    | /storage-backends/{id}            |
| DELETE | /storage-backends/{id}            |
| POST   | /storage-backends/{id}/test       |
| PUT    | /storage-backends/{id}/default    |

Auth: header trio on the authed client; unauth tests build a bare
``TestClient`` and assert the 401 from ``require_auth``.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.common.exception import StorageBackendError
from src.core.infra.storage_backends.service.storage_backend_service import (
    StorageBackendService,
)
from src.core.infra.storage_backends.types import (
    REDACTED_SECRET_PLACEHOLDER,
    StorageBackendConfigInfo,
)
from src.db.dao.storage_backend_repository import StorageBackendRepository
from src.db.models.storage_backend import (
    STORAGE_BACKEND_SOURCE_USER,
    STORAGE_BACKEND_STATUS_ACTIVE,
    StorageBackend,
)
from src.web.deps.infra_storage_backends import get_storage_backend_service

_NOW = datetime(2026, 1, 1, tzinfo=UTC)

_MINIO_CONFIG = StorageBackendConfigInfo(
    mode="remote",
    endpoint="storage.example.com:9000",
    access_key_id="AKIA_EXAMPLE",
    secret_access_key="secret-example",
    bucket_name="documents",
)

_MINIO_BODY_CONFIG: dict[str, Any] = {
    "mode": "remote",
    "endpoint": "storage.example.com:9000",
    "access_key_id": "AKIA_EXAMPLE",
    "secret_access_key": "secret-example",
    "bucket_name": "documents",
}


# ── Repository double ─────────────────────────────────────────────────


class _RepoDouble(StorageBackendRepository):
    """In-memory drop-in for ``StorageBackendRepository``.

    Bypasses the SQL path entirely; every method returns rows that
    ``StorageBackendInfo.map_from_db`` can consume. The class derives
    from the real repo so type compatibility with the service is
    preserved.
    """

    def __init__(self) -> None:
        # Skip the parent ``__init__`` (which expects an ``AsyncSession``);
        # only the data slots are needed for the in-memory backend.
        self.rows: dict[str, StorageBackend] = {}
        self.default_backend_ids: dict[int, str] = {}
        self.kb_refs = 0
        self.resource_refs = 0

    def _live(self, tenant_id: int) -> dict[str, StorageBackend]:
        return {
            bid: r
            for bid, r in self.rows.items()
            if r.tenant_id == tenant_id and r.deleted_at is None
        }

    async def get_by_id(self, *, tenant_id: int, id: str) -> StorageBackend | None:
        row = self.rows.get(id)
        if row is None or row.tenant_id != tenant_id or row.deleted_at is not None:
            return None
        return row

    async def list_for_tenant(self, tenant_id: int) -> list[StorageBackend]:
        rows = sorted(
            self._live(tenant_id).values(),
            key=lambda r: (r.created_at, r.id),
        )
        return list(rows)

    async def find_legacy_alias(self, *, tenant_id: int, provider: str) -> StorageBackend | None:
        for r in await self.list_for_tenant(tenant_id):
            if r.provider == provider and r.legacy_alias:
                return r
        return None

    async def find_by_name(self, *, tenant_id: int, name: str) -> StorageBackend | None:
        for r in await self.list_for_tenant(tenant_id):
            if r.name == name:
                return r
        return None

    async def create(self, row: StorageBackend) -> StorageBackend:
        self.rows[row.id] = row
        return row

    async def update_columns(
        self, *, tenant_id: int, id: str, columns: dict[str, Any]
    ) -> StorageBackend | None:
        existing = await self.get_by_id(tenant_id=tenant_id, id=id)
        if existing is None:
            return None
        updated = existing.model_copy(update=dict(columns))
        self.rows[id] = updated
        return updated

    async def soft_delete(self, *, tenant_id: int, id: str) -> bool:
        existing = await self.get_by_id(tenant_id=tenant_id, id=id)
        if existing is None:
            return False
        now = datetime.now(UTC)
        self.rows[id] = existing.model_copy(update={"deleted_at": now, "updated_at": now})
        return True

    async def get_default_backend_id(self, tenant_id: int) -> str | None:
        return self.default_backend_ids.get(tenant_id)

    async def set_default_backend_id(self, *, tenant_id: int, id: str) -> bool:
        self.default_backend_ids[tenant_id] = id
        return True

    async def count_knowledge_base_references(self, *, tenant_id: int, id: str) -> int:
        return self.kb_refs

    async def count_active_resource_references(self, *, tenant_id: int, id: str) -> int:
        return self.resource_refs


# ── Fixtures ──────────────────────────────────────────────────────────


class _PassingAdapter:
    """Adapter double whose probe always succeeds."""

    async def check_connectivity(self) -> None:
        return None


@pytest.fixture(autouse=True)
def _stub_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralise the outbound probe + SSRF check for every request."""

    async def _ok(**_kwargs: object) -> None:
        return None

    monkeypatch.setattr(
        StorageBackendService,
        "_adapter_for",
        lambda _self, **_kwargs: _PassingAdapter(),
    )
    monkeypatch.setattr(
        StorageBackendService,
        "_validate_endpoint",
        lambda _self, **_kwargs: _ok(),
    )


@pytest.fixture
def repo() -> _RepoDouble:
    return _RepoDouble()


@pytest.fixture
def app(
    web_app: FastAPI,
    repo: _RepoDouble,
) -> FastAPI:
    """Override the per-request storage-backend service factory."""

    def _override() -> StorageBackendService:
        return StorageBackendService(backend_repo=repo)

    web_app.dependency_overrides[get_storage_backend_service] = _override
    return web_app


@pytest.fixture
def client(app: FastAPI, web_authed_client: TestClient) -> TestClient:
    return web_authed_client


@pytest.fixture
def anon_client(app: FastAPI) -> Iterator[TestClient]:
    """A ``TestClient`` without the auth header trio — 401 surface."""
    with TestClient(app=app) as c:
        yield c


def _seed(
    repo: _RepoDouble,
    *,
    id: str = "backend-1",
    name: str = "Primary MinIO",
    provider: str = "minio",
    source: str = STORAGE_BACKEND_SOURCE_USER,
    status: str = STORAGE_BACKEND_STATUS_ACTIVE,
    tenant_id: int | None = None,
) -> StorageBackend:
    # ``tenant_id`` is resolved against the authed client's tenant at
    # call time so each test sees its own principal.
    resolved_tenant = tenant_id if tenant_id is not None else 0
    row = StorageBackend(
        id=id,
        tenant_id=resolved_tenant,
        name=name,
        provider=provider,
        config=_MINIO_CONFIG.to_json(),
        source=source,
        status=status,
        legacy_alias=False,
        created_at=_NOW,
        updated_at=_NOW,
    )
    repo.rows[id] = row
    return row


# ── GET /storage-backends/types ──────────────────────────────────────


async def test_list_provider_types_returns_the_allowed_set(client: TestClient) -> None:
    """The /types endpoint returns the providers from the allow-list."""
    resp = client.get("/api/v1/storage-backends/types")

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert "local" in body["data"]
    assert "minio" in body["data"]


# ── POST /storage-backends ──────────────────────────────────────────


async def test_create_returns_201_with_the_created_backend(
    client: TestClient,
    repo: _RepoDouble,
) -> None:
    """A create call returns the wrapped envelope and persists the row."""
    resp = client.post(
        "/api/v1/storage-backends",
        json={"name": "Primary MinIO", "provider": "minio", "config": _MINIO_BODY_CONFIG},
    )

    assert resp.status_code == 201
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["name"] == "Primary MinIO"
    assert body["data"]["provider"] == "minio"
    assert body["data"]["source"] == STORAGE_BACKEND_SOURCE_USER
    # Credentials are masked on the wire.
    assert body["data"]["config"]["access_key_id"] == REDACTED_SECRET_PLACEHOLDER
    assert body["data"]["config"]["secret_access_key"] == REDACTED_SECRET_PLACEHOLDER
    # The row is persisted in the in-memory repo.
    assert len(repo.rows) == 1


async def test_create_rejects_an_unsupported_provider_with_422(client: TestClient) -> None:
    """An unsupported provider fails validation with 422."""
    resp = client.post(
        "/api/v1/storage-backends",
        json={"name": "Nope", "provider": "dropbox", "config": _MINIO_BODY_CONFIG},
    )

    assert resp.status_code == 422


async def test_create_rejects_blank_name_with_422(client: TestClient) -> None:
    """A blank name fails validation with 422."""
    resp = client.post(
        "/api/v1/storage-backends",
        json={"name": "   ", "provider": "minio", "config": _MINIO_BODY_CONFIG},
    )

    assert resp.status_code == 422


# ── GET /storage-backends ───────────────────────────────────────────


async def test_list_returns_tenant_scoped_backends_in_envelope(
    client: TestClient,
    repo: _RepoDouble,
    admin_user: tuple[str, int],
) -> None:
    """The list endpoint returns the workspace's backends with masked creds."""
    _seed(repo, id="backend-1", name="A", tenant_id=admin_user[1])
    _seed(repo, id="backend-2", name="B", tenant_id=admin_user[1])
    repo.default_backend_ids[admin_user[1]] = "backend-2"

    resp = client.get("/api/v1/storage-backends")

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert [b["id"] for b in body["data"]] == ["backend-1", "backend-2"]
    assert body["default_storage_backend_id"] == "backend-2"
    # Both rows have masked credentials on the wire.
    for entry in body["data"]:
        assert entry["config"]["secret_access_key"] == REDACTED_SECRET_PLACEHOLDER


# ── GET /storage-backends/{id} ──────────────────────────────────────


async def test_get_returns_backend_with_masked_credentials(
    client: TestClient,
    repo: _RepoDouble,
    admin_user: tuple[str, int],
) -> None:
    """A get for an existing id returns the wrapped backend, creds masked."""
    _seed(repo, tenant_id=admin_user[1])

    resp = client.get("/api/v1/storage-backends/backend-1")

    assert resp.status_code == 200
    body = resp.json()
    assert body["data"]["id"] == "backend-1"
    assert body["data"]["config"]["access_key_id"] == REDACTED_SECRET_PLACEHOLDER


async def test_get_missing_backend_returns_404(
    client: TestClient,
) -> None:
    """An unknown id is rejected with 404."""
    resp = client.get("/api/v1/storage-backends/ghost")

    assert resp.status_code == 404


# ── PUT /storage-backends/{id} ──────────────────────────────────────


async def test_update_renames_and_preserves_redacted_secrets(
    client: TestClient,
    repo: _RepoDouble,
    admin_user: tuple[str, int],
) -> None:
    """A put with redacted credentials keeps the stored secrets untouched."""
    _seed(repo, tenant_id=admin_user[1])
    masked = dict(_MINIO_BODY_CONFIG)
    masked["access_key_id"] = REDACTED_SECRET_PLACEHOLDER
    masked["secret_access_key"] = REDACTED_SECRET_PLACEHOLDER

    resp = client.put(
        "/api/v1/storage-backends/backend-1",
        json={"name": "Renamed", "config": masked},
    )

    assert resp.status_code == 200
    assert resp.json()["data"]["name"] == "Renamed"
    # The underlying stored secret survives the masked re-submission.
    stored = StorageBackendConfigInfo.from_json(repo.rows["backend-1"].config)
    assert stored.secret_access_key == "secret-example"


async def test_update_missing_backend_returns_404(client: TestClient) -> None:
    """Updating an unknown id returns 404."""
    resp = client.put("/api/v1/storage-backends/ghost", json={"name": "X"})
    assert resp.status_code == 404


# ── DELETE /storage-backends/{id} ───────────────────────────────────


async def test_delete_returns_success_and_soft_deletes(
    client: TestClient,
    repo: _RepoDouble,
    admin_user: tuple[str, int],
) -> None:
    """A successful delete returns the ack envelope and clears the row."""
    _seed(repo, tenant_id=admin_user[1])

    resp = client.delete("/api/v1/storage-backends/backend-1")

    assert resp.status_code == 200
    assert resp.json() == {"success": True}
    assert repo.rows["backend-1"].deleted_at is not None


async def test_delete_missing_backend_returns_404(client: TestClient) -> None:
    """Deleting an unknown id returns 404."""
    resp = client.delete("/api/v1/storage-backends/ghost")
    assert resp.status_code == 404


# ── POST /storage-backends/test ─────────────────────────────────────


async def test_test_raw_returns_success_envelope(client: TestClient) -> None:
    """A raw probe returns 200 with ``success=true``."""
    resp = client.post(
        "/api/v1/storage-backends/test",
        json={"name": "Probe", "provider": "minio", "config": _MINIO_BODY_CONFIG},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["error"] is None


async def test_test_raw_reports_probe_failure_inside_envelope(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed probe answers 200 with ``success=false`` (probe-as-data)."""

    class _FailingAdapter:
        async def check_connectivity(self) -> None:
            raise StorageBackendError(
                code="storage_backend.bucket_unavailable",
                message="bucket unavailable",
            )

    monkeypatch.setattr(
        StorageBackendService,
        "_adapter_for",
        lambda _self, **_kwargs: _FailingAdapter(),
    )

    resp = client.post(
        "/api/v1/storage-backends/test",
        json={"name": "Probe", "provider": "minio", "config": _MINIO_BODY_CONFIG},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is False
    assert body["error"] == "bucket unavailable"


# ── POST /storage-backends/{id}/test ────────────────────────────────


async def test_test_by_id_returns_success_envelope(
    client: TestClient,
    repo: _RepoDouble,
    admin_user: tuple[str, int],
) -> None:
    """The by-id probe surfaces ``success=true``."""
    _seed(repo, tenant_id=admin_user[1])

    resp = client.post("/api/v1/storage-backends/backend-1/test")

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True


async def test_test_by_id_missing_returns_404(client: TestClient) -> None:
    """Probing an unknown id returns 404."""
    resp = client.post("/api/v1/storage-backends/ghost/test")
    assert resp.status_code == 404


# ── PUT /storage-backends/{id}/default ──────────────────────────────


async def test_set_default_returns_ack_and_records_default(
    client: TestClient,
    repo: _RepoDouble,
    admin_user: tuple[str, int],
) -> None:
    """Setting the workspace default returns the ack envelope."""
    _seed(repo, tenant_id=admin_user[1])

    resp = client.put("/api/v1/storage-backends/backend-1/default")

    assert resp.status_code == 200
    assert resp.json() == {"success": True}
    assert repo.default_backend_ids[admin_user[1]] == "backend-1"


async def test_set_default_missing_backend_returns_404(client: TestClient) -> None:
    """Setting the default to an unknown id returns 404."""
    resp = client.put("/api/v1/storage-backends/ghost/default")
    assert resp.status_code == 404


# ── Auth gate ────────────────────────────────────────────────────────


async def test_unauthed_request_returns_401(anon_client: TestClient) -> None:
    """A read without the header trio is rejected with 401."""
    resp = anon_client.get("/api/v1/storage-backends")
    assert resp.status_code == 401


async def test_unauthed_post_returns_401(anon_client: TestClient) -> None:
    """Writes also require the header trio."""
    resp = anon_client.post(
        "/api/v1/storage-backends",
        json={"name": "X", "provider": "minio", "config": _MINIO_BODY_CONFIG},
    )
    assert resp.status_code == 401


__all__ = [
    "_RepoDouble",
    "anon_client",
    "app",
    "client",
    "repo",
]
