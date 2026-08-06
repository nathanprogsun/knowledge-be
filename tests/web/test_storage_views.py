"""Web-layer tests for the storage-backend router.

Per AGENTS.md §9, web routers are tested via ``httpx.AsyncClient`` against
the app. The service dependency is overridden with a service backed by an
in-memory repository, so the tests exercise the full HTTP path (routing,
role gates, serialization, exception handling) without a database.

Uses the shared ``web_app`` fixture (header-based auth) and applies
the service dep override on it; the real ``require_auth`` dep resolves
the principal via the ``x-knowledge-*`` header trio.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from src.common.exception import StorageBackendError
from src.core.infra.storage_backends.service.storage_backend_service import (
    StorageBackendService,
)
from src.core.infra.storage_backends.types import (
    REDACTED_SECRET_PLACEHOLDER,
    StorageBackendConfigInfo,
)
from src.db.models.storage_backend import (
    STORAGE_BACKEND_SOURCE_ENV,
    STORAGE_BACKEND_SOURCE_USER,
    STORAGE_BACKEND_STATUS_ACTIVE,
    STORAGE_BACKEND_STATUS_DISABLED,
    StorageBackend,
)
from src.web.deps.infra_storage_backends import get_storage_backend_service
from tests.integration.web.conftest import web_app, web_authed_client  # noqa: F401
from tests.unit.fakes.storage_backends import FakeStorageBackendRepository

# The header auth channel pins the active workspace to 1.
_TENANT_ID = 1
_NOW = datetime(2026, 1, 1, tzinfo=UTC)

_MINIO_CONFIG = StorageBackendConfigInfo(
    mode="remote",
    endpoint="storage.example.com:9000",
    access_key_id="AKIA_EXAMPLE",
    secret_access_key="secret-example",
    bucket_name="documents",
)

_MINIO_BODY_CONFIG = {
    "mode": "remote",
    "endpoint": "storage.example.com:9000",
    "access_key_id": "AKIA_EXAMPLE",
    "secret_access_key": "secret-example",
    "bucket_name": "documents",
}


class _PassingAdapter:
    """Adapter double whose probe always succeeds."""

    async def check_connectivity(self) -> None:
        return None


@pytest.fixture(autouse=True)
def _stub_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralise the outbound probe and SSRF check for every request."""

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
def repo() -> FakeStorageBackendRepository:
    return FakeStorageBackendRepository()


@pytest.fixture
def app(
    request: pytest.FixtureRequest,  # noqa: ARG001 - explicit fixture-param
    web_app: FastAPI,  # noqa: ARG001 - resolved from the parent conftest
    repo: FakeStorageBackendRepository,
) -> FastAPI:
    """Override ``get_storage_backend_service`` on the shared web app."""
    web_app.dependency_overrides[get_storage_backend_service] = (
        lambda: StorageBackendService(backend_repo=repo)  # type: ignore[arg-type]
    )
    return web_app


@pytest.fixture
def client(app: FastAPI, web_authed_client: AsyncClient) -> AsyncClient:  # noqa: ARG001
    """Alias ``web_authed_client``; depending on ``app`` forces the
    dep-override fixture to run before the test executes."""
    return web_authed_client


def _seed(
    repo: FakeStorageBackendRepository,
    *,
    id: str = "backend-1",
    tenant_id: int = _TENANT_ID,
    name: str = "Primary MinIO",
    provider: str = "minio",
    source: str = STORAGE_BACKEND_SOURCE_USER,
    status: str = STORAGE_BACKEND_STATUS_ACTIVE,
    legacy_alias: bool = False,
) -> StorageBackend:
    row = StorageBackend(
        id=id,
        tenant_id=tenant_id,
        name=name,
        provider=provider,
        config=_MINIO_CONFIG.to_json(),
        source=source,
        status=status,
        legacy_alias=legacy_alias,
        created_at=_NOW,
        updated_at=_NOW,
    )
    repo.rows[id] = row
    return row


# ── GET /storage-backends/types ─────────────────────────────────────


async def test_list_provider_types_returns_the_allowed_set(client: AsyncClient) -> None:
    resp = await client.get("/storage-backends/types")

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert "local" in body["data"]
    assert "minio" in body["data"]


# ── POST /storage-backends ──────────────────────────────────────────


async def test_create_returns_201_with_the_created_backend(
    client: AsyncClient,
    repo: FakeStorageBackendRepository,
) -> None:
    resp = await client.post(
        "/storage-backends",
        json={"name": "Primary MinIO", "provider": "minio", "config": _MINIO_BODY_CONFIG},
    )

    assert resp.status_code == 201
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["name"] == "Primary MinIO"
    assert body["data"]["provider"] == "minio"
    assert body["data"]["source"] == STORAGE_BACKEND_SOURCE_USER
    assert len(repo.rows) == 1


async def test_create_rejects_a_duplicate_name_with_409(
    client: AsyncClient,
    repo: FakeStorageBackendRepository,
) -> None:
    _seed(repo, name="Primary MinIO")

    resp = await client.post(
        "/storage-backends",
        json={"name": "Primary MinIO", "provider": "minio", "config": _MINIO_BODY_CONFIG},
    )

    assert resp.status_code == 409


async def test_create_rejects_an_unsupported_provider_with_422(
    client: AsyncClient,
) -> None:
    resp = await client.post(
        "/storage-backends",
        json={"name": "Nope", "provider": "dropbox", "config": _MINIO_BODY_CONFIG},
    )

    assert resp.status_code == 422


async def test_create_rejects_a_missing_required_config_field_with_422(
    client: AsyncClient,
) -> None:
    resp = await client.post(
        "/storage-backends",
        json={"name": "Incomplete", "provider": "minio", "config": {"mode": "remote"}},
    )

    assert resp.status_code == 422


# ── GET /storage-backends ───────────────────────────────────────────


async def test_list_masks_credentials_and_reports_the_default(
    client: AsyncClient,
    repo: FakeStorageBackendRepository,
) -> None:
    _seed(repo, id="backend-1", name="A")
    _seed(repo, id="backend-2", name="B")
    repo.default_backend_id[_TENANT_ID] = "backend-2"

    resp = await client.get("/storage-backends")

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert [b["id"] for b in body["data"]] == ["backend-1", "backend-2"]
    assert body["default_storage_backend_id"] == "backend-2"
    assert body["data"][0]["config"]["secret_access_key"] == REDACTED_SECRET_PLACEHOLDER


async def test_list_is_empty_for_a_workspace_with_no_backends(
    client: AsyncClient,
) -> None:
    resp = await client.get("/storage-backends")

    assert resp.status_code == 200
    body = resp.json()
    assert body["data"] == []
    assert body["default_storage_backend_id"] is None


async def test_list_excludes_other_workspaces(
    client: AsyncClient,
    repo: FakeStorageBackendRepository,
) -> None:
    _seed(repo, id="mine")
    _seed(repo, id="theirs", tenant_id=99)

    resp = await client.get("/storage-backends")

    assert [b["id"] for b in resp.json()["data"]] == ["mine"]


# ── GET /storage-backends/{id} ──────────────────────────────────────


async def test_get_returns_the_backend_with_masked_credentials(
    client: AsyncClient,
    repo: FakeStorageBackendRepository,
) -> None:
    _seed(repo)

    resp = await client.get("/storage-backends/backend-1")

    assert resp.status_code == 200
    body = resp.json()
    assert body["data"]["id"] == "backend-1"
    assert body["data"]["config"]["access_key_id"] == REDACTED_SECRET_PLACEHOLDER


async def test_get_of_a_missing_backend_returns_404(client: AsyncClient) -> None:
    resp = await client.get("/storage-backends/ghost")

    assert resp.status_code == 404


async def test_get_does_not_cross_workspaces(
    client: AsyncClient,
    repo: FakeStorageBackendRepository,
) -> None:
    _seed(repo, tenant_id=99)

    resp = await client.get("/storage-backends/backend-1")

    assert resp.status_code == 404


# ── PUT /storage-backends/{id} ──────────────────────────────────────


async def test_update_renames_and_preserves_redacted_secrets(
    client: AsyncClient,
    repo: FakeStorageBackendRepository,
) -> None:
    _seed(repo)
    masked_config = dict(_MINIO_BODY_CONFIG)
    masked_config["access_key_id"] = REDACTED_SECRET_PLACEHOLDER
    masked_config["secret_access_key"] = REDACTED_SECRET_PLACEHOLDER

    resp = await client.put(
        "/storage-backends/backend-1",
        json={"name": "Renamed", "config": masked_config},
    )

    assert resp.status_code == 200
    assert resp.json()["data"]["name"] == "Renamed"
    stored = StorageBackendConfigInfo.from_json(repo.rows["backend-1"].config)
    assert stored.secret_access_key == "secret-example"


async def test_update_rejects_a_location_change_with_422(
    client: AsyncClient,
    repo: FakeStorageBackendRepository,
) -> None:
    _seed(repo)
    moved = dict(_MINIO_BODY_CONFIG)
    moved["bucket_name"] = "elsewhere"

    resp = await client.put("/storage-backends/backend-1", json={"config": moved})

    assert resp.status_code == 422


async def test_update_of_an_env_backend_returns_422(
    client: AsyncClient,
    repo: FakeStorageBackendRepository,
) -> None:
    _seed(repo, source=STORAGE_BACKEND_SOURCE_ENV)

    resp = await client.put("/storage-backends/backend-1", json={"name": "Renamed"})

    assert resp.status_code == 422


async def test_update_can_disable_an_unreferenced_backend(
    client: AsyncClient,
    repo: FakeStorageBackendRepository,
) -> None:
    _seed(repo)

    resp = await client.put(
        "/storage-backends/backend-1",
        json={"status": STORAGE_BACKEND_STATUS_DISABLED},
    )

    assert resp.status_code == 200
    assert resp.json()["data"]["status"] == STORAGE_BACKEND_STATUS_DISABLED


async def test_update_of_a_missing_backend_returns_404(client: AsyncClient) -> None:
    resp = await client.put("/storage-backends/ghost", json={"name": "X"})

    assert resp.status_code == 404


# ── DELETE /storage-backends/{id} ───────────────────────────────────


async def test_delete_acknowledges_and_soft_deletes(
    client: AsyncClient,
    repo: FakeStorageBackendRepository,
) -> None:
    _seed(repo)

    resp = await client.delete("/storage-backends/backend-1")

    assert resp.status_code == 200
    assert resp.json() == {"success": True}
    assert repo.rows["backend-1"].deleted_at is not None


async def test_delete_of_the_default_returns_422(
    client: AsyncClient,
    repo: FakeStorageBackendRepository,
) -> None:
    _seed(repo)
    repo.default_backend_id[_TENANT_ID] = "backend-1"

    resp = await client.delete("/storage-backends/backend-1")

    assert resp.status_code == 422


async def test_delete_of_a_bound_backend_returns_422(
    client: AsyncClient,
    repo: FakeStorageBackendRepository,
) -> None:
    _seed(repo)
    repo.knowledge_base_references = 2

    resp = await client.delete("/storage-backends/backend-1")

    assert resp.status_code == 422


async def test_delete_of_a_missing_backend_returns_404(client: AsyncClient) -> None:
    resp = await client.delete("/storage-backends/ghost")

    assert resp.status_code == 404


# ── POST /storage-backends/test ─────────────────────────────────────


async def test_test_raw_config_returns_success_true(client: AsyncClient) -> None:
    resp = await client.post(
        "/storage-backends/test",
        json={"name": "Probe", "provider": "minio", "config": _MINIO_BODY_CONFIG},
    )

    assert resp.status_code == 200
    assert resp.json()["success"] is True


async def test_test_raw_config_reports_a_probe_failure_as_200(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    resp = await client.post(
        "/storage-backends/test",
        json={"name": "Probe", "provider": "minio", "config": _MINIO_BODY_CONFIG},
    )

    # A connectivity failure travels as data, not as an HTTP error.
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is False
    assert body["error"] == "bucket unavailable"


async def test_test_raw_config_reports_an_invalid_body_with_200(
    client: AsyncClient,
) -> None:
    """A validation failure answers 200 with ``success=false`` (Go keeps
    the HTTP status at 200 and reports the error in the body)."""
    resp = await client.post(
        "/storage-backends/test",
        json={"name": "Probe", "provider": "minio", "config": {"mode": "remote"}},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is False
    assert body["error"]


# ── POST /storage-backends/{id}/test ────────────────────────────────


async def test_test_saved_backend_returns_success_true(
    client: AsyncClient,
    repo: FakeStorageBackendRepository,
) -> None:
    _seed(repo)

    resp = await client.post("/storage-backends/backend-1/test")

    assert resp.status_code == 200
    assert resp.json()["success"] is True


async def test_test_of_a_missing_backend_returns_404(client: AsyncClient) -> None:
    resp = await client.post("/storage-backends/ghost/test")

    assert resp.status_code == 404


# ── PUT /storage-backends/{id}/default ──────────────────────────────


async def test_set_default_acknowledges_and_points_the_workspace(
    client: AsyncClient,
    repo: FakeStorageBackendRepository,
) -> None:
    _seed(repo)

    resp = await client.put("/storage-backends/backend-1/default")

    assert resp.status_code == 200
    assert resp.json() == {"success": True}
    assert repo.default_backend_id[_TENANT_ID] == "backend-1"


async def test_set_default_of_a_disabled_backend_returns_422(
    client: AsyncClient,
    repo: FakeStorageBackendRepository,
) -> None:
    _seed(repo, status=STORAGE_BACKEND_STATUS_DISABLED)

    resp = await client.put("/storage-backends/backend-1/default")

    assert resp.status_code == 422


async def test_set_default_of_a_missing_backend_returns_404(
    client: AsyncClient,
) -> None:
    resp = await client.put("/storage-backends/ghost/default")

    assert resp.status_code == 404


# ── Literal segments are not captured as ids ────────────────────────


async def test_types_route_wins_over_the_id_route(
    client: AsyncClient,
    repo: FakeStorageBackendRepository,
) -> None:
    # No row named "types" exists, so a captured id would 404.
    resp = await client.get("/storage-backends/types")

    assert resp.status_code == 200
    assert isinstance(resp.json()["data"], list)


__all__: list[str] = []
