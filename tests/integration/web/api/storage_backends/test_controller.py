"""Web-layer tests for the storage-backend router.

Exercises the router over HTTP via ``TestClient`` against the
app. The service dependency is overridden with a service backed by an
``AsyncMock(spec=StorageBackendRepository)`` configured with stateful
closures, so the tests exercise the full HTTP path (routing, role
gates, serialization, exception handling) without a database.

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
    STORAGE_BACKEND_SOURCE_ENV,
    STORAGE_BACKEND_SOURCE_USER,
    STORAGE_BACKEND_STATUS_ACTIVE,
    STORAGE_BACKEND_STATUS_DISABLED,
    StorageBackend,
)
from src.web.deps.infra_storage_backends import get_storage_backend_service

# The header auth channel pins the active workspace to 1.
_TENANT_ID = 1


@pytest.fixture(autouse=True)
def _bind_tenant_id_to_admin(
    admin_user: tuple[int, int],
) -> None:
    """Rewrite the module-level ``_TENANT_ID`` to the minted admin tenant.

    Per-test conftest mints a fresh ``tenant_id``; this rebind keeps the
    helper closures (which seed mocks keyed by ``_TENANT_ID``) aligned
    with the principal the authed client presents.
    """
    global _TENANT_ID
    _TENANT_ID = admin_user[1]
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
def repo() -> AsyncMock:
    """``AsyncMock(spec=StorageBackendRepository)`` with stateful closuress."""
    repo = AsyncMock(spec=StorageBackendRepository)
    rows: dict[str, StorageBackend] = {}
    default_backend_id: dict[int, str] = {}
    kb_refs = [0]
    resource_refs = [0]

    def _live_for_tenant(tenant_id: int) -> dict[str, StorageBackend]:
        return {
            bid: r for bid, r in rows.items() if r.tenant_id == tenant_id and r.deleted_at is None
        }

    async def _get_by_id(*, tenant_id: int, id: str) -> StorageBackend | None:
        row = rows.get(id)
        if row is None or row.tenant_id != tenant_id or row.deleted_at is not None:
            return None
        return row

    async def _list_for_tenant(tenant_id: int) -> list[StorageBackend]:
        return sorted(_live_for_tenant(tenant_id).values(), key=lambda r: (r.created_at, r.id))

    async def _find_legacy_alias(*, tenant_id: int, provider: str) -> StorageBackend | None:
        for r in await _list_for_tenant(tenant_id):
            if r.provider == provider and r.legacy_alias:
                return r
        return None

    async def _find_by_name(*, tenant_id: int, name: str) -> StorageBackend | None:
        for r in await _list_for_tenant(tenant_id):
            if r.name == name:
                return r
        return None

    async def _create(row: StorageBackend) -> StorageBackend:
        rows[row.id] = row
        return row

    async def _update_columns(
        *, tenant_id: int, id: str, columns: dict[str, object]
    ) -> StorageBackend | None:
        existing = await _get_by_id(tenant_id=tenant_id, id=id)
        if existing is None:
            return None
        updated = existing.model_copy(update=dict(columns))
        rows[id] = updated
        return updated

    async def _soft_delete(*, tenant_id: int, id: str) -> bool:
        existing = await _get_by_id(tenant_id=tenant_id, id=id)
        if existing is None:
            return False
        now = datetime.now(UTC)
        rows[id] = existing.model_copy(update={"deleted_at": now, "updated_at": now})
        return True

    async def _get_default_backend_id(tenant_id: int) -> str | None:
        return default_backend_id.get(tenant_id)

    async def _set_default_backend_id(*, tenant_id: int, id: str) -> bool:
        default_backend_id[tenant_id] = id
        return True

    async def _count_kb_references(*, tenant_id: int, id: str) -> int:
        return kb_refs[0]

    async def _count_active_resource_references(*, tenant_id: int, id: str) -> int:
        return resource_refs[0]

    repo.get_by_id.side_effect = _get_by_id
    repo.list_for_tenant.side_effect = _list_for_tenant
    repo.find_legacy_alias.side_effect = _find_legacy_alias
    repo.find_by_name.side_effect = _find_by_name
    repo.create.side_effect = _create
    repo.update_columns.side_effect = _update_columns
    repo.soft_delete.side_effect = _soft_delete
    repo.get_default_backend_id.side_effect = _get_default_backend_id
    repo.set_default_backend_id.side_effect = _set_default_backend_id
    repo.count_knowledge_base_references.side_effect = _count_kb_references
    repo.count_active_resource_references.side_effect = _count_active_resource_references
    repo._rows = rows  # type: ignore[attr-defined]
    repo._default_backend_id = default_backend_id  # type: ignore[attr-defined]
    repo._kb_refs = kb_refs  # type: ignore[attr-defined]
    return repo


@pytest.fixture
def app(
    request: pytest.FixtureRequest,
    web_app: FastAPI,
    repo: AsyncMock,
) -> FastAPI:
    """Override ``get_storage_backend_service`` on the shared web app."""
    web_app.dependency_overrides[get_storage_backend_service] = lambda: StorageBackendService(
        backend_repo=repo
    )
    return web_app


@pytest.fixture
def client(app: FastAPI, web_authed_client: TestClient) -> TestClient:
    """Alias ``web_authed_client``; depending on ``app`` forces the
    dep-override fixture to run before the test executes."""
    return web_authed_client


def _seed(
    repo: AsyncMock,
    *,
    id: str = "backend-1",
    tenant_id: int | None = None,
    name: str = "Primary MinIO",
    provider: str = "minio",
    source: str = STORAGE_BACKEND_SOURCE_USER,
    status: str = STORAGE_BACKEND_STATUS_ACTIVE,
    legacy_alias: bool = False,
) -> StorageBackend:
    # ``tenant_id`` default is resolved at call time so the
    # ``_bind_tenant_id_to_admin`` autouse fixture's rebind is honoured.
    if tenant_id is None:
        tenant_id = _TENANT_ID
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
    repo._rows[id] = row  # type: ignore[attr-defined]
    return row


# ── GET /storage-backends/types ─────────────────────────────────────


async def test_list_provider_types_returns_the_allowed_set(client: TestClient) -> None:
    resp = client.get("/storage-backends/types")

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert "local" in body["data"]
    assert "minio" in body["data"]


# ── POST /storage-backends ──────────────────────────────────────────


async def test_create_returns_201_with_the_created_backend(
    client: TestClient,
    repo: AsyncMock,
) -> None:
    resp = client.post(
        "/storage-backends",
        json={"name": "Primary MinIO", "provider": "minio", "config": _MINIO_BODY_CONFIG},
    )

    assert resp.status_code == 201
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["name"] == "Primary MinIO"
    assert body["data"]["provider"] == "minio"
    assert body["data"]["source"] == STORAGE_BACKEND_SOURCE_USER
    rows = repo._rows  # type: ignore[attr-defined]
    assert len(rows) == 1


async def test_create_rejects_a_duplicate_name_with_409(
    client: TestClient,
    repo: AsyncMock,
) -> None:
    _seed(repo, name="Primary MinIO")

    resp = client.post(
        "/storage-backends",
        json={"name": "Primary MinIO", "provider": "minio", "config": _MINIO_BODY_CONFIG},
    )

    assert resp.status_code == 409


async def test_create_rejects_an_unsupported_provider_with_422(
    client: TestClient,
) -> None:
    resp = client.post(
        "/storage-backends",
        json={"name": "Nope", "provider": "dropbox", "config": _MINIO_BODY_CONFIG},
    )

    assert resp.status_code == 422


async def test_create_rejects_a_missing_required_config_field_with_422(
    client: TestClient,
) -> None:
    resp = client.post(
        "/storage-backends",
        json={"name": "Incomplete", "provider": "minio", "config": {"mode": "remote"}},
    )

    assert resp.status_code == 422


# ── GET /storage-backends ───────────────────────────────────────────


async def test_list_masks_credentials_and_reports_the_default(
    client: TestClient,
    repo: AsyncMock,
) -> None:
    _seed(repo, id="backend-1", name="A")
    _seed(repo, id="backend-2", name="B")
    repo._default_backend_id[_TENANT_ID] = "backend-2"  # type: ignore[attr-defined]

    resp = client.get("/storage-backends")

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert [b["id"] for b in body["data"]] == ["backend-1", "backend-2"]
    assert body["default_storage_backend_id"] == "backend-2"
    assert body["data"][0]["config"]["secret_access_key"] == REDACTED_SECRET_PLACEHOLDER


async def test_list_is_empty_for_a_workspace_with_no_backends(
    client: TestClient,
) -> None:
    resp = client.get("/storage-backends")

    assert resp.status_code == 200
    body = resp.json()
    assert body["data"] == []
    assert body["default_storage_backend_id"] is None


async def test_list_excludes_other_workspaces(
    client: TestClient,
    repo: AsyncMock,
) -> None:
    _seed(repo, id="mine")
    _seed(repo, id="theirs", tenant_id=99)

    resp = client.get("/storage-backends")

    assert [b["id"] for b in resp.json()["data"]] == ["mine"]


# ── GET /storage-backends/{id} ──────────────────────────────────────


async def test_get_returns_the_backend_with_masked_credentials(
    client: TestClient,
    repo: AsyncMock,
) -> None:
    _seed(repo)

    resp = client.get("/storage-backends/backend-1")

    assert resp.status_code == 200
    body = resp.json()
    assert body["data"]["id"] == "backend-1"
    assert body["data"]["config"]["access_key_id"] == REDACTED_SECRET_PLACEHOLDER


async def test_get_of_a_missing_backend_returns_404(client: TestClient) -> None:
    resp = client.get("/storage-backends/ghost")

    assert resp.status_code == 404


async def test_get_does_not_cross_workspaces(
    client: TestClient,
    repo: AsyncMock,
) -> None:
    _seed(repo, tenant_id=99)

    resp = client.get("/storage-backends/backend-1")

    assert resp.status_code == 404


# ── PUT /storage-backends/{id} ──────────────────────────────────────


async def test_update_renames_and_preserves_redacted_secrets(
    client: TestClient,
    repo: AsyncMock,
) -> None:
    _seed(repo)
    masked_config = dict(_MINIO_BODY_CONFIG)
    masked_config["access_key_id"] = REDACTED_SECRET_PLACEHOLDER
    masked_config["secret_access_key"] = REDACTED_SECRET_PLACEHOLDER

    resp = client.put(
        "/storage-backends/backend-1",
        json={"name": "Renamed", "config": masked_config},
    )

    assert resp.status_code == 200
    assert resp.json()["data"]["name"] == "Renamed"
    rows = repo._rows  # type: ignore[attr-defined]
    stored = StorageBackendConfigInfo.from_json(rows["backend-1"].config)
    assert stored.secret_access_key == "secret-example"


async def test_update_rejects_a_location_change_with_422(
    client: TestClient,
    repo: AsyncMock,
) -> None:
    _seed(repo)
    moved = dict(_MINIO_BODY_CONFIG)
    moved["bucket_name"] = "elsewhere"

    resp = client.put("/storage-backends/backend-1", json={"config": moved})

    assert resp.status_code == 422


async def test_update_of_an_env_backend_returns_422(
    client: TestClient,
    repo: AsyncMock,
) -> None:
    _seed(repo, source=STORAGE_BACKEND_SOURCE_ENV)

    resp = client.put("/storage-backends/backend-1", json={"name": "Renamed"})

    assert resp.status_code == 422


async def test_update_can_disable_an_unreferenced_backend(
    client: TestClient,
    repo: AsyncMock,
) -> None:
    _seed(repo)

    resp = client.put(
        "/storage-backends/backend-1",
        json={"status": STORAGE_BACKEND_STATUS_DISABLED},
    )

    assert resp.status_code == 200
    assert resp.json()["data"]["status"] == STORAGE_BACKEND_STATUS_DISABLED


async def test_update_of_a_missing_backend_returns_404(client: TestClient) -> None:
    resp = client.put("/storage-backends/ghost", json={"name": "X"})

    assert resp.status_code == 404


# ── DELETE /storage-backends/{id} ───────────────────────────────────


async def test_delete_acknowledges_and_soft_deletes(
    client: TestClient,
    repo: AsyncMock,
) -> None:
    _seed(repo)

    resp = client.delete("/storage-backends/backend-1")

    assert resp.status_code == 200
    assert resp.json() == {"success": True}
    rows = repo._rows  # type: ignore[attr-defined]
    assert rows["backend-1"].deleted_at is not None


async def test_delete_of_the_default_returns_422(
    client: TestClient,
    repo: AsyncMock,
) -> None:
    _seed(repo)
    repo._default_backend_id[_TENANT_ID] = "backend-1"  # type: ignore[attr-defined]

    resp = client.delete("/storage-backends/backend-1")

    assert resp.status_code == 422


async def test_delete_of_a_bound_backend_returns_422(
    client: TestClient,
    repo: AsyncMock,
) -> None:
    _seed(repo)
    repo._kb_refs[0] = 2  # type: ignore[attr-defined]

    resp = client.delete("/storage-backends/backend-1")

    assert resp.status_code == 422


async def test_delete_of_a_missing_backend_returns_404(client: TestClient) -> None:
    resp = client.delete("/storage-backends/ghost")

    assert resp.status_code == 404


# ── POST /storage-backends/test ─────────────────────────────────────


async def test_test_raw_config_returns_success_true(client: TestClient) -> None:
    resp = client.post(
        "/storage-backends/test",
        json={"name": "Probe", "provider": "minio", "config": _MINIO_BODY_CONFIG},
    )

    assert resp.status_code == 200
    assert resp.json()["success"] is True


async def test_test_raw_config_reports_a_probe_failure_as_200(
    client: TestClient,
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

    resp = client.post(
        "/storage-backends/test",
        json={"name": "Probe", "provider": "minio", "config": _MINIO_BODY_CONFIG},
    )

    # A connectivity failure travels as data, not as an HTTP error.
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is False
    assert body["error"] == "bucket unavailable"


async def test_test_raw_config_reports_an_invalid_body_with_200(
    client: TestClient,
) -> None:
    """A validation failure answers 200 with ``success=false`` (Go keeps
    the HTTP status at 200 and reports the error in the body)."""
    resp = client.post(
        "/storage-backends/test",
        json={"name": "Probe", "provider": "minio", "config": {"mode": "remote"}},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is False
    assert body["error"]


# ── POST /storage-backends/{id}/test ────────────────────────────────


async def test_test_saved_backend_returns_success_true(
    client: TestClient,
    repo: AsyncMock,
) -> None:
    _seed(repo)

    resp = client.post("/storage-backends/backend-1/test")

    assert resp.status_code == 200
    assert resp.json()["success"] is True


async def test_test_of_a_missing_backend_returns_404(client: TestClient) -> None:
    resp = client.post("/storage-backends/ghost/test")

    assert resp.status_code == 404


# ── PUT /storage-backends/{id}/default ──────────────────────────────


async def test_set_default_acknowledges_and_points_the_workspace(
    client: TestClient,
    repo: AsyncMock,
) -> None:
    _seed(repo)

    resp = client.put("/storage-backends/backend-1/default")

    assert resp.status_code == 200
    assert resp.json() == {"success": True}
    assert repo._default_backend_id[_TENANT_ID] == "backend-1"  # type: ignore[attr-defined]


async def test_set_default_of_a_disabled_backend_returns_422(
    client: TestClient,
    repo: AsyncMock,
) -> None:
    _seed(repo, status=STORAGE_BACKEND_STATUS_DISABLED)

    resp = client.put("/storage-backends/backend-1/default")

    assert resp.status_code == 422


async def test_set_default_of_a_missing_backend_returns_404(
    client: TestClient,
) -> None:
    resp = client.put("/storage-backends/ghost/default")

    assert resp.status_code == 404


# ── Literal segments are not captured as ids ────────────────────────


async def test_types_route_wins_over_the_id_route(
    client: TestClient,
    repo: AsyncMock,
) -> None:
    # No row named "types" exists, so a captured id would 404.
    resp = client.get("/storage-backends/types")

    assert resp.status_code == 200
    assert isinstance(resp.json()["data"], list)


__all__: list[str] = []
