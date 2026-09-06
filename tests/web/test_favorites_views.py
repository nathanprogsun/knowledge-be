"""Unit tests for the user-favorites and system-service HTTP endpoints.

The favorites endpoints are tested against a minimal FastAPI app that
mounts only the favorites router; the service layer is replaced with a
fake so the tests exercise the request handling (principal resolution,
wire-shape envelopes, error mapping) without a database.

The system-service endpoints are tested with the service_views router
mounted the same way — they read configuration from the environment
rather than the DB, so no service override is needed.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from src.common.exception import ValidationError
from src.db.models.user_resource_favorite import UserResourceFavorite
from src.web.api.favorites.router import router as favorites_router
from src.web.api.system.service_views import router as system_service_router
from src.web.deps.context import get_tenant_id_dep, get_user_id_dep
from src.web.deps.system import get_favorite_service, get_system_info_service
from src.web.exception_handler import register_exception_handlers
from src.web.middleware.auth import require_auth

# ── Helpers ──────────────────────────────────────────────────────────


def _favorite_row(
    *,
    resource_type: str,
    resource_id: str,
    created_at: datetime | None = None,
) -> UserResourceFavorite:
    """Build a storage row the fake service returns."""
    return UserResourceFavorite(
        user_id="u-1",
        tenant_id=7,
        resource_type=resource_type,
        resource_id=resource_id,
        created_at=created_at or datetime(2026, 8, 1, tzinfo=UTC),
    )


class _FakeFavoriteService:
    """In-memory fake of ``UserResourceFavoriteService``."""

    def __init__(self) -> None:
        self.rows: list[UserResourceFavorite] = []
        self.added: list[tuple[str, str, str]] = []
        self.removed: list[tuple[str, str, str]] = []
        self.list_calls: list[str] = []
        self.raise_on_list: ValidationError | None = None

    async def list_favorites(
        self,
        *,
        user_id: str,
        tenant_id: int,
        resource_type: str,
    ) -> list[UserResourceFavorite]:
        self.list_calls.append(resource_type)
        if self.raise_on_list is not None:
            raise self.raise_on_list
        return [r for r in self.rows if r.resource_type == resource_type]

    async def add_favorite(
        self,
        *,
        user_id: str,
        tenant_id: int,
        resource_type: str,
        resource_id: str,
    ) -> UserResourceFavorite | None:
        self.added.append((resource_type, resource_id, str(tenant_id)))
        return None

    async def remove_favorite(
        self,
        *,
        user_id: str,
        tenant_id: int,
        resource_type: str,
        resource_id: str,
    ) -> bool:
        self.removed.append((resource_type, resource_id, str(tenant_id)))
        return True


@pytest.fixture
def favorite_service() -> _FakeFavoriteService:
    return _FakeFavoriteService()


@pytest.fixture
def favorites_client(
    favorite_service: _FakeFavoriteService,
) -> TestClient:
    """A minimal app with the favorites router and a fake service.

    ``require_auth`` is replaced with a no-op and the principal deps are
    pinned to a fixed (user, tenant) pair so the tests exercise the view
    logic without touching the auth middleware or a database.
    """
    app = FastAPI()
    app.include_router(favorites_router, prefix="/api/v1")
    register_exception_handlers(app)
    app.dependency_overrides[require_auth] = lambda: None
    app.dependency_overrides[get_tenant_id_dep] = lambda: 7
    app.dependency_overrides[get_user_id_dep] = lambda: "u-1"
    app.dependency_overrides[get_favorite_service] = lambda: favorite_service
    return TestClient(app)


@pytest.fixture
def system_client() -> TestClient:
    """A minimal app with the system routers (info + service + admin info).

    The endpoints that read only env or static config need no overrides;
    the ones that touch the session or lifespan get explicit fakes so
    the test stays a pure unit test.
    """
    from datetime import UTC, datetime

    from src.core.contracts.system import SystemInfo
    from src.core.system.info_service import SystemInfoSnapshot
    from src.web.api.system.router import info_router

    class _FakeInfoService:
        async def get_info(self) -> SystemInfoSnapshot:
            return SystemInfoSnapshot(
                info=SystemInfo(
                    version="1.0.0",
                    edition="standard",
                    commit_id="abc",
                    build_time=datetime(2026, 1, 1, tzinfo=UTC),
                    go_version="go1.22",
                    keyword_index_engine="未配置",
                    vector_store_engine="未配置",
                    graph_database_engine="未配置",
                    minio_enabled=False,
                    db_version="0000",
                ),
                db_migration_error=None,
                started_at=datetime(2026, 1, 1, tzinfo=UTC),
                uptime_seconds=42,
            )

    app = FastAPI()
    app.state.started_at = datetime(2026, 1, 1, tzinfo=UTC)
    app.include_router(info_router, prefix="/api/v1")
    app.include_router(system_service_router, prefix="/api/v1")
    register_exception_handlers(app)

    def _fake_admin_auth(request: Request) -> None:
        request.state.tenant_id = "7"
        request.state.tenant_role = "admin"
        return

    app.dependency_overrides[require_auth] = _fake_admin_auth
    app.dependency_overrides[get_system_info_service] = lambda: _FakeInfoService()
    return TestClient(app)


# ── Favorites: list ──────────────────────────────────────────────────


def test_list_favorites_returns_envelope(favorites_client: TestClient) -> None:
    favorite_service = favorites_client.app.dependency_overrides[get_favorite_service]()
    favorite_service.rows = [
        _favorite_row(resource_type="kb", resource_id="kb-1"),
        _favorite_row(resource_type="kb", resource_id="kb-2"),
    ]

    response = favorites_client.get("/api/v1/user/favorites", params={"type": "kb"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert [entry["resource_id"] for entry in payload["data"]] == ["kb-1", "kb-2"]
    assert payload["data"][0]["user_id"] == "u-1"
    assert payload["data"][0]["tenant_id"] == 7
    assert payload["data"][0]["resource_type"] == "kb"
    assert favorite_service.list_calls == ["kb"]


def test_list_favorites_empty_when_none_starred(favorites_client: TestClient) -> None:
    response = favorites_client.get("/api/v1/user/favorites", params={"type": "agent"})

    assert response.status_code == 200
    assert response.json() == {"success": True, "data": []}


def test_list_favorites_invalid_type_is_422(favorites_client: TestClient) -> None:
    favorite_service = favorites_client.app.dependency_overrides[get_favorite_service]()
    favorite_service.raise_on_list = ValidationError(
        code="favorite.invalid_type",
        message="invalid favorite resource type 'wiki'",
    )

    response = favorites_client.get("/api/v1/user/favorites", params={"type": "wiki"})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "favorite.invalid_type"


def test_list_favorites_requires_type_query(favorites_client: TestClient) -> None:
    response = favorites_client.get("/api/v1/user/favorites")

    assert response.status_code == 422


# ── Favorites: add / remove ──────────────────────────────────────────


def test_add_favorite_returns_success(favorites_client: TestClient) -> None:
    favorite_service = favorites_client.app.dependency_overrides[get_favorite_service]()

    response = favorites_client.post(
        "/api/v1/user/favorites",
        json={"type": "kb", "id": "kb-42"},
    )

    assert response.status_code == 200
    assert response.json() == {"success": True}
    assert favorite_service.added == [("kb", "kb-42", "7")]


def test_add_favorite_rejects_missing_id(favorites_client: TestClient) -> None:
    response = favorites_client.post(
        "/api/v1/user/favorites",
        json={"type": "kb"},
    )

    assert response.status_code == 422


def test_remove_favorite_returns_success(favorites_client: TestClient) -> None:
    favorite_service = favorites_client.app.dependency_overrides[get_favorite_service]()

    response = favorites_client.delete("/api/v1/user/favorites/kb/kb-42")

    assert response.status_code == 200
    assert response.json() == {"success": True}
    assert favorite_service.removed == [("kb", "kb-42", "7")]


# ── Favorites: principal resolution ──────────────────────────────────


def test_favorites_missing_principal_is_401(favorites_client: TestClient) -> None:
    """A request without a resolved user context is rejected.

    The fixture pins ``get_user_id_dep`` to ``"u-1"``; overriding it to
    ``None`` simulates a request that reached the handler without an
    authenticated principal.
    """
    favorite_service = favorites_client.app.dependency_overrides[get_favorite_service]()
    favorites_client.app.dependency_overrides[get_user_id_dep] = lambda: None

    response = favorites_client.get("/api/v1/user/favorites", params={"type": "kb"})

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "auth.principal_context_missing"
    assert favorite_service.list_calls == []


# ── System service: info ─────────────────────────────────────────────


def test_get_system_info_returns_envelope(system_client: TestClient) -> None:
    response = system_client.get("/api/v1/system/info")

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    data = payload["data"]
    assert isinstance(data["version"], str)
    assert isinstance(data["edition"], str)
    assert isinstance(data["minio_enabled"], bool)
    assert isinstance(data["uptime_seconds"], int)
    assert data["started_at"].endswith("Z")


# ── System service: parser engines ───────────────────────────────────


def test_list_parser_engines_returns_registry(system_client: TestClient) -> None:
    response = system_client.get("/api/v1/system/parser-engines")

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert isinstance(payload["connected"], bool)
    assert isinstance(payload["docreader_addr"], str)
    names = [engine["Name"] for engine in payload["data"]]
    assert "builtin" in names
    assert "simple" in names


def test_check_parser_engines_returns_registry(system_client: TestClient) -> None:
    response = system_client.post(
        "/api/v1/system/parser-engines/check",
        json={"addr": "http://docreader:8000"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["docreader_addr"] == "http://docreader:8000"
    assert any(engine["Name"] == "builtin" for engine in payload["data"])


# ── System service: docreader reconnect ──────────────────────────────


def test_reconnect_docreader_accepts_valid_addr(
    system_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SSRF_WHITELIST", "docreader.internal")

    response = system_client.post(
        "/api/v1/system/docreader/reconnect",
        json={"addr": "http://docreader.internal:8000"},
    )

    assert response.status_code == 200
    assert response.json() == {"success": True}


def test_reconnect_docreader_rejects_blank_addr(system_client: TestClient) -> None:
    response = system_client.post(
        "/api/v1/system/docreader/reconnect",
        json={"addr": "  "},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "system.docreader_addr_empty"


def test_reconnect_docreader_rejects_unverifiable_host(
    system_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SSRF_WHITELIST", "")
    monkeypatch.delenv("SSRF_WHITELIST", raising=False)

    response = system_client.post(
        "/api/v1/system/docreader/reconnect",
        json={"addr": "http://docreader.internal:8000"},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "system.docreader_ssrf_blocked"


# ── System service: storage engine ───────────────────────────────────


def test_get_storage_engine_status_returns_providers(system_client: TestClient) -> None:
    response = system_client.get("/api/v1/system/storage-engine-status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    data = payload["data"]
    engine_names = [engine["name"] for engine in data["engines"]]
    assert "local" in engine_names
    assert "minio" in engine_names
    assert data["minio_env_available"] is False


def test_check_storage_engine_accepts_supported_provider(system_client: TestClient) -> None:
    response = system_client.post(
        "/api/v1/system/storage-engine-check",
        json={"provider": "minio", "config": {"endpoint": "localhost:9000"}},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["data"]["ok"] is False


def test_check_storage_engine_rejects_unknown_provider(system_client: TestClient) -> None:
    response = system_client.post(
        "/api/v1/system/storage-engine-check",
        json={"provider": "s3-private", "config": {}},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "system.unknown_storage_provider"


# ── Service validation (real service, fake repo) ─────────────────────


class _RecordingFavoriteRepo:
    """Repository stub that records calls for the service under test."""

    def __init__(self) -> None:
        self.add_calls: list[tuple[str, int, str, str]] = []
        self.remove_calls: list[tuple[str, int, str, str]] = []
        self.list_calls: list[tuple[str, int, str]] = []

    async def add(
        self,
        *,
        user_id: str,
        tenant_id: int,
        resource_type: str,
        resource_id: str,
    ) -> UserResourceFavorite | None:
        self.add_calls.append((user_id, tenant_id, resource_type, resource_id))
        return None

    async def remove(
        self,
        *,
        user_id: str,
        tenant_id: int,
        resource_type: str,
        resource_id: str,
    ) -> bool:
        self.remove_calls.append((user_id, tenant_id, resource_type, resource_id))
        return True

    async def list_by_user(
        self,
        *,
        user_id: str,
        tenant_id: int,
        resource_type: str,
    ) -> list[UserResourceFavorite]:
        self.list_calls.append((user_id, tenant_id, resource_type))
        return []

    async def is_favorite(
        self,
        *,
        user_id: str,
        tenant_id: int,
        resource_type: str,
        resource_id: str,
    ) -> bool:
        return False


@pytest.fixture
def service_with_repo() -> tuple[Any, _RecordingFavoriteRepo]:
    """A real ``UserResourceFavoriteService`` over a recording repo."""
    from src.core.system.favorite_service import UserResourceFavoriteService

    repo = _RecordingFavoriteRepo()
    return UserResourceFavoriteService(repo=repo), repo


async def test_service_rejects_invalid_type(
    service_with_repo: tuple[Any, _RecordingFavoriteRepo],
) -> None:
    service, repo = service_with_repo

    with pytest.raises(ValidationError) as excinfo:
        await service.list_favorites(user_id="u-1", tenant_id=7, resource_type="wiki")

    assert excinfo.value.code == "favorite.invalid_type"
    assert repo.list_calls == []


async def test_service_rejects_empty_resource_id(
    service_with_repo: tuple[Any, _RecordingFavoriteRepo],
) -> None:
    service, repo = service_with_repo

    with pytest.raises(ValidationError) as excinfo:
        await service.add_favorite(
            user_id="u-1",
            tenant_id=7,
            resource_type="kb",
            resource_id="   ",
        )

    assert excinfo.value.code == "favorite.empty_id"
    assert repo.add_calls == []


async def test_service_forwards_valid_add(
    service_with_repo: tuple[Any, _RecordingFavoriteRepo],
) -> None:
    service, repo = service_with_repo

    result = await service.add_favorite(
        user_id="u-1",
        tenant_id=7,
        resource_type="kb",
        resource_id="kb-42",
    )

    assert result is None
    assert repo.add_calls == [("u-1", 7, "kb", "kb-42")]


__all__: list[Any] = []
