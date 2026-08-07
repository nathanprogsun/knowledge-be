"""Tests for the retrieval engine registry and the env-driven registry.

Covers the two lookup maps (env-store ``by_engine_type`` and DB-store
``by_store_id``), generation-based stale-build rejection, the failure
cooldown, the single-flight collapse of on-demand engine rebuilds, and the
``RETRIEVE_DRIVER`` registration loop. No vector database is contacted —
the store repository, the engine factory, and the per-driver repository
builders are faked.
"""

from __future__ import annotations

import asyncio
from typing import cast

import pytest

from src.ai.embedding import TaskContext
from src.ai.retrieval import env_registry
from src.ai.retrieval.base import (
    Context,
    EngineFactory,
    RetrieveEngineRepository,
    RetrieveEngineService,
    VectorStoreRepositoryLike,
)
from src.ai.retrieval.env_registry import (
    init_retrieve_engine_registry,
    load_db_stores_into_registry,
    parse_retrieve_driver,
)
from src.ai.retrieval.kv_hybrid import KVHybridRetrieveEngine
from src.ai.retrieval.registry import (
    RetrieveEngineRegistry,
    VectorStoreNotFoundError,
    VectorStoreUnavailableError,
    new_retrieve_engine_registry,
)
from src.ai.retrieval.types import RetrieverEngineType, VectorStore, VectorStoreLike

_CTX = TaskContext()


class _DB:
    """Stand-in for the opaque database handle."""


class _Cfg:
    """Stand-in for the opaque application config."""


class _FakeService:
    """Minimal engine service satisfying the registry's structural needs."""

    def __init__(self, engine_type: RetrieverEngineType, name: str = "") -> None:
        self._engine_type = engine_type
        self.name = name

    def engine_type(self) -> RetrieverEngineType:
        return self._engine_type


class _FakeEngineRepo:
    """Stand-in for a concrete engine repository built by the env path."""


class _FakeRepo:
    """Store repository returning a fixed store (or ``None``)."""

    def __init__(self, store: VectorStoreLike | None) -> None:
        self._store = store
        self.calls: list[tuple[int, str]] = []

    async def get_by_id(
        self, _ctx: Context, tenant_id: int, store_id: str
    ) -> VectorStoreLike | None:
        self.calls.append((tenant_id, store_id))
        return self._store


def _store(
    store_id: str = "store-1",
    engine_type: RetrieverEngineType = RetrieverEngineType.QDRANT,
    name: str = "Store",
) -> VectorStore:
    return VectorStore(
        id=store_id,
        tenant_id=1,
        name=name,
        engine_type=engine_type,
    )


def _svc(engine_type: RetrieverEngineType = RetrieverEngineType.QDRANT) -> RetrieveEngineService:
    return cast("RetrieveEngineService", _FakeService(engine_type))


async def _fake_factory(_ctx: Context, _store: VectorStoreLike) -> RetrieveEngineService:
    return cast("RetrieveEngineService", _FakeService(RetrieverEngineType.QDRANT, name="built"))


def _registry(
    repo: VectorStoreRepositoryLike | None = None,
    factory: EngineFactory | None = None,
) -> RetrieveEngineRegistry:
    return new_retrieve_engine_registry(repo, factory)


# ── env-store (byEngineType) API ────────────────────────────────────


def test_register_and_get_by_engine_type() -> None:
    registry = _registry()
    registry.register(_svc(RetrieverEngineType.POSTGRES))
    service = registry.get_retrieve_engine_service(RetrieverEngineType.POSTGRES)
    assert service.engine_type() == RetrieverEngineType.POSTGRES


def test_get_all_returns_only_env_engines() -> None:
    registry = _registry()
    registry.register(_svc(RetrieverEngineType.POSTGRES))
    registry.register(_svc(RetrieverEngineType.QDRANT))
    assert len(registry.get_all_retrieve_engine_services()) == 2


def test_register_duplicate_engine_type_raises_conflict() -> None:
    registry = _registry()
    registry.register(_svc(RetrieverEngineType.POSTGRES))
    with pytest.raises(Exception, match="already registered"):
        registry.register(_svc(RetrieverEngineType.POSTGRES))


def test_get_missing_engine_type_raises_not_found() -> None:
    registry = _registry()
    with pytest.raises(Exception, match="not found"):
        registry.get_retrieve_engine_service(RetrieverEngineType.MILVUS)


# ── DB-store (byStoreID) API ────────────────────────────────────────


def test_register_and_get_by_store_id() -> None:
    registry = _registry()
    service = _svc()
    registry.register_with_store_id("store-1", service)
    assert registry.get_by_store_id("store-1") is service


def test_get_missing_store_raises_not_found() -> None:
    registry = _registry()
    with pytest.raises(Exception, match="store store-9 not found"):
        registry.get_by_store_id("store-9")


def test_register_with_store_id_upserts() -> None:
    registry = _registry()
    first = _svc()
    second = _svc()
    registry.register_with_store_id("store-1", first)
    registry.register_with_store_id("store-1", second)
    assert registry.get_by_store_id("store-1") is second


def test_unregister_by_store_id_is_idempotent() -> None:
    registry = _registry()
    registry.register_with_store_id("store-1", _svc())
    registry.unregister_by_store_id("store-1")
    registry.unregister_by_store_id("store-1")  # no-op
    with pytest.raises(Exception, match="not found"):
        registry.get_by_store_id("store-1")


# ── rebuild capability ──────────────────────────────────────────────


def test_can_rebuild_stores_requires_repo_and_factory() -> None:
    assert _registry().can_rebuild_stores() is False
    assert (
        _registry(repo=cast("VectorStoreRepositoryLike", _FakeRepo(None))).can_rebuild_stores()
        is False
    )
    assert _registry(factory=_fake_factory).can_rebuild_stores() is False
    assert (
        _registry(
            repo=cast("VectorStoreRepositoryLike", _FakeRepo(None)), factory=_fake_factory
        ).can_rebuild_stores()
        is True
    )


# ── on-demand rebuild ───────────────────────────────────────────────


async def test_get_or_load_returns_registered_store() -> None:
    repo = _FakeRepo(_store())
    registry = _registry(repo=cast("VectorStoreRepositoryLike", repo), factory=_fake_factory)
    expected = _svc()
    registry.register_with_store_id("store-1", expected)
    service = await registry.get_or_load_by_store_id(_CTX, 1, "store-1")
    assert service is expected
    assert repo.calls == []  # never touched the database


async def test_get_or_load_without_repo_or_factory_raises_not_found() -> None:
    registry = _registry()
    with pytest.raises(VectorStoreNotFoundError):
        await registry.get_or_load_by_store_id(_CTX, 1, "store-1")


async def test_get_or_load_rebuilds_from_database() -> None:
    repo = _FakeRepo(_store())
    registry = _registry(repo=cast("VectorStoreRepositoryLike", repo), factory=_fake_factory)
    service = await registry.get_or_load_by_store_id(_CTX, 1, "store-1")
    assert service.engine_type() == RetrieverEngineType.QDRANT
    assert repo.calls == [(1, "store-1")]
    # Now registered; a followup lookup hits the map.
    assert registry.get_by_store_id("store-1").engine_type() == RetrieverEngineType.QDRANT


async def test_get_or_load_missing_store_raises_not_found() -> None:
    repo = _FakeRepo(None)
    registry = _registry(repo=cast("VectorStoreRepositoryLike", repo), factory=_fake_factory)
    with pytest.raises(VectorStoreNotFoundError):
        await registry.get_or_load_by_store_id(_CTX, 1, "store-1")
    # A missing store is non-retryable: the failure cooldown is NOT armed.
    assert not registry._in_failure_cooldown("store-1")


async def test_get_or_load_factory_failure_raises_and_arms_cooldown() -> None:
    repo = _FakeRepo(_store())

    async def _boom_factory(_ctx: Context, _store: VectorStoreLike) -> RetrieveEngineService:
        raise RuntimeError("backend down")

    registry = _registry(repo=cast("VectorStoreRepositoryLike", repo), factory=_boom_factory)
    with pytest.raises(VectorStoreUnavailableError):
        await registry.get_or_load_by_store_id(_CTX, 1, "store-1")
    assert registry._in_failure_cooldown("store-1")
    # Within the cooldown a retry is short-circuited and does not rebuild.
    with pytest.raises(VectorStoreUnavailableError):
        await registry.get_or_load_by_store_id(_CTX, 1, "store-1")
    assert len(repo.calls) == 1


async def test_stale_build_is_not_published_after_generation_bump() -> None:
    repo = _FakeRepo(_store())
    started = asyncio.Event()
    release = asyncio.Event()

    async def _slow_factory(_ctx: Context, _store: VectorStoreLike) -> RetrieveEngineService:
        started.set()
        await release.wait()
        return cast("RetrieveEngineService", _FakeService(RetrieverEngineType.QDRANT, name="stale"))

    registry = _registry(repo=cast("VectorStoreRepositoryLike", repo), factory=_slow_factory)
    build = asyncio.create_task(registry.get_or_load_by_store_id(_CTX, 1, "store-1"))
    await started.wait()
    # A concurrent registration bumps the generation for the same store.
    authoritative = _svc()
    registry.register_with_store_id("store-1", authoritative)
    release.set()
    with pytest.raises(VectorStoreUnavailableError):
        await build
    # The authoritative engine is what stays registered.
    assert registry.get_by_store_id("store-1") is authoritative


async def test_concurrent_callers_collapse_onto_one_build() -> None:
    repo = _FakeRepo(_store())
    started = asyncio.Event()
    release = asyncio.Event()
    build_count = 0

    async def _slow_factory(_ctx: Context, _store: VectorStoreLike) -> RetrieveEngineService:
        nonlocal build_count
        build_count += 1
        started.set()
        await release.wait()
        return cast("RetrieveEngineService", _FakeService(RetrieverEngineType.QDRANT, name="built"))

    registry = _registry(repo=cast("VectorStoreRepositoryLike", repo), factory=_slow_factory)
    first = asyncio.create_task(registry.get_or_load_by_store_id(_CTX, 1, "store-1"))
    await started.wait()
    second = asyncio.create_task(registry.get_or_load_by_store_id(_CTX, 1, "store-1"))
    await asyncio.sleep(0)  # let the second caller attach to the flight
    release.set()
    svc_first = await first
    svc_second = await second
    assert svc_first is svc_second
    assert build_count == 1
    assert len(repo.calls) == 1


async def test_get_or_load_scopes_flight_by_tenant() -> None:
    repo = _FakeRepo(_store())
    registry = _registry(repo=cast("VectorStoreRepositoryLike", repo), factory=_fake_factory)
    await registry.get_or_load_by_store_id(_CTX, 1, "store-1")
    assert repo.calls == [(1, "store-1")]


# ── env-driven registry (RETRIEVE_DRIVER) ───────────────────────────


def test_parse_retrieve_driver_trims_and_skips_empty() -> None:
    assert parse_retrieve_driver("") == []
    assert parse_retrieve_driver("   ") == []
    assert parse_retrieve_driver("postgres, sqlite ,,elasticsearch_v8") == [
        "postgres",
        "sqlite",
        "elasticsearch_v8",
    ]


async def _fake_engine_repo(
    _driver: str = "",
    _db: object | None = None,
    _cfg: object | None = None,
    _audit_sink: object | None = None,
) -> RetrieveEngineRepository:
    return cast("RetrieveEngineRepository", _FakeEngineRepo())


async def test_init_registry_without_drivers_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RETRIEVE_DRIVER", raising=False)
    registry = await init_retrieve_engine_registry(_DB(), _Cfg())
    assert registry.get_all_retrieve_engine_services() == []


async def test_init_registry_registers_driver_engines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RETRIEVE_DRIVER", "postgres,sqlite")
    monkeypatch.setattr(env_registry, "_build_driver_repository", _fake_engine_repo)
    registry = await init_retrieve_engine_registry(_DB(), _Cfg())
    assert len(registry.get_all_retrieve_engine_services()) == 2

    postgres = registry.get_retrieve_engine_service(RetrieverEngineType.POSTGRES)
    assert isinstance(postgres, KVHybridRetrieveEngine)
    assert postgres.engine_type() == RetrieverEngineType.POSTGRES
    assert isinstance(postgres._index_repository, _FakeEngineRepo)

    sqlite = registry.get_retrieve_engine_service(RetrieverEngineType.SQLITE)
    assert isinstance(sqlite, KVHybridRetrieveEngine)
    assert sqlite.engine_type() == RetrieverEngineType.SQLITE


async def test_init_registry_skips_driver_build_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RETRIEVE_DRIVER", "postgres,qdrant")

    async def _flaky_build(
        _driver: str, _db: object, _cfg: object, _audit_sink: object | None
    ) -> RetrieveEngineRepository:
        if _driver == "qdrant":
            raise RuntimeError("backend down")
        return cast("RetrieveEngineRepository", _FakeEngineRepo())

    monkeypatch.setattr(env_registry, "_build_driver_repository", _flaky_build)
    registry = await init_retrieve_engine_registry(_DB(), _Cfg())
    assert registry.get_retrieve_engine_service(RetrieverEngineType.POSTGRES)
    with pytest.raises(Exception, match="not found"):
        registry.get_retrieve_engine_service(RetrieverEngineType.QDRANT)


async def test_init_registry_conflict_on_shared_engine_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Both ES drivers register the same engine type; the duplicate
    # registration is logged and the first wins.
    monkeypatch.setenv("RETRIEVE_DRIVER", "elasticsearch_v8,elasticsearch_v7")
    monkeypatch.setattr(env_registry, "_build_driver_repository", _fake_engine_repo)
    registry = await init_retrieve_engine_registry(_DB(), _Cfg())
    engine = registry.get_retrieve_engine_service(RetrieverEngineType.ELASTICSEARCH)
    assert engine.engine_type() == RetrieverEngineType.ELASTICSEARCH


async def test_init_registry_loads_db_stores(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RETRIEVE_DRIVER", raising=False)
    sentinel = cast("RetrieveEngineService", _FakeService(RetrieverEngineType.QDRANT))

    async def _fake_create(
        _ctx: object,
        _store: VectorStoreLike,
        _db: object,
        _cfg: object,
        _audit_sink: object | None = None,
    ) -> RetrieveEngineService:
        return sentinel

    monkeypatch.setattr(env_registry, "create_engine_service_from_store", _fake_create)

    async def _loader() -> list[VectorStore]:
        return [_store("store-1"), _store("store-2")]

    registry = await init_retrieve_engine_registry(_DB(), _Cfg(), store_loader=_loader)
    assert registry.get_by_store_id("store-1") is sentinel
    assert registry.get_by_store_id("store-2") is sentinel


async def test_load_db_stores_skips_failed_engines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _registry()

    async def _fake_create(
        _ctx: object,
        store: VectorStoreLike,
        _db: object,
        _cfg: object,
        _audit_sink: object | None = None,
    ) -> RetrieveEngineService:
        if store.id == "bad":
            raise RuntimeError("boom")
        return cast("RetrieveEngineService", _FakeService(RetrieverEngineType.QDRANT))

    monkeypatch.setattr(env_registry, "create_engine_service_from_store", _fake_create)
    stores = [_store("good"), _store("bad")]
    await load_db_stores_into_registry(registry, stores, _DB(), _Cfg())
    assert registry.get_by_store_id("good")
    with pytest.raises(Exception, match="not found"):
        registry.get_by_store_id("bad")


def test_env_helpers_default_and_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QDRANT_PORT", raising=False)
    assert env_registry._env_int("QDRANT_PORT", 6334) == 6334
    monkeypatch.setenv("QDRANT_PORT", "7000")
    assert env_registry._env_int("QDRANT_PORT", 6334) == 7000
    monkeypatch.setenv("QDRANT_PORT", "not-a-number")
    assert env_registry._env_int("QDRANT_PORT", 6334) == 6334

    monkeypatch.delenv("QDRANT_USE_TLS", raising=False)
    assert env_registry._env_tls_enabled("QDRANT_USE_TLS") is False
    monkeypatch.setenv("QDRANT_USE_TLS", "true")
    assert env_registry._env_tls_enabled("QDRANT_USE_TLS") is True
    monkeypatch.setenv("QDRANT_USE_TLS", "0")
    assert env_registry._env_tls_enabled("QDRANT_USE_TLS") is False


def test_env_weaviate_api_key_requires_auth_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WEAVIATE_AUTH_ENABLED", raising=False)
    monkeypatch.delenv("WEAVIATE_API_KEY", raising=False)
    assert env_registry._env_weaviate_api_key() == ""
    monkeypatch.setenv("WEAVIATE_AUTH_ENABLED", "true")
    monkeypatch.setenv("WEAVIATE_API_KEY", "  secret  ")
    assert env_registry._env_weaviate_api_key() == "secret"
    monkeypatch.setenv("WEAVIATE_AUTH_ENABLED", "false")
    assert env_registry._env_weaviate_api_key() == ""
