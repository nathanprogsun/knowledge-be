"""Tests for the retrieval engine factory.

The concrete engine repositories are delivered by followup PRs; the factory
tests pin the switch routing, the ES v7/v8 version detection, the
per-engine config defaults, and the requirement that every engine path
wraps its repository in a ``KVHybridRetrieveEngine``. Repository
constructors are monkeypatched with fakes — no vector database is
contacted.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest

from src.ai.embedding import TaskContext
from src.ai.retrieval import factory as factory_module
from src.ai.retrieval.base import RetrieveEngineRepository, RetrieveEngineService
from src.ai.retrieval.factory import (
    WeaviateClientConfig,
    build_milvus_client_config,
    create_engine_service_from_store,
    host_from_addr,
    is_es_v7,
    new_engine_factory,
)
from src.ai.retrieval.kv_hybrid import KVHybridRetrieveEngine
from src.ai.retrieval.types import (
    ConnectionConfig,
    IndexConfig,
    RetrieverEngineType,
    VectorStore,
    VectorStoreLike,
)
from src.common.exception import ValidationError

_CTX = TaskContext()


class _DB:
    """Stand-in for the opaque database handle."""


class _Cfg:
    """Stand-in for the opaque application config."""


class _FakeService:
    def __init__(self, name: str = "svc") -> None:
        self.name = name

    def engine_type(self) -> RetrieverEngineType:
        return RetrieverEngineType.POSTGRES


class _FakeRepo:
    """Stand-in for a future engine repository."""


def _store(
    engine_type: RetrieverEngineType,
    cc: ConnectionConfig | None = None,
    index_config: IndexConfig | None = None,
    store_id: str = "store-1",
) -> VectorStore:
    return VectorStore(
        id=store_id,
        tenant_id=1,
        name="Store",
        engine_type=engine_type,
        connection_config=cc or ConnectionConfig(),
        index_config=index_config or IndexConfig(),
    )


def _fake_repo() -> RetrieveEngineRepository:
    return cast("RetrieveEngineRepository", _FakeRepo())


# ── pure helpers ────────────────────────────────────────────────────


def test_is_es_v7_detects_7x_only() -> None:
    assert is_es_v7("7.10.1") is True
    assert is_es_v7("7.0") is True
    assert is_es_v7("8.0.0") is False
    assert is_es_v7("6.8") is False
    assert is_es_v7("") is False


def test_host_from_addr_splits_last_colon() -> None:
    assert host_from_addr("doris-fe:9030") == "doris-fe"
    assert host_from_addr("localhost:19530") == "localhost"
    assert host_from_addr("http://host:9200") == "http://host"
    assert host_from_addr("localhost") == "localhost"


def test_build_milvus_client_config_defaults() -> None:
    cfg = build_milvus_client_config(ConnectionConfig())
    assert cfg.address == "localhost:19530"
    assert cfg.username == ""
    assert cfg.password == ""
    assert cfg.db_name == ""
    assert cfg.dial_timeout_seconds == 5


def test_build_milvus_client_config_uses_connection_fields() -> None:
    cfg = build_milvus_client_config(
        ConnectionConfig(addr="milvus-host:19530", username="root", password="pw", database="db1")
    )
    assert cfg.address == "milvus-host:19530"
    assert cfg.username == "root"
    assert cfg.password == "pw"
    assert cfg.db_name == "db1"


# ── switch routing ──────────────────────────────────────────────────

_ROUTES: list[tuple[RetrieverEngineType, str]] = [
    (RetrieverEngineType.POSTGRES, "_create_postgres_engine"),
    (RetrieverEngineType.ELASTICSEARCH, "_create_elasticsearch_engine"),
    (RetrieverEngineType.QDRANT, "_create_qdrant_engine"),
    (RetrieverEngineType.MILVUS, "_create_milvus_engine"),
    (RetrieverEngineType.WEAVIATE, "_create_weaviate_engine"),
    (RetrieverEngineType.DORIS, "_create_doris_engine"),
    (RetrieverEngineType.SQLITE, "_create_sqlite_engine"),
    (RetrieverEngineType.TENCENT_VECTORDB, "_create_tencent_vectordb_engine"),
    (RetrieverEngineType.OPENSEARCH, "_create_opensearch_engine"),
]


@pytest.mark.parametrize(("engine_type", "target"), _ROUTES)
async def test_create_engine_service_routes_each_engine_type(
    monkeypatch: pytest.MonkeyPatch,
    engine_type: RetrieverEngineType,
    target: str,
) -> None:
    sentinel = cast("RetrieveEngineService", _FakeService(name=target))
    monkeypatch.setattr(factory_module, target, AsyncMock(return_value=sentinel))
    store = _store(engine_type)
    engine = await create_engine_service_from_store(_CTX, store, _DB(), _Cfg())
    assert engine is sentinel


async def test_create_engine_service_unknown_engine_raises() -> None:
    store = cast("VectorStoreLike", SimpleNamespace(engine_type="unknown"))
    with pytest.raises(ValidationError, match="unsupported engine type: unknown"):
        await create_engine_service_from_store(_CTX, store, _DB(), _Cfg())


async def test_create_engine_service_rejects_legacy_engine_types() -> None:
    for legacy in (RetrieverEngineType.INFINITY, RetrieverEngineType.ELASTICFAISS):
        store = _store(legacy)
        with pytest.raises(ValidationError, match="unsupported engine type"):
            await create_engine_service_from_store(_CTX, store, _DB(), _Cfg())


# ── per-engine builders wrap the repository in KVHybrid ─────────────


async def test_create_postgres_engine_requires_default_connection() -> None:
    store = _store(RetrieverEngineType.POSTGRES, cc=ConnectionConfig(use_default_connection=False))
    with pytest.raises(ValidationError, match="use_default_connection=true"):
        await factory_module._create_postgres_engine(store, _DB())


async def test_create_postgres_engine_wraps_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_repo = _fake_repo()
    monkeypatch.setattr(
        factory_module,
        "_new_postgres_retrieve_engine_repository",
        AsyncMock(return_value=fake_repo),
    )
    store = _store(RetrieverEngineType.POSTGRES, cc=ConnectionConfig(use_default_connection=True))
    engine = await factory_module._create_postgres_engine(store, _DB())
    assert isinstance(engine, KVHybridRetrieveEngine)
    assert engine.engine_type() == RetrieverEngineType.POSTGRES
    assert engine._index_repository is fake_repo


async def test_create_sqlite_engine_wraps_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_repo = _fake_repo()
    monkeypatch.setattr(
        factory_module,
        "_new_sqlite_retrieve_engine_repository",
        AsyncMock(return_value=fake_repo),
    )
    engine = await factory_module._create_sqlite_engine(_store(RetrieverEngineType.SQLITE), _DB())
    assert isinstance(engine, KVHybridRetrieveEngine)
    assert engine.engine_type() == RetrieverEngineType.SQLITE
    assert engine._index_repository is fake_repo


async def test_create_elasticsearch_engine_routes_by_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v7 = cast("RetrieveEngineService", _FakeService(name="v7"))
    v8 = cast("RetrieveEngineService", _FakeService(name="v8"))
    monkeypatch.setattr(
        factory_module, "_create_elasticsearch_v7_engine", AsyncMock(return_value=v7)
    )
    monkeypatch.setattr(
        factory_module, "_create_elasticsearch_v8_engine", AsyncMock(return_value=v8)
    )
    store_7x = _store(RetrieverEngineType.ELASTICSEARCH, cc=ConnectionConfig(version="7.10.1"))
    assert await factory_module._create_elasticsearch_engine(store_7x, _Cfg()) is v7

    store_8x = _store(RetrieverEngineType.ELASTICSEARCH, cc=ConnectionConfig(version="8.0.0"))
    assert await factory_module._create_elasticsearch_engine(store_8x, _Cfg()) is v8

    store_default = _store(RetrieverEngineType.ELASTICSEARCH, cc=ConnectionConfig(version=""))
    assert await factory_module._create_elasticsearch_engine(store_default, _Cfg()) is v8


async def test_create_elasticsearch_v8_engine_wraps_repo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_repo = _fake_repo()
    monkeypatch.setattr(
        factory_module,
        "_new_elasticsearch_v8_retrieve_engine_repository",
        AsyncMock(return_value=fake_repo),
    )
    store = _store(
        RetrieverEngineType.ELASTICSEARCH,
        cc=ConnectionConfig(addr="http://es:9200", username="u", password="p"),
    )
    engine = await factory_module._create_elasticsearch_v8_engine(store, _Cfg())
    assert isinstance(engine, KVHybridRetrieveEngine)
    assert engine.engine_type() == RetrieverEngineType.ELASTICSEARCH
    assert engine._index_repository is fake_repo


async def test_create_qdrant_engine_applies_default_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    async def _fake_repo_ctor(
        host: str, port: int, api_key: str, use_tls: bool, index_config: IndexConfig
    ) -> RetrieveEngineRepository:
        captured["host"] = host
        captured["port"] = port
        captured["api_key"] = api_key
        captured["use_tls"] = use_tls
        return _fake_repo()

    monkeypatch.setattr(factory_module, "_new_qdrant_retrieve_engine_repository", _fake_repo_ctor)
    store = _store(
        RetrieverEngineType.QDRANT,
        cc=ConnectionConfig(host="qdrant-host", port=0, api_key="key", use_tls=True),
    )
    engine = await factory_module._create_qdrant_engine(store)
    assert captured["host"] == "qdrant-host"
    assert captured["port"] == 6334
    assert captured["api_key"] == "key"
    assert captured["use_tls"] is True
    assert engine.engine_type() == RetrieverEngineType.QDRANT


async def test_create_weaviate_engine_applies_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    async def _fake_repo_ctor(
        client_cfg: WeaviateClientConfig, index_config: IndexConfig
    ) -> RetrieveEngineRepository:
        captured["client_cfg"] = client_cfg
        return _fake_repo()

    monkeypatch.setattr(factory_module, "_new_weaviate_retrieve_engine_repository", _fake_repo_ctor)
    store = _store(RetrieverEngineType.WEAVIATE, cc=ConnectionConfig(host=""))
    engine = await factory_module._create_weaviate_engine(store)
    client_cfg = captured["client_cfg"]
    assert isinstance(client_cfg, WeaviateClientConfig)
    assert client_cfg.host == "weaviate:8080"
    assert client_cfg.grpc_address == "weaviate:50051"
    assert client_cfg.scheme == "http"
    assert engine.engine_type() == RetrieverEngineType.WEAVIATE


async def test_create_milvus_engine_wraps_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_repo = _fake_repo()
    monkeypatch.setattr(
        factory_module,
        "_new_milvus_retrieve_engine_repository",
        AsyncMock(return_value=fake_repo),
    )
    store = _store(
        RetrieverEngineType.MILVUS,
        cc=ConnectionConfig(addr="", username="root", password="pw", database="db"),
    )
    engine = await factory_module._create_milvus_engine(_CTX, store)
    assert isinstance(engine, KVHybridRetrieveEngine)
    assert engine.engine_type() == RetrieverEngineType.MILVUS
    assert engine._index_repository is fake_repo


async def test_create_doris_engine_requires_addr_and_database() -> None:
    with pytest.raises(ValidationError, match="requires addr"):
        await factory_module._create_doris_engine(
            _store(RetrieverEngineType.DORIS, cc=ConnectionConfig(database="kb"))
        )
    with pytest.raises(ValidationError, match="requires database"):
        await factory_module._create_doris_engine(
            _store(RetrieverEngineType.DORIS, cc=ConnectionConfig(addr="doris-fe:9030"))
        )


async def test_create_doris_engine_computes_http_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    async def _fake_repo_ctor(
        addr: str,
        http_base: str,
        username: str,
        password: str,
        database: str,
        index_config: IndexConfig,
    ) -> RetrieveEngineRepository:
        captured["addr"] = addr
        captured["http_base"] = http_base
        return _fake_repo()

    monkeypatch.setattr(factory_module, "_new_doris_retrieve_engine_repository", _fake_repo_ctor)
    store = _store(
        RetrieverEngineType.DORIS,
        cc=ConnectionConfig(addr="doris-fe:9030", database="kb", username="root"),
    )
    engine = await factory_module._create_doris_engine(store)
    assert captured["addr"] == "doris-fe:9030"
    assert captured["http_base"] == "http://doris-fe:8030"
    assert engine.engine_type() == RetrieverEngineType.DORIS


async def test_create_tencent_vectordb_engine_wraps_repo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_repo = _fake_repo()
    monkeypatch.setattr(
        factory_module,
        "_new_tencent_vectordb_retrieve_engine_repository",
        AsyncMock(return_value=fake_repo),
    )
    store = _store(
        RetrieverEngineType.TENCENT_VECTORDB,
        cc=ConnectionConfig(addr="tvdb:8080", username="u", api_key="key", database="db"),
    )
    engine = await factory_module._create_tencent_vectordb_engine(store)
    assert isinstance(engine, KVHybridRetrieveEngine)
    assert engine.engine_type() == RetrieverEngineType.TENCENT_VECTORDB
    assert engine._index_repository is fake_repo


async def test_create_opensearch_engine_clears_env_store_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_ids: list[str] = []

    async def _fake_repo_ctor(
        cc: ConnectionConfig, audit_sink: object, store_id: str, index_config: IndexConfig
    ) -> RetrieveEngineRepository:
        store_ids.append(store_id)
        return _fake_repo()

    monkeypatch.setattr(
        factory_module, "_new_opensearch_retrieve_engine_repository", _fake_repo_ctor
    )
    env_store = _store(
        RetrieverEngineType.OPENSEARCH,
        cc=ConnectionConfig(addr="https://os:9200"),
        store_id="__env_opensearch__",
    )
    engine = await factory_module._create_opensearch_engine(_CTX, env_store, None)
    assert engine.engine_type() == RetrieverEngineType.OPENSEARCH

    db_store = _store(RetrieverEngineType.OPENSEARCH, cc=ConnectionConfig(addr="https://os:9200"))
    await factory_module._create_opensearch_engine(_CTX, db_store, None)
    assert store_ids == ["", "store-1"]


# ── factory closure ─────────────────────────────────────────────────


async def test_new_engine_factory_returns_closure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_switch(
        _ctx: object,
        _store: VectorStoreLike,
        _db: object,
        _cfg: object,
        _audit_sink: object | None = None,
    ) -> RetrieveEngineService:
        return cast("RetrieveEngineService", _FakeService(name="built"))

    monkeypatch.setattr(factory_module, "create_engine_service_from_store", _fake_switch)
    engine_factory = new_engine_factory(_DB(), _Cfg())
    store = _store(RetrieverEngineType.QDRANT)
    engine = await engine_factory(_CTX, store)
    assert isinstance(engine, _FakeService)
    assert engine.name == "built"
