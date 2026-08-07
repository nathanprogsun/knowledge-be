"""Retrieval engine factory (upstream ``engine_factory.go``).

``new_engine_factory`` returns the ``EngineFactory`` closure wired by the
container; ``create_engine_service_from_store`` is the DB-store switch that
builds an engine service from a ``VectorStore`` config. The switch mirrors
the upstream factory: each engine path wraps its repository in a
``KVHybridRetrieveEngine``. The Postgres repository is wired to its concrete
implementation; every other engine repository constructor is a placeholder
that raises ``NotImplementedError`` so wiring is explicit rather than silent.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.ai.retrieval.base import (
    AppConfig,
    AuditSink,
    Context,
    Database,
    EngineFactory,
    RetrieveEngineRepository,
    RetrieveEngineService,
)
from src.ai.retrieval.elasticsearch_v7 import new_elasticsearch_v7_repository
from src.ai.retrieval.elasticsearch_v8 import new_elasticsearch_v8_repository
from src.ai.retrieval.kv_hybrid import new_kv_hybrid_retrieve_engine
from src.ai.retrieval.milvus import new_milvus_retrieve_engine_repository
from src.ai.retrieval.opensearch import new_opensearch_repository
from src.ai.retrieval.pgvector import new_postgres_retrieve_engine_repository
from src.ai.retrieval.qdrant import new_qdrant_retrieve_engine_repository
from src.ai.retrieval.types import (
    ConnectionConfig,
    IndexConfig,
    MilvusClientConfig,
    RetrieverEngineType,
    VectorStoreLike,
    is_env_store_id,
)
from src.ai.retrieval.weaviate import new_weaviate_retrieve_engine_repository_from_env


@dataclass(frozen=True, slots=True)
class WeaviateClientConfig:
    """Weaviate client settings (upstream ``createWeaviateEngine``)."""

    host: str = "weaviate:8080"
    grpc_address: str = "weaviate:50051"
    scheme: str = "http"
    api_key: str = ""


def build_milvus_client_config(cc: ConnectionConfig) -> MilvusClientConfig:
    """Build Milvus client settings from a store connection config."""
    addr = cc.addr
    if addr == "":
        addr = "localhost:19530"
    return MilvusClientConfig(
        address=addr,
        username=cc.username,
        password=cc.password,
        db_name=cc.database,
        dial_timeout_seconds=5,
    )


def is_es_v7(version: str) -> bool:
    """Report whether a detected ES version is 7.x (upstream ``isESv7``)."""
    return version.startswith("7.")


def host_from_addr(addr: str) -> str:
    """Split the host out of a ``host:port`` address (upstream ``hostFromAddr``)."""
    index = addr.rfind(":")
    if index > 0:
        return addr[:index]
    return addr


# ── Engine repository placeholders ───────────────────────────────────
#
# Each placeholder mirrors the signature the concrete engine PR will fill
# in. They raise ``NotImplementedError`` so the factory fails loudly until
# the engine repository module lands.


async def _new_postgres_retrieve_engine_repository(db: Database) -> RetrieveEngineRepository:
    return new_postgres_retrieve_engine_repository(db)


async def _new_sqlite_retrieve_engine_repository(db: Database) -> RetrieveEngineRepository:
    raise NotImplementedError("sqlite retrieval repository lands with the sqlite-vec engine")


async def _new_elasticsearch_v8_retrieve_engine_repository(
    addr: str,
    username: str,
    password: str,
    cfg: AppConfig,
    index_config: IndexConfig,
) -> RetrieveEngineRepository:
    return await new_elasticsearch_v8_repository(addr, username, password, cfg, index_config)


async def _new_elasticsearch_v7_retrieve_engine_repository(
    addr: str,
    username: str,
    password: str,
    cfg: AppConfig,
    index_config: IndexConfig,
) -> RetrieveEngineRepository:
    return await new_elasticsearch_v7_repository(addr, username, password, cfg, index_config)


async def _new_opensearch_retrieve_engine_repository(
    cc: ConnectionConfig,
    audit_sink: AuditSink | None,
    store_id: str,
    index_config: IndexConfig,
) -> RetrieveEngineRepository:
    return await new_opensearch_repository(cc, audit_sink, store_id, index_config)


async def _new_qdrant_retrieve_engine_repository(
    host: str,
    port: int,
    api_key: str,
    use_tls: bool,
    index_config: IndexConfig,
) -> RetrieveEngineRepository:
    return await new_qdrant_retrieve_engine_repository(
        host, port, api_key, use_tls, index_config
    )


async def _new_milvus_retrieve_engine_repository(
    client_cfg: MilvusClientConfig,
    index_config: IndexConfig,
) -> RetrieveEngineRepository:
    return await new_milvus_retrieve_engine_repository(client_cfg, index_config)


async def _new_weaviate_retrieve_engine_repository(
    client_cfg: WeaviateClientConfig,
    index_config: IndexConfig,
) -> RetrieveEngineRepository:
    """Build a Weaviate repository from a store connection config (DB-store path)."""
    return await new_weaviate_retrieve_engine_repository_from_env(
        client_cfg.host,
        client_cfg.grpc_address,
        client_cfg.scheme,
        client_cfg.api_key,
        index_config,
    )


async def _new_doris_retrieve_engine_repository(
    http_base: str,
    username: str,
    password: str,
    database: str,
    index_config: IndexConfig,
) -> RetrieveEngineRepository:
    raise NotImplementedError("doris repository lands with the doris engine")


async def _new_tencent_vectordb_retrieve_engine_repository(
    addr: str,
    username: str,
    api_key: str,
    database: str,
    index_config: IndexConfig,
) -> RetrieveEngineRepository:
    raise NotImplementedError("tencent_vectordb repository lands with the tencent vectordb engine")


# ── Per-engine builders ──────────────────────────────────────────────


async def _create_postgres_engine(store: VectorStoreLike, db: Database) -> RetrieveEngineService:
    cc = store.connection_config
    if not cc.use_default_connection:
        raise ValueError(
            "custom postgres connections not yet supported; use use_default_connection=true"
        )
    repo = await _new_postgres_retrieve_engine_repository(db)
    return new_kv_hybrid_retrieve_engine(repo, RetrieverEngineType.POSTGRES)


async def _create_sqlite_engine(_store: VectorStoreLike, db: Database) -> RetrieveEngineService:
    repo = await _new_sqlite_retrieve_engine_repository(db)
    return new_kv_hybrid_retrieve_engine(repo, RetrieverEngineType.SQLITE)


async def _create_elasticsearch_engine(
    store: VectorStoreLike, cfg: AppConfig
) -> RetrieveEngineService:
    # Version-based v7/v8 selection; empty version defaults to v8.
    if is_es_v7(store.connection_config.version):
        return await _create_elasticsearch_v7_engine(store, cfg)
    return await _create_elasticsearch_v8_engine(store, cfg)


async def _create_elasticsearch_v8_engine(
    store: VectorStoreLike, cfg: AppConfig
) -> RetrieveEngineService:
    cc = store.connection_config
    repo = await _new_elasticsearch_v8_retrieve_engine_repository(
        cc.addr, cc.username, cc.password, cfg, store.index_config
    )
    return new_kv_hybrid_retrieve_engine(repo, RetrieverEngineType.ELASTICSEARCH)


async def _create_elasticsearch_v7_engine(
    store: VectorStoreLike, cfg: AppConfig
) -> RetrieveEngineService:
    cc = store.connection_config
    repo = await _new_elasticsearch_v7_retrieve_engine_repository(
        cc.addr, cc.username, cc.password, cfg, store.index_config
    )
    return new_kv_hybrid_retrieve_engine(repo, RetrieverEngineType.ELASTICSEARCH)


async def _create_qdrant_engine(store: VectorStoreLike) -> RetrieveEngineService:
    cc = store.connection_config
    port = cc.port
    if port == 0:
        port = 6334
    repo = await _new_qdrant_retrieve_engine_repository(
        cc.host, port, cc.api_key, cc.use_tls, store.index_config
    )
    return new_kv_hybrid_retrieve_engine(repo, RetrieverEngineType.QDRANT)


async def _create_milvus_engine(_ctx: Context, store: VectorStoreLike) -> RetrieveEngineService:
    client_cfg = build_milvus_client_config(store.connection_config)
    repo = await _new_milvus_retrieve_engine_repository(client_cfg, store.index_config)
    return new_kv_hybrid_retrieve_engine(repo, RetrieverEngineType.MILVUS)


async def _create_weaviate_engine(store: VectorStoreLike) -> RetrieveEngineService:
    cc = store.connection_config
    host = cc.host
    if host == "":
        host = "weaviate:8080"
    grpc_address = cc.grpc_address
    if grpc_address == "":
        grpc_address = "weaviate:50051"
    scheme = cc.scheme
    if scheme == "":
        scheme = "http"
    client_cfg = WeaviateClientConfig(
        host=host,
        grpc_address=grpc_address,
        scheme=scheme,
        api_key=cc.api_key,
    )
    repo = await _new_weaviate_retrieve_engine_repository(client_cfg, store.index_config)
    return new_kv_hybrid_retrieve_engine(repo, RetrieverEngineType.WEAVIATE)


async def _create_doris_engine(store: VectorStoreLike) -> RetrieveEngineService:
    cc = store.connection_config
    if cc.addr == "":
        raise ValueError("doris connection requires addr (host:port)")
    if cc.database == "":
        raise ValueError("doris connection requires database")
    http_port = cc.http_port
    if http_port <= 0:
        http_port = 8030
    http_base = f"http://{host_from_addr(cc.addr)}:{http_port}"
    repo = await _new_doris_retrieve_engine_repository(
        http_base, cc.username, cc.password, cc.database, store.index_config
    )
    return new_kv_hybrid_retrieve_engine(repo, RetrieverEngineType.DORIS)


async def _create_tencent_vectordb_engine(store: VectorStoreLike) -> RetrieveEngineService:
    cc = store.connection_config
    repo = await _new_tencent_vectordb_retrieve_engine_repository(
        cc.addr, cc.username, cc.api_key, cc.database, store.index_config
    )
    return new_kv_hybrid_retrieve_engine(repo, RetrieverEngineType.TENCENT_VECTORDB)


async def _create_opensearch_engine(
    _ctx: Context,
    store: VectorStoreLike,
    audit_sink: AuditSink | None,
) -> RetrieveEngineService:
    cc = store.connection_config
    # Env stores share the cluster without a per-store index prefix; DB
    # stores fold their (>=16-char) ID into the index name.
    store_id = store.id
    if is_env_store_id(store_id):
        store_id = ""
    repo = await _new_opensearch_retrieve_engine_repository(
        cc, audit_sink, store_id, store.index_config
    )
    return new_kv_hybrid_retrieve_engine(repo, RetrieverEngineType.OPENSEARCH)


# ── Switch + factory ─────────────────────────────────────────────────


async def create_engine_service_from_store(
    ctx: Context,
    store: VectorStoreLike,
    db: Database,
    cfg: AppConfig,
    audit_sink: AuditSink | None = None,
) -> RetrieveEngineService:
    """Build an engine service from a VectorStore config (upstream ``createEngineServiceFromStore``)."""
    engine_type = store.engine_type
    if engine_type == RetrieverEngineType.POSTGRES:
        return await _create_postgres_engine(store, db)
    if engine_type == RetrieverEngineType.ELASTICSEARCH:
        return await _create_elasticsearch_engine(store, cfg)
    if engine_type == RetrieverEngineType.QDRANT:
        return await _create_qdrant_engine(store)
    if engine_type == RetrieverEngineType.MILVUS:
        return await _create_milvus_engine(ctx, store)
    if engine_type == RetrieverEngineType.WEAVIATE:
        return await _create_weaviate_engine(store)
    if engine_type == RetrieverEngineType.DORIS:
        return await _create_doris_engine(store)
    if engine_type == RetrieverEngineType.SQLITE:
        return await _create_sqlite_engine(store, db)
    if engine_type == RetrieverEngineType.TENCENT_VECTORDB:
        return await _create_tencent_vectordb_engine(store)
    if engine_type == RetrieverEngineType.OPENSEARCH:
        return await _create_opensearch_engine(ctx, store, audit_sink)
    raise ValueError(f"unsupported engine type: {store.engine_type}")


def new_engine_factory(
    db: Database,
    cfg: AppConfig,
    audit_svc: AuditSink | None = None,
) -> EngineFactory:
    """Return an ``EngineFactory`` closure over ``db``, ``cfg`` and the audit sink."""

    async def _build(ctx: Context, store: VectorStoreLike) -> RetrieveEngineService:
        return await create_engine_service_from_store(ctx, store, db, cfg, audit_svc)

    return _build


__all__ = [
    "MilvusClientConfig",
    "WeaviateClientConfig",
    "build_milvus_client_config",
    "create_engine_service_from_store",
    "host_from_addr",
    "is_es_v7",
    "new_engine_factory",
]
