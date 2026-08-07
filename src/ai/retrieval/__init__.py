"""Retrieval engines: types, interfaces, factory, registry, and hybrid engine.

Public surface: the domain types (``RetrieverEngineType`` / ``RetrieverType`` /
``RetrieveParams`` / ``RetrieveResult`` / ``IndexInfo``), the interfaces
(``RetrieveEngine`` / ``RetrieveEngineRepository`` / ``RetrieveEngineService`` /
``RetrieveEngineRegistry``), the engine factory
(``new_engine_factory`` / ``create_engine_service_from_store``), the env-driven
registry (``init_retrieve_engine_registry`` / ``RetrieveEngineRegistry``), and
the keywords-vector hybrid engine (``new_kv_hybrid_retrieve_engine``).

The ai layer never imports core or storage: stores and the database handle
are supplied structurally through protocols declared in ``base.py``.
"""

from __future__ import annotations

from src.ai.retrieval.base import (
    AppConfig,
    AuditSink,
    Context,
    Database,
    Embedder,
    EngineFactory,
    RetrieveEngine,
    RetrieveEngineRepository,
    RetrieveEngineService,
    StoreRegistry,
    VectorStoreRepositoryLike,
)
from src.ai.retrieval.base import (
    RetrieveEngineRegistry as RetrieveEngineRegistryProtocol,
)
from src.ai.retrieval.elasticsearch_v7 import (
    ElasticsearchV7Repository,
    new_elasticsearch_v7_client,
    new_elasticsearch_v7_repository,
)
from src.ai.retrieval.elasticsearch_v8 import (
    ElasticsearchV8Repository,
    new_elasticsearch_v8_client,
    new_elasticsearch_v8_repository,
)
from src.ai.retrieval.doris import (
    DorisRepository,
    new_doris_retrieve_engine_repository,
)
)
from src.ai.retrieval.env_registry import (
    StoreLoaderFn,
    init_retrieve_engine_registry,
    load_db_stores_into_registry,
    parse_retrieve_driver,
)
from src.ai.retrieval.factory import (
    MilvusClientConfig,
    WeaviateClientConfig,
    build_milvus_client_config,
    create_engine_service_from_store,
    host_from_addr,
    is_es_v7,
    new_engine_factory,
)
from src.ai.retrieval.kv_hybrid import (
    KVHybridRetrieveEngine,
    new_kv_hybrid_retrieve_engine,
    sanitize_for_embedding,
)
from src.ai.retrieval.opensearch import (
    OpenSearchRepository,
    new_opensearch_client,
    new_opensearch_repository,
)
from src.ai.retrieval.qdrant import (
    QdrantRetrieveEngineRepository,
    new_qdrant_retrieve_engine_repository,
)
from src.ai.retrieval.registry import (
    ENGINE_BUILD_TIMEOUT_SECONDS,
    REBUILD_COOLDOWN_SECONDS,
    RetrieveEngineRegistry,
    VectorStoreNotFoundError,
    VectorStoreUnavailableError,
    new_retrieve_engine_registry,
)
from src.ai.retrieval.sqlite_vec import (
    SQLiteRepository,
    new_sqlite_retrieve_engine_repository,
)
from src.ai.retrieval.tencent_vectordb import (
    TencentVectorDBRepository,
    new_tencent_vectordb_retrieve_engine_repository,
)
from src.ai.retrieval.types import (
    ENV_STORE_ID_PREFIX,
    ConnectionConfig,
    IndexConfig,
    IndexInfo,
    IndexSaveParams,
    IndexWithScore,
    MatchType,
    RetrieveParams,
    RetrieverEngineParams,
    RetrieverEngineType,
    RetrieveResult,
    RetrieverType,
    SourceType,
    VectorStore,
    VectorStoreLike,
    is_env_store_id,
)

__all__ = [
    "ENGINE_BUILD_TIMEOUT_SECONDS",
    "ENV_STORE_ID_PREFIX",
    "REBUILD_COOLDOWN_SECONDS",
    "AppConfig",
    "AuditSink",
    "ConnectionConfig",
    "Context",
    "Database",
<<<<<<< HEAD
    "ElasticsearchV7Repository",
    "ElasticsearchV8Repository",
    "DorisRepository",
    "Embedder",
    "EngineFactory",
    "IndexConfig",
    "IndexInfo",
    "IndexSaveParams",
    "IndexWithScore",
    "KVHybridRetrieveEngine",
    "MatchType",
    "MilvusClientConfig",
    "OpenSearchRepository",
    "QdrantRetrieveEngineRepository",

    "RetrieveEngine",
    "RetrieveEngineRegistry",
    "RetrieveEngineRegistryProtocol",
    "RetrieveEngineRepository",
    "RetrieveEngineService",
    "RetrieveParams",
    "RetrieveResult",
    "RetrieverEngineParams",
    "RetrieverEngineType",
    "RetrieverType",
    "SQLiteRepository",
    "SourceType",
    "StoreLoaderFn",
    "StoreRegistry",
    "TencentVectorDBRepository",
    "VectorStore",
    "VectorStoreLike",
    "VectorStoreNotFoundError",
    "VectorStoreRepositoryLike",
    "VectorStoreUnavailableError",
    "WeaviateClientConfig",
    "build_milvus_client_config",
    "create_engine_service_from_store",
    "host_from_addr",
    "init_retrieve_engine_registry",
    "is_env_store_id",
    "is_es_v7",
    "load_db_stores_into_registry",
<<<<<<< HEAD
    "new_elasticsearch_v7_client",
    "new_elasticsearch_v7_repository",
    "new_elasticsearch_v8_client",
    "new_elasticsearch_v8_repository",
    "new_doris_retrieve_engine_repository",
    "new_engine_factory",
    "new_kv_hybrid_retrieve_engine",
    "new_opensearch_client",
    "new_opensearch_repository",
    "new_qdrant_retrieve_engine_repository",

    "new_retrieve_engine_registry",
    "new_sqlite_retrieve_engine_repository",
    "new_tencent_vectordb_retrieve_engine_repository",
    "parse_retrieve_driver",
    "sanitize_for_embedding",
]
