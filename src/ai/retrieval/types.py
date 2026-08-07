"""Retrieval domain types.

Mirrors the upstream ``retriever.go`` contract: engine / retriever type
enums, retrieval parameters, retrieval results, and the index payload
(upstream ``embedding.go``). The ``ConnectionConfig`` / ``IndexConfig`` /
``VectorStore`` models mirror the upstream ``vectorstore.go`` contract so
the ai layer can build engines from a store config without importing the
core or storage layers — the store is supplied as a structural protocol
and coerced to these models at the boundary.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum
from typing import Protocol, TypeAlias, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from src.common.json import JsonObject


class RetrieverEngineType(StrEnum):
    """Retriever engine type (upstream ``RetrieverEngineType``)."""

    POSTGRES = "postgres"
    ELASTICSEARCH = "elasticsearch"
    INFINITY = "infinity"
    ELASTICFAISS = "elasticfaiss"
    QDRANT = "qdrant"
    MILVUS = "milvus"
    WEAVIATE = "weaviate"
    DORIS = "doris"
    SQLITE = "sqlite"
    TENCENT_VECTORDB = "tencent_vectordb"
    OPENSEARCH = "opensearch"


class RetrieverType(StrEnum):
    """Retriever type (upstream ``RetrieverType``)."""

    KEYWORDS = "keywords"
    VECTOR = "vector"
    WEB_SEARCH = "websearch"


class SourceType(IntEnum):
    """Content source type (upstream ``SourceType``)."""

    CHUNK = 0
    PASSAGE = 1
    SUMMARY = 2


class MatchType(IntEnum):
    """Matching algorithm type (upstream ``MatchType``).

    ``DIRECT_LOAD`` is reserved to preserve serialized enum values; it is
    no longer produced by the retrieval pipeline.
    """

    EMBEDDING = 0
    KEYWORDS = 1
    NEAR_BY_CHUNK = 2
    HISTORY = 3
    PARENT_CHUNK = 4
    RELATION_CHUNK = 5
    GRAPH = 6
    WEB_SEARCH = 7
    DIRECT_LOAD = 8
    DATA_ANALYSIS = 9


class IndexInfo(BaseModel):
    """Metadata about one indexed chunk (upstream ``IndexInfo``)."""

    model_config = ConfigDict(frozen=True)

    id: str = ""
    content: str = ""
    source_id: str = ""
    source_type: SourceType = SourceType.CHUNK
    chunk_id: str = ""
    knowledge_id: str = ""
    knowledge_base_id: str = ""
    knowledge_type: str = ""
    tag_id: str = ""
    is_enabled: bool = False
    is_recommended: bool = False


class RetrieveParams(BaseModel):
    """Parameters for one retrieval call (upstream ``RetrieveParams``).

    ``embedding`` carries the query embedding for vector retrieval;
    ``additional_params`` is free-form and interpreted by each engine.
    """

    model_config = ConfigDict(frozen=True)

    query: str = ""
    embedding: list[float] = Field(default_factory=list)
    knowledge_base_ids: list[str] = Field(default_factory=list)
    knowledge_ids: list[str] = Field(default_factory=list)
    tag_ids: list[str] = Field(default_factory=list)
    exclude_knowledge_ids: list[str] = Field(default_factory=list)
    exclude_chunk_ids: list[str] = Field(default_factory=list)
    top_k: int = 0
    threshold: float = 0.0
    knowledge_type: str = ""
    additional_params: JsonObject = Field(default_factory=dict)
    retriever_type: RetrieverType = RetrieverType.VECTOR


class RetrieverEngineParams(BaseModel):
    """Engine + retriever selection (upstream ``RetrieverEngineParams``)."""

    model_config = ConfigDict(frozen=True)

    retriever_engine_type: RetrieverEngineType
    retriever_type: RetrieverType


class IndexWithScore(BaseModel):
    """One indexed record with its match score (upstream ``IndexWithScore``)."""

    model_config = ConfigDict(frozen=True)

    id: str = ""
    content: str = ""
    source_id: str = ""
    source_type: SourceType = SourceType.CHUNK
    chunk_id: str = ""
    knowledge_id: str = ""
    knowledge_base_id: str = ""
    tag_id: str = ""
    score: float = 0.0
    match_type: MatchType = MatchType.EMBEDDING
    is_enabled: bool = False


class RetrieveResult(BaseModel):
    """Outcome of one retrieval call (upstream ``RetrieveResult``).

    ``error`` carries a retrieval failure message instead of an exception
    object — a result either holds ``results`` or an ``error`` string.
    """

    model_config = ConfigDict(frozen=True)

    results: list[IndexWithScore] = Field(default_factory=list)
    retriever_engine_type: RetrieverEngineType
    retriever_type: RetrieverType
    error: str | None = None


#: Storage parameters handed to the repository ``save`` / ``batch_save`` /
#: ``estimate_storage_size`` calls. The embedding map is keyed by source
#: (or chunk) id; engine followups may extend the shape with driver-specific
#: keys.
IndexSaveParams: TypeAlias = dict[str, dict[str, list[float]]]


class ConnectionConfig(BaseModel):
    """Driver-specific connection parameters (upstream ``ConnectionConfig``).

    Sensitive fields (``password`` / ``api_key``) are expected to be
    already decrypted by the caller.
    """

    model_config = ConfigDict(frozen=True)

    addr: str = ""
    username: str = ""
    password: str = ""
    api_key: str = ""
    insecure_skip_verify: bool = False
    host: str = ""
    port: int = 0
    use_tls: bool = False
    grpc_address: str = ""
    scheme: str = ""
    database: str = ""
    use_default_connection: bool = False
    http_port: int = 0
    version: str = ""


class IndexConfig(BaseModel):
    """Index / collection configuration (upstream ``IndexConfig``).

    Zero values fall back to engine-specific defaults at build time.
    """

    model_config = ConfigDict(frozen=True)

    index_name: str = ""
    number_of_shards: int = 0
    number_of_replicas: int = 0
    collection_prefix: str = ""
    collection_name: str = ""
    shard_number: int = 0
    replication_factor: int = 0
    shards_num: int = 0
    replica_number: int = 0
    desired_shard_count: int = 0
    buckets_num: int = 0
    replication_num: int = 0
    hnsw_m: int = 0
    hnsw_ef_construction: int = 0
    hnsw_ef_search: int = 0
    knn_engine: str = ""


@runtime_checkable
class VectorStoreLike(Protocol):
    """Structural shape of a configured vector store.

    Satisfied by the storage-row projection and by the concrete
    ``VectorStore`` model; the ai layer never imports the storage or core
    layers, so the boundary accepts any object with these attributes.
    """

    id: str
    tenant_id: int
    name: str
    engine_type: RetrieverEngineType
    connection_config: ConnectionConfig
    index_config: IndexConfig


class VectorStore(BaseModel):
    """A configured vector database instance (upstream ``types.VectorStore``)."""

    model_config = ConfigDict(frozen=True)

    id: str = ""
    tenant_id: int = 0
    name: str = ""
    engine_type: RetrieverEngineType = RetrieverEngineType.ELASTICSEARCH
    connection_config: ConnectionConfig = Field(default_factory=ConnectionConfig)
    index_config: IndexConfig = Field(default_factory=IndexConfig)


#: Prefix of virtual env-store ids synthesized from ``RETRIEVE_DRIVER``.
ENV_STORE_ID_PREFIX: str = "__env_"


def is_env_store_id(store_id: str) -> bool:
    """Report whether ``store_id`` is a virtual env-store id."""
    return store_id.startswith(ENV_STORE_ID_PREFIX)


__all__ = [
    "ENV_STORE_ID_PREFIX",
    "ConnectionConfig",
    "IndexConfig",
    "IndexInfo",
    "IndexSaveParams",
    "IndexWithScore",
    "MatchType",
    "RetrieveParams",
    "RetrieveResult",
    "RetrieverEngineParams",
    "RetrieverEngineType",
    "RetrieverType",
    "SourceType",
    "VectorStore",
    "VectorStoreLike",
    "is_env_store_id",
]
