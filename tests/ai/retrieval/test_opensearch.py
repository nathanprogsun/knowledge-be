"""Tests for the OpenSearch k-NN engine repository.

The SDK client is mocked with ``MagicMock`` - no OpenSearch cluster is
contacted. Pinned: per-dimension lazy index creation, kNN vector retrieval,
keyword retrieval, save/batch-save with size caps, delete-by-query, copy-
indices, batch update-by-query, audit sink events, and error sentinels.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from opensearchpy.exceptions import NotFoundError

from src.ai.embedding import TaskContext
from src.ai.retrieval.opensearch import (
    BatchTooLargeError,
    ConfigInvalidError,
    DimensionMismatchError,
    OpenSearchRepository,
    _build_filter_must,
    _build_index_mapping,
    _build_keywords_mapping,
    _build_knn_query,
    _resolve_base_index,
    _transform_source_id,
    new_opensearch_client,
)
from src.ai.retrieval.types import (
    ConnectionConfig,
    IndexInfo,
    MatchType,
    RetrieveParams,
    RetrieverEngineType,
    RetrieverType,
    SourceType,
)

_CTX = TaskContext()
_BASE = "weknora_test"


# ── helpers ──────────────────────────────────────────────────────────


def _mock_os_client() -> MagicMock:
    """Build a mocked opensearchpy client."""
    client = MagicMock()
    client.indices.exists_alias.return_value = True
    client.indices.create.return_value = {"acknowledged": True}
    client.indices.put_alias.return_value = {"acknowledged": True}
    client.indices.delete.return_value = {"acknowledged": True}
    client.search.return_value = {"hits": {"hits": []}}
    client.index.return_value = {"result": "created"}
    client.bulk.return_value = {"errors": False}
    client.delete_by_query.return_value = {"deleted": 1}
    client.update_by_query.return_value = {"updated": 1}
    return client


def _repo(
    client: MagicMock | None = None,
    base: str = _BASE,
    cfg: dict[str, Any] | None = None,
    audit: Any = None,
) -> OpenSearchRepository:
    return OpenSearchRepository(
        client or _mock_os_client(),
        base,
        cfg or {"shards": 4, "replicas": 1, "knn_engine": "lucene",
                "hnsw_m": 16, "hnsw_ef_construction": 100, "ef_search": 100},
        audit,
    )


def _index_info(**overrides: Any) -> IndexInfo:
    defaults: dict[str, Any] = {
        "content": "hello world",
        "source_id": "src-1",
        "chunk_id": "chunk-1",
        "knowledge_id": "k-1",
        "knowledge_base_id": "kb-1",
        "source_type": SourceType.CHUNK,
    }
    defaults.update(overrides)
    return IndexInfo(**defaults)


def _search_hit(doc_id: str = "doc1", score: float = 0.95, **source: Any) -> dict[str, Any]:
    src = {"chunk_id": "c1", "content": "text", "source_id": "s1",
           "knowledge_id": "k1", "knowledge_base_id": "kb1",
           "source_type": 0, "is_enabled": True}
    src.update(source)
    return {"_id": doc_id, "_score": score, "_source": src}


def _alias_not_found(client: MagicMock) -> None:
    """Configure mock so exists_alias raises NotFoundError."""
    client.indices.exists_alias.side_effect = NotFoundError(404, "not found")


# ── pure functions ───────────────────────────────────────────────────


def test_build_index_mapping_has_knn_vector_field() -> None:
    cfg = {"knn_engine": "lucene", "hnsw_m": 16, "hnsw_ef_construction": 100, "ef_search": 100}
    mapping = _build_index_mapping(cfg, 768)
    props = mapping["mappings"]["properties"]
    assert props["embedding"]["type"] == "knn_vector"
    assert props["embedding"]["dimension"] == 768
    assert props["embedding"]["method"]["space_type"] == "cosinesimil"
    assert props["content"]["type"] == "text"
    assert props["chunk_id"]["type"] == "keyword"


def test_build_keywords_mapping_omits_embedding() -> None:
    mapping = _build_keywords_mapping({"shards": 1, "replicas": 0})
    props = mapping["mappings"]["properties"]
    assert "embedding" not in props
    assert props["content"]["type"] == "text"


def test_build_filter_must_includes_is_enabled() -> None:
    params = RetrieveParams(knowledge_base_ids=["kb1"])
    must = _build_filter_must(params)
    assert {"term": {"is_enabled": True}} in must
    assert {"terms": {"knowledge_base_id": ["kb1"]}} in must


def test_build_filter_must_excludes() -> None:
    params = RetrieveParams(exclude_chunk_ids=["c1"], exclude_knowledge_ids=["k1"])
    must = _build_filter_must(params)
    assert any("must_not" in str(m) for m in must if "bool" in m)


def test_build_knn_query_includes_filter_and_threshold() -> None:
    must = [{"term": {"is_enabled": True}}]
    body = _build_knn_query([0.1, 0.2], 10, 0.5, must)
    assert body["size"] == 10
    assert body["min_score"] == 0.5
    knn = body["query"]["knn"]["embedding"]
    assert knn["k"] == 10
    assert knn["filter"]["bool"]["must"] == must


def test_build_knn_query_omits_threshold_when_zero() -> None:
    body = _build_knn_query([0.1], 5, 0.0, [])
    assert "min_score" not in body


def test_transform_source_id_regular_chunk() -> None:
    assert _transform_source_id("chunk-1", "chunk-1", "target-1") == "target-1"


def test_transform_source_id_generated_question() -> None:
    assert _transform_source_id("chunk-1-q1", "chunk-1", "target-1") == "target-1-q1"


def test_resolve_base_index_env_store_no_prefix() -> None:
    assert _resolve_base_index("", None) == "weknora"


def test_resolve_base_index_db_store_folds_id() -> None:
    base = _resolve_base_index("abcdef0123456789", None)
    assert base == "weknora_abcdef012345"


def test_resolve_base_index_rejects_short_store_id() -> None:
    with pytest.raises(ConfigInvalidError, match="storeID"):
        _resolve_base_index("short", None)


# ── repository: engine_type / support ───────────────────────────────


def test_engine_type() -> None:
    assert _repo().engine_type() == RetrieverEngineType.OPENSEARCH


def test_support_returns_keywords_and_vector() -> None:
    assert _repo().support() == [RetrieverType.KEYWORDS, RetrieverType.VECTOR]


# ── repository: save ───────────────────────────────────────────────


def test_save_with_embedding_uses_dim_alias() -> None:
    client = _mock_os_client()
    _alias_not_found(client)
    repo = _repo(client)
    info = _index_info()
    params = {"embedding": {"src-1": [0.1, 0.2]}}
    import asyncio
    asyncio.run(repo.save(_CTX, info, params))
    client.indices.create.assert_called_once()
    _, kwargs = client.indices.create.call_args
    assert "_v1" in kwargs["index"]
    client.index.assert_called_once()
    _, ikwargs = client.index.call_args
    assert ikwargs["id"] == "chunk-1"


def test_save_without_embedding_uses_keywords_index() -> None:
    client = _mock_os_client()
    _alias_not_found(client)
    repo = _repo(client)
    info = _index_info()
    import asyncio
    asyncio.run(repo.save(_CTX, info, {}))
    client.indices.create.assert_called_once()
    _, kwargs = client.indices.create.call_args
    assert "_keywords" in kwargs["index"]


# ── repository: batch_save ───────────────────────────────────────────


def test_batch_save_rejects_too_many_docs() -> None:
    repo = _repo()
    infos = [_index_info(chunk_id=f"c{i}") for i in range(1001)]
    import asyncio
    with pytest.raises(BatchTooLargeError, match="1000-doc cap"):
        asyncio.run(repo.batch_save(_CTX, infos, {}))


def test_batch_save_rejects_oversized_body() -> None:
    repo = _repo()
    infos = [_index_info(chunk_id=f"c{i}", source_id=f"src-{i}") for i in range(1000)]
    params = {"embedding": {f"src-{i}": [0.0] * 2000 for i in range(1000)}}
    import asyncio
    with pytest.raises(BatchTooLargeError, match="exceeds"):
        asyncio.run(repo.batch_save(_CTX, infos, params))


def test_batch_save_empty_list_skips() -> None:
    client = _mock_os_client()
    repo = _repo(client)
    import asyncio
    asyncio.run(repo.batch_save(_CTX, [], {}))
    client.bulk.assert_not_called()


# ── repository: retrieve ────────────────────────────────────────────


def test_vector_retrieve_builds_knn_query() -> None:
    client = _mock_os_client()
    client.search.return_value = {"hits": {"hits": [_search_hit()]}}
    repo = _repo(client)
    params = RetrieveParams(
        embedding=[0.1, 0.2], top_k=5, threshold=0.5,
        retriever_type=RetrieverType.VECTOR,
    )
    import asyncio
    results = asyncio.run(repo.retrieve(_CTX, params))
    assert len(results) == 1
    assert results[0].retriever_type == RetrieverType.VECTOR
    assert results[0].results[0].match_type == MatchType.EMBEDDING
    _, kwargs = client.search.call_args
    body = kwargs["body"]
    assert "knn" in body["query"]
    assert body["min_score"] == 0.5


def test_keyword_retrieve_builds_match_query() -> None:
    client = _mock_os_client()
    client.search.return_value = {"hits": {"hits": [_search_hit()]}}
    repo = _repo(client)
    params = RetrieveParams(
        query="hello", top_k=10,
        retriever_type=RetrieverType.KEYWORDS,
    )
    import asyncio
    results = asyncio.run(repo.retrieve(_CTX, params))
    assert len(results) == 1
    assert results[0].retriever_type == RetrieverType.KEYWORDS
    _, kwargs = client.search.call_args
    body = kwargs["body"]
    assert "bool" in body["query"]
    assert {"match": {"content": "hello"}} in body["query"]["bool"]["must"]


def test_retrieve_vector_requires_dim() -> None:
    repo = _repo()
    params = RetrieveParams(
        retriever_type=RetrieverType.VECTOR, top_k=5,
    )
    import asyncio
    with pytest.raises(DimensionMismatchError):
        asyncio.run(repo.retrieve(_CTX, params))


def test_retrieve_rejects_unknown_type() -> None:
    repo = _repo()
    params = RetrieveParams(
        retriever_type=RetrieverType.WEB_SEARCH, top_k=5,
    )
    import asyncio
    with pytest.raises(ValueError, match="unsupported retriever type"):
        asyncio.run(repo.retrieve(_CTX, params))


# ── repository: delete ───────────────────────────────────────────────


def test_delete_by_chunk_id_list() -> None:
    client = _mock_os_client()
    repo = _repo(client)
    import asyncio
    asyncio.run(repo.delete_by_chunk_id_list(_CTX, ["c1", "c2"], 128, "doc"))
    client.delete_by_query.assert_called_once()
    _, kwargs = client.delete_by_query.call_args
    assert "chunk_id" in kwargs["body"]["query"]["terms"]


def test_delete_empty_list_skips() -> None:
    client = _mock_os_client()
    repo = _repo(client)
    import asyncio
    asyncio.run(repo.delete_by_chunk_id_list(_CTX, [], 128, "doc"))
    client.delete_by_query.assert_not_called()


def test_delete_rejects_too_many() -> None:
    repo = _repo()
    ids = [f"c{i}" for i in range(1001)]
    import asyncio
    with pytest.raises(BatchTooLargeError):
        asyncio.run(repo.delete_by_chunk_id_list(_CTX, ids, 128, "doc"))


# ── repository: copy_indices ─────────────────────────────────────────


def test_copy_indices_empty_map_skips() -> None:
    client = _mock_os_client()
    repo = _repo(client)
    import asyncio
    asyncio.run(repo.copy_indices(
        _CTX, "kb", {}, {}, "tgt", 128, "doc",
    ))
    client.search.assert_not_called()


def test_copy_indices_requires_positive_dim() -> None:
    repo = _repo()
    import asyncio
    with pytest.raises(DimensionMismatchError, match="dim > 0"):
        asyncio.run(repo.copy_indices(
            _CTX, "kb", {"k": "k"}, {"c": "c"}, "tgt", 0, "doc",
        ))


def test_copy_indices_paginates_and_saves() -> None:
    client = _mock_os_client()
    client.search.return_value = {
        "hits": {"hits": [
            _search_hit(doc_id="d1", chunk_id="src_c1", knowledge_id="src_k1",
                        embedding=[0.1, 0.2]),
        ]}
    }
    repo = _repo(client)
    import asyncio
    asyncio.run(repo.copy_indices(
        _CTX,
        source_knowledge_base_id="src_kb",
        source_to_target_kb_id_map={"src_k1": "tgt_k1"},
        source_to_target_chunk_id_map={"src_c1": "tgt_c1"},
        target_knowledge_base_id="tgt_kb",
        dimension=128,
        knowledge_type="doc",
    ))
    client.bulk.assert_called_once()


# ── repository: batch_update ─────────────────────────────────────────


def test_batch_update_chunk_enabled_status_groups_by_value() -> None:
    client = _mock_os_client()
    repo = _repo(client)
    import asyncio
    asyncio.run(repo.batch_update_chunk_enabled_status(_CTX, {"c1": True, "c2": False}))
    assert client.update_by_query.call_count == 2


def test_batch_update_chunk_tag_id_groups_by_tag() -> None:
    client = _mock_os_client()
    repo = _repo(client)
    import asyncio
    asyncio.run(repo.batch_update_chunk_tag_id(_CTX, {"c1": "t1", "c2": "t2"}))
    assert client.update_by_query.call_count == 2


def test_batch_update_empty_map_skips() -> None:
    client = _mock_os_client()
    repo = _repo(client)
    import asyncio
    asyncio.run(repo.batch_update_chunk_enabled_status(_CTX, {}))
    client.update_by_query.assert_not_called()


# ── repository: audit sink ──────────────────────────────────────────


class _CapturingAuditSink:
    def __init__(self) -> None:
        self.index_created: list[tuple[str, int]] = []
        self.reindex_executed: list[tuple[str, str, int]] = []

    async def emit_index_created(self, ctx: Any, alias: str, dim: int) -> None:
        self.index_created.append((alias, dim))

    async def emit_reindex_executed(
        self, ctx: Any, src_alias: str, dst_alias: str, docs: int
    ) -> None:
        self.reindex_executed.append((src_alias, dst_alias, docs))


def test_audit_sink_emits_on_index_creation() -> None:
    client = _mock_os_client()
    _alias_not_found(client)
    sink = _CapturingAuditSink()
    repo = _repo(client, audit=sink)
    info = _index_info()
    params = {"embedding": {"src-1": [0.1, 0.2]}}
    import asyncio
    asyncio.run(repo.save(_CTX, info, params))
    assert len(sink.index_created) == 1
    alias, dim = sink.index_created[0]
    assert dim == 2
    assert "_v1" not in alias


def test_audit_sink_emits_on_reindex() -> None:
    client = _mock_os_client()
    client.search.return_value = {
        "hits": {"hits": [
            _search_hit(chunk_id="src_c1", knowledge_id="src_k1", embedding=[0.1, 0.2]),
        ]}
    }
    sink = _CapturingAuditSink()
    repo = _repo(client, audit=sink)
    import asyncio
    asyncio.run(repo.copy_indices(
        _CTX, "src_kb", {"src_k1": "tgt_k1"}, {"src_c1": "tgt_c1"},
        "tgt_kb", 128, "doc",
    ))
    assert len(sink.reindex_executed) == 1
    assert sink.reindex_executed[0][2] == 1


# ── repository: estimate_storage_size ───────────────────────────────


def test_estimate_storage_size_returns_lower_bound() -> None:
    repo = _repo()
    infos = [_index_info() for _ in range(3)]
    size = repo.estimate_storage_size(_CTX, infos, {})
    assert size == 3 * (1024 + 4 * 768 + 128)


def test_estimate_storage_size_empty_list() -> None:
    repo = _repo()
    assert repo.estimate_storage_size(_CTX, [], {}) == 0


# ── client construction ─────────────────────────────────────────────


def test_new_opensearch_client_requires_addr() -> None:
    with pytest.raises(ConfigInvalidError, match="Addr required"):
        new_opensearch_client(ConnectionConfig())


def test_new_opensearch_client_builds_with_auth() -> None:
    cc = ConnectionConfig(
        addr="https://os:9200",
        username="admin",
        password="secret",
    )
    client = new_opensearch_client(cc)
    assert client is not None
