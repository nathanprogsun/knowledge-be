"""Tests for the Milvus retrieval engine repository.

The pymilvus ``MilvusClient`` is faked with ``MagicMock`` - no Milvus server
is contacted. Pinned here: collection schema/index creation and caching,
single/batch upsert with UUID generation, vector + keywords retrieval,
delete-by-field, copy-indices with paginated reads, batch chunk status / tag
updates, storage-size estimation, the filter expression builder, and the
module-level constructor that wraps a real client.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.ai.embedding import TaskContext
from src.ai.retrieval.milvus import (
    ALL_FIELDS,
    FIELD_CHUNK_ID,
    FIELD_CONTENT,
    FIELD_CONTENT_SPARSE,
    FIELD_EMBEDDING,
    FIELD_ID,
    FIELD_IS_ENABLED,
    FIELD_KNOWLEDGE_BASE_ID,
    FIELD_KNOWLEDGE_ID,
    FIELD_SOURCE_ID,
    FIELD_TAG_ID,
    MilvusFilterConverter,
    MilvusRetrieveEngineRepository,
    MilvusVectorEmbedding,
    UniversalFilterCondition,
    _calculate_storage_size,
    _from_milvus_vector_embedding,
    _to_milvus_vector_embedding,
    _to_upsert_row,
    new_milvus_retrieve_engine_repository,
)
from src.ai.retrieval.types import (
    IndexConfig,
    IndexInfo,
    MatchType,
    MilvusClientConfig,
    RetrieveParams,
    RetrieverEngineType,
    RetrieverType,
    SourceType,
)
from src.common.exception import ValidationError

_CTX = TaskContext()

_DIM = 4
_COLLECTION = "weknora_embeddings_4"


# ── Fixtures ─────────────────────────────────────────────────────────


def _mock_client() -> MagicMock:
    """Return a ``MagicMock`` standing in for ``pymilvus.MilvusClient``."""
    client = MagicMock()
    client.has_collection.return_value = False
    client.list_collections.return_value = []
    client.search.return_value = [[]]
    client.query.return_value = []
    client.upsert.return_value = {"upsert_count": 1}
    client.delete.return_value = {"delete_count": 1}
    return client


def _repo(
    client: MagicMock | None = None,
    index_config: IndexConfig | None = None,
) -> MilvusRetrieveEngineRepository:
    return MilvusRetrieveEngineRepository(client or _mock_client(), index_config)


def _index_info(
    source_id: str = "src-1",
    content: str = "hello",
    chunk_id: str = "chunk-1",
    knowledge_id: str = "k-1",
    knowledge_base_id: str = "kb-1",
    tag_id: str = "tag-1",
    is_enabled: bool = True,
) -> IndexInfo:
    return IndexInfo(
        id="id-1",
        content=content,
        source_id=source_id,
        chunk_id=chunk_id,
        knowledge_id=knowledge_id,
        knowledge_base_id=knowledge_base_id,
        tag_id=tag_id,
        is_enabled=is_enabled,
    )


def _embedding_map(source_id: str, dim: int = _DIM) -> dict[str, list[float]]:
    return {source_id: [0.1] * dim}


def _search_row(
    row_id: str = "rid-1",
    content: str = "hit",
    chunk_id: str = "chunk-1",
    score: float = 0.9,
    is_enabled: bool = True,
    tag_id: str = "tag-1",
) -> dict[str, Any]:
    return {
        FIELD_ID: row_id,
        FIELD_CONTENT: content,
        "source_id": "src-1",
        "source_type": 0,
        FIELD_CHUNK_ID: chunk_id,
        FIELD_KNOWLEDGE_ID: "k-1",
        FIELD_KNOWLEDGE_BASE_ID: "kb-1",
        FIELD_TAG_ID: tag_id,
        FIELD_IS_ENABLED: is_enabled,
        FIELD_EMBEDDING: [0.1] * _DIM,
        "distance": score,
    }


# ── Construction / engine_type / support ─────────────────────────────


def test_engine_type_is_milvus() -> None:
    assert _repo().engine_type() == RetrieverEngineType.MILVUS


def test_supports_keywords_and_vector() -> None:
    assert _repo().support() == [RetrieverType.KEYWORDS, RetrieverType.VECTOR]


def test_collection_name_is_dimension_scoped() -> None:
    repo = _repo()
    assert repo._get_collection_name(4) == "weknora_embeddings_4"
    assert repo._get_collection_name(8) == "weknora_embeddings_8"


def test_collection_name_uses_index_config_prefix() -> None:
    repo = _repo(index_config=IndexConfig(collection_prefix="custom_idx"))
    assert repo._get_collection_name(4) == "custom_idx_4"


def test_collection_name_uses_index_config_name() -> None:
    repo = _repo(index_config=IndexConfig(collection_name="named_idx"))
    assert repo._get_collection_name(4) == "named_idx_4"


def test_collection_name_prefix_wins_over_name() -> None:
    repo = _repo(
        index_config=IndexConfig(collection_prefix="prefix", collection_name="named")
    )
    assert repo._get_collection_name(4) == "prefix_4"


# ── ensure_collection ────────────────────────────────────────────────


def test_ensure_collection_creates_and_loads() -> None:
    client = _mock_client()
    client.has_collection.return_value = False
    repo = _repo(client)
    repo._ensure_collection(_CTX, _DIM)
    client.create_collection.assert_called_once()
    client.create_index.assert_called_once()
    client.load_collection.assert_called_once_with(collection_name=_COLLECTION)


def test_ensure_collection_skips_existing_collection() -> None:
    client = _mock_client()
    client.has_collection.return_value = True
    repo = _repo(client)
    repo._ensure_collection(_CTX, _DIM)
    client.create_collection.assert_not_called()
    client.create_index.assert_not_called()
    client.load_collection.assert_called_once_with(collection_name=_COLLECTION)


def test_ensure_collection_is_cached_per_dimension() -> None:
    client = _mock_client()
    client.has_collection.return_value = True
    repo = _repo(client)
    repo._ensure_collection(_CTX, _DIM)
    repo._ensure_collection(_CTX, _DIM)
    client.load_collection.assert_called_once()


def test_ensure_collection_passes_shards_and_replica() -> None:
    client = _mock_client()
    client.has_collection.return_value = False
    repo = _repo(
        client,
        index_config=IndexConfig(shards_num=2, replica_number=3),
    )
    repo._ensure_collection(_CTX, _DIM)
    create_kwargs = client.create_collection.call_args.kwargs
    assert create_kwargs["shards_num"] == 2
    load_kwargs = client.load_collection.call_args.kwargs
    assert load_kwargs["replica_number"] == 3


def test_ensure_collection_omits_shards_when_zero() -> None:
    client = _mock_client()
    client.has_collection.return_value = False
    repo = _repo(client)
    repo._ensure_collection(_CTX, _DIM)
    create_kwargs = client.create_collection.call_args.kwargs
    assert "shards_num" not in create_kwargs
    load_kwargs = client.load_collection.call_args.kwargs
    assert "replica_number" not in load_kwargs


# ── Save ─────────────────────────────────────────────────────────────


async def test_save_upserts_single_row() -> None:
    client = _mock_client()
    client.has_collection.return_value = True
    repo = _repo(client)
    info = _index_info()
    await repo.save(_CTX, info, {"embedding": _embedding_map("src-1")})
    client.upsert.assert_called_once()
    data = client.upsert.call_args.kwargs["data"]
    assert len(data) == 1
    row = data[0]
    assert row[FIELD_ID]  # UUID generated
    assert row[FIELD_EMBEDDING] == [0.1] * _DIM
    assert row[FIELD_CONTENT] == "hello"
    assert row[FIELD_CHUNK_ID] == "chunk-1"


async def test_save_rejects_empty_embedding() -> None:
    repo = _repo()
    with pytest.raises(ValidationError, match="empty embedding vector"):
        await repo.save(_CTX, _index_info(), {})


# ── BatchSave ────────────────────────────────────────────────────────


async def test_batch_save_groups_by_dimension() -> None:
    client = _mock_client()
    client.has_collection.return_value = True
    repo = _repo(client)
    items = [
        _index_info(source_id=f"src-{i}", chunk_id=f"c-{i}") for i in range(3)
    ]
    params = {
        "embedding": {
            f"src-{i}": [0.1] * _DIM for i in range(3)
        }
    }
    await repo.batch_save(_CTX, items, params)
    assert client.upsert.call_count == 1
    data = client.upsert.call_args.kwargs["data"]
    assert len(data) == 3


async def test_batch_save_skips_empty_embeddings() -> None:
    client = _mock_client()
    repo = _repo(client)
    items = [_index_info(source_id="src-1"), _index_info(source_id="src-2")]
    params = {"embedding": {"src-1": [0.1] * _DIM}}  # src-2 missing
    await repo.batch_save(_CTX, items, params)
    client.upsert.assert_called_once()
    data = client.upsert.call_args.kwargs["data"]
    assert len(data) == 1
    assert data[0][FIELD_SOURCE_ID] == "src-1"


async def test_batch_save_empty_list_is_noop() -> None:
    client = _mock_client()
    repo = _repo(client)
    await repo.batch_save(_CTX, [], {})
    client.upsert.assert_not_called()


# ── Retrieve dispatch ────────────────────────────────────────────────


async def test_retrieve_dispatches_vector() -> None:
    client = _mock_client()
    client.has_collection.return_value = True
    client.search.return_value = [[_search_row()]]
    repo = _repo(client)
    params = RetrieveParams(
        embedding=[0.1] * _DIM,
        top_k=5,
        retriever_type=RetrieverType.VECTOR,
    )
    results = await repo.retrieve(_CTX, params)
    assert len(results) == 1
    assert results[0].retriever_type == RetrieverType.VECTOR
    assert len(results[0].results) == 1


async def test_retrieve_dispatches_keywords() -> None:
    client = _mock_client()
    client.list_collections.return_value = [_COLLECTION]
    client.search.return_value = [[_search_row()]]
    repo = _repo(client)
    params = RetrieveParams(
        query="hello",
        top_k=5,
        retriever_type=RetrieverType.KEYWORDS,
    )
    results = await repo.retrieve(_CTX, params)
    assert len(results) == 1
    assert results[0].retriever_type == RetrieverType.KEYWORDS
    assert results[0].results[0].score == 1.0


async def test_retrieve_invalid_type_raises() -> None:
    repo = _repo()
    params = RetrieveParams(retriever_type=RetrieverType.WEB_SEARCH)
    with pytest.raises(ValidationError, match="invalid retriever type"):
        await repo.retrieve(_CTX, params)


# ── VectorRetrieve ───────────────────────────────────────────────────


async def test_vector_retrieve_returns_empty_when_collection_missing() -> None:
    client = _mock_client()
    client.has_collection.return_value = False
    repo = _repo(client)
    params = RetrieveParams(embedding=[0.1] * _DIM, top_k=5)
    results = await repo.vector_retrieve(_CTX, params)
    assert len(results) == 1
    assert results[0].results == []


async def test_vector_retrieve_searches_with_filter() -> None:
    client = _mock_client()
    client.has_collection.return_value = True
    client.search.return_value = [[_search_row(score=0.95)]]
    repo = _repo(client)
    params = RetrieveParams(
        embedding=[0.1] * _DIM,
        top_k=10,
        knowledge_base_ids=["kb-1"],
        retriever_type=RetrieverType.VECTOR,
    )
    results = await repo.vector_retrieve(_CTX, params)
    assert len(results[0].results) == 1
    assert results[0].results[0].score == 0.95
    assert results[0].results[0].match_type == MatchType.EMBEDDING
    search_kwargs = client.search.call_args.kwargs
    assert search_kwargs["anns_field"] == FIELD_EMBEDDING
    assert search_kwargs["limit"] == 10
    assert search_kwargs["output_fields"] == list(ALL_FIELDS)
    assert "filter_params" in search_kwargs


async def test_vector_retrieve_sets_radius_when_threshold_positive() -> None:
    client = _mock_client()
    client.has_collection.return_value = True
    client.search.return_value = [[]]
    repo = _repo(client)
    params = RetrieveParams(embedding=[0.1] * _DIM, top_k=5, threshold=0.5)
    await repo.vector_retrieve(_CTX, params)
    search_params = client.search.call_args.kwargs["search_params"]
    assert search_params["params"]["radius"] == 0.5


async def test_vector_retrieve_no_search_params_when_threshold_zero() -> None:
    client = _mock_client()
    client.has_collection.return_value = True
    client.search.return_value = [[]]
    repo = _repo(client)
    params = RetrieveParams(embedding=[0.1] * _DIM, top_k=5)
    await repo.vector_retrieve(_CTX, params)
    assert client.search.call_args.kwargs["search_params"] is None


# ── KeywordsRetrieve ─────────────────────────────────────────────────


async def test_keywords_retrieve_searches_owned_collections_only() -> None:
    client = _mock_client()
    client.list_collections.return_value = [
        "weknora_embeddings_4",
        "other_collection",
        "weknora_embeddings_8",
    ]
    client.search.return_value = [[_search_row()]]
    repo = _repo(client)
    params = RetrieveParams(query="hello", top_k=5, retriever_type=RetrieverType.KEYWORDS)
    await repo.keywords_retrieve(_CTX, params)
    assert client.search.call_count == 2
    for call in client.search.call_args_list:
        assert call.kwargs["anns_field"] == FIELD_CONTENT_SPARSE


async def test_keywords_retrieve_truncates_to_top_k() -> None:
    client = _mock_client()
    client.list_collections.return_value = [_COLLECTION]
    client.search.return_value = [
        [_search_row(row_id=f"r{i}") for i in range(10)]
    ]
    repo = _repo(client)
    params = RetrieveParams(query="hello", top_k=3, retriever_type=RetrieverType.KEYWORDS)
    results = await repo.keywords_retrieve(_CTX, params)
    assert len(results[0].results) == 3


async def test_keywords_retrieve_skips_failed_collection() -> None:
    client = _mock_client()
    client.list_collections.return_value = [_COLLECTION]
    client.search.side_effect = RuntimeError("boom")
    repo = _repo(client)
    params = RetrieveParams(query="hello", top_k=5, retriever_type=RetrieverType.KEYWORDS)
    results = await repo.keywords_retrieve(_CTX, params)
    assert results[0].results == []


# ── Delete by * ──────────────────────────────────────────────────────


async def test_delete_by_chunk_id_list_calls_delete() -> None:
    client = _mock_client()
    repo = _repo(client)
    await repo.delete_by_chunk_id_list(_CTX, ["c1", "c2"], _DIM, "manual")
    client.delete.assert_called_once()
    kwargs = client.delete.call_args.kwargs
    assert kwargs["collection_name"] == _COLLECTION
    assert FIELD_CHUNK_ID in kwargs["filter"]
    assert kwargs["filter_params"]


async def test_delete_by_knowledge_id_list_calls_delete() -> None:
    client = _mock_client()
    repo = _repo(client)
    await repo.delete_by_knowledge_id_list(_CTX, ["k1"], _DIM, "manual")
    assert FIELD_KNOWLEDGE_ID in client.delete.call_args.kwargs["filter"]


async def test_delete_by_source_id_list_calls_delete() -> None:
    client = _mock_client()
    repo = _repo(client)
    await repo.delete_by_source_id_list(_CTX, ["s1"], _DIM, "manual")
    assert FIELD_SOURCE_ID in client.delete.call_args.kwargs["filter"]


async def test_delete_by_chunk_id_list_empty_is_noop() -> None:
    client = _mock_client()
    repo = _repo(client)
    await repo.delete_by_chunk_id_list(_CTX, [], _DIM, "manual")
    client.delete.assert_not_called()


# ── CopyIndices ──────────────────────────────────────────────────────


async def test_copy_indices_copies_and_remaps_ids() -> None:
    client = _mock_client()
    client.has_collection.return_value = True
    # First page returns 2 rows, second returns empty (count < batch).
    row1 = _search_row(row_id="r1", chunk_id="src-chunk", is_enabled=True)
    row1["knowledge_id"] = "src-k"
    row2 = _search_row(row_id="r2", chunk_id="src-chunk-2", is_enabled=False)
    row2["knowledge_id"] = "src-k"
    client.query.side_effect = [[row1, row2], []]
    repo = _repo(client)
    await repo.copy_indices(
        _CTX,
        "src-kb",
        {"src-k": "tgt-k"},
        {"src-chunk": "tgt-chunk", "src-chunk-2": "tgt-chunk-2"},
        "tgt-kb",
        _DIM,
        "manual",
    )
    client.upsert.assert_called_once()
    data = client.upsert.call_args.kwargs["data"]
    assert len(data) == 2
    assert data[0][FIELD_CHUNK_ID] == "tgt-chunk"
    assert data[0][FIELD_KNOWLEDGE_ID] == "tgt-k"
    assert data[0][FIELD_KNOWLEDGE_BASE_ID] == "tgt-kb"


async def test_copy_indices_empty_mapping_is_noop() -> None:
    client = _mock_client()
    repo = _repo(client)
    await repo.copy_indices(_CTX, "kb", {}, {}, "tgt", _DIM, "manual")
    client.upsert.assert_not_called()


async def test_copy_indices_resolves_question_source_id() -> None:
    client = _mock_client()
    client.has_collection.return_value = True
    row = _search_row(row_id="r1", chunk_id="src-chunk")
    row["source_id"] = "src-chunk-q42"
    client.query.side_effect = [[row], []]
    repo = _repo(client)
    await repo.copy_indices(
        _CTX,
        "src-kb",
        {"k-1": "tgt-k"},
        {"src-chunk": "tgt-chunk"},
        "tgt-kb",
        _DIM,
        "manual",
    )
    data = client.upsert.call_args.kwargs["data"]
    assert data[0]["source_id"] == "tgt-chunk-q42"


# ── BatchUpdateChunkEnabledStatus ────────────────────────────────────


async def test_batch_update_chunk_enabled_status_updates_all_collections() -> None:
    client = _mock_client()
    client.list_collections.return_value = [
        "weknora_embeddings_4",
        "weknora_embeddings_8",
        "unrelated",
    ]
    client.query.return_value = [_search_row(row_id="r1", chunk_id="c1")]
    repo = _repo(client)
    await repo.batch_update_chunk_enabled_status(_CTX, {"c1": True, "c2": False})
    # 2 owned collections * (enabled + disabled) = 4 query calls,
    # but disabled list [c2] returns empty (no rows) so only enabled upserts.
    assert client.upsert.call_count >= 1


async def test_batch_update_chunk_enabled_status_empty_is_noop() -> None:
    client = _mock_client()
    repo = _repo(client)
    await repo.batch_update_chunk_enabled_status(_CTX, {})
    client.list_collections.assert_not_called()


async def test_batch_update_chunk_enabled_status_propagates_error() -> None:
    client = _mock_client()
    client.list_collections.return_value = [_COLLECTION]
    client.query.side_effect = RuntimeError("boom")
    repo = _repo(client)
    with pytest.raises(RuntimeError, match="boom"):
        await repo.batch_update_chunk_enabled_status(_CTX, {"c1": True})


# ── BatchUpdateChunkTagID ─────────────────────────────────────────────


async def test_batch_update_chunk_tag_id_updates_tag() -> None:
    client = _mock_client()
    client.list_collections.return_value = [_COLLECTION]
    client.query.return_value = [_search_row(row_id="r1", chunk_id="c1", tag_id="old")]
    repo = _repo(client)
    await repo.batch_update_chunk_tag_id(_CTX, {"c1": "new-tag"})
    client.upsert.assert_called_once()
    data = client.upsert.call_args.kwargs["data"]
    assert data[0][FIELD_TAG_ID] == "new-tag"


async def test_batch_update_chunk_tag_id_empty_is_noop() -> None:
    client = _mock_client()
    repo = _repo(client)
    await repo.batch_update_chunk_tag_id(_CTX, {})
    client.list_collections.assert_not_called()


async def test_batch_update_chunk_tag_id_skips_failed_collection() -> None:
    client = _mock_client()
    client.list_collections.return_value = [_COLLECTION]
    client.query.side_effect = RuntimeError("boom")
    repo = _repo(client)
    # Should not raise - per-collection failures are logged and skipped.
    await repo.batch_update_chunk_tag_id(_CTX, {"c1": "tag"})
    client.upsert.assert_not_called()


# ── EstimateStorageSize ──────────────────────────────────────────────


def test_estimate_storage_size_sums_all_items() -> None:
    client = _mock_client()
    repo = _repo(client)
    items = [_index_info(source_id=f"s{i}") for i in range(3)]
    params = {"embedding": {f"s{i}": [0.1] * 4 for i in range(3)}}
    total = repo.estimate_storage_size(_CTX, items, params)
    assert total > 0
    # Each item: 4 dims * 4 bytes = 16 vector + 16 index + 32 meta + payload
    single = _calculate_storage_size(
        _to_milvus_vector_embedding(items[0], params)
    )
    assert total == single * 3


def test_estimate_storage_size_without_embedding() -> None:
    client = _mock_client()
    repo = _repo(client)
    items = [_index_info()]
    total = repo.estimate_storage_size(_CTX, items, {})
    # No vector: only payload + metadata(32)
    assert total >= 32


# ── Filter converter ─────────────────────────────────────────────────


def test_filter_eq_uses_template() -> None:
    converter = MilvusFilterConverter()
    result = converter.convert(
        UniversalFilterCondition(field="is_enabled", operator="eq", value=True)
    )
    assert "is_enabled == {" in result.expr_str
    assert len(result.params) == 1
    assert next(iter(result.params.values())) is True


def test_filter_in_builds_in_expression() -> None:
    converter = MilvusFilterConverter()
    result = converter.convert(
        UniversalFilterCondition(field="chunk_id", operator="in", value=["a", "b"])
    )
    assert "chunk_id in {" in result.expr_str
    assert next(iter(result.params.values())) == ["a", "b"]


def test_filter_not_in_builds_not_in_expression() -> None:
    converter = MilvusFilterConverter()
    result = converter.convert(
        UniversalFilterCondition(field="chunk_id", operator="not in", value=["a"])
    )
    assert "chunk_id not in {" in result.expr_str


def test_filter_and_joins_children() -> None:
    converter = MilvusFilterConverter()
    result = converter.convert(
        UniversalFilterCondition(
            operator="and",
            value=[
                UniversalFilterCondition(field="a", operator="eq", value=1),
                UniversalFilterCondition(field="b", operator="eq", value=2),
            ],
        )
    )
    assert "and" in result.expr_str
    assert len(result.params) == 2


def test_filter_between_builds_range() -> None:
    converter = MilvusFilterConverter()
    result = converter.convert(
        UniversalFilterCondition(field="x", operator="between", value=[1, 10])
    )
    assert ">=" in result.expr_str
    assert "<=" in result.expr_str
    assert len(result.params) == 2


def test_filter_rejects_nil_condition() -> None:
    converter = MilvusFilterConverter()
    with pytest.raises(ValidationError, match="condition is nil"):
        converter.convert(None)


def test_filter_rejects_unknown_operator() -> None:
    converter = MilvusFilterConverter()
    with pytest.raises(ValidationError, match="unsupported operator"):
        converter.convert(
            UniversalFilterCondition(field="x", operator="bogus", value=1)
        )


def test_filter_in_rejects_non_list() -> None:
    converter = MilvusFilterConverter()
    with pytest.raises(ValidationError, match="must be a slice"):
        converter.convert(
            UniversalFilterCondition(field="x", operator="in", value="not-a-list")
        )


def test_filter_in_rejects_empty_list() -> None:
    converter = MilvusFilterConverter()
    with pytest.raises(ValidationError, match="must be a slice"):
        converter.convert(
            UniversalFilterCondition(field="x", operator="in", value=[])
        )


def test_filter_param_name_replaces_dots() -> None:
    converter = MilvusFilterConverter()
    result = converter.convert(
        UniversalFilterCondition(field="meta.field", operator="eq", value=1)
    )
    assert "meta_field_" in result.expr_str


def test_filter_nested_and_or() -> None:
    converter = MilvusFilterConverter()
    result = converter.convert(
        UniversalFilterCondition(
            operator="or",
            value=[
                UniversalFilterCondition(
                    operator="and",
                    value=[
                        UniversalFilterCondition(field="a", operator="eq", value=1),
                        UniversalFilterCondition(field="b", operator="eq", value=2),
                    ],
                ),
                UniversalFilterCondition(field="c", operator="eq", value=3),
            ],
        )
    )
    assert "or" in result.expr_str
    assert "and" in result.expr_str
    assert len(result.params) == 3


# ── Helpers ──────────────────────────────────────────────────────────


def test_to_milvus_vector_embedding_pulls_from_map() -> None:
    info = _index_info(source_id="src-1")
    emb = _to_milvus_vector_embedding(info, {"embedding": {"src-1": [0.1, 0.2]}})
    assert emb.embedding == [0.1, 0.2]
    assert emb.content == "hello"
    assert emb.chunk_id == "chunk-1"


def test_to_milvus_vector_embedding_missing_source_returns_empty() -> None:
    info = _index_info(source_id="src-1")
    emb = _to_milvus_vector_embedding(info, {"embedding": {"other": [0.1]}})
    assert emb.embedding == []


def test_to_upsert_row_excludes_sparse_field() -> None:
    emb = MilvusVectorEmbedding(id="x", content="c", embedding=[0.1])
    row = _to_upsert_row(emb)
    assert FIELD_CONTENT_SPARSE not in row
    assert row[FIELD_ID] == "x"
    assert row[FIELD_EMBEDDING] == [0.1]


def test_from_milvus_vector_embedding_maps_fields() -> None:
    from src.ai.retrieval.milvus import MilvusVectorEmbeddingWithScore

    emb = MilvusVectorEmbedding(
        id="rid",
        content="hit",
        source_id="s1",
        source_type=1,
        chunk_id="c1",
        knowledge_id="k1",
        knowledge_base_id="kb1",
        tag_id="t1",
        is_enabled=True,
    )
    scored = MilvusVectorEmbeddingWithScore(embedding=emb, score=0.8)
    result = _from_milvus_vector_embedding("rid", scored, MatchType.EMBEDDING)
    assert result.id == "rid"
    assert result.content == "hit"
    assert result.score == 0.8
    assert result.match_type == MatchType.EMBEDDING
    assert result.is_enabled is True
    assert result.source_type == SourceType.PASSAGE


def test_calculate_storage_size_with_embedding() -> None:
    emb = MilvusVectorEmbedding(content="hello", embedding=[0.1] * 4)
    size = _calculate_storage_size(emb)
    # payload(5 + 8 for source_type) + vector(16) + index(16+16) + meta(32) = 93
    assert size == 93


def test_calculate_storage_size_without_embedding() -> None:
    emb = MilvusVectorEmbedding(content="hello")
    size = _calculate_storage_size(emb)
    # payload(5 + 8) + 0 + 0 + meta(32) = 45
    assert size == 45


# ── Constructor ──────────────────────────────────────────────────────


async def test_new_milvus_retrieve_engine_repository_builds_client() -> None:
    cfg = MilvusClientConfig(
        address="milvus:19530", username="root", password="pw", db_name="db1"
    )
    with patch("src.ai.retrieval.milvus.MilvusClient") as mock_ctor:
        mock_client = MagicMock()
        mock_ctor.return_value = mock_client
        repo = await new_milvus_retrieve_engine_repository(cfg, None)
    mock_ctor.assert_called_once_with(
        uri="milvus:19530", user="root", password="pw", db_name="db1"
    )
    assert isinstance(repo, MilvusRetrieveEngineRepository)


async def test_new_milvus_retrieve_engine_repository_minimal_config() -> None:
    cfg = MilvusClientConfig()
    with patch("src.ai.retrieval.milvus.MilvusClient") as mock_ctor:
        mock_ctor.return_value = MagicMock()
        await new_milvus_retrieve_engine_repository(cfg, None)
    mock_ctor.assert_called_once_with(uri="localhost:19530")


async def test_new_milvus_retrieve_engine_repository_with_index_config() -> None:
    cfg = MilvusClientConfig()
    with patch("src.ai.retrieval.milvus.MilvusClient") as mock_ctor:
        mock_ctor.return_value = MagicMock()
        repo = await new_milvus_retrieve_engine_repository(
            cfg, IndexConfig(collection_name="custom")
        )
    assert repo._collection_base_name == "custom"


# ── Retrieve result shape ────────────────────────────────────────────


async def test_vector_retrieve_result_has_engine_type() -> None:
    client = _mock_client()
    client.has_collection.return_value = True
    client.search.return_value = [[]]
    repo = _repo(client)
    params = RetrieveParams(embedding=[0.1] * _DIM, top_k=5)
    results = await repo.vector_retrieve(_CTX, params)
    assert results[0].retriever_engine_type == RetrieverEngineType.MILVUS
    assert results[0].error is None
