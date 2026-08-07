"""Tests for the Weaviate retrieval engine repository.

The ``WeaviateAsyncClient`` is mocked with an ``AsyncMock`` double; no live
Weaviate instance is contacted. Pinned here: collection creation with the
``self_provided`` vector + HNSW settings, save / batch_save (per-dimension
bucketing + replication / sharding config), vector and keyword retrieval
(filters / certainty threshold / top-k), delete / batch update, copy-indices
(source-id preservation + base-name filtering), storage-size estimate, the
filter helpers, and the factory + env wiring.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from weaviate.classes.config import (
    Configure,
    DataType,
    Property,
    VectorDistances,
)
from weaviate.classes.query import MetadataQuery

from src.ai.embedding import TaskContext
from src.ai.retrieval.factory import WeaviateClientConfig
from src.ai.retrieval.types import (
    IndexConfig,
    IndexInfo,
    MatchType,
    RetrieveParams,
    RetrieverEngineType,
    RetrieverType,
    SourceType,
)
from src.ai.retrieval.weaviate import (
    FIELD_CHUNK_ID,
    FIELD_CONTENT,
    FIELD_IS_ENABLED,
    FIELD_KNOWLEDGE_BASE_ID,
    FIELD_KNOWLEDGE_ID,
    FIELD_SOURCE_ID,
    FIELD_SOURCE_TYPE,
    FIELD_TAG_ID,
    VECTOR_NAME,
    WeaviateRetrieveEngineRepository,
    _calculate_storage_size,
    _collection_name,
    _resolve_collection_name,
    _resolve_target_source_id,
    new_weaviate_retrieve_engine_repository,
    new_weaviate_retrieve_engine_repository_from_env,
)

_CTX = TaskContext()


# ── Mock helpers ──────────────────────────────────────────────────────


def _obj(
    uuid: str,
    properties: dict[str, Any] | None,
    *,
    certainty: float = 0.0,
    score: float = 0.0,
    vector: Any = None,
) -> MagicMock:
    """Build a minimal mock for a Weaviate query return object."""
    obj = MagicMock()
    obj.uuid = uuid
    obj.properties = properties or {}
    meta = MagicMock()
    meta.certainty = certainty
    meta.score = score
    obj.metadata = meta
    obj.vector = vector
    return obj


class _FakeQueryReturn:
    """Async-result wrapper for query responses (mirrors Weaviate v4)."""

    def __init__(self, objects: list[MagicMock]) -> None:
        self.objects = objects


class _FakeCollection:
    """Mock collection handle (async-capable)."""

    def __init__(self) -> None:
        self.data = MagicMock()
        self.data.insert = AsyncMock(return_value=None)
        self.data.insert_many = AsyncMock(return_value=MagicMock())
        self.data.delete_many = AsyncMock(return_value=MagicMock())
        self.data.update = AsyncMock(return_value=None)
        self.query = MagicMock()
        self.query.near_vector = AsyncMock()
        self.query.bm25 = AsyncMock()
        self.query.fetch_objects = AsyncMock()


class _FakeClient:
    """Async-capable mock of the subset of ``WeaviateAsyncClient`` the repo uses."""

    def __init__(self) -> None:
        self.collections = MagicMock()
        self.collections.exists = AsyncMock(return_value=False)
        self.collections.create = AsyncMock(return_value=None)
        self.collections.list_all = AsyncMock(return_value=[])
        self.collections.get = MagicMock(return_value=_FakeCollection())
        self._collection_handle = _FakeCollection()
        self.close = AsyncMock(return_value=None)

    # The v4 client exposes ``client.collection(name)`` as a shortcut; we
    # emulate both shapes.
    def collection(self, name: str) -> _FakeCollection:
        return self._collection_handle


def _repo(
    base: str = "weknora_embeddings",
    *,
    client: _FakeClient | None = None,
    replication: int = 0,
    shard: int = 0,
    ef_construction: int = 128,
    ef: int = 64,
    m: int = 32,
) -> WeaviateRetrieveEngineRepository:
    fake = client or _FakeClient()
    return WeaviateRetrieveEngineRepository(
        client=fake,  # type: ignore[arg-type]
        collection_base_name=base,
        replication_factor=replication,
        desired_shard_count=shard,
        hnsw_ef_construction=ef_construction,
        hnsw_ef=ef,
        hnsw_m=m,
    )


def _index_info(
    source_id: str = "src-1",
    chunk_id: str = "chunk-1",
    knowledge_id: str = "know-1",
    knowledge_base_id: str = "kb-1",
    content: str = "hello world",
    tag_id: str = "",
    source_type: SourceType = SourceType.CHUNK,
    is_enabled: bool = True,
) -> IndexInfo:
    return IndexInfo(
        id=source_id,
        content=content,
        source_id=source_id,
        source_type=source_type,
        chunk_id=chunk_id,
        knowledge_id=knowledge_id,
        knowledge_base_id=knowledge_base_id,
        tag_id=tag_id,
        is_enabled=is_enabled,
    )


def _save_params(source_id: str, embedding: list[float]) -> dict[str, dict[str, list[float]]]:
    return {"embedding": {source_id: embedding}}


def _collection_entry(name: str) -> MagicMock:
    """Build a MagicMock entry whose ``.name`` attribute equals ``name``.

    The retrieval layer reads each entry's ``.name`` via ``getattr``; using
    ``MagicMock(name=...)`` alone leaks a child MagicMock on access instead of
    the desired string.
    """
    m = MagicMock()
    m.name = name
    return m


# ── Pure helpers ──────────────────────────────────────────────────────


def test_resolve_collection_name_prefers_collection_prefix() -> None:
    cfg = IndexConfig(collection_prefix="prefix", collection_name="name")
    assert _resolve_collection_name(cfg) == "prefix"


def test_resolve_collection_name_falls_back_to_collection_name() -> None:
    cfg = IndexConfig(collection_prefix="", collection_name="name")
    assert _resolve_collection_name(cfg) == "name"


def test_resolve_collection_name_uses_env_when_no_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WEAVIATE_COLLECTION", "env_coll")
    assert _resolve_collection_name(None) == "env_coll"


def test_resolve_collection_name_uses_default_when_nothing_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WEAVIATE_COLLECTION", raising=False)
    assert _resolve_collection_name(None) == "weknora_embeddings"


def test_resolve_collection_name_config_overrides_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WEAVIATE_COLLECTION", "env_coll")
    cfg = IndexConfig(collection_prefix="prefix")
    assert _resolve_collection_name(cfg) == "prefix"


def test_collection_name_combines_base_and_dimension() -> None:
    assert _collection_name("weknora_embeddings", 768) == "weknora_embeddings_768"


def test_calculate_storage_size_includes_vector_and_id_tracker() -> None:
    size_no_vec = _calculate_storage_size("abcde", "s", "c", "k", "kb", [])
    assert size_no_vec == len("abcde") + len("s") + len("c") + len("k") + len("kb") + 8 + 24

    size_with_vec = _calculate_storage_size(
        "abcde", "s", "c", "k", "kb", [0.0] * 16, hnsw_m=32
    )
    assert size_with_vec == size_no_vec + 16 * 4 + 32 * 2 * 8


def test_resolve_target_source_id_preserves_question_suffix() -> None:
    assert _resolve_target_source_id("c1", "c1", "tc1") == "tc1"
    assert _resolve_target_source_id("c1-q1", "c1", "tc1") == "tc1-q1"
    # Mismatched prefix falls back to a new UUID (not asserted exact value).
    new_id = _resolve_target_source_id("other", "c1", "tc1")
    assert new_id != "tc1"
    assert new_id != "other"


# ── Construction / engine_type / support ──────────────────────────────


def test_engine_type_is_weaviate() -> None:
    r = _repo()
    assert r.engine_type() == RetrieverEngineType.WEAVIATE


def test_supports_keywords_and_vector() -> None:
    r = _repo()
    assert set(r.support()) == {RetrieverType.KEYWORDS, RetrieverType.VECTOR}


# ── Property schema ──────────────────────────────────────────────────


def test_property_schema_marks_filterable_and_searchable_correctly() -> None:
    schema = WeaviateRetrieveEngineRepository._build_properties_schema()
    by_name = {p.name: p for p in schema}
    # ``content`` is the searchable text field (BM25) but not filterable.
    assert isinstance(by_name[FIELD_CONTENT], Property)
    assert by_name[FIELD_CONTENT].dataType == DataType.TEXT
    assert by_name[FIELD_CONTENT].indexSearchable is True
    assert by_name[FIELD_CONTENT].indexFilterable is False
    # The five filterable id-style fields share the same shape.
    for name in (
        FIELD_CHUNK_ID,
        FIELD_KNOWLEDGE_ID,
        FIELD_KNOWLEDGE_BASE_ID,
        FIELD_TAG_ID,
        FIELD_IS_ENABLED,
    ):
        prop = by_name[name]
        assert prop.indexFilterable is True, name
        assert prop.indexSearchable is False, name
    assert by_name[FIELD_IS_ENABLED].dataType == DataType.BOOL
    assert by_name[FIELD_SOURCE_TYPE].dataType == DataType.INT
    assert by_name[FIELD_SOURCE_ID].dataType == DataType.TEXT


# ── Filter assembly ──────────────────────────────────────────────────


def test_build_base_filter_only_enabled_when_no_constraints() -> None:
    params = RetrieveParams(retriever_type=RetrieverType.VECTOR)
    flt = WeaviateRetrieveEngineRepository._build_base_filter(params)
    assert flt is not None
    # A single operand is returned unwrapped (the ``_FilterByProperty`` builder).
    assert hasattr(flt, "operator") or hasattr(flt, "value")


def test_build_base_filter_combines_kb_kw_tag_and_excludes() -> None:
    params = RetrieveParams(
        knowledge_base_ids=["kb-1"],
        knowledge_ids=["k-1"],
        tag_ids=["t-1"],
        exclude_knowledge_ids=["x-1"],
        exclude_chunk_ids=["xc-1"],
        retriever_type=RetrieverType.VECTOR,
    )
    flt = WeaviateRetrieveEngineRepository._build_base_filter(params)
    assert flt is not None


# ── Save / batch_save ────────────────────────────────────────────────


async def test_save_creates_collection_with_expected_vector_config() -> None:
    client = _FakeClient()
    client.collections.exists.return_value = False
    r = _repo(client=client)
    info = _index_info(source_id="s-1")
    await r.save(_CTX, info, _save_params("s-1", [0.1] * 4))
    client.collections.create.assert_awaited_once()
    kwargs = client.collections.create.await_args.kwargs
    assert kwargs["name"] == "weknora_embeddings_4"
    assert kwargs["vector_config"] == Configure.Vectors.self_provided(
        name=VECTOR_NAME,
        vector_index_config=Configure.VectorIndex.hnsw(
            distance_metric=VectorDistances.COSINE,
            ef_construction=128,
            ef=64,
            max_connections=32,
        ),
    )
    assert "replication_config" not in kwargs
    assert "sharding_config" not in kwargs
    properties = kwargs["properties"]
    schema_names = {prop.name for prop in properties}
    assert schema_names == {
        FIELD_CONTENT,
        FIELD_SOURCE_ID,
        FIELD_SOURCE_TYPE,
        FIELD_CHUNK_ID,
        FIELD_KNOWLEDGE_ID,
        FIELD_KNOWLEDGE_BASE_ID,
        FIELD_TAG_ID,
        FIELD_IS_ENABLED,
    }


async def test_save_passes_replication_and_sharding_when_configured() -> None:
    client = _FakeClient()
    client.collections.exists.return_value = False
    r = _repo(client=client, replication=3, shard=5)
    info = _index_info(source_id="s-1")
    await r.save(_CTX, info, _save_params("s-1", [0.1] * 4))
    kwargs = client.collections.create.await_args.kwargs
    assert kwargs["replication_config"].factor == 3
    assert kwargs["sharding_config"].desiredCount == 5


async def test_save_skips_creation_when_collection_already_exists() -> None:
    client = _FakeClient()
    client.collections.exists.return_value = True
    r = _repo(client=client)
    info = _index_info(source_id="s-1")
    await r.save(_CTX, info, _save_params("s-1", [0.1] * 4))
    client.collections.create.assert_not_awaited()


async def test_save_rejects_empty_embedding() -> None:
    r = _repo()
    info = _index_info(source_id="s-1")
    with pytest.raises(ValueError, match="empty embedding"):
        await r.save(_CTX, info, _save_params("s-1", []))


async def test_save_inserts_object_with_properties_and_vector() -> None:
    client = _FakeClient()
    client.collections.exists.return_value = True
    r = _repo(client=client)
    info = _index_info(source_id="s-1", chunk_id="c-1", knowledge_id="k-1")
    await r.save(_CTX, info, _save_params("s-1", [0.1, 0.2, 0.3, 0.4]))
    collection = client.collections.get.return_value
    insert = collection.data.insert.await_args
    assert insert.kwargs["uuid"] == "c-1"
    assert insert.kwargs["properties"][FIELD_CHUNK_ID] == "c-1"
    assert insert.kwargs["properties"][FIELD_KNOWLEDGE_ID] == "k-1"
    assert insert.kwargs["properties"][FIELD_IS_ENABLED] is True
    assert insert.kwargs["vector"] == {VECTOR_NAME: [0.1, 0.2, 0.3, 0.4]}


async def test_batch_save_groups_objects_by_dimension() -> None:
    client = _FakeClient()
    client.collections.exists.return_value = True
    r = _repo(client=client)
    infos = [
        _index_info(source_id="a", chunk_id="ca"),
        _index_info(source_id="b", chunk_id="cb"),
        _index_info(source_id="c", chunk_id="cc"),
    ]
    params = {
        "embedding": {
            "a": [0.1] * 4,
            "b": [0.1] * 8,
            "c": [0.1] * 8,
        }
    }
    await r.batch_save(_CTX, infos, params)
    insert_many_calls = client.collections.get.return_value.data.insert_many.await_args_list
    # Two insert_many invocations, one per dimension.
    assert len(insert_many_calls) == 2


async def test_batch_save_skips_empty_embeddings() -> None:
    client = _FakeClient()
    client.collections.exists.return_value = True
    r = _repo(client=client)
    infos = [_index_info(source_id="a"), _index_info(source_id="b")]
    params = {"embedding": {"a": [0.1] * 4}}
    await r.batch_save(_CTX, infos, params)
    insert_many_calls = client.collections.get.return_value.data.insert_many.await_args_list
    assert len(insert_many_calls) == 1


async def test_batch_save_no_op_on_empty_list() -> None:
    client = _FakeClient()
    r = _repo(client=client)
    await r.batch_save(_CTX, [], _save_params("x", [0.1] * 4))
    client.collections.get.return_value.data.insert_many.assert_not_awaited()


# ── Vector retrieval ─────────────────────────────────────────────────


async def test_vector_retrieve_returns_empty_when_collection_missing() -> None:
    client = _FakeClient()
    client.collections.exists.return_value = False
    r = _repo(client=client)
    params = RetrieveParams(
        query="hi",
        embedding=[0.1] * 4,
        top_k=3,
        retriever_type=RetrieverType.VECTOR,
    )
    results = await r.retrieve(_CTX, params)
    assert len(results) == 1
    assert results[0].results == []
    client.collections.exists.assert_awaited()


async def test_vector_retrieve_queries_with_filter_and_threshold() -> None:
    client = _FakeClient()
    client.collections.exists.return_value = True
    client.collections.get.return_value.query.near_vector.return_value = _FakeQueryReturn([])
    r = _repo(client=client)
    params = RetrieveParams(
        query="hi",
        embedding=[0.1] * 4,
        knowledge_base_ids=["kb-1"],
        knowledge_ids=["k-1"],
        tag_ids=["t-1"],
        exclude_knowledge_ids=["x-1"],
        exclude_chunk_ids=["xc-1"],
        top_k=7,
        threshold=0.5,
        retriever_type=RetrieverType.VECTOR,
    )
    await r.retrieve(_CTX, params)
    near = client.collections.get.return_value.query.near_vector.await_args
    assert near.kwargs["near_vector"] == {VECTOR_NAME: [0.1] * 4}
    assert near.kwargs["limit"] == 7
    assert near.kwargs["certainty"] == 0.5
    assert near.kwargs["filters"] is not None
    assert isinstance(near.kwargs["return_metadata"], MetadataQuery)


async def test_vector_retrieve_omits_threshold_when_zero() -> None:
    client = _FakeClient()
    client.collections.exists.return_value = True
    client.collections.get.return_value.query.near_vector.return_value = _FakeQueryReturn([])
    r = _repo(client=client)
    params = RetrieveParams(
        embedding=[0.1] * 4,
        top_k=5,
        threshold=0.0,
        retriever_type=RetrieverType.VECTOR,
    )
    await r.retrieve(_CTX, params)
    near = client.collections.get.return_value.query.near_vector.await_args
    assert "certainty" not in near.kwargs


async def test_vector_retrieve_converts_objects_to_index_with_score() -> None:
    client = _FakeClient()
    client.collections.exists.return_value = True
    properties = {
        FIELD_CONTENT: "hi",
        FIELD_SOURCE_ID: "s",
        FIELD_SOURCE_TYPE: 0,
        FIELD_CHUNK_ID: "c",
        FIELD_KNOWLEDGE_ID: "k",
        FIELD_KNOWLEDGE_BASE_ID: "kb",
        FIELD_TAG_ID: "t",
        FIELD_IS_ENABLED: True,
    }
    client.collections.get.return_value.query.near_vector.return_value = _FakeQueryReturn(
        [
            _obj("p-1", properties, certainty=0.9),
            _obj("p-2", properties, certainty=0.7),
        ]
    )
    r = _repo(client=client)
    params = RetrieveParams(
        query="hi",
        embedding=[0.1] * 4,
        top_k=5,
        retriever_type=RetrieverType.VECTOR,
    )
    results = await r.retrieve(_CTX, params)
    assert len(results[0].results) == 2
    assert results[0].results[0].score == 0.9
    assert results[0].results[0].match_type == MatchType.EMBEDDING
    assert results[0].results[0].chunk_id == "c"
    assert results[0].retriever_type == RetrieverType.VECTOR


# ── Keywords retrieval ───────────────────────────────────────────────


async def test_keywords_retrieve_filters_collections_by_base_name() -> None:
    client = _FakeClient()

    def _collection_entry(name: str) -> MagicMock:
        m = MagicMock()
        m.name = name
        return m

    client.collections.list_all.return_value = [
        _collection_entry("weknora_embeddings_4"),
        _collection_entry("weknora_embeddings_8"),
        _collection_entry("other_collection"),
    ]
    client.collections.get.return_value.query.bm25.return_value = _FakeQueryReturn([])
    r = _repo(client=client)
    params = RetrieveParams(query="hello", top_k=3, retriever_type=RetrieverType.KEYWORDS)
    await r.retrieve(_CTX, params)
    bm25_calls = client.collections.get.return_value.query.bm25.await_args_list
    # 2 collections share the same get() handle; one query per collection.
    assert len(bm25_calls) == 2
    call = bm25_calls[0]
    assert call.kwargs["query"] == "hello"
    assert call.kwargs["query_properties"] == [FIELD_CONTENT]
    assert call.kwargs["limit"] == 3
    assert call.kwargs["filters"] is not None


async def test_keywords_retrieve_caps_results_to_top_k() -> None:
    client = _FakeClient()

    def _collection_entry(name: str) -> MagicMock:
        m = MagicMock()
        m.name = name
        return m

    client.collections.list_all.return_value = [
        _collection_entry("weknora_embeddings_4")
    ]
    properties = {FIELD_CONTENT: "x", FIELD_CHUNK_ID: "c"}
    client.collections.get.return_value.query.bm25.return_value = _FakeQueryReturn(
        [_obj(f"p-{i}", properties) for i in range(5)]
    )
    r = _repo(client=client)
    params = RetrieveParams(query="x", top_k=2, retriever_type=RetrieverType.KEYWORDS)
    results = await r.retrieve(_CTX, params)
    assert len(results[0].results) == 2
    assert results[0].results[0].match_type == MatchType.KEYWORDS


async def test_keywords_retrieve_returns_empty_when_no_match() -> None:
    client = _FakeClient()
    client.collections.list_all.return_value = [
        _collection_entry("other_collection")
    ]
    r = _repo(client=client)
    params = RetrieveParams(query="x", top_k=3, retriever_type=RetrieverType.KEYWORDS)
    results = await r.retrieve(_CTX, params)
    assert results[0].results == []
    client.collections.get.return_value.query.bm25.assert_not_awaited()


# ── Retrieve dispatch ────────────────────────────────────────────────


async def test_retrieve_dispatch_rejects_unknown_retriever_type() -> None:
    r = _repo()
    params = RetrieveParams(retriever_type=RetrieverType.VECTOR)
    object.__setattr__(params, "retriever_type", "bogus")
    with pytest.raises(ValueError, match="invalid retriever type"):
        await r.retrieve(_CTX, params)


# ── Delete ────────────────────────────────────────────────────────────


async def test_delete_by_chunk_id_list_uses_contains_any() -> None:
    client = _FakeClient()
    client.collections.exists.return_value = True
    r = _repo(client=client)
    await r.delete_by_chunk_id_list(_CTX, ["c-1", "c-2"], 4, "")
    flt = client.collections.get.return_value.data.delete_many.await_args.kwargs["where"]
    assert flt is not None


async def test_delete_by_chunk_id_list_skips_empty_input() -> None:
    client = _FakeClient()
    r = _repo(client=client)
    await r.delete_by_chunk_id_list(_CTX, [], 4, "")
    client.collections.get.return_value.data.delete_many.assert_not_awaited()


async def test_delete_by_source_id_list_targets_source_field() -> None:
    client = _FakeClient()
    client.collections.exists.return_value = True
    r = _repo(client=client)
    await r.delete_by_source_id_list(_CTX, ["s-1"], 4, "")
    flt = client.collections.get.return_value.data.delete_many.await_args.kwargs["where"]
    assert flt is not None


async def test_delete_by_knowledge_id_list_targets_knowledge_field() -> None:
    client = _FakeClient()
    client.collections.exists.return_value = True
    r = _repo(client=client)
    await r.delete_by_knowledge_id_list(_CTX, ["k-1"], 4, "")
    flt = client.collections.get.return_value.data.delete_many.await_args.kwargs["where"]
    assert flt is not None


# ── Batch update ─────────────────────────────────────────────────────


async def test_batch_update_chunk_enabled_status_updates_all_collections() -> None:
    client = _FakeClient()
    client.collections.list_all.return_value = [
        _collection_entry("weknora_embeddings_4"),
        _collection_entry("weknora_embeddings_8"),
        _collection_entry("unrelated"),
    ]
    r = _repo(client=client)
    await r.batch_update_chunk_enabled_status(_CTX, {"c-1": True, "c-2": False})
    update_calls = client.collections.get.return_value.data.update.await_args_list
    # Two payload writes per matching collection: two matching collections.
    assert len(update_calls) == 4


async def test_batch_update_chunk_enabled_status_skips_empty_map() -> None:
    client = _FakeClient()
    r = _repo(client=client)
    await r.batch_update_chunk_enabled_status(_CTX, {})
    client.collections.get.return_value.data.update.assert_not_awaited()


async def test_batch_update_chunk_tag_id_writes_payload() -> None:
    client = _FakeClient()
    client.collections.list_all.return_value = [
        _collection_entry("weknora_embeddings_4")
    ]
    r = _repo(client=client)
    await r.batch_update_chunk_tag_id(_CTX, {"c-1": "tag-a", "c-2": "tag-b"})
    update_calls = client.collections.get.return_value.data.update.await_args_list
    payloads = [c.kwargs["properties"][FIELD_TAG_ID] for c in update_calls]
    assert set(payloads) == {"tag-a", "tag-b"}


# ── Copy indices ─────────────────────────────────────────────────────


async def test_copy_indices_paginates_and_remaps_ids() -> None:
    client = _FakeClient()
    client.collections.exists.return_value = True

    src_payload_a = {
        FIELD_CONTENT: "alpha",
        FIELD_SOURCE_ID: "src-c1",
        FIELD_SOURCE_TYPE: 0,
        FIELD_CHUNK_ID: "c1",
        FIELD_KNOWLEDGE_ID: "k1",
        FIELD_KNOWLEDGE_BASE_ID: "src-kb",
        FIELD_TAG_ID: "",
        FIELD_IS_ENABLED: True,
    }
    src_payload_b = {
        FIELD_CONTENT: "beta",
        FIELD_SOURCE_ID: "src-c2",
        FIELD_SOURCE_TYPE: 0,
        FIELD_CHUNK_ID: "c2",
        FIELD_KNOWLEDGE_ID: "k2",
        FIELD_KNOWLEDGE_BASE_ID: "src-kb",
        FIELD_TAG_ID: "",
        FIELD_IS_ENABLED: True,
    }
    client.collections.get.return_value.query.fetch_objects.side_effect = [
        _FakeQueryReturn(
            [
                _obj("p-a", src_payload_a, vector={VECTOR_NAME: [0.1] * 4}),
                _obj("p-b", src_payload_b, vector={VECTOR_NAME: [0.2] * 4}),
            ]
        ),
        _FakeQueryReturn([]),
    ]
    r = _repo(client=client)
    await r.copy_indices(
        _CTX,
        source_knowledge_base_id="src-kb",
        source_to_target_kb_id_map={"k1": "tk1", "k2": "tk2"},
        source_to_target_chunk_id_map={"c1": "tc1", "c2": "tc2"},
        target_knowledge_base_id="tgt-kb",
        dimension=4,
        knowledge_type="",
    )
    insert_many = client.collections.get.return_value.data.insert_many
    insert_many.assert_awaited()
    objects = insert_many.await_args.kwargs["objects"]
    assert len(objects) == 2
    target_kbs = {obj["properties"][FIELD_KNOWLEDGE_BASE_ID] for obj in objects}
    assert target_kbs == {"tgt-kb"}
    target_chunks = {obj["properties"][FIELD_CHUNK_ID] for obj in objects}
    assert target_chunks == {"tc1", "tc2"}


async def test_copy_indices_preserves_question_suffix() -> None:
    client = _FakeClient()
    client.collections.exists.return_value = True
    src_payload = {
        FIELD_CONTENT: "q",
        FIELD_SOURCE_ID: "c1-q1",
        FIELD_SOURCE_TYPE: 0,
        FIELD_CHUNK_ID: "c1",
        FIELD_KNOWLEDGE_ID: "k1",
        FIELD_KNOWLEDGE_BASE_ID: "src-kb",
        FIELD_TAG_ID: "",
        FIELD_IS_ENABLED: True,
    }
    client.collections.get.return_value.query.fetch_objects.side_effect = [
        _FakeQueryReturn(
            [_obj("p", src_payload, vector={VECTOR_NAME: [0.1] * 4})]
        ),
        _FakeQueryReturn([]),
    ]
    r = _repo(client=client)
    await r.copy_indices(
        _CTX,
        source_knowledge_base_id="src-kb",
        source_to_target_kb_id_map={"k1": "tk1"},
        source_to_target_chunk_id_map={"c1": "tc1"},
        target_knowledge_base_id="tgt-kb",
        dimension=4,
        knowledge_type="",
    )
    objects = client.collections.get.return_value.data.insert_many.await_args.kwargs["objects"]
    assert objects[0]["properties"][FIELD_SOURCE_ID] == "tc1-q1"


async def test_copy_indices_no_op_on_empty_map() -> None:
    client = _FakeClient()
    r = _repo(client=client)
    await r.copy_indices(_CTX, "src-kb", {}, {}, "tgt-kb", 4, "")
    client.collections.get.return_value.query.fetch_objects.assert_not_awaited()
    client.collections.get.return_value.data.insert_many.assert_not_awaited()


# ── Estimate storage size ─────────────────────────────────────────────


def test_estimate_storage_size_sums_all_entries() -> None:
    r = _repo()
    infos = [_index_info(source_id=f"s-{i}", chunk_id=f"c-{i}") for i in range(3)]
    params = _save_params("s-0", [0.0] * 8)
    size = r.estimate_storage_size(_CTX, infos, params)
    # Each entry adds at least 32 bytes (id tracker + source_type).
    assert size >= 3 * 32


# ── Factory wiring ────────────────────────────────────────────────────


async def test_new_weaviate_retrieve_engine_repository_applies_config() -> None:
    repo = await new_weaviate_retrieve_engine_repository(
        _FakeClient(),  # type: ignore[arg-type]
        IndexConfig(
            collection_prefix="custom_prefix",
            replication_factor=2,
            desired_shard_count=4,
            hnsw_ef_construction=256,
            hnsw_ef_search=128,
            hnsw_m=64,
        ),
    )
    assert repo._collection_base_name == "custom_prefix"
    assert repo._replication_factor == 2
    assert repo._desired_shard_count == 4
    assert repo._hnsw_ef_construction == 256
    assert repo._hnsw_ef == 128
    assert repo._hnsw_m == 64


async def test_new_weaviate_retrieve_engine_repository_falls_back_to_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WEAVIATE_COLLECTION", "env_only")
    repo = await new_weaviate_retrieve_engine_repository(
        _FakeClient(),  # type: ignore[arg-type]
        None,
    )
    assert repo._collection_base_name == "env_only"


async def test_new_weaviate_retrieve_engine_repository_from_env_uses_connect_to_custom() -> None:
    with patch("src.ai.retrieval.weaviate.connect_to_custom") as connect:
        connect.return_value = _FakeClient()
        repo = await new_weaviate_retrieve_engine_repository_from_env(
            "weaviate.local",
            "weaviate.local:50051",
            "http",
            "secret-key",
            IndexConfig(collection_prefix="custom"),
        )
    kwargs = connect.call_args.kwargs
    assert kwargs["http_host"] == "weaviate.local"
    assert kwargs["http_secure"] is False
    assert kwargs["grpc_host"] == "weaviate.local"
    assert kwargs["grpc_port"] == 50051
    assert kwargs["grpc_secure"] is False
    assert kwargs["auth_credentials"] is not None
    assert kwargs["skip_init_checks"] is True
    assert repo._collection_base_name == "custom"


async def test_new_weaviate_retrieve_engine_repository_from_env_https_and_no_key() -> None:
    with patch("src.ai.retrieval.weaviate.connect_to_custom") as connect:
        connect.return_value = _FakeClient()
        await new_weaviate_retrieve_engine_repository_from_env(
            "weaviate.local", "weaviate.local:50051", "https", "", None
        )
    kwargs = connect.call_args.kwargs
    assert kwargs["http_secure"] is True
    assert kwargs["grpc_secure"] is True
    assert kwargs["auth_credentials"] is None


# ── Public surface contract ──────────────────────────────────────────


def test_factory_weaviate_placeholder_is_replaced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The factory placeholder for Weaviate now delegates to the real builder."""
    from src.ai.retrieval import factory as factory_module

    captured: dict[str, object] = {}

    async def _fake_ctor(
        host: str,
        grpc_address: str,
        scheme: str,
        api_key: str,
        index_config: IndexConfig | None = None,
    ) -> WeaviateRetrieveEngineRepository:
        captured["host"] = host
        captured["grpc"] = grpc_address
        captured["scheme"] = scheme
        captured["api_key"] = api_key
        captured["index_config"] = index_config
        return _repo()

    monkeypatch.setattr(
        factory_module,
        "new_weaviate_retrieve_engine_repository_from_env",
        _fake_ctor,
    )

    async def _exercise() -> WeaviateRetrieveEngineRepository:
        return await factory_module._new_weaviate_retrieve_engine_repository(
            WeaviateClientConfig(
                host="weaviate.local",
                grpc_address="weaviate.local:50051",
                scheme="http",
                api_key="secret",
            ),
            IndexConfig(collection_prefix="custom"),
        )

    import asyncio

    result = asyncio.run(_exercise())
    assert result is not None
    assert captured["host"] == "weaviate.local"
    assert captured["grpc"] == "weaviate.local:50051"
    assert captured["scheme"] == "http"
    assert captured["api_key"] == "secret"
    assert isinstance(captured["index_config"], IndexConfig)
    assert captured["index_config"].collection_prefix == "custom"
