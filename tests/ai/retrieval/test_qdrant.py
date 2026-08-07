"""Tests for the Qdrant retrieval engine repository.

The ``AsyncQdrantClient`` is mocked; no real Qdrant instance is contacted.
Pinned here: collection creation + payload indexes, save / batch_save
(bucketing by dimension), vector and keyword retrieval, delete / batch
update / copy-indices, storage-size estimate, factory wiring, and the
``score_threshold`` / empty-list short-circuits.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from qdrant_client.http.models import (
    CollectionDescription,
    CollectionsResponse,
    Distance,
    Filter,
    PayloadSchemaType,
    PointStruct,
    ScoredPoint,
    TextIndexParams,
    TextIndexType,
    TokenizerType,
    VectorParams,
)

from src.ai.embedding import TaskContext
from src.ai.retrieval.qdrant import (
    FIELD_CHUNK_ID,
    FIELD_CONTENT,
    FIELD_IS_ENABLED,
    FIELD_KNOWLEDGE_BASE_ID,
    FIELD_KNOWLEDGE_ID,
    FIELD_SOURCE_ID,
    FIELD_SOURCE_TYPE,
    FIELD_TAG_ID,
    QdrantRetrieveEngineRepository,
    _calculate_storage_size,
    _collection_name,
    _resolve_collection_name,
    _to_embedding_data,
    _tokenize_query,
    new_qdrant_retrieve_engine_repository,
)
from src.ai.retrieval.types import (
    IndexConfig,
    IndexInfo,
    MatchType,
    RetrieveParams,
    RetrieverType,
    SourceType,
)

_CTX = TaskContext()


# ── Mock helpers ──────────────────────────────────────────────────────


def _scored_point(
    point_id: str, score: float, payload: dict[str, object]
) -> ScoredPoint:
    """Build a ScoredPoint with just the fields the repo reads."""
    return ScoredPoint(id=point_id, score=score, payload=payload, version=0)


def _record(
    point_id: str, payload: dict[str, object], vector: list[float] | None = None
) -> MagicMock:
    """Build a minimal Record mock for scroll responses."""
    record = MagicMock()
    record.id = point_id
    record.payload = payload
    record.vector = vector
    return record


class _FakeClient:
    """Async-capable mock of the subset of ``AsyncQdrantClient`` the repo uses."""

    def __init__(self) -> None:
        self.collection_exists = AsyncMock(return_value=False)
        self.create_collection = AsyncMock(return_value=True)
        self.create_payload_index = AsyncMock(return_value=None)
        self.upsert = AsyncMock(return_value=None)
        self.delete = AsyncMock(return_value=None)
        self.set_payload = AsyncMock(return_value=None)
        self.query_points = AsyncMock()
        self.scroll = AsyncMock(return_value=([], None))
        self.get_collections = AsyncMock(return_value=CollectionsResponse(collections=[]))
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def _repo(
    base: str = "weknora_embeddings",
    shard: int = 0,
    replication: int = 0,
    client: _FakeClient | None = None,
) -> QdrantRetrieveEngineRepository:
    fake = client or _FakeClient()
    return QdrantRetrieveEngineRepository(
        client=fake,  # type: ignore[arg-type]
        collection_base_name=base,
        shard_number=shard,
        replication_factor=replication,
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


def _embedding(
    source_id: str = "src-1", dim: int = 4, extra: list[tuple[str, list[float]]] | None = None
) -> list[float]:
    base = [0.1 * (i + 1) for i in range(dim)]
    del source_id, extra
    return base


# ── Pure helpers ──────────────────────────────────────────────────────


def test_resolve_collection_name_prefers_collection_prefix() -> None:
    cfg = IndexConfig(collection_prefix="prefix", collection_name="name")
    assert _resolve_collection_name(cfg) == "prefix"


def test_resolve_collection_name_falls_back_to_collection_name() -> None:
    cfg = IndexConfig(collection_prefix="", collection_name="name")
    assert _resolve_collection_name(cfg) == "name"


def test_resolve_collection_name_uses_env_when_no_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT_COLLECTION", "env_coll")
    assert _resolve_collection_name(None) == "env_coll"


def test_resolve_collection_name_uses_default_when_nothing_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("QDRANT_COLLECTION", raising=False)
    assert _resolve_collection_name(None) == "weknora_embeddings"


def test_resolve_collection_name_config_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT_COLLECTION", "env_coll")
    cfg = IndexConfig(collection_prefix="prefix")
    assert _resolve_collection_name(cfg) == "prefix"


def test_collection_name_combines_base_and_dimension() -> None:
    assert _collection_name("weknora_embeddings", 768) == "weknora_embeddings_768"


def test_tokenize_query_returns_empty_for_blank() -> None:
    assert _tokenize_query("") == []
    assert _tokenize_query("   ") == []


def test_tokenize_query_deduplicates_and_filters_short_tokens() -> None:
    # ASCII words; single chars and repeats are dropped.
    tokens = _tokenize_query("alpha alpha beta a b")
    assert "alpha" in tokens
    assert "beta" in tokens
    # 'a' and 'b' are single-character and must be dropped.
    assert all(len(t) >= 2 for t in tokens)


def test_to_embedding_data_extracts_embedding_by_source_id() -> None:
    info = _index_info(source_id="src-7")
    params = _save_params("src-7", [0.1, 0.2])
    data = _to_embedding_data(info, params)
    assert data.embedding == [0.1, 0.2]
    assert data.source_id == "src-7"
    assert data.is_enabled is True


def test_to_embedding_data_returns_empty_when_source_id_missing() -> None:
    info = _index_info(source_id="src-7")
    params = _save_params("src-other", [0.1, 0.2])
    data = _to_embedding_data(info, params)
    assert data.embedding == []


def test_calculate_storage_size_includes_vector_and_id_tracker() -> None:
    info = _index_info(content="abcde", source_id="s", chunk_id="c", knowledge_id="k", knowledge_base_id="kb")
    data = _to_embedding_data(info, {})
    # No embedding => vector_size = 0, hnsw_index = 0.
    size_no_vec = _calculate_storage_size(data)
    assert size_no_vec == len("abcde") + len("s") + len("c") + len("k") + len("kb") + 8 + 24

    data_with_vec = _to_embedding_data(info, _save_params("s", [0.0] * 16))
    size_with_vec = _calculate_storage_size(data_with_vec)
    assert size_with_vec == size_no_vec + 16 * 4 + 16 * 2 * 8


# ── Construction / engine_type / support ──────────────────────────────


def test_engine_type_is_qdrant() -> None:
    from src.ai.retrieval.types import RetrieverEngineType

    r = _repo()
    assert r.engine_type() == RetrieverEngineType.QDRANT


def test_supports_keywords_and_vector() -> None:
    r = _repo()
    from src.ai.retrieval.types import RetrieverType

    assert set(r.support()) == {RetrieverType.KEYWORDS, RetrieverType.VECTOR}


# ── Collection management ─────────────────────────────────────────────


async def test_save_creates_collection_with_expected_schema() -> None:
    client = _FakeClient()
    client.collection_exists.return_value = False
    r = _repo(client=client)
    info = _index_info(source_id="s-1")
    await r.save(_CTX, info, _save_params("s-1", [0.1] * 4))
    client.create_collection.assert_awaited_once()
    kwargs = client.create_collection.await_args
    assert kwargs is not None
    assert kwargs.kwargs["collection_name"] == "weknora_embeddings_4"
    vc = kwargs.kwargs["vectors_config"]
    assert isinstance(vc, VectorParams)
    assert vc.size == 4
    assert vc.distance == Distance.COSINE


async def test_save_creates_payload_indexes_after_collection() -> None:
    client = _FakeClient()
    client.collection_exists.return_value = False
    r = _repo(client=client)
    info = _index_info(source_id="s-1")
    await r.save(_CTX, info, _save_params("s-1", [0.1] * 4))
    indexes = client.create_payload_index.await_args_list
    schema_by_field = {
        c.kwargs["field_name"]: c.kwargs["field_schema"] for c in indexes
    }
    for keyword_field in (
        FIELD_CHUNK_ID,
        FIELD_KNOWLEDGE_ID,
        FIELD_KNOWLEDGE_BASE_ID,
        FIELD_SOURCE_ID,
    ):
        assert schema_by_field[keyword_field] == PayloadSchemaType.KEYWORD
    assert schema_by_field[FIELD_IS_ENABLED] == PayloadSchemaType.BOOL
    text_schema = schema_by_field[FIELD_CONTENT]
    assert isinstance(text_schema, TextIndexParams)
    assert text_schema.type == TextIndexType.TEXT
    assert text_schema.tokenizer == TokenizerType.MULTILINGUAL
    assert text_schema.lowercase is True


async def test_save_skips_index_creation_when_collection_already_exists() -> None:
    client = _FakeClient()
    client.collection_exists.return_value = True
    r = _repo(client=client)
    info = _index_info(source_id="s-1")
    await r.save(_CTX, info, _save_params("s-1", [0.1] * 4))
    client.create_collection.assert_not_awaited()
    client.create_payload_index.assert_not_awaited()


async def test_save_rejects_empty_embedding() -> None:
    r = _repo()
    info = _index_info(source_id="s-1")
    with pytest.raises(ValueError, match="empty embedding"):
        await r.save(_CTX, info, _save_params("s-1", []))


async def test_save_passes_point_with_payload_and_vector() -> None:
    client = _FakeClient()
    client.collection_exists.return_value = True
    r = _repo(client=client)
    info = _index_info(source_id="s-1", chunk_id="c-1", knowledge_id="k-1")
    await r.save(_CTX, info, _save_params("s-1", [0.1, 0.2, 0.3, 0.4]))
    upsert_call = client.upsert.await_args
    assert upsert_call is not None
    assert upsert_call.kwargs["collection_name"] == "weknora_embeddings_4"
    points = upsert_call.kwargs["points"]
    assert len(points) == 1
    point = points[0]
    assert isinstance(point, PointStruct)
    assert point.vector == [0.1, 0.2, 0.3, 0.4]
    payload = point.payload
    assert payload is not None
    assert payload[FIELD_CHUNK_ID] == "c-1"
    assert payload[FIELD_KNOWLEDGE_ID] == "k-1"
    assert payload[FIELD_IS_ENABLED] is True


async def test_batch_save_groups_points_by_dimension() -> None:
    client = _FakeClient()
    client.collection_exists.return_value = True
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
    upsert_calls = client.upsert.await_args_list
    collections = {c.kwargs["collection_name"] for c in upsert_calls}
    assert collections == {"weknora_embeddings_4", "weknora_embeddings_8"}


async def test_batch_save_skips_empty_embeddings() -> None:
    client = _FakeClient()
    client.collection_exists.return_value = True
    r = _repo(client=client)
    infos = [_index_info(source_id="a"), _index_info(source_id="b")]
    params = {"embedding": {"a": [0.1] * 4}}
    await r.batch_save(_CTX, infos, params)
    assert client.upsert.await_count == 1


async def test_batch_save_no_op_on_empty_list() -> None:
    client = _FakeClient()
    r = _repo(client=client)
    await r.batch_save(_CTX, [], _save_params("x", [0.1] * 4))
    client.upsert.assert_not_awaited()


# ── Vector retrieval ─────────────────────────────────────────────────


async def test_vector_retrieve_returns_empty_when_collection_missing() -> None:
    client = _FakeClient()
    client.collection_exists.return_value = False
    r = _repo(client=client)
    params = RetrieveParams(
        query="hi",
        embedding=[0.1] * 4,
        top_k=3,
        threshold=0.0,
        retriever_type=RetrieverType.VECTOR,
    )
    results = await r.retrieve(_CTX, params)
    assert len(results) == 1
    assert results[0].results == []


async def test_vector_retrieve_queries_with_filter_and_threshold() -> None:
    client = _FakeClient()
    client.collection_exists.return_value = True
    client.query_points.return_value = MagicMock(points=[])
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
    qp = client.query_points.await_args
    assert qp is not None
    assert qp.kwargs["collection_name"] == "weknora_embeddings_4"
    assert qp.kwargs["limit"] == 7
    assert qp.kwargs["score_threshold"] == 0.5
    assert qp.kwargs["with_payload"] is True
    flt = qp.kwargs["query_filter"]
    assert isinstance(flt, Filter)
    must = flt.must
    must_not = flt.must_not
    assert must is not None and isinstance(must, list)
    # Enabled + 3 positive + 2 negative conditions
    assert len(must) == 4
    assert must_not is not None
    assert isinstance(must_not, list)
    assert len(must_not) == 2


async def test_vector_retrieve_converts_points_to_index_with_score() -> None:
    client = _FakeClient()
    client.collection_exists.return_value = True
    payload = {
        FIELD_CONTENT: "hi",
        FIELD_SOURCE_ID: "s",
        FIELD_SOURCE_TYPE: int(SourceType.CHUNK),
        FIELD_CHUNK_ID: "c",
        FIELD_KNOWLEDGE_ID: "k",
        FIELD_KNOWLEDGE_BASE_ID: "kb",
        FIELD_TAG_ID: "t",
        FIELD_IS_ENABLED: True,
    }
    client.query_points.return_value = MagicMock(
        points=[_scored_point("p-1", 0.9, payload), _scored_point("p-2", 0.7, payload)]
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


async def test_vector_retrieve_omits_threshold_when_zero() -> None:
    client = _FakeClient()
    client.collection_exists.return_value = True
    client.query_points.return_value = MagicMock(points=[])
    r = _repo(client=client)
    params = RetrieveParams(
        embedding=[0.1] * 4,
        top_k=5,
        threshold=0.0,
        retriever_type=RetrieverType.VECTOR,
    )
    await r.retrieve(_CTX, params)
    qp_args = client.query_points.await_args
    assert qp_args is not None
    assert qp_args.kwargs["score_threshold"] is None


# ── Keywords retrieval ───────────────────────────────────────────────


async def test_keywords_retrieve_filters_collections_by_base_name() -> None:
    client = _FakeClient()
    client.get_collections.return_value = CollectionsResponse(
        collections=[
            CollectionDescription(name="weknora_embeddings_4"),
            CollectionDescription(name="weknora_embeddings_8"),
            CollectionDescription(name="other_collection"),
        ]
    )
    client.scroll.return_value = ([], None)
    r = _repo(client=client)
    params = RetrieveParams(query="hello", top_k=3, retriever_type=RetrieverType.KEYWORDS)
    await r.retrieve(_CTX, params)
    # Only the two base-name-matching collections are scrolled.
    scrolled = {c.kwargs["collection_name"] for c in client.scroll.await_args_list}
    assert scrolled == {"weknora_embeddings_4", "weknora_embeddings_8"}


async def test_keywords_retrieve_uses_should_conditions_for_tokens() -> None:
    client = _FakeClient()
    client.get_collections.return_value = CollectionsResponse(
        collections=[CollectionDescription(name="weknora_embeddings_4")]
    )
    client.scroll.return_value = ([], None)
    r = _repo(client=client)
    params = RetrieveParams(
        query="machine learning",
        top_k=3,
        retriever_type=RetrieverType.KEYWORDS,
    )
    await r.retrieve(_CTX, params)
    scroll_args = client.scroll.await_args
    assert scroll_args is not None
    flt = scroll_args.kwargs["scroll_filter"]
    assert isinstance(flt, Filter)
    should = flt.should
    must = flt.must
    assert should is not None and isinstance(should, list) and len(should) >= 2
    assert must is not None


async def test_keywords_retrieve_falls_back_to_must_match_for_empty_tokens() -> None:
    client = _FakeClient()
    client.get_collections.return_value = CollectionsResponse(
        collections=[CollectionDescription(name="weknora_embeddings_4")]
    )
    client.scroll.return_value = ([], None)
    r = _repo(client=client)
    params = RetrieveParams(query="a", top_k=3, retriever_type=RetrieverType.KEYWORDS)
    await r.retrieve(_CTX, params)
    scroll_args = client.scroll.await_args
    assert scroll_args is not None
    flt = scroll_args.kwargs["scroll_filter"]
    should = flt.should
    must = flt.must
    assert should is None or len(should) == 0
    assert must is not None


async def test_keywords_retrieve_caps_results_to_top_k() -> None:
    client = _FakeClient()
    client.get_collections.return_value = CollectionsResponse(
        collections=[CollectionDescription(name="weknora_embeddings_4")]
    )
    payload: dict[str, object] = {FIELD_CONTENT: "x", FIELD_CHUNK_ID: "c"}
    records = [_record(f"p-{i}", payload) for i in range(5)]
    client.scroll.return_value = (records, None)
    r = _repo(client=client)
    params = RetrieveParams(query="x", top_k=2, retriever_type=RetrieverType.KEYWORDS)
    results = await r.retrieve(_CTX, params)
    assert len(results[0].results) == 2
    assert results[0].results[0].match_type == MatchType.KEYWORDS


# ── Delete ────────────────────────────────────────────────────────────


async def test_delete_by_chunk_id_list_emits_filter_selector() -> None:
    client = _FakeClient()
    r = _repo(client=client)
    await r.delete_by_chunk_id_list(_CTX, ["c-1", "c-2"], 4, "")
    delete_call = client.delete.await_args
    assert delete_call is not None
    assert delete_call.kwargs["collection_name"] == "weknora_embeddings_4"
    selector = delete_call.kwargs["points_selector"]
    assert isinstance(selector, Filter)


async def test_delete_by_chunk_id_list_skips_empty_input() -> None:
    client = _FakeClient()
    r = _repo(client=client)
    await r.delete_by_chunk_id_list(_CTX, [], 4, "")
    client.delete.assert_not_awaited()


async def test_delete_by_source_id_list_targets_source_field() -> None:
    client = _FakeClient()
    r = _repo(client=client)
    await r.delete_by_source_id_list(_CTX, ["s-1"], 4, "")
    delete_args = client.delete.await_args
    assert delete_args is not None
    flt = delete_args.kwargs["points_selector"]
    assert any(c.key == FIELD_SOURCE_ID for c in (flt.must or []))


async def test_delete_by_knowledge_id_list_targets_knowledge_field() -> None:
    client = _FakeClient()
    r = _repo(client=client)
    await r.delete_by_knowledge_id_list(_CTX, ["k-1"], 4, "")
    delete_args = client.delete.await_args
    assert delete_args is not None
    flt = delete_args.kwargs["points_selector"]
    assert any(c.key == FIELD_KNOWLEDGE_ID for c in (flt.must or []))


# ── Batch update ──────────────────────────────────────────────────────


async def test_batch_update_chunk_enabled_status_updates_all_collections() -> None:
    client = _FakeClient()
    client.get_collections.return_value = CollectionsResponse(
        collections=[
            CollectionDescription(name="weknora_embeddings_4"),
            CollectionDescription(name="weknora_embeddings_8"),
            CollectionDescription(name="unrelated"),
        ]
    )
    r = _repo(client=client)
    await r.batch_update_chunk_enabled_status(
        _CTX, {"c-1": True, "c-2": False}
    )
    set_payload_calls = client.set_payload.await_args_list
    targeted = {c.kwargs["collection_name"] for c in set_payload_calls}
    assert targeted == {"weknora_embeddings_4", "weknora_embeddings_8"}
    # Two payload writes per collection (enabled + disabled).
    assert len(set_payload_calls) == 4


async def test_batch_update_chunk_enabled_status_skips_empty_map() -> None:
    client = _FakeClient()
    r = _repo(client=client)
    await r.batch_update_chunk_enabled_status(_CTX, {})
    client.set_payload.assert_not_awaited()


async def test_batch_update_chunk_tag_id_groups_by_tag() -> None:
    client = _FakeClient()
    client.get_collections.return_value = CollectionsResponse(
        collections=[CollectionDescription(name="weknora_embeddings_4")]
    )
    r = _repo(client=client)
    await r.batch_update_chunk_tag_id(
        _CTX, {"c-1": "tag-a", "c-2": "tag-a", "c-3": "tag-b"}
    )
    payloads = [c.kwargs["payload"][FIELD_TAG_ID] for c in client.set_payload.await_args_list]
    assert payloads == ["tag-a", "tag-b"]


# ── Copy indices ──────────────────────────────────────────────────────


async def test_copy_indices_paginates_and_remaps_ids() -> None:
    client = _FakeClient()
    client.collection_exists.return_value = True
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
    # First call returns 2 records; second call returns empty.
    client.scroll.side_effect = [
        ([_record("p-a", src_payload_a, [0.1] * 4),
          _record("p-b", src_payload_b, [0.2] * 4)], "p-b"),
        ([], None),
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
    # ensure_collection checks existence; collection already exists so no create.
    client.collection_exists.assert_awaited()
    client.create_collection.assert_not_awaited()
    upsert_args = client.upsert.await_args
    assert upsert_args is not None
    points = upsert_args.kwargs["points"]
    assert len(points) == 2
    target_kbs = set()
    for p in points:
        payload = p.payload
        assert payload is not None
        target_kbs.add(payload[FIELD_KNOWLEDGE_BASE_ID])
    assert target_kbs == {"tgt-kb"}


async def test_copy_indices_preserves_question_suffix() -> None:
    client = _FakeClient()
    client.collection_exists.return_value = True
    src_payload = {
        FIELD_CONTENT: "q",
        FIELD_SOURCE_ID: "c1-q1",  # generated question for chunk c1
        FIELD_SOURCE_TYPE: 0,
        FIELD_CHUNK_ID: "c1",
        FIELD_KNOWLEDGE_ID: "k1",
        FIELD_KNOWLEDGE_BASE_ID: "src-kb",
        FIELD_TAG_ID: "",
        FIELD_IS_ENABLED: True,
    }
    client.scroll.side_effect = [
        ([_record("p", src_payload, [0.1] * 4)], "p"),
        ([], None),
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
    points_args = client.upsert.await_args
    assert points_args is not None
    points = points_args.kwargs["points"]
    assert points[0].payload[FIELD_SOURCE_ID] == "tc1-q1"


async def test_copy_indices_no_op_on_empty_map() -> None:
    client = _FakeClient()
    r = _repo(client=client)
    await r.copy_indices(_CTX, "src-kb", {}, {}, "tgt-kb", 4, "")
    client.scroll.assert_not_awaited()
    client.upsert.assert_not_awaited()


# ── Estimate storage size ─────────────────────────────────────────────


def test_estimate_storage_size_sums_all_entries() -> None:
    r = _repo()
    infos = [_index_info(source_id=f"s-{i}", chunk_id=f"c-{i}") for i in range(3)]
    params = _save_params("s-0", [0.0] * 8)
    size = r.estimate_storage_size(_CTX, infos, params)
    # Each entry adds at least 24 bytes (id tracker) + 8 (source_type).
    assert size >= 3 * 32


# ── Factory wiring ────────────────────────────────────────────────────


async def test_new_qdrant_retrieve_engine_repository_constructs_client() -> None:
    with patch("src.ai.retrieval.qdrant.AsyncQdrantClient") as client_cls:
        client_cls.return_value = MagicMock()
        repo = await new_qdrant_retrieve_engine_repository(
            "qdrant.example.com", 6334, "key-1", True, IndexConfig(shard_number=2)
        )
    kwargs = client_cls.call_args.kwargs
    assert kwargs["host"] == "qdrant.example.com"
    assert kwargs["grpc_port"] == 6334
    assert kwargs["prefer_grpc"] is True
    assert kwargs["api_key"] == "key-1"
    assert kwargs["https"] is True
    assert repo._collection_base_name == "weknora_embeddings"
    assert repo._shard_number == 2


async def test_new_qdrant_retrieve_engine_repository_uses_index_config_prefix() -> None:
    with patch("src.ai.retrieval.qdrant.AsyncQdrantClient") as client_cls:
        client_cls.return_value = MagicMock()
        repo = await new_qdrant_retrieve_engine_repository(
            "h", 6334, "", False, IndexConfig(collection_prefix="custom_prefix")
        )
    assert repo._collection_base_name == "custom_prefix"


async def test_new_qdrant_retrieve_engine_repository_without_config_falls_back_to_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QDRANT_COLLECTION", "env_only")
    with patch("src.ai.retrieval.qdrant.AsyncQdrantClient") as client_cls:
        client_cls.return_value = MagicMock()
        repo = await new_qdrant_retrieve_engine_repository("h", 6334, "", False, None)
    assert repo._collection_base_name == "env_only"


async def test_new_qdrant_retrieve_engine_repository_passes_none_api_key_when_empty() -> None:
    with patch("src.ai.retrieval.qdrant.AsyncQdrantClient") as client_cls:
        client_cls.return_value = MagicMock()
        await new_qdrant_retrieve_engine_repository("h", 6334, "", False, None)
    assert client_cls.call_args.kwargs["api_key"] is None


# ── Retrieve dispatch ────────────────────────────────────────────────


async def test_retrieve_dispatch_rejects_unknown_retriever_type() -> None:
    r = _repo()
    from src.ai.retrieval.types import RetrieverType as RT

    # Build a params object with a non-keywords/non-vector type via construction
    # bypass: a custom RetrieverType is not creatable, so just patch the field.
    params = RetrieveParams(retriever_type=RT.VECTOR)
    # Replace retriever_type with an unknown value via object.__setattr__ bypass.
    object.__setattr__(params, "retriever_type", "bogus")
    with pytest.raises(ValueError, match="invalid retriever type"):
        await r.retrieve(_CTX, params)
