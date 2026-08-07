"""Tests for the Tencent VectorDB retrieve engine repository.

Mock the ``tcvectordb.RpcClient`` and its ``Database`` / ``Collection`` chain
with configurable fakes. Every async method is replaced by ``AsyncMock`` so
no real RPC call is issued. Pinned here: the embedding extraction, the
delete-by-filter routing, the base filter string, the source-id translation,
the vector/keyword retrieve dispatch, and the batch-update path.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.ai.embedding import TaskContext
from src.ai.retrieval.tencent_vectordb import (
    TencentVectorDBRepository,
    _base_filter,
    _clean_invalid_utf8,
    _default_if_zero,
    _from_document,
    _in_expr,
    _resolve_collection_name,
    _resolve_replica_number,
    _retrieve_result,
    _should_use_dimension_suffix,
    _to_document,
    _to_vector_embedding,
    _translate_source_id,
    new_tencent_vectordb_retrieve_engine_repository,
)
from src.ai.retrieval.types import (
    IndexConfig,
    IndexInfo,
    IndexSaveParams,
    MatchType,
    RetrieveParams,
    RetrieverEngineType,
    RetrieverType,
)

_CTX = TaskContext()


# ── Fakes ───────────────────────────────────────────────────────────


class _FakeCollection:
    """Records calls and returns canned responses for upsert/search/query/etc.

    SDK methods are sync (the real tcvectordb SDK is sync); the repository
    wraps them in ``asyncio.to_thread`` so the fake can be plain functions.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def upsert(self, documents: Any, **_kw: Any) -> dict[str, Any]:
        self.calls.append(("upsert", {"documents": documents, **_kw}))
        return {"affectedCount": len(documents) if isinstance(documents, list) else 1}

    def search(self, vectors: Any, **_kw: Any) -> list[list[dict[str, Any]]]:
        self.calls.append(("search", {"vectors": vectors, **_kw}))
        return [[{"id": "row-1", "score": 0.9, "fields": {
            "content": "hello", "source_id": "src-1", "source_type": 0,
            "chunk_id": "c-1", "knowledge_id": "k-1",
            "knowledge_base_id": "kb-1", "tag_id": "t-1", "is_enabled": 1,
        }}]]

    def search_by_text(self, texts: Any, **_kw: Any) -> list[dict[str, Any]]:
        self.calls.append(("search_by_text", {"texts": texts, **_kw}))
        return [{"id": "row-2", "score": 0.8, "fields": {
            "content": "world", "source_id": "src-2", "source_type": 0,
            "chunk_id": "c-2", "knowledge_id": "k-2",
            "knowledge_base_id": "kb-2", "tag_id": "t-2", "is_enabled": 1,
        }}]

    def fulltext_search(self, data: Any, **_kw: Any) -> list[Any]:
        self.calls.append(("fulltext_search", {"data": data, **_kw}))
        return []

    def query(self, **_kw: Any) -> list[Any]:
        self.calls.append(("query", dict(_kw)))
        return []

    def delete(self, **_kw: Any) -> dict[str, Any]:
        self.calls.append(("delete", dict(_kw)))
        return {"affectedCount": 0}

    def update(self, **_kw: Any) -> dict[str, Any]:
        self.calls.append(("update", dict(_kw)))
        return {"affectedCount": 0}


class _FakeDatabase:
    """Records ``exists_collection`` / ``create_collection`` / ``list_collections`` calls.

    All methods are sync; the real tcvectordb SDK is sync and the repository
    wraps the calls in ``asyncio.to_thread``.
    """

    def __init__(self, name: str, collections: list[str] | None = None) -> None:
        self.name = name
        self._collections = collections or []
        self.collections_map: dict[str, _FakeCollection] = {}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def Collection(self, name: str) -> _FakeCollection:
        coll = self.collections_map.get(name)
        if coll is None:
            coll = _FakeCollection(name)
            self.collections_map[name] = coll
        return coll

    def exists_collection(self, name: str) -> bool:
        self.calls.append(("exists_collection", {"name": name}))
        return name in self._collections

    def create_collection_if_not_exists(self, name: str) -> None:
        self.calls.append(("create_database", {"name": name}))

    def create_collection(self, name: str, *_args: Any, **_kw: Any) -> None:
        self.calls.append(("create_collection", {"name": name}))
        self._collections.append(name)

    def list_collections(self) -> list[str]:
        self.calls.append(("list_collections", {}))
        return list(self._collections)


class _FakeClient:
    """Root client fake that hands out a single ``_FakeDatabase``."""

    def __init__(self, database: _FakeDatabase) -> None:
        self._database = database
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def Database(self, name: str) -> _FakeDatabase:
        return self._database

    def create_database_if_not_exists(self, name: str) -> None:
        self.calls.append(("create_database_if_not_exists", {"name": name}))


def _new_repo(
    database_name: str = "test_db",
    index_cfg: IndexConfig | None = None,
    existing_collections: list[str] | None = None,
    dimensions: list[int] | None = None,
    collection_base: str = "emb",
) -> tuple[TencentVectorDBRepository, _FakeDatabase, _FakeClient]:
    db = _FakeDatabase(database_name, existing_collections)
    client = _FakeClient(db)
    repo = new_tencent_vectordb_retrieve_engine_repository(client, database_name, index_cfg)
    repo._collection_base_name = collection_base
    # Existing collections keyed by the same prefix
    if existing_collections:
        existing_collections[:] = [c.replace("weknora_embeddings", collection_base) for c in existing_collections]
        repo._collection_base_name = collection_base
    # Pretend collections are pre-initialized so upsert skips create.
    if dimensions:
        for d in dimensions:
            repo._initialized.add(d)
    return repo, db, client


# ── Pure helpers ────────────────────────────────────────────────────


def test_resolve_collection_name_priority() -> None:
    cfg = IndexConfig(collection_prefix="prefix_x", collection_name="name_y")
    assert _resolve_collection_name(cfg, "ENV_KEY", "default") == "prefix_x"
    cfg2 = IndexConfig(collection_name="name_y")
    assert _resolve_collection_name(cfg2, "ENV_KEY", "default") == "name_y"
    cfg3 = IndexConfig()
    assert _resolve_collection_name(cfg3, "UNLIKELY_ENV_KEY_xyz", "default") == "default"


def test_resolve_replica_number_uses_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TENCENT_VECTORDB_REPLICA_NUMBER", "3")
    assert _resolve_replica_number(IndexConfig()) == 3
    assert _resolve_replica_number(IndexConfig(replica_number=5)) == 5
    monkeypatch.setenv("TENCENT_VECTORDB_REPLICA_NUMBER", "not-a-number")
    assert _resolve_replica_number(IndexConfig()) == 1
    monkeypatch.delenv("TENCENT_VECTORDB_REPLICA_NUMBER", raising=False)
    assert _resolve_replica_number(IndexConfig()) == 1


def test_should_use_dimension_suffix() -> None:
    assert _should_use_dimension_suffix(None) is True
    assert _should_use_dimension_suffix(IndexConfig()) is True
    assert _should_use_dimension_suffix(IndexConfig(collection_name="explicit")) is False


def test_default_if_zero() -> None:
    assert _default_if_zero(0, 5) == 5
    assert _default_if_zero(7, 5) == 7


def test_clean_invalid_utf8_strips_nulls() -> None:
    assert _clean_invalid_utf8("hello\x00world") == "helloworld"


def test_in_expr_quotes_values() -> None:
    expr = _in_expr("chunk_id", ["c-1", "c-2"])
    assert "chunk_id in" in expr
    assert '"c-1"' in expr and '"c-2"' in expr


def test_translate_source_id_same_chunk() -> None:
    assert _translate_source_id("c-1", "c-1", "c-2") == "c-2"


def test_translate_source_id_question_suffix() -> None:
    assert _translate_source_id("c-1-q-1", "c-1", "c-2") == "c-2-q-1"


def test_translate_source_id_other_uses_sha256() -> None:
    result = _translate_source_id("orphan", "c-1", "c-2")
    assert result.startswith("c-2-")
    assert len(result.split("-", 2)[2]) == 16


def test_matches_collection_with_dimension_suffix() -> None:
    # Use the repo's _matches_collection indirectly
    repo, _, _ = _new_repo()
    repo._use_dimension_suffix = True
    repo._collection_base_name = "emb"
    assert repo._matches_collection("emb_768") is True
    assert repo._matches_collection("other_768") is False


def test_matches_collection_without_dimension_suffix() -> None:
    repo, _, _ = _new_repo()
    repo._use_dimension_suffix = False
    repo._collection_base_name = "emb"
    assert repo._matches_collection("emb") is True
    assert repo._matches_collection("other") is False


def test_base_filter_includes_is_enabled() -> None:
    expr = _base_filter(RetrieveParams(retriever_type=RetrieverType.VECTOR))
    assert "is_enabled=1" in expr


def test_base_filter_includes_optional_filters() -> None:
    expr = _base_filter(RetrieveParams(
        retriever_type=RetrieverType.VECTOR,
        knowledge_base_ids=["kb-1"],
        knowledge_ids=["k-1"],
        tag_ids=["t-1"],
        exclude_knowledge_ids=["k-ex"],
        exclude_chunk_ids=["c-ex"],
    ))
    assert "knowledge_base_id in" in expr
    assert "not (knowledge_id in" in expr
    assert "not (chunk_id in" in expr


def test_to_vector_embedding_extracts_by_source_id() -> None:
    info = IndexInfo(
        source_id="src-1", chunk_id="c-1", content="hello", is_enabled=True,
    )
    params: IndexSaveParams = {"embedding": {"src-1": [1.0, 2.0, 3.0]}}
    emb = _to_vector_embedding(info, params)
    assert emb.embedding == [1.0, 2.0, 3.0]
    assert emb.source_id == "src-1"


def test_to_vector_embedding_falls_back_to_chunk_id() -> None:
    info = IndexInfo(source_id="", chunk_id="c-1")
    params: IndexSaveParams = {"embedding": {"c-1": [4.0, 5.0]}}
    emb = _to_vector_embedding(info, params)
    assert emb.embedding == [4.0, 5.0]


def test_to_vector_embedding_no_embedding_returns_empty() -> None:
    info = IndexInfo(source_id="s1", chunk_id="c1")
    emb = _to_vector_embedding(info, {})
    assert emb.embedding == []


def test_to_document_builds_correct_fields() -> None:
    emb = _to_vector_embedding(
        IndexInfo(
            source_id="src-1", chunk_id="c-1", knowledge_id="k-1",
            knowledge_base_id="kb-1", tag_id="t-1", is_enabled=True,
        ),
        {"embedding": {"src-1": [1.0]}},
    )
    doc = _to_document(emb)
    assert doc["id"] == "src-1"
    assert doc["vector"] == [1.0]
    assert doc["content"] == ""
    assert doc["source_id"] == "src-1"
    assert doc["is_enabled"] == 1


def test_from_document_parses_dict() -> None:
    doc = {
        "id": "row-1",
        "vector": [1.0, 2.0],
        "fields": {
            "content": "hello", "source_id": "s-1", "source_type": 0,
            "chunk_id": "c-1", "knowledge_id": "k-1",
            "knowledge_base_id": "kb-1", "tag_id": "t-1", "is_enabled": 1,
        },
        "score": 0.85,
    }
    emb = _from_document(doc)
    assert emb.id == "row-1"
    assert emb.score == 0.85
    assert emb.is_enabled is True
    assert emb.embedding == [1.0, 2.0]


def test_retrieve_result_envelope() -> None:
    result = _retrieve_result([], RetrieverType.VECTOR)
    assert len(result) == 1
    assert result[0].retriever_type == RetrieverType.VECTOR
    assert result[0].retriever_engine_type == RetrieverEngineType.TENCENT_VECTORDB


# ── RetrieveEngine contract ────────────────────────────────────────


def test_engine_type_returns_tencent_vectordb() -> None:
    repo, _, _ = _new_repo()
    assert repo.engine_type() == RetrieverEngineType.TENCENT_VECTORDB


def test_support_returns_keywords_and_vector() -> None:
    repo, _, _ = _new_repo()
    assert set(repo.support()) == {RetrieverType.KEYWORDS, RetrieverType.VECTOR}


def test_estimate_storage_size_accumulates() -> None:
    repo, _, _ = _new_repo()
    infos = [
        IndexInfo(source_id="s1", content="hello"),
        IndexInfo(source_id="s2", content="world"),
    ]
    total = repo.estimate_storage_size(_CTX, infos, {"embedding": {"s1": [1.0], "s2": [2.0]}})
    assert total > 0


# ── save / batch_save ──────────────────────────────────────────────


async def test_save_delegates_to_batch_save() -> None:
    repo, _, _ = _new_repo()
    bs = AsyncMock(return_value=None)
    repo.batch_save = bs  # type: ignore[method-assign]
    info = IndexInfo(source_id="s1", chunk_id="c1")
    await repo.save(_CTX, info, {"embedding": {"s1": [1.0]}})
    bs.assert_awaited_once()


async def test_batch_save_with_empty_list_is_noop() -> None:
    repo, db, _ = _new_repo()
    await repo.batch_save(_CTX, [], {})
    assert db.calls == []


async def test_batch_save_groups_by_dimension() -> None:
    repo, db, _ = _new_repo(dimensions=[4, 768])
    infos = [
        IndexInfo(source_id="s1", chunk_id="c1"),
        IndexInfo(source_id="s2", chunk_id="c2"),
    ]
    await repo.batch_save(_CTX, infos, {
        "embedding": {"s1": [1.0, 2.0, 3.0, 4.0], "s2": [0.1] * 768}
    })
    # Each dimension got its own collection upsert
    calls = sum(1 for c in db.collections_map["emb_4"].calls + db.collections_map["emb_768"].calls if c[0] == "upsert")
    assert calls == 2


async def test_batch_save_skips_empty_embedding() -> None:
    repo, db, _ = _new_repo(dimensions=[4])
    infos = [IndexInfo(source_id="s1", chunk_id="c1")]
    await repo.batch_save(_CTX, infos, {"embedding": {}})
    assert all(c[0] != "upsert" for c in db.collections_map.get("emb_4", _FakeCollection("x")).calls)


# ── retrieve ───────────────────────────────────────────────────────


async def test_retrieve_dispatches_by_type() -> None:
    repo, _, _ = _new_repo(existing_collections=["emb_4"], dimensions=[4])
    seen: list[str] = []

    async def _kw(*_a: Any, **_k: Any) -> list[Any]:
        seen.append("kw")
        return []

    async def _vc(*_a: Any, **_k: Any) -> list[Any]:
        seen.append("vc")
        return []

    repo._keywords_retrieve = _kw  # type: ignore[method-assign]
    repo._vector_retrieve = _vc  # type: ignore[method-assign]
    await repo.retrieve(_CTX, RetrieveParams(retriever_type=RetrieverType.KEYWORDS))
    await repo.retrieve(_CTX, RetrieveParams(retriever_type=RetrieverType.VECTOR))
    assert seen == ["kw", "vc"]


async def test_retrieve_invalid_type_raises() -> None:
    repo, _, _ = _new_repo()
    with pytest.raises(ValueError, match="invalid retriever type"):
        await repo.retrieve(_CTX, RetrieveParams(retriever_type=RetrieverType.WEB_SEARCH))


async def test_vector_retrieve_returns_empty_when_no_embedding() -> None:
    repo, _, _ = _new_repo()
    results = await repo._vector_retrieve(_CTX, RetrieveParams(retriever_type=RetrieverType.VECTOR))
    assert results[0].results == []


async def test_vector_retrieve_returns_empty_when_collection_missing() -> None:
    repo, _, _ = _new_repo()
    results = await repo._vector_retrieve(
        _CTX, RetrieveParams(embedding=[1.0, 2.0, 3.0, 4.0], retriever_type=RetrieverType.VECTOR)
    )
    assert results[0].results == []


async def test_vector_retrieve_runs_search() -> None:
    repo, db, _ = _new_repo(existing_collections=["emb_4"])
    results = await repo._vector_retrieve(
        _CTX,
        RetrieveParams(
            embedding=[1.0, 2.0, 3.0, 4.0],
            retriever_type=RetrieverType.VECTOR,
            top_k=5,
            threshold=0.5,
        ),
    )
    assert len(results) == 1
    assert len(results[0].results) == 1
    item = results[0].results[0]
    assert item.id == "row-1"
    assert item.match_type == MatchType.EMBEDDING
    # Search was called with radius (threshold)
    search_calls = [c for c in db.collections_map["emb_4"].calls if c[0] == "search"]
    assert search_calls
    assert search_calls[0][1].get("radius") == 0.5


async def test_keywords_retrieve_empty_query_returns_empty() -> None:
    repo, _, _ = _new_repo()
    results = await repo._keywords_retrieve(_CTX, RetrieveParams(query=""))
    assert results[0].results == []


async def test_keywords_retrieve_searches_all_collections() -> None:
    repo, db, _ = _new_repo(existing_collections=["emb_4", "emb_768"])
    await repo._keywords_retrieve(
        _CTX,
        RetrieveParams(query="hello world", top_k=5, retriever_type=RetrieverType.KEYWORDS),
    )
    # Both collections tried
    text_calls: list[tuple[str, dict[str, Any]]] = []
    for name in ("emb_4", "emb_768"):
        if name in db.collections_map:
            text_calls.extend(c for c in db.collections_map[name].calls if c[0] == "search_by_text")
    assert len(text_calls) == 2


# ── delete_by_* ────────────────────────────────────────────────────


async def test_delete_by_chunk_id_list() -> None:
    repo, db, _ = _new_repo()
    coll = db.collections_map.setdefault("emb_4", _FakeCollection("emb_4"))
    await repo.delete_by_chunk_id_list(_CTX, ["c1", "c2"], dimension=4, knowledge_type="")
    assert any(c[0] == "delete" for c in coll.calls)


async def test_delete_by_source_id_list() -> None:
    repo, db, _ = _new_repo()
    coll = db.collections_map.setdefault("emb_4", _FakeCollection("emb_4"))
    await repo.delete_by_source_id_list(_CTX, ["s1"], dimension=4, knowledge_type="")
    assert any(c[0] == "delete" for c in coll.calls)


async def test_delete_by_knowledge_id_list() -> None:
    repo, db, _ = _new_repo()
    coll = db.collections_map.setdefault("emb_4", _FakeCollection("emb_4"))
    await repo.delete_by_knowledge_id_list(_CTX, ["k1"], dimension=4, knowledge_type="")
    assert any(c[0] == "delete" for c in coll.calls)


# ── copy_indices ───────────────────────────────────────────────────


async def test_copy_indices_empty_mapping_is_noop() -> None:
    repo, db, _ = _new_repo()
    await repo.copy_indices(
        _CTX, "src-kb", {}, {}, "tgt-kb", dimension=4, knowledge_type=""
    )
    assert db.calls == []


async def test_copy_indices_queries_and_upserts() -> None:
    repo, db, _ = _new_repo(dimensions=[4])
    coll = db.collections_map.setdefault("emb_4", _FakeCollection("emb_4"))

    def _query(**_kw: Any) -> list[dict[str, Any]]:
        coll.calls.append(("query", dict(_kw)))
        return [{
            "id": "src-c-1", "vector": [1.0, 2.0, 3.0, 4.0], "fields": {
                "content": "hello", "source_id": "src-c-1", "source_type": 0,
                "chunk_id": "src-c-1", "knowledge_id": "src-k-1",
                "knowledge_base_id": "src-kb-1", "tag_id": "t-1", "is_enabled": 1,
            }
        }]
    coll.query = _query  # type: ignore[method-assign]
    await repo.copy_indices(
        _CTX,
        "src-kb",
        {"src-k-1": "tgt-k-1"},
        {"src-c-1": "tgt-c-1"},
        "tgt-kb",
        dimension=4,
        knowledge_type="",
    )
    query_calls = [c for c in coll.calls if c[0] == "query"]
    upsert_calls = [c for c in coll.calls if c[0] == "upsert"]
    assert len(query_calls) >= 1
    assert len(upsert_calls) == 1


# ── batch_update_chunk_* ──────────────────────────────────────────


async def test_batch_update_chunk_enabled_status_groups_by_value() -> None:
    repo, db, _ = _new_repo(existing_collections=["emb_4"])
    await repo.batch_update_chunk_enabled_status(_CTX, {
        "c-1": True, "c-2": False, "c-3": True,
    })
    update_calls = [c for c in db.collections_map["emb_4"].calls if c[0] == "update"]
    assert len(update_calls) == 2


async def test_batch_update_chunk_tag_id_groups_by_tag() -> None:
    repo, db, _ = _new_repo(existing_collections=["emb_4"])
    await repo.batch_update_chunk_tag_id(_CTX, {
        "c-1": "tag-a", "c-2": "tag-b", "c-3": "tag-a",
    })
    update_calls = [c for c in db.collections_map["emb_4"].calls if c[0] == "update"]
    assert len(update_calls) == 2


async def test_batch_update_with_empty_map_is_noop() -> None:
    repo, db, _ = _new_repo(existing_collections=["emb_4"])
    await repo.batch_update_chunk_enabled_status(_CTX, {})
    await repo.batch_update_chunk_tag_id(_CTX, {})
    coll = db.collections_map.get("emb_4", _FakeCollection("emb_4"))
    update_calls = [c for c in coll.calls if c[0] == "update"]
    assert update_calls == []


# ── ensure_collection ──────────────────────────────────────────────


async def test_ensure_collection_creates_when_missing() -> None:
    repo, db, _ = _new_repo()
    await repo._ensure_collection(_CTX, dimension=4)
    assert "emb_4" in db._collections


async def test_ensure_collection_skips_when_present() -> None:
    repo, db, _ = _new_repo(existing_collections=["emb_4"])
    # Clear calls so we can detect new create calls
    db.calls.clear()
    await repo._ensure_collection(_CTX, dimension=4)
    assert all(c[0] != "create_collection" for c in db.calls)


# ── Constructor ────────────────────────────────────────────────────


def test_new_tencent_vectordb_retrieve_engine_repository_returns_repo() -> None:
    client = _FakeClient(_FakeDatabase("db"))
    repo = new_tencent_vectordb_retrieve_engine_repository(client, "db", None)
    assert isinstance(repo, TencentVectorDBRepository)
    assert repo._database_name == "db"
    assert repo._use_dimension_suffix is True


def test_constructor_reads_index_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TENCENT_VECTORDB_DATABASE", raising=False)
    client = _FakeClient(_FakeDatabase("from_env_db"))
    repo = new_tencent_vectordb_retrieve_engine_repository(
        client, "from_env_db", IndexConfig(collection_name="explicit_coll", replica_number=4)
    )
    assert repo._collection_base_name == "explicit_coll"
    assert repo._use_dimension_suffix is False
    assert repo._replicas_num == 4


def test_constructor_uses_env_when_database_name_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TENCENT_VECTORDB_DATABASE", "env_db")
    client = _FakeClient(_FakeDatabase("env_db"))
    repo = new_tencent_vectordb_retrieve_engine_repository(client, "", None)
    assert repo._database_name == "env_db"


def test_collection_name_uses_dimension_suffix_by_default() -> None:
    repo, _, _ = _new_repo()
    assert repo._collection_name(4) == "emb_4"
    assert repo._collection_name(768) == "emb_768"


def test_collection_name_omits_dimension_when_disabled() -> None:
    repo, _, _ = _new_repo()
    # Override after construction (constructor already set use_dimension_suffix)
    repo._collection_base_name = "single_coll"
    repo._use_dimension_suffix = False
    assert repo._collection_name(4) == "single_coll"
