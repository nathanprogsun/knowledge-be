"""Tests for the Elasticsearch v7 and v8 engine repositories.

The SDK clients are mocked with ``MagicMock`` - no Elasticsearch cluster is
contacted. Pinned: index management, field-type detection, keyword/vector
retrieval DSL shapes, save/batch-save/delete operations, copy-indices, and
batch update-by-query.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from src.ai.embedding import TaskContext
from src.ai.retrieval._es_common import (
    build_base_conds,
    calculate_storage_size,
    resolve_index_name,
    to_db_vector_embedding,
)
from src.ai.retrieval.elasticsearch_v7 import ElasticsearchV7Repository
from src.ai.retrieval.elasticsearch_v8 import ElasticsearchV8Repository
from src.ai.retrieval.types import (
    IndexConfig,
    IndexInfo,
    MatchType,
    RetrieveParams,
    RetrieverEngineType,
    RetrieverType,
    SourceType,
)
from src.common.exception import ValidationError

_CTX = TaskContext()
_INDEX = "test_index"


# ── helpers ──────────────────────────────────────────────────────────


def _mock_v7_client(
    index_exists: bool = True,
    mapping_type: str = "keyword",
) -> MagicMock:
    """Build a mocked elasticsearch7 client."""
    client = MagicMock()
    client.indices.exists.return_value = index_exists
    client.indices.create.return_value = {"acknowledged": True}
    client.indices.get_mapping.return_value = {
        _INDEX: {"mappings": {"properties": {"chunk_id": {"type": mapping_type}}}}
    }
    client.search.return_value = {"hits": {"hits": []}}
    client.create.return_value = {"result": "created"}
    client.bulk.return_value = {"errors": False}
    client.delete_by_query.return_value = {"deleted": 1}
    client.update_by_query.return_value = {"updated": 1}
    return client


def _mock_v8_client(
    index_exists: bool = True,
    mapping_type: str = "keyword",
) -> MagicMock:
    """Build a mocked elasticsearch v8 client."""
    client = MagicMock()
    client.indices.exists.return_value = index_exists
    client.indices.create.return_value = {"acknowledged": True}
    client.indices.get_mapping.return_value = {
        _INDEX: {"mappings": {"properties": {"chunk_id": {"type": mapping_type}}}}
    }
    client.search.return_value = {"hits": {"hits": []}}
    client.index.return_value = {"result": "created"}
    client.bulk.return_value = {"errors": False}
    client.delete_by_query.return_value = {"deleted": 1}
    client.update_by_query.return_value = {"updated": 1}
    return client


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


def _search_hit(doc_id: str = "doc1", score: float = 1.5, **source: Any) -> dict[str, Any]:
    src = {"chunk_id": "c1", "content": "text", "source_id": "s1", "knowledge_id": "k1",
           "knowledge_base_id": "kb1", "source_type": 0, "is_enabled": True}
    src.update(source)
    return {"_id": doc_id, "_score": score, "_source": src}


# ── shared helpers ──────────────────────────────────────────────────


def test_resolve_index_name_uses_config() -> None:
    cfg = IndexConfig(index_name="from_config")
    assert resolve_index_name(cfg, "ENV_KEY", "default") == "from_config"


def test_resolve_index_name_uses_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENV_KEY", "from_env")
    assert resolve_index_name(None, "ENV_KEY", "default") == "from_env"


def test_resolve_index_name_uses_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ENV_KEY", raising=False)
    assert resolve_index_name(None, "ENV_KEY", "default") == "default"


def test_to_db_vector_embedding_looks_up_embedding() -> None:
    info = _index_info()
    params = {"embedding": {"src-1": [0.1, 0.2]}}
    doc = to_db_vector_embedding(info, params)
    assert doc["embedding"] == [0.1, 0.2]
    assert doc["content"] == "hello world"


def test_to_db_vector_embedding_overrides_is_enabled() -> None:
    info = _index_info()
    params = {"chunk_enabled": {"chunk-1": False}}
    doc = to_db_vector_embedding(info, params)
    assert doc["is_enabled"] is False


def test_calculate_storage_size_includes_vector() -> None:
    doc = {"content": "abc", "embedding": [1.0] * 100}
    size = calculate_storage_size(doc)
    assert size > 0
    assert size == 3 + 400 + 250 + (3 + 400) * 5 // 10


def test_build_base_conds_builds_must_and_must_not() -> None:
    params = RetrieveParams(
        knowledge_base_ids=["kb1"],
        knowledge_ids=["k1"],
        exclude_chunk_ids=["c2"],
    )
    conds = build_base_conds(params, lambda name: name)
    assert len(conds) == 1
    bool_q = conds[0]["bool"]
    assert len(bool_q["must"]) == 2
    assert {"term": {"is_enabled": False}} in bool_q["must_not"]


# ── v7 repository ────────────────────────────────────────────────────


def _v7_repo(
    client: MagicMock | None = None,
    index_config: IndexConfig | None = None,
) -> ElasticsearchV7Repository:
    return ElasticsearchV7Repository(
        client or _mock_v7_client(),
        index_config or IndexConfig(index_name=_INDEX),
    )


def test_v7_engine_type() -> None:
    repo = _v7_repo()
    assert repo.engine_type() == RetrieverEngineType.ELASTICSEARCH


def test_v7_support_returns_keywords_only() -> None:
    repo = _v7_repo()
    assert repo.support() == [RetrieverType.KEYWORDS]


def test_v7_detects_keyword_suffix_false_for_keyword_type() -> None:
    client = _mock_v7_client(mapping_type="keyword")
    repo = ElasticsearchV7Repository(client, IndexConfig(index_name=_INDEX))
    assert repo._use_keyword_suffix is False


def test_v7_detects_keyword_suffix_true_for_text_type() -> None:
    client = _mock_v7_client(mapping_type="text")
    repo = ElasticsearchV7Repository(client, IndexConfig(index_name=_INDEX))
    assert repo._use_keyword_suffix is True


def test_v7_creates_index_when_missing() -> None:
    client = _mock_v7_client(index_exists=False)
    ElasticsearchV7Repository(client, IndexConfig(index_name=_INDEX))
    client.indices.create.assert_called_once()


def test_v7_skips_index_creation_when_exists() -> None:
    client = _mock_v7_client(index_exists=True)
    ElasticsearchV7Repository(client, IndexConfig(index_name=_INDEX))
    client.indices.create.assert_not_called()


def test_v7_save_creates_document_with_uuid() -> None:
    client = _mock_v7_client()
    repo = _v7_repo(client)
    info = _index_info()
    params = {"embedding": {"src-1": [0.1]}}
    import asyncio
    asyncio.run(repo.save(_CTX, info, params))
    client.create.assert_called_once()
    _, kwargs = client.create.call_args
    assert kwargs["index"] == _INDEX
    assert "id" in kwargs
    assert kwargs["body"]["embedding"] == [0.1]


def test_v7_save_rejects_empty_embedding() -> None:
    repo = _v7_repo()
    info = _index_info()
    import asyncio
    with pytest.raises(ValidationError, match="empty embedding"):
        asyncio.run(repo.save(_CTX, info, {}))


def test_v7_batch_save_builds_bulk_actions() -> None:
    client = _mock_v7_client()
    repo = _v7_repo(client)
    infos = [_index_info(), _index_info(source_id="src-2", chunk_id="c2")]
    params = {"embedding": {"src-1": [0.1], "src-2": [0.2]}}
    import asyncio
    asyncio.run(repo.batch_save(_CTX, infos, params))
    client.bulk.assert_called_once()
    _, kwargs = client.bulk.call_args
    actions = kwargs["body"]
    assert len(actions) == 4


def test_v7_batch_save_empty_list_skips() -> None:
    client = _mock_v7_client()
    repo = _v7_repo(client)
    import asyncio
    asyncio.run(repo.batch_save(_CTX, [], {}))
    client.bulk.assert_not_called()


def test_v7_keywords_retrieve_builds_match_query() -> None:
    client = _mock_v7_client()
    client.search.return_value = {"hits": {"hits": [_search_hit()]}}
    repo = _v7_repo(client)
    params = RetrieveParams(query="hello", top_k=10, retriever_type=RetrieverType.KEYWORDS)
    import asyncio
    results = asyncio.run(repo.retrieve(_CTX, params))
    assert len(results) == 1
    assert results[0].retriever_type == RetrieverType.KEYWORDS
    assert len(results[0].results) == 1
    assert results[0].results[0].id == "doc1"
    assert results[0].results[0].score == 1.5
    _, kwargs = client.search.call_args
    body = kwargs["body"]
    assert "match" in body["query"]["bool"]["must"][0]


def test_v7_retrieve_rejects_vector_type() -> None:
    repo = _v7_repo()
    params = RetrieveParams(retriever_type=RetrieverType.VECTOR, top_k=5)
    import asyncio
    with pytest.raises(ValidationError, match="invalid retriever type"):
        asyncio.run(repo.retrieve(_CTX, params))


def test_v7_delete_by_chunk_id_list() -> None:
    client = _mock_v7_client()
    repo = _v7_repo(client)
    import asyncio
    asyncio.run(repo.delete_by_chunk_id_list(_CTX, ["c1", "c2"], 128, "doc"))
    client.delete_by_query.assert_called_once()
    _, kwargs = client.delete_by_query.call_args
    assert "chunk_id" in kwargs["body"]["query"]["terms"]


def test_v7_delete_by_empty_list_skips() -> None:
    client = _mock_v7_client()
    repo = _v7_repo(client)
    import asyncio
    asyncio.run(repo.delete_by_chunk_id_list(_CTX, [], 128, "doc"))
    client.delete_by_query.assert_not_called()


def test_v7_batch_update_chunk_enabled_status() -> None:
    client = _mock_v7_client()
    repo = _v7_repo(client)
    import asyncio
    asyncio.run(repo.batch_update_chunk_enabled_status(_CTX, {"c1": True, "c2": False}))
    assert client.update_by_query.call_count == 2


def test_v7_batch_update_chunk_tag_id() -> None:
    client = _mock_v7_client()
    repo = _v7_repo(client)
    import asyncio
    asyncio.run(repo.batch_update_chunk_tag_id(_CTX, {"c1": "t1", "c2": "t1", "c3": "t2"}))
    assert client.update_by_query.call_count == 2


def test_v7_estimate_storage_size() -> None:
    repo = _v7_repo()
    infos = [_index_info()]
    params = {"embedding": {"src-1": [0.1] * 100}}
    size = repo.estimate_storage_size(_CTX, infos, params)
    assert size > 0


def test_v7_delete_by_source_id_list() -> None:
    client = _mock_v7_client()
    repo = _v7_repo(client)
    import asyncio
    asyncio.run(repo.delete_by_source_id_list(_CTX, ["s1", "s2"], 128, "doc"))
    client.delete_by_query.assert_called_once()
    _, kwargs = client.delete_by_query.call_args
    assert "source_id" in kwargs["body"]["query"]["terms"]
    assert kwargs["body"]["query"]["terms"]["source_id"] == ["s1", "s2"]
    assert kwargs["index"] == _INDEX


def test_v7_delete_by_source_id_list_empty_skips() -> None:
    client = _mock_v7_client()
    repo = _v7_repo(client)
    import asyncio
    asyncio.run(repo.delete_by_source_id_list(_CTX, [], 128, "doc"))
    client.delete_by_query.assert_not_called()


def test_v7_delete_by_knowledge_id_list() -> None:
    client = _mock_v7_client()
    repo = _v7_repo(client)
    import asyncio
    asyncio.run(repo.delete_by_knowledge_id_list(_CTX, ["k1", "k2"], 128, "doc"))
    client.delete_by_query.assert_called_once()
    _, kwargs = client.delete_by_query.call_args
    assert "knowledge_id" in kwargs["body"]["query"]["terms"]
    assert kwargs["body"]["query"]["terms"]["knowledge_id"] == ["k1", "k2"]
    assert kwargs["index"] == _INDEX


def test_v7_delete_by_knowledge_id_list_empty_skips() -> None:
    client = _mock_v7_client()
    repo = _v7_repo(client)
    import asyncio
    asyncio.run(repo.delete_by_knowledge_id_list(_CTX, [], 128, "doc"))
    client.delete_by_query.assert_not_called()


def test_v7_copy_indices_paginates_and_saves() -> None:
    client = _mock_v7_client()
    client.search.return_value = {
        "hits": {"hits": [
            _search_hit(doc_id="d1", chunk_id="src_c1", knowledge_id="src_k1"),
            _search_hit(doc_id="d2", chunk_id="src_c2", knowledge_id="src_k2"),
        ]}
    }
    repo = _v7_repo(client)
    import asyncio
    asyncio.run(repo.copy_indices(
        _CTX,
        source_knowledge_base_id="src_kb",
        source_to_target_kb_id_map={"src_k1": "tgt_k1", "src_k2": "tgt_k2"},
        source_to_target_chunk_id_map={"src_c1": "tgt_c1", "src_c2": "tgt_c2"},
        target_knowledge_base_id="tgt_kb",
        dimension=128,
        knowledge_type="doc",
    ))
    client.bulk.assert_called_once()


def test_v7_copy_indices_empty_map_skips() -> None:
    client = _mock_v7_client()
    repo = _v7_repo(client)
    import asyncio
    asyncio.run(repo.copy_indices(
        _CTX, "kb", {}, {}, "tgt", 128, "doc",
    ))
    client.search.assert_not_called()
    client.bulk.assert_not_called()


# ── v8 repository ────────────────────────────────────────────────────


def _v8_repo(
    client: MagicMock | None = None,
    index_config: IndexConfig | None = None,
) -> ElasticsearchV8Repository:
    return ElasticsearchV8Repository(
        client or _mock_v8_client(),
        index_config or IndexConfig(index_name=_INDEX),
    )


def test_v8_engine_type() -> None:
    repo = _v8_repo()
    assert repo.engine_type() == RetrieverEngineType.ELASTICSEARCH


def test_v8_support_returns_keywords_and_vector() -> None:
    repo = _v8_repo()
    assert repo.support() == [RetrieverType.KEYWORDS, RetrieverType.VECTOR]


def test_v8_save_uses_index_not_create() -> None:
    client = _mock_v8_client()
    repo = _v8_repo(client)
    info = _index_info()
    params = {"embedding": {"src-1": [0.1]}}
    import asyncio
    asyncio.run(repo.save(_CTX, info, params))
    client.index.assert_called_once()
    _, kwargs = client.index.call_args
    assert kwargs["index"] == _INDEX
    assert "document" in kwargs
    client.create.assert_not_called()


def test_v8_keywords_retrieve_excludes_embedding_source() -> None:
    client = _mock_v8_client()
    client.search.return_value = {"hits": {"hits": [_search_hit()]}}
    repo = _v8_repo(client)
    params = RetrieveParams(query="hello", top_k=10, retriever_type=RetrieverType.KEYWORDS)
    import asyncio
    results = asyncio.run(repo.retrieve(_CTX, params))
    assert len(results) == 1
    assert results[0].retriever_type == RetrieverType.KEYWORDS
    _, kwargs = client.search.call_args
    body = kwargs["body"]
    assert body["_source"]["excludes"] == ["embedding"]


def test_v8_vector_retrieve_uses_script_score() -> None:
    client = _mock_v8_client()
    client.search.return_value = {"hits": {"hits": [_search_hit(doc_id="v1", score=0.9)]}}
    repo = _v8_repo(client)
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
    assert "script_score" in body["query"]
    assert body["query"]["script_score"]["script"]["source"] == "cosineSimilarity(params.query_vector, 'embedding')"
    assert body["_source"]["excludes"] == ["embedding"]


def test_v8_retrieve_rejects_unknown_type() -> None:
    repo = _v8_repo()
    params = RetrieveParams(retriever_type=RetrieverType.WEB_SEARCH, top_k=5)
    import asyncio
    with pytest.raises(ValidationError, match="invalid retriever type"):
        asyncio.run(repo.retrieve(_CTX, params))


def test_v8_batch_save_uses_operations() -> None:
    client = _mock_v8_client()
    repo = _v8_repo(client)
    infos = [_index_info(), _index_info(source_id="src-2", chunk_id="c2")]
    params = {"embedding": {"src-1": [0.1], "src-2": [0.2]}}
    import asyncio
    asyncio.run(repo.batch_save(_CTX, infos, params))
    client.bulk.assert_called_once()
    _, kwargs = client.bulk.call_args
    ops = kwargs["operations"]
    assert len(ops) == 4


def test_v8_delete_by_source_id_list() -> None:
    client = _mock_v8_client()
    repo = _v8_repo(client)
    import asyncio
    asyncio.run(repo.delete_by_source_id_list(_CTX, ["s1", "s2"], 128, "doc"))
    client.delete_by_query.assert_called_once()
    _, kwargs = client.delete_by_query.call_args
    assert "source_id" in kwargs["query"]["terms"]
    assert kwargs["query"]["terms"]["source_id"] == ["s1", "s2"]
    assert kwargs["index"] == _INDEX


def test_v8_delete_by_source_id_list_empty_skips() -> None:
    client = _mock_v8_client()
    repo = _v8_repo(client)
    import asyncio
    asyncio.run(repo.delete_by_source_id_list(_CTX, [], 128, "doc"))
    client.delete_by_query.assert_not_called()


def test_v8_delete_by_chunk_id_list() -> None:
    client = _mock_v8_client()
    repo = _v8_repo(client)
    import asyncio
    asyncio.run(repo.delete_by_chunk_id_list(_CTX, ["c1", "c2"], 128, "doc"))
    client.delete_by_query.assert_called_once()
    _, kwargs = client.delete_by_query.call_args
    assert "chunk_id" in kwargs["query"]["terms"]
    assert kwargs["query"]["terms"]["chunk_id"] == ["c1", "c2"]
    assert kwargs["index"] == _INDEX


def test_v8_delete_by_chunk_id_list_empty_skips() -> None:
    client = _mock_v8_client()
    repo = _v8_repo(client)
    import asyncio
    asyncio.run(repo.delete_by_chunk_id_list(_CTX, [], 128, "doc"))
    client.delete_by_query.assert_not_called()


def test_v8_delete_by_knowledge_id_list() -> None:
    client = _mock_v8_client()
    repo = _v8_repo(client)
    import asyncio
    asyncio.run(repo.delete_by_knowledge_id_list(_CTX, ["k1", "k2"], 128, "doc"))
    client.delete_by_query.assert_called_once()
    _, kwargs = client.delete_by_query.call_args
    assert "knowledge_id" in kwargs["query"]["terms"]
    assert kwargs["query"]["terms"]["knowledge_id"] == ["k1", "k2"]
    assert kwargs["index"] == _INDEX


def test_v8_delete_by_knowledge_id_list_empty_skips() -> None:
    client = _mock_v8_client()
    repo = _v8_repo(client)
    import asyncio
    asyncio.run(repo.delete_by_knowledge_id_list(_CTX, [], 128, "doc"))
    client.delete_by_query.assert_not_called()


def test_v8_batch_update_chunk_enabled_status() -> None:
    client = _mock_v8_client()
    repo = _v8_repo(client)
    import asyncio
    asyncio.run(repo.batch_update_chunk_enabled_status(_CTX, {"c1": True, "c2": False}))
    assert client.update_by_query.call_count == 2


def test_v8_batch_update_chunk_tag_id_groups_by_tag() -> None:
    client = _mock_v8_client()
    repo = _v8_repo(client)
    import asyncio
    asyncio.run(repo.batch_update_chunk_tag_id(_CTX, {"c1": "t1", "c2": "t2"}))
    assert client.update_by_query.call_count == 2


def test_v8_copy_indices_paginates_and_saves() -> None:
    client = _mock_v8_client()
    client.search.return_value = {
        "hits": {"hits": [
            _search_hit(doc_id="d1", chunk_id="src_c1", knowledge_id="src_k1"),
            _search_hit(doc_id="d2", chunk_id="src_c2", knowledge_id="src_k2"),
        ]}
    }
    repo = _v8_repo(client)
    import asyncio
    asyncio.run(repo.copy_indices(
        _CTX,
        source_knowledge_base_id="src_kb",
        source_to_target_kb_id_map={"src_k1": "tgt_k1", "src_k2": "tgt_k2"},
        source_to_target_chunk_id_map={"src_c1": "tgt_c1", "src_c2": "tgt_c2"},
        target_knowledge_base_id="tgt_kb",
        dimension=128,
        knowledge_type="doc",
    ))
    client.bulk.assert_called_once()


def test_v8_copy_indices_empty_map_skips() -> None:
    client = _mock_v8_client()
    repo = _v8_repo(client)
    import asyncio
    asyncio.run(repo.copy_indices(
        _CTX, "kb", {}, {}, "tgt", 128, "doc",
    ))
    client.search.assert_not_called()
