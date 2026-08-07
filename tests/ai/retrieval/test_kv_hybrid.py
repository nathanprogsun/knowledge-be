"""Tests for the keywords-vector hybrid retrieve engine.

The repository and embedder are faked with ``AsyncMock`` — no vector
database or embedding API is contacted. Pinned here: the embedding step
added by ``index`` / ``batch_index`` (with sanitized content), the
keyword-only path, batching + bounded concurrency, the backoff wrapper,
the storage-size estimate, and the straight delegation of every other
method.
"""

from __future__ import annotations

from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest

from src.ai.embedding import Embedder, TaskContext
from src.ai.retrieval.base import RetrieveEngineRepository
from src.ai.retrieval.kv_hybrid import (
    KVHybridRetrieveEngine,
    _batch_embed_with_backoff,
    _chunk_slice,
    new_kv_hybrid_retrieve_engine,
    sanitize_for_embedding,
)
from src.ai.retrieval.types import (
    IndexInfo,
    RetrieveParams,
    RetrieverEngineType,
    RetrieverType,
)

_CTX = TaskContext()


class _FakeRepository:
    """Structural fake with AsyncMock methods (satisfies the repo protocol)."""

    def __init__(self) -> None:
        self.save = AsyncMock()
        self.batch_save = AsyncMock()
        self.estimate_storage_size = Mock(return_value=42)
        self.retrieve = AsyncMock(return_value=[])
        self.support = Mock(return_value=[RetrieverType.VECTOR, RetrieverType.KEYWORDS])
        self.copy_indices = AsyncMock()
        self.delete_by_chunk_id_list = AsyncMock()
        self.delete_by_source_id_list = AsyncMock()
        self.delete_by_knowledge_id_list = AsyncMock()
        self.batch_update_chunk_enabled_status = AsyncMock()
        self.batch_update_chunk_tag_id = AsyncMock()


class _FakeEmbedder:
    """Structural fake with a fixed embedding dimension."""

    def __init__(self, dimensions: int = 4) -> None:
        self._dimensions = dimensions
        self.embed = AsyncMock(side_effect=self._embed_one)
        self.batch_embed_with_pool = AsyncMock(side_effect=self._embed_many)
        self.embed_calls: list[str] = []

    def get_dimensions(self) -> int:
        return self._dimensions

    async def _embed_one(self, _ctx: object, text: str) -> list[float]:
        self.embed_calls.append(text)
        return [0.1] * self._dimensions

    async def _embed_many(
        self, _ctx: object, _model: object, texts: list[str]
    ) -> list[list[float]]:
        return [[0.1] * self._dimensions for _ in texts]


def _repo() -> _FakeRepository:
    return _FakeRepository()


def _engine(repo: _FakeRepository | None = None) -> tuple[KVHybridRetrieveEngine, _FakeRepository]:
    fake = repo or _repo()
    engine = new_kv_hybrid_retrieve_engine(
        cast("RetrieveEngineRepository", fake), RetrieverEngineType.QDRANT
    )
    return engine, fake


def _index_info(
    source_id: str = "src-1", content: str = "hello", chunk_id: str = "chunk-1"
) -> IndexInfo:
    return IndexInfo(id="id-1", content=content, source_id=source_id, chunk_id=chunk_id)


def _embedder() -> _FakeEmbedder:
    return _FakeEmbedder()


# ── sanitize_for_embedding ──────────────────────────────────────────


def test_sanitize_for_embedding_passes_plain_content() -> None:
    assert sanitize_for_embedding(_CTX, "plain text content") == "plain text content"


def test_sanitize_for_embedding_scrubs_inline_base64_images() -> None:
    content = (
        'before <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAE="> after '
        "![alt](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAE=) tail"
    )
    sanitized = sanitize_for_embedding(_CTX, content)
    assert "[image]" in sanitized
    assert "data:image/png;base64," not in sanitized


def test_sanitize_for_embedding_truncates_over_limit() -> None:
    content = "x" * 50000
    sanitized = sanitize_for_embedding(_CTX, content)
    assert len(sanitized) == 20000


# ── construction / engine_type / support ────────────────────────────


def test_new_engine_reports_engine_type() -> None:
    engine, _ = _engine()
    assert engine.engine_type() == RetrieverEngineType.QDRANT


def test_support_delegates_to_repository() -> None:
    engine, fake = _engine()
    assert engine.support() == [RetrieverType.VECTOR, RetrieverType.KEYWORDS]
    fake.support.assert_called_once_with()


# ── retrieve ────────────────────────────────────────────────────────


async def test_retrieve_delegates_to_repository() -> None:
    engine, fake = _engine()
    params = RetrieveParams(query="q")
    await engine.retrieve(_CTX, params)
    fake.retrieve.assert_awaited_once_with(_CTX, params)


# ── index ───────────────────────────────────────────────────────────


async def test_index_embeds_and_saves_with_vector_retriever() -> None:
    engine, fake = _engine()
    embedder = _embedder()
    index_info = _index_info(content="content-to-embed")
    await engine.index(_CTX, cast("Embedder", embedder), index_info, [RetrieverType.VECTOR])
    embedder.embed.assert_awaited_once_with(_CTX, "content-to-embed")
    fake.save.assert_awaited_once()
    args = fake.save.await_args
    assert args is not None
    assert args.args[1] == index_info
    assert args.args[2] == {"embedding": {"src-1": [0.1, 0.1, 0.1, 0.1]}}


async def test_index_without_vector_retriever_saves_empty_embeddings() -> None:
    engine, fake = _engine()
    embedder = _embedder()
    await engine.index(_CTX, cast("Embedder", embedder), _index_info(), [RetrieverType.KEYWORDS])
    embedder.embed.assert_not_called()
    fake.save.assert_awaited_once()
    args = fake.save.await_args
    assert args is not None
    assert args.args[2] == {"embedding": {}}


# ── batch_index ─────────────────────────────────────────────────────


async def test_batch_index_with_vector_retriever_embeds_then_saves() -> None:
    engine, fake = _engine()
    embedder = _embedder()
    items = [_index_info(source_id=f"src-{i}", content=f"c{i}") for i in range(3)]
    await engine.batch_index(_CTX, cast("Embedder", embedder), items, [RetrieverType.VECTOR])
    embedder.batch_embed_with_pool.assert_awaited_once_with(_CTX, embedder, ["c0", "c1", "c2"])
    fake.batch_save.assert_awaited_once()
    args = fake.batch_save.await_args
    assert args is not None
    assert args.args[1] == items
    assert args.args[2] == {
        "embedding": {
            "src-0": [0.1, 0.1, 0.1, 0.1],
            "src-1": [0.1, 0.1, 0.1, 0.1],
            "src-2": [0.1, 0.1, 0.1, 0.1],
        }
    }


async def test_batch_index_without_vector_retriever_saves_keyword_path() -> None:
    engine, fake = _engine()
    embedder = _embedder()
    items = [_index_info(source_id=f"src-{i}") for i in range(3)]
    await engine.batch_index(_CTX, cast("Embedder", embedder), items, [RetrieverType.KEYWORDS])
    embedder.batch_embed_with_pool.assert_not_called()
    fake.batch_save.assert_awaited_once()
    args = fake.batch_save.await_args
    assert args is not None
    assert args.args[1] == items
    assert args.args[2] == {}


async def test_batch_index_empty_list_is_noop() -> None:
    engine, fake = _engine()
    await engine.batch_index(_CTX, cast("Embedder", _embedder()), [], [RetrieverType.VECTOR])
    fake.batch_save.assert_not_called()


async def test_batch_index_splits_large_vector_batch_into_40s() -> None:
    engine, fake = _engine()
    embedder = _embedder()
    items = [_index_info(source_id=f"src-{i}") for i in range(50)]
    await engine.batch_index(_CTX, cast("Embedder", embedder), items, [RetrieverType.VECTOR])
    assert fake.batch_save.await_count == 2  # 40 + 10
    first = fake.batch_save.await_args_list[0]
    assert first.args[1] == items[:40]
    assert len(first.args[2]["embedding"]) == 40
    second = fake.batch_save.await_args_list[1]
    assert second.args[1] == items[40:]
    assert len(second.args[2]["embedding"]) == 10


async def test_batch_index_bounded_concurrency_for_many_batches() -> None:
    engine, fake = _engine()
    embedder = _embedder()
    # 250 items -> 7 vector batches (> max concurrency 5).
    items = [_index_info(source_id=f"src-{i}") for i in range(250)]
    await engine.batch_index(_CTX, cast("Embedder", embedder), items, [RetrieverType.VECTOR])
    assert fake.batch_save.await_count == 7


# ── estimate_storage_size ───────────────────────────────────────────


def test_estimate_storage_size_uses_embedder_dimensions() -> None:
    engine, fake = _engine()
    embedder = _FakeEmbedder(dimensions=8)
    items = [_index_info(chunk_id=f"chunk-{i}") for i in range(2)]
    total = engine.estimate_storage_size(
        _CTX, cast("Embedder", embedder), items, [RetrieverType.VECTOR]
    )
    assert total == 42
    args = fake.estimate_storage_size.call_args
    assert args is not None
    assert args.args[1] == items
    assert args.args[2] == {"embedding": {"chunk-0": [0.0] * 8, "chunk-1": [0.0] * 8}}


def test_estimate_storage_size_without_vector_retriever_is_plain() -> None:
    engine, fake = _engine()
    embedder = _embedder()
    items = [_index_info() for _ in range(2)]
    engine.estimate_storage_size(_CTX, cast("Embedder", embedder), items, [RetrieverType.KEYWORDS])
    args = fake.estimate_storage_size.call_args
    assert args is not None
    assert args.args[2] == {}


# ── delegated mutations ─────────────────────────────────────────────


async def test_delegated_methods_forward_to_repository() -> None:
    engine, fake = _engine()
    await engine.copy_indices(_CTX, "kb-src", {"kb": "t"}, {"c": "c2"}, "kb-t", 4, "manual")
    fake.copy_indices.assert_awaited_once_with(
        _CTX, "kb-src", {"kb": "t"}, {"c": "c2"}, "kb-t", 4, "manual"
    )

    await engine.delete_by_chunk_id_list(_CTX, ["c1"], 4, "manual")
    fake.delete_by_chunk_id_list.assert_awaited_once_with(_CTX, ["c1"], 4, "manual")

    await engine.delete_by_source_id_list(_CTX, ["s1"], 4, "manual")
    fake.delete_by_source_id_list.assert_awaited_once_with(_CTX, ["s1"], 4, "manual")

    await engine.delete_by_knowledge_id_list(_CTX, ["k1"], 4, "manual")
    fake.delete_by_knowledge_id_list.assert_awaited_once_with(_CTX, ["k1"], 4, "manual")

    await engine.batch_update_chunk_enabled_status(_CTX, {"c1": True})
    fake.batch_update_chunk_enabled_status.assert_awaited_once_with(_CTX, {"c1": True})

    await engine.batch_update_chunk_tag_id(_CTX, {"c1": "tag-1"})
    fake.batch_update_chunk_tag_id.assert_awaited_once_with(_CTX, {"c1": "tag-1"})


# ── helpers ─────────────────────────────────────────────────────────


def test_chunk_slice_splits_contiguous_ranges() -> None:
    chunks = _chunk_slice([_index_info(source_id=f"s{i}") for i in range(5)], 2)
    assert [len(chunk) for chunk in chunks] == [2, 2, 1]


async def test_batch_embed_with_backoff_retries_then_raises() -> None:
    embedder = _FakeEmbedder()
    embedder.batch_embed_with_pool = AsyncMock(side_effect=RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        await _batch_embed_with_backoff(_CTX, cast("Embedder", embedder), ["a"])
    assert embedder.batch_embed_with_pool.await_count == 5


async def test_batch_embed_with_backoff_recovers_after_failure() -> None:
    embedder = _FakeEmbedder()
    attempts = 0

    async def _flaky(_ctx: object, _model: object, texts: list[str]) -> list[list[float]]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient")
        return [[0.1]] * len(texts)

    embedder.batch_embed_with_pool = AsyncMock(side_effect=_flaky)
    embeddings = await _batch_embed_with_backoff(_CTX, cast("Embedder", embedder), ["a", "b"])
    assert embeddings == [[0.1], [0.1]]
    assert attempts == 2
