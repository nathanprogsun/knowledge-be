"""Unit + integration tests for the knowledge-base cascade delete module.

Unit tests exercise ``process_kb_delete`` against ``AsyncMock``
repositories (AAA pattern), covering validation, hook ordering, and
error propagation. Integration tests run the real cascade against the
applied schema.

``documents.tenant_id`` is BIGINT, but ``chunks.tenant_id`` is INTEGER
(32-bit), so the integration tests seed rows under a small local tenant
counter rather than ``make_test_tenant_id`` (whose BIGINT-range values
overflow the chunk column).
"""

from __future__ import annotations

import itertools
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from random import randint
from unittest.mock import AsyncMock, call

import pytest
import pytest_asyncio
from faker import Faker
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from src.common.exception import DataError, ValidationError
from src.core.knowledge.knowledge_bases.delete import KBDeleteResult, process_kb_delete
from src.core.knowledge.knowledge_bases.service.kb_service import KBService
from src.db.dao.chunk_repository import ChunkRepository
from src.db.dao.knowledge_base_repository import KnowledgeBaseRepository
from src.db.dao.knowledge_repository import KnowledgeRepository
from src.db.models.chunk import Chunk
from src.db.models.knowledge import Document
from src.settings import get_settings, reset_settings_cache
from tests.integration.conftest import make_test_tenant_id

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_FAKER_SEED_MAX = 100_000_000
_TENANT_COUNTER = itertools.count(6_000_000)


@pytest.fixture(autouse=True)
def faker_seed() -> None:
    """Re-seed Faker per test for varied-but-reproducible generation."""
    Faker.seed(randint(1, _FAKER_SEED_MAX))


def _int32_tenant_id() -> int:
    """Return a unique tenant id inside the 32-bit INTEGER range."""
    return next(_TENANT_COUNTER)


def _doc_row(*, tenant_id: int, kb_id: str, doc_id: str) -> Document:
    """Build a ``documents`` row with the minimal required columns."""
    return Document(
        id=doc_id,
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
        type="manual",
        title=f"document {doc_id}",
        source="manual",
        parse_status="unprocessed",
        created_at=_NOW,
        updated_at=_NOW,
    )


def _chunk_row(
    *,
    tenant_id: int,
    kb_id: str,
    knowledge_id: str,
    chunk_index: int,
    chunk_id: str,
) -> Chunk:
    """Build a ``chunks`` row with the minimal required columns."""
    return Chunk(
        id=chunk_id,
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
        knowledge_id=knowledge_id,
        content=f"chunk {chunk_index} of {knowledge_id}",
        chunk_index=chunk_index,
        is_enabled=True,
        start_at=0,
        end_at=5,
        chunk_type="text",
        flags=1,
        source_content="",
        content_revision=0,
        index_status="ready",
        last_editor_id="",
        created_at=_NOW,
        updated_at=_NOW,
    )


def _make_repos() -> tuple[AsyncMock, AsyncMock]:
    """Return signature-bound document / chunk repository mocks."""
    knowledge_repo = AsyncMock(spec=KnowledgeRepository)
    chunk_repo = AsyncMock(spec=ChunkRepository)
    return knowledge_repo, chunk_repo


# ── Unit: validation ─────────────────────────────────────────────────


async def test_blank_knowledge_base_id_is_rejected() -> None:
    knowledge_repo, chunk_repo = _make_repos()

    with pytest.raises(ValidationError) as exc:
        await process_kb_delete(
            tenant_id=7,
            knowledge_base_id="  ",
            knowledge_repo=knowledge_repo,
            chunk_repo=chunk_repo,
        )

    assert exc.value.code == "knowledge_base.id_required"
    knowledge_repo.list_by_knowledge_base.assert_not_awaited()


async def test_non_positive_tenant_id_is_rejected() -> None:
    knowledge_repo, chunk_repo = _make_repos()

    with pytest.raises(ValidationError) as exc:
        await process_kb_delete(
            tenant_id=0,
            knowledge_base_id="kb-1",
            knowledge_repo=knowledge_repo,
            chunk_repo=chunk_repo,
        )

    assert exc.value.code == "knowledge_base.tenant_required"
    knowledge_repo.list_by_knowledge_base.assert_not_awaited()


# ── Unit: cascade ────────────────────────────────────────────────────


async def test_cascade_soft_deletes_chunks_then_documents() -> None:
    knowledge_repo, chunk_repo = _make_repos()
    tid = make_test_tenant_id()
    rows = [
        _doc_row(tenant_id=tid, kb_id="kb-1", doc_id="doc-1"),
        _doc_row(tenant_id=tid, kb_id="kb-1", doc_id="doc-2"),
    ]
    knowledge_repo.list_by_knowledge_base.return_value = rows
    chunk_repo.delete_by_knowledge_id.side_effect = [2, 3]
    knowledge_repo.soft_delete_list.return_value = 2

    result = await process_kb_delete(
        tenant_id=tid,
        knowledge_base_id="kb-1",
        knowledge_repo=knowledge_repo,
        chunk_repo=chunk_repo,
        now=_NOW,
    )

    assert isinstance(result, KBDeleteResult)
    assert result.knowledge_ids == ("doc-1", "doc-2")
    assert result.deleted_chunks == 5
    assert result.deleted_knowledge == 2
    knowledge_repo.list_by_knowledge_base.assert_awaited_once_with(tid, "kb-1")
    assert chunk_repo.delete_by_knowledge_id.await_args_list == [
        call(tenant_id=tid, knowledge_id="doc-1", now=_NOW),
        call(tenant_id=tid, knowledge_id="doc-2", now=_NOW),
    ]
    knowledge_repo.soft_delete_list.assert_awaited_once_with(
        tenant_id=tid,
        ids=["doc-1", "doc-2"],
        now=_NOW,
    )


async def test_no_documents_is_a_noop() -> None:
    knowledge_repo, chunk_repo = _make_repos()
    tid = make_test_tenant_id()
    knowledge_repo.list_by_knowledge_base.return_value = []

    result = await process_kb_delete(
        tenant_id=tid,
        knowledge_base_id="kb-1",
        knowledge_repo=knowledge_repo,
        chunk_repo=chunk_repo,
        vector_store_id="vs-1",
    )

    assert result.knowledge_ids == ()
    assert result.deleted_chunks == 0
    assert result.deleted_knowledge == 0
    assert result.vector_store_id == "vs-1"
    chunk_repo.delete_by_knowledge_id.assert_not_awaited()
    knowledge_repo.soft_delete_list.assert_not_awaited()


async def test_result_reports_the_repo_batch_delete_count() -> None:
    knowledge_repo, chunk_repo = _make_repos()
    tid = make_test_tenant_id()
    rows = [
        _doc_row(tenant_id=tid, kb_id="kb-1", doc_id="doc-1"),
        _doc_row(tenant_id=tid, kb_id="kb-1", doc_id="doc-2"),
    ]
    knowledge_repo.list_by_knowledge_base.return_value = rows
    chunk_repo.delete_by_knowledge_id.side_effect = [1, 1]
    knowledge_repo.soft_delete_list.return_value = 1

    result = await process_kb_delete(
        tenant_id=tid,
        knowledge_base_id="kb-1",
        knowledge_repo=knowledge_repo,
        chunk_repo=chunk_repo,
        now=_NOW,
    )

    assert result.knowledge_ids == ("doc-1", "doc-2")
    assert result.deleted_knowledge == 1


# ── Unit: index-cleanup hook ─────────────────────────────────────────


async def test_index_cleanup_receives_documents_and_store_snapshot() -> None:
    knowledge_repo, chunk_repo = _make_repos()
    tid = make_test_tenant_id()
    rows = [_doc_row(tenant_id=tid, kb_id="kb-1", doc_id="doc-1")]
    knowledge_repo.list_by_knowledge_base.return_value = rows
    chunk_repo.delete_by_knowledge_id.return_value = 1
    knowledge_repo.soft_delete_list.return_value = 1
    seen: list[dict[str, object]] = []

    async def _hook(**kwargs: object) -> None:
        seen.append(kwargs)

    result = await process_kb_delete(
        tenant_id=tid,
        knowledge_base_id="kb-1",
        knowledge_repo=knowledge_repo,
        chunk_repo=chunk_repo,
        vector_store_id="vs-9",
        index_cleanup=_hook,
    )

    assert seen == [
        {
            "tenant_id": tid,
            "knowledge_base_id": "kb-1",
            "knowledge": rows,
            "vector_store_id": "vs-9",
        }
    ]
    assert result.vector_store_id == "vs-9"


async def test_index_cleanup_runs_before_any_row_is_written() -> None:
    knowledge_repo, chunk_repo = _make_repos()
    tid = make_test_tenant_id()
    knowledge_repo.list_by_knowledge_base.return_value = [
        _doc_row(tenant_id=tid, kb_id="kb-1", doc_id="doc-1")
    ]
    chunk_repo.delete_by_knowledge_id.return_value = 1
    knowledge_repo.soft_delete_list.return_value = 1

    async def _hook(**kwargs: object) -> None:
        chunk_repo.delete_by_knowledge_id.assert_not_awaited()
        knowledge_repo.soft_delete_list.assert_not_awaited()

    await process_kb_delete(
        tenant_id=tid,
        knowledge_base_id="kb-1",
        knowledge_repo=knowledge_repo,
        chunk_repo=chunk_repo,
        index_cleanup=_hook,
    )


async def test_index_cleanup_failure_aborts_before_db_writes() -> None:
    knowledge_repo, chunk_repo = _make_repos()
    tid = make_test_tenant_id()
    knowledge_repo.list_by_knowledge_base.return_value = [
        _doc_row(tenant_id=tid, kb_id="kb-1", doc_id="doc-1")
    ]

    async def _hook(**kwargs: object) -> None:
        raise RuntimeError("store unreachable")

    with pytest.raises(RuntimeError, match="store unreachable"):
        await process_kb_delete(
            tenant_id=tid,
            knowledge_base_id="kb-1",
            knowledge_repo=knowledge_repo,
            chunk_repo=chunk_repo,
            index_cleanup=_hook,
        )

    chunk_repo.delete_by_knowledge_id.assert_not_awaited()
    knowledge_repo.soft_delete_list.assert_not_awaited()


# ── Unit: error handling ─────────────────────────────────────────────


async def test_chunk_sweep_failure_is_best_effort() -> None:
    knowledge_repo, chunk_repo = _make_repos()
    tid = make_test_tenant_id()
    rows = [
        _doc_row(tenant_id=tid, kb_id="kb-1", doc_id="doc-1"),
        _doc_row(tenant_id=tid, kb_id="kb-1", doc_id="doc-2"),
        _doc_row(tenant_id=tid, kb_id="kb-1", doc_id="doc-3"),
    ]
    knowledge_repo.list_by_knowledge_base.return_value = rows
    chunk_repo.delete_by_knowledge_id.side_effect = [2, RuntimeError("sweep failed"), 1]
    knowledge_repo.soft_delete_list.return_value = 3

    result = await process_kb_delete(
        tenant_id=tid,
        knowledge_base_id="kb-1",
        knowledge_repo=knowledge_repo,
        chunk_repo=chunk_repo,
        now=_NOW,
    )

    assert result.deleted_chunks == 3
    assert result.deleted_knowledge == 3
    knowledge_repo.soft_delete_list.assert_awaited_once()


async def test_knowledge_batch_delete_failure_propagates() -> None:
    knowledge_repo, chunk_repo = _make_repos()
    tid = make_test_tenant_id()
    knowledge_repo.list_by_knowledge_base.return_value = [
        _doc_row(tenant_id=tid, kb_id="kb-1", doc_id="doc-1")
    ]
    chunk_repo.delete_by_knowledge_id.return_value = 1
    knowledge_repo.soft_delete_list.side_effect = DataError(
        message="document batch delete failed",
        code="document.delete_failed",
    )

    with pytest.raises(DataError, match="batch delete failed"):
        await process_kb_delete(
            tenant_id=tid,
            knowledge_base_id="kb-1",
            knowledge_repo=knowledge_repo,
            chunk_repo=chunk_repo,
        )


# ── Integration (real applied schema) ─────────────────────────────────


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    """Per-test session against the real applied schema (no cleanup)."""
    reset_settings_cache()
    settings = get_settings()
    engine = create_async_engine(settings.database_url, poolclass=NullPool)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with factory() as s:
            yield s
    finally:
        await engine.dispose()


async def test_integration_cascade_soft_deletes_documents_and_chunks(
    session: AsyncSession,
) -> None:
    tid = _int32_tenant_id()
    kb_id = f"kb-{uuid.uuid4().hex[:12]}"
    doc_ids = [f"doc-{uuid.uuid4().hex[:12]}", f"doc-{uuid.uuid4().hex[:12]}"]
    knowledge_repo = KnowledgeRepository(session)
    chunk_repo = ChunkRepository(session)
    for doc_id in doc_ids:
        await knowledge_repo.create(_doc_row(tenant_id=tid, kb_id=kb_id, doc_id=doc_id))
    chunk_ids = [f"chunk-{uuid.uuid4().hex[:12]}" for _ in range(3)]
    await chunk_repo.create_many(
        [
            _chunk_row(
                tenant_id=tid,
                kb_id=kb_id,
                knowledge_id=doc_ids[0],
                chunk_index=0,
                chunk_id=chunk_ids[0],
            ),
            _chunk_row(
                tenant_id=tid,
                kb_id=kb_id,
                knowledge_id=doc_ids[0],
                chunk_index=1,
                chunk_id=chunk_ids[1],
            ),
            _chunk_row(
                tenant_id=tid,
                kb_id=kb_id,
                knowledge_id=doc_ids[1],
                chunk_index=0,
                chunk_id=chunk_ids[2],
            ),
        ]
    )
    await session.commit()

    result = await process_kb_delete(
        tenant_id=tid,
        knowledge_base_id=kb_id,
        knowledge_repo=knowledge_repo,
        chunk_repo=chunk_repo,
        now=_NOW,
    )

    assert result.knowledge_ids == tuple(doc_ids)
    assert result.deleted_chunks == 3
    assert result.deleted_knowledge == 2
    for doc_id in doc_ids:
        assert await knowledge_repo.get_by_id(tid, doc_id) is None
    for chunk_id in chunk_ids:
        assert await chunk_repo.get_by_id_or_none(tid, chunk_id) is None


async def test_integration_second_pass_is_a_noop(session: AsyncSession) -> None:
    tid = _int32_tenant_id()
    kb_id = f"kb-{uuid.uuid4().hex[:12]}"
    doc_id = f"doc-{uuid.uuid4().hex[:12]}"
    knowledge_repo = KnowledgeRepository(session)
    chunk_repo = ChunkRepository(session)
    await knowledge_repo.create(_doc_row(tenant_id=tid, kb_id=kb_id, doc_id=doc_id))
    await chunk_repo.create_many(
        [
            _chunk_row(
                tenant_id=tid,
                kb_id=kb_id,
                knowledge_id=doc_id,
                chunk_index=0,
                chunk_id=f"chunk-{uuid.uuid4().hex[:12]}",
            )
        ]
    )
    await session.commit()

    first = await process_kb_delete(
        tenant_id=tid,
        knowledge_base_id=kb_id,
        knowledge_repo=knowledge_repo,
        chunk_repo=chunk_repo,
        now=_NOW,
    )
    second = await process_kb_delete(
        tenant_id=tid,
        knowledge_base_id=kb_id,
        knowledge_repo=knowledge_repo,
        chunk_repo=chunk_repo,
        now=_NOW,
    )

    assert first.deleted_chunks == 1
    assert first.deleted_knowledge == 1
    assert second.knowledge_ids == ()
    assert second.deleted_chunks == 0
    assert second.deleted_knowledge == 0


async def test_integration_chunk_sweep_failure_keeps_document_delete(
    session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tid = _int32_tenant_id()
    kb_id = f"kb-{uuid.uuid4().hex[:12]}"
    doc_id = f"doc-{uuid.uuid4().hex[:12]}"
    chunk_id = f"chunk-{uuid.uuid4().hex[:12]}"
    knowledge_repo = KnowledgeRepository(session)
    chunk_repo = ChunkRepository(session)
    await knowledge_repo.create(_doc_row(tenant_id=tid, kb_id=kb_id, doc_id=doc_id))
    await chunk_repo.create_many(
        [
            _chunk_row(
                tenant_id=tid,
                kb_id=kb_id,
                knowledge_id=doc_id,
                chunk_index=0,
                chunk_id=chunk_id,
            )
        ]
    )
    await session.commit()

    async def _failing_sweep(*, tenant_id: int, knowledge_id: str, now: datetime) -> int:
        raise RuntimeError("sweep failed")

    monkeypatch.setattr(chunk_repo, "delete_by_knowledge_id", _failing_sweep)

    result = await process_kb_delete(
        tenant_id=tid,
        knowledge_base_id=kb_id,
        knowledge_repo=knowledge_repo,
        chunk_repo=chunk_repo,
        now=_NOW,
    )

    assert result.deleted_chunks == 0
    assert result.deleted_knowledge == 1
    assert await knowledge_repo.get_by_id(tid, doc_id) is None
    assert await chunk_repo.get_by_id_or_none(tid, chunk_id) is not None


async def test_integration_soft_deleted_kb_cascades_via_service_flow(
    session: AsyncSession,
) -> None:
    tid = _int32_tenant_id()
    kb_repo = KnowledgeBaseRepository(session)
    knowledge_repo = KnowledgeRepository(session)
    chunk_repo = ChunkRepository(session)
    kb_service = KBService(kb_repo=kb_repo)
    kb = await kb_service.create_knowledge_base(tenant_id=tid, name="docs")
    doc_id = f"doc-{uuid.uuid4().hex[:12]}"
    await knowledge_repo.create(_doc_row(tenant_id=tid, kb_id=kb.id, doc_id=doc_id))
    await chunk_repo.create_many(
        [
            _chunk_row(
                tenant_id=tid,
                kb_id=kb.id,
                knowledge_id=doc_id,
                chunk_index=0,
                chunk_id=f"chunk-{uuid.uuid4().hex[:12]}",
            )
        ]
    )
    await session.commit()

    deleted = await kb_service.delete_knowledge_base(knowledge_base_id=kb.id)
    await session.commit()
    assert deleted is True

    result = await process_kb_delete(
        tenant_id=tid,
        knowledge_base_id=kb.id,
        knowledge_repo=knowledge_repo,
        chunk_repo=chunk_repo,
        vector_store_id=kb.vector_store_id,
        now=_NOW,
    )

    assert result.deleted_chunks == 1
    assert result.deleted_knowledge == 1
    assert result.vector_store_id == kb.vector_store_id
    assert await knowledge_repo.get_by_id(tid, doc_id) is None
