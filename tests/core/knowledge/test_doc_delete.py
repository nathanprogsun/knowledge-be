"""Unit + integration tests for the document delete modules.

``delete_knowledge`` (single) and ``delete_knowledge_list`` (batch) are
standalone cascade deletes: they validate the tenant / document scope,
resolve the live row(s), mark each mid-deletion, soft-delete their
chunks, then soft-delete the rows.

Unit tests drive the functions with stateful repository mocks (closure
captured storage, the pattern used across the core service tests).

Integration tests run against the real applied schema (``documents`` /
``chunks`` tables) using the tenant factories; they require a reachable
database — run with ``DATABASE_URL_OVERRIDE``.
"""

from __future__ import annotations

import itertools
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from random import randint
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from faker import Faker
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from src.common.exception import NotFoundError, ValidationError
from src.core.knowledge.documents.delete import (
    delete_knowledge,
)
from src.core.knowledge.documents.list_delete import delete_knowledge_list
from src.core.knowledge.documents.service.knowledge_service import KnowledgeService
from src.core.knowledge.documents.types import PARSE_STATUS_DELETING
from src.db.dao.chunk_repository import ChunkRepository
from src.db.dao.knowledge_repository import KnowledgeRepository
from src.db.models.chunk import Chunk
from src.db.models.knowledge import Document
from src.settings import get_settings, reset_settings_cache
from tests.integration.conftest import make_test_tenant_id

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_FAKER_SEED_MAX = 100_000_000
# ``chunks.tenant_id`` is a 32-bit INTEGER, so integration rows need a
# 32-bit-safe unique id (the ``tenants`` table's 64-bit ids overflow it).
_int32_tenant_counter = itertools.count(9_000_000)


@pytest.fixture(autouse=True)
def faker_seed() -> None:
    """Re-seed Faker per test for varied-but-reproducible generation."""
    Faker.seed(randint(1, _FAKER_SEED_MAX))


def _did() -> str:
    return f"doc-{uuid.uuid4().hex[:12]}"


def _kbid() -> str:
    return f"kb-{uuid.uuid4().hex[:12]}"


def _cid() -> str:
    return f"chunk-{uuid.uuid4().hex[:12]}"


def _int32_tenant_id() -> int:
    """Return a unique 32-bit tenant id for the ``chunks`` table."""
    return next(_int32_tenant_counter)


def _sample_document(
    *,
    id: str | None = None,
    tenant_id: int,
    knowledge_base_id: str | None = None,
    title: str = "Q3 budget",
    **columns: object,
) -> Document:
    """Build a persisted-shape document row for seeding mocks / DB."""
    return Document.model_validate(
        {
            "id": id or _did(),
            "tenant_id": tenant_id,
            "knowledge_base_id": knowledge_base_id or _kbid(),
            "type": "file",
            "title": title,
            "description": None,
            "source": "budget-2026.pdf",
            "channel": "web",
            "parse_status": "completed",
            "pending_subtasks_count": 0,
            "summary_status": "none",
            "enable_status": "enabled",
            "embedding_model_id": None,
            "file_name": "budget-2026.pdf",
            "file_type": "pdf",
            "file_size": 1024,
            "file_hash": None,
            "file_path": None,
            "storage_size": 2048,
            "metadata": None,
            "custom_metadata": {},
            "last_faq_import_result": None,
            "created_at": _NOW,
            "updated_at": _NOW,
            "processed_at": None,
            "error_message": None,
            "deleted_at": None,
            **columns,
        }
    )


def _sample_chunk(
    *,
    id: str | None = None,
    tenant_id: int,
    knowledge_base_id: str,
    knowledge_id: str,
    content: str = "The quick brown fox jumps over the lazy dog.",
    chunk_index: int = 0,
    **columns: object,
) -> Chunk:
    """Build a persisted-shape chunk row for seeding mocks / DB."""
    return Chunk.model_validate(
        {
            "id": id or _cid(),
            "tenant_id": tenant_id,
            "knowledge_base_id": knowledge_base_id,
            "knowledge_id": knowledge_id,
            "content": content,
            "chunk_index": chunk_index,
            "is_enabled": True,
            "start_at": 0,
            "end_at": len(content),
            "chunk_type": "text",
            "flags": 1,
            "source_content": "",
            "content_revision": 0,
            "index_status": "ready",
            "last_editor_id": "",
            "created_at": _NOW,
            "updated_at": _NOW,
            "deleted_at": None,
            **columns,
        }
    )


# ── Repository mocks (stateful via side_effect closures) ───────────────


def _make_repos() -> tuple[AsyncMock, AsyncMock, dict[str, Document], dict[str, Chunk]]:
    """Return knowledge + chunk mocks with closure-captured storage."""
    knowledge_repo = AsyncMock(spec=KnowledgeRepository)
    chunk_repo = AsyncMock(spec=ChunkRepository)
    documents: dict[str, Document] = {}
    chunks: dict[str, Chunk] = {}

    async def _get_by_id(tenant_id: int, id: str) -> Document | None:
        row = documents.get(id)
        if row is not None and row.tenant_id == tenant_id and row.deleted_at is None:
            return row
        return None

    async def _get_batch(tenant_id: int, ids: list[str]) -> list[Document]:
        out: list[Document] = []
        for id in ids:
            row = documents.get(id)
            if row is not None and row.tenant_id == tenant_id and row.deleted_at is None:
                out.append(row)
        return out

    async def _update_columns(id: str, values: dict[str, object]) -> Document | None:
        row = documents.get(id)
        if row is None or row.deleted_at is not None:
            return None
        updated = row.model_copy(update=values)
        documents[id] = updated
        return updated

    async def _soft_delete(*, tenant_id: int, id: str, now: datetime) -> bool:
        row = documents.get(id)
        if row is None or row.tenant_id != tenant_id or row.deleted_at is not None:
            return False
        documents[id] = row.model_copy(update={"deleted_at": now, "updated_at": now})
        return True

    async def _soft_delete_list(*, tenant_id: int, ids: list[str], now: datetime) -> int:
        removed = 0
        for id in ids:
            row = documents.get(id)
            if row is not None and row.tenant_id == tenant_id and row.deleted_at is None:
                documents[id] = row.model_copy(update={"deleted_at": now, "updated_at": now})
                removed += 1
        return removed

    async def _delete_by_knowledge_id(
        *,
        tenant_id: int,
        knowledge_id: str,
        now: datetime,
    ) -> int:
        affected = 0
        for cid, chunk in list(chunks.items()):
            if (
                chunk.tenant_id == tenant_id
                and chunk.knowledge_id == knowledge_id
                and chunk.deleted_at is None
            ):
                chunks[cid] = chunk.model_copy(update={"deleted_at": now, "updated_at": now})
                affected += 1
        return affected

    knowledge_repo.get_by_id.side_effect = _get_by_id
    knowledge_repo.get_batch.side_effect = _get_batch
    knowledge_repo.update_columns.side_effect = _update_columns
    knowledge_repo.soft_delete.side_effect = _soft_delete
    knowledge_repo.soft_delete_list.side_effect = _soft_delete_list
    chunk_repo.delete_by_knowledge_id.side_effect = _delete_by_knowledge_id
    return knowledge_repo, chunk_repo, documents, chunks


def _seed_document_with_chunks(
    documents: dict[str, Document],
    chunks: dict[str, Chunk],
    *,
    tenant_id: int,
    knowledge_base_id: str | None = None,
) -> tuple[Document, list[Chunk]]:
    """Seed one live document plus two live chunks into mock storage."""
    doc = _sample_document(tenant_id=tenant_id, knowledge_base_id=knowledge_base_id)
    documents[doc.id] = doc
    seeded = [
        _sample_chunk(
            tenant_id=tenant_id,
            knowledge_base_id=doc.knowledge_base_id,
            knowledge_id=doc.id,
            chunk_index=0,
        ),
        _sample_chunk(
            tenant_id=tenant_id,
            knowledge_base_id=doc.knowledge_base_id,
            knowledge_id=doc.id,
            chunk_index=1,
            content="A second sentence to split over two chunks.",
        ),
    ]
    for chunk in seeded:
        chunks[chunk.id] = chunk
    return doc, seeded


# ── Single delete (unit) ───────────────────────────────────────────────


async def test_delete_knowledge_marks_deleting_cascades_and_soft_deletes() -> None:
    knowledge_repo, chunk_repo, documents, chunks = _make_repos()
    tenant_id = make_test_tenant_id()
    doc, seeded = _seed_document_with_chunks(
        documents,
        chunks,
        tenant_id=tenant_id,
    )

    removed = await delete_knowledge(
        tenant_id=tenant_id,
        id=doc.id,
        knowledge_repo=knowledge_repo,
        chunk_repo=chunk_repo,
    )

    assert removed is True
    stored = documents[doc.id]
    assert stored.deleted_at is not None
    assert stored.parse_status == PARSE_STATUS_DELETING
    for chunk in seeded:
        assert chunks[chunk.id].deleted_at is not None


async def test_delete_knowledge_raises_for_missing_row() -> None:
    knowledge_repo, chunk_repo, _documents, _chunks = _make_repos()
    with pytest.raises(NotFoundError) as exc_info:
        await delete_knowledge(
            tenant_id=1,
            id="doc-missing",
            knowledge_repo=knowledge_repo,
            chunk_repo=chunk_repo,
        )
    assert exc_info.value.code == "knowledge.not_found"


async def test_delete_knowledge_raises_for_cross_tenant_row() -> None:
    knowledge_repo, chunk_repo, documents, chunks = _make_repos()
    tenant_id = make_test_tenant_id()
    doc, _seeded = _seed_document_with_chunks(
        documents,
        chunks,
        tenant_id=tenant_id,
    )
    with pytest.raises(NotFoundError):
        await delete_knowledge(
            tenant_id=tenant_id + 1,
            id=doc.id,
            knowledge_repo=knowledge_repo,
            chunk_repo=chunk_repo,
        )


async def test_delete_knowledge_raises_when_already_deleted() -> None:
    knowledge_repo, chunk_repo, documents, chunks = _make_repos()
    tenant_id = make_test_tenant_id()
    doc, seeded = _seed_document_with_chunks(
        documents,
        chunks,
        tenant_id=tenant_id,
    )
    assert (
        await delete_knowledge(
            tenant_id=tenant_id,
            id=doc.id,
            knowledge_repo=knowledge_repo,
            chunk_repo=chunk_repo,
        )
        is True
    )
    with pytest.raises(NotFoundError):
        await delete_knowledge(
            tenant_id=tenant_id,
            id=doc.id,
            knowledge_repo=knowledge_repo,
            chunk_repo=chunk_repo,
        )
    for chunk in seeded:
        assert chunks[chunk.id].deleted_at is not None


async def test_delete_knowledge_validates_scope() -> None:
    knowledge_repo, chunk_repo, _documents, _chunks = _make_repos()
    with pytest.raises(ValidationError) as exc_info:
        await delete_knowledge(
            tenant_id=0,
            id="doc-1",
            knowledge_repo=knowledge_repo,
            chunk_repo=chunk_repo,
        )
    assert exc_info.value.code == "knowledge.tenant_required"
    with pytest.raises(ValidationError) as exc_info:
        await delete_knowledge(
            tenant_id=1,
            id=" ",
            knowledge_repo=knowledge_repo,
            chunk_repo=chunk_repo,
        )
    assert exc_info.value.code == "knowledge.id_required"


async def test_delete_knowledge_does_not_touch_other_tenants_chunks() -> None:
    knowledge_repo, chunk_repo, documents, chunks = _make_repos()
    tenant_id = make_test_tenant_id()
    doc, seeded = _seed_document_with_chunks(
        documents,
        chunks,
        tenant_id=tenant_id,
    )
    # A chunk owned by another tenant under the same knowledge id must survive.
    foreign_chunk = _sample_chunk(
        tenant_id=tenant_id + 1,
        knowledge_base_id=doc.knowledge_base_id,
        knowledge_id=doc.id,
    )
    chunks[foreign_chunk.id] = foreign_chunk

    assert (
        await delete_knowledge(
            tenant_id=tenant_id,
            id=doc.id,
            knowledge_repo=knowledge_repo,
            chunk_repo=chunk_repo,
        )
        is True
    )
    for chunk in seeded:
        assert chunks[chunk.id].deleted_at is not None
    assert chunks[foreign_chunk.id].deleted_at is None


# ── Batch delete (unit) ────────────────────────────────────────────────


async def test_delete_knowledge_list_cascades_and_returns_count() -> None:
    knowledge_repo, chunk_repo, documents, chunks = _make_repos()
    tenant_id = make_test_tenant_id()
    doc_a, seeded_a = _seed_document_with_chunks(
        documents,
        chunks,
        tenant_id=tenant_id,
    )
    doc_b, seeded_b = _seed_document_with_chunks(
        documents,
        chunks,
        tenant_id=tenant_id,
    )

    removed = await delete_knowledge_list(
        tenant_id=tenant_id,
        ids=[doc_a.id, doc_b.id],
        knowledge_repo=knowledge_repo,
        chunk_repo=chunk_repo,
    )

    assert removed == 2
    assert documents[doc_a.id].deleted_at is not None
    assert documents[doc_b.id].deleted_at is not None
    assert documents[doc_a.id].parse_status == PARSE_STATUS_DELETING
    for chunk in seeded_a + seeded_b:
        assert chunks[chunk.id].deleted_at is not None


async def test_delete_knowledge_list_empty_and_blank_ids_skip_database() -> None:
    knowledge_repo, chunk_repo, _documents, _chunks = _make_repos()
    assert (
        await delete_knowledge_list(
            tenant_id=1,
            ids=[],
            knowledge_repo=knowledge_repo,
            chunk_repo=chunk_repo,
        )
        == 0
    )
    assert (
        await delete_knowledge_list(
            tenant_id=1,
            ids=["", "  "],
            knowledge_repo=knowledge_repo,
            chunk_repo=chunk_repo,
        )
        == 0
    )
    knowledge_repo.get_batch.assert_not_awaited()


async def test_delete_knowledge_list_drops_absent_and_cross_tenant_ids() -> None:
    knowledge_repo, chunk_repo, documents, chunks = _make_repos()
    tenant_id = make_test_tenant_id()
    doc, seeded = _seed_document_with_chunks(
        documents,
        chunks,
        tenant_id=tenant_id,
    )
    other, seeded_other = _seed_document_with_chunks(
        documents,
        chunks,
        tenant_id=tenant_id + 1,
    )

    removed = await delete_knowledge_list(
        tenant_id=tenant_id,
        ids=[doc.id, other.id, "doc-missing", ""],
        knowledge_repo=knowledge_repo,
        chunk_repo=chunk_repo,
    )

    assert removed == 1
    assert documents[doc.id].deleted_at is not None
    assert documents[other.id].deleted_at is None
    for chunk in seeded:
        assert chunks[chunk.id].deleted_at is not None
    for chunk in seeded_other:
        assert chunks[chunk.id].deleted_at is None


async def test_delete_knowledge_list_validates_tenant() -> None:
    knowledge_repo, chunk_repo, _documents, _chunks = _make_repos()
    with pytest.raises(ValidationError) as exc_info:
        await delete_knowledge_list(
            tenant_id=-1,
            ids=["doc-1"],
            knowledge_repo=knowledge_repo,
            chunk_repo=chunk_repo,
        )
    assert exc_info.value.code == "knowledge.tenant_required"


async def test_delete_knowledge_list_no_live_rows_returns_zero() -> None:
    knowledge_repo, chunk_repo, documents, chunks = _make_repos()
    tenant_id = make_test_tenant_id()
    doc, _seeded = _seed_document_with_chunks(
        documents,
        chunks,
        tenant_id=tenant_id,
    )
    # Soft-delete the row first: batch delete treats it as absent.
    await knowledge_repo.soft_delete(tenant_id=tenant_id, id=doc.id, now=_NOW)
    removed = await delete_knowledge_list(
        tenant_id=tenant_id,
        ids=[doc.id],
        knowledge_repo=knowledge_repo,
        chunk_repo=chunk_repo,
    )
    assert removed == 0


# ── Integration (real applied schema) ──────────────────────────────────


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


async def test_integration_delete_knowledge_cascades_chunks(session: AsyncSession) -> None:
    tenant_id = _int32_tenant_id()
    kb_id = _kbid()
    knowledge_repo = KnowledgeRepository(session)
    chunk_repo = ChunkRepository(session)
    service = KnowledgeService(knowledge_repo=knowledge_repo)

    doc = await service.create_document(
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
        type="file",
        title="Integration delete",
        source="delete-2026.pdf",
    )
    await chunk_repo.create_many(
        [
            _sample_chunk(
                tenant_id=tenant_id,
                knowledge_base_id=kb_id,
                knowledge_id=doc.id,
                chunk_index=0,
            ),
            _sample_chunk(
                tenant_id=tenant_id,
                knowledge_base_id=kb_id,
                knowledge_id=doc.id,
                chunk_index=1,
                content="A second sentence to split over two chunks.",
            ),
        ]
    )
    assert len(await chunk_repo.list_by_knowledge_id(tenant_id, doc.id)) == 2

    removed = await delete_knowledge(
        tenant_id=tenant_id,
        id=doc.id,
        knowledge_repo=knowledge_repo,
        chunk_repo=chunk_repo,
    )

    assert removed is True
    assert await knowledge_repo.get_by_id(tenant_id, doc.id) is None
    assert await chunk_repo.list_by_knowledge_id(tenant_id, doc.id) == []
    with pytest.raises(NotFoundError):
        await service.get_document(tenant_id=tenant_id, id=doc.id)


async def test_integration_delete_knowledge_raises_for_missing(session: AsyncSession) -> None:
    tenant_id = make_test_tenant_id()
    knowledge_repo = KnowledgeRepository(session)
    chunk_repo = ChunkRepository(session)
    with pytest.raises(NotFoundError) as exc_info:
        await delete_knowledge(
            tenant_id=tenant_id,
            id="doc-missing",
            knowledge_repo=knowledge_repo,
            chunk_repo=chunk_repo,
        )
    assert exc_info.value.code == "knowledge.not_found"


async def test_integration_delete_knowledge_list_cascades(session: AsyncSession) -> None:
    tenant_id = _int32_tenant_id()
    kb_id = _kbid()
    knowledge_repo = KnowledgeRepository(session)
    chunk_repo = ChunkRepository(session)
    service = KnowledgeService(knowledge_repo=knowledge_repo)

    doc_a = await service.create_document(
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
        type="file",
        title="a",
        source="a.pdf",
    )
    doc_b = await service.create_document(
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
        type="file",
        title="b",
        source="b.pdf",
    )
    await chunk_repo.create_many(
        [
            _sample_chunk(
                tenant_id=tenant_id,
                knowledge_base_id=kb_id,
                knowledge_id=doc_a.id,
                chunk_index=0,
            )
        ]
    )

    removed = await delete_knowledge_list(
        tenant_id=tenant_id,
        ids=[doc_a.id, doc_b.id],
        knowledge_repo=knowledge_repo,
        chunk_repo=chunk_repo,
    )

    assert removed == 2
    assert await knowledge_repo.get_by_id(tenant_id, doc_a.id) is None
    assert await knowledge_repo.get_by_id(tenant_id, doc_b.id) is None
    assert await chunk_repo.list_by_knowledge_id(tenant_id, doc_a.id) == []


async def test_integration_delete_knowledge_list_drops_absent(session: AsyncSession) -> None:
    tenant_id = _int32_tenant_id()
    kb_id = _kbid()
    knowledge_repo = KnowledgeRepository(session)
    chunk_repo = ChunkRepository(session)
    service = KnowledgeService(knowledge_repo=knowledge_repo)

    doc = await service.create_document(
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
        type="file",
        title="keep",
        source="keep.pdf",
    )

    removed = await delete_knowledge_list(
        tenant_id=tenant_id,
        ids=[doc.id, "doc-missing", ""],
        knowledge_repo=knowledge_repo,
        chunk_repo=chunk_repo,
    )

    assert removed == 1
    assert await knowledge_repo.get_by_id(tenant_id, doc.id) is None
