"""Unit + integration tests for the reparse / cancel / image-update modules.

Unit tests drive each standalone function with stateful repository mocks
(closure-captured storage, the same pattern used across the core service
tests): they cover validation, error classification, and the happy paths.

Integration tests run against the real applied schema (``documents`` and
``chunks`` tables). Tests that seed ``chunks`` rows use an int32-safe
tenant id (the ``chunks.tenant_id`` column is INTEGER); tests that only
touch ``documents`` use the BIGINT ``make_test_tenant_id`` factory. They
require a reachable database - run with ``DATABASE_URL_OVERRIDE``.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from hashlib import md5
from random import randint
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from faker import Faker
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from src.common.exception import DataError, NotFoundError, ValidationError
from src.common.json import BindParams, JsonObject
from src.core.contracts.knowledge import Knowledge
from src.core.knowledge.chunks.service.chunk_service import ChunkService
from src.core.knowledge.chunks.types import (
    CHUNK_TYPE_IMAGE_CAPTION,
    CHUNK_TYPE_IMAGE_OCR,
    CHUNK_TYPE_TEXT,
)
from src.core.knowledge.documents.cancel import cancel_knowledge_parse
from src.core.knowledge.documents.image_update import ImageInfo, update_document_image
from src.core.knowledge.documents.reparse import DocumentProcessPayload, reparse_knowledge
from src.core.knowledge.documents.service.knowledge_service import KnowledgeService
from src.core.knowledge.knowledge_bases.service.kb_service import KBService
from src.core.knowledge.knowledge_bases.types import KnowledgeBaseInfo
from src.db.dao.chunk_repository import ChunkRepository
from src.db.dao.knowledge_base_repository import KnowledgeBaseRepository
from src.db.dao.knowledge_repository import KnowledgeRepository
from src.db.models.chunk import Chunk
from src.db.models.knowledge import Document
from src.settings import get_settings, reset_settings_cache
from tests.integration.conftest import make_test_tenant_id

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_FAKER_SEED_MAX = 100_000_000

# ``chunks.tenant_id`` is an INTEGER column; ids minted here stay inside
# the int32 range so integration rows never overflow.
_INT32_TENANT_MAX = 2**31 - 1
_used_int32_tenant_ids: set[int] = set()


def _int32_tenant_id() -> int:
    """Return a tenant id that fits the ``chunks.tenant_id`` INTEGER column."""
    while True:
        candidate = randint(1, _INT32_TENANT_MAX)
        if candidate not in _used_int32_tenant_ids:
            _used_int32_tenant_ids.add(candidate)
            return candidate


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


def _doc_row(
    *,
    id: str | None = None,
    tenant_id: int | None = None,
    knowledge_base_id: str | None = None,
    type: str = "file",
    title: str = "budget",
    source: str = "budget-2026.pdf",
    parse_status: str = "completed",
    file_name: str | None = "budget-2026.pdf",
    file_type: str | None = "pdf",
    file_path: str | None = "/tmp/budget-2026.pdf",
    file_hash: str | None = None,
    metadata: JsonObject | None = None,
    **columns: object,
) -> Document:
    """Build a persisted-shape document row for seeding mocks / DB."""
    return Document.model_validate(
        {
            "id": id or _did(),
            "tenant_id": tenant_id if tenant_id is not None else _int32_tenant_id(),
            "knowledge_base_id": knowledge_base_id or _kbid(),
            "type": type,
            "title": title,
            "description": None,
            "source": source,
            "channel": "web",
            "parse_status": parse_status,
            "pending_subtasks_count": 0,
            "summary_status": "none",
            "enable_status": "enabled",
            "embedding_model_id": None,
            "file_name": file_name,
            "file_type": file_type,
            "file_size": 1024,
            "file_hash": file_hash,
            "file_path": file_path,
            "storage_size": 2048,
            "metadata": metadata,
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


def _chunk_row(
    *,
    id: str | None = None,
    tenant_id: int,
    knowledge_base_id: str,
    knowledge_id: str,
    content: str = "chunk content",
    chunk_type: str = CHUNK_TYPE_TEXT,
    parent_chunk_id: str | None = None,
    image_info: str | None = None,
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
            "chunk_index": 0,
            "is_enabled": True,
            "start_at": 0,
            "end_at": 0,
            "chunk_type": chunk_type,
            "parent_chunk_id": parent_chunk_id,
            "image_info": image_info,
            "created_at": _NOW,
            "updated_at": _NOW,
            **columns,
        }
    )


def _kb_info(
    *,
    tenant_id: int,
    kb_id: str,
    **overrides: object,
) -> KnowledgeBaseInfo:
    """Build a knowledge-base service projection for the mock service."""
    return KnowledgeBaseInfo.model_validate(
        {
            "id": kb_id,
            "name": "docs",
            "tenant_id": tenant_id,
            "created_at": _NOW,
            "updated_at": _NOW,
            **overrides,
        }
    )


# ── Repository mocks (stateful via side_effect closures) ────────────────


def _make_repo() -> tuple[AsyncMock, dict[str, Document]]:
    """Knowledge-repository mock with closure-captured storage."""
    repo = AsyncMock(spec=KnowledgeRepository)
    rows: dict[str, Document] = {}

    async def _get_by_id(tenant_id: int, id: str) -> Document | None:
        row = rows.get(id)
        if row is not None and row.tenant_id == tenant_id and row.deleted_at is None:
            return row
        return None

    async def _update(row: Document) -> Document:
        rows[row.id] = row
        return row

    async def _update_columns(id: str, values: BindParams) -> Document | None:
        row = rows.get(id)
        if row is None:
            return None
        updated = row.model_copy(update=dict(values))
        rows[id] = updated
        return updated

    repo.get_by_id.side_effect = _get_by_id
    repo.update.side_effect = _update
    repo.update_columns.side_effect = _update_columns
    return repo, rows


def _make_kb_service(kb: KnowledgeBaseInfo) -> AsyncMock:
    """Knowledge-base service mock returning one projection."""
    service = AsyncMock(spec=KBService)

    async def _get_by_id_and_tenant(*, tenant_id: int, knowledge_base_id: str) -> KnowledgeBaseInfo:
        return kb

    service.get_knowledge_base_by_id_and_tenant.side_effect = _get_by_id_and_tenant
    return service


class _FakeChunkService:
    """Stateful stand-in for ``ChunkService`` covering the methods used here."""

    def __init__(self, chunks: dict[str, Chunk] | None = None) -> None:
        self._chunks: dict[str, Chunk] = chunks if chunks is not None else {}
        self.created: list[Chunk] = []
        self.updated: list[Chunk] = []
        self.soft_deleted: list[str] = []

    async def get_chunk_by_id(self, *, tenant_id: int, id: str) -> Chunk:
        row = self._chunks.get(id)
        if row is None or row.tenant_id != tenant_id:
            raise NotFoundError(code="chunk.not_found", message=f"chunk {id} not found")
        return row

    async def list_chunk_by_parent_id(self, *, tenant_id: int, parent_id: str) -> list[Chunk]:
        return [
            row
            for row in self._chunks.values()
            if row.tenant_id == tenant_id and row.parent_chunk_id == parent_id
        ]

    async def create_chunks(self, *, chunks: list[Chunk]) -> list[Chunk]:
        for chunk in chunks:
            self._chunks[chunk.id] = chunk
            self.created.append(chunk)
        return chunks

    async def update_chunks(self, *, chunks: list[Chunk]) -> list[Chunk]:
        for chunk in chunks:
            self._chunks[chunk.id] = chunk
            self.updated.append(chunk)
        return chunks

    async def delete_chunks_by_knowledge_id(self, *, tenant_id: int, knowledge_id: str) -> int:
        removed = 0
        for chunk in list(self._chunks.values()):
            if chunk.tenant_id == tenant_id and chunk.knowledge_id == knowledge_id:
                self._chunks[chunk.id] = chunk.model_copy(update={"deleted_at": _NOW})
                self.soft_deleted.append(chunk.id)
                removed += 1
        return removed


class _FakeEnqueuer:
    """Records reparse submissions; optionally fails when ``error`` is set."""

    def __init__(self) -> None:
        self.manual: list[tuple[int, str, str]] = []
        self.document: list[DocumentProcessPayload] = []
        self.error: Exception | None = None

    async def enqueue_manual_process(
        self,
        *,
        tenant_id: int,
        knowledge_id: str,
        content: str,
    ) -> None:
        if self.error is not None:
            raise self.error
        self.manual.append((tenant_id, knowledge_id, content))

    async def enqueue_document_process(
        self,
        *,
        tenant_id: int,
        payload: DocumentProcessPayload,
    ) -> None:
        if self.error is not None:
            raise self.error
        self.document.append(payload)


class _FakeInspector:
    """Records parse-task dequeue requests; optionally fails when set."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.error: Exception | None = None

    async def cancel_tasks_for_knowledge(self, *, knowledge_id: str) -> None:
        if self.error is not None:
            raise self.error
        self.calls.append(knowledge_id)


# ═══════════════════════════════════════════════════════════════════════
# reparse_knowledge - unit tests
# ═══════════════════════════════════════════════════════════════════════


async def test_reparse_file_document_resets_and_submits_task() -> None:
    repo, rows = _make_repo()
    tenant_id = _int32_tenant_id()
    doc = _doc_row(
        tenant_id=tenant_id,
        knowledge_base_id="kb-1",
        file_path="/tmp/budget-2026.pdf",
        file_name="budget-2026.pdf",
        file_type="pdf",
    )
    rows[doc.id] = doc
    enqueuer = _FakeEnqueuer()
    chunk_service = _FakeChunkService()

    result = await reparse_knowledge(
        tenant_id=tenant_id,
        knowledge_id=doc.id,
        knowledge_repo=repo,
        kb_service=_make_kb_service(
            _kb_info(tenant_id=tenant_id, kb_id="kb-1", embedding_model_id="embed-9")
        ),
        chunk_service=chunk_service,
        enqueuer=enqueuer,
    )

    assert isinstance(result, Knowledge)
    assert result.id == doc.id
    assert result.parse_status == "pending"
    assert result.enable_status == "disabled"
    stored = rows[doc.id]
    assert stored.parse_status == "pending"
    assert stored.enable_status == "disabled"
    assert stored.embedding_model_id == "embed-9"
    assert stored.pending_subtasks_count == 0
    assert chunk_service.soft_deleted == []
    assert len(enqueuer.document) == 1
    payload = enqueuer.document[0]
    assert payload.tenant_id == tenant_id
    assert payload.knowledge_id == doc.id
    assert payload.knowledge_base_id == "kb-1"
    assert payload.file_path == "/tmp/budget-2026.pdf"
    assert payload.file_name == "budget-2026.pdf"
    assert payload.file_type == "pdf"
    assert payload.file_url is None
    assert payload.url is None
    # The counter reset goes through the explicit column update.
    repo.update_columns.assert_awaited_with(doc.id, {"pending_subtasks_count": 0})


async def test_reparse_non_manual_cleans_existing_chunks() -> None:
    repo, rows = _make_repo()
    tenant_id = _int32_tenant_id()
    doc = _doc_row(tenant_id=tenant_id, knowledge_base_id="kb-1")
    rows[doc.id] = doc
    chunk = _chunk_row(
        tenant_id=tenant_id,
        knowledge_base_id="kb-1",
        knowledge_id=doc.id,
    )
    chunk_service = _FakeChunkService({chunk.id: chunk})

    await reparse_knowledge(
        tenant_id=tenant_id,
        knowledge_id=doc.id,
        knowledge_repo=repo,
        kb_service=_make_kb_service(_kb_info(tenant_id=tenant_id, kb_id="kb-1")),
        chunk_service=chunk_service,
        enqueuer=_FakeEnqueuer(),
    )

    assert chunk_service.soft_deleted == [chunk.id]
    assert chunk_service._chunks[chunk.id].deleted_at is not None


async def test_reparse_manual_path_submits_manual_task_without_cleanup() -> None:
    repo, rows = _make_repo()
    tenant_id = _int32_tenant_id()
    doc = _doc_row(
        tenant_id=tenant_id,
        knowledge_base_id="kb-1",
        type="manual",
        source="manual",
        file_name=None,
        file_type="manual",
        file_path=None,
        metadata={"content": "hello manual"},
    )
    rows[doc.id] = doc
    enqueuer = _FakeEnqueuer()
    chunk_service = _FakeChunkService()

    result = await reparse_knowledge(
        tenant_id=tenant_id,
        knowledge_id=doc.id,
        knowledge_repo=repo,
        kb_service=_make_kb_service(_kb_info(tenant_id=tenant_id, kb_id="kb-1")),
        chunk_service=chunk_service,
        enqueuer=enqueuer,
    )

    assert result.parse_status == "pending"
    assert enqueuer.manual == [(tenant_id, doc.id, "hello manual")]
    assert enqueuer.document == []
    assert chunk_service.soft_deleted == []
    repo.update_columns.assert_awaited_with(doc.id, {"pending_subtasks_count": 0})


async def test_reparse_missing_document_raises_not_found() -> None:
    repo, _rows = _make_repo()
    with pytest.raises(NotFoundError) as exc_info:
        await reparse_knowledge(
            tenant_id=_int32_tenant_id(),
            knowledge_id="doc-missing",
            knowledge_repo=repo,
            kb_service=_make_kb_service(_kb_info(tenant_id=1, kb_id="kb-1")),
            chunk_service=_FakeChunkService(),
            enqueuer=_FakeEnqueuer(),
        )
    assert exc_info.value.code == "knowledge.not_found"


async def test_reparse_manual_without_content_raises_validation() -> None:
    repo, rows = _make_repo()
    tenant_id = _int32_tenant_id()
    doc = _doc_row(
        tenant_id=tenant_id,
        knowledge_base_id="kb-1",
        type="manual",
        source="manual",
        file_path=None,
        metadata=None,
    )
    rows[doc.id] = doc

    with pytest.raises(ValidationError) as exc_info:
        await reparse_knowledge(
            tenant_id=tenant_id,
            knowledge_id=doc.id,
            knowledge_repo=repo,
            kb_service=_make_kb_service(_kb_info(tenant_id=tenant_id, kb_id="kb-1")),
            chunk_service=_FakeChunkService(),
            enqueuer=_FakeEnqueuer(),
        )
    assert exc_info.value.code == "knowledge.manual_content_missing"


async def test_reparse_no_parseable_content_marks_failed() -> None:
    repo, rows = _make_repo()
    tenant_id = _int32_tenant_id()
    for file_path in (None, ""):
        doc = _doc_row(
            tenant_id=tenant_id,
            knowledge_base_id="kb-1",
            file_path=file_path,
            file_name=None,
            file_type=None,
        )
        rows[doc.id] = doc
        with pytest.raises(ValidationError) as exc_info:
            await reparse_knowledge(
                tenant_id=tenant_id,
                knowledge_id=doc.id,
                knowledge_repo=repo,
                kb_service=_make_kb_service(_kb_info(tenant_id=tenant_id, kb_id="kb-1")),
                chunk_service=_FakeChunkService(),
                enqueuer=_FakeEnqueuer(),
            )
        assert exc_info.value.code == "knowledge.not_parseable"
        assert rows[doc.id].parse_status == "failed"
        assert rows[doc.id].error_message == "Knowledge has no parseable content"
        del rows[doc.id]


async def test_reparse_enqueue_failure_marks_failed_and_reraises() -> None:
    repo, rows = _make_repo()
    tenant_id = _int32_tenant_id()
    doc = _doc_row(tenant_id=tenant_id, knowledge_base_id="kb-1")
    rows[doc.id] = doc
    enqueuer = _FakeEnqueuer()
    enqueuer.error = DataError(code="task.queue_unavailable", message="queue down")

    with pytest.raises(DataError):
        await reparse_knowledge(
            tenant_id=tenant_id,
            knowledge_id=doc.id,
            knowledge_repo=repo,
            kb_service=_make_kb_service(_kb_info(tenant_id=tenant_id, kb_id="kb-1")),
            chunk_service=_FakeChunkService(),
            enqueuer=enqueuer,
        )

    assert rows[doc.id].parse_status == "failed"
    assert rows[doc.id].error_message == "Failed to enqueue processing task"


async def test_reparse_persists_process_overrides_and_applies_flags() -> None:
    repo, rows = _make_repo()
    tenant_id = _int32_tenant_id()
    doc = _doc_row(tenant_id=tenant_id, knowledge_base_id="kb-1")
    rows[doc.id] = doc
    enqueuer = _FakeEnqueuer()

    await reparse_knowledge(
        tenant_id=tenant_id,
        knowledge_id=doc.id,
        knowledge_repo=repo,
        kb_service=_make_kb_service(_kb_info(tenant_id=tenant_id, kb_id="kb-1")),
        chunk_service=_FakeChunkService(),
        enqueuer=enqueuer,
        process_overrides={
            "enable_multimodel": True,
            "question_generation_config": {"enabled": True, "question_count": 5},
        },
    )

    stored = rows[doc.id]
    assert stored.metadata is not None
    assert stored.metadata["process_overrides"] == {
        "enable_multimodel": True,
        "question_generation_config": {"enabled": True, "question_count": 5},
    }
    assert enqueuer.document[0].enable_multimodel is True
    assert enqueuer.document[0].enable_question_generation is True
    assert enqueuer.document[0].question_count == 5


async def test_reparse_resolves_question_generation_from_kb_defaults() -> None:
    repo, rows = _make_repo()
    tenant_id = _int32_tenant_id()
    doc = _doc_row(tenant_id=tenant_id, knowledge_base_id="kb-1")
    rows[doc.id] = doc
    enqueuer = _FakeEnqueuer()

    await reparse_knowledge(
        tenant_id=tenant_id,
        knowledge_id=doc.id,
        knowledge_repo=repo,
        kb_service=_make_kb_service(
            _kb_info(
                tenant_id=tenant_id,
                kb_id="kb-1",
                chunking_config={"enable_multimodal": True},
                question_generation_config={"enabled": True, "question_count": 7},
            )
        ),
        chunk_service=_FakeChunkService(),
        enqueuer=enqueuer,
    )

    payload = enqueuer.document[0]
    assert payload.enable_multimodel is True
    assert payload.enable_question_generation is True
    assert payload.question_count == 7


async def test_reparse_routes_file_url_and_url_sources() -> None:
    repo, rows = _make_repo()
    tenant_id = _int32_tenant_id()
    file_url_doc = _doc_row(
        tenant_id=tenant_id,
        knowledge_base_id="kb-1",
        type="file_url",
        source="https://example.com/a.pdf",
        file_name="a.pdf",
        file_type="pdf",
        file_path=None,
    )
    url_doc = _doc_row(
        tenant_id=tenant_id,
        knowledge_base_id="kb-1",
        type="url",
        source="https://example.com/page",
        file_name=None,
        file_type=None,
        file_path=None,
    )
    rows[file_url_doc.id] = file_url_doc
    rows[url_doc.id] = url_doc
    enqueuer = _FakeEnqueuer()

    await reparse_knowledge(
        tenant_id=tenant_id,
        knowledge_id=file_url_doc.id,
        knowledge_repo=repo,
        kb_service=_make_kb_service(_kb_info(tenant_id=tenant_id, kb_id="kb-1")),
        chunk_service=_FakeChunkService(),
        enqueuer=enqueuer,
    )
    await reparse_knowledge(
        tenant_id=tenant_id,
        knowledge_id=url_doc.id,
        knowledge_repo=repo,
        kb_service=_make_kb_service(_kb_info(tenant_id=tenant_id, kb_id="kb-1")),
        chunk_service=_FakeChunkService(),
        enqueuer=enqueuer,
    )

    assert enqueuer.document[0].file_url == "https://example.com/a.pdf"
    assert enqueuer.document[0].file_type == "pdf"
    assert enqueuer.document[1].url == "https://example.com/page"
    assert enqueuer.document[1].file_type is None


async def test_reparse_validates_scope() -> None:
    repo, _rows = _make_repo()
    with pytest.raises(ValidationError) as exc_info:
        await reparse_knowledge(
            tenant_id=0,
            knowledge_id="doc-1",
            knowledge_repo=repo,
            kb_service=_make_kb_service(_kb_info(tenant_id=1, kb_id="kb-1")),
            chunk_service=_FakeChunkService(),
            enqueuer=_FakeEnqueuer(),
        )
    assert exc_info.value.code == "knowledge.tenant_required"

    with pytest.raises(ValidationError) as exc_info:
        await reparse_knowledge(
            tenant_id=1,
            knowledge_id="  ",
            knowledge_repo=repo,
            kb_service=_make_kb_service(_kb_info(tenant_id=1, kb_id="kb-1")),
            chunk_service=_FakeChunkService(),
            enqueuer=_FakeEnqueuer(),
        )
    assert exc_info.value.code == "knowledge.id_required"


# ═══════════════════════════════════════════════════════════════════════
# cancel_knowledge_parse - unit tests
# ═══════════════════════════════════════════════════════════════════════


async def test_cancel_processing_document_flips_to_cancelled() -> None:
    repo, rows = _make_repo()
    tenant_id = _int32_tenant_id()
    doc = _doc_row(tenant_id=tenant_id, parse_status="processing")
    rows[doc.id] = doc
    inspector = _FakeInspector()

    result = await cancel_knowledge_parse(
        tenant_id=tenant_id,
        knowledge_id=doc.id,
        knowledge_repo=repo,
        task_inspector=inspector,
    )

    assert result.parse_status == "cancelled"
    assert result.error_message == "用户已取消解析"
    stored = rows[doc.id]
    assert stored.parse_status == "cancelled"
    assert stored.error_message == "用户已取消解析"
    assert stored.pending_subtasks_count == 0
    assert inspector.calls == [doc.id]
    repo.update_columns.assert_awaited_once()


async def test_cancel_is_idempotent_for_already_cancelled_row() -> None:
    repo, rows = _make_repo()
    tenant_id = _int32_tenant_id()
    doc = _doc_row(tenant_id=tenant_id, parse_status="cancelled")
    rows[doc.id] = doc
    inspector = _FakeInspector()

    result = await cancel_knowledge_parse(
        tenant_id=tenant_id,
        knowledge_id=doc.id,
        knowledge_repo=repo,
        task_inspector=inspector,
    )

    assert result.parse_status == "cancelled"
    # No row write; the dequeue is still retried.
    repo.update_columns.assert_not_awaited()
    assert inspector.calls == [doc.id]


async def test_cancel_without_inspector_still_marks_cancelled() -> None:
    repo, rows = _make_repo()
    tenant_id = _int32_tenant_id()
    doc = _doc_row(tenant_id=tenant_id, parse_status="finalizing")
    rows[doc.id] = doc

    result = await cancel_knowledge_parse(
        tenant_id=tenant_id,
        knowledge_id=doc.id,
        knowledge_repo=repo,
    )

    assert result.parse_status == "cancelled"
    assert rows[doc.id].parse_status == "cancelled"


async def test_cancel_rejects_finished_and_deleting_states() -> None:
    repo, rows = _make_repo()
    tenant_id = _int32_tenant_id()
    for status, code in (
        ("completed", "knowledge.parse_not_cancellable"),
        ("failed", "knowledge.parse_not_cancellable"),
        ("deleting", "knowledge.parse_deleting"),
    ):
        doc = _doc_row(tenant_id=tenant_id, parse_status=status)
        rows[doc.id] = doc
        with pytest.raises(ValidationError) as exc_info:
            await cancel_knowledge_parse(
                tenant_id=tenant_id,
                knowledge_id=doc.id,
                knowledge_repo=repo,
            )
        assert exc_info.value.code == code
        del rows[doc.id]


async def test_cancel_missing_document_raises_not_found() -> None:
    repo, _rows = _make_repo()
    with pytest.raises(NotFoundError) as exc_info:
        await cancel_knowledge_parse(
            tenant_id=_int32_tenant_id(),
            knowledge_id="doc-missing",
            knowledge_repo=repo,
        )
    assert exc_info.value.code == "knowledge.not_found"


async def test_cancel_validates_scope() -> None:
    repo, _rows = _make_repo()
    with pytest.raises(ValidationError) as exc_info:
        await cancel_knowledge_parse(
            tenant_id=0,
            knowledge_id="doc-1",
            knowledge_repo=repo,
        )
    assert exc_info.value.code == "knowledge.tenant_required"

    with pytest.raises(ValidationError) as exc_info:
        await cancel_knowledge_parse(
            tenant_id=1,
            knowledge_id="",
            knowledge_repo=repo,
        )
    assert exc_info.value.code == "knowledge.id_required"


# ═══════════════════════════════════════════════════════════════════════
# update_document_image - unit tests
# ═══════════════════════════════════════════════════════════════════════


def _image_info_json(
    *,
    original_url: str = "u1",
    caption: str = "",
    ocr_text: str = "",
) -> str:
    return (
        '[{"original_url":"' + original_url + '","caption":"' + caption
        + '","ocr_text":"' + ocr_text + '"}]'
    )


async def test_image_update_refreshes_caption_ocr_and_document_hash() -> None:
    repo, rows = _make_repo()
    tenant_id = _int32_tenant_id()
    kb_id = "kb-1"
    doc = _doc_row(tenant_id=tenant_id, knowledge_base_id=kb_id, file_hash="old-hash")
    rows[doc.id] = doc
    parent = _chunk_row(tenant_id=tenant_id, knowledge_base_id=kb_id, knowledge_id=doc.id)
    caption = _chunk_row(
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
        knowledge_id=doc.id,
        content="old caption",
        chunk_type=CHUNK_TYPE_IMAGE_CAPTION,
        parent_chunk_id=parent.id,
        image_info=_image_info_json(caption="old caption"),
    )
    ocr = _chunk_row(
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
        knowledge_id=doc.id,
        content="old ocr",
        chunk_type=CHUNK_TYPE_IMAGE_OCR,
        parent_chunk_id=parent.id,
        image_info=_image_info_json(ocr_text="old ocr"),
    )
    chunk_service = _FakeChunkService({parent.id: parent, caption.id: caption, ocr.id: ocr})

    payload = _image_info_json(caption="new caption", ocr_text="new ocr")
    result = await update_document_image(
        tenant_id=tenant_id,
        knowledge_id=doc.id,
        chunk_id=parent.id,
        image_info=payload,
        knowledge_repo=repo,
        chunk_service=chunk_service,
    )

    assert result is None
    assert chunk_service._chunks[parent.id].image_info == payload
    assert chunk_service._chunks[caption.id].content == "new caption"
    assert chunk_service._chunks[caption.id].image_info == payload
    assert chunk_service._chunks[ocr.id].content == "new ocr"
    assert chunk_service._chunks[ocr.id].image_info == payload
    expected_hash = md5((doc.id + "old-hash" + payload).encode("utf-8")).hexdigest()
    assert rows[doc.id].file_hash == expected_hash


async def test_image_update_creates_missing_caption_and_ocr_children() -> None:
    repo, rows = _make_repo()
    tenant_id = _int32_tenant_id()
    kb_id = "kb-1"
    doc = _doc_row(tenant_id=tenant_id, knowledge_base_id=kb_id)
    rows[doc.id] = doc
    parent = _chunk_row(tenant_id=tenant_id, knowledge_base_id=kb_id, knowledge_id=doc.id)
    chunk_service = _FakeChunkService({parent.id: parent})

    payload = _image_info_json(caption="cap", ocr_text="ocr")
    await update_document_image(
        tenant_id=tenant_id,
        knowledge_id=doc.id,
        chunk_id=parent.id,
        image_info=payload,
        knowledge_repo=repo,
        chunk_service=chunk_service,
    )

    created_types = {c.chunk_type for c in chunk_service.created}
    assert created_types == {CHUNK_TYPE_IMAGE_CAPTION, CHUNK_TYPE_IMAGE_OCR}
    for created in chunk_service.created:
        assert created.parent_chunk_id == parent.id
        assert created.knowledge_id == doc.id
        assert created.image_info == payload
    # Only the parent row was updated.
    assert len(chunk_service.updated) == 1
    assert chunk_service.updated[0].id == parent.id


async def test_image_update_skips_non_matching_children() -> None:
    repo, rows = _make_repo()
    tenant_id = _int32_tenant_id()
    kb_id = "kb-1"
    doc = _doc_row(tenant_id=tenant_id, knowledge_base_id=kb_id)
    rows[doc.id] = doc
    parent = _chunk_row(tenant_id=tenant_id, knowledge_base_id=kb_id, knowledge_id=doc.id)
    caption = _chunk_row(
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
        knowledge_id=doc.id,
        content="old caption",
        chunk_type=CHUNK_TYPE_IMAGE_CAPTION,
        parent_chunk_id=parent.id,
        image_info=_image_info_json(original_url="other", caption="old caption"),
    )
    chunk_service = _FakeChunkService({parent.id: parent, caption.id: caption})

    payload = _image_info_json(caption="new caption")
    await update_document_image(
        tenant_id=tenant_id,
        knowledge_id=doc.id,
        chunk_id=parent.id,
        image_info=payload,
        knowledge_repo=repo,
        chunk_service=chunk_service,
    )

    # The URL mismatch leaves the child untouched.
    assert chunk_service._chunks[caption.id].content == "old caption"
    assert chunk_service._chunks[parent.id].image_info == payload
    assert len(chunk_service.updated) == 1


async def test_image_update_non_single_record_is_noop() -> None:
    repo, rows = _make_repo()
    tenant_id = _int32_tenant_id()
    doc = _doc_row(tenant_id=tenant_id)
    rows[doc.id] = doc
    parent = _chunk_row(tenant_id=tenant_id, knowledge_base_id=doc.knowledge_base_id, knowledge_id=doc.id)
    chunk_service = _FakeChunkService({parent.id: parent})

    for payload in (
        "[]",
        '[{"original_url":"u1","caption":"","ocr_text":""},'
        '{"original_url":"u2","caption":"","ocr_text":""}]',
    ):
        await update_document_image(
            tenant_id=tenant_id,
            knowledge_id=doc.id,
            chunk_id=parent.id,
            image_info=payload,
            knowledge_repo=repo,
            chunk_service=chunk_service,
        )

    assert chunk_service.created == []
    assert chunk_service.updated == []
    assert rows[doc.id].file_hash is None


async def test_image_update_rejects_malformed_payload() -> None:
    repo, _rows = _make_repo()
    with pytest.raises(ValidationError) as exc_info:
        await update_document_image(
            tenant_id=1,
            knowledge_id="doc-1",
            chunk_id="chunk-1",
            image_info="not-json",
            knowledge_repo=repo,
            chunk_service=_FakeChunkService(),
        )
    assert exc_info.value.code == "knowledge.invalid_image_info"

    with pytest.raises(ValidationError) as exc_info:
        await update_document_image(
            tenant_id=1,
            knowledge_id="doc-1",
            chunk_id="chunk-1",
            image_info='{"url":"x"}',
            knowledge_repo=repo,
            chunk_service=_FakeChunkService(),
        )
    assert exc_info.value.code == "knowledge.invalid_image_info"


async def test_image_update_missing_chunk_raises() -> None:
    repo, _rows = _make_repo()
    with pytest.raises(NotFoundError) as exc_info:
        await update_document_image(
            tenant_id=1,
            knowledge_id="doc-1",
            chunk_id="chunk-missing",
            image_info=_image_info_json(),
            knowledge_repo=repo,
            chunk_service=_FakeChunkService(),
        )
    assert exc_info.value.code == "chunk.not_found"


async def test_image_update_validates_scope() -> None:
    repo, _rows = _make_repo()
    with pytest.raises(ValidationError) as exc_info:
        await update_document_image(
            tenant_id=0,
            knowledge_id="doc-1",
            chunk_id="chunk-1",
            image_info=_image_info_json(),
            knowledge_repo=repo,
            chunk_service=_FakeChunkService(),
        )
    assert exc_info.value.code == "knowledge.tenant_required"

    with pytest.raises(ValidationError) as exc_info:
        await update_document_image(
            tenant_id=1,
            knowledge_id="",
            chunk_id="chunk-1",
            image_info=_image_info_json(),
            knowledge_repo=repo,
            chunk_service=_FakeChunkService(),
        )
    assert exc_info.value.code == "knowledge.id_required"

    with pytest.raises(ValidationError) as exc_info:
        await update_document_image(
            tenant_id=1,
            knowledge_id="doc-1",
            chunk_id="",
            image_info=_image_info_json(),
            knowledge_repo=repo,
            chunk_service=_FakeChunkService(),
        )
    assert exc_info.value.code == "knowledge.chunk_id_required"


async def test_image_info_model_accepts_partial_records() -> None:
    info = ImageInfo.model_validate({"original_url": "u1"})
    assert info.caption == ""
    assert info.ocr_text == ""
    assert info.start_pos == 0
    assert info.url is None


# ═══════════════════════════════════════════════════════════════════════
# Integration (real applied schema)
# ═══════════════════════════════════════════════════════════════════════


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


async def _seed_kb_and_doc(
    session: AsyncSession,
    *,
    tenant_id: int,
    **columns: object,
) -> tuple[KnowledgeBaseInfo, Knowledge]:
    kb_service = KBService(kb_repo=KnowledgeBaseRepository(session))
    kb = await kb_service.create_knowledge_base(tenant_id=tenant_id, name="docs")
    doc_service = KnowledgeService(knowledge_repo=KnowledgeRepository(session))
    defaults: dict[str, object] = {
        "tenant_id": tenant_id,
        "knowledge_base_id": kb.id,
        "type": "file",
        "title": "budget",
        "source": "budget-2026.pdf",
        "file_name": "budget-2026.pdf",
        "file_type": "pdf",
        "file_path": "/tmp/budget-2026.pdf",
        "parse_status": "completed",
    }
    defaults.update(columns)
    doc = await doc_service.create_document(**defaults)  # type: ignore[arg-type]
    return kb, doc


async def test_integration_reparse_resets_and_cleans_chunks(session: AsyncSession) -> None:
    tenant_id = _int32_tenant_id()
    kb, doc = await _seed_kb_and_doc(session, tenant_id=tenant_id)
    chunk = _chunk_row(
        tenant_id=tenant_id,
        knowledge_base_id=kb.id,
        knowledge_id=doc.id,
        content="stale chunk",
    )
    chunk_repo = ChunkRepository(session)
    await chunk_repo.create(chunk)
    await session.commit()
    enqueuer = _FakeEnqueuer()

    result = await reparse_knowledge(
        tenant_id=tenant_id,
        knowledge_id=doc.id,
        knowledge_repo=KnowledgeRepository(session),
        kb_service=KBService(kb_repo=KnowledgeBaseRepository(session)),
        chunk_service=ChunkService(chunk_repo=chunk_repo),
        enqueuer=enqueuer,
    )
    await session.commit()

    assert result.parse_status == "pending"
    assert result.enable_status == "disabled"
    stored = await KnowledgeRepository(session).get_by_id(tenant_id, doc.id)
    assert stored is not None
    assert stored.parse_status == "pending"
    assert stored.pending_subtasks_count == 0
    assert stored.embedding_model_id == ""
    assert await chunk_repo.get_by_id_or_none(tenant_id, chunk.id) is None
    assert len(enqueuer.document) == 1
    assert enqueuer.document[0].file_path == "/tmp/budget-2026.pdf"


async def test_integration_reparse_persists_overrides(session: AsyncSession) -> None:
    tenant_id = _int32_tenant_id()
    _kb, doc = await _seed_kb_and_doc(session, tenant_id=tenant_id)
    await session.commit()
    enqueuer = _FakeEnqueuer()

    await reparse_knowledge(
        tenant_id=tenant_id,
        knowledge_id=doc.id,
        knowledge_repo=KnowledgeRepository(session),
        kb_service=KBService(kb_repo=KnowledgeBaseRepository(session)),
        chunk_service=ChunkService(chunk_repo=ChunkRepository(session)),
        enqueuer=enqueuer,
        process_overrides={"enable_multimodel": True},
    )
    await session.commit()

    stored = await KnowledgeRepository(session).get_by_id(tenant_id, doc.id)
    assert stored is not None
    assert stored.metadata is not None
    assert stored.metadata["process_overrides"] == {"enable_multimodel": True}
    assert enqueuer.document[0].enable_multimodel is True


async def test_integration_cancel_marks_document_cancelled(session: AsyncSession) -> None:
    tenant_id = make_test_tenant_id()
    _kb, doc = await _seed_kb_and_doc(session, tenant_id=tenant_id, parse_status="processing")
    await session.commit()
    inspector = _FakeInspector()

    result = await cancel_knowledge_parse(
        tenant_id=tenant_id,
        knowledge_id=doc.id,
        knowledge_repo=KnowledgeRepository(session),
        task_inspector=inspector,
    )
    await session.commit()

    assert result.parse_status == "cancelled"
    stored = await KnowledgeRepository(session).get_by_id(tenant_id, doc.id)
    assert stored is not None
    assert stored.parse_status == "cancelled"
    assert stored.error_message == "用户已取消解析"
    assert stored.pending_subtasks_count == 0
    assert inspector.calls == [doc.id]


async def test_integration_cancel_rejects_completed_document(session: AsyncSession) -> None:
    tenant_id = make_test_tenant_id()
    _kb, doc = await _seed_kb_and_doc(session, tenant_id=tenant_id, parse_status="completed")
    await session.commit()

    with pytest.raises(ValidationError) as exc_info:
        await cancel_knowledge_parse(
            tenant_id=tenant_id,
            knowledge_id=doc.id,
            knowledge_repo=KnowledgeRepository(session),
        )
    assert exc_info.value.code == "knowledge.parse_not_cancellable"


async def test_integration_image_update_refreshes_caption_and_hash(
    session: AsyncSession,
) -> None:
    tenant_id = _int32_tenant_id()
    kb, doc = await _seed_kb_and_doc(session, tenant_id=tenant_id, file_hash="old-hash")
    chunk_repo = ChunkRepository(session)
    now = datetime.now(UTC)
    parent = _chunk_row(
        tenant_id=tenant_id,
        knowledge_base_id=kb.id,
        knowledge_id=doc.id,
        created_at=now,
        updated_at=now,
    )
    caption = _chunk_row(
        tenant_id=tenant_id,
        knowledge_base_id=kb.id,
        knowledge_id=doc.id,
        content="old caption",
        chunk_type=CHUNK_TYPE_IMAGE_CAPTION,
        parent_chunk_id=parent.id,
        image_info=_image_info_json(caption="old caption"),
        created_at=now,
        updated_at=now,
    )
    await chunk_repo.create_many([parent, caption])
    await session.commit()

    payload = _image_info_json(caption="new caption")
    await update_document_image(
        tenant_id=tenant_id,
        knowledge_id=doc.id,
        chunk_id=parent.id,
        image_info=payload,
        knowledge_repo=KnowledgeRepository(session),
        chunk_service=ChunkService(chunk_repo=chunk_repo),
    )
    await session.commit()

    stored_caption = await chunk_repo.get_by_id(tenant_id, caption.id)
    assert stored_caption.content == "new caption"
    assert stored_caption.image_info == payload
    stored_parent = await chunk_repo.get_by_id(tenant_id, parent.id)
    assert stored_parent.image_info == payload
    stored_doc = await KnowledgeRepository(session).get_by_id(tenant_id, doc.id)
    assert stored_doc is not None
    expected_hash = md5((doc.id + "old-hash" + payload).encode("utf-8")).hexdigest()
    assert stored_doc.file_hash == expected_hash


async def test_integration_image_update_creates_missing_children(session: AsyncSession) -> None:
    tenant_id = _int32_tenant_id()
    kb, doc = await _seed_kb_and_doc(session, tenant_id=tenant_id)
    chunk_repo = ChunkRepository(session)
    now = datetime.now(UTC)
    parent = _chunk_row(
        tenant_id=tenant_id,
        knowledge_base_id=kb.id,
        knowledge_id=doc.id,
        created_at=now,
        updated_at=now,
    )
    await chunk_repo.create(parent)
    await session.commit()

    payload = _image_info_json(caption="cap", ocr_text="ocr")
    await update_document_image(
        tenant_id=tenant_id,
        knowledge_id=doc.id,
        chunk_id=parent.id,
        image_info=payload,
        knowledge_repo=KnowledgeRepository(session),
        chunk_service=ChunkService(chunk_repo=chunk_repo),
    )
    await session.commit()

    children = await chunk_repo.list_by_parent_id(tenant_id, parent.id)
    assert len(children) == 2
    assert {child.chunk_type for child in children} == {
        CHUNK_TYPE_IMAGE_CAPTION,
        CHUNK_TYPE_IMAGE_OCR,
    }
