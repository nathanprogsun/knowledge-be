"""Unit + integration tests for ``clone_knowledge`` / ``move_knowledge``.

Unit tests drive the standalone modules with stateful repository mocks
(closure-captured storage, the same pattern used across the core service
tests): they cover validation, error classification, deep-copy semantics
(relationship / tag / image remapping), and the move-mode compatibility
gates.

Integration tests run against the real applied schema. ``chunks`` carries
an INTEGER (32-bit) ``tenant_id`` column, so those tests use an int32-safe
tenant id (a local counter) instead of ``make_test_tenant_id``'s BIGINT
range, which would overflow it.
"""

from __future__ import annotations

import itertools
import json
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

from src.common.exception import DataError, NotFoundError, ValidationError
from src.core.contracts.knowledge import Knowledge
from src.core.knowledge.chunks.types import (
    CHUNK_FLAG_RECOMMENDED,
    CHUNK_STATUS_STORED,
    CHUNK_TYPE_IMAGE_OCR,
    CHUNK_TYPE_PARENT_TEXT,
    CHUNK_TYPE_TEXT,
)
from src.core.knowledge.documents.clone import clone_knowledge
from src.core.knowledge.documents.move import (
    MOVE_MODE_REPARSE,
    MOVE_MODE_REUSE_VECTORS,
    move_knowledge,
)
from src.core.knowledge.documents.service.knowledge_service import KnowledgeService
from src.core.knowledge.documents.types import (
    CHANNEL_WEB,
    PARSE_STATUS_COMPLETED,
    PARSE_STATUS_FAILED,
    PARSE_STATUS_PENDING,
    PARSE_STATUS_PROCESSING,
)
from src.core.knowledge.knowledge_bases.service.kb_service import KBService
from src.core.knowledge.knowledge_bases.types import KnowledgeBaseInfo
from src.core.knowledge.tags.service.tag_service import TagService
from src.db.dao.chunk_repository import ChunkRepository
from src.db.dao.knowledge_base_repository import KnowledgeBaseRepository
from src.db.dao.knowledge_repository import KnowledgeRepository
from src.db.dao.knowledge_tag_repository import TagRepository
from src.db.models.chunk import Chunk
from src.db.models.knowledge import Document
from src.db.models.knowledge_tag import KnowledgeTag
from src.settings import get_settings, reset_settings_cache
from tests.integration.conftest import make_test_tenant_id

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_FAKER_SEED_MAX = 100_000_000

# ``chunks.tenant_id`` is INTEGER (32-bit); integration tests mint ids from
# this counter so they stay inside the range.
_INT32_TENANT_BASE = 2_000_000
_INT32_TENANT_SEQ = itertools.count(start=1)


def _int32_tenant_id() -> int:
    """Return a tenant id unique within the session, safe for INTEGER."""
    return _INT32_TENANT_BASE + next(_INT32_TENANT_SEQ)


@pytest.fixture(autouse=True)
def faker_seed() -> None:
    """Re-seed Faker per test for varied-but-reproducible generation."""
    Faker.seed(randint(1, _FAKER_SEED_MAX))


# ── Sample rows ───────────────────────────────────────────────────────


def _sample_doc(
    *,
    id: str | None = None,
    tenant_id: int | None = None,
    knowledge_base_id: str | None = None,
    parse_status: str = PARSE_STATUS_COMPLETED,
    file_path: str | None = None,
    embedding_model_id: str | None = None,
    **columns: object,
) -> Document:
    """Build a persisted-shape document row for seeding mocks."""
    return Document.model_validate(
        {
            "id": id or f"doc-{uuid.uuid4().hex[:12]}",
            "tenant_id": tenant_id if tenant_id is not None else make_test_tenant_id(),
            "knowledge_base_id": knowledge_base_id or f"kb-{uuid.uuid4().hex[:12]}",
            "type": "file",
            "title": "Q3 budget",
            "description": None,
            "source": "budget-2026.pdf",
            "channel": CHANNEL_WEB,
            "parse_status": parse_status,
            "pending_subtasks_count": 0,
            "summary_status": "none",
            "enable_status": "enabled",
            "embedding_model_id": embedding_model_id,
            "file_name": "budget-2026.pdf",
            "file_type": "pdf",
            "file_size": 1024,
            "file_hash": None,
            "file_path": file_path,
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
    tenant_id: int | None = None,
    knowledge_base_id: str | None = None,
    knowledge_id: str | None = None,
    chunk_index: int = 0,
    chunk_type: str = CHUNK_TYPE_TEXT,
    content: str = "chunk text",
    pre_chunk_id: str | None = None,
    next_chunk_id: str | None = None,
    parent_chunk_id: str | None = None,
    image_info: str | None = None,
    tag_id: str | None = None,
    **columns: object,
) -> Chunk:
    """Build a persisted-shape chunk row for seeding mocks."""
    return Chunk.model_validate(
        {
            "id": id or f"ch-{uuid.uuid4().hex[:12]}",
            "tenant_id": tenant_id if tenant_id is not None else make_test_tenant_id(),
            "knowledge_base_id": knowledge_base_id or f"kb-{uuid.uuid4().hex[:12]}",
            "knowledge_id": knowledge_id or f"kn-{uuid.uuid4().hex[:12]}",
            "content": content,
            "chunk_index": chunk_index,
            "is_enabled": True,
            "start_at": 0,
            "end_at": len(content),
            "pre_chunk_id": pre_chunk_id,
            "next_chunk_id": next_chunk_id,
            "chunk_type": chunk_type,
            "parent_chunk_id": parent_chunk_id,
            "image_info": image_info,
            "relation_chunks": None,
            "indirect_relation_chunks": None,
            "metadata": None,
            "tag_id": tag_id,
            "status": CHUNK_STATUS_STORED,
            "content_hash": None,
            "flags": CHUNK_FLAG_RECOMMENDED,
            "seq_id": 0,
            "source_content": "",
            "content_revision": 0,
            "index_status": "ready",
            "last_editor_id": "",
            "context_header": "",
            "created_at": _NOW,
            "updated_at": _NOW,
            "deleted_at": None,
            **columns,
        }
    )


def _sample_kb(
    *,
    id: str | None = None,
    tenant_id: int | None = None,
    kb_type: str = "document",
    embedding_model_id: str = "",
    vector_store_id: str | None = None,
) -> KnowledgeBaseInfo:
    """Build a knowledge-base service shape for mocking ``KBService``."""
    return KnowledgeBaseInfo(
        id=id or f"kb-{uuid.uuid4().hex[:12]}",
        name="test-kb",
        type=kb_type,
        tenant_id=tenant_id if tenant_id is not None else make_test_tenant_id(),
        embedding_model_id=embedding_model_id,
        vector_store_id=vector_store_id,
        created_at=_NOW,
        updated_at=_NOW,
    )


# ── Fake seams ────────────────────────────────────────────────────────


class _FakeObjectCopier:
    """Records copy calls and returns a synthetic destination path."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, str]] = []

    async def copy_object(
        self,
        *,
        src_path: str,
        dst_tenant_id: int,
        dst_knowledge_id: str,
    ) -> str:
        self.calls.append((src_path, dst_tenant_id, dst_knowledge_id))
        return f"new://{dst_tenant_id}/{dst_knowledge_id}/obj"


class _FakeReplicator:
    """Records retrieval-index copy calls."""

    def __init__(self) -> None:
        self.copies: list[dict[str, object]] = []

    async def copy_indices(
        self,
        *,
        source_kb_id: str,
        target_kb_id: str,
        knowledge_id_map: dict[str, str],
        chunk_id_map: dict[str, str],
    ) -> None:
        self.copies.append(
            {
                "source_kb_id": source_kb_id,
                "target_kb_id": target_kb_id,
                "knowledge_id_map": knowledge_id_map,
                "chunk_id_map": chunk_id_map,
            }
        )


class _FakeReparseTrigger:
    """Records reparse-trigger calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, str, str]] = []

    async def trigger_reparse(
        self,
        *,
        tenant_id: int,
        knowledge_id: str,
        knowledge_base_id: str,
    ) -> None:
        self.calls.append((tenant_id, knowledge_id, knowledge_base_id))


# ── Repository mocks (stateful via side_effect closures) ──────────────


def _make_knowledge_repo() -> tuple[AsyncMock, dict[str, Document]]:
    """Knowledge-repository mock with closure-captured storage."""
    repo = AsyncMock(spec=KnowledgeRepository)
    rows: dict[str, Document] = {}

    async def _get_by_id(tenant_id: int, id: str) -> Document | None:
        row = rows.get(id)
        if row is not None and row.tenant_id == tenant_id and row.deleted_at is None:
            return row
        return None

    async def _create(row: Document) -> Document:
        rows[row.id] = row
        return row

    async def _update(row: Document) -> Document:
        rows[row.id] = row
        return row

    async def _update_columns(id: str, values: dict[str, object]) -> Document | None:
        row = rows.get(id)
        if row is None:
            return None
        updated = row.model_copy(update=values)
        rows[id] = updated
        return updated

    repo.get_by_id.side_effect = _get_by_id
    repo.create.side_effect = _create
    repo.update.side_effect = _update
    repo.update_columns.side_effect = _update_columns
    return repo, rows


def _make_kb_service(*kbs: KnowledgeBaseInfo) -> AsyncMock:
    """``KBService`` mock resolving the supplied knowledge bases by id."""
    svc = AsyncMock(spec=KBService)
    by_id = {kb.id: kb for kb in kbs}

    async def _get_by_id(*, knowledge_base_id: str) -> KnowledgeBaseInfo:
        kb = by_id.get(knowledge_base_id)
        if kb is None:
            raise NotFoundError(
                code="knowledge_base.not_found",
                message=f"knowledge base {knowledge_base_id} not found",
            )
        return kb

    svc.get_knowledge_base_by_id.side_effect = _get_by_id
    return svc


def _make_chunk_repo() -> tuple[AsyncMock, dict[str, Chunk]]:
    """Chunk-repository mock with closure-captured storage."""
    repo = AsyncMock(spec=ChunkRepository)
    rows: dict[str, Chunk] = {}

    async def _find_all(cols: dict[str, object]) -> list[Chunk]:
        out: list[Chunk] = []
        for row in rows.values():
            if row.deleted_at is not None:
                continue
            if all(getattr(row, key) == value for key, value in cols.items()):
                out.append(row)
        return out

    async def _create_many(chunks: list[Chunk]) -> list[Chunk]:
        for chunk in chunks:
            rows[chunk.id] = chunk
        return chunks

    async def _move(tenant_id: int, knowledge_id: str, target_kb_id: str) -> int:
        moved = 0
        for chunk in list(rows.values()):
            if chunk.tenant_id == tenant_id and chunk.knowledge_id == knowledge_id:
                rows[chunk.id] = chunk.model_copy(update={"knowledge_base_id": target_kb_id})
                moved += 1
        return moved

    async def _delete_by_kg(tenant_id: int, knowledge_id: str, now: datetime) -> int:
        removed = 0
        for chunk in list(rows.values()):
            if (
                chunk.tenant_id == tenant_id
                and chunk.knowledge_id == knowledge_id
                and chunk.deleted_at is None
            ):
                rows[chunk.id] = chunk.model_copy(update={"deleted_at": now})
                removed += 1
        return removed

    repo.find_all_by_column_values.side_effect = _find_all
    repo.create_many.side_effect = _create_many
    repo.move_by_knowledge_id.side_effect = _move
    repo.delete_by_knowledge_id.side_effect = _delete_by_kg
    return repo, rows


def _make_tag_repo() -> tuple[AsyncMock, dict[str, KnowledgeTag]]:
    """Tag-repository mock with closure-captured storage."""
    repo = AsyncMock(spec=TagRepository)
    rows: dict[str, KnowledgeTag] = {}

    async def _get_by_id(tenant_id: int, id: str) -> KnowledgeTag | None:
        row = rows.get(id)
        if row is not None and row.tenant_id == tenant_id:
            return row
        return None

    async def _get_by_name(
        tenant_id: int,
        knowledge_base_id: str,
        name: str,
    ) -> KnowledgeTag | None:
        for row in rows.values():
            if (
                row.tenant_id == tenant_id
                and row.knowledge_base_id == knowledge_base_id
                and row.name == name
            ):
                return row
        return None

    async def _create(row: KnowledgeTag) -> KnowledgeTag:
        rows[row.id] = row
        return row

    repo.get_by_id.side_effect = _get_by_id
    repo.get_by_name.side_effect = _get_by_name
    repo.create.side_effect = _create
    return repo, rows


def _make_tag_service() -> AsyncMock:
    """``TagService`` mock whose relation clear reports success."""
    svc = AsyncMock(spec=TagService)
    svc.delete_knowledge_tag_relations.return_value = 1
    return svc


# ── clone_knowledge unit tests ────────────────────────────────────────


async def test_clone_knowledge_deep_copies_knowledge_and_chunks() -> None:
    tenant_id = 7
    dst_kb = _sample_kb(id="kb-dst", tenant_id=tenant_id, embedding_model_id="embed-a")
    source = _sample_doc(
        id="kn-1",
        tenant_id=tenant_id,
        knowledge_base_id="kb-src",
        embedding_model_id="embed-a",
    )
    k_repo, k_rows = _make_knowledge_repo()
    k_rows[source.id] = source
    c_repo, c_rows = _make_chunk_repo()
    c_rows["c-1"] = _sample_chunk(
        id="c-1",
        tenant_id=tenant_id,
        knowledge_base_id="kb-src",
        knowledge_id=source.id,
        chunk_index=0,
    )
    c_rows["c-2"] = _sample_chunk(
        id="c-2",
        tenant_id=tenant_id,
        knowledge_base_id="kb-src",
        knowledge_id=source.id,
        chunk_index=1,
        chunk_type=CHUNK_TYPE_PARENT_TEXT,
    )

    result = await clone_knowledge(
        tenant_id=tenant_id,
        knowledge_id=source.id,
        target_kb_id=dst_kb.id,
        knowledge_repo=k_repo,
        chunk_repo=c_repo,
        tag_repo=_make_tag_repo()[0],
        kb_service=_make_kb_service(dst_kb),
    )

    assert isinstance(result, Knowledge)
    assert result.id != source.id
    assert result.tenant_id == tenant_id
    assert result.knowledge_base_id == dst_kb.id
    assert result.parse_status == PARSE_STATUS_COMPLETED
    assert result.enable_status == "enabled"
    assert result.embedding_model_id == "embed-a"

    dst_doc = k_rows[result.id]
    assert dst_doc.parse_status == PARSE_STATUS_COMPLETED
    dst_chunks = [chunk for chunk in c_rows.values() if chunk.knowledge_id == result.id]
    assert len(dst_chunks) == 2
    assert {chunk.chunk_type for chunk in dst_chunks} == {
        CHUNK_TYPE_TEXT,
        CHUNK_TYPE_PARENT_TEXT,
    }
    # source chunks are untouched
    assert {chunk.id for chunk in c_rows.values() if chunk.knowledge_id == source.id} == {
        "c-1",
        "c-2",
    }


async def test_clone_knowledge_remaps_chunk_relationships() -> None:
    tenant_id = 7
    dst_kb = _sample_kb(id="kb-dst", tenant_id=tenant_id)
    source = _sample_doc(id="kn-1", tenant_id=tenant_id, knowledge_base_id="kb-src")
    k_repo, k_rows = _make_knowledge_repo()
    k_rows[source.id] = source
    c_repo, c_rows = _make_chunk_repo()
    c_rows["c-1"] = _sample_chunk(
        id="c-1",
        tenant_id=tenant_id,
        knowledge_base_id="kb-src",
        knowledge_id=source.id,
        chunk_index=0,
        next_chunk_id="c-2",
    )
    c_rows["c-2"] = _sample_chunk(
        id="c-2",
        tenant_id=tenant_id,
        knowledge_base_id="kb-src",
        knowledge_id=source.id,
        chunk_index=1,
        pre_chunk_id="c-1",
        parent_chunk_id="c-3",
    )
    # ``parent_chunk_id`` points at c-9, a chunk outside the cloned set, so
    # the reference resets to "".
    c_rows["c-3"] = _sample_chunk(
        id="c-3",
        tenant_id=tenant_id,
        knowledge_base_id="kb-src",
        knowledge_id=source.id,
        chunk_index=2,
        chunk_type=CHUNK_TYPE_IMAGE_OCR,
        parent_chunk_id="c-9",
    )

    result = await clone_knowledge(
        tenant_id=tenant_id,
        knowledge_id=source.id,
        target_kb_id=dst_kb.id,
        knowledge_repo=k_repo,
        chunk_repo=c_repo,
        tag_repo=_make_tag_repo()[0],
        kb_service=_make_kb_service(dst_kb),
    )
    assert result is not None

    dst_by_index = {
        chunk.chunk_index: chunk for chunk in c_rows.values() if chunk.knowledge_id == result.id
    }
    assert dst_by_index[0].next_chunk_id == dst_by_index[1].id
    assert dst_by_index[1].pre_chunk_id == dst_by_index[0].id
    assert dst_by_index[1].parent_chunk_id == dst_by_index[2].id
    assert dst_by_index[2].parent_chunk_id == ""


async def test_clone_knowledge_reuses_existing_target_tag() -> None:
    tenant_id = 7
    dst_kb = _sample_kb(id="kb-dst", tenant_id=tenant_id)
    source = _sample_doc(id="kn-1", tenant_id=tenant_id, knowledge_base_id="kb-src")
    k_repo, k_rows = _make_knowledge_repo()
    k_rows[source.id] = source
    c_repo, c_rows = _make_chunk_repo()
    c_rows["c-1"] = _sample_chunk(
        id="c-1",
        tenant_id=tenant_id,
        knowledge_base_id="kb-src",
        knowledge_id=source.id,
        tag_id="tag-src",
    )
    t_repo, t_rows = _make_tag_repo()
    t_rows["tag-src"] = KnowledgeTag(
        id="tag-src",
        tenant_id=tenant_id,
        knowledge_base_id="kb-src",
        name="finance",
        created_at=_NOW,
        updated_at=_NOW,
    )
    t_rows["tag-dst"] = KnowledgeTag(
        id="tag-dst",
        tenant_id=tenant_id,
        knowledge_base_id="kb-dst",
        name="finance",
        created_at=_NOW,
        updated_at=_NOW,
    )

    result = await clone_knowledge(
        tenant_id=tenant_id,
        knowledge_id=source.id,
        target_kb_id=dst_kb.id,
        knowledge_repo=k_repo,
        chunk_repo=c_repo,
        tag_repo=t_repo,
        kb_service=_make_kb_service(dst_kb),
    )
    assert result is not None

    dst_chunk = next(chunk for chunk in c_rows.values() if chunk.knowledge_id == result.id)
    assert dst_chunk.tag_id == "tag-dst"
    t_repo.create.assert_not_awaited()


async def test_clone_knowledge_creates_missing_target_tag_with_untagged_pin() -> None:
    tenant_id = 7
    dst_kb = _sample_kb(id="kb-dst", tenant_id=tenant_id)
    source = _sample_doc(id="kn-1", tenant_id=tenant_id, knowledge_base_id="kb-src")
    k_repo, k_rows = _make_knowledge_repo()
    k_rows[source.id] = source
    c_repo, c_rows = _make_chunk_repo()
    c_rows["c-1"] = _sample_chunk(
        id="c-1",
        tenant_id=tenant_id,
        knowledge_base_id="kb-src",
        knowledge_id=source.id,
        tag_id="tag-src",
    )
    t_repo, t_rows = _make_tag_repo()
    t_rows["tag-src"] = KnowledgeTag(
        id="tag-src",
        tenant_id=tenant_id,
        knowledge_base_id="kb-src",
        name="未分类",
        color="#efefef",
        sort_order=5,
        created_at=_NOW,
        updated_at=_NOW,
    )

    result = await clone_knowledge(
        tenant_id=tenant_id,
        knowledge_id=source.id,
        target_kb_id=dst_kb.id,
        knowledge_repo=k_repo,
        chunk_repo=c_repo,
        tag_repo=t_repo,
        kb_service=_make_kb_service(dst_kb),
    )
    assert result is not None

    created = [tag for tag in t_rows.values() if tag.knowledge_base_id == "kb-dst"]
    assert len(created) == 1
    assert created[0].name == "未分类"
    assert created[0].color == "#efefef"
    assert created[0].sort_order == -1
    dst_chunk = next(chunk for chunk in c_rows.values() if chunk.knowledge_id == result.id)
    assert dst_chunk.tag_id == created[0].id


async def test_clone_knowledge_copies_file_via_object_copier() -> None:
    tenant_id = 7
    dst_kb = _sample_kb(id="kb-dst", tenant_id=tenant_id)
    source = _sample_doc(
        id="kn-1",
        tenant_id=tenant_id,
        knowledge_base_id="kb-src",
        file_path="src://t1/kn-1/doc.pdf",
    )
    k_repo, k_rows = _make_knowledge_repo()
    k_rows[source.id] = source
    copier = _FakeObjectCopier()

    result = await clone_knowledge(
        tenant_id=tenant_id,
        knowledge_id=source.id,
        target_kb_id=dst_kb.id,
        knowledge_repo=k_repo,
        chunk_repo=_make_chunk_repo()[0],
        tag_repo=_make_tag_repo()[0],
        kb_service=_make_kb_service(dst_kb),
        object_copier=copier,
    )
    assert result is not None
    assert result.file_path == f"new://{tenant_id}/{result.id}/obj"
    assert copier.calls == [("src://t1/kn-1/doc.pdf", tenant_id, result.id)]


async def test_clone_knowledge_deep_copies_chunk_images_and_rewrites_content() -> None:
    tenant_id = 7
    dst_kb = _sample_kb(id="kb-dst", tenant_id=tenant_id)
    source = _sample_doc(id="kn-1", tenant_id=tenant_id, knowledge_base_id="kb-src")
    k_repo, k_rows = _make_knowledge_repo()
    k_rows[source.id] = source
    c_repo, c_rows = _make_chunk_repo()
    c_rows["c-1"] = _sample_chunk(
        id="c-1",
        tenant_id=tenant_id,
        knowledge_base_id="kb-src",
        knowledge_id=source.id,
        content="![a](src://img/a.png) ![b](src://img/b.jpg)",
        image_info=(
            '[{"url": "src://img/a.png", "original_url": "src://img/a.png"}, '
            '{"url": "src://img/b.jpg", "original_url": "http://external/x.jpg"}]'
        ),
    )
    copier = _FakeObjectCopier()

    result = await clone_knowledge(
        tenant_id=tenant_id,
        knowledge_id=source.id,
        target_kb_id=dst_kb.id,
        knowledge_repo=k_repo,
        chunk_repo=c_repo,
        tag_repo=_make_tag_repo()[0],
        kb_service=_make_kb_service(dst_kb),
        object_copier=copier,
    )
    assert result is not None

    dst_chunk = next(chunk for chunk in c_rows.values() if chunk.knowledge_id == result.id)
    parsed = json.loads(dst_chunk.image_info or "")
    new_url = f"new://{tenant_id}/{result.id}/obj"
    assert parsed[0]["url"] == new_url
    assert parsed[0]["original_url"] == new_url  # matched original rewritten
    assert parsed[1]["url"] == new_url
    assert parsed[1]["original_url"] == "http://external/x.jpg"  # foreign original preserved
    assert new_url in dst_chunk.content
    assert "src://img/a.png" not in dst_chunk.content


async def test_clone_knowledge_copies_indices_via_replicator() -> None:
    tenant_id = 7
    dst_kb = _sample_kb(id="kb-dst", tenant_id=tenant_id)
    source = _sample_doc(id="kn-1", tenant_id=tenant_id, knowledge_base_id="kb-src")
    k_repo, k_rows = _make_knowledge_repo()
    k_rows[source.id] = source
    c_repo, c_rows = _make_chunk_repo()
    c_rows["c-1"] = _sample_chunk(
        id="c-1", tenant_id=tenant_id, knowledge_base_id="kb-src", knowledge_id=source.id
    )
    replicator = _FakeReplicator()

    result = await clone_knowledge(
        tenant_id=tenant_id,
        knowledge_id=source.id,
        target_kb_id=dst_kb.id,
        knowledge_repo=k_repo,
        chunk_repo=c_repo,
        tag_repo=_make_tag_repo()[0],
        kb_service=_make_kb_service(dst_kb),
        index_replicator=replicator,
    )
    assert result is not None
    dst_chunks = [chunk for chunk in c_rows.values() if chunk.knowledge_id == result.id]
    assert len(dst_chunks) == 1
    assert len(replicator.copies) == 1
    assert replicator.copies[0]["source_kb_id"] == "kb-src"
    assert replicator.copies[0]["target_kb_id"] == "kb-dst"
    assert replicator.copies[0]["knowledge_id_map"] == {"kn-1": result.id}
    assert replicator.copies[0]["chunk_id_map"] == {"c-1": dst_chunks[0].id}


async def test_clone_knowledge_skips_non_completed_source() -> None:
    tenant_id = 7
    source = _sample_doc(
        id="kn-1",
        tenant_id=tenant_id,
        knowledge_base_id="kb-src",
        parse_status=PARSE_STATUS_PROCESSING,
    )
    k_repo, k_rows = _make_knowledge_repo()
    k_rows[source.id] = source
    c_repo, _c_rows = _make_chunk_repo()

    result = await clone_knowledge(
        tenant_id=tenant_id,
        knowledge_id=source.id,
        target_kb_id="kb-dst",
        knowledge_repo=k_repo,
        chunk_repo=c_repo,
        tag_repo=_make_tag_repo()[0],
        kb_service=_make_kb_service(),
    )
    assert result is None
    k_repo.create.assert_not_awaited()
    c_repo.create_many.assert_not_awaited()


async def test_clone_knowledge_requires_object_copier_for_stored_file() -> None:
    tenant_id = 7
    source = _sample_doc(
        id="kn-1",
        tenant_id=tenant_id,
        knowledge_base_id="kb-src",
        file_path="src://t1/kn-1/doc.pdf",
    )
    k_repo, k_rows = _make_knowledge_repo()
    k_rows[source.id] = source

    with pytest.raises(DataError) as exc_info:
        await clone_knowledge(
            tenant_id=tenant_id,
            knowledge_id=source.id,
            target_kb_id="kb-dst",
            knowledge_repo=k_repo,
            chunk_repo=_make_chunk_repo()[0],
            tag_repo=_make_tag_repo()[0],
            kb_service=_make_kb_service(_sample_kb(id="kb-dst", tenant_id=tenant_id)),
        )
    assert exc_info.value.code == "knowledge.clone_storage_hook_required"
    k_repo.create.assert_not_awaited()


async def test_clone_knowledge_marks_failed_on_chunk_error() -> None:
    tenant_id = 7
    dst_kb = _sample_kb(id="kb-dst", tenant_id=tenant_id)
    source = _sample_doc(id="kn-1", tenant_id=tenant_id, knowledge_base_id="kb-src")
    k_repo, k_rows = _make_knowledge_repo()
    k_rows[source.id] = source
    c_repo, c_rows = _make_chunk_repo()
    c_rows["c-1"] = _sample_chunk(
        id="c-1", tenant_id=tenant_id, knowledge_base_id="kb-src", knowledge_id=source.id
    )
    c_repo.create_many.side_effect = RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await clone_knowledge(
            tenant_id=tenant_id,
            knowledge_id=source.id,
            target_kb_id=dst_kb.id,
            knowledge_repo=k_repo,
            chunk_repo=c_repo,
            tag_repo=_make_tag_repo()[0],
            kb_service=_make_kb_service(dst_kb),
        )
    dst_docs = [doc for doc in k_rows.values() if doc.id != source.id]
    assert len(dst_docs) == 1
    assert dst_docs[0].parse_status == PARSE_STATUS_FAILED
    assert dst_docs[0].error_message == "boom"


async def test_clone_knowledge_raises_not_found_for_missing_source() -> None:
    with pytest.raises(NotFoundError) as exc_info:
        await clone_knowledge(
            tenant_id=7,
            knowledge_id="kn-missing",
            target_kb_id="kb-dst",
            knowledge_repo=_make_knowledge_repo()[0],
            chunk_repo=_make_chunk_repo()[0],
            tag_repo=_make_tag_repo()[0],
            kb_service=_make_kb_service(),
        )
    assert exc_info.value.code == "knowledge.not_found"


async def test_clone_knowledge_raises_not_found_for_missing_target_kb() -> None:
    tenant_id = 7
    source = _sample_doc(id="kn-1", tenant_id=tenant_id, knowledge_base_id="kb-src")
    k_repo, k_rows = _make_knowledge_repo()
    k_rows[source.id] = source

    with pytest.raises(NotFoundError) as exc_info:
        await clone_knowledge(
            tenant_id=tenant_id,
            knowledge_id=source.id,
            target_kb_id="kb-missing",
            knowledge_repo=k_repo,
            chunk_repo=_make_chunk_repo()[0],
            tag_repo=_make_tag_repo()[0],
            kb_service=_make_kb_service(),
        )
    assert exc_info.value.code == "knowledge_base.not_found"


async def test_clone_knowledge_validates_scope() -> None:
    with pytest.raises(ValidationError) as exc_info:
        await clone_knowledge(
            tenant_id=0,
            knowledge_id="kn-1",
            target_kb_id="kb-dst",
            knowledge_repo=_make_knowledge_repo()[0],
            chunk_repo=_make_chunk_repo()[0],
            tag_repo=_make_tag_repo()[0],
            kb_service=_make_kb_service(),
        )
    assert exc_info.value.code == "knowledge.tenant_required"

    with pytest.raises(ValidationError) as exc_info:
        await clone_knowledge(
            tenant_id=7,
            knowledge_id="",
            target_kb_id="kb-dst",
            knowledge_repo=_make_knowledge_repo()[0],
            chunk_repo=_make_chunk_repo()[0],
            tag_repo=_make_tag_repo()[0],
            kb_service=_make_kb_service(),
        )
    assert exc_info.value.code == "knowledge.id_required"

    with pytest.raises(ValidationError) as exc_info:
        await clone_knowledge(
            tenant_id=7,
            knowledge_id="kn-1",
            target_kb_id="  ",
            knowledge_repo=_make_knowledge_repo()[0],
            chunk_repo=_make_chunk_repo()[0],
            tag_repo=_make_tag_repo()[0],
            kb_service=_make_kb_service(),
        )
    assert exc_info.value.code == "knowledge.kb_required"


# ── move_knowledge unit tests ─────────────────────────────────────────


async def test_move_knowledge_reuse_vectors_rehomes_rows() -> None:
    tenant_id = 7
    src_kb = _sample_kb(id="kb-src", tenant_id=tenant_id)
    dst_kb = _sample_kb(id="kb-dst", tenant_id=tenant_id)
    source = _sample_doc(id="kn-1", tenant_id=tenant_id, knowledge_base_id="kb-src")
    k_repo, k_rows = _make_knowledge_repo()
    k_rows[source.id] = source
    c_repo, c_rows = _make_chunk_repo()
    c_rows["c-1"] = _sample_chunk(
        id="c-1", tenant_id=tenant_id, knowledge_base_id="kb-src", knowledge_id=source.id
    )
    tag_service = _make_tag_service()

    moved = await move_knowledge(
        tenant_id=tenant_id,
        knowledge_id=source.id,
        source_kb_id=src_kb.id,
        target_kb_id=dst_kb.id,
        mode=MOVE_MODE_REUSE_VECTORS,
        knowledge_repo=k_repo,
        chunk_repo=c_repo,
        tag_service=tag_service,
        kb_service=_make_kb_service(src_kb, dst_kb),
    )
    assert moved.knowledge_base_id == dst_kb.id
    assert moved.parse_status == PARSE_STATUS_COMPLETED
    assert c_rows["c-1"].knowledge_base_id == dst_kb.id
    tag_service.delete_knowledge_tag_relations.assert_awaited_once_with(source.id)


async def test_move_knowledge_reuse_vectors_copies_indices() -> None:
    tenant_id = 7
    src_kb = _sample_kb(id="kb-src", tenant_id=tenant_id, vector_store_id="store-1")
    dst_kb = _sample_kb(id="kb-dst", tenant_id=tenant_id, vector_store_id="store-1")
    source = _sample_doc(
        id="kn-1",
        tenant_id=tenant_id,
        knowledge_base_id="kb-src",
        embedding_model_id="embed-a",
    )
    k_repo, k_rows = _make_knowledge_repo()
    k_rows[source.id] = source
    c_repo, c_rows = _make_chunk_repo()
    c_rows["c-1"] = _sample_chunk(
        id="c-1", tenant_id=tenant_id, knowledge_base_id="kb-src", knowledge_id=source.id
    )
    replicator = _FakeReplicator()

    await move_knowledge(
        tenant_id=tenant_id,
        knowledge_id=source.id,
        source_kb_id=src_kb.id,
        target_kb_id=dst_kb.id,
        mode=MOVE_MODE_REUSE_VECTORS,
        knowledge_repo=k_repo,
        chunk_repo=c_repo,
        tag_service=_make_tag_service(),
        kb_service=_make_kb_service(src_kb, dst_kb),
        index_replicator=replicator,
    )
    assert len(replicator.copies) == 1
    assert replicator.copies[0]["source_kb_id"] == "kb-src"
    assert replicator.copies[0]["target_kb_id"] == "kb-dst"
    assert replicator.copies[0]["knowledge_id_map"] == {"kn-1": "kn-1"}
    assert replicator.copies[0]["chunk_id_map"] == {"c-1": "c-1"}


async def test_move_knowledge_reparse_resets_and_triggers() -> None:
    tenant_id = 7
    src_kb = _sample_kb(id="kb-src", tenant_id=tenant_id)
    dst_kb = _sample_kb(id="kb-dst", tenant_id=tenant_id)
    source = _sample_doc(
        id="kn-1",
        tenant_id=tenant_id,
        knowledge_base_id="kb-src",
        description="keep?",
        embedding_model_id="embed-a",
    )
    k_repo, k_rows = _make_knowledge_repo()
    k_rows[source.id] = source
    c_repo, c_rows = _make_chunk_repo()
    c_rows["c-1"] = _sample_chunk(
        id="c-1", tenant_id=tenant_id, knowledge_base_id="kb-src", knowledge_id=source.id
    )
    trigger = _FakeReparseTrigger()

    moved = await move_knowledge(
        tenant_id=tenant_id,
        knowledge_id=source.id,
        source_kb_id=src_kb.id,
        target_kb_id=dst_kb.id,
        mode=MOVE_MODE_REPARSE,
        knowledge_repo=k_repo,
        chunk_repo=c_repo,
        tag_service=_make_tag_service(),
        kb_service=_make_kb_service(src_kb, dst_kb),
        reparse_trigger=trigger,
    )
    assert moved.knowledge_base_id == dst_kb.id
    assert moved.parse_status == PARSE_STATUS_PENDING
    assert moved.enable_status == "disabled"
    assert moved.description == ""
    assert moved.embedding_model_id == ""
    assert c_rows["c-1"].deleted_at is not None
    assert trigger.calls == [(tenant_id, "kn-1", dst_kb.id)]


async def test_move_knowledge_rejects_non_completed_source() -> None:
    tenant_id = 7
    src_kb = _sample_kb(id="kb-src", tenant_id=tenant_id)
    dst_kb = _sample_kb(id="kb-dst", tenant_id=tenant_id)
    source = _sample_doc(
        id="kn-1",
        tenant_id=tenant_id,
        knowledge_base_id="kb-src",
        parse_status=PARSE_STATUS_PROCESSING,
    )
    k_repo, k_rows = _make_knowledge_repo()
    k_rows[source.id] = source

    with pytest.raises(ValidationError) as exc_info:
        await move_knowledge(
            tenant_id=tenant_id,
            knowledge_id=source.id,
            source_kb_id=src_kb.id,
            target_kb_id=dst_kb.id,
            mode=MOVE_MODE_REUSE_VECTORS,
            knowledge_repo=k_repo,
            chunk_repo=_make_chunk_repo()[0],
            tag_service=_make_tag_service(),
            kb_service=_make_kb_service(src_kb, dst_kb),
        )
    assert exc_info.value.code == "knowledge.move_not_completed"


async def test_move_knowledge_rejects_type_mismatch() -> None:
    tenant_id = 7
    src_kb = _sample_kb(id="kb-src", tenant_id=tenant_id, kb_type="document")
    dst_kb = _sample_kb(id="kb-dst", tenant_id=tenant_id, kb_type="faq")
    source = _sample_doc(id="kn-1", tenant_id=tenant_id, knowledge_base_id="kb-src")
    k_repo, k_rows = _make_knowledge_repo()
    k_rows[source.id] = source

    with pytest.raises(ValidationError) as exc_info:
        await move_knowledge(
            tenant_id=tenant_id,
            knowledge_id=source.id,
            source_kb_id=src_kb.id,
            target_kb_id=dst_kb.id,
            mode=MOVE_MODE_REUSE_VECTORS,
            knowledge_repo=k_repo,
            chunk_repo=_make_chunk_repo()[0],
            tag_service=_make_tag_service(),
            kb_service=_make_kb_service(src_kb, dst_kb),
        )
    assert exc_info.value.code == "knowledge.move_type_mismatch"


async def test_move_knowledge_rejects_embedding_model_mismatch() -> None:
    tenant_id = 7
    src_kb = _sample_kb(id="kb-src", tenant_id=tenant_id, embedding_model_id="embed-a")
    dst_kb = _sample_kb(id="kb-dst", tenant_id=tenant_id, embedding_model_id="embed-b")
    source = _sample_doc(id="kn-1", tenant_id=tenant_id, knowledge_base_id="kb-src")
    k_repo, k_rows = _make_knowledge_repo()
    k_rows[source.id] = source

    with pytest.raises(ValidationError) as exc_info:
        await move_knowledge(
            tenant_id=tenant_id,
            knowledge_id=source.id,
            source_kb_id=src_kb.id,
            target_kb_id=dst_kb.id,
            mode=MOVE_MODE_REUSE_VECTORS,
            knowledge_repo=k_repo,
            chunk_repo=_make_chunk_repo()[0],
            tag_service=_make_tag_service(),
            kb_service=_make_kb_service(src_kb, dst_kb),
        )
    assert exc_info.value.code == "knowledge.move_embedding_mismatch"


async def test_move_knowledge_rejects_cross_store_reuse_vectors_before_status_change() -> None:
    tenant_id = 7
    src_kb = _sample_kb(id="kb-src", tenant_id=tenant_id, vector_store_id="store-1")
    dst_kb = _sample_kb(id="kb-dst", tenant_id=tenant_id, vector_store_id="store-2")
    source = _sample_doc(id="kn-1", tenant_id=tenant_id, knowledge_base_id="kb-src")
    k_repo, k_rows = _make_knowledge_repo()
    k_rows[source.id] = source

    with pytest.raises(ValidationError) as exc_info:
        await move_knowledge(
            tenant_id=tenant_id,
            knowledge_id=source.id,
            source_kb_id=src_kb.id,
            target_kb_id=dst_kb.id,
            mode=MOVE_MODE_REUSE_VECTORS,
            knowledge_repo=k_repo,
            chunk_repo=_make_chunk_repo()[0],
            tag_service=_make_tag_service(),
            kb_service=_make_kb_service(src_kb, dst_kb),
        )
    assert exc_info.value.code == "knowledge.move_cross_store_not_supported"
    k_repo.update.assert_not_awaited()


async def test_move_knowledge_rejects_unknown_mode() -> None:
    with pytest.raises(ValidationError) as exc_info:
        await move_knowledge(
            tenant_id=7,
            knowledge_id="kn-1",
            source_kb_id="kb-src",
            target_kb_id="kb-dst",
            mode="teleport",
            knowledge_repo=_make_knowledge_repo()[0],
            chunk_repo=_make_chunk_repo()[0],
            tag_service=_make_tag_service(),
            kb_service=_make_kb_service(),
        )
    assert exc_info.value.code == "knowledge.move_mode_invalid"


async def test_move_knowledge_restores_status_on_failure() -> None:
    tenant_id = 7
    src_kb = _sample_kb(id="kb-src", tenant_id=tenant_id)
    dst_kb = _sample_kb(id="kb-dst", tenant_id=tenant_id)
    source = _sample_doc(id="kn-1", tenant_id=tenant_id, knowledge_base_id="kb-src")
    k_repo, k_rows = _make_knowledge_repo()
    k_rows[source.id] = source
    c_repo, _c_rows = _make_chunk_repo()
    c_repo.move_by_knowledge_id.side_effect = RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await move_knowledge(
            tenant_id=tenant_id,
            knowledge_id=source.id,
            source_kb_id=src_kb.id,
            target_kb_id=dst_kb.id,
            mode=MOVE_MODE_REUSE_VECTORS,
            knowledge_repo=k_repo,
            chunk_repo=c_repo,
            tag_service=_make_tag_service(),
            kb_service=_make_kb_service(src_kb, dst_kb),
        )
    assert k_rows[source.id].parse_status == PARSE_STATUS_COMPLETED


async def test_move_knowledge_raises_not_found() -> None:
    with pytest.raises(NotFoundError) as exc_info:
        await move_knowledge(
            tenant_id=7,
            knowledge_id="kn-missing",
            source_kb_id="kb-src",
            target_kb_id="kb-dst",
            mode=MOVE_MODE_REUSE_VECTORS,
            knowledge_repo=_make_knowledge_repo()[0],
            chunk_repo=_make_chunk_repo()[0],
            tag_service=_make_tag_service(),
            kb_service=_make_kb_service(),
        )
    assert exc_info.value.code == "knowledge.not_found"


async def test_move_knowledge_validates_scope() -> None:
    with pytest.raises(ValidationError) as exc_info:
        await move_knowledge(
            tenant_id=0,
            knowledge_id="kn-1",
            source_kb_id="kb-src",
            target_kb_id="kb-dst",
            mode=MOVE_MODE_REUSE_VECTORS,
            knowledge_repo=_make_knowledge_repo()[0],
            chunk_repo=_make_chunk_repo()[0],
            tag_service=_make_tag_service(),
            kb_service=_make_kb_service(),
        )
    assert exc_info.value.code == "knowledge.tenant_required"

    with pytest.raises(ValidationError) as exc_info:
        await move_knowledge(
            tenant_id=7,
            knowledge_id="",
            source_kb_id="kb-src",
            target_kb_id="kb-dst",
            mode=MOVE_MODE_REUSE_VECTORS,
            knowledge_repo=_make_knowledge_repo()[0],
            chunk_repo=_make_chunk_repo()[0],
            tag_service=_make_tag_service(),
            kb_service=_make_kb_service(),
        )
    assert exc_info.value.code == "knowledge.id_required"

    with pytest.raises(ValidationError) as exc_info:
        await move_knowledge(
            tenant_id=7,
            knowledge_id="kn-1",
            source_kb_id="",
            target_kb_id="kb-dst",
            mode=MOVE_MODE_REUSE_VECTORS,
            knowledge_repo=_make_knowledge_repo()[0],
            chunk_repo=_make_chunk_repo()[0],
            tag_service=_make_tag_service(),
            kb_service=_make_kb_service(),
        )
    assert exc_info.value.code == "knowledge.kb_required"


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


def _integration_chunk(
    *,
    id: str | None = None,
    tenant_id: int,
    knowledge_base_id: str,
    knowledge_id: str,
    chunk_index: int,
    chunk_type: str = CHUNK_TYPE_TEXT,
    content: str = "chunk text",
    pre_chunk_id: str | None = None,
    next_chunk_id: str | None = None,
    image_info: str | None = None,
) -> Chunk:
    """Build a chunk row ready for real-DB inserts."""
    return Chunk(
        id=id or str(uuid.uuid4()),
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        knowledge_id=knowledge_id,
        content=content,
        chunk_index=chunk_index,
        is_enabled=True,
        start_at=0,
        end_at=len(content),
        pre_chunk_id=pre_chunk_id,
        next_chunk_id=next_chunk_id,
        chunk_type=chunk_type,
        parent_chunk_id=None,
        image_info=image_info,
        metadata=None,
        tag_id=None,
        status=CHUNK_STATUS_STORED,
        content_hash=None,
        flags=CHUNK_FLAG_RECOMMENDED,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


async def test_integration_clone_knowledge_round_trip(session: AsyncSession) -> None:
    tenant_id = _int32_tenant_id()
    kb_service = KBService(kb_repo=KnowledgeBaseRepository(session))
    src_kb = await kb_service.create_knowledge_base(
        tenant_id=tenant_id, name="clone-src", kb_type="document"
    )
    dst_kb = await kb_service.create_knowledge_base(
        tenant_id=tenant_id, name="clone-dst", kb_type="document"
    )
    knowledge_service = KnowledgeService(knowledge_repo=KnowledgeRepository(session))
    source = await knowledge_service.create_document(
        tenant_id=tenant_id,
        knowledge_base_id=src_kb.id,
        type="manual",
        title="Source doc",
        source="manual",
        parse_status=PARSE_STATUS_COMPLETED,
    )
    chunk_repo = ChunkRepository(session)
    c1_id, c2_id = str(uuid.uuid4()), str(uuid.uuid4())
    await chunk_repo.create_many(
        [
            _integration_chunk(
                id=c1_id,
                tenant_id=tenant_id,
                knowledge_base_id=src_kb.id,
                knowledge_id=source.id,
                chunk_index=0,
                next_chunk_id=c2_id,
            ),
            _integration_chunk(
                id=c2_id,
                tenant_id=tenant_id,
                knowledge_base_id=src_kb.id,
                knowledge_id=source.id,
                chunk_index=1,
                pre_chunk_id=c1_id,
            ),
        ]
    )

    result = await clone_knowledge(
        tenant_id=tenant_id,
        knowledge_id=source.id,
        target_kb_id=dst_kb.id,
        knowledge_repo=KnowledgeRepository(session),
        chunk_repo=chunk_repo,
        tag_repo=TagRepository(session),
        kb_service=kb_service,
    )
    assert result is not None
    assert result.id != source.id
    assert result.knowledge_base_id == dst_kb.id
    assert result.parse_status == PARSE_STATUS_COMPLETED

    dst_chunks = await chunk_repo.find_all_by_column_values(
        {"tenant_id": tenant_id, "knowledge_id": result.id}
    )
    assert len(dst_chunks) == 2
    by_index = {chunk.chunk_index: chunk for chunk in dst_chunks}
    assert by_index[0].next_chunk_id == by_index[1].id
    assert by_index[1].pre_chunk_id == by_index[0].id

    # the source item and its chunks are untouched
    src_chunks = await chunk_repo.find_all_by_column_values(
        {"tenant_id": tenant_id, "knowledge_id": source.id}
    )
    assert len(src_chunks) == 2


async def test_integration_clone_knowledge_deep_copies_file_and_images(
    session: AsyncSession,
) -> None:
    tenant_id = _int32_tenant_id()
    kb_service = KBService(kb_repo=KnowledgeBaseRepository(session))
    src_kb = await kb_service.create_knowledge_base(
        tenant_id=tenant_id, name="img-src", kb_type="document"
    )
    dst_kb = await kb_service.create_knowledge_base(
        tenant_id=tenant_id, name="img-dst", kb_type="document"
    )
    knowledge_service = KnowledgeService(knowledge_repo=KnowledgeRepository(session))
    source = await knowledge_service.create_document(
        tenant_id=tenant_id,
        knowledge_base_id=src_kb.id,
        type="file",
        title="Doc",
        source="doc.pdf",
        parse_status=PARSE_STATUS_COMPLETED,
        file_name="doc.pdf",
        file_type="pdf",
        file_path="src://doc.pdf",
    )
    chunk_repo = ChunkRepository(session)
    await chunk_repo.create_many(
        [
            _integration_chunk(
                tenant_id=tenant_id,
                knowledge_base_id=src_kb.id,
                knowledge_id=source.id,
                chunk_index=0,
                content="![a](src://img/a.png)",
                image_info=json.dumps(
                    [{"url": "src://img/a.png", "original_url": "src://img/a.png"}]
                ),
            )
        ]
    )
    copier = _FakeObjectCopier()

    result = await clone_knowledge(
        tenant_id=tenant_id,
        knowledge_id=source.id,
        target_kb_id=dst_kb.id,
        knowledge_repo=KnowledgeRepository(session),
        chunk_repo=chunk_repo,
        tag_repo=TagRepository(session),
        kb_service=kb_service,
        object_copier=copier,
    )
    assert result is not None
    assert result.file_path == f"new://{tenant_id}/{result.id}/obj"

    dst_chunks = await chunk_repo.find_all_by_column_values(
        {"tenant_id": tenant_id, "knowledge_id": result.id}
    )
    assert len(dst_chunks) == 1
    parsed = json.loads(dst_chunks[0].image_info or "")
    assert parsed[0]["url"] == f"new://{tenant_id}/{result.id}/obj"
    assert "src://img/a.png" not in dst_chunks[0].content


async def test_integration_move_knowledge_reuse_vectors(session: AsyncSession) -> None:
    tenant_id = _int32_tenant_id()
    kb_service = KBService(kb_repo=KnowledgeBaseRepository(session))
    src_kb = await kb_service.create_knowledge_base(
        tenant_id=tenant_id, name="move-src", kb_type="document"
    )
    dst_kb = await kb_service.create_knowledge_base(
        tenant_id=tenant_id, name="move-dst", kb_type="document"
    )
    knowledge_service = KnowledgeService(knowledge_repo=KnowledgeRepository(session))
    doc = await knowledge_service.create_document(
        tenant_id=tenant_id,
        knowledge_base_id=src_kb.id,
        type="manual",
        title="Move me",
        source="manual",
        parse_status=PARSE_STATUS_COMPLETED,
    )
    chunk_repo = ChunkRepository(session)
    await chunk_repo.create_many(
        [
            _integration_chunk(
                tenant_id=tenant_id,
                knowledge_base_id=src_kb.id,
                knowledge_id=doc.id,
                chunk_index=0,
            )
        ]
    )
    tag_service = TagService(tag_repo=TagRepository(session))

    moved = await move_knowledge(
        tenant_id=tenant_id,
        knowledge_id=doc.id,
        source_kb_id=src_kb.id,
        target_kb_id=dst_kb.id,
        mode=MOVE_MODE_REUSE_VECTORS,
        knowledge_repo=KnowledgeRepository(session),
        chunk_repo=chunk_repo,
        tag_service=tag_service,
        kb_service=kb_service,
    )
    assert moved.knowledge_base_id == dst_kb.id
    assert moved.parse_status == PARSE_STATUS_COMPLETED

    chunks = await chunk_repo.find_all_by_column_values(
        {"tenant_id": tenant_id, "knowledge_id": doc.id}
    )
    assert len(chunks) == 1
    assert chunks[0].knowledge_base_id == dst_kb.id


async def test_integration_move_knowledge_reparse(session: AsyncSession) -> None:
    tenant_id = _int32_tenant_id()
    kb_service = KBService(kb_repo=KnowledgeBaseRepository(session))
    src_kb = await kb_service.create_knowledge_base(
        tenant_id=tenant_id, name="reparse-src", kb_type="document"
    )
    dst_kb = await kb_service.create_knowledge_base(
        tenant_id=tenant_id, name="reparse-dst", kb_type="document"
    )
    knowledge_service = KnowledgeService(knowledge_repo=KnowledgeRepository(session))
    doc = await knowledge_service.create_document(
        tenant_id=tenant_id,
        knowledge_base_id=src_kb.id,
        type="manual",
        title="Reparse me",
        source="manual",
        parse_status=PARSE_STATUS_COMPLETED,
    )
    chunk_repo = ChunkRepository(session)
    await chunk_repo.create_many(
        [
            _integration_chunk(
                tenant_id=tenant_id,
                knowledge_base_id=src_kb.id,
                knowledge_id=doc.id,
                chunk_index=0,
            )
        ]
    )
    tag_service = TagService(tag_repo=TagRepository(session))

    moved = await move_knowledge(
        tenant_id=tenant_id,
        knowledge_id=doc.id,
        source_kb_id=src_kb.id,
        target_kb_id=dst_kb.id,
        mode=MOVE_MODE_REPARSE,
        knowledge_repo=KnowledgeRepository(session),
        chunk_repo=chunk_repo,
        tag_service=tag_service,
        kb_service=kb_service,
    )
    assert moved.knowledge_base_id == dst_kb.id
    assert moved.parse_status == PARSE_STATUS_PENDING
    assert moved.enable_status == "disabled"

    chunks = await chunk_repo.find_all_by_column_values(
        {"tenant_id": tenant_id, "knowledge_id": doc.id}
    )
    assert chunks == []
    deleted = await chunk_repo.find_all_by_column_values(
        {"tenant_id": tenant_id, "knowledge_id": doc.id},
        exclude_deleted_or_archived=False,
    )
    assert len(deleted) == 1
    assert deleted[0].deleted_at is not None
