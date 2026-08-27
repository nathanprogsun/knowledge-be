"""Unit and integration tests for ``ChunkService``.

Unit tests drive the service against spec'd ``AsyncMock`` repositories with
closure-captured state (the CRUD paths) or canned returns (the document
edit). The integration section runs against the real applied schema and
skips when the database is unreachable, so the unit suite stays runnable
without Postgres.
"""

from __future__ import annotations

import itertools
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from unittest.mock import ANY, AsyncMock

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.pool import NullPool

from src.common.exception import ConflictError, DataError, NotFoundError, ValidationError
from src.core.knowledge.chunks.factory import build_chunk_service
from src.core.knowledge.chunks.service.chunk_service import (
    ChunkIndexSyncer,
    ChunkService,
    image_urls_in_content,
    validate_edited_chunk_images,
)
from src.db.base import DatabaseEngine
from src.db.dao.chunk_repository import ChunkRepository
from src.db.models.chunk import Chunk
from src.settings import get_settings, reset_settings_cache
from tests.util.service_test import ServiceTest

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_TENANT_ID = 7_100_000
# The ``chunks.tenant_id`` column is a 32-bit INTEGER, so integration rows
# need a 32-bit-safe unique id (the tenants table's 64-bit ids do not fit).
_tenant_counter = itertools.count(8_000_000)


def _tenant_id() -> int:
    """Return a unique 32-bit tenant id for the ``chunks`` table."""
    return next(_tenant_counter)


def _cid() -> str:
    return f"chunk-{uuid.uuid4().hex[:12]}"


def _sample_chunk(
    *,
    tenant_id: int,
    id: str | None = None,
    knowledge_id: str = "knowledge-doc-1",
    knowledge_base_id: str = "kb-1",
    content: str = "The quick brown fox jumps over the lazy dog.",
    chunk_index: int = 0,
    **overrides: object,
) -> Chunk:
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
            **overrides,
        }
    )


# ── CRUD mock with closure-captured state ─────────────────────────────


def _make_crud_repo() -> tuple[AsyncMock, dict[str, Chunk]]:
    repo = AsyncMock(spec=ChunkRepository)
    rows: dict[str, Chunk] = {}

    async def _create_many(chunks: list[Chunk]) -> list[Chunk]:
        stored: list[Chunk] = []
        for i, chunk in enumerate(chunks, start=1):
            persisted = chunk.model_copy(update={"seq_id": len(rows) + i})
            rows[persisted.id] = persisted
            stored.append(persisted)
        return stored

    async def _get_by_id(tenant_id: int, id: str) -> Chunk:
        row = rows.get(id)
        if row is None or row.tenant_id != tenant_id or row.deleted_at is not None:
            raise NotFoundError(code="chunk.not_found", message=f"chunk {id} not found")
        return row

    async def _get_by_id_only(id: str) -> Chunk | None:
        row = rows.get(id)
        if row is None or row.deleted_at is not None:
            return None
        return row

    async def _list_by_knowledge_id(tenant_id: int, knowledge_id: str) -> list[Chunk]:
        return [
            row
            for row in rows.values()
            if row.tenant_id == tenant_id
            and row.knowledge_id == knowledge_id
            and row.chunk_type == "text"
            and row.deleted_at is None
        ]

    async def _list_by_parent_id(tenant_id: int, parent_id: str) -> list[Chunk]:
        return [
            row
            for row in rows.values()
            if row.tenant_id == tenant_id
            and row.parent_chunk_id == parent_id
            and row.deleted_at is None
        ]

    async def _update(row: Chunk) -> Chunk:
        existing = rows.get(row.id)
        if existing is None:
            raise DataError(
                code="chunk.update_no_row",
                message=f"chunk {row.id} not found for update",
            )
        stored = row.model_copy(
            update={
                "seq_id": existing.seq_id,
                "tenant_id": existing.tenant_id,
                "created_at": existing.created_at,
            }
        )
        rows[stored.id] = stored
        return stored

    async def _soft_delete(*, tenant_id: int, id: str, now: datetime) -> bool:
        row = rows.get(id)
        if row is None or row.tenant_id != tenant_id or row.deleted_at is not None:
            return False
        rows[id] = row.model_copy(update={"deleted_at": now, "updated_at": now})
        return True

    async def _delete_by_knowledge_id(*, tenant_id: int, knowledge_id: str, now: datetime) -> int:
        affected = 0
        for cid, row in list(rows.items()):
            if (
                row.tenant_id == tenant_id
                and row.knowledge_id == knowledge_id
                and row.deleted_at is None
            ):
                rows[cid] = row.model_copy(update={"deleted_at": now, "updated_at": now})
                affected += 1
        return affected

    repo.create_many.side_effect = _create_many
    repo.get_by_id.side_effect = _get_by_id
    repo.get_by_id_only.side_effect = _get_by_id_only
    repo.list_by_knowledge_id.side_effect = _list_by_knowledge_id
    repo.list_by_parent_id.side_effect = _list_by_parent_id
    repo.update.side_effect = _update
    repo.soft_delete.side_effect = _soft_delete
    repo.delete_by_knowledge_id.side_effect = _delete_by_knowledge_id
    return repo, rows


@pytest.fixture
def crud_repo_and_rows() -> tuple[AsyncMock, dict[str, Chunk]]:
    return _make_crud_repo()


@pytest.fixture
def crud_repo(crud_repo_and_rows: tuple[AsyncMock, dict[str, Chunk]]) -> AsyncMock:
    return crud_repo_and_rows[0]


@pytest.fixture
def rows(crud_repo_and_rows: tuple[AsyncMock, dict[str, Chunk]]) -> dict[str, Chunk]:
    return crud_repo_and_rows[1]


@pytest.fixture
def service(crud_repo: AsyncMock) -> ChunkService:
    return ChunkService(chunk_repo=crud_repo)


# ── CRUD ──────────────────────────────────────────────────────────────


class TestChunkServiceCrud(ServiceTest):
    async def test_create_chunks_persists_and_returns_rows(
        self,
        service: ChunkService,
        rows: dict[str, Chunk],
    ) -> None:
        persisted = await service.create_chunks(
            chunks=[_sample_chunk(tenant_id=_TENANT_ID, id="c1")]
        )

        assert [c.id for c in persisted] == ["c1"]
        assert rows["c1"].id == "c1"

    async def test_get_chunk_by_id_found(
        self,
        service: ChunkService,
        rows: dict[str, Chunk],
    ) -> None:
        cid = _cid()
        rows[cid] = _sample_chunk(tenant_id=_TENANT_ID, id=cid)

        result = await service.get_chunk_by_id(tenant_id=_TENANT_ID, id=cid)

        assert result.id == cid

    async def test_get_chunk_by_id_raises_not_found(self, service: ChunkService) -> None:
        with pytest.raises(NotFoundError) as excinfo:
            await service.get_chunk_by_id(tenant_id=_TENANT_ID, id=_cid())
        assert excinfo.value.code == "chunk.not_found"

    async def test_get_chunk_by_id_only_ignores_tenant(
        self,
        service: ChunkService,
        rows: dict[str, Chunk],
    ) -> None:
        cid = _cid()
        rows[cid] = _sample_chunk(tenant_id=_TENANT_ID, id=cid)

        result = await service.get_chunk_by_id_only(id=cid)

        assert result.id == cid

    async def test_get_chunk_by_id_only_raises_not_found(self, service: ChunkService) -> None:
        with pytest.raises(NotFoundError) as excinfo:
            await service.get_chunk_by_id_only(id=_cid())
        assert excinfo.value.code == "chunk.not_found"

    async def test_list_chunks_by_knowledge_id_scopes_and_orders(
        self,
        service: ChunkService,
        rows: dict[str, Chunk],
    ) -> None:
        cid = _cid()
        rows[cid] = _sample_chunk(tenant_id=_TENANT_ID, id=cid, knowledge_id="kid-x")
        rows[_cid()] = _sample_chunk(tenant_id=_TENANT_ID, knowledge_id="kid-other")

        result = await service.list_chunks_by_knowledge_id(
            tenant_id=_TENANT_ID,
            knowledge_id="kid-x",
        )

        assert [c.id for c in result] == [cid]

    async def test_list_chunk_by_parent_id(
        self,
        service: ChunkService,
        rows: dict[str, Chunk],
    ) -> None:
        parent = _cid()
        child = _cid()
        rows[child] = _sample_chunk(tenant_id=_TENANT_ID, id=child, parent_chunk_id=parent)
        rows[_cid()] = _sample_chunk(tenant_id=_TENANT_ID)

        result = await service.list_chunk_by_parent_id(tenant_id=_TENANT_ID, parent_id=parent)

        assert [c.id for c in result] == [child]

    async def test_update_chunk_overwrites_columns(
        self,
        service: ChunkService,
        rows: dict[str, Chunk],
    ) -> None:
        cid = _cid()
        rows[cid] = _sample_chunk(tenant_id=_TENANT_ID, id=cid, content="before")
        edited = rows[cid].model_copy(update={"content": "after"})

        result = await service.update_chunk(chunk=edited)

        assert result.content == "after"
        assert rows[cid].content == "after"

    async def test_update_chunks_empty_is_noop(
        self,
        service: ChunkService,
        crud_repo: AsyncMock,
    ) -> None:
        assert await service.update_chunks(chunks=[]) == []
        crud_repo.update.assert_not_awaited()

    async def test_update_chunks_batch(
        self,
        service: ChunkService,
        rows: dict[str, Chunk],
    ) -> None:
        c1, c2 = _cid(), _cid()
        rows[c1] = _sample_chunk(tenant_id=_TENANT_ID, id=c1, content="a")
        rows[c2] = _sample_chunk(tenant_id=_TENANT_ID, id=c2, content="b")

        result = await service.update_chunks(
            chunks=[
                rows[c1].model_copy(update={"content": "A"}),
                rows[c2].model_copy(update={"content": "B"}),
            ]
        )

        assert [c.content for c in result] == ["A", "B"]

    async def test_delete_chunk_soft_deletes_once(
        self,
        service: ChunkService,
        rows: dict[str, Chunk],
    ) -> None:
        cid = _cid()
        rows[cid] = _sample_chunk(tenant_id=_TENANT_ID, id=cid)

        assert await service.delete_chunk(tenant_id=_TENANT_ID, id=cid) is True
        assert rows[cid].deleted_at is not None
        assert await service.delete_chunk(tenant_id=_TENANT_ID, id=cid) is False

    async def test_delete_chunks_returns_affected_count(
        self,
        service: ChunkService,
        rows: dict[str, Chunk],
    ) -> None:
        c1, c2 = _cid(), _cid()
        rows[c1] = _sample_chunk(tenant_id=_TENANT_ID, id=c1)
        rows[c2] = _sample_chunk(tenant_id=_TENANT_ID, id=c2)

        assert await service.delete_chunks(tenant_id=_TENANT_ID, ids=[c1, c2, _cid()]) == 2

    async def test_delete_chunks_empty_is_noop(
        self,
        service: ChunkService,
        crud_repo: AsyncMock,
    ) -> None:
        assert await service.delete_chunks(tenant_id=_TENANT_ID, ids=[]) == 0
        crud_repo.soft_delete.assert_not_awaited()

    async def test_delete_chunks_by_knowledge_id(
        self,
        service: ChunkService,
        rows: dict[str, Chunk],
    ) -> None:
        c1, c2 = _cid(), _cid()
        rows[c1] = _sample_chunk(tenant_id=_TENANT_ID, id=c1, knowledge_id="kid-x")
        rows[c2] = _sample_chunk(tenant_id=_TENANT_ID, id=c2, knowledge_id="kid-x")

        affected = await service.delete_chunks_by_knowledge_id(
            tenant_id=_TENANT_ID,
            knowledge_id="kid-x",
        )

        assert affected == 2

    async def test_delete_by_knowledge_list(
        self,
        service: ChunkService,
        rows: dict[str, Chunk],
    ) -> None:
        rows[_cid()] = _sample_chunk(tenant_id=_TENANT_ID, knowledge_id="kid-a")
        rows[_cid()] = _sample_chunk(tenant_id=_TENANT_ID, knowledge_id="kid-b")

        affected = await service.delete_by_knowledge_list(
            tenant_id=_TENANT_ID,
            ids=["kid-a", "kid-b"],
        )

        assert affected == 2

    async def test_delete_by_knowledge_list_empty_is_noop(
        self,
        service: ChunkService,
        crud_repo: AsyncMock,
    ) -> None:
        assert await service.delete_by_knowledge_list(tenant_id=_TENANT_ID, ids=[]) == 0
        crud_repo.delete_by_knowledge_id.assert_not_awaited()


# ── update_document_chunk ─────────────────────────────────────────────


class _Syncer:
    """Stub ``ChunkIndexSyncer`` that records calls and can fail."""

    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[int, Chunk]] = []

    async def sync_chunk(self, *, tenant_id: int, chunk: Chunk) -> None:
        self.calls.append((tenant_id, chunk))
        if self.error is not None:
            raise self.error


class TestUpdateDocumentChunk(ServiceTest):
    def _service(
        self,
        *,
        current: Chunk,
        updated: Chunk,
        syncer: ChunkIndexSyncer | None = None,
    ) -> tuple[ChunkService, AsyncMock]:
        repo = AsyncMock(spec=ChunkRepository)
        repo.get_by_id.side_effect = [current]
        repo.update_document_chunk.return_value = updated
        return ChunkService(chunk_repo=repo, index_syncer=syncer), repo

    async def test_edit_bumps_revision_and_settles_ready(self) -> None:
        cid = _cid()
        current = _sample_chunk(
            tenant_id=_TENANT_ID, id=cid, content="original", source_content="original"
        )
        updated = current.model_copy(
            update={
                "content": "revised",
                "content_revision": 1,
                "index_status": "processing",
                "last_editor_id": "usr-1",
            }
        )
        service, repo = self._service(current=current, updated=updated)
        repo.update.return_value = updated.model_copy(update={"index_status": "ready"})

        result = await service.update_document_chunk(
            tenant_id=_TENANT_ID,
            chunk_id=cid,
            content="  revised  ",
            last_editor_id="usr-1",
        )

        assert result.content_revision == 1
        assert result.index_status == "ready"
        repo.update_document_chunk.assert_awaited_once_with(
            tenant_id=_TENANT_ID,
            chunk_id=cid,
            content="  revised  ",
            is_enabled=None,
            expected_revision=0,
            last_editor_id="usr-1",
            now=ANY,
        )
        repo.update.assert_awaited_once()

    async def test_missing_expected_revision_resolves_to_current(self) -> None:
        cid = _cid()
        current = _sample_chunk(tenant_id=_TENANT_ID, id=cid, content_revision=3)
        updated = current.model_copy(update={"content_revision": 4, "index_status": "processing"})
        service, repo = self._service(current=current, updated=updated)
        repo.update.return_value = updated.model_copy(update={"index_status": "ready"})

        await service.update_document_chunk(
            tenant_id=_TENANT_ID,
            chunk_id=cid,
            content="edited",
            last_editor_id="usr-1",
        )

        repo.update_document_chunk.assert_awaited_once_with(
            tenant_id=_TENANT_ID,
            chunk_id=cid,
            content="edited",
            is_enabled=None,
            expected_revision=3,
            last_editor_id="usr-1",
            now=ANY,
        )

    async def test_explicit_expected_revision_is_passed_through(self) -> None:
        cid = _cid()
        current = _sample_chunk(tenant_id=_TENANT_ID, id=cid, content_revision=3)
        updated = current.model_copy(update={"content_revision": 4, "index_status": "processing"})
        service, repo = self._service(current=current, updated=updated)
        repo.update.return_value = updated.model_copy(update={"index_status": "ready"})

        await service.update_document_chunk(
            tenant_id=_TENANT_ID,
            chunk_id=cid,
            content="edited",
            expected_revision=2,
            last_editor_id="usr-1",
        )

        repo.update_document_chunk.assert_awaited_once_with(
            tenant_id=_TENANT_ID,
            chunk_id=cid,
            content="edited",
            is_enabled=None,
            expected_revision=2,
            last_editor_id="usr-1",
            now=ANY,
        )

    async def test_noop_edit_returns_without_settling(self) -> None:
        cid = _cid()
        current = _sample_chunk(tenant_id=_TENANT_ID, id=cid, content="same")
        service, repo = self._service(current=current, updated=current)

        result = await service.update_document_chunk(
            tenant_id=_TENANT_ID,
            chunk_id=cid,
            content="same",
            last_editor_id="usr-1",
        )

        assert result.index_status == "ready"
        repo.update.assert_not_awaited()

    async def test_noop_edit_on_failed_row_retries_and_settles_ready(self) -> None:
        cid = _cid()
        current = _sample_chunk(tenant_id=_TENANT_ID, id=cid, content="same", index_status="failed")
        service, repo = self._service(current=current, updated=current)
        repo.update.return_value = current.model_copy(update={"index_status": "ready"})

        result = await service.update_document_chunk(
            tenant_id=_TENANT_ID,
            chunk_id=cid,
            content="same",
            last_editor_id="usr-1",
        )

        assert result.index_status == "ready"
        repo.update.assert_awaited_once()

    async def test_edit_rejects_new_image_before_write(self) -> None:
        cid = _cid()
        source = "before ![a](https://a/1.png)"
        current = _sample_chunk(tenant_id=_TENANT_ID, id=cid, content=source, source_content=source)
        service, repo = self._service(current=current, updated=current)

        with pytest.raises(ValidationError) as excinfo:
            await service.update_document_chunk(
                tenant_id=_TENANT_ID,
                chunk_id=cid,
                content="before ![a](https://a/1.png) ![b](https://a/2.png)",
                last_editor_id="usr-1",
            )

        assert excinfo.value.code == "chunk.image_add_unsupported"
        repo.update_document_chunk.assert_not_awaited()

    async def test_edit_keeps_existing_images(self) -> None:
        cid = _cid()
        source = "before ![a](https://a/1.png)"
        current = _sample_chunk(tenant_id=_TENANT_ID, id=cid, content=source, source_content=source)
        updated = current.model_copy(
            update={
                "content": "after ![a](https://a/1.png)",
                "content_revision": 1,
                "index_status": "processing",
            }
        )
        service, repo = self._service(current=current, updated=updated)
        repo.update.return_value = updated.model_copy(update={"index_status": "ready"})

        result = await service.update_document_chunk(
            tenant_id=_TENANT_ID,
            chunk_id=cid,
            content="after ![a](https://a/1.png)",
            last_editor_id="usr-1",
        )

        assert result.index_status == "ready"
        repo.update_document_chunk.assert_awaited_once()

    async def test_edit_passthrough_not_found(self) -> None:
        cid = _cid()
        repo = AsyncMock(spec=ChunkRepository)
        repo.get_by_id.side_effect = [
            NotFoundError(code="chunk.not_found", message=f"chunk {cid} not found")
        ]
        service = ChunkService(chunk_repo=repo)

        with pytest.raises(NotFoundError) as excinfo:
            await service.update_document_chunk(
                tenant_id=_TENANT_ID,
                chunk_id=cid,
                content="x",
                last_editor_id="usr-1",
            )
        assert excinfo.value.code == "chunk.not_found"

    async def test_edit_passthrough_revision_conflict(self) -> None:
        cid = _cid()
        current = _sample_chunk(tenant_id=_TENANT_ID, id=cid)
        repo = AsyncMock(spec=ChunkRepository)
        repo.get_by_id.side_effect = [current]
        repo.update_document_chunk.side_effect = ConflictError(
            code="chunk.revision_conflict",
            message="chunk changed",
        )
        service = ChunkService(chunk_repo=repo)

        with pytest.raises(ConflictError) as excinfo:
            await service.update_document_chunk(
                tenant_id=_TENANT_ID,
                chunk_id=cid,
                content="x",
                expected_revision=0,
                last_editor_id="usr-1",
            )
        assert excinfo.value.code == "chunk.revision_conflict"

    async def test_edit_passthrough_content_empty(self) -> None:
        cid = _cid()
        current = _sample_chunk(tenant_id=_TENANT_ID, id=cid)
        repo = AsyncMock(spec=ChunkRepository)
        repo.get_by_id.side_effect = [current]
        repo.update_document_chunk.side_effect = ValidationError(
            code="chunk.content_empty",
            message="chunk content cannot be empty",
        )
        service = ChunkService(chunk_repo=repo)

        with pytest.raises(ValidationError) as excinfo:
            await service.update_document_chunk(
                tenant_id=_TENANT_ID,
                chunk_id=cid,
                content="   ",
                last_editor_id="usr-1",
            )
        assert excinfo.value.code == "chunk.content_empty"

    async def test_syncer_failure_marks_row_failed(self) -> None:
        cid = _cid()
        current = _sample_chunk(tenant_id=_TENANT_ID, id=cid, content="original")
        updated = current.model_copy(
            update={"content": "edited", "content_revision": 1, "index_status": "processing"}
        )
        syncer = _Syncer(error=RuntimeError("vector store down"))
        service, repo = self._service(current=current, updated=updated, syncer=syncer)
        captured: list[Chunk] = []

        async def _update(row: Chunk) -> Chunk:
            captured.append(row)
            return row

        repo.update.side_effect = _update

        result = await service.update_document_chunk(
            tenant_id=_TENANT_ID,
            chunk_id=cid,
            content="edited",
            last_editor_id="usr-1",
        )

        assert result.index_status == "failed"
        assert captured[0].index_status == "failed"
        assert syncer.calls[0][0] == _TENANT_ID
        assert syncer.calls[0][1].content == "edited"

    async def test_syncer_success_marks_row_ready(self) -> None:
        cid = _cid()
        current = _sample_chunk(tenant_id=_TENANT_ID, id=cid, content="original")
        updated = current.model_copy(
            update={"content": "edited", "content_revision": 1, "index_status": "processing"}
        )
        syncer = _Syncer()
        service, repo = self._service(current=current, updated=updated, syncer=syncer)
        repo.update.return_value = updated.model_copy(update={"index_status": "ready"})

        result = await service.update_document_chunk(
            tenant_id=_TENANT_ID,
            chunk_id=cid,
            content="edited",
            last_editor_id="usr-1",
        )

        assert result.index_status == "ready"
        assert len(syncer.calls) == 1


# ── Image helpers and factory ─────────────────────────────────────────


class TestImageHelpers(ServiceTest):
    def test_extracts_markdown_and_html_urls(self) -> None:
        content = 'before ![alt](https://a/x.png) after <img src="https://b/y.jpg"> end'

        assert image_urls_in_content(content) == {"https://a/x.png", "https://b/y.jpg"}

    def test_ignores_data_src_attribute(self) -> None:
        content = '<img data-src="https://lazy.png" src="https://real.png">'

        assert image_urls_in_content(content) == {"https://real.png"}

    def test_returns_empty_set_without_images(self) -> None:
        assert image_urls_in_content("plain text") == set()

    def test_validate_keeps_existing_images(self) -> None:
        validate_edited_chunk_images(
            "![a](https://a/1.png) original",
            "revised ![a](https://a/1.png)",
        )

    def test_validate_rejects_new_image(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            validate_edited_chunk_images(
                "![a](https://a/1.png)",
                "![a](https://a/1.png) ![b](https://a/2.png)",
            )
        assert excinfo.value.code == "chunk.image_add_unsupported"


class TestChunkServiceFactory(ServiceTest):
    def test_build_chunk_service_returns_service(self) -> None:
        service = build_chunk_service(AsyncMock(spec=AsyncSession))
        assert isinstance(service, ChunkService)

    def test_build_chunk_service_wires_index_syncer(self) -> None:
        syncer = _Syncer()
        service = build_chunk_service(AsyncMock(spec=AsyncSession), index_syncer=syncer)
        assert isinstance(service, ChunkService)


# ── Integration against the real schema ───────────────────────────────


@pytest_asyncio.fixture
async def db_session() -> AsyncIterator[AsyncSession]:
    """Per-test session against the real applied schema; skips without a DB."""
    reset_settings_cache()
    engine = DatabaseEngine(url=get_settings().database_url, poolclass=NullPool)
    try:
        await engine.prewarm()
    except Exception as exc:
        await engine.close()
        pytest.skip(f"integration database unavailable: {exc}")
    async with engine.session_factory() as s:
        yield s
        await s.rollback()
    await engine.close()


class TestChunkServiceIntegration(ServiceTest):
    async def test_update_document_chunk_persists_and_settles_ready(
        self,
        db_session: AsyncSession,
    ) -> None:
        tid = _tenant_id()
        service = ChunkService(chunk_repo=ChunkRepository(db_session))
        persisted = await service.create_chunks(
            chunks=[
                _sample_chunk(
                    tenant_id=tid,
                    content="  original  ",
                    source_content="original",
                    knowledge_id="kid-it-1",
                    knowledge_base_id="kb-it-1",
                )
            ]
        )
        cid = persisted[0].id

        updated = await service.update_document_chunk(
            tenant_id=tid,
            chunk_id=cid,
            content="  revised body  ",
            expected_revision=0,
            last_editor_id="usr-it-1",
        )

        assert updated.content == "revised body"
        assert updated.content_revision == 1
        assert updated.last_editor_id == "usr-it-1"
        assert updated.index_status == "ready"
        reloaded = await ChunkRepository(db_session).get_by_id(tid, cid)
        assert reloaded.content == "revised body"
        assert reloaded.content_revision == 1
        assert reloaded.index_status == "ready"

    async def test_update_document_chunk_revision_conflict(
        self,
        db_session: AsyncSession,
    ) -> None:
        tid = _tenant_id()
        service = ChunkService(chunk_repo=ChunkRepository(db_session))
        persisted = await service.create_chunks(chunks=[_sample_chunk(tenant_id=tid)])
        cid = persisted[0].id
        await service.update_document_chunk(
            tenant_id=tid,
            chunk_id=cid,
            content="first edit",
            expected_revision=0,
            last_editor_id="usr-it-1",
        )

        with pytest.raises(ConflictError):
            await service.update_document_chunk(
                tenant_id=tid,
                chunk_id=cid,
                content="second edit",
                expected_revision=0,
                last_editor_id="usr-it-1",
            )

    async def test_update_document_chunk_not_found(self, db_session: AsyncSession) -> None:
        tid = _tenant_id()
        service = ChunkService(chunk_repo=ChunkRepository(db_session))

        with pytest.raises(NotFoundError) as excinfo:
            await service.update_document_chunk(
                tenant_id=tid,
                chunk_id=_cid(),
                content="x",
                last_editor_id="usr-it-1",
            )
        assert excinfo.value.code == "chunk.not_found"

    async def test_soft_delete_and_get_round_trip(self, db_session: AsyncSession) -> None:
        tid = _tenant_id()
        service = ChunkService(chunk_repo=ChunkRepository(db_session))
        persisted = await service.create_chunks(chunks=[_sample_chunk(tenant_id=tid)])
        cid = persisted[0].id

        assert await service.delete_chunk(tenant_id=tid, id=cid) is True

        with pytest.raises(NotFoundError):
            await service.get_chunk_by_id(tenant_id=tid, id=cid)
