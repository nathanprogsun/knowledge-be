"""Unit and integration tests for chunk revision + generated-question logic.

Covers ``src/core/knowledge/chunks``:

- the pure helpers: ``ChunkRevision`` model defaults, the
  ``ChunkRevisionInfo`` projection, the read-side queries
  (``list_chunk_revisions`` / ``get_chunk_revision``) against a mocked
  repository, and the generated-question metadata helpers (bind/unbind,
  source id, currency, parse/serialize).
- the service-level operations (``revert_document_chunk``,
  ``upsert_generated_question``, ``delete_generated_question``) against
  mocked services and sync hooks.
- integration tests against the real applied schema that exercise the
  service operations end to end.

The DB-backed repository behavior is exercised separately in
``tests/integration/db/dao/test_chunk_revisions_repository.py``.
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
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.pool import NullPool

from src.common.exception import ConflictError, NotFoundError, ValidationError
from src.common.json import JsonObject
from src.core.knowledge.chunks.questions import (
    DocumentChunkMetadata,
    GeneratedQuestion,
    bind_generated_question,
    delete_generated_question,
    generated_question_source_id,
    get_question_strings,
    is_question_current,
    parse_document_metadata,
    unbind_generated_question,
    upsert_generated_question,
)
from src.core.knowledge.chunks.revisions import (
    ChunkRevisionInfo,
    get_chunk_revision,
    list_chunk_revisions,
    revert_document_chunk,
)
from src.core.knowledge.chunks.service.chunk_service import ChunkService
from src.db.base import DatabaseEngine
from src.db.dao.chunk_repository import ChunkRepository
from src.db.dao.chunk_revision_repository import ChunkRevisionRepository
from src.db.models.chunk import Chunk
from src.db.models.chunk_revision import ChunkRevision
from src.settings import get_settings, reset_settings_cache

TENANT_ID = 7
CHUNK_ID = "chunk-1"
NOW = datetime(2026, 1, 1, tzinfo=UTC)
_FAKER_SEED_MAX = 100_000_000


def _revision(*, revision: int, **overrides: object) -> ChunkRevision:
    values: dict[str, object] = {
        "id": f"rev-{revision}",
        "tenant_id": TENANT_ID,
        "knowledge_base_id": "kb-1",
        "knowledge_id": "knowledge-1",
        "chunk_id": CHUNK_ID,
        "revision": revision,
        "content": f"body-{revision}",
        "is_enabled": True,
        "editor_id": "editor-1",
        "edit_source": "user",
        "edited_at": NOW,
        "created_at": NOW,
    }
    values.update(overrides)
    return ChunkRevision.model_validate(values)


# ── ChunkRevision model ─────────────────────────────────────────────


def test_model_defaults_mirror_sql_defaults() -> None:
    row = ChunkRevision(
        id="rev-1",
        tenant_id=TENANT_ID,
        knowledge_base_id="kb-1",
        knowledge_id="knowledge-1",
        chunk_id=CHUNK_ID,
        revision=1,
        edited_at=NOW,
        created_at=NOW,
    )
    assert row.content == ""
    assert row.is_enabled is True
    assert row.editor_id == ""
    assert row.edit_source == "user"


def test_revision_info_map_from_db_copies_every_column() -> None:
    row = _revision(revision=2, content="edited body", is_enabled=False)

    info = ChunkRevisionInfo.map_from_db(row)

    assert info.id == "rev-2"
    assert info.tenant_id == TENANT_ID
    assert info.chunk_id == CHUNK_ID
    assert info.revision == 2
    assert info.content == "edited body"
    assert info.is_enabled is False
    assert info.editor_id == "editor-1"
    assert info.edit_source == "user"
    assert info.edited_at == NOW
    assert info.created_at == NOW


# ── read-side queries (mocked repository) ───────────────────────────


def _make_repo(rows: list[ChunkRevision]) -> AsyncMock:
    repo = AsyncMock(spec=ChunkRevisionRepository)

    async def _list(*, tenant_id: int, chunk_id: str) -> list[ChunkRevision]:
        return [r for r in rows if r.tenant_id == tenant_id and r.chunk_id == chunk_id]

    async def _get(
        *,
        tenant_id: int,
        chunk_id: str,
        revision: int,
    ) -> ChunkRevision | None:
        return next(
            (
                r
                for r in rows
                if r.tenant_id == tenant_id
                and r.chunk_id == chunk_id
                and r.revision == revision
            ),
            None,
        )

    repo.list_chunk_revisions.side_effect = _list
    repo.get_chunk_revision.side_effect = _get
    return repo


async def test_list_chunk_revisions_returns_newest_first() -> None:
    repo = _make_repo([_revision(revision=1), _revision(revision=2), _revision(revision=3)])

    infos = await list_chunk_revisions(repo, tenant_id=TENANT_ID, chunk_id=CHUNK_ID)

    assert [i.revision for i in infos] == [1, 2, 3]


async def test_list_chunk_revisions_is_tenant_scoped() -> None:
    repo = _make_repo(
        [
            _revision(revision=1),
            _revision(revision=1, id="rev-other", tenant_id=99),
        ]
    )

    infos = await list_chunk_revisions(repo, tenant_id=TENANT_ID, chunk_id=CHUNK_ID)

    assert [i.revision for i in infos] == [1]


async def test_get_chunk_revision_returns_snapshot() -> None:
    repo = _make_repo([_revision(revision=1), _revision(revision=2)])

    info = await get_chunk_revision(
        repo,
        tenant_id=TENANT_ID,
        chunk_id=CHUNK_ID,
        revision=2,
    )

    assert info.revision == 2
    assert info.content == "body-2"


async def test_get_chunk_revision_raises_when_absent() -> None:
    repo = _make_repo([_revision(revision=1)])

    with pytest.raises(NotFoundError) as exc_info:
        await get_chunk_revision(
            repo,
            tenant_id=TENANT_ID,
            chunk_id=CHUNK_ID,
            revision=99,
        )

    assert exc_info.value.code == "chunk.revision_not_found"


# ── generated-question source id ────────────────────────────────────


def test_source_id_keeps_short_question_verbatim() -> None:
    assert generated_question_source_id("chunk-1", "question-1") == "chunk-1-question-1"


def test_source_id_hashes_oversized_question() -> None:
    chunk_id = "c" * 36
    question_id = "q" * 40
    source_id = generated_question_source_id(chunk_id, question_id)

    assert source_id.startswith(f"{chunk_id}-q")
    assert len(source_id) == len(chunk_id) + 2 + 24


# ── metadata parse / serialize ──────────────────────────────────────


def test_parse_document_metadata_none_for_absent() -> None:
    assert parse_document_metadata(None) is None
    assert parse_document_metadata("") is None


def test_parse_document_metadata_accepts_dict() -> None:
    meta = parse_document_metadata(
        {"generated_questions": [{"id": "q-1", "question": "what?", "content_revision": 3}]}
    )

    assert meta is not None
    assert meta.generated_questions[0].question == "what?"
    assert meta.generated_questions[0].content_revision == 3


def test_parse_document_metadata_accepts_json_string() -> None:
    meta = parse_document_metadata('{"generated_questions": []}')

    assert meta is not None
    assert meta.generated_questions == []


def test_parse_document_metadata_ignores_invalid_json() -> None:
    assert parse_document_metadata("{not json") is None


def test_metadata_to_json_omits_defaults() -> None:
    meta = DocumentChunkMetadata()
    assert meta.to_json() == {}


# ── bind / unbind generated questions ──────────────────────────────


def test_bind_appends_new_question() -> None:
    meta = DocumentChunkMetadata()

    updated, bound = bind_generated_question(
        meta,
        question_id=None,
        question="  what is the answer?  ",
        content_revision=3,
    )

    assert bound.id
    assert bound.question == "what is the answer?"
    assert bound.content_revision == 3
    assert updated.generated_questions == [bound]
    # immutable: the source metadata is untouched
    assert meta.generated_questions == []


def test_bind_updates_existing_question_text() -> None:
    meta = DocumentChunkMetadata(generated_questions=[GeneratedQuestion(id="q-1", question="old")])

    updated, bound = bind_generated_question(
        meta,
        question_id="q-1",
        question="new text",
        content_revision=5,
    )

    assert bound.question == "new text"
    assert bound.content_revision == 5
    assert len(updated.generated_questions) == 1
    assert meta.generated_questions[0].question == "old"


def test_bind_rejects_empty_question() -> None:
    with pytest.raises(ValidationError) as exc_info:
        bind_generated_question(
            DocumentChunkMetadata(),
            question_id=None,
            question="   ",
            content_revision=1,
        )
    assert exc_info.value.code == "chunk.question_empty"


def test_bind_rejects_unknown_question_id() -> None:
    with pytest.raises(ValidationError) as exc_info:
        bind_generated_question(
            DocumentChunkMetadata(),
            question_id="missing",
            question="text",
            content_revision=1,
        )
    assert exc_info.value.code == "chunk.question_not_found"


def test_unbind_removes_question() -> None:
    meta = DocumentChunkMetadata(
        generated_questions=[
            GeneratedQuestion(id="q-1", question="a"),
            GeneratedQuestion(id="q-2", question="b"),
        ]
    )

    updated, removed = unbind_generated_question(meta, question_id="q-1")

    assert removed.id == "q-1"
    assert [q.id for q in updated.generated_questions] == ["q-2"]
    assert len(meta.generated_questions) == 2


def test_unbind_rejects_empty_metadata() -> None:
    with pytest.raises(ValidationError) as exc_info:
        unbind_generated_question(DocumentChunkMetadata(), question_id="q-1")
    assert exc_info.value.code == "chunk.no_questions"


def test_unbind_rejects_unknown_question_id() -> None:
    meta = DocumentChunkMetadata(generated_questions=[GeneratedQuestion(id="q-1", question="a")])

    with pytest.raises(ValidationError) as exc_info:
        unbind_generated_question(meta, question_id="missing")
    assert exc_info.value.code == "chunk.question_not_found"


# ── question currency / strings ─────────────────────────────────────


def test_is_question_current_uses_per_question_revision() -> None:
    question = GeneratedQuestion(id="q-1", question="a", content_revision=4)
    assert is_question_current(None, question, chunk_revision=4) is True
    assert is_question_current(None, question, chunk_revision=5) is False


def test_is_question_current_falls_back_to_metadata_revision() -> None:
    question = GeneratedQuestion(id="q-1", question="a")
    meta = DocumentChunkMetadata(generated_questions_revision=3)

    assert is_question_current(meta, question, chunk_revision=3) is True
    assert is_question_current(None, question, chunk_revision=3) is False


def test_get_question_strings_returns_in_order() -> None:
    meta = DocumentChunkMetadata(
        generated_questions=[
            GeneratedQuestion(id="q-1", question="first"),
            GeneratedQuestion(id="q-2", question="second"),
        ]
    )
    assert get_question_strings(meta) == ["first", "second"]
    assert get_question_strings(None) == []


# ── shared service-level test scaffolding ────────────────────────────


def _chunk(
    *,
    tenant_id: int = TENANT_ID,
    id: str | None = None,
    content: str = "The quick brown fox jumps over the lazy dog.",
    content_revision: int = 0,
    metadata: JsonObject | None = None,
    **overrides: object,
) -> Chunk:
    """Build a persisted-shape chunk row for seeding mocks / the DB."""
    return Chunk.model_validate(
        {
            "id": id or f"chunk-{uuid.uuid4().hex[:12]}",
            "tenant_id": tenant_id,
            "knowledge_base_id": "kb-1",
            "knowledge_id": "knowledge-1",
            "content": content,
            "chunk_index": 0,
            "is_enabled": True,
            "start_at": 0,
            "end_at": len(content),
            "chunk_type": "text",
            "flags": 1,
            "source_content": "",
            "content_revision": content_revision,
            "index_status": "ready",
            "last_editor_id": "",
            "metadata": metadata,
            "created_at": NOW,
            "updated_at": NOW,
            "deleted_at": None,
            **overrides,
        }
    )


def _question_metadata(*questions: dict[str, object]) -> JsonObject:
    """Build the stored ``generated_questions`` metadata payload."""
    return {"generated_questions": list(questions)}


class _Syncer:
    """Stub index syncer that records calls and can fail."""

    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.sync_calls: list[tuple[int, Chunk]] = []
        self.delete_calls: list[tuple[int, str]] = []

    async def sync_chunk(self, *, tenant_id: int, chunk: Chunk) -> None:
        self.sync_calls.append((tenant_id, chunk))
        if self.error is not None:
            raise self.error

    async def delete_question(self, *, tenant_id: int, source_id: str) -> None:
        self.delete_calls.append((tenant_id, source_id))
        if self.error is not None:
            raise self.error


def _mocked_question_service(chunk: Chunk) -> tuple[AsyncMock, list[Chunk]]:
    """Return a mocked chunk service plus the chunks passed to ``update_chunk``."""
    chunk_service = AsyncMock(spec=ChunkService)
    chunk_service.get_chunk_by_id.return_value = chunk
    persisted: list[Chunk] = []

    async def _update(*, chunk: Chunk) -> Chunk:
        persisted.append(chunk)
        return chunk

    chunk_service.update_chunk.side_effect = _update
    return chunk_service, persisted


# ── revert_document_chunk (service) ──────────────────────────────────


async def test_revert_replays_snapshot_through_edit_pipeline() -> None:
    snapshot = _revision(revision=0, content="original body")
    edited = _chunk(id=CHUNK_ID, content="original body", content_revision=1)
    revision_repo = AsyncMock(spec=ChunkRevisionRepository)
    revision_repo.get_chunk_revision.return_value = snapshot
    chunk_service = AsyncMock(spec=ChunkService)
    chunk_service.update_document_chunk.return_value = edited

    result = await revert_document_chunk(
        revision_repo=revision_repo,
        chunk_service=chunk_service,
        tenant_id=TENANT_ID,
        chunk_id=CHUNK_ID,
        revision=0,
        last_editor_id="editor-1",
    )

    assert result is edited
    revision_repo.get_chunk_revision.assert_awaited_once_with(
        tenant_id=TENANT_ID,
        chunk_id=CHUNK_ID,
        revision=0,
    )
    chunk_service.update_document_chunk.assert_awaited_once_with(
        tenant_id=TENANT_ID,
        chunk_id=CHUNK_ID,
        content="original body",
        is_enabled=True,
        expected_revision=None,
        last_editor_id="editor-1",
    )


async def test_revert_passes_explicit_expected_revision() -> None:
    snapshot = _revision(revision=1, content="second body")
    edited = _chunk(id=CHUNK_ID, content="second body", content_revision=3)
    revision_repo = AsyncMock(spec=ChunkRevisionRepository)
    revision_repo.get_chunk_revision.return_value = snapshot
    chunk_service = AsyncMock(spec=ChunkService)
    chunk_service.update_document_chunk.return_value = edited

    await revert_document_chunk(
        revision_repo=revision_repo,
        chunk_service=chunk_service,
        tenant_id=TENANT_ID,
        chunk_id=CHUNK_ID,
        revision=1,
        expected_revision=2,
        last_editor_id="editor-1",
    )

    chunk_service.update_document_chunk.assert_awaited_once_with(
        tenant_id=TENANT_ID,
        chunk_id=CHUNK_ID,
        content="second body",
        is_enabled=True,
        expected_revision=2,
        last_editor_id="editor-1",
    )


async def test_revert_raises_when_snapshot_absent() -> None:
    revision_repo = AsyncMock(spec=ChunkRevisionRepository)
    revision_repo.get_chunk_revision.return_value = None
    chunk_service = AsyncMock(spec=ChunkService)

    with pytest.raises(NotFoundError) as exc_info:
        await revert_document_chunk(
            revision_repo=revision_repo,
            chunk_service=chunk_service,
            tenant_id=TENANT_ID,
            chunk_id=CHUNK_ID,
            revision=99,
            last_editor_id="editor-1",
        )

    assert exc_info.value.code == "chunk.revision_not_found"
    chunk_service.update_document_chunk.assert_not_awaited()


async def test_revert_propagates_edit_conflict() -> None:
    snapshot = _revision(revision=0, content="original body")
    revision_repo = AsyncMock(spec=ChunkRevisionRepository)
    revision_repo.get_chunk_revision.return_value = snapshot
    chunk_service = AsyncMock(spec=ChunkService)
    chunk_service.update_document_chunk.side_effect = ConflictError(
        code="chunk.revision_conflict",
        message="chunk changed",
    )

    with pytest.raises(ConflictError) as exc_info:
        await revert_document_chunk(
            revision_repo=revision_repo,
            chunk_service=chunk_service,
            tenant_id=TENANT_ID,
            chunk_id=CHUNK_ID,
            revision=0,
            expected_revision=1,
            last_editor_id="editor-1",
        )

    assert exc_info.value.code == "chunk.revision_conflict"


# ── upsert_generated_question (service) ──────────────────────────────


async def test_upsert_appends_new_question_and_persists() -> None:
    chunk = _chunk(id=CHUNK_ID, content_revision=2)
    chunk_service, persisted = _mocked_question_service(chunk)

    bound = await upsert_generated_question(
        chunk_service=chunk_service,
        tenant_id=TENANT_ID,
        chunk_id=CHUNK_ID,
        question="  what is the answer?  ",
    )

    assert bound.question == "what is the answer?"
    assert bound.content_revision == 2
    meta = parse_document_metadata(persisted[0].metadata)
    assert meta is not None
    assert [(q.id, q.question, q.content_revision) for q in meta.generated_questions] == [
        (bound.id, "what is the answer?", 2)
    ]
    chunk_service.get_chunk_by_id.assert_awaited_once_with(tenant_id=TENANT_ID, id=CHUNK_ID)


async def test_upsert_updates_existing_question_text() -> None:
    chunk = _chunk(
        id=CHUNK_ID,
        content_revision=2,
        metadata=_question_metadata({"id": "q-1", "question": "old text", "content_revision": 1}),
    )
    chunk_service, persisted = _mocked_question_service(chunk)

    bound = await upsert_generated_question(
        chunk_service=chunk_service,
        tenant_id=TENANT_ID,
        chunk_id=CHUNK_ID,
        question_id="q-1",
        question="new text",
    )

    assert bound.id == "q-1"
    assert bound.question == "new text"
    assert bound.content_revision == 2
    meta = parse_document_metadata(persisted[0].metadata)
    assert meta is not None
    assert [(q.id, q.question, q.content_revision) for q in meta.generated_questions] == [
        ("q-1", "new text", 2)
    ]


async def test_upsert_rejects_blank_question_before_chunk_lookup() -> None:
    chunk_service = AsyncMock(spec=ChunkService)

    with pytest.raises(ValidationError) as exc_info:
        await upsert_generated_question(
            chunk_service=chunk_service,
            tenant_id=TENANT_ID,
            chunk_id=CHUNK_ID,
            question="   ",
        )

    assert exc_info.value.code == "chunk.question_empty"
    chunk_service.get_chunk_by_id.assert_not_awaited()


async def test_upsert_rejects_unknown_question_id() -> None:
    chunk = _chunk(id=CHUNK_ID)
    chunk_service, persisted = _mocked_question_service(chunk)

    with pytest.raises(ValidationError) as exc_info:
        await upsert_generated_question(
            chunk_service=chunk_service,
            tenant_id=TENANT_ID,
            chunk_id=CHUNK_ID,
            question_id="missing",
            question="text",
        )

    assert exc_info.value.code == "chunk.question_not_found"
    assert persisted == []


async def test_upsert_raises_not_found_for_missing_chunk() -> None:
    chunk_service = AsyncMock(spec=ChunkService)
    chunk_service.get_chunk_by_id.side_effect = NotFoundError(
        code="chunk.not_found",
        message=f"chunk {CHUNK_ID} not found",
    )

    with pytest.raises(NotFoundError) as exc_info:
        await upsert_generated_question(
            chunk_service=chunk_service,
            tenant_id=TENANT_ID,
            chunk_id=CHUNK_ID,
            question="text",
        )

    assert exc_info.value.code == "chunk.not_found"
    chunk_service.update_chunk.assert_not_awaited()


async def test_upsert_syncs_index_with_persisted_chunk() -> None:
    chunk = _chunk(id=CHUNK_ID, content_revision=1)
    chunk_service = AsyncMock(spec=ChunkService)
    chunk_service.get_chunk_by_id.return_value = chunk
    persisted = chunk.model_copy(update={"metadata": {"generated_questions": []}})
    chunk_service.update_chunk.return_value = persisted
    syncer = _Syncer()

    await upsert_generated_question(
        chunk_service=chunk_service,
        tenant_id=TENANT_ID,
        chunk_id=CHUNK_ID,
        question="text",
        syncer=syncer,
    )

    assert len(syncer.sync_calls) == 1
    assert syncer.sync_calls[0][0] == TENANT_ID
    assert syncer.sync_calls[0][1] is persisted


async def test_upsert_propagates_index_sync_failure() -> None:
    chunk = _chunk(id=CHUNK_ID)
    chunk_service = AsyncMock(spec=ChunkService)
    chunk_service.get_chunk_by_id.return_value = chunk
    chunk_service.update_chunk.return_value = chunk
    syncer = _Syncer(error=RuntimeError("vector store down"))

    with pytest.raises(RuntimeError):
        await upsert_generated_question(
            chunk_service=chunk_service,
            tenant_id=TENANT_ID,
            chunk_id=CHUNK_ID,
            question="text",
            syncer=syncer,
        )


# ── delete_generated_question (service) ──────────────────────────────


async def test_delete_removes_question_and_persists() -> None:
    chunk = _chunk(
        id=CHUNK_ID,
        metadata=_question_metadata(
            {"id": "q-1", "question": "a", "content_revision": 1},
            {"id": "q-2", "question": "b", "content_revision": 1},
        ),
    )
    chunk_service, persisted = _mocked_question_service(chunk)

    await delete_generated_question(
        chunk_service=chunk_service,
        tenant_id=TENANT_ID,
        chunk_id=CHUNK_ID,
        question_id="q-1",
    )

    meta = parse_document_metadata(persisted[0].metadata)
    assert meta is not None
    assert [q.id for q in meta.generated_questions] == ["q-2"]


async def test_delete_rejects_when_no_questions() -> None:
    chunk = _chunk(id=CHUNK_ID)
    chunk_service, persisted = _mocked_question_service(chunk)

    with pytest.raises(ValidationError) as exc_info:
        await delete_generated_question(
            chunk_service=chunk_service,
            tenant_id=TENANT_ID,
            chunk_id=CHUNK_ID,
            question_id="q-1",
        )

    assert exc_info.value.code == "chunk.no_questions"
    assert persisted == []


async def test_delete_rejects_unknown_question_id() -> None:
    chunk = _chunk(
        id=CHUNK_ID,
        metadata=_question_metadata({"id": "q-1", "question": "a", "content_revision": 1}),
    )
    chunk_service, persisted = _mocked_question_service(chunk)

    with pytest.raises(ValidationError) as exc_info:
        await delete_generated_question(
            chunk_service=chunk_service,
            tenant_id=TENANT_ID,
            chunk_id=CHUNK_ID,
            question_id="missing",
        )

    assert exc_info.value.code == "chunk.question_not_found"
    assert persisted == []


async def test_delete_raises_not_found_for_missing_chunk() -> None:
    chunk_service = AsyncMock(spec=ChunkService)
    chunk_service.get_chunk_by_id.side_effect = NotFoundError(
        code="chunk.not_found",
        message=f"chunk {CHUNK_ID} not found",
    )

    with pytest.raises(NotFoundError) as exc_info:
        await delete_generated_question(
            chunk_service=chunk_service,
            tenant_id=TENANT_ID,
            chunk_id=CHUNK_ID,
            question_id="q-1",
        )

    assert exc_info.value.code == "chunk.not_found"
    chunk_service.update_chunk.assert_not_awaited()


async def test_delete_drops_question_index_row() -> None:
    chunk = _chunk(
        id=CHUNK_ID,
        metadata=_question_metadata({"id": "q-1", "question": "a", "content_revision": 1}),
    )
    chunk_service = AsyncMock(spec=ChunkService)
    chunk_service.get_chunk_by_id.return_value = chunk
    chunk_service.update_chunk.return_value = chunk
    syncer = _Syncer()

    await delete_generated_question(
        chunk_service=chunk_service,
        tenant_id=TENANT_ID,
        chunk_id=CHUNK_ID,
        question_id="q-1",
        syncer=syncer,
    )

    assert syncer.delete_calls == [
        (TENANT_ID, generated_question_source_id(CHUNK_ID, "q-1"))
    ]
    chunk_service.update_chunk.assert_awaited_once()


async def test_delete_ignores_index_sync_failure() -> None:
    chunk = _chunk(
        id=CHUNK_ID,
        metadata=_question_metadata({"id": "q-1", "question": "a", "content_revision": 1}),
    )
    chunk_service = AsyncMock(spec=ChunkService)
    chunk_service.get_chunk_by_id.return_value = chunk
    chunk_service.update_chunk.return_value = chunk
    syncer = _Syncer(error=RuntimeError("vector store down"))

    await delete_generated_question(
        chunk_service=chunk_service,
        tenant_id=TENANT_ID,
        chunk_id=CHUNK_ID,
        question_id="q-1",
        syncer=syncer,
    )

    assert len(syncer.delete_calls) == 1
    chunk_service.update_chunk.assert_awaited_once()


# ── faker seeding (integration) ──────────────────────────────────────


@pytest.fixture(autouse=True)
def faker_seed() -> None:
    """Re-seed Faker per test for varied-but-reproducible generation."""
    Faker.seed(randint(1, _FAKER_SEED_MAX))


# ── Integration against the real schema ──────────────────────────────

# The ``chunks.tenant_id`` column is a 32-bit INTEGER, so integration
# rows need a 32-bit-safe unique id (the tenants table's 64-bit ids do
# not fit).
_tenant_counter = itertools.count(9_000_000)


def _tenant_id() -> int:
    """Return a unique 32-bit tenant id for the ``chunks`` table."""
    return next(_tenant_counter)


@pytest.fixture(scope="session")
def _db_engine() -> DatabaseEngine:
    """Session-scoped engine against the configured Postgres (NullPool)."""
    reset_settings_cache()
    settings = get_settings()
    return DatabaseEngine(url=settings.database_url, poolclass=NullPool)


@pytest_asyncio.fixture
async def db_session(_db_engine: DatabaseEngine) -> AsyncIterator[AsyncSession]:
    """Per-test session; skips the test when Postgres is unreachable."""
    factory = async_sessionmaker(_db_engine.engine, expire_on_commit=False)
    async with factory() as session:
        try:
            await session.execute(text("select 1"))
        except Exception:
            pytest.skip("integration Postgres is not reachable; set DATABASE_URL_OVERRIDE")
        yield session
        await session.rollback()


async def test_integration_revert_replays_snapshot(db_session: AsyncSession) -> None:
    tid = _tenant_id()
    chunk_service = ChunkService(chunk_repo=ChunkRepository(db_session))
    revision_repo = ChunkRevisionRepository(db_session)
    created = await chunk_service.create_chunks(
        chunks=[_chunk(tenant_id=tid, content="current body", content_revision=1)]
    )
    cid = created[0].id
    await revision_repo.create(
        ChunkRevision.model_validate(
            {
                "id": f"{cid}-rev-0",
                "tenant_id": tid,
                "knowledge_base_id": "kb-1",
                "knowledge_id": "knowledge-1",
                "chunk_id": cid,
                "revision": 0,
                "content": "historical body",
                "is_enabled": True,
                "editor_id": "editor-it-1",
                "edit_source": "user",
                "edited_at": NOW,
                "created_at": NOW,
            }
        )
    )
    await db_session.commit()

    reverted = await revert_document_chunk(
        revision_repo=revision_repo,
        chunk_service=chunk_service,
        tenant_id=tid,
        chunk_id=cid,
        revision=0,
        expected_revision=1,
        last_editor_id="editor-it-2",
    )

    assert reverted.content == "historical body"
    assert reverted.content_revision == 2
    assert reverted.index_status == "ready"
    reloaded = await ChunkRepository(db_session).get_by_id(tid, cid)
    assert reloaded.content == "historical body"
    assert reloaded.content_revision == 2


async def test_integration_revert_raises_when_snapshot_absent(
    db_session: AsyncSession,
) -> None:
    tid = _tenant_id()
    chunk_service = ChunkService(chunk_repo=ChunkRepository(db_session))
    created = await chunk_service.create_chunks(chunks=[_chunk(tenant_id=tid)])
    cid = created[0].id

    with pytest.raises(NotFoundError) as exc_info:
        await revert_document_chunk(
            revision_repo=ChunkRevisionRepository(db_session),
            chunk_service=chunk_service,
            tenant_id=tid,
            chunk_id=cid,
            revision=99,
            last_editor_id="editor-it-1",
        )

    assert exc_info.value.code == "chunk.revision_not_found"


async def test_integration_question_bind_unbind_round_trip(
    db_session: AsyncSession,
) -> None:
    tid = _tenant_id()
    chunk_service = ChunkService(chunk_repo=ChunkRepository(db_session))
    created = await chunk_service.create_chunks(
        chunks=[_chunk(tenant_id=tid, content_revision=2)]
    )
    cid = created[0].id

    bound = await upsert_generated_question(
        chunk_service=chunk_service,
        tenant_id=tid,
        chunk_id=cid,
        question=Faker().sentence(),
    )

    meta = parse_document_metadata((await ChunkRepository(db_session).get_by_id(tid, cid)).metadata)
    assert meta is not None
    assert [q.id for q in meta.generated_questions] == [bound.id]
    assert meta.generated_questions[0].content_revision == 2

    updated = await upsert_generated_question(
        chunk_service=chunk_service,
        tenant_id=tid,
        chunk_id=cid,
        question_id=bound.id,
        question="revised question",
    )
    assert updated.question == "revised question"

    await delete_generated_question(
        chunk_service=chunk_service,
        tenant_id=tid,
        chunk_id=cid,
        question_id=bound.id,
    )

    meta = parse_document_metadata((await ChunkRepository(db_session).get_by_id(tid, cid)).metadata)
    assert meta is None or meta.generated_questions == []


async def test_integration_question_ops_validate(db_session: AsyncSession) -> None:
    tid = _tenant_id()
    chunk_service = ChunkService(chunk_repo=ChunkRepository(db_session))
    missing = f"chunk-{uuid.uuid4().hex[:12]}"

    with pytest.raises(ValidationError) as exc_info:
        await upsert_generated_question(
            chunk_service=chunk_service,
            tenant_id=tid,
            chunk_id=missing,
            question="   ",
        )
    assert exc_info.value.code == "chunk.question_empty"

    with pytest.raises(NotFoundError) as exc_info:
        await upsert_generated_question(
            chunk_service=chunk_service,
            tenant_id=tid,
            chunk_id=missing,
            question="text",
        )
    assert exc_info.value.code == "chunk.not_found"

    created = await chunk_service.create_chunks(chunks=[_chunk(tenant_id=tid)])
    cid = created[0].id

    with pytest.raises(ValidationError) as exc_info:
        await delete_generated_question(
            chunk_service=chunk_service,
            tenant_id=tid,
            chunk_id=cid,
            question_id="q-1",
        )
    assert exc_info.value.code == "chunk.no_questions"

    await upsert_generated_question(
        chunk_service=chunk_service,
        tenant_id=tid,
        chunk_id=cid,
        question="a question",
    )

    with pytest.raises(ValidationError) as exc_info:
        await delete_generated_question(
            chunk_service=chunk_service,
            tenant_id=tid,
            chunk_id=cid,
            question_id="missing",
        )
    assert exc_info.value.code == "chunk.question_not_found"

    with pytest.raises(NotFoundError) as exc_info:
        await delete_generated_question(
            chunk_service=chunk_service,
            tenant_id=tid,
            chunk_id=missing,
            question_id="q-1",
        )
    assert exc_info.value.code == "chunk.not_found"
