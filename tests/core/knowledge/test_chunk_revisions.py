"""Unit tests for chunk revision + generated-question domain logic.

Covers the pure parts of ``src/core/knowledge/chunks``:

- ``ChunkRevision`` model defaults and the ``ChunkRevisionInfo``
  projection (``map_from_db``).
- the read-side queries (``list_chunk_revisions`` /
  ``get_chunk_revision``) against a mocked repository.
- the generated-question metadata helpers (bind/unbind, source id,
  currency, parse/serialize).

The DB-backed repository behavior is exercised separately in
``tests/integration/db/dao/test_chunk_revisions_repository.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from src.common.exception import NotFoundError, ValidationError
from src.core.knowledge.chunks.questions import (
    DocumentChunkMetadata,
    GeneratedQuestion,
    bind_generated_question,
    generated_question_source_id,
    get_question_strings,
    is_question_current,
    parse_document_metadata,
    unbind_generated_question,
)
from src.core.knowledge.chunks.revisions import (
    ChunkRevisionInfo,
    get_chunk_revision,
    list_chunk_revisions,
)
from src.db.dao.chunk_revision_repository import ChunkRevisionRepository
from src.db.models.chunk_revision import ChunkRevision

TENANT_ID = 7
CHUNK_ID = "chunk-1"
NOW = datetime(2026, 1, 1, tzinfo=UTC)


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
