"""Generated retrieval questions — domain types and metadata helpers.

Maps the question-binding half of the upstream chunk service
(``UpsertGeneratedQuestion`` / ``DeleteGeneratedQuestion``). Generated
questions live in the chunk's document metadata (``generated_questions``
array) and are tied to the chunk's ``content_revision`` so the UI can
mark them stale after an edit.

This module contains the self-contained parts:

- ``GeneratedQuestion`` / ``DocumentChunkMetadata`` types.
- ``generated_question_source_id`` — the retrieval ``source_id``.
- ``bind_generated_question`` / ``unbind_generated_question`` — the
  immutable metadata-level upsert / delete.
- ``is_question_current`` / ``get_question_strings`` and the
  metadata parse/dump helpers.

The orchestration that persists the updated metadata to the ``chunks``
row (``upsert_generated_question`` / ``delete_generated_question``) sits
on top of those helpers. It depends on the merged chunk service and an
injectable retrieval-index hook (``GeneratedQuestionIndexSyncer``); the
hook is ``None`` until the retrieval engine is wired, at which point the
web layer supplies a concrete syncer.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from contextlib import suppress
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from src.common.exception import ValidationError
from src.common.json import JsonObject
from src.core.knowledge.chunks.service.chunk_service import ChunkService
from src.db.models.chunk import Chunk

# The retrieval ``source_id`` column is varchar(64). A chunk UUID plus a
# question UUID would be 73 bytes; short identifiers are kept verbatim and
# oversized question ids are hashed so existing index rows remain
# addressable by delete/reindex operations.
_MAX_SOURCE_ID_LENGTH = 64


class GeneratedQuestion(BaseModel):
    """One AI-generated retrieval question for a chunk."""

    model_config = ConfigDict(frozen=True)

    id: str
    question: str
    content_revision: int | None = Field(default=None)


class DocumentChunkMetadata(BaseModel):
    """Chunk-level document metadata (``generated_questions`` payload)."""

    model_config = ConfigDict(frozen=True)

    generated_questions: list[GeneratedQuestion] = Field(default_factory=list)
    generated_questions_revision: int = 0

    def with_question(self, question: GeneratedQuestion) -> DocumentChunkMetadata:
        """Return a copy with ``question`` replaced in place (new object)."""
        questions = [
            question if q.id == question.id else q for q in self.generated_questions
        ]
        return self.model_copy(update={"generated_questions": questions})

    def without_question(self, question_id: str) -> DocumentChunkMetadata:
        """Return a copy with ``question_id`` removed (new object)."""
        return self.model_copy(
            update={
                "generated_questions": [
                    q for q in self.generated_questions if q.id != question_id
                ]
            }
        )

    def to_json(self) -> JsonObject:
        """Serialize to the JSON shape stored in the chunk metadata column.

        ``exclude_defaults`` mirrors the upstream ``omitempty`` tags: an
        empty question list, an unset per-question revision, and a zero
        metadata revision are omitted from the persisted payload.
        """
        return self.model_dump(mode="json", exclude_defaults=True)


def generated_question_source_id(chunk_id: str, question_id: str) -> str:
    """Build the retrieval ``source_id`` for a generated question.

    Returns ``{chunk_id}-{question_id}`` while it fits the column;
    otherwise ``{chunk_id}-q{12-byte-sha256}`` (24 hex characters), which
    keeps the identifier at 62 bytes for a UUID chunk id.
    """
    candidate = f"{chunk_id}-{question_id}"
    if len(candidate) <= _MAX_SOURCE_ID_LENGTH:
        return candidate
    digest = hashlib.sha256(question_id.encode("utf-8")).digest()
    return f"{chunk_id}-q{digest[:12].hex()}"


def parse_document_metadata(raw: JsonObject | str | None) -> DocumentChunkMetadata | None:
    """Decode the chunk metadata column; ``None`` for absent / empty values."""
    if not raw:
        return None
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return DocumentChunkMetadata.model_validate(decoded)
    return DocumentChunkMetadata.model_validate(raw)


def bind_generated_question(
    metadata: DocumentChunkMetadata,
    *,
    question_id: str | None,
    question: str,
    content_revision: int,
) -> tuple[DocumentChunkMetadata, GeneratedQuestion]:
    """Upsert one generated question into the chunk metadata (new objects).

    ``question_id`` of ``None`` (or empty) appends a fresh question with
    a new id; a known id replaces the stored question text. Both paths
    stamp ``content_revision`` so the question is tied to the current
    body. Raises ``ValidationError`` for empty text or an unknown id,
    mirroring the upstream bad-request paths.
    """
    question = question.strip()
    if question == "":
        raise ValidationError(
            code="chunk.question_empty",
            message="question cannot be empty",
        )
    if question_id:
        stored = next(
            (q for q in metadata.generated_questions if q.id == question_id),
            None,
        )
        if stored is None:
            raise ValidationError(
                code="chunk.question_not_found",
                message=f"question not found: {question_id}",
            )
        bound = stored.model_copy(
            update={"question": question, "content_revision": content_revision}
        )
        return metadata.with_question(bound), bound
    bound = GeneratedQuestion(
        id=str(uuid.uuid4()),
        question=question,
        content_revision=content_revision,
    )
    updated = metadata.model_copy(
        update={"generated_questions": [*metadata.generated_questions, bound]}
    )
    return updated, bound


def unbind_generated_question(
    metadata: DocumentChunkMetadata,
    *,
    question_id: str,
) -> tuple[DocumentChunkMetadata, GeneratedQuestion]:
    """Remove one generated question from the chunk metadata (new objects).

    Raises ``ValidationError`` when the metadata carries no questions or
    the id is unknown, mirroring the upstream delete path.
    """
    if not metadata.generated_questions:
        raise ValidationError(
            code="chunk.no_questions",
            message="no generated questions found for chunk",
        )
    removed = next(
        (q for q in metadata.generated_questions if q.id == question_id),
        None,
    )
    if removed is None:
        raise ValidationError(
            code="chunk.question_not_found",
            message=f"question not found: {question_id}",
        )
    return metadata.without_question(question_id), removed


def is_question_current(
    metadata: DocumentChunkMetadata | None,
    question: GeneratedQuestion,
    chunk_revision: int,
) -> bool:
    """Report whether a question was authored for the current chunk body.

    Questions with a per-question revision compare against it directly;
    legacy rows fall back to the metadata-level revision.
    """
    if question.content_revision is not None:
        return question.content_revision == chunk_revision
    return metadata is not None and metadata.generated_questions_revision == chunk_revision


def get_question_strings(metadata: DocumentChunkMetadata | None) -> list[str]:
    """Return the question texts of the metadata, in stored order."""
    if metadata is None:
        return []
    return [q.question for q in metadata.generated_questions]


class GeneratedQuestionIndexSyncer(Protocol):
    """Optional retrieval-index hook for generated-question mutations.

    ``sync_chunk`` re-indexes a chunk's content and generated questions
    after a bind; ``delete_question`` drops the vector row for one
    question's ``source_id`` after an unbind. The hook is injectable so
    this module stays independent of the retrieval engine, which lands in
    a later wave.
    """

    async def sync_chunk(self, *, tenant_id: int, chunk: Chunk) -> None:
        """Re-index ``chunk``'s content and generated questions."""

    async def delete_question(self, *, tenant_id: int, source_id: str) -> None:
        """Drop one question vector row; callers treat absence as success."""


async def upsert_generated_question(
    *,
    chunk_service: ChunkService,
    tenant_id: int,
    chunk_id: str,
    question: str,
    question_id: str | None = None,
    syncer: GeneratedQuestionIndexSyncer | None = None,
) -> GeneratedQuestion:
    """Bind one generated question to a chunk and persist the metadata.

    Loads the chunk tenant-scoped, upserts the question into its
    ``generated_questions`` metadata (stamped with the current content
    revision), persists the row, and re-indexes through the optional
    ``syncer``. Blank questions are rejected before the chunk lookup,
    mirroring the upstream validation order; an unknown ``question_id``
    raises ``ValidationError``.
    """
    if not question.strip():
        raise ValidationError(
            code="chunk.question_empty",
            message="question cannot be empty",
        )
    chunk = await chunk_service.get_chunk_by_id(tenant_id=tenant_id, id=chunk_id)
    metadata = parse_document_metadata(chunk.metadata)
    if metadata is None:
        metadata = DocumentChunkMetadata()
    updated, bound = bind_generated_question(
        metadata,
        question_id=question_id,
        question=question,
        content_revision=chunk.content_revision,
    )
    persisted = await chunk_service.update_chunk(
        chunk=chunk.model_copy(update={"metadata": updated.to_json()}),
    )
    if syncer is not None:
        await syncer.sync_chunk(tenant_id=tenant_id, chunk=persisted)
    return bound


async def delete_generated_question(
    *,
    chunk_service: ChunkService,
    tenant_id: int,
    chunk_id: str,
    question_id: str,
    syncer: GeneratedQuestionIndexSyncer | None = None,
) -> None:
    """Unbind one generated question from a chunk and persist the metadata.

    Loads the chunk tenant-scoped, removes the question from its
    metadata (``ValidationError`` when the metadata carries no questions
    or the id is unknown), drops the question's retrieval-index row as a
    best-effort (a missing vector row is not an error, mirroring the
    upstream delete path), and persists the row.
    """
    chunk = await chunk_service.get_chunk_by_id(tenant_id=tenant_id, id=chunk_id)
    metadata = parse_document_metadata(chunk.metadata)
    if metadata is None:
        metadata = DocumentChunkMetadata()
    updated, _removed = unbind_generated_question(metadata, question_id=question_id)
    if syncer is not None:
        # Best-effort index cleanup: the vector row may never have been
        # indexed, so a failure must not block the metadata removal
        # (upstream logs and continues).
        with suppress(Exception):
            await syncer.delete_question(
                tenant_id=tenant_id,
                source_id=generated_question_source_id(chunk_id, question_id),
            )
    await chunk_service.update_chunk(
        chunk=chunk.model_copy(update={"metadata": updated.to_json()}),
    )


__all__ = [
    "DocumentChunkMetadata",
    "GeneratedQuestion",
    "GeneratedQuestionIndexSyncer",
    "bind_generated_question",
    "delete_generated_question",
    "generated_question_source_id",
    "get_question_strings",
    "is_question_current",
    "parse_document_metadata",
    "unbind_generated_question",
    "upsert_generated_question",
]
