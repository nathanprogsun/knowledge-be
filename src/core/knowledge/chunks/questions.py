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
row and syncs the retrieval index needs the current-chunk repository and
the retrieval engine from earlier waves; it is deferred here until those
dependencies are merged.
"""

from __future__ import annotations

import hashlib
import json
import uuid

from pydantic import BaseModel, ConfigDict, Field

from src.common.exception import ValidationError
from src.common.json import JsonObject

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


__all__ = [
    "DocumentChunkMetadata",
    "GeneratedQuestion",
    "bind_generated_question",
    "generated_question_source_id",
    "get_question_strings",
    "is_question_current",
    "parse_document_metadata",
    "unbind_generated_question",
]
