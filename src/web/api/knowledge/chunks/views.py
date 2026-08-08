"""Wire-shape models for the chunk endpoints.

Defines the request bodies and response envelopes the chunk router
exchanges. The chunk payloads are the frozen ``Chunk`` contract from
``src.core.contracts.knowledge``, projected from the storage rows by
``chunk_to_contract``; the response field set matches the documented
API response (retrieval bookkeeping columns stay off the wire).

The request bodies mirror the handler-local shapes on the upstream side:
the update body carries ``content`` / ``is_enabled`` / ``expected_revision``
and the revert body carries a required ``revision`` plus an optional
``expected_revision``. The question bodies are the ``question_id`` /
``question`` pair.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from src.core.contracts.knowledge import Chunk
from src.core.knowledge.chunks.questions import GeneratedQuestion
from src.core.knowledge.chunks.revisions import ChunkRevisionInfo


class UpdateChunkRequest(BaseModel):
    """Body for ``PUT /chunks/{knowledge_id}/{id}`` - every field optional.

    ``expected_revision`` is the optimistic-lock guard: when present it
    must equal the chunk's current ``content_revision`` or the write
    fails with a conflict; when absent the client-side staleness check is
    skipped (the write's WHERE guard still rejects a concurrent edit).
    """

    model_config = ConfigDict(frozen=True)

    content: str | None = None
    is_enabled: bool | None = None
    expected_revision: int | None = None


class RevertChunkRequest(BaseModel):
    """Body for ``POST /chunks/{knowledge_id}/{id}/revert``."""

    model_config = ConfigDict(frozen=True)

    revision: int
    expected_revision: int | None = None


class UpsertGeneratedQuestionRequest(BaseModel):
    """Body for ``PUT /chunks/by-id/{id}/questions``.

    An empty ``question_id`` appends a fresh question; a known id replaces
    the stored text. ``question`` is required.
    """

    model_config = ConfigDict(frozen=True)

    question_id: str = ""
    question: str


class DeleteGeneratedQuestionRequest(BaseModel):
    """Body for ``DELETE /chunks/by-id/{id}/questions``."""

    model_config = ConfigDict(frozen=True)

    question_id: str


class ChunkEnvelope(BaseModel):
    """``{"success": true, "data": {...}}`` - single-chunk responses."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: Chunk


class ChunkListEnvelope(BaseModel):
    """``{"success": true, "data": [...], "total", "page", "page_size"}``."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: list[Chunk]
    total: int
    page: int
    page_size: int


class ChunkMessageEnvelope(BaseModel):
    """``{"success": true, "message": "..."}`` - deletion acknowledgements."""

    model_config = ConfigDict(frozen=True)

    success: bool
    message: str


class ChunkRevisionsEnvelope(BaseModel):
    """``{"success": true, "data": [...]}`` - revision history, newest first."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: list[ChunkRevisionInfo]


class GeneratedQuestionEnvelope(BaseModel):
    """``{"success": true, "data": {...}}`` - one upserted question."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: GeneratedQuestion


class GeneratedQuestionsEnvelope(BaseModel):
    """``{"success": true, "data": [...]}`` - a question list response."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: list[GeneratedQuestion]


__all__ = [
    "ChunkEnvelope",
    "ChunkListEnvelope",
    "ChunkMessageEnvelope",
    "ChunkRevisionsEnvelope",
    "DeleteGeneratedQuestionRequest",
    "GeneratedQuestionEnvelope",
    "GeneratedQuestionsEnvelope",
    "RevertChunkRequest",
    "UpdateChunkRequest",
    "UpsertGeneratedQuestionRequest",
]
