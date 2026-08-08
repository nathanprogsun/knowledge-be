"""Wire-shape conversion for the knowledge document endpoints.

Projects the document domain outputs (``Knowledge``,
``KnowledgeMoveResponse``) onto the success envelopes consumed by the
document router. Request shapes the upstream contract file does not
carry — passage create, single-document clone, reparse override — are
declared here next to their endpoints.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src.common.json import JsonObject
from src.core.contracts.knowledge import Knowledge, KnowledgeMoveResponse


class KnowledgeEnvelope(BaseModel):
    """``{"success": true, "data": {...}}`` - single-document responses."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: Knowledge


class KnowledgeUpdatedEnvelope(BaseModel):
    """``{"success": true, "message": "...", "data": {...}}`` - update ack."""

    model_config = ConfigDict(frozen=True)

    success: bool
    message: str
    data: Knowledge


class KnowledgeListEnvelope(BaseModel):
    """Paged list response, matching the upstream list payload shape.

    ``total`` / ``page`` / ``page_size`` sit next to ``data`` rather than
    nested, mirroring the upstream list handler.
    """

    model_config = ConfigDict(frozen=True)

    success: bool
    total: int
    page: int
    page_size: int
    data: list[Knowledge]


class KnowledgeTaskEnvelope(BaseModel):
    """``{"success": true, "message": "...", "data": {...}}`` - lifecycle ack.

    Reparse and cancel-parse both answer with the refreshed document
    shape plus the upstream message.
    """

    model_config = ConfigDict(frozen=True)

    success: bool
    message: str
    data: Knowledge


class DeleteResult(BaseModel):
    """``{"deleted": true}`` - single-delete ack payload."""

    model_config = ConfigDict(frozen=True)

    deleted: bool


class DeleteEnvelope(BaseModel):
    """``{"success": true, "message": "...", "data": {"deleted": bool}}``."""

    model_config = ConfigDict(frozen=True)

    success: bool
    message: str
    data: DeleteResult


class MoveEnvelope(BaseModel):
    """``{"success": true, "data": {...}}`` - move submission response."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: KnowledgeMoveResponse


class ReparseRequest(BaseModel):
    """Optional reparse body: ``{"process_config": {...}}``."""

    model_config = ConfigDict(frozen=True)

    process_config: JsonObject | None = Field(default=None)


class CreatePassageKnowledgeRequest(BaseModel):
    """``{"passages": [...], "channel": "...", "sync": bool}`` - passage create."""

    model_config = ConfigDict(frozen=True)

    passages: list[str] = Field(min_length=1)
    channel: str | None = Field(default=None)
    sync: bool = False


class CloneKnowledgeRequest(BaseModel):
    """``{"target_kb_id": "..."}`` - single-document clone body."""

    model_config = ConfigDict(frozen=True)

    target_kb_id: str


__all__ = [
    "CloneKnowledgeRequest",
    "CreatePassageKnowledgeRequest",
    "DeleteEnvelope",
    "DeleteResult",
    "KnowledgeEnvelope",
    "KnowledgeListEnvelope",
    "KnowledgeTaskEnvelope",
    "KnowledgeUpdatedEnvelope",
    "MoveEnvelope",
    "ReparseRequest",
]
