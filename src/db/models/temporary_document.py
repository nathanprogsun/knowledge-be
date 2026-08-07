"""Storage row for the `temporary_documents` table.

Session-scoped, expiring documents uploaded for chat turns. Parsed
artifacts (``content`` / ``chunks`` / ``image_refs``) are retained
separately from the source file so a later turn can select only the
useful parts without re-parsing the upload.

The row mirrors the upstream entity column-for-column:

- ``id`` is caller-assigned (a UUID minted by the service), so it
  participates in INSERT.
- ``chunks`` / ``image_refs`` are JSON arrays; ``metadata`` /
  ``processing_options`` are JSON objects. All four are JSONB and
  ``not null`` with server defaults, matching the wire defaults the
  entity applies before create.
- ``deleted_at`` is the soft-delete marker (expiring documents are
  cleaned up by a sweep, not hard-deleted on the hot path).

``status`` lifecycle: ``uploaded`` -> ``processing`` -> ``ready`` /
``failed``.
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from pydantic import Field

from src.common.json import JsonObject
from src.common.table_model import TableModel

# ── Lifecycle statuses ─────────────────────────────────────────────────

TEMPORARY_DOCUMENT_STATUS_UPLOADED = "uploaded"
TEMPORARY_DOCUMENT_STATUS_PROCESSING = "processing"
TEMPORARY_DOCUMENT_STATUS_READY = "ready"
TEMPORARY_DOCUMENT_STATUS_FAILED = "failed"

# Maximum number of pre-uploaded temporary attachment ids a single chat
# turn may reference.
MAX_TEMPORARY_ATTACHMENTS_PER_MESSAGE = 5


class TemporaryDocument(TableModel):
    """One row of the `temporary_documents` table."""

    table: ClassVar[str] = "temporary_documents"
    primary_keys: ClassVar[tuple[str, ...]] = ("id",)
    json_columns: ClassVar[tuple[str, ...]] = (
        "chunks",
        "image_refs",
        "metadata",
        "processing_options",
    )
    # ``id`` is application-assigned (UUID), so it participates in INSERT.
    db_generated_columns: ClassVar[tuple[str, ...]] = ()

    id: str
    tenant_id: int
    session_id: str
    resource_ref: str
    file_name: str
    file_type: str
    mime_type: str = ""
    file_size: int
    status: str = TEMPORARY_DOCUMENT_STATUS_UPLOADED
    content: str | None = None
    chunks: list[JsonObject] = Field(default_factory=list)
    image_refs: list[JsonObject] = Field(default_factory=list)
    metadata: JsonObject = Field(default_factory=dict)
    processing_options: JsonObject = Field(default_factory=dict)
    token_count: int = 0
    chunk_count: int = 0
    error_message: str | None = None
    expires_at: datetime
    started_at: datetime | None = None
    ready_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None


__all__ = [
    "MAX_TEMPORARY_ATTACHMENTS_PER_MESSAGE",
    "TEMPORARY_DOCUMENT_STATUS_FAILED",
    "TEMPORARY_DOCUMENT_STATUS_PROCESSING",
    "TEMPORARY_DOCUMENT_STATUS_READY",
    "TEMPORARY_DOCUMENT_STATUS_UPLOADED",
    "TemporaryDocument",
]
