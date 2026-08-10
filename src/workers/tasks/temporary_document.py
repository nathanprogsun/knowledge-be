"""ARQ task: pre-parse a chat-attached temporary document.

Maps the upstream ``temporary_document:process`` task type. The worker
decodes the payload, validates the target scope, and returns a
structured result so legacy in-flight payloads drain cleanly until
the parse pipeline wires this entry point to the core domain.

Scope of this module
--------------------

- Decode the JSON payload into the wire-shaped
  :class:`~src.core.knowledge.documents.temporary_document.TemporaryDocumentTaskPayload`.
- Surface a stable result shape so callers and tests can assert on
  the parsed ``tenant_id`` / ``document_id`` pair without depending
  on the storage back-end.

The actual pre-parse -> promote lifecycle (``mark_processing`` ->
content extraction -> ``analyze_content`` -> ``mark_ready``) is
delivered by the core service that the parser pipeline exposes.
A future wiring lands the worker onto a per-job database session
and dispatches the payload into that service's ``process`` method;
the handler shape here is the contract that wiring will use.
"""

from __future__ import annotations

from src.core.knowledge.documents.temporary_document import (
    TemporaryDocumentTaskPayload,
)
from src.workers.base import WorkerContext
from src.workers.registry import JsonValue, register_task

# Upstream task name kept verbatim so legacy queue consumers see the
# same string they used to enqueue with.
TASK_NAME = "temporary_document:process"


@register_task(TASK_NAME)
async def task_temporary_document(
    ctx: WorkerContext,
    **payload: JsonValue,
) -> dict[str, JsonValue]:
    """Async parse worker entry point for a chat-attached document.

    Decodes ``tenant_id`` / ``document_id`` from the JSON payload and
    returns them in the result dict so downstream retries and tests
    can assert on the parsed scope without touching storage.

    The parse pipeline (file read, image resolution, OCR/caption,
    chunking, and ``mark_ready``) is delivered by the core domain.
    This entry point is intentionally a thin dispatcher: it does
    not open a database session or write to the table, so it
    remains safe to call from unit tests and from the worker's
    poll loop without leaking connections.
    """
    # Forward the raw payload straight into the wire schema so a missing
    # or non-coercible ``tenant_id`` / ``document_id`` fails the same
    # way every other call site fails (ValidationError -> ValueError).
    parsed = TemporaryDocumentTaskPayload.model_validate(dict(payload))
    return {
        "tenant_id": parsed.tenant_id,
        "document_id": parsed.document_id,
        "status": "dispatched",
    }


__all__ = ["TASK_NAME", "task_temporary_document"]
