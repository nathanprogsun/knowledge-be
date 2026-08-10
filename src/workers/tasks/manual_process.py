"""Worker task for manual Markdown knowledge processing.

Maps the upstream ``manual_process`` task type. The worker pulls the
manual-content payload, validates it against the persisted task
contract, and delegates to the core manual-processing seam. The
``manual_process`` task fires for both the publish (create) and update
flows of manual Markdown knowledge; the upstream behaviour controls
whether stale indexes/chunks are reaped before the new run via the
``need_cleanup`` flag.

The worker sticks to payload parsing and dispatch — the actual parse /
chunk / embed / index orchestration lives in the core layer. The core
seam is exposed as :func:`process_document_manual` so callers (tests,
future composition) can override the dispatch without touching the
worker surface.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from src.workers.base import WorkerContext
from src.workers.registry import JsonValue, register_task


class ManualProcessPayload(BaseModel):
    """Async manual-knowledge parse task payload.

    Mirrors the upstream ``ManualProcessPayload`` shape: tenant and
    knowledge identifiers, the cleaned Markdown content that drives
    chunking, and a ``need_cleanup`` flag the upstream service uses to
    reap old indexes / chunks before a re-index on update flows. A
    ``request_id`` is carried for log correlation; the upstream tracing
    context fields are accepted via the JSON payload but not modelled
    on this surface (no consumer reads them yet).
    """

    model_config = ConfigDict(frozen=True)

    request_id: str = ""
    tenant_id: int
    knowledge_id: str
    knowledge_base_id: str
    content: str
    need_cleanup: bool = False


async def process_document_manual(
    *,
    tenant_id: int,
    knowledge_id: str,
    knowledge_base_id: str,
    content: str,
    need_cleanup: bool,
    request_id: str,
) -> dict[str, JsonValue]:
    """Run manual-document processing for one knowledge entry.

    Core implementation lands in a later wave; this entry point stays
    so the worker task has a stable seam to call. Tests swap it for an
    ``AsyncMock`` to exercise the dispatch contract without the full
    pipeline.
    """
    raise NotImplementedError(
        "Manual document processing lands in a later wave.",
    )


@register_task("manual_process")
async def manual_process(
    ctx: WorkerContext,
    **payload: JsonValue,
) -> dict[str, JsonValue]:
    """Worker handler for the ``manual_process`` task type.

    Parses the JSON payload into :class:`ManualProcessPayload` and
    delegates to :func:`process_document_manual`. The ``ctx`` argument
    is currently unused — the worker context carries the ARQ-Redis
    pool, but the manual-processing seam is invoked without database
    access at this stage.
    """
    parsed = ManualProcessPayload.model_validate(payload)
    return await process_document_manual(
        tenant_id=parsed.tenant_id,
        knowledge_id=parsed.knowledge_id,
        knowledge_base_id=parsed.knowledge_base_id,
        content=parsed.content,
        need_cleanup=parsed.need_cleanup,
        request_id=parsed.request_id,
    )


__all__ = [
    "ManualProcessPayload",
    "manual_process",
    "process_document_manual",
]
