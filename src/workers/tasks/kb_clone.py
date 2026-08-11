"""ARQ worker task: ``kb:clone``.

Maps the upstream knowledge-base-clone task: receives the serialized
clone payload over ARQ, validates it, and dispatches it to the core
clone composition :func:`src.core.knowledge.knowledge_bases.copy.copy_kb`.

The handler stays thin — payload parsing, logging, and result shaping
live here; the clone defenses (embedding-model, vector-store and
storage-instance match) and the settings copy live in the core layer.
A session-bound :class:`KBService` and the shared ``AsyncSession`` are
injected by the worker wiring layer (the worker context carries no
database engine). Until that wiring lands, the seam raises rather than
silently skipping a clone, so a miswired worker fails loudly.

Wire field names mirror the upstream contract so payloads enqueued by
the existing web/CLI paths deserialize without translation.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession

from src.app_logging import logger
from src.core.knowledge.knowledge_bases.copy import copy_kb
from src.core.knowledge.knowledge_bases.service.kb_service import KBService
from src.workers.base import WorkerContext
from src.workers.registry import JsonValue, register_task

# Upstream task name kept verbatim so legacy queue consumers see the
# same string they used to enqueue with.
TASK_NAME = "kb:clone"


class KBClonePayload(BaseModel):
    """ARQ-side payload for the ``kb:clone`` task.

    Mirrors the upstream ``KBClonePayload`` wire contract: the tenant,
    task, source and target ids are required. The upstream initiator and
    tracing-context fields are accepted via the JSON payload but not
    modelled on this surface (no consumer reads them yet), so
    ``extra="ignore"`` drops them without failing the parse.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    tenant_id: int
    task_id: str
    source_id: str
    target_id: str


@register_task(TASK_NAME)
async def task_kb_clone(
    ctx: WorkerContext,
    *,
    service: KBService | None = None,
    session: AsyncSession | None = None,
    **payload: JsonValue,
) -> dict[str, JsonValue]:
    """ARQ handler for the ``kb:clone`` task.

    Parses the ARQ payload into :class:`KBClonePayload` and delegates to
    :func:`process_kb_clone`. A session-bound ``service`` and the shared
    ``session`` may be injected by the worker wiring layer; when either
    is omitted the seam raises so a miswired clone fails loudly.
    ``ctx`` is currently unused — the worker context carries the
    ARQ-Redis pool, which the clone seam does not need at this stage.
    """
    parsed = KBClonePayload.model_validate(payload)
    logger.info(
        "kb:clone: tenant={} task={} source={} target={}",
        parsed.tenant_id,
        parsed.task_id,
        parsed.source_id,
        parsed.target_id,
    )
    return await process_kb_clone(
        tenant_id=parsed.tenant_id,
        task_id=parsed.task_id,
        source_kb_id=parsed.source_id,
        target_kb_id=parsed.target_id,
        service=service,
        session=session,
    )


async def process_kb_clone(
    *,
    tenant_id: int,
    task_id: str,
    source_kb_id: str,
    target_kb_id: str,
    service: KBService | None = None,
    session: AsyncSession | None = None,
) -> dict[str, JsonValue]:
    """Run one knowledge-base clone to completion via the core composition.

    Delegates to :func:`src.core.knowledge.knowledge_bases.copy.copy_kb`
    and returns a JSON-serialisable summary of the cloned pair. The
    clone defences (embedding model / vector store / storage instance
    match) are enforced by the core composition, whose errors propagate
    to the worker caller for ARQ retry.

    ``service`` and ``session`` are injected by the worker wiring layer
    — a :class:`KBService` bound to a per-job ``AsyncSession``. No
    service can be constructed here (the worker context carries no
    database engine), so an uninjected call raises ``NotImplementedError``
    instead of silently skipping the clone.
    """
    if service is None or session is None:
        raise NotImplementedError(
            "knowledge-base clone requires a session-bound KBService "
            "injected by the worker wiring layer",
        )
    source, target = await copy_kb(
        service=service,
        session=session,
        tenant_id=tenant_id,
        source_kb_id=source_kb_id,
        target_kb_id=target_kb_id,
    )
    return {
        "task_id": task_id,
        "source_id": source.id,
        "target_id": target.id,
        "status": "completed",
    }


def parse_payload(payload: Mapping[str, JsonValue]) -> KBClonePayload:
    """Validate a raw payload mapping into the task payload model.

    Exposed for tests and the wiring layer so the handler and its
    callers agree on the schema.
    """
    return KBClonePayload.model_validate(dict(payload))


__all__ = [
    "TASK_NAME",
    "KBClonePayload",
    "parse_payload",
    "process_kb_clone",
    "task_kb_clone",
]
