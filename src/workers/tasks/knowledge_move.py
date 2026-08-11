"""ARQ worker task: ``knowledge:move``.

Maps the upstream knowledge-move task: receives the serialized payload
over ARQ, validates it, and dispatches it to the core move domain
(:func:`src.core.knowledge.documents.move.move_knowledge`).

The handler stays thin: payload parsing, logging, and result shaping
live here; the compatibility gates and per-item re-home / re-ingest
orchestration live in the core layer. Composing the core dependencies
(repositories, KB / tag services, and the optional index-replication /
reparse seams bound to the job session) is the worker wiring layer's
responsibility — it supplies a :class:`KnowledgeMoveRunner` that closes
over those dependencies and invokes the core move for one item. Until
the wiring lands, the task refuses to run so a miswired worker fails
loudly instead of silently skipping real knowledge moves.

Batch semantics mirror the upstream handler: every item is attempted,
per-item failures are counted, and the task completes normally with the
processed / failed tallies in its result.

Wire field names mirror the upstream contract so payloads enqueued by
the existing web/CLI paths deserialize without translation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from src.app_logging import logger
from src.core.contracts.knowledge import Knowledge
from src.workers.base import WorkerContext
from src.workers.registry import JsonValue, register_task

#: Upstream task type kept verbatim so legacy queue consumers see the
#: same string they used to enqueue with.
TASK_NAME = "knowledge:move"


class KnowledgeMovePayload(BaseModel):
    """ARQ-side payload for the ``knowledge:move`` task.

    Mirrors the upstream wire contract: the tenant id, the source and
    target knowledge-base ids, the move mode (``reuse_vectors`` or
    ``reparse``), and the list of knowledge ids to move are mandatory.
    The ``task_id`` correlates the batch with its progress / audit trail
    and defaults to blank when omitted. The upstream tracing-context and
    initiator fields are accepted via the JSON payload but not modelled
    on this surface, so ``extra="ignore"`` drops them without failing
    the parse.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    tenant_id: int
    task_id: str = ""
    knowledge_ids: list[str]
    source_kb_id: str
    target_kb_id: str
    mode: str


class KnowledgeMoveRunner(Protocol):
    """Composition seam the worker wiring layer provides.

    Builds the core dependencies for one item move (repositories, KB /
    tag services, and the optional index-replication / reparse seams
    bound to the job session) and invokes
    :func:`src.core.knowledge.documents.move.move_knowledge`.
    """

    async def __call__(
        self,
        *,
        tenant_id: int,
        knowledge_id: str,
        source_kb_id: str,
        target_kb_id: str,
        mode: str,
    ) -> Knowledge:
        """Move one item; raise on failure."""


@register_task(TASK_NAME)
async def task_knowledge_move(
    ctx: WorkerContext,
    *,
    runner: KnowledgeMoveRunner | None = None,
    **payload: JsonValue,
) -> dict[str, JsonValue]:
    """ARQ handler for the ``knowledge:move`` task.

    Parses the ARQ payload into :class:`KnowledgeMovePayload`, logs the
    dispatch scope, delegates to :func:`process_knowledge_move`, and
    returns a JSON-serialisable summary of the batch move. A
    wiring-provided ``runner`` composes the core dependencies; without
    one the task refuses to run. The ``ctx`` argument is currently
    unused — the worker context carries the ARQ-Redis pool, but the move
    seam is invoked without database access at this stage.
    """
    parsed = KnowledgeMovePayload.model_validate(payload)
    logger.info(
        "knowledge:move: tenant={} task={} source={} target={} mode={} count={}",
        parsed.tenant_id,
        parsed.task_id,
        parsed.source_kb_id,
        parsed.target_kb_id,
        parsed.mode,
        len(parsed.knowledge_ids),
    )
    return await process_knowledge_move(
        tenant_id=parsed.tenant_id,
        knowledge_ids=parsed.knowledge_ids,
        source_kb_id=parsed.source_kb_id,
        target_kb_id=parsed.target_kb_id,
        mode=parsed.mode,
        runner=runner,
    )


async def process_knowledge_move(
    *,
    tenant_id: int,
    knowledge_ids: list[str],
    source_kb_id: str,
    target_kb_id: str,
    mode: str,
    runner: KnowledgeMoveRunner | None = None,
) -> dict[str, JsonValue]:
    """Dispatch a parsed move payload to the core domain.

    A ``runner`` supplied by the worker wiring layer composes the core
    dependencies and invokes
    :func:`src.core.knowledge.documents.move.move_knowledge` for one item
    at a time. Without a runner the task refuses to run so a miswired
    worker fails loudly instead of silently skipping knowledge moves.

    Every item is attempted; a per-item failure is logged and counted in
    the ``failed`` tally rather than aborting the batch, mirroring the
    upstream per-item move loop.
    """
    if runner is None:
        raise NotImplementedError(
            "knowledge:move requires a wiring-provided runner; the worker "
            "wiring layer composes the core dependencies.",
        )
    processed = 0
    failed = 0
    for knowledge_id in knowledge_ids:
        try:
            await runner(
                tenant_id=tenant_id,
                knowledge_id=knowledge_id,
                source_kb_id=source_kb_id,
                target_kb_id=target_kb_id,
                mode=mode,
            )
            processed += 1
        except Exception:
            logger.exception(
                "knowledge:move: tenant={} knowledge={} move failed; "
                "continuing with the rest of the batch",
                tenant_id,
                knowledge_id,
            )
            failed += 1
    logger.info(
        "knowledge:move: tenant={} finished, processed={} failed={}",
        tenant_id,
        processed,
        failed,
    )
    return {"processed": processed, "failed": failed}


def parse_payload(payload: Mapping[str, JsonValue]) -> KnowledgeMovePayload:
    """Validate a raw payload mapping into the task payload model.

    Exposed for tests and the wiring layer so the handler and its
    callers agree on the schema.
    """
    return KnowledgeMovePayload.model_validate(dict(payload))


__all__ = [
    "TASK_NAME",
    "KnowledgeMovePayload",
    "KnowledgeMoveRunner",
    "parse_payload",
    "process_knowledge_move",
    "task_knowledge_move",
]
