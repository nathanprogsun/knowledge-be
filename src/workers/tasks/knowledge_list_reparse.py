"""ARQ worker task: ``knowledge:list_reparse``.

Maps the upstream batch-knowledge-reparse task: receives the serialized
payload over ARQ, validates it, and dispatches it to the core reparse
domain (:func:`src.core.knowledge.documents.reparse.reparse_knowledge`).

The handler stays thin: payload parsing, logging, and result shaping
live here; the per-item reset / re-submit orchestration lives in the
core layer. Composing the core dependencies (repositories, KB / chunk
services, and the parse enqueuer bound to the job session) is the
worker wiring layer's responsibility — it supplies a
:class:`KnowledgeReparseRunner` that closes over those dependencies and
invokes the core reparse for one item. Until the wiring lands, the task
refuses to run so a miswired worker fails loudly instead of silently
skipping batch reparses.

Batch semantics mirror the upstream handler: every item is attempted so
one bad document cannot block the remainder of the batch, and a partial
failure is intentionally not retried as a whole (retrying would
destructively reparse items that were already submitted successfully).
Failed rows stay selectable for an explicit retry; the submitted /
failed counts are surfaced in the task result.

Wire field names mirror the upstream contract so payloads enqueued by
the existing web/CLI paths deserialize without translation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from src.app_logging import logger
from src.common.json import JsonObject
from src.core.contracts.knowledge import Knowledge
from src.workers.base import WorkerContext
from src.workers.registry import JsonValue, register_task

#: Upstream task type kept verbatim so legacy queue consumers see the
#: same string they used to enqueue with.
TASK_NAME = "knowledge:list_reparse"


class KnowledgeListReparsePayload(BaseModel):
    """ARQ-side payload for the ``knowledge:list_reparse`` task.

    Mirrors the upstream wire contract: the tenant id and the list of
    knowledge ids to reparse are mandatory; the parse-config override
    rides under ``process_config`` (the upstream JSON field name) and
    is forwarded to the core reparse seam as ``process_overrides``. The
    upstream tracing-context and initiator fields are accepted via the
    JSON payload but not modelled on this surface, so ``extra="ignore"``
    drops them without failing the parse.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    tenant_id: int
    knowledge_ids: list[str]
    process_config: JsonObject | None = None


class KnowledgeReparseRunner(Protocol):
    """Composition seam the worker wiring layer provides.

    Builds the core dependencies for one item reparse (repositories,
    KB / chunk services, and the parse enqueuer bound to the job session)
    and invokes
    :func:`src.core.knowledge.documents.reparse.reparse_knowledge`.
    """

    async def __call__(
        self,
        *,
        tenant_id: int,
        knowledge_id: str,
        process_overrides: JsonObject | None,
    ) -> Knowledge:
        """Reset one item and submit its reparse; raise on failure."""


@register_task(TASK_NAME)
async def task_knowledge_list_reparse(
    ctx: WorkerContext,
    *,
    runner: KnowledgeReparseRunner | None = None,
    **payload: JsonValue,
) -> dict[str, JsonValue]:
    """ARQ handler for the ``knowledge:list_reparse`` task.

    Parses the ARQ payload into :class:`KnowledgeListReparsePayload`,
    logs the dispatch scope, delegates to
    :func:`process_knowledge_list_reparse`, and returns a JSON-serialisable
    summary of the batch reparse. A wiring-provided ``runner`` composes
    the core dependencies; without one the task refuses to run. The
    ``ctx`` argument is currently unused — the worker context carries the
    ARQ-Redis pool, but the reparse seam is invoked without database
    access at this stage.
    """
    parsed = KnowledgeListReparsePayload.model_validate(payload)
    logger.info(
        "knowledge:list_reparse: tenant={} count={} process_config={}",
        parsed.tenant_id,
        len(parsed.knowledge_ids),
        parsed.process_config is not None,
    )
    return await process_knowledge_list_reparse(
        tenant_id=parsed.tenant_id,
        knowledge_ids=parsed.knowledge_ids,
        process_overrides=parsed.process_config,
        runner=runner,
    )


async def process_knowledge_list_reparse(
    *,
    tenant_id: int,
    knowledge_ids: list[str],
    process_overrides: JsonObject | None = None,
    runner: KnowledgeReparseRunner | None = None,
) -> dict[str, JsonValue]:
    """Dispatch a parsed batch-reparse payload to the core domain.

    A ``runner`` supplied by the worker wiring layer composes the core
    dependencies and invokes
    :func:`src.core.knowledge.documents.reparse.reparse_knowledge` for one
    item at a time. Without a runner the task refuses to run so a
    miswired worker fails loudly instead of silently skipping batch
    reparses.

    Every item is attempted; a per-item failure is logged and counted in
    the ``failed`` tally rather than aborting the batch, and the whole
    task completes normally so a partial failure is never retried as a
    destructive full-batch reparse.
    """
    if runner is None:
        raise NotImplementedError(
            "knowledge:list_reparse requires a wiring-provided runner; the "
            "worker wiring layer composes the core dependencies.",
        )
    submitted = 0
    failed = 0
    for knowledge_id in knowledge_ids:
        try:
            await runner(
                tenant_id=tenant_id,
                knowledge_id=knowledge_id,
                process_overrides=process_overrides,
            )
            submitted += 1
        except Exception:
            logger.exception(
                "knowledge:list_reparse: tenant={} knowledge={} reparse failed; "
                "continuing with the rest of the batch",
                tenant_id,
                knowledge_id,
            )
            failed += 1
    logger.info(
        "knowledge:list_reparse: tenant={} finished, submitted={} failed={}",
        tenant_id,
        submitted,
        failed,
    )
    return {"submitted": submitted, "failed": failed}


def parse_payload(payload: Mapping[str, JsonValue]) -> KnowledgeListReparsePayload:
    """Validate a raw payload mapping into the task payload model.

    Exposed for tests and the wiring layer so the handler and its
    callers agree on the schema.
    """
    return KnowledgeListReparsePayload.model_validate(dict(payload))


__all__ = [
    "TASK_NAME",
    "KnowledgeListReparsePayload",
    "KnowledgeReparseRunner",
    "parse_payload",
    "process_knowledge_list_reparse",
    "task_knowledge_list_reparse",
]
