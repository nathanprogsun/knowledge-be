"""ARQ worker task: ``knowledge:list_delete``.

Maps the upstream batch-knowledge-delete task: receives the serialized
payload over ARQ, validates it, and dispatches it to the core batch
delete domain
(:func:`src.core.knowledge.documents.list_delete.delete_knowledge_list`).

The handler stays thin: payload parsing, logging, and result shaping
live here; the soft-delete plus cascade chunk cleanup lives in the core
layer. Composing the core dependencies (repositories bound to a
per-job session) is the worker wiring layer's responsibility — it
supplies a :class:`KnowledgeListDeleteRunner` that closes over those
dependencies and invokes the core batch delete. Until the wiring lands,
the task refuses to run so a miswired worker fails loudly instead of
silently dropping real batch deletes.

Wire field names mirror the upstream contract so payloads enqueued by
the existing web/CLI paths deserialize without translation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from src.app_logging import logger
from src.workers.base import WorkerContext
from src.workers.registry import JsonValue, register_task

#: Upstream task type kept verbatim so legacy queue consumers see the
#: same string they used to enqueue with.
TASK_NAME = "knowledge:list_delete"


class KnowledgeListDeletePayload(BaseModel):
    """ARQ-side payload for the ``knowledge:list_delete`` task.

    Mirrors the upstream wire contract: the tenant id and the list of
    knowledge ids to delete are mandatory. The upstream tracing-context
    and initiator fields are accepted via the JSON payload but not
    modelled on this surface (no consumer reads them yet), so
    ``extra="ignore"`` drops them without failing the parse.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    tenant_id: int
    knowledge_ids: list[str]


class KnowledgeListDeleteRunner(Protocol):
    """Composition seam the worker wiring layer provides.

    Builds the core dependencies for one batch delete (repositories
    bound to the job session) and invokes
    :func:`src.core.knowledge.documents.list_delete.delete_knowledge_list`.
    """

    async def __call__(
        self,
        *,
        tenant_id: int,
        knowledge_ids: list[str],
    ) -> int:
        """Soft-delete the batch and return the number of rows removed."""


@register_task(TASK_NAME)
async def task_knowledge_list_delete(
    ctx: WorkerContext,
    *,
    runner: KnowledgeListDeleteRunner | None = None,
    **payload: JsonValue,
) -> dict[str, JsonValue]:
    """ARQ handler for the ``knowledge:list_delete`` task.

    Parses the ARQ payload into :class:`KnowledgeListDeletePayload`,
    logs the dispatch scope, delegates to
    :func:`process_knowledge_list_delete`, and returns a JSON-serialisable
    summary of the batch delete. A wiring-provided ``runner`` composes the
    core dependencies; without one the task refuses to run. The ``ctx``
    argument is currently unused — the worker context carries the
    ARQ-Redis pool, but the delete seam is invoked without database access
    at this stage.
    """
    parsed = KnowledgeListDeletePayload.model_validate(payload)
    logger.info(
        "knowledge:list_delete: tenant={} count={}",
        parsed.tenant_id,
        len(parsed.knowledge_ids),
    )
    return await process_knowledge_list_delete(
        tenant_id=parsed.tenant_id,
        knowledge_ids=parsed.knowledge_ids,
        runner=runner,
    )


async def process_knowledge_list_delete(
    *,
    tenant_id: int,
    knowledge_ids: list[str],
    runner: KnowledgeListDeleteRunner | None = None,
) -> dict[str, JsonValue]:
    """Dispatch a parsed batch-delete payload to the core domain.

    A ``runner`` supplied by the worker wiring layer composes the core
    dependencies and invokes
    :func:`src.core.knowledge.documents.list_delete.delete_knowledge_list`.
    Without a runner the task refuses to run so a miswired worker fails
    loudly instead of silently skipping batch deletes.
    """
    if runner is None:
        raise NotImplementedError(
            "knowledge:list_delete requires a wiring-provided runner; the "
            "worker wiring layer composes the core repositories.",
        )
    deleted = await runner(tenant_id=tenant_id, knowledge_ids=knowledge_ids)
    return {"deleted": deleted}


def parse_payload(payload: Mapping[str, JsonValue]) -> KnowledgeListDeletePayload:
    """Validate a raw payload mapping into the task payload model.

    Exposed for tests and the wiring layer so the handler and its
    callers agree on the schema.
    """
    return KnowledgeListDeletePayload.model_validate(dict(payload))


__all__ = [
    "TASK_NAME",
    "KnowledgeListDeletePayload",
    "KnowledgeListDeleteRunner",
    "parse_payload",
    "process_knowledge_list_delete",
    "task_knowledge_list_delete",
]
