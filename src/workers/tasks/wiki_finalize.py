"""ARQ worker task: ``wiki:finalize``.

Maps the upstream wiki-finalize task: the debounced, knowledge-base-wide
convergence pass that runs after a burst of ingest batches — index-intro
rebuild, dead-link cleanup, cross-link injection, and empty-folder
pruning. It carries the same per-KB trigger payload as ``wiki:ingest``
(tenant id, knowledge-base id, language); the affected pages are
discovered from the durable finalize lane at run time, not from the
payload.

The worker sticks to payload parsing and dispatch. The convergence pass
itself is delivered by the core wiki layer in a later wave; this entry
point keeps the seam so the handler contract stays stable, and raises
rather than silently dropping a finalize so a miswired worker fails
loudly.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from src.app_logging import logger
from src.workers.base import WorkerContext
from src.workers.registry import JsonValue, register_task


class WikiFinalizePayload(BaseModel):
    """ARQ-side payload for the ``wiki:finalize`` trigger task.

    Mirrors the upstream wire contract: only the tenant and knowledge-base
    id are mandatory; ``language`` is optional, matching the ``omitempty``
    JSON tag on the upstream side. The affected slugs and change
    description are persisted in the durable finalize lane and resolved by
    the core layer at run time. The upstream tracing-context fields are
    accepted via the JSON payload but not modelled here, so
    ``extra="ignore"`` drops them without failing the parse.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    tenant_id: int
    knowledge_base_id: str
    language: str = ""


@register_task("wiki:finalize")
async def task_wiki_finalize(
    ctx: WorkerContext,
    **payload: JsonValue,
) -> dict[str, JsonValue]:
    """ARQ handler for the ``wiki:finalize`` task.

    Parses the ARQ payload into :class:`WikiFinalizePayload` and delegates
    to :func:`process_wiki_finalize`. ``ctx`` is currently unused — the
    worker context carries the ARQ-Redis pool, which the finalize seam
    does not need at this stage.
    """
    parsed = WikiFinalizePayload.model_validate(payload)
    logger.info(
        "wiki_finalize: tenant={} kb={} language={!r}",
        parsed.tenant_id,
        parsed.knowledge_base_id,
        parsed.language,
    )
    return await process_wiki_finalize(
        tenant_id=parsed.tenant_id,
        knowledge_base_id=parsed.knowledge_base_id,
        language=parsed.language,
    )


async def process_wiki_finalize(
    *,
    tenant_id: int,
    knowledge_base_id: str,
    language: str,
) -> dict[str, JsonValue]:
    """Run the knowledge-base-wide wiki convergence pass via the core layer.

    Core implementation lands in a later wave; this entry point stays so
    the worker task has a stable seam to call. Tests swap it for an
    ``AsyncMock`` to exercise the dispatch contract without the full
    pipeline.
    """
    raise NotImplementedError(
        "the wiki finalize convergence pass (index-intro rebuild, "
        "dead-link cleanup, cross-link injection) is delivered by the "
        "core wiki layer in a later wave.",
    )


__all__ = [
    "WikiFinalizePayload",
    "process_wiki_finalize",
    "task_wiki_finalize",
]
