"""ARQ worker task: ``wiki:ingest``.

Maps the upstream wiki-ingest batch trigger task. The trigger carries
only the per-KB metadata — tenant, knowledge-base id, and language —
because the actual per-document operations live in the durable
pending-ops queue; the worker resolves whatever batch of rows is queued
under that knowledge base and processes it.

The handler stays thin: payload parsing, logging, and result shaping
live here; the map -> taxonomy -> reduce -> settle orchestration lives
in the core layer. A session-bound :class:`WikiIngestService` is
injected by the worker wiring layer — the worker context carries no
database engine, so an uninjected seam raises rather than silently
dropping a batch.

Wire field names mirror the upstream contract so payloads enqueued by
the existing web/CLI paths deserialize without translation.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from src.app_logging import logger
from src.core.knowledge.wiki.ingest_service import WikiIngestService
from src.core.knowledge.wiki.ingest_types import WikiBatchOutcome
from src.workers.base import WorkerContext
from src.workers.registry import JsonValue, register_task


class WikiIngestPayload(BaseModel):
    """ARQ-side payload for the ``wiki:ingest`` trigger task.

    Mirrors the upstream wire contract: only the tenant and knowledge-base
    id are mandatory; ``language`` is optional, matching the ``omitempty``
    JSON tag on the upstream side. The upstream tracing-context fields are
    accepted via the JSON payload but not modelled here (no consumer reads
    them yet), so ``extra="ignore"`` drops them without failing the parse.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    tenant_id: int
    knowledge_base_id: str
    language: str = ""


@register_task("wiki:ingest")
async def task_wiki_ingest(
    ctx: WorkerContext,
    *,
    service: WikiIngestService | None = None,
    **payload: JsonValue,
) -> dict[str, JsonValue]:
    """ARQ handler for the ``wiki:ingest`` task.

    Parses the ARQ payload into :class:`WikiIngestPayload` and delegates
    to :func:`process_wiki_ingest`. A session-bound ``service`` may be
    injected by the worker wiring layer; when omitted, the seam raises so
    a miswired batch fails loudly. ``ctx`` is currently unused — the
    worker context carries the ARQ-Redis pool, which the ingest seam does
    not need at this stage.
    """
    parsed = WikiIngestPayload.model_validate(payload)
    logger.info(
        "wiki_ingest: tenant={} kb={} language={!r}",
        parsed.tenant_id,
        parsed.knowledge_base_id,
        parsed.language,
    )
    return await process_wiki_ingest(
        tenant_id=parsed.tenant_id,
        knowledge_base_id=parsed.knowledge_base_id,
        language=parsed.language,
        service=service,
    )


async def process_wiki_ingest(
    *,
    tenant_id: int,
    knowledge_base_id: str,
    language: str,
    service: WikiIngestService | None = None,
) -> dict[str, JsonValue]:
    """Run one wiki ingest batch via the core wiki engine.

    Delegates to :meth:`WikiIngestService.process_batch`, which peeks the
    durable pending queue and drains up to a batch of per-document ops
    (parse -> chunk -> embed -> index plus the wiki content passes).
    Returns the aggregate outcome as a JSON-serialisable dict.

    ``service`` is injected by the worker wiring layer — a
    :class:`WikiIngestService` bound to a per-job session with all seams
    wired. No service can be constructed here (the worker context carries
    no database engine), so an uninjected call raises
    ``NotImplementedError`` instead of silently skipping the batch.
    """
    if service is None:
        raise NotImplementedError(
            "wiki ingest requires a session-bound WikiIngestService "
            "injected by the worker wiring layer",
        )
    outcome = await service.process_batch(
        None,
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        language=language,
    )
    return _serialise_outcome(outcome)


def _serialise_outcome(outcome: WikiBatchOutcome) -> dict[str, JsonValue]:
    """Project a :class:`WikiBatchOutcome` onto a JSON-serialisable dict."""
    return {
        "pending_ops": outcome.pending_ops,
        "ingest_succeeded": outcome.ingest_succeeded,
        "ingest_failed": outcome.ingest_failed,
        "retract_handled": outcome.retract_handled,
        "pages_affected": outcome.pages_affected,
        "follow_up_scheduled": outcome.follow_up_scheduled,
        "rate_limited": outcome.rate_limited,
    }


__all__ = [
    "WikiIngestPayload",
    "process_wiki_ingest",
    "task_wiki_ingest",
]
