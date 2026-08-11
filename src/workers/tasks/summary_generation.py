"""ARQ worker task: ``summary:generation``.

Maps the upstream summary-generation task: receives the serialized
summary payload over ARQ, validates it, and dispatches it to the core
LLM summary pipeline. The handler stays thin — payload parsing, logging,
and result shaping live here; the actual model call, chunk write, and
re-indexing live in the core layer
(:func:`src.core.knowledge.documents.summary.process_summary`).

The core summary pipeline needs a composed set of session-scoped
dependencies (chat client, repositories, knowledge-base service, prompt)
that the worker wiring layer must construct per job. Until that wiring
lands, :func:`process_summary_generation` is the stable seam the handler
delegates to; it raises ``NotImplementedError`` so an unwired invocation
fails loudly instead of reaching the core with missing dependencies.

Wire field names mirror the upstream contract so payloads enqueued by
the existing web paths deserialize without translation. The ``refresh``
flag distinguishes an independently queued refresh (updates existing
summary chunks in place) from the parse-attempt summary; ``attempt``
links the run back to its parent parse attempt for span recording.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict

from src.app_logging import logger
from src.workers.base import WorkerContext
from src.workers.registry import JsonValue, register_task

# Upstream task type constant, kept verbatim so legacy queue consumers
# see the same string they used to enqueue with.
TASK_NAME = "summary:generation"


class SummaryGenerationTaskPayload(BaseModel):
    """ARQ-side payload for the ``summary:generation`` task.

    Mirrors the upstream wire contract: the three identifiers are
    required; ``language``, ``refresh``, and ``attempt`` are optional,
    matching the omitempty JSON tags on the Go side. Field names use
    snake_case so ARQ's JSON deserializer maps transparently onto this
    model.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    tenant_id: int
    knowledge_base_id: str
    knowledge_id: str
    language: str = ""
    refresh: bool = False
    attempt: int = 0


async def process_summary_generation(
    *,
    tenant_id: int,
    knowledge_id: str,
    knowledge_base_id: str,
    language: str,
    refresh: bool,
    attempt: int,
) -> dict[str, JsonValue]:
    """Run one summary-generation pass for a knowledge item.

    Core implementation (model call + summary chunk write + re-index)
    lands in a later wave; this entry point stays so the worker task has
    a stable seam to call. Tests swap it for an ``AsyncMock`` to
    exercise the dispatch contract without the composed core pipeline.
    """
    raise NotImplementedError(
        "Summary generation dependency composition lands in a later wave.",
    )


@register_task(TASK_NAME)
async def task_summary_generation(
    ctx: WorkerContext,
    **payload: JsonValue,
) -> dict[str, JsonValue]:
    """ARQ handler for the ``summary:generation`` task.

    Parses the JSON payload into :class:`SummaryGenerationTaskPayload`
    and delegates to :func:`process_summary_generation`. The ``ctx``
    argument is currently unused — the worker context carries the
    ARQ-Redis pool, but the summary seam is invoked without database
    access at this stage.
    """
    parsed = SummaryGenerationTaskPayload.model_validate(payload)
    logger.info(
        "summary:generation: tenant={} knowledge={} kb={} language={!r} refresh={} attempt={}",
        parsed.tenant_id,
        parsed.knowledge_id,
        parsed.knowledge_base_id,
        parsed.language,
        parsed.refresh,
        parsed.attempt,
    )
    return await process_summary_generation(
        tenant_id=parsed.tenant_id,
        knowledge_id=parsed.knowledge_id,
        knowledge_base_id=parsed.knowledge_base_id,
        language=parsed.language,
        refresh=parsed.refresh,
        attempt=parsed.attempt,
    )


def parse_payload(payload: Mapping[str, JsonValue]) -> SummaryGenerationTaskPayload:
    """Validate a raw payload mapping into the task payload model.

    Exposed for tests and the wiring layer so the handler and its
    callers agree on the schema.
    """
    return SummaryGenerationTaskPayload.model_validate(dict(payload))


__all__ = [
    "TASK_NAME",
    "SummaryGenerationTaskPayload",
    "parse_payload",
    "process_summary_generation",
    "task_summary_generation",
]
