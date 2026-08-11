"""ARQ worker task: ``knowledge:post_process``.

Maps the upstream knowledge-post-process task: receives the serialized
post-process payload over ARQ, validates it, and delegates to the core
orchestrator :func:`src.core.knowledge.documents.post_process_service.run_post_process`.

The handler stays thin: payload parsing, logging, and result shaping
live here; the enrichment fan-out orchestration (summary / question
generation / chunk extract / wiki ingest dispatch, span tracking,
multimodal stage close, and the pending-subtask reconciliation) lives in
the core layer. The worker wiring layer is responsible for constructing
a fully composed :class:`PostProcessService` (with all seams wired)
before any real run — until then the default service runs in its
deferred-seam mode and refuses to act rather than guessing.

Wire field names mirror the upstream contract so payloads enqueued by
the existing web/CLI paths deserialize without translation.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from src.app_logging import logger
from src.core.knowledge.documents.post_process_service import (
    PostProcessOutcome,
    PostProcessService,
)
from src.core.knowledge.documents.post_process_service import (
    run_post_process as _core_run_post_process,
)
from src.workers.base import WorkerContext
from src.workers.registry import JsonValue, register_task


class KnowledgePostProcessTaskPayload(BaseModel):
    """ARQ-side payload for the ``knowledge:post_process`` task.

    Mirrors the upstream wire contract: the three ids are required and
    ``language`` / ``attempt`` are optional, matching the ``omitempty``
    JSON tags on the Go side. Field names use snake_case so ARQ's JSON
    deserializer maps transparently onto this model.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    tenant_id: int
    knowledge_id: str
    knowledge_base_id: str
    language: str = ""
    attempt: int = 0


@register_task("knowledge:post_process")
async def task_knowledge_post_process(
    ctx: WorkerContext,
    *,
    service: PostProcessService | None = None,
    **payload: JsonValue,
) -> dict[str, JsonValue]:
    """ARQ handler for the ``knowledge:post_process`` task.

    Parses the ARQ payload into :class:`KnowledgePostProcessTaskPayload`,
    delegates to :func:`src.core.knowledge.documents.post_process_service.run_post_process`,
    and returns a JSON-serialisable summary of the outcome. A fully
    composed ``service`` may be injected by the worker wiring layer; when
    omitted, the core layer constructs a default instance whose deferred
    seams refuse to run before any external work.
    """
    parsed = KnowledgePostProcessTaskPayload.model_validate(payload)
    logger.info(
        "knowledge_post_process: tenant={} knowledge={} kb={} attempt={}",
        parsed.tenant_id,
        parsed.knowledge_id,
        parsed.knowledge_base_id,
        parsed.attempt,
    )

    outcome = await _core_run_post_process(
        tenant_id=parsed.tenant_id,
        knowledge_id=parsed.knowledge_id,
        knowledge_base_id=parsed.knowledge_base_id,
        language=parsed.language,
        attempt=parsed.attempt,
        service=service,
    )
    return _serialise_outcome(outcome)


def _serialise_outcome(outcome: PostProcessOutcome) -> dict[str, JsonValue]:
    """Project a :class:`PostProcessOutcome` onto a JSON-serialisable dict."""
    return {
        "skipped": outcome.skipped,
        "reason": outcome.reason,
        "chunks_total": outcome.chunks_total,
        "enqueued_summary": outcome.enqueued_summary,
        "enqueued_question": outcome.enqueued_question,
        "enqueued_question_count": outcome.enqueued_question_count,
        "enqueued_wiki": outcome.enqueued_wiki,
        "wiki_slot_owned": outcome.wiki_slot_owned,
        "enqueued_graph": outcome.enqueued_graph,
        "enqueued_graph_count": outcome.enqueued_graph_count,
    }


def parse_payload(payload: dict[str, JsonValue]) -> KnowledgePostProcessTaskPayload:
    """Validate a raw payload mapping into the task payload model.

    Exposed for tests and the wiring layer so the handler and its
    callers agree on the schema.
    """
    return KnowledgePostProcessTaskPayload.model_validate(dict(payload))


__all__ = [
    "KnowledgePostProcessTaskPayload",
    "parse_payload",
    "task_knowledge_post_process",
]
