"""ARQ worker task: ``question:generation``.

Maps the upstream question-generation task: receives the serialized
question payload over ARQ, validates it, and dispatches it to the core
LLM question-generation pipeline. The handler stays thin — payload
parsing, logging, and result shaping live here; the actual model call,
chunk-metadata binding, and re-indexing live in the core layer
(:func:`src.core.knowledge.documents.question_gen.generate_questions`).

The core question pipeline needs a composed set of session-scoped
dependencies (chat client, repositories, knowledge-base service, prompt)
that the worker wiring layer must construct per job. Until that wiring
lands, :func:`process_question_generation` is the stable seam the
handler delegates to; it raises ``NotImplementedError`` so an unwired
invocation fails loudly instead of reaching the core with missing
dependencies.

Wire field names mirror the upstream contract so payloads enqueued by
the existing web paths deserialize without translation. The optional
batch fields (``chunk_ids`` / ``chunk_id``, ``batch_index``,
``prev_chunk_id`` / ``next_chunk_id``) switch the handler into the
batched fan-out mode where questions are generated for an ordered window
of text chunks only; when empty, the legacy whole-knowledge mode runs.
``attempt`` links the run back to its parent parse attempt for span
recording.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict

from src.app_logging import logger
from src.workers.base import WorkerContext
from src.workers.registry import JsonValue, register_task

# Upstream task type constant, kept verbatim so legacy queue consumers
# see the same string they used to enqueue with.
TASK_NAME = "question:generation"

# Default question count when the payload omits one — mirrors the
# shared ingestion default used by the web layer.
_DEFAULT_QUESTION_COUNT = 3


class QuestionGenerationTaskPayload(BaseModel):
    """ARQ-side payload for the ``question:generation`` task.

    Mirrors the upstream wire contract: the three identifiers are
    required; everything else is optional, matching the omitempty JSON
    tags on the Go side. An explicitly enqueued ``question_count`` of 0
    means "use the default"; the core clamps the resolved value.
    Field names use snake_case so ARQ's JSON deserializer maps
    transparently onto this model.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    tenant_id: int
    knowledge_base_id: str
    knowledge_id: str
    question_count: int = _DEFAULT_QUESTION_COUNT
    language: str = ""
    attempt: int = 0
    chunk_ids: list[str] = []
    chunk_id: str = ""
    batch_index: int = 0
    prev_chunk_id: str = ""
    next_chunk_id: str = ""


async def process_question_generation(
    *,
    tenant_id: int,
    knowledge_id: str,
    knowledge_base_id: str,
    question_count: int,
    language: str,
    attempt: int,
    chunk_ids: list[str],
    chunk_id: str,
    batch_index: int,
    prev_chunk_id: str,
    next_chunk_id: str,
) -> dict[str, JsonValue]:
    """Run one question-generation pass for a knowledge item.

    Core implementation (model call + chunk-metadata binding +
    re-indexing) lands in a later wave; this entry point stays so the
    worker task has a stable seam to call. Tests swap it for an
    ``AsyncMock`` to exercise the dispatch contract without the composed
    core pipeline.
    """
    raise NotImplementedError(
        "Question generation dependency composition lands in a later wave.",
    )


@register_task(TASK_NAME)
async def task_question_generation(
    ctx: WorkerContext,
    **payload: JsonValue,
) -> dict[str, JsonValue]:
    """ARQ handler for the ``question:generation`` task.

    Parses the JSON payload into :class:`QuestionGenerationTaskPayload`
    and delegates to :func:`process_question_generation`. The ``ctx``
    argument is currently unused — the worker context carries the
    ARQ-Redis pool, but the question seam is invoked without database
    access at this stage.
    """
    parsed = QuestionGenerationTaskPayload.model_validate(payload)
    logger.info(
        "question:generation: tenant={} knowledge={} kb={} question_count={} "
        "language={!r} attempt={} batch={} chunks={}",
        parsed.tenant_id,
        parsed.knowledge_id,
        parsed.knowledge_base_id,
        parsed.question_count,
        parsed.language,
        parsed.attempt,
        parsed.batch_index,
        len(parsed.chunk_ids),
    )
    return await process_question_generation(
        tenant_id=parsed.tenant_id,
        knowledge_id=parsed.knowledge_id,
        knowledge_base_id=parsed.knowledge_base_id,
        question_count=parsed.question_count,
        language=parsed.language,
        attempt=parsed.attempt,
        chunk_ids=parsed.chunk_ids,
        chunk_id=parsed.chunk_id,
        batch_index=parsed.batch_index,
        prev_chunk_id=parsed.prev_chunk_id,
        next_chunk_id=parsed.next_chunk_id,
    )


def parse_payload(payload: Mapping[str, JsonValue]) -> QuestionGenerationTaskPayload:
    """Validate a raw payload mapping into the task payload model.

    Exposed for tests and the wiring layer so the handler and its
    callers agree on the schema.
    """
    return QuestionGenerationTaskPayload.model_validate(dict(payload))


__all__ = [
    "TASK_NAME",
    "QuestionGenerationTaskPayload",
    "parse_payload",
    "process_question_generation",
    "task_question_generation",
]
