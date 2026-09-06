"""ARQ worker task: ``document_process``.

Maps the upstream document-process task: receives the serialized
``DocumentProcessPayload`` over ARQ, validates it, and dispatches it
to the core pipeline :func:`src.core.knowledge.documents.process_document.process_document`.

The handler stays thin: payload parsing, logging, and result shaping
live here; the actual parse / chunk / embed / index orchestration lives
in the core layer. The worker runtime composes the pipeline, including
the file-service resolver used to persist ``file_url`` bytes.

Wire field names mirror the upstream contract so payloads enqueued by
the existing web/CLI paths deserialize without translation.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict

from src.app_logging import logger
from src.common.json import JsonObject, JsonValue
from src.core.knowledge.documents.process_document import (
    DocumentProcessPipeline,
    ProcessOutcome,
)
from src.core.knowledge.documents.process_document import (
    process_document as _core_process_document,
)
from src.workers.base import WorkerContext
from src.workers.registry import register_task
from src.workers.runtime_ctx import document_process_runtime_from_ctx

# Default question count when the payload omits one — mirrors the
# shared ingestion default used by the web layer.
_DEFAULT_QUESTION_COUNT = 3


class DocumentProcessTaskPayload(BaseModel):
    """ARQ-side payload for the ``document_process`` task.

    Mirrors the upstream wire contract: every field except the three
    ids is optional, matching the omitempty JSON tags on the Go side.
    Field names use snake_case so ARQ's JSON deserializer maps
    transparently onto this model.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    tenant_id: int
    knowledge_id: str
    knowledge_base_id: str
    request_id: str = ""
    file_path: str = ""
    file_name: str = ""
    file_type: str = ""
    url: str = ""
    file_url: str = ""
    enable_multimodel: bool = False
    enable_question_generation: bool = False
    question_count: int = _DEFAULT_QUESTION_COUNT
    language: str = ""


@register_task("document_process")
async def task_document_process(
    ctx: WorkerContext,
    *,
    pipeline: DocumentProcessPipeline | None = None,
    **payload: JsonValue,
) -> JsonObject:
    """ARQ handler for the ``document_process`` task.

    Parses the ARQ payload into :class:`DocumentProcessTaskPayload`,
    delegates to :func:`src.core.knowledge.documents.process_document.process_document`,
    and returns a JSON-serialisable summary of the outcome. A fully
    composed ``pipeline`` may be injected by the worker wiring layer;
    when omitted, the core layer constructs a default instance whose
    deferred seams short-circuit before any external work.
    """
    parsed = DocumentProcessTaskPayload.model_validate(payload)
    source_url = parsed.url or parsed.file_url
    logger.info(
        "document_process: tenant={} knowledge={} kb={} file_path={!r} url={!r}",
        parsed.tenant_id,
        parsed.knowledge_id,
        parsed.knowledge_base_id,
        parsed.file_path,
        source_url,
    )

    outcome = await _run_document_process(
        ctx,
        parsed=parsed,
        source_url=source_url,
        pipeline=pipeline,
    )
    return _serialise_outcome(outcome)


async def _run_document_process(
    ctx: WorkerContext,
    *,
    parsed: DocumentProcessTaskPayload,
    source_url: str,
    pipeline: DocumentProcessPipeline | None,
) -> ProcessOutcome:
    """Prefer the worker-wired runtime; tests still inject or patch the core."""
    runtime = None if pipeline is not None else document_process_runtime_from_ctx(ctx)
    if runtime is not None:
        return await runtime.run_document_process(
            tenant_id=parsed.tenant_id,
            knowledge_id=parsed.knowledge_id,
            knowledge_base_id=parsed.knowledge_base_id,
            file_path=parsed.file_path,
            file_name=parsed.file_name,
            file_type=parsed.file_type,
            url=source_url,
            enable_multimodel=parsed.enable_multimodel,
            language=parsed.language,
            request_id=parsed.request_id,
        )
    return await _core_process_document(
        tenant_id=parsed.tenant_id,
        knowledge_id=parsed.knowledge_id,
        knowledge_base_id=parsed.knowledge_base_id,
        file_path=parsed.file_path,
        file_name=parsed.file_name,
        file_type=parsed.file_type,
        url=source_url,
        enable_multimodel=parsed.enable_multimodel,
        language=parsed.language,
        request_id=parsed.request_id,
        now=datetime.now(UTC),
        pipeline=pipeline,
    )


def _serialise_outcome(outcome: ProcessOutcome) -> JsonObject:
    """Project a :class:`ProcessOutcome` onto a JSON-serialisable dict."""
    return {
        "parse_status": outcome.parse_status,
        "enable_status": outcome.enable_status,
        "summary_status": outcome.summary_status,
        "storage_size": outcome.storage_size,
        "error_message": outcome.error_message,
        "text_chunk_count": outcome.text_chunk_count,
        "skipped": outcome.skipped,
    }


def parse_payload(payload: Mapping[str, JsonValue]) -> DocumentProcessTaskPayload:
    """Validate a raw payload mapping into the task payload model.

    Exposed for tests and the wiring layer so the handler and its
    callers agree on the schema.
    """
    return DocumentProcessTaskPayload.model_validate(dict(payload))


__all__ = [
    "DocumentProcessTaskPayload",
    "parse_payload",
    "task_document_process",
]
