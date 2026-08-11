"""ARQ worker task: ``datatable:summary``.

Maps the reference data-table summary task: receives the serialized
payload over ARQ, validates it, and dispatches it to the core
data-table summary generation
(:func:`src.core.knowledge.documents.datatable_summary.process_datatable_summary`).

The handler stays thin: payload parsing, logging, and result shaping
live here; the actual table load / description generation / chunk and
vector index orchestration lives in the core layer. Composing the core
dependencies (chat model, embedder, retrieval engine, repositories, and
the table-data tool) is the worker wiring layer's responsibility — it
supplies a :class:`DatatableSummaryRunner` that closes over those
dependencies and invokes the core generator. Until the wiring lands, the
task refuses to run so a miswired worker fails loudly instead of silently
dropping real data-table summaries.

Wire field names mirror the upstream contract so payloads enqueued by
the existing web/CLI paths deserialize without translation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from src.app_logging import logger
from src.core.knowledge.documents.datatable_summary import DataTableSummaryResult
from src.workers.base import WorkerContext
from src.workers.registry import JsonValue, register_task

#: Upstream task type kept verbatim so legacy queue consumers see the
#: same string they used to enqueue with.
TASK_NAME = "datatable:summary"


class DatatableSummaryPayload(BaseModel):
    """ARQ-side payload for the ``datatable:summary`` task.

    Mirrors the upstream wire contract: tenant and knowledge identifiers
    plus the summary / embedding model ids the wiring layer uses to build
    the chat and embedder seams. The model ids default to blank so an
    omitted value lets the wiring layer fall back to the knowledge base's
    configured models; the upstream tracing-context fields ride along in
    the JSON payload and are ignored here.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    tenant_id: int
    knowledge_id: str
    summary_model: str = ""
    embedding_model: str = ""


class DatatableSummaryRunner(Protocol):
    """Composition seam the worker wiring layer provides.

    Builds the core dependencies for one data-table summary run (chat
    model, embedder, retrieval engine, repositories, and the table-data
    tool) and invokes
    :func:`src.core.knowledge.documents.datatable_summary.process_datatable_summary`.
    """

    async def __call__(
        self, *, payload: DatatableSummaryPayload
    ) -> DataTableSummaryResult: ...


@register_task(TASK_NAME)
async def task_datatable_summary(
    ctx: WorkerContext,
    *,
    runner: DatatableSummaryRunner | None = None,
    **payload: JsonValue,
) -> dict[str, JsonValue]:
    """ARQ handler for the ``datatable:summary`` task.

    Parses the ARQ payload into :class:`DatatableSummaryPayload`, logs the
    dispatch scope, delegates to :func:`run_datatable_summary`, and returns
    a JSON-serialisable summary of the run. A wiring-provided ``runner``
    composes the core dependencies; without one the task refuses to run.
    """
    parsed = DatatableSummaryPayload.model_validate(payload)
    logger.info(
        "datatable_summary: tenant={} knowledge={} summary_model={!r} embedding_model={!r}",
        parsed.tenant_id,
        parsed.knowledge_id,
        parsed.summary_model,
        parsed.embedding_model,
    )
    result = await run_datatable_summary(payload=parsed, runner=runner)
    return _serialise_result(result)


async def run_datatable_summary(
    *,
    payload: DatatableSummaryPayload,
    runner: DatatableSummaryRunner | None = None,
) -> DataTableSummaryResult:
    """Dispatch a parsed data-table summary payload to the core generator.

    A ``runner`` supplied by the worker wiring layer composes the core
    dependencies and invokes
    :func:`src.core.knowledge.documents.datatable_summary.process_datatable_summary`.
    Without a runner the task refuses to run so a miswired worker fails
    loudly instead of silently skipping data-table summaries.
    """
    if runner is not None:
        return await runner(payload=payload)
    raise NotImplementedError(
        "datatable_summary requires a wiring-provided runner; the worker "
        "wiring layer composes the core dependencies.",
    )


def parse_payload(payload: Mapping[str, JsonValue]) -> DatatableSummaryPayload:
    """Validate a raw payload mapping into the task payload model.

    Exposed for tests and the wiring layer so the handler and its
    callers agree on the schema.
    """
    return DatatableSummaryPayload.model_validate(dict(payload))


def _serialise_result(result: DataTableSummaryResult) -> dict[str, JsonValue]:
    """Project a :class:`DataTableSummaryResult` onto a JSON-serialisable dict."""
    return {
        "knowledge_id": result.knowledge_id,
        "summary_chunk_id": result.summary_chunk_id,
        "column_chunk_id": result.column_chunk_id,
    }


__all__ = [
    "DatatableSummaryPayload",
    "DatatableSummaryRunner",
    "TASK_NAME",
    "parse_payload",
    "run_datatable_summary",
    "task_datatable_summary",
]
