"""ARQ worker task: ``index:delete``.

Maps the upstream index-delete task: receives the serialized index
cleanup payload over ARQ, validates it, and delegates to the core
index-cleanup seam :func:`process_index_delete`.

The handler stays thin — payload parsing and logging live here; the
engine resolution and the batched vector-store delete-by-chunk-id run
live in the core layer. The core index-cleanup composition (engine
factory, model-dimension resolution, batched deletes) lands in a later
wave; until then :func:`process_index_delete` is the stable seam the
handler delegates to, and it raises ``NotImplementedError`` so an
unwired invocation fails loudly instead of reaching the vector store
with missing dependencies.

Wire field names mirror the upstream contract so payloads enqueued by
the existing web/CLI paths deserialize without translation.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict

from src.app_logging import logger
from src.workers.base import WorkerContext
from src.workers.registry import JsonValue, register_task

# Upstream task name kept verbatim so legacy queue consumers see the
# same string they used to enqueue with.
TASK_NAME = "index:delete"


class RetrieverEngineParams(BaseModel):
    """One retriever-engine selection inside ``effective_engines``.

    Mirrors the upstream ``RetrieverEngineParams`` wire pair. Plain
    strings (rather than the enum-typed core model) keep the worker
    robust to engine types the core enum has not caught up with yet, so
    a legacy payload with an unknown engine still parses.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    retriever_engine_type: str = ""
    retriever_type: str = ""


class IndexDeletePayload(BaseModel):
    """ARQ-side payload for the ``index:delete`` task.

    Mirrors the upstream ``IndexDeletePayload`` wire contract: the
    tenant, knowledge-base and embedding-model ids are required; the
    knowledge-base type, chunk ids, effective engines and the bound
    vector-store snapshot are optional, matching the ``omitempty`` JSON
    tags on the upstream side. The upstream tracing-context fields are
    accepted via the JSON payload but not modelled on this surface (no
    consumer reads them yet), so ``extra="ignore"`` drops them without
    failing the parse.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    tenant_id: int
    knowledge_base_id: str
    embedding_model_id: str
    kb_type: str = ""
    chunk_ids: list[str] = []
    effective_engines: list[RetrieverEngineParams] = []
    vector_store_id: str | None = None


@register_task(TASK_NAME)
async def task_index_delete(
    ctx: WorkerContext,
    **payload: JsonValue,
) -> dict[str, JsonValue]:
    """ARQ handler for the ``index:delete`` task.

    Parses the JSON payload into :class:`IndexDeletePayload` and
    delegates to :func:`process_index_delete`. The ``ctx`` argument is
    currently unused — the worker context carries the ARQ-Redis pool,
    but the index-cleanup seam is invoked without database access at
    this stage.
    """
    parsed = IndexDeletePayload.model_validate(payload)
    logger.info(
        "index:delete: tenant={} kb={} model={} kb_type={!r} chunks={} "
        "engines={} store={!r}",
        parsed.tenant_id,
        parsed.knowledge_base_id,
        parsed.embedding_model_id,
        parsed.kb_type,
        len(parsed.chunk_ids),
        len(parsed.effective_engines),
        parsed.vector_store_id,
    )
    return await process_index_delete(
        tenant_id=parsed.tenant_id,
        knowledge_base_id=parsed.knowledge_base_id,
        embedding_model_id=parsed.embedding_model_id,
        kb_type=parsed.kb_type,
        chunk_ids=parsed.chunk_ids,
        effective_engines=parsed.effective_engines,
        vector_store_id=parsed.vector_store_id,
    )


async def process_index_delete(
    *,
    tenant_id: int,
    knowledge_base_id: str,
    embedding_model_id: str,
    kb_type: str,
    chunk_ids: list[str],
    effective_engines: list[RetrieverEngineParams],
    vector_store_id: str | None,
) -> dict[str, JsonValue]:
    """Run one index-cleanup pass for a set of chunk ids.

    Core implementation (engine resolution, embedding-model dimension
    lookup, batched vector-store delete-by-chunk-id) lands in a later
    wave; this entry point stays so the worker task has a stable seam to
    call. Tests swap it for an ``AsyncMock`` to exercise the dispatch
    contract without the composed core pipeline.
    """
    raise NotImplementedError(
        "Index cleanup composition lands in a later wave.",
    )


def parse_payload(payload: Mapping[str, JsonValue]) -> IndexDeletePayload:
    """Validate a raw payload mapping into the task payload model.

    Exposed for tests and the wiring layer so the handler and its
    callers agree on the schema.
    """
    return IndexDeletePayload.model_validate(dict(payload))


__all__ = [
    "TASK_NAME",
    "IndexDeletePayload",
    "RetrieverEngineParams",
    "parse_payload",
    "process_index_delete",
    "task_index_delete",
]
