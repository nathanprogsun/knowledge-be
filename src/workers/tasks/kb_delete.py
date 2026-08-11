"""ARQ worker task: ``kb:delete``.

Maps the upstream knowledge-base-delete task: receives the serialized
delete payload over ARQ, validates it, and dispatches it to the core
cascade delete :func:`src.core.knowledge.knowledge_bases.delete.process_kb_delete`.

The handler stays thin — payload parsing, logging, and result shaping
live here; the cascade (index-cleanup hook, chunk sweep, document batch
soft delete) lives in the core layer. The worker wiring layer injects
the repository pair bound to a per-job ``AsyncSession``; until that
wiring lands, the seam raises rather than silently skipping a delete,
so a miswired worker fails loudly.

Wire field names mirror the upstream contract so payloads enqueued by
the existing web/CLI paths deserialize without translation. The
``data_source_ids`` / ``effective_engines`` fields are carried for the
deferred task-cancellation and engine-resolution seams; the Python
cascade consumes the bound ``vector_store_id`` snapshot today.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from src.app_logging import logger
from src.core.knowledge.knowledge_bases.delete import (
    KBDeleteResult,
)
from src.core.knowledge.knowledge_bases.delete import (
    process_kb_delete as _core_process_kb_delete,
)
from src.db.models.knowledge import Document
from src.workers.base import WorkerContext
from src.workers.registry import JsonValue, register_task

# Upstream task name kept verbatim so legacy queue consumers see the
# same string they used to enqueue with.
TASK_NAME = "kb:delete"


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


class KBDeletePayload(BaseModel):
    """ARQ-side payload for the ``kb:delete`` task.

    Mirrors the upstream ``KBDeletePayload`` wire contract: the tenant
    and knowledge-base ids are required; the data-source ids, effective
    engines and the bound vector-store snapshot are optional, matching
    the ``omitempty`` JSON tags on the upstream side. The upstream
    tracing-context fields are accepted via the JSON payload but not
    modelled on this surface (no consumer reads them yet), so
    ``extra="ignore"`` drops them without failing the parse.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    tenant_id: int
    knowledge_base_id: str
    data_source_ids: list[str] = []
    effective_engines: list[RetrieverEngineParams] = []
    vector_store_id: str | None = None


class KnowledgeDeleteRepo(Protocol):
    """Structural view of the knowledge repo the core cascade consumes.

    Matches the concrete repository's method shapes so the wiring layer
    can hand the worker a session-bound repo without the worker reaching
    into the storage layer's implementation. ``Document`` is the storage
    row type the core cascade reads and is imported for typing only.
    """

    async def list_by_knowledge_base(
        self,
        tenant_id: int,
        knowledge_base_id: str,
    ) -> Sequence[Document]: ...
    async def soft_delete_list(
        self,
        *,
        tenant_id: int,
        ids: Sequence[str],
        now: datetime,
    ) -> int: ...


class ChunkDeleteRepo(Protocol):
    """Structural view of the chunk repo the core cascade consumes.

    Matches the concrete repository's method shape so the wiring layer
    can hand the worker a session-bound repo without the worker reaching
    into the storage layer's implementation.
    """

    async def delete_by_knowledge_id(
        self,
        *,
        tenant_id: int,
        knowledge_id: str,
        now: datetime,
    ) -> int: ...


@register_task(TASK_NAME)
async def task_kb_delete(
    ctx: WorkerContext,
    *,
    knowledge_repo: KnowledgeDeleteRepo | None = None,
    chunk_repo: ChunkDeleteRepo | None = None,
    **payload: JsonValue,
) -> dict[str, JsonValue]:
    """ARQ handler for the ``kb:delete`` task.

    Parses the ARQ payload into :class:`KBDeletePayload` and delegates to
    :func:`process_kb_delete`. The repository pair bound to a per-job
    session may be injected by the worker wiring layer; when either is
    omitted the seam raises so a miswired delete fails loudly. ``ctx`` is
    currently unused — the worker context carries the ARQ-Redis pool,
    which the delete seam does not need at this stage.
    """
    parsed = KBDeletePayload.model_validate(payload)
    logger.info(
        "kb:delete: tenant={} kb={} data_sources={} engines={} store={!r}",
        parsed.tenant_id,
        parsed.knowledge_base_id,
        len(parsed.data_source_ids),
        len(parsed.effective_engines),
        parsed.vector_store_id,
    )
    return await process_kb_delete(
        tenant_id=parsed.tenant_id,
        knowledge_base_id=parsed.knowledge_base_id,
        data_source_ids=parsed.data_source_ids,
        vector_store_id=parsed.vector_store_id,
        knowledge_repo=knowledge_repo,
        chunk_repo=chunk_repo,
    )


async def process_kb_delete(
    *,
    tenant_id: int,
    knowledge_base_id: str,
    data_source_ids: Sequence[str],
    vector_store_id: str | None,
    knowledge_repo: KnowledgeDeleteRepo | None = None,
    chunk_repo: ChunkDeleteRepo | None = None,
) -> dict[str, JsonValue]:
    """Run one knowledge-base cascade delete via the core composition.

    Delegates to
    :func:`src.core.knowledge.knowledge_bases.delete.process_kb_delete`
    and returns a JSON-serialisable summary of the cascade. A failing
    durable step (document batch soft delete) propagates to the worker
    caller for ARQ retry.

    ``knowledge_repo`` and ``chunk_repo`` are injected by the worker
    wiring layer — session-bound repositories over the per-job
    ``AsyncSession``. No repositories can be constructed here (the
    worker context carries no database engine), so an uninjected call
    raises ``NotImplementedError`` instead of silently skipping the
    delete. ``data_source_ids`` is carried for the deferred
    task-cancellation seam and is only correlated here.
    """
    if knowledge_repo is None or chunk_repo is None:
        raise NotImplementedError(
            "knowledge-base delete requires session-bound repositories "
            "injected by the worker wiring layer",
        )
    result: KBDeleteResult = await _core_process_kb_delete(
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        # The structural repo views are satisfied by the concrete
        # session-bound repositories at runtime; mypy cannot see across
        # the Protocol boundary, so the call is narrowed explicitly.
        knowledge_repo=knowledge_repo,  # type: ignore[arg-type]
        chunk_repo=chunk_repo,  # type: ignore[arg-type]
        vector_store_id=vector_store_id,
    )
    return {
        "knowledge_base_id": knowledge_base_id,
        "knowledge_ids": list(result.knowledge_ids),
        "deleted_chunks": result.deleted_chunks,
        "deleted_knowledge": result.deleted_knowledge,
        "vector_store_id": result.vector_store_id,
        "status": "completed",
    }


def parse_payload(payload: Mapping[str, JsonValue]) -> KBDeletePayload:
    """Validate a raw payload mapping into the task payload model.

    Exposed for tests and the wiring layer so the handler and its
    callers agree on the schema.
    """
    return KBDeletePayload.model_validate(dict(payload))


__all__ = [
    "TASK_NAME",
    "ChunkDeleteRepo",
    "KBDeletePayload",
    "KnowledgeDeleteRepo",
    "RetrieverEngineParams",
    "parse_payload",
    "process_kb_delete",
    "task_kb_delete",
]
