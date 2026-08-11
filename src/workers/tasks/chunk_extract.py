"""ARQ worker task: ``chunk:extract``.

Maps the upstream per-chunk graph-extraction task: the worker receives
the serialized extract payload over ARQ, validates it, resolves the
extraction model client, and delegates to the core
:class:`src.core.knowledge.documents.chunk_extract.ChunkExtractor`.

The handler stays thin: payload parsing, model resolution, logging, and
result shaping live here; the extraction itself (cancel / delete guards,
effective extract-config resolution, the LLM call, and graph
persistence) lives in the core layer. A fully composed ``ChunkExtractor``
and a chat resolver are injected by the worker wiring layer; until then
the task short-circuits with a skipped outcome before any external work.

Wire field names mirror the upstream contract so payloads enqueued by
the existing web/CLI paths deserialize without translation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from src.ai.embedding import TaskContext
from src.ai.llm import Chat
from src.app_logging import logger
from src.core.knowledge.documents.chunk_extract import (
    ChunkExtractor,
    ExtractionOutcome,
)
from src.workers.base import WorkerContext
from src.workers.registry import JsonValue, register_task

# Upstream task name kept verbatim so legacy queue consumers see the
# same string they used to enqueue with.
TASK_NAME = "chunk:extract"


class ChunkExtractTaskPayload(BaseModel):
    """ARQ-side payload for the ``chunk:extract`` task.

    Mirrors the upstream wire contract: the tenant, chunk, and model ids
    are required; the parent-knowledge link (``knowledge_id``), the
    ``attempt`` ordinal, and the ``chunk_index`` are optional and feed
    trace correlation / span naming. The upstream tracing-context fields
    are accepted via the JSON payload but not modelled on this surface
    (no consumer reads them yet).
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    tenant_id: int
    chunk_id: str
    model_id: str
    knowledge_id: str = ""
    attempt: int = 0
    chunk_index: int = 0


@runtime_checkable
class ChatResolver(Protocol):
    """Resolves a chat client for the extraction model id."""

    async def resolve_chat(self, *, model_id: str) -> Chat | None:
        """Return the chat client, or ``None`` when the model is unavailable."""
        ...


@register_task(TASK_NAME)
async def task_chunk_extract(
    ctx: WorkerContext,
    *,
    extractor: ChunkExtractor | None = None,
    chat_resolver: ChatResolver | None = None,
    **payload: JsonValue,
) -> dict[str, JsonValue]:
    """ARQ handler for the ``chunk:extract`` task.

    Parses the ARQ payload into :class:`ChunkExtractTaskPayload`,
    resolves the extraction model, and delegates to the core
    :class:`ChunkExtractor`. A fully composed ``extractor`` and
    ``chat_resolver`` may be injected by the worker wiring layer; when
    either is omitted the task short-circuits with a skipped outcome
    rather than touching external services.
    """
    parsed = parse_payload(payload)
    logger.info(
        "chunk:extract: tenant={} chunk={} model={} knowledge={} attempt={} index={}",
        parsed.tenant_id,
        parsed.chunk_id,
        parsed.model_id,
        parsed.knowledge_id,
        parsed.attempt,
        parsed.chunk_index,
    )
    if extractor is None or chat_resolver is None:
        return _serialise_outcome(ExtractionOutcome(skipped=True, reason="not_wired"))
    outcome = await _run_extraction(
        extractor=extractor,
        chat_resolver=chat_resolver,
        tenant_id=parsed.tenant_id,
        chunk_id=parsed.chunk_id,
        model_id=parsed.model_id,
        knowledge_id=parsed.knowledge_id,
        chunk_index=parsed.chunk_index,
    )
    return _serialise_outcome(outcome)


async def _run_extraction(
    *,
    extractor: ChunkExtractor,
    chat_resolver: ChatResolver,
    tenant_id: int,
    chunk_id: str,
    model_id: str,
    knowledge_id: str,
    chunk_index: int,
) -> ExtractionOutcome:
    """Resolve the extraction model and run one chunk extraction.

    A model that cannot be resolved short-circuits with a
    ``model_unavailable`` skip, mirroring the core's other skip guards.
    """
    chat = await chat_resolver.resolve_chat(model_id=model_id)
    if chat is None:
        return ExtractionOutcome(skipped=True, reason="model_unavailable")
    return await extractor.extract_chunk(
        ctx=TaskContext(is_background_task=True),
        tenant_id=tenant_id,
        chunk_id=chunk_id,
        chat=chat,
        knowledge_id=knowledge_id,
        chunk_index=chunk_index,
    )


def _serialise_outcome(outcome: ExtractionOutcome) -> dict[str, JsonValue]:
    """Project an :class:`ExtractionOutcome` onto a JSON-serialisable dict."""
    return {
        "skipped": outcome.skipped,
        "reason": outcome.reason,
        "node_count": outcome.node_count,
        "relation_count": outcome.relation_count,
    }


def parse_payload(payload: Mapping[str, JsonValue]) -> ChunkExtractTaskPayload:
    """Validate a raw payload mapping into the task payload model.

    Exposed for tests and the wiring layer so the handler and its
    callers agree on the schema.
    """
    return ChunkExtractTaskPayload.model_validate(dict(payload))


__all__ = [
    "TASK_NAME",
    "ChatResolver",
    "ChunkExtractTaskPayload",
    "parse_payload",
    "task_chunk_extract",
]
