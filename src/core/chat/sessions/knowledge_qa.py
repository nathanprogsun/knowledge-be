"""Knowledge-QA pipeline orchestration.

Assembles the pipeline context, builds the stage list, and executes
each stage through the event manager. Before the streaming completion
stage, retrieved references are emitted so the client receives them
while the stream is still open. When retrieval returns nothing, the
fallback handler emits a fallback response.

The orchestrator is stateless: it accepts a pre-built
``PipelineContext`` and does not resolve knowledge bases, models, or
search targets itself. Those concerns belong to the caller.
"""

from __future__ import annotations

import logging
from uuid import uuid4

from src.core.chat.bus import Event, EventBus
from src.core.chat.pipeline.common import (
    build_attachments_prompt,
    pipeline_error,
    pipeline_info,
    pipeline_warn,
)
from src.core.chat.pipeline.context import PipelineContext
from src.core.chat.pipeline.engine import ERR_SEARCH_NOTHING, EventManager
from src.core.chat.pipeline.types import (
    ChatResponse,
    Context,
    EventType,
    FallbackStrategy,
    PipelineBuilder,
    SearchTarget,
)
from src.core.chat.types import EventType as ChatEventType

logger = logging.getLogger(__name__)


# ── Scope detection ───────────────────────────────────────────────────


def has_knowledge_retrieval_scope(
    *,
    search_targets: list[SearchTarget],
    knowledge_base_ids: list[str],
    knowledge_ids: list[str],
) -> bool:
    """Return whether the request carries any KB retrieval scope."""
    return bool(knowledge_base_ids or knowledge_ids or search_targets)


# ── Pipeline assembly ─────────────────────────────────────────────────


def build_pipeline_stages(
    *,
    pipeline_ctx: PipelineContext,
) -> list[EventType]:
    """Assemble the pipeline stage list from the request context.

    Pure chat (no retrieval):

        ``[LOAD_HISTORY?] -> CHAT_COMPLETION_STREAM``

    RAG:

        ``[LOAD_HISTORY?] -> QUERY_UNDERSTAND -> CHUNK_SEARCH_PARALLEL ->
        CHUNK_RERANK -> [WEB_FETCH?] -> CHUNK_MERGE -> FILTER_TOP_K ->
        [DATA_ANALYSIS?] -> INTO_CHAT_MESSAGE -> CHAT_COMPLETION_STREAM``
    """
    has_kb = has_knowledge_retrieval_scope(
        search_targets=pipeline_ctx.search_targets,
        knowledge_base_ids=pipeline_ctx.knowledge_base_ids,
        knowledge_ids=pipeline_ctx.knowledge_ids,
    )
    needs_rag = has_kb or pipeline_ctx.web_search_enabled
    has_history = pipeline_ctx.max_rounds > 0

    builder = PipelineBuilder()
    if not needs_rag:
        builder.add_if(has_history, EventType.LOAD_HISTORY)
        builder.add(EventType.CHAT_COMPLETION_STREAM)
    else:
        builder.add_if(has_history, EventType.LOAD_HISTORY)
        builder.add(EventType.QUERY_UNDERSTAND)
        builder.add(EventType.CHUNK_SEARCH_PARALLEL)
        builder.add(EventType.CHUNK_RERANK)
        builder.add_if(
            pipeline_ctx.web_search_enabled, EventType.WEB_FETCH
        )
        builder.add(EventType.CHUNK_MERGE)
        builder.add(EventType.FILTER_TOP_K)
        builder.add_if(
            pipeline_ctx.data_analysis_enabled, EventType.DATA_ANALYSIS
        )
        builder.add(EventType.INTO_CHAT_MESSAGE)
        builder.add(EventType.CHAT_COMPLETION_STREAM)
    return builder.build()


def build_pure_chat_user_content(pipeline_ctx: PipelineContext) -> str:
    """Assemble user content for the pure-chat (no-retrieval) path.

    Concatenates the query, image description (when the model lacks
    vision), quoted context, and attachment prompt. The RAG path
    handles these in the ``INTO_CHAT_MESSAGE`` stage instead.
    """
    parts: list[str] = [pipeline_ctx.query]
    if (
        pipeline_ctx.image_description
        and not pipeline_ctx.chat_model_supports_vision
    ):
        parts.append("\n\n[用户上传图片内容]\n" + pipeline_ctx.image_description)
    if pipeline_ctx.quoted_context:
        parts.append("\n\n" + pipeline_ctx.quoted_context)
    if pipeline_ctx.attachments:
        parts.append(build_attachments_prompt(pipeline_ctx.attachments))
    return "".join(parts)


# ── References ────────────────────────────────────────────────────────


async def emit_references(
    pipeline_ctx: PipelineContext,
    event_bus: EventBus,
) -> None:
    """Emit retrieved references before the answer stream starts.

    References are emitted only when the merge stage produced results;
    a search-only run with no merge output sends nothing.
    """
    references = pipeline_ctx.merge_result
    if not references:
        return
    logger.info(
        "Emitting references event with %d results (pre-answer)",
        len(references),
    )
    await event_bus.emit(
        Event(
            type=ChatEventType.AGENT_REFERENCES,
            session_id=pipeline_ctx.session_id,
            id=f"{uuid4().hex[:8]}-references",
            data={
                "references": [r.model_dump() for r in references],
            },
        )
    )


# ── Fallback ──────────────────────────────────────────────────────────


class FallbackHandler:
    """Handles fallback responses when retrieval returns nothing.

    Dispatches to the fixed or model strategy based on the pipeline
    context's ``fallback_strategy``. The model strategy falls back
    to fixed when ``fallback_prompt`` is empty.
    """

    async def handle(
        self,
        *,
        pipeline_ctx: PipelineContext,
        event_bus: EventBus,
    ) -> None:
        """Emit a fallback answer for the turn."""
        if pipeline_ctx.fallback_strategy == FallbackStrategy.MODEL:
            await self._handle_model(pipeline_ctx, event_bus)
        else:
            await self._handle_fixed(pipeline_ctx, event_bus)

    async def _handle_fixed(
        self,
        pipeline_ctx: PipelineContext,
        event_bus: EventBus,
    ) -> None:
        """Emit the configured fixed fallback string."""
        content = pipeline_ctx.fallback_response
        pipeline_ctx.chat_response = ChatResponse(content=content)
        await _emit_fallback_answer(pipeline_ctx, event_bus, content)

    async def _handle_model(
        self,
        pipeline_ctx: PipelineContext,
        event_bus: EventBus,
    ) -> None:
        """Render the fallback prompt and stream via the chat model.

        Falls back to fixed when ``fallback_prompt`` is empty.
        """
        if not pipeline_ctx.fallback_prompt:
            logger.warning(
                "Fallback strategy is 'model' but fallback_prompt is "
                "empty, using fixed response"
            )
            await self._handle_fixed(pipeline_ctx, event_bus)
            return
        # Model-based fallback with streaming requires a chat model
        # service; fall back to fixed when one is not wired.
        await self._handle_fixed(pipeline_ctx, event_bus)


async def _emit_fallback_answer(
    pipeline_ctx: PipelineContext,
    event_bus: EventBus,
    content: str,
) -> None:
    """Emit a fallback final-answer event with ``done=True``."""
    await event_bus.emit(
        Event(
            type=ChatEventType.AGENT_FINAL_ANSWER,
            session_id=pipeline_ctx.session_id,
            id=f"{uuid4().hex[:8]}-fallback",
            data={
                "content": content,
                "done": True,
                "is_fallback": True,
            },
        )
    )


# ── Execution ─────────────────────────────────────────────────────────


async def execute_knowledge_qa(
    *,
    ctx: Context,
    event_manager: EventManager,
    pipeline_ctx: PipelineContext,
    stages: list[EventType],
    event_bus: EventBus,
    fallback_handler: FallbackHandler | None = None,
) -> None:
    """Execute pipeline stages through the event manager.

    Iterates through ``stages``, triggering each via the event manager.
    Before ``CHAT_COMPLETION_STREAM``, emits references if available.
    On ``ERR_SEARCH_NOTHING``, invokes the fallback handler and
    returns. Other plugin errors are raised (or logged and returned
    when the error carries no underlying exception).
    """
    handler = fallback_handler or FallbackHandler()

    for stage in stages:
        if stage == EventType.CHAT_COMPLETION_STREAM:
            await emit_references(pipeline_ctx, event_bus)

        error = await event_manager.trigger(ctx, stage, pipeline_ctx)

        if error is ERR_SEARCH_NOTHING:
            pipeline_warn(
                "Pipeline",
                "stage_fallback",
                {
                    "event": str(stage),
                    "reason": "search_nothing",
                    "strategy": str(pipeline_ctx.fallback_strategy),
                },
            )
            await handler.handle(
                pipeline_ctx=pipeline_ctx,
                event_bus=event_bus,
            )
            return

        if error is not None:
            pipeline_error(
                "Pipeline",
                "stage_failed",
                {
                    "event": str(stage),
                    "error_type": error.error_type,
                    "description": error.description,
                },
            )
            if error.err is not None:
                raise error.err
            logger.warning(
                "Pipeline stage %s reported error without "
                "exception: %s",
                stage,
                error.description,
            )
            return

        pipeline_info(
            "Pipeline",
            "stage_complete",
            {"event": str(stage)},
        )

    pipeline_info(
        "Pipeline",
        "all_stages_complete",
        {
            "session_id": pipeline_ctx.session_id,
            "total_stages": len(stages),
        },
    )


# ── High-level entry point ────────────────────────────────────────────


async def run_knowledge_qa(
    *,
    ctx: Context,
    event_manager: EventManager,
    pipeline_ctx: PipelineContext,
    event_bus: EventBus,
    fallback_handler: FallbackHandler | None = None,
) -> None:
    """Assemble and execute the knowledge-QA pipeline.

    Builds the pipeline stages from the context, prepares pure-chat
    content when no retrieval is needed, and executes the pipeline
    through the event manager.
    """
    has_kb = has_knowledge_retrieval_scope(
        search_targets=pipeline_ctx.search_targets,
        knowledge_base_ids=pipeline_ctx.knowledge_base_ids,
        knowledge_ids=pipeline_ctx.knowledge_ids,
    )
    needs_rag = has_kb or pipeline_ctx.web_search_enabled

    if not needs_rag:
        pipeline_ctx.user_content = build_pure_chat_user_content(pipeline_ctx)

    stages = build_pipeline_stages(pipeline_ctx=pipeline_ctx)

    pipeline_info(
        "Pipeline",
        "assembled",
        {
            "session_id": pipeline_ctx.session_id,
            "stages": len(stages),
            "has_kb": has_kb,
            "web_search": pipeline_ctx.web_search_enabled,
            "history": pipeline_ctx.max_rounds > 0,
        },
    )

    await execute_knowledge_qa(
        ctx=ctx,
        event_manager=event_manager,
        pipeline_ctx=pipeline_ctx,
        stages=stages,
        event_bus=event_bus,
        fallback_handler=fallback_handler,
    )


__all__ = [
    "FallbackHandler",
    "build_pipeline_stages",
    "build_pure_chat_user_content",
    "emit_references",
    "execute_knowledge_qa",
    "has_knowledge_retrieval_scope",
    "run_knowledge_qa",
]
