"""Streaming chat-completion step (upstream ``PluginChatCompletionStream``).

Resolves the chat model, prepares the model-facing messages with retrieval
handles encoded, and streams the completion through the request-scoped
event bus: reasoning chunks are forwarded as ``thought`` events and answer
tokens as ``final_answer`` events, matching the agent streaming contract.
The answer is persisted downstream from those events, so the step only
emits and returns.

Resource handles split across provider chunks are bridged by the stream
decoders; their tails are flushed on stream end and on cancellation.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Protocol, runtime_checkable
from uuid import uuid4

from src.ai.llm.types import ResponseType, StreamResponse
from src.core.agents.engine.modelcontext.registry import Registry
from src.core.chat.bus import Event
from src.core.chat.pipeline.common import pipeline_error, pipeline_info
from src.core.chat.pipeline.context import PipelineContext
from src.core.chat.pipeline.engine import (
    ERR_GET_CHAT_MODEL,
    ERR_MODEL_CALL,
    Next,
    PluginError,
)
from src.core.chat.pipeline.steps.model_context import (
    KNOWLEDGE_QA_PURPOSE,
    ModelService,
    prepare_chat_model_for_step,
    prepare_messages_with_model_context,
    with_prompt_cache_metadata,
)
from src.core.chat.pipeline.types import Context, EventType
from src.core.chat.types import EventType as ChatEventType

#: Stage label surfaced in error events emitted for failed stream chunks.
_STREAM_STAGE = "chat_completion_stream"


@runtime_checkable
class StreamBus(Protocol):
    """The event-bus surface the stream step emits through."""

    async def emit(self, event: Event) -> None: ...


class ChatCompletionStreamStep:
    """Runs the streaming chat completion stage of the pipeline."""

    def __init__(self, model_service: ModelService, event_bus: StreamBus) -> None:
        self._model_service = model_service
        self._event_bus = event_bus

    def activation_events(self) -> Sequence[EventType]:
        return (EventType.CHAT_COMPLETION_STREAM,)

    async def on_event(
        self,
        ctx: Context,
        event_type: EventType | str,
        pipeline_ctx: PipelineContext,
        next: Next,
    ) -> PluginError | None:
        pipeline_info(
            "Stream",
            "input",
            {
                "session_id": pipeline_ctx.session_id,
                "user_question": pipeline_ctx.user_content,
                "history_rounds": len(pipeline_ctx.history),
                "chat_model": pipeline_ctx.chat_model_id,
            },
        )

        try:
            chat_model, options = await prepare_chat_model_for_step(
                ctx, self._model_service, pipeline_ctx
            )
        except Exception as exc:
            return ERR_GET_CHAT_MODEL.with_error(exc)

        chat_messages, model_context = prepare_messages_with_model_context(pipeline_ctx)
        chat_messages = model_context.encode_messages(chat_messages)

        pipeline_info(
            "Stream",
            "messages_ready",
            {
                "message_count": len(chat_messages),
                "system_prompt": chat_messages[0].content if chat_messages else "",
            },
        )
        pipeline_info(
            "Stream",
            "user_message",
            {
                "content": chat_messages[-1].content if chat_messages else "",
            },
        )

        if self._event_bus is None:
            pipeline_error(
                "Stream",
                "eventbus_missing",
                {"session_id": pipeline_ctx.session_id},
            )
            return ERR_MODEL_CALL.with_error(
                RuntimeError("EventBus is required for streaming")
            )

        pipeline_info(
            "Stream",
            "model_call",
            {"chat_model": pipeline_ctx.chat_model_id},
        )
        try:
            with with_prompt_cache_metadata(
                chat_messages, options, KNOWLEDGE_QA_PURPOSE
            ):
                stream = chat_model.chat_stream(chat_messages, options)
        except Exception as exc:
            pipeline_error(
                "Stream",
                "model_call",
                {
                    "chat_model": pipeline_ctx.chat_model_id,
                    "error": str(exc),
                },
            )
            return ERR_MODEL_CALL.with_error(exc)
        if stream is None:
            pipeline_error(
                "Stream",
                "model_call",
                {
                    "chat_model": pipeline_ctx.chat_model_id,
                    "error": "nil_channel",
                },
            )
            return ERR_MODEL_CALL.with_error(
                RuntimeError("chat stream returned nil channel")
            )

        pipeline_info(
            "Stream",
            "model_started",
            {"session_id": pipeline_ctx.session_id},
        )
        await self._emit_stream_events(
            ctx,
            pipeline_ctx,
            model_context,
            stream,
            self._event_bus,
        )
        return await next()

    async def _emit_stream_events(
        self,
        ctx: Context,
        pipeline_ctx: PipelineContext,
        model_context: Registry,
        stream: AsyncIterator[StreamResponse],
        event_bus: StreamBus,
    ) -> None:
        """Consume ``stream`` and forward reasoning / answer / error events."""
        thinking_decoder = model_context.stream_decoder()
        answer_decoder = model_context.stream_decoder()
        thinking_id = f"{uuid4().hex[:8]}-thinking"
        answer_id = f"{uuid4().hex[:8]}-answer"
        thinking_open = False
        answer_completed = False

        async def close_thinking() -> None:
            nonlocal thinking_open
            if not thinking_open:
                return
            await event_bus.emit(
                Event(
                    type=ChatEventType.AGENT_THOUGHT,
                    session_id=pipeline_ctx.session_id,
                    id=thinking_id,
                    data={"content": "", "done": True},
                )
            )
            thinking_open = False

        async def flush_decoders() -> None:
            thinking_tail = thinking_decoder.flush()
            if thinking_tail:
                await event_bus.emit(
                    Event(
                        type=ChatEventType.AGENT_THOUGHT,
                        session_id=pipeline_ctx.session_id,
                        id=thinking_id,
                        data={"content": thinking_tail},
                    )
                )
            answer_tail = answer_decoder.flush()
            if answer_tail:
                await event_bus.emit(
                    Event(
                        type=ChatEventType.AGENT_FINAL_ANSWER,
                        session_id=pipeline_ctx.session_id,
                        id=answer_id,
                        data={"content": answer_tail},
                    )
                )

        try:
            async for response in stream:
                if response.response_type == ResponseType.ERROR:
                    pipeline_error(
                        "Stream",
                        "stream_error",
                        {
                            "session_id": pipeline_ctx.session_id,
                            "error": response.content,
                        },
                    )
                    await event_bus.emit(
                        Event(
                            type=ChatEventType.ERROR,
                            session_id=pipeline_ctx.session_id,
                            id=f"{uuid4().hex[:8]}-error",
                            data={
                                "error": response.content,
                                "stage": _STREAM_STAGE,
                                "session_id": pipeline_ctx.session_id,
                            },
                        )
                    )
                    continue

                if response.response_type == ResponseType.THINKING:
                    content = thinking_decoder.feed(response.content)
                    if response.done:
                        content += thinking_decoder.flush()
                    if content:
                        thinking_open = True
                        await event_bus.emit(
                            Event(
                                type=ChatEventType.AGENT_THOUGHT,
                                session_id=pipeline_ctx.session_id,
                                id=thinking_id,
                                data={"content": content, "done": False},
                            )
                        )
                    if response.done:
                        await close_thinking()
                    continue

                if response.response_type == ResponseType.ANSWER:
                    # Providers can emit a completion once for the finish reason
                    # and again for an EOF sentinel; a final answer is terminal
                    # for one stream, so duplicates are dropped.
                    if answer_completed:
                        continue
                    content = answer_decoder.feed(response.content)
                    if response.done:
                        content += answer_decoder.flush()
                        answer_completed = True
                    await close_thinking()
                    await event_bus.emit(
                        Event(
                            type=ChatEventType.AGENT_FINAL_ANSWER,
                            session_id=pipeline_ctx.session_id,
                            id=answer_id,
                            data={"content": content, "done": response.done},
                        )
                    )
        finally:
            # The decoder tails are drained on both normal stream end and
            # cancellation so a resource reference split across chunks is never
            # dropped silently.
            await flush_decoders()
            await close_thinking()


__all__ = [
    "ChatCompletionStreamStep",
    "StreamBus",
]
