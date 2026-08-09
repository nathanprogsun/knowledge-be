"""Chat-completion step (upstream ``PluginChatCompletion``).

Resolves the chat model, assembles the model-facing messages (including
conversation history) with retrieval handles encoded, runs the non-streaming
completion, decodes the response, and stores it on the run carrier for the
downstream persistence steps.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from src.ai.llm.types import ChatResponse as LLMChatResponse
from src.common.json import JsonValue
from src.core.chat.pipeline.common import pipeline_error, pipeline_info, pipeline_warn
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
from src.core.chat.pipeline.types import (
    ChatResponse,
    Context,
    EventType,
    TokenUsage,
)


class ChatCompletionStep:
    """Runs the non-streaming chat completion stage of the pipeline."""

    def __init__(self, model_service: ModelService) -> None:
        self._model_service = model_service

    def activation_events(self) -> Sequence[EventType]:
        return (EventType.CHAT_COMPLETION,)

    async def on_event(
        self,
        ctx: Context,
        event_type: EventType | str,
        pipeline_ctx: PipelineContext,
        next: Next,
    ) -> PluginError | None:
        pipeline_info(
            "Completion",
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

        pipeline_info(
            "Completion",
            "messages_ready",
            {"message_count": len(pipeline_ctx.history) + 2},
        )
        chat_messages, model_context = prepare_messages_with_model_context(pipeline_ctx)
        chat_messages = model_context.encode_messages(chat_messages)

        pipeline_info(
            "Completion",
            "model_call",
            {"chat_model": pipeline_ctx.chat_model_id},
        )
        try:
            with with_prompt_cache_metadata(
                chat_messages, options, KNOWLEDGE_QA_PURPOSE
            ):
                chat_response = await chat_model.chat(chat_messages, options)
        except Exception as exc:
            pipeline_error(
                "Completion",
                "model_call",
                {
                    "chat_model": pipeline_ctx.chat_model_id,
                    "error": str(exc),
                },
            )
            return ERR_MODEL_CALL.with_error(exc)

        model_context.decode_response(chat_response)
        orphans = model_context.orphan_resource_handles(chat_response.content)
        if orphans:
            pipeline_warn(
                "Completion",
                "orphan_resource_handles",
                {
                    "session_id": pipeline_ctx.session_id,
                    "handles": cast(JsonValue, orphans),
                },
            )

        pipeline_info(
            "Completion",
            "output",
            {
                "answer_preview": chat_response.content,
                "finish_reason": chat_response.finish_reason,
                "completion_tokens": chat_response.usage.completion_tokens,
                "prompt_tokens": chat_response.usage.prompt_tokens,
            },
        )
        pipeline_ctx.chat_response = to_pipeline_chat_response(chat_response)
        return await next()


def to_pipeline_chat_response(response: LLMChatResponse) -> ChatResponse:
    """Project an LLM completion onto the pipeline's frozen response model."""
    usage = response.usage
    return ChatResponse(
        content=response.content,
        reasoning_content=response.reasoning_content,
        finish_reason=response.finish_reason,
        usage=TokenUsage(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            cached_tokens=usage.cached_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            cache_miss_tokens=usage.cache_miss_tokens,
            cache_reported=usage.cache_reported,
            cache_status=str(usage.cache_status or ""),
        ),
    )


__all__ = [
    "ChatCompletionStep",
    "to_pipeline_chat_response",
]
