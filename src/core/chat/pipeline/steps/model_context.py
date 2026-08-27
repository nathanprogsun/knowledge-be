"""Model-facing message preparation for the completion steps.

Ports the shared ``prepareChatModel`` / ``prepareMessagesWithHistory`` /
``withPromptCacheMetadata`` wiring into step-local helpers that talk
directly to the LLM layer: building the sampling options from the run
config, assembling system + history + user messages, encoding retrieval
handles through the request-scoped ``Registry``, and tagging the model
call with cache-observability labels.

The pipeline carrier types and the LLM wire types differ (frozen dataclass
vs. pydantic message), so the conversion helpers live here too.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Protocol, runtime_checkable

from src.ai.llm.prompt_cache import prompt_prefix_fingerprint
from src.ai.llm.types import Chat, ChatOptions, Message
from src.ai.llm.types import SearchResult as LLMSearchResult
from src.ai.llm.usage import with_llm_call_metadata
from src.common.json import JsonValue
from src.core.agents.engine.modelcontext.model_output import ToolResult
from src.core.agents.engine.modelcontext.registry import Registry
from src.core.chat.pipeline.common import (
    ChatMessage,
    pipeline_info,
    prepare_messages_with_history,
)
from src.core.chat.pipeline.context import PipelineContext
from src.core.chat.pipeline.steps.passage import (
    CHUNK_TYPE_FAQ,
    CHUNK_TYPE_WEB_SEARCH,
    get_enriched_passage_for_chat,
)
from src.core.chat.pipeline.types import Context, SearchResult

#: Purpose label attached to knowledge-QA model calls for cache observability.
KNOWLEDGE_QA_PURPOSE = "knowledge_qa"


@runtime_checkable
class ModelService(Protocol):
    """Resolves chat models by id for the completion steps."""

    async def get_chat_model(self, ctx: Context, model_id: str) -> Chat: ...


async def prepare_chat_model_for_step(
    ctx: Context,
    model_service: ModelService,
    pipeline_ctx: PipelineContext,
) -> tuple[Chat, ChatOptions]:
    """Resolve the chat model and build the sampling options for the turn.

    Raises the underlying service error on failure; callers wrap it into
    ``ERR_GET_CHAT_MODEL``.
    """
    chat_model = await model_service.get_chat_model(ctx, pipeline_ctx.chat_model_id)
    config = pipeline_ctx.summary_config
    options = ChatOptions(
        temperature=config.temperature,
        top_p=config.top_p,
        seed=config.seed,
        max_tokens=config.max_tokens,
        max_completion_tokens=config.max_completion_tokens,
        frequency_penalty=config.frequency_penalty,
        presence_penalty=config.presence_penalty,
        thinking=config.thinking,
    )
    if options.thinking is not None:
        pipeline_info("Stream", "thinking_option", {"enabled": options.thinking})
    return chat_model, options


def prepare_messages_with_model_context(
    pipeline_ctx: PipelineContext,
) -> tuple[list[Message], Registry]:
    """Assemble the model-facing message list and encode retrieval handles.

    The system prompt gains the registry's handle protocol, and the merged
    retrieval context is replaced by the compact handle-encoded rendering
    when citations are enabled. Returns the encoded-ready messages plus the
    request-scoped registry used to decode the response.
    """
    registry = Registry(pipeline_ctx.citations_enabled())
    llm_messages = [
        chat_message_to_llm(message) for message in prepare_messages_with_history(pipeline_ctx)
    ]
    if llm_messages:
        first = llm_messages[0]
        content = first.content.rstrip(" \t\r\n") + registry.protocol_prompt()
        llm_messages[0] = first.model_copy(update={"content": content})
    if not pipeline_ctx.merge_result or not llm_messages:
        return llm_messages, registry

    ordered = ordered_pipeline_references(pipeline_ctx)
    knowledge_results: list[LLMSearchResult] = []
    knowledge_rows: list[JsonValue] = []
    web_rows: list[JsonValue] = []
    for result in ordered:
        if is_pipeline_web_reference(result):
            web_rows.append(
                {
                    "url": result.id,
                    "title": first_pipeline_title(result),
                    "snippet": result.content,
                    "published_at": result.metadata.get("published_at"),
                }
            )
            continue
        knowledge_results.append(pipeline_search_result_to_llm(result))
        knowledge_rows.append(
            {
                "chunk_id": result.id,
                "knowledge_id": result.knowledge_id,
                "knowledge_base_id": result.knowledge_base_id,
                "knowledge_title": first_pipeline_title(result),
                "chunk_index": result.chunk_index,
                "chunk_type": result.chunk_type,
                "content": get_enriched_passage_for_chat(result),
            }
        )
    registry.register_search_results(knowledge_results)

    context_parts: list[str] = []
    if knowledge_rows:
        context_parts.append(
            registry.model_tool_result(
                ToolResult(
                    success=True,
                    data={"display_type": "search_results", "results": knowledge_rows},
                )
            )
        )
    if web_rows:
        context_parts.append(
            registry.model_tool_result(
                ToolResult(
                    success=True,
                    data={
                        "display_type": "web_search_results",
                        "results": web_rows,
                    },
                )
            )
        )
    model_contexts = "\n".join(context_parts)
    if not model_contexts.strip():
        return llm_messages, registry

    rendered = pipeline_ctx.rendered_contexts
    last = len(llm_messages) - 1
    replaced = False
    for index in (0, last):
        message = llm_messages[index]
        if rendered and rendered in message.content:
            content = message.content.replace(rendered, model_contexts)
            llm_messages[index] = message.model_copy(update={"content": content})
            replaced = True
    if not replaced:
        last_message = llm_messages[last]
        content = model_contexts + "\n\n" + last_message.content
        llm_messages[last] = last_message.model_copy(update={"content": content})
    return llm_messages, registry


@contextmanager
def with_prompt_cache_metadata(
    messages: list[Message],
    options: ChatOptions,
    purpose: str,
) -> Iterator[None]:
    """Annotate the surrounding call scope with cache-observability labels."""
    prefix_fingerprint = prompt_prefix_fingerprint(messages, options)
    with with_llm_call_metadata(
        purpose=purpose,
        prefix_fingerprint=prefix_fingerprint,
    ):
        yield


def ordered_pipeline_references(pipeline_ctx: PipelineContext) -> list[SearchResult]:
    """Return merged references with FAQ hits first when FAQ priority is on."""
    if not pipeline_ctx.faq_priority_enabled:
        return list(pipeline_ctx.merge_result)
    faq = [result for result in pipeline_ctx.merge_result if result.chunk_type == CHUNK_TYPE_FAQ]
    non_faq = [
        result for result in pipeline_ctx.merge_result if result.chunk_type != CHUNK_TYPE_FAQ
    ]
    return [*faq, *non_faq]


def is_pipeline_web_reference(result: SearchResult) -> bool:
    """Return whether a merged hit is a web-search reference."""
    return result.chunk_type.lower() == CHUNK_TYPE_WEB_SEARCH or (
        result.knowledge_source.lower() == "web_search"
    )


def first_pipeline_title(result: SearchResult) -> str:
    """Return the display title of a hit (knowledge title, else filename)."""
    if result.knowledge_title:
        return result.knowledge_title
    return result.knowledge_filename


def chat_message_to_llm(message: ChatMessage) -> Message:
    """Convert the pipeline message shape to the LLM wire message."""
    return Message(role=message.role, content=message.content, images=list(message.images))


def pipeline_search_result_to_llm(result: SearchResult) -> LLMSearchResult:
    """Convert a pipeline search hit to the LLM-layer search hit."""
    return LLMSearchResult(**result.model_dump())


__all__ = [
    "KNOWLEDGE_QA_PURPOSE",
    "ModelService",
    "chat_message_to_llm",
    "first_pipeline_title",
    "is_pipeline_web_reference",
    "ordered_pipeline_references",
    "pipeline_search_result_to_llm",
    "prepare_chat_model_for_step",
    "prepare_messages_with_model_context",
    "with_prompt_cache_metadata",
]
