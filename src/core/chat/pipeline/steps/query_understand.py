"""Query-understanding pipeline step (upstream ``PluginQueryUnderstand``).

Rewrites the user query and classifies its intent with a chat model,
falling back to a vision model when the turn carries images. The
rewritten query and intent drive the downstream pipeline stages; a
generated image caption is persisted back to the stored user message so
later turns see it in history.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from src.ai.llm.types import Chat, ChatOptions, Message
from src.ai.llm.usage import with_llm_call_metadata
from src.common.json import JsonValue
from src.core.chat.pipeline.common import (
    StoredMessage,
    build_attachments_prompt,
    load_and_process_history,
    pipeline_error,
    pipeline_info,
    pipeline_warn,
    render_prompt_placeholders,
)
from src.core.chat.pipeline.context import PipelineContext
from src.core.chat.pipeline.engine import Next, PluginError
from src.core.chat.pipeline.types import (
    Context,
    EventType,
    History,
    MessageImage,
    QueryIntent,
)

#: Token budget for the rewrite completion; images need the larger budget.
_MAX_TEXT_TOKENS = 150
_MAX_IMAGE_TOKENS = 500


class ChatModelService(Protocol):
    """Resolves chat-capable models by id (structural seam)."""

    async def get_chat_model(self, ctx: Context, model_id: str) -> Chat: ...


class MessageStore(Protocol):
    """Message storage surface the step consumes (structural seam)."""

    async def get_recent_messages_by_session(
        self,
        ctx: Context,
        session_id: str,
        count: int,
    ) -> Sequence[StoredMessage]: ...

    async def get_message(
        self,
        ctx: Context,
        session_id: str,
        user_message_id: str,
    ) -> StoredMessage: ...

    async def update_message_images(
        self,
        ctx: Context,
        session_id: str,
        user_message_id: str,
        images: Sequence[MessageImage],
    ) -> None: ...


class QueryUnderstandPlugin:
    """Rewrites + classifies the user query, optionally describing images."""

    def __init__(
        self,
        *,
        model_service: ChatModelService,
        message_service: MessageStore,
        rewrite_prompt_system: str = "",
        rewrite_prompt_user: str = "",
        intent_system_prompts: Mapping[str, str] | None = None,
    ) -> None:
        self._model_service = model_service
        self._message_service = message_service
        self._rewrite_prompt_system = rewrite_prompt_system
        self._rewrite_prompt_user = rewrite_prompt_user
        self._intent_system_prompts = dict(intent_system_prompts or {})
        #: Fire-and-forget caption persistence; kept referenced so the task
        #: is not collected before it completes.
        self._background_tasks: list[asyncio.Task[None]] = []

    def activation_events(self) -> list[EventType]:
        return [EventType.QUERY_UNDERSTAND]

    async def on_event(
        self,
        ctx: Context,
        event_type: EventType | str,
        pipeline_ctx: PipelineContext,
        next: Next,
    ) -> PluginError | None:
        pipeline_ctx.rewrite_query = pipeline_ctx.query

        has_images = bool(pipeline_ctx.images)
        if not pipeline_ctx.enable_rewrite and not has_images:
            pipeline_info(
                "QueryUnderstand",
                "skip",
                {"session_id": pipeline_ctx.session_id, "reason": "rewrite_disabled_no_images"},
            )
            return await next()

        pipeline_info(
            "QueryUnderstand",
            "input",
            {
                "session_id": pipeline_ctx.session_id,
                "tenant_id": pipeline_ctx.tenant_id,
                "user_query": pipeline_ctx.query,
                "has_images": has_images,
                "enable_rewrite": pipeline_ctx.enable_rewrite,
            },
        )

        history_list = list(pipeline_ctx.history)
        if not history_list:
            history_list = await self._load_history(ctx, pipeline_ctx)
        else:
            pipeline_info(
                "QueryUnderstand",
                "history_reused",
                {"session_id": pipeline_ctx.session_id, "rounds": len(history_list)},
            )

        rewrite_model, use_images = await self._select_model(ctx, pipeline_ctx, has_images)
        if rewrite_model is None:
            pipeline_error(
                "QueryUnderstand",
                "get_model",
                {"session_id": pipeline_ctx.session_id, "chat_model_id": pipeline_ctx.chat_model_id},
            )
            return await next()

        system_content, user_content = self._build_prompts(pipeline_ctx, history_list)
        messages = [Message(role="system", content=system_content)]
        user_message = Message(role="user", content=user_content)
        if use_images:
            user_message = Message(role="user", content=user_content, images=list(pipeline_ctx.images))
        messages.append(user_message)

        max_tokens = _MAX_IMAGE_TOKENS if use_images else _MAX_TEXT_TOKENS
        try:
            with with_llm_call_metadata(purpose="query_rewrite"):
                response = await rewrite_model.chat(
                    messages,
                    ChatOptions(temperature=0.3, max_completion_tokens=max_tokens, thinking=False),
                )
        except Exception as exc:
            pipeline_error(
                "QueryUnderstand",
                "model_call",
                {"session_id": pipeline_ctx.session_id, "error": str(exc)},
            )
            return await next()

        self._parse_output(pipeline_ctx, response.content)

        # Persist the image description asynchronously — this DB write does
        # not affect the current pipeline result.
        if pipeline_ctx.image_description and pipeline_ctx.user_message_id:
            task = asyncio.create_task(self._update_user_message_image_caption(ctx, pipeline_ctx))
            self._background_tasks.append(task)

        if not pipeline_ctx.needs_retrieval() and self._apply_intent_prompt_override(pipeline_ctx):
            pipeline_info(
                "QueryUnderstand",
                "prompt_override",
                {"session_id": pipeline_ctx.session_id, "intent": str(pipeline_ctx.intent)},
            )

        pipeline_info(
            "QueryUnderstand",
            "output",
            {
                "session_id": pipeline_ctx.session_id,
                "rewrite_query": pipeline_ctx.rewrite_query,
                "intent": str(pipeline_ctx.intent),
                "has_image_desc": pipeline_ctx.image_description != "",
                "has_prompt_override": pipeline_ctx.system_prompt_override != "",
                "original_output": response.content,
            },
        )
        return await next()

    async def _load_history(self, ctx: Context, pipeline_ctx: PipelineContext) -> list[History]:
        """Fetch and process conversation history for the rewrite context.

        ``max_rounds <= 0`` is set explicitly when multi-turn is disabled;
        it must not fall back to the global default, otherwise rewrite would
        still pull old turns into the context.
        """
        if pipeline_ctx.max_rounds <= 0:
            return []
        try:
            history_list = await load_and_process_history(
                ctx,
                self._message_service,
                pipeline_ctx.session_id,
                pipeline_ctx.max_rounds,
                20,
            )
        except Exception as exc:
            pipeline_warn(
                "QueryUnderstand",
                "history_fetch",
                {"session_id": pipeline_ctx.session_id, "error": str(exc)},
            )
            return []
        pipeline_ctx.history = history_list
        if history_list:
            pipeline_info(
                "QueryUnderstand",
                "history_ready",
                {"session_id": pipeline_ctx.session_id, "history_rounds": len(history_list)},
            )
        return history_list

    async def _select_model(
        self,
        ctx: Context,
        pipeline_ctx: PipelineContext,
        has_images: bool,
    ) -> tuple[Chat | None, bool]:
        """Pick the rewrite model, preferring a vision-capable model for images."""
        if has_images:
            if pipeline_ctx.chat_model_supports_vision:
                try:
                    model = await self._model_service.get_chat_model(ctx, pipeline_ctx.chat_model_id)
                    return model, True
                except Exception as exc:
                    pipeline_warn(
                        "QueryUnderstand",
                        "vision_model_fallback",
                        {"session_id": pipeline_ctx.session_id, "error": str(exc)},
                    )
            if pipeline_ctx.vlm_model_id:
                try:
                    model = await self._model_service.get_chat_model(ctx, pipeline_ctx.vlm_model_id)
                    return model, True
                except Exception as exc:
                    pipeline_warn(
                        "QueryUnderstand",
                        "vlm_model_fallback",
                        {
                            "session_id": pipeline_ctx.session_id,
                            "vlm_model_id": pipeline_ctx.vlm_model_id,
                            "error": str(exc),
                        },
                    )
            pipeline_warn("QueryUnderstand", "no_vision_model", {"session_id": pipeline_ctx.session_id})

        text_model_id = pipeline_ctx.chat_model_id
        if pipeline_ctx.query_understand_model_id:
            text_model_id = pipeline_ctx.query_understand_model_id
        try:
            model = await self._model_service.get_chat_model(ctx, text_model_id)
            return model, False
        except Exception as exc:
            if not (
                pipeline_ctx.query_understand_model_id and text_model_id != pipeline_ctx.chat_model_id
            ):
                pipeline_error(
                    "QueryUnderstand",
                    "get_model",
                    {"session_id": pipeline_ctx.session_id, "chat_model_id": text_model_id, "error": str(exc)},
                )
                return None, False
            pipeline_warn(
                "QueryUnderstand",
                "query_understand_model_fallback",
                {
                    "session_id": pipeline_ctx.session_id,
                    "query_understand_model_id": pipeline_ctx.query_understand_model_id,
                    "error": str(exc),
                },
            )
            try:
                fallback = await self._model_service.get_chat_model(ctx, pipeline_ctx.chat_model_id)
                return fallback, False
            except Exception as fb_exc:
                pipeline_error(
                    "QueryUnderstand",
                    "get_model",
                    {"session_id": pipeline_ctx.session_id, "chat_model_id": pipeline_ctx.chat_model_id, "error": str(fb_exc)},
                )
                return None, False

    def _build_prompts(
        self,
        pipeline_ctx: PipelineContext,
        history_list: Sequence[History],
    ) -> tuple[str, str]:
        """Construct system and user prompts with placeholder replacement."""
        user_prompt = pipeline_ctx.rewrite_prompt_user or self._rewrite_prompt_user
        system_prompt = pipeline_ctx.rewrite_prompt_system or self._rewrite_prompt_system

        conversation_text = format_conversation_history(history_list)

        query_content = pipeline_ctx.query
        if pipeline_ctx.images:
            query_content += f'\n\n<images_uploaded count="{len(pipeline_ctx.images)}" />'
        else:
            query_content += "\n\n<no_image_attached />"
        if pipeline_ctx.attachments:
            query_content += build_attachments_prompt(pipeline_ctx.attachments)
        else:
            query_content += "\n<no_document_attached />"

        values = {
            "conversation": conversation_text,
            "query": query_content,
            "language": pipeline_ctx.language,
        }
        return (
            render_prompt_placeholders(system_prompt, values),
            render_prompt_placeholders(user_prompt, values),
        )

    def _parse_output(self, pipeline_ctx: PipelineContext, raw: str) -> None:
        """Extract rewrite / intent / image description from the model output."""
        content = raw.strip()
        if not content:
            return
        output = parse_structured_query_output(content)
        if output is not None:
            rewrite = output.rewrite_query.strip()
            if rewrite:
                pipeline_ctx.rewrite_query = rewrite
            pipeline_ctx.intent = coerce_query_intent(output.intent)
            pipeline_ctx.image_description = output.image_description.strip()
            return
        # JSON parsing failed entirely: treat the raw text as the rewritten
        # query and leave the intent unclassified (retrieval for safety).
        pipeline_ctx.rewrite_query = content

    def _apply_intent_prompt_override(self, pipeline_ctx: PipelineContext) -> bool:
        """Resolve the system-prompt override for the current intent.

        Agent-level overrides take precedence; whitespace-only agent
        overrides are treated as unset and fall through to the global map.
        """
        intent_key = pipeline_ctx.intent.value if pipeline_ctx.intent is not None else ""
        agent_override = pipeline_ctx.intent_prompt_overrides.get(intent_key, "")
        if agent_override.strip():
            pipeline_ctx.system_prompt_override = agent_override
        if not pipeline_ctx.system_prompt_override:
            global_override = self._intent_system_prompts.get(intent_key, "")
            if global_override:
                pipeline_ctx.system_prompt_override = global_override
        return pipeline_ctx.system_prompt_override != ""

    async def _update_user_message_image_caption(
        self,
        ctx: Context,
        pipeline_ctx: PipelineContext,
    ) -> None:
        """Write the generated image description back to the stored user message."""
        try:
            message = await self._message_service.get_message(
                ctx,
                pipeline_ctx.session_id,
                pipeline_ctx.user_message_id,
            )
        except Exception as exc:
            pipeline_warn(
                "QueryUnderstand",
                "get_user_message",
                {
                    "session_id": pipeline_ctx.session_id,
                    "user_message_id": pipeline_ctx.user_message_id,
                    "error": str(exc),
                },
            )
            return
        if not message.images:
            return
        first = message.images[0]
        updated_images = [
            MessageImage(url=first.url, caption=pipeline_ctx.image_description),
            *list(message.images[1:]),
        ]
        try:
            await self._message_service.update_message_images(
                ctx,
                pipeline_ctx.session_id,
                pipeline_ctx.user_message_id,
                updated_images,
            )
        except Exception as exc:
            pipeline_warn(
                "QueryUnderstand",
                "update_image_caption",
                {
                    "session_id": pipeline_ctx.session_id,
                    "user_message_id": pipeline_ctx.user_message_id,
                    "error": str(exc),
                },
            )


def coerce_query_intent(value: str) -> QueryIntent | None:
    """Map a raw intent string to the enum, or ``None`` when unrecognized.

    The run carrier treats ``None`` as unclassified and defaults to KB
    retrieval for safety, mirroring the upstream empty-intent behaviour.
    """
    if not value:
        return None
    try:
        return QueryIntent(value)
    except ValueError:
        return None


def format_conversation_history(history_list: Sequence[History]) -> str:
    """Format conversation history for the prompt template."""
    if not history_list:
        return ""
    parts: list[str] = []
    for entry in history_list:
        parts.append("------BEGIN------\n")
        parts.append("User question: ")
        parts.append(entry.query)
        parts.append("\nAssistant answer: ")
        parts.append(entry.answer)
        parts.append("\n------END------\n")
    return "".join(parts)


@dataclass(frozen=True, slots=True)
class _StructuredQueryOutput:
    """Parsed query-understanding output (rewrite / intent / image text)."""

    rewrite_query: str = ""
    intent: str = ""
    image_description: str = ""


def parse_structured_query_output(raw: str) -> _StructuredQueryOutput | None:
    """Parse the model's structured JSON output, tolerating markdown wrappers.

    Returns ``None`` when the content is not a JSON object so callers fall
    back to treating the raw text as the rewritten query.
    """
    content = raw.strip()
    if not content:
        return None
    output = _parse_structured_query_output_json(content)
    if output is not None:
        return output
    start = content.find("{")
    end = content.rfind("}")
    if start < 0 or end <= start:
        return None
    return _parse_structured_query_output_json(content[start : end + 1])


def _parse_structured_query_output_json(content: str) -> _StructuredQueryOutput | None:
    """Parse ``content`` as a JSON object into structured output."""
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    rewrite_query = _first_string_field(obj, "rewrite_query", "rewritten_query", "query", "question")
    intent = _first_string_field(obj, "intent")
    description = _first_string_field(
        obj,
        "image_description",
        "image_desc",
        "image_text",
        "image_ocr_text",
        "description",
    )
    ocr = _first_string_field(obj, "ocr_text", "ocr", "full_ocr", "image_ocr", "ocr_content")
    image_description, is_set = merge_image_desc_and_ocr(description, ocr)
    return _StructuredQueryOutput(
        rewrite_query=rewrite_query.strip(),
        intent=intent.strip(),
        image_description=image_description.strip() if is_set else "",
    )


def _first_string_field(obj: Mapping[str, JsonValue], *keys: str) -> str:
    """Return the first string value among ``keys``, or ``""``."""
    for key in keys:
        value = obj.get(key)
        if isinstance(value, str):
            return value
    return ""


def merge_image_desc_and_ocr(description: str, ocr: str) -> tuple[str, bool]:
    """Combine a description with OCR text, deduplicating embedded copies."""
    if not description and not ocr:
        return "", False
    if not description:
        return ocr, True
    if not ocr:
        return description, True
    if ocr in description:
        return description, True
    return f"{description}\n\n[OCR]\n{ocr}", True


__all__ = [
    "ChatModelService",
    "MessageStore",
    "QueryUnderstandPlugin",
    "coerce_query_intent",
    "format_conversation_history",
    "merge_image_desc_and_ocr",
    "parse_structured_query_output",
]
