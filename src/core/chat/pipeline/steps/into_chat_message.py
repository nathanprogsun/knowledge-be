"""Context-render step (upstream ``PluginIntoChatMessage``).

Assembles the RAG-augmented user message: separates FAQ and document
results when FAQ priority is on, renders the merged context into the
configured ``context_template``, appends image / quoted / attachment
sections the current model cannot process natively, and stores the
rendered content back to the stored user message so later turns see the
full retrieval context in history.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from src.core.chat.pipeline.common import (
    build_attachments_prompt,
    pipeline_info,
    pipeline_warn,
    render_prompt_placeholders,
)
from src.core.chat.pipeline.context import PipelineContext
from src.core.chat.pipeline.engine import ERR_TEMPLATE_EXECUTE, Next, PluginError
from src.core.chat.pipeline.steps.passage import (
    CHUNK_TYPE_FAQ,
    build_document_header,
    get_enriched_passage_for_chat,
)
from src.core.chat.pipeline.types import Context, EventType, SearchResult
from src.core.knowledge.documents.create_common import validate_input

#: Text marker injected for image content when the model cannot see images.
_IMAGE_DESCRIPTION_PREFIX = "\n\n[用户上传图片内容]\n"

#: Human-readable label for the high-confidence FAQ log.
_HIGH_CONFIDENCE_FAQ_LABEL = "high_confidence_faq"


@runtime_checkable
class RenderedContentMessageService(Protocol):
    """Persists the rendered user message for later conversation turns."""

    async def update_message_rendered_content(
        self,
        ctx: Context,
        session_id: str,
        user_message_id: str,
        content: str,
    ) -> None: ...


class IntoChatMessageStep:
    """Renders the RAG context into the user message for the current turn."""

    def __init__(self, message_service: RenderedContentMessageService) -> None:
        self._message_service = message_service
        #: Strong references to fire-and-forget persistence tasks so they are
        #: never garbage-collected mid-flight.
        self._background_tasks: set[asyncio.Task[None]] = set()

    def activation_events(self) -> Sequence[EventType]:
        return (EventType.INTO_CHAT_MESSAGE,)

    async def on_event(
        self,
        ctx: Context,
        event_type: EventType | str,
        pipeline_ctx: PipelineContext,
        next: Next,
    ) -> PluginError | None:
        pipeline_info(
            "IntoChatMessage",
            "input",
            {
                "session_id": pipeline_ctx.session_id,
                "merge_result_cnt": len(pipeline_ctx.merge_result),
                "template_len": len(pipeline_ctx.summary_config.context_template),
            },
        )

        faq_results, doc_results, has_high_confidence_faq = self._separate_faq_results(pipeline_ctx)

        safe_query, is_valid = validate_input(pipeline_ctx.query)
        if not is_valid:
            pipeline_warn(
                "IntoChatMessage",
                "invalid_query",
                {"session_id": pipeline_ctx.session_id},
            )
            return ERR_TEMPLATE_EXECUTE.with_error(
                ValueError("user query contains invalid content")
            )

        if not pipeline_ctx.needs_retrieval():
            self._render_no_search(ctx, pipeline_ctx, safe_query)
            return await next()

        context_parts = self._render_contexts(
            pipeline_ctx,
            faq_results,
            doc_results,
            has_high_confidence_faq,
        )
        pipeline_ctx.rendered_contexts = "".join(context_parts)

        user_content = render_prompt_placeholders(
            pipeline_ctx.summary_config.context_template,
            {
                "query": safe_query,
                "contexts": pipeline_ctx.rendered_contexts,
                "language": pipeline_ctx.language,
            },
        )
        user_content = self._append_auxiliary_sections(pipeline_ctx, user_content)
        pipeline_ctx.user_content = user_content

        pipeline_info(
            "IntoChatMessage",
            "output",
            {
                "session_id": pipeline_ctx.session_id,
                "user_content_len": len(pipeline_ctx.user_content),
                "faq_priority": pipeline_ctx.faq_priority_enabled,
                "intent": str(pipeline_ctx.intent or ""),
                "image_description": pipeline_ctx.image_description,
                "chat_model_supports_vision": pipeline_ctx.chat_model_supports_vision,
            },
        )

        self.persist_rendered_content(ctx, pipeline_ctx)
        return await next()

    def _separate_faq_results(
        self,
        pipeline_ctx: PipelineContext,
    ) -> tuple[list[SearchResult], list[SearchResult], bool]:
        """Split merged results into FAQ / document buckets when enabled."""
        faq_results: list[SearchResult] = []
        doc_results: list[SearchResult] = []
        has_high_confidence_faq = False
        if not pipeline_ctx.faq_priority_enabled:
            return faq_results, doc_results, has_high_confidence_faq
        for result in pipeline_ctx.merge_result:
            if result.chunk_type == CHUNK_TYPE_FAQ:
                faq_results.append(result)
                if (
                    result.score >= pipeline_ctx.faq_direct_answer_threshold
                    and not has_high_confidence_faq
                ):
                    has_high_confidence_faq = True
                    pipeline_info(
                        "IntoChatMessage",
                        _HIGH_CONFIDENCE_FAQ_LABEL,
                        {
                            "chunk_id": result.id,
                            "score": f"{result.score:.4f}",
                            "threshold": pipeline_ctx.faq_direct_answer_threshold,
                        },
                    )
            else:
                doc_results.append(result)
        pipeline_info(
            "IntoChatMessage",
            "faq_separation",
            {
                "faq_count": len(faq_results),
                "doc_count": len(doc_results),
                "has_high_confidence": has_high_confidence_faq,
            },
        )
        return faq_results, doc_results, has_high_confidence_faq

    def _render_no_search(
        self,
        ctx: Context,
        pipeline_ctx: PipelineContext,
        safe_query: str,
    ) -> None:
        """Render the no-retrieval user content through the context template."""
        user_content = safe_query
        rewrite = pipeline_ctx.rewrite_query.strip()
        if rewrite:
            safe_rewrite, ok = validate_input(rewrite)
            if ok:
                user_content = safe_rewrite
            else:
                pipeline_warn(
                    "IntoChatMessage",
                    "invalid_rewrite_query_fallback",
                    {"session_id": pipeline_ctx.session_id},
                )
        user_content = self._append_auxiliary_sections(pipeline_ctx, user_content)

        template = pipeline_ctx.summary_config.context_template
        if template:
            pipeline_ctx.user_content = render_prompt_placeholders(
                template,
                {
                    "query": user_content,
                    "contexts": "",
                    "language": pipeline_ctx.language,
                },
            )
        else:
            pipeline_ctx.user_content = user_content
        pipeline_info(
            "IntoChatMessage",
            "no_search_with_template",
            {
                "session_id": pipeline_ctx.session_id,
                "user_content_len": len(pipeline_ctx.user_content),
                "has_template": pipeline_ctx.summary_config.context_template != "",
            },
        )

    def _append_auxiliary_sections(
        self,
        pipeline_ctx: PipelineContext,
        user_content: str,
    ) -> str:
        """Append image / quoted-context / attachment sections the model needs."""
        if pipeline_ctx.image_description and not pipeline_ctx.chat_model_supports_vision:
            user_content += _IMAGE_DESCRIPTION_PREFIX + pipeline_ctx.image_description
        if pipeline_ctx.quoted_context:
            user_content += "\n\n" + pipeline_ctx.quoted_context
        if pipeline_ctx.attachments:
            user_content += build_attachments_prompt(pipeline_ctx.attachments)
        return user_content

    def _render_contexts(
        self,
        pipeline_ctx: PipelineContext,
        faq_results: list[SearchResult],
        doc_results: list[SearchResult],
        has_high_confidence_faq: bool,
    ) -> list[str]:
        """Build the ``<context>`` XML section from the merged results."""
        parts: list[str] = []
        all_results = pipeline_ctx.merge_result
        if pipeline_ctx.faq_priority_enabled and faq_results:
            all_results = [*faq_results, *doc_results]
        document_header = build_document_header(all_results)
        if document_header:
            parts.append(document_header)
            parts.append("\n")

        if pipeline_ctx.faq_priority_enabled and faq_results:
            parts.append('<source type="faq" priority="high">\n')
            for index, result in enumerate(faq_results):
                passage = get_enriched_passage_for_chat(result)
                if has_high_confidence_faq and index == 0:
                    parts.append(
                        f'<context id="FAQ-{index + 1}" match="exact">{passage}</context>\n'
                    )
                else:
                    parts.append(f'<context id="FAQ-{index + 1}">{passage}</context>\n')
            parts.append("</source>\n")
            if doc_results:
                parts.append('<source type="document" priority="supplementary">\n')
                for index, result in enumerate(doc_results):
                    passage = get_enriched_passage_for_chat(result)
                    parts.append(f'<context id="DOC-{index + 1}">{passage}</context>\n')
                parts.append("</source>")
        else:
            for index, result in enumerate(pipeline_ctx.merge_result):
                passage = get_enriched_passage_for_chat(result)
                if index > 0:
                    parts.append("\n")
                parts.append(f'<context id="{index + 1}">{passage}</context>')
        return parts

    def persist_rendered_content(
        self,
        ctx: Context,
        pipeline_ctx: PipelineContext,
    ) -> None:
        """Schedule persisting the rendered user content to the stored message."""
        if pipeline_ctx.user_message_id == "" or pipeline_ctx.user_content == "":
            pipeline_info(
                "IntoChatMessage",
                "persist_rendered_content_skip",
                {
                    "session_id": pipeline_ctx.session_id,
                    "user_message_id": pipeline_ctx.user_message_id,
                    "has_user_content": pipeline_ctx.user_content != "",
                    "reason": "empty_id_or_content",
                },
            )
            return
        if pipeline_ctx.user_content == pipeline_ctx.query:
            return
        pipeline_info(
            "IntoChatMessage",
            "persist_rendered_content",
            {
                "session_id": pipeline_ctx.session_id,
                "user_message_id": pipeline_ctx.user_message_id,
                "rendered_content_len": len(pipeline_ctx.user_content),
            },
        )
        task = asyncio.create_task(self._update_rendered_content(ctx, pipeline_ctx))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _update_rendered_content(
        self,
        ctx: Context,
        pipeline_ctx: PipelineContext,
    ) -> None:
        """Best-effort persistence of the rendered content; failures are logged."""
        try:
            await self._message_service.update_message_rendered_content(
                ctx,
                pipeline_ctx.session_id,
                pipeline_ctx.user_message_id,
                pipeline_ctx.user_content,
            )
        except Exception as exc:
            pipeline_warn(
                "IntoChatMessage",
                "persist_rendered_content_error",
                {
                    "session_id": pipeline_ctx.session_id,
                    "user_message_id": pipeline_ctx.user_message_id,
                    "error": str(exc),
                },
            )


__all__ = [
    "IntoChatMessageStep",
    "RenderedContentMessageService",
]
