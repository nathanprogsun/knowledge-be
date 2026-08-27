"""Reference handling for the chat pipeline (upstream ``references.go``).

Replaces positional retrieval ids with request-local model handles. The
persisted rendered context stays unchanged; public citations are expanded
only when the request setting enables them.

``prepare_messages_with_model_context`` is the entry point: it rebuilds the
message list with the source-handling protocol prompt, registers every
knowledge hit on a fresh model-context registry, and injects the rendered
chunk / web context back into the system or user message. Pure helpers
(web-reference detection, FAQ-first ordering, title resolution, enriched
passage text) are exported separately so downstream steps and tests can
reuse them.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import replace
from typing import cast

from src.ai.llm.types import SearchResult as LlmSearchResult
from src.common.json import JsonObject, JsonValue
from src.core.agents.engine.modelcontext.model_output import ToolResult
from src.core.agents.engine.modelcontext.registry import Registry
from src.core.agents.tools.text_utils import parse_image_infos
from src.core.chat.pipeline.common import ChatMessage, prepare_messages_with_history
from src.core.chat.pipeline.context import PipelineContext
from src.core.chat.pipeline.types import SearchResult
from src.core.knowledge.chunks.types import CHUNK_TYPE_FAQ, CHUNK_TYPE_WEB_SEARCH

#: Matches a Markdown image link ``![alt](url)`` (shared shape with the
#: document parser so an image the parser stored can always be matched back).
_MARKDOWN_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

#: Matches an HTML ``<img>`` tag carrying a quoted ``src``. ``src`` must be
#: preceded by whitespace so ``data-src`` is not mistaken for it; an unquoted
#: ``src`` and a ``srcset``-only tag are deliberately out of scope.
_HTML_IMAGE_SRC_RE = re.compile(r'(?i)<img\b([^>]*?)\ssrc\s*=\s*[\'"]([^\'"]+)[\'"]([^>]*)>')

#: Submatch index of the ``src`` value in :data:`_HTML_IMAGE_SRC_RE`.
_HTML_IMAGE_SRC_URL_GROUP = 2


def _as_str(value: JsonValue) -> str:
    """Return ``value`` when it is a string, else the empty string."""
    return value if isinstance(value, str) else ""


def is_pipeline_web_reference(result: SearchResult) -> bool:
    """Return whether ``result`` is a web-search hit rather than a chunk."""
    return result.chunk_type.lower() == CHUNK_TYPE_WEB_SEARCH or (
        result.knowledge_source.lower() == "web_search"
    )


def ordered_pipeline_references(pipeline_ctx: PipelineContext) -> list[SearchResult]:
    """Return the merged references, FAQ chunks first when enabled.

    FAQ priority only reorders; it never drops or duplicates a reference.
    """
    merge_result = pipeline_ctx.merge_result
    if not pipeline_ctx.faq_priority_enabled:
        return list(merge_result)
    faq = [result for result in merge_result if result.chunk_type == CHUNK_TYPE_FAQ]
    others = [result for result in merge_result if result.chunk_type != CHUNK_TYPE_FAQ]
    return [*faq, *others]


def first_pipeline_title(result: SearchResult) -> str:
    """Return the display title for ``result``, falling back to its filename."""
    if result.knowledge_title != "":
        return result.knowledge_title
    return result.knowledge_filename


def _to_registry_search_result(result: SearchResult) -> LlmSearchResult:
    """Project a pipeline search hit onto the registry's search-result shape."""
    return LlmSearchResult(
        id=result.id,
        content=result.content,
        knowledge_id=result.knowledge_id,
        chunk_index=result.chunk_index,
        knowledge_title=result.knowledge_title,
        start_at=result.start_at,
        end_at=result.end_at,
        seq=result.seq,
        score=result.score,
        match_type=int(result.match_type),
        sub_chunk_id=list(result.sub_chunk_id),
        metadata=dict(result.metadata),
        chunk_type=result.chunk_type,
        parent_chunk_id=result.parent_chunk_id,
        image_info=result.image_info,
        knowledge_filename=result.knowledge_filename,
        knowledge_source=result.knowledge_source,
        knowledge_channel=result.knowledge_channel,
        chunk_metadata=result.chunk_metadata,
        matched_content=result.matched_content,
        knowledge_description=result.knowledge_description,
        knowledge_custom_metadata=result.knowledge_custom_metadata,
        knowledge_base_id=result.knowledge_base_id,
    )


def _image_info_map(infos: list[JsonObject]) -> dict[str, JsonObject]:
    """Index image records by both their URL and original URL."""
    result: dict[str, JsonObject] = {}
    for info in infos:
        url = _as_str(info.get("url"))
        if url:
            result[url] = info
        original_url = _as_str(info.get("original_url"))
        if original_url:
            result[original_url] = info
    return result


def _image_info_markdown_metadata(info: JsonObject) -> str:
    """Render caption / OCR text as a blockquote for chat context."""
    lines: list[str] = []
    caption = _as_str(info.get("caption")).strip()
    if caption:
        lines.append("**Image caption:** " + caption)
    ocr = _as_str(info.get("ocr_text")).strip()
    if ocr:
        lines.append("**Image text (OCR):** " + ocr)
    if not lines:
        return ""
    joined = "\n\n".join(lines)
    return "> " + joined.replace("\n", "\n> ")


def enrich_content_with_image_info_for_chat(content: str, image_info_json: str) -> str:
    """Enrich matching Markdown / HTML images with caption / OCR text.

    Only images with a matching ``image_info`` entry are enriched; the image
    itself stays Markdown so a model that copies it still renders the copy.
    Injections are spliced right-to-left so an injected block is never
    rescanned by the other syntax's pattern.
    """
    infos = parse_image_infos(image_info_json)
    if not infos:
        return content
    image_info_map = _image_info_map(infos)

    injections: list[tuple[int, str]] = []

    def append_for(matches: Iterator[re.Match[str]], url_group: int, trim: bool) -> None:
        for match in matches:
            key = match.group(url_group)
            if key is None:
                continue
            if trim:
                key = key.strip()
            info = image_info_map.get(key)
            if info is None:
                continue
            metadata = _image_info_markdown_metadata(info)
            if metadata == "":
                continue
            injections.append((match.end(), "\n\n" + metadata))

    append_for(_MARKDOWN_IMAGE_RE.finditer(content), 2, False)
    append_for(_HTML_IMAGE_SRC_RE.finditer(content), _HTML_IMAGE_SRC_URL_GROUP, True)

    if not injections:
        return content
    injections.sort(key=lambda item: item[0], reverse=True)
    for at, text in injections:
        content = content[:at] + text + content[at:]
    return content


def get_enriched_passage_for_chat(content: str, image_info: str) -> str:
    """Merge a chunk's content and image-info text for chat messages."""
    if content == "" and image_info == "":
        return ""
    if image_info == "":
        return content
    return enrich_content_with_image_info_for_chat(content, image_info)


def prepare_messages_with_model_context(
    pipeline_ctx: PipelineContext,
) -> tuple[list[ChatMessage], Registry]:
    """Replace positional retrieval ids with request-local model handles.

    The system message gains the source-handling protocol prompt; the
    rendered chunk / web context is spliced in place of the persisted
    rendered-context placeholder, or prepended to the user message when no
    placeholder matches. The registry is returned alongside so the caller
    can encode / decode the messages against the same handles.
    """
    registry = Registry(pipeline_ctx.citations_enabled())
    messages = prepare_messages_with_history(pipeline_ctx)
    if messages:
        first = messages[0]
        updated_first = replace(
            first, content=first.content.rstrip(" \t\r\n") + registry.protocol_prompt()
        )
        messages = [updated_first, *messages[1:]]
    if not pipeline_ctx.merge_result or not messages:
        return messages, registry

    ordered = ordered_pipeline_references(pipeline_ctx)
    knowledge_results: list[LlmSearchResult] = []
    knowledge_rows: list[JsonObject] = []
    web_rows: list[JsonObject] = []
    for result in ordered:
        if is_pipeline_web_reference(result):
            web_row: JsonObject = {
                "url": result.id,
                "title": first_pipeline_title(result),
                "snippet": result.content,
                "published_at": result.metadata.get("published_at", ""),
            }
            web_rows.append(web_row)
            continue
        knowledge_results.append(_to_registry_search_result(result))
        knowledge_row: JsonObject = {
            "chunk_id": result.id,
            "knowledge_id": result.knowledge_id,
            "knowledge_base_id": result.knowledge_base_id,
            "knowledge_title": first_pipeline_title(result),
            "chunk_index": result.chunk_index,
            "chunk_type": result.chunk_type,
            "content": get_enriched_passage_for_chat(result.content, result.image_info),
        }
        knowledge_rows.append(knowledge_row)
    registry.register_search_results(knowledge_results)

    context_parts: list[str] = []
    if knowledge_rows:
        knowledge_data: JsonObject = {
            "display_type": "search_results",
            "results": cast("list[JsonValue]", knowledge_rows),
        }
        context_parts.append(
            registry.model_tool_result(ToolResult(success=True, data=knowledge_data))
        )
    if web_rows:
        web_data: JsonObject = {
            "display_type": "web_search_results",
            "results": cast("list[JsonValue]", web_rows),
        }
        context_parts.append(registry.model_tool_result(ToolResult(success=True, data=web_data)))

    model_contexts = "\n".join(context_parts)
    if model_contexts.strip() == "":
        return messages, registry

    last = len(messages) - 1
    replaced = False
    for index in (0, last):
        message = messages[index]
        if (
            pipeline_ctx.rendered_contexts != ""
            and pipeline_ctx.rendered_contexts in message.content
        ):
            updated = replace(
                message,
                content=message.content.replace(pipeline_ctx.rendered_contexts, model_contexts),
            )
            messages = [updated if i == index else m for i, m in enumerate(messages)]
            replaced = True
    if not replaced:
        last_message = messages[last]
        messages = [
            *messages[:last],
            replace(last_message, content=model_contexts + "\n\n" + last_message.content),
        ]
    return messages, registry


__all__ = [
    "enrich_content_with_image_info_for_chat",
    "first_pipeline_title",
    "get_enriched_passage_for_chat",
    "is_pipeline_web_reference",
    "ordered_pipeline_references",
    "prepare_messages_with_model_context",
]
