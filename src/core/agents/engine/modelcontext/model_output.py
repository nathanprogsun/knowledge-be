"""Compact, source-centric renderings of tool results for the model.

The canonical tool result output remains untouched for UI, logs and storage;
only the model-facing rendering is produced here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import cast

from pydantic import BaseModel, ConfigDict, Field

from src.common.json import JsonObject, JsonValue
from src.core.agents.engine.modelcontext.citations import escape_attr
from src.core.agents.engine.modelcontext.sources import (
    ChunkReference,
    SourceRegistry,
    first_non_empty,
)

MODEL_WEB_SEARCH_EVIDENCE_MAX_RUNES = 500
MODEL_WEB_FETCH_SUMMARY_MAX_RUNES = 4000
MODEL_WEB_FETCH_CONTENT_MAX_RUNES = 8000
MODEL_WEB_FETCH_TOTAL_MAX_RUNES = 16000


class ToolResult(BaseModel):
    """Result of one agent tool execution, rendered with model handles.

    ``output`` is the canonical human-readable text; ``data`` carries the
    structured payload that drives the compact model renderings.
    """

    model_config = ConfigDict(frozen=True)

    success: bool
    output: str = ""
    data: JsonObject | None = None
    error: str = ""
    images: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class ModelChunk:
    """One knowledge chunk prepared for the compact retrieval rendering."""

    handle: str
    doc_handle: str
    kb_handle: str
    title: str
    metadata: str
    chunk_type: str
    index: int
    view: str
    match: str
    content: str
    question: str
    answers: list[str] = field(default_factory=list)
    images: list[JsonObject] = field(default_factory=list)
    doc_real_id: str = ""
    kb_real_id: str = ""
    chunk_real: str = ""
    input_order: int = 0


@dataclass
class _DocGroup:
    """Knowledge chunks grouped under one document in the rendering."""

    handle: str
    kb_handle: str
    title: str
    metadata: str
    order: int
    chunks: list[ModelChunk] = field(default_factory=list)


def render_model_output(registry: SourceRegistry, result: ToolResult) -> str:
    """Return a compact source-centric representation for the LLM."""
    data = result.data if result.data is not None else {}
    display_type = string_value(data, "display_type")
    if display_type == "web_fetch_results":
        return render_web_fetch_output(registry, maps_value(data.get("results")), result.output)
    if not result.success:
        if result.error != "":
            return "Error: " + registry.compact_known_text(result.error)
        return "Error: tool call failed"
    match display_type:
        case "grep_results":
            return render_knowledge_output(
                registry, "keyword", maps_value(data.get("chunk_results")), result.output
            )
        case "search_results":
            return render_knowledge_output(
                registry, "semantic", maps_value(data.get("results")), result.output
            )
        case "knowledge_chunks_list":
            return render_knowledge_chunks_output(registry, data, result.output)
        case "document_info":
            return render_document_info_output(
                registry, maps_value(data.get("documents")), result.output
            )
        case "graph_query_results":
            return render_knowledge_output(
                registry, "graph", maps_value(data.get("results")), result.output
            )
        case "web_search_results":
            return render_web_search_output(
                registry, maps_value(data.get("results")), result.output
            )
        case "database_query":
            return render_database_query_output(
                registry, maps_value(data.get("rows")), result.output
            )
        case _:
            registry.register_labeled_references(result.output)
            register_structured_references(registry, result.output)
            return registry.compact_known_text(result.output)


def register_structured_references(registry: SourceRegistry, raw: str) -> None:
    """Register durable IDs under explicitly labeled keys of a JSON result.

    Uses the same key dispatch as tool arguments; non-JSON output is a no-op.
    """
    value = _loads_json(raw)
    if value is None:
        return
    walk_structured_references("", value, registry)


def walk_structured_references(key: str, value: JsonValue, registry: SourceRegistry) -> None:
    if isinstance(value, str):
        registry.register_source_id_by_key(key, value)
    elif isinstance(value, list):
        for item in value:
            walk_structured_references(key, item, registry)
    elif isinstance(value, dict):
        for child_key, item in value.items():
            walk_structured_references(child_key, item, registry)


def render_database_query_output(
    registry: SourceRegistry, rows: list[JsonObject], fallback: str
) -> str:
    """Register durable IDs found in database rows and compact the fallback text."""
    for row in rows:
        for key, raw in row.items():
            if isinstance(raw, str):
                registry.register_source_id_by_key(key, raw)
    return registry.compact_known_text(fallback)


def render_document_info_output(
    registry: SourceRegistry, rows: list[JsonObject], fallback: str
) -> str:
    if not rows:
        return registry.compact_known_text(fallback)
    out: list[str] = ["<documents>"]
    count = 0
    for row in rows:
        knowledge_id = string_value(row, "knowledge_id")
        doc_handle = registry.register_document(knowledge_id)
        if bool_value(row, "is_faq"):
            chunk_id = string_value(row, "faq_id")
            if chunk_id == "":
                continue
            title = first_non_empty(string_value(row, "faq_question"), string_value(row, "title"))
            chunk_handle = registry.register_chunk(
                ChunkReference(
                    chunk_id=chunk_id,
                    knowledge_id=knowledge_id,
                    document_title=title,
                    chunk_type="faq",
                )
            )
            out.append(f'  <document id="{escape_attr(doc_handle)}" type="faq">')
            out.append(f'    <chunk id="{escape_attr(chunk_handle)}" type="faq">')
            if title != "":
                out.append(f"      <question>{escape_text(title)}</question>")
            for answer in string_slice_value(row.get("faq_answers")):
                out.append(f"      <answer>{escape_text(answer)}</answer>")
            out.append("    </chunk>\n  </document>")
            count += 1
            continue
        if doc_handle == "":
            continue
        parts = [f'  <document id="{escape_attr(doc_handle)}"']
        if title := string_value(row, "title"):
            parts.append(f' title="{escape_attr(title)}"')
        if doc_type := string_value(row, "type"):
            parts.append(f' type="{escape_attr(doc_type)}"')
        if file_type := string_value(row, "file_type"):
            parts.append(f' file_type="{escape_attr(file_type)}"')
        parts.append(f' chunk_count="{int_value(row, "chunk_count")}">')
        out.append("".join(parts))
        if description := string_value(row, "description"):
            out.append(f"    <description>{escape_text(description)}</description>")
        out.append("  </document>")
        count += 1
    out.append("</documents>")
    if count == 0:
        return registry.compact_known_text(fallback)
    return "\n".join(out)


def render_knowledge_output(
    registry: SourceRegistry, mode: str, rows: list[JsonObject], fallback: str
) -> str:
    """Register chunk/document/kb handles and render the grouped retrieval."""
    chunks: list[ModelChunk] = []
    for idx, row in enumerate(rows):
        chunk_id = first_non_empty(
            string_value(row, "chunk_id"), string_value(row, "faq_id"), string_value(row, "id")
        )
        knowledge_id = string_value(row, "knowledge_id")
        kb_id = first_non_empty(
            string_value(row, "knowledge_base_id"), string_value(row, "knowledge_base")
        )
        title = first_non_empty(string_value(row, "knowledge_title"), string_value(row, "title"))
        if chunk_id == "":
            continue
        chunk_type = string_value(row, "chunk_type")
        if string_value(row, "faq_id") != "" and chunk_type == "":
            chunk_type = "faq"
        chunk_index = int_value(row, "chunk_index")
        if chunk_index == 0:
            chunk_index = int_value(row, "index")
        chunk_handle = registry.register_chunk(
            ChunkReference(
                chunk_id=chunk_id,
                knowledge_id=knowledge_id,
                knowledge_base_id=kb_id,
                document_title=title,
                chunk_index=chunk_index,
                chunk_type=chunk_type,
            )
        )
        chunks.append(
            ModelChunk(
                handle=chunk_handle,
                doc_handle=registry.register_document(knowledge_id),
                kb_handle=registry.register_knowledge_base(kb_id),
                title=title,
                metadata=string_value(row, "knowledge_metadata"),
                chunk_type=chunk_type,
                index=chunk_index,
                view=view_for_row(row, mode),
                match=first_non_empty(
                    string_value(row, "match_snippet"), string_value(row, "matched_content")
                ),
                content=string_value(row, "content"),
                question=first_non_empty(
                    string_value(row, "faq_question"), string_value(row, "faq_standard_question")
                ),
                answers=string_slice_value(row.get("faq_answers")),
                images=maps_value(row.get("images")),
                doc_real_id=knowledge_id,
                kb_real_id=kb_id,
                chunk_real=chunk_id,
                input_order=idx,
            )
        )
    if not chunks:
        return registry.compact_known_text(fallback)
    return render_knowledge_chunks(mode, chunks)


def view_for_row(row: JsonObject, mode: str) -> str:
    """Choose the chunk view: full content or a match snippet."""
    if string_value(row, "content") != "":
        return "full"
    if mode == "deep_read":
        return "full"
    return "match"


def render_knowledge_chunks_output(
    registry: SourceRegistry, data: JsonObject, fallback: str
) -> str:
    """Render a knowledge-chunk list, defaulting knowledge fields per row."""
    rows = maps_value(data.get("chunks"))
    title = string_value(data, "knowledge_title")
    knowledge_id = string_value(data, "knowledge_id")
    copied: list[JsonObject] = []
    for row in rows:
        updated = dict(row)
        if string_value(updated, "knowledge_id") == "":
            updated["knowledge_id"] = knowledge_id
        if string_value(updated, "knowledge_title") == "":
            updated["knowledge_title"] = title
        copied.append(updated)
    output = render_knowledge_output(registry, "deep_read", copied, fallback)
    if not copied:
        return output
    remaining = int_value(data, "total_chunks") - int_value(data, "fetched_chunks")
    if remaining > 0:
        output = output.removesuffix("</retrieval>")
        output += (
            f'  <pagination remaining="{remaining}" page="{int_value(data, "page")}" '
            f'page_size="{int_value(data, "page_size")}" />\n</retrieval>'
        )
    return output


def render_knowledge_chunks(mode: str, chunks: list[ModelChunk]) -> str:
    """Group chunks under their documents and render the ``<retrieval>`` block."""
    groups_by_key: dict[str, _DocGroup] = {}
    groups: list[_DocGroup] = []
    for chunk in chunks:
        key = chunk.doc_handle
        if key == "":
            key = f"chunk:{chunk.handle}"
        group = groups_by_key.get(key)
        if group is None:
            group = _DocGroup(
                handle=chunk.doc_handle,
                kb_handle=chunk.kb_handle,
                title=chunk.title,
                metadata=chunk.metadata,
                order=chunk.input_order,
            )
            groups_by_key[key] = group
            groups.append(group)
        elif group.metadata == "":
            group.metadata = chunk.metadata
        group.chunks.append(chunk)
    groups.sort(key=lambda item: item.order)

    out: list[str] = [f'<retrieval type="knowledge" mode="{escape_attr(mode)}">']
    for group in groups:
        parts = ["  <document"]
        if group.handle != "":
            parts.append(f' id="{escape_attr(group.handle)}"')
        if group.kb_handle != "":
            parts.append(f' kb="{escape_attr(group.kb_handle)}"')
        if group.title != "":
            parts.append(f' title="{escape_attr(group.title)}"')
        parts.append(">")
        out.append("".join(parts))
        if group.metadata != "":
            out.append(f"    <metadata>{escape_text(group.metadata)}</metadata>")
        for chunk in group.chunks:
            parts = [
                f'    <chunk id="{escape_attr(chunk.handle)}" index="{chunk.index}" view="{chunk.view}"'
            ]
            if chunk.chunk_type != "":
                parts.append(f' type="{escape_attr(chunk.chunk_type)}"')
            parts.append(">")
            out.append("".join(parts))
            if chunk.question != "":
                out.append(f"      <question>{escape_text(chunk.question)}</question>")
            if chunk.match != "":
                out.append(f"      <match>{escape_text(chunk.match)}</match>")
            if chunk.content != "":
                out.append(f"      <content>{escape_text(chunk.content)}</content>")
            for answer in chunk.answers:
                out.append(f"      <answer>{escape_text(answer)}</answer>")
            for image in chunk.images:
                image_url = string_value(image, "url")
                if image_url == "":
                    continue
                caption = string_value(image, "caption")
                out.append(f"      ![{caption}]({image_url})")
            out.append("    </chunk>")
        out.append("  </document>")
    out.append("</retrieval>")
    return "\n".join(out)


def render_web_search_output(
    registry: SourceRegistry, rows: list[JsonObject], fallback: str
) -> str:
    if not rows:
        return registry.compact_known_text(fallback)
    out: list[str] = ['<retrieval type="web" mode="search">']
    count = 0
    for row in rows:
        raw_url = string_value(row, "url")
        if raw_url == "":
            continue
        handle = registry.register_web(raw_url, string_value(row, "title"))
        out.append(f'  <page id="{handle}" title="{escape_attr(string_value(row, "title"))}">')
        out.append('    <evidence type="search_summary" verified="false" />')
        snippet = string_value(row, "snippet")
        if snippet != "":
            write_limited_web_evidence(
                out, "match", snippet, MODEL_WEB_SEARCH_EVIDENCE_MAX_RUNES, None
            )
        content = string_value(row, "content")
        if content != "" and content != snippet:
            write_limited_web_evidence(
                out, "content", content, MODEL_WEB_SEARCH_EVIDENCE_MAX_RUNES, None
            )
        published = string_value(row, "published_at")
        if published != "":
            out.append(f"    <published>{escape_text(published)}</published>")
        out.append("  </page>")
        count += 1
    out.append("</retrieval>")
    if count == 0:
        return registry.compact_known_text(fallback)
    return "\n".join(out)


def render_web_fetch_output(registry: SourceRegistry, rows: list[JsonObject], fallback: str) -> str:
    if not rows:
        return registry.compact_known_text(fallback)
    out: list[str] = ['<retrieval type="web" mode="fetch">']
    count = 0
    success_count = 0
    failed_count = 0
    remaining_evidence: int | None = MODEL_WEB_FETCH_TOTAL_MAX_RUNES
    for row in rows:
        raw_url = string_value(row, "url")
        if raw_url == "":
            continue
        title = string_value(row, "title")
        handle = registry.register_web(raw_url, title)
        status = string_value(row, "status")
        if status == "":
            status = "success"
        parts = [f'  <page id="{handle}" status="{escape_attr(status)}"']
        if title != "":
            parts.append(f' title="{escape_attr(title)}"')
        if status == "success":
            parts.append(' view="full">')
            out.append("".join(parts))
            success_count += 1
            summary = string_value(row, "summary")
            if summary != "":
                remaining_evidence = write_limited_web_evidence(
                    out, "summary", summary, MODEL_WEB_FETCH_SUMMARY_MAX_RUNES, remaining_evidence
                )
            if string_value(row, "summary_status") == "failed":
                out.append(
                    f'    <summary_error code="{escape_attr(string_value(row, "summary_error_code"))}">'
                    f"{escape_text(string_value(row, 'summary_error_message'))}</summary_error>"
                )
            content = string_value(row, "raw_content")
            if content != "":
                remaining_evidence = write_limited_web_evidence(
                    out, "content", content, MODEL_WEB_FETCH_CONTENT_MAX_RUNES, remaining_evidence
                )
        else:
            parts.append(f' retryable="{bool_value(row, "retryable")}"')
            error_code = string_value(row, "error_code")
            if error_code != "":
                parts.append(f' error_code="{escape_attr(error_code)}"')
            parts.append(">")
            out.append("".join(parts))
            error_message = string_value(row, "error_message")
            if error_message != "":
                out.append(f"    <error>{escape_text(error_message)}</error>")
            if status == "failed":
                failed_count += 1
        out.append("  </page>")
        count += 1
    out.append("</retrieval>")
    if count == 0:
        return registry.compact_known_text(fallback)
    if failed_count > 0:
        out.append("")
        out.append("=== Next Steps ===")
        if success_count == 0:
            out.append(
                "- All page fetches failed. Stop expanding web searches and answer from "
                "existing web_search titles, URLs, and snippets."
            )
            out.append(
                "- Explicitly state that page content was not verified and treat dynamic "
                "facts as uncertain."
            )
        else:
            out.append(
                "- Use successful page content together with existing search snippets; "
                "failed URLs do not invalidate successful evidence."
            )
            out.append(
                "- Do not retry non-retryable failures. If evidence is sufficient, answer now."
            )
    return "\n".join(out)


def write_limited_web_evidence(
    out: list[str],
    tag: str,
    value: str,
    max_runes: int,
    remaining: int | None,
) -> int | None:
    """Write a bounded evidence element, returning the updated shared budget."""
    if value == "":
        return remaining
    limit = max_runes
    if remaining is not None and remaining < limit:
        limit = remaining
    limited, truncated = truncate_model_evidence(value, limit)
    if remaining is not None:
        remaining = max(remaining - len(limited), 0)
    parts = [f"    <{tag}"]
    if truncated:
        parts.append(' truncated="true"')
    parts.append(f">{escape_text(limited)}</{tag}>")
    out.append("".join(parts))
    return remaining


def truncate_model_evidence(value: str, max_runes: int) -> tuple[str, bool]:
    """Truncate ``value`` to ``max_runes`` code points; report whether it was cut."""
    runes = list(value)
    if len(runes) <= max_runes:
        return value, False
    if max_runes <= 0:
        return "", True
    return "".join(runes[:max_runes]), True


def maps_value(value: JsonValue | None) -> list[JsonObject]:
    """Coerce a JSON array value into a list of object dictionaries."""
    if not isinstance(value, list):
        return []
    rows: list[JsonObject] = []
    for item in value:
        if isinstance(item, dict):
            rows.append(item)
    return rows


def string_value(values: JsonObject, key: str) -> str:
    """Return the string value for ``key`` or ``""`` when absent or non-string."""
    value = values.get(key)
    if isinstance(value, str):
        return value
    return ""


def int_value(values: JsonObject, key: str) -> int:
    """Return the integer value for ``key`` or ``0`` when absent or non-integer."""
    value = values.get(key)
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def bool_value(values: JsonObject, key: str) -> bool:
    """Return the boolean value for ``key``, defaulting to ``False``."""
    value = values.get(key)
    if isinstance(value, bool):
        return value
    return False


def string_slice_value(value: JsonValue | None) -> list[str]:
    """Coerce a JSON array value into a list of strings."""
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str):
            out.append(item)
    return out


def escape_text(value: str) -> str:
    """Escape text for XML/HTML element content."""
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _loads_json(raw: str) -> JsonValue | None:
    try:
        value = json.loads(raw)
    except ValueError:
        return None
    return cast(JsonValue, value)


__all__ = [
    "ModelChunk",
    "ToolResult",
    "bool_value",
    "escape_text",
    "int_value",
    "maps_value",
    "render_model_output",
    "render_web_fetch_output",
    "render_web_search_output",
    "string_slice_value",
    "string_value",
    "truncate_model_evidence",
]
