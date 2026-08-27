"""List-knowledge-chunks tool: full chunk content for one document or FAQ.

Retrieves the complete chunk content of a document (paged) or of a single
FAQ entry / chunk, after authorizing the target against the session scope.
The output mirrors the sibling retrieval tools' XML shape so agents and
downstream consumers see one consistent format.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import cast

from src.ai.embedding.base import Context
from src.common.json import JsonObject, JsonValue
from src.core.agents.tools.base import ToolDefinition, ToolResult
from src.core.agents.tools.chunk_store import PagedChunkStore
from src.core.agents.tools.faq_utils import (
    FAQ_CHUNK_TYPE,
    append_faq_chunk_data,
    faq_standard_question,
    normalize_faq_chunk_data_map,
    write_faq_entry_xml,
)
from src.core.agents.tools.scope_auth import (
    ChunkLookup,
    KnowledgeLookup,
    KnowledgeTagsFetcher,
    authorize_chunk_in_search_targets,
    authorize_knowledge_in_search_targets,
)
from src.core.agents.tools.search_target import SearchTargets
from src.core.agents.tools.text_utils import (
    parse_image_infos,
    summarize_content,
    xml_escape,
)
from src.db.models.chunk import Chunk

_DEFAULT_LIMIT = 20
_DEFAULT_OFFSET = 0

#: Tool name constant (kept here to avoid a dependency cycle with base).
LIST_TOOL_NAME = "list_knowledge_chunks"

LIST_TOOL_DESCRIPTION = (
    "Retrieve full chunk content for a document or a single FAQ entry.\n"
    "\n"
    "## Use After grep_chunks or knowledge_search:\n"
    '- **FAQ hit** (type faq): list_knowledge_chunks(faq_id="cN") — reads that '
    "one FAQ chunk with answers from metadata.\n"
    '- **Document hit**: list_knowledge_chunks(knowledge_id="dN") — pages '
    "through all chunks.\n"
    "\n"
    "## Parameters (provide exactly one id target):\n"
    "- faq_id (optional): Short cN ID for an FAQ chunk from grep_chunks / "
    "knowledge_search.\n"
    "- chunk_id (optional): Short cN ID for a single non-FAQ chunk.\n"
    "- knowledge_id (optional): Short dN document ID to page through all "
    "chunks.\n"
    "- limit / offset: Only for knowledge_id paging (default limit 20, max 100).\n"
    "\n"
    "## Output:\n"
    "Full chunk content. FAQ entries include <faq> with <answer> from metadata."
)

LIST_TOOL_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "faq_id": {
                "type": "string",
                "description": "Short cN FAQ chunk ID. Use for FAQ hits instead "
                "of the parent dN document ID.",
            },
            "chunk_id": {
                "type": "string",
                "description": "Short cN ID for one non-FAQ chunk",
            },
            "knowledge_id": {
                "type": "string",
                "description": "Short dN document ID to list all chunks",
            },
            "limit": {
                "type": "integer",
                "description": "Chunks per page when using knowledge_id (default 20, max 100)",
                "default": 20,
                "minimum": 1,
                "maximum": 100,
            },
            "offset": {
                "type": "integer",
                "description": "Start position when using knowledge_id (default 0)",
                "default": 0,
                "minimum": 0,
            },
        },
    },
    ensure_ascii=False,
)


def build_list_knowledge_chunks_definition() -> ToolDefinition:
    """Return the default tool definition for the list tool."""
    return ToolDefinition(
        name=LIST_TOOL_NAME,
        description=LIST_TOOL_DESCRIPTION,
        parameters=LIST_TOOL_SCHEMA,
    )


@dataclass(frozen=True, slots=True)
class ListKnowledgeChunksInput:
    """Parsed input for the list-knowledge-chunks tool."""

    knowledge_id: str = ""
    faq_id: str = ""
    chunk_id: str = ""
    limit: int = 0
    offset: int = 0

    @classmethod
    def from_json(cls, raw: JsonObject) -> ListKnowledgeChunksInput:
        return cls(
            knowledge_id=_as_str(raw.get("knowledge_id")),
            faq_id=_as_str(raw.get("faq_id")),
            chunk_id=_as_str(raw.get("chunk_id")),
            limit=_as_int(raw.get("limit")),
            offset=_as_int(raw.get("offset")),
        )


def _as_str(value: JsonValue) -> str:
    return value if isinstance(value, str) else ""


def _as_int(value: JsonValue) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


class ListKnowledgeChunksTool:
    """Retrieves chunk snapshots for a specific knowledge document."""

    def __init__(
        self,
        *,
        definition: ToolDefinition,
        chunk_store: PagedChunkStore,
        search_targets: SearchTargets,
        knowledge_service: KnowledgeLookup | None = None,
        chunk_service: ChunkLookup | None = None,
        tag_fetcher: KnowledgeTagsFetcher | None = None,
    ) -> None:
        self._definition = definition
        self._chunk_store = chunk_store
        self._search_targets = search_targets
        self._knowledge_service = knowledge_service
        self._chunk_service = chunk_service
        self._tag_fetcher = tag_fetcher

    def name(self) -> str:
        return self._definition.name

    def description(self) -> str:
        return self._definition.description

    def parameters(self) -> str:
        return self._definition.parameters

    async def execute(self, ctx: Context, args: str) -> ToolResult:
        """Fetch chunk content by knowledge id, or by a single chunk id."""
        input_ = _parse_input(args)

        chunk_id = input_.faq_id.strip()
        if not chunk_id:
            chunk_id = input_.chunk_id.strip()
        if chunk_id:
            return await self._execute_by_chunk_id(ctx, chunk_id)

        knowledge_id = input_.knowledge_id.strip()
        if not knowledge_id:
            return ToolResult(
                success=False,
                error="one of faq_id, chunk_id, or knowledge_id is required",
            )

        knowledge = await authorize_knowledge_in_search_targets(
            ctx,
            self._search_targets,
            knowledge_id,
            self._knowledge_service,
            self._tag_fetcher,
        )
        # Use the knowledge's actual tenant id so cross-tenant shared KBs
        # resolve their chunks under the owning tenant.
        effective_tenant_id = knowledge.tenant_id

        chunk_limit = input_.limit if input_.limit > 0 else _DEFAULT_LIMIT
        offset = input_.offset if input_.offset > 0 else _DEFAULT_OFFSET

        page = offset // chunk_limit + 1
        chunks, total = await self._chunk_store.list_paged_chunks(
            tenant_id=effective_tenant_id,
            knowledge_id=knowledge_id,
            page=page,
            page_size=chunk_limit,
            enabled_only=True,
        )
        if chunks is None:
            return ToolResult(success=False, error="chunk query returned no data")

        total_chunks = total
        fetched = len(chunks)

        # Explicit out-of-range guidance: a silently-empty page past the end
        # confuses models that just saw the document in search results.
        if fetched == 0 and total_chunks > 0 and offset >= total_chunks:
            suggested_offset = max(total_chunks - chunk_limit, 0)
            return ToolResult(
                success=False,
                error=(
                    f"offset {offset} is out of range: document has only "
                    f"{total_chunks} chunks (valid offset range: 0.."
                    f"{total_chunks - 1}). Retry with offset={suggested_offset} "
                    f"(or any value < {total_chunks})."
                ),
                data={
                    "knowledge_id": knowledge_id,
                    "total_chunks": total_chunks,
                    "requested_offset": offset,
                    "requested_limit": chunk_limit,
                    "suggested_offset": suggested_offset,
                },
            )

        knowledge_title = await self._lookup_knowledge_title(ctx, knowledge_id)
        output = _build_output(knowledge_id, knowledge_title, total_chunks, fetched, chunks)
        formatted_chunks = [
            _build_chunk_data(seq, chunk) for seq, chunk in enumerate(chunks, start=1)
        ]

        return ToolResult(
            success=True,
            output=output,
            data={
                "display_type": "knowledge_chunks_list",
                "knowledge_id": knowledge_id,
                "knowledge_title": knowledge_title,
                "total_chunks": total_chunks,
                "fetched_chunks": fetched,
                "page": page,
                "page_size": chunk_limit,
                "chunks": cast("list[JsonValue]", formatted_chunks),
            },
        )

    async def _execute_by_chunk_id(self, ctx: Context, chunk_id: str) -> ToolResult:
        """Load one chunk by faq_id / chunk_id (FAQ entry or any chunk)."""
        chunk = await authorize_chunk_in_search_targets(
            ctx,
            self._search_targets,
            chunk_id,
            self._chunk_service,
            self._knowledge_service,
            self._tag_fetcher,
        )

        chunks = [chunk]
        knowledge_title = await self._lookup_knowledge_title(ctx, chunk.knowledge_id)
        output = _build_output(chunk.knowledge_id, knowledge_title, 1, 1, chunks)

        formatted_chunks = [_build_chunk_data(1, chunk)]
        data: JsonObject = {
            "display_type": "knowledge_chunks_list",
            "knowledge_id": chunk.knowledge_id,
            "knowledge_title": knowledge_title,
            "total_chunks": 1,
            "fetched_chunks": 1,
            "page": 1,
            "page_size": 1,
            "chunks": cast("list[JsonValue]", formatted_chunks),
            "faq_id": chunk.id,
            "single_chunk": True,
        }
        question = faq_standard_question(chunk)
        if question:
            data["faq_question"] = question

        return ToolResult(success=True, output=output, data=data)

    async def _lookup_knowledge_title(self, ctx: Context, knowledge_id: str) -> str:
        """Look up a document title (supports cross-tenant shared KBs)."""
        if self._knowledge_service is None:
            return ""
        knowledge = await self._knowledge_service.get_document_by_id_only(id=knowledge_id)
        if knowledge is None:
            return ""
        return knowledge.title.strip() if knowledge.title else ""


def _parse_input(args: str) -> ListKnowledgeChunksInput:
    try:
        raw = json.loads(args)
    except json.JSONDecodeError:
        raw = {}
    return ListKnowledgeChunksInput.from_json(raw if isinstance(raw, dict) else {})


def _build_output(
    knowledge_id: str,
    knowledge_title: str,
    total: int,
    fetched: int,
    chunks: list[Chunk],
) -> str:
    """Build the XML output for the list-knowledge-chunks tool."""
    parts: list[str] = []
    title_attr = f' title="{xml_escape(knowledge_title)}"' if knowledge_title else ""
    parts.append(
        f'<knowledge_chunks knowledge_id="{xml_escape(knowledge_id)}"'
        f'{title_attr} total="{total}" fetched="{fetched}">\n'
    )
    if fetched == 0:
        parts.append("</knowledge_chunks>")
        return "".join(parts)

    for chunk in chunks:
        if chunk.chunk_type == FAQ_CHUNK_TYPE:
            write_faq_entry_xml(parts, chunk)
            continue
        question = faq_standard_question(chunk)
        if question:
            parts.append(
                f'<chunk chunk_id="{xml_escape(chunk.id)}" chunk_index="{chunk.chunk_index}" '
                f'type="{xml_escape(chunk.chunk_type)}" question="{xml_escape(question)}">\n'
            )
        else:
            parts.append(
                f'<chunk chunk_id="{xml_escape(chunk.id)}" chunk_index="{chunk.chunk_index}" '
                f'type="{xml_escape(chunk.chunk_type)}">\n'
            )
        parts.append(f"<content>{xml_escape(summarize_content(chunk.content))}</content>\n")
        parts.append("</chunk>\n")

    if fetched < total:
        parts.append(f'<pagination remaining="{total - fetched}" />\n')
    parts.append("</knowledge_chunks>")
    return "".join(parts)


def _images_for_chunk(chunk: Chunk) -> list[JsonObject]:
    """Extract the url / caption / ocr entries of a chunk's image info."""
    images: list[JsonObject] = []
    for img in parse_image_infos(chunk.image_info):
        entry: JsonObject = {}
        url = _as_str(img.get("url"))
        if url:
            entry["url"] = url
        caption = _as_str(img.get("caption"))
        if caption:
            entry["caption"] = caption
        ocr = _as_str(img.get("ocr_text"))
        if ocr:
            entry["ocr_text"] = ocr
        if entry:
            images.append(entry)
    return images


def _build_chunk_data(seq: int, chunk: Chunk) -> JsonObject:
    """Project one chunk onto the structured result map."""
    data: JsonObject = {
        "seq": seq,
        "chunk_id": chunk.id,
        "chunk_index": chunk.chunk_index,
        "content": chunk.content,
        "chunk_type": chunk.chunk_type,
        "knowledge_id": chunk.knowledge_id,
        "knowledge_base": chunk.knowledge_base_id,
        "start_at": chunk.start_at,
        "end_at": chunk.end_at,
        "parent_chunk_id": chunk.parent_chunk_id or "",
    }
    append_faq_chunk_data(data, chunk)
    normalize_faq_chunk_data_map(data, chunk)
    images = _images_for_chunk(chunk)
    if images:
        data["images"] = cast("list[JsonValue]", images)
    return data


__all__ = [
    "LIST_TOOL_DESCRIPTION",
    "LIST_TOOL_NAME",
    "LIST_TOOL_SCHEMA",
    "ListKnowledgeChunksInput",
    "ListKnowledgeChunksTool",
    "build_list_knowledge_chunks_definition",
]
