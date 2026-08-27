"""Chunk structure extraction and table metadata prompts.

Two extraction concerns from the reference service are kept here because
they share the same prompt-building vocabulary:

1. **Structure extraction** — a chunk's entity/relation graph is pulled out
   of its text with a structured LLM prompt. The system prompt is rendered
   from a description (with an optional ``%s`` tag list), a JSON-fenced
   example, and user-authored custom instructions; the model's reply is
   stripped of fences, parsed as JSON, and rebuilt into a deduplicated
   node/relation graph. This mirrors the reference ``Extractor`` /
   ``Formater`` semantics: duplicate nodes merge attributes, self
   relations are dropped, and relation endpoints that were not extracted
   as nodes are synthesised.

2. **Table metadata prompts** — the table-description and column-description
   templates used by the data-table summary flow. They live here so the
   summary module and any future callers share one source of truth.

The chat client is an injected seam (``src.ai.llm.Chat``); the module never
calls a provider directly. Repository and knowledge-base dependencies are
injected per call so the worker layer composes them on the request session.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from src.ai.embedding import Context
from src.ai.llm import Chat, ChatOptions, Message
from src.app_logging import logger
from src.common.exception import ApplicationError
from src.common.json import JsonObject, JsonValue
from src.core.knowledge.documents.types import (
    PARSE_STATUS_CANCELLED,
    PARSE_STATUS_DELETING,
)
from src.core.knowledge.documents.upload_pipeline import (
    DATA_TABLE_EXTENSIONS,
    file_type_of,
    normalize_file_extension,
)
from src.core.knowledge.knowledge_bases.service.kb_service import KBService
from src.core.knowledge.knowledge_bases.types import KnowledgeBaseInfo
from src.db.dao.chunk_repository import ChunkRepository
from src.db.dao.knowledge_repository import KnowledgeRepository
from src.db.models.knowledge import Document

# ── Table metadata prompt templates (reference ``extract.go``) ─────────

TABLE_DESCRIPTION_PROMPT_TEMPLATE = """You are a data analysis expert. Based on the following table structure information and data samples, generate a concise table metadata description (200-300 words).

Table name: %s

%s

%s

Please describe the table from the following dimensions:
1. **Data Subject**: What type of data does this table record? (e.g., user information, sales records, log data, etc.)
2. **Core Fields**: List 3-5 most important fields and their meanings
3. **Data Scale**: Total number of rows and columns
4. **Business Scenarios**: What business analysis or application scenarios might this table be used for?
5. **Key Characteristics**: What notable features does the data have? (e.g., contains geographic locations, has category labels, has hierarchical relationships, etc.)

**Important Notes**:
- Do not output specific data values or sample content
- Use general descriptions so users can quickly determine if this table contains the information they need
- Use concise and professional language for easy retrieval and understanding
- Write the description in the same language as the data content"""

COLUMN_DESCRIPTIONS_PROMPT_TEMPLATE = """You are a data analysis expert. Based on the following table structure information and data samples, generate structured description information for each column.

Table name: %s

%s

%s

Please generate a detailed description for each column, including the following information:
1. **Field Meaning**: What information does this column store? (e.g., user ID, order amount, creation time, etc.)
2. **Data Type**: The type and format of the data (e.g., integer, string, datetime, boolean, etc.)
3. **Business Purpose**: The role of this field in business (e.g., for user identification, amount calculation, time sorting, etc.)
4. **Data Characteristics**: Notable features of the data (e.g., unique identifier, nullable, has enum values, has units, etc.)

Please output in the following format (one paragraph per column):

**Column1** (data type)
- Field Meaning: xxx
- Business Purpose: xxx
- Data Characteristics: xxx

**Column2** (data type)
- Field Meaning: xxx
- Business Purpose: xxx
- Data Characteristics: xxx

**Important Notes**:
- Do not output specific data values, only describe the field metadata
- Use clear business terms for easy user understanding and search
- If enum value ranges can be inferred from sample data, provide a summary (e.g., status field contains pending/in-progress/completed states)
- Write descriptions in the same language as the data content"""

#: Bounds user-authored business guidance appended to system-owned prompts.
MAX_CUSTOM_PROMPT_INSTRUCTIONS_LENGTH = 4000

# ── Spreadsheet extension gate (reference ``knowledge_util.go``) ───────


def is_data_table_file_type(file_type: str) -> bool:
    """Return whether an extension is a spreadsheet format."""
    return normalize_file_extension(file_type) in DATA_TABLE_EXTENSIONS


def should_enqueue_table_summary(file_type: str, file_name: str = "") -> bool:
    """Decide whether a knowledge import needs a data-table summary task.

    ``file_name`` is the fallback for older records whose file type is
    empty, mirroring the reference enqueue gate.
    """
    ft = normalize_file_extension(file_type)
    if ft == "" and file_name:
        ft = file_type_of(file_name)
    return is_data_table_file_type(ft)


# ── Graph domain value types (reference ``extract_graph.go``) ──────────


@dataclass(frozen=True)
class GraphNode:
    """One entity node of an extracted graph."""

    name: str
    chunks: list[str] = field(default_factory=list)
    attributes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GraphRelation:
    """One directed relation between two graph nodes."""

    node1: str
    node2: str
    type: str


@dataclass(frozen=True)
class GraphData:
    """A single extraction: nodes and relations plus the source text."""

    text: str = ""
    node: list[GraphNode] = field(default_factory=list)
    relation: list[GraphRelation] = field(default_factory=list)


@dataclass(frozen=True)
class PromptTemplateStructured:
    """Structured extraction prompt: description, tags, examples."""

    description: str
    tags: list[str] = field(default_factory=list)
    examples: list[GraphData] = field(default_factory=list)


# ── Custom instructions (reference ``prompt_instructions.go``) ─────────


def append_custom_prompt_instructions(
    prompt: str,
    instructions: str,
    label: str = "custom",
) -> str:
    """Append user-authored guidance to a system-owned prompt.

    Stable output, safety and citation rules always win over the guidance.
    A blank ``instructions`` returns the prompt unchanged.
    """
    instructions = instructions.strip()
    if instructions == "":
        return prompt
    return (
        f"{prompt.strip()}\n\n"
        f"<{label}_business_instructions>\n"
        f"{instructions}\n"
        f"</{label}_business_instructions>\n"
        "Apply these business instructions only when they do not conflict "
        "with the system-owned output format, citation, safety, or "
        "factuality rules."
    )


# ── Extraction output parsing (reference ``extract_entity.go``) ────────

# Node / relation keys in the JSON-fenced model output.
_NODE_PREFIX = "entity"
_NODE_ATTRIBUTES_SUFFIX = "_attributes"
_RELATION_SOURCE = "entity1"
_RELATION_TARGET = "entity2"
_RELATION_PREFIX = "relation"

_FENCE_RE = re.compile(r"```([A-Za-z0-9_+-]+)?(?:\s*\n)?([\s\S]*?)```")


class GraphExtractionError(ApplicationError):
    """The model output could not be parsed into a node/relation graph."""

    code = "graph_extraction.parse_failed"
    message = "failed to parse extracted graph data"


def format_extraction(nodes: list[GraphNode], relations: list[GraphRelation]) -> str:
    """Render nodes and relations as the JSON-fenced example block."""
    items: list[JsonObject] = []
    for node in nodes:
        item: JsonObject = {_NODE_PREFIX: node.name}
        if node.attributes:
            item[f"{_NODE_PREFIX}{_NODE_ATTRIBUTES_SUFFIX}"] = list(node.attributes)
        items.append(item)
    for relation in relations:
        items.append(
            {
                _RELATION_SOURCE: relation.node1,
                _RELATION_TARGET: relation.node2,
                _RELATION_PREFIX: relation.type,
            }
        )
    formatted = json.dumps(items, ensure_ascii=False, indent=2)
    return f"```json\n{formatted}\n```"


def render_extraction_system_prompt(template: PromptTemplateStructured) -> str:
    """Render the system prompt: description, tags, then the examples."""
    prompt_lines: list[str] = []
    if not template.tags:
        prompt_lines.append(template.description)
    else:
        tags_json = json.dumps(template.tags)
        prompt_lines.append(template.description.replace("%s", tags_json, 1))
    if template.examples:
        prompt_lines.append("# Examples")
        for example in template.examples:
            prompt_lines.append(f"Q: {example.text.strip()}")
            prompt_lines.append(f"A: {format_extraction(example.node, example.relation)}")
            prompt_lines.append("")
    return "\n".join(prompt_lines)


def render_extraction_user_prompt(content: str) -> str:
    """Render the user prompt for one chunk's text."""
    return f"# Question\nQ: {content}\nA: "


def render_extraction_messages(
    template: PromptTemplateStructured,
    content: str,
) -> list[Message]:
    """Build the two-message extraction conversation."""
    return [
        Message(role="system", content=render_extraction_system_prompt(template)),
        Message(role="user", content=render_extraction_user_prompt(content)),
    ]


def _valid_json_fence_languages() -> set[str]:
    return {"json", ""}


def _is_likely_language_tag(value: str) -> bool:
    if value == "" or len(value) > 16:
        return False
    return all(ch.isalnum() or ch in "_+-" for ch in value)


def _extract_json_like(text: str) -> str:
    """Return the outermost JSON object/array substring of ``text``.

    Mirrors the reference fallback that recovers a payload when the fence
    regex fails: whichever bracket type appears first wins and the scan
    respects string literals so braces inside JSON strings do not unbalance
    the depth count.
    """
    obj_start = text.find("{")
    arr_start = text.find("[")
    if obj_start < 0 and arr_start < 0:
        return ""
    if obj_start < 0:
        open_ch, close_ch, start = "[", "]", arr_start
    elif arr_start < 0 or obj_start < arr_start:
        open_ch, close_ch, start = "{", "}", obj_start
    else:
        open_ch, close_ch, start = "[", "]", arr_start
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        char = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == open_ch:
            depth += 1
        elif char == close_ch:
            depth -= 1
            if depth == 0:
                return text[start : i + 1].strip()
    return ""


def _strip_fences_and_extract(text: str) -> str:
    """Recover a parseable payload when the strict fence regex fails.

    Handles a truncated opening fence (no closing fence), malformed
    fences with a recognizable JSON block, and prose-wrapped JSON.
    """
    trimmed = text.strip()
    if trimmed == "":
        return ""
    fence_index = trimmed.find("```")
    if fence_index >= 0:
        rest = trimmed[fence_index + 3 :]
        newline = rest.find("\n")
        if newline >= 0:
            first_line = rest[:newline].strip()
            if first_line == "" or _is_likely_language_tag(first_line):
                rest = rest[newline + 1 :]
        closing = rest.find("```")
        if closing >= 0:
            rest = rest[:closing]
        rest = rest.strip().strip("`").strip()
        if rest != "":
            return rest
    return _extract_json_like(trimmed)


def _extract_fenced_content(text: str) -> str:
    """Pull the JSON body out of a fenced model reply.

    Accepts a single fenced block whose language tag is ``json`` (or
    absent); multiple candidates resolve to the first with a warning.
    Falls back to fence-stripping heuristics when nothing matches.
    """
    matches = list(_FENCE_RE.finditer(text))
    candidates: list[str] = []
    for match in matches:
        lang = match.group(1)
        body = match.group(2)
        if (lang or "") in _valid_json_fence_languages():
            candidates.append(body)
    if len(candidates) == 1:
        return candidates[0].strip()
    if len(candidates) > 1:
        logger.warning("multiple json candidates found: {}", len(candidates))
        return candidates[0].strip()
    if len(matches) == 1:
        return matches[0].group(2).strip()
    if len(matches) > 1:
        logger.warning("multiple fence matches found: {}", len(matches))
        return matches[0].group(2).strip()
    recovered = _strip_fences_and_extract(text)
    if recovered != "":
        return recovered
    logger.warning("no parseable fence found in extraction output")
    return text.strip()


def _string_attributes(raw: JsonValue | None) -> list[str]:
    """Narrow an attribute payload to a string list."""
    if not isinstance(raw, list):
        return []
    attributes: list[str] = []
    for value in raw:
        attributes.append(str(value))
    return attributes


def _rebuild_graph(
    nodes: list[GraphNode],
    relations: list[GraphRelation],
) -> GraphData:
    """Deduplicate the graph and materialise relation endpoints as nodes.

    Duplicate node names merge their attributes; a relation whose subject
    and object coincide is dropped; relation endpoints absent from the
    node set are synthesised so the graph stays closed.
    """
    node_by_name: dict[str, GraphNode] = {}
    ordered_nodes: list[GraphNode] = []
    for node in nodes:
        existing = node_by_name.get(node.name)
        if existing is not None:
            merged = GraphNode(
                name=node.name,
                chunks=existing.chunks,
                attributes=existing.attributes + node.attributes,
            )
            node_by_name[node.name] = merged
            ordered_nodes = [merged if n.name == node.name else n for n in ordered_nodes]
            continue
        node_by_name[node.name] = node
        ordered_nodes.append(node)

    relations_out: list[GraphRelation] = []
    for relation in relations:
        if relation.node1 == relation.node2:
            continue
        for endpoint in (relation.node1, relation.node2):
            if endpoint not in node_by_name:
                node = GraphNode(name=endpoint)
                node_by_name[endpoint] = node
                ordered_nodes.append(node)
        relations_out.append(relation)
    return GraphData(node=ordered_nodes, relation=relations_out)


def parse_graph_output(text: str) -> GraphData:
    """Parse a model reply into a deduplicated node/relation graph.

    Raises ``GraphExtractionError`` when the fenced payload is missing or
    is not a JSON object / array of objects.
    """
    content = _extract_fenced_content(text)
    if content == "":
        raise GraphExtractionError(message="empty extraction output")
    try:
        parsed: JsonValue = json.loads(content)
    except json.JSONDecodeError as exc:
        raise GraphExtractionError(
            message=f"failed to parse JSON extraction content: {exc}"
        ) from exc
    if isinstance(parsed, dict):
        items: list[JsonValue] = [parsed]
    elif isinstance(parsed, list):
        items = parsed
    else:
        raise GraphExtractionError(message="content must be a list of extractions or a dict")

    nodes: list[GraphNode] = []
    relations: list[GraphRelation] = []
    for item in items:
        if not isinstance(item, dict):
            raise GraphExtractionError(message="each item in the sequence must be a mapping")
        if item.get(_NODE_PREFIX) is not None:
            nodes.append(
                GraphNode(
                    name=str(item[_NODE_PREFIX]),
                    attributes=_string_attributes(
                        item.get(f"{_NODE_PREFIX}{_NODE_ATTRIBUTES_SUFFIX}")
                    ),
                )
            )
        elif item.get(_RELATION_SOURCE) is not None and item.get(_RELATION_TARGET) is not None:
            relations.append(
                GraphRelation(
                    node1=str(item[_RELATION_SOURCE]),
                    node2=str(item[_RELATION_TARGET]),
                    type=str(item.get(_RELATION_PREFIX) or ""),
                )
            )
    return _rebuild_graph(nodes, relations)


class StructureExtractor:
    """Extracts a node/relation graph from text via the chat seam."""

    def __init__(self, chat: Chat, template: PromptTemplateStructured) -> None:
        self._chat = chat
        self._template = template

    async def extract(self, ctx: Context, content: str) -> GraphData:
        """Run the structured extraction call and parse the reply."""
        response = await self._chat.chat(
            render_extraction_messages(self._template, content),
            ChatOptions(temperature=0.3, max_tokens=4096, thinking=False),
        )
        return parse_graph_output(response.content)


# ── Effective extract configuration (reference ``knowledge_process_config.go``) ─


@dataclass(frozen=True)
class ExtractRunConfig:
    """Effective per-chunk extraction settings after KB/override merge."""

    enabled: bool
    text: str
    tags: list[str]
    nodes: list[GraphNode]
    relations: list[GraphRelation]
    custom_instructions: str


def _coerce_nodes(raw: JsonValue | None) -> list[GraphNode]:
    """Coerce the extract-config node payload to graph nodes.

    Accepts both the structured form (``{"name": ..., "attributes": [...]}``)
    and a plain name string.
    """
    if not isinstance(raw, list):
        return []
    nodes: list[GraphNode] = []
    for value in raw:
        if isinstance(value, dict):
            name = value.get("name")
            if name is None:
                continue
            nodes.append(
                GraphNode(name=str(name), attributes=_string_attributes(value.get("attributes")))
            )
        elif value is not None:
            nodes.append(GraphNode(name=str(value)))
    return nodes


def _coerce_relations(raw: JsonValue | None) -> list[GraphRelation]:
    """Coerce the extract-config relation payload to graph relations."""
    if not isinstance(raw, list):
        return []
    relations: list[GraphRelation] = []
    for value in raw:
        if not isinstance(value, dict):
            continue
        node1 = value.get("node1")
        node2 = value.get("node2")
        if node1 is None or node2 is None:
            continue
        relations.append(
            GraphRelation(node1=str(node1), node2=str(node2), type=str(value.get("type") or ""))
        )
    return relations


def _extract_config_dict(
    kb: KnowledgeBaseInfo,
    process_overrides: JsonObject | None,
) -> JsonObject:
    """Resolve the effective extract-config dict after the override merge.

    The knowledge base config is the default; a per-document
    ``process_overrides.extract_config`` entry replaces it, keeping the
    base fields the override leaves blank (mirroring the reference merge).
    """
    base = kb.extract_config if isinstance(kb.extract_config, dict) else {}
    override = None
    if isinstance(process_overrides, dict):
        candidate = process_overrides.get("extract_config")
        if isinstance(candidate, dict):
            override = candidate
    if override is None:
        return base
    merged: JsonObject = dict(base)
    merged["enabled"] = bool(override.get("enabled", base.get("enabled", False)))
    for key in ("text", "tags", "nodes", "relations", "custom_instructions"):
        value = override.get(key)
        if value is not None and value != "":
            merged[key] = value
    return merged


def resolve_extract_config(
    kb: KnowledgeBaseInfo,
    process_overrides: JsonObject | None,
) -> ExtractRunConfig:
    """Merge KB defaults with per-upload overrides for the extraction run."""
    raw = _extract_config_dict(kb, process_overrides)
    enabled = bool(raw.get("enabled", False))
    text = raw.get("text")
    tags = raw.get("tags")
    custom = raw.get("custom_instructions")
    return ExtractRunConfig(
        enabled=enabled,
        text=text if isinstance(text, str) else "",
        tags=[str(tag) for tag in tags] if isinstance(tags, list) else [],
        nodes=_coerce_nodes(raw.get("nodes")),
        relations=_coerce_relations(raw.get("relations")),
        custom_instructions=custom if isinstance(custom, str) else "",
    )


# ── Chunk extraction orchestration (reference ``extract.go``) ──────────


@runtime_checkable
class GraphStore(Protocol):
    """Persists extracted graphs scoped to a knowledge base + knowledge."""

    async def add_graph(
        self,
        *,
        knowledge_base_id: str,
        knowledge_id: str,
        graphs: list[GraphData],
    ) -> None:
        """Store ``graphs`` under the knowledge's namespace."""


@dataclass(frozen=True)
class ExtractionOutcome:
    """Result of one chunk-extraction run."""

    skipped: bool = False
    reason: str = ""
    node_count: int = 0
    relation_count: int = 0


def _process_overrides_of(row: Document | None) -> JsonObject | None:
    """Read the per-upload process overrides from a document's metadata."""
    if row is None or not isinstance(row.metadata, dict):
        return None
    overrides = row.metadata.get("process_overrides")
    return overrides if isinstance(overrides, dict) else None


def _build_extract_template(
    base_description: str,
    cfg: ExtractRunConfig,
) -> PromptTemplateStructured:
    """Build the per-chunk extraction prompt template.

    The description is the default graph-extraction template with the
    user's custom instructions appended; tags come from the effective
    config (an empty list keeps the literal description, mirroring the
    reference); the example is always the configured text/node/relation
    triple, even when empty.
    """
    description = append_custom_prompt_instructions(
        base_description, cfg.custom_instructions, "graph_extraction"
    )
    return PromptTemplateStructured(
        description=description,
        tags=list(cfg.tags),
        examples=[GraphData(text=cfg.text, node=cfg.nodes, relation=cfg.relations)],
    )


class ChunkExtractor:
    """Runs the per-chunk graph-extraction task over injected seams.

    The graph store is a protocol seam; the chat client is injected per
    call so the worker layer resolves the model and composes the request
    session. ``graph_enabled`` mirrors the environment flag that gates
    the whole extraction pipeline.
    """

    def __init__(
        self,
        *,
        chunk_repo: ChunkRepository,
        knowledge_repo: KnowledgeRepository,
        kb_service: KBService,
        graph_store: GraphStore,
        graph_enabled: bool = False,
        default_description: str = "",
    ) -> None:
        self._chunk_repo = chunk_repo
        self._knowledge_repo = knowledge_repo
        self._kb_service = kb_service
        self._graph_store = graph_store
        self._graph_enabled = graph_enabled
        self._default_description = default_description

    async def extract_chunk(
        self,
        *,
        ctx: Context,
        tenant_id: int,
        chunk_id: str,
        chat: Chat,
        knowledge_id: str = "",
        chunk_index: int = 0,
    ) -> ExtractionOutcome:
        """Extract the chunk's entity/relation graph and persist it.

        Skips (with a reason) when graph extraction is disabled, the
        parent knowledge is being cancelled / deleted, the effective
        extract config is disabled, or the chunk disappears mid-run.
        """
        if not self._graph_enabled:
            return ExtractionOutcome(skipped=True, reason="graph_disabled")

        chunk = await self._chunk_repo.get_by_id(tenant_id, chunk_id)

        # The parent knowledge may have been cancelled / deleted after the
        # task was enqueued; skip rather than enrich a dead document.
        knowledge = await self._knowledge_row(tenant_id, knowledge_id or chunk.knowledge_id)
        if knowledge is not None:
            if knowledge.parse_status == PARSE_STATUS_CANCELLED:
                return ExtractionOutcome(skipped=True, reason="knowledge_cancelled")
            if knowledge.parse_status == PARSE_STATUS_DELETING:
                return ExtractionOutcome(skipped=True, reason="knowledge_deleting")

        kb = await self._kb_service.get_knowledge_base_by_id(
            knowledge_base_id=chunk.knowledge_base_id
        )
        cfg = resolve_extract_config(kb, _process_overrides_of(knowledge))
        if not cfg.enabled:
            return ExtractionOutcome(skipped=True, reason="extract_disabled")

        template = _build_extract_template(self._default_description, cfg)
        extractor = StructureExtractor(chat, template)
        graph = await extractor.extract(ctx, chunk.content)

        latest = await self._chunk_repo.get_by_id_or_none(tenant_id, chunk_id)
        if latest is None:
            return ExtractionOutcome(skipped=True, reason="chunk_disappeared")

        chunked_nodes = [
            GraphNode(name=node.name, chunks=[latest.id], attributes=node.attributes)
            for node in graph.node
        ]
        await self._graph_store.add_graph(
            knowledge_base_id=latest.knowledge_base_id,
            knowledge_id=latest.knowledge_id,
            graphs=[GraphData(node=chunked_nodes, relation=graph.relation)],
        )
        return ExtractionOutcome(
            skipped=False,
            node_count=len(chunked_nodes),
            relation_count=len(graph.relation),
        )

    async def _knowledge_row(self, tenant_id: int, knowledge_id: str) -> Document | None:
        if not knowledge_id:
            return None
        return await self._knowledge_repo.get_by_id(tenant_id, knowledge_id)


__all__ = [
    "COLUMN_DESCRIPTIONS_PROMPT_TEMPLATE",
    "MAX_CUSTOM_PROMPT_INSTRUCTIONS_LENGTH",
    "TABLE_DESCRIPTION_PROMPT_TEMPLATE",
    "ChunkExtractor",
    "ExtractRunConfig",
    "ExtractionOutcome",
    "GraphData",
    "GraphExtractionError",
    "GraphNode",
    "GraphRelation",
    "GraphStore",
    "PromptTemplateStructured",
    "StructureExtractor",
    "append_custom_prompt_instructions",
    "format_extraction",
    "is_data_table_file_type",
    "parse_graph_output",
    "render_extraction_messages",
    "render_extraction_system_prompt",
    "render_extraction_user_prompt",
    "resolve_extract_config",
    "should_enqueue_table_summary",
]
