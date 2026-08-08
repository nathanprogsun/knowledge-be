"""Knowledge graph builder - entity/relation extraction and graph weighting.

Faithful port of the upstream graph service semantics: given the text
chunks of a document, the builder drives an injectable chat client to
extract entities and relationships, aggregates them across chunks, then
computes relationship weights from Pointwise Mutual Information (PMI)
and relationship strength, entity degrees, and the chunk-relation graph
used by retrieval-time expansion.

The chat client is injected through the ``Chat`` protocol so the builder
never calls a provider directly. Prompt templates are constructor
parameters with the upstream default templates. The module is a
standalone service module: the web layer composes it with the document
and chunk services and a concrete provider later.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
import uuid
from collections.abc import Awaitable, Callable, Sequence
from typing import TypeVar, cast

from pydantic import ValidationError as PydanticValidationError

from src.ai.llm.types import Chat, ChatOptions, Message
from src.common.exception import AIProviderError, ApplicationError, ValidationError
from src.common.json import JsonValue
from src.core.knowledge.graph.prompts import (
    DEFAULT_EXTRACT_ENTITIES_PROMPT,
    DEFAULT_EXTRACT_RELATIONSHIPS_PROMPT,
)
from src.core.knowledge.graph.types import (
    ChunkInput,
    ChunkRelation,
    Entity,
    GraphBuildResult,
    Relationship,
)

logger = logging.getLogger(__name__)

# ── Tuning constants (mirror the upstream graph service) ──────────────

# Low temperature for more deterministic extraction results.
DEFAULT_LLM_TEMPERATURE = 0.1

# Proportion of PMI in the combined relationship weight.
PMI_WEIGHT = 0.6

# Proportion of relationship strength in the combined relationship weight.
STRENGTH_WEIGHT = 0.4

# Decay coefficient applied to second-degree (indirect) relationship weights.
INDIRECT_RELATION_WEIGHT_DECAY = 0.5

# Concurrency cap for entity extraction across chunks.
MAX_CONCURRENT_ENTITY_EXTRACTIONS = 4

# Concurrency cap for relationship extraction across batches.
MAX_CONCURRENT_RELATION_EXTRACTIONS = 4

# Default batch size (in chunks) for relationship extraction.
DEFAULT_RELATION_BATCH_SIZE = 5

# Minimum number of entities required to form relationships.
MIN_ENTITIES_FOR_RELATION = 2

# Minimum weight value, avoids division by zero during normalization.
MIN_WEIGHT_VALUE = 1.0

# Scaling factor that normalizes combined weights to the 1-10 range.
WEIGHT_SCALE_FACTOR = 9.0

# Default prompt language used to fill the ``{{language}}`` placeholder.
DEFAULT_PROMPT_LANGUAGE = "English"

# Shortest suffix that participates in chunk overlap dedup (mirrors the
# upstream chunk-merge helper); anything shorter risks false matches.
_MIN_OVERLAP_RUNES = 12

# Lower bound for the overlap search window when chunk positions are missing.
_DEFAULT_SEARCH_SPAN = 400

# Lower bound for how far a merged prefix may skip into the next chunk.
_MIN_HEAD_SLACK = 320

# Matches a JSON payload wrapped in a Markdown code fence.
_JSON_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```")


# ── Shared prompt / JSON helpers ───────────────────────────────────────


def render_prompt_placeholders(template: str, values: dict[str, str]) -> str:
    """Replace every ``{{key}}`` placeholder with its value.

    Unknown placeholders are left untouched, mirroring the upstream
    prompt placeholder renderer used by the extraction templates.
    """
    result = template
    for key, value in values.items():
        placeholder = "{{" + key + "}}"
        if placeholder in result:
            result = result.replace(placeholder, value)
    return result


def parse_llm_json_response(content: str) -> JsonValue:
    """Parse an LLM JSON payload, unwrapping a Markdown code fence.

    Mirrors the upstream parse helper: the content is first tried as a
    plain JSON document; when that fails, a JSON payload wrapped in a
    fenced code block is extracted and parsed. A payload that is neither
    raises the underlying decode error.
    """
    try:
        return cast(JsonValue, json.loads(content))
    except json.JSONDecodeError as decode_error:
        match = _JSON_CODE_FENCE_RE.search(content)
        if match is None:
            raise decode_error
        return cast(JsonValue, json.loads(match.group(1).strip()))


# ── Chunk content merge (overlap-aware reconstruction) ─────────────────


def _index_runes(haystack: str, needle: str, max_start: int) -> int:
    """Return the first code-point index of ``needle`` within ``max_start``.

    Returns -1 when the needle is absent or its first occurrence starts
    past ``max_start``, mirroring the upstream rune-index helper.
    """
    if not needle or len(needle) > len(haystack):
        return -1
    limit = len(haystack) - len(needle)
    if max_start < limit:
        limit = max_start
    for index in range(limit + 1):
        if haystack[index : index + len(needle)] == needle:
            return index
    return -1


def _append_with_overlap(acc: str, next_: str, position_overlap: int) -> str:
    """Append ``next_`` to ``acc`` removing any real text overlap.

    The overlap is matched by text, not by position offsets, so it stays
    correct when chunk content carries synthesized headers or HTML
    entities whose character count differs from the source offsets. When
    no overlap is found the bodies are joined verbatim rather than
    dropping edited content.
    """
    if acc == "":
        return next_
    if next_ == "":
        return acc
    span = max(position_overlap, 0)
    max_k = min(len(acc), len(next_))
    if max_k > max(span * 3, _DEFAULT_SEARCH_SPAN):
        max_k = max(span * 3, _DEFAULT_SEARCH_SPAN)
    head_slack = max(span * 2, _MIN_HEAD_SLACK)
    for k in range(max_k, _MIN_OVERLAP_RUNES - 1, -1):
        needle = acc[-k:]
        position = _index_runes(next_, needle, head_slack)
        if position >= 0:
            return acc + next_[position + k :]
    return acc + next_


def merge_text_chunks(chunks: Sequence[ChunkInput], gap_separator: str = "") -> str:
    """Merge ``chunks`` into one text, removing overlapping boundaries.

    Chunks are ordered by ``start_at`` (ties broken by ``chunk_index``).
    A chunk whose start position is inside the already-merged text has its
    overlapping prefix dropped; a chunk that does not touch the merged
    text is appended verbatim after ``gap_separator`` (empty here, so a
    direct concatenation, matching the graph-merge call site).
    """
    if not chunks:
        return ""
    ordered = sorted(
        chunks,
        key=lambda chunk: (chunk.start_at if chunk.start_at is not None else 0, chunk.chunk_index),
    )
    merged = ""
    merged_end = -1
    for chunk in ordered:
        content = chunk.content or ""
        if content == "":
            continue
        start_at = chunk.start_at if chunk.start_at is not None else 0
        end_at = chunk.end_at if chunk.end_at is not None else 0
        if merged == "":
            merged = content
            if end_at > 0:
                merged_end = end_at
            continue
        if start_at > merged_end or end_at == 0:
            if gap_separator != "":
                merged += gap_separator
            merged += content
            if end_at > 0:
                merged_end = end_at
            continue
        if end_at > merged_end:
            merged = _append_with_overlap(merged, content, merged_end - start_at)
            merged_end = end_at
    return merged


# ── LLM output coercion ────────────────────────────────────────────────


def _as_entity_list(raw: JsonValue) -> list[Entity]:
    """Validate the LLM entity array, skipping ``null`` entries.

    A non-array payload or a non-object element is treated as malformed
    output (the upstream unmarshal fails the whole parse in that case),
    raising a provider-classified error.
    """
    if not isinstance(raw, list):
        raise AIProviderError(
            code="graph.entity_extraction_invalid_json",
            message="entity extraction response must be a JSON array",
        )
    entities: list[Entity] = []
    for item in raw:
        if item is None:
            continue
        if not isinstance(item, dict):
            raise AIProviderError(
                code="graph.entity_extraction_invalid_json",
                message="entity extraction response entries must be objects",
            )
        cleaned = {key: "" if value is None else value for key, value in item.items()}
        entities.append(Entity.model_validate(cleaned))
    return entities


def _as_relationship_list(raw: JsonValue) -> list[Relationship]:
    """Validate the LLM relationship array, skipping ``null`` entries."""
    if not isinstance(raw, list):
        raise AIProviderError(
            code="graph.relationship_extraction_invalid_json",
            message="relationship extraction response must be a JSON array",
        )
    relationships: list[Relationship] = []
    for item in raw:
        if item is None:
            continue
        if not isinstance(item, dict):
            raise AIProviderError(
                code="graph.relationship_extraction_invalid_json",
                message="relationship extraction response entries must be objects",
            )
        cleaned = {key: "" if value is None else value for key, value in item.items()}
        relationships.append(Relationship.model_validate(cleaned))
    return relationships


# ── Concurrency helper ─────────────────────────────────────────────────

_T = TypeVar("_T")
_R = TypeVar("_R")


async def _run_limited(
    items: Sequence[_T],
    *,
    limit: int,
    worker: Callable[[_T], Awaitable[_R]],
) -> list[_R]:
    """Apply ``worker`` over ``items`` with at most ``limit`` in flight.

    On the first failure, still-pending work is cancelled and the failure
    is re-raised, mirroring the upstream errgroup semantics used by the
    concurrent extraction phases.
    """
    semaphore = asyncio.Semaphore(limit)

    async def _guarded(item: _T) -> _R:
        async with semaphore:
            return await worker(item)

    tasks: list[asyncio.Task[_R]] = [asyncio.create_task(_guarded(item)) for item in items]
    if not tasks:
        return []
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
    failed = [task for task in done if not task.cancelled() and task.exception() is not None]
    if failed:
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        first_error = failed[0].exception()
        if first_error is not None:
            raise first_error
    return [task.result() for task in tasks]


# ── Service-boundary validation ────────────────────────────────────────


def _validate_chunks(chunks: Sequence[ChunkInput]) -> None:
    """Reject a chunk without a usable id at the service boundary."""
    for chunk in chunks:
        if not chunk.id.strip():
            raise ValidationError(
                code="graph.chunk_id_required",
                message="each chunk must carry a non-blank id",
            )


def _validate_chunk_lookup(chunk_id: str, top_k: int) -> None:
    """Reject a blank chunk id or negative top-k at the service boundary."""
    if not chunk_id.strip():
        raise ValidationError(
            code="graph.chunk_id_required",
            message="chunk ID is required",
        )
    if top_k < 0:
        raise ValidationError(
            code="graph.invalid_top_k",
            message="top_k must be >= 0",
        )


def _as_provider_error(exc: BaseException, *, code: str, phase: str) -> ApplicationError:
    """Project an extraction failure onto a sanctioned exception type.

    Application-layer errors (e.g. an ``AIProviderError`` raised by the
    chat client) keep their type; anything else is wrapped so the builder
    never leaks an arbitrary exception type.
    """
    if isinstance(exc, ApplicationError):
        return exc
    return AIProviderError(f"{phase}: {exc}", code=code)


# ── Mermaid diagram helpers ────────────────────────────────────────────


def _dfs(
    entity_title: str,
    adjacency: dict[str, dict[str, Relationship]],
    visited: set[str],
    component: list[str],
) -> None:
    """Depth-first traversal collecting one connected component.

    Both forward edges (entity -> target) and reverse edges (other
    entities pointing at this one) are followed, so a component contains
    every entity reachable in the undirected relationship graph.
    """
    visited.add(entity_title)
    component.append(entity_title)
    for target in adjacency.get(entity_title, {}):
        if target not in visited:
            _dfs(target, adjacency, visited, component)
    for source, targets in adjacency.items():
        if entity_title in targets and source not in visited:
            _dfs(source, adjacency, visited, component)


# ── Graph builder ──────────────────────────────────────────────────────


class GraphBuilder:
    """Build a knowledge graph from a document's text chunks.

    The builder is stateful: after :meth:`build_graph` the extracted
    entities, relationships and chunk-relation graph are queryable through
    the ``get_*`` accessors. A subsequent build resets the state, so one
    builder instance can process multiple documents.
    """

    def __init__(
        self,
        chat: Chat,
        *,
        extract_entities_prompt: str = DEFAULT_EXTRACT_ENTITIES_PROMPT,
        extract_relationships_prompt: str = DEFAULT_EXTRACT_RELATIONSHIPS_PROMPT,
        language: str = DEFAULT_PROMPT_LANGUAGE,
        max_concurrent_entity_extractions: int = MAX_CONCURRENT_ENTITY_EXTRACTIONS,
        max_concurrent_relation_extractions: int = MAX_CONCURRENT_RELATION_EXTRACTIONS,
        relation_batch_size: int = DEFAULT_RELATION_BATCH_SIZE,
    ) -> None:
        if not extract_entities_prompt.strip():
            raise ValidationError(
                code="graph.prompt_required",
                message="entity extraction prompt is required",
            )
        if not extract_relationships_prompt.strip():
            raise ValidationError(
                code="graph.prompt_required",
                message="relationship extraction prompt is required",
            )
        self._chat = chat
        self._extract_entities_prompt = extract_entities_prompt
        self._extract_relationships_prompt = extract_relationships_prompt
        self._language = language
        self._max_concurrent_entity_extractions = max_concurrent_entity_extractions
        self._max_concurrent_relation_extractions = max_concurrent_relation_extractions
        self._relation_batch_size = relation_batch_size
        self._entities_by_id: dict[str, Entity] = {}
        self._entities_by_title: dict[str, Entity] = {}
        self._relationships: dict[str, Relationship] = {}
        self._chunk_graph: dict[str, dict[str, ChunkRelation]] = {}

    # ── Public API ───────────────────────────────────────────────────

    async def build_graph(self, chunks: Sequence[ChunkInput]) -> GraphBuildResult:
        """Build the knowledge graph from ``chunks`` and return a snapshot.

        Entities are extracted per chunk under the entity concurrency
        cap; a failed extraction aborts the build. Relationships are then
        extracted per batch of chunks (batch failures are non-fatal, so a
        single bad batch does not discard the rest), and the result is
        finalized with relationship weights, entity degrees and the chunk
        relation graph.
        """
        _validate_chunks(chunks)
        self._reset_state()
        start_time = time.monotonic()
        chunk_list = list(chunks)

        chunk_entities = await _run_limited(
            chunk_list,
            limit=self._max_concurrent_entity_extractions,
            worker=self._extract_entities,
        )

        relation_batches: list[tuple[Sequence[ChunkInput], list[Entity]]] = []
        for index in range(0, len(chunk_list), self._relation_batch_size):
            batch_chunks = chunk_list[index : index + self._relation_batch_size]
            batch_entities = [
                entity
                for entities in chunk_entities[index : index + self._relation_batch_size]
                for entity in entities
            ]
            if len(batch_entities) >= MIN_ENTITIES_FOR_RELATION:
                relation_batches.append((batch_chunks, batch_entities))

        await _run_limited(
            relation_batches,
            limit=self._max_concurrent_relation_extractions,
            worker=self._extract_relationships_batch,
        )

        self._calculate_weights()
        self._calculate_degrees()
        self._build_chunk_graph()
        elapsed_seconds = time.monotonic() - start_time
        return GraphBuildResult(
            entities=self.get_all_entities(),
            relationships=self.get_all_relationships(),
            chunk_count=len(self._chunk_graph),
            elapsed_seconds=elapsed_seconds,
        )

    def get_all_entities(self) -> list[Entity]:
        """Return every entity currently in the graph."""
        return list(self._entities_by_id.values())

    def get_all_relationships(self) -> list[Relationship]:
        """Return every relationship currently in the graph."""
        return list(self._relationships.values())

    def get_relation_chunks(self, chunk_id: str, top_k: int) -> list[str]:
        """Return chunks directly related to ``chunk_id``.

        Results are ordered by relation weight (descending) then degree
        (descending); ``top_k`` truncates the list, and a non-positive
        ``top_k`` returns everything.
        """
        _validate_chunk_lookup(chunk_id, top_k)
        weighted = [
            (related_id, relation.weight, relation.degree)
            for related_id, relation in self._chunk_graph.get(chunk_id, {}).items()
        ]
        weighted.sort(key=lambda entry: (-entry[1], -entry[2]))
        result_count = len(weighted)
        if top_k > 0 and top_k < result_count:
            result_count = top_k
        return [entry[0] for entry in weighted[:result_count]]

    def get_indirect_relation_chunks(self, chunk_id: str, top_k: int) -> list[str]:
        """Return chunks found through second-degree connections.

        Direct neighbours of ``chunk_id`` are excluded. An indirect edge
        takes the product of the two traversed relation weights times the
        decay coefficient, and its degree is the larger of the two; when
        several paths reach the same chunk the highest weight wins.
        """
        _validate_chunk_lookup(chunk_id, top_k)
        chunk_relations = self._chunk_graph.get(chunk_id, {})
        direct_chunks = {chunk_id} | set(chunk_relations)
        indirect_map: dict[str, ChunkRelation] = {}
        for direct_chunk_id, direct_relation in chunk_relations.items():
            for indirect_chunk_id, indirect_relation in self._chunk_graph.get(
                direct_chunk_id, {}
            ).items():
                if indirect_chunk_id in direct_chunks:
                    continue
                combined_weight = (
                    direct_relation.weight
                    * indirect_relation.weight
                    * INDIRECT_RELATION_WEIGHT_DECAY
                )
                combined_degree = max(direct_relation.degree, indirect_relation.degree)
                existing = indirect_map.get(indirect_chunk_id)
                if existing is None or combined_weight > existing.weight:
                    indirect_map[indirect_chunk_id] = ChunkRelation(
                        weight=combined_weight, degree=combined_degree
                    )
        weighted = [
            (related_id, relation.weight, relation.degree)
            for related_id, relation in indirect_map.items()
        ]
        weighted.sort(key=lambda entry: (-entry[1], -entry[2]))
        result_count = len(weighted)
        if top_k > 0 and top_k < result_count:
            result_count = top_k
        return [entry[0] for entry in weighted[:result_count]]

    def generate_knowledge_graph_diagram(self) -> str:
        """Render the current graph as a Mermaid diagram.

        Entities are grouped into connected subgraphs (drawn as Mermaid
        subgraphs) and sorted by frequency; relationship edges are styled
        by strength. Entities with no relationships are omitted.
        """
        entities = sorted(self._entities_by_id.values(), key=lambda entity: -entity.frequency)
        relationships = sorted(self._relationships.values(), key=lambda rel: -rel.weight)

        entity_ids = {entity.title: f"E{index}" for index, entity in enumerate(entities)}
        adjacency: dict[str, dict[str, Relationship]] = {entity.title: {} for entity in entities}
        for rel in relationships:
            if rel.source in entity_ids and rel.target in entity_ids:
                adjacency[rel.source][rel.target] = rel

        visited: set[str] = set()
        subgraphs: list[list[str]] = []
        for entity in entities:
            if entity.title not in visited:
                component: list[str] = []
                _dfs(entity.title, adjacency, visited, component)
                if component:
                    subgraphs.append(component)

        parts = [
            "```mermaid\n",
            "graph TD\n",
            "  %% entity style definition\n",
            "  classDef entity fill:#f9f,stroke:#333,stroke-width:1px;\n",
            "  classDef highFreq fill:#bbf,stroke:#333,stroke-width:2px;\n",
            "\n",
        ]
        subgraph_count = 0
        for component in subgraphs:
            node_count = len(component)
            if node_count == 1:
                title = component[0]
                has_relations = any(
                    rel.source == title or rel.target == title for rel in relationships
                )
            else:
                has_relations = True
            if not has_relations:
                continue
            subgraph_count += 1
            parts.append(f"\n  subgraph Subgraph{subgraph_count}\n")
            component_set = set(component)
            for entity_title in component:
                node_id = entity_ids[entity_title]
                if self._entities_by_title.get(entity_title) is not None:
                    parts.append(f'    {node_id}["{entity_title}"]\n')
            for rel in relationships:
                if rel.source in component_set and rel.target in component_set:
                    source_id = entity_ids[rel.source]
                    target_id = entity_ids[rel.target]
                    link_style = "-->" if rel.strength <= 7 else "==>"
                    parts.append(f"    {source_id} {link_style}|{rel.description}| {target_id}\n")
            parts.append("  end\n")
            for entity_title in component:
                node_id = entity_ids[entity_title]
                entity_by_title = self._entities_by_title.get(entity_title)
                if entity_by_title is not None:
                    style = "highFreq" if entity_by_title.frequency > 5 else "entity"
                    parts.append(f"  class {node_id} {style};\n")
        parts.append("```\n")
        return "".join(parts)

    # ── Entity extraction ────────────────────────────────────────────

    async def _extract_entities(self, chunk: ChunkInput) -> list[Entity]:
        """Extract and register the entities of one chunk.

        A chunk with empty content contributes no entities. Entities are
        de-duplicated by title: a title seen before gets the new chunk id
        appended and its frequency incremented, matching the upstream
        aggregation.
        """
        if chunk.content == "":
            return []
        messages = [
            Message(
                role="system",
                content=self._render_graph_extraction_prompt(self._extract_entities_prompt),
            ),
            Message(role="user", content=chunk.content),
        ]
        try:
            response = await self._chat.chat(
                messages,
                ChatOptions(temperature=DEFAULT_LLM_TEMPERATURE, thinking=False),
            )
            extracted = _as_entity_list(parse_llm_json_response(response.content))
        except json.JSONDecodeError as decode_error:
            raise AIProviderError(
                code="graph.entity_extraction_invalid_json",
                message="entity extraction returned an unparseable payload",
            ) from decode_error
        except PydanticValidationError as validation_error:
            raise AIProviderError(
                code="graph.entity_extraction_invalid_json",
                message="entity extraction returned an invalid entity object",
            ) from validation_error
        except Exception as exc:
            raise _as_provider_error(
                exc, code="graph.entity_extraction_failed", phase="entity extraction"
            ) from exc

        result: list[Entity] = []
        for entity in extracted:
            if entity.title == "" or entity.description == "":
                continue
            existing = self._entities_by_title.get(entity.title)
            if existing is None:
                fresh = entity.model_copy(
                    update={
                        "id": str(uuid.uuid4()),
                        "chunk_ids": [chunk.id],
                        "frequency": 1,
                    }
                )
                self._entities_by_title[entity.title] = fresh
                self._entities_by_id[fresh.id] = fresh
                result.append(fresh)
            else:
                chunk_ids = list(existing.chunk_ids)
                if chunk.id not in chunk_ids:
                    chunk_ids.append(chunk.id)
                updated = existing.model_copy(
                    update={"chunk_ids": chunk_ids, "frequency": existing.frequency + 1}
                )
                self._entities_by_title[entity.title] = updated
                self._entities_by_id[updated.id] = updated
                result.append(updated)
        return result

    # ── Relationship extraction ──────────────────────────────────────

    async def _extract_relationships_batch(
        self, batch: tuple[Sequence[ChunkInput], list[Entity]]
    ) -> None:
        """Extract relationships for one batch, never propagating failure.

        A failing batch is logged and skipped so the remaining batches
        still contribute their relationships, mirroring the upstream
        per-batch error handling.
        """
        batch_chunks, entities = batch
        try:
            await self._extract_relationships(batch_chunks, entities)
        except Exception as exc:
            logger.warning("relationship extraction batch failed: %s", exc)

    async def _extract_relationships(
        self, chunks: Sequence[ChunkInput], entities: list[Entity]
    ) -> None:
        """Extract and register the relationships between ``entities``.

        Fewer than two entities yields nothing. Relationships whose source
        and target share no chunk are dropped. An existing relationship is
        replaced by an updated copy: new chunk ids are merged and strength
        becomes a weighted average with integer division, matching the
        upstream aggregation.
        """
        if len(entities) < MIN_ENTITIES_FOR_RELATION:
            return
        entities_json = json.dumps([entity.model_dump() for entity in entities], ensure_ascii=False)
        content = merge_text_chunks(chunks)
        if content == "":
            return
        messages = [
            Message(
                role="system",
                content=self._render_graph_extraction_prompt(self._extract_relationships_prompt),
            ),
            Message(
                role="user",
                content=f"Entities: {entities_json}\n\nText: {content}",
            ),
        ]
        try:
            response = await self._chat.chat(
                messages,
                ChatOptions(temperature=DEFAULT_LLM_TEMPERATURE, thinking=False),
            )
            extracted = _as_relationship_list(parse_llm_json_response(response.content))
        except json.JSONDecodeError as decode_error:
            raise AIProviderError(
                code="graph.relationship_extraction_invalid_json",
                message="relationship extraction returned an unparseable payload",
            ) from decode_error
        except PydanticValidationError as validation_error:
            raise AIProviderError(
                code="graph.relationship_extraction_invalid_json",
                message="relationship extraction returned an invalid relationship object",
            ) from validation_error
        except Exception as exc:
            raise _as_provider_error(
                exc,
                code="graph.relationship_extraction_failed",
                phase="relationship extraction",
            ) from exc

        for relationship in extracted:
            key = f"{relationship.source}#{relationship.target}"
            relation_chunk_ids = self._find_relation_chunk_ids(
                relationship.source, relationship.target, entities
            )
            if not relation_chunk_ids:
                continue
            existing = self._relationships.get(key)
            if existing is None:
                self._relationships[key] = relationship.model_copy(
                    update={"id": str(uuid.uuid4()), "chunk_ids": relation_chunk_ids}
                )
            else:
                merged_chunk_ids = list(existing.chunk_ids)
                for chunk_id in relation_chunk_ids:
                    if chunk_id not in merged_chunk_ids:
                        merged_chunk_ids.append(chunk_id)
                strength = existing.strength
                if merged_chunk_ids:
                    strength = (
                        existing.strength * len(merged_chunk_ids) + relationship.strength
                    ) // (len(merged_chunk_ids) + 1)
                self._relationships[key] = existing.model_copy(
                    update={"chunk_ids": merged_chunk_ids, "strength": strength}
                )

    def _find_relation_chunk_ids(
        self, source: str, target: str, entities: Sequence[Entity]
    ) -> list[str]:
        """Return the union of chunk ids held by ``source`` and ``target``."""
        relation_chunk_ids: set[str] = set()
        for entity in entities:
            if entity.title == source or entity.title == target:
                relation_chunk_ids.update(entity.chunk_ids)
        return sorted(relation_chunk_ids)

    # ── Post-processing ──────────────────────────────────────────────

    def _calculate_weights(self) -> None:
        """Assign each relationship a 1-10 weight from PMI and strength.

        PMI (pointwise mutual information between the two entity
        frequencies and the co-occurrence count) and relationship
        strength are each normalized to 0-1 and combined with the fixed
        PMI / strength proportions, then scaled into the 1-10 range.
        """
        total_entity_occurrences = 0
        entity_frequency: dict[str, int] = {}
        for entity in self._entities_by_id.values():
            frequency = len(entity.chunk_ids)
            entity_frequency[entity.title] = frequency
            total_entity_occurrences += frequency

        total_rel_occurrences = 0
        for rel in self._relationships.values():
            total_rel_occurrences += len(rel.chunk_ids)

        if total_entity_occurrences == 0 or total_rel_occurrences == 0:
            return

        max_pmi = 0.0
        max_strength = MIN_WEIGHT_VALUE
        pmi_values: dict[str, float] = {}
        for rel in self._relationships.values():
            source_freq = entity_frequency.get(rel.source, 0)
            target_freq = entity_frequency.get(rel.target, 0)
            rel_freq = len(rel.chunk_ids)
            if source_freq > 0 and target_freq > 0 and rel_freq > 0:
                source_probability = source_freq / total_entity_occurrences
                target_probability = target_freq / total_entity_occurrences
                rel_probability = rel_freq / total_rel_occurrences
                pmi = max(
                    math.log2(rel_probability / (source_probability * target_probability)),
                    0.0,
                )
                pmi_values[rel.id] = pmi
                if pmi > max_pmi:
                    max_pmi = pmi
            if rel.strength > max_strength:
                max_strength = float(rel.strength)

        updated: dict[str, Relationship] = {}
        for key, rel in self._relationships.items():
            pmi = pmi_values.get(rel.id, 0.0)
            normalized_pmi = pmi / max_pmi if max_pmi > 0 else 0.0
            normalized_strength = rel.strength / max_strength
            combined_weight = normalized_pmi * PMI_WEIGHT + normalized_strength * STRENGTH_WEIGHT
            scaled_weight = 1.0 + WEIGHT_SCALE_FACTOR * combined_weight
            updated[key] = rel.model_copy(update={"weight": scaled_weight})
        self._relationships = updated

    def _calculate_degrees(self) -> None:
        """Assign entity degrees and relationship combined degrees.

        An entity's degree is the number of relationships it participates
        in (incoming plus outgoing). A relationship's combined degree is
        the sum of its two endpoints' degrees.
        """
        in_degree: dict[str, int] = {}
        out_degree: dict[str, int] = {}
        for rel in self._relationships.values():
            out_degree[rel.source] = out_degree.get(rel.source, 0) + 1
            in_degree[rel.target] = in_degree.get(rel.target, 0) + 1

        updated_entities: dict[str, Entity] = {}
        for entity in self._entities_by_id.values():
            degree = in_degree.get(entity.title, 0) + out_degree.get(entity.title, 0)
            updated_entities[entity.id] = entity.model_copy(update={"degree": degree})
        self._entities_by_id = updated_entities
        self._entities_by_title = {entity.title: entity for entity in updated_entities.values()}

        updated_relationships: dict[str, Relationship] = {}
        for key, rel in self._relationships.items():
            source_entity = self._entities_by_title.get(rel.source)
            target_entity = self._entities_by_title.get(rel.target)
            combined = 0
            if source_entity is not None and target_entity is not None:
                combined = source_entity.degree + target_entity.degree
            updated_relationships[key] = rel.model_copy(update={"combined_degree": combined})
        self._relationships = updated_relationships

    def _build_chunk_graph(self) -> None:
        """Connect every pair of chunks co-referenced by a relationship.

        Two chunks are linked when one relationship's source and target
        entities both mention them; the edge carries the relationship's
        weight and combined degree, symmetric in both directions.
        """
        chunk_graph: dict[str, dict[str, ChunkRelation]] = {}
        for rel in self._relationships.values():
            source_entity = self._entities_by_title.get(rel.source)
            target_entity = self._entities_by_title.get(rel.target)
            if source_entity is None or target_entity is None:
                continue
            relation = ChunkRelation(weight=rel.weight, degree=rel.combined_degree)
            for source_chunk_id in source_entity.chunk_ids:
                for target_chunk_id in target_entity.chunk_ids:
                    chunk_graph.setdefault(source_chunk_id, {})[target_chunk_id] = relation
                    chunk_graph.setdefault(target_chunk_id, {})[source_chunk_id] = relation
        self._chunk_graph = chunk_graph

    # ── Internal helpers ─────────────────────────────────────────────

    def _reset_state(self) -> None:
        """Drop all extracted state so the builder can process a new document."""
        self._entities_by_id = {}
        self._entities_by_title = {}
        self._relationships = {}
        self._chunk_graph = {}

    def _render_graph_extraction_prompt(self, template: str) -> str:
        """Fill the shared prompt placeholders for a graph extraction template."""
        return render_prompt_placeholders(template, {"language": self._language})


__all__ = [
    "DEFAULT_EXTRACT_ENTITIES_PROMPT",
    "DEFAULT_EXTRACT_RELATIONSHIPS_PROMPT",
    "DEFAULT_LLM_TEMPERATURE",
    "DEFAULT_PROMPT_LANGUAGE",
    "DEFAULT_RELATION_BATCH_SIZE",
    "INDIRECT_RELATION_WEIGHT_DECAY",
    "MAX_CONCURRENT_ENTITY_EXTRACTIONS",
    "MAX_CONCURRENT_RELATION_EXTRACTIONS",
    "MIN_ENTITIES_FOR_RELATION",
    "MIN_WEIGHT_VALUE",
    "PMI_WEIGHT",
    "STRENGTH_WEIGHT",
    "WEIGHT_SCALE_FACTOR",
    "GraphBuilder",
    "merge_text_chunks",
    "parse_llm_json_response",
    "render_prompt_placeholders",
]
