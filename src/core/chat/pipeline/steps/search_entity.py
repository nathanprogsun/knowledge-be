"""Entity-graph search pipeline step (upstream ``PluginSearchEntity``).

Resolves the knowledge-graph subgraph matching the classified entities,
collects the fresh chunk ids it references (skipping chunks already
surfaced by earlier search steps), hydrates chunk + knowledge rows,
projects them onto search results (``match_type=GRAPH``) and enriches
image metadata for hits that carry none.

The graph store is a structural seam (a disabled repository returns
``None`` and is skipped); chunk / knowledge hydration happens through the
injected repository protocols so the step stays free of storage access.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from hashlib import md5
from typing import Protocol

from pydantic import ValidationError

from src.ai.graph.types import NameSpace, RetrieveGraphRepository
from src.ai.retrieval.types import MatchType
from src.common.json import JsonObject
from src.core.chat.pipeline.common import pipeline_info, pipeline_warn
from src.core.chat.pipeline.context import PipelineContext
from src.core.chat.pipeline.engine import ERR_SEARCH_NOTHING, Next, PluginError
from src.core.chat.pipeline.types import (
    Context,
    EventType,
    GraphData,
    GraphNode,
    GraphRelation,
    SearchResult,
)
from src.core.knowledge.chunks.types import (
    CHUNK_TYPE_IMAGE_CAPTION,
    CHUNK_TYPE_IMAGE_OCR,
    CHUNK_TYPE_TEXT,
)
from src.core.knowledge.documents.image_update import ImageInfo
from src.db.models.chunk import Chunk
from src.db.models.knowledge import Document


class ChunkStore(Protocol):
    """Chunk rows the entity step hydrates (structural seam)."""

    async def list_by_ids(self, tenant_id: int, ids: list[str]) -> list[Chunk]: ...

    async def list_by_parent_id(self, tenant_id: int, parent_id: str) -> list[Chunk]: ...


class KnowledgeStore(Protocol):
    """Knowledge rows the entity step hydrates (structural seam)."""

    async def get_batch(self, tenant_id: int, ids: list[str]) -> list[Document]: ...


class SearchEntityPlugin:
    """Entity-graph search pipeline step.

    Activates on ``ENTITY_SEARCH``: searches the graph for the classified
    entities, then hydrates and projects the referenced chunks.
    """

    def __init__(
        self,
        *,
        graph_repo: RetrieveGraphRepository,
        chunk_repo: ChunkStore,
        knowledge_repo: KnowledgeStore,
    ) -> None:
        self._graph_repo = graph_repo
        self._chunk_repo = chunk_repo
        self._knowledge_repo = knowledge_repo

    def activation_events(self) -> list[EventType]:
        return [EventType.ENTITY_SEARCH]

    async def on_event(
        self,
        ctx: Context,
        event_type: EventType | str,
        pipeline_ctx: PipelineContext,
        next: Next,
    ) -> PluginError | None:
        if not pipeline_ctx.entity:
            pipeline_info("SearchEntity", "skip", {"reason": "no_entities"})
            return await next()

        knowledge_base_ids = list(pipeline_ctx.entity_kb_ids)
        entity_knowledge = dict(pipeline_ctx.entity_knowledge)
        if not knowledge_base_ids and not entity_knowledge:
            pipeline_warn(
                "SearchEntity",
                "no_kb_scope",
                {"session_id": pipeline_ctx.session_id},
            )
            return await next()

        nodes, relations = await self._search_graphs(
            ctx,
            pipeline_ctx.entity,
            knowledge_base_ids,
            entity_knowledge,
        )
        pipeline_ctx.graph_result = GraphData(node=nodes, relation=relations)

        chunk_ids = filter_seen_chunks(pipeline_ctx.graph_result, pipeline_ctx.search_result)
        if not chunk_ids:
            pipeline_info(
                "SearchEntity",
                "no_new_chunks",
                {"session_id": pipeline_ctx.session_id},
            )
            return await next()

        chunks = await self._chunk_repo.list_by_ids(pipeline_ctx.tenant_id, chunk_ids)
        knowledge_map = {
            knowledge.id: knowledge
            for knowledge in await self._knowledge_repo.get_batch(
                pipeline_ctx.tenant_id,
                [chunk.knowledge_id for chunk in chunks],
            )
        }

        entity_results: list[SearchResult] = []
        for chunk in chunks:
            knowledge = knowledge_map.get(chunk.knowledge_id)
            if knowledge is None:
                pipeline_warn(
                    "SearchEntity",
                    "chunk_knowledge_missing",
                    {"chunk_id": chunk.id, "knowledge_id": chunk.knowledge_id},
                )
                continue
            entity_results.append(chunk_to_search_result(chunk, knowledge))

        entity_results = await enrich_search_results_image_info(
            self._chunk_repo,
            pipeline_ctx.tenant_id,
            entity_results,
        )

        pipeline_ctx.search_result = remove_duplicate_results(
            [*pipeline_ctx.search_result, *entity_results],
        )
        if not pipeline_ctx.search_result:
            pipeline_info(
                "SearchEntity",
                "no_results",
                {"session_id": pipeline_ctx.session_id},
            )
            return ERR_SEARCH_NOTHING

        pipeline_info(
            "SearchEntity",
            "output",
            {
                "session_id": pipeline_ctx.session_id,
                "result_count": len(pipeline_ctx.search_result),
            },
        )
        return await next()

    async def _search_graphs(
        self,
        ctx: Context,
        entity: Sequence[str],
        knowledge_base_ids: Sequence[str],
        entity_knowledge: Mapping[str, str],
    ) -> tuple[list[GraphNode], list[GraphRelation]]:
        """Search the graph per knowledge file when files are known, else per KB."""
        scopes = (
            [
                NameSpace(knowledge_base=kb_id, knowledge=knowledge_id)
                for knowledge_id, kb_id in entity_knowledge.items()
            ]
            if entity_knowledge
            else [NameSpace(knowledge_base=kb_id) for kb_id in knowledge_base_ids]
        )

        async def _search_one(
            namespace: NameSpace,
        ) -> tuple[list[GraphNode], list[GraphRelation]]:
            graph = await self._graph_repo.search_node(namespace, list(entity))
            if graph is None:
                pipeline_warn(
                    "SearchEntity",
                    "graph_search_empty",
                    {"knowledge_base": namespace.knowledge_base, "knowledge": namespace.knowledge},
                )
                return [], []
            pipeline_info(
                "SearchEntity",
                "graph_hits",
                {
                    "knowledge_base": namespace.knowledge_base,
                    "knowledge": namespace.knowledge,
                    "nodes": len(graph.node),
                    "relations": len(graph.relation),
                },
            )
            nodes = [
                GraphNode(name=node.name, chunks=list(node.chunks), attributes=list(node.attributes))
                for node in graph.node
            ]
            relations = [
                GraphRelation(node1=rel.node1, node2=rel.node2, type=rel.type)
                for rel in graph.relation
            ]
            return nodes, relations

        pairs = await asyncio.gather(*(_search_one(scope) for scope in scopes))
        nodes = [node for pair in pairs for node in pair[0]]
        relations = [relation for pair in pairs for relation in pair[1]]
        return nodes, relations


def filter_seen_chunks(graph: GraphData, search_result: Sequence[SearchResult]) -> list[str]:
    """Return node chunk ids not already surfaced in ``search_result``.

    A chunk id seen once (from any earlier search step) is not collected
    again, even when several graph nodes reference it.
    """
    seen = {result.id for result in search_result}
    chunk_ids: list[str] = []
    for node in graph.node:
        for chunk_id in node.chunks:
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            chunk_ids.append(chunk_id)
    return chunk_ids


def chunk_to_search_result(chunk: Chunk, knowledge: Document) -> SearchResult:
    """Project one chunk + knowledge pair as a graph search hit."""
    return SearchResult(
        id=chunk.id,
        content=chunk.content,
        knowledge_id=chunk.knowledge_id,
        chunk_index=chunk.chunk_index,
        knowledge_title=knowledge.title,
        start_at=chunk.start_at,
        end_at=chunk.end_at,
        seq=chunk.chunk_index,
        score=1.0,
        match_type=MatchType.GRAPH,
        metadata=_stringify_metadata(knowledge.metadata),
        chunk_type=chunk.chunk_type,
        parent_chunk_id=chunk.parent_chunk_id or "",
        image_info=chunk.image_info or "",
        knowledge_filename=knowledge.file_name or "",
        knowledge_source=knowledge.source,
        knowledge_channel=knowledge.channel,
        chunk_metadata=chunk.metadata,
        knowledge_base_id=knowledge.knowledge_base_id,
    )


def remove_duplicate_results(results: Sequence[SearchResult]) -> list[SearchResult]:
    """Deduplicate results by exact chunk id, then by content signature.

    Shared parent ids are deliberately not treated as duplicates: child
    chunks of the same parent carry distinct content segments that may all
    be relevant.
    """
    seen: set[str] = set()
    content_sig: dict[str, str] = {}
    unique: list[SearchResult] = []
    for result in results:
        if result.id in seen:
            continue
        signature = build_content_signature(result.content)
        if signature:
            first_id = content_sig.get(signature)
            if first_id is not None:
                continue
            content_sig[signature] = result.id
        seen.add(result.id)
        unique.append(result)
    return unique


def build_content_signature(content: str) -> str:
    """Return a normalized MD5 signature of ``content`` for dedup checks."""
    normalized = " ".join(content.lower().strip().split())
    if not normalized:
        return ""
    return md5(normalized.encode("utf-8")).hexdigest()


async def enrich_search_results_image_info(
    chunk_repo: ChunkStore,
    tenant_id: int,
    results: Sequence[SearchResult],
) -> list[SearchResult]:
    """Fill ``image_info`` for results that carry none by querying image children.

    Frozen results are rebuilt with the enriched field; the returned list
    is a new copy.
    """
    missing = [result for result in results if not result.image_info]
    if not missing:
        return list(results)
    info_map = await _collect_image_info_by_chunk_ids(
        chunk_repo,
        tenant_id,
        [result.id for result in missing],
    )
    if not info_map:
        return list(results)
    return [
        result.model_copy(update={"image_info": info_map[result.id]})
        if result.id in info_map and not result.image_info
        else result
        for result in results
    ]


async def _collect_image_info_by_chunk_ids(
    chunk_repo: ChunkStore,
    tenant_id: int,
    chunk_ids: Sequence[str],
) -> dict[str, str]:
    """Collect merged ``image_info`` JSON per parent chunk id.

    Two-level resolution: text chunks resolve their direct image children
    first; parent-text chunks resolve the text children whose image
    children carry the records (a second query per text child).
    """
    if not chunk_ids:
        return {}

    children: list[Chunk] = []
    for parent_id in chunk_ids:
        children.extend(await chunk_repo.list_by_parent_id(tenant_id, parent_id))
    if not children:
        return {}

    aggregate: dict[str, dict[str, ImageInfo]] = {}

    def add_info(target_id: str, child: Chunk) -> None:
        if not child.image_info:
            return
        infos = _parse_image_info_list(child.image_info)
        if not infos:
            return
        bucket = aggregate.setdefault(target_id, {})
        for info in infos:
            key = info.url or info.original_url
            if not key:
                continue
            existing = bucket.get(key)
            if existing is None:
                bucket[key] = info
            elif info.ocr_text or info.caption:
                bucket[key] = existing.model_copy(
                    update={
                        "ocr_text": info.ocr_text or existing.ocr_text,
                        "caption": info.caption or existing.caption,
                    },
                )

    text_child_ids: list[str] = []
    text_to_parent: dict[str, str] = {}
    for child in children:
        if child.chunk_type in (CHUNK_TYPE_IMAGE_OCR, CHUNK_TYPE_IMAGE_CAPTION):
            add_info(child.parent_chunk_id or "", child)
        elif child.chunk_type == CHUNK_TYPE_TEXT:
            text_child_ids.append(child.id)
            text_to_parent[child.id] = child.parent_chunk_id or ""

    if text_child_ids:
        for text_id in text_child_ids:
            for grandchild in await chunk_repo.list_by_parent_id(tenant_id, text_id):
                if grandchild.chunk_type not in (CHUNK_TYPE_IMAGE_OCR, CHUNK_TYPE_IMAGE_CAPTION):
                    continue
                parent_text_id = text_to_parent.get(grandchild.parent_chunk_id or "")
                if parent_text_id:
                    add_info(parent_text_id, grandchild)

    out: dict[str, str] = {}
    for target_id, bucket in aggregate.items():
        if not bucket:
            continue
        merged = [bucket[key] for key in sorted(bucket)]
        out[target_id] = json.dumps(
            [info.model_dump(mode="json") for info in merged],
            ensure_ascii=False,
        )
    return out


def _parse_image_info_list(image_info: str) -> list[ImageInfo]:
    """Decode an ``image_info`` JSON list of image records, or ``[]``."""
    try:
        raw = json.loads(image_info)
        if not isinstance(raw, list):
            return []
        return [ImageInfo.model_validate(item) for item in raw]
    except (json.JSONDecodeError, ValidationError):
        return []


def _stringify_metadata(raw: JsonObject | None) -> dict[str, str]:
    """Narrow a JSON metadata object to a string map, stringifying values."""
    if not raw:
        return {}
    return {key: "" if value is None else str(value) for key, value in raw.items()}


__all__ = [
    "ChunkStore",
    "KnowledgeStore",
    "SearchEntityPlugin",
    "build_content_signature",
    "chunk_to_search_result",
    "enrich_search_results_image_info",
    "filter_seen_chunks",
    "remove_duplicate_results",
]
