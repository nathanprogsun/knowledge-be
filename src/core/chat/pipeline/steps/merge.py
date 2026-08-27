"""Chunk-merge pipeline step (upstream ``PluginMerge``).

Selects the retrieval input, deduplicates by id and content signature,
injects relevant history references, resolves parent chunks, groups and
merges sequential current bodies, populates FAQ answers, expands short
contexts with neighbouring chunks, and finally removes partial overlaps
before writing the merged reference list back to the run carrier.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from src.common.json import JsonObject
from src.core.agents.tools.text_utils import parse_image_infos
from src.core.chat.pipeline.common import parallel_map, pipeline_info, pipeline_warn
from src.core.chat.pipeline.context import PipelineContext
from src.core.chat.pipeline.engine import Next, PluginError
from src.core.chat.pipeline.steps.merge_expand import expand_short_context_with_neighbors
from src.core.chat.pipeline.steps.merge_faq import populate_faq_answers
from src.core.chat.pipeline.steps.merge_history import filter_history_results
from src.core.chat.pipeline.steps.merge_overlap import merge_sequential_chunks
from src.core.chat.pipeline.steps.merge_utils import (
    ChunkSource,
    contains_id,
    filter_image_info_by_content_urls,
    join_chunk_content,
    prune_markdown_images_by_image_info,
    remove_duplicate_results,
    remove_partial_overlaps,
    rune_len,
    search_result_sort_key,
)
from src.core.chat.pipeline.types import Context, EventType, SearchResult
from src.core.knowledge.chunks.types import (
    CHUNK_TYPE_IMAGE_CAPTION,
    CHUNK_TYPE_IMAGE_OCR,
    CHUNK_TYPE_PARENT_TEXT,
    CHUNK_TYPE_TEXT,
)
from src.db.models.chunk import Chunk

_JOIN_SEPARATOR = "\n\n"


class PluginMerge:
    """The ``CHUNK_MERGE`` pipeline step.

    ``chunk_repo`` is optional: without it the parent-resolution and
    short-context-expansion stages are skipped (their guards mirror the
    upstream nil-repository behaviour).
    """

    def __init__(self, chunk_repo: ChunkSource | None = None) -> None:
        self._chunk_repo = chunk_repo

    def activation_events(self) -> Sequence[EventType]:
        return [EventType.CHUNK_MERGE]

    async def on_event(
        self,
        ctx: Context,
        event_type: EventType | str,
        pipeline_ctx: PipelineContext,
        next: Next,
    ) -> PluginError | None:
        if not pipeline_ctx.needs_retrieval():
            return await next()
        pipeline_info(
            "Merge",
            "input",
            {
                "session_id": pipeline_ctx.session_id,
                "candidate_cnt": len(pipeline_ctx.rerank_result),
            },
        )

        search_result = self._select_input(pipeline_ctx)
        search_result = self._dedup("dedup_summary", search_result)
        search_result = self._inject_history_results(pipeline_ctx, search_result)
        pipeline_info("Merge", "candidate_ready", {"chunk_cnt": len(search_result)})

        if not search_result:
            pipeline_warn(
                "Merge",
                "output",
                {"chunk_cnt": 0, "reason": "no_candidates"},
            )
            return await next()

        tenant_id = pipeline_ctx.tenant_id
        search_result = await resolve_parent_chunks(ctx, self._chunk_repo, tenant_id, search_result)
        merged_chunks = await group_and_merge_current_content(ctx, search_result)
        merged_chunks = await populate_faq_answers(ctx, self._chunk_repo, tenant_id, merged_chunks)
        merged_chunks = await expand_short_context_with_neighbors(
            ctx, self._chunk_repo, tenant_id, merged_chunks
        )
        merged_chunks = await group_and_merge_current_content(ctx, merged_chunks)
        merged_chunks = self._dedup("final_dedup", merged_chunks)
        merged_chunks = remove_partial_overlaps(merged_chunks)

        pipeline_ctx.merge_result = merged_chunks
        return await next()

    def _select_input(self, pipeline_ctx: PipelineContext) -> list[SearchResult]:
        """Pick rerank results when present, else search results by score."""
        if pipeline_ctx.rerank_result:
            return list(pipeline_ctx.rerank_result)
        pipeline_warn("Merge", "fallback", {"reason": "empty_rerank_result"})
        return sorted(pipeline_ctx.search_result, key=lambda result: result.score, reverse=True)

    def _dedup(self, label: str, results: list[SearchResult]) -> list[SearchResult]:
        """Deduplicate with before/after logging when anything was removed."""
        before = len(results)
        out = remove_duplicate_results(results)
        if len(out) < before:
            pipeline_info("Merge", label, {"before": before, "after": len(out)})
        return out

    def _inject_history_results(
        self,
        pipeline_ctx: PipelineContext,
        current: list[SearchResult],
    ) -> list[SearchResult]:
        """Append relevant history references and deduplicate the combined set."""
        history_results = filter_history_results(pipeline_ctx, current)
        if not history_results:
            return current
        pipeline_info(
            "Merge",
            "history_inject",
            {
                "session_id": pipeline_ctx.session_id,
                "history_hits": len(history_results),
            },
        )
        return remove_duplicate_results([*current, *history_results])


# ── Group + sequential merge ───────────────────────────────────────────


async def group_and_merge_current_content(
    ctx: Context,
    results: list[SearchResult],
) -> list[SearchResult]:
    """Group chunks by knowledge + chunk type and merge sequential bodies.

    Each (knowledge, chunk type) bucket is sorted by ``chunk_index`` and
    merged independently; a final deterministic sort restores the global
    relevance order after map-based grouping.
    """
    knowledge_group: dict[str, dict[str, list[SearchResult]]] = {}
    for chunk in results:
        by_type = knowledge_group.setdefault(chunk.knowledge_id, {})
        by_type.setdefault(chunk.chunk_type, []).append(chunk)
    pipeline_info("Merge", "group_summary", {"knowledge_cnt": len(knowledge_group)})

    units: list[tuple[str, list[SearchResult]]] = [
        (knowledge_id, chunks)
        for knowledge_id, by_type in knowledge_group.items()
        for chunks in by_type.values()
    ]

    async def _process(
        _index: int,
        unit: tuple[str, list[SearchResult]],
    ) -> list[SearchResult]:
        knowledge_id, chunks = unit
        pipeline_info(
            "Merge",
            "group_process",
            {"knowledge_id": knowledge_id, "chunk_cnt": len(chunks)},
        )
        ordered = sorted(chunks, key=lambda chunk: (chunk.chunk_index, chunk.id))
        grouped = merge_sequential_chunks(ctx, knowledge_id, ordered)
        pipeline_info(
            "Merge",
            "group_output",
            {"knowledge_id": knowledge_id, "merged_chunks": len(grouped)},
        )
        return grouped

    group_results = await parallel_map(units, 0, _process)
    merged_chunks = [chunk for group in group_results for chunk in group]
    merged_chunks.sort(key=search_result_sort_key)
    pipeline_info("Merge", "output", {"merged_total": len(merged_chunks)})
    return merged_chunks


# ── Parent-child resolution ────────────────────────────────────────────


async def resolve_parent_chunks(
    ctx: Context,
    chunk_repo: ChunkSource | None,
    tenant_id: int,
    results: list[SearchResult],
) -> list[SearchResult]:
    """Expand parent-child retrieval hits with current parent text context.

    Text children and image grandchildren share the same behaviour, while
    image Markdown is scoped to the matched text child by durable URLs
    rather than parser coordinates.
    """
    if not results or chunk_repo is None:
        return results
    if tenant_id == 0:
        pipeline_warn("Merge", "parent_resolve_skip", {"reason": "missing_tenant"})
        return results

    parent_ids = {result.parent_chunk_id for result in results if result.parent_chunk_id}
    if not parent_ids:
        return results
    try:
        parent_chunks = await chunk_repo.list_chunks_by_ids(ctx, tenant_id, list(parent_ids))
    except Exception as err:
        pipeline_warn("Merge", "parent_resolve_failed", {"error": str(err)})
        return results
    parent_map: dict[str, Chunk] = {chunk.id: chunk for chunk in parent_chunks}

    # Image hits have an image -> text -> parent_text chain. Fetch the
    # grandparent only for those results so they retain parent-child context
    # without using editable StartAt/EndAt coordinates.
    image_text_parent_ids = {
        result.parent_chunk_id
        for result in results
        if result.chunk_type in (CHUNK_TYPE_IMAGE_OCR, CHUNK_TYPE_IMAGE_CAPTION)
    }
    if image_text_parent_ids:
        grandparent_ids: list[str] = []
        grandparent_seen: set[str] = set()
        for candidate_parent in parent_chunks:
            if candidate_parent.id not in image_text_parent_ids:
                continue
            if (
                not candidate_parent.parent_chunk_id
                or candidate_parent.chunk_type != CHUNK_TYPE_TEXT
            ):
                continue
            if (
                candidate_parent.parent_chunk_id in parent_map
                or candidate_parent.parent_chunk_id in grandparent_seen
            ):
                continue
            grandparent_seen.add(candidate_parent.parent_chunk_id)
            grandparent_ids.append(candidate_parent.parent_chunk_id)
        if grandparent_ids:
            try:
                grandparents = await chunk_repo.list_chunks_by_ids(ctx, tenant_id, grandparent_ids)
            except Exception as err:
                pipeline_warn("Merge", "grandparent_fetch_failed", {"error": str(err)})
            else:
                for grandparent_row in grandparents:
                    parent_map[grandparent_row.id] = grandparent_row

    text_child_ids = collect_scoped_text_child_ids(results, parent_map)
    scoped_image_info: dict[str, str] = {}
    if text_child_ids:
        scoped_image_info = await collect_image_info_by_chunk_ids(
            ctx, chunk_repo, tenant_id, text_child_ids
        )

    out: list[SearchResult] = []
    for result in results:
        if not result.parent_chunk_id:
            out.append(result)
            continue

        if result.chunk_type == CHUNK_TYPE_TEXT:
            parent = parent_map.get(result.parent_chunk_id)
            if parent is None or not parent.content or parent.chunk_type != CHUNK_TYPE_PARENT_TEXT:
                out.append(result)
                continue
            pipeline_info(
                "Merge",
                "parent_resolve",
                {
                    "child_id": result.id,
                    "parent_id": result.parent_chunk_id,
                    "child_len": rune_len(result.content),
                    "parent_len": rune_len(parent.content),
                    "scoped_img": True,
                },
            )
            result = assign_scoped_image_info(result, scoped_image_info, result.id)
            parent_content = prune_markdown_images_by_image_info(parent.content, result.image_info)
            content = join_chunk_content(parent_content, result.content, _JOIN_SEPARATOR)
            sub_ids = [*result.sub_chunk_id]
            if not contains_id(sub_ids, result.id):
                sub_ids.append(result.id)
            out.append(result.model_copy(update={"content": content, "sub_chunk_id": sub_ids}))

        elif result.chunk_type in (CHUNK_TYPE_IMAGE_OCR, CHUNK_TYPE_IMAGE_CAPTION):
            text_parent = parent_map.get(result.parent_chunk_id)
            if (
                text_parent is None
                or not text_parent.content
                or text_parent.chunk_type != CHUNK_TYPE_TEXT
            ):
                out.append(result)
                continue
            hit_image_info = result.image_info
            content_source = text_parent
            if text_parent.parent_chunk_id:
                grandparent = parent_map.get(text_parent.parent_chunk_id)
                if (
                    grandparent is not None
                    and grandparent.chunk_type == CHUNK_TYPE_PARENT_TEXT
                    and grandparent.content
                ):
                    content_source = grandparent
            result = result.model_copy(
                update={"content": text_parent.content, "chunk_index": text_parent.chunk_index}
            )
            result = assign_scoped_image_info(result, scoped_image_info, text_parent.id)
            if not result.image_info and hit_image_info:
                result = result.model_copy(
                    update={
                        "image_info": filter_image_info_by_content_urls(
                            text_parent.content, hit_image_info
                        )
                    }
                )
            text_content = prune_markdown_images_by_image_info(
                text_parent.content, result.image_info
            )
            parent_content = prune_markdown_images_by_image_info(
                content_source.content, result.image_info
            )
            content = join_chunk_content(parent_content, text_content, _JOIN_SEPARATOR)
            sub_ids = [*result.sub_chunk_id]
            if not contains_id(sub_ids, result.id):
                sub_ids.append(result.id)
            result = result.model_copy(update={"content": content, "sub_chunk_id": sub_ids})
            pipeline_info(
                "Merge",
                "image_parent_resolve",
                {
                    "child_id": result.id,
                    "child_type": result.chunk_type,
                    "text_id": text_parent.id,
                    "parent_id": content_source.id,
                    "match_len": rune_len(result.content),
                    "parent_len": rune_len(content_source.content),
                    "scoped": True,
                },
            )
            out.append(result)

        else:
            out.append(result)
    return out


def collect_scoped_text_child_ids(
    results: list[SearchResult],
    parent_map: dict[str, Chunk],
) -> list[str]:
    """Return text chunk ids whose ``image_info`` should scope the parent merge."""
    seen: set[str] = set()
    ids: list[str] = []
    for result in results:
        if not result.parent_chunk_id:
            continue
        if result.chunk_type == CHUNK_TYPE_TEXT:
            parent = parent_map.get(result.parent_chunk_id)
            if parent is None or parent.chunk_type != CHUNK_TYPE_PARENT_TEXT:
                continue
            if result.id in seen:
                continue
            seen.add(result.id)
            ids.append(result.id)
        elif result.chunk_type in (CHUNK_TYPE_IMAGE_OCR, CHUNK_TYPE_IMAGE_CAPTION):
            if result.parent_chunk_id in seen:
                continue
            seen.add(result.parent_chunk_id)
            ids.append(result.parent_chunk_id)
    return ids


def assign_scoped_image_info(
    result: SearchResult,
    scoped: dict[str, str] | None,
    text_child_id: str,
) -> SearchResult:
    """Set ``image_info`` from the per-text-child map, falling back to content URLs."""
    if scoped:
        info = scoped.get(text_child_id)
        if info:
            return result.model_copy(update={"image_info": info})
    if result.image_info:
        return result.model_copy(
            update={
                "image_info": filter_image_info_by_content_urls(result.content, result.image_info)
            }
        )
    return result


async def collect_image_info_by_chunk_ids(
    ctx: Context,
    chunk_repo: ChunkSource,
    tenant_id: int,
    chunk_ids: list[str],
) -> dict[str, str]:
    """Collect merged ``image_info`` JSON for each chunk id from image children.

    Supports two-level resolution: text chunks' direct image children, and
    parent_text chunks' text children's image grandchildren.
    """
    if not chunk_ids:
        return {}
    try:
        children = await chunk_repo.list_chunks_by_parent_ids(ctx, tenant_id, chunk_ids)
    except Exception as err:
        pipeline_warn("Merge", "image_children_fetch_failed", {"error": str(err)})
        return {}
    if not children:
        return {}

    aggregations: dict[str, dict[str, JsonObject]] = {}

    def _add_info(target_id: str, child: Chunk) -> None:
        if not child.image_info:
            return
        infos = parse_image_infos(child.image_info)
        if not infos:
            return
        bucket = aggregations.setdefault(target_id, {})
        for info in infos:
            key = info.get("url") or info.get("original_url") or ""
            if not isinstance(key, str) or not key:
                continue
            existing = bucket.get(key)
            if existing is None:
                bucket[key] = info
            else:
                ocr_text = info.get("ocr_text")
                if isinstance(ocr_text, str) and ocr_text:
                    existing["ocr_text"] = ocr_text
                caption = info.get("caption")
                if isinstance(caption, str) and caption:
                    existing["caption"] = caption

    text_child_ids: list[str] = []
    text_to_parent: dict[str, str] = {}
    for child in children:
        if child.chunk_type in (CHUNK_TYPE_IMAGE_OCR, CHUNK_TYPE_IMAGE_CAPTION):
            _add_info(child.parent_chunk_id or "", child)
        elif child.chunk_type == CHUNK_TYPE_TEXT:
            text_child_ids.append(child.id)
            text_to_parent[child.id] = child.parent_chunk_id or ""

    if text_child_ids:
        try:
            grandchildren = await chunk_repo.list_chunks_by_parent_ids(
                ctx, tenant_id, text_child_ids
            )
        except Exception as err:
            pipeline_warn("Merge", "image_grandchildren_fetch_failed", {"error": str(err)})
            grandchildren = []
        for grandchild in grandchildren:
            if grandchild.chunk_type not in (CHUNK_TYPE_IMAGE_OCR, CHUNK_TYPE_IMAGE_CAPTION):
                continue
            parent_text_id = text_to_parent.get(grandchild.parent_chunk_id or "")
            if parent_text_id is not None:
                _add_info(parent_text_id, grandchild)

    out: dict[str, str] = {}
    for chunk_id, bucket in aggregations.items():
        if not bucket:
            continue
        out[chunk_id] = json.dumps(list(bucket.values()), ensure_ascii=False)
    return out


__all__ = [
    "PluginMerge",
    "assign_scoped_image_info",
    "collect_image_info_by_chunk_ids",
    "collect_scoped_text_child_ids",
    "group_and_merge_current_content",
    "resolve_parent_chunks",
]
