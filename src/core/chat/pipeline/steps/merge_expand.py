"""Short-context expansion for the chunk-merge step.

Text hits shorter than a floor length are expanded with their document
neighbours (previous / next chunk bodies) fetched from the chunk table,
bounded to a ceiling length and scoped to the same knowledge item.
"""

from __future__ import annotations

from typing import cast

from src.common.json import JsonValue
from src.core.chat.pipeline.common import pipeline_info, pipeline_warn
from src.core.chat.pipeline.steps.merge_utils import (
    ChunkSource,
    contains_id,
    join_chunk_content,
    rune_len,
)
from src.core.chat.pipeline.types import Context, SearchResult
from src.core.knowledge.chunks.types import CHUNK_TYPE_TEXT
from src.db.models.chunk import Chunk

#: Bodies shorter than this (in code points) are candidates for expansion.
MIN_EXPAND_RUNES = 350
#: Expanded bodies are capped at this many code points.
MAX_EXPAND_RUNES = 850

_JOIN_SEPARATOR = "\n\n"


def merge_ordered_content(prev: str, base: str, next_: str, max_len: int) -> str:
    """Join ordered chunk bodies and cap the result to ``max_len`` code points."""
    content = base
    if prev:
        content = join_chunk_content(prev, content, _JOIN_SEPARATOR)
    if next_:
        content = join_chunk_content(content, next_, _JOIN_SEPARATOR)
    if rune_len(content) > max_len:
        return content[:max_len]
    return content


async def fetch_chunks_if_missing(
    ctx: Context,
    chunk_repo: ChunkSource,
    tenant_id: int,
    chunk_map: dict[str, Chunk | None],
    *chunk_ids: str | None,
) -> None:
    """Load any unknown chunk ids into ``chunk_map``; missing ids map to ``None``."""
    missing = [chunk_id for chunk_id in chunk_ids if chunk_id and chunk_id not in chunk_map]
    if not missing:
        return
    try:
        chunks = await chunk_repo.list_chunks_by_ids(ctx, tenant_id, missing)
    except Exception as err:
        pipeline_warn(
            "Merge",
            "expand_fetch_missing_failed",
            {"missing_cnt": len(missing), "error": str(err)},
        )
        chunks = []
    found = {chunk.id for chunk in chunks}
    for chunk in chunks:
        chunk_map[chunk.id] = chunk
    for chunk_id in missing:
        if chunk_id not in found:
            chunk_map[chunk_id] = None


async def expand_short_context_with_neighbors(
    ctx: Context,
    chunk_repo: ChunkSource | None,
    tenant_id: int,
    results: list[SearchResult],
) -> list[SearchResult]:
    """Expand short text hits with neighbouring chunk bodies.

    Neighbouring bodies must belong to the same knowledge item as the base
    chunk; the walk stops when a neighbour belongs elsewhere.
    """
    if not results or chunk_repo is None:
        return results
    if tenant_id == 0:
        pipeline_warn("Merge", "expand_skip", {"reason": "missing_tenant"})
        return results

    targets: list[tuple[int, SearchResult]] = []
    base_ids: set[str] = set()
    for index, result in enumerate(results):
        if not result.id or not result.content:
            continue
        if result.chunk_type != CHUNK_TYPE_TEXT:
            continue
        if rune_len(result.content) >= MIN_EXPAND_RUNES:
            continue
        targets.append((index, result))
        base_ids.add(result.id)
        pipeline_info(
            "Merge",
            "need_expand",
            {
                "chunk_id": result.id,
                "content": result.content,
                "chunk_type": result.chunk_type,
                "len": rune_len(result.content),
            },
        )
    if not targets:
        return results

    chunk_map: dict[str, Chunk | None] = {}
    try:
        chunks = await chunk_repo.list_chunks_by_ids(ctx, tenant_id, list(base_ids))
    except Exception as err:
        pipeline_warn("Merge", "expand_list_base_failed", {"error": str(err)})
        return results
    for chunk in chunks:
        chunk_map[chunk.id] = chunk

    neighbor_ids: set[str] = set()
    for candidate in chunk_map.values():
        if candidate is None:
            continue
        if candidate.pre_chunk_id and candidate.pre_chunk_id not in chunk_map:
            neighbor_ids.add(candidate.pre_chunk_id)
        if candidate.next_chunk_id and candidate.next_chunk_id not in chunk_map:
            neighbor_ids.add(candidate.next_chunk_id)
    if neighbor_ids:
        try:
            neighbors = await chunk_repo.list_chunks_by_ids(ctx, tenant_id, list(neighbor_ids))
        except Exception as err:
            pipeline_warn("Merge", "expand_list_neighbor_failed", {"error": str(err)})
        else:
            for neighbor in neighbors:
                chunk_map[neighbor.id] = neighbor
                pipeline_info(
                    "Merge",
                    "expand_list_neighbor_success",
                    {
                        "neighbor_chunk_id": neighbor.id,
                        "neighbor_content": neighbor.content,
                        "neighbor_chunk_type": neighbor.chunk_type,
                        "neighbor_len": rune_len(neighbor.content),
                    },
                )

    out = list(results)
    for target_index, result in targets:
        await fetch_chunks_if_missing(ctx, chunk_repo, tenant_id, chunk_map, result.id)
        base_chunk = chunk_map.get(result.id)
        if base_chunk is None or not base_chunk.content or base_chunk.chunk_type != CHUNK_TYPE_TEXT:
            continue

        prev_content = ""
        next_content = ""
        prev_ids: list[str] = []
        next_ids: list[str] = []
        prev_cursor = base_chunk.pre_chunk_id
        next_cursor = base_chunk.next_chunk_id

        await fetch_chunks_if_missing(
            ctx, chunk_repo, tenant_id, chunk_map, prev_cursor, next_cursor
        )
        if prev_cursor:
            prev_chunk = chunk_map.get(prev_cursor)
            if prev_chunk is not None and prev_chunk.knowledge_id == base_chunk.knowledge_id:
                prev_content = prev_chunk.content
                prev_ids.append(prev_chunk.id)
                prev_cursor = prev_chunk.pre_chunk_id
            else:
                prev_cursor = ""
        if next_cursor:
            next_chunk = chunk_map.get(next_cursor)
            if next_chunk is not None and next_chunk.knowledge_id == base_chunk.knowledge_id:
                next_content = next_chunk.content
                next_ids.append(next_chunk.id)
                next_cursor = next_chunk.next_chunk_id
            else:
                next_cursor = ""

        merged = ""
        while True:
            merged = merge_ordered_content(
                prev_content, base_chunk.content, next_content, MAX_EXPAND_RUNES
            )
            if merged == "":
                break
            if rune_len(merged) >= MIN_EXPAND_RUNES:
                break
            if not prev_cursor and not next_cursor:
                break

            expanded = False
            if prev_cursor:
                await fetch_chunks_if_missing(ctx, chunk_repo, tenant_id, chunk_map, prev_cursor)
                prev_chunk = chunk_map.get(prev_cursor)
                if prev_chunk is not None and prev_chunk.knowledge_id == base_chunk.knowledge_id:
                    prev_content = join_chunk_content(
                        prev_chunk.content, prev_content, _JOIN_SEPARATOR
                    )
                    prev_ids = [prev_chunk.id, *prev_ids]
                    prev_cursor = prev_chunk.pre_chunk_id
                    expanded = True
                else:
                    prev_cursor = ""

            merged = merge_ordered_content(
                prev_content, base_chunk.content, next_content, MAX_EXPAND_RUNES
            )
            if rune_len(merged) >= MIN_EXPAND_RUNES:
                break

            if next_cursor:
                await fetch_chunks_if_missing(ctx, chunk_repo, tenant_id, chunk_map, next_cursor)
                next_chunk = chunk_map.get(next_cursor)
                if next_chunk is not None and next_chunk.knowledge_id == base_chunk.knowledge_id:
                    next_content = join_chunk_content(
                        next_content, next_chunk.content, _JOIN_SEPARATOR
                    )
                    next_ids.append(next_chunk.id)
                    next_cursor = next_chunk.next_chunk_id
                    expanded = True
                else:
                    next_cursor = ""

            if not expanded:
                break

        if merged == "":
            continue

        before_len = rune_len(result.content)
        sub_ids = [*result.sub_chunk_id]
        for chunk_id in [*prev_ids, *next_ids]:
            if chunk_id and not contains_id(sub_ids, chunk_id):
                sub_ids.append(chunk_id)
        out[target_index] = result.model_copy(update={"content": merged, "sub_chunk_id": sub_ids})
        fields: dict[str, JsonValue] = {
            "chunk_id": result.id,
            "prev_ids": cast("list[JsonValue]", prev_ids),
            "next_ids": cast("list[JsonValue]", next_ids),
            "before_len": before_len,
            "after_len": rune_len(merged),
            "base_content": base_chunk.content,
            "after_content": merged,
            "chunk_type": result.chunk_type,
            "remaining_prev": prev_cursor,
            "remaining_next": next_cursor,
        }
        pipeline_info("Merge", "expand_short_chunk", fields)
    return out


__all__ = [
    "MAX_EXPAND_RUNES",
    "MIN_EXPAND_RUNES",
    "expand_short_context_with_neighbors",
    "fetch_chunks_if_missing",
    "merge_ordered_content",
]
