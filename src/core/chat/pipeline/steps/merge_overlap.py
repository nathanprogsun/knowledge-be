"""Sequential body merging for the chunk-merge step.

Joins current chunk bodies without using parser coordinates. Chunks MUST
be pre-sorted by ``chunk_index``; neighbouring bodies are collapsed when
one is textually contained in the other or the indices are consecutive,
keeping the higher score and merging ``image_info`` payloads.
"""

from __future__ import annotations

import json

from src.common.json import JsonObject
from src.core.chat.pipeline.common import pipeline_info
from src.core.chat.pipeline.steps.merge_utils import (
    contains_chunk_content,
    contains_id,
    join_chunk_content,
)
from src.core.chat.pipeline.types import Context, SearchResult


def _decode_image_infos(raw: str) -> list[JsonObject] | None:
    """Decode an ``image_info`` JSON array; ``None`` when the payload is invalid."""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None
    return [item for item in parsed if isinstance(item, dict)]


def merge_image_info_json(target_json: str, source_json: str) -> tuple[str, bool]:
    """Merge ``source_json`` image_info into ``target_json``, dedup by URL.

    Returns the merged payload plus whether the merge reported an error
    (an undecodable *source*). An undecodable *target* payload is replaced
    wholesale by the source, mirroring the upstream behaviour.
    """
    if not source_json:
        return target_json, False
    source_infos = _decode_image_infos(source_json)
    if source_infos is None:
        return target_json, True
    if not source_infos:
        return target_json, False
    target_infos = _decode_image_infos(target_json) if target_json else []
    if target_infos is None:
        return source_json, False
    unique: list[JsonObject] = []
    seen: set[str] = set()
    for info in [*target_infos, *source_infos]:
        url = info.get("url") or info.get("original_url") or ""
        if isinstance(url, str) and url and url not in seen:
            seen.add(url)
            unique.append(info)
    return json.dumps(unique, ensure_ascii=False), False


def merge_sequential_chunks(
    ctx: Context,
    knowledge_id: str,
    chunks: list[SearchResult],
) -> list[SearchResult]:
    """Join sequential or textually-contained chunk bodies into groups.

    ``chunks`` MUST already be sorted by ``chunk_index`` (tie-broken by
    id). Returns the merged results ordered by score descending.
    """
    if not chunks:
        return []
    groups: list[tuple[SearchResult, int]] = [(chunks[0], chunks[0].chunk_index)]
    for current in chunks[1:]:
        last, last_index = groups[-1]
        text_contained = contains_chunk_content(
            last.content, current.content
        ) or contains_chunk_content(current.content, last.content)
        sequential = current.chunk_index == last_index + 1
        if not text_contained and not sequential:
            groups.append((current, current.chunk_index))
            continue

        sub_ids = [*last.sub_chunk_id]
        if not contains_id(sub_ids, current.id):
            sub_ids.append(current.id)
        merged_image, image_error = merge_image_info_json(last.image_info, current.image_info)
        updates: dict[str, str | list[str] | float] = {
            "content": join_chunk_content(last.content, current.content, "\n\n"),
            "sub_chunk_id": sub_ids,
        }
        if not image_error:
            updates["image_info"] = merged_image
            if merged_image:
                refs = len(_decode_image_infos(merged_image) or [])
                pipeline_info(
                    "Merge",
                    "image_merged",
                    {"knowledge_id": knowledge_id, "image_refs": refs},
                )
        if current.score > last.score:
            updates["score"] = current.score
        groups[-1] = (last.model_copy(update=updates), max(last_index, current.chunk_index))

    merged = [result for result, _ in groups]
    return sorted(merged, key=lambda result: result.score, reverse=True)


__all__ = ["merge_image_info_json", "merge_sequential_chunks"]
