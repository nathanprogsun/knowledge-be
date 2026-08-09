"""Shared helpers for the chunk-merge pipeline steps.

Pure content utilities (overlap-aware chunk joining, containment checks,
token-level overlap, ``image_info`` filtering) plus the chunk-table seam
the merge steps fetch parent / neighbour / FAQ chunks through. Token and
signature primitives are reused from the retrieval-tool text helpers so
the whole chat domain agrees on one tokenization / signature vocabulary.
"""

from __future__ import annotations

import json
import re
from typing import Protocol, runtime_checkable

from src.common.json import JsonObject
from src.core.agents.tools.text_utils import (
    build_content_signature,
    parse_image_infos,
    tokenize_simple,
)
from src.core.chat.pipeline.types import Context, SearchResult
from src.db.models.chunk import Chunk

#: Shortest suffix/prefix that participates in overlap detection.
MIN_OVERLAP_RUNES = 12
#: Upper bound for the suffix-match window when joining chunk bodies.
DEFAULT_SEARCH_SPAN = 400
#: Partial-overlap removal threshold (token overlap coefficient).
OVERLAP_THRESHOLD = 0.85

#: Matches a Markdown image link ``![alt](url)``.
_MARKDOWN_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
#: Matches an HTML ``<img>`` tag carrying a quoted ``src`` attribute.
_HTML_IMG_SRC_RE = re.compile(r"(?i)<img\b([^>]*?)\ssrc\s*=\s*['\"]([^'\"]+)['\"]([^>]*)>")


@runtime_checkable
class ChunkSource(Protocol):
    """Chunk-table reads the merge steps depend on (upstream repository)."""

    async def list_chunks_by_ids(
        self,
        ctx: Context,
        tenant_id: int,
        ids: list[str],
    ) -> list[Chunk]: ...

    async def list_chunks_by_parent_ids(
        self,
        ctx: Context,
        tenant_id: int,
        parent_ids: list[str],
    ) -> list[Chunk]: ...


class _ChunkTable(Protocol):
    """Minimal chunk-table surface the merge adapter wraps."""

    async def list_by_ids(self, tenant_id: int, ids: list[str]) -> list[Chunk]: ...

    async def list_by_parent_id(self, tenant_id: int, parent_id: str) -> list[Chunk]: ...


class SqlChunkSource:
    """Adapts the chunks-table DAO to the merge step seam.

    The DAO pages parent-child reads one parent at a time; the adapter
    fans out so the merge steps keep a single batched call shape.
    """

    def __init__(self, table: _ChunkTable) -> None:
        self._table = table

    async def list_chunks_by_ids(
        self,
        ctx: Context,
        tenant_id: int,
        ids: list[str],
    ) -> list[Chunk]:
        return await self._table.list_by_ids(tenant_id, ids)

    async def list_chunks_by_parent_ids(
        self,
        ctx: Context,
        tenant_id: int,
        parent_ids: list[str],
    ) -> list[Chunk]:
        out: list[Chunk] = []
        for parent_id in parent_ids:
            out.extend(await self._table.list_by_parent_id(tenant_id, parent_id))
        return out


# ── Rune / list helpers ────────────────────────────────────────────────


def rune_len(text: str) -> int:
    """Return the length of ``text`` in code points."""
    return len(text)


def contains_id(ids: list[str], target: str) -> bool:
    """Return whether ``target`` appears in ``ids``."""
    return target in ids


# ── Chunk-body joining ─────────────────────────────────────────────────


def contains_chunk_content(container: str, contained: str) -> bool:
    """Return whether one current body is safely represented by another.

    Very short substrings are not treated as containment because common
    words and punctuation would create false drops.
    """
    if not container or not contained:
        return False
    if container == contained:
        return True
    return len(contained) >= MIN_OVERLAP_RUNES and contained in container


def join_chunk_content(acc: str, next_: str, separator: str) -> str:
    """Join two current chunk bodies without relying on parser offsets.

    Exact containment is collapsed, a real suffix/prefix overlap is
    removed, and otherwise both bodies are retained with ``separator``
    between them. The conservative fallback intentionally prefers small
    duplication over silently dropping edited content.
    """
    if not acc:
        return next_
    if not next_:
        return acc
    if contains_chunk_content(acc, next_):
        return acc
    if contains_chunk_content(next_, acc):
        return next_
    max_overlap = min(len(acc), len(next_))
    # Editable chunks may be much larger than parser-produced chunks. Bound
    # suffix matching so an adversarial large edit cannot turn retrieval
    # into quadratic work; larger unmatched overlap is safely retained as
    # duplication.
    if max_overlap > DEFAULT_SEARCH_SPAN:
        max_overlap = DEFAULT_SEARCH_SPAN
    for overlap in range(max_overlap, MIN_OVERLAP_RUNES - 1, -1):
        if acc[-overlap:] == next_[:overlap]:
            return acc + next_[overlap:]
    return acc + separator + next_


# ── Normalization / overlap scoring ────────────────────────────────────


def normalize_content(text: str) -> str:
    """Return a lowercased, whitespace-collapsed form of ``text``."""
    collapsed = text.lower().strip()
    if not collapsed:
        return ""
    return " ".join(collapsed.split())


def is_content_contained(normalized_short: str, normalized_long: str) -> bool:
    """Return whether the normalized short text is a substring of the long."""
    if not normalized_short or not normalized_long:
        return False
    if len(normalized_short) > len(normalized_long):
        return False
    return normalized_short in normalized_long


def content_overlap_ratio(left: str, right: str) -> float:
    """Estimate how much of the smaller content's tokens appear in the larger.

    Uses the overlap coefficient (``|intersection| / |smaller set|``), so a
    value of ``1`` means the smaller set is fully contained.
    """
    left_tokens = tokenize_simple(left)
    right_tokens = tokenize_simple(right)
    if not left_tokens or not right_tokens:
        return 0.0
    small, large = (
        (left_tokens, right_tokens)
        if len(left_tokens) <= len(right_tokens)
        else (right_tokens, left_tokens)
    )
    intersection = len(small & large)
    return intersection / len(small)


# ── image_info helpers ─────────────────────────────────────────────────


def image_urls_in_content(content: str) -> set[str]:
    """Return image URLs referenced by content (Markdown links + HTML ``<img>``)."""
    urls: set[str] = set()
    for match in _MARKDOWN_IMAGE_RE.finditer(content):
        url = match.group(2)
        if url:
            urls.add(url)
    for match in _HTML_IMG_SRC_RE.finditer(content):
        src = match.group(2).strip()
        if src:
            urls.add(src)
    return urls


def filter_image_info_by_content_urls(content: str, image_info_json: str) -> str:
    """Keep only ``image_info`` entries whose URL is referenced by ``content``.

    Returns an empty string when nothing matches or the payload is invalid.
    """
    if not image_info_json:
        return ""
    infos = parse_image_infos(image_info_json)
    if not infos:
        return ""
    urls = image_urls_in_content(content)
    if not urls:
        return ""
    filtered = [info for info in infos if _image_url_key(info) in urls]
    return _marshal_image_infos(filtered)


def prune_markdown_images_by_image_info(content: str, image_info_json: str) -> str:
    """Keep only Markdown images represented by the chunk-scoped metadata.

    Stable after text edits because it matches durable image URLs instead
    of parser character offsets.
    """
    allowed: set[str] = set()
    if image_info_json:
        for info in parse_image_infos(image_info_json):
            url = _image_url_key(info)
            if url:
                allowed.add(url)

    def _replace(match: re.Match[str]) -> str:
        if match.group(2) in allowed:
            return match.group(0)
        return ""

    filtered = _MARKDOWN_IMAGE_RE.sub(_replace, content)
    return collapse_blank_lines(filtered)


def _image_url_key(info: JsonObject) -> str:
    url = info.get("url")
    if isinstance(url, str) and url:
        return url
    original = info.get("original_url")
    if isinstance(original, str) and original:
        return original
    return ""


def _marshal_image_infos(infos: list[JsonObject]) -> str:
    if not infos:
        return ""
    return json.dumps(infos, ensure_ascii=False)


def collapse_blank_lines(text: str) -> str:
    """Collapse runs of blank lines and trim surrounding whitespace."""
    result = text
    while "\n\n\n" in result:
        result = result.replace("\n\n\n", "\n\n")
    return result.strip()


# ── Deduplication ──────────────────────────────────────────────────────


def remove_duplicate_results(results: list[SearchResult]) -> list[SearchResult]:
    """Deduplicate by exact chunk ID and normalized content signature.

    Shared ``parent_chunk_id`` values are deliberately NOT treated as
    duplicates: different child chunks of the same parent carry different
    content segments that may all be relevant.
    """
    seen_ids: set[str] = set()
    seen_signatures: dict[str, str] = {}
    unique: list[SearchResult] = []
    for result in results:
        if result.id in seen_ids:
            continue
        signature = build_content_signature(result.content)
        if signature:
            if signature in seen_signatures:
                continue
            seen_signatures[signature] = result.id
        seen_ids.add(result.id)
        unique.append(result)
    return unique


def remove_partial_overlaps(results: list[SearchResult]) -> list[SearchResult]:
    """Drop chunks largely contained within a higher-scored chunk.

    The input MUST already be deduplicated by id/signature. Within each
    pair the lower-scored chunk is the removal candidate; ties are broken
    by content length (longer wins).
    """
    if len(results) <= 1:
        return results
    entries = [(normalize_content(result.content), result) for result in results]
    removed: set[int] = set()
    for i in range(len(entries)):
        if i in removed:
            continue
        for j in range(i + 1, len(entries)):
            if j in removed:
                continue
            if len(entries[i][0]) > len(entries[j][0]):
                short_idx, long_idx = j, i
            else:
                short_idx, long_idx = i, j
            contained = is_content_contained(entries[short_idx][0], entries[long_idx][0])
            if not contained:
                ratio = content_overlap_ratio(
                    entries[short_idx][1].content,
                    entries[long_idx][1].content,
                )
                if ratio < OVERLAP_THRESHOLD:
                    continue
            victim = short_idx
            if entries[short_idx][1].score > entries[long_idx][1].score:
                victim = long_idx
            removed.add(victim)
    return [result for index, (_, result) in enumerate(entries) if index not in removed]


# ── Deterministic ordering ─────────────────────────────────────────────


def search_result_sort_key(result: SearchResult) -> tuple[float, str, str, int, str]:
    """Deterministic relevance-order key after map-based grouping."""
    return (-result.score, result.knowledge_id, result.chunk_type, result.chunk_index, result.id)


__all__ = [
    "DEFAULT_SEARCH_SPAN",
    "MIN_OVERLAP_RUNES",
    "OVERLAP_THRESHOLD",
    "ChunkSource",
    "SqlChunkSource",
    "collapse_blank_lines",
    "contains_chunk_content",
    "contains_id",
    "content_overlap_ratio",
    "filter_image_info_by_content_urls",
    "image_urls_in_content",
    "is_content_contained",
    "join_chunk_content",
    "normalize_content",
    "prune_markdown_images_by_image_info",
    "remove_duplicate_results",
    "remove_partial_overlaps",
    "rune_len",
    "search_result_sort_key",
]
