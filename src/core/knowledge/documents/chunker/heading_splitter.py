"""Tier 1: Markdown heading-aware chunking.

Documents with proper heading structure are split at heading boundaries and
each chunk is prefixed with a breadcrumb of active heading context (e.g.
``# Chapter 1\\n## Section 1.2``) delivered via :attr:`Chunk.context_header`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from src.core.knowledge.documents.chunker.heading_hierarchy import HeadingHierarchy
from src.core.knowledge.documents.chunker.patterns import MARKDOWN_HEADING_PATTERN
from src.core.knowledge.documents.chunker.profiler import DocProfile, profile_document
from src.core.knowledge.documents.chunker.splitter import Chunk, SplitterConfig, split_text


@dataclass(frozen=True)
class HeadingBoundary:
    """Marks where a section starts.

    The first boundary is at rune offset 0 (covers any preamble before the
    first heading); subsequent boundaries sit at headings whose level is
    <= ``primary_level``.
    """

    rune_start: int
    line: str  # raw heading line, "" when this is the leading boundary


def split_by_headings(
    text: str, cfg: SplitterConfig, profile: DocProfile | None = None
) -> list[Chunk]:
    """Tier-1 implementation; falls through to the legacy splitter when the
    document has no usable heading structure or the split would produce a
    single section anyway.

    ``profile`` may be None; it is computed on demand. When the strategy
    resolver already ran the profiler (auto strategy), the same profile is
    threaded through here so the document is not re-scanned.
    """
    if text == "":
        return []
    if profile is None:
        profile = profile_document(text)
    primary_level = profile.dominant_heading_level()
    if primary_level == 0:
        return split_text(text, cfg)

    bounds = find_heading_boundaries(text, primary_level)
    if len(bounds) <= 1:
        return split_text(text, cfg)

    hierarchy = HeadingHierarchy()

    # Pre-walk every heading (not just primary-level) so the hierarchy
    # reflects the full nesting context for each section's start. Only the
    # breadcrumb is snapshotted at section boundaries; deeper sub-headings
    # inside a section update the hierarchy but do not change the chunk's
    # breadcrumb (chunks within a section share one breadcrumb).
    out: list[Chunk] = []
    seq = 0

    for i, boundary in enumerate(bounds):
        end_rune = len(text)
        if i + 1 < len(bounds):
            end_rune = bounds[i + 1].rune_start
        if boundary.line != "":
            _, _, hierarchy = hierarchy.observe(boundary.line)
        # Catch sub-headings that occur between this primary boundary and
        # the next so the hierarchy stays in sync for subsequent sections.
        # This runs after observing the section header so the breadcrumb
        # reflects the section-leading heading.
        breadcrumb = hierarchy.breadcrumb_with_hashes()
        section_start = hierarchy
        hierarchy = observe_sub_headings(
            text[boundary.rune_start : end_rune], primary_level, hierarchy
        )

        section_content = text[boundary.rune_start : end_rune]
        sec_len = len(section_content)
        if sec_len == 0:
            continue

        bc_len = len(breadcrumb)
        # Single-chunk section: emit as-is, breadcrumb tracked separately.
        if bc_len + 2 + sec_len <= cfg.chunk_size:
            out.append(
                Chunk(
                    content=section_content,
                    context_header=breadcrumb,
                    seq=seq,
                    start=boundary.rune_start,
                    end=end_rune,
                )
            )
            seq += 1
            continue

        # Section too large: defer to the legacy splitter for inner
        # segmentation. Each sub-chunk gets a breadcrumb reflecting the
        # deepest heading active at its start.
        sub_breadcrumbs = section_breadcrumbs(section_content, primary_level, section_start)
        sub_chunks = split_text(section_content, cfg)
        for sub in sub_chunks:
            out.append(
                Chunk(
                    content=sub.content,
                    context_header=breadcrumb_at_offset(sub_breadcrumbs, sub.start, breadcrumb),
                    seq=seq,
                    start=boundary.rune_start + sub.start,
                    end=boundary.rune_start + sub.end,
                )
            )
            seq += 1

    return coalesce_tiny_chunks(out, cfg.chunk_size)


def coalesce_tiny_chunks(chunks: list[Chunk], chunk_size: int) -> list[Chunk]:
    """Merge adjacent small chunks under their shared heading context.

    Documents whose primary sections are mostly short (FAQs, install logs,
    change-lists) otherwise trip the validator's "too many tiny chunks" rule
    and fall through all the way to legacy. Safety:

    - Only merge when ``cur.end == next.start``. That preserves the
      ``End - Start == len(Content)`` invariant and naturally skips legacy
      sub-chunks (which may overlap due to chunk overlap).
    - Stop accumulating once the running chunk reaches the merge target
      (~ChunkSize/2) so chunks aren't packed beyond what the validator
      considers comfortable.
    """
    if len(chunks) <= 1 or chunk_size <= 0:
        return chunks
    target = chunk_size // 2
    if target < 200:
        target = 200

    out: list[Chunk] = []
    cur = chunks[0]
    cur_len = len(cur.content)

    for i in range(1, len(chunks)):
        next_chunk = chunks[i]
        next_len = len(next_chunk.content)
        shared_header = common_heading_prefix(cur.context_header, next_chunk.context_header)
        # Adjacent + still-small + would not blow the size budget -> merge.
        if (
            shared_header != ""
            and cur.end == next_chunk.start
            and cur_len < target
            and cur_len + next_len <= chunk_size
        ):
            cur = replace(
                cur,
                content=cur.content + next_chunk.content,
                context_header=shared_header,
                end=next_chunk.end,
            )
            cur_len += next_len
            continue
        out.append(cur)
        cur = next_chunk
        cur_len = next_len
    out.append(cur)

    # Re-sequence — downstream code expects seq to be a dense 0..N-1 range.
    return [replace(chunk, seq=i) for i, chunk in enumerate(out)]


def common_heading_prefix(a: str, b: str) -> str:
    """Longest line-aligned prefix shared by two breadcrumb strings.

    Heading hierarchies are emitted as ``# Top\\n## Section\\n### Sub``, so a
    line-by-line comparison is sufficient and avoids partial-line truncation
    that would corrupt the breadcrumb.
    """
    if a == b:
        return a
    lines_a = a.split("\n")
    lines_b = b.split("\n")
    common = 0
    for i in range(min(len(lines_a), len(lines_b))):
        if lines_a[i] != lines_b[i]:
            break
        common = i + 1
    if common == 0:
        return ""
    return "\n".join(lines_a[:common])


def find_heading_boundaries(text: str, primary_level: int) -> list[HeadingBoundary]:
    """One boundary at offset 0 plus one per Markdown heading at level
    <= ``primary_level`` that sits outside fenced code blocks.

    Heading detection is line-oriented — a heading must occupy a whole line
    to be recognized.
    """
    bounds = [HeadingBoundary(rune_start=0, line="")]
    if text == "":
        return bounds

    pos = 0
    in_fence = False
    lines = text.split("\n")
    for i, line in enumerate(lines):
        trimmed = line.strip()
        if trimmed.startswith("```"):
            in_fence = not in_fence
            pos += len(line)
            if i < len(lines) - 1:
                pos += 1  # newline
            continue
        if not in_fence:
            m = MARKDOWN_HEADING_PATTERN.search(line)
            if m is not None:
                level = len(m.group(1))
                if 1 <= level <= primary_level and pos > 0:
                    bounds.append(HeadingBoundary(rune_start=pos, line=line))
                if 1 <= level <= primary_level and pos == 0:
                    # First line is a heading — replace the leading boundary.
                    bounds[0] = replace(bounds[0], line=line)
        pos += len(line)
        if i < len(lines) - 1:
            pos += 1  # account for the \n that str.split removed
    return bounds


def observe_sub_headings(text: str, primary_level: int, h: HeadingHierarchy) -> HeadingHierarchy:
    """Feed every heading deeper than ``primary_level`` into the hierarchy.

    Keeps the hierarchy state correct so the breadcrumb at the next primary
    section reflects the truly active stack.
    """
    if text == "":
        return h
    in_fence = False
    for line in text.split("\n"):
        trimmed = line.strip()
        if trimmed.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = MARKDOWN_HEADING_PATTERN.search(line)
        if m is None:
            continue
        level = len(m.group(1))
        if level > primary_level:
            _, _, h = h.observe(line)
    return h


@dataclass(frozen=True)
class SectionBreadcrumb:
    """Pairs a rune offset within a section with the breadcrumb in effect there."""

    rune_start: int
    breadcrumb: str


def section_breadcrumbs(
    text: str, primary_level: int, seed: HeadingHierarchy
) -> list[SectionBreadcrumb]:
    """Record, for each deeper sub-heading, the offset where it takes effect.

    ``seed`` is the hierarchy state at the section's start (already including
    the section heading and its ancestors). The returned list is ordered by
    rune_start and always begins with the seed breadcrumb at offset 0.
    """
    h = seed
    result = [SectionBreadcrumb(rune_start=0, breadcrumb=h.breadcrumb_with_hashes())]
    pos = 0
    in_fence = False
    lines = text.split("\n")
    for i, line in enumerate(lines):
        trimmed = line.strip()
        if trimmed.startswith("```"):
            in_fence = not in_fence
            pos += len(line)
            if i < len(lines) - 1:
                pos += 1
            continue
        if not in_fence:
            m = MARKDOWN_HEADING_PATTERN.search(line)
            if m is not None and len(m.group(1)) > primary_level:
                _, _, h = h.observe(line)
                result.append(
                    SectionBreadcrumb(rune_start=pos, breadcrumb=h.breadcrumb_with_hashes())
                )
        pos += len(line)
        if i < len(lines) - 1:
            pos += 1
    return result


def breadcrumb_at_offset(breadcrumbs: list[SectionBreadcrumb], offset: int, fallback: str) -> str:
    """Breadcrumb in effect at ``offset`` — the last entry whose start <= offset."""
    bc = fallback
    for entry in breadcrumbs:
        if entry.rune_start > offset:
            break
        bc = entry.breadcrumb
    return bc


__all__ = [
    "HeadingBoundary",
    "breadcrumb_at_offset",
    "coalesce_tiny_chunks",
    "common_heading_prefix",
    "find_heading_boundaries",
    "observe_sub_headings",
    "section_breadcrumbs",
    "split_by_headings",
]
