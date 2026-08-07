"""Tier 2: boundary-driven chunking.

For documents that lack proper Markdown headings but contain recognizable
structural cues (page breaks, numbered sections, multilingual chapter
markers, visual separators, all-caps section titles, page footers). The
algorithm finds all candidate boundary positions, then performs greedy
bin-packing — accumulating blocks between boundaries into chunks until the
next block would exceed ``chunk_size``. Blocks larger than ``chunk_size`` are
recursively delegated to the legacy splitter for inner segmentation.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.core.knowledge.documents.chunker.patterns import (
    ALL_CAPS_HEADING_PATTERN,
    EXCESSIVE_BLANKS_PATTERN,
    NUMBERED_SECTION_PATTERN,
    PAGE_FOOTER_PATTERN,
    PRIO_ALL_CAPS_HEADING,
    PRIO_BLANK_BLOCK,
    PRIO_CHAPTER_MARKER,
    PRIO_FORM_FEED,
    PRIO_NUMBERED_HEAD,
    PRIO_PAGE_FOOTER,
    PRIO_VISUAL_SEP,
    VISUAL_SEPARATOR_PATTERN,
    chapter_patterns_for_langs,
)
from src.core.knowledge.documents.chunker.profiler import DocProfile
from src.core.knowledge.documents.chunker.splitter import (
    Chunk,
    Span,
    SplitterConfig,
    protected_spans,
    split_text,
)


@dataclass(frozen=True)
class Boundary:
    """A candidate split point in the document."""

    rune_start: int  # rune offset where the next chunk should start
    priority: int = 0


def split_by_heuristics(
    text: str, cfg: SplitterConfig, profile: DocProfile | None = None
) -> list[Chunk]:
    """Tier-2 implementation; falls through to the legacy splitter when no
    heuristic boundaries are found.

    ``profile`` is accepted to keep the tier signatures uniform but is
    currently unused — this tier scans for boundaries directly.
    """
    del profile  # reserved for signature parity across tiers
    if text == "":
        return []
    total_runes = len(text)
    if total_runes <= cfg.chunk_size:
        return split_text(text, cfg)

    bounds = find_heuristic_boundaries(text, cfg.languages)
    # Drop any boundary that falls strictly inside a protected region (table,
    # fenced code block, LaTeX block, etc.) — splitting there would cut
    # through atomic content. Boundaries on a span edge are kept since they
    # align with the protected region start/end.
    protected = protected_spans(text)
    if protected:
        bounds = drop_bounds_inside_spans(bounds, protected)
    if not bounds:
        return split_text(text, cfg)

    # Append a sentinel at end-of-document so the bin-packer can flush.
    bounds = [*bounds, Boundary(rune_start=total_runes)]
    # Always start with a boundary at offset 0 if not already there.
    if bounds[0].rune_start != 0:
        bounds = [Boundary(rune_start=0), *bounds]

    # Greedy bin-packing.
    out: list[Chunk] = []
    seq = 0
    chunk_start = bounds[0].rune_start
    cur_end = chunk_start
    min_chunk_size = cfg.chunk_size // 4
    if min_chunk_size < 50:
        min_chunk_size = 50

    for i in range(1, len(bounds)):
        next_end = bounds[i].rune_start
        block_len = next_end - cur_end

        if block_len > cfg.chunk_size:
            # The block between the previous and this boundary is itself too
            # large to fit in any chunk. Flush current accumulation, then
            # recursively chunk the oversize block via the legacy splitter.
            if cur_end - chunk_start > 0:
                out, seq = append_chunk(out, text, chunk_start, cur_end, seq)
                chunk_start = cur_end
            out, seq = append_oversize_block(out, text, cur_end, next_end, cfg, seq)
            cur_end = next_end
            chunk_start = next_end
            continue

        # Would adding this block exceed the budget?
        accumulated = next_end - chunk_start
        if accumulated > cfg.chunk_size and cur_end - chunk_start >= min_chunk_size:
            # Flush accumulated content as a chunk, restart at cur_end.
            out, seq = append_chunk(out, text, chunk_start, cur_end, seq)
            # Snap overlap start to the nearest semantic boundary or line
            # break instead of slicing mid-line / mid-word.
            chunk_start = apply_overlap_aligned(text, cur_end, cfg.chunk_overlap, bounds)
        cur_end = next_end

    # Flush remaining content.
    if cur_end > chunk_start:
        out, seq = append_chunk(out, text, chunk_start, cur_end, seq)
    return out


def find_heuristic_boundaries(text: str, langs: list[str]) -> list[Boundary]:
    """Scan ``text`` and return boundary positions in ascending order.

    Lower-priority duplicates at the same offset are dropped.
    """
    bounds: list[Boundary] = []

    # Form feeds — strongest single-character boundary.
    for idx in all_rune_indices(text, "\f"):
        bounds.append(Boundary(rune_start=idx, priority=PRIO_FORM_FEED))

    # Per-line patterns walk the text once, line by line.
    lines = text.split("\n")
    chapter_patterns = chapter_patterns_for_langs(langs)
    pos = 0
    in_fence = False
    for i, line in enumerate(lines):
        trimmed = line.strip()
        if trimmed.startswith("```"):
            in_fence = not in_fence
        elif not in_fence:
            rune_start = pos
            added = False
            for pattern in chapter_patterns:
                if pattern.search(line) is not None:
                    bounds.append(Boundary(rune_start=rune_start, priority=PRIO_CHAPTER_MARKER))
                    added = True
                    break
            if not added and NUMBERED_SECTION_PATTERN.search(line) is not None:
                bounds.append(Boundary(rune_start=rune_start, priority=PRIO_NUMBERED_HEAD))
                added = True
            if not added and ALL_CAPS_HEADING_PATTERN.search(line) is not None:
                bounds.append(Boundary(rune_start=rune_start, priority=PRIO_ALL_CAPS_HEADING))
                added = True
            if not added and VISUAL_SEPARATOR_PATTERN.search(line) is not None:
                bounds.append(Boundary(rune_start=rune_start, priority=PRIO_VISUAL_SEP))
                added = True
            if not added and PAGE_FOOTER_PATTERN.search(line) is not None:
                bounds.append(Boundary(rune_start=rune_start, priority=PRIO_PAGE_FOOTER))
        pos += len(line)
        if i < len(lines) - 1:
            pos += 1  # \n

    # Excessive blank blocks (\n{3,}). Match at the *start* of the run so we
    # drop into the next paragraph cleanly.
    for m in EXCESSIVE_BLANKS_PATTERN.finditer(text):
        rune_start = len(text[: m.end()])
        bounds.append(Boundary(rune_start=rune_start, priority=PRIO_BLANK_BLOCK))

    if not bounds:
        return []

    # Sort by position; drop near-duplicate offsets keeping the highest priority.
    bounds.sort(key=lambda b: (b.rune_start, -b.priority))
    deduped: list[Boundary] = []
    prev = -1
    for b in bounds:
        if b.rune_start != prev:
            deduped.append(b)
            prev = b.rune_start
    return deduped


def drop_bounds_inside_spans(bounds: list[Boundary], spans: list[Span]) -> list[Boundary]:
    """Return bounds with entries strictly inside any protected span removed.

    Bounds at a span's start or end are kept — they align with the span edge
    and don't split protected content. ``spans`` must be sorted by start.
    """
    if not spans:
        return bounds
    out: list[Boundary] = []
    for b in bounds:
        dropped = False
        for span in spans:
            if span.start >= b.rune_start:
                break  # remaining spans start at or after b — can't contain b
            if b.rune_start < span.end:
                dropped = True
                break
        if not dropped:
            out.append(b)
    return out


def all_rune_indices(text: str, needle: str) -> list[int]:
    """Every code-point offset where ``needle`` starts in ``text``.

    Only used for single-rune needles like form-feed.
    """
    return [i for i, ch in enumerate(text) if ch == needle]


def append_chunk(
    out: list[Chunk], text: str, start: int, end: int, seq: int
) -> tuple[list[Chunk], int]:
    """Slice ``text[start:end]`` into a Chunk and append it to ``out``.

    Pure-whitespace slices are skipped (boundary clustering can occasionally
    produce them). The content stored is the raw slice — Start/End offsets
    must match its length for downstream reconstruction code; whitespace
    stripping for embedding happens in ``Chunk.embedding_content``.
    """
    if end <= start:
        return out, seq
    raw = text[start:end]
    if raw.strip() == "":
        return out, seq
    return [*out, Chunk(content=raw, seq=seq, start=start, end=end)], seq + 1


def append_oversize_block(
    out: list[Chunk],
    text: str,
    start: int,
    end: int,
    cfg: SplitterConfig,
    seq: int,
) -> tuple[list[Chunk], int]:
    """Recursively chunk a region larger than ``chunk_size`` via the legacy splitter."""
    if end <= start:
        return out, seq
    sub_text = text[start:end]
    subs = split_text(sub_text, cfg)
    for s in subs:
        out = [
            *out,
            Chunk(
                content=s.content,
                seq=seq,
                start=start + s.start,
                end=start + s.end,
            ),
        ]
        seq += 1
    return out, seq


def apply_overlap_aligned(text: str, cur_end: int, overlap: int, bounds: list[Boundary]) -> int:
    """Rune offset where the next chunk should start.

    The target is ``cur_end - overlap``, snapped to the nearest preceding
    boundary (within 2x overlap) or, failing that, the previous newline so
    chunks don't begin mid-line / mid-word. Falls back to the raw target only
    if neither option is available.
    """
    if overlap <= 0:
        return cur_end
    target = cur_end - overlap
    if target < 0:
        target = 0
    # Allowed search window: [cur_end - 2*overlap, cur_end).
    window_start = cur_end - 2 * overlap
    if window_start < 0:
        window_start = 0

    # Prefer a semantic boundary strictly inside the window.
    best_bound = -1
    for b in bounds:
        if window_start <= b.rune_start < cur_end and b.rune_start > best_bound:
            best_bound = b.rune_start
    if best_bound >= 0:
        return best_bound

    # Fallback: scan backwards from ``target`` to the previous newline, but
    # not past window_start so we keep the overlap roughly the right size.
    for i in range(target, window_start, -1):
        if i < len(text) and text[i] == "\n":
            return i + 1
    return target


__all__ = [
    "Boundary",
    "all_rune_indices",
    "apply_overlap_aligned",
    "drop_bounds_inside_spans",
    "find_heuristic_boundaries",
    "split_by_heuristics",
]
