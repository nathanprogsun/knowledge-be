"""Recursive text splitting for the adaptive text chunker.

``Chunk`` represents a piece of split text with position tracking:
``Content`` holds exactly the text from the original document between
``Start`` and ``End`` (code-point offsets), so ``End - Start == len(Content)``
for chunks that are not decorated with a synthetic context header. The
splitting itself is a recursive priority-separator splitter with protected
regions (tables, code blocks, LaTeX, Markdown links/images) kept atomic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.core.knowledge.documents.chunker.header_tracker import (
    HeaderTracker,
    header_column_mismatch,
)

# Default chunk sizing constants. Single source of truth for the whole
# chunker package.
#
# DEFAULT_CHUNK_SIZE = 512 chars: ~100-130 English tokens / ~300 Chinese
# tokens. Validated as a strong baseline. Use 200-400 for FAQ-style atomic
# content, 1000-2000 for narrative / argumentative documents.
#
# DEFAULT_CHUNK_OVERLAP = 80 chars (~15% of DEFAULT_CHUNK_SIZE): community-
# recommended sweet spot between recall and storage cost. Use 0 for strictly
# atomic data, 150-200 for long narratives.
DEFAULT_CHUNK_SIZE: int = 512
DEFAULT_CHUNK_OVERLAP: int = 80
DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", "。")

# Maximum size for a protected unit / absolute chunk ceiling. Protected spans
# (e.g. a giant code block) are forcibly re-split when they exceed this so
# downstream embedding APIs never see an oversized payload.
_MAX_PROTECTED_SIZE: int = 7500
_ABSOLUTE_MAX_SIZE: int = 7500

# Regex patterns for content that must not be split.
_PROTECTED_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\$\$.*?\$\$", re.DOTALL),  # LaTeX block math
    re.compile(r"!\[[^\]]*\]\([^)]+\)"),  # Markdown images
    re.compile(r"\[[^\]]*\]\([^)]+\)"),  # Markdown links
    re.compile(  # Table header + separator
        r"[ ]*(?:\|[^|\n]*)+\|[\r\n]+\s*(?:\|\s*:?-{3,}:?\s*)+\|[\r\n]+",
        re.MULTILINE,
    ),
    re.compile(r"[ ]*(?:\|[^|\n]*)+\|[\r\n]+", re.MULTILINE),  # Table rows
    re.compile(r"```(?:\w+)?[\r\n].*?```", re.DOTALL),  # Fenced code blocks
    re.compile(r"`[^`\r\n]+`"),  # Markdown inline code
)


@dataclass(frozen=True)
class Chunk:
    """A piece of split text with position tracking.

    ``context_header`` is a separately-tracked context string (e.g. a
    Markdown heading breadcrumb or a repeated table header) that should be
    prepended at embedding/retrieval time but is NOT part of ``content``.
    Keeping the two apart preserves the position invariant while still
    letting embedding pipelines see the section context.
    """

    content: str
    context_header: str = ""
    seq: int = 0
    start: int = 0
    end: int = 0

    def embedding_content(self) -> str:
        """Text to feed an embedding model: context header + trimmed content.

        ``content`` is returned verbatim from the source document (the
        position invariant requires that), but for embedding the surrounding
        whitespace is trimmed so leading/trailing newlines from boundary
        slices don't dilute the embedded vector or waste tokens.
        """
        body = self.content.strip()
        if not self.context_header:
            return body
        return self.context_header + "\n\n" + body


@dataclass(frozen=True)
class ImageRef:
    """An image reference found within a chunk's content."""

    original_ref: str
    alt_text: str
    start: int  # offset within the chunk content
    end: int


@dataclass(frozen=True)
class SplitterConfig:
    """Configuration for the text splitter.

    ``strategy`` and ``token_limit`` are honored by the strategy entry point;
    the legacy :func:`split_text` path uses only ``chunk_size`` /
    ``chunk_overlap`` / ``separators``.
    """

    chunk_size: int = 0
    chunk_overlap: int = 0
    separators: list[str] = field(default_factory=list)
    strategy: str = ""  # empty = legacy (backwards-compatible)
    token_limit: int = 0  # caps chunk size in approximate tokens; 0 = chars
    languages: list[str] = field(default_factory=list)  # empty = auto-detect


@dataclass(frozen=True)
class ParentChildResult:
    """Two-level chunking output.

    Parent chunks provide context (large window); child chunks are used for
    embedding/retrieval (small window). Each child carries its
    ``parent_index`` so the caller can wire up parent IDs after DB insertion.
    """

    parents: list[Chunk]
    children: list[ChildChunk]


@dataclass(frozen=True)
class ChildChunk:
    """Extends :class:`Chunk` with a reference to its parent."""

    chunk: Chunk
    parent_index: int  # index into ParentChildResult.parents; -1 when standalone


def default_config() -> SplitterConfig:
    """Return sensible defaults."""
    return SplitterConfig(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        separators=list(DEFAULT_SEPARATORS),
    )


@dataclass(frozen=True)
class Span:
    """A half-open ``[start, end)`` region."""

    start: int
    end: int


@dataclass(frozen=True)
class SplitUnit:
    """A piece of text with its original position."""

    text: str
    start: int
    end: int


def _utf8_char_len(seq: bytes) -> int:
    """Length in bytes of the UTF-8 character whose lead byte is ``seq[0]``."""
    b = seq[0]
    if b < 0x80:
        return 1
    if b >> 5 == 0b110:
        return 2
    if b >> 4 == 0b1110:
        return 3
    if b >> 3 == 0b11110:
        return 4
    return 1


def protected_spans_rune(text: str, byte_spans: list[Span]) -> list[Span]:
    """Convert byte-offset protected spans to code-point offsets.

    Python's ``re`` reports matches in code points rather than bytes, so the
    conversion is only meaningful for callers that imported spans from a
    byte-oriented source; for spans produced by :func:`protected_spans` this
    is the identity.
    """
    if not byte_spans:
        return []
    out: list[Span] = []
    rune_idx = 0
    byte_idx = 0
    text_bytes = text.encode("utf-8")
    n = len(text_bytes)
    for s in byte_spans:
        while byte_idx < s.start and byte_idx < n:
            byte_idx += _utf8_char_len(text_bytes[byte_idx:])
            rune_idx += 1
        start_rune = rune_idx
        while byte_idx < s.end and byte_idx < n:
            byte_idx += _utf8_char_len(text_bytes[byte_idx:])
            rune_idx += 1
        out.append(Span(start=start_rune, end=rune_idx))
    return out


def protected_spans(text: str) -> list[Span]:
    """Find all non-overlapping protected regions in ``text``.

    Returns code-point (rune) offsets, which is exactly the coordinate space
    every consumer in this module works in.
    """
    all_matches: list[Span] = []
    for pattern in _PROTECTED_PATTERNS:
        for m in pattern.finditer(text):
            start, end = m.span()
            if end - start > 0:
                all_matches.append(Span(start=start, end=end))
    if not all_matches:
        return []

    # Sort by start ascending, then by length descending.
    all_matches.sort(key=lambda s: (s.start, -(s.end - s.start)))

    # Remove overlaps.
    result: list[Span] = []
    last_end = 0
    for span in all_matches:
        if span.start >= last_end:
            result.append(span)
            last_end = span.end
    return result


def split_by_separators(text: str, separators: list[str], chunk_size: int) -> list[str]:
    """Split ``text`` by separators in priority order, recursively.

    Applies the next separator to any piece still larger than ``chunk_size``.
    ``chunk_size == 0`` disables the recursion guard; callers that don't care
    about size budget (e.g. a final merge pass) pass 0.
    """
    if text == "" or not separators:
        return [text]
    if chunk_size > 0 and len(text) <= chunk_size:
        return [text]

    for i, sep in enumerate(separators):
        if sep == "":
            continue
        pieces = [p for p in re.split("(" + re.escape(sep) + ")", text) if p != ""]
        if len(pieces) <= 1:
            continue

        # Recursively split any piece that is still too large with the
        # remaining (lower-priority) separators.
        out: list[str] = []
        remaining = separators[i + 1 :]
        for piece in pieces:
            if chunk_size > 0 and len(piece) > chunk_size and remaining:
                out.extend(split_by_separators(piece, remaining, chunk_size))
            else:
                out.append(piece)
        return out
    return [text]


def split_text(text: str, cfg: SplitterConfig) -> list[Chunk]:
    """Split ``text`` into chunks with overlap, respecting protected patterns."""
    if text == "":
        return []

    chunk_size = cfg.chunk_size
    chunk_overlap = cfg.chunk_overlap
    separators = cfg.separators

    if chunk_size <= 0:
        chunk_size = 512
    if chunk_overlap < 0:
        chunk_overlap = 0

    # Step 1: Find protected spans.
    protected = protected_spans(text)

    # Step 2: Split non-protected regions by separators, keep protected as
    # atomic units.
    units = build_units_with_protection(text, protected, separators, chunk_size)

    # Step 3: Merge units into chunks with overlap.
    return merge_units(units, chunk_size, chunk_overlap)


def build_units_with_protection(
    text: str,
    protected: list[Span],
    separators: list[str],
    chunk_size: int,
) -> list[SplitUnit]:
    """Split text into units, preserving protected spans as atomic.

    Start/End positions in the returned units are code-point offsets. If a
    protected span exceeds ``_MAX_PROTECTED_SIZE`` it is forcibly split so
    downstream processing never sees an oversized chunk.
    """
    units: list[SplitUnit] = []
    pos = 0
    rune_pos = 0

    for span in protected:
        if span.start > pos:
            pre = text[pos : span.start]
            parts = split_by_separators(pre, separators, chunk_size)
            rune_offset = rune_pos
            for part in parts:
                part_rune_len = len(part)
                units.append(
                    SplitUnit(text=part, start=rune_offset, end=rune_offset + part_rune_len)
                )
                rune_offset += part_rune_len
            rune_pos += len(pre)

        prot_text = text[span.start : span.end]
        prot_rune_len = len(prot_text)

        # If protected content is too large, forcibly split it.
        if prot_rune_len > _MAX_PROTECTED_SIZE:
            offset = 0
            while offset < prot_rune_len:
                chunk_end = offset + _MAX_PROTECTED_SIZE
                if chunk_end > prot_rune_len:
                    chunk_end = prot_rune_len
                else:
                    # Try to break at a newline or space.
                    for i in range(chunk_end - 1, offset, -1):
                        if i <= chunk_end - 200:
                            break
                        if prot_text[i] == "\n" or prot_text[i] == " ":
                            chunk_end = i + 1
                            break

                chunk_text = prot_text[offset:chunk_end]
                chunk_len = chunk_end - offset
                units.append(
                    SplitUnit(
                        text=chunk_text, start=rune_pos + offset, end=rune_pos + offset + chunk_len
                    )
                )
                offset = chunk_end
        else:
            # Normal case: keep protected content as a single unit.
            units.append(SplitUnit(text=prot_text, start=rune_pos, end=rune_pos + prot_rune_len))
        rune_pos += prot_rune_len
        pos = span.end

    if pos < len(text):
        remaining = text[pos:]
        parts = split_by_separators(remaining, separators, chunk_size)
        rune_offset = rune_pos
        for part in parts:
            part_rune_len = len(part)
            units.append(SplitUnit(text=part, start=rune_offset, end=rune_offset + part_rune_len))
            rune_offset += part_rune_len

    return units


def units_text(units: list[SplitUnit]) -> str:
    """Concatenate the text of all units."""
    return "".join(u.text for u in units)


def header_already_present(headers: str, overlap_text: str, unit_text: str) -> bool:
    """True if the column-name row from the header is already in the overlap or unit."""
    if headers in overlap_text or headers in unit_text:
        return True

    # Extract the column-name row (first meaningful non-separator line).
    col_row = header_column_row(headers)
    if col_row == "":
        return False

    return col_row in overlap_text or col_row in unit_text


def header_column_row(header: str) -> str:
    """Extract the column-name line from a header string; "" if none."""
    for line in header.split("\n"):
        line = line.strip()
        if line == "" or "---" in line:
            continue
        # Skip lines that are only pipes/whitespace (empty header rows).
        if not all(ch in "|\t " for ch in line):
            return line
    return ""


def merge_units(units: list[SplitUnit], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Combine split units into chunks with overlap tracking.

    Enforces an absolute maximum chunk size to prevent exceeding downstream
    limits. Active contextual headers (e.g. Markdown table headers) are
    prepended to new chunks so every chunk carries its own header context.
    """
    if not units:
        return []

    ht = HeaderTracker()

    chunks: list[Chunk] = []
    current: list[SplitUnit] = []
    cur_len = 0

    for u in units:
        u_len = len(u.text)

        # If this single unit exceeds absolute max, force split it further.
        if u_len > _ABSOLUTE_MAX_SIZE:
            # Flush current chunk if any.
            if current:
                chunks.append(build_chunk(current, len(chunks)))
                current = []
                cur_len = 0

            # Update header state even for oversized units.
            ht.update(u.text)

            # Split this oversized unit into smaller chunks.
            offset = 0
            while offset < len(u.text):
                chunk_end = offset + _ABSOLUTE_MAX_SIZE
                if chunk_end > len(u.text):
                    chunk_end = len(u.text)
                else:
                    for i in range(chunk_end - 1, offset, -1):
                        if i <= chunk_end - 200:
                            break
                        if u.text[i] == "\n" or u.text[i] == " ":
                            chunk_end = i + 1
                            break

                chunk_text = u.text[offset:chunk_end]
                chunks.append(
                    Chunk(
                        content=chunk_text,
                        seq=len(chunks),
                        start=u.start + offset,
                        end=u.start + chunk_end,
                    )
                )
                offset = chunk_end
            continue

        # Update header tracking.
        ht.update(u.text)
        # Flush at table boundary so the next table is not merged into a chunk
        # that still carries the previous table's prepended header context.
        if ht.header_ended_this_unit and current:
            chunks.append(build_chunk(current, len(chunks)))
            current = []
            cur_len = 0
        headers = ht.get_headers()
        headers_len = len(headers)
        if headers_len > chunk_size:
            headers = ""
            headers_len = 0

        # If adding this unit (plus reserving space for headers in a potential
        # next chunk) would exceed chunk size, flush the current chunk.
        if cur_len + u_len + headers_len > chunk_size and current:
            chunks.append(build_chunk(current, len(chunks)))

            # Keep overlap from the end of current.
            current, cur_len = compute_overlap(current, chunk_overlap, chunk_size, u_len)

            # Shrink overlap further if needed to fit headers + next unit.
            if headers != "" and headers_len + u_len <= chunk_size:
                while current and cur_len + u_len + headers_len > chunk_size:
                    cur_len -= len(current[0].text)
                    current = current[1:]

                # Prepend headers if the column-name context is not already
                # present in the overlap or the next unit being added.
                overlap_text = units_text(current)
                if not header_already_present(
                    headers, overlap_text, u.text
                ) and not header_column_mismatch(headers, u.text):
                    start_pos = u.start if not current else current[0].start
                    h_unit = SplitUnit(text=headers, start=start_pos, end=start_pos)
                    current = [h_unit, *current]
                    cur_len += headers_len

        # Check if adding this unit would exceed absolute max.
        if cur_len + u_len > _ABSOLUTE_MAX_SIZE and current:
            chunks.append(build_chunk(current, len(chunks)))
            current = []
            cur_len = 0

        current = [*current, u]
        cur_len += u_len

    # Flush remaining.
    if current:
        chunks.append(build_chunk(current, len(chunks)))

    return chunks


def build_chunk(units: list[SplitUnit], seq: int) -> Chunk:
    return Chunk(
        content=units_text(units),
        seq=seq,
        start=units[0].start,
        end=units[-1].end,
    )


# Rune length of the longest separator considered by semantic overlap: \r\n\r\n.
_SEMANTIC_OVERLAP_LOOKBEHIND: int = 4


def compute_overlap(
    current: list[SplitUnit],
    chunk_overlap: int,
    chunk_size: int,
    next_len: int,
) -> tuple[list[SplitUnit], int]:
    """Return the semantic suffix to keep for overlap and its total rune length.

    The configured overlap is a hard upper bound, not a raw character slice.
    Boundary detection may inspect up to four additional runes immediately
    before that tail window so a separator cut by the window boundary remains
    visible. Eligible boundaries use this priority: paragraph break
    (1), line break (2), sentence end (3). For boundaries with the same
    priority, the earliest one in the window wins so the useful overlap is as
    large as possible. If the window has no valid semantic boundary, no
    overlap is retained.
    """
    if chunk_overlap <= 0:
        return [], 0

    # Overlap is part of the next chunk's total size. When the incoming unit
    # is already close to the chunk budget, shrink the search window rather
    # than creating an oversized chunk.
    max_overlap = chunk_overlap
    remaining = chunk_size - next_len
    if remaining < max_overlap:
        max_overlap = remaining
    if max_overlap <= 0:
        return [], 0

    window = semantic_overlap_window(current, max_overlap + _SEMANTIC_OVERLAP_LOOKBEHIND)
    if not window:
        return [], 0

    window_text = units_text(window)
    # Coordinates before original_window_start are the lookbehind region.
    original_window_start = len(window_text) - max_overlap
    if original_window_start < 0:
        original_window_start = 0
    boundary_end, ok = find_semantic_overlap_boundary_ending_at_or_after(
        window_text, original_window_start
    )
    if not ok:
        return [], 0

    overlap = trim_units_prefix(window, boundary_end)
    overlap_len = sum(len(u.text) for u in overlap)
    if overlap_len <= 0 or overlap_len > max_overlap or units_text(overlap).strip() == "":
        return [], 0
    return overlap, overlap_len


def semantic_overlap_window(current: list[SplitUnit], max_len: int) -> list[SplitUnit]:
    """Return at most ``max_len`` source-backed runes from the tail of ``current``.

    May slice inside the first retained split unit so a semantic boundary
    inside a large paragraph remains discoverable. Synthetic zero-width units
    (for example repeated table headers) form a hard barrier: crossing them
    would break the Start/End-to-content source invariant.
    """
    if max_len <= 0 or not current:
        return []

    remaining = max_len
    reversed_units: list[SplitUnit] = []
    for u in reversed(current):
        if remaining <= 0:
            break
        u_len = len(u.text)
        if u_len == 0:
            continue
        # Header markers contain generated text but occupy no source range.
        # Do not include or cross them when deriving an overlap suffix.
        if u.start == u.end or u.end - u.start != u_len:
            break

        if u_len <= remaining:
            reversed_units.append(u)
            remaining -= u_len
            continue

        start = u_len - remaining
        reversed_units.append(SplitUnit(text=u.text[start:], start=u.start + start, end=u.end))
        remaining = 0

    if not reversed_units:
        return []
    window: list[SplitUnit] = list(reversed(reversed_units))
    return window


def find_semantic_overlap_boundary(text: str) -> tuple[int, bool]:
    """Rune offset immediately after the selected separator; (0, False) if none."""
    return find_semantic_overlap_boundary_ending_at_or_after(text, 0)


def find_semantic_overlap_boundary_ending_at_or_after(text: str, min_end: int) -> tuple[int, bool]:
    """Apply the priority / earliest-position rules to eligible boundaries.

    Boundaries inside protected regions are ignored, a boundary is invalid
    when only whitespace follows it, and only candidates whose end-exclusive
    rune offset is at least ``min_end`` are considered.
    """
    n = len(text)
    if n == 0:
        return 0, False
    if min_end < 0:
        min_end = 0
    if min_end > n:
        return 0, False

    protected = protected_spans(text)

    def inside_protected(pos: int) -> bool:
        for s in protected:
            if pos < s.start:
                return False
            if pos < s.end:
                return True
        return False

    def has_meaningful_tail(end: int) -> bool:
        return 0 <= end < n and text[end:].strip() != ""

    best_start = 0
    best_end = 0
    best_priority = 0
    found = False

    def consider(start: int, end: int, priority: int) -> None:
        nonlocal best_start, best_end, best_priority, found
        if (
            start < 0
            or end <= start
            or end < min_end
            or end > n
            or inside_protected(start)
            or not has_meaningful_tail(end)
        ):
            return
        candidate_priority = priority
        if (
            not found
            or candidate_priority < best_priority
            or (candidate_priority == best_priority and start < best_start)
        ):
            best_start = start
            best_end = end
            best_priority = candidate_priority
            found = True

    # Mark paragraph-break runes so their component newlines are not also
    # emitted as lower-priority line-break candidates.
    paragraph_rune = [False] * n
    i = 0
    while i < n:
        if (
            i + 3 < n
            and text[i] == "\r"
            and text[i + 1] == "\n"
            and text[i + 2] == "\r"
            and text[i + 3] == "\n"
        ):
            consider(i, i + 4, 1)
            for j in range(i, i + 4):
                paragraph_rune[j] = True
            i += 4
        elif i + 1 < n and text[i] == "\n" and text[i + 1] == "\n":
            consider(i, i + 2, 1)
            paragraph_rune[i] = True
            paragraph_rune[i + 1] = True
            i += 2
        else:
            i += 1

    i = 0
    while i < n:
        if paragraph_rune[i]:
            i += 1
            continue
        if text[i] == "\r" and i + 1 < n and text[i + 1] == "\n" and not paragraph_rune[i + 1]:
            consider(i, i + 2, 2)
            i += 2  # skip the \n so it is not considered as a separate line break
            continue
        if text[i] == "\n":
            consider(i, i + 1, 2)
        i += 1

    for i in range(n):
        ch = text[i]
        if ch in "。？！":  # noqa: RUF001
            consider(i, i + 1, 3)
        elif ch in ".?!" and i + 1 < n and text[i + 1] == " ":
            consider(i, i + 2, 3)

    if not found:
        return 0, False
    return best_end, True


def trim_units_prefix(units: list[SplitUnit], prefix_len: int) -> list[SplitUnit]:
    """Remove ``prefix_len`` source runes while preserving remaining positions."""
    if prefix_len <= 0:
        return list(units)

    remaining = prefix_len
    out: list[SplitUnit] = []
    for u in units:
        u_len = len(u.text)
        if remaining >= u_len:
            remaining -= u_len
            continue
        if remaining > 0:
            u = SplitUnit(text=u.text[remaining:], start=u.start + remaining, end=u.end)
            remaining = 0
        out.append(u)
    return out


def split_text_parent_child(
    text: str, parent_cfg: SplitterConfig, child_cfg: SplitterConfig
) -> ParentChildResult:
    """Perform two-level chunking: large parent chunks, then smaller children.

    The child ``seq`` is globally unique across the entire document.
    """
    parents = split_text(text, parent_cfg)
    if not parents:
        return ParentChildResult(parents=[], children=[])

    new_parents: list[Chunk] = []
    children: list[ChildChunk] = []
    child_seq = 0
    for parent in parents:
        subs = split_text(parent.content, child_cfg)

        parent_index = -1
        if len(subs) > 1 or (len(subs) == 1 and subs[0].content != parent.content):
            parent_index = len(new_parents)
            new_parents.append(parent)

        for sub in subs:
            # Adjust offsets: sub positions are relative to parent content,
            # shift to document-level offsets.
            sub = Chunk(
                content=sub.content,
                context_header=sub.context_header,
                seq=child_seq,
                start=sub.start + parent.start,
                end=sub.end + parent.start,
            )
            children.append(ChildChunk(chunk=sub, parent_index=parent_index))
            child_seq += 1
    return ParentChildResult(parents=new_parents, children=children)


# Matches the nested [![alt](img_url)](link_url) pattern where an image is
# wrapped inside a Markdown link. The URL groups support one level of balanced
# parentheses.
_LINKED_IMAGE_PATTERN: re.Pattern[str] = re.compile(
    r"\[!\[([^\]]*)\]\(([^()\s]*(?:\([^)]*\)[^()\s]*)*)\)\]"
    r"\([^()\s]*(?:\([^)]*\)[^()\s]*)*\)"
)

# Matches Markdown image syntax: ![alt](url). The URL group supports one level
# of balanced parentheses so URLs like https://example.com/item_(abc)/123 are
# captured in full.
_IMAGE_REF_PATTERN: re.Pattern[str] = re.compile(
    r"!\[([^\]]*)\]\(([^()\s]*(?:\([^)]*\)[^()\s]*)*)\)"
)


def unwrap_linked_images(markdown: str) -> str:
    """Replace ``[![alt](img_url)](link_url)`` with just ``![alt](img_url)``.

    Called before any image-extraction regex so only the flat ``![alt](url)``
    form needs to be handled.
    """
    return _LINKED_IMAGE_PATTERN.sub(r"![\1](\2)", markdown)


def extract_image_refs(text: str) -> list[ImageRef]:
    """Extract Markdown image references from ``text``."""
    unwrapped = unwrap_linked_images(text)
    refs: list[ImageRef] = []
    for m in _IMAGE_REF_PATTERN.finditer(unwrapped):
        refs.append(
            ImageRef(
                original_ref=m.group(2),  # URL
                alt_text=m.group(1),  # alt text
                start=m.start(),
                end=m.end(),
            )
        )
    return refs


__all__ = [
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_SEPARATORS",
    "ChildChunk",
    "Chunk",
    "ImageRef",
    "ParentChildResult",
    "Span",
    "SplitUnit",
    "SplitterConfig",
    "build_chunk",
    "compute_overlap",
    "default_config",
    "extract_image_refs",
    "find_semantic_overlap_boundary",
    "find_semantic_overlap_boundary_ending_at_or_after",
    "header_already_present",
    "header_column_row",
    "merge_units",
    "protected_spans",
    "protected_spans_rune",
    "split_by_separators",
    "split_text",
    "split_text_parent_child",
    "trim_units_prefix",
    "units_text",
    "unwrap_linked_images",
]
