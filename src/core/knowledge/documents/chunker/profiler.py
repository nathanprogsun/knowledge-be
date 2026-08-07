"""Document profiling for the adaptive text chunker.

Scans a document once to gather structure indicators that drive strategy
selection (heading-aware vs heuristic vs recursive). Profiling is cheap (a
few regex passes plus rune counting) and runs before any chunking decision is
made.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

from src.core.knowledge.documents.chunker.patterns import (
    ALL_CAPS_HEADING_PATTERN,
    CHINESE_CHAPTER_PATTERN,
    ENGLISH_CHAPTER_PATTERN,
    GERMAN_CHAPTER_PATTERN,
    MARKDOWN_HEADING_PATTERN,
    NUMBERED_SECTION_PATTERN,
    PAGE_FOOTER_PATTERN,
    VISUAL_SEPARATOR_PATTERN,
)
from src.core.knowledge.documents.chunker.tokens import (
    LangChinese,
    LangEnglish,
    LangGerman,
    LangMixed,
    detect_language,
)


class StrategyTier(StrEnum):
    """Identifies which chunking implementation should run."""

    HEADING = "heading"
    HEURISTIC = "heuristic"
    LEGACY = "legacy"


@dataclass
class DocProfile:
    """Document-level signals used to choose a chunking tier."""

    total_chars: int = 0
    total_lines: int = 0
    avg_line_len: float = 0.0
    std_line_len: float = 0.0

    # Markdown structure.
    md_heading_counts: dict[int, int] = field(default_factory=dict)  # level (1..6) -> count
    md_heading_total: int = 0

    # Heuristic indicators.
    numbered_section_count: int = 0
    all_caps_short_line_count: int = 0
    blank_paragraph_breaks: int = 0
    form_feed_count: int = 0
    visual_sep_count: int = 0
    german_chapter_count: int = 0
    english_chapter_count: int = 0
    chinese_chapter_count: int = 0
    repeated_footer_count: int = 0

    # Content characteristics.
    has_tables: bool = False
    has_code: bool = False
    code_ratio: float = 0.0

    # Detected language hints (best-effort).
    detected_langs: list[str] = field(default_factory=list)

    def heading_density(self) -> float:
        """Share of lines that are Markdown headings."""
        if self.total_lines == 0:
            return 0.0
        return self.md_heading_total / self.total_lines

    def dominant_heading_level(self) -> int:
        """Heading level (1..6) that should drive section splitting.

        Preference order:
        1. The lowest level (closest to root) that has at least 3
           occurrences — a "real" structural backbone of the document.
        2. Otherwise the deepest level present at least once — gives
           finer-grained boundaries for small documents.

        Returns 0 when no Markdown headings exist.
        """
        if self.md_heading_total == 0:
            return 0
        for level in range(1, 7):
            if self.md_heading_counts.get(level, 0) >= 3:
                return level
        for level in range(6, 0, -1):
            if self.md_heading_counts.get(level, 0) > 0:
                return level
        return 0

    def heuristic_marker_total(self) -> int:
        """Sum of the non-Markdown structural markers."""
        return (
            self.numbered_section_count
            + self.german_chapter_count
            + self.english_chapter_count
            + self.chinese_chapter_count
            + self.all_caps_short_line_count
            + self.visual_sep_count
            + self.form_feed_count
        )


def match_heading(line: str, counts: dict[int, int]) -> bool:
    """Increment the appropriate level counter when ``line`` is an ATX heading."""
    m = MARKDOWN_HEADING_PATTERN.search(line)
    if m is None:
        return False
    level = len(m.group(1))
    if level < 1 or level > 6:
        return False
    counts[level] = counts.get(level, 0) + 1
    return True


def profile_document(text: str) -> DocProfile:
    """Run a single pass over ``text`` and return its profile."""
    p = DocProfile()
    if text == "":
        return p

    p.total_chars = len(text)
    p.form_feed_count = text.count("\f")

    lines = text.split("\n")
    p.total_lines = len(lines)

    # First pass: per-line markers and length stats.
    lengths: list[float] = []
    in_fence = False
    code_chars = 0
    for line in lines:
        trimmed = line.strip()

        # Toggle fenced-code state. A 3-backtick prefix detector is used
        # rather than a full regex so we don't have to fight with the
        # protected-pattern logic later.
        if trimmed.startswith("```"):
            in_fence = not in_fence
            p.has_code = True
            continue
        if in_fence:
            code_chars += len(line)
            continue

        line_len = len(line)
        lengths.append(float(line_len))

        if match_heading(line, p.md_heading_counts):
            p.md_heading_total += 1
            continue
        if NUMBERED_SECTION_PATTERN.search(line) is not None:
            p.numbered_section_count += 1
        if GERMAN_CHAPTER_PATTERN.search(line) is not None:
            p.german_chapter_count += 1
        if ENGLISH_CHAPTER_PATTERN.search(line) is not None:
            p.english_chapter_count += 1
        if CHINESE_CHAPTER_PATTERN.search(line) is not None:
            p.chinese_chapter_count += 1
        if ALL_CAPS_HEADING_PATTERN.search(line) is not None:
            p.all_caps_short_line_count += 1
        if VISUAL_SEPARATOR_PATTERN.search(line) is not None:
            p.visual_sep_count += 1
        if PAGE_FOOTER_PATTERN.search(line) is not None:
            p.repeated_footer_count += 1
        if trimmed.startswith("|") and trimmed.endswith("|"):
            p.has_tables = True

    if lengths:
        total = sum(lengths)
        p.avg_line_len = total / len(lengths)
        variance = sum((line_len - p.avg_line_len) ** 2 for line_len in lengths) / len(lengths)
        p.std_line_len = math.sqrt(variance)

    if p.total_chars > 0:
        p.code_ratio = code_chars / p.total_chars

    p.blank_paragraph_breaks = text.count("\n\n\n")

    # Sample a slice of the document for language detection — avoids paying an
    # O(N) scan cost on huge inputs while still giving a stable signal.
    sample = text[:4096]
    lang = detect_language(sample)
    p.detected_langs = [lang]
    if lang == LangMixed:
        # Provide all three for downstream pattern selection.
        p.detected_langs = [LangEnglish, LangGerman, LangChinese]

    return p


def select_strategy(p: DocProfile | None) -> list[StrategyTier]:
    """Return the ordered tier chain to attempt for this document.

    The first tier is the primary choice; subsequent tiers are fallbacks if
    validation rejects the previous output. The "legacy" tier is appended as
    a final safety net so callers always receive at least one chunk-set.
    """
    if p is None:
        return [StrategyTier.LEGACY]
    chain: list[StrategyTier] = []

    # Tier 1 candidate: Markdown heading-aware.
    if p.md_heading_total >= 3 and p.heading_density() > 0.005 and p.dominant_heading_level() > 0:
        chain.append(StrategyTier.HEADING)

    # Tier 2 candidate: heuristic boundary detection.
    if (
        p.heuristic_marker_total() >= 5
        or p.form_feed_count > 0
        or p.german_chapter_count + p.english_chapter_count + p.chinese_chapter_count > 0
    ):
        chain.append(StrategyTier.HEURISTIC)

    # Legacy is the ultimate fallback: always returns chunks even when
    # validation fails, so callers never get an empty result.
    chain.append(StrategyTier.LEGACY)
    return chain


__all__ = [
    "DocProfile",
    "StrategyTier",
    "match_heading",
    "profile_document",
    "select_strategy",
]
