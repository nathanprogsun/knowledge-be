"""Multilingual regex patterns for the adaptive text chunker.

This module is the single source of truth for the regex patterns used by the
heading-aware and heuristic splitters. Patterns are grouped by purpose
(chapter markers, numbering, separators) and tagged with a priority that the
heuristic splitter uses to rank candidate chunk boundaries.
"""

from __future__ import annotations

import re
from typing import Final

from src.core.knowledge.documents.chunker.tokens import (
    LangChinese,
    LangEnglish,
    LangGerman,
)

# Boundary priority levels for heuristic chunk boundaries. Higher = stronger.
PRIO_FORM_FEED: Final = 100
PRIO_NUMBERED_HEAD: Final = 90
PRIO_CHAPTER_MARKER: Final = 85
PRIO_ALL_CAPS_HEADING: Final = 70
PRIO_VISUAL_SEP: Final = 60
PRIO_PAGE_FOOTER: Final = 50
PRIO_BLANK_BLOCK: Final = 40

# Matches an ATX-style Markdown heading at line start.
# Capture groups: (1) hashes, (2) heading text.
MARKDOWN_HEADING_PATTERN: Final = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$", re.MULTILINE)

# Matches the form-feed control character used by some PDF converters as a
# page break marker.
FORM_FEED_PATTERN: Final = re.compile(r"\f")

# Matches lines starting with numeric or roman numbering followed by a
# non-empty title, e.g. "1. Intro", "2.3 Methods", "IV. Results",
# "2.2.1 用户与权限". The trailing dot after a multi-level numeral is optional
# because many technical documents write "1.1 Foo" without a closing dot.
NUMBERED_SECTION_PATTERN: Final = re.compile(
    r"^[ \t]*(?:\d+(?:\.\d+){1,3}\.?|(?:\d+|[IVX]{1,5})\.)[ \t]+\S.{0,200}$",
    re.MULTILINE,
)

# Matches short all-caps lines (likely section titles rendered without
# Markdown headings). Requires at least 4 letters and up to ~10 words.
# Trailing colons are tolerated.
ALL_CAPS_HEADING_PATTERN: Final = re.compile(
    r"^[ \t]*([A-ZÄÖÜ][A-ZÄÖÜ \-]{3,80}):?\s*$", re.MULTILINE
)

# Matches horizontal rules / divider lines used as section separators in
# plain text or pre-Markdown documents.
VISUAL_SEPARATOR_PATTERN: Final = re.compile(
    r"^[ \t]*(?:-{3,}|={3,}|\*{3,}|_{3,})[ \t]*$", re.MULTILINE
)

# Matches three or more consecutive newlines, which usually denote a hard
# section break.
EXCESSIVE_BLANKS_PATTERN: Final = re.compile(r"\n{3,}")

# Matches typical "Seite X von Y" / "Page X of Y" lines.
PAGE_FOOTER_PATTERN: Final = re.compile(
    r"^[ \t]*(?:Seite|Page|页码?)\s+\d+(?:\s*(?:von|of|/)\s*\d+)?[ \t]*$",
    re.MULTILINE | re.IGNORECASE,
)

# Matches German chapter / section markers.
GERMAN_CHAPTER_PATTERN: Final = re.compile(
    r"^[ \t]*(?:Kapitel|Abschnitt|Teil)\s+(?:[0-9]+|[IVX]{1,5})[\.: ].{0,200}$",
    re.MULTILINE,
)

# Matches English chapter / section markers.
ENGLISH_CHAPTER_PATTERN: Final = re.compile(
    r"^[ \t]*(?:Chapter|Section|Part)\s+(?:[0-9]+|[IVX]{1,5})[\.: ].{0,200}$",
    re.MULTILINE,
)

# Matches CJK chapter / section markers like 第一章, 第3节, 第 1 章
# (whitespace between 第 / numeral / unit is tolerated).
CHINESE_CHAPTER_PATTERN: Final = re.compile(
    r"^[ \t]*第[ \t]*[一二三四五六七八九十百千零〇0-9]+[ \t]*(?:章|节|節|部分|篇)[ \t]?.{0,200}$",  # noqa: RUF001
    re.MULTILINE,
)

# Sentence-level separators tuned per language. Used for fine-grained
# sub-splitting when a section is still too large.
_LANG_SEPARATORS: Final = {
    LangChinese: ("。", "！", "？", "；", "\n"),  # noqa: RUF001
    LangGerman: (". ", "! ", "? ", "; ", "\n"),
    LangEnglish: (". ", "! ", "? ", "; ", "\n"),
}


def sentence_separators(lang: str) -> list[str]:
    """Return sentence-level separators tuned for the given language."""
    if lang in _LANG_SEPARATORS:
        return list(_LANG_SEPARATORS[lang])
    return ["。", "！", "？", "；", ". ", "! ", "? ", "; ", "\n"]  # noqa: RUF001


def chapter_patterns_for_langs(langs: list[str]) -> list[re.Pattern[str]]:
    """Return the chapter-marker regexes that apply for the given language hints.

    An empty / unknown list returns all of them so that auto-detected
    documents still match.
    """
    if not langs:
        return [GERMAN_CHAPTER_PATTERN, ENGLISH_CHAPTER_PATTERN, CHINESE_CHAPTER_PATTERN]
    out: list[re.Pattern[str]] = []
    for lang in langs:
        if lang == LangGerman:
            out.append(GERMAN_CHAPTER_PATTERN)
        elif lang == LangEnglish:
            out.append(ENGLISH_CHAPTER_PATTERN)
        elif lang == LangChinese:
            out.append(CHINESE_CHAPTER_PATTERN)
    if not out:
        out = [GERMAN_CHAPTER_PATTERN, ENGLISH_CHAPTER_PATTERN, CHINESE_CHAPTER_PATTERN]
    return out


__all__ = [
    "ALL_CAPS_HEADING_PATTERN",
    "CHINESE_CHAPTER_PATTERN",
    "ENGLISH_CHAPTER_PATTERN",
    "EXCESSIVE_BLANKS_PATTERN",
    "FORM_FEED_PATTERN",
    "GERMAN_CHAPTER_PATTERN",
    "MARKDOWN_HEADING_PATTERN",
    "NUMBERED_SECTION_PATTERN",
    "PAGE_FOOTER_PATTERN",
    "PRIO_ALL_CAPS_HEADING",
    "PRIO_BLANK_BLOCK",
    "PRIO_CHAPTER_MARKER",
    "PRIO_FORM_FEED",
    "PRIO_NUMBERED_HEAD",
    "PRIO_PAGE_FOOTER",
    "PRIO_VISUAL_SEP",
    "VISUAL_SEPARATOR_PATTERN",
    "chapter_patterns_for_langs",
    "sentence_separators",
]
