"""Shared text helpers for the retrieval tools.

Content signatures, tokenization, Jaccard similarity, XML escaping,
regex helpers, and the match-snippet builders shared by the grep and
knowledge-search tools. Length bounds are expressed in code points so
multi-byte text is handled consistently.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from re import Pattern

from src.common.json import JsonObject, JsonValue

#: Bounds for retrieval-tool match snippets.
SNIPPET_CONTEXT_RUNES = 200
SNIPPET_MAX_MATCH_RUNES = 200
SNIPPET_MAX_TOTAL_RUNES = 800
SNIPPET_MAX_ANSWER_RUNES = 600

#: Characters that split a search query into tokens.
_SEARCH_TOKEN_DELIMITERS = frozenset(" \t\n\r,.;:?!()[]{}'\"")

#: CJK unified ideograph ranges used to detect Chinese text.
_CJK_RANGES: tuple[tuple[int, int], ...] = (
    (0x4E00, 0x9FFF),
    (0x3400, 0x4DBF),
    (0xF900, 0xFAFF),
)


def build_content_signature(content: str) -> str:
    """Return a normalized MD5 signature for content duplicate detection.

    The content is lowercased, trimmed, and whitespace-collapsed before
    hashing; empty content yields ``""``.
    """
    normalized = content.lower().strip()
    if not normalized:
        return ""
    normalized = " ".join(normalized.split())
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def _is_cjk_char(char: str) -> bool:
    code = ord(char)
    return any(low <= code <= high for low, high in _CJK_RANGES)


def _contains_chinese(text: str) -> bool:
    return any(_is_cjk_char(char) for char in text)


def _is_all_punct(value: str) -> bool:
    for char in value:
        if char.isspace():
            continue
        category = unicodedata.category(char)
        if not category.startswith(("P", "S")):
            return False
    return True


def _cjk_bigrams(text: str) -> list[str]:
    """Emit consecutive two-character sequences for CJK text.

    Stand-in for a segmentation model so CJK content still yields
    multi-character tokens for similarity work.
    """
    return [text[i : i + 2] for i in range(max(len(text) - 1, 0))]


def tokenize_simple(text: str) -> set[str]:
    """Tokenize text into a set of lowercase multi-character tokens.

    CJK text is segmented into character bigrams; other text is split on
    whitespace. Tokens that are pure punctuation are dropped.
    """
    text = text.lower().strip()
    if not text:
        return set()
    words = _cjk_bigrams(text) if _contains_chinese(text) else text.split()
    tokens: set[str] = set()
    for word in words:
        word = word.strip()
        if len(word) > 1 and not _is_all_punct(word):
            tokens.add(word)
    return tokens


def jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity of two token sets (0..1, 1 = identical)."""
    if not a and not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    intersection = len(a & b)
    union = len(a) + len(b) - intersection
    if union == 0:
        return 0.0
    return intersection / union


def clamp_float(value: float, minimum: float, maximum: float) -> float:
    """Clamp ``value`` into ``[minimum, maximum]``."""
    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value


_XML_REPLACEMENTS: dict[str, str] = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&apos;",
}


def xml_escape(value: str) -> str:
    """Escape a value for safe inclusion in simple XML text / attributes."""
    return "".join(_XML_REPLACEMENTS.get(char, char) for char in value)


def dedup_non_empty_strings(values: list[str]) -> list[str]:
    """Return ``values`` with empties dropped and duplicates collapsed."""
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def truncate_runes(text: str, max_runes: int) -> str:
    """Truncate ``text`` to ``max_runes`` code points (``""`` for invalid)."""
    if max_runes <= 0:
        return ""
    return text[:max_runes]


def parse_image_infos(raw: str | None) -> list[JsonObject]:
    """Decode a chunk's ``image_info`` JSON list leniently."""
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return [item for item in parsed if isinstance(item, dict)]


def _image_info_markdown_metadata(img: JsonObject) -> str:
    lines: list[str] = []
    caption = _as_str(img.get("caption")).strip()
    if caption:
        lines.append("**Image caption:** " + caption)
    ocr = _as_str(img.get("ocr_text")).strip()
    if ocr:
        lines.append("**Image text (OCR):** " + ocr)
    if not lines:
        return ""
    joined = "\n\n".join(lines)
    return "> " + joined.replace("\n", "\n> ")


def build_image_info_markdown(url: str, img: JsonObject) -> str:
    """Format one image as answer-ready Markdown for LLM-facing context.

    The URL is preserved verbatim; the caption becomes the alt text.
    """
    url = url.strip()
    metadata = _image_info_markdown_metadata(img)
    if not url:
        return metadata
    alt = " ".join(_as_str(img.get("caption")).split())
    if not alt:
        alt = "image"
    alt = alt.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")
    image = f"![{alt}]({url})"
    if not metadata:
        return image
    return image + "\n\n" + metadata


def _as_str(value: JsonValue) -> str:
    return value if isinstance(value, str) else ""


def search_query_tokens(queries: list[str]) -> list[str]:
    """Split the queries into lowercase tokens of length >= 2, deduplicated."""
    tokens: list[str] = []
    seen: set[str] = set()
    for query in queries:
        for token in _split_on_delimiters(query):
            token = token.lower().strip()
            if len(token) < 2 or token in seen:
                continue
            seen.add(token)
            tokens.append(token)
    return tokens


def _split_on_delimiters(text: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    for char in text:
        if char in _SEARCH_TOKEN_DELIMITERS:
            if current:
                parts.append("".join(current))
                current = []
        else:
            current.append(char)
    if current:
        parts.append("".join(current))
    return parts


def text_matches_search_queries(text: str, queries: list[str], tokens: list[str]) -> bool:
    """Whether any query or extracted token appears in ``text``."""
    if not text:
        return False
    lowered = text.lower()
    for query in queries:
        clean = query.lower().strip()
        if clean and clean in lowered:
            return True
    return any(token in lowered for token in tokens)


def regex_matches_any(text: str, compiled: list[Pattern[str]]) -> bool:
    """Whether ``text`` matches at least one of the compiled patterns."""
    if not text or not compiled:
        return False
    return any(
        pattern is not None and pattern.search(text) is not None
        for pattern in compiled
    )


def count_regex_hits(
    content: str,
    compiled: list[Pattern[str]],
    patterns: list[str],
) -> dict[str, int]:
    """Count non-overlapping matches per pattern, keyed by pattern string."""
    counts: dict[str, int] = {}
    if not content or not compiled:
        return counts
    for i, pattern in enumerate(compiled):
        if pattern is None or i >= len(patterns):
            continue
        counts[patterns[i]] = len(pattern.findall(content))
    return counts


def extract_snippet_for_queries(content: str, queries: list[str]) -> str:
    """Produce a short contextual snippet around the first query-token hit.

    Falls back to the leading 400 code points of content when no token
    matches so callers always get something to scan.
    """
    content = content.strip()
    if not content:
        return ""

    tokens = search_query_tokens(queries)
    lowered = content.lower()
    earliest = -1
    earliest_end = -1
    for token in tokens:
        index = lowered.find(token)
        if index < 0:
            continue
        end = index + len(token)
        if earliest < 0 or index < earliest:
            earliest = index
            earliest_end = end

    if earliest < 0:
        if len(content) > SNIPPET_CONTEXT_RUNES * 2:
            return content[: SNIPPET_CONTEXT_RUNES * 2].strip() + " ..."
        return content

    match_str = content[earliest:earliest_end]
    before = content[:earliest]
    after = content[earliest_end:]
    before_runes = before[-SNIPPET_CONTEXT_RUNES:]
    after_runes = after[:SNIPPET_CONTEXT_RUNES]
    snippet = before_runes + match_str + after_runes
    snippet = snippet.replace("\n", " ")
    while "  " in snippet:
        snippet = snippet.replace("  ", " ")
    return "... " + snippet.strip() + " ..."


def extract_snippet_regex(content: str, compiled: list[Pattern[str]]) -> str:
    """Return a short context snippet around the earliest regex match."""
    if not content or not compiled:
        return ""

    earliest = -1
    earliest_end = -1
    for pattern in compiled:
        if pattern is None:
            continue
        match = pattern.search(content)
        if match is None:
            continue
        if earliest < 0 or match.start() < earliest:
            earliest = match.start()
            earliest_end = match.end()

    if earliest < 0:
        return ""

    match_str = content[earliest:earliest_end]
    before = content[:earliest]
    after = content[earliest_end:]
    before_runes = before[-SNIPPET_CONTEXT_RUNES:]
    after_runes = after[:SNIPPET_CONTEXT_RUNES]
    match_runes = match_str[:SNIPPET_MAX_MATCH_RUNES]
    if len(match_str) > SNIPPET_MAX_MATCH_RUNES:
        match_runes = match_runes + "..."

    snippet = before_runes + match_runes + after_runes
    snippet = snippet.replace("\n", " ")
    while "  " in snippet:
        snippet = snippet.replace("  ", " ")
    snippet = snippet.strip()
    if len(snippet) > SNIPPET_MAX_TOTAL_RUNES:
        snippet = snippet[:SNIPPET_MAX_TOTAL_RUNES] + "..."
    return "... " + snippet + " ..."


def summarize_content(content: str) -> str:
    """Return the trimmed content, or ``(empty)`` when blank."""
    cleaned = content.strip()
    if not cleaned:
        return "(empty)"
    return cleaned


__all__ = [
    "SNIPPET_CONTEXT_RUNES",
    "SNIPPET_MAX_ANSWER_RUNES",
    "SNIPPET_MAX_MATCH_RUNES",
    "SNIPPET_MAX_TOTAL_RUNES",
    "build_content_signature",
    "build_image_info_markdown",
    "clamp_float",
    "count_regex_hits",
    "dedup_non_empty_strings",
    "extract_snippet_for_queries",
    "extract_snippet_regex",
    "jaccard",
    "parse_image_infos",
    "regex_matches_any",
    "search_query_tokens",
    "summarize_content",
    "text_matches_search_queries",
    "tokenize_simple",
    "truncate_runes",
    "xml_escape",
]
