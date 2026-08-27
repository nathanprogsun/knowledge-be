"""Query-expansion helpers for the chat pipeline search step.

When initial recall is low, the search step generates local query variants
and re-runs retrieval with a widened keyword gate. Variant generation is
deterministic and model-free: stopword removal, quoted-phrase extraction,
delimiter splitting, and question-word stripping over a small English +
Chinese vocabulary.

The helpers here are pure text transforms over a ``PipelineContext``; the
retrieval fan-out that consumes the variants lives in the search step.
"""

from __future__ import annotations

import re

import jieba

from src.core.chat.pipeline.common import pipeline_info
from src.core.chat.pipeline.context import PipelineContext

#: Common Chinese and English stopwords (upstream ``stopwords`` table).
_STOPWORDS: frozenset[str] = frozenset(
    {
        "的",
        "是",
        "在",
        "了",
        "和",
        "与",
        "或",
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "can",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "about",
        "what",
        "how",
        "why",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
    }
)

#: Leading question words stripped from a query (upstream ``questionWords``).
_QUESTION_WORDS_RE = re.compile(
    r"^(什么是|什么|如何|怎么|怎样|为什么|为何|哪个|哪些|谁|何时|何地|请问|请告诉我|帮我|我想知道|我想了解)"
)

#: Common punctuation / whitespace that split a query into candidate segments.
_DELIMITER_RE = re.compile(r"[,，;；、。！？!?\s]+")

#: Quote characters that bracket an extractable phrase.
_QUOTE_CHARS = "\"'「」『』"
_QUOTED_PHRASE_RE = re.compile(f"[{_QUOTE_CHARS}]([^{_QUOTE_CHARS}]+)[{_QUOTE_CHARS}]")

#: CJK unified-ideograph ranges treated as Han text by the tokenizer.
_HAN_RANGES: tuple[tuple[int, int], ...] = (
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0x20000, 0x2A6DF),
)

#: Upper bound on generated query variants per turn.
_MAX_EXPANSIONS = 5


def _byte_len(value: str) -> int:
    """Return the UTF-8 byte length of ``value`` (upstream ``len``)."""
    return len(value.encode("utf-8"))


def _is_han_char(char: str) -> bool:
    """Report whether ``char`` is a CJK unified ideograph."""
    code = ord(char)
    return any(low <= code <= high for low, high in _HAN_RANGES)


def tokenize(text: str) -> list[str]:
    """Tokenize ``text``, running search-mode segmentation over Han runs.

    Non-Han letter/digit runs pass through whole; continuous Han runs are
    segmented with search-mode word splitting (upstream ``tokenize``).
    """
    tokens: list[str] = []
    current: list[str] = []
    current_is_han = False

    def flush() -> None:
        nonlocal current, current_is_han
        if not current:
            return
        segment = "".join(current)
        if current_is_han:
            for word in jieba.cut_for_search(segment, HMM=True):
                word = word.strip()
                if word:
                    tokens.append(word)
        else:
            tokens.append(segment)
        current = []
        current_is_han = False

    for char in text:
        if _is_han_char(char):
            if current and not current_is_han:
                flush()
            current_is_han = True
            current.append(char)
        elif char.isalpha() or char.isdigit():
            if current and current_is_han:
                flush()
            current_is_han = False
            current.append(char)
        else:
            flush()
    flush()
    return tokens


def extract_keywords(text: str) -> list[str]:
    """Return tokens that are not stopwords and span more than one rune."""
    keywords: list[str] = []
    for word in tokenize(text):
        if word.lower() not in _STOPWORDS and len(word) > 1:
            keywords.append(word)
    return keywords


def extract_phrases(text: str) -> list[str]:
    """Return the contents of quote-bracketed phrases in ``text``."""
    phrases: list[str] = []
    for match in _QUOTED_PHRASE_RE.finditer(text):
        body = match.group(1)
        if _byte_len(body) > 2:
            phrases.append(body)
    return phrases


def split_by_delimiters(text: str) -> list[str]:
    """Split ``text`` on common punctuation / whitespace delimiters."""
    parts: list[str] = []
    for part in _DELIMITER_RE.split(text):
        part = part.strip()
        if part:
            parts.append(part)
    return parts


def remove_question_words(text: str) -> str:
    """Strip a leading question word from ``text``."""
    return _QUESTION_WORDS_RE.sub("", text).strip()


def expand_queries(pipeline_ctx: PipelineContext) -> list[str]:
    """Generate local keyword-focused query variants (upstream ``expandQueries``).

    The rewritten query seeds the variant set; the original user query is
    excluded from variants. Produces at most ``_MAX_EXPANSIONS`` variants.
    """
    query = pipeline_ctx.rewrite_query.strip()
    if not query:
        return []

    expansions: list[str] = []
    seen: set[str] = {query.lower()}
    original = pipeline_ctx.query
    if original:
        seen.add(original.lower())

    def add_if_new(value: str) -> None:
        candidate = value.strip()
        if not candidate or _byte_len(candidate) < 3:
            return
        key = candidate.lower()
        if key in seen:
            return
        seen.add(key)
        expansions.append(candidate)

    keywords = extract_keywords(query)
    if len(keywords) >= 2:
        add_if_new(" ".join(keywords))

    for phrase in extract_phrases(query):
        add_if_new(phrase)

    for segment in split_by_delimiters(query):
        if _byte_len(segment) > 5:
            add_if_new(segment)

    cleaned = remove_question_words(query)
    if cleaned != query:
        add_if_new(cleaned)

    if len(expansions) > _MAX_EXPANSIONS:
        expansions = expansions[:_MAX_EXPANSIONS]
    pipeline_info("Search", "local_expansion_result", {"variants": len(expansions)})
    return expansions


__all__ = [
    "expand_queries",
    "extract_keywords",
    "extract_phrases",
    "remove_question_words",
    "split_by_delimiters",
    "tokenize",
]
