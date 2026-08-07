"""Language-aware approximate token counting for the adaptive text chunker.

No tokenizer dependency (e.g. tiktoken) is required: per-language
chars-per-token ratios derived from common embedding model vocabularies are
used instead. The numbers are conservative — they tend to slightly
over-estimate token counts so that chunks stay safely under model limits.
"""

from __future__ import annotations

from typing import Final

# Language identifiers used by the token estimator and the heuristic splitter.
LangEnglish = "en"
LangGerman = "de"
LangChinese = "zh"
LangMixed = "mixed"

# Approximate chars/token ratios per language. Numbers err on the
# conservative side so estimates over-shoot a little.
_CHARS_PER_TOKEN: Final = {
    LangEnglish: 4.0,
    LangGerman: 4.5,
    LangChinese: 1.7,
    LangMixed: 3.0,
}

# German stop words used to bias language detection towards "de".
_GERMAN_STOPWORDS: Final = (
    " der ",
    " die ",
    " das ",
    " und ",
    " ist ",
    " nicht ",
    " mit ",
    " auf ",
)

_UMLAUTS = frozenset("äöüÄÖÜß")

# CJK code-point ranges used for cheap script counting.
_CJK_RANGES: Final = (
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0x3040, 0x309F),  # Hiragana
    (0x30A0, 0x30FF),  # Katakana
    (0xAC00, 0xD7AF),  # Hangul Syllables
    (0x1100, 0x11FF),  # Hangul Jamo
    (0x3130, 0x318F),  # Hangul Compatibility Jamo
)


def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def _is_german_umlaut(ch: str) -> bool:
    return ch in _UMLAUTS


def _contains_lower(haystack: str, needle: str) -> bool:
    """Case-insensitive substring search folding ASCII uppercase only."""
    if len(haystack) < len(needle):
        return False
    needle_len = len(needle)
    for i in range(len(haystack) - needle_len + 1):
        match = True
        for j in range(needle_len):
            h = haystack[i + j]
            if "A" <= h <= "Z":
                h = chr(ord(h) + 32)
            if h != needle[j]:
                match = False
                break
        if match:
            return True
    return False


def _has_german_words(s: str) -> bool:
    """Tiny stop-word check biasing towards "de" for common German function words."""
    sample = s[:512]
    return any(_contains_lower(sample, word) for word in _GERMAN_STOPWORDS)


def approx_token_count(s: str, lang: str) -> int:
    """Conservative token estimate for ``s`` in the given language.

    An empty or unknown language falls back to "mixed".
    """
    if s == "":
        return 0
    return approx_token_count_from_rune_len(len(s), lang)


def approx_token_count_from_rune_len(rune_len: int, lang: str) -> int:
    """Allocation-free variant of :func:`approx_token_count`.

    Use in hot loops where the same content's rune count would otherwise be
    recomputed multiple times.
    """
    if rune_len <= 0:
        return 0
    ratio = _CHARS_PER_TOKEN.get(lang, _CHARS_PER_TOKEN[LangMixed])
    approx = rune_len / ratio
    if approx < 1:
        return 1
    return int(approx + 0.5)


def detect_language(s: str) -> str:
    """Coarse language label by counting CJK runes vs Latin runes.

    Returns one of ``zh``, ``de``, ``en`` or ``mixed``. Detection is cheap and
    meant only for heuristic dispatch — it is NOT a replacement for proper
    language identification.
    """
    if s == "":
        return LangMixed
    cjk = 0
    latin = 0
    umlaut = 0
    for ch in s:
        if _is_cjk(ch):
            cjk += 1
        elif _is_german_umlaut(ch):
            umlaut += 1
            latin += 1
        elif "a" <= ch <= "z" or "A" <= ch <= "Z":
            latin += 1
    total = cjk + latin
    if total == 0:
        return LangMixed
    cjk_ratio = cjk / total
    latin_ratio = latin / total
    # Mixed: meaningful presence of both scripts (>=15% each).
    if cjk_ratio >= 0.15 and latin_ratio >= 0.15:
        return LangMixed
    if cjk_ratio > 0.3:
        return LangChinese
    if umlaut > 0 or _has_german_words(s):
        return LangGerman
    return LangEnglish


def chars_for_token_limit(tokens: int, lang: str) -> int:
    """Convert a token limit into an approximate character budget.

    Used to size chunks so they fit within an embedding model's max-token
    window with a small safety margin.
    """
    if tokens <= 0:
        return 0
    ratio = _CHARS_PER_TOKEN.get(lang, _CHARS_PER_TOKEN[LangMixed])
    # 0.9 safety factor so we under-shoot the model limit instead of overshooting.
    return int(tokens * ratio * 0.9)


__all__ = [
    "LangChinese",
    "LangEnglish",
    "LangGerman",
    "LangMixed",
    "approx_token_count",
    "approx_token_count_from_rune_len",
    "chars_for_token_limit",
    "detect_language",
]
