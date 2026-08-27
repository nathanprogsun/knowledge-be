"""Wiki-link parsing and cross-link injection helpers (pure, I/O-free).

Ports the link semantics of the wiki domain: extracting outbound
``[[slug]]`` references from markdown bodies, stripping the internal
chunk-citation handles the ingest classification pass emits, rewriting
dead wiki links, and the code-aware cross-link injection pass that wraps
title mentions in ``[[slug|title]]`` without touching fenced code blocks,
inline code, existing links, or autolinks.

Every function here is pure — no database, no I/O — so the link policy is
unit-testable in isolation from the service wiring.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from src.core.knowledge.wiki.types import WIKI_PAGE_TYPE_INDEX
from src.db.models.wiki_page import WikiPage

# Matches ``[[wiki-link]]`` / ``[[slug|display]]`` syntax in markdown.
WIKI_LINK_RE = re.compile(r"\[\[([^\]]+)\]\]")

# Matches the short internal chunk-citation handles (``[c123]`` /
# ``[c001, c002]``) emitted by the ingest classification pass. The stable
# source relationship lives in the page's chunk references, so these
# handles have no meaning to readers and must not leak into markdown.
WIKI_INLINE_CHUNK_CITATION_RE = re.compile(r"[ \t]*\[c\d{3,}(?:\s*[,;]\s*c\d{3,})*\]")


def strip_wiki_inline_chunk_citations(content: str) -> str:
    """Remove internal chunk-citation handles from ``content``."""
    return WIKI_INLINE_CHUNK_CITATION_RE.sub("", content)


def strip_page_chunk_citations(page: WikiPage) -> WikiPage:
    """Return a copy of ``page`` with chunk-citation handles stripped."""
    return page.model_copy(
        update={
            "content": strip_wiki_inline_chunk_citations(page.content),
            "summary": strip_wiki_inline_chunk_citations(page.summary),
        }
    )


def normalize_slug(slug: str) -> str:
    """Normalize a wiki link slug: lowercased, trimmed, spaces to hyphens."""
    return slug.lower().strip().replace(" ", "-")


def parse_out_links(content: str) -> list[str]:
    """Extract deduplicated outbound wiki-link slugs from markdown content.

    Handles both ``[[slug]]`` and ``[[slug|display name]]`` forms (the
    display name is dropped). Slugs are normalized and deduplicated,
    preserving first-occurrence order.
    """
    seen: set[str] = set()
    links: list[str] = []
    for match in WIKI_LINK_RE.findall(content):
        raw = match.strip()
        if "|" in raw:
            raw = raw.split("|", 1)[0].strip()
        slug = normalize_slug(raw)
        if slug and slug not in seen:
            seen.add(slug)
            links.append(slug)
    return links


def slug_namespace(slug: str) -> str:
    """Return the prefix of a slug up to (but excluding) the first '/'.

    ``"summary/abc"`` -> ``"summary"``; slugs without a '/' map to ``""``.
    """
    if "/" in slug:
        return slug.split("/", 1)[0]
    return ""


@dataclass(frozen=True)
class LinkRef:
    """A single (slug, match_text) candidate for cross-link injection."""

    slug: str
    match_text: str


def collect_link_refs(pages: list[WikiPage]) -> list[LinkRef]:
    """Flatten (title + aliases) of non-system pages into link refs."""
    refs: list[LinkRef] = []
    for page in pages:
        if page.page_type == WIKI_PAGE_TYPE_INDEX:
            continue
        if page.title:
            refs.append(LinkRef(slug=page.slug, match_text=page.title))
        for alias in page.aliases:
            if alias:
                refs.append(LinkRef(slug=page.slug, match_text=alias))
    return refs


Resolve = Callable[[str, str], tuple[str, bool]]


def rewrite_dead_wiki_links(content: str, resolve: Resolve) -> tuple[str, bool]:
    """Rewrite every ``[[slug]]`` / ``[[slug|display]]`` occurrence via ``resolve``.

    ``resolve(normalized_slug, display)`` returns the replacement slug and
    whether to rewrite; a non-rewrite leaves the link untouched. Display
    text is preserved verbatim.
    """
    changed = False

    def _replace(match: re.Match[str]) -> str:
        nonlocal changed
        inner = match.group(1)
        raw_slug = inner
        display = ""
        if "|" in inner:
            raw_slug, display = inner.split("|", 1)
            display = display.strip()
        norm = normalize_slug(raw_slug)
        if not norm:
            return match.group(0)
        new_slug, ok = resolve(norm, display)
        if not ok or not new_slug or new_slug == norm:
            return match.group(0)
        changed = True
        if display:
            return f"[[{new_slug}|{display}]]"
        return f"[[{new_slug}]]"

    out = WIKI_LINK_RE.sub(_replace, content)
    return out, changed


# ── Cross-link injection ─────────────────────────────────────────────


@dataclass(frozen=True)
class Span:
    """A half-open ``[start, end)`` range inside content that must not be touched."""

    start: int
    end: int


def linkify_content(content: str, refs: list[LinkRef], self_slug: str) -> tuple[str, bool]:
    """Inject ``[[slug|match_text]]`` cross-links into ``content``.

    For every ref, at most the FIRST eligible occurrence is wrapped, and
    occurrences inside code or existing links are skipped. Refs already
    linked to their slug, and refs pointing at ``self_slug``, are skipped.
    Longer match texts win over their substrings.

    Returns the possibly-updated content and whether any change was made.
    The input ``refs`` is not mutated.
    """
    if not content or not refs:
        return content, False

    sorted_refs = sorted(
        (r for r in refs if r.slug and r.match_text and r.slug != self_slug),
        key=lambda r: len(r.match_text),
        reverse=True,
    )
    if not sorted_refs:
        return content, False

    forbidden, used = compute_forbidden_spans(content)
    changed = False

    for ref in sorted_refs:
        if ref.slug in used:
            continue
        pos = find_first_safe_match(content, ref.match_text, forbidden)
        if pos < 0:
            continue
        replacement = f"[[{ref.slug}|{ref.match_text}]]"
        content = content[:pos] + replacement + content[pos + len(ref.match_text) :]
        # Shift / extend the forbidden spans so later refs cannot nest a
        # link inside the newly created ``[[...]]``.
        delta = len(replacement) - len(ref.match_text)
        shifted = shift_spans_after(forbidden, pos, delta)
        forbidden = sort_spans([*shifted, Span(pos, pos + len(replacement))])
        used.add(ref.slug)
        changed = True

    return content, changed


def find_first_safe_match(haystack: str, needle: str, forbidden: list[Span]) -> int:
    """Return the offset of the first occurrence of ``needle`` outside forbidden spans.

    ASCII-letter-led needles also require non-word-character boundaries.
    Returns ``-1`` when no such occurrence exists.
    """
    if not needle:
        return -1
    needs_boundary = has_ascii_letter_edge(needle)

    start = 0
    while start <= len(haystack) - len(needle):
        pos = haystack.find(needle, start)
        if pos < 0:
            return -1
        end = pos + len(needle)
        if span_contains(forbidden, pos, end):
            start = pos + 1
            continue
        if needs_boundary and not has_word_boundary(haystack, pos, end):
            start = pos + 1
            continue
        return pos
    return -1


def has_ascii_letter_edge(s: str) -> bool:
    """Whether ``s`` starts or ends with an ASCII letter / digit / underscore."""
    if not s:
        return False
    return is_ascii_word_rune(s[0]) or is_ascii_word_rune(s[-1])


def is_ascii_word_rune(ch: str) -> bool:
    """Whether ``ch`` is an ASCII word rune (letter / digit / underscore)."""
    return ch == "_" or "0" <= ch <= "9" or "a" <= ch <= "z" or "A" <= ch <= "Z"


def has_word_boundary(s: str, pos: int, end: int) -> bool:
    """Whether the chars adjacent to ``[pos, end)`` are not ASCII word runes.

    Non-ASCII runes (e.g. CJK) count as boundaries so a CJK mention inside
    a longer CJK string still matches.
    """
    if pos > 0 and is_ascii_word_rune(s[pos - 1]):
        return False
    return not (end < len(s) and is_ascii_word_rune(s[end]))


def span_contains(spans: list[Span], pos: int, end: int) -> bool:
    """Whether any span overlaps ``[pos, end)``."""
    return any(pos < sp.end and end > sp.start for sp in spans)


def shift_spans_after(spans: list[Span], pivot: int, delta: int) -> list[Span]:
    """Shift every span at or after ``pivot`` by ``delta``."""
    if delta == 0:
        return spans
    return [Span(sp.start + delta, sp.end + delta) if sp.start >= pivot else sp for sp in spans]


def sort_spans(spans: list[Span]) -> list[Span]:
    """Return ``spans`` sorted by start then end."""
    return sorted(spans, key=lambda sp: (sp.start, sp.end))


def compute_forbidden_spans(s: str) -> tuple[list[Span], set[str]]:
    """Return ranges in ``s`` that cross-link injection must leave untouched.

    Covered regions: fenced code blocks (`` ``` `` / ``~~~``), inline code
    runs, existing ``[[slug|...]]`` wiki links, inline markdown links and
    images, reference-style links, reference link definitions, and
    autolinks. Also returns the set of wiki-link slugs already referenced
    so callers can skip already-linked refs without a second scan.
    """
    spans: list[Span] = list(_scan_reference_definitions(s))
    used: set[str] = set()
    i = 0
    n = len(s)

    while i < n:
        if is_fence_start(s, i):
            fence_len, fence_ch = fence_run(s, i)
            end = find_fence_end(s, i + fence_len, fence_ch, fence_len)
            spans.append(Span(i, end))
            i = end
            continue

        ch = s[i]
        if ch == "`":
            run = 1
            while i + run < n and s[i + run] == "`":
                run += 1
            close = find_inline_code_close(s, i + run, run)
            if close < 0:
                i += run
                continue
            spans.append(Span(i, close + run))
            i = close + run
        elif ch == "[":
            if i + 1 < n and s[i + 1] == "[":
                close_rel = s.find("]]", i + 2)
                if close_rel >= 0:
                    inner = s[i + 2 : close_rel]
                    slug = extract_wiki_slug(inner)
                    if slug:
                        used.add(slug)
                    spans.append(Span(i, close_rel + 2))
                    i = close_rel + 2
                    continue
            end, ok = match_markdown_link(s, i)
            if ok:
                spans.append(Span(i, end))
                i = end
                continue
            end, ok = match_reference_style_link(s, i)
            if ok:
                spans.append(Span(i, end))
                i = end
                continue
            i += 1
        elif ch == "!":
            if i + 1 < n and s[i + 1] == "[":
                end, ok = match_markdown_link(s, i + 1)
                if ok:
                    spans.append(Span(i, end))
                    i = end
                    continue
                end, ok = match_reference_style_link(s, i + 1)
                if ok:
                    spans.append(Span(i, end))
                    i = end
                    continue
            i += 1
        elif ch == "<":
            end, ok = match_autolink(s, i)
            if ok:
                spans.append(Span(i, end))
                i = end
                continue
            i += 1
        else:
            i += 1

    return sort_spans(spans), used


def extract_wiki_slug(inner: str) -> str:
    """Return the slug part of ``[[...]]`` inner text (``[[slug|display]]`` -> ``slug``)."""
    if "|" in inner:
        inner = inner.split("|", 1)[0]
    return inner.strip()


def match_reference_style_link(s: str, i: int) -> tuple[int, bool]:
    """Match ``[text][label]`` starting at ``[``; return the offset past the closing ``]``."""
    if i >= len(s) or s[i] != "[":
        return 0, False
    text_end, ok = find_closing_bracket(s, i)
    if not ok:
        return 0, False
    if text_end + 1 >= len(s) or s[text_end + 1] != "[":
        return 0, False
    label_end, ok = find_closing_bracket(s, text_end + 1)
    if not ok:
        return 0, False
    return label_end + 1, True


def find_closing_bracket(s: str, i: int) -> tuple[int, bool]:
    """Return the offset of the matching ``]`` for the ``[`` at ``i``.

    Honors ``\\[`` / ``\\]`` escapes and gives up on newlines.
    """
    if i >= len(s) or s[i] != "[":
        return 0, False
    depth = 1
    j = i + 1
    while j < len(s):
        ch = s[j]
        if ch == "\\":
            j += 2
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return j, True
        elif ch == "\n":
            return 0, False
        j += 1
    return 0, False


def _scan_reference_definitions(s: str) -> list[Span]:
    """Find ``[label]: url ...`` definition lines and return their ranges.

    Only lines whose first non-space character is ``[`` (up to 3 spaces of
    CommonMark indent) are considered.
    """
    out: list[Span] = []
    line_start = 0
    n = len(s)
    while line_start < n:
        nl = s.find("\n", line_start)
        line_end = n if nl < 0 else nl + 1

        indent = 0
        while indent < 3 and line_start + indent < line_end and s[line_start + indent] == " ":
            indent += 1
        start = line_start + indent

        if start < line_end and s[start] == "[":
            label_end, ok = find_closing_bracket(s, start)
            if ok and label_end + 1 < line_end and s[label_end + 1] == ":":
                out.append(Span(line_start, line_end))

        line_start = line_end
    return out


def is_fence_start(s: str, i: int) -> bool:
    """Whether index ``i`` is at the start of a line and begins a fence run."""
    if i > 0 and s[i - 1] != "\n":
        return False
    if i + 2 >= len(s):
        return False
    ch = s[i]
    return (ch == "`" or ch == "~") and s[i + 1] == ch and s[i + 2] == ch


def fence_run(s: str, i: int) -> tuple[int, str]:
    """Return the length and char of the fence run starting at ``i``."""
    ch = s[i]
    j = i
    while j < len(s) and s[j] == ch:
        j += 1
    return j - i, ch


def find_fence_end(s: str, start: int, ch: str, min_len: int) -> int:
    """Return the offset just past the closing fence, or ``len(s)`` if none."""
    nl = s.find("\n", start)
    if nl < 0:
        return len(s)
    pos = nl + 1
    while pos < len(s):
        if s[pos] == ch:
            run_len, _ = fence_run(s, pos)
            if run_len >= min_len:
                end_line = s.find("\n", pos)
                if end_line < 0:
                    return len(s)
                return end_line + 1
        nl = s.find("\n", pos)
        if nl < 0:
            return len(s)
        pos = nl + 1
    return len(s)


def find_inline_code_close(s: str, start: int, run_len: int) -> int:
    """Return the offset of a closing backtick run of exactly ``run_len``, or ``-1``."""
    i = start
    n = len(s)
    while i < n:
        if i + 1 < n and s[i] == "\n" and s[i + 1] == "\n":
            return -1
        if s[i] == "`":
            j = i
            while j < n and s[j] == "`":
                j += 1
            if j - i == run_len:
                return i
            i = j
            continue
        i += 1
    return -1


def match_markdown_link(s: str, i: int) -> tuple[int, bool]:
    """Match ``[text](url)`` starting at ``[``; return the offset past the closing ``)``."""
    if i >= len(s) or s[i] != "[":
        return 0, False
    depth = 1
    j = i + 1
    while j < len(s) and depth > 0:
        ch = s[j]
        if ch == "\\":
            j += 2
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        elif ch == "\n":
            return 0, False
        if depth == 0:
            break
        j += 1
    if j >= len(s) or s[j] != "]":
        return 0, False
    if j + 1 >= len(s) or s[j + 1] != "(":
        return 0, False
    k = j + 2
    paren = 1
    while k < len(s) and paren > 0:
        ch = s[k]
        if ch == "\\":
            k += 2
            continue
        if ch == "(":
            paren += 1
        elif ch == ")":
            paren -= 1
            if paren == 0:
                return k + 1, True
        elif ch == "\n":
            return 0, False
        k += 1
    return 0, False


def match_autolink(s: str, i: int) -> tuple[int, bool]:
    """Match ``<scheme://...>`` or ``<mailto:...>`` starting at ``[``."""
    if i >= len(s) or s[i] != "<":
        return 0, False
    close = s.find(">", i + 1)
    if close < 0:
        return 0, False
    inner = s[i + 1 : close]
    if not inner or any(c in inner for c in " \t\n"):
        return 0, False
    if "://" not in inner and not inner.startswith("mailto:"):
        return 0, False
    return close + 1, True


__all__ = [
    "WIKI_INLINE_CHUNK_CITATION_RE",
    "WIKI_LINK_RE",
    "LinkRef",
    "Resolve",
    "Span",
    "collect_link_refs",
    "compute_forbidden_spans",
    "extract_wiki_slug",
    "find_first_safe_match",
    "linkify_content",
    "match_autolink",
    "match_markdown_link",
    "match_reference_style_link",
    "normalize_slug",
    "parse_out_links",
    "rewrite_dead_wiki_links",
    "slug_namespace",
    "strip_page_chunk_citations",
    "strip_wiki_inline_chunk_citations",
]
