"""Fuzzy slug resolution and short model handles for the wiki domain.

Two companion pieces of the wiki link machinery:

- :func:`resolve_dead_slug` rescues LLM-generated ``[[slug|display]]`` links
  whose slug is *almost* right but not exactly a live page: the display text
  is usually still correct, so a reverse lookup plus a tolerant string
  compare recover the real slug without an LLM round-trip.
- :class:`WikiSlugHandles` gives the ingest editor LLM short, low-entropy
  handles (``ref-1``, ``ref-2``, ...) for real slugs so it never has to
  reproduce a high-entropy slug verbatim; the handles are translated back
  on output.

Both helpers are pure and unit-testable in isolation.
"""

from __future__ import annotations

from src.core.knowledge.wiki.link_utils import rewrite_dead_wiki_links

# Minimum char-bigram Jaccard similarity for the fuzzy slug fallback.
# Conservative: close-but-distinct slugs must NOT resolve to the wrong
# page, which would be worse than leaving the link dead.
SLUG_RESOLVE_BIGRAM_THRESHOLD = 0.8


def normalize_slug_for_compare(slug: str) -> str:
    """Collapse cosmetic slug variations: lowercase, strip hyphens/underscores.

    Hyphens in wiki slugs are visual separators between word fragments;
    removing them treats "shang-hai-tower" and "shanghai-tower" as the
    same logical token bag. CJK runes are preserved verbatim.
    """
    return slug.lower().replace("-", "").replace("_", "")


def slug_char_bigrams(s: str) -> set[str]:
    """Return the character-bigram set of an already-normalized slug.

    Single-rune slugs degrade to a 1-gram so they still contribute a
    comparable signal.
    """
    if not s:
        return set()
    if len(s) == 1:
        return {s}
    return {s[i : i + 2] for i in range(len(s) - 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two sets (0.0 when either is empty)."""
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def resolve_dead_slug(
    dead_slug: str,
    display_text: str,
    live_slugs: set[str],
    title_to_slug: dict[str, str],
) -> tuple[str, bool]:
    """Map a dead ``[[slug]]`` reference back to a live KB slug.

    Progressive heuristics, in order:

    1. Display-text reverse lookup against page titles/aliases (the LLM
       copies the title correctly even when it mangles the slug).
    2. Hyphen/case-normalized slug equality.
    3. Bigram Jaccard similarity at or above :data:`SLUG_RESOLVE_BIGRAM_THRESHOLD`.

    Returns ``("", False)`` when no candidate is close enough to be safe.
    Both maps are consulted only, never mutated.
    """
    if not dead_slug:
        return "", False
    # Already live? No-op success.
    if dead_slug in live_slugs:
        return dead_slug, True

    # (1) Display-text reverse lookup. Trim stray spaces around the pipe
    # that models occasionally emit.
    display = display_text.strip()
    if display:
        slug = title_to_slug.get(display)
        if slug and slug in live_slugs:
            return slug, True

    # (2) Normalized-equality check.
    dead_norm = normalize_slug_for_compare(dead_slug)
    if not dead_norm:
        return "", False
    for candidate in live_slugs:
        if normalize_slug_for_compare(candidate) == dead_norm:
            return candidate, True

    # (3) Bigram Jaccard fallback over the normalized forms.
    dead_grams = slug_char_bigrams(dead_norm)
    if not dead_grams:
        return "", False
    best_slug = ""
    best_score = 0.0
    for candidate in live_slugs:
        candidate_norm = normalize_slug_for_compare(candidate)
        if not candidate_norm:
            continue
        candidate_grams = slug_char_bigrams(candidate_norm)
        if not candidate_grams:
            continue
        score = _jaccard(dead_grams, candidate_grams)
        if score > best_score:
            best_score = score
            best_slug = candidate
    if best_score >= SLUG_RESOLVE_BIGRAM_THRESHOLD:
        return best_slug, True
    return "", False


class HandleTable:
    """Stable slug -> short-handle allocation with reverse lookup."""

    def __init__(self, prefix: str, start: int, step: int) -> None:
        self._prefix = prefix
        self._next = start
        self._step = step
        self._slug_to_handle: dict[str, str] = {}
        self._handle_to_slug: dict[str, str] = {}

    def register(self, real_slug: str) -> str:
        """Return the stable handle for ``real_slug``, assigning one on first use."""
        if not real_slug:
            return ""
        existing = self._slug_to_handle.get(real_slug)
        if existing is not None:
            return existing
        handle = f"{self._prefix}{self._next}"
        self._next += self._step
        self._slug_to_handle[real_slug] = handle
        self._handle_to_slug[handle] = real_slug
        return handle

    def resolve(self, handle: str) -> tuple[str, bool]:
        """Reverse-lookup a handle to its real slug."""
        real_slug = self._handle_to_slug.get(handle)
        if real_slug is None:
            return "", False
        return real_slug, True

    def empty(self) -> bool:
        """Whether no handles have been assigned yet."""
        return not self._slug_to_handle


class WikiSlugHandles:
    """Rewrite wiki links to short model handles and back.

    The editor LLM never has to reproduce a high-entropy slug verbatim —
    most importantly the UUID-based summary slugs it routinely mangles by
    inserting or dropping hex digits. It copies the tiny handle instead;
    this class translates handles back to real slugs on output.
    """

    def __init__(self) -> None:
        self._table = HandleTable("ref-", 0, 1)

    @property
    def empty(self) -> bool:
        """Whether no handles have been assigned yet."""
        return self._table.empty()

    def handle(self, real_slug: str) -> str:
        """Return the stable handle for ``real_slug``, assigning one on first use."""
        return self._table.register(real_slug)

    def encode_content(self, content: str, known: set[str]) -> str:
        """Rewrite ``[[real_slug|disp]]`` / ``[[real_slug]]`` to handle form.

        Only slugs present in ``known`` are rewritten; unknown links are
        left untouched. Handles are assigned on demand so a slug appearing
        only in the body still gets a consistent handle.
        """
        if not content or not known:
            return content

        def _resolve(norm: str, _display: str) -> tuple[str, bool]:
            if norm not in known:
                return "", False
            return self._table.register(norm), True

        out, _ = rewrite_dead_wiki_links(content, _resolve)
        return out

    def decode_content(self, content: str) -> str:
        """Rewrite ``[[handle|disp]]`` / ``[[handle]]`` occurrences back to real slugs.

        Handles with no mapping are left untouched — they fall through to
        the ordinary parse / dead-link-cleanup path.
        """
        if not content or self._table.empty():
            return content

        def _resolve(norm: str, _display: str) -> tuple[str, bool]:
            return self._table.resolve(norm)

        out, _ = rewrite_dead_wiki_links(content, _resolve)
        return out


__all__ = [
    "SLUG_RESOLVE_BIGRAM_THRESHOLD",
    "HandleTable",
    "WikiSlugHandles",
    "normalize_slug_for_compare",
    "resolve_dead_slug",
    "slug_char_bigrams",
]
