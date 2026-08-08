"""Dedup pre-filter for the wiki ingest pipeline.

Before the (LLM-backed) merge decision, the candidate existing pages a new
item could merge into are narrowed to pages that share at least some cheap
surface-level signal with it — slug tokens and character bigrams of the
surface forms. This keeps the merge prompt small on large knowledge bases
and — more importantly — stops weak models from hallucinating merges
between unrelated slugs (the pre-filter only ever *removes* candidates;
the downstream merge validation still guards the final write).

Everything in this module is pure and deterministic: no IO, no model
calls. The merge decision itself is an injectable ``DedupMerger`` seam
(LLM-backed in the worker layer); until it is wired the default merger
keeps every item, matching the "no dedup" safe default.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.core.knowledge.wiki.ingest_types import (
    WIKI_PAGE_TYPE_CONCEPT,
    WIKI_PAGE_TYPE_ENTITY,
    DedupMerger,
    WikiExtractedItem,
)
from src.db.models.wiki_page import WikiPage

#: Bounds how many trigram-similar existing pages each new item's similarity
#: probe contributes. A tight K keeps each item's candidate list short,
#: which improves the model's precision and cuts tokens.
DEDUP_CANDIDATE_TOP_K: int = 5

#: Jaccard floor: pairs at or above this similarity are always included.
DEDUP_CANDIDATE_SCORE_FLOOR: float = 0.08

#: Pre-filter is bypassed when the existing-page corpus is small enough to
#: fit the prompt without degrading the model.
DEDUP_SMALL_CORPUS_BYPASS: int = 25


@dataclass(frozen=True, slots=True)
class DedupSurface:
    """Pre-computed similarity features for one side of a comparison."""

    slug_tokens: frozenset[str]
    name_gram_sets: tuple[frozenset[str], ...]


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Jaccard similarity of two sets (0 when both are empty)."""
    if not a and not b:
        return 0
    if len(a) > len(b):
        return jaccard(b, a)
    inter = sum(1 for token in a if token in b)
    if inter == 0:
        return 0
    return inter / (len(a) + len(b) - inter)


def slug_base_tokens(slug: str) -> frozenset[str]:
    """Kebab-case tokens of a slug's base component.

    ``entity/beijing-nongshang-yinxing`` -> ``{"beijing", "nongshang",
    "yinxing"}``. Slugs are an orthogonal signal to the surface names —
    CJK pages carry their pinyin here.
    """
    if not slug:
        return frozenset()
    base = slug.split("/", 1)[1] if "/" in slug else slug
    base = base.lower()
    return frozenset(
        token
        for token in base.replace("-", " ").replace("_", " ").replace(".", " ").split()
        if token != ""
    )


def surface_grams(s: str) -> frozenset[str]:
    """Character-bigram set of a surface form after normalising.

    Lowercases and strips non-letter/digit runes. Bigrams work well across
    both CJK (where each bigram approximates a word) and Latin (where they
    catch stem overlap). Single-rune strings fall back to a 1-gram.
    """
    if not s:
        return frozenset()
    compact = "".join(ch for ch in s.lower() if ch.isalnum())
    if not compact:
        return frozenset()
    runes = list(compact)
    if len(runes) == 1:
        return frozenset({runes[0]})
    return frozenset("".join(runes[i : i + 2]) for i in range(len(runes) - 1))


def grams_per_surface(surfaces: list[str]) -> tuple[frozenset[str], ...]:
    """One gram set per non-empty surface form."""
    return tuple(gram for s in surfaces if (gram := surface_grams(s)))


def dedup_pair_score(a: DedupSurface, b: DedupSurface) -> float:
    """Max similarity between any surface form of ``a`` and ``b``.

    Slug and name signals live in different symbol spaces (ASCII pinyin vs
    raw surface form) so the max is taken rather than an average.
    """
    best = jaccard(a.slug_tokens, b.slug_tokens)
    for ag in a.name_gram_sets:
        for bg in b.name_gram_sets:
            if (score := jaccard(ag, bg)) > best:
                best = score
    return best


def _dedup_surface(title: str, aliases: list[str], slug: str) -> DedupSurface:
    surfaces = [title, *aliases]
    return DedupSurface(
        slug_tokens=slug_base_tokens(slug),
        name_gram_sets=grams_per_surface(surfaces),
    )


def select_dedup_candidate_pages(
    new_items: list[WikiExtractedItem],
    all_pages: list[WikiPage],
) -> list[WikiPage]:
    """Return the existing pages plausibly related to at least one new item.

    Non-entity/concept pages are dropped unconditionally, and on small
    corpora (at most ``DEDUP_SMALL_CORPUS_BYPASS`` entries) the pre-filter
    is a no-op aside from that type filter. The returned list preserves
    input order so the downstream prompt stays stable across runs.
    """
    pages = [
        page
        for page in all_pages
        if page.page_type == WIKI_PAGE_TYPE_ENTITY or page.page_type == WIKI_PAGE_TYPE_CONCEPT
    ]
    if not pages or not new_items or len(pages) <= DEDUP_SMALL_CORPUS_BYPASS:
        return pages

    page_features = [
        _dedup_surface(page.title, list(page.aliases), page.slug) for page in pages
    ]
    selected: set[int] = set()
    for item in new_items:
        item_features = _dedup_surface(item.name, list(item.aliases), item.slug)
        if not item_features.slug_tokens and not item_features.name_gram_sets:
            continue
        ranked = sorted(
            range(len(page_features)),
            key=lambda idx: dedup_pair_score(item_features, page_features[idx]),
            reverse=True,
        )
        top_k_remaining = DEDUP_CANDIDATE_TOP_K
        for idx in ranked:
            score = dedup_pair_score(item_features, page_features[idx])
            if score >= DEDUP_CANDIDATE_SCORE_FLOOR:
                selected.add(idx)
                continue
            # Below the floor but the prompt still needs some candidates to
            # decline cleanly: fill the top-K budget with the best remaining
            # non-zero scores (a flat zero means nothing in common and just
            # invites hallucination).
            if top_k_remaining > 0 and score > 0:
                selected.add(idx)
                top_k_remaining -= 1
                continue
            break

    return [page for i, page in enumerate(pages) if i in selected]


def dedup_merge_reject_reason(
    src_slug: str,
    dst_slug: str,
    src_candidates: set[str],
) -> str:
    """Validate one proposed merge against deterministic, model-independent rules.

    Returns an empty string when the merge is allowed, or a short reason
    when it must be rejected. ``src_candidates`` is the set of existing
    page slugs that surfaced for ``src_slug``'s own similarity probe; a
    merge whose target was only similar to a *different* item is rejected.
    """
    if dst_slug not in src_candidates:
        return "target is not a similarity candidate for this item"
    src_slash = src_slug.find("/")
    dst_slash = dst_slug.find("/")
    if src_slash <= 0 or dst_slash <= 0:
        return "missing type prefix"
    if src_slug[: src_slash + 1] != dst_slug[: dst_slash + 1]:
        return (
            "type mismatch: "
            + src_slug[: src_slash + 1]
            + " vs "
            + dst_slug[: dst_slash + 1]
        )
    return ""


def candidate_slugs_for_item(
    item: WikiExtractedItem,
    candidate_pages: list[WikiPage],
) -> set[str]:
    """Return the candidate-page slugs whose surface features match ``item``.

    Re-runs the pair scoring so callers have the per-item scoping the merge
    validation needs without keeping the full feature set around.
    """
    item_features = _dedup_surface(item.name, list(item.aliases), item.slug)
    return {
        page.slug
        for page in candidate_pages
        if dedup_pair_score(
            item_features,
            _dedup_surface(page.title, list(page.aliases), page.slug),
        )
        >= DEDUP_CANDIDATE_SCORE_FLOOR
    }


class NoopDedupMerger:
    """Default merger: keep every item (safe no-dedup default)."""

    async def decide(
        self,
        *,
        item: WikiExtractedItem,
        candidate_slugs: list[str],
    ) -> str:
        return ""


async def deduplicate_items(
    items: list[WikiExtractedItem],
    candidate_pages: list[WikiPage],
    merger: DedupMerger | None,
) -> tuple[list[WikiExtractedItem], dict[str, str]]:
    """Run the dedup merge pass over ``items``.

    Returns ``(kept_items, merged_refs)`` where ``merged_refs`` maps an
    item slug to the existing page slug it merged into. Items that merge
    are dropped from the kept set (their contribution now lives on the
    target page). Without a merger seam every item is kept unchanged.
    """
    if merger is None or not items or not candidate_pages:
        return items, {}
    kept: list[WikiExtractedItem] = []
    merged_refs: dict[str, str] = {}
    for item in items:
        candidates = sorted(candidate_slugs_for_item(item, candidate_pages))
        target = await merger.decide(item=item, candidate_slugs=candidates)
        if target == "" or dedup_merge_reject_reason(item.slug, target, set(candidates)):
            kept.append(item)
        else:
            merged_refs[item.slug] = target
    return kept, merged_refs


__all__ = [
    "DEDUP_CANDIDATE_SCORE_FLOOR",
    "DEDUP_CANDIDATE_TOP_K",
    "DEDUP_SMALL_CORPUS_BYPASS",
    "DedupMerger",
    "DedupSurface",
    "NoopDedupMerger",
    "candidate_slugs_for_item",
    "dedup_merge_reject_reason",
    "dedup_pair_score",
    "deduplicate_items",
    "grams_per_surface",
    "jaccard",
    "select_dedup_candidate_pages",
    "slug_base_tokens",
    "surface_grams",
]
