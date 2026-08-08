"""Batch taxonomy planning for the wiki ingest pipeline.

Before the reduce phase writes pages in parallel it assigns every new
entity / concept slug a coherent directory path in ONE planning pass
(chunked for large batches), so the whole batch lands on a single tree
that reuses existing folders. This replaces per-page folder invention,
which could not converge — worst of all on the founding batch when the
knowledge base has no folders to anchor on. Reduce applies these paths
only to pages that do not already have a category.

The planner itself is an injectable seam (LLM-backed in the worker layer);
the folder reuse / similarity preprocessing and the path-to-folder
resolution are deterministic.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from src.ai.embedding import Context, Embedder
from src.core.knowledge.wiki.ingest_types import (
    WIKI_PAGE_TYPE_CONCEPT,
    WIKI_PAGE_TYPE_ENTITY,
    TaxonomyPlanner,
    WikiFolderStore,
    WikiSlugUpdate,
)
from src.core.knowledge.wiki.types import clean_category_path

#: Caps how many existing folders are rendered into a planning prompt as the
#: set to reuse. Reached only for pathologically large taxonomies.
WIKI_TAXONOMY_PROMPT_MAX_PATHS: int = 150

#: Bounds the existing folders pulled from storage as the candidate pool for
#: similarity selection.
WIKI_TAXONOMY_FOLDER_POOL_MAX: int = 400

#: Folder count at or below which the whole set is fed to the planner
#: as-is (best reuse recall with no embedding cost).
WIKI_TAXONOMY_FEED_ALL_MAX_FOLDERS: int = 60

#: How many nearest existing deeper folders each item contributes to the
#: reuse set when similarity preprocessing kicks in.
WIKI_TAXONOMY_RELEVANT_TOP_K: int = 3

#: Caps how many items go into a single planning call; larger batches are
#: split, feeding earlier assignments forward so later chunks converge.
WIKI_TAXONOMY_PLAN_CHUNK_SIZE: int = 60

WIKI_TAXONOMY_EMPTY_TREE_HINT: str = (
    "(none yet — this knowledge base has no folders, design a fresh directory)"
)


@dataclass(frozen=True, slots=True)
class WikiTaxonomyItem:
    """One entity / concept page to be filed into the directory."""

    slug: str
    title: str
    page_type: str
    about: str


@dataclass(slots=True)
class _TaxonomyNode:
    """One node of the existing-folder tree rendered into a prompt."""

    children: dict[str, _TaxonomyNode] = field(default_factory=dict)


def collect_taxonomy_items(
    slug_updates: Mapping[str, Sequence[WikiSlugUpdate]],
) -> list[WikiTaxonomyItem]:
    """Extract the entity / concept pages from a batch's slug updates.

    Summary and retract-only slugs are skipped (they carry no directory
    category). Slugs are visited in sorted order so chunk boundaries are
    stable across runs; one entry per slug is enough for classification.
    """
    items: list[WikiTaxonomyItem] = []
    for slug in sorted(slug_updates):
        for update in slug_updates[slug]:
            if update.type != WIKI_PAGE_TYPE_ENTITY and update.type != WIKI_PAGE_TYPE_CONCEPT:
                continue
            title = update.item.name.strip() if update.item is not None else ""
            if title == "":
                title = slug
            about = update.item.description.strip() if update.item is not None else ""
            items.append(
                WikiTaxonomyItem(slug=slug, title=title, page_type=update.type, about=about)
            )
            break
    return items


def insert_taxonomy_path(root: _TaxonomyNode, path: Sequence[str]) -> None:
    """Insert one cleaned path into the tree, skipping blank segments."""
    current = root
    for part in path:
        label = part.strip()
        if label == "":
            continue
        current = current.children.setdefault(label, _TaxonomyNode())


def _append_tree(buf: list[str], label: str, node: _TaxonomyNode, depth: int) -> None:
    if label != "":
        buf.append(f"{'  ' * depth}{label}")
    if node is None or not node.children:
        return
    for key in sorted(node.children):
        _append_tree(buf, key, node.children[key], depth + 1)


def format_existing_taxonomy_for_prompt(paths: Sequence[Sequence[str]]) -> str:
    """Render distinct category paths as an indented folder tree."""
    root = _TaxonomyNode()
    for path in paths:
        insert_taxonomy_path(root, path)
    if not root.children:
        return ""
    buf: list[str] = []
    for key in sorted(root.children):
        _append_tree(buf, key, root.children[key], 0)
    return "\n".join(buf).strip()


def parse_taxonomy_assignments(raw: str) -> dict[str, list[str]]:
    """Parse the planner's JSON into a slug -> path map.

    Malformed output yields an empty map; entries with a blank slug are
    dropped.
    """
    raw = raw.strip()
    if raw == "":
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError:
        return {}
    assignments = parsed.get("assignments")
    if not isinstance(assignments, list):
        return {}
    out: dict[str, list[str]] = {}
    for entry in assignments:
        if not isinstance(entry, dict):
            continue
        slug = str(entry.get("slug", "")).strip()
        if slug == "":
            continue
        path = entry.get("path")
        if isinstance(path, list):
            out[slug] = [str(part) for part in path]
        else:
            out[slug] = []
    return out


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine of two equal-length vectors, or 0 for empty / mismatched inputs."""
    if not a or len(a) != len(b):
        return 0
    dot = sum(av * bv for av, bv in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(av * av for av in a))
    norm_b = math.sqrt(sum(bv * bv for bv in b))
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot / (norm_a * norm_b)


def select_folders_by_vectors(
    deeper: list[list[str]],
    folder_vecs: list[list[float]],
    item_vecs: list[list[float]],
    top_k: int,
) -> list[list[str]]:
    """Return the deeper folders that rank in any item's top-K by cosine.

    Preserves the input order for determinism.
    """
    if len(deeper) != len(folder_vecs) or not item_vecs or top_k <= 0:
        return []
    chosen: set[int] = set()
    for item_vec in item_vecs:
        ranked = sorted(
            range(len(folder_vecs)),
            key=lambda idx: cosine_similarity(item_vec, folder_vecs[idx]),
            reverse=True,
        )
        for idx in ranked[:top_k]:
            chosen.add(idx)
    return [deeper[idx] for idx in range(len(deeper)) if idx in chosen]


def cap_folders(paths: list[list[str]], max_paths: int) -> list[list[str]]:
    """Truncate a folder list to at most ``max_paths`` entries (<= 0 = no cap)."""
    if max_paths > 0 and len(paths) > max_paths:
        return paths[:max_paths]
    return paths


def _preview_text(text: str, max_chars: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars] + "...(truncated)"


async def select_relevant_folders(
    ctx: Context,
    *,
    items: list[WikiTaxonomyItem],
    pool: list[list[str]],
    embedder: Embedder | None,
) -> list[list[str]]:
    """Narrow the existing-folder pool to the subset worth showing the planner.

    A healthy navigation directory is small, so it is fed whole. Only once
    folders are numerous does similarity preprocessing kick in: all level-1
    folders are always kept as coarse anchors, and each item pulls in its
    nearest deeper folders by embedding similarity. Without an embedder the
    pool is simply capped.
    """
    if len(pool) <= WIKI_TAXONOMY_FEED_ALL_MAX_FOLDERS:
        return pool

    l1_seen: set[str] = set()
    l1_paths: list[list[str]] = []
    deeper: list[list[str]] = []
    for path in pool:
        if not path:
            continue
        if path[0] not in l1_seen:
            l1_seen.add(path[0])
            l1_paths.append([path[0]])
        if len(path) >= 2:
            deeper.append(path)

    if embedder is None or not deeper:
        return cap_folders(pool, WIKI_TAXONOMY_PROMPT_MAX_PATHS)

    folder_texts = [" / ".join(path) for path in deeper]
    item_texts = [
        " ".join([item.title, _preview_text(item.about, 120)]).strip() for item in items
    ]
    if not item_texts:
        return cap_folders(pool, WIKI_TAXONOMY_PROMPT_MAX_PATHS)
    try:
        folder_vecs = await embedder.batch_embed(ctx, folder_texts)
        item_vecs = await embedder.batch_embed(ctx, item_texts)
    except Exception:
        return cap_folders(pool, WIKI_TAXONOMY_PROMPT_MAX_PATHS)

    selected = l1_paths + select_folders_by_vectors(
        deeper, folder_vecs, item_vecs, WIKI_TAXONOMY_RELEVANT_TOP_K
    )
    return cap_folders(selected, WIKI_TAXONOMY_PROMPT_MAX_PATHS)


async def plan_batch_taxonomy(
    ctx: Context,
    *,
    folder_service: WikiFolderStore,
    knowledge_base_id: str,
    language: str,
    slug_updates: Mapping[str, Sequence[WikiSlugUpdate]],
    planner: TaxonomyPlanner | None,
    embedder: Embedder | None = None,
) -> dict[str, list[str]]:
    """Assign a directory path to every new entity / concept slug in the batch.

    Returns a slug -> cleaned category path map; an item may map to an
    empty path when it is unclassifiable. Without a planner seam nothing is
    assigned (the batch falls back to un-filed pages).
    """
    items = collect_taxonomy_items(slug_updates)
    if not items:
        return {}

    pool = await folder_service.list_distinct_category_paths(
        knowledge_base_id=knowledge_base_id,
        max_paths=WIKI_TAXONOMY_FOLDER_POOL_MAX,
    )
    existing = await select_relevant_folders(ctx, items=items, pool=pool, embedder=embedder)
    if planner is None:
        return {}

    result: dict[str, list[str]] = {}
    for start in range(0, len(items), WIKI_TAXONOMY_PLAN_CHUNK_SIZE):
        chunk = items[start : start + WIKI_TAXONOMY_PLAN_CHUNK_SIZE]
        tree = format_existing_taxonomy_for_prompt(existing)
        if tree.strip() == "":
            tree = WIKI_TAXONOMY_EMPTY_TREE_HINT
        items_block = "\n".join(
            f"- slug: {it.slug} | title: {it.title} | type: {it.page_type} | about: {_preview_text(it.about, 120)}"
            for it in chunk
        )
        raw = await planner.plan_assignments(
            existing_taxonomy=tree,
            items_block=items_block,
            language=language,
        )
        for slug, path in parse_taxonomy_assignments(raw).items():
            clean = clean_category_path(path)
            result[slug] = clean
            if clean:
                existing = [*existing, clean]  # feed forward for later chunks
    return result


async def resolve_planned_folders(
    *,
    folder_service: WikiFolderStore,
    knowledge_base_id: str,
    tenant_id: int,
    planned: Mapping[str, list[str]],
) -> dict[str, str]:
    """Reify the planner's per-slug paths into real folder ids.

    Folder creation happens here, sequentially and before the parallel
    reduce phase, so reduce only assigns pre-resolved ids and never races
    two writers into creating the same folder. Blank paths map to the root
    and are omitted.
    """
    out: dict[str, str] = {}
    path_cache: dict[str, str] = {}
    for slug, path in planned.items():
        clean = clean_category_path(path)
        if not clean:
            continue
        key = "/".join(clean)
        folder_id = path_cache.get(key)
        if folder_id is None:
            resolved, _ = await folder_service.find_or_create_folder_path(
                knowledge_base_id=knowledge_base_id,
                tenant_id=tenant_id,
                path=clean,
            )
            path_cache[key] = resolved
            folder_id = resolved
        if folder_id != "":
            out[slug] = folder_id
    return out


__all__ = [
    "WIKI_TAXONOMY_EMPTY_TREE_HINT",
    "WIKI_TAXONOMY_FEED_ALL_MAX_FOLDERS",
    "WIKI_TAXONOMY_FOLDER_POOL_MAX",
    "WIKI_TAXONOMY_PLAN_CHUNK_SIZE",
    "WIKI_TAXONOMY_PROMPT_MAX_PATHS",
    "WIKI_TAXONOMY_RELEVANT_TOP_K",
    "WikiTaxonomyItem",
    "cap_folders",
    "collect_taxonomy_items",
    "cosine_similarity",
    "format_existing_taxonomy_for_prompt",
    "parse_taxonomy_assignments",
    "plan_batch_taxonomy",
    "resolve_planned_folders",
    "select_folders_by_vectors",
    "select_relevant_folders",
]
