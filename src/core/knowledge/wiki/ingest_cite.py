"""Chunk-citation classification for the wiki ingest pipeline.

Given a document's candidate slugs (the pass-0 extraction output) and its
text chunks, ask the classifier which chunks substantively support each
candidate, merge the per-batch results into a single slug -> union of
chunk ids map, and surface any genuinely new slugs the classifier
discovered. Chunk ids are hidden behind request-local handles in the
prompt so the model never has to reproduce a UUID; the stage translates
handles back to real chunk ids before any result reaches application
state.

The classifier itself is an injectable seam (LLM-backed in the worker
layer); everything else in this module is pure text / collection work.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace

from src.ai.embedding import Context
from src.core.knowledge.wiki.ingest_types import (
    ChunkCitationClassifier,
    CitationBatchResult,
    NewSlugFromCitation,
    WikiExtractedItem,
)
from src.core.knowledge.wiki.slug_utils import HandleTable
from src.db.models.chunk import Chunk

#: Bounds the rune size of a single citation batch (a fast approximation of
#: token count) so batches stay inside the classifier's context budget.
MAX_RUNES_PER_CITATION_BATCH: int = 12000

#: Caps how many citation batches run concurrently so one long document
#: cannot saturate the synthesis model.
MAX_CITATION_BATCH_CONCURRENCY: int = 4


@dataclass(frozen=True, slots=True)
class CitationChunkBatch:
    """One citation batch: its chunks plus the request-local handle table.

    Each chunk's id is registered with ``handles`` so the rendered prompt
    references compact aliases instead of raw UUIDs.
    """

    chunks: tuple[Chunk, ...]
    handles: HandleTable


def split_chunks_into_citation_batches(chunks: list[Chunk]) -> list[CitationChunkBatch]:
    """Partition chunks into citation batches whose total runes stay bounded.

    Only text chunks with content are cited — image / ocr chunks are
    already merged into the text content and are not standalone units.
    Chunk order (by ``chunk_index`` then ``start_at``) is preserved, and a
    chunk that by itself exceeds the budget occupies its own batch so no
    content is silently dropped.
    """
    filtered = [
        chunk
        for chunk in chunks
        if chunk is not None
        and chunk.content != ""
        and (chunk.chunk_type == "text" or chunk.chunk_type == "")
    ]
    filtered.sort(key=lambda chunk: (chunk.chunk_index, chunk.start_at))
    if not filtered:
        return []

    batches: list[CitationChunkBatch] = []
    current_chunks: list[Chunk] = []
    current_runes = 0
    current_handles = HandleTable("c", 0, 1)
    for chunk in filtered:
        runes = len(chunk.content)
        if current_chunks and current_runes + runes > MAX_RUNES_PER_CITATION_BATCH:
            batches.append(CitationChunkBatch(tuple(current_chunks), current_handles))
            current_chunks = []
            current_runes = 0
            current_handles = HandleTable("c", 0, 1)
        current_handles.register(chunk.id)
        current_chunks.append(chunk)
        current_runes += runes
    if current_chunks:
        batches.append(CitationChunkBatch(tuple(current_chunks), current_handles))
    return batches


def render_candidate_slugs_xml(
    entities: list[WikiExtractedItem], concepts: list[WikiExtractedItem]
) -> str:
    """Render candidate slugs as the compact list for the prompt.

    Only entries carrying both a slug and a name are rendered; each line
    carries the slug, its type, name, aliases and short description.
    """
    lines: list[str] = []
    for item in entities:
        if item.slug == "" or item.name == "":
            continue
        lines.append(_render_slug_line(item, "entity"))
    for item in concepts:
        if item.slug == "" or item.name == "":
            continue
        lines.append(_render_slug_line(item, "concept"))
    return "\n".join(lines)


def _render_slug_line(item: WikiExtractedItem, kind: str) -> str:
    aliases = ""
    if item.aliases:
        aliases = f" aliases={item.aliases!r}"
    return f"- slug: {item.slug}, type: {kind}, name: {item.name!r}{aliases}, description: {item.description}"


def render_chunks_xml(batch: CitationChunkBatch) -> str:
    """Format one batch's chunks into the prompt's ``<chunks>`` block.

    Uses the batch's request-local handles instead of raw chunk ids; the
    id lookup is idempotent so handles stay stable across renders.
    """
    blocks: list[str] = []
    for chunk in batch.chunks:
        handle = batch.handles.register(chunk.id)
        blocks.append(f'<c id={handle!r} index="{chunk.chunk_index}">\n{chunk.content}\n</c>')
    return "\n".join(blocks)


async def classify_chunk_citations(
    ctx: Context,
    *,
    classifier: ChunkCitationClassifier,
    candidates_xml: str,
    chunks: list[Chunk],
    language: str,
) -> tuple[dict[str, list[str]], list[NewSlugFromCitation], int]:
    """Classify chunk citations across all batches, merging the results.

    Returns ``(citations, new_slugs, batch_count)``. ``citations`` maps a
    slug to the real chunk ids (already translated from handles) ordered by
    document position. ``new_slugs`` carry real chunk ids in their source
    chunks. When there is nothing to classify (no text chunks or no
    candidates), an empty result is returned with ``batch_count == 0``.
    """
    batches = split_chunks_into_citation_batches(chunks)
    if not batches or candidates_xml.strip() == "":
        return {}, [], 0

    semaphore = asyncio.Semaphore(MAX_CITATION_BATCH_CONCURRENCY)

    async def _classify_one(
        batch: CitationChunkBatch,
    ) -> tuple[dict[str, list[str]], list[NewSlugFromCitation]]:
        async with semaphore:
            result: CitationBatchResult = await classifier.classify_batch(
                candidates_xml=candidates_xml,
                chunks_xml=render_chunks_xml(batch),
                language=language,
            )
        return _translate_batch_result(batch, result)

    per_batch = await asyncio.gather(*(_classify_one(batch) for batch in batches))

    # Merge per-batch citations into a slug -> set of real chunk ids.
    citation_set: dict[str, set[str]] = {}
    new_slugs: list[NewSlugFromCitation] = []
    for batch_citations, batch_new_slugs in per_batch:
        for slug, ids in batch_citations.items():
            if slug == "":
                continue
            citation_set.setdefault(slug, set()).update(ids)
        new_slugs.extend(batch_new_slugs)

    chunk_order = {chunk.id: chunk.chunk_index for chunk in chunks}
    citations = {
        slug: sorted(ids, key=lambda chunk_id: chunk_order.get(chunk_id, 0))
        for slug, ids in citation_set.items()
    }
    return citations, new_slugs, len(batches)


def _translate_batch_result(
    batch: CitationChunkBatch,
    result: CitationBatchResult,
) -> tuple[dict[str, list[str]], list[NewSlugFromCitation]]:
    """Translate a batch's handle-keyed classifier output to real chunk ids.

    Handles the model referenced but never registered for the batch are
    dropped (logged upstream by the classifier seam).
    """
    citations: dict[str, list[str]] = {}
    for slug, handle_list in result.citations.items():
        if slug == "":
            continue
        real_ids = [real for handle in handle_list if (real := _resolve(batch, handle))]
        if real_ids:
            citations[slug] = real_ids

    new_slugs: list[NewSlugFromCitation] = []
    for new_slug in result.new_slugs:
        if new_slug.slug == "" or new_slug.name == "":
            continue
        real_chunks = tuple(
            real for handle in new_slug.source_chunks if (real := _resolve(batch, handle))
        )
        new_slugs.append(
            NewSlugFromCitation(
                type=new_slug.type,
                name=new_slug.name,
                slug=new_slug.slug,
                aliases=new_slug.aliases,
                description=new_slug.description,
                details=new_slug.details,
                source_chunks=real_chunks,
            )
        )
    return citations, new_slugs


def _resolve(batch: CitationChunkBatch, handle: str) -> str:
    real_id, known = batch.handles.resolve(handle)
    return real_id if known else ""


def merge_citations_into_items(
    entities: list[WikiExtractedItem],
    concepts: list[WikiExtractedItem],
    citations: dict[str, list[str]],
    new_slugs: list[NewSlugFromCitation],
) -> tuple[list[WikiExtractedItem], list[WikiExtractedItem], int]:
    """Backfill source chunks on every item and append newly discovered slugs.

    Items whose slug is not in ``citations`` keep their description /
    details fallback. Returns ``(entities, concepts, uncited_count)``.
    """
    entities = [replace(item, source_chunks=tuple(citations.get(item.slug, []))) for item in entities]
    concepts = [replace(item, source_chunks=tuple(citations.get(item.slug, []))) for item in concepts]
    uncited = sum(1 for item in entities if not item.source_chunks) + sum(
        1 for item in concepts if not item.source_chunks
    )

    # Aggregate newly discovered slugs per slug, unioning their source
    # chunks across batches.
    existing_slugs = {item.slug for item in entities} | {item.slug for item in concepts}
    merged: dict[str, tuple[str, WikiExtractedItem]] = {}
    slug_order: list[str] = []
    for new_slug in new_slugs:
        if new_slug.slug in existing_slugs:
            continue
        kind = new_slug.type.strip().lower()
        if kind == "":
            kind = "concept" if new_slug.slug.startswith("concept/") else "entity"
        previous = merged.get(new_slug.slug)
        if previous is None:
            merged[new_slug.slug] = (
                kind,
                WikiExtractedItem(
                    name=new_slug.name,
                    slug=new_slug.slug,
                    aliases=tuple(new_slug.aliases),
                    description=new_slug.description,
                    details=new_slug.details,
                    source_chunks=tuple(new_slug.source_chunks),
                ),
            )
            slug_order.append(new_slug.slug)
            continue
        seen = set(previous[1].source_chunks)
        union = tuple(previous[1].source_chunks) + tuple(
            chunk_id for chunk_id in new_slug.source_chunks if chunk_id not in seen
        )
        merged[new_slug.slug] = (kind, replace(previous[1], source_chunks=union))

    appended_entities: list[WikiExtractedItem] = []
    appended_concepts: list[WikiExtractedItem] = []
    for slug in slug_order:
        kind, item = merged[slug]
        if kind == "concept":
            appended_concepts.append(item)
        else:
            appended_entities.append(item)
    return entities + appended_entities, concepts + appended_concepts, uncited


def collect_cited_chunk_content(chunk_ids: list[str], content_by_id: dict[str, str]) -> str:
    """Materialise the verbatim content of every referenced chunk.

    Chunk ids that cannot be resolved are silently dropped. Content is
    concatenated in the order the ids are provided.
    """
    blocks: list[str] = []
    for chunk_id in chunk_ids:
        content = content_by_id.get(chunk_id)
        if content is None or content.strip() == "":
            continue
        blocks.append(content)
    return "\n\n".join(blocks)


__all__ = [
    "MAX_CITATION_BATCH_CONCURRENCY",
    "MAX_RUNES_PER_CITATION_BATCH",
    "CitationChunkBatch",
    "classify_chunk_citations",
    "collect_cited_chunk_content",
    "merge_citations_into_items",
    "render_candidate_slugs_xml",
    "render_chunks_xml",
    "split_chunks_into_citation_batches",
]
