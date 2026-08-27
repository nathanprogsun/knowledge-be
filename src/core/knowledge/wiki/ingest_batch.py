"""Wiki ingest batch driver.

Pulls a batch of queued operations for a knowledge base, maps each
document through the document pipeline (parse -> chunk -> embed -> index)
plus the wiki content passes (extraction, summary, chunk citation, dedup,
taxonomy), then reduces the collected updates onto wiki pages and settles
the queue — trimming consumed rows, re-queuing failed ones through a
retry budget, and dead-lettering rows that exhaust it.

The map phase runs sequentially here; the worker layer may fan it out once
the async task infrastructure lands. Every external dependency (parser,
embedder, index writer, synthesis seams, stores) arrives through the
:class:`WikiIngestDeps` bundle, so the stages stay testable with fakes and
the real wiring happens at the orchestrator boundary.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import replace
from datetime import UTC, datetime

from src.ai.embedding import Context
from src.common.exception import NotFoundError
from src.common.json import SqlValue
from src.core.knowledge.documents.chunker import default_config, split
from src.core.knowledge.documents.types import (
    PARSE_STATUS_COMPLETED,
    PARSE_STATUS_DELETING,
    PARSE_STATUS_FAILED,
    PARSE_STATUS_FINALIZING,
    PARSE_STATUS_PENDING,
    PARSE_STATUS_PROCESSING,
)
from src.core.knowledge.wiki.ingest_cite import (
    classify_chunk_citations,
    merge_citations_into_items,
    render_candidate_slugs_xml,
)
from src.core.knowledge.wiki.ingest_dedup import (
    deduplicate_items,
    select_dedup_candidate_pages,
)
from src.core.knowledge.wiki.ingest_taxonomy import plan_batch_taxonomy, resolve_planned_folders
from src.core.knowledge.wiki.ingest_types import (
    INDEX_STATUS_FAILED,
    INDEX_STATUS_PROCESSING,
    INDEX_STATUS_READY,
    MAX_CONTENT_FOR_WIKI,
    WIKI_MAX_DOCS_PER_BATCH,
    WIKI_MAX_FAIL_RETRIES,
    WIKI_OP_RETRACT,
    WIKI_PAGE_TYPE_CONCEPT,
    WIKI_PAGE_TYPE_ENTITY,
    WIKI_PAGE_TYPE_INDEX,
    WIKI_PAGE_TYPE_SUMMARY,
    WIKI_UPDATE_RETRACT,
    WIKI_UPDATE_RETRACT_STALE,
    WikiBatchContext,
    WikiBatchOutcome,
    WikiDocIngestResult,
    WikiExtractedItem,
    WikiIngestDeps,
    WikiIngestOp,
    WikiPageRef,
    WikiSlugUpdate,
    chunk_embedding_text,
)
from src.core.knowledge.wiki.types import WIKI_PAGE_STATUS_DRAFT
from src.db.models.chunk import Chunk
from src.db.models.knowledge import Document
from src.db.models.wiki_page import WikiPage

#: Minimum non-whitespace, non-image-reference runes for a document to be
#: worth the wiki content passes (primary defence against filename-driven
#: hallucinations on scanned PDFs with no usable text).
_MIN_TEXT_CONTENT_RUNES: int = 10

_MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_HTML_IMG_RE = re.compile(r"(?i)<img\b[^>]*>")

# Manual-knowledge bodies are stored under this metadata key.
_METADATA_KEY_MANUAL_CONTENT = "content"


# ── Pure helpers ──────────────────────────────────────────────────────


def slugify(value: str) -> str:
    """Return a URL-safe slug for ``value`` (used for summary page slugs)."""
    slug = value.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-")


def strip_image_markup(content: str) -> str:
    """Strip image references so text-yield checks ignore image-only docs."""
    content = _MARKDOWN_IMAGE_RE.sub("", content)
    return _HTML_IMG_RE.sub("", content)


def real_text_runes(content: str) -> int:
    """Rune count of the content after image markup is stripped."""
    return len(strip_image_markup(content).strip())


def has_sufficient_text_content(content: str) -> bool:
    """Report whether the content carries enough real text for the wiki passes."""
    return real_text_runes(content) >= _MIN_TEXT_CONTENT_RUNES


#: Headline prefixes the summariser emits (half- and full-width colon).
_SUMMARY_PREFIXES: tuple[str, ...] = ("SUMMARY:", "SUMMARY：")


def split_summary_line(raw: str) -> tuple[str, str]:
    """Extract the ``SUMMARY: ...`` headline line from summariser output.

    Returns ``(summary, content)``; when no headline is found the summary
    is empty and the whole text is treated as content.
    """
    text = raw.strip()
    if text.startswith(_SUMMARY_PREFIXES):
        idx = text.find("\n")
        if idx < 0:
            return _strip_summary_prefix(text), ""
        return _strip_summary_prefix(text[:idx]), text[idx + 1 :].strip()
    return "", text


def _strip_summary_prefix(line: str) -> str:
    for prefix in _SUMMARY_PREFIXES:
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return line.strip()


def manual_document_content(document: Document) -> str:
    """Return a manual knowledge's stored markdown body ("" when absent)."""
    metadata = document.metadata or {}
    value = metadata.get(_METADATA_KEY_MANUAL_CONTENT)
    return value if isinstance(value, str) else ""


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _resolve_doc_title(document: Document, chunks: list[Chunk]) -> str:
    if document.title.strip():
        return document.title.strip()
    for chunk in chunks:
        if not chunk.content:
            continue
        first_line = chunk.content.split("\n", 1)[0].strip()
        if first_line and len(first_line) < 200:
            return first_line.lstrip("# ").strip()
    return document.id


def _append_unique(values: list[str], *items: str) -> list[str]:
    seen = set(values)
    out = list(values)
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _merge_chunk_refs(current: list[str], additions: list[WikiSlugUpdate]) -> list[str]:
    """Union chunk ids already on a page with those cited by the additions."""
    seen = set(current)
    out = list(current)
    for addition in additions:
        for chunk_id in addition.source_chunks:
            if chunk_id and chunk_id not in seen:
                seen.add(chunk_id)
                out.append(chunk_id)
    return out


def _ref_knowledge_id(ref: str) -> str:
    return ref.split("|", 1)[0]


def _has_additions(updates: list[WikiSlugUpdate]) -> bool:
    return any(
        update.type in (WIKI_PAGE_TYPE_ENTITY, WIKI_PAGE_TYPE_CONCEPT, WIKI_PAGE_TYPE_SUMMARY)
        for update in updates
    )


def _render_additions_content(additions: list[WikiSlugUpdate]) -> str:
    """Deterministic markdown body for a page fed by entity / concept additions.

    The content-editor seam (which would quote cited chunks verbatim and
    reconcile with existing prose) is deferred to the worker layer; this
    deterministic rendering keeps the reduce stage self-contained.
    """
    blocks: list[str] = []
    for addition in additions:
        if addition.item is None:
            continue
        body = addition.item.details or addition.item.description
        blocks.append(
            f"<document>\n<title>{addition.doc_title}</title>\n<content>\n"
            f"**{addition.item.name}**: {addition.item.description}\n\n{body}\n"
            f"</content>\n</document>"
        )
    return "\n\n".join(blocks)


def _new_page(*, tenant_id: int, knowledge_base_id: str, slug: str) -> WikiPage:
    now = datetime.now(UTC)
    return WikiPage(
        id=str(uuid.uuid4()),
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        slug=slug,
        page_type="",
        status=WIKI_PAGE_STATUS_DRAFT,
        source_refs=[],
        aliases=[],
        created_at=now,
        updated_at=now,
    )


# ── Document pipeline: parse -> chunk -> embed -> index ──────────────


async def _chunk_and_persist(
    ctx: Context,
    *,
    deps: WikiIngestDeps,
    tenant_id: int,
    knowledge_base_id: str,
    knowledge_id: str,
    content: str,
) -> list[Chunk]:
    """Split ``content`` with the merged chunker and persist the chunk rows.

    Re-ingesting a document replaces its previous chunk set (soft-deleted
    before the new rows are written). Rows are stamped ``processing`` so
    the index pass can settle them to ``ready`` / ``failed``.
    """
    cfg = deps.splitter_config or default_config()
    parts = split(content, cfg)
    if not parts:
        return []
    now = datetime.now(UTC)
    rows: list[Chunk] = []
    for i, part in enumerate(parts):
        previous_id = rows[i - 1].id if i > 0 else None
        rows.append(
            Chunk(
                id=str(uuid.uuid4()),
                tenant_id=tenant_id,
                knowledge_base_id=knowledge_base_id,
                knowledge_id=knowledge_id,
                content=part.content,
                chunk_index=part.seq,
                is_enabled=True,
                start_at=part.start,
                end_at=part.end,
                pre_chunk_id=previous_id,
                next_chunk_id=None,
                chunk_type="text",
                parent_chunk_id=None,
                status=0,
                content_hash=_content_hash(part.content),
                index_status=INDEX_STATUS_PROCESSING,
                context_header=part.context_header,
                created_at=now,
                updated_at=now,
            )
        )
    rows = [
        row.model_copy(update={"next_chunk_id": rows[i + 1].id}) if i < len(rows) - 1 else row
        for i, row in enumerate(rows)
    ]
    await deps.chunk_store.delete_by_knowledge_id(
        tenant_id=tenant_id, knowledge_id=knowledge_id, now=now
    )
    return await deps.chunk_store.create_many(rows)


async def _embed_and_index(
    ctx: Context,
    *,
    deps: WikiIngestDeps,
    tenant_id: int,
    knowledge_base_id: str,
    knowledge_id: str,
    chunks: list[Chunk],
) -> None:
    """Embed the chunk rows and push them into the retrieval index.

    Settles each row's ``index_status`` to ``ready`` on success or
    ``failed`` when the index write trips — the rows stay saved either way.
    Without an embedder or index writer the rows keep their ``processing``
    stamp (the wiring layer promotes them later).
    """
    if deps.embedder is None or deps.index_writer is None or not chunks:
        return
    texts = [chunk_embedding_text(chunk) for chunk in chunks]
    embeddings = await deps.embedder.batch_embed_with_pool(ctx, deps.embedder, texts)
    status = INDEX_STATUS_READY
    try:
        await deps.index_writer.write_chunks(
            ctx=ctx,
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            knowledge_id=knowledge_id,
            chunks=chunks,
            embeddings=embeddings,
            embedder=deps.embedder,
        )
    except Exception:
        status = INDEX_STATUS_FAILED
    now = datetime.now(UTC)
    for row in chunks:
        if row.index_status == status:
            continue
        await deps.chunk_store.update(
            row.model_copy(update={"index_status": status, "updated_at": now})
        )


async def _run_dedup(
    ctx: Context,
    *,
    deps: WikiIngestDeps,
    entities: list[WikiExtractedItem],
    concepts: list[WikiExtractedItem],
    existing_pages: list[WikiPage],
) -> tuple[list[WikiExtractedItem], list[WikiExtractedItem]]:
    """Apply the dedup merge pass to the extracted items.

    Without a merger seam the extraction passes through unchanged (the
    safe no-dedup default).
    """
    if deps.merger is None or not existing_pages:
        return entities, concepts
    candidates = select_dedup_candidate_pages([*entities, *concepts], existing_pages)
    if not candidates:
        return entities, concepts
    kept_entities, _ = await deduplicate_items(entities, candidates, deps.merger)
    kept_concepts, _ = await deduplicate_items(concepts, candidates, deps.merger)
    return kept_entities, kept_concepts


# ── Map phase ─────────────────────────────────────────────────────────


async def map_one_document(
    ctx: Context,
    *,
    deps: WikiIngestDeps,
    tenant_id: int,
    knowledge_base_id: str,
    op: WikiIngestOp,
    batch_ctx: WikiBatchContext,
    existing_pages: list[WikiPage],
) -> tuple[WikiDocIngestResult | None, list[WikiSlugUpdate]]:
    """Map one queued operation onto wiki page updates.

    Retract operations expand their slug set against the live pages that
    reference the knowledge and return only retract updates. Ingest
    operations run the document pipeline (parse -> chunk -> embed -> index)
    and then the wiki content passes, returning ``(result, updates)`` or
    ``(None, [])`` when the document reached a terminal skip state (gone,
    mid-deletion, no chunks, or insufficient text).
    """
    if op.op == WIKI_OP_RETRACT:
        updates = await _build_retract_updates(ctx, deps, knowledge_base_id, op)
        return None, updates

    document = await deps.document_store.get_by_id(tenant_id, op.knowledge_id)
    if document is None or document.parse_status == PARSE_STATUS_DELETING:
        return None, []
    if document.parse_status == PARSE_STATUS_PENDING:
        await deps.document_store.update_columns(
            document.id, {"parse_status": PARSE_STATUS_PROCESSING}
        )

    content = await _parse_document(ctx, deps, tenant_id, document)
    if not has_sufficient_text_content(content):
        return None, []
    if len(content) > MAX_CONTENT_FOR_WIKI:
        content = content[:MAX_CONTENT_FOR_WIKI]

    chunks = await _chunk_and_persist(
        ctx,
        deps=deps,
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        knowledge_id=op.knowledge_id,
        content=content,
    )
    if not chunks:
        return None, []
    await _embed_and_index(
        ctx,
        deps=deps,
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        knowledge_id=op.knowledge_id,
        chunks=chunks,
    )

    doc_title = _resolve_doc_title(document, chunks)
    source_ref = op.knowledge_id
    old_slug_set = {
        slug
        for slug in await deps.page_service.list_slugs_by_source_ref(
            knowledge_base_id=knowledge_base_id, source_knowledge_id=op.knowledge_id
        )
        if slug != WIKI_PAGE_TYPE_INDEX
    }

    entities: list[WikiExtractedItem] = []
    concepts: list[WikiExtractedItem] = []
    if deps.extractor is not None:
        entities, concepts = await deps.extractor.extract_candidate_slugs(
            content=content,
            language=op.language,
            previous_slugs=tuple(sorted(old_slug_set)),
            granularity=batch_ctx.extraction_granularity,
            extraction_instructions=batch_ctx.extraction_instructions,
        )
        entities, concepts = await _run_dedup(
            ctx, deps=deps, entities=entities, concepts=concepts, existing_pages=existing_pages
        )

    summary_content = ""
    if deps.summarizer is not None:
        slug_items = [item for item in [*entities, *concepts] if item.slug and item.name]
        summary_content = await deps.summarizer.summarize(
            content=content,
            language=op.language,
            extracted_slugs=tuple(sorted({item.slug for item in slug_items})),
            custom_instructions=batch_ctx.content_instructions,
        )

    if deps.classifier is not None and (entities or concepts):
        candidates_xml = render_candidate_slugs_xml(entities, concepts)
        citations, new_slugs, _batch_count = await classify_chunk_citations(
            ctx,
            classifier=deps.classifier,
            candidates_xml=candidates_xml,
            chunks=chunks,
            language=op.language,
        )
        entities, concepts, _uncited = merge_citations_into_items(
            entities, concepts, citations, new_slugs
        )

    prior_summary_map = await deps.page_service.list_summaries_by_knowledge_ids(
        knowledge_base_id=knowledge_base_id, knowledge_ids=[op.knowledge_id]
    )
    return _build_map_updates(
        deps,
        knowledge_id=op.knowledge_id,
        doc_title=doc_title,
        source_ref=source_ref,
        language=op.language,
        old_slug_set=old_slug_set,
        entities=entities,
        concepts=concepts,
        summary_content=summary_content,
        content=content,
        prior_summary=prior_summary_map.get(op.knowledge_id, ""),
    )


async def _parse_document(
    ctx: Context, deps: WikiIngestDeps, tenant_id: int, document: Document
) -> str:
    """Run the parse stage: parser seam first, then the manual-content fallback."""
    if deps.parser is not None:
        content = await deps.parser.parse_text(tenant_id=tenant_id, document=document)
        if content.strip():
            return content
    return manual_document_content(document)


def _build_map_updates(
    deps: WikiIngestDeps,
    *,
    knowledge_id: str,
    doc_title: str,
    source_ref: str,
    language: str,
    old_slug_set: set[str],
    entities: list[WikiExtractedItem],
    concepts: list[WikiExtractedItem],
    summary_content: str,
    content: str,
    prior_summary: str,
) -> tuple[WikiDocIngestResult | None, list[WikiSlugUpdate]]:
    """Assemble the map-phase update list and per-document result."""
    updates: list[WikiSlugUpdate] = []
    extracted_pages: list[WikiPageRef] = []
    sum_line = doc_title
    sum_body = summary_content

    if deps.summarizer is not None and summary_content.strip():
        line, body = split_summary_line(summary_content)
        sum_body = body or summary_content
        sum_line = line or doc_title
        summary_slug = f"summary/{slugify(knowledge_id)}"
        updates.append(
            WikiSlugUpdate(
                slug=summary_slug,
                type=WIKI_PAGE_TYPE_SUMMARY,
                doc_title=doc_title,
                knowledge_id=knowledge_id,
                source_ref=source_ref,
                language=language,
                summary_line=sum_line,
                summary_body=sum_body,
            )
        )
        extracted_pages.append(WikiPageRef(summary_slug, doc_title))

    for item in entities:
        if not item.slug:
            continue
        updates.append(
            WikiSlugUpdate(
                slug=item.slug,
                type=WIKI_PAGE_TYPE_ENTITY,
                item=item,
                doc_title=doc_title,
                knowledge_id=knowledge_id,
                source_ref=source_ref,
                language=language,
                source_chunks=item.source_chunks,
                doc_summary=sum_body,
            )
        )
        extracted_pages.append(WikiPageRef(item.slug, item.name or item.slug))
    for item in concepts:
        if not item.slug:
            continue
        updates.append(
            WikiSlugUpdate(
                slug=item.slug,
                type=WIKI_PAGE_TYPE_CONCEPT,
                item=item,
                doc_title=doc_title,
                knowledge_id=knowledge_id,
                source_ref=source_ref,
                language=language,
                source_chunks=item.source_chunks,
                doc_summary=sum_body,
            )
        )
        extracted_pages.append(WikiPageRef(item.slug, item.name or item.slug))

    new_slug_set = {ref.slug for ref in extracted_pages}

    for old_slug in sorted(old_slug_set):
        if old_slug in new_slug_set:
            # Reparse overlap: the page keeps its identity; emit a retract
            # carrying the prior summary so the content pass can replace
            # (not append to) the document's section.
            if old_slug.startswith("summary/"):
                continue
            updates.append(
                WikiSlugUpdate(
                    slug=old_slug,
                    type=WIKI_UPDATE_RETRACT,
                    retract_doc_content=prior_summary,
                    doc_title=doc_title,
                    knowledge_id=knowledge_id,
                    language=language,
                )
            )
            continue
        # Stale: the document no longer produces this page.
        updates.append(
            WikiSlugUpdate(
                slug=old_slug,
                type=WIKI_UPDATE_RETRACT_STALE,
                retract_doc_content=content,
                doc_title=doc_title,
                knowledge_id=knowledge_id,
                language=language,
            )
        )

    result = WikiDocIngestResult(
        knowledge_id=knowledge_id,
        doc_title=doc_title,
        summary=sum_line,
        pages=tuple(extracted_pages),
    )
    return result, updates


async def _build_retract_updates(
    ctx: Context,
    deps: WikiIngestDeps,
    knowledge_base_id: str,
    op: WikiIngestOp,
) -> list[WikiSlugUpdate]:
    """Expand a retract op's slug set and emit retract updates for each page.

    Re-queries the live pages that reference the knowledge at run time so
    pages created after the op was enqueued are still retracted.
    """
    slug_set = {slug for slug in op.page_slugs if slug and slug != WIKI_PAGE_TYPE_INDEX}
    if op.knowledge_id:
        live = await deps.page_service.list_slugs_by_source_ref(
            knowledge_base_id=knowledge_base_id, source_knowledge_id=op.knowledge_id
        )
        slug_set.update(slug for slug in live if slug and slug != WIKI_PAGE_TYPE_INDEX)
    return [
        WikiSlugUpdate(
            slug=slug,
            type=WIKI_UPDATE_RETRACT,
            retract_doc_content=op.doc_summary,
            doc_title=op.doc_title,
            knowledge_id=op.knowledge_id,
            language=op.language,
        )
        for slug in sorted(slug_set)
    ]


# ── Reduce phase ──────────────────────────────────────────────────────


async def reduce_slug_updates(
    ctx: Context,
    *,
    deps: WikiIngestDeps,
    tenant_id: int,
    knowledge_base_id: str,
    slug: str,
    updates: list[WikiSlugUpdate],
    batch_ctx: WikiBatchContext,
) -> tuple[bool, str, bool]:
    """Reduce one slug's updates onto its wiki page.

    Returns ``(changed, affected_type, addition_failed)``. ``affected_type``
    is ``"ingest"`` or ``"retract"`` and drives downstream bookkeeping;
    ``addition_failed`` is always ``False`` here (the content pass is
    deterministic and cannot fail independently of the write).
    """
    live_updates = await _filter_live_updates(ctx, deps, knowledge_base_id, updates)
    if not live_updates:
        return False, "", False

    page = await _get_page_or_none(deps, knowledge_base_id, slug)
    exists = page is not None
    if page is None:
        if not _has_additions(live_updates):
            return False, "", False
        page = _new_page(tenant_id=tenant_id, knowledge_base_id=knowledge_base_id, slug=slug)

    summary_update = next((u for u in live_updates if u.type == WIKI_PAGE_TYPE_SUMMARY), None)
    retracts = [
        u for u in live_updates if u.type in (WIKI_UPDATE_RETRACT, WIKI_UPDATE_RETRACT_STALE)
    ]
    additions = [
        u for u in live_updates if u.type in (WIKI_PAGE_TYPE_ENTITY, WIKI_PAGE_TYPE_CONCEPT)
    ]

    if summary_update is not None:
        page = _apply_summary(page, summary_update)
        await _save_page(deps, page, exists)
        return True, "ingest", False

    if retracts:
        page = _strip_retracted_refs(page, retracts)

    if additions:
        page = _apply_additions(page, additions, batch_ctx)
        await _save_page(deps, page, exists)
        return True, "ingest", False

    if retracts:
        if not page.source_refs:
            await deps.page_service.delete_page(knowledge_base_id=knowledge_base_id, slug=slug)
            return True, "retract", False
        await deps.page_service.update_page_meta(page=page)
        return True, "retract", False

    return False, "", False


async def _filter_live_updates(
    ctx: Context,
    deps: WikiIngestDeps,
    knowledge_base_id: str,
    updates: list[WikiSlugUpdate],
) -> list[WikiSlugUpdate]:
    """Drop addition updates whose source document is already gone.

    Retract updates are kept — they actively remove references, which is
    the desired behaviour when the source is deleted.
    """
    gone: set[str] = set()
    for update in updates:
        kid = update.knowledge_id
        if not kid or update.type in (WIKI_UPDATE_RETRACT, WIKI_UPDATE_RETRACT_STALE):
            continue
        if kid in gone:
            continue
        if await deps.document_store.get_by_id_only(kid) is None:
            gone.add(kid)
    return [
        update
        for update in updates
        if update.type in (WIKI_UPDATE_RETRACT, WIKI_UPDATE_RETRACT_STALE)
        or update.knowledge_id not in gone
    ]


def _apply_summary(page: WikiPage, update: WikiSlugUpdate) -> WikiPage:
    return page.model_copy(
        update={
            "title": f"{update.doc_title} - Summary" if update.doc_title else page.title,
            "content": update.summary_body,
            "summary": update.summary_line,
            "page_type": WIKI_PAGE_TYPE_SUMMARY,
            "source_refs": _append_unique(page.source_refs, update.source_ref),
            "chunk_refs": [],
        }
    )


def _strip_retracted_refs(page: WikiPage, retracts: list[WikiSlugUpdate]) -> WikiPage:
    retract_kids = {r.knowledge_id for r in retracts if r.knowledge_id}
    new_refs = [ref for ref in page.source_refs if _ref_knowledge_id(ref) not in retract_kids]
    return page.model_copy(update={"source_refs": new_refs})


def _apply_additions(
    page: WikiPage, additions: list[WikiSlugUpdate], batch_ctx: WikiBatchContext
) -> WikiPage:
    first = additions[0]
    item_name = first.item.name if first.item is not None else ""
    aliases = list(page.aliases)
    for addition in additions:
        if addition.item is None:
            continue
        for alias in addition.item.aliases:
            if alias and alias not in aliases:
                aliases.append(alias)
    source_refs = list(page.source_refs)
    for addition in additions:
        if addition.source_ref and addition.source_ref not in source_refs:
            source_refs.append(addition.source_ref)
    chunk_refs = _merge_chunk_refs(page.chunk_refs, additions)
    folder_id = page.folder_id
    if folder_id == "":
        folder_id = batch_ctx.planned_folder_id.get(first.slug, "")
    return page.model_copy(
        update={
            "title": page.title or item_name,
            "page_type": page.page_type or first.type,
            "content": _render_additions_content(additions),
            "aliases": aliases,
            "source_refs": source_refs,
            "chunk_refs": chunk_refs,
            "folder_id": folder_id,
        }
    )


async def _save_page(deps: WikiIngestDeps, page: WikiPage, exists: bool) -> None:
    if exists:
        await deps.page_service.update_page(page=page)
    else:
        await deps.page_service.create_page(page=page)


async def _get_page_or_none(
    deps: WikiIngestDeps, knowledge_base_id: str, slug: str
) -> WikiPage | None:
    """Load one page, tolerating both ``None`` and raise-on-missing stores."""
    try:
        return await deps.page_service.get_page_by_slug(
            knowledge_base_id=knowledge_base_id, slug=slug
        )
    except NotFoundError:
        return None


# ── Batch driver ──────────────────────────────────────────────────────


async def process_wiki_ingest_batch(
    ctx: Context,
    *,
    deps: WikiIngestDeps,
    tenant_id: int,
    knowledge_base_id: str,
    language: str,
    batch_ctx: WikiBatchContext,
    max_docs: int = WIKI_MAX_DOCS_PER_BATCH,
    max_fail_retries: int = WIKI_MAX_FAIL_RETRIES,
) -> WikiBatchOutcome:
    """Run one batch: peek ops, map each document, reduce, settle the queue.

    Returns an aggregate :class:`WikiBatchOutcome`. A batch that yields
    nothing still reports ``follow_up_scheduled`` based on the remaining
    queue depth so the worker layer can chain the next drain.
    """
    ops = await deps.pending_store.peek(knowledge_base_id=knowledge_base_id, limit=max_docs)
    if not ops:
        return WikiBatchOutcome(
            pending_ops=0,
            ingest_succeeded=0,
            ingest_failed=0,
            retract_handled=0,
            pages_affected=0,
            follow_up_scheduled=False,
        )

    existing_pages = await _load_existing_pages(ctx, deps, knowledge_base_id)

    results: list[WikiDocIngestResult] = []
    failed_ops: list[WikiIngestOp] = []
    updates_by_slug: dict[str, list[WikiSlugUpdate]] = {}
    ingest_succeeded = 0
    ingest_failed = 0
    retract_handled = 0

    for op in ops:
        try:
            result, updates = await map_one_document(
                ctx,
                deps=deps,
                tenant_id=tenant_id,
                knowledge_base_id=knowledge_base_id,
                op=op,
                batch_ctx=batch_ctx,
                existing_pages=existing_pages,
            )
        except Exception:
            ingest_failed += 1
            failed_ops.append(op)
            continue
        if op.op == WIKI_OP_RETRACT:
            retract_handled += 1
        elif result is not None:
            ingest_succeeded += 1
            results.append(result)
        for update in updates:
            updates_by_slug.setdefault(update.slug, []).append(update)

    # Taxonomy planning is batch-global so the whole set converges onto one
    # coherent tree; reduce then only assigns pre-resolved folder ids.
    planned_folders: dict[str, str] = {}
    if deps.planner is not None:
        planned = await plan_batch_taxonomy(
            ctx,
            folder_service=deps.folder_service,
            knowledge_base_id=knowledge_base_id,
            language=language,
            slug_updates=updates_by_slug,
            planner=deps.planner,
            embedder=deps.embedder,
        )
        planned_folders = await resolve_planned_folders(
            folder_service=deps.folder_service,
            knowledge_base_id=knowledge_base_id,
            tenant_id=tenant_id,
            planned=planned,
        )
    batch_ctx = replace(batch_ctx, planned_folder_id=planned_folders)

    pages_affected = 0
    for slug, updates in updates_by_slug.items():
        changed, _affected_type, _addition_failed = await reduce_slug_updates(
            ctx,
            deps=deps,
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            slug=slug,
            updates=updates,
            batch_ctx=batch_ctx,
        )
        if changed:
            pages_affected += 1

    await _settle_queue(
        ctx,
        deps=deps,
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        ops=ops,
        failed_ops=failed_ops,
        results=results,
        max_fail_retries=max_fail_retries,
    )

    remaining = await deps.pending_store.pending_count(knowledge_base_id=knowledge_base_id)
    return WikiBatchOutcome(
        pending_ops=len(ops),
        ingest_succeeded=ingest_succeeded,
        ingest_failed=ingest_failed,
        retract_handled=retract_handled,
        pages_affected=pages_affected,
        follow_up_scheduled=remaining > 0,
    )


async def _load_existing_pages(
    ctx: Context, deps: WikiIngestDeps, knowledge_base_id: str
) -> list[WikiPage]:
    pages = await deps.page_service.list_all_pages(knowledge_base_id=knowledge_base_id)
    return [
        page for page in pages if page.page_type in (WIKI_PAGE_TYPE_ENTITY, WIKI_PAGE_TYPE_CONCEPT)
    ]


async def _settle_queue(
    ctx: Context,
    *,
    deps: WikiIngestDeps,
    tenant_id: int,
    knowledge_base_id: str,
    ops: list[WikiIngestOp],
    failed_ops: list[WikiIngestOp],
    results: list[WikiDocIngestResult],
    max_fail_retries: int,
) -> None:
    """Trim consumed rows and run failed ops through the retry budget.

    Successful and skipped ops are removed from the queue; failed ops get a
    fail-count bump (released for retry) or, once the budget is exhausted,
    are archived to the dead-letter store and removed.
    """
    failed_ids = {op.row_id for op in failed_ops if op.row_id != 0}
    trim_ids = [op.row_id for op in ops if op.row_id != 0 and op.row_id not in failed_ids]
    if trim_ids:
        await deps.pending_store.delete_by_ids(trim_ids)

    for op in failed_ops:
        if op.row_id == 0:
            continue
        count = await deps.pending_store.increment_fail_count(op.row_id)
        if count > max_fail_retries:
            await deps.pending_store.archive(
                tenant_id=tenant_id,
                knowledge_base_id=knowledge_base_id,
                op=op,
                fail_count=count,
                last_error=(f"exceeded max_fail_retries={max_fail_retries} (in-batch retries)"),
            )
            await deps.pending_store.delete_by_ids([op.row_id])
            await _settle_document_terminal(ctx, deps, tenant_id, op.knowledge_id, failed=True)
        else:
            await deps.pending_store.release_by_ids([op.row_id])

    for result in results:
        await _settle_document_terminal(ctx, deps, tenant_id, result.knowledge_id, failed=False)


async def _settle_document_terminal(
    ctx: Context,
    deps: WikiIngestDeps,
    tenant_id: int,
    knowledge_id: str,
    failed: bool,
) -> None:
    """Move a document to its terminal parse status once its wiki op settles.

    On success the wiki subtask slot is drained (``pending_subtasks_count``
    decremented, promoting to ``completed`` when the counter hits zero);
    on terminal failure the row is marked ``failed``.
    """
    document = await deps.document_store.get_by_id(tenant_id, knowledge_id)
    if document is None:
        return
    if failed:
        await deps.document_store.update_columns(
            knowledge_id, {"parse_status": PARSE_STATUS_FAILED}
        )
        return
    if document.parse_status == PARSE_STATUS_PROCESSING:
        await deps.document_store.update_columns(
            knowledge_id, {"parse_status": PARSE_STATUS_COMPLETED}
        )
        return
    if document.pending_subtasks_count > 0:
        remaining = document.pending_subtasks_count - 1
        values: dict[str, SqlValue] = {"pending_subtasks_count": remaining}
        if document.parse_status == PARSE_STATUS_FINALIZING and remaining == 0:
            values["parse_status"] = PARSE_STATUS_COMPLETED
        await deps.document_store.update_columns(knowledge_id, values)


__all__ = [
    "has_sufficient_text_content",
    "manual_document_content",
    "map_one_document",
    "process_wiki_ingest_batch",
    "real_text_runes",
    "reduce_slug_updates",
    "slugify",
    "split_summary_line",
    "strip_image_markup",
]
