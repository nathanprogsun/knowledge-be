"""Shared types and injectable seams for the wiki ingest pipeline.

This module holds the domain value types, the injectable protocol seams,
and the pipeline tunables shared by the four sub-stages (batch, cite,
dedup, taxonomy) and the :class:`WikiIngestService` orchestrator. Keeping
them in a single module avoids import cycles between the sub-stages: every
sub-stage imports its types from here and no sub-stage imports another.

The pipeline mirrors the upstream batch ingest contract: pull a batch of
per-document operations, then for each document run parse -> chunk -> embed
-> index, attach the wiki content passes (extraction, summary, chunk
citation, dedup, taxonomy planning) and finally materialise the generated
pages. The LLM-backed and broker-backed pieces are injected as protocol
seams so the worker layer can wire the real document parser, embedding
model, retrieval composite, and synthesis model without changing this
module — until those are wired, the pipeline degrades gracefully (an
absent seam skips the stage it feeds).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from src.ai.embedding import Context, Embedder
from src.common.json import SqlValue
from src.core.knowledge.documents.chunker import SplitterConfig
from src.db.models.chunk import Chunk
from src.db.models.knowledge import Document
from src.db.models.wiki_page import WikiPage, WikiPageLite

# ── Pipeline tunables (batch sizing / retry budgets) ──────────────────

# Caps the document content fed to the wiki content passes. Prevents a
# single oversized document from dominating the batch.
MAX_CONTENT_FOR_WIKI: int = 32768

# How many documents a single batch processes; remaining ops stay queued
# for the follow-up batch.
WIKI_MAX_DOCS_PER_BATCH: int = 5

# Maximum number of times a document op may fail before it is archived.
WIKI_MAX_FAIL_RETRIES: int = 5

# Follow-up debounce after a batch drains (short breather, not a
# lock-release wait).
WIKI_FOLLOW_UP_DELAY_SECONDS: int = 5

# Longer follow-up delay used when the batch tripped an upstream rate
# limit, so retries do not hammer an already-saturated budget.
WIKI_RATE_LIMIT_BACKOFF_SECONDS: int = 60

# ── Pending-op / task identifiers ─────────────────────────────────────

# Op kinds recorded in the pending queue.
WIKI_OP_INGEST: str = "ingest"
WIKI_OP_RETRACT: str = "retract"

# Task-type / scope stamps used by the durable pending queue and the
# dead-letter archive.
WIKI_TASK_TYPE: str = "wiki:ingest"
WIKI_TASK_SCOPE: str = "knowledge_base"

# ── Wiki page types / update kinds ────────────────────────────────────

WIKI_PAGE_TYPE_SUMMARY: str = "summary"
WIKI_PAGE_TYPE_ENTITY: str = "entity"
WIKI_PAGE_TYPE_CONCEPT: str = "concept"
WIKI_PAGE_TYPE_INDEX: str = "index"

# Update kinds emitted by the map phase. Entity / concept additions build
# or refresh a page; retract removes one document's contribution; a stale
# retract removes a page the document no longer produces.
WIKI_UPDATE_RETRACT: str = "retract"
WIKI_UPDATE_RETRACT_STALE: str = "retractStale"

# ── Extraction granularity (scoped per knowledge base) ────────────────

WIKI_EXTRACTION_STANDARD: str = "standard"
WIKI_EXTRACTION_GRANULAR: str = "granular"
WIKI_EXTRACTION_CONCISE: str = "concise"

# ── Retrieval bookkeeping ─────────────────────────────────────────────

# knowledge_type stamp written onto indexed chunks.
WIKI_KNOWLEDGE_TYPE: str = "wiki"

# Chunk retrieval-index lifecycle states the pipeline settles.
INDEX_STATUS_PROCESSING: str = "processing"
INDEX_STATUS_READY: str = "ready"
INDEX_STATUS_FAILED: str = "failed"


def chunk_embedding_text(chunk: Chunk) -> str:
    """The text an embedding model sees for a stored chunk row.

    Mirrors the chunker's embedding-content rule: the body with the
    tracked context header prepended when present.
    """
    body = chunk.content.strip()
    if chunk.context_header:
        return f"{chunk.context_header}\n\n{body}"
    return body


# ── Domain value types ────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class WikiIngestOp:
    """One queued wiki operation (the pending-op payload shape).

    ``op`` is ``ingest`` or ``retract``; ``row_id`` is the pending queue's
    row id (zero when the op was never persisted, e.g. in tests).
    """

    op: str = WIKI_OP_INGEST
    knowledge_id: str = ""
    language: str = ""
    doc_title: str = ""
    doc_summary: str = ""
    page_slugs: tuple[str, ...] = ()
    folder_ids: tuple[str, ...] = ()
    row_id: int = 0


@dataclass(frozen=True, slots=True)
class WikiExtractedItem:
    """One extracted entity or concept candidate.

    ``source_chunks`` holds the chunk ids that substantively support this
    item (populated by the chunk-citation pass); when non-empty the reduce
    stage quotes those chunks verbatim instead of the short description.
    """

    name: str
    slug: str
    aliases: tuple[str, ...] = ()
    description: str = ""
    details: str = ""
    source_chunks: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class NewSlugFromCitation:
    """An entry of the citation pass's ``new_slugs`` array.

    Mirrors :class:`WikiExtractedItem` but carries a ``type`` tag because
    the classifier emits entities and concepts in a single array. Chunk
    ids in ``source_chunks`` are still batch handles at this point and are
    translated to real ids by the citation stage.
    """

    type: str
    name: str
    slug: str
    aliases: tuple[str, ...] = ()
    description: str = ""
    details: str = ""
    source_chunks: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CitationBatchResult:
    """One classifier response for a chunk-citation batch.

    ``citations`` maps a slug to the batch chunk handles the model cited
    for it; ``new_slugs`` are candidates the classifier discovered that
    the extraction pass missed. Handles are translated to real chunk ids
    by :func:`src.core.knowledge.wiki.ingest_cite.classify_chunk_citations`.
    """

    citations: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    new_slugs: tuple[NewSlugFromCitation, ...] = ()


@dataclass(frozen=True, slots=True)
class WikiSlugUpdate:
    """A single update destined for one wiki page slug.

    ``type`` is ``entity`` / ``concept`` / ``summary`` for additions, or
    ``retract`` / ``retractStale`` for removals. ``item`` carries the
    extracted entity/concept for addition kinds; the summary and retract
    kinds carry their payload fields directly.
    """

    slug: str
    type: str
    item: WikiExtractedItem | None = None
    doc_title: str = ""
    knowledge_id: str = ""
    source_ref: str = ""
    language: str = ""
    summary_line: str = ""
    summary_body: str = ""
    retract_doc_content: str = ""
    source_chunks: tuple[str, ...] = ()
    doc_summary: str = ""


@dataclass(frozen=True, slots=True)
class WikiPageRef:
    """One page a document materialised, for link / retract bookkeeping."""

    slug: str
    title: str


@dataclass(frozen=True, slots=True)
class WikiDocIngestResult:
    """Per-document outcome of the map phase.

    ``pages`` records every page the document touched; ``summary`` is the
    one-line headline for log / audit previews.
    """

    knowledge_id: str
    doc_title: str
    summary: str = ""
    pages: tuple[WikiPageRef, ...] = ()


@dataclass(frozen=True, slots=True)
class WikiBatchContext:
    """Batch-scoped settings resolved once per batch.

    Extraction granularity and the knowledge-base instructions drive every
    document in the batch consistently. ``planned_folder_id`` is filled by
    the taxonomy stage before reduce runs and maps a page slug to the
    folder id the batch assigned it.
    """

    extraction_granularity: str = WIKI_EXTRACTION_STANDARD
    content_instructions: str = ""
    extraction_instructions: str = ""
    planned_folder_id: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class WikiBatchOutcome:
    """Aggregate result of one batch run."""

    pending_ops: int
    ingest_succeeded: int
    ingest_failed: int
    retract_handled: int
    pages_affected: int
    follow_up_scheduled: bool
    rate_limited: bool = False


# ── Storage seams ─────────────────────────────────────────────────────


class WikiPendingOpsStore(Protocol):
    """Durable per-document wiki queue (the pending-ops table shape).

    The batch driver peeks a window, settles consumed rows by id, and
    pushes in-batch failures through the fail-count budget. The worker
    layer wires the durable implementation; an in-memory store backs
    tests.
    """

    async def enqueue(self, *, tenant_id: int, knowledge_base_id: str, op: WikiIngestOp) -> bool:
        """Append ``op`` to the queue; return whether it was accepted."""

    async def peek(self, *, knowledge_base_id: str, limit: int) -> list[WikiIngestOp]:
        """Return up to ``limit`` queued ops for a knowledge base, FIFO."""

    async def delete_by_ids(self, ids: list[int]) -> None:
        """Remove consumed rows by their queue row ids."""

    async def increment_fail_count(self, op_id: int) -> int:
        """Bump a row's fail counter and return the new total."""

    async def release_by_ids(self, ids: list[int]) -> None:
        """Release claimed rows so a later batch can re-claim them."""

    async def pending_count(self, *, knowledge_base_id: str) -> int:
        """Return how many ops are still queued for a knowledge base."""

    async def archive(
        self,
        *,
        tenant_id: int,
        knowledge_base_id: str,
        op: WikiIngestOp,
        fail_count: int,
        last_error: str,
    ) -> None:
        """Move an op whose retries are exhausted into the dead-letter store."""


class DocumentStore(Protocol):
    """Knowledge-document row reads and status updates."""

    async def get_by_id(self, tenant_id: int, id: str) -> Document | None:
        """Return one live document within the tenant scope, or ``None``."""

    async def get_by_id_only(self, id: str) -> Document | None:
        """Return one live document by id without a tenant filter."""

    async def update_columns(self, id: str, values: Mapping[str, SqlValue]) -> Document | None:
        """Write several columns of one live row in a single statement."""


class DocumentParser(Protocol):
    """Document parsing seam (docreader-backed in the worker layer).

    Turns a stored document into the text content the chunker and the wiki
    content passes consume. The real implementation reads the file bytes
    through the document-reader service; until that is wired the pipeline
    skips documents whose parse stage cannot run.
    """

    async def parse_text(self, *, tenant_id: int, document: Document) -> str:
        """Return the extracted text of ``document``."""


class ChunkStore(Protocol):
    """``chunks``-table persistence used by the pipeline."""

    async def create_many(self, chunks: list[Chunk]) -> list[Chunk]:
        """Persist a batch of chunk rows, returning the stored rows."""

    async def list_by_knowledge_id(self, *, tenant_id: int, knowledge_id: str) -> list[Chunk]:
        """Return a knowledge item's text chunks in document order."""

    async def delete_by_knowledge_id(
        self, *, tenant_id: int, knowledge_id: str, now: datetime
    ) -> int:
        """Soft-delete every live chunk of a knowledge item; return the count."""

    async def update(self, row: Chunk) -> Chunk:
        """Overwrite every mutable column of a chunk row."""


class WikiIndexWriter(Protocol):
    """Retrieval-index write seam (the merged composite in the worker layer).

    ``write_chunks`` embeds-and-indexes a document's chunk rows;
    ``delete_by_source_id_list`` removes a document's indexed records. The
    default :class:`CompositeIndexWriter` in the service module wraps the
    composite retrieval engine directly.
    """

    async def write_chunks(
        self,
        *,
        ctx: Context,
        tenant_id: int,
        knowledge_base_id: str,
        knowledge_id: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        embedder: Embedder,
    ) -> None:
        """Persist the chunk embeddings into every registered index."""

    async def delete_by_source_id_list(
        self,
        *,
        ctx: Context,
        tenant_id: int,
        knowledge_base_id: str,
        source_id_list: list[str],
        dimension: int,
    ) -> None:
        """Remove indexed records by their source (chunk) ids."""


class WikiPageStore(Protocol):
    """Wiki page reads and writes used by the reduce stage.

    Structurally satisfied by the merged page service; the protocol keeps
    the pipeline testable with lightweight fakes.
    """

    async def get_page_by_slug(self, *, knowledge_base_id: str, slug: str) -> WikiPage | None:
        """Return one live page or ``None`` when absent."""

    async def create_page(self, *, page: WikiPage) -> WikiPage:
        """Create a new page and return the persisted row."""

    async def update_page(self, *, page: WikiPage) -> WikiPage:
        """Apply a user-visible edit to a page, bumping its version."""

    async def update_page_meta(self, *, page: WikiPage) -> WikiPage:
        """Persist page bookkeeping without a version bump."""

    async def delete_page(self, *, knowledge_base_id: str, slug: str) -> None:
        """Soft-delete a page and drop its inbound references."""

    async def list_slugs_by_source_ref(
        self, *, knowledge_base_id: str, source_knowledge_id: str
    ) -> list[str]:
        """Return the slugs of pages citing a source knowledge id."""

    async def list_summaries_by_knowledge_ids(
        self, *, knowledge_base_id: str, knowledge_ids: list[str]
    ) -> dict[str, str]:
        """Return summary content keyed by the knowledge id that authored it."""

    async def list_by_slugs(
        self, *, knowledge_base_id: str, slugs: list[str]
    ) -> dict[str, WikiPageLite]:
        """Resolve slugs to slim page projections in one IN query."""

    async def list_all_pages(self, *, knowledge_base_id: str) -> list[WikiPage]:
        """Return every live page in the knowledge base without pagination."""


class WikiFolderStore(Protocol):
    """Folder-tree reads and creates used by the taxonomy stage.

    Structurally satisfied by the merged folder service.
    """

    async def list_distinct_category_paths(
        self, *, knowledge_base_id: str, max_paths: int
    ) -> list[list[str]]:
        """Return the existing folder paths, each cleaned into segments."""

    async def find_or_create_folder_path(
        self, *, knowledge_base_id: str, tenant_id: int, path: list[str]
    ) -> tuple[str, list[str]]:
        """Resolve a category path to a leaf folder id, creating missing folders."""


# ── LLM-backed seams ──────────────────────────────────────────────────


class WikiExtractor(Protocol):
    """Candidate slug extraction (the pass-0 extractor).

    LLM-backed in production; returns the extracted entities and concepts
    for one document.
    """

    async def extract_candidate_slugs(
        self,
        *,
        content: str,
        language: str,
        previous_slugs: tuple[str, ...],
        granularity: str,
        extraction_instructions: str,
    ) -> tuple[list[WikiExtractedItem], list[WikiExtractedItem]]:
        """Return ``(entities, concepts)`` extracted from ``content``."""


class WikiSummarizer(Protocol):
    """Document summariser producing the summary page body.

    LLM-backed in production. Output is expected to carry an optional
    ``SUMMARY: <one-liner>`` headline line followed by the body.
    """

    async def summarize(
        self,
        *,
        content: str,
        language: str,
        extracted_slugs: tuple[str, ...],
        custom_instructions: str,
    ) -> str:
        """Return the summary text for one document."""


class ChunkCitationClassifier(Protocol):
    """Chunk-citation classifier: which chunks substantively support which slug.

    LLM-backed in production. The prompt is assembled by the citation
    stage (rendered XML + handles); the classifier returns slug -> cited
    chunk handles plus any newly discovered slugs.
    """

    async def classify_batch(
        self,
        *,
        candidates_xml: str,
        chunks_xml: str,
        language: str,
    ) -> CitationBatchResult:
        """Classify one chunk batch against the candidate slugs."""


class TaxonomyPlanner(Protocol):
    """Batch taxonomy planner: assign a directory path to every new page slug.

    LLM-backed in production; returns raw JSON text whose ``assignments``
    array maps each slug to a category path.
    """

    async def plan_assignments(
        self,
        *,
        existing_taxonomy: str,
        items_block: str,
        language: str,
    ) -> str:
        """Return the planning JSON for the given items."""


class DedupMerger(Protocol):
    """Merge-decision seam for the dedup stage.

    LLM-backed in production: given one new item and its candidate page
    slugs, return the target slug to merge into (or ``""`` to keep the item
    as a fresh page). Until wired, the pipeline keeps every item.
    """

    async def decide(
        self,
        *,
        item: WikiExtractedItem,
        candidate_slugs: list[str],
    ) -> str:
        """Return the target slug to merge ``item`` into, or ``""``."""


# ── Dependency bundle ─────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class WikiIngestDeps:
    """Every storage / seam dependency the pipeline stages consume.

    Built by :class:`WikiIngestService` (or the worker layer) and threaded
    into the sub-stage functions as a single immutable bundle. Optional
    seams are ``None`` until the worker wires them; the stages skip the
    corresponding work and degrade gracefully.
    """

    page_service: WikiPageStore
    folder_service: WikiFolderStore
    document_store: DocumentStore
    chunk_store: ChunkStore
    pending_store: WikiPendingOpsStore
    parser: DocumentParser | None = None
    embedder: Embedder | None = None
    index_writer: WikiIndexWriter | None = None
    extractor: WikiExtractor | None = None
    summarizer: WikiSummarizer | None = None
    classifier: ChunkCitationClassifier | None = None
    planner: TaxonomyPlanner | None = None
    merger: DedupMerger | None = None
    splitter_config: SplitterConfig | None = None


__all__ = [
    "INDEX_STATUS_FAILED",
    "INDEX_STATUS_PROCESSING",
    "INDEX_STATUS_READY",
    "MAX_CONTENT_FOR_WIKI",
    "WIKI_EXTRACTION_CONCISE",
    "WIKI_EXTRACTION_GRANULAR",
    "WIKI_EXTRACTION_STANDARD",
    "WIKI_FOLLOW_UP_DELAY_SECONDS",
    "WIKI_KNOWLEDGE_TYPE",
    "WIKI_MAX_DOCS_PER_BATCH",
    "WIKI_MAX_FAIL_RETRIES",
    "WIKI_OP_INGEST",
    "WIKI_OP_RETRACT",
    "WIKI_PAGE_TYPE_CONCEPT",
    "WIKI_PAGE_TYPE_ENTITY",
    "WIKI_PAGE_TYPE_INDEX",
    "WIKI_PAGE_TYPE_SUMMARY",
    "WIKI_RATE_LIMIT_BACKOFF_SECONDS",
    "WIKI_TASK_SCOPE",
    "WIKI_TASK_TYPE",
    "WIKI_UPDATE_RETRACT",
    "WIKI_UPDATE_RETRACT_STALE",
    "ChunkCitationClassifier",
    "ChunkStore",
    "CitationBatchResult",
    "DedupMerger",
    "DocumentParser",
    "DocumentStore",
    "NewSlugFromCitation",
    "TaxonomyPlanner",
    "WikiBatchContext",
    "WikiBatchOutcome",
    "WikiDocIngestResult",
    "WikiExtractedItem",
    "WikiExtractor",
    "WikiFolderStore",
    "WikiIndexWriter",
    "WikiIngestDeps",
    "WikiIngestOp",
    "WikiPageRef",
    "WikiPageStore",
    "WikiPendingOpsStore",
    "WikiSlugUpdate",
    "WikiSummarizer",
    "chunk_embedding_text",
]
