"""Unit + integration tests for the wiki ingest pipeline.

Unit tests drive the pure sub-stage functions (cite, dedup, taxonomy) and
the map / reduce / batch stages with lightweight in-memory fakes for every
storage seam (pytest, AAA). Integration tests run the batch driver against
the real applied schema (revision 0022+): real chunk / wiki page / folder /
knowledge repositories with the parser / embedding / index / synthesis
seams replaced by fakes. They are skipped when Postgres is not reachable
(set ``DATABASE_URL_OVERRIDE``).
"""

from __future__ import annotations

import json
import re
import secrets
import uuid
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.pool import NullPool

from src.ai.embedding import TaskContext
from src.core.knowledge.documents.types import (
    PARSE_STATUS_COMPLETED,
    PARSE_STATUS_FAILED,
    PARSE_STATUS_PENDING,
    PARSE_STATUS_PROCESSING,
)
from src.core.knowledge.wiki.ingest_batch import (
    has_sufficient_text_content,
    manual_document_content,
    map_one_document,
    process_wiki_ingest_batch,
    real_text_runes,
    reduce_slug_updates,
    slugify,
    split_summary_line,
    strip_image_markup,
)
from src.core.knowledge.wiki.ingest_cite import (
    MAX_RUNES_PER_CITATION_BATCH,
    CitationChunkBatch,
    classify_chunk_citations,
    collect_cited_chunk_content,
    merge_citations_into_items,
    render_candidate_slugs_xml,
    render_chunks_xml,
    split_chunks_into_citation_batches,
)
from src.core.knowledge.wiki.ingest_dedup import (
    DedupSurface,
    dedup_merge_reject_reason,
    dedup_pair_score,
    deduplicate_items,
    grams_per_surface,
    jaccard,
    select_dedup_candidate_pages,
    slug_base_tokens,
    surface_grams,
)
from src.core.knowledge.wiki.ingest_service import (
    CompositeIndexWriter,
    WikiIngestService,
)
from src.core.knowledge.wiki.ingest_taxonomy import (
    WIKI_TAXONOMY_PLAN_CHUNK_SIZE,
    cap_folders,
    collect_taxonomy_items,
    cosine_similarity,
    format_existing_taxonomy_for_prompt,
    parse_taxonomy_assignments,
    plan_batch_taxonomy,
    resolve_planned_folders,
    select_folders_by_vectors,
)
from src.core.knowledge.wiki.ingest_types import (
    INDEX_STATUS_PROCESSING,
    INDEX_STATUS_READY,
    WIKI_OP_INGEST,
    WIKI_OP_RETRACT,
    WIKI_PAGE_TYPE_CONCEPT,
    WIKI_PAGE_TYPE_ENTITY,
    WIKI_PAGE_TYPE_SUMMARY,
    WIKI_UPDATE_RETRACT,
    CitationBatchResult,
    NewSlugFromCitation,
    WikiBatchContext,
    WikiIngestDeps,
    WikiIngestOp,
    WikiSlugUpdate,
    chunk_embedding_text,
)
from src.db.base import DatabaseEngine
from src.db.dao.chunk_repository import ChunkRepository
from src.db.dao.knowledge_repository import KnowledgeRepository
from src.db.dao.wiki_page_repository import WikiFolderRepository, WikiPageRepository
from src.db.models.chunk import Chunk
from src.db.models.knowledge import Document
from src.db.models.wiki_page import WikiPage, WikiPageLite
from src.settings import get_settings, reset_settings_cache

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_CTX = TaskContext(is_background_task=True)

# ── Shared helpers ────────────────────────────────────────────────────


def _kid(prefix: str = "kid") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _kb() -> str:
    return f"kb-{uuid.uuid4().hex[:8]}"


def _doc(
    *,
    tenant_id: int,
    knowledge_base_id: str,
    id: str,
    title: str = "Test doc",
    content: str = "",
    parse_status: str = PARSE_STATUS_PENDING,
    pending_subtasks_count: int = 0,
) -> Document:
    metadata = {"content": content} if content else None
    return Document(
        id=id,
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        type="manual",
        title=title,
        description=None,
        source="manual",
        channel="web",
        parse_status=parse_status,
        pending_subtasks_count=pending_subtasks_count,
        summary_status="none",
        enable_status="enabled",
        embedding_model_id=None,
        file_name=None,
        file_type=None,
        file_size=None,
        file_hash=None,
        file_path=None,
        storage_size=0,
        metadata=metadata,
        custom_metadata={},
        last_faq_import_result=None,
        created_at=_NOW,
        updated_at=_NOW,
        processed_at=None,
        error_message=None,
        deleted_at=None,
    )


def _chunk_row(
    *,
    tenant_id: int,
    knowledge_base_id: str,
    knowledge_id: str,
    content: str,
    chunk_index: int = 0,
) -> Chunk:
    return Chunk(
        id=f"chunk-{uuid.uuid4().hex[:12]}",
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        knowledge_id=knowledge_id,
        content=content,
        chunk_index=chunk_index,
        is_enabled=True,
        start_at=0,
        end_at=len(content),
        chunk_type="text",
        status=0,
        index_status=INDEX_STATUS_PROCESSING,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _entity(name: str, slug: str, description: str = "desc", aliases: tuple[str, ...] = ()):
    from src.core.knowledge.wiki.ingest_types import WikiExtractedItem

    return WikiExtractedItem(name=name, slug=slug, description=description, aliases=aliases)


# ── In-memory storage fakes ───────────────────────────────────────────


class FakePendingStore:
    """In-memory pending-ops queue satisfying the store protocol."""

    def __init__(self) -> None:
        self._ops: dict[int, WikiIngestOp] = {}
        self._fails: dict[int, int] = {}
        self._next_id = 1
        self.archived: list[WikiIngestOp] = []
        self.archived_errors: list[str] = []

    async def enqueue(self, *, tenant_id: int, knowledge_base_id: str, op: WikiIngestOp) -> bool:
        self._ops[self._next_id] = replace(op, row_id=self._next_id)
        self._next_id += 1
        return True

    async def peek(self, *, knowledge_base_id: str, limit: int) -> list[WikiIngestOp]:
        return list(self._ops.values())[:limit]

    async def delete_by_ids(self, ids: list[int]) -> None:
        for row_id in ids:
            self._ops.pop(row_id, None)
            self._fails.pop(row_id, None)

    async def increment_fail_count(self, op_id: int) -> int:
        count = self._fails.get(op_id, 0) + 1
        self._fails[op_id] = count
        return count

    async def release_by_ids(self, ids: list[int]) -> None:
        return None

    async def pending_count(self, *, knowledge_base_id: str) -> int:
        return len(self._ops)

    async def archive(
        self,
        *,
        tenant_id: int,
        knowledge_base_id: str,
        op: WikiIngestOp,
        fail_count: int,
        last_error: str,
    ) -> None:
        self.archived.append(op)
        self.archived_errors.append(last_error)


class FakeDocumentStore:
    """In-memory document store tracking column updates."""

    def __init__(self) -> None:
        self._docs: dict[str, Document] = {}
        self.updates: list[tuple[str, dict[str, object]]] = []

    def seed(self, document: Document) -> None:
        self._docs[document.id] = document

    async def get_by_id(self, tenant_id: int, id: str) -> Document | None:
        doc = self._docs.get(id)
        if doc is None or doc.tenant_id != tenant_id:
            return None
        return doc

    async def get_by_id_only(self, id: str) -> Document | None:
        return self._docs.get(id)

    async def update_columns(self, id: str, values: dict[str, object]) -> Document | None:
        doc = self._docs.get(id)
        if doc is None:
            return None
        updated = doc.model_copy(update=dict(values))
        self._docs[id] = updated
        self.updates.append((id, dict(values)))
        return updated


class FakeChunkStore:
    """In-memory chunk store that honours soft deletes."""

    def __init__(self) -> None:
        self._rows: list[Chunk] = []

    async def create_many(self, chunks: list[Chunk]) -> list[Chunk]:
        self._rows.extend(chunks)
        return chunks

    async def list_by_knowledge_id(self, *, tenant_id: int, knowledge_id: str) -> list[Chunk]:
        return [
            row
            for row in self._rows
            if row.tenant_id == tenant_id
            and row.knowledge_id == knowledge_id
            and row.deleted_at is None
        ]

    async def delete_by_knowledge_id(
        self, *, tenant_id: int, knowledge_id: str, now: datetime
    ) -> int:
        count = 0
        remaining: list[Chunk] = []
        for row in self._rows:
            if row.tenant_id == tenant_id and row.knowledge_id == knowledge_id:
                count += 1
                remaining.append(row.model_copy(update={"deleted_at": now}))
            else:
                remaining.append(row)
        self._rows = remaining
        return count

    async def update(self, row: Chunk) -> Chunk:
        for i, existing in enumerate(self._rows):
            if existing.id == row.id:
                self._rows[i] = row
                break
        return row


class FakePageStore:
    """In-memory wiki page store backed by a slug-keyed map."""

    def __init__(self) -> None:
        self.pages: dict[str, WikiPage] = {}
        self.deleted: list[str] = []

    def seed(self, page: WikiPage) -> None:
        self.pages[page.slug] = page

    async def get_page_by_slug(self, *, knowledge_base_id: str, slug: str) -> WikiPage | None:
        return self.pages.get(slug)

    async def create_page(self, *, page: WikiPage) -> WikiPage:
        self.pages[page.slug] = page
        return page

    async def update_page(self, *, page: WikiPage) -> WikiPage:
        self.pages[page.slug] = page
        return page

    async def update_page_meta(self, *, page: WikiPage) -> WikiPage:
        self.pages[page.slug] = page
        return page

    async def delete_page(self, *, knowledge_base_id: str, slug: str) -> None:
        self.deleted.append(slug)
        self.pages.pop(slug, None)

    async def list_slugs_by_source_ref(
        self, *, knowledge_base_id: str, source_knowledge_id: str
    ) -> list[str]:
        return [
            page.slug
            for page in self.pages.values()
            if any(_ref_kid(ref) == source_knowledge_id for ref in page.source_refs)
        ]

    async def list_summaries_by_knowledge_ids(
        self, *, knowledge_base_id: str, knowledge_ids: list[str]
    ) -> dict[str, str]:
        out: dict[str, str] = {}
        for page in self.pages.values():
            if page.page_type != WIKI_PAGE_TYPE_SUMMARY:
                continue
            for kid in knowledge_ids:
                if any(_ref_kid(ref) == kid for ref in page.source_refs):
                    out[kid] = page.content
        return out

    async def list_by_slugs(
        self, *, knowledge_base_id: str, slugs: list[str]
    ) -> dict[str, WikiPageLite]:
        out: dict[str, WikiPageLite] = {}
        for slug in slugs:
            page = self.pages.get(slug)
            if page is None:
                continue
            out[slug] = WikiPageLite(
                slug=page.slug,
                title=page.title,
                page_type=page.page_type,
                status=page.status,
                aliases=list(page.aliases),
                out_links=list(page.out_links),
            )
        return out

    async def list_all_pages(self, *, knowledge_base_id: str) -> list[WikiPage]:
        return list(self.pages.values())


def _ref_kid(ref: str) -> str:
    return ref.split("|", 1)[0]


class FakeFolderStore:
    """In-memory folder store resolving paths to stable synthetic ids."""

    def __init__(self) -> None:
        self.paths: dict[tuple[str, ...], str] = {}
        self._next = 0

    async def list_distinct_category_paths(
        self, *, knowledge_base_id: str, max_paths: int
    ) -> list[list[str]]:
        return [list(path) for path in self.paths]

    async def find_or_create_folder_path(
        self, *, knowledge_base_id: str, tenant_id: int, path: list[str]
    ) -> tuple[str, list[str]]:
        key = tuple(path)
        folder_id = self.paths.get(key)
        if folder_id is None:
            self._next += 1
            folder_id = f"folder-{self._next}"
            self.paths[key] = folder_id
        return folder_id, list(path)


# ── Seam fakes ────────────────────────────────────────────────────────


class FakeParser:
    def __init__(self, text: str = "") -> None:
        self._text = text

    async def parse_text(self, *, tenant_id: int, document: Document) -> str:
        return self._text


class FakeEmbedder:
    def __init__(self, dimensions: int = 3) -> None:
        self._dimensions = dimensions
        self.calls: list[list[str]] = []

    async def batch_embed_with_pool(
        self, ctx: object, model: object, texts: list[str]
    ) -> list[list[float]]:
        self.calls.append(texts)
        return [[0.1 * (i + 1)] * self._dimensions for i in range(len(texts))]

    async def embed(self, ctx: object, text: str) -> list[float]:
        return [0.1] * self._dimensions

    async def batch_embed(self, ctx: object, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self._dimensions for _ in texts]

    def get_model_name(self) -> str:
        return "fake-embedder"

    def get_dimensions(self) -> int:
        return self._dimensions

    def get_model_id(self) -> str:
        return "fake-embedder"


class FakeIndexWriter:
    def __init__(self) -> None:
        self.writes: list[dict[str, object]] = []
        self.deletes: list[list[str]] = []
        self.fail = False

    async def write_chunks(
        self,
        *,
        ctx: object,
        tenant_id: int,
        knowledge_base_id: str,
        knowledge_id: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        embedder: object,
    ) -> None:
        if self.fail:
            raise RuntimeError("index write failed")
        self.writes.append(
            {
                "knowledge_base_id": knowledge_base_id,
                "knowledge_id": knowledge_id,
                "chunks": chunks,
                "embeddings": embeddings,
            }
        )

    async def delete_by_source_id_list(
        self,
        *,
        ctx: object,
        tenant_id: int,
        knowledge_base_id: str,
        source_id_list: list[str],
        dimension: int,
    ) -> None:
        self.deletes.append(source_id_list)


class FakeExtractor:
    def __init__(self, entities: list = (), concepts: list = ()) -> None:
        self.entities = list(entities)
        self.concepts = list(concepts)
        self.calls: list[dict[str, object]] = []

    async def extract_candidate_slugs(
        self,
        *,
        content: str,
        language: str,
        previous_slugs: tuple[str, ...],
        granularity: str,
        extraction_instructions: str,
    ) -> tuple[list, list]:
        self.calls.append({"language": language, "previous_slugs": previous_slugs})
        return self.entities, self.concepts


class FakeSummarizer:
    def __init__(self, text: str = "SUMMARY: A test document\n\nBody summary.") -> None:
        self._text = text
        self.fail = False
        self.calls: list[dict[str, object]] = []

    async def summarize(
        self,
        *,
        content: str,
        language: str,
        extracted_slugs: tuple[str, ...],
        custom_instructions: str,
    ) -> str:
        self.calls.append({"extracted_slugs": extracted_slugs})
        if self.fail:
            raise RuntimeError("summary failed")
        return self._text


class FakeClassifier:
    """Classifier that cites chunk handles by index parsed from the prompt XML."""

    def __init__(self, slug_to_indexes: dict[str, list[int]] | None = None) -> None:
        self._slug_indexes = slug_to_indexes or {}
        self.calls: list[str] = []

    async def classify_batch(
        self, *, candidates_xml: str, chunks_xml: str, language: str
    ) -> CitationBatchResult:
        self.calls.append(chunks_xml)
        handle_by_index: dict[int, str] = {}
        for match in re.finditer(r"<c id='(c\d+)' index=\"(\d+)\">", chunks_xml):
            handle_by_index[int(match.group(2))] = match.group(1)
        citations = {
            slug: [handle_by_index[i] for i in indexes if i in handle_by_index]
            for slug, indexes in self._slug_indexes.items()
        }
        return CitationBatchResult(citations=citations, new_slugs=())


class FakePlanner:
    def __init__(self, assignments: dict[str, list[str]] | None = None) -> None:
        self._assignments = assignments or {}
        self.calls: list[dict[str, object]] = []

    async def plan_assignments(
        self, *, existing_taxonomy: str, items_block: str, language: str
    ) -> str:
        self.calls.append({"existing_taxonomy": existing_taxonomy})
        payload = {
            "assignments": [
                {"slug": slug, "path": path} for slug, path in self._assignments.items()
            ]
        }
        return json.dumps(payload)


class FakeMerger:
    def __init__(self, decisions: dict[str, str] | None = None) -> None:
        self._decisions = decisions or {}
        self.calls: list[list[str]] = []

    async def decide(self, *, item, candidate_slugs: list[str]) -> str:
        self.calls.append(candidate_slugs)
        return self._decisions.get(item.slug, "")


def _deps(
    *,
    parser: FakeParser | None = None,
    embedder: FakeEmbedder | None = None,
    index_writer: FakeIndexWriter | None = None,
    extractor: FakeExtractor | None = None,
    summarizer: FakeSummarizer | None = None,
    classifier: FakeClassifier | None = None,
    planner: FakePlanner | None = None,
    merger: FakeMerger | None = None,
    page_store: FakePageStore | None = None,
    folder_store: FakeFolderStore | None = None,
    document_store: FakeDocumentStore | None = None,
    chunk_store: FakeChunkStore | None = None,
    pending_store: FakePendingStore | None = None,
) -> WikiIngestDeps:
    return WikiIngestDeps(
        page_service=page_store or FakePageStore(),
        folder_service=folder_store or FakeFolderStore(),
        document_store=document_store or FakeDocumentStore(),
        chunk_store=chunk_store or FakeChunkStore(),
        pending_store=pending_store or FakePendingStore(),
        parser=parser,
        embedder=embedder,
        index_writer=index_writer,
        extractor=extractor,
        summarizer=summarizer,
        classifier=classifier,
        planner=planner,
        merger=merger,
    )


# ── Pure ingest-batch helpers ─────────────────────────────────────────


class TestIngestHelpers:
    def test_slugify_lowercases_and_dashes(self) -> None:
        assert slugify("Hello World!") == "hello-world"

    def test_slugify_preserves_uuid_hyphens(self) -> None:
        raw = "AbC-1234-xYz"
        assert slugify(raw) == "abc-1234-xyz"

    def test_split_summary_line_parses_headline(self) -> None:
        summary, body = split_summary_line("SUMMARY: One-liner\n\nFull body text.")
        assert summary == "One-liner"
        assert body == "Full body text."

    def test_split_summary_line_fullwidth_colon(self) -> None:
        summary, body = split_summary_line("SUMMARY：One-liner\n\nBody.")
        assert summary == "One-liner"
        assert body == "Body."

    def test_split_summary_line_without_headline_keeps_content(self) -> None:
        summary, body = split_summary_line("Just some prose.")
        assert summary == ""
        assert body == "Just some prose."

    def test_strip_image_markup_removes_markdown_and_html(self) -> None:
        raw = "Text ![alt](img.png) more <img src='x.png'> tail"
        assert strip_image_markup(raw) == "Text  more  tail"

    def test_real_text_runes_counts_only_real_text(self) -> None:
        assert real_text_runes("   ![a](b.png)   ") == 0
        assert real_text_runes("  hello  ") == 5

    def test_has_sufficient_text_content_threshold(self) -> None:
        assert not has_sufficient_text_content("![a](b.png)")
        assert has_sufficient_text_content("hello world!")

    def test_manual_document_content_reads_metadata(self) -> None:
        document = _doc(tenant_id=1, knowledge_base_id="kb", id="d1", content="body")
        assert manual_document_content(document) == "body"

    def test_manual_document_content_empty_without_metadata(self) -> None:
        document = _doc(tenant_id=1, knowledge_base_id="kb", id="d1")
        assert manual_document_content(document) == ""

    def test_chunk_embedding_text_prepends_context_header(self) -> None:
        row = _chunk_row(tenant_id=1, knowledge_base_id="kb", knowledge_id="k", content="body")
        assert chunk_embedding_text(row) == "body"
        with_header = row.model_copy(update={"context_header": "# Section"})
        assert chunk_embedding_text(with_header) == "# Section\n\nbody"


# ── Citation sub-stage ────────────────────────────────────────────────


class TestCitationBatching:
    def test_split_chunks_respects_rune_budget(self) -> None:
        rows = [
            _chunk_row(
                tenant_id=1,
                knowledge_base_id="kb",
                knowledge_id="k",
                content=f"c{i} " + "x" * 1500,
                chunk_index=i,
            )
            for i in range(10)
        ]
        batches = split_chunks_into_citation_batches(rows)
        assert len(batches) > 1
        assert all(
            sum(len(chunk.content) for chunk in batch.chunks) <= MAX_RUNES_PER_CITATION_BATCH
            for batch in batches
        )
        # all chunks preserved, in document order
        flattened = [chunk.id for batch in batches for chunk in batch.chunks]
        assert flattened == [row.id for row in rows]

    def test_split_chunks_filters_non_text_and_empty(self) -> None:
        text_row = _chunk_row(tenant_id=1, knowledge_base_id="kb", knowledge_id="k", content="a")
        image_row = text_row.model_copy(update={"id": "img", "chunk_type": "image", "content": "b"})
        empty_row = text_row.model_copy(update={"id": "empty", "content": ""})
        batches = split_chunks_into_citation_batches([image_row, empty_row, text_row])
        assert len(batches) == 1
        assert batches[0].chunks == (text_row,)

    def test_split_chunks_empty_input(self) -> None:
        assert split_chunks_into_citation_batches([]) == []


class TestCitationRendering:
    def test_render_candidate_slugs_xml(self) -> None:
        entity = _entity("Acme", "entity/acme", "A company")
        concept = _entity("RAG", "concept/rag", "Retrieval", aliases=("RAGs",))
        xml = render_candidate_slugs_xml([entity], [concept])
        assert "- slug: entity/acme, type: entity, name: 'Acme'" in xml
        assert "aliases=('RAGs',)" in xml
        assert "- slug: concept/rag, type: concept" in xml

    def test_render_candidate_slugs_xml_skips_blank(self) -> None:
        assert render_candidate_slugs_xml([_entity("", "entity/x")], []) == ""

    def test_render_chunks_xml_uses_handles(self) -> None:
        from src.core.knowledge.wiki.slug_utils import HandleTable

        row = _chunk_row(tenant_id=1, knowledge_base_id="kb", knowledge_id="k", content="hi")
        table = HandleTable("c", 0, 1)
        table.register(row.id)
        batch = CitationChunkBatch((row,), table)
        xml = render_chunks_xml(batch)
        assert "c0" in xml
        assert row.id not in xml


class TestCitationClassification:
    async def test_classify_translates_handles_to_real_ids(self) -> None:
        rows = [
            _chunk_row(
                tenant_id=1,
                knowledge_base_id="kb",
                knowledge_id="k",
                content=f"text {i}",
                chunk_index=i,
            )
            for i in range(3)
        ]
        classifier = FakeClassifier(slug_to_indexes={"entity/acme": [0, 2]})
        citations, new_slugs, batch_count = await classify_chunk_citations(
            _CTX,
            classifier=classifier,
            candidates_xml="- slug: entity/acme, type: entity, name: 'Acme'",
            chunks=rows,
            language="en",
        )
        assert citations["entity/acme"] == [rows[0].id, rows[2].id]
        assert new_slugs == []
        assert batch_count == 1

    async def test_classify_returns_empty_without_candidates(self) -> None:
        rows = [_chunk_row(tenant_id=1, knowledge_base_id="kb", knowledge_id="k", content="x")]
        citations, new_slugs, count = await classify_chunk_citations(
            _CTX,
            classifier=FakeClassifier(),
            candidates_xml="",
            chunks=rows,
            language="en",
        )
        assert citations == {}
        assert new_slugs == []
        assert count == 0


class TestCitationMerge:
    def test_merge_backfills_source_chunks(self) -> None:
        entities = [_entity("Acme", "entity/acme")]
        concepts = [_entity("RAG", "concept/rag")]
        entities, concepts, uncited = merge_citations_into_items(
            entities, concepts, {"entity/acme": ["c1", "c2"]}, []
        )
        assert entities[0].source_chunks == ("c1", "c2")
        assert concepts[0].source_chunks == ()
        assert uncited == 1

    def test_merge_appends_new_slugs(self) -> None:
        new = NewSlugFromCitation(
            type="entity",
            name="NewCo",
            slug="entity/newco",
            source_chunks=("c9",),
        )
        entities, concepts, _ = merge_citations_into_items([], [], {}, [new])
        assert [item.slug for item in entities] == ["entity/newco"]
        assert concepts == []

    def test_merge_skips_existing_slugs_in_new_list(self) -> None:
        existing = _entity("Acme", "entity/acme")
        new = NewSlugFromCitation(type="entity", name="Acme", slug="entity/acme")
        entities, _, _ = merge_citations_into_items([existing], [], {}, [new])
        assert len(entities) == 1

    def test_collect_cited_chunk_content(self) -> None:
        content = collect_cited_chunk_content(["c1", "missing", "c2"], {"c1": "a", "c2": "b"})
        assert content == "a\n\nb"
        assert collect_cited_chunk_content([], {}) == ""


# ── Dedup sub-stage ───────────────────────────────────────────────────


class TestDedupSimilarity:
    def test_jaccard(self) -> None:
        assert jaccard(frozenset({"a", "b"}), frozenset({"b", "c"})) == pytest.approx(1 / 3)
        assert jaccard(frozenset(), frozenset()) == 0
        assert jaccard(frozenset({"a"}), frozenset()) == 0

    def test_slug_base_tokens(self) -> None:
        tokens = slug_base_tokens("entity/beijing-nongshang-yinxing")
        assert tokens == frozenset({"beijing", "nongshang", "yinxing"})
        assert slug_base_tokens("") == frozenset()

    def test_surface_grams(self) -> None:
        grams = surface_grams("Acme Corp")
        assert "ac" in grams and "me" in grams
        assert surface_grams("") == frozenset()

    def test_dedup_pair_score_is_max_over_surfaces(self) -> None:
        a = DedupSurface(slug_tokens=frozenset(), name_gram_sets=grams_per_surface(["Acme Corp"]))
        b = DedupSurface(
            slug_tokens=frozenset(), name_gram_sets=grams_per_surface(["Acme Corporation"])
        )
        assert 0 < dedup_pair_score(a, b) < 1


class TestDedupCandidateSelection:
    def test_filters_to_entity_concept_pages(self) -> None:
        pages = [
            _page("entity/acme", WIKI_PAGE_TYPE_ENTITY, "Acme"),
            _page("summary/x", WIKI_PAGE_TYPE_SUMMARY, "Sum"),
            _page("concept/rag", WIKI_PAGE_TYPE_CONCEPT, "RAG"),
        ]
        candidates = select_dedup_candidate_pages([_entity("Acme", "entity/acme")], pages)
        assert {p.slug for p in candidates} == {"entity/acme", "concept/rag"}

    def test_small_corpus_bypass_keeps_all_entity_pages(self) -> None:
        pages = [_page(f"entity/p{i}", WIKI_PAGE_TYPE_ENTITY, f"Page {i}") for i in range(5)]
        candidates = select_dedup_candidate_pages([_entity("zzz", "entity/zzz")], pages)
        assert len(candidates) == len(pages)

    def test_similarity_selection_keeps_related_pages(self) -> None:
        pages = [_page(f"entity/p{i}", WIKI_PAGE_TYPE_ENTITY, f"Page {i}") for i in range(40)]
        # An item that shares bigrams with exactly one page's title.
        pages[0] = _page("entity/acme", WIKI_PAGE_TYPE_ENTITY, "Acme Corporation")
        candidates = select_dedup_candidate_pages([_entity("Acme Corp", "entity/acme")], pages)
        assert any(p.slug == "entity/acme" for p in candidates)

    def test_dedup_merge_reject_reason(self) -> None:
        assert dedup_merge_reject_reason("entity/a", "entity/b", {"entity/b"}) == ""
        assert dedup_merge_reject_reason("entity/a", "entity/b", {"entity/c"}) != ""
        assert "missing type prefix" in dedup_merge_reject_reason("a", "entity/b", {"entity/b"})
        assert "type mismatch" in dedup_merge_reject_reason("entity/a", "concept/b", {"concept/b"})

    async def test_deduplicate_items_with_merger(self) -> None:
        item = _entity("Acme Corp", "entity/acme")
        target = _page("entity/acme", WIKI_PAGE_TYPE_ENTITY, "Acme Corporation")
        merger = FakeMerger(decisions={"entity/acme": "entity/acme"})
        kept, merged = await deduplicate_items([item], [target], merger)
        assert kept == []
        assert merged == {"entity/acme": "entity/acme"}

    async def test_deduplicate_items_without_merger_keeps_all(self) -> None:
        item = _entity("Acme", "entity/acme")
        kept, merged = await deduplicate_items([item], [], None)
        assert kept == [item]
        assert merged == {}


def _page(slug: str, page_type: str, title: str) -> WikiPage:
    return WikiPage(
        id=f"page-{uuid.uuid4().hex[:8]}",
        tenant_id=1,
        knowledge_base_id="kb",
        slug=slug,
        title=title,
        page_type=page_type,
        status="published",
        source_refs=[],
        aliases=[],
        created_at=_NOW,
        updated_at=_NOW,
    )


# ── Taxonomy sub-stage ────────────────────────────────────────────────


class TestTaxonomyPure:
    def test_collect_taxonomy_items(self) -> None:
        updates: dict[str, list[WikiSlugUpdate]] = {
            "entity/a": [
                WikiSlugUpdate(
                    slug="entity/a",
                    type=WIKI_PAGE_TYPE_ENTITY,
                    item=_entity("A", "entity/a", "about A"),
                )
            ],
            "summary/s": [
                WikiSlugUpdate(slug="summary/s", type=WIKI_PAGE_TYPE_SUMMARY, summary_body="x")
            ],
        }
        items = collect_taxonomy_items(updates)
        assert [(item.slug, item.about) for item in items] == [("entity/a", "about A")]

    def test_format_existing_taxonomy_for_prompt(self) -> None:
        tree = format_existing_taxonomy_for_prompt([["AI", "RAG"], ["AI", "Agents"], ["HR"]])
        assert "AI" in tree and "RAG" in tree and "Agents" in tree and "HR" in tree
        assert format_existing_taxonomy_for_prompt([]) == ""

    def test_parse_taxonomy_assignments(self) -> None:
        raw = '{"assignments": [{"slug": "entity/a", "path": ["AI", "RAG"]}]}'
        assert parse_taxonomy_assignments(raw) == {"entity/a": ["AI", "RAG"]}
        assert parse_taxonomy_assignments("not json") == {}

    def test_cosine_similarity(self) -> None:
        assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
        assert cosine_similarity([], []) == 0.0

    def test_select_folders_by_vectors(self) -> None:
        deeper = [["AI"], ["HR"], ["Ops"]]
        folder_vecs = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        item_vecs = [[1.0, 0.0]]
        selected = select_folders_by_vectors(deeper, folder_vecs, item_vecs, 2)
        assert selected == [["AI"], ["Ops"]]

    def test_cap_folders(self) -> None:
        paths = [[str(i)] for i in range(5)]
        assert cap_folders(paths, 2) == [["0"], ["1"]]
        assert cap_folders(paths, 0) == paths


class TestTaxonomyPlanning:
    async def test_resolve_planned_folders(self) -> None:
        store = FakeFolderStore()
        resolved = await resolve_planned_folders(
            folder_service=store,
            knowledge_base_id="kb",
            tenant_id=1,
            planned={"entity/a": ["AI", "RAG"], "entity/b": [], "entity/c": ["AI", "RAG"]},
        )
        assert resolved["entity/a"] == resolved["entity/c"]
        assert "entity/b" not in resolved

    async def test_plan_batch_taxonomy_with_planner(self) -> None:
        store = FakeFolderStore()
        await store.find_or_create_folder_path(knowledge_base_id="kb", tenant_id=1, path=["AI"])
        updates = {
            "entity/a": [
                WikiSlugUpdate(
                    slug="entity/a",
                    type=WIKI_PAGE_TYPE_ENTITY,
                    item=_entity("A", "entity/a", "about A"),
                )
            ]
        }
        planner = FakePlanner(assignments={"entity/a": ["AI", "RAG"]})
        planned = await plan_batch_taxonomy(
            _CTX,
            folder_service=store,
            knowledge_base_id="kb",
            language="en",
            slug_updates=updates,
            planner=planner,
            embedder=None,
        )
        assert planned["entity/a"] == ["AI", "RAG"]
        assert planner.calls[0]["existing_taxonomy"] == "AI"

    async def test_plan_batch_taxonomy_without_planner_is_noop(self) -> None:
        store = FakeFolderStore()
        updates = {
            "entity/a": [
                WikiSlugUpdate(
                    slug="entity/a",
                    type=WIKI_PAGE_TYPE_ENTITY,
                    item=_entity("A", "entity/a", "about A"),
                )
            ]
        }
        planned = await plan_batch_taxonomy(
            _CTX,
            folder_service=store,
            knowledge_base_id="kb",
            language="en",
            slug_updates=updates,
            planner=None,
        )
        assert planned == {}

    def test_plan_chunk_size_constant_sane(self) -> None:
        assert WIKI_TAXONOMY_PLAN_CHUNK_SIZE == 60


# ── Map phase ─────────────────────────────────────────────────────────


class TestMapOneDocument:
    async def test_retract_op_builds_retract_updates(self) -> None:
        page_store = FakePageStore()
        page = _page("entity/a", WIKI_PAGE_TYPE_ENTITY, "A")
        page = page.model_copy(update={"source_refs": ["kid-1|A"]})
        page_store.seed(page)
        deps = _deps(page_store=page_store)
        op = WikiIngestOp(
            op=WIKI_OP_RETRACT,
            knowledge_id="kid-1",
            doc_title="Doc",
            page_slugs=("entity/a",),
            language="en",
        )
        result, updates = await map_one_document(
            _CTX,
            deps=deps,
            tenant_id=1,
            knowledge_base_id="kb",
            op=op,
            batch_ctx=WikiBatchContext(),
            existing_pages=[],
        )
        assert result is None
        assert [(u.slug, u.type) for u in updates] == [("entity/a", WIKI_UPDATE_RETRACT)]

    async def test_gone_document_skips(self) -> None:
        deps = _deps()
        op = WikiIngestOp(op=WIKI_OP_INGEST, knowledge_id="missing", language="en")
        result, updates = await map_one_document(
            _CTX,
            deps=deps,
            tenant_id=1,
            knowledge_base_id="kb",
            op=op,
            batch_ctx=WikiBatchContext(),
            existing_pages=[],
        )
        assert result is None
        assert updates == []

    async def test_insufficient_text_skips(self) -> None:
        document_store = FakeDocumentStore()
        document_store.seed(
            _doc(tenant_id=1, knowledge_base_id="kb", id="d1", content="![a](b.png)")
        )
        deps = _deps(document_store=document_store, parser=FakeParser("![a](b.png)"))
        op = WikiIngestOp(op=WIKI_OP_INGEST, knowledge_id="d1", language="en")
        result, updates = await map_one_document(
            _CTX,
            deps=deps,
            tenant_id=1,
            knowledge_base_id="kb",
            op=op,
            batch_ctx=WikiBatchContext(),
            existing_pages=[],
        )
        assert result is None
        assert updates == []

    async def test_full_ingest_pipeline(self) -> None:
        tenant_id = 1
        kb_id = _kb()
        document_store = FakeDocumentStore()
        document_store.seed(
            _doc(
                tenant_id=tenant_id,
                knowledge_base_id=kb_id,
                id="d1",
                content="Acme makes widgets.",
            )
        )
        chunk_store = FakeChunkStore()
        index_writer = FakeIndexWriter()
        embedder = FakeEmbedder()
        classifier = FakeClassifier(slug_to_indexes={"entity/acme": [0]})
        deps = _deps(
            document_store=document_store,
            chunk_store=chunk_store,
            parser=FakeParser("Acme makes widgets."),
            embedder=embedder,
            index_writer=index_writer,
            extractor=FakeExtractor(entities=[_entity("Acme", "entity/acme", "A widget maker")]),
            summarizer=FakeSummarizer("SUMMARY: Acme overview\n\nAcme makes widgets."),
            classifier=classifier,
        )
        op = WikiIngestOp(op=WIKI_OP_INGEST, knowledge_id="d1", language="en")
        result, updates = await map_one_document(
            _CTX,
            deps=deps,
            tenant_id=tenant_id,
            knowledge_base_id=kb_id,
            op=op,
            batch_ctx=WikiBatchContext(),
            existing_pages=[],
        )
        assert result is not None
        assert result.doc_title == "Test doc"
        assert len(chunk_store._rows) == 1
        assert chunk_store._rows[0].index_status == INDEX_STATUS_READY
        assert len(index_writer.writes) == 1
        # summary + entity updates, with the citation attached to the entity
        kinds = {u.slug: u.type for u in updates}
        assert kinds["summary/" + slugify("d1")] == WIKI_PAGE_TYPE_SUMMARY
        assert kinds["entity/acme"] == WIKI_PAGE_TYPE_ENTITY
        entity_update = next(u for u in updates if u.slug == "entity/acme")
        assert entity_update.source_chunks == (chunk_store._rows[0].id,)
        assert document_store.updates[-1][1] == {"parse_status": PARSE_STATUS_PROCESSING}

    async def test_summary_failure_propagates(self) -> None:
        document_store = FakeDocumentStore()
        document_store.seed(
            _doc(tenant_id=1, knowledge_base_id="kb", id="d1", content="A long enough body.")
        )
        summarizer = FakeSummarizer()
        summarizer.fail = True
        deps = _deps(
            document_store=document_store,
            chunk_store=FakeChunkStore(),
            parser=FakeParser("A long enough body."),
            extractor=FakeExtractor(entities=[_entity("Acme", "entity/acme")]),
            summarizer=summarizer,
        )
        op = WikiIngestOp(op=WIKI_OP_INGEST, knowledge_id="d1", language="en")
        with pytest.raises(RuntimeError):
            await map_one_document(
                _CTX,
                deps=deps,
                tenant_id=1,
                knowledge_base_id="kb",
                op=op,
                batch_ctx=WikiBatchContext(),
                existing_pages=[],
            )


# ── Reduce phase ──────────────────────────────────────────────────────


class TestReduce:
    async def test_summary_update_overwrites_page(self) -> None:
        page_store = FakePageStore()
        page_store.seed(
            _page("summary/old", WIKI_PAGE_TYPE_SUMMARY, "Old title").model_copy(
                update={"source_refs": ["k1|Doc"], "chunk_refs": ["c-old"]}
            )
        )
        document_store = FakeDocumentStore()
        document_store.seed(_doc(tenant_id=1, knowledge_base_id="kb", id="k1"))
        deps = _deps(page_store=page_store, document_store=document_store)
        updates = [
            WikiSlugUpdate(
                slug="summary/k1",
                type=WIKI_PAGE_TYPE_SUMMARY,
                doc_title="New doc",
                knowledge_id="k1",
                source_ref="k1",
                language="en",
                summary_line="Headline",
                summary_body="New body.",
            )
        ]
        changed, affected, failed = await reduce_slug_updates(
            _CTX,
            deps=deps,
            tenant_id=1,
            knowledge_base_id="kb",
            slug="summary/k1",
            updates=updates,
            batch_ctx=WikiBatchContext(),
        )
        assert changed and affected == "ingest" and not failed
        page = page_store.pages["summary/k1"]
        assert page.content == "New body."
        assert page.summary == "Headline"
        assert page.title == "New doc - Summary"
        assert page.chunk_refs == []

    async def test_additions_create_page_and_apply_folder(self) -> None:
        page_store = FakePageStore()
        document_store = FakeDocumentStore()
        document_store.seed(_doc(tenant_id=1, knowledge_base_id="kb", id="k1"))
        deps = _deps(page_store=page_store, document_store=document_store)
        updates = [
            WikiSlugUpdate(
                slug="entity/acme",
                type=WIKI_PAGE_TYPE_ENTITY,
                item=_entity("Acme", "entity/acme", "A widget maker"),
                doc_title="Doc",
                knowledge_id="k1",
                source_ref="k1",
                language="en",
                source_chunks=("c1",),
            )
        ]
        batch_ctx = WikiBatchContext(planned_folder_id={"entity/acme": "folder-9"})
        changed, affected, _ = await reduce_slug_updates(
            _CTX,
            deps=deps,
            tenant_id=1,
            knowledge_base_id="kb",
            slug="entity/acme",
            updates=updates,
            batch_ctx=batch_ctx,
        )
        assert changed and affected == "ingest"
        page = page_store.pages["entity/acme"]
        assert page.title == "Acme"
        assert page.page_type == WIKI_PAGE_TYPE_ENTITY
        assert page.source_refs == ["k1"]
        assert page.chunk_refs == ["c1"]
        assert page.folder_id == "folder-9"

    async def test_retract_only_strips_refs_and_deletes_when_empty(self) -> None:
        page_store = FakePageStore()
        page = _page("entity/a", WIKI_PAGE_TYPE_ENTITY, "A").model_copy(
            update={"source_refs": ["k1|Doc", "k2|Other"]}
        )
        page_store.seed(page)
        deps = _deps(page_store=page_store, document_store=FakeDocumentStore())
        updates = [
            WikiSlugUpdate(
                slug="entity/a",
                type=WIKI_UPDATE_RETRACT,
                knowledge_id="k1",
                doc_title="Doc",
                language="en",
            )
        ]
        changed, affected, _ = await reduce_slug_updates(
            _CTX,
            deps=deps,
            tenant_id=1,
            knowledge_base_id="kb",
            slug="entity/a",
            updates=updates,
            batch_ctx=WikiBatchContext(),
        )
        assert changed and affected == "retract"
        assert page_store.pages["entity/a"].source_refs == ["k2|Other"]

        # Retracting the last source deletes the page.
        updates = [
            WikiSlugUpdate(
                slug="entity/a",
                type=WIKI_UPDATE_RETRACT,
                knowledge_id="k2",
                doc_title="Other",
                language="en",
            )
        ]
        changed, affected, _ = await reduce_slug_updates(
            _CTX,
            deps=deps,
            tenant_id=1,
            knowledge_base_id="kb",
            slug="entity/a",
            updates=updates,
            batch_ctx=WikiBatchContext(),
        )
        assert changed and affected == "retract"
        assert page_store.deleted == ["entity/a"]

    async def test_retract_only_for_new_page_is_noop(self) -> None:
        deps = _deps()
        updates = [
            WikiSlugUpdate(
                slug="entity/a", type=WIKI_UPDATE_RETRACT, knowledge_id="k1", doc_title="Doc"
            )
        ]
        changed, affected, _ = await reduce_slug_updates(
            _CTX,
            deps=deps,
            tenant_id=1,
            knowledge_base_id="kb",
            slug="entity/a",
            updates=updates,
            batch_ctx=WikiBatchContext(),
        )
        assert (changed, affected) == (False, "")


# ── Batch driver ──────────────────────────────────────────────────────


class TestProcessBatch:
    async def test_batch_processes_and_trims_queue(self) -> None:
        tenant_id = 1
        kb_id = _kb()
        document_store = FakeDocumentStore()
        document_store.seed(
            _doc(
                tenant_id=tenant_id, knowledge_base_id=kb_id, id="d1", content="Acme makes widgets."
            )
        )
        pending_store = FakePendingStore()
        await pending_store.enqueue(
            tenant_id=tenant_id,
            knowledge_base_id=kb_id,
            op=WikiIngestOp(op=WIKI_OP_INGEST, knowledge_id="d1", language="en"),
        )
        page_store = FakePageStore()
        deps = _deps(
            document_store=document_store,
            chunk_store=FakeChunkStore(),
            pending_store=pending_store,
            page_store=page_store,
            parser=FakeParser("Acme makes widgets."),
            extractor=FakeExtractor(entities=[_entity("Acme", "entity/acme")]),
            summarizer=FakeSummarizer(),
            planner=FakePlanner(assignments={"entity/acme": ["Companies"]}),
        )
        outcome = await process_wiki_ingest_batch(
            _CTX,
            deps=deps,
            tenant_id=tenant_id,
            knowledge_base_id=kb_id,
            language="en",
            batch_ctx=WikiBatchContext(),
        )
        assert outcome.ingest_succeeded == 1
        assert outcome.pages_affected >= 1
        assert await pending_store.pending_count(knowledge_base_id=kb_id) == 0
        assert page_store.pages["entity/acme"].folder_id == "folder-1"
        doc = await document_store.get_by_id(tenant_id, "d1")
        assert doc is not None and doc.parse_status == PARSE_STATUS_COMPLETED

    async def test_batch_requeues_failed_ops(self) -> None:
        tenant_id = 1
        kb_id = _kb()
        document_store = FakeDocumentStore()
        document_store.seed(
            _doc(
                tenant_id=tenant_id, knowledge_base_id=kb_id, id="d1", content="A long enough body."
            )
        )
        pending_store = FakePendingStore()
        await pending_store.enqueue(
            tenant_id=tenant_id,
            knowledge_base_id=kb_id,
            op=WikiIngestOp(op=WIKI_OP_INGEST, knowledge_id="d1", language="en"),
        )
        summarizer = FakeSummarizer()
        summarizer.fail = True
        deps = _deps(
            document_store=document_store,
            chunk_store=FakeChunkStore(),
            pending_store=pending_store,
            parser=FakeParser("A long enough body."),
            summarizer=summarizer,
        )
        outcome = await process_wiki_ingest_batch(
            _CTX,
            deps=deps,
            tenant_id=tenant_id,
            knowledge_base_id=kb_id,
            language="en",
            batch_ctx=WikiBatchContext(),
        )
        assert outcome.ingest_failed == 1
        assert outcome.pages_affected == 0
        assert await pending_store.pending_count(knowledge_base_id=kb_id) == 1
        assert pending_store.archived == []

    async def test_batch_dead_letters_after_retry_budget(self) -> None:
        tenant_id = 1
        kb_id = _kb()
        document_store = FakeDocumentStore()
        document_store.seed(
            _doc(
                tenant_id=tenant_id, knowledge_base_id=kb_id, id="d1", content="A long enough body."
            )
        )
        pending_store = FakePendingStore()
        await pending_store.enqueue(
            tenant_id=tenant_id,
            knowledge_base_id=kb_id,
            op=WikiIngestOp(op=WIKI_OP_INGEST, knowledge_id="d1", language="en"),
        )
        summarizer = FakeSummarizer()
        summarizer.fail = True
        deps = _deps(
            document_store=document_store,
            chunk_store=FakeChunkStore(),
            pending_store=pending_store,
            parser=FakeParser("A long enough body."),
            summarizer=summarizer,
        )
        for _ in range(6):  # initial + max_fail_retries=5 retries
            await process_wiki_ingest_batch(
                _CTX,
                deps=deps,
                tenant_id=tenant_id,
                knowledge_base_id=kb_id,
                language="en",
                batch_ctx=WikiBatchContext(),
            )
        assert len(pending_store.archived) == 1
        assert await pending_store.pending_count(knowledge_base_id=kb_id) == 0
        doc = await document_store.get_by_id(tenant_id, "d1")
        assert doc is not None and doc.parse_status == PARSE_STATUS_FAILED

    async def test_batch_empty_queue(self) -> None:
        deps = _deps()
        outcome = await process_wiki_ingest_batch(
            _CTX,
            deps=deps,
            tenant_id=1,
            knowledge_base_id="kb",
            language="en",
            batch_ctx=WikiBatchContext(),
        )
        assert outcome.pending_ops == 0
        assert not outcome.follow_up_scheduled


# ── Service + composite writer ────────────────────────────────────────


class TestWikiIngestService:
    async def test_enqueue_ingest_delegates(self) -> None:
        pending_store = FakePendingStore()
        service = WikiIngestService(deps=_deps(pending_store=pending_store))
        accepted = await service.enqueue_ingest(
            tenant_id=1, knowledge_base_id="kb", knowledge_id="d1", language="en"
        )
        assert accepted
        ops = await pending_store.peek(knowledge_base_id="kb", limit=5)
        assert ops[0].op == WIKI_OP_INGEST
        assert ops[0].knowledge_id == "d1"

    async def test_enqueue_retract_delegates(self) -> None:
        pending_store = FakePendingStore()
        service = WikiIngestService(deps=_deps(pending_store=pending_store))
        accepted = await service.enqueue_retract(
            tenant_id=1,
            knowledge_base_id="kb",
            knowledge_id="d1",
            doc_title="Doc",
            page_slugs=["entity/a"],
        )
        assert accepted
        ops = await pending_store.peek(knowledge_base_id="kb", limit=5)
        assert ops[0].op == WIKI_OP_RETRACT
        assert ops[0].page_slugs == ("entity/a",)

    async def test_process_batch_defaults_context(self) -> None:
        deps = _deps()
        service = WikiIngestService(deps=deps)
        outcome = await service.process_batch(
            None, tenant_id=1, knowledge_base_id="kb", language="en"
        )
        assert outcome.pending_ops == 0


class TestCompositeIndexWriter:
    def test_write_chunks_builds_index_infos(self) -> None:
        calls: list[list] = []

        class FakeComposite:
            async def batch_index(self, ctx, embedder, infos) -> None:
                calls.append(infos)

        writer = CompositeIndexWriter(composite=FakeComposite())  # type: ignore[arg-type]
        chunks = [_chunk_row(tenant_id=1, knowledge_base_id="kb", knowledge_id="k", content="body")]
        # Await the coroutine via anyio/anyio.run in asyncio context.
        import asyncio

        async def run() -> None:
            await writer.write_chunks(
                ctx=_CTX,
                tenant_id=1,
                knowledge_base_id="kb",
                knowledge_id="k",
                chunks=chunks,
                embeddings=[[0.1, 0.2]],
                embedder=FakeEmbedder(),
            )

        asyncio.run(run())
        infos = calls[0]
        assert infos[0].chunk_id == chunks[0].id
        assert infos[0].content == "body"
        assert infos[0].knowledge_type == "wiki"

    def test_delete_by_source_id_list(self) -> None:
        calls: list[tuple] = []

        class FakeComposite:
            async def delete_by_source_id_list(
                self, ctx, source_id_list, dimension, knowledge_type
            ) -> None:
                calls.append((source_id_list, dimension, knowledge_type))

        writer = CompositeIndexWriter(composite=FakeComposite())  # type: ignore[arg-type]
        import asyncio

        async def run() -> None:
            await writer.delete_by_source_id_list(
                ctx=_CTX,
                tenant_id=1,
                knowledge_base_id="kb",
                source_id_list=["c1"],
                dimension=3,
            )

        asyncio.run(run())
        assert calls[0] == (["c1"], 3, "wiki")


# ── Integration (real Postgres, revision 0022+) ───────────────────────

# tenant ids for integration tests must fit chunks.tenant_id (INTEGER).
_used_tenant_ids: set[int] = set()
_int32_tenant_ids: set[int] = set()


def _int32_tenant_id() -> int:
    """A tenant id unique in the session and safe for ``chunks.tenant_id``."""
    while True:
        candidate = secrets.randbelow(2**31 - 1) + 1
        if candidate not in _used_tenant_ids and candidate not in _int32_tenant_ids:
            _int32_tenant_ids.add(candidate)
            return candidate


@pytest.fixture(scope="session")
def _db_engine() -> DatabaseEngine:
    reset_settings_cache()
    settings = get_settings()
    return DatabaseEngine(url=settings.database_url, poolclass=NullPool)


@pytest_asyncio.fixture
async def db_session(_db_engine: DatabaseEngine) -> AsyncIterator[AsyncSession]:
    factory = async_sessionmaker(_db_engine.engine, expire_on_commit=False)
    async with factory() as session:
        try:
            await session.execute(text("select 1"))
        except Exception:
            pytest.skip("integration Postgres is not reachable; set DATABASE_URL_OVERRIDE")
        yield session
        await session.rollback()


async def test_integration_batch_end_to_end(db_session: AsyncSession) -> None:
    tenant_id = _int32_tenant_id()
    kb_id = _kb()
    knowledge_id = _kid("doc")
    now = datetime.now(UTC)

    knowledge_repo = KnowledgeRepository(db_session)
    document = _doc(
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
        id=knowledge_id,
        title="Integration doc",
        content="",
    )
    document = document.model_copy(
        update={"type": "manual", "source": "manual", "created_at": now, "updated_at": now}
    )
    await knowledge_repo.create(document)

    page_repo = WikiPageRepository(db_session)
    folder_repo = WikiFolderRepository(db_session)
    from src.core.knowledge.wiki.folders import WikiFolderService
    from src.core.knowledge.wiki.page_service import WikiPageService

    page_service = WikiPageService(page_repo=page_repo, folder_repo=folder_repo)
    folder_service = WikiFolderService(folder_repo=folder_repo, page_repo=page_repo)
    chunk_store = ChunkRepository(db_session)
    pending_store = FakePendingStore()
    await pending_store.enqueue(
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
        op=WikiIngestOp(op=WIKI_OP_INGEST, knowledge_id=knowledge_id, language="en"),
    )

    deps = WikiIngestDeps(
        page_service=page_service,
        folder_service=folder_service,
        document_store=knowledge_repo,
        chunk_store=chunk_store,
        pending_store=pending_store,
        parser=FakeParser("Acme Corporation builds widgets and ships them globally."),
        embedder=FakeEmbedder(),
        index_writer=FakeIndexWriter(),
        extractor=FakeExtractor(
            entities=[_entity("Acme", "entity/acme", "A widget maker", aliases=("ACME",))]
        ),
        summarizer=FakeSummarizer("SUMMARY: Acme overview\n\nAcme builds widgets."),
        classifier=FakeClassifier(slug_to_indexes={"entity/acme": [0]}),
        planner=FakePlanner(assignments={"entity/acme": ["Companies", "Widgets"]}),
    )

    outcome = await process_wiki_ingest_batch(
        _CTX,
        deps=deps,
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
        language="en",
        batch_ctx=WikiBatchContext(),
    )

    assert outcome.ingest_succeeded == 1
    assert outcome.pages_affected >= 1

    # Chunk rows persisted and settled to ready.
    chunks = await chunk_store.list_by_knowledge_id(tenant_id=tenant_id, knowledge_id=knowledge_id)
    assert len(chunks) == 1
    assert chunks[0].index_status == INDEX_STATUS_READY
    assert chunks[0].tenant_id == tenant_id

    # Wiki pages created: the summary and the entity page with its folder.
    summary_slug = f"summary/{slugify(knowledge_id)}"
    summary_page = await page_service.get_page_by_slug(knowledge_base_id=kb_id, slug=summary_slug)
    assert summary_page.content == "Acme builds widgets."
    entity_page = await page_service.get_page_by_slug(knowledge_base_id=kb_id, slug="entity/acme")
    assert entity_page.page_type == WIKI_PAGE_TYPE_ENTITY
    assert entity_page.title == "Acme"
    assert entity_page.source_refs == [knowledge_id]
    assert entity_page.chunk_refs == [chunks[0].id]
    assert entity_page.category_path == ["Companies", "Widgets"]

    # The document row reached a terminal parse status and the queue drained.
    stored = await knowledge_repo.get_by_id(tenant_id, knowledge_id)
    assert stored is not None
    assert stored.parse_status == PARSE_STATUS_COMPLETED
    assert await pending_store.pending_count(knowledge_base_id=kb_id) == 0


async def test_integration_chunk_rows_tenant_scoped(db_session: AsyncSession) -> None:
    """Chunk writes honour the caller's tenant id and are int32-safe."""
    tenant_id = _int32_tenant_id()
    assert 1 <= tenant_id < 2**31
    kb_id = _kb()
    chunk_store = ChunkRepository(db_session)
    row = _chunk_row(tenant_id=tenant_id, knowledge_base_id=kb_id, knowledge_id="k", content="x")
    row = row.model_copy(update={"created_at": datetime.now(UTC), "updated_at": datetime.now(UTC)})
    created = await chunk_store.create_many([row])
    assert created[0].id == row.id
    loaded = await chunk_store.list_by_knowledge_id(tenant_id=tenant_id, knowledge_id="k")
    assert [c.id for c in loaded] == [row.id]
