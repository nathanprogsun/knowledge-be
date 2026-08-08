"""Unit and integration tests for chunk extraction and data-table summary.

Unit tests drive the standalone modules with stateful repository mocks
(closure-captured storage, the same pattern used across the core service
tests), a scripted chat seam, and a real composite retrieve engine built
over a fake engine service so the datatable-summary index fan-out is
exercised without any vector store.

Integration tests run against the real applied schema and skip when the
database is unreachable. ``chunks`` carries an INTEGER (32-bit)
``tenant_id`` column, so those tests use an int32-safe tenant id (a local
counter) instead of ``make_test_tenant_id``'s BIGINT range.
"""

from __future__ import annotations

import itertools
import uuid
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime
from random import randint

import pytest
import pytest_asyncio
from faker import Faker
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.pool import NullPool

from src.ai.embedding import Context, Embedder
from src.ai.llm import ChatOptions, ChatResponse, Message
from src.ai.retrieval.composite import CompositeRetrieveEngine, new_composite_retrieve_engine
from src.ai.retrieval.types import (
    IndexInfo,
    RetrieveParams,
    RetrieverEngineParams,
    RetrieverEngineType,
    RetrieveResult,
    RetrieverType,
)
from src.common.exception import NotFoundError, ValidationError
from src.common.json import JsonObject, JsonValue
from src.core.knowledge.chunks.service.chunk_service import ChunkService
from src.core.knowledge.chunks.types import (
    CHUNK_STATUS_INDEXED,
    CHUNK_STATUS_STORED,
    CHUNK_TYPE_TABLE_COLUMN,
    CHUNK_TYPE_TABLE_SUMMARY,
    CHUNK_TYPE_TEXT,
)
from src.core.knowledge.documents.chunk_extract import (
    ChunkExtractor,
    ExtractionOutcome,
    GraphData,
    GraphExtractionError,
    GraphNode,
    GraphRelation,
    PromptTemplateStructured,
    StructureExtractor,
    append_custom_prompt_instructions,
    format_extraction,
    parse_graph_output,
    render_extraction_messages,
    render_extraction_system_prompt,
    render_extraction_user_prompt,
    resolve_extract_config,
    should_enqueue_table_summary,
)
from src.core.knowledge.documents.datatable_summary import (
    DataTableSummaryResult,
    SummaryContext,
    TableColumn,
    TableSchema,
    TenantEffectiveEngines,
    build_datatable_chunks,
    build_sample_data_description,
    cleanup_on_failure,
    generate_column_descriptions,
    generate_table_description,
    index_to_vector_db,
    process_datatable_summary,
    resolve_datatable_engine,
    resolve_table_metadata_instructions,
)
from src.core.knowledge.documents.service.knowledge_service import KnowledgeService
from src.core.knowledge.documents.types import (
    PARSE_STATUS_CANCELLED,
    PARSE_STATUS_COMPLETED,
    PARSE_STATUS_DELETING,
    PARSE_STATUS_FAILED,
    SUMMARY_STATUS_NONE,
)
from src.core.knowledge.knowledge_bases.service.kb_service import KBService
from src.core.knowledge.knowledge_bases.types import KnowledgeBaseInfo
from src.core.tenants.types import RetrieverEngineEntry, TenantInfo
from src.db.base import DatabaseEngine
from src.db.dao.chunk_repository import ChunkRepository
from src.db.dao.knowledge_base_repository import KnowledgeBaseRepository
from src.db.dao.knowledge_repository import KnowledgeRepository
from src.db.models.chunk import Chunk
from src.db.models.knowledge import Document
from src.settings import get_settings, reset_settings_cache

_NOW = datetime(2026, 2, 1, 12, 0, tzinfo=UTC)
_FAKER_SEED_MAX = 100_000_000

# ``chunks.tenant_id`` is INTEGER (32-bit); integration tests mint ids from
# this counter so they stay inside the range.
_INT32_TENANT_BASE = 4_000_000
_INT32_TENANT_SEQ = itertools.count(start=1)


def _int32_tenant_id() -> int:
    """Return a tenant id that fits the ``chunks.tenant_id`` INTEGER column."""
    return _INT32_TENANT_BASE + next(_INT32_TENANT_SEQ)


@pytest.fixture(autouse=True)
def faker_seed() -> None:
    """Re-seed Faker per test for varied-but-reproducible generation."""
    Faker.seed(randint(1, _FAKER_SEED_MAX))


def _did() -> str:
    return f"doc-{uuid.uuid4().hex[:12]}"


def _kbid() -> str:
    return f"kb-{uuid.uuid4().hex[:12]}"


def _doc_row(
    *,
    tenant_id: int,
    knowledge_base_id: str,
    file_type: str | None = None,
    parse_status: str = PARSE_STATUS_COMPLETED,
    metadata: JsonObject | None = None,
    id: str | None = None,
) -> Document:
    """Build a persisted-shape document row for seeding mocks / DB."""
    return Document.model_validate(
        {
            "id": id or _did(),
            "tenant_id": tenant_id,
            "knowledge_base_id": knowledge_base_id,
            "type": "file",
            "title": "fixture dataset",
            "description": None,
            "source": "fixture.csv",
            "channel": "web",
            "parse_status": parse_status,
            "pending_subtasks_count": 0,
            "summary_status": SUMMARY_STATUS_NONE,
            "enable_status": "enabled",
            "embedding_model_id": None,
            "file_name": "fixture.csv",
            "file_type": file_type,
            "file_size": 1024,
            "file_hash": None,
            "file_path": "obj/fixture.csv",
            "storage_size": 0,
            "metadata": metadata,
            "custom_metadata": {},
            "last_faq_import_result": None,
            "created_at": _NOW,
            "updated_at": _NOW,
            "processed_at": None,
            "error_message": None,
            "deleted_at": None,
        }
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
        id=str(uuid.uuid4()),
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        knowledge_id=knowledge_id,
        content=content,
        chunk_index=chunk_index,
        is_enabled=True,
        start_at=0,
        end_at=len(content),
        pre_chunk_id=None,
        next_chunk_id=None,
        chunk_type=CHUNK_TYPE_TEXT,
        parent_chunk_id=None,
        image_info=None,
        relation_chunks=None,
        indirect_relation_chunks=None,
        metadata=None,
        tag_id=None,
        status=CHUNK_STATUS_STORED,
        content_hash=None,
        flags=1,
        seq_id=0,
        source_content="",
        content_revision=0,
        index_status="ready",
        last_editor_id="",
        context_header="",
        created_at=_NOW,
        updated_at=_NOW,
        deleted_at=None,
    )


def _kb(
    *,
    tenant_id: int,
    id: str | None = None,
    extract_config: JsonObject | None = None,
    chunking_config: JsonObject | None = None,
) -> KnowledgeBaseInfo:
    return KnowledgeBaseInfo.model_validate(
        {
            "id": id or _kbid(),
            "name": "extract-kb",
            "type": "document",
            "tenant_id": tenant_id,
            "extract_config": extract_config,
            "chunking_config": chunking_config,
            "created_at": _NOW,
            "updated_at": _NOW,
        }
    )


def _tenant(
    *,
    id: int,
    engines: list[RetrieverEngineEntry] | None = None,
) -> TenantInfo:
    return TenantInfo.model_validate(
        {
            "id": id,
            "name": "ws",
            "status": "active",
            "retriever_engines": {"engines": engines or []},
            "created_at": _NOW,
            "updated_at": _NOW,
        }
    )


# ── Fake seams ─────────────────────────────────────────────────────────


class _FakeChat:
    """Scripted chat seam: records calls and returns canned output."""

    def __init__(self, *, content: str = "", error: Exception | None = None) -> None:
        self.content = content
        self.error = error
        self.calls: list[tuple[list[Message], ChatOptions | None]] = []

    async def chat(
        self,
        messages: list[Message],
        opts: ChatOptions | None = None,
    ) -> ChatResponse:
        if self.error is not None:
            raise self.error
        self.calls.append((messages, opts))
        return ChatResponse(content=self.content)


class _FakeEmbedder:
    """Embedding double satisfying the ``Embedder`` protocol."""

    def __init__(self, dimensions: int = 8) -> None:
        self.dimensions = dimensions

    def get_model_name(self) -> str:
        return "fake-embed"

    def get_model_id(self) -> str:
        return "embed-model"

    def get_dimensions(self) -> int:
        return self.dimensions

    async def embed(self, ctx: Context, text: str) -> list[float]:
        return [0.0] * self.dimensions

    async def batch_embed(self, ctx: Context, texts: list[str]) -> list[list[float]]:
        return [[0.0] * self.dimensions for _ in texts]

    async def batch_embed_with_pool(
        self,
        ctx: Context,
        model: Embedder,
        texts: list[str],
    ) -> list[list[float]]:
        return await self.batch_embed(ctx, texts)


class _FakeEngineService:
    """Retrieve-engine double recording every fan-out call."""

    def __init__(
        self,
        *,
        engine_type: RetrieverEngineType = RetrieverEngineType.SQLITE,
        support: list[RetrieverType] | None = None,
        batch_error: Exception | None = None,
    ) -> None:
        self._engine_type = engine_type
        self._support = support or [
            RetrieverType.KEYWORDS,
            RetrieverType.VECTOR,
        ]
        self.batch_error = batch_error
        self.indexed: list[IndexInfo] = []
        self.deleted_sources: list[str] = []
        self.deleted_chunks: list[str] = []
        self.batch_calls: list[tuple[int, tuple[RetrieverType, ...]]] = []

    def engine_type(self) -> RetrieverEngineType:
        return self._engine_type

    def support(self) -> list[RetrieverType]:
        return list(self._support)

    async def retrieve(
        self, _ctx: Context, params: RetrieveParams
    ) -> list[RetrieveResult]:
        return []

    async def index(
        self,
        _ctx: Context,
        _embedder: Embedder,
        index_info: IndexInfo,
        _retriever_types: list[RetrieverType],
    ) -> None:
        self.indexed.append(index_info)

    async def batch_index(
        self,
        _ctx: Context,
        _embedder: Embedder,
        index_info_list: list[IndexInfo],
        retriever_types: list[RetrieverType],
    ) -> None:
        if self.batch_error is not None:
            raise self.batch_error
        self.indexed.extend(index_info_list)
        self.batch_calls.append((len(index_info_list), tuple(retriever_types)))

    def estimate_storage_size(
        self,
        _ctx: Context,
        _embedder: Embedder,
        index_info_list: list[IndexInfo],
        _retriever_types: list[RetrieverType],
    ) -> int:
        return len(index_info_list)

    async def delete_by_chunk_id_list(
        self,
        _ctx: Context,
        index_id_list: list[str],
        _dimension: int,
        _knowledge_type: str,
    ) -> None:
        self.deleted_chunks.extend(index_id_list)

    async def delete_by_source_id_list(
        self,
        _ctx: Context,
        source_id_list: list[str],
        _dimension: int,
        _knowledge_type: str,
    ) -> None:
        self.deleted_sources.extend(source_id_list)

    async def delete_by_knowledge_id_list(
        self,
        _ctx: Context,
        _knowledge_id_list: list[str],
        _dimension: int,
        _knowledge_type: str,
    ) -> None:
        pass

    async def batch_update_chunk_enabled_status(
        self,
        _ctx: Context,
        _chunk_status_map: Mapping[str, bool],
    ) -> None:
        pass

    async def batch_update_chunk_tag_id(
        self,
        _ctx: Context,
        _chunk_tag_map: Mapping[str, str],
    ) -> None:
        pass

    async def copy_indices(
        self,
        _ctx: Context,
        _source_knowledge_base_id: str,
        _source_to_target_kb_id_map: Mapping[str, str],
        _source_to_target_chunk_id_map: Mapping[str, str],
        _target_knowledge_base_id: str,
        _dimension: int,
        _knowledge_type: str,
    ) -> None:
        pass


class _FakeRegistry:
    """Registry double serving one engine service per engine type."""

    def __init__(self, service: _FakeEngineService) -> None:
        self._service = service

    def register(self, service: _FakeEngineService) -> None:
        pass

    def get_retrieve_engine_service(
        self, engine_type: RetrieverEngineType
    ) -> _FakeEngineService:
        return self._service

    def register_with_store_id(self, store_id: str, svc: _FakeEngineService) -> None:
        pass

    def get_by_store_id(self, store_id: str) -> _FakeEngineService:
        return self._service

    def unregister_by_store_id(self, store_id: str) -> None:
        pass

    async def get_or_load_by_store_id(
        self, ctx: Context, tenant_id: int, store_id: str
    ) -> _FakeEngineService:
        return self._service


class _FakeGraphStore:
    """Graph-store double recording every ``add_graph`` call."""

    def __init__(self) -> None:
        self.added: list[tuple[str, str, list[GraphData]]] = []

    async def add_graph(
        self,
        *,
        knowledge_base_id: str,
        knowledge_id: str,
        graphs: list[GraphData],
    ) -> None:
        self.added.append((knowledge_base_id, knowledge_id, graphs))


class _FakeKnowledgeRepo:
    """Knowledge-repository double with closure-captured storage."""

    def __init__(self) -> None:
        self.rows: dict[str, Document] = {}

    def seed(self, row: Document) -> None:
        self.rows[row.id] = row

    async def get_by_id(self, tenant_id: int, id: str) -> Document | None:
        row = self.rows.get(id)
        if row is not None and row.tenant_id == tenant_id and row.deleted_at is None:
            return row
        return None

    async def update(self, row: Document) -> Document:
        self.rows[row.id] = row
        return row


class _FakeChunkRepo:
    """Chunk-repository double with closure-captured storage."""

    def __init__(self) -> None:
        self.rows: dict[str, Chunk] = {}

    def seed(self, row: Chunk) -> None:
        self.rows[row.id] = row

    async def get_by_id(self, tenant_id: int, id: str) -> Chunk:
        row = self.rows.get(id)
        if row is not None and row.tenant_id == tenant_id and row.deleted_at is None:
            return row
        raise NotFoundError(code="chunk.not_found", message=f"chunk {id} not found")

    async def get_by_id_or_none(self, tenant_id: int, id: str) -> Chunk | None:
        try:
            return await self.get_by_id(tenant_id, id)
        except NotFoundError:
            return None


class _FakeKBService:
    """Knowledge-base service double returning canned knowledge bases."""

    def __init__(self, kb: KnowledgeBaseInfo) -> None:
        self.kb = kb
        self.requests: list[str] = []

    async def get_knowledge_base_by_id(self, *, knowledge_base_id: str) -> KnowledgeBaseInfo:
        self.requests.append(knowledge_base_id)
        return self.kb


class _FakeChunkService:
    """Chunk-service double recording create/update/delete calls."""

    def __init__(self) -> None:
        self.created: list[Chunk] = []
        self.updated: list[Chunk] = []
        self.deleted_ids: list[str] = []

    async def create_chunks(self, *, chunks: list[Chunk]) -> list[Chunk]:
        self.created.extend(chunks)
        return chunks

    async def update_chunks(self, *, chunks: list[Chunk]) -> list[Chunk]:
        self.updated.extend(chunks)
        return chunks

    async def delete_chunks(self, *, tenant_id: int, ids: list[str]) -> int:
        self.deleted_ids.extend(ids)
        return len(ids)


class _FakeTableTool:
    """Table-data tool double loading a canned schema and sample rows."""

    def __init__(
        self,
        schema: TableSchema,
        rows: list[dict[str, JsonValue]],
    ) -> None:
        self.schema = schema
        self.rows = rows
        self.loaded: list[Document] = []
        self.cleanups = 0

    async def load_from_knowledge(self, *, knowledge: Document) -> TableSchema:
        self.loaded.append(knowledge)
        return self.schema

    async def sample_rows(
        self, *, table_name: str, limit: int
    ) -> list[dict[str, JsonValue]]:
        return self.rows[:limit]

    def cleanup(self) -> None:
        self.cleanups += 1


# ── chunk_extract: pure helpers ────────────────────────────────────────


def test_append_custom_prompt_instructions_blank_is_noop() -> None:
    assert append_custom_prompt_instructions("prompt", "   ") == "prompt"


def test_append_custom_prompt_instructions_appends_labeled_block() -> None:
    result = append_custom_prompt_instructions("prompt", "domain rules", "table_metadata")
    assert result.startswith("prompt\n\n")
    assert "<table_metadata_business_instructions>\ndomain rules\n</table_metadata_business_instructions>" in result
    assert "do not conflict" in result


def test_is_data_table_file_type_gate() -> None:
    assert should_enqueue_table_summary("csv") is True
    assert should_enqueue_table_summary(".XLSX") is True
    assert should_enqueue_table_summary("xls") is True
    assert should_enqueue_table_summary("pdf") is False
    # Empty file type falls back to the file name's extension.
    assert should_enqueue_table_summary("", "data.csv") is True
    assert should_enqueue_table_summary("", "notes.pdf") is False


def test_format_extraction_renders_json_fence() -> None:
    graph = format_extraction(
        [GraphNode(name="A", attributes=["x"])],
        [GraphRelation(node1="A", node2="B", type="link")],
    )
    assert graph.startswith("```json\n")
    assert graph.endswith("\n```")
    assert '"entity": "A"' in graph
    assert '"entity1": "A"' in graph


def test_render_extraction_system_prompt_with_tags_substitutes_placeholder() -> None:
    template = PromptTemplateStructured(
        description="Allowed relation types are: %s.",
        tags=["Author", "Alias"],
        examples=[GraphData(text="t", node=[GraphNode(name="A")])],
    )
    rendered = render_extraction_system_prompt(template)
    assert '["Author", "Alias"]' in rendered
    assert "# Examples" in rendered
    assert "Q: t" in rendered


def test_render_extraction_user_prompt() -> None:
    assert render_extraction_user_prompt("body") == "# Question\nQ: body\nA: "


def test_render_extraction_messages_roles() -> None:
    messages = render_extraction_messages(PromptTemplateStructured(description="d"), "c")
    assert [m.role for m in messages] == ["system", "user"]
    assert messages[1].content == "# Question\nQ: c\nA: "


# ── chunk_extract: graph output parsing ────────────────────────────────


def test_parse_graph_output_fenced_json() -> None:
    graph = parse_graph_output('```json\n[{"entity": "A"}, {"entity1": "A", "entity2": "B", "relation": "link"}]\n```')
    assert [n.name for n in graph.node] == ["A", "B"]
    assert graph.relation[0].type == "link"


def test_parse_graph_output_merges_duplicate_nodes_and_attributes() -> None:
    graph = parse_graph_output(
        '[{"entity": "A", "entity_attributes": ["x"]}, '
        '{"entity": "A", "entity_attributes": ["y"]}]'
    )
    assert len(graph.node) == 1
    assert graph.node[0].attributes == ["x", "y"]


def test_parse_graph_output_drops_self_relations_and_synthesises_endpoints() -> None:
    graph = parse_graph_output(
        '[{"entity1": "A", "entity2": "A", "relation": "self"}, '
        '{"entity1": "X", "entity2": "Y", "relation": "goes"}]'
    )
    assert len(graph.relation) == 1
    assert graph.relation[0].node1 == "X"
    assert {n.name for n in graph.node} == {"X", "Y"}


def test_parse_graph_output_recovers_unfenced_json() -> None:
    graph = parse_graph_output('Here is the answer: [{"entity": "A"}]')
    assert [n.name for n in graph.node] == ["A"]


def test_parse_graph_output_raises_on_empty_or_invalid() -> None:
    with pytest.raises(GraphExtractionError):
        parse_graph_output("")
    with pytest.raises(GraphExtractionError):
        parse_graph_output("```json\nnot json\n```")


def test_structure_extractor_calls_chat_and_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    chat = _FakeChat(content='```json\n[{"entity": "Romeo"}]\n```')
    extractor = StructureExtractor(chat, PromptTemplateStructured(description="d"))
    graph = None

    async def run() -> GraphData:
        nonlocal graph
        graph = await extractor.extract(_ctx(), "chunk body")
        return graph

    import asyncio

    asyncio.run(run())
    assert graph is not None
    assert [n.name for n in graph.node] == ["Romeo"]
    assert chat.calls[0][1] is not None
    assert chat.calls[0][1].temperature == 0.3
    assert chat.calls[0][1].max_tokens == 4096


def _ctx() -> Context:
    from src.ai.embedding import TaskContext

    return TaskContext()


# ── chunk_extract: effective config resolution ─────────────────────────


def test_resolve_extract_config_defaults() -> None:
    cfg = resolve_extract_config(_kb(tenant_id=1), None)
    assert cfg.enabled is False
    assert cfg.tags == []
    assert cfg.custom_instructions == ""


def test_resolve_extract_config_from_kb() -> None:
    kb = _kb(
        tenant_id=1,
        extract_config={
            "enabled": True,
            "tags": ["Author"],
            "text": "sample",
            "nodes": [{"name": "A"}],
            "relations": [{"node1": "A", "node2": "B", "type": "wrote"}],
            "custom_instructions": "be strict",
        },
    )
    cfg = resolve_extract_config(kb, None)
    assert cfg.enabled is True
    assert cfg.tags == ["Author"]
    assert cfg.text == "sample"
    assert cfg.nodes[0].name == "A"
    assert cfg.relations[0].type == "wrote"
    assert cfg.custom_instructions == "be strict"


def test_resolve_extract_config_override_keeps_blank_base() -> None:
    kb = _kb(
        tenant_id=1,
        extract_config={"enabled": False, "tags": ["Base"], "custom_instructions": "base"},
    )
    cfg = resolve_extract_config(
        kb,
        {"extract_config": {"enabled": True, "tags": ["Override"]}},
    )
    assert cfg.enabled is True
    assert cfg.tags == ["Override"]
    # Blank override fields keep the knowledge-base values.
    assert cfg.custom_instructions == "base"


# ── chunk_extract: orchestration ───────────────────────────────────────


def _extractor(
    *,
    kb: KnowledgeBaseInfo,
    chunk: Chunk,
    knowledge: Document | None,
    graph_store: _FakeGraphStore | None = None,
    graph_enabled: bool = True,
) -> ChunkExtractor:
    chunk_repo = _FakeChunkRepo()
    chunk_repo.seed(chunk)
    knowledge_repo = _FakeKnowledgeRepo()
    if knowledge is not None:
        knowledge_repo.seed(knowledge)
    return ChunkExtractor(
        chunk_repo=chunk_repo,
        knowledge_repo=knowledge_repo,
        kb_service=_FakeKBService(kb),
        graph_store=graph_store or _FakeGraphStore(),
        graph_enabled=graph_enabled,
        default_description="Extract from %s.",
    )


def _run_extract(
    extractor: ChunkExtractor, chat: _FakeChat, chunk_id: str
) -> ExtractionOutcome:
    import asyncio

    return asyncio.run(
        extractor.extract_chunk(
            ctx=_ctx(),
            tenant_id=1,
            chunk_id=chunk_id,
            chat=chat,
            knowledge_id="k-1",
        )
    )


def test_extract_chunk_skips_when_graph_disabled() -> None:
    kb = _kb(tenant_id=1, extract_config={"enabled": True})
    chunk = _chunk_row(tenant_id=1, knowledge_base_id=kb.id, knowledge_id="k-1", content="x")
    extractor = _extractor(kb=kb, chunk=chunk, knowledge=None, graph_enabled=False)
    outcome = _run_extract(extractor, _FakeChat(), chunk.id)
    assert outcome.skipped is True
    assert outcome.reason == "graph_disabled"


def test_extract_chunk_skips_when_knowledge_cancelled() -> None:
    kb = _kb(tenant_id=1, extract_config={"enabled": True})
    chunk = _chunk_row(tenant_id=1, knowledge_base_id=kb.id, knowledge_id="k-1", content="x")
    knowledge = _doc_row(
        tenant_id=1, knowledge_base_id=kb.id, parse_status=PARSE_STATUS_CANCELLED
    )
    knowledge = knowledge.model_copy(update={"id": "k-1"})
    extractor = _extractor(kb=kb, chunk=chunk, knowledge=knowledge)
    outcome = _run_extract(extractor, _FakeChat(), chunk.id)
    assert outcome.skipped is True
    assert outcome.reason == "knowledge_cancelled"


def test_extract_chunk_skips_when_knowledge_deleting() -> None:
    kb = _kb(tenant_id=1, extract_config={"enabled": True})
    chunk = _chunk_row(tenant_id=1, knowledge_base_id=kb.id, knowledge_id="k-1", content="x")
    knowledge = _doc_row(
        tenant_id=1, knowledge_base_id=kb.id, parse_status=PARSE_STATUS_DELETING
    )
    knowledge = knowledge.model_copy(update={"id": "k-1"})
    extractor = _extractor(kb=kb, chunk=chunk, knowledge=knowledge)
    outcome = _run_extract(extractor, _FakeChat(), chunk.id)
    assert outcome.skipped is True
    assert outcome.reason == "knowledge_deleting"


def test_extract_chunk_skips_when_extract_config_disabled() -> None:
    kb = _kb(tenant_id=1, extract_config={"enabled": False})
    chunk = _chunk_row(tenant_id=1, knowledge_base_id=kb.id, knowledge_id="k-1", content="x")
    extractor = _extractor(kb=kb, chunk=chunk, knowledge=None)
    outcome = _run_extract(extractor, _FakeChat(), chunk.id)
    assert outcome.skipped is True
    assert outcome.reason == "extract_disabled"


def test_extract_chunk_persists_graph_with_chunk_id() -> None:
    kb = _kb(
        tenant_id=1,
        extract_config={
            "enabled": True,
            "text": "sample",
            "nodes": [{"name": "A"}],
            "relations": [{"node1": "A", "node2": "B", "type": "link"}],
        },
    )
    chunk = _chunk_row(tenant_id=1, knowledge_base_id=kb.id, knowledge_id="k-1", content="x")
    graph_store = _FakeGraphStore()
    extractor = _extractor(kb=kb, chunk=chunk, knowledge=None, graph_store=graph_store)
    chat = _FakeChat(content='[{"entity": "A"}]')
    outcome = _run_extract(extractor, chat, chunk.id)
    assert outcome.skipped is False
    assert outcome.node_count == 1
    assert len(graph_store.added) == 1
    kb_id, k_id, graphs = graph_store.added[0]
    assert kb_id == kb.id
    assert k_id == "k-1"
    assert graphs[0].node[0].chunks == [chunk.id]


def test_extract_chunk_skips_when_chunk_disappears(monkeypatch: pytest.MonkeyPatch) -> None:
    kb = _kb(tenant_id=1, extract_config={"enabled": True})
    chunk = _chunk_row(tenant_id=1, knowledge_base_id=kb.id, knowledge_id="k-1", content="x")
    extractor = _extractor(kb=kb, chunk=chunk, knowledge=None)

    async def disappear(*args: object, **kwargs: object) -> None:
        extractor._chunk_repo.rows.pop(chunk.id, None)  # type: ignore[attr-defined]

    monkeypatch.setattr(
        "src.core.knowledge.documents.chunk_extract.StructureExtractor.extract",
        disappear,
    )
    outcome = _run_extract(extractor, _FakeChat(), chunk.id)
    assert outcome.skipped is True
    assert outcome.reason == "chunk_disappeared"


# ── datatable_summary: pure helpers ────────────────────────────────────


def test_table_schema_description() -> None:
    schema = TableSchema(
        table_name="k_orders",
        columns=[TableColumn(name="id", type="BIGINT"), TableColumn(name="name", type="VARCHAR")],
        row_count=42,
    )
    desc = schema.description()
    assert "Table name: k_orders" in desc
    assert "Columns: 2" in desc
    assert "Rows: 42" in desc
    assert "- id (BIGINT)" in desc
    assert "- name (VARCHAR)" in desc


def test_build_sample_data_description_limits_rows() -> None:
    rows: list[dict[str, object]] = [
        {"id": 1, "name": "a"},
        {"id": 2, "name": "b"},
        {"id": 3, "name": "c"},
    ]
    desc = build_sample_data_description(rows, 2)
    assert "Sample data (first 2 rows):" in desc
    assert '"name": "a"' in desc
    assert '"name": "c"' not in desc


def test_generate_table_description_wraps_output() -> None:
    chat = _FakeChat(content="a sales table")
    desc = None

    import asyncio

    async def run() -> str:
        nonlocal desc
        desc = await generate_table_description(chat, "k_orders", "schema", "sample", "custom")
        return desc

    asyncio.run(run())
    assert desc is not None
    assert desc.startswith("# Table Summary\n\nTable name: k_orders")
    assert "a sales table" in desc
    assert chat.calls[0][1] is not None
    assert chat.calls[0][1].max_tokens == 512
    # Custom instructions are folded into the user prompt.
    assert "<table_metadata_business_instructions>" in chat.calls[0][0][0].content


def test_generate_column_descriptions_wraps_output() -> None:
    chat = _FakeChat(content="id is a key")
    desc = None

    import asyncio

    async def run() -> str:
        nonlocal desc
        desc = await generate_column_descriptions(chat, "k_orders", "schema", "sample", "")
        return desc

    asyncio.run(run())
    assert desc is not None
    assert desc.startswith("# Table Column Information\n\nTable name: k_orders")
    assert chat.calls[0][1] is not None
    assert chat.calls[0][1].max_tokens == 2048


def test_build_datatable_chunks_links_pair() -> None:
    chunks = build_datatable_chunks(
        tenant_id=1,
        knowledge_id="k-1",
        knowledge_base_id="kb-1",
        table_description="summary",
        column_description="columns",
        now=_NOW,
    )
    assert [c.chunk_type for c in chunks] == [CHUNK_TYPE_TABLE_SUMMARY, CHUNK_TYPE_TABLE_COLUMN]
    assert [c.chunk_index for c in chunks] == [0, 1]
    assert [c.status for c in chunks] == [CHUNK_STATUS_STORED, CHUNK_STATUS_STORED]
    assert chunks[0].next_chunk_id == chunks[1].id
    assert chunks[1].pre_chunk_id == chunks[0].id
    assert chunks[1].parent_chunk_id == chunks[0].id


def test_resolve_table_metadata_instructions() -> None:
    kb = _kb(tenant_id=1, chunking_config={"table_metadata_instructions": "base rules"})
    assert resolve_table_metadata_instructions(kb, None) == "base rules"
    assert (
        resolve_table_metadata_instructions(
            kb, {"chunking_config": {"table_metadata_instructions": "override rules"}}
        )
        == "override rules"
    )
    # Blank override keeps the knowledge-base value.
    assert resolve_table_metadata_instructions(kb, {"chunking_config": {}}) == "base rules"


# ── datatable_summary: engine resolution ───────────────────────────────


def test_tenant_effective_engines_uses_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    tenant = _tenant(id=1, engines=[RetrieverEngineEntry(retriever_type="vector", retriever_engine_type="qdrant")])
    carrier = TenantEffectiveEngines(tenant)
    engines = carrier.get_effective_engines()
    assert len(engines) == 1
    assert engines[0].retriever_type == RetrieverType.VECTOR
    assert engines[0].retriever_engine_type == RetrieverEngineType.QDRANT


def test_tenant_effective_engines_falls_back_to_driver_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RETRIEVE_DRIVER", "postgres,sqlite")
    carrier = TenantEffectiveEngines(_tenant(id=1))
    engines = carrier.get_effective_engines()
    pairs = {(e.retriever_type, e.retriever_engine_type) for e in engines}
    assert (RetrieverType.KEYWORDS, RetrieverEngineType.POSTGRES) in pairs
    assert (RetrieverType.VECTOR, RetrieverEngineType.POSTGRES) in pairs
    assert (RetrieverType.VECTOR, RetrieverEngineType.SQLITE) in pairs


def test_resolve_datatable_engine_unbound_builds_composite(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETRIEVE_DRIVER", "sqlite")
    service = _FakeEngineService(engine_type=RetrieverEngineType.SQLITE)
    registry = _FakeRegistry(service)
    ctx, engine = None, None

    import asyncio

    async def run() -> None:
        nonlocal ctx, engine
        ctx, engine = await resolve_datatable_engine(
            registry=registry,
            ownership=_FakeOwnership(),
            tenant_id=1,
            tenant_info=_tenant(id=1),
            vector_store_id=None,
        )

    asyncio.run(run())
    assert ctx is not None and isinstance(ctx, SummaryContext)
    assert engine is not None
    assert engine.support_retriever(RetrieverType.KEYWORDS)


class _FakeOwnership:
    """Tenant-store ownership double accepting every store."""

    async def store_owned_by(self, ctx: Context, store_id: str, tenant_id: int) -> bool:
        return True


# ── datatable_summary: indexing + orchestration ────────────────────────


def _summary_engine(service: _FakeEngineService) -> CompositeRetrieveEngine:
    return new_composite_retrieve_engine(
        _FakeRegistry(service),
        [
            RetrieverEngineParams(
                retriever_engine_type=RetrieverEngineType.SQLITE,
                retriever_type=RetrieverType.KEYWORDS,
            ),
            RetrieverEngineParams(
                retriever_engine_type=RetrieverEngineType.SQLITE,
                retriever_type=RetrieverType.VECTOR,
            ),
        ],
    )


def test_index_to_vector_db_persists_indexes_and_settles_status() -> None:
    service = _FakeEngineService(engine_type=RetrieverEngineType.SQLITE)
    engine = _summary_engine(service)
    chunk_service = _FakeChunkService()
    chunks = build_datatable_chunks(
        tenant_id=1,
        knowledge_id="k-1",
        knowledge_base_id="kb-1",
        table_description="summary",
        column_description="columns",
        now=_NOW,
    )

    import asyncio

    asyncio.run(
        index_to_vector_db(
            ctx=_ctx(),
            engine=engine,
            embedder=_FakeEmbedder(),
            chunks=chunks,
            chunk_service=chunk_service,
        )
    )
    assert {c.id for c in chunk_service.created} == {c.id for c in chunks}
    assert {c.id for c in chunk_service.updated} == {c.id for c in chunks}
    assert all(c.status == CHUNK_STATUS_INDEXED for c in chunk_service.updated)
    assert {info.source_id for info in service.indexed} == {c.id for c in chunks}
    assert service.batch_calls[0][0] == 2


def test_process_datatable_summary_happy_path() -> None:
    tenant_id = 1
    kb = _kb(tenant_id=tenant_id)
    row = _doc_row(tenant_id=tenant_id, knowledge_base_id=kb.id, file_type="csv")
    schema = TableSchema(table_name="k_orders", columns=[TableColumn(name="id", type="BIGINT")], row_count=5)
    chat = _FakeChat(content="fixture summary")
    service = _FakeEngineService(engine_type=RetrieverEngineType.SQLITE)
    engine = _summary_engine(service)
    chunk_service = _FakeChunkService()
    knowledge_repo = _FakeKnowledgeRepo()
    knowledge_repo.seed(row)

    import asyncio

    result = asyncio.run(
        process_datatable_summary(
            ctx=_ctx(),
            tenant_id=tenant_id,
            knowledge_id=row.id,
            chat=chat,
            embedder=_FakeEmbedder(),
            engine=engine,
            knowledge_repo=knowledge_repo,
            kb_service=_FakeKBService(kb),
            chunk_service=chunk_service,
            table_tool=_FakeTableTool(schema, [{"id": 1}]),
        )
    )
    assert isinstance(result, DataTableSummaryResult)
    assert result.summary_chunk_id != result.column_chunk_id
    assert len(chunk_service.created) == 2
    assert all(c.status == CHUNK_STATUS_INDEXED for c in chunk_service.updated)
    assert len(chat.calls) == 2
    assert len(service.indexed) == 2


def test_process_datatable_summary_rejects_non_spreadsheet() -> None:
    row = _doc_row(tenant_id=1, knowledge_base_id="kb-1", file_type="pdf")
    knowledge_repo = _FakeKnowledgeRepo()
    knowledge_repo.seed(row)
    import asyncio

    with pytest.raises(ValidationError):
        asyncio.run(
            process_datatable_summary(
                ctx=_ctx(),
                tenant_id=1,
                knowledge_id=row.id,
                chat=_FakeChat(),
                embedder=_FakeEmbedder(),
                engine=_summary_engine(_FakeEngineService()),
                knowledge_repo=knowledge_repo,
                kb_service=_FakeKBService(_kb(tenant_id=1)),
                chunk_service=_FakeChunkService(),
                table_tool=_FakeTableTool(
                    TableSchema(table_name="t", row_count=0), []
                ),
            )
        )


def test_process_datatable_summary_raises_on_missing_document() -> None:
    import asyncio

    with pytest.raises(NotFoundError):
        asyncio.run(
            process_datatable_summary(
                ctx=_ctx(),
                tenant_id=1,
                knowledge_id="nope",
                chat=_FakeChat(),
                embedder=_FakeEmbedder(),
                engine=_summary_engine(_FakeEngineService()),
                knowledge_repo=_FakeKnowledgeRepo(),
                kb_service=_FakeKBService(_kb(tenant_id=1)),
                chunk_service=_FakeChunkService(),
                table_tool=_FakeTableTool(
                    TableSchema(table_name="t", row_count=0), []
                ),
            )
        )


def test_process_datatable_summary_index_failure_rolls_back() -> None:
    tenant_id = 1
    kb = _kb(tenant_id=tenant_id)
    row = _doc_row(tenant_id=tenant_id, knowledge_base_id=kb.id, file_type="csv")
    knowledge_repo = _FakeKnowledgeRepo()
    knowledge_repo.seed(row)
    service = _FakeEngineService(
        engine_type=RetrieverEngineType.SQLITE,
        batch_error=RuntimeError("engine down"),
    )
    chunk_service = _FakeChunkService()
    import asyncio

    with pytest.raises(RuntimeError, match="engine down"):
        asyncio.run(
            process_datatable_summary(
                ctx=_ctx(),
                tenant_id=tenant_id,
                knowledge_id=row.id,
                chat=_FakeChat(content="summary"),
                embedder=_FakeEmbedder(),
                engine=_summary_engine(service),
                knowledge_repo=knowledge_repo,
                kb_service=_FakeKBService(kb),
                chunk_service=chunk_service,
                table_tool=_FakeTableTool(
                    TableSchema(table_name="t", columns=[TableColumn(name="id", type="BIGINT")], row_count=1),
                    [{"id": 1}],
                ),
            )
        )
    # The two chunks were created then deleted; the row is marked failed.
    assert len(chunk_service.created) == 2
    assert len(chunk_service.deleted_ids) == 2
    failed = knowledge_repo.rows[row.id]
    assert failed.parse_status == PARSE_STATUS_FAILED
    assert "engine down" in (failed.error_message or "")


def test_cleanup_on_failure_marks_failed_and_drops_index() -> None:
    tenant_id = 1
    row = _doc_row(tenant_id=tenant_id, knowledge_base_id="kb-1", file_type="csv")
    knowledge_repo = _FakeKnowledgeRepo()
    knowledge_repo.seed(row)
    chunks = build_datatable_chunks(
        tenant_id=tenant_id,
        knowledge_id=row.id,
        knowledge_base_id="kb-1",
        table_description="s",
        column_description="c",
        now=_NOW,
    )
    service = _FakeEngineService(engine_type=RetrieverEngineType.SQLITE)
    engine = _summary_engine(service)
    chunk_service = _FakeChunkService()
    import asyncio

    asyncio.run(
        cleanup_on_failure(
            ctx=_ctx(),
            knowledge_repo=knowledge_repo,
            chunk_service=chunk_service,
            engine=engine,
            embedder=_FakeEmbedder(dimensions=8),
            row=row,
            chunks=chunks,
            error=RuntimeError("boom"),
        )
    )
    assert knowledge_repo.rows[row.id].parse_status == PARSE_STATUS_FAILED
    assert len(chunk_service.deleted_ids) == 2
    assert set(service.deleted_sources) == {c.id for c in chunks}


# ── Integration against the real schema ────────────────────────────────


@pytest_asyncio.fixture
async def db_session() -> AsyncIterator[AsyncSession]:
    """Per-test session against the real applied schema; skips without a DB."""
    reset_settings_cache()
    engine = DatabaseEngine(url=get_settings().database_url, poolclass=NullPool)
    try:
        await engine.prewarm()
    except Exception as exc:
        await engine.close()
        pytest.skip(f"integration database unavailable: {exc}")
    async with engine.session_factory() as s:
        yield s
        await s.rollback()
    await engine.close()


async def test_integration_chunk_extract_round_trip(db_session: AsyncSession) -> None:
    tenant_id = _int32_tenant_id()
    kb_service = KBService(kb_repo=KnowledgeBaseRepository(db_session))
    kb = await kb_service.create_knowledge_base(
        tenant_id=tenant_id,
        name="extract-docs",
        kb_type="document",
        extract_config={
            "enabled": True,
            "text": "sample",
            "nodes": [{"name": "Romeo"}],
            "relations": [{"node1": "Romeo", "node2": "Juliet", "type": "loves"}],
        },
    )
    knowledge_service = KnowledgeService(knowledge_repo=KnowledgeRepository(db_session))
    doc = await knowledge_service.create_document(
        tenant_id=tenant_id,
        knowledge_base_id=kb.id,
        type="manual",
        title="Extract me",
        source="manual",
        parse_status=PARSE_STATUS_COMPLETED,
    )
    chunk_repo = ChunkRepository(db_session)
    chunk = _chunk_row(
        tenant_id=tenant_id,
        knowledge_base_id=kb.id,
        knowledge_id=doc.id,
        content="Romeo loves Juliet.",
    )
    stored = (await chunk_repo.create_many([chunk]))[0]

    chat = _FakeChat(
        content='```json\n[{"entity": "Romeo", "entity_attributes": ["suitor"]}, {"entity1": "Romeo", "entity2": "Juliet", "relation": "loves"}]\n```'
    )
    graph_store = _FakeGraphStore()
    extractor = ChunkExtractor(
        chunk_repo=chunk_repo,
        knowledge_repo=KnowledgeRepository(db_session),
        kb_service=kb_service,
        graph_store=graph_store,
        graph_enabled=True,
        default_description="Extract entities and relations from the text.",
    )
    outcome = await extractor.extract_chunk(
        ctx=_ctx(),
        tenant_id=tenant_id,
        chunk_id=stored.id,
        chat=chat,
        knowledge_id=doc.id,
    )
    assert outcome.skipped is False
    assert outcome.node_count == 2
    assert outcome.relation_count == 1
    assert len(graph_store.added) == 1
    _kb_id, _k_id, graphs = graph_store.added[0]
    assert {node.name for node in graphs[0].node} == {"Romeo", "Juliet"}
    assert graphs[0].node[0].chunks == [stored.id]


async def test_integration_datatable_summary_round_trip(db_session: AsyncSession) -> None:
    tenant_id = _int32_tenant_id()
    kb_service = KBService(kb_repo=KnowledgeBaseRepository(db_session))
    kb = await kb_service.create_knowledge_base(
        tenant_id=tenant_id,
        name="summary-docs",
        kb_type="document",
    )
    knowledge_service = KnowledgeService(knowledge_repo=KnowledgeRepository(db_session))
    doc = await knowledge_service.create_document(
        tenant_id=tenant_id,
        knowledge_base_id=kb.id,
        type="file",
        title="orders.csv",
        source="file.csv",
        file_name="orders.csv",
        file_type="csv",
        parse_status=PARSE_STATUS_COMPLETED,
    )
    schema = TableSchema(
        table_name="k_orders",
        columns=[TableColumn(name="id", type="BIGINT"), TableColumn(name="total", type="VARCHAR")],
        row_count=10,
    )
    service = _FakeEngineService(engine_type=RetrieverEngineType.SQLITE)
    engine = _summary_engine(service)
    chunk_service = ChunkService(chunk_repo=ChunkRepository(db_session))

    result = await process_datatable_summary(
        ctx=_ctx(),
        tenant_id=tenant_id,
        knowledge_id=doc.id,
        chat=_FakeChat(content="orders summary"),
        embedder=_FakeEmbedder(),
        engine=engine,
        knowledge_repo=KnowledgeRepository(db_session),
        kb_service=kb_service,
        chunk_service=chunk_service,
        table_tool=_FakeTableTool(schema, [{"id": 1, "total": "10"}]),
    )
    assert result.summary_chunk_id != result.column_chunk_id
    stored_chunks = await ChunkRepository(db_session).find_all_by_column_values(
        {"tenant_id": tenant_id, "knowledge_id": doc.id}
    )
    assert {c.chunk_type for c in stored_chunks} == {
        CHUNK_TYPE_TABLE_SUMMARY,
        CHUNK_TYPE_TABLE_COLUMN,
    }
    assert all(c.status == CHUNK_STATUS_INDEXED for c in stored_chunks)
    assert len(service.indexed) == 2
