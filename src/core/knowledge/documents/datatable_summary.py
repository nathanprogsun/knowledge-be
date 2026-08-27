"""Data-table summary generation and indexing.

``process_datatable_summary`` turns a CSV / Excel knowledge item into two
indexed chunks: a ``table_summary`` chunk carrying the model-generated
table description and a ``table_column`` chunk carrying the per-column
descriptions. The flow mirrors the reference data-table summary service:

- the knowledge file is validated as a spreadsheet and loaded into a table
  through an injected ``TableDataTool`` seam (the reference backs this with
  an in-memory SQL engine; that dependency lands with the tool layer, so
  the core module depends only on the seam);
- a schema description and a JSON-serialised sample of the first rows are
  fed to the chat seam (``src.ai.llm.Chat``) to produce the table and
  column descriptions;
- the two chunks are persisted and indexed through the composite retrieval
  engine (``CompositeRetrieveEngine.batch_index``); a failed index rolls
  back the chunks and the vector entries and marks the knowledge failed.

The retrieval engine is resolved through the shared KB resolver, so an
unbound knowledge base falls back to the tenant's effective engines (or
the driver defaults) while a store-bound one is ownership verified. The
tenant carrier adapter lives here because the wiring layer that would
otherwise supply it is not merged yet.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from src.ai.embedding import Context, Embedder
from src.ai.llm import Chat, ChatOptions, Message
from src.ai.retrieval.base import RetrieveEngineRegistry
from src.ai.retrieval.composite import CompositeRetrieveEngine
from src.ai.retrieval.kb_engine_resolver import (
    TenantEnginesCarrier,
    create_retrieve_engine_for_kb,
)
from src.ai.retrieval.ownership import TenantStoreOwnership
from src.ai.retrieval.types import (
    IndexInfo,
    RetrieverEngineParams,
    RetrieverEngineType,
    RetrieverType,
    SourceType,
)
from src.app_logging import logger
from src.common.exception import NotFoundError, ValidationError
from src.common.json import JsonObject, JsonValue
from src.core.knowledge.chunks.service.chunk_service import ChunkService
from src.core.knowledge.chunks.types import (
    CHUNK_STATUS_INDEXED,
    CHUNK_STATUS_STORED,
    CHUNK_TYPE_TABLE_COLUMN,
    CHUNK_TYPE_TABLE_SUMMARY,
)
from src.core.knowledge.documents.chunk_extract import (
    COLUMN_DESCRIPTIONS_PROMPT_TEMPLATE,
    TABLE_DESCRIPTION_PROMPT_TEMPLATE,
    append_custom_prompt_instructions,
)
from src.core.knowledge.documents.types import PARSE_STATUS_FAILED
from src.core.knowledge.knowledge_bases.service.kb_service import KBService
from src.core.knowledge.knowledge_bases.types import KnowledgeBaseInfo
from src.core.tenants.types import RetrieverEngineEntry, TenantInfo
from src.db.dao.knowledge_repository import KnowledgeRepository
from src.db.models.chunk import Chunk
from src.db.models.knowledge import Document

_NOT_FOUND_CODE = "knowledge.not_found"

#: Knowledge-base type used for the vector-index namespace of table chunks.
KNOWLEDGE_BASE_TYPE_DOCUMENT = "document"

#: Sample size fed to the description prompts.
SAMPLE_ROW_LIMIT = 10

#: Bounds the error message written to the documents row on failure.
_MAX_ERROR_MESSAGE_CHARS = 2048

#: Sampling parameters for the two description calls (deliberately low).
_DESCRIPTION_TEMPERATURE = 0.3
_TABLE_DESCRIPTION_MAX_TOKENS = 512
_COLUMN_DESCRIPTIONS_MAX_TOKENS = 2048

#: Accepted spreadsheet file types for the summary flow.
_DATA_TABLE_FILE_TYPES: frozenset[str] = frozenset({"csv", "xlsx", "xls"})


def _engine_pair(engine: RetrieverEngineType, retriever: RetrieverType) -> RetrieverEngineParams:
    """Build one engine/retriever selection pair."""
    return RetrieverEngineParams(
        retriever_engine_type=engine,
        retriever_type=retriever,
    )


#: ``RETRIEVE_DRIVER`` -> engine/retriever pairs (reference tenant defaults).
_DRIVER_ENGINES: dict[str, list[RetrieverEngineParams]] = {
    "postgres": [
        _engine_pair(RetrieverEngineType.POSTGRES, RetrieverType.KEYWORDS),
        _engine_pair(RetrieverEngineType.POSTGRES, RetrieverType.VECTOR),
    ],
    "elasticsearch_v7": [
        _engine_pair(RetrieverEngineType.ELASTICSEARCH, RetrieverType.KEYWORDS),
    ],
    "elasticsearch_v8": [
        _engine_pair(RetrieverEngineType.ELASTICSEARCH, RetrieverType.KEYWORDS),
        _engine_pair(RetrieverEngineType.ELASTICSEARCH, RetrieverType.VECTOR),
    ],
    "qdrant": [
        _engine_pair(RetrieverEngineType.QDRANT, RetrieverType.KEYWORDS),
        _engine_pair(RetrieverEngineType.QDRANT, RetrieverType.VECTOR),
    ],
    "milvus": [
        _engine_pair(RetrieverEngineType.MILVUS, RetrieverType.VECTOR),
        _engine_pair(RetrieverEngineType.MILVUS, RetrieverType.KEYWORDS),
    ],
    "weaviate": [
        _engine_pair(RetrieverEngineType.WEAVIATE, RetrieverType.KEYWORDS),
        _engine_pair(RetrieverEngineType.WEAVIATE, RetrieverType.VECTOR),
    ],
    "doris": [
        _engine_pair(RetrieverEngineType.DORIS, RetrieverType.KEYWORDS),
        _engine_pair(RetrieverEngineType.DORIS, RetrieverType.VECTOR),
    ],
    "sqlite": [
        _engine_pair(RetrieverEngineType.SQLITE, RetrieverType.KEYWORDS),
        _engine_pair(RetrieverEngineType.SQLITE, RetrieverType.VECTOR),
    ],
    "opensearch": [
        _engine_pair(RetrieverEngineType.OPENSEARCH, RetrieverType.KEYWORDS),
        _engine_pair(RetrieverEngineType.OPENSEARCH, RetrieverType.VECTOR),
    ],
}


def _parse_retrieve_driver(raw: str) -> list[str]:
    """Split the ``RETRIEVE_DRIVER`` value into trimmed driver names."""
    if not raw:
        return []
    return [segment.strip() for segment in raw.split(",") if segment.strip()]


def _to_retriever_engine_params(
    entries: list[RetrieverEngineEntry],
) -> list[RetrieverEngineParams]:
    """Coerce the tenant's stored engine entries, dropping unknown values."""
    result: list[RetrieverEngineParams] = []
    for entry in entries:
        try:
            engine_type = RetrieverEngineType(entry.retriever_engine_type)
            retriever_type = RetrieverType(entry.retriever_type)
        except ValueError:
            continue
        result.append(
            RetrieverEngineParams(
                retriever_engine_type=engine_type,
                retriever_type=retriever_type,
            )
        )
    return result


def default_retriever_engines() -> list[RetrieverEngineParams]:
    """Return the driver-default engine/retriever pairs.

    Mirrors the reference ``GetDefaultRetrieverEngines``: engines are
    collected from ``RETRIEVE_DRIVER`` in order and deduplicated by
    ``(retriever_type, engine_type)``.
    """
    seen: set[tuple[RetrieverType, RetrieverEngineType]] = set()
    result: list[RetrieverEngineParams] = []
    for driver in _parse_retrieve_driver(os.getenv("RETRIEVE_DRIVER", "")):
        for params in _DRIVER_ENGINES.get(driver, ()):
            key = (params.retriever_type, params.retriever_engine_type)
            if key not in seen:
                seen.add(key)
                result.append(params)
    return result


class TenantEffectiveEngines:
    """Adapts a tenant row to the effective-engine carrier.

    A tenant with configured retriever engines uses them; an unconfigured
    tenant falls back to the driver defaults, mirroring the reference
    ``GetEffectiveEngines`` semantics.
    """

    def __init__(self, tenant: TenantInfo) -> None:
        self._tenant = tenant

    def get_effective_engines(self) -> list[RetrieverEngineParams]:
        configured = self._tenant.retriever_engines.engines
        if configured:
            return _to_retriever_engine_params(configured)
        return default_retriever_engines()


@dataclass
class SummaryContext:
    """Embedding-context carrier used by datatable-summary engine resolution."""

    is_background_task: bool = True
    tenant_info: TenantEnginesCarrier | None = None


async def resolve_datatable_engine(
    *,
    registry: RetrieveEngineRegistry,
    ownership: TenantStoreOwnership,
    tenant_id: int,
    tenant_info: TenantInfo,
    vector_store_id: str | None,
) -> tuple[SummaryContext, CompositeRetrieveEngine]:
    """Resolve the knowledge base's effective retrieval engine.

    An unbound knowledge base falls back to the tenant's effective engines
    from the context; a store-bound one is ownership verified and resolved
    through the registry.
    """
    ctx = SummaryContext(
        is_background_task=True,
        tenant_info=TenantEffectiveEngines(tenant_info),
    )
    engine = await create_retrieve_engine_for_kb(
        ctx, registry, ownership, tenant_id, vector_store_id
    )
    return ctx, engine


# ── Table loading seam (reference data-analysis tool, deferred) ────────


@runtime_checkable
class TableDataTool(Protocol):
    """Loads a knowledge file into a queryable table and reads sample rows.

    The reference implementation backs this with an in-memory SQL engine
    that also powers the data-analysis tool; the core module only depends
    on the seam.
    """

    async def load_from_knowledge(self, *, knowledge: Document) -> TableSchema:
        """Load the knowledge's file and return the resulting table schema."""
        ...

    async def sample_rows(self, *, table_name: str, limit: int) -> list[dict[str, JsonValue]]:
        """Return up to ``limit`` rows from ``table_name`` as JSON objects."""
        ...

    def cleanup(self) -> None:
        """Drop any session-scoped tables created by the load."""
        ...


@dataclass(frozen=True)
class TableColumn:
    """One column of a loaded data table."""

    name: str
    type: str
    nullable: str = ""


@dataclass(frozen=True)
class TableSchema:
    """Schema of a loaded data table."""

    table_name: str
    columns: list[TableColumn] = field(default_factory=list)
    row_count: int = 0
    metadata: JsonObject | None = None

    def description(self) -> str:
        """Render the schema as the model-facing description."""
        lines = [
            f"Table name: {self.table_name}",
            f"Columns: {len(self.columns)}",
            f"Rows: {self.row_count}",
            "",
            "Column info:",
        ]
        for column in self.columns:
            lines.append(f"- {column.name} ({column.type})")
        return "\n".join(lines)


@dataclass(frozen=True)
class DataTableSummaryPayload:
    """Payload of a data-table summary task (reference ``extract.go``)."""

    tenant_id: int
    knowledge_id: str
    summary_model: str = ""
    embedding_model: str = ""


@dataclass(frozen=True)
class DataTableSummaryResult:
    """Outcome of one data-table summary run."""

    knowledge_id: str
    summary_chunk_id: str
    column_chunk_id: str


# ── Prompt helpers ─────────────────────────────────────────────────────


def build_sample_data_description(
    rows: list[dict[str, JsonValue]],
    max_rows: int,
) -> str:
    """Render the first rows as one JSON object per line."""
    builder = [f"Sample data (first {max_rows} rows):"]
    for row in rows[:max_rows]:
        builder.append(json.dumps(row, ensure_ascii=False))
    return "\n".join(builder) + "\n"


def resolve_table_metadata_instructions(
    kb: KnowledgeBaseInfo,
    process_overrides: JsonObject | None,
) -> str:
    """Resolve the effective table-metadata instructions after the merge.

    The knowledge base chunking config is the default; a per-document
    ``process_overrides.chunking_config`` entry replaces it when its value
    is non-blank.
    """
    base = kb.chunking_config if isinstance(kb.chunking_config, dict) else {}
    raw = base.get("table_metadata_instructions")
    instructions = raw if isinstance(raw, str) else ""
    if isinstance(process_overrides, dict):
        override_chunking = process_overrides.get("chunking_config")
        if isinstance(override_chunking, dict):
            value = override_chunking.get("table_metadata_instructions")
            if isinstance(value, str) and value.strip() != "":
                instructions = value
    return instructions


async def generate_table_description(
    chat: Chat,
    table_name: str,
    schema_desc: str,
    sample_desc: str,
    custom_instructions: str = "",
) -> str:
    """Generate the whole-table metadata description via the chat seam."""
    prompt = TABLE_DESCRIPTION_PROMPT_TEMPLATE % (table_name, schema_desc, sample_desc)
    prompt = append_custom_prompt_instructions(prompt, custom_instructions, "table_metadata")
    response = await chat.chat(
        [Message(role="user", content=prompt)],
        ChatOptions(
            temperature=_DESCRIPTION_TEMPERATURE,
            max_tokens=_TABLE_DESCRIPTION_MAX_TOKENS,
            thinking=False,
        ),
    )
    return f"# Table Summary\n\nTable name: {table_name}\n\n{response.content}"


async def generate_column_descriptions(
    chat: Chat,
    table_name: str,
    schema_desc: str,
    sample_desc: str,
    custom_instructions: str = "",
) -> str:
    """Generate the per-column descriptions in one chat call."""
    prompt = COLUMN_DESCRIPTIONS_PROMPT_TEMPLATE % (table_name, schema_desc, sample_desc)
    prompt = append_custom_prompt_instructions(prompt, custom_instructions, "table_metadata")
    response = await chat.chat(
        [Message(role="user", content=prompt)],
        ChatOptions(
            temperature=_DESCRIPTION_TEMPERATURE,
            max_tokens=_COLUMN_DESCRIPTIONS_MAX_TOKENS,
            thinking=False,
        ),
    )
    return f"# Table Column Information\n\nTable name: {table_name}\n\n{response.content}"


# ── Chunk construction and indexing ────────────────────────────────────


def build_datatable_chunks(
    *,
    tenant_id: int,
    knowledge_id: str,
    knowledge_base_id: str,
    table_description: str,
    column_description: str,
    now: datetime,
) -> list[Chunk]:
    """Build the summary chunk and the linked column chunk.

    The summary chunk (index 0) is the parent of the column chunk (index
    1); the pair is prev/next linked like any text run.
    """
    summary_id = str(uuid.uuid4())
    column_id = str(uuid.uuid4())
    summary_chunk = Chunk(
        id=summary_id,
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        knowledge_id=knowledge_id,
        content=table_description,
        chunk_index=0,
        is_enabled=True,
        start_at=0,
        end_at=0,
        pre_chunk_id=None,
        next_chunk_id=column_id,
        chunk_type=CHUNK_TYPE_TABLE_SUMMARY,
        parent_chunk_id=None,
        status=CHUNK_STATUS_STORED,
        created_at=now,
        updated_at=now,
    )
    column_chunk = Chunk(
        id=column_id,
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        knowledge_id=knowledge_id,
        content=column_description,
        chunk_index=1,
        is_enabled=True,
        start_at=0,
        end_at=0,
        pre_chunk_id=summary_id,
        next_chunk_id=None,
        chunk_type=CHUNK_TYPE_TABLE_COLUMN,
        parent_chunk_id=summary_id,
        status=CHUNK_STATUS_STORED,
        created_at=now,
        updated_at=now,
    )
    return [summary_chunk, column_chunk]


def build_datatable_index_infos(chunks: list[Chunk]) -> list[IndexInfo]:
    """Build one index entry per table chunk, referencing the chunk row."""
    return [
        IndexInfo(
            content=chunk.content,
            source_id=chunk.id,
            source_type=SourceType.CHUNK,
            chunk_id=chunk.id,
            knowledge_id=chunk.knowledge_id,
            knowledge_base_id=chunk.knowledge_base_id,
            is_enabled=True,
        )
        for chunk in chunks
    ]


async def index_to_vector_db(
    *,
    ctx: Context,
    engine: CompositeRetrieveEngine,
    embedder: Embedder,
    chunks: list[Chunk],
    chunk_service: ChunkService,
) -> None:
    """Persist the chunks, index them, and settle their status to indexed."""
    if not chunks:
        return
    await chunk_service.create_chunks(chunks=chunks)
    await engine.batch_index(ctx, embedder, build_datatable_index_infos(chunks))
    indexed = [chunk.model_copy(update={"status": CHUNK_STATUS_INDEXED}) for chunk in chunks]
    await chunk_service.update_chunks(chunks=indexed)


async def cleanup_on_failure(
    *,
    ctx: Context,
    knowledge_repo: KnowledgeRepository,
    chunk_service: ChunkService,
    engine: CompositeRetrieveEngine | None,
    embedder: Embedder | None,
    row: Document,
    chunks: list[Chunk],
    error: Exception,
) -> None:
    """Roll back a failed summary run: mark failed, drop chunks and vectors.

    Every step is best-effort — a cleanup failure must not mask the
    original index error, which the caller re-raises.
    """
    now = datetime.now(UTC)
    message = str(error)[:_MAX_ERROR_MESSAGE_CHARS]
    failed = row.model_copy(
        update={
            "parse_status": PARSE_STATUS_FAILED,
            "error_message": message,
            "updated_at": now,
        }
    )
    try:
        await knowledge_repo.update(failed)
    except Exception as exc:
        logger.warning("failed to mark knowledge {} failed: {}", row.id, exc)
    chunk_ids = [chunk.id for chunk in chunks]
    if chunk_ids:
        try:
            await chunk_service.delete_chunks(tenant_id=row.tenant_id, ids=chunk_ids)
        except Exception as exc:
            logger.warning("failed to delete datatable chunks: {}", exc)
        if engine is not None and embedder is not None:
            try:
                await engine.delete_by_source_id_list(
                    ctx,
                    chunk_ids,
                    embedder.get_dimensions(),
                    KNOWLEDGE_BASE_TYPE_DOCUMENT,
                )
            except Exception as exc:
                logger.warning("failed to delete datatable vector index: {}", exc)


# ── Orchestration ──────────────────────────────────────────────────────


def _process_overrides_of(row: Document) -> JsonObject | None:
    """Read the per-upload process overrides from a document's metadata."""
    if not isinstance(row.metadata, dict):
        return None
    overrides = row.metadata.get("process_overrides")
    return overrides if isinstance(overrides, dict) else None


def _require_scope(*, tenant_id: int, knowledge_id: str) -> None:
    """Reject an invalid tenant/knowledge scope at the service boundary."""
    if not isinstance(tenant_id, int) or tenant_id <= 0:
        raise ValidationError(
            code="knowledge.tenant_required",
            message="tenant ID is required",
        )
    if not knowledge_id.strip():
        raise ValidationError(
            code="knowledge.id_required",
            message="knowledge ID is required",
        )


async def process_datatable_summary(
    *,
    ctx: Context,
    tenant_id: int,
    knowledge_id: str,
    chat: Chat,
    embedder: Embedder,
    engine: CompositeRetrieveEngine,
    knowledge_repo: KnowledgeRepository,
    kb_service: KBService,
    chunk_service: ChunkService,
    table_tool: TableDataTool,
) -> DataTableSummaryResult:
    """Generate and index a data-table summary for a knowledge item.

    Raises ``NotFoundError`` for an absent document and ``ValidationError``
    for an invalid scope or a non-spreadsheet file type. A failed index
    rolls back the freshly written chunks and vector entries before the
    error propagates.
    """
    _require_scope(tenant_id=tenant_id, knowledge_id=knowledge_id)
    row = await knowledge_repo.get_by_id(tenant_id, knowledge_id)
    if row is None:
        raise NotFoundError(code=_NOT_FOUND_CODE, message="knowledge not found")
    file_type = (row.file_type or "").lower()
    if file_type not in _DATA_TABLE_FILE_TYPES:
        raise ValidationError(
            code="datatable.unsupported_file_type",
            message=f"unsupported file type: {row.file_type}",
        )

    schema = await table_tool.load_from_knowledge(knowledge=row)
    rows = await table_tool.sample_rows(table_name=schema.table_name, limit=SAMPLE_ROW_LIMIT)
    schema_desc = schema.description()
    sample_desc = build_sample_data_description(rows, SAMPLE_ROW_LIMIT)

    kb = await kb_service.get_knowledge_base_by_id(knowledge_base_id=row.knowledge_base_id)
    custom_instructions = resolve_table_metadata_instructions(kb, _process_overrides_of(row))

    table_description = await generate_table_description(
        chat, schema.table_name, schema_desc, sample_desc, custom_instructions
    )
    column_description = await generate_column_descriptions(
        chat, schema.table_name, schema_desc, sample_desc, custom_instructions
    )

    chunks = build_datatable_chunks(
        tenant_id=tenant_id,
        knowledge_id=knowledge_id,
        knowledge_base_id=row.knowledge_base_id,
        table_description=table_description,
        column_description=column_description,
        now=datetime.now(UTC),
    )
    try:
        await index_to_vector_db(
            ctx=ctx,
            engine=engine,
            embedder=embedder,
            chunks=chunks,
            chunk_service=chunk_service,
        )
    except Exception as exc:
        await cleanup_on_failure(
            ctx=ctx,
            knowledge_repo=knowledge_repo,
            chunk_service=chunk_service,
            engine=engine,
            embedder=embedder,
            row=row,
            chunks=chunks,
            error=exc,
        )
        raise
    return DataTableSummaryResult(
        knowledge_id=knowledge_id,
        summary_chunk_id=chunks[0].id,
        column_chunk_id=chunks[1].id,
    )


__all__ = [
    "KNOWLEDGE_BASE_TYPE_DOCUMENT",
    "SAMPLE_ROW_LIMIT",
    "DataTableSummaryPayload",
    "DataTableSummaryResult",
    "SummaryContext",
    "TableColumn",
    "TableDataTool",
    "TableSchema",
    "TenantEffectiveEngines",
    "build_datatable_chunks",
    "build_datatable_index_infos",
    "build_sample_data_description",
    "cleanup_on_failure",
    "default_retriever_engines",
    "generate_column_descriptions",
    "generate_table_description",
    "index_to_vector_db",
    "process_datatable_summary",
    "resolve_datatable_engine",
    "resolve_table_metadata_instructions",
]
