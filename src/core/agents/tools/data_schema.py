"""Data-schema tool: table schema information for a data document.

``data_schema`` returns the schema of a CSV / Excel document that was
indexed into the data-analysis engine. The schema is reconstructed from
the ``table_summary`` and ``table_column`` chunks the document pipeline
produced, so the tool never reads the engine directly — it only needs the
document row and a chunk listing seam.

Scope enforcement mirrors the sibling retrieval tools: when the tool is
constructed with search targets, the target document must be authorized
against the session scope; otherwise the un-scoped document lookup is
used (for callers that already own their scope).
"""

from __future__ import annotations

import json
import logging
from typing import Protocol, cast, runtime_checkable

from sqlalchemy import text
from sqlalchemy.engine import RowMapping
from sqlalchemy.ext.asyncio import AsyncSession

from src.ai.embedding.base import Context
from src.common.exception import ApplicationError
from src.common.json import JsonObject, SqlValue
from src.core.agents.tools.base import ToolDefinition, ToolResult
from src.core.agents.tools.scope_auth import (
    KnowledgeLookup,
    KnowledgeTagsFetcher,
    authorize_knowledge_in_search_targets,
)
from src.core.agents.tools.search_target import SearchTargets
from src.core.contracts.knowledge import Knowledge
from src.core.knowledge.chunks.types import (
    CHUNK_STATUS_DEFAULT,
    CHUNK_STATUS_INDEXED,
    CHUNK_TYPE_TABLE_COLUMN,
    CHUNK_TYPE_TABLE_SUMMARY,
)
from src.db.models.chunk import Chunk

logger = logging.getLogger(__name__)

#: Tool name constant (kept here to avoid a dependency cycle with base).
DATA_SCHEMA_TOOL_NAME = "data_schema"

DATA_SCHEMA_TOOL_DESCRIPTION = (
    "Use this tool to get the schema information of a CSV or Excel file "
    "loaded into DuckDB. It returns the table name, columns, and row count."
)

DATA_SCHEMA_TOOL_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "knowledge_id": {
                "type": "string",
                "description": "short dN document ID to query",
            },
        },
        "required": ["knowledge_id"],
    },
    ensure_ascii=False,
)

#: Chunk types that carry table schema fragments.
DEFAULT_SCHEMA_CHUNK_TYPES: tuple[str, ...] = (
    CHUNK_TYPE_TABLE_SUMMARY,
    CHUNK_TYPE_TABLE_COLUMN,
)

#: Page size for the schema chunk listing (a schema is small).
_SCHEMA_CHUNK_PAGE_SIZE = 100


def build_data_schema_definition() -> ToolDefinition:
    """Return the default tool definition for the data-schema tool."""
    return ToolDefinition(
        name=DATA_SCHEMA_TOOL_NAME,
        description=DATA_SCHEMA_TOOL_DESCRIPTION,
        parameters=DATA_SCHEMA_TOOL_SCHEMA,
    )


@runtime_checkable
class SchemaChunkStore(Protocol):
    """Lists the enabled schema chunks (summary + column) of a document."""

    async def list_schema_chunks(
        self,
        *,
        tenant_id: int,
        knowledge_id: str,
        page_size: int,
    ) -> list[Chunk]: ...


class SqlSchemaChunkStore:
    """``SchemaChunkStore`` implementation over the ``chunks`` table.

    Only enabled, live schema chunks are returned, ordered by ``chunk_index``
    so the summary fragment precedes its column detail.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list_schema_chunks(
        self,
        *,
        tenant_id: int,
        knowledge_id: str,
        page_size: int,
    ) -> list[Chunk]:
        limit = max(page_size, 1)
        rows_result = await self._session.execute(
            text(
                "select * from chunks "
                "where tenant_id = :tenant_id and knowledge_id = :knowledge_id "
                "and chunk_type in (:summary_type, :column_type) "
                "and status in (:status_default, :status_indexed) "
                "and is_enabled = true and deleted_at is null "
                "order by chunk_index asc limit :limit"
            ).bindparams(
                tenant_id=tenant_id,
                knowledge_id=knowledge_id,
                summary_type=CHUNK_TYPE_TABLE_SUMMARY,
                column_type=CHUNK_TYPE_TABLE_COLUMN,
                status_default=CHUNK_STATUS_DEFAULT,
                status_indexed=CHUNK_STATUS_INDEXED,
                limit=limit,
            )
        )
        return [
            _chunk_from_mapping(mapping) for mapping in rows_result.mappings().all()
        ]


def _chunk_from_mapping(mapping: RowMapping) -> Chunk:
    row = cast("dict[str, SqlValue]", dict(mapping))
    return cast("Chunk", Chunk.from_row(row))


class DataSchemaTool:
    """Returns the table schema fragments of one data document.

    ``chunk_types`` controls which chunk types are scanned for the summary
    and column content; the default covers the table schema chunks.
    """

    def __init__(
        self,
        *,
        definition: ToolDefinition,
        knowledge_service: KnowledgeLookup | None,
        chunk_store: SchemaChunkStore,
        search_targets: SearchTargets | None = None,
        tag_fetcher: KnowledgeTagsFetcher | None = None,
        chunk_types: tuple[str, ...] = DEFAULT_SCHEMA_CHUNK_TYPES,
    ) -> None:
        self._definition = definition
        self._knowledge_service = knowledge_service
        self._chunk_store = chunk_store
        self._search_targets = search_targets
        self._tag_fetcher = tag_fetcher
        self._chunk_types = chunk_types

    def name(self) -> str:
        return self._definition.name

    def description(self) -> str:
        return self._definition.description

    def parameters(self) -> str:
        return self._definition.parameters

    async def execute(self, ctx: Context, args: str) -> ToolResult:
        """Return the summary + column schema content for a document."""
        knowledge_id = _parse_input(args).strip()
        if not knowledge_id:
            return ToolResult(
                success=False,
                error="knowledge_id is required",
            )

        knowledge: Knowledge | None
        if self._search_targets is not None:
            try:
                knowledge = await authorize_knowledge_in_search_targets(
                    ctx,
                    self._search_targets,
                    knowledge_id,
                    self._knowledge_service,
                    self._tag_fetcher,
                )
            except ApplicationError as exc:
                return ToolResult(
                    success=False,
                    error=f"Failed to get knowledge '{knowledge_id}': {exc.message}",
                )
        else:
            knowledge = await self._load_knowledge(knowledge_id)
            if knowledge is None:
                return ToolResult(
                    success=False,
                    error=(
                        f"Failed to get knowledge '{knowledge_id}': "
                        "knowledge service returned an empty result"
                    ),
                )
        assert knowledge is not None

        try:
            chunks = await self._chunk_store.list_schema_chunks(
                tenant_id=knowledge.tenant_id,
                knowledge_id=knowledge_id,
                page_size=_SCHEMA_CHUNK_PAGE_SIZE,
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                error=f"Failed to list chunks for knowledge ID '{knowledge_id}': {exc}",
            )

        summary_content = ""
        column_content = ""
        for chunk in chunks:
            if chunk.chunk_type == CHUNK_TYPE_TABLE_SUMMARY:
                summary_content = chunk.content
            elif chunk.chunk_type == CHUNK_TYPE_TABLE_COLUMN:
                column_content = chunk.content

        if not summary_content or not column_content:
            return ToolResult(
                success=False,
                error=f"No table schema information found for knowledge ID '{knowledge_id}'",
            )

        output = f"{summary_content}\n\n{column_content}"
        data: JsonObject = {
            "summary": summary_content,
            "columns": column_content,
        }
        return ToolResult(success=True, output=output, data=data)

    async def _load_knowledge(self, knowledge_id: str) -> Knowledge | None:
        """Load a document by id alone (no scope filter, callers own scope)."""
        if self._knowledge_service is None:
            return None
        try:
            return await self._knowledge_service.get_document_by_id_only(id=knowledge_id)
        except Exception as exc:
            logger.warning(
                "[Tool][DataSchema] Failed to get knowledge '%s': %s", knowledge_id, exc
            )
            return None


def _parse_input(args: str) -> str:
    try:
        raw = json.loads(args)
    except json.JSONDecodeError:
        return ""
    if not isinstance(raw, dict):
        return ""
    value = raw.get("knowledge_id")
    return value if isinstance(value, str) else ""


__all__ = [
    "DATA_SCHEMA_TOOL_DESCRIPTION",
    "DATA_SCHEMA_TOOL_NAME",
    "DATA_SCHEMA_TOOL_SCHEMA",
    "DEFAULT_SCHEMA_CHUNK_TYPES",
    "DataSchemaTool",
    "SchemaChunkStore",
    "SqlSchemaChunkStore",
    "build_data_schema_definition",
]
