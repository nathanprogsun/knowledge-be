"""Database-query tool: read-only SQL over the tenant's own tables.

``database_query`` lets the agent run a read-only SELECT against the
``knowledge_bases`` / ``knowledges`` / ``chunks`` tables. The query is
validated and then rewritten server-side to inject the caller's tenant
filter, soft-delete filtering, hidden-knowledge-base filtering,
enabled-chunk filtering, and the session's search scope — the model can
never widen its own read access by writing SQL.

The tool executes through a ``DatabaseQueryRunner`` seam so the layer
stays free of a direct engine dependency; ``SqlAlchemyQueryRunner`` runs
the secured statement over an ``AsyncSession``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Protocol, cast, runtime_checkable

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.ai.embedding.base import Context
from src.common.json import JsonObject, JsonValue, SqlValue
from src.core.agents.tools.base import ToolDefinition, ToolResult
from src.core.agents.tools.scope_auth import (
    search_target_is_whole_kb,
    search_target_scope,
)
from src.core.agents.tools.search_target import SearchTargets
from src.core.agents.tools.sql_security import (
    SearchScope,
    SQLValidationConfig,
    validate_and_secure_sql,
    with_chunk_enabled_filter,
    with_hidden_kb_filter,
    with_injection_risk_check,
    with_search_scopes,
    with_security_defaults,
    with_soft_delete_filter,
)

logger = logging.getLogger(__name__)

#: Tool name constant (kept here to avoid a dependency cycle with base).
DATABASE_QUERY_TOOL_NAME = "database_query"

DATABASE_QUERY_TOOL_DESCRIPTION = (
    "Execute SQL queries to retrieve information from the database.\n"
    "\n"
    "## Security Features\n"
    "- Automatic tenant_id injection: All queries are automatically filtered by the logged-in user's tenant_id\n"
    "- Automatic soft-delete filtering: All queries are automatically filtered to include only records with deleted_at IS NULL\n"
    "- Read-only queries: Only SELECT statements are allowed\n"
    "- Safe tables: Only allow queries on authorized tables (knowledge_bases, knowledges, chunks)\n"
    "\n"
    "## Available Tables and Columns\n"
    "\n"
    "### knowledge_bases\n"
    "- id (VARCHAR): Knowledge base ID\n"
    "- name (VARCHAR): Knowledge base name\n"
    "- description (TEXT): Description\n"
    "- tenant_id (INTEGER): Owner tenant ID\n"
    "- embedding_model_id, summary_model_id, rerank_model_id (VARCHAR): Model IDs\n"
    "- vlm_config (JSON): Includes VLM settings such as enabled flag and model_id\n"
    "- created_at, updated_at, deleted_at (TIMESTAMP)\n"
    "\n"
    "### knowledges (documents)\n"
    "- id (VARCHAR): Document ID\n"
    "- tenant_id (INTEGER): Owner tenant ID\n"
    "- knowledge_base_id (VARCHAR): Parent knowledge base ID\n"
    "- type (VARCHAR): Document type\n"
    "- title (VARCHAR): Document title\n"
    "- description (TEXT): Description\n"
    "- source (VARCHAR): Source location\n"
    "- parse_status (VARCHAR): Processing status (unprocessed/processing/completed/failed)\n"
    "- enable_status (VARCHAR): Enable status (enabled/disabled)\n"
    "- file_name, file_type (VARCHAR): File information\n"
    "- file_size, storage_size (BIGINT): Size in bytes\n"
    "- created_at, updated_at, processed_at, deleted_at (TIMESTAMP)\n"
    "\n"
    "### chunks\n"
    "- id (VARCHAR): Chunk ID\n"
    "- tenant_id (INTEGER): Owner tenant ID\n"
    "- knowledge_base_id (VARCHAR): Parent knowledge base ID\n"
    "- knowledge_id (VARCHAR): Parent document ID\n"
    "- content (TEXT): Chunk content\n"
    "- chunk_index (INTEGER): Index in document\n"
    "- is_enabled (BOOLEAN): Enable status\n"
    "- chunk_type (VARCHAR): Type (text/image/table)\n"
    "- created_at, updated_at, deleted_at (TIMESTAMP)\n"
    "\n"
    "## Usage Examples\n"
    "\n"
    "Query knowledge base information:\n"
    '{\n  "sql": "SELECT id, name, description FROM knowledge_bases ORDER BY created_at DESC LIMIT 10"\n}\n'
    "\n"
    "Count documents by status:\n"
    '{\n  "sql": "SELECT parse_status, COUNT(*) as count FROM knowledges GROUP BY parse_status"\n}\n'
    "\n"
    "Get storage usage:\n"
    '{\n  "sql": "SELECT SUM(storage_size) as total_storage FROM knowledges"\n}\n'
    "\n"
    "Join knowledge bases and documents:\n"
    '{\n  "sql": "SELECT kb.name as kb_name, COUNT(k.id) as doc_count FROM knowledge_bases kb LEFT JOIN knowledges k ON kb.id = k.knowledge_base_id GROUP BY kb.id, kb.name"\n}\n'
    "\n"
    "## Important Notes\n"
    "- DO NOT include tenant_id in WHERE clause - it's automatically added\n"
    "- DO NOT include deleted_at filtering manually unless needed - default query already enforces deleted_at IS NULL\n"
    "- Only SELECT queries are allowed\n"
    "- Limit results with LIMIT clause for better performance\n"
    "- Use appropriate JOINs when querying across tables\n"
    "- All timestamps are in UTC with time zone"
)

DATABASE_QUERY_TOOL_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": (
                    "The SELECT SQL query to execute. DO NOT include tenant_id "
                    "condition - it will be automatically added for security."
                ),
            },
        },
        "required": ["sql"],
    },
    ensure_ascii=False,
)


def build_database_query_definition() -> ToolDefinition:
    """Return the default tool definition for the database-query tool."""
    return ToolDefinition(
        name=DATABASE_QUERY_TOOL_NAME,
        description=DATABASE_QUERY_TOOL_DESCRIPTION,
        parameters=DATABASE_QUERY_TOOL_SCHEMA,
    )


@dataclass(frozen=True, slots=True)
class DatabaseQueryInput:
    """Parsed input for the database-query tool."""

    sql: str = ""


def _parse_input(args: str) -> DatabaseQueryInput:
    try:
        raw = json.loads(args)
    except json.JSONDecodeError:
        raw = {}
    if not isinstance(raw, dict):
        raw = {}
    sql_value = raw.get("sql")
    return DatabaseQueryInput(sql=sql_value if isinstance(sql_value, str) else "")


def search_scopes_from_targets(search_targets: SearchTargets) -> list[SearchScope]:
    """Map the session's search targets to SQL search scopes.

    A target that narrows to nothing (no document whitelist and no tag
    scope on a non-whole-KB target) contributes no scope, mirroring the
    upstream behaviour.
    """
    scopes: list[SearchScope] = []
    for target in search_targets:
        if target is None or not target.knowledge_base_id:
            continue
        knowledge_ids, tag_ids = search_target_scope(target)
        if not search_target_is_whole_kb(target) and not knowledge_ids and not tag_ids:
            continue
        scopes.append(
            SearchScope(
                knowledge_base_id=target.knowledge_base_id,
                knowledge_ids=tuple(knowledge_ids),
                tag_ids=tuple(tag_ids),
            )
        )
    return scopes


@runtime_checkable
class DatabaseQueryRunner(Protocol):
    """Executes a secured read-only statement, returning columns + rows."""

    async def query(self, sql: str) -> tuple[list[str], list[dict[str, JsonValue]]]: ...


class SqlAlchemyQueryRunner:
    """``DatabaseQueryRunner`` over an ``AsyncSession``.

    Column values are projected to JSON-safe scalars (``bytes`` become
    text, ``datetime`` becomes ISO-8601 text, ``Decimal`` becomes float)
    so the result map is directly serializable.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def query(self, sql: str) -> tuple[list[str], list[dict[str, JsonValue]]]:
        result = await self._session.execute(text(sql))
        columns = list(result.keys())
        rows: list[dict[str, JsonValue]] = []
        for mapping in result.mappings().all():
            row = dict(mapping)
            rows.append({column: _to_json_value(row[column]) for column in columns})
        return columns, rows


def _to_json_value(value: SqlValue) -> JsonValue:
    """Project one SQL scalar onto a JSON-safe value."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, Decimal):
        return float(value)
    return str(value)


class DatabaseQueryTool:
    """Validates, secures, and executes a read-only database query."""

    def __init__(
        self,
        *,
        definition: ToolDefinition,
        runner: DatabaseQueryRunner,
        search_targets: SearchTargets,
        tenant_id: int = 0,
    ) -> None:
        self._definition = definition
        self._runner = runner
        self._search_targets = search_targets
        self._tenant_id = tenant_id

    def name(self) -> str:
        return self._definition.name

    def description(self) -> str:
        return self._definition.description

    def parameters(self) -> str:
        return self._definition.parameters

    async def execute(self, ctx: Context, args: str) -> ToolResult:
        """Validate, secure, and run one read-only SQL statement."""
        input_ = _parse_input(args)
        sql_text = input_.sql.strip()
        if not sql_text:
            return ToolResult(
                success=False,
                error="Missing or invalid 'sql' parameter",
            )

        scopes = search_scopes_from_targets(self._search_targets)
        if not scopes:
            return ToolResult(
                success=False,
                error="no effective Agent knowledge scope is available",
            )

        config = with_security_defaults(SQLValidationConfig(), self._tenant_id)
        config = with_soft_delete_filter(config)
        config = with_hidden_kb_filter(config)
        config = with_chunk_enabled_filter(config)
        config = with_injection_risk_check(config)
        config = with_search_scopes(config, scopes)

        secured_sql, validation = validate_and_secure_sql(sql_text, config)
        if not validation.valid:
            details = "; ".join(f"{error.type}: {error.message}" for error in validation.errors)
            if not details:
                details = "invalid query"
            return ToolResult(
                success=False,
                error=f"SQL validation failed: {details}",
            )

        try:
            columns, rows = await self._runner.query(secured_sql)
        except Exception as exc:
            logger.warning("[Tool][DatabaseQuery] Query execution failed: %s", exc)
            return ToolResult(
                success=False,
                error=f"Query execution failed: {exc}",
            )

        output = format_query_results(columns, rows)
        data: JsonObject = {
            "columns": cast("list[JsonValue]", columns),
            "rows": cast("list[JsonValue]", rows),
            "row_count": len(rows),
            "display_type": "database_query",
        }
        return ToolResult(success=True, output=output, data=data)


def _format_value(value: JsonValue) -> str:
    """Format one result value for the human-readable output."""
    if value is None:
        return "<NULL>"
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def format_query_results(columns: list[str], rows: list[dict[str, JsonValue]]) -> str:
    """Format query results into readable text."""
    parts: list[str] = []
    parts.append("=== Query Results ===\n\n")
    parts.append(f"Returned {len(rows)} rows\n\n")
    if not rows:
        parts.append("No matching records found.\n")
        return "".join(parts)

    parts.append("=== Data Details ===\n\n")
    for index, row in enumerate(rows, start=1):
        parts.append(f"--- Record #{index} ---\n")
        for column in columns:
            parts.append(f"  {column}: {_format_value(row.get(column))}\n")
        parts.append("\n")

    if len(rows) > 10:
        parts.append(
            f"Note: Showing {len(rows)} records out of {len(rows)} total. "
            "Consider using a LIMIT clause to restrict the result count.\n"
        )
    return "".join(parts)


__all__ = [
    "DATABASE_QUERY_TOOL_DESCRIPTION",
    "DATABASE_QUERY_TOOL_NAME",
    "DATABASE_QUERY_TOOL_SCHEMA",
    "DatabaseQueryInput",
    "DatabaseQueryRunner",
    "DatabaseQueryTool",
    "SqlAlchemyQueryRunner",
    "build_database_query_definition",
    "format_query_results",
    "search_scopes_from_targets",
]
