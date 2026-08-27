"""Data-analysis tool: SQL over CSV / Excel knowledge documents.

``data_analysis`` loads a CSV / Excel document into an in-memory analysis
engine (DuckDB upstream) and runs a read-only SQL statement against it.
Multi-sheet workbooks are unioned into one table with a synthetic
``__sheet_name`` column so the model can filter per sheet. The tool
enforces read-only queries, validates the statement against the loaded
table, auto-reconciles quoted identifiers against the real schema, and
drops its session tables on cleanup.

The engine is an injected ``AnalysisEngine`` seam so unit tests can drive
the tool without DuckDB; ``DuckDbAnalysisEngine`` reproduces the upstream
DuckDB SQL verbatim and lazily loads the optional dependency.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from types import ModuleType
from typing import Protocol, cast, runtime_checkable

from src.ai.embedding.base import Context
from src.common.exception import ApplicationError
from src.common.json import JsonObject, JsonValue
from src.core.agents.tools.base import Cleanable, ToolDefinition, ToolResult
from src.core.agents.tools.scope_auth import (
    KnowledgeLookup,
    KnowledgeTagsFetcher,
    authorize_knowledge_in_search_targets,
)
from src.core.agents.tools.search_target import SearchTargets
from src.core.agents.tools.sql_security import (
    SQLValidationConfig,
    validate_sql,
    with_allowed_tables,
    with_no_dangerous_functions,
    with_single_statement,
)
from src.core.contracts.knowledge import Knowledge

logger = logging.getLogger(__name__)

#: Tool name constant (kept here to avoid a dependency cycle with base).
DATA_ANALYSIS_TOOL_NAME = "data_analysis"

DATA_ANALYSIS_TOOL_DESCRIPTION = (
    "Use this tool when the knowledge is CSV or Excel files. It loads the "
    "data into memory and executes SQL for data analysis. "
    "For Excel files with multiple sheets, every sheet is loaded into the "
    "same table and the source sheet name is exposed as a '__sheet_name' "
    "column so you can filter/aggregate per sheet. "
    "If the user's question requires data statistics, convert the question "
    "into SQL and execute it."
)

DATA_ANALYSIS_TOOL_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "knowledge_id": {
                "type": "string",
                "description": "short dN document ID to query",
            },
            "sql": {
                "type": "string",
                "description": "SQL to be executed on knowledge",
            },
        },
        "required": ["knowledge_id", "sql"],
    },
    ensure_ascii=False,
)

#: Name of the synthetic column identifying the source Excel sheet.
EXCEL_SHEET_NAME_COLUMN = "__sheet_name"

#: Read-only statement prefixes (the only SQL the analysis engine accepts).
READ_ONLY_PREFIXES = ("select", "show", "describe", "explain", "pragma")

#: Message returned for a rejected modification query.
READ_ONLY_ERROR_MESSAGE = (
    "DuckDB tool only supports read-only queries (SELECT, SHOW, DESCRIBE, "
    "EXPLAIN, PRAGMA). Modification operations (INSERT, UPDATE, DELETE, "
    "CREATE, DROP, etc.) are not allowed."
)

#: File types the analysis engine can load.
_SUPPORTED_FILE_TYPES = frozenset({"csv", "xlsx", "xls"})

_RE_QUOTED_IDENTIFIER = re.compile(r'"([^"]+)"')
_RE_MISSING_COLUMN = re.compile(r'Referenced column "([^"]+)" not found')


def _resolve_duckdb_module() -> tuple[ModuleType | None, str]:
    """Resolve the optional duckdb package, returning ``(module, error)``.

    duckdb is an optional runtime dependency of the engine implementation; it
    is resolved lazily via ``importlib`` so the module (and the rest of the
    tool layer) imports cleanly without the package installed.
    """
    try:
        return importlib.import_module("duckdb"), ""
    except ImportError:  # pragma: no cover - depends on an optional dependency
        return None, "duckdb package is not installed"


_duckdb_module, _DUCKDB_IMPORT_ERROR = _resolve_duckdb_module()


def build_data_analysis_definition() -> ToolDefinition:
    """Return the default tool definition for the data-analysis tool."""
    return ToolDefinition(
        name=DATA_ANALYSIS_TOOL_NAME,
        description=DATA_ANALYSIS_TOOL_DESCRIPTION,
        parameters=DATA_ANALYSIS_TOOL_SCHEMA,
    )


@dataclass(frozen=True, slots=True)
class DataAnalysisInput:
    """Parsed input for the data-analysis tool."""

    knowledge_id: str = ""
    sql: str = ""


@dataclass(frozen=True, slots=True)
class ColumnInfo:
    """One column of a loaded analysis table."""

    name: str
    type: str
    nullable: str = "YES"


@dataclass(frozen=True, slots=True)
class TableSchema:
    """Schema of a loaded analysis table."""

    table_name: str
    columns: tuple[ColumnInfo, ...] = ()
    row_count: int = 0
    metadata: JsonObject = field(default_factory=dict)

    def description(self) -> str:
        """Build a human-readable schema description."""
        parts = [
            f"Table name: {self.table_name}\n",
            f"Columns: {len(self.columns)}\n",
            f"Rows: {self.row_count}\n\n",
            "Column info:\n",
        ]
        for column in self.columns:
            parts.append(f"- {column.name} ({column.type})\n")
        return "".join(parts)

    def describe(self) -> str:
        """Alias of :meth:`description` (pipeline-side callers)."""
        return self.description()


class AnalysisExecutionError(RuntimeError):
    """Raised when the analysis engine rejects an operation."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


# ── Engine seam ───────────────────────────────────────────────────────


@runtime_checkable
class AnalysisEngine(Protocol):
    """In-memory SQL engine the data-analysis tool executes against."""

    async def load_csv(self, table_name: str, path: str) -> None: ...

    async def load_excel(self, table_name: str, path: str, sheet_names: list[str]) -> None: ...

    async def list_excel_sheets(self, path: str) -> list[str]: ...

    async def describe_table(self, table_name: str) -> TableSchema: ...

    async def execute_query(self, sql: str) -> list[dict[str, str]]: ...

    async def drop_table(self, table_name: str) -> None: ...


@runtime_checkable
class _DuckDBResult(Protocol):
    """Minimal DuckDB result surface used by the engine."""

    columns: list[str]

    def fetchall(self) -> list[tuple[JsonValue, ...]]: ...


@runtime_checkable
class _DuckDBConnection(Protocol):
    """Minimal DuckDB connection surface used by the engine."""

    def execute(self, sql: str) -> _DuckDBResult: ...

    def close(self) -> None: ...


@runtime_checkable
class _DuckDBModule(Protocol):
    """Minimal DuckDB module surface used by the engine."""

    def connect(self) -> _DuckDBConnection: ...


class DuckDbAnalysisEngine:
    """``AnalysisEngine`` over a lazily-created in-memory DuckDB connection.

    The SQL reproduces the upstream statements verbatim (``read_csv_auto``,
    ``read_xlsx``, ``st_read_meta``, ``DESCRIBE``); the DuckDB package is
    resolved on first use so the tool layer stays importable without it.
    """

    def __init__(self) -> None:
        self._conn: _DuckDBConnection | None = None

    def close(self) -> None:
        """Close the engine connection, if one was opened."""
        if self._conn is None:
            return
        with contextlib.suppress(Exception):
            self._conn.close()
        self._conn = None

    def _connection(self) -> _DuckDBConnection:
        if self._conn is None:
            module = _duckdb_module
            if module is None:
                raise AnalysisExecutionError(_DUCKDB_IMPORT_ERROR)
            self._conn = cast("_DuckDBModule", module).connect()
        return self._conn

    def _execute(self, sql: str, error_prefix: str) -> None:
        connection = self._connection()
        try:
            connection.execute(sql)
        except Exception as exc:
            raise AnalysisExecutionError(f"{error_prefix}: {exc}") from exc

    def _query(self, sql: str, error_prefix: str) -> tuple[list[str], list[tuple[JsonValue, ...]]]:
        connection = self._connection()
        try:
            result = connection.execute(sql)
        except Exception as exc:
            raise AnalysisExecutionError(f"{error_prefix}: {exc}") from exc
        return list(result.columns), result.fetchall()

    async def load_csv(self, table_name: str, path: str) -> None:
        self._execute(
            f'CREATE TABLE "{table_name}" AS SELECT * FROM read_csv_auto('
            f"'{sql_single_quote_escape(path)}', header=true, all_varchar=true)",
            "failed to create table from CSV",
        )

    async def load_excel(self, table_name: str, path: str, sheet_names: list[str]) -> None:
        self._execute(
            build_excel_create_table_sql(table_name, path, sheet_names),
            "failed to create table from Excel file",
        )

    async def list_excel_sheets(self, path: str) -> list[str]:
        meta_sql = (
            f"SELECT UNNEST(layers).name FROM st_read_meta('{sql_single_quote_escape(path)}')"
        )
        _columns, rows = self._query(meta_sql, "failed to query sheet metadata")
        names: list[str] = []
        for row in rows:
            if row and row[0] is not None and str(row[0]).strip():
                names.append(str(row[0]))
        return names

    async def describe_table(self, table_name: str) -> TableSchema:
        _columns, rows = self._query(f'DESCRIBE "{table_name}"', "failed to get table schema")
        column_infos: list[ColumnInfo] = []
        for row in rows:
            if not row or len(row) < 3:
                continue
            name = str(row[0]) if row[0] is not None else ""
            if not name:
                continue
            col_type = str(row[1]) if len(row) > 1 and row[1] is not None else ""
            nullable = str(row[2]) if len(row) > 2 and row[2] is not None else "YES"
            column_infos.append(ColumnInfo(name=name, type=col_type, nullable=nullable))

        _count_columns, count_rows = self._query(
            f'SELECT COUNT(*) FROM "{table_name}"',
            "failed to get row count",
        )
        row_count = 0
        if count_rows and count_rows[0] and count_rows[0][0] is not None:
            row_count = int(cast("int | str", count_rows[0][0]))

        return TableSchema(
            table_name=table_name,
            columns=tuple(column_infos),
            row_count=row_count,
            metadata={"column_count": len(column_infos)},
        )

    async def execute_query(self, sql: str) -> list[dict[str, str]]:
        columns, rows = self._query(sql, "query execution failed")
        results: list[dict[str, str]] = []
        for row in rows:
            record: dict[str, str] = {}
            for index, column in enumerate(columns):
                value = row[index] if index < len(row) else None
                if isinstance(value, bytes):
                    record[column] = value.decode("utf-8", errors="replace")
                else:
                    record[column] = "" if value is None else str(value)
            results.append(record)
        return results

    async def drop_table(self, table_name: str) -> None:
        self._execute(f'DROP TABLE IF EXISTS "{table_name}"', "failed to drop table")


# ── File materialization seam ─────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class MaterializedFile:
    """A knowledge blob copied to a local temp path for the engine."""

    path: str
    file_type: str


@runtime_checkable
class AnalysisFileResolver(Protocol):
    """Materializes a knowledge file to a local path the engine can open."""

    async def materialize(
        self,
        ctx: Context,
        *,
        file_path: str,
        file_type: str,
    ) -> MaterializedFile: ...


@runtime_checkable
class FileBytesReader(Protocol):
    """Reads a knowledge blob by storage path."""

    async def read_file(self, *, file_path: str) -> bytes: ...


class LocalTempFileResolver:
    """``AnalysisFileResolver`` that writes a blob to a temp file.

    The suffix preserves the original extension so the engine's format
    auto-detection keeps working.
    """

    def __init__(self, reader: FileBytesReader) -> None:
        self._reader = reader

    async def materialize(
        self,
        ctx: Context,
        *,
        file_path: str,
        file_type: str,
    ) -> MaterializedFile:
        data = await self._reader.read_file(file_path=file_path)
        ext = file_type.strip().lower() if file_type else ""
        suffix = f".{ext}" if ext else ""
        fd, path = tempfile.mkstemp(prefix="knowledge-data-analysis-", suffix=suffix)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
        except BaseException:
            _remove_file(path)
            raise
        return MaterializedFile(path=path, file_type=file_type)


def _remove_file(path: str) -> None:
    with contextlib.suppress(OSError):
        os.remove(path)


# ── SQL helpers ───────────────────────────────────────────────────────


def sql_single_quote_escape(value: str) -> str:
    """Escape single quotes so a value is safe inside a quoted literal."""
    return value.replace("'", "''")


def normalize_identifier_for_match(value: str) -> str:
    """Normalize an identifier for fuzzy schema matching."""
    return value.strip().lower().replace(" ", "").replace("　", "")


def reconcile_sql_columns_with_schema(
    sql_text: str,
    schema: TableSchema | None,
) -> tuple[str, list[str]]:
    """Rewrite quoted identifiers to their canonical schema names.

    Returns ``(rewritten_sql, fixes)`` where ``fixes`` describes every
    rewrite performed (empty when the SQL already matches the schema).
    """
    if schema is None or not schema.columns:
        return sql_text, []

    normalized_to_canonical: dict[str, str] = {}
    for column in schema.columns:
        key = normalize_identifier_for_match(column.name)
        if not key:
            continue
        if key not in normalized_to_canonical:
            normalized_to_canonical[key] = column.name

    fixes: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        name = match.group(1)
        canonical = normalized_to_canonical.get(normalize_identifier_for_match(name))
        if canonical is None or canonical == name:
            return match.group(0)
        fixes.append(f'"{name}" -> "{canonical}"')
        return f'"{canonical}"'

    return _RE_QUOTED_IDENTIFIER.sub(_replace, sql_text), fixes


def build_missing_column_suggestion(err_message: str, schema: TableSchema | None) -> str:
    """Suggest a canonical column when DuckDB reports a missing column."""
    if not err_message or schema is None or not schema.columns:
        return ""
    if 'Referenced column "' not in err_message or "not found" not in err_message:
        return ""

    match = _RE_MISSING_COLUMN.search(err_message)
    if match is None:
        return ""
    missing = match.group(1)
    normalized_missing = normalize_identifier_for_match(missing)
    if not normalized_missing:
        return ""

    for column in schema.columns:
        if normalize_identifier_for_match(column.name) == normalized_missing:
            return (
                f'Column "{missing}" does not exist. Did you mean "{column.name}"? '
                "Please use the exact column name from schema."
            )
    return ""


def build_excel_create_table_sql(table_name: str, filename: str, sheet_names: list[str]) -> str:
    """Assemble the CREATE TABLE statement for an Excel workbook.

    With multiple sheets the statement unions every sheet (``UNION ALL BY
    NAME``) and tags each row with its source sheet; a single sheet still
    carries the tag so downstream SQL behaves identically.
    """
    escaped_file = sql_single_quote_escape(filename)

    if not sheet_names:
        return (
            f'CREATE TABLE "{table_name}" AS SELECT * FROM read_xlsx('
            f"'{escaped_file}', header=true, all_varchar=true)"
        )

    if len(sheet_names) == 1:
        escaped_sheet = sql_single_quote_escape(sheet_names[0])
        return (
            f"CREATE TABLE \"{table_name}\" AS SELECT *, '{escaped_sheet}' AS "
            f"{EXCEL_SHEET_NAME_COLUMN} FROM read_xlsx('{escaped_file}', sheet = "
            f"'{escaped_sheet}', header=true, all_varchar=true)"
        )

    parts: list[str] = []
    for sheet in sheet_names:
        escaped_sheet = sql_single_quote_escape(sheet)
        parts.append(
            f"SELECT *, '{escaped_sheet}' AS {EXCEL_SHEET_NAME_COLUMN} "
            f"FROM read_xlsx('{escaped_file}', sheet = '{escaped_sheet}', "
            "header=true, all_varchar=true)"
        )
    return f'CREATE TABLE "{table_name}" AS ' + "\nUNION ALL BY NAME\n".join(parts)


def format_query_results(results: list[dict[str, str]], query: str) -> str:
    """Format query results into JSONL text (one JSON object per line)."""
    parts: list[str] = []
    parts.append("=== DuckDB Query Results ===\n\n")
    parts.append(f"Executed SQL: {query}\n\n")
    parts.append(f"Returned {len(results)} rows\n\n")
    if not results:
        parts.append("No matching records found.\n")
        return "".join(parts)

    parts.append("=== Data Details ===\n\n")
    if len(results) > 10:
        parts.append(
            f"Showing all {len(results)} records. Consider using a LIMIT clause "
            "to restrict the result count for better performance.\n\n"
        )
    for index, record in enumerate(results, start=1):
        record_str = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        parts.append(f"record {index}: {record_str}\n")
    return "".join(parts)


def _parse_input(args: str) -> DataAnalysisInput:
    try:
        raw = json.loads(args)
    except json.JSONDecodeError:
        raw = {}
    if not isinstance(raw, dict):
        raw = {}
    knowledge_id = raw.get("knowledge_id")
    sql_value = raw.get("sql")
    return DataAnalysisInput(
        knowledge_id=knowledge_id if isinstance(knowledge_id, str) else "",
        sql=sql_value if isinstance(sql_value, str) else "",
    )


# ── Tool ──────────────────────────────────────────────────────────────


class DataAnalysisTool(Cleanable):
    """Loads a data document into the engine and runs read-only SQL."""

    def __init__(
        self,
        *,
        definition: ToolDefinition,
        knowledge_service: KnowledgeLookup | None,
        analysis_engine: AnalysisEngine,
        file_resolver: AnalysisFileResolver,
        search_targets: SearchTargets | None = None,
        tag_fetcher: KnowledgeTagsFetcher | None = None,
        session_id: str = "",
    ) -> None:
        self._definition = definition
        self._knowledge_service = knowledge_service
        self._analysis_engine = analysis_engine
        self._file_resolver = file_resolver
        self._search_targets = search_targets
        self._tag_fetcher = tag_fetcher
        self._session_id = session_id
        #: Tables created by this tool instance, dropped on cleanup.
        self._created_tables: list[str] = []

    def name(self) -> str:
        return self._definition.name

    def description(self) -> str:
        return self._definition.description

    def parameters(self) -> str:
        return self._definition.parameters

    def _table_name_for_knowledge(self, knowledge: Knowledge) -> str:
        return "k_" + knowledge.id.replace("-", "_")

    def _record_created_table(self, table_name: str) -> bool:
        """Record a table for cleanup; returns False when already recorded."""
        if table_name in self._created_tables:
            return False
        self._created_tables.append(table_name)
        return True

    async def execute(self, ctx: Context, args: str) -> ToolResult:
        """Load the document and run one read-only SQL statement."""
        input_ = _parse_input(args)

        try:
            knowledge = await self._resolve_knowledge(ctx, input_.knowledge_id)
        except ApplicationError as exc:
            return ToolResult(success=False, error=exc.message)
        if knowledge is None:
            return ToolResult(
                success=False,
                error=(
                    f"Failed to load knowledge ID '{input_.knowledge_id}': "
                    "knowledge service returned an empty result"
                ),
            )

        try:
            schema = await self._load_from_knowledge(ctx, knowledge)
        except AnalysisExecutionError as exc:
            return ToolResult(
                success=False,
                error=f"Failed to load knowledge ID '{input_.knowledge_id}': {exc.message}",
            )

        sql_text = input_.sql.replace(input_.knowledge_id, schema.table_name)
        rewritten_sql, fixes = reconcile_sql_columns_with_schema(sql_text, schema)
        if fixes:
            sql_text = rewritten_sql

        normalized_sql = sql_text.strip().lower()
        is_read_only = normalized_sql.startswith(READ_ONLY_PREFIXES)
        if not is_read_only:
            return ToolResult(success=False, error=READ_ONLY_ERROR_MESSAGE)

        config = SQLValidationConfig()
        config = with_allowed_tables(config, schema.table_name)
        config = with_single_statement(config)
        config = with_no_dangerous_functions(config)
        validation = validate_sql(sql_text, config)
        if not validation.valid:
            details = "; ".join(f"{error.type}: {error.message}" for error in validation.errors)
            return ToolResult(success=False, error=f"SQL validation failed: {details}")

        try:
            results = await self._analysis_engine.execute_query(sql_text)
        except AnalysisExecutionError as exc:
            suggestion = build_missing_column_suggestion(exc.message, schema)
            if suggestion:
                return ToolResult(
                    success=False,
                    error=f"Query execution failed: {exc.message}. {suggestion}",
                )
            return ToolResult(
                success=False,
                error=f"Query execution failed: {exc.message}",
            )

        output = format_query_results(results, sql_text)
        data: JsonObject = {
            "rows": cast("list[JsonValue]", results),
            "row_count": len(results),
            "query": sql_text,
            "display_type": DATA_ANALYSIS_TOOL_NAME,
            "session_id": self._session_id,
        }
        return ToolResult(success=True, output=output, data=data)

    async def _resolve_knowledge(self, ctx: Context, knowledge_id: str) -> Knowledge | None:
        """Resolve the document, applying scope authorization when configured.

        A scope-enforced failure raises an ``ApplicationError`` (the auth
        boundary's message becomes the tool error); the un-scoped lookup
        returns ``None`` when the document cannot be resolved.
        """
        if self._search_targets is not None:
            return await authorize_knowledge_in_search_targets(
                ctx,
                self._search_targets,
                knowledge_id,
                self._knowledge_service,
                self._tag_fetcher,
            )
        if self._knowledge_service is None:
            return None
        try:
            return await self._knowledge_service.get_document_by_id_only(id=knowledge_id)
        except Exception as exc:
            logger.warning(
                "[Tool][DataAnalysis] Failed to get knowledge '%s': %s", knowledge_id, exc
            )
            return None

    async def _load_from_knowledge(self, ctx: Context, knowledge: Knowledge) -> TableSchema:
        """Materialize the knowledge file and load it into the engine."""
        file_type = (knowledge.file_type or "").strip().lower()
        if file_type not in _SUPPORTED_FILE_TYPES:
            raise AnalysisExecutionError(
                f"unsupported file type: {file_type} (supported types: csv, xlsx, xls)"
            )

        table_name = self._table_name_for_knowledge(knowledge)
        material = await self._file_resolver.materialize(
            ctx,
            file_path=knowledge.file_path or "",
            file_type=knowledge.file_type or "",
        )
        try:
            if file_type == "csv":
                await self._load_csv(table_name, material.path)
            else:
                await self._load_excel(table_name, material.path)
            return await self._analysis_engine.describe_table(table_name)
        finally:
            _remove_file(material.path)

    async def _load_csv(self, table_name: str, path: str) -> None:
        if self._record_created_table(table_name):
            await self._analysis_engine.load_csv(table_name, path)

    async def _load_excel(self, table_name: str, path: str) -> None:
        if self._record_created_table(table_name):
            try:
                sheet_names = await self._analysis_engine.list_excel_sheets(path)
            except AnalysisExecutionError:
                # Sheet enumeration failed: fall back to the first sheet only.
                sheet_names = []
            await self._analysis_engine.load_excel(table_name, path, sheet_names)

    async def cleanup(self, ctx: Context) -> None:
        """Drop every table created by this tool instance."""
        if not self._created_tables:
            return
        for table_name in self._created_tables:
            try:
                await self._analysis_engine.drop_table(table_name)
            except Exception:
                # Best-effort cleanup: continue dropping the remaining tables.
                continue
        self._created_tables = []


__all__ = [
    "DATA_ANALYSIS_TOOL_DESCRIPTION",
    "DATA_ANALYSIS_TOOL_NAME",
    "DATA_ANALYSIS_TOOL_SCHEMA",
    "EXCEL_SHEET_NAME_COLUMN",
    "READ_ONLY_ERROR_MESSAGE",
    "READ_ONLY_PREFIXES",
    "AnalysisEngine",
    "AnalysisExecutionError",
    "AnalysisFileResolver",
    "ColumnInfo",
    "DataAnalysisInput",
    "DataAnalysisTool",
    "DuckDbAnalysisEngine",
    "FileBytesReader",
    "LocalTempFileResolver",
    "MaterializedFile",
    "TableSchema",
    "build_data_analysis_definition",
    "build_excel_create_table_sql",
    "build_missing_column_suggestion",
    "format_query_results",
    "normalize_identifier_for_match",
    "reconcile_sql_columns_with_schema",
    "sql_single_quote_escape",
]


def data_analysis_input_schema() -> JsonObject:
    """Return the JSON output schema for :class:`DataAnalysisInput`.

    Mirrors the generated schema used to pin the model's structured
    response in the data-analysis planning call.
    """
    return {
        "type": "object",
        "properties": {
            "knowledge_id": {
                "type": "string",
                "description": "short dN document ID to query",
            },
            "sql": {
                "type": "string",
                "description": "SQL to be executed on knowledge",
            },
        },
        "required": ["knowledge_id", "sql"],
    }
