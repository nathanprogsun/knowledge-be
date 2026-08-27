"""Unit tests for the data-analysis, database-query, and data-schema tools.

Each tool is driven through injected seams — a stub knowledge service, an
in-memory analysis engine, a file resolver, a query runner, and a chunk
store — so no test touches DuckDB or a live database. The shared SQL
security layer these tools build on is covered directly alongside the tool
flows (dangerous-function blocking, subquery inspection, scope injection,
and the read-only / soft-delete / tenant rewrites).
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from decimal import Decimal
from typing import cast

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.ai.embedding.base import Context
from src.common.json import JsonValue, SqlValue
from src.core.agents.tools.data_analysis import (
    READ_ONLY_ERROR_MESSAGE,
    AnalysisExecutionError,
    AnalysisFileResolver,
    ColumnInfo,
    DataAnalysisTool,
    DuckDbAnalysisEngine,
    LocalTempFileResolver,
    MaterializedFile,
    TableSchema,
    build_data_analysis_definition,
    build_excel_create_table_sql,
    build_missing_column_suggestion,
    format_query_results,
    normalize_identifier_for_match,
    reconcile_sql_columns_with_schema,
    sql_single_quote_escape,
)
from src.core.agents.tools.data_schema import (
    DEFAULT_SCHEMA_CHUNK_TYPES,
    DataSchemaTool,
    build_data_schema_definition,
)
from src.core.agents.tools.database_query import (
    DATABASE_QUERY_TOOL_NAME,
    DatabaseQueryTool,
    SqlAlchemyQueryRunner,
    build_database_query_definition,
    search_scopes_from_targets,
)
from src.core.agents.tools.database_query import (
    format_query_results as format_database_query_results,
)
from src.core.agents.tools.search_target import SearchTarget, SearchTargets, SearchTargetType
from src.core.agents.tools.sql_security import (
    SearchScope,
    SQLValidationConfig,
    inject_and_conditions,
    parse_sql,
    validate_and_secure_sql,
    validate_sql,
    with_allowed_tables,
    with_chunk_enabled_filter,
    with_hidden_kb_filter,
    with_injection_risk_check,
    with_no_dangerous_functions,
    with_search_scopes,
    with_security_defaults,
    with_single_statement,
    with_soft_delete_filter,
)
from src.core.contracts.knowledge import Knowledge
from src.core.knowledge.chunks.types import (
    CHUNK_STATUS_INDEXED,
    CHUNK_TYPE_TABLE_COLUMN,
    CHUNK_TYPE_TABLE_SUMMARY,
)
from src.db.models.chunk import Chunk

_NOW = datetime(2026, 2, 1, tzinfo=UTC)


# ── Shared doubles ────────────────────────────────────────────────────


class _Context:
    """Minimal task context satisfying the ``Context`` protocol."""

    is_background_task: bool = False


class _FakeKnowledgeService:
    """Stub of the tools' ``KnowledgeLookup`` seam."""

    def __init__(self, doc: Knowledge | None = None, error: Exception | None = None) -> None:
        self._doc = doc
        self._error = error
        self.calls: list[str] = []

    async def get_document_by_id_only(self, *, id: str) -> Knowledge | None:
        self.calls.append(id)
        if self._error is not None:
            raise self._error
        return self._doc


def _knowledge(
    id: str = "d1",
    knowledge_base_id: str = "kb-1",
    tenant_id: int = 7,
    *,
    file_type: str = "csv",
    file_path: str = "files/sample.csv",
) -> Knowledge:
    """Build one document contract record."""
    return Knowledge(
        id=id,
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        type="file",
        title="Doc",
        parse_status="completed",
        enable_status="enabled",
        file_type=file_type,
        file_path=file_path,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _targets(*targets: SearchTarget) -> SearchTargets:
    return SearchTargets(targets=tuple(targets))


def _kb_target(
    knowledge_base_id: str = "kb-1",
    tenant_id: int = 7,
    *,
    knowledge_ids: tuple[str, ...] = (),
    tag_ids: tuple[str, ...] = (),
) -> SearchTarget:
    return SearchTarget(
        type=SearchTargetType.KNOWLEDGE_BASE,
        knowledge_base_id=knowledge_base_id,
        tenant_id=tenant_id,
        knowledge_ids=knowledge_ids,
        tag_ids=tag_ids,
    )


def _schema(rows: int = 2) -> TableSchema:
    return TableSchema(
        table_name="k_d1",
        columns=(
            ColumnInfo(name="id", type="VARCHAR"),
            ColumnInfo(name="name", type="VARCHAR"),
        ),
        row_count=rows,
    )


class _FakeAnalysisEngine:
    """Stub of the data-analysis tool's ``AnalysisEngine`` seam."""

    def __init__(
        self,
        *,
        schema: TableSchema | None = None,
        results: list[dict[str, str]] | None = None,
        sheets: list[str] | None = None,
        query_error: Exception | None = None,
        sheets_error: Exception | None = None,
    ) -> None:
        self._schema = schema
        self._results = results or []
        self._sheets = sheets or []
        self._query_error = query_error
        self._sheets_error = sheets_error
        self.loaded_csv: list[str] = []
        self.loaded_excel: list[tuple[str, str, list[str]]] = []
        self.dropped: list[str] = []
        self.executed_queries: list[str] = []

    async def load_csv(self, table_name: str, path: str) -> None:
        self.loaded_csv.append(f"{table_name}:{path}")

    async def load_excel(self, table_name: str, path: str, sheet_names: list[str]) -> None:
        self.loaded_excel.append((table_name, path, sheet_names))

    async def list_excel_sheets(self, path: str) -> list[str]:
        if self._sheets_error is not None:
            raise self._sheets_error
        return list(self._sheets)

    async def describe_table(self, table_name: str) -> TableSchema:
        if self._schema is not None:
            return self._schema
        return _schema()

    async def execute_query(self, sql: str) -> list[dict[str, str]]:
        self.executed_queries.append(sql)
        if self._query_error is not None:
            raise self._query_error
        return list(self._results)

    async def drop_table(self, table_name: str) -> None:
        self.dropped.append(table_name)


class _FakeFileResolver:
    """Stub of the data-analysis tool's ``AnalysisFileResolver`` seam."""

    def __init__(self, path: str = "/tmp/knowledge-data-analysis-sample.csv") -> None:
        self._path = path
        self.materialized: list[tuple[str, str]] = []

    async def materialize(
        self,
        ctx: Context,
        *,
        file_path: str,
        file_type: str,
    ) -> MaterializedFile:
        self.materialized.append((file_path, file_type))
        return MaterializedFile(path=self._path, file_type=file_type)


class _FakeRunner:
    """Stub of the database-query tool's ``DatabaseQueryRunner`` seam."""

    def __init__(
        self,
        columns: list[str] | None = None,
        rows: list[dict[str, JsonValue]] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._columns = columns or ["id"]
        self._rows = rows or []
        self._error = error
        self.queries: list[str] = []

    async def query(self, sql: str) -> tuple[list[str], list[dict[str, JsonValue]]]:
        self.queries.append(sql)
        if self._error is not None:
            raise self._error
        return list(self._columns), list(self._rows)


class _FakeChunkStore:
    """Stub of the data-schema tool's ``SchemaChunkStore`` seam."""

    def __init__(
        self,
        chunks: list[Chunk] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._chunks = chunks or []
        self._error = error
        self.calls: list[tuple[int, str, int]] = []

    async def list_schema_chunks(
        self,
        *,
        tenant_id: int,
        knowledge_id: str,
        page_size: int,
    ) -> list[Chunk]:
        self.calls.append((tenant_id, knowledge_id, page_size))
        if self._error is not None:
            raise self._error
        return list(self._chunks)


def _schema_chunk(
    chunk_type: str,
    content: str,
    *,
    chunk_index: int = 1,
) -> Chunk:
    return Chunk(
        id=f"c-{chunk_type}-{chunk_index}",
        tenant_id=7,
        knowledge_base_id="kb-1",
        knowledge_id="d1",
        content=content,
        chunk_index=chunk_index,
        is_enabled=True,
        start_at=0,
        end_at=len(content),
        chunk_type=chunk_type,
        status=CHUNK_STATUS_INDEXED,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _data_analysis_tool(
    *,
    engine: _FakeAnalysisEngine | None = None,
    resolver: AnalysisFileResolver | None = None,
    doc: Knowledge | None = None,
    error: Exception | None = None,
    targets: SearchTargets | None = None,
    session_id: str = "sess-1",
) -> tuple[DataAnalysisTool, _FakeAnalysisEngine, _FakeFileResolver]:
    service = _FakeKnowledgeService(doc=doc, error=error)
    fake_engine = engine or _FakeAnalysisEngine()
    fake_resolver = _FakeFileResolver() if resolver is None else cast("_FakeFileResolver", resolver)
    tool = DataAnalysisTool(
        definition=build_data_analysis_definition(),
        knowledge_service=service,
        analysis_engine=fake_engine,
        file_resolver=fake_resolver,
        search_targets=targets,
        session_id=session_id,
    )
    return tool, fake_engine, fake_resolver


# ═══════════════════════════════════════════════════════════════════════
# SQL security layer (shared by the data tools)
# ═══════════════════════════════════════════════════════════════════════


class TestSqlSecurity:
    async def test_parse_sql_extracts_tables_and_functions(self) -> None:
        parsed = parse_sql("SELECT name, count(*) FROM k_d1 WHERE name = 'x'")

        assert parsed.is_select is True
        assert parsed.table_names == ("k_d1",)
        assert "count" in parsed.functions
        assert parsed.where_clause == "name = 'x'"

    async def test_subquery_aliases_do_not_become_tables(self) -> None:
        parsed = parse_sql("SELECT * FROM (SELECT * FROM k_d1) AS sub")

        assert parsed.table_names == ("k_d1",)
        assert parsed.has_subquery is True
        assert parsed.table_aliases == {"k_d1": "k_d1"}

    async def test_dangerous_function_blocked_at_any_depth(self) -> None:
        config = SQLValidationConfig()
        config = with_allowed_tables(config, "k_d1")
        config = with_single_statement(config)
        config = with_no_dangerous_functions(config)

        nested = validate_sql(
            "SELECT * FROM (SELECT * FROM read_text('/etc/passwd')) AS t",
            config,
        )
        assert nested.valid is False
        assert any("read_text" in error.details for error in nested.errors)

        top = validate_sql("SELECT * FROM read_csv_auto('/etc/passwd')", config)
        assert top.valid is False

    async def test_safe_function_and_subquery_pass(self) -> None:
        config = SQLValidationConfig()
        config = with_allowed_tables(config, "k_d1")
        config = with_single_statement(config)
        config = with_no_dangerous_functions(config)

        result = validate_sql(
            "SELECT * FROM (SELECT count(*) AS c FROM k_d1) t WHERE t.c > 0",
            config,
        )
        assert result.valid is True

    async def test_table_names_validated_inside_subqueries(self) -> None:
        config = with_allowed_tables(SQLValidationConfig(), "k_d1")

        result = validate_sql("SELECT * FROM (SELECT * FROM knowledge_bases) t", config)
        assert result.valid is False
        assert any("knowledge_bases" in error.message for error in result.errors)

    async def test_compound_queries_rejected(self) -> None:
        config = with_allowed_tables(SQLValidationConfig(), "k_d1")

        result = validate_sql("SELECT * FROM k_d1 UNION SELECT * FROM k_d1", config)
        assert result.valid is False
        assert any("compound" in error.details for error in result.errors)

    async def test_inject_and_conditions_wraps_existing_where(self) -> None:
        sql = inject_and_conditions(
            "SELECT * FROM k_d1 WHERE name = 'x' OR id = '1'",
            "tenant_id = 7",
        )

        assert "WHERE tenant_id = 7 AND (name = 'x' OR id = '1')" in sql

    async def test_inject_and_conditions_before_tail_clauses(self) -> None:
        sql = inject_and_conditions(
            "SELECT * FROM k_d1 ORDER BY id LIMIT 10",
            "deleted_at IS NULL",
        )

        assert sql.startswith("SELECT * FROM k_d1 WHERE deleted_at IS NULL ORDER BY id LIMIT 10")

    async def test_security_defaults_rewrite_scoped_query(self) -> None:
        config = with_security_defaults(SQLValidationConfig(), tenant_id=7)
        config = with_soft_delete_filter(config)
        config = with_hidden_kb_filter(config)
        config = with_chunk_enabled_filter(config)
        config = with_injection_risk_check(config)
        config = with_search_scopes(config, (SearchScope(knowledge_base_id="kb-1"),))

        secured, validation = validate_and_secure_sql(
            "SELECT id, name FROM knowledge_bases WHERE name = 'x'",
            config,
        )
        assert validation.valid is True
        assert "knowledge_bases.tenant_id = 7" in secured
        assert "knowledge_bases.deleted_at IS NULL" in secured
        assert "knowledge_bases.is_temporary = false" in secured
        assert "knowledge_bases.id IN ('kb-1')" in secured

    async def test_search_scopes_from_targets_mapping(self) -> None:
        targets = _targets(
            _kb_target("kb-1"),
            _kb_target("kb-2", knowledge_ids=("d1", "d2")),
            _kb_target("kb-3", tag_ids=("t1",)),
        )
        scopes = search_scopes_from_targets(targets)

        assert scopes[0] == SearchScope(knowledge_base_id="kb-1")
        assert scopes[1] == SearchScope(
            knowledge_base_id="kb-2",
            knowledge_ids=("d1", "d2"),
        )
        assert scopes[2] == SearchScope(knowledge_base_id="kb-3", tag_ids=("t1",))


# ═══════════════════════════════════════════════════════════════════════
# DataAnalysisTool
# ═══════════════════════════════════════════════════════════════════════


class TestDataAnalysisTool:
    async def test_happy_path_csv(self) -> None:
        engine = _FakeAnalysisEngine(
            schema=_schema(),
            results=[{"id": "1", "name": "alpha"}],
        )
        tool, fake_engine, resolver = _data_analysis_tool(
            engine=engine,
            doc=_knowledge(),
        )

        result = await tool.execute(
            _Context(),
            '{"knowledge_id": "d1", "sql": "SELECT id, name FROM d1"}',
        )

        assert result.success is True
        assert fake_engine.loaded_csv == ["k_d1:/tmp/knowledge-data-analysis-sample.csv"]
        assert resolver.materialized == [("files/sample.csv", "csv")]
        assert fake_engine.executed_queries == ["SELECT id, name FROM k_d1"]
        assert "=== DuckDB Query Results ===" in result.output
        assert "Executed SQL: SELECT id, name FROM k_d1" in result.output
        assert "Returned 1 rows" in result.output
        assert 'record 1: {"id":"1","name":"alpha"}' in result.output

        data = result.data
        assert data is not None
        assert data["display_type"] == "data_analysis"
        assert data["row_count"] == 1
        assert data["session_id"] == "sess-1"

    async def test_happy_path_excel_loads_all_sheets(self) -> None:
        engine = _FakeAnalysisEngine(schema=_schema(), sheets=["Sheet1", "Sheet2"])
        tool, fake_engine, _resolver = _data_analysis_tool(
            engine=engine,
            doc=_knowledge(file_type="xlsx", file_path="files/book.xlsx"),
        )

        result = await tool.execute(
            _Context(),
            '{"knowledge_id": "d1", "sql": "SELECT * FROM d1"}',
        )

        assert result.success is True
        assert fake_engine.loaded_excel == [
            ("k_d1", "/tmp/knowledge-data-analysis-sample.csv", ["Sheet1", "Sheet2"])
        ]

    async def test_excel_sheet_enumeration_failure_falls_back(self) -> None:
        engine = _FakeAnalysisEngine(
            schema=_schema(),
            sheets_error=AnalysisExecutionError("failed to query sheet metadata"),
        )
        tool, fake_engine, _resolver = _data_analysis_tool(
            engine=engine,
            doc=_knowledge(file_type="xlsx", file_path="files/book.xlsx"),
        )

        result = await tool.execute(
            _Context(),
            '{"knowledge_id": "d1", "sql": "SELECT * FROM d1"}',
        )

        assert result.success is True
        assert fake_engine.loaded_excel == [("k_d1", "/tmp/knowledge-data-analysis-sample.csv", [])]

    async def test_unknown_knowledge_fails(self) -> None:
        tool, _fake_engine, _resolver = _data_analysis_tool(doc=None)

        result = await tool.execute(
            _Context(),
            '{"knowledge_id": "d1", "sql": "SELECT * FROM d1"}',
        )

        assert result.success is False
        assert result.error == (
            "Failed to load knowledge ID 'd1': knowledge service returned an empty result"
        )

    async def test_scope_enforced_auth_failure_surfaces_message(self) -> None:
        tool, _fake_engine, _resolver = _data_analysis_tool(
            doc=_knowledge(knowledge_base_id="kb-1"),
            targets=_targets(_kb_target("kb-other")),
        )

        result = await tool.execute(
            _Context(),
            '{"knowledge_id": "d1", "sql": "SELECT * FROM d1"}',
        )

        assert result.success is False
        assert result.error == "knowledge base kb-1 is not within the current Agent scope"

    async def test_unsupported_file_type_fails(self) -> None:
        tool, fake_engine, resolver = _data_analysis_tool(
            doc=_knowledge(file_type="pdf", file_path="files/doc.pdf"),
        )

        result = await tool.execute(
            _Context(),
            '{"knowledge_id": "d1", "sql": "SELECT * FROM d1"}',
        )

        assert result.success is False
        assert "unsupported file type: pdf" in result.error
        assert fake_engine.loaded_csv == []
        assert resolver.materialized == []

    async def test_modification_query_rejected(self) -> None:
        tool, fake_engine, _resolver = _data_analysis_tool(doc=_knowledge())

        result = await tool.execute(
            _Context(),
            '{"knowledge_id": "d1", "sql": "DELETE FROM d1 WHERE id = 1"}',
        )

        assert result.success is False
        assert result.error == READ_ONLY_ERROR_MESSAGE
        assert fake_engine.executed_queries == []

    async def test_sql_validation_rejects_unallowed_table(self) -> None:
        tool, fake_engine, _resolver = _data_analysis_tool(doc=_knowledge())

        result = await tool.execute(
            _Context(),
            '{"knowledge_id": "d1", "sql": "SELECT * FROM knowledge_bases"}',
        )

        assert result.success is False
        assert "SQL validation failed" in result.error
        assert "knowledge_bases" in result.error
        assert fake_engine.executed_queries == []

    async def test_query_error_suggests_missing_column(self) -> None:
        error = AnalysisExecutionError(
            'query execution failed: Binder Error: Referenced column "Name" not found in table'
        )
        engine = _FakeAnalysisEngine(
            schema=TableSchema(
                table_name="k_d1",
                columns=(ColumnInfo(name="Name", type="VARCHAR"),),
            ),
            query_error=error,
        )
        tool, _fake_engine, _resolver = _data_analysis_tool(
            engine=engine,
            doc=_knowledge(),
        )

        result = await tool.execute(
            _Context(),
            '{"knowledge_id": "d1", "sql": "SELECT \\"Name\\" FROM d1"}',
        )

        assert result.success is False
        assert 'Did you mean "Name"?' in result.error

    async def test_cleanup_drops_created_tables(self) -> None:
        tool, fake_engine, _resolver = _data_analysis_tool(doc=_knowledge())

        await tool.execute(
            _Context(),
            '{"knowledge_id": "d1", "sql": "SELECT * FROM d1"}',
        )

        await tool.cleanup(_Context())

        assert fake_engine.dropped == ["k_d1"]

    async def test_cleanup_noop_when_no_tables(self) -> None:
        tool, fake_engine, _resolver = _data_analysis_tool(doc=None)

        await tool.cleanup(_Context())

        assert fake_engine.dropped == []

    async def test_duckdb_engine_missing_package(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.core.agents.tools import data_analysis as module

        monkeypatch.setattr(module, "_duckdb_module", None)
        engine = DuckDbAnalysisEngine()

        with pytest.raises(AnalysisExecutionError) as exc_info:
            await engine.describe_table("k_d1")

        assert "duckdb package is not installed" in str(exc_info.value)

    async def test_duckdb_engine_describe_and_query(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.core.agents.tools import data_analysis as module

        conn = _FakeDuckDBConnection()
        conn.results = [
            _FakeDuckDBResult(
                ["column_name", "column_type", "null"],
                [("id", "VARCHAR", "YES"), ("name", "VARCHAR", "YES")],
            ),
            _FakeDuckDBResult(["count(*)"], [(2,)]),
            _FakeDuckDBResult(["id", "name"], [(1, "alpha"), (2, "beta")]),
        ]
        monkeypatch.setattr(module, "_duckdb_module", _FakeDuckDBModule(conn))

        engine = DuckDbAnalysisEngine()
        schema = await engine.describe_table("k_d1")
        rows = await engine.execute_query("SELECT id, name FROM k_d1")

        assert schema.table_name == "k_d1"
        assert [column.name for column in schema.columns] == ["id", "name"]
        assert schema.row_count == 2
        assert rows == [
            {"id": "1", "name": "alpha"},
            {"id": "2", "name": "beta"},
        ]
        assert conn.executed[0].startswith('DESCRIBE "k_d1"')

    async def test_duckdb_engine_list_excel_sheets(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.core.agents.tools import data_analysis as module

        conn = _FakeDuckDBConnection()
        conn.results = [_FakeDuckDBResult(["name"], [("Sheet1",), ("Sheet2",)])]
        monkeypatch.setattr(module, "_duckdb_module", _FakeDuckDBModule(conn))

        engine = DuckDbAnalysisEngine()
        sheets = await engine.list_excel_sheets("files/book.xlsx")

        assert sheets == ["Sheet1", "Sheet2"]
        assert "st_read_meta" in conn.executed[0]

    async def test_local_temp_file_resolver_roundtrip(self) -> None:
        resolver = LocalTempFileResolver(_FakeBytesReader(b"a,b\n1,2\n"))

        material = await resolver.materialize(
            _Context(),
            file_path="files/sample.csv",
            file_type="csv",
        )

        assert material.path.endswith(".csv")
        with open(material.path, "rb") as handle:
            assert handle.read() == b"a,b\n1,2\n"
        os.remove(material.path)


class TestDataAnalysisHelpers:
    async def test_sql_single_quote_escape(self) -> None:
        assert sql_single_quote_escape("it's") == "it''s"

    async def test_normalize_identifier_for_match(self) -> None:
        assert normalize_identifier_for_match("  first name ") == "firstname"
        assert normalize_identifier_for_match("first　name") == "firstname"

    async def test_reconcile_sql_columns_with_schema(self) -> None:
        schema = TableSchema(
            table_name="k_d1",
            columns=(ColumnInfo(name="first name", type="VARCHAR"),),
        )

        rewritten, fixes = reconcile_sql_columns_with_schema(
            'SELECT "firstname" FROM k_d1',
            schema,
        )

        assert rewritten == 'SELECT "first name" FROM k_d1'
        assert fixes == ['"firstname" -> "first name"']

    async def test_build_missing_column_suggestion(self) -> None:
        schema = TableSchema(
            table_name="k_d1",
            columns=(ColumnInfo(name="Name", type="VARCHAR"),),
        )

        suggestion = build_missing_column_suggestion(
            'Binder Error: Referenced column "name" not found in table',
            schema,
        )

        assert suggestion == (
            'Column "name" does not exist. Did you mean "Name"? '
            "Please use the exact column name from schema."
        )

    async def test_build_missing_column_suggestion_empty_when_no_match(self) -> None:
        assert build_missing_column_suggestion("some other error", _schema()) == ""

    async def test_build_excel_create_table_sql_multi_sheet(self) -> None:
        sql = build_excel_create_table_sql("k_d1", "book.xlsx", ["A", "B"])

        assert 'CREATE TABLE "k_d1" AS' in sql
        assert "UNION ALL BY NAME" in sql
        assert sql.count("read_xlsx(") == 2
        assert sql.count("__sheet_name") == 2

    async def test_build_excel_create_table_sql_single_sheet(self) -> None:
        sql = build_excel_create_table_sql("k_d1", "book.xlsx", ["A"])

        assert "UNION ALL BY NAME" not in sql
        assert "'A' AS __sheet_name" in sql

    async def test_build_excel_create_table_sql_no_sheets(self) -> None:
        sql = build_excel_create_table_sql("k_d1", "book.xlsx", [])

        assert "read_xlsx(" in sql
        assert "__sheet_name" not in sql

    async def test_table_schema_description(self) -> None:
        description = _schema().description()

        assert "Table name: k_d1" in description
        assert "Columns: 2" in description
        assert "Rows: 2" in description
        assert "- id (VARCHAR)" in description

    async def test_format_query_results_empty(self) -> None:
        output = format_query_results([], "SELECT * FROM k_d1")

        assert "Returned 0 rows" in output
        assert "No matching records found." in output

    async def test_format_query_results_shows_limit_hint_above_ten(self) -> None:
        rows = [{"id": str(i)} for i in range(11)]

        output = format_query_results(rows, "SELECT id FROM k_d1")

        assert "Showing all 11 records." in output
        assert "record 1:" in output
        assert "record 11:" in output


# ═══════════════════════════════════════════════════════════════════════
# DatabaseQueryTool
# ═══════════════════════════════════════════════════════════════════════


class TestDatabaseQueryTool:
    async def test_happy_path_secures_and_runs(self) -> None:
        runner = _FakeRunner(
            columns=["id", "name"],
            rows=[{"id": "1", "name": "alpha"}],
        )
        tool = DatabaseQueryTool(
            definition=build_database_query_definition(),
            runner=runner,
            search_targets=_targets(_kb_target("kb-1")),
            tenant_id=7,
        )

        result = await tool.execute(
            _Context(),
            '{"sql": "SELECT id, name FROM knowledge_bases WHERE name = \'x\'"}',
        )

        assert result.success is True
        secured = runner.queries[0]
        assert "knowledge_bases.tenant_id = 7" in secured
        assert "knowledge_bases.deleted_at IS NULL" in secured
        assert "knowledge_bases.is_temporary = false" in secured
        assert "knowledge_bases.id IN ('kb-1')" in secured
        assert "=== Query Results ===" in result.output
        assert "Returned 1 rows" in result.output

        data = result.data
        assert data is not None
        assert data["display_type"] == DATABASE_QUERY_TOOL_NAME
        assert data["columns"] == ["id", "name"]
        assert data["row_count"] == 1

    async def test_missing_sql_fails(self) -> None:
        tool = DatabaseQueryTool(
            definition=build_database_query_definition(),
            runner=_FakeRunner(),
            search_targets=_targets(_kb_target("kb-1")),
            tenant_id=7,
        )

        result = await tool.execute(_Context(), "{}")

        assert result.success is False
        assert result.error == "Missing or invalid 'sql' parameter"

    async def test_no_effective_scope_fails(self) -> None:
        tool = DatabaseQueryTool(
            definition=build_database_query_definition(),
            runner=_FakeRunner(),
            search_targets=_targets(_kb_target("")),
            tenant_id=7,
        )

        result = await tool.execute(
            _Context(),
            '{"sql": "SELECT * FROM knowledge_bases"}',
        )

        assert result.success is False
        assert "no effective Agent knowledge scope is available" in result.error

    async def test_validation_rejects_write_statement(self) -> None:
        tool = DatabaseQueryTool(
            definition=build_database_query_definition(),
            runner=_FakeRunner(),
            search_targets=_targets(_kb_target("kb-1")),
            tenant_id=7,
        )

        result = await tool.execute(
            _Context(),
            '{"sql": "DELETE FROM knowledge_bases"}',
        )

        assert result.success is False
        assert "SQL validation failed" in result.error

    async def test_validation_rejects_unallowed_table(self) -> None:
        tool = DatabaseQueryTool(
            definition=build_database_query_definition(),
            runner=_FakeRunner(),
            search_targets=_targets(_kb_target("kb-1")),
            tenant_id=7,
        )

        result = await tool.execute(
            _Context(),
            '{"sql": "SELECT * FROM users"}',
        )

        assert result.success is False
        assert "SQL validation failed" in result.error
        assert "users" in result.error

    async def test_runner_exception_fails(self) -> None:
        runner = _FakeRunner(error=RuntimeError("db down"))
        tool = DatabaseQueryTool(
            definition=build_database_query_definition(),
            runner=runner,
            search_targets=_targets(_kb_target("kb-1")),
            tenant_id=7,
        )

        result = await tool.execute(
            _Context(),
            '{"sql": "SELECT * FROM knowledge_bases"}',
        )

        assert result.success is False
        assert "Query execution failed: db down" in result.error

    async def test_name_and_description_exposed(self) -> None:
        tool = DatabaseQueryTool(
            definition=build_database_query_definition(),
            runner=_FakeRunner(),
            search_targets=_targets(_kb_target("kb-1")),
            tenant_id=7,
        )

        assert tool.name() == DATABASE_QUERY_TOOL_NAME
        assert "Available Tables and Columns" in tool.description()


class TestDatabaseQueryFormatting:
    async def test_format_query_results_null_and_empty(self) -> None:
        output = format_database_query_results([], [])

        assert "Returned 0 rows" in output
        assert "No matching records found." in output

    async def test_format_query_results_renders_values(self) -> None:
        rows: list[dict[str, JsonValue]] = [{"id": "1", "name": None, "meta": {"a": 1}}]

        output = format_database_query_results(["id", "name", "meta"], rows)

        assert "--- Record #1 ---" in output
        assert "  id: 1" in output
        assert "  name: <NULL>" in output
        assert '  meta: {"a":1}' in output

    async def test_format_query_results_limit_hint(self) -> None:
        rows: list[dict[str, JsonValue]] = [{"id": str(i)} for i in range(11)]

        output = format_database_query_results(["id"], rows)

        assert "Showing 11 records out of 11 total." in output


# ═══════════════════════════════════════════════════════════════════════
# SqlAlchemyQueryRunner (value projection)
# ═══════════════════════════════════════════════════════════════════════


class TestSqlAlchemyQueryRunner:
    async def test_to_json_value_projection(self) -> None:
        from src.core.agents.tools.database_query import _to_json_value

        assert _to_json_value(None) is None
        assert _to_json_value("x") == "x"
        assert _to_json_value(True) is True
        assert _to_json_value(42) == 42
        assert _to_json_value(datetime(2026, 2, 1, tzinfo=UTC)) == "2026-02-01T00:00:00+00:00"
        assert _to_json_value(cast("SqlValue", b"\xff")) == "�"
        assert _to_json_value(cast("SqlValue", Decimal("1.5"))) == 1.5
        assert isinstance(_to_json_value(cast("SqlValue", object())), str)

    async def test_query_returns_columns_and_rows(self) -> None:
        session = _FakeAsyncSession(
            keys=["id", "name"],
            mappings=[{"id": "1", "name": "alpha"}],
        )
        runner = SqlAlchemyQueryRunner(cast("AsyncSession", session))

        columns, rows = await runner.query("SELECT id, name FROM knowledge_bases")

        assert columns == ["id", "name"]
        assert rows == [{"id": "1", "name": "alpha"}]


# ═══════════════════════════════════════════════════════════════════════
# DataSchemaTool
# ═══════════════════════════════════════════════════════════════════════


class TestDataSchemaTool:
    async def test_happy_path_returns_summary_and_columns(self) -> None:
        store = _FakeChunkStore(
            chunks=[
                _schema_chunk(CHUNK_TYPE_TABLE_SUMMARY, "Table: d1\nRows: 3"),
                _schema_chunk(
                    CHUNK_TYPE_TABLE_COLUMN,
                    "- id (VARCHAR)\n- name (VARCHAR)",
                    chunk_index=2,
                ),
            ]
        )
        tool = DataSchemaTool(
            definition=build_data_schema_definition(),
            knowledge_service=_FakeKnowledgeService(doc=_knowledge()),
            chunk_store=store,
        )

        result = await tool.execute(_Context(), '{"knowledge_id": "d1"}')

        assert result.success is True
        assert result.output == "Table: d1\nRows: 3\n\n- id (VARCHAR)\n- name (VARCHAR)"
        data = result.data
        assert data is not None
        assert data["summary"] == "Table: d1\nRows: 3"
        assert data["columns"] == "- id (VARCHAR)\n- name (VARCHAR)"
        assert store.calls == [(7, "d1", 100)]

    async def test_missing_knowledge_id_fails(self) -> None:
        tool = DataSchemaTool(
            definition=build_data_schema_definition(),
            knowledge_service=_FakeKnowledgeService(doc=_knowledge()),
            chunk_store=_FakeChunkStore(),
        )

        result = await tool.execute(_Context(), "{}")

        assert result.success is False
        assert result.error == "knowledge_id is required"

    async def test_unknown_knowledge_fails(self) -> None:
        tool = DataSchemaTool(
            definition=build_data_schema_definition(),
            knowledge_service=_FakeKnowledgeService(doc=None),
            chunk_store=_FakeChunkStore(),
        )

        result = await tool.execute(_Context(), '{"knowledge_id": "d1"}')

        assert result.success is False
        assert result.error == (
            "Failed to get knowledge 'd1': knowledge service returned an empty result"
        )

    async def test_scope_enforced_auth_failure_surfaces_message(self) -> None:
        tool = DataSchemaTool(
            definition=build_data_schema_definition(),
            knowledge_service=_FakeKnowledgeService(doc=_knowledge(knowledge_base_id="kb-1")),
            chunk_store=_FakeChunkStore(),
            search_targets=_targets(_kb_target("kb-other")),
        )

        result = await tool.execute(_Context(), '{"knowledge_id": "d1"}')

        assert result.success is False
        assert result.error == (
            "Failed to get knowledge 'd1': knowledge base kb-1 is not within "
            "the current Agent scope"
        )

    async def test_missing_schema_chunks_fails(self) -> None:
        tool = DataSchemaTool(
            definition=build_data_schema_definition(),
            knowledge_service=_FakeKnowledgeService(doc=_knowledge()),
            chunk_store=_FakeChunkStore(),
        )

        result = await tool.execute(_Context(), '{"knowledge_id": "d1"}')

        assert result.success is False
        assert result.error == "No table schema information found for knowledge ID 'd1'"

    async def test_chunk_store_error_fails(self) -> None:
        tool = DataSchemaTool(
            definition=build_data_schema_definition(),
            knowledge_service=_FakeKnowledgeService(doc=_knowledge()),
            chunk_store=_FakeChunkStore(error=RuntimeError("boom")),
        )

        result = await tool.execute(_Context(), '{"knowledge_id": "d1"}')

        assert result.success is False
        assert "Failed to list chunks for knowledge ID 'd1'" in result.error

    async def test_partial_chunks_fails(self) -> None:
        store = _FakeChunkStore(chunks=[_schema_chunk(CHUNK_TYPE_TABLE_SUMMARY, "only summary")])
        tool = DataSchemaTool(
            definition=build_data_schema_definition(),
            knowledge_service=_FakeKnowledgeService(doc=_knowledge()),
            chunk_store=store,
        )

        result = await tool.execute(_Context(), '{"knowledge_id": "d1"}')

        assert result.success is False
        assert "No table schema information found" in result.error

    async def test_default_schema_chunk_types(self) -> None:
        assert DEFAULT_SCHEMA_CHUNK_TYPES == (CHUNK_TYPE_TABLE_SUMMARY, CHUNK_TYPE_TABLE_COLUMN)


# ═══════════════════════════════════════════════════════════════════════
# Registry integration seam
# ═══════════════════════════════════════════════════════════════════════


class TestRegistryIntegration:
    async def test_registry_executes_and_cleans_up_data_analysis(self) -> None:
        from src.core.agents.tools.registry import ToolRegistry

        registry = ToolRegistry()
        engine = _FakeAnalysisEngine(schema=_schema(), results=[{"id": "1"}])
        tool, _fake_engine, _resolver = _data_analysis_tool(engine=engine, doc=_knowledge())
        registry.register_tool(tool)

        result = await registry.execute_tool(
            _Context(),
            "data_analysis",
            '{"knowledge_id": "d1", "sql": "SELECT id FROM d1"}',
        )

        assert result.success is True
        assert engine.executed_queries == ["SELECT id FROM k_d1"]

        await registry.cleanup(_Context())
        assert engine.dropped == ["k_d1"]

    async def test_registry_rejects_tool_outside_scope(self) -> None:
        from src.core.agents.tools.registry import ToolRegistry

        registry = ToolRegistry()
        tool, _fake_engine, _resolver = _data_analysis_tool(
            doc=_knowledge(knowledge_base_id="kb-1"),
            targets=_targets(_kb_target("kb-other")),
        )
        registry.register_tool(tool)

        result = await registry.execute_tool(
            _Context(),
            "data_analysis",
            '{"knowledge_id": "d1", "sql": "SELECT id FROM d1"}',
        )

        assert result.success is False
        assert "not within the current Agent scope" in result.error

    async def test_custom_chunk_types_do_not_match_content_loop(self) -> None:
        # The content loop only matches the canonical summary / column chunk
        # types, mirroring the upstream tool; custom chunk types are fetched
        # but never contribute summary/column content.
        store = _FakeChunkStore(
            chunks=[
                _schema_chunk("custom_summary", "summary"),
                _schema_chunk("custom_column", "columns", chunk_index=2),
            ]
        )
        tool = DataSchemaTool(
            definition=build_data_schema_definition(),
            knowledge_service=_FakeKnowledgeService(doc=_knowledge()),
            chunk_store=store,
            chunk_types=("custom_summary", "custom_column"),
        )

        result = await tool.execute(_Context(), '{"knowledge_id": "d1"}')

        assert result.success is False
        assert result.error == "No table schema information found for knowledge ID 'd1'"


# ═══════════════════════════════════════════════════════════════════════
# DuckDB engine doubles
# ═══════════════════════════════════════════════════════════════════════


class _FakeDuckDBResult:
    def __init__(
        self,
        columns: list[str],
        rows: list[tuple[JsonValue, ...]],
    ) -> None:
        self.columns = columns
        self._rows = rows

    def fetchall(self) -> list[tuple[JsonValue, ...]]:
        return self._rows


class _FakeDuckDBConnection:
    def __init__(self) -> None:
        self.executed: list[str] = []
        self.results: list[_FakeDuckDBResult] = []
        self.closed = False

    def execute(self, sql: str) -> _FakeDuckDBResult:
        self.executed.append(sql)
        if not self.results:
            return _FakeDuckDBResult([], [])
        return self.results.pop(0)

    def close(self) -> None:
        self.closed = True


class _FakeDuckDBModule:
    def __init__(self, conn: _FakeDuckDBConnection) -> None:
        self._conn = conn

    def connect(self) -> _FakeDuckDBConnection:
        return self._conn


class _FakeBytesReader:
    """Stub of the resolver's ``FileBytesReader`` seam."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    async def read_file(self, *, file_path: str) -> bytes:
        return self._data


class _FakeAsyncSession:
    """Stub of an ``AsyncSession`` for ``SqlAlchemyQueryRunner``."""

    def __init__(
        self,
        keys: list[str],
        mappings: list[dict[str, JsonValue]],
    ) -> None:
        self._keys = keys
        self._mappings = mappings
        self.last_sql: str | None = None

    async def execute(self, statement: object) -> _FakeResult:
        sql = getattr(statement, "text", None)
        if sql is not None:
            self.last_sql = str(sql)
        return _FakeResult(
            keys=list(self._keys),
            mappings=list(self._mappings),
        )


class _FakeResult:
    """Stub of ``sqlalchemy.engine.Result``."""

    def __init__(
        self,
        keys: list[str],
        mappings: list[dict[str, JsonValue]],
    ) -> None:
        self._keys = keys
        self._mappings = mappings

    def keys(self) -> list[str]:
        return list(self._keys)

    def mappings(self) -> _FakeMappings:
        return _FakeMappings(list(self._mappings))


class _FakeMappings:
    """Stub of ``result.mappings()`` with an ``all()`` method."""

    def __init__(self, rows: list[dict[str, JsonValue]]) -> None:
        self._rows = rows

    def all(self) -> list[dict[str, JsonValue]]:
        return list(self._rows)
