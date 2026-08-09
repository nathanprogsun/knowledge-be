"""Data-analysis tool contract: schema shapes and the execution seam.

``data_analysis`` runs SQL over tabular knowledge (CSV / Excel) loaded into
a local analysis engine. This module pins the contract the rest of the
system depends on: the request payload (``DataAnalysisInput``), its JSON
output schema, the ``TableSchema`` / ``ColumnInfo`` shapes used to describe
a loaded table to the model, the JSONL result renderer, and the structural
``DataAnalysisTool`` protocol the pipeline step executes against. The
engine-backed implementation satisfying this protocol lives behind the
tool registry and is wired at the request entry point.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from src.ai.embedding.base import Context
from src.common.json import JsonObject
from src.core.agents.tools.base import ToolResult
from src.core.contracts.knowledge import Knowledge

#: Name of the synthetic column tagging which Excel sheet a row came from
#: when multiple sheets are unioned into one table.
EXCEL_SHEET_NAME_COLUMN = "__sheet_name"


class DataAnalysisInput(BaseModel):
    """LLM-provided arguments for one data-analysis execution."""

    model_config = ConfigDict(frozen=True)

    knowledge_id: str = Field(description="short dN document ID to query")
    sql: str = Field(description="SQL to be executed on knowledge")


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


class ColumnInfo(BaseModel):
    """One column of a loaded data table."""

    model_config = ConfigDict(frozen=True)

    name: str = ""
    type: str = ""
    nullable: str = ""


class TableSchema(BaseModel):
    """Schema of a table loaded into the analysis engine."""

    model_config = ConfigDict(frozen=True)

    table_name: str = ""
    columns: list[ColumnInfo] = Field(default_factory=list)
    row_count: int = 0
    metadata: JsonObject | None = None

    def describe(self) -> str:
        """Render a human-readable schema description for the model prompt."""
        parts = [
            f"Table name: {self.table_name}\n",
            f"Columns: {len(self.columns)}\n",
            f"Rows: {self.row_count}\n\n",
            "Column info:\n",
        ]
        for column in self.columns:
            parts.append(f"- {column.name} ({column.type})\n")
        return "".join(parts)


def sql_single_quote_escape(value: str) -> str:
    """Escape single quotes so ``value`` embeds in a single-quoted literal."""
    return value.replace("'", "''")


def format_query_results(results: Sequence[Mapping[str, str]], query: str) -> str:
    """Render query rows as JSONL-style records with a header.

    ``results`` carries one mapping per row (column name to string value);
    keys are sorted to match the deterministic reference ordering.
    """
    parts = [
        "=== DuckDB Query Results ===\n\n",
        f"Executed SQL: {query}\n\n",
        f"Returned {len(results)} rows\n\n",
    ]
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
        record_str = json.dumps(record, sort_keys=True)
        parts.append(f"record {index}: {record_str}\n")
    return "".join(parts)


@runtime_checkable
class DataAnalysisTool(Protocol):
    """Execution surface the data-analysis pipeline step depends on.

    A concrete implementation loads a knowledge's tabular backing file into
    an analysis engine, executes read-only SQL against it, and releases the
    session-scoped tables when the run ends.
    """

    async def load_from_knowledge(self, ctx: Context, knowledge: Knowledge) -> TableSchema:
        """Load ``knowledge``'s backing file and return its table schema."""
        ...

    async def execute(self, ctx: Context, args_json: str) -> ToolResult:
        """Execute the SQL carried by ``args_json`` and return its output."""
        ...

    async def cleanup(self, ctx: Context) -> None:
        """Release session-scoped tables created by this tool."""
        ...


__all__ = [
    "EXCEL_SHEET_NAME_COLUMN",
    "ColumnInfo",
    "DataAnalysisInput",
    "DataAnalysisTool",
    "TableSchema",
    "data_analysis_input_schema",
    "format_query_results",
    "sql_single_quote_escape",
]
