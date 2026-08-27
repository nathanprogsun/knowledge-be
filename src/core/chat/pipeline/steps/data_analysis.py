"""Data-analysis pipeline step (upstream ``PluginDataAnalysis``).

Runs before the context template renders when the merged retrieval results
contain tabular knowledge (CSV / Excel). The step asks the chat model
whether the user's question needs statistical / aggregate SQL over the
first data file, executes the generated SQL through the data-analysis
tool, and appends the output as a ``DATA_ANALYSIS`` search hit so the
downstream context renderer can cite it.

Every failure is a soft short-circuit: the step logs the reason and hands
control to the next listener, mirroring the upstream contract where data
analysis must never break the normal QA path.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

from src.ai.llm.types import ChatOptions, Message
from src.ai.llm.usage import with_llm_call_metadata
from src.ai.retrieval.types import MatchType
from src.core.agents.tools.base import ToolResult
from src.core.agents.tools.data_analysis import (
    TableSchema,
    data_analysis_input_schema,
)
from src.core.chat.pipeline.common import pipeline_error
from src.core.chat.pipeline.context import PipelineContext
from src.core.chat.pipeline.engine import ERR_GET_CHAT_MODEL, Next, PluginError
from src.core.chat.pipeline.steps.model_context import ModelService
from src.core.chat.pipeline.steps.passage import (
    CHUNK_TYPE_TABLE_COLUMN,
    CHUNK_TYPE_TABLE_SUMMARY,
)
from src.core.chat.pipeline.types import Context, EventType, SearchResult
from src.core.contracts.knowledge import Knowledge

#: File extensions treated as analyzable data files.
_DATA_FILE_EXTENSIONS = (".csv", ".xlsx", ".xls")

#: Purpose label attached to the data-analysis planning model call.
_DATA_ANALYSIS_PLAN_PURPOSE = "data_analysis_plan"

#: Chunk types filtered out of the retrieval context before data analysis
#: runs, since table schema probes carry no answer content.
_TABLE_CHUNK_TYPES = frozenset({CHUNK_TYPE_TABLE_COLUMN, CHUNK_TYPE_TABLE_SUMMARY})


def is_data_file(filename: str) -> bool:
    """Return whether ``filename`` looks like a CSV / Excel data file."""
    lower = filename.lower()
    return lower.endswith(_DATA_FILE_EXTENSIONS)


def filter_out_table_chunks(results: Sequence[SearchResult]) -> list[SearchResult]:
    """Drop table-column and table-summary chunks from ``results``."""
    return [result for result in results if result.chunk_type not in _TABLE_CHUNK_TYPES]


@runtime_checkable
class DataAnalysisTool(Protocol):
    """Execution surface for the data-analysis tool.

    Mirrors the tools-layer ``DataAnalysisTool`` contract with the pipeline
    context type, so the step stays decoupled from the engine-backed
    implementation.
    """

    async def load_from_knowledge(self, ctx: Context, knowledge: Knowledge) -> TableSchema: ...
    async def execute(self, ctx: Context, args_json: str) -> ToolResult: ...
    async def cleanup(self, ctx: Context) -> None: ...


@runtime_checkable
class KnowledgeService(Protocol):
    """Resolves a knowledge document by id for data analysis."""

    async def get_knowledge_by_id(self, ctx: Context, knowledge_id: str) -> Knowledge | None: ...


class DataAnalysisStep:
    """Runs the data-analysis stage of the chat pipeline."""

    def __init__(
        self,
        model_service: ModelService,
        knowledge_service: KnowledgeService,
        data_analysis_tool_factory: Callable[[str], DataAnalysisTool],
    ) -> None:
        self._model_service = model_service
        self._knowledge_service = knowledge_service
        self._tool_factory = data_analysis_tool_factory

    def activation_events(self) -> Sequence[EventType]:
        return (EventType.DATA_ANALYSIS,)

    async def on_event(
        self,
        ctx: Context,
        event_type: EventType | str,
        pipeline_ctx: PipelineContext,
        next: Next,
    ) -> PluginError | None:
        if not pipeline_ctx.needs_retrieval():
            return await next()

        data_files = [
            result
            for result in pipeline_ctx.merge_result
            if is_data_file(result.knowledge_filename)
        ]
        pipeline_ctx.merge_result = filter_out_table_chunks(pipeline_ctx.merge_result)
        if not data_files:
            return await next()

        # Only the first data file is processed for now to keep the turn bounded.
        target = data_files[0]
        try:
            knowledge = await self._knowledge_service.get_knowledge_by_id(ctx, target.knowledge_id)
        except Exception as exc:
            pipeline_error(
                "DataAnalysis",
                "get_knowledge_failed",
                {
                    "session_id": pipeline_ctx.session_id,
                    "knowledge_id": target.knowledge_id,
                    "error": str(exc),
                },
            )
            return await next()
        if knowledge is None:
            pipeline_error(
                "DataAnalysis",
                "get_knowledge_failed",
                {"session_id": pipeline_ctx.session_id, "knowledge_id": target.knowledge_id},
            )
            return await next()

        tool = self._tool_factory(pipeline_ctx.session_id)
        try:
            try:
                schema = await tool.load_from_knowledge(ctx, knowledge)
            except Exception as exc:
                pipeline_error(
                    "DataAnalysis",
                    "load_schema_failed",
                    {
                        "session_id": pipeline_ctx.session_id,
                        "knowledge_id": knowledge.id,
                        "error": str(exc),
                    },
                )
                return await next()

            try:
                chat_model = await self._model_service.get_chat_model(
                    ctx, pipeline_ctx.chat_model_id
                )
            except Exception as exc:
                return ERR_GET_CHAT_MODEL.with_error(exc)

            analysis_prompt = (
                f"\nUser Question: {pipeline_ctx.query}\n"
                f"Knowledge ID: {knowledge.id}\n"
                f"Table Schema: {schema.describe()}\n\n"
                "Determine if the user's question requires data analysis (e.g., "
                "statistics, aggregation, filtering) on this table.\n"
                "If YES, generate a DuckDB SQL query to answer the user's question and "
                "fill in the knowledge_id and sql fields.\n"
                "If NO, leave the sql field empty.\n\n"
                "Return your response in the specified JSON format."
            )
            try:
                with with_llm_call_metadata(purpose=_DATA_ANALYSIS_PLAN_PURPOSE):
                    response = await chat_model.chat(
                        [Message(role="user", content=analysis_prompt)],
                        ChatOptions(temperature=0.1, format=data_analysis_input_schema()),
                    )
            except Exception as exc:
                pipeline_error(
                    "DataAnalysis",
                    "plan_model_call_failed",
                    {
                        "session_id": pipeline_ctx.session_id,
                        "knowledge_id": knowledge.id,
                        "error": str(exc),
                    },
                )
                return await next()

            try:
                result = await tool.execute(ctx, response.content)
            except Exception as exc:
                pipeline_error(
                    "DataAnalysis",
                    "execute_sql_failed",
                    {
                        "session_id": pipeline_ctx.session_id,
                        "knowledge_id": knowledge.id,
                        "error": str(exc),
                    },
                )
                return await next()

            if not result.success:
                return await next()

            analysis_result = SearchResult(
                id=f"analysis_{knowledge.id}",
                content=result.output,
                score=1.0,
                match_type=MatchType.DATA_ANALYSIS,
                knowledge_id=knowledge.id,
                knowledge_title=knowledge.title or "",
                knowledge_filename=knowledge.file_name or "",
                knowledge_description=knowledge.description or "",
            )
            pipeline_ctx.merge_result = [*pipeline_ctx.merge_result, analysis_result]
            return await next()
        finally:
            # Session-scoped analysis tables are released whether or not the
            # turn short-circuits. Cleanup is best-effort: a failure is logged
            # and never masks the turn's own outcome.
            try:
                await tool.cleanup(ctx)
            except Exception as exc:
                pipeline_error(
                    "DataAnalysis",
                    "cleanup_failed",
                    {
                        "session_id": pipeline_ctx.session_id,
                        "knowledge_id": knowledge.id,
                        "error": str(exc),
                    },
                )


__all__ = [
    "DataAnalysisStep",
    "DataAnalysisTool",
    "KnowledgeService",
    "filter_out_table_chunks",
    "is_data_file",
]
