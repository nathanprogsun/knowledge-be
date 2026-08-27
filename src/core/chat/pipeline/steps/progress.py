"""Pipeline progress events (upstream ``progress.go``).

Emits ``tool_call`` / ``tool_result`` notifications on the chat event bus so
the frontend can render a single user-visible progress window for the
consolidated retrieval stage, plus a separate window for query
understanding. The tool names and structured payloads mirror the agent
tool-call vocabulary so existing streaming consumers render them unchanged.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from src.common.json import JsonObject, JsonValue
from src.core.agents.tools.base import TOOL_KNOWLEDGE_SEARCH
from src.core.chat.bus import Event
from src.core.chat.pipeline.common import pipeline_warn
from src.core.chat.pipeline.context import PipelineContext
from src.core.chat.pipeline.engine import ERR_SEARCH_NOTHING, PluginError
from src.core.chat.pipeline.types import EventType, SearchResult, SearchTargetType
from src.core.chat.types import EventType as ChatEventType
from src.core.knowledge.chunks.types import CHUNK_TYPE_WEB_SEARCH

#: Tool names the progress windows impersonate.
RETRIEVAL_PROGRESS_TOOL = TOOL_KNOWLEDGE_SEARCH
QUERY_UNDERSTAND_PROGRESS_TOOL = "query_understand"

#: Search-source labels carried in the progress payloads.
RETRIEVAL_SOURCE_KNOWLEDGE = "knowledge"
RETRIEVAL_SOURCE_WEB = "web"
RETRIEVAL_SOURCE_MIXED = "mixed"


@dataclass(frozen=True, slots=True)
class StageProgress:
    """An in-flight pipeline progress ``tool_call``."""

    tool_call_id: str
    tool_name: str


@runtime_checkable
class ProgressEventBus(Protocol):
    """The event-bus surface progress notifications need."""

    async def emit(self, event: Event) -> None: ...


def should_emit_query_understand_progress(pipeline_ctx: PipelineContext) -> bool:
    """Return whether the query-understand stage will actually run."""
    return pipeline_ctx.enable_rewrite or len(pipeline_ctx.images) > 0


def is_consolidated_retrieval_stage(
    stage: EventType,
    pipeline_ctx: PipelineContext,
) -> bool:
    """Return whether ``stage`` belongs to the single retrieval window."""
    if stage in {
        EventType.CHUNK_SEARCH_PARALLEL,
        EventType.CHUNK_RERANK,
        EventType.CHUNK_MERGE,
        EventType.FILTER_TOP_K,
    }:
        return pipeline_ctx.needs_retrieval()
    if stage == EventType.WEB_FETCH:
        return pipeline_ctx.web_search_enabled
    if stage == EventType.DATA_ANALYSIS:
        return pipeline_ctx.data_analysis_enabled and pipeline_ctx.needs_retrieval()
    return False


def last_consolidated_retrieval_stage(
    event_list: list[EventType],
    pipeline_ctx: PipelineContext,
) -> EventType | None:
    """Return the last retrieval-related stage in the assembled pipeline."""
    last: EventType | None = None
    for stage in event_list:
        if is_consolidated_retrieval_stage(stage, pipeline_ctx):
            last = stage
    return last


def should_close_retrieval_progress(
    stage: EventType,
    last_retrieval_stage: EventType | None,
    stage_err: PluginError | None,
) -> bool:
    """Return whether the retrieval progress window must close after ``stage``.

    Closing on the error paths prevents the frontend spinner from hanging
    forever when the pipeline early-returns before the last retrieval stage
    — including ``ERR_SEARCH_NOTHING``, which routes into the fallback
    response.
    """
    return stage == last_retrieval_stage or stage_err is not None


def _duration_ms(start: float) -> int:
    """Return the elapsed wall time in milliseconds since ``start``."""
    return max(0, int((time.monotonic() - start) * 1000))


def _stage_error_message(stage_err: PluginError | None) -> str:
    """Return the wrapped exception text, or the empty string."""
    if stage_err is None or stage_err.err is None:
        return ""
    return str(stage_err.err)


async def _emit_tool_call(
    pipeline_ctx: PipelineContext,
    event_bus: ProgressEventBus,
    *,
    tool_call_id: str,
    tool_name: str,
    arguments: dict[str, JsonValue],
) -> None:
    payload: JsonObject = {
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
        "arguments": arguments,
    }
    try:
        await event_bus.emit(
            Event(
                type=ChatEventType.AGENT_TOOL_CALL,
                session_id=pipeline_ctx.session_id,
                data=payload,
            )
        )
    except Exception as exc:
        pipeline_warn(
            "Progress",
            "tool_call_emit",
            {"tool_name": tool_name, "error": str(exc)},
        )


async def _emit_tool_result(
    pipeline_ctx: PipelineContext,
    event_bus: ProgressEventBus,
    *,
    tool_call_id: str,
    tool_name: str,
    output: str,
    error: str,
    success: bool,
    duration_ms: int,
    data: JsonObject | None = None,
) -> None:
    payload: JsonObject = {
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
        "output": output,
        "error": error,
        "success": success,
        "duration_ms": duration_ms,
    }
    if data is not None:
        payload["data"] = data
    try:
        await event_bus.emit(
            Event(
                type=ChatEventType.AGENT_TOOL_RESULT,
                session_id=pipeline_ctx.session_id,
                data=payload,
            )
        )
    except Exception as exc:
        pipeline_warn(
            "Progress",
            "tool_result_emit",
            {"tool_name": tool_name, "error": str(exc)},
        )


async def begin_retrieval_progress(
    pipeline_ctx: PipelineContext,
    event_bus: ProgressEventBus | None,
) -> StageProgress | None:
    """Emit a single pending ``knowledge_search`` tool_call."""
    if event_bus is None:
        return None

    tool_call_id = str(uuid.uuid4())
    arguments: dict[str, JsonValue] = {"search_source": _retrieval_search_source(pipeline_ctx)}
    if pipeline_ctx.rewrite_query != "":
        arguments["query"] = pipeline_ctx.rewrite_query
    elif pipeline_ctx.query != "":
        arguments["query"] = pipeline_ctx.query
    await _emit_tool_call(
        pipeline_ctx,
        event_bus,
        tool_call_id=tool_call_id,
        tool_name=RETRIEVAL_PROGRESS_TOOL,
        arguments=arguments,
    )
    return StageProgress(tool_call_id=tool_call_id, tool_name=RETRIEVAL_PROGRESS_TOOL)


async def begin_query_understand_progress(
    pipeline_ctx: PipelineContext,
    event_bus: ProgressEventBus | None,
) -> StageProgress | None:
    """Emit a pending ``query_understand`` tool_call when the stage will run."""
    if event_bus is None or not should_emit_query_understand_progress(pipeline_ctx):
        return None

    tool_call_id = str(uuid.uuid4())
    arguments: dict[str, JsonValue] = {}
    if pipeline_ctx.query != "":
        arguments["query"] = pipeline_ctx.query
    if len(pipeline_ctx.images) > 0:
        arguments["has_images"] = True
    await _emit_tool_call(
        pipeline_ctx,
        event_bus,
        tool_call_id=tool_call_id,
        tool_name=QUERY_UNDERSTAND_PROGRESS_TOOL,
        arguments=arguments,
    )
    return StageProgress(tool_call_id=tool_call_id, tool_name=QUERY_UNDERSTAND_PROGRESS_TOOL)


async def end_query_understand_progress(
    pipeline_ctx: PipelineContext,
    progress: StageProgress | None,
    start: float,
    stage_err: PluginError | None,
    event_bus: ProgressEventBus | None,
) -> None:
    """Emit the matching ``query_understand`` tool_result."""
    if progress is None or event_bus is None:
        return

    success = stage_err is None
    output = "已完成问题理解" if success else ""
    error = "" if success else _stage_error_message(stage_err)
    await _emit_tool_result(
        pipeline_ctx,
        event_bus,
        tool_call_id=progress.tool_call_id,
        tool_name=QUERY_UNDERSTAND_PROGRESS_TOOL,
        output=output,
        error=error,
        success=success,
        duration_ms=_duration_ms(start),
    )


async def end_retrieval_progress(
    pipeline_ctx: PipelineContext,
    progress: StageProgress | None,
    start: float,
    stage_err: PluginError | None,
    event_bus: ProgressEventBus | None,
) -> None:
    """Emit the matching ``knowledge_search`` tool_result."""
    if progress is None or event_bus is None:
        return

    count, doc_count, web_count = _retrieval_result_breakdown(pipeline_ctx)
    search_source = _retrieval_search_source(pipeline_ctx)
    if count > 0:
        if doc_count > 0 and web_count > 0:
            search_source = RETRIEVAL_SOURCE_MIXED
        elif web_count > 0:
            search_source = RETRIEVAL_SOURCE_WEB
        else:
            search_source = RETRIEVAL_SOURCE_KNOWLEDGE

    success = stage_err is None or stage_err is ERR_SEARCH_NOTHING
    output = (
        "" if not success else ("未检索到相关内容" if count == 0 else f"检索到 {count} 条相关内容")
    )
    error = "" if success else _stage_error_message(stage_err)
    data: JsonObject = {
        "count": count,
        "doc_count": doc_count,
        "web_count": web_count,
        "search_source": search_source,
    }
    await _emit_tool_result(
        pipeline_ctx,
        event_bus,
        tool_call_id=progress.tool_call_id,
        tool_name=RETRIEVAL_PROGRESS_TOOL,
        output=output,
        error=error,
        success=success,
        duration_ms=_duration_ms(start),
        data=data,
    )


def _has_knowledge_retrieval_scope(pipeline_ctx: PipelineContext) -> bool:
    """Return whether any knowledge retrieval target is configured."""
    if any(kb_id != "" for kb_id in pipeline_ctx.knowledge_base_ids):
        return True
    if any(knowledge_id != "" for knowledge_id in pipeline_ctx.knowledge_ids):
        return True
    for target in pipeline_ctx.search_targets:
        if target.knowledge_base_id == "":
            continue
        if (
            target.type == SearchTargetType.KNOWLEDGE_BASE
            or len(target.knowledge_ids) > 0
            or len(target.tag_ids) > 0
            or len(target.scope_tag_ids) > 0
        ):
            return True
    return False


def _retrieval_search_source(pipeline_ctx: PipelineContext) -> str:
    """Return the search-source label for the begin / end payloads."""
    has_kb = _has_knowledge_retrieval_scope(pipeline_ctx)
    has_web = pipeline_ctx.web_search_enabled
    if has_kb and has_web:
        return RETRIEVAL_SOURCE_MIXED
    if has_web:
        return RETRIEVAL_SOURCE_WEB
    return RETRIEVAL_SOURCE_KNOWLEDGE


def _retrieval_results(pipeline_ctx: PipelineContext) -> list[SearchResult]:
    """Return the best available result set for the turn."""
    if len(pipeline_ctx.merge_result) > 0:
        return pipeline_ctx.merge_result
    if len(pipeline_ctx.rerank_result) > 0:
        return pipeline_ctx.rerank_result
    return pipeline_ctx.search_result


def _retrieval_result_breakdown(
    pipeline_ctx: PipelineContext,
) -> tuple[int, int, int]:
    """Count total / knowledge / web results across the best available set."""
    total = 0
    doc_count = 0
    web_count = 0
    for result in _retrieval_results(pipeline_ctx):
        total += 1
        if _is_web_search_result(result):
            web_count += 1
        else:
            doc_count += 1
    return total, doc_count, web_count


def _is_web_search_result(result: SearchResult) -> bool:
    """Return whether ``result`` is a web-search hit."""
    return result.chunk_type.lower() == CHUNK_TYPE_WEB_SEARCH or (
        result.knowledge_source.lower() == "web_search"
    )


__all__ = [
    "QUERY_UNDERSTAND_PROGRESS_TOOL",
    "RETRIEVAL_PROGRESS_TOOL",
    "RETRIEVAL_SOURCE_KNOWLEDGE",
    "RETRIEVAL_SOURCE_MIXED",
    "RETRIEVAL_SOURCE_WEB",
    "ProgressEventBus",
    "StageProgress",
    "begin_query_understand_progress",
    "begin_retrieval_progress",
    "end_query_understand_progress",
    "end_retrieval_progress",
    "is_consolidated_retrieval_stage",
    "last_consolidated_retrieval_stage",
    "should_close_retrieval_progress",
    "should_emit_query_understand_progress",
]
