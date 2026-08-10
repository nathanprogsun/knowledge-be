"""Act phase: tool execution through the tool registry.

The act phase runs every tool call the model requested in one round,
sequentially or concurrently, converting each outcome into the engine's
persisted ``ToolCall`` shape. Argument payloads are parsed (with malformed
JSON repair), model handles are decoded through the model-context registry,
and any failure — unknown tool, unresolved handle, timeout, or raised
exception — becomes a failed ``ToolResult`` the model can react to rather
than a hard engine error.

The phase consumes the merged engine types (``ToolCall`` / ``ToolResult`` /
``AgentConfig``) and reaches tools only through the injected
``ToolRegistry``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time

from src.ai.embedding.base import Context
from src.ai.llm.types import ChatResponse, LLMToolCall
from src.common.json import JsonObject
from src.core.agents.engine.modelcontext import Registry
from src.core.agents.engine.types import AgentConfig, ToolCall, ToolResult
from src.core.agents.tools.exec_context import (
    DEFAULT_TOOL_EXEC_TIMEOUT,
    ToolExecContext,
    with_tool_exec_context,
)
from src.core.agents.tools.registry import ToolRegistry
from src.core.chat.bus import Event, EventBus
from src.core.chat.types import EventType

logger = logging.getLogger(__name__)

#: Hint appended to tool failures so the model retries a different approach.
_TOOL_ERROR_HINT = "\n\n[Analyze the error above and try a different approach.]"

#: Maximum tool-call id length; longer ids are truncated with a hash suffix.
_MAX_TOOL_CALL_ID_LEN = 64

#: Characters stripped from tool-call ids for provider compatibility.
_VALID_ID_CHARS_RE = re.compile(r"[^a-zA-Z0-9_-]")

#: User-friendly display labels for internal tool names (UI hints).
_TOOL_DISPLAY_NAMES: dict[str, str] = {
    "thinking": "深度思考",
    "todo_write": "制定计划",
    "grep_chunks": "关键词搜索",
    "knowledge_search": "知识搜索",
    "list_knowledge_chunks": "查看文档分块",
    "query_knowledge_graph": "查询知识图谱",
    "get_document_info": "获取文档信息",
    "database_query": "查询数据",
    "data_analysis": "数据分析",
    "data_schema": "查看数据结构",
    "web_search": "搜索网页",
    "web_fetch": "获取网页",
    "execute_skill_script": "执行技能脚本",
    "read_skill": "读取技能",
}

#: Tools whose arguments must not be shown in UI hints (e.g. raw SQL).
_TOOL_SENSITIVE_ARGS: frozenset[str] = frozenset({"database_query"})


# ── Tool argument helpers ─────────────────────────────────────────────


def _parse_tool_arguments(raw: str) -> JsonObject | None:
    """Parse ``raw`` as a JSON object; ``None`` when it is not an object."""
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _repair_json(raw: str) -> str:
    """Attempt to fix common JSON malformations from model outputs."""
    repaired = raw.strip()
    if repaired == "":
        return "{}"
    if not repaired.startswith("{"):
        if ":" in repaired or "=" in repaired:
            repaired = "{" + repaired + "}"
        else:
            return repaired
    repaired = _fix_invalid_escapes(repaired)
    repaired = _fix_trailing_commas(repaired)
    return _balance_brackets(repaired)


def _fix_invalid_escapes(text: str) -> str:
    """Rewrite invalid JSON string escapes into literal backslash sequences."""
    out: list[str] = []
    in_string = False
    index = 0
    runes = list(text)
    while index < len(runes):
        rune = runes[index]
        if not in_string:
            out.append(rune)
            if rune == '"':
                in_string = True
            index += 1
            continue
        if rune == '"':
            out.append(rune)
            in_string = False
            index += 1
            continue
        if rune != "\\":
            out.append(rune)
            index += 1
            continue
        if index + 1 >= len(runes):
            out.append("\\\\")
            index += 1
            continue
        nxt = runes[index + 1]
        if nxt in '"\\/bfnrtu':
            out.append(rune)
            out.append(nxt)
        else:
            out.append("\\\\")
            out.append(nxt)
        index += 2
    return "".join(out)


def _fix_trailing_commas(text: str) -> str:
    """Remove trailing commas before closing brackets and braces."""
    out: list[str] = []
    in_string = False
    escaped = False
    runes = list(text)
    for index, rune in enumerate(runes):
        if escaped:
            escaped = False
            out.append(rune)
            continue
        if rune == "\\" and in_string:
            escaped = True
            out.append(rune)
            continue
        if rune == '"':
            in_string = not in_string
            out.append(rune)
            continue
        if in_string:
            out.append(rune)
            continue
        if rune == ",":
            nxt = _find_next_non_space(runes, index + 1)
            if nxt >= 0 and runes[nxt] in "}]":
                continue
        out.append(rune)
    return "".join(out)


def _find_next_non_space(runes: list[str], start: int) -> int:
    """Return the index of the next non-whitespace rune, or ``-1``."""
    for index in range(start, len(runes)):
        if not runes[index].isspace():
            return index
    return -1


def _balance_brackets(text: str) -> str:
    """Append the missing closing brackets and braces."""
    stack: list[str] = []
    in_string = False
    escaped = False
    for rune in text:
        if escaped:
            escaped = False
            continue
        if rune == "\\" and in_string:
            escaped = True
            continue
        if rune == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if rune == "{":
            stack.append("}")
        elif rune == "[":
            stack.append("]")
        elif rune in "}]" and stack and stack[-1] == rune:
            stack.pop()
    if in_string:
        text += '"'
    for closer in reversed(stack):
        text += closer
    return text


def _normalize_tool_call_id(call_id: str, tool_name: str, index: int) -> str:
    """Normalize a model tool-call id for cross-provider compatibility."""
    call_id = call_id.strip()
    if not call_id:
        digest = hashlib.sha256(f"{tool_name}_{index}".encode()).digest()
        call_id = f"call_{digest[:6].hex()}"
    call_id = _VALID_ID_CHARS_RE.sub("_", call_id)
    if len(call_id) > _MAX_TOOL_CALL_ID_LEN:
        digest = hashlib.sha256(call_id.encode()).digest()
        suffix = f"_{digest[:4].hex()}"
        call_id = call_id[: _MAX_TOOL_CALL_ID_LEN - len(suffix)] + suffix
    return call_id


def _format_tool_hint(name: str, args: JsonObject) -> str:
    """Return a concise human-readable hint for one tool call."""
    display_name = _TOOL_DISPLAY_NAMES.get(name, name)
    if not args or name in _TOOL_SENSITIVE_ARGS:
        return display_name
    for value in args.values():
        if isinstance(value, str):
            if len(value) > 40:
                value = value[:40] + "…"
            return f'{display_name}("{value}")'
    return display_name


class ActPhase:
    """Executes the model's tool calls through the tool registry."""

    def __init__(
        self,
        config: AgentConfig,
        tool_registry: ToolRegistry,
        event_bus: EventBus,
        model_context: Registry,
        *,
        tool_exec_timeout: float = 0.0,
    ) -> None:
        self._config = config
        self._tool_registry = tool_registry
        self._event_bus = event_bus
        self._model_context = model_context
        self._tool_exec_timeout = (
            tool_exec_timeout if tool_exec_timeout > 0 else DEFAULT_TOOL_EXEC_TIMEOUT
        )

    async def execute_tool_calls(
        self,
        ctx: Context,
        response: ChatResponse,
        iteration: int,
        session_id: str,
        message_id: str,
    ) -> list[ToolCall]:
        """Run every tool call in ``response``, in order, emitting events."""
        calls = list(response.tool_calls)
        if not calls:
            return []
        if self._config.parallel_tool_calls and len(calls) >= 2:
            tool_calls = list(
                await asyncio.gather(
                    *(
                        self.run_tool_call(ctx, call, index, iteration, session_id, message_id)
                        for index, call in enumerate(calls)
                    )
                )
            )
        else:
            tool_calls = [
                await self.run_tool_call(ctx, call, index, iteration, session_id, message_id)
                for index, call in enumerate(calls)
            ]

        for tool_call in tool_calls:
            result = (
                tool_call.result
                if tool_call.result is not None
                else ToolResult(success=False, error="no result")
            )
            await self._emit(
                Event(
                    id=f"{tool_call.id}-tool-result",
                    type=EventType.AGENT_TOOL_RESULT,
                    session_id=session_id,
                    data={
                        "tool_call_id": tool_call.id,
                        "tool_name": tool_call.name,
                        "output": result.output,
                        "error": result.error,
                        "success": result.success,
                        "duration": tool_call.duration,
                        "iteration": iteration,
                        "data": result.data,
                    },
                )
            )
            await self._emit(
                Event(
                    id=f"{tool_call.id}-tool-exec",
                    type=EventType.AGENT_TOOL,
                    session_id=session_id,
                    data={
                        "iteration": iteration,
                        "tool_name": tool_call.name,
                        "tool_input": tool_call.args,
                        "tool_output": result.output,
                        "success": result.success,
                        "error": result.error,
                        "duration": tool_call.duration,
                    },
                )
            )
        return tool_calls

    async def run_tool_call(
        self,
        ctx: Context,
        tc: LLMToolCall,
        index: int,
        iteration: int,
        session_id: str,
        message_id: str,
    ) -> ToolCall:
        """Execute one tool call, converting any failure into a failed result."""
        call_id = _normalize_tool_call_id(tc.id, tc.function.name, index)
        args_str = tc.function.arguments

        args = _parse_tool_arguments(args_str)
        if args is None:
            # Malformed JSON: repair, then decode the repaired payload so model
            # handles can never survive a raw fallback into execution.
            repaired = _repair_json(args_str)
            args = _parse_tool_arguments(repaired)
            if args is None:
                return ToolCall(
                    id=call_id,
                    name=tc.function.name,
                    args={"_raw": args_str},
                    provider_metadata=dict(tc.provider_metadata),
                    result=ToolResult(
                        success=False,
                        error=f"Failed to parse tool arguments: {args_str}" + _TOOL_ERROR_HINT,
                    ),
                )
            decoded = tc.model_copy(
                update={
                    "model_arguments": "",
                    "function": tc.function.model_copy(update={"arguments": repaired}),
                }
            )
            self._model_context.decode_tool_calls([decoded])
            effective_args = decoded.function.arguments
            args = _parse_tool_arguments(effective_args) or {}
            unresolved = list(decoded.unresolved_handles)
        else:
            effective_args = args_str
            unresolved = list(tc.unresolved_handles)

        await self._emit(
            Event(
                id=f"{call_id}-tool-hint",
                type=EventType.AGENT_TOOL_CALL,
                session_id=session_id,
                data={
                    "tool_call_id": call_id,
                    "tool_name": tc.function.name,
                    "arguments": args,
                    "iteration": iteration,
                    "hint": _format_tool_hint(tc.function.name, args),
                },
            )
        )

        started = time.monotonic()
        if unresolved:
            # A temporary handle is not an application identity: fail before
            # execution so a hallucinated token can never reach a tool.
            result = ToolResult(
                success=False,
                error=f"tool arguments contain unresolved model handles: {unresolved}",
            )
        else:
            meta = ToolExecContext(
                session_id=session_id,
                assistant_message_id=message_id,
                tool_call_id=call_id,
                exec_timeout=self._tool_exec_timeout,
            )
            with with_tool_exec_context(meta):
                try:
                    registry_result = await asyncio.wait_for(
                        self._tool_registry.execute_tool(ctx, tc.function.name, effective_args),
                        timeout=self._tool_exec_timeout,
                    )
                except TimeoutError:
                    result = ToolResult(
                        success=False,
                        error=(f"tool execution timed out after {self._tool_exec_timeout:g}s"),
                    )
                except Exception as exc:
                    result = ToolResult(success=False, error=str(exc))
                else:
                    result = ToolResult.from_tools_result(registry_result)
        duration_ms = int((time.monotonic() - started) * 1000)
        return ToolCall(
            id=call_id,
            name=tc.function.name,
            args=args,
            result=result,
            duration=duration_ms,
            provider_metadata=dict(tc.provider_metadata),
        )

    async def _emit(self, event: Event) -> None:
        """Publish one event to the phase's event bus."""
        await self._event_bus.emit(event)


__all__ = ["ActPhase"]
