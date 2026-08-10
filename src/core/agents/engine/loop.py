"""Agent engine core loop: think, act, observe, finalize.

``AgentEngine`` runs the ReAct cycle against a tool registry and a chat
model:

- *think* streams the model response, emitting thought / answer / tool-call
  events, with retry on transient LLM errors and graceful degradation when
  prior tool results exist;
- *analyze* inspects the response for stop conditions (natural stop, content
  filter, empty content, repeated content) and either ends the turn or
  continues;
- *act* executes the model's tool calls through the registry, sequentially or
  in parallel, converting every failure into a failed tool result the model
  can react to;
- *observe* appends the assistant step and the rendered tool results back
  onto the message list for the next round;
- *finalize* synthesizes a final answer from the gathered tool results when
  the loop exhausts its iteration ceiling, and always emits exactly one
  completion event per run.

Conversation history is rebuilt by the caller once per turn and passed into
:meth:`AgentEngine.execute`; the engine keeps no cross-turn cache.

The loop is async throughout. Cancellation is signalled through the
``is_cancelled`` callable seam (the async analogue of a cancellable context):
checked at the loop head (salvage + :class:`AgentCancelledError`) and after
each streaming LLM call (preserve the partial step and exit).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

from src.ai.embedding.base import Context
from src.ai.llm.types import (
    Chat,
    ChatOptions,
    ChatResponse,
    FunctionCall,
    FunctionDef,
    LLMToolCall,
    Message,
    ResponseType,
    SearchResult,
    StreamResponse,
    TokenUsage,
    Tool,
)
from src.ai.llm.types import (
    ToolCall as MessageToolCall,
)
from src.common.json import JsonObject
from src.core.agents.engine.modelcontext import Registry
from src.core.agents.engine.modelcontext.model_output import ToolResult as ModelToolResult
from src.core.agents.engine.types import (
    CONTENT_FILTER_FALLBACK,
    EMPTY_RESPONSE_FALLBACK,
    EMPTY_RESPONSE_NUDGE,
    MAX_EMPTY_RESPONSE_RETRIES,
    MAX_ITERATIONS_FALLBACK,
    MAX_LLM_RETRIES,
    MAX_REPEATED_RESPONSE_ROUNDS,
    AgentCancelledError,
    AgentConfig,
    AgentExecutionError,
    AgentLLMError,
    AgentState,
    AgentStep,
    IterOutcome,
    KnowledgeBaseInfo,
    ResponseVerdict,
    SelectedDocumentInfo,
    ToolCall,
    ToolResult,
    generate_event_id,
    is_natural_stop_finish_reason,
    is_transient_error,
)
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

#: Markdown image link matcher used by the final-answer image requirement.
_MARKDOWN_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

#: Inline reasoning markers some models embed in the plain content channel.
_THINK_OPEN_TAG = "<think>"
_THINK_CLOSE_TAG = "</think>"

#: Strips complete ``<think>…</think>`` blocks from accumulated content.
_THINK_BLOCK_RE = re.compile(r"(?s)<think>.*?</think>")

#: Tool names whose historical results may be stale across turns and are
#: therefore redacted when replaying prior context (forces fresh retrieval).
_KB_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "knowledge_search",
        "grep_chunks",
        "list_knowledge_chunks",
        "query_knowledge_graph",
        "get_document_info",
        "wiki_search",
        "wiki_read_page",
        "wiki_read_source_doc",
    }
)

#: Default system prompt used when neither a template nor a configured
#: custom prompt is supplied.
DEFAULT_SYSTEM_PROMPT = """You are a helpful knowledge assistant.

Answer the user's question by gathering information with the available tools.
Keep using tools until you have enough information; do not give a partial
answer mid-investigation. When you are ready, write your complete user-facing
answer as your reply and stop — do not request any more tools in that final
message. If the retrieved information is insufficient, say so honestly.
Respond in the same language as the user's question."""

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

#: Maximum tool-call id length; longer ids are truncated with a hash suffix.
_MAX_TOOL_CALL_ID_LEN = 64

#: Characters stripped from tool-call ids for provider compatibility.
_VALID_ID_CHARS_RE = re.compile(r"[^a-zA-Z0-9_-]")


def _never_cancelled() -> bool:
    """Default cancellation seam: the run never observes cancellation."""
    return False


# ── Text helpers ──────────────────────────────────────────────────────


def strip_think_blocks(content: str) -> str:
    """Remove ``<think>…</think>`` blocks and trim leftover whitespace."""
    if not content:
        return ""
    return _THINK_BLOCK_RE.sub("", content).strip(" \t\r\n")


class _ThinkStreamSplitter:
    """Incrementally separate inline thinking from answer text.

    Streaming counterpart to :func:`strip_think_blocks`: tag boundaries
    straddling two chunks are buffered until the next feed so a partial tag can
    never leak into the answer channel. Not safe for concurrent use.
    """

    def __init__(self) -> None:
        self._in_think = False
        self._pending = ""

    def feed(self, chunk: str) -> tuple[str, str]:
        """Consume one chunk; return ``(think_out, answer_out)`` portions."""
        if chunk == "":
            return "", ""
        self._pending += chunk
        think: list[str] = []
        answer: list[str] = []
        while True:
            if self._in_think:
                idx = self._pending.find(_THINK_CLOSE_TAG)
                if idx >= 0:
                    think.append(self._pending[:idx])
                    self._pending = self._pending[idx + len(_THINK_CLOSE_TAG) :]
                    self._in_think = False
                    continue
                safe, hold = _hold_back_partial_tag(self._pending, _THINK_CLOSE_TAG)
                think.append(safe)
                self._pending = hold
                return "".join(think), "".join(answer)
            idx = self._pending.find(_THINK_OPEN_TAG)
            if idx >= 0:
                answer.append(self._pending[:idx])
                self._pending = self._pending[idx + len(_THINK_OPEN_TAG) :]
                self._in_think = True
                continue
            safe, hold = _hold_back_partial_tag(self._pending, _THINK_OPEN_TAG)
            answer.append(safe)
            self._pending = hold
            return "".join(think), "".join(answer)

    def flush(self) -> tuple[str, str]:
        """Drain buffered remainder at end-of-stream."""
        rest = self._pending
        self._pending = ""
        if rest == "":
            return "", ""
        if self._in_think:
            return rest, ""
        return "", rest


def _hold_back_partial_tag(text: str, tag: str) -> tuple[str, str]:
    """Split ``text`` into an emit-now part and a trailing tag prefix."""
    max_k = min(len(tag) - 1, len(text))
    for k in range(max_k, 0, -1):
        if text.endswith(tag[:k]):
            return text[: len(text) - k], text[len(text) - k :]
    return text, ""


def _rfc3339_now() -> str:
    """Return the current UTC time in RFC 3339 (second) format."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _escape_xml_attr(value: str) -> str:
    """Escape a string for safe inclusion in an XML attribute value."""
    return (
        value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def _escape_xml_text(value: str) -> str:
    """Escape a string for safe inclusion in XML element content."""
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _indent_lines(text: str, indent: str) -> str:
    """Prefix every non-empty line of ``text`` with ``indent``."""
    if not text:
        return ""
    return "\n".join(indent + line if line else line for line in text.split("\n"))


def _compose_user_turn(*parts: str) -> str:
    """Join the non-empty user-turn parts with blank lines."""
    return "\n\n".join(part for part in parts if part.strip())


# ── Message sanitisation ──────────────────────────────────────────────


def _sanitize_messages(messages: Sequence[Message]) -> list[Message]:
    """Validate and repair a message array for provider compatibility.

    Drops empty non-system messages, merges consecutive same-role messages,
    and converts orphaned tool results (no matching assistant tool call) into
    system messages.
    """
    if not messages:
        return list(messages)
    result: list[Message] = []
    for index, msg in enumerate(messages):
        if not msg.content and msg.role not in {"system", "tool"} and not msg.tool_calls:
            continue
        if result and msg.role != "tool":
            prev = result[-1]
            if prev.role == msg.role and prev.role != "tool":
                result[-1] = prev.model_copy(update={"content": f"{prev.content}\n\n{msg.content}"})
                continue
        if (
            msg.role == "tool"
            and msg.tool_call_id
            and not _has_matching_tool_call(messages[:index], msg.tool_call_id)
        ):
            result.append(
                msg.model_copy(
                    update={
                        "role": "system",
                        "content": f"[Tool result for {msg.name}]: {msg.content}",
                        "tool_call_id": "",
                        "name": "",
                    }
                )
            )
            continue
        result.append(msg)
    return result


def _has_matching_tool_call(messages: Sequence[Message], tool_call_id: str) -> bool:
    """Return whether a preceding assistant message references the id."""
    for msg in reversed(messages):
        if msg.role != "assistant":
            continue
        for call in msg.tool_calls:
            if call.id == tool_call_id:
                return True
    return False


def _redact_history_kb_results(llm_context: Sequence[Message]) -> list[Message]:
    """Replace stale knowledge-tool results in history with brief markers."""
    redacted: list[Message] = []
    for msg in llm_context:
        if msg.role == "tool" and msg.name in _KB_TOOL_NAMES:
            redacted.append(
                Message(
                    role="tool",
                    content=(
                        "[Previous retrieval result omitted — knowledge base may "
                        "have changed. Please perform a fresh search.]"
                    ),
                    tool_call_id=msg.tool_call_id,
                    name=msg.name,
                )
            )
        else:
            redacted.append(msg)
    return redacted


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


def _to_model_output_result(result: ToolResult) -> ModelToolResult:
    """Map the engine's tool result onto the model-context rendering shape."""
    return ModelToolResult(
        success=result.success,
        output=result.output,
        data=result.data,
        error=result.error,
        images=list(result.images),
    )


def _dump_args(args: JsonObject) -> str:
    """Serialize tool arguments back to a JSON string for replay."""
    return json.dumps(args, ensure_ascii=False)


# ── Prompt helpers ────────────────────────────────────────────────────


def _render_prompt_placeholders(
    template: str,
    knowledge_bases: Sequence[KnowledgeBaseInfo],
    web_search_enabled: bool,
    current_time: str,
) -> str:
    """Render the system-prompt placeholders with runtime values."""
    result = template
    if "{{knowledge_bases}}" in result:
        if not knowledge_bases:
            result = result.replace(
                "{{knowledge_bases}}", "(no knowledge bases bound to this session)"
            )
        else:
            result = result.replace(
                "{{knowledge_bases}}",
                "(see `<bound_knowledge_bases>` inside the user message's "
                "`<runtime_context>` for the current bound KB list and their capabilities)",
            )
    result = result.replace(
        "{{web_search_status}}", "Enabled" if web_search_enabled else "Disabled"
    )
    result = result.replace("{{current_time}}", current_time)
    return result.replace("{{language}}", "")


def _format_knowledge_base_list(kbs: Sequence[KnowledgeBaseInfo]) -> str:
    """Render knowledge-base metadata as the XML block used in prompts."""
    if not kbs:
        return "<knowledge_bases />"
    out = ["<knowledge_bases>"]
    for kb in kbs:
        kb_type = kb.type or "document"
        capabilities = ""
        if kb.capabilities:
            capabilities = f' capabilities="{",".join(kb.capabilities)}"'
        out.append(
            f'<knowledge_base id="{_escape_xml_attr(kb.id)}" '
            f'name="{_escape_xml_attr(kb.name)}" type="{kb_type}" '
            f'doc_count="{kb.doc_count}"{capabilities}>'
        )
        if kb.description:
            out.append(f"<description>{_escape_xml_text(kb.description)}</description>")
        out.append("</knowledge_base>")
    out.append("</knowledge_bases>")
    return "\n".join(out)


def _final_answer_image_requirement(has_retrieved_image: bool) -> str:
    """Return the image-output requirement when retrieved results show images."""
    if not has_retrieved_image:
        return ""
    return (
        "5. Retrieved tool results contain Markdown images. Unless the user "
        "explicitly requested text-only output or every image is clearly "
        "unrelated, the final answer MUST include at least one relevant "
        "Markdown image copied verbatim from the tool results. Preserve its "
        "complete URL exactly. Use ASCII half-width parentheses exactly as "
        "![alt](url) and never use full-width \uff08 or \uff09. Place the image "
        "immediately after the paragraph it supports. When multiple images "
        "support different sections, distribute them across those sections "
        "instead of stopping after the first image.\n"
        "6. Before finishing, silently verify that the answer contains a "
        "Markdown image when requirement 5 applies."
    )


# ── Stream / loop carriers ────────────────────────────────────────────


@dataclass
class _StreamResult:
    """Accumulated output of one streaming LLM call."""

    content: str = ""
    reasoning_content: str = ""
    tool_calls: list[LLMToolCall] = field(default_factory=list)
    usage: TokenUsage | None = None
    finish_reason: str = ""
    stream_error: str = ""


@dataclass
class _RunState:
    """Mutable working state for one run (frozen into ``AgentState`` at exit)."""

    current_round: int = 0
    round_steps: list[AgentStep] = field(default_factory=list)
    is_complete: bool = False
    final_answer: str = ""
    knowledge_refs: list[SearchResult] = field(default_factory=list)
    empty_retries: int = 0
    consecutive_same_content: int = 0
    last_response_content: str = ""

    def total_tool_calls(self) -> int:
        """Return the number of tool calls across every step."""
        return sum(len(step.tool_calls) for step in self.round_steps)

    def freeze(self) -> AgentState:
        """Project the working state onto the frozen contract shape."""
        return AgentState(
            current_round=self.current_round,
            round_steps=list(self.round_steps),
            is_complete=self.is_complete,
            final_answer=self.final_answer,
            knowledge_refs=list(self.knowledge_refs),
        )


# ── Agent engine ──────────────────────────────────────────────────────


class AgentEngine:
    """Core ReAct agent engine (think / act / observe / finalize)."""

    def __init__(
        self,
        config: AgentConfig,
        tool_registry: ToolRegistry,
        chat_model: Chat,
        event_bus: EventBus,
        *,
        model_context: Registry | None = None,
        system_prompt_template: str = "",
        knowledge_bases: Sequence[KnowledgeBaseInfo] = (),
        selected_docs: Sequence[SelectedDocumentInfo] = (),
        tool_exec_timeout: float = 0.0,
    ) -> None:
        self._config = config
        self._tool_registry = tool_registry
        self._chat_model = chat_model
        self._event_bus = event_bus
        self._model_context = model_context or Registry(config.citations_enabled())
        self._system_prompt_template = system_prompt_template
        self._knowledge_bases = list(knowledge_bases)
        self._selected_docs = list(selected_docs)
        self._tool_exec_timeout = (
            tool_exec_timeout if tool_exec_timeout > 0 else DEFAULT_TOOL_EXEC_TIMEOUT
        )

    # ── public API ──────────────────────────────────────────────────

    async def execute(
        self,
        ctx: Context,
        *,
        query: str,
        session_id: str,
        message_id: str,
        llm_context: Sequence[Message] | None = None,
        image_urls: Sequence[str] | None = None,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> AgentState:
        """Run the full agent loop and return the terminal frozen state.

        Raises ``AgentExecutionError`` subclasses on irrecoverable failures;
        the cancelled run raises ``AgentCancelledError`` carrying the salvaged
        partial state.
        """
        system_prompt = self._build_system_prompt()
        messages = self._build_messages_with_llm_context(
            system_prompt, query, session_id, list(llm_context or ()), list(image_urls or ())
        )
        tools = self._build_tools_for_llm()
        state = _RunState()
        try:
            await self._execute_loop(
                ctx,
                state,
                query,
                messages,
                tools,
                session_id,
                message_id,
                is_cancelled or _never_cancelled,
            )
        except AgentExecutionError as exc:
            await self._emit(
                Event(
                    type=EventType.ERROR,
                    session_id=session_id,
                    data={
                        "error": str(exc),
                        "stage": "agent_execution",
                        "session_id": session_id,
                    },
                )
            )
            raise
        finally:
            await self._tool_registry.cleanup(ctx)
        return state.freeze()

    # ── loop ────────────────────────────────────────────────────────

    async def _execute_loop(
        self,
        ctx: Context,
        state: _RunState,
        query: str,
        messages: list[Message],
        tools: list[Tool],
        session_id: str,
        message_id: str,
        is_cancelled: Callable[[], bool],
    ) -> _RunState:
        start_time = time.monotonic()
        completion_emitted = False

        async def emit_completion() -> None:
            nonlocal completion_emitted
            if completion_emitted:
                return
            completion_emitted = True
            await self._emit_completion_event(state, session_id, message_id, start_time)

        try:
            while state.current_round < self._config.max_iterations:
                if is_cancelled():
                    # Salvage what completed before the stop.
                    if state.total_tool_calls() > 0:
                        await self._best_effort_synthesize(ctx, query, state, session_id)
                        state.is_complete = True
                    raise AgentCancelledError(state=state.freeze())
                outcome = await self._run_iteration(
                    ctx, state, messages, tools, session_id, message_id, query, is_cancelled
                )
                if outcome is IterOutcome.CONTINUE:
                    continue
                if outcome is IterOutcome.BREAK:
                    break
                state.current_round += 1
        finally:
            await emit_completion()

        if not state.is_complete and not is_cancelled():
            await self._handle_max_iterations(ctx, query, state, session_id)
        return state

    async def _run_iteration(
        self,
        ctx: Context,
        state: _RunState,
        messages: list[Message],
        tools: list[Tool],
        session_id: str,
        message_id: str,
        query: str,
        is_cancelled: Callable[[], bool],
    ) -> IterOutcome:
        # Think: call the LLM (retry + graceful degradation handled inside).
        response = await self._call_llm_with_retry(
            ctx, messages, tools, state, query, state.current_round, session_id
        )
        if response is None:
            # Graceful degradation succeeded; state is already complete.
            return IterOutcome.BREAK

        # Detect stuck loops: repeated identical content with no tool calls.
        if not response.tool_calls and response.content:
            if response.content == state.last_response_content:
                state.consecutive_same_content += 1
            else:
                state.consecutive_same_content = 0
            state.last_response_content = response.content
            if state.consecutive_same_content >= MAX_REPEATED_RESPONSE_ROUNDS:
                state.final_answer = response.content
                state.is_complete = True
                return IterOutcome.BREAK
        else:
            state.consecutive_same_content = 0
            state.last_response_content = ""

        # Cancelled mid-stream: preserve the partial step and exit quietly.
        if is_cancelled():
            if response.content or response.tool_calls:
                state.round_steps.append(self._build_step(state, response))
            return IterOutcome.BREAK

        # Analyze: natural stop, content filter, or continue to act.
        verdict = await self._analyze_response(ctx, response, state, session_id)
        if verdict.is_done:
            if verdict.empty_content:
                state.empty_retries += 1
                if state.empty_retries <= MAX_EMPTY_RESPONSE_RETRIES:
                    messages.append(Message(role="user", content=EMPTY_RESPONSE_NUDGE))
                    return IterOutcome.CONTINUE
                state.final_answer = EMPTY_RESPONSE_FALLBACK
                state.is_complete = True
                if verdict.step is not None:
                    state.round_steps.append(verdict.step)
                return IterOutcome.BREAK
            state.final_answer = verdict.final_answer
            state.is_complete = True
            if verdict.step is not None:
                state.round_steps.append(verdict.step)
            return IterOutcome.BREAK

        # Act: execute the model's tool calls through the registry.
        tool_calls = await self._execute_tool_calls(
            ctx, response, state.current_round, session_id, message_id
        )

        # Observe: persist the step and append rendered tool results.
        step = AgentStep(
            iteration=state.current_round,
            thought=response.content,
            reasoning_content=response.reasoning_content,
            tool_calls=tool_calls,
            timestamp=datetime.now(UTC),
        )
        state.round_steps.append(step)
        self._append_tool_results(messages, step)
        return IterOutcome.NEXT

    def _build_step(self, state: _RunState, response: ChatResponse) -> AgentStep:
        """Build a step for a terminal iteration (no tool calls yet)."""
        return AgentStep(
            iteration=state.current_round,
            thought=response.content,
            reasoning_content=response.reasoning_content,
            tool_calls=[],
            timestamp=datetime.now(UTC),
        )

    # ── think ───────────────────────────────────────────────────────

    async def _call_llm_with_retry(
        self,
        ctx: Context,
        messages: list[Message],
        tools: list[Tool],
        state: _RunState,
        query: str,
        iteration: int,
        session_id: str,
    ) -> ChatResponse | None:
        sanitized = _sanitize_messages(messages)
        response: ChatResponse | None = None
        err: AgentLLMError | None = None
        try:
            response = await self._stream_thinking_to_events(
                ctx, sanitized, tools, iteration, session_id
            )
        except AgentLLMError as exc:
            err = exc
        if err is not None and is_transient_error(err):
            for retry in range(1, MAX_LLM_RETRIES + 1):
                await asyncio.sleep(retry)
                try:
                    response = await self._stream_thinking_to_events(
                        ctx, sanitized, tools, iteration, session_id
                    )
                    err = None
                    break
                except AgentLLMError as retry_err:
                    err = retry_err
                    if not is_transient_error(retry_err):
                        break
        if err is not None:
            if state.total_tool_calls() > 0:
                # Graceful degradation: synthesize from the existing results.
                try:
                    await self._synthesize_final_answer(ctx, query, state, session_id)
                except AgentExecutionError as synth_err:
                    raise AgentLLMError(
                        f"LLM call failed: {err} (synthesis also failed: {synth_err})"
                    ) from synth_err
                state.is_complete = True
                return None
            raise AgentLLMError(f"LLM call failed: {err}") from err
        if response is None:
            raise AgentLLMError("LLM call failed")
        return response

    async def _stream_thinking_to_events(
        self,
        ctx: Context,
        messages: list[Message],
        tools: list[Tool],
        iteration: int,
        session_id: str,
    ) -> ChatResponse:
        opts = ChatOptions(
            temperature=self._config.temperature,
            tools=list(tools),
            thinking=self._config.thinking,
            parallel_tool_calls=True,
        )
        thinking_id = generate_event_id("thinking")
        answer_id = generate_event_id("answer")
        splitter = _ThinkStreamSplitter()
        thinking_open = False
        answer_streamed = False
        thinking_tool_ids: dict[str, str] = {}
        pending_tool_calls: set[str] = set()

        async def emit_thought(content: str, done: bool) -> None:
            if content == "" and not done:
                return
            await self._emit(
                Event(
                    id=thinking_id,
                    type=EventType.AGENT_THOUGHT,
                    session_id=session_id,
                    data={"content": content, "iteration": iteration, "done": done},
                )
            )

        async def close_thinking() -> None:
            nonlocal thinking_open
            if thinking_open:
                await emit_thought("", True)
                thinking_open = False

        async def emit_answer(content: str) -> None:
            nonlocal answer_streamed
            if content == "":
                return
            if not answer_streamed and content.strip() == "":
                return
            await close_thinking()
            answer_streamed = True
            await self._emit(
                Event(
                    id=answer_id,
                    type=EventType.AGENT_FINAL_ANSWER,
                    session_id=session_id,
                    data={"content": content, "done": False},
                )
            )

        async def emit_chunk(chunk: StreamResponse, full_content: str) -> None:
            nonlocal thinking_open
            if chunk.response_type is ResponseType.TOOL_CALL and chunk.data is not None:
                tool_call_id = str(chunk.data.get("tool_call_id", ""))
                tool_name = str(chunk.data.get("tool_name", ""))
                if tool_call_id and tool_name and tool_call_id not in pending_tool_calls:
                    pending_tool_calls.add(tool_call_id)
                    await self._emit(
                        Event(
                            id=f"{tool_call_id}-tool-call-pending",
                            type=EventType.AGENT_TOOL_CALL,
                            session_id=session_id,
                            data={
                                "tool_call_id": tool_call_id,
                                "tool_name": tool_name,
                                "iteration": iteration,
                            },
                        )
                    )
            if (
                chunk.response_type is ResponseType.THINKING
                and chunk.data is not None
                and chunk.data.get("source") == "thinking_tool"
            ):
                tool_call_id = str(chunk.data.get("tool_call_id", ""))
                event_id = thinking_tool_ids.get(tool_call_id)
                if event_id is None:
                    event_id = generate_event_id("thinking-tool")
                    thinking_tool_ids[tool_call_id] = event_id
                await self._emit(
                    Event(
                        id=event_id,
                        type=EventType.AGENT_THOUGHT,
                        session_id=session_id,
                        data={"content": chunk.content, "iteration": iteration, "done": False},
                    )
                )
                return
            if chunk.response_type is ResponseType.THINKING:
                if chunk.content:
                    thinking_open = True
                    await emit_thought(chunk.content, False)
                elif chunk.done and thinking_open:
                    await close_thinking()
                return
            if chunk.content:
                think_part, answer_part = splitter.feed(chunk.content)
                if think_part:
                    thinking_open = True
                    await emit_thought(think_part, False)
                await emit_answer(answer_part)
            if chunk.done:
                think_part, answer_part = splitter.flush()
                if think_part:
                    thinking_open = True
                    await emit_thought(think_part, False)
                await emit_answer(answer_part)
                await close_thinking()

        llm_result = await self._stream_llm_to_events(
            ctx, messages, opts, session_id, emit_func=emit_chunk
        )
        content = strip_think_blocks(llm_result.content)
        finish_reason = llm_result.finish_reason or "stop"
        response = ChatResponse(
            content=content,
            reasoning_content=llm_result.reasoning_content,
            tool_calls=list(llm_result.tool_calls),
            finish_reason=finish_reason,
        )
        if answer_streamed:
            response.answer_streamed = True
            response.answer_event_id = answer_id
        if llm_result.usage is not None:
            response.usage = llm_result.usage
        return response

    async def _stream_llm_to_events(
        self,
        ctx: Context,
        messages: list[Message],
        opts: ChatOptions,
        session_id: str,
        emit_func: Callable[[StreamResponse, str], Awaitable[None]] | None = None,
    ) -> _StreamResult:
        llm_messages = self._model_context.encode_messages(messages)
        stream = self._chat_model.chat_stream(llm_messages, opts)
        result = _StreamResult()
        answer_decoder = self._model_context.stream_decoder()
        thinking_decoder = self._model_context.stream_decoder()

        async for chunk in stream:
            if chunk.response_type is ResponseType.ERROR:
                result.stream_error = chunk.content
                continue
            if chunk.response_type is ResponseType.THINKING:
                content = thinking_decoder.feed(chunk.content)
                if chunk.done:
                    content += thinking_decoder.flush()
            else:
                content = answer_decoder.feed(chunk.content)
                if chunk.done:
                    content += answer_decoder.flush()
            self._model_context.decode_tool_calls(chunk.tool_calls)

            if content:
                is_extracted = chunk.data is not None and chunk.data.get("source") is not None
                if not is_extracted:
                    if chunk.response_type is ResponseType.THINKING:
                        result.reasoning_content += content
                    else:
                        result.content += content
            if chunk.tool_calls:
                result.tool_calls = list(chunk.tool_calls)
            if chunk.usage is not None:
                result.usage = chunk.usage
            if chunk.finish_reason:
                result.finish_reason = chunk.finish_reason
            if emit_func is not None:
                await emit_func(chunk, result.content)

        answer_tail = answer_decoder.flush()
        thinking_tail = thinking_decoder.flush()
        result.content += answer_tail
        result.reasoning_content += thinking_tail
        if emit_func is not None:
            if thinking_tail:
                await emit_func(
                    StreamResponse(response_type=ResponseType.THINKING, content=thinking_tail),
                    result.content,
                )
            if answer_tail:
                await emit_func(
                    StreamResponse(response_type=ResponseType.ANSWER, content=answer_tail),
                    result.content,
                )
        self._model_context.decode_tool_calls(result.tool_calls)
        if result.stream_error and not result.content and not result.tool_calls:
            raise AgentLLMError(f"LLM stream error: {result.stream_error}")
        return result

    # ── analyze ────────────────────────────────────────────────────

    async def _analyze_response(
        self,
        ctx: Context,
        response: ChatResponse,
        state: _RunState,
        session_id: str,
    ) -> ResponseVerdict:
        step = self._build_step(state, response)

        # Content blocked by the model's safety filter is terminal.
        if response.finish_reason == "content_filter" and not response.tool_calls:
            answer = response.content or CONTENT_FILTER_FALLBACK
            answer_id = generate_event_id("answer")
            await self._emit(
                Event(
                    id=answer_id,
                    type=EventType.AGENT_FINAL_ANSWER,
                    session_id=session_id,
                    data={"content": answer, "done": False},
                )
            )
            await self._emit(
                Event(
                    id=answer_id,
                    type=EventType.AGENT_FINAL_ANSWER,
                    session_id=session_id,
                    data={"content": "", "done": True},
                )
            )
            return ResponseVerdict(is_done=True, final_answer=answer, step=step)

        # Natural stop with no tool calls ends the turn.
        if is_natural_stop_finish_reason(response.finish_reason) and not response.tool_calls:
            content = strip_think_blocks(response.content)
            if response.answer_streamed and response.answer_event_id:
                answer_id = response.answer_event_id
            else:
                answer_id = generate_event_id("answer")
                if content:
                    await self._emit(
                        Event(
                            id=answer_id,
                            type=EventType.AGENT_FINAL_ANSWER,
                            session_id=session_id,
                            data={"content": content, "done": False},
                        )
                    )
            await self._emit(
                Event(
                    id=answer_id,
                    type=EventType.AGENT_FINAL_ANSWER,
                    session_id=session_id,
                    data={"content": "", "done": True},
                )
            )
            return ResponseVerdict(
                is_done=True,
                final_answer=content,
                empty_content=content == "",
                step=step,
            )

        # A round still requesting tools is non-terminal.
        return ResponseVerdict(is_done=False, step=step)

    # ── act ────────────────────────────────────────────────────────

    async def _execute_tool_calls(
        self,
        ctx: Context,
        response: ChatResponse,
        iteration: int,
        session_id: str,
        message_id: str,
    ) -> list[ToolCall]:
        calls = list(response.tool_calls)
        if not calls:
            return []
        if self._config.parallel_tool_calls and len(calls) >= 2:
            tool_calls = list(
                await asyncio.gather(
                    *(
                        self._run_tool_call(ctx, call, index, iteration, session_id, message_id)
                        for index, call in enumerate(calls)
                    )
                )
            )
        else:
            tool_calls = [
                await self._run_tool_call(ctx, call, index, iteration, session_id, message_id)
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

    async def _run_tool_call(
        self,
        ctx: Context,
        tc: LLMToolCall,
        index: int,
        iteration: int,
        session_id: str,
        message_id: str,
    ) -> ToolCall:
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

    # ── observe ────────────────────────────────────────────────────

    def _append_tool_results(self, messages: list[Message], step: AgentStep) -> None:
        """Append the assistant step and rendered tool results onto messages."""
        if step.thought or step.tool_calls or step.reasoning_content:
            assistant_msg = Message(
                role="assistant",
                content=step.thought,
                reasoning_content=step.reasoning_content,
                tool_calls=[
                    MessageToolCall(
                        id=tc.id,
                        type="function",
                        function=FunctionCall(name=tc.name, arguments=_dump_args(tc.args)),
                        provider_metadata=tc.provider_metadata,
                    )
                    for tc in step.tool_calls
                ],
            )
            messages.append(assistant_msg)
        for tc in step.tool_calls:
            result = (
                tc.result if tc.result is not None else ToolResult(success=False, error="no result")
            )
            rendered = self._model_context.model_tool_result_for_tool(
                tc.name, _to_model_output_result(result)
            )
            messages.append(
                Message(role="tool", content=rendered, tool_call_id=tc.id, name=tc.name)
            )

    # ── finalize ───────────────────────────────────────────────────

    async def _synthesize_final_answer(
        self,
        ctx: Context,
        query: str,
        state: _RunState,
        session_id: str,
    ) -> None:
        system_prompt = self._build_system_prompt()
        user_turn = self._render_user_turn(session_id, query)
        messages: list[Message] = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_turn),
        ]
        has_retrieved_image = False
        for step in state.round_steps:
            for tool_call in step.tool_calls:
                result = (
                    tool_call.result
                    if tool_call.result is not None
                    else ToolResult(success=False, error="no result")
                )
                if _MARKDOWN_IMAGE_RE.search(result.output):
                    has_retrieved_image = True
                rendered = self._model_context.model_tool_result_for_tool(
                    tool_call.name, _to_model_output_result(result)
                )
                messages.append(
                    Message(role="user", content=f"Tool {tool_call.name} returned: {rendered}")
                )
        image_requirement = _final_answer_image_requirement(has_retrieved_image)
        final_prompt = (
            "Based on the above tool call results, generate a complete answer "
            "for the user's question.\n\n"
            f"User question: {query}\n\n"
            "Requirements:\n"
            "1. Answer based on the actually retrieved content\n"
            "2. Organize the answer in a structured format\n"
            "3. If information is insufficient, honestly state so\n"
            "4. IMPORTANT: Respond in the same language as the user's question\n"
            f"{image_requirement}\n\n"
            "Now generate the final answer:"
        )
        messages.append(Message(role="user", content=final_prompt))

        answer_id = generate_event_id("answer")
        answer_done_emitted = False

        async def emit_final(chunk: StreamResponse, full_content: str) -> None:
            nonlocal answer_done_emitted
            if chunk.response_type is ResponseType.THINKING:
                return
            if chunk.content:
                await self._emit(
                    Event(
                        id=answer_id,
                        type=EventType.AGENT_FINAL_ANSWER,
                        session_id=session_id,
                        data={"content": chunk.content, "done": chunk.done},
                    )
                )
                if chunk.done:
                    answer_done_emitted = True

        llm_result = await self._stream_llm_to_events(
            ctx,
            messages,
            ChatOptions(temperature=self._config.temperature),
            session_id,
            emit_func=emit_final,
        )
        if not answer_done_emitted:
            await self._emit(
                Event(
                    id=answer_id,
                    type=EventType.AGENT_FINAL_ANSWER,
                    session_id=session_id,
                    data={"content": "", "done": True},
                )
            )
        state.final_answer = strip_think_blocks(llm_result.content)

    async def _handle_max_iterations(
        self, ctx: Context, query: str, state: _RunState, session_id: str
    ) -> None:
        """Synthesize a final answer when the iteration ceiling was exhausted."""
        try:
            await self._synthesize_final_answer(ctx, query, state, session_id)
        except AgentExecutionError:
            state.final_answer = MAX_ITERATIONS_FALLBACK
        state.is_complete = True

    async def _best_effort_synthesize(
        self, ctx: Context, query: str, state: _RunState, session_id: str
    ) -> None:
        """Attempt a salvage final answer on a cancelled run; never fails."""
        try:
            await self._synthesize_final_answer(ctx, query, state, session_id)
        except AgentExecutionError:
            return

    async def _emit_completion_event(
        self, state: _RunState, session_id: str, message_id: str, start_time: float
    ) -> None:
        """Emit the exactly-once completion event with the run summary."""
        await self._emit(
            Event(
                id=generate_event_id("complete"),
                type=EventType.AGENT_COMPLETE,
                session_id=session_id,
                data={
                    "session_id": session_id,
                    "total_steps": len(state.round_steps),
                    "final_answer": state.final_answer,
                    "knowledge_refs": [ref.model_dump(mode="json") for ref in state.knowledge_refs],
                    "agent_steps": [step.model_dump(mode="json") for step in state.round_steps],
                    "total_duration_ms": int((time.monotonic() - start_time) * 1000),
                    "message_id": message_id,
                },
            )
        )

    # ── assembly helpers ───────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        """Compose the system prompt, appending the model-context protocol."""
        if self._system_prompt_template:
            template = self._system_prompt_template
        elif self._config.use_custom_system_prompt and self._config.system_prompt:
            template = self._config.system_prompt
        else:
            template = DEFAULT_SYSTEM_PROMPT
        rendered = _render_prompt_placeholders(
            template,
            self._knowledge_bases,
            self._config.web_search_enabled,
            _rfc3339_now(),
        )
        return rendered.rstrip(" \t\r\n") + self._model_context.protocol_prompt()

    def _build_messages_with_llm_context(
        self,
        system_prompt: str,
        query: str,
        session_id: str,
        llm_context: Sequence[Message],
        image_urls: Sequence[str],
    ) -> list[Message]:
        messages: list[Message] = [Message(role="system", content=system_prompt)]
        if llm_context:
            if self._config.retain_retrieval_history:
                sanitized = list(llm_context)
            else:
                sanitized = _redact_history_kb_results(llm_context)
            for msg in sanitized:
                if msg.role in {"user", "assistant", "tool"}:
                    messages.append(msg)
        messages.append(
            Message(
                role="user",
                content=self._render_user_turn(session_id, query),
                images=list(image_urls),
            )
        )
        return messages

    def _render_user_turn(self, session_id: str, query: str) -> str:
        """Build the current user turn (runtime context + query)."""
        for kb in self._knowledge_bases:
            self._model_context.register_knowledge_base(kb.id)
        for doc in self._selected_docs:
            self._model_context.register_document(doc.knowledge_id)
            if doc.knowledge_base_id:
                self._model_context.register_knowledge_base(doc.knowledge_base_id)
        runtime_context = self._model_context.compact_known_text(
            self._build_runtime_context_block(session_id)
        )
        return _compose_user_turn(runtime_context, query)

    def _build_runtime_context_block(self, session_id: str) -> str:
        """Render the per-turn scope block injected into the user message."""
        lines = ['<runtime_context scope="this_turn">']
        lines.append(f"  <current_time>{_rfc3339_now()}</current_time>")
        lines.append(f"  <session>{_escape_xml_attr(session_id)}</session>")
        if self._knowledge_bases:
            lines.append("  <bound_knowledge_bases>")
            for line in _indent_lines(_format_knowledge_base_list(self._knowledge_bases), "    "):
                lines.append(line)
            lines.append("  </bound_knowledge_bases>")
        if self._selected_docs:
            lines.append('  <pinned_documents scope="authoritative_for_this_turn">')
            for doc in self._selected_docs:
                title = doc.title or doc.file_name or doc.knowledge_id
                lines.append(
                    f'    <document knowledge_id="{_escape_xml_attr(doc.knowledge_id)}" '
                    f'title="{_escape_xml_attr(title)}" '
                    f'file_type="{_escape_xml_attr(doc.file_type)}" />'
                )
            lines.append("  </pinned_documents>")
        lines.append(
            "  <communication_instruction>Do not use internal tool names or "
            'identifiers in your answers or in Thought. Say "keyword retrieval" '
            'instead of grep_chunks, "semantic retrieval" instead of '
            'knowledge_search, "browse full document" instead of '
            "list_knowledge_chunks; likewise never expose chunk_id, "
            "knowledge_id, or other internal IDs — refer to documents by title "
            "or name.</communication_instruction>"
        )
        lines.append(
            "  <answer_instruction>When you have gathered enough information, "
            "write your complete user-facing answer as your reply and stop — do "
            "not request any more tools in that final message. Until then, keep "
            "using tools; do not give a partial answer "
            "mid-investigation.</answer_instruction>"
        )
        lines.append("</runtime_context>")
        return "\n".join(lines)

    def _build_tools_for_llm(self) -> list[Tool]:
        """Build the function-calling tool list from the registry."""
        tools: list[Tool] = []
        for definition in self._tool_registry.get_function_definitions():
            tools.append(
                Tool(
                    type="function",
                    function=FunctionDef(
                        name=definition.name,
                        description=definition.description,
                        parameters=_parse_tool_arguments(definition.parameters),
                    ),
                )
            )
        return tools

    async def _emit(self, event: Event) -> None:
        """Publish one event to the engine's event bus."""
        await self._event_bus.emit(event)


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "AgentEngine",
]
