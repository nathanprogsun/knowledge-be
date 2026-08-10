"""Think phase: streaming LLM reasoning with event emission and retry.

The think phase performs the reasoning half of one ReAct iteration. It
assembles the model-facing prompt payload (system prompt, per-turn runtime
context, sanitised history), streams the model response while emitting
thought / answer / tool-call events, and wraps the call in transient-error
retry with graceful degradation to the finalize phase when the model fails
after prior tool results already exist.

The phase consumes the merged engine types (``AgentState`` / ``AgentConfig``)
and the chat-layer wire types (``Message`` / ``ChatResponse`` /
``StreamResponse``). It stays free of storage and web layers; the LLM is
reached only through the injected ``Chat`` protocol and the model-context
``Registry``.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TypeAlias

from src.ai.embedding.base import Context
from src.ai.llm.types import (
    Chat,
    ChatOptions,
    ChatResponse,
    LLMToolCall,
    Message,
    ResponseType,
    StreamResponse,
    TokenUsage,
    Tool,
)
from src.core.agents.engine.modelcontext import Registry
from src.core.agents.engine.types import (
    MAX_LLM_RETRIES,
    AgentConfig,
    AgentExecutionError,
    AgentLLMError,
    AgentState,
    KnowledgeBaseInfo,
    SelectedDocumentInfo,
    generate_event_id,
    is_transient_error,
)
from src.core.chat.bus import Event, EventBus
from src.core.chat.types import EventType

logger = logging.getLogger(__name__)

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

#: Finalize seam injected for graceful degradation: given the query and the
#: current run state, produce an updated state carrying a synthesized answer.
SynthesizeFn: TypeAlias = Callable[[Context, str, AgentState, str], Awaitable[AgentState]]


def strip_think_blocks(content: str) -> str:
    """Remove ``<think>…</think>`` blocks and trim leftover whitespace."""
    if not content:
        return ""
    return _THINK_BLOCK_RE.sub("", content).strip(" \t\r\n")


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


def _has_matching_tool_call(messages: Sequence[Message], tool_call_id: str) -> bool:
    """Return whether a preceding assistant message references the id."""
    for msg in reversed(messages):
        if msg.role != "assistant":
            continue
        for call in msg.tool_calls:
            if call.id == tool_call_id:
                return True
    return False


def sanitize_messages(messages: Sequence[Message]) -> list[Message]:
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


# ── Prompt assembly ───────────────────────────────────────────────────


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


def _build_runtime_context_block(
    session_id: str,
    knowledge_bases: Sequence[KnowledgeBaseInfo],
    selected_docs: Sequence[SelectedDocumentInfo],
) -> str:
    """Render the per-turn scope block injected into the user message."""
    lines = ['<runtime_context scope="this_turn">']
    lines.append(f"  <current_time>{_rfc3339_now()}</current_time>")
    lines.append(f"  <session>{_escape_xml_attr(session_id)}</session>")
    if knowledge_bases:
        lines.append("  <bound_knowledge_bases>")
        for line in _indent_lines(_format_knowledge_base_list(knowledge_bases), "    "):
            lines.append(line)
        lines.append("  </bound_knowledge_bases>")
    if selected_docs:
        lines.append('  <pinned_documents scope="authoritative_for_this_turn">')
        for doc in selected_docs:
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


# ── Stream / carrier ──────────────────────────────────────────────────


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


@dataclass
class _StreamResult:
    """Accumulated output of one streaming LLM call."""

    content: str = ""
    reasoning_content: str = ""
    tool_calls: list[LLMToolCall] = field(default_factory=list)
    usage: TokenUsage | None = None
    finish_reason: str = ""
    stream_error: str = ""


@dataclass(frozen=True, slots=True)
class ThinkRound:
    """Outcome of one think-phase call.

    ``response`` is ``None`` when graceful degradation succeeded (the injected
    finalize seam produced a synthesized answer and the loop should stop).
    ``state`` carries the possibly updated run state; on graceful degradation
    ``state.is_complete`` is ``True``.
    """

    response: ChatResponse | None
    state: AgentState


class ThinkPhase:
    """Streaming think phase: prompt assembly, event emission, and retry."""

    def __init__(
        self,
        config: AgentConfig,
        chat_model: Chat,
        event_bus: EventBus,
        model_context: Registry,
        *,
        system_prompt_template: str = "",
        knowledge_bases: Sequence[KnowledgeBaseInfo] = (),
        selected_docs: Sequence[SelectedDocumentInfo] = (),
    ) -> None:
        self._config = config
        self._chat_model = chat_model
        self._event_bus = event_bus
        self._model_context = model_context
        self._system_prompt_template = system_prompt_template
        self._knowledge_bases = list(knowledge_bases)
        self._selected_docs = list(selected_docs)

    # ── prompt assembly ─────────────────────────────────────────────

    def build_system_prompt(self) -> str:
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

    def render_user_turn(self, session_id: str, query: str) -> str:
        """Build the current user turn (runtime context + query)."""
        for kb in self._knowledge_bases:
            self._model_context.register_knowledge_base(kb.id)
        for doc in self._selected_docs:
            self._model_context.register_document(doc.knowledge_id)
            if doc.knowledge_base_id:
                self._model_context.register_knowledge_base(doc.knowledge_base_id)
        runtime_context = self._model_context.compact_known_text(
            _build_runtime_context_block(session_id, self._knowledge_bases, self._selected_docs)
        )
        return _compose_user_turn(runtime_context, query)

    def build_messages_with_llm_context(
        self,
        system_prompt: str,
        query: str,
        session_id: str,
        llm_context: Sequence[Message],
        image_urls: Sequence[str],
    ) -> list[Message]:
        """Compose the full message array for the first think call."""
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
                content=self.render_user_turn(session_id, query),
                images=list(image_urls),
            )
        )
        return messages

    # ── streaming ───────────────────────────────────────────────────

    async def stream_llm_to_events(
        self,
        ctx: Context,
        messages: list[Message],
        opts: ChatOptions,
        session_id: str,
        emit_func: Callable[[StreamResponse, str], Awaitable[None]] | None = None,
    ) -> _StreamResult:
        """Stream one LLM call, decoding model handles and routing chunks."""
        llm_messages = self._model_context.encode_messages(list(messages))
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

    async def stream_thinking_to_events(
        self,
        ctx: Context,
        messages: list[Message],
        tools: list[Tool],
        iteration: int,
        session_id: str,
    ) -> ChatResponse:
        """Stream the thinking round, emitting thought / answer / tool events."""
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

        llm_result = await self.stream_llm_to_events(
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

    # ── retry / graceful degradation ───────────────────────────────

    async def call_with_retry(
        self,
        ctx: Context,
        messages: list[Message],
        tools: list[Tool],
        state: AgentState,
        query: str,
        iteration: int,
        session_id: str,
        *,
        synthesize: SynthesizeFn | None = None,
    ) -> ThinkRound:
        """Call the LLM with transient-error retry and graceful degradation.

        Returns ``ThinkRound(response=None, ...)`` with ``state.is_complete``
        set when the injected ``synthesize`` seam succeeds after a failure and
        the loop should stop. Raises ``AgentLLMError`` on irrecoverable
        failures (or when degradation is unavailable).
        """
        sanitized = sanitize_messages(messages)
        response: ChatResponse | None = None
        err: AgentLLMError | None = None
        try:
            response = await self.stream_thinking_to_events(
                ctx, sanitized, tools, iteration, session_id
            )
        except AgentLLMError as exc:
            err = exc
        if err is not None and is_transient_error(err):
            for retry in range(1, MAX_LLM_RETRIES + 1):
                await asyncio.sleep(retry)
                try:
                    response = await self.stream_thinking_to_events(
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
                if synthesize is None:
                    raise AgentLLMError(f"LLM call failed: {err}")
                try:
                    state = await synthesize(ctx, query, state, session_id)
                except AgentExecutionError as synth_err:
                    raise AgentLLMError(
                        f"LLM call failed: {err} (synthesis also failed: {synth_err})"
                    ) from synth_err
                return ThinkRound(
                    response=None, state=state.model_copy(update={"is_complete": True})
                )
            raise AgentLLMError(f"LLM call failed: {err}") from err
        if response is None:
            raise AgentLLMError("LLM call failed")
        return ThinkRound(response=response, state=state)

    async def _emit(self, event: Event) -> None:
        """Publish one event to the phase's event bus."""
        await self._event_bus.emit(event)


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "SynthesizeFn",
    "ThinkPhase",
    "ThinkRound",
    "sanitize_messages",
    "strip_think_blocks",
]
