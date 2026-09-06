"""Request-scoped chat execution service for the chat view layer.

``ChatService`` is the single facade the chat endpoints talk to. It owns
the request-level orchestration that the upstream handler performs inline:

- resolve the custom agent (if any) for the request;
- persist the user / assistant message shells via an injectable gateway
  (message persistence lives behind the gateway seam);
- open the per-request event bus and translate the pipeline / agent
  domain events onto the wire ``response_type`` vocabulary;
- run the selected QA executor (knowledge pipeline or agent loop) and
  surface every event as an async stream the view can forward to SSE.

The heavy execution (LLM chat models, retrieval engines, message rows) is
deliberately behind injectable seams — ``AgentResolver``,
``KnowledgeSearcher``, ``QARunner`` and ``MessageGateway`` — so the HTTP
layer stays testable without a live model / store, and a later change can
wire real implementations into ``src.core.chat.factory`` without touching
the endpoints. The pipeline is assembled per request by the runner seam,
mirroring the upstream composition-root style.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Protocol, TypeVar, runtime_checkable

from src.common.exception import ValidationError
from src.core.agents.types import (
    AGENT_MODE_SMART_REASONING,
    CustomAgentInfo,
)
from src.core.chat.bus import Event, EventBus
from src.core.chat.pipeline.types import Context, SearchResult
from src.core.chat.stream.manager import StreamManager
from src.core.chat.types import EventType

_T = TypeVar("_T")

logger = logging.getLogger(__name__)

#: SSE ``response_type`` for each chat-domain event forwarded to the wire.
#: Only these event types are visible to the client; the remaining domain
#: events (query lifecycle, retrieval progress, ...) stay internal.
#: The view layer reuses this mapping when projecting events onto the SSE
#: ``StreamResponse`` shape, so the two layers can never drift apart.
WIRE_RESPONSE_TYPE: dict[EventType, str] = {
    EventType.AGENT_QUERY: "agent_query",
    EventType.AGENT_THOUGHT: "thinking",
    EventType.AGENT_TOOL_CALL: "tool_call",
    EventType.AGENT_TOOL_RESULT: "tool_result",
    EventType.AGENT_REFERENCES: "references",
    EventType.AGENT_FINAL_ANSWER: "answer",
    EventType.AGENT_COMPLETE: "complete",
    EventType.AGENT_REFLECTION: "reflection",
    EventType.ERROR: "error",
    EventType.SESSION_TITLE: "session_title",
    EventType.STOP: "stop",
}

#: Event types the stream bridge subscribes to on the per-request bus.
_STREAM_EVENT_TYPES: tuple[EventType, ...] = tuple(WIRE_RESPONSE_TYPE)


def _require_session_id(session_id: str) -> None:
    """Reject an empty session id."""
    if not session_id or not session_id.strip():
        raise ValidationError(
            code="chat.session_required",
            message="Session ID is empty",
        )


def _require_query(query: str) -> None:
    """Reject an empty query."""
    if not query or not query.strip():
        raise ValidationError(
            code="chat.query_required",
            message="Query content cannot be empty",
        )


def answer_delta(event: Event) -> str:
    """Return the streamed answer fragment on a final-answer event."""
    if event.type != EventType.AGENT_FINAL_ANSWER:
        return ""
    data = event.data
    if not isinstance(data, dict):
        return ""
    content = data.get("content")
    return content if isinstance(content, str) else ""


# ── Request shapes (structural — satisfied by the web wire models) ────


class MentionedItemLike(Protocol):
    """One ``@mentioned`` item carried by a chat request."""

    id: str
    type: str
    kb_id: str | None


class KnowledgeQARequestLike(Protocol):
    """Knowledge-QA body surface used by the service."""

    query: str
    knowledge_base_ids: Sequence[str] | None
    knowledge_ids: Sequence[str] | None
    agent_id: str | None
    summary_model_id: str | None
    mcp_service_ids: Sequence[str] | None
    skill_names: Sequence[str] | None
    tag_ids: Sequence[str] | None
    mentioned_items: Sequence[MentionedItemLike] | None
    disable_title: bool
    channel: str | None
    attachment_ids: Sequence[str] | None


class AgentQARequestLike(KnowledgeQARequestLike, Protocol):
    """Agent-QA body surface — adds the agent-mode flags."""

    agent_enabled: bool
    web_search_enabled: bool


# ── Injectable seams ──────────────────────────────────────────────────


@runtime_checkable
class AgentResolver(Protocol):
    """Resolves a custom agent for the caller's workspace."""

    async def resolve(
        self,
        *,
        tenant_id: int,
        agent_id: str,
    ) -> CustomAgentInfo | None: ...


@runtime_checkable
class KnowledgeSearcher(Protocol):
    """Runs a retrieval-only knowledge search (no LLM summarization)."""

    async def search(
        self,
        *,
        tenant_id: int,
        query: str,
        knowledge_base_ids: list[str],
        knowledge_ids: list[str],
        tag_scopes: list[TagScope],
    ) -> list[SearchResult]: ...


@runtime_checkable
class QARunner(Protocol):
    """Executes one QA turn, emitting domain events onto ``event_bus``."""

    async def run(
        self,
        *,
        ctx: Context,
        session_id: str,
        request: KnowledgeQARequestLike,
        agent: CustomAgentInfo | None,
        event_bus: EventBus,
    ) -> None: ...


@runtime_checkable
class MessageGateway(Protocol):
    """Persists the message shells of a QA turn (deferred seam)."""

    async def create_user_message(self, *, session_id: str, query: str) -> str: ...

    async def create_assistant_message(
        self,
        *,
        session_id: str,
        request_id: str,
        agent: CustomAgentInfo | None,
        model_id: str,
    ) -> AssistantMessage: ...

    async def complete_assistant_message(
        self,
        *,
        assistant_message_id: str,
        content: str,
        is_fallback: bool = False,
    ) -> None: ...


# ── Value shapes ──────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class TagScope:
    """Tags scoped to one knowledge base (upstream ``TagScope``)."""

    knowledge_base_id: str
    tag_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AssistantMessage:
    """A freshly created assistant message shell."""

    id: str
    session_id: str


@dataclass(frozen=True, slots=True)
class RequestContext:
    """Opaque execution context threaded through the QA runner.

    Satisfies the pipeline ``Context`` protocol structurally; the runner
    passes it to the pipeline steps as-is.
    """

    tenant_id: int
    user_id: str
    request_id: str


# ── Request normalisation helpers ─────────────────────────────────────


def merge_knowledge_targets(
    *,
    knowledge_base_ids: Sequence[str],
    knowledge_ids: Sequence[str],
    mentioned_items: Sequence[MentionedItemLike] | None,
) -> tuple[list[str], list[str]]:
    """Merge ``@mention`` KB/file ids into the request target lists.

    Pure function — the input lists are never mutated. Mentioned items of
    type ``kb`` join the knowledge-base list and type ``file`` the
    knowledge list, preserving order and dropping duplicates.
    """
    kb_seen: set[str] = set()
    merged_kb: list[str] = []
    for kb_id in knowledge_base_ids:
        if kb_id and kb_id not in kb_seen:
            kb_seen.add(kb_id)
            merged_kb.append(kb_id)

    knowledge_seen: set[str] = set()
    merged_knowledge: list[str] = []
    for knowledge_id in knowledge_ids:
        if knowledge_id and knowledge_id not in knowledge_seen:
            knowledge_seen.add(knowledge_id)
            merged_knowledge.append(knowledge_id)

    for item in mentioned_items or ():
        if not item.id:
            continue
        if item.type == "kb" and item.id not in kb_seen:
            kb_seen.add(item.id)
            merged_kb.append(item.id)
        elif item.type == "file" and item.id not in knowledge_seen:
            knowledge_seen.add(item.id)
            merged_knowledge.append(item.id)

    return merged_kb, merged_knowledge


def build_tag_scopes(
    *,
    tag_ids: Sequence[str],
    mentioned_items: Sequence[MentionedItemLike] | None,
    knowledge_base_ids: Sequence[str],
) -> list[TagScope]:
    """Build scoped tag filters from request tag ids and mentions.

    Tag mentions (``type == "tag"``) carry their owning knowledge base in
    ``kb_id`` and become a scope of their own. Unscoped request ``tag_ids``
    are applied to every knowledge base in scope. An unscoped tag id with
    no knowledge base to attach to is rejected — it cannot be matched.
    """
    scopes: list[TagScope] = []
    seen: set[tuple[str, str]] = set()
    mentioned_tag_ids = {
        item.id for item in mentioned_items or () if item.type == "tag" and item.id
    }

    for item in mentioned_items or ():
        if item.type != "tag" or not item.id or not item.kb_id:
            continue
        key = (item.kb_id, item.id)
        if key in seen:
            continue
        seen.add(key)
        scopes.append(TagScope(knowledge_base_id=item.kb_id, tag_ids=(item.id,)))

    orphan = [tag_id for tag_id in tag_ids if tag_id and tag_id not in mentioned_tag_ids]
    if orphan and not knowledge_base_ids:
        raise ValidationError(
            code="chat.tag_scope_required",
            message="tag_ids require a knowledge_base_ids or a scoped tag mention",
        )
    for kb_id in knowledge_base_ids:
        for tag_id in orphan:
            key = (kb_id, tag_id)
            if key in seen:
                continue
            seen.add(key)
            scopes.append(TagScope(knowledge_base_id=kb_id, tag_ids=(tag_id,)))

    return scopes


def resolve_agent_mode(
    *,
    agent_enabled: bool,
    agent: CustomAgentInfo | None,
) -> bool:
    """Return whether this turn runs in agent (ReAct) mode.

    A resolved custom agent's ``agent_mode`` wins over the request flag,
    mirroring the upstream resolution rule.
    """
    if agent is not None:
        return agent.config.get("agent_mode") == AGENT_MODE_SMART_REASONING
    return agent_enabled


# ── The service ───────────────────────────────────────────────────────


def _bind_qa_queue() -> tuple[asyncio.Queue[Event | None], EventBus]:
    queue: asyncio.Queue[Event | None] = asyncio.Queue()
    event_bus = EventBus()

    async def _sink(event: Event) -> None:
        await queue.put(event)

    for event_type in _STREAM_EVENT_TYPES:
        event_bus.on(event_type, _sink)
    return queue, event_bus


async def _run_qa_runner(
    *,
    runner: QARunner,
    ctx: RequestContext,
    session_id: str,
    request: KnowledgeQARequestLike,
    agent: CustomAgentInfo | None,
    event_bus: EventBus,
    queue: asyncio.Queue[Event | None],
    request_id: str,
) -> None:
    try:
        await runner.run(
            ctx=ctx,
            session_id=session_id,
            request=request,
            agent=agent,
            event_bus=event_bus,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception("QA execution failed for session %s", session_id)
        await queue.put(
            Event(
                type=EventType.ERROR,
                session_id=session_id,
                request_id=request_id,
                data={
                    "error": str(exc),
                    "error_code": getattr(exc, "code", None),
                    "stage": "qa_execution",
                    "session_id": session_id,
                },
            )
        )
    finally:
        await queue.put(None)


async def _cancel_task(task: asyncio.Task[_T] | None) -> None:
    if task is None or task.done():
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def _next_qa_event(
    queue: asyncio.Queue[Event | None],
    cancel_task: asyncio.Task[None] | None,
) -> Event | None:
    if cancel_task is None:
        return await queue.get()
    get_task: asyncio.Task[Event | None] = asyncio.create_task(queue.get())
    done, _pending = await asyncio.wait(
        {get_task, cancel_task},
        return_when=asyncio.FIRST_COMPLETED,
    )
    if cancel_task in done:
        await _cancel_task(get_task)
        return None
    return get_task.result()


async def _iter_qa_events(
    queue: asyncio.Queue[Event | None],
    *,
    session_id: str,
    message_id: str,
    stream_manager: StreamManager | None,
) -> AsyncIterator[Event]:
    # Stop and the model loop share this manager. The first of drain-end
    # or cancel wins so POST /stop can interrupt tokens.
    cancel_task: asyncio.Task[None] | None = None
    if stream_manager is not None:
        cancel_task = asyncio.create_task(stream_manager.wait_cancelled(session_id, message_id))
    try:
        while True:
            event = await _next_qa_event(queue, cancel_task)
            if event is None:
                return
            yield event
    finally:
        await _cancel_task(cancel_task)


async def _open_qa_turn(
    gateway: MessageGateway,
    *,
    session_id: str,
    query: str,
    request_id: str,
    agent: CustomAgentInfo | None,
    model_id: str,
) -> tuple[str, AssistantMessage, asyncio.Queue[Event | None], EventBus]:
    user_message_id = await gateway.create_user_message(
        session_id=session_id,
        query=query,
    )
    assistant = await gateway.create_assistant_message(
        session_id=session_id,
        request_id=request_id,
        agent=agent,
        model_id=model_id,
    )
    queue, event_bus = _bind_qa_queue()
    return user_message_id, assistant, queue, event_bus


def _agent_query_event(
    session_id: str,
    request_id: str,
    user_message_id: str,
    assistant_id: str,
) -> Event:
    return Event(
        type=EventType.AGENT_QUERY,
        session_id=session_id,
        request_id=request_id,
        data={
            "session_id": session_id,
            "assistant_message_id": assistant_id,
            "user_message_id": user_message_id,
        },
    )


class ChatService:
    """Request-scoped chat facade (one instance per request).

    Immutable after construction: every method returns new values or
    raises; nothing on the service is mutated between calls.
    """

    def __init__(
        self,
        *,
        tenant_id: int,
        user_id: str,
        request_id: str,
        agent_resolver: AgentResolver,
        searcher: KnowledgeSearcher,
        knowledge_runner: QARunner,
        agent_runner: QARunner,
        message_gateway: MessageGateway,
        stream_manager: StreamManager | None = None,
    ) -> None:
        self._tenant_id = tenant_id
        self._user_id = user_id
        self._request_id = request_id
        self._agent_resolver = agent_resolver
        self._searcher = searcher
        self._knowledge_runner = knowledge_runner
        self._agent_runner = agent_runner
        self._message_gateway = message_gateway
        self._stream_manager = stream_manager

    @property
    def tenant_id(self) -> int:
        """The caller's active workspace id."""
        return self._tenant_id

    @property
    def user_id(self) -> str:
        """The caller's user id."""
        return self._user_id

    @property
    def request_id(self) -> str:
        """The request id stamped on streamed wire events."""
        return self._request_id

    # ── Agent resolution ───────────────────────────────────────────

    async def resolve_agent(self, agent_id: str | None) -> CustomAgentInfo | None:
        """Resolve ``agent_id`` in the caller's workspace, or ``None``.

        A missing id is a no-op; an id that cannot be resolved in the
        caller's workspace is treated as absent (the caller may still
        request agent mode and then receive a validation error).
        """
        if not agent_id:
            return None
        return await self._agent_resolver.resolve(
            tenant_id=self._tenant_id,
            agent_id=agent_id,
        )

    # ── Knowledge search (no LLM) ──────────────────────────────────

    async def search_knowledge(
        self,
        *,
        query: str,
        knowledge_base_id: str | None = None,
        knowledge_base_ids: list[str] | None = None,
        knowledge_ids: list[str] | None = None,
        tag_ids: list[str] | None = None,
        mentioned_items: Sequence[MentionedItemLike] | None = None,
    ) -> list[SearchResult]:
        """Run a retrieval-only search and return the hits.

        Validates the request per the upstream contract: a non-empty query
        and at least one knowledge scope (base ids, knowledge ids, or
        scoped tags). The single ``knowledge_base_id`` field is merged
        into ``knowledge_base_ids`` for backward compatibility.
        """
        if not query or not query.strip():
            raise ValidationError(
                code="chat.query_required",
                message="Query content cannot be empty",
            )

        merged_base = [kb_id for kb_id in [knowledge_base_id, *(knowledge_base_ids or [])] if kb_id]
        merged_kb, merged_knowledge = merge_knowledge_targets(
            knowledge_base_ids=merged_base,
            knowledge_ids=knowledge_ids or [],
            mentioned_items=mentioned_items,
        )
        tag_scopes = build_tag_scopes(
            tag_ids=tag_ids or [],
            mentioned_items=mentioned_items,
            knowledge_base_ids=merged_kb,
        )
        if not merged_kb and not merged_knowledge and not tag_scopes:
            raise ValidationError(
                code="chat.search_target_required",
                message=(
                    "At least one knowledge_base_id, knowledge_base_ids, "
                    "knowledge_ids, or scoped tag must be provided"
                ),
            )

        return await self._searcher.search(
            tenant_id=self._tenant_id,
            query=query.strip(),
            knowledge_base_ids=merged_kb,
            knowledge_ids=merged_knowledge,
            tag_scopes=tag_scopes,
        )

    # ── Knowledge QA (RAG / pure chat, SSE) ────────────────────────

    async def stream_knowledge_qa(
        self,
        *,
        session_id: str,
        request: KnowledgeQARequestLike,
    ) -> AsyncIterator[Event]:
        """Run knowledge QA, yielding domain events for the SSE bridge.

        The leading ``agent_query`` event is always emitted first, then the
        events produced by the knowledge runner until the turn completes.
        Request validation (a non-empty session id, non-empty query) runs
        before the stream is returned so errors surface as a clean HTTP
        response instead of mid-stream.
        """
        _require_session_id(session_id)
        _require_query(getattr(request, "query", ""))
        agent = await self.resolve_agent(request.agent_id)
        return self._stream_qa(
            session_id=session_id,
            request=request,
            agent=agent,
            runner=self._knowledge_runner,
        )

    # ── Agent QA (agent loop, SSE) ─────────────────────────────────

    async def stream_agent_qa(
        self,
        *,
        session_id: str,
        request: AgentQARequestLike,
    ) -> AsyncIterator[Event]:
        """Run agent QA, yielding domain events for the SSE bridge.

        Agent mode requires a resolvable custom agent; when the request
        enables agent mode without an ``agent_id`` the turn is rejected
        with a validation error before the stream is returned.
        """
        _require_session_id(session_id)
        _require_query(getattr(request, "query", ""))
        agent = await self._resolve_agent_for_turn(request)
        return self._stream_qa(
            session_id=session_id,
            request=request,
            agent=agent,
            runner=self._agent_runner,
        )

    async def _resolve_agent_for_turn(self, request: AgentQARequestLike) -> CustomAgentInfo | None:
        """Resolve the agent and enforce the agent-mode gate."""
        agent = await self.resolve_agent(request.agent_id)
        if resolve_agent_mode(agent_enabled=request.agent_enabled, agent=agent) and agent is None:
            raise ValidationError(
                code="chat.agent_required",
                message="agent_id is required when agent mode is enabled",
            )
        return agent

    # ── Shared stream bridge ───────────────────────────────────────

    async def _yield_qa_events(
        self,
        queue: asyncio.Queue[Event | None],
        task: asyncio.Task[None],
        session_id: str,
        assistant: AssistantMessage,
    ) -> AsyncIterator[Event]:
        answer_parts: list[str] = []
        try:
            async for event in _iter_qa_events(
                queue,
                session_id=session_id,
                message_id=assistant.id,
                stream_manager=self._stream_manager,
            ):
                piece = answer_delta(event)
                if piece:
                    answer_parts.append(piece)
                yield event
        finally:
            await _cancel_task(task)
            await self._message_gateway.complete_assistant_message(
                assistant_message_id=assistant.id,
                content="".join(answer_parts),
                is_fallback=False,
            )

    async def _stream_qa(
        self,
        *,
        session_id: str,
        request: KnowledgeQARequestLike,
        agent: CustomAgentInfo | None,
        runner: QARunner,
    ) -> AsyncIterator[Event]:
        # Message shells: the assistant message id rides the leading
        # agent_query event so the client can correlate the stream.
        user_message_id, assistant, queue, event_bus = await _open_qa_turn(
            self._message_gateway,
            session_id=session_id,
            query=request.query,
            request_id=self._request_id,
            agent=agent,
            model_id=request.summary_model_id or "",
        )
        yield _agent_query_event(session_id, self._request_id, user_message_id, assistant.id)
        task = asyncio.create_task(
            _run_qa_runner(
                runner=runner,
                ctx=RequestContext(
                    tenant_id=self._tenant_id,
                    user_id=self._user_id,
                    request_id=self._request_id,
                ),
                session_id=session_id,
                request=request,
                agent=agent,
                event_bus=event_bus,
                queue=queue,
                request_id=self._request_id,
            )
        )
        async for event in self._yield_qa_events(queue, task, session_id, assistant):
            yield event


__all__ = [
    "WIRE_RESPONSE_TYPE",
    "AgentQARequestLike",
    "AgentResolver",
    "AssistantMessage",
    "ChatService",
    "Context",
    "KnowledgeQARequestLike",
    "KnowledgeSearcher",
    "MentionedItemLike",
    "MessageGateway",
    "QARunner",
    "RequestContext",
    "SearchResult",
    "TagScope",
    "answer_delta",
    "build_tag_scopes",
    "merge_knowledge_targets",
    "resolve_agent_mode",
]
