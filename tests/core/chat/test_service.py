"""Unit tests for the request-scoped chat service.

Exercises the orchestration that the web tests stub out: knowledge
target merging, tag-scope building, request validation, the agent-mode
gate, and the SSE stream bridge (leading ``agent_query`` + runner events).
All heavy seams are replaced with tiny in-memory fakes.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.common.exception import ValidationError
from src.core.agents.types import (
    AGENT_MODE_SMART_REASONING,
    CustomAgentInfo,
)
from src.core.chat.bus import Event, EventBus
from src.core.chat.pipeline.types import Context, SearchResult
from src.core.chat.service import (
    AgentResolver,
    AssistantMessage,
    ChatService,
    KnowledgeSearcher,
    MessageGateway,
    QARunner,
    TagScope,
    answer_delta,
    build_tag_scopes,
    merge_knowledge_targets,
    resolve_agent_mode,
)
from src.core.chat.types import EventType

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


# ── Value fakes ───────────────────────────────────────────────────────


class _Item:
    """Minimal structural stand-in for a ``MentionedItemRequest``."""

    def __init__(self, item_id: str, item_type: str, kb_id: str | None = None) -> None:
        self.id = item_id
        self.type = item_type
        self.kb_id = kb_id


class _AgentResolver(AgentResolver):
    def __init__(self, agents: dict[str, CustomAgentInfo]) -> None:
        self._agents = agents

    async def resolve(
        self,
        *,
        tenant_id: int,
        agent_id: str,
    ) -> CustomAgentInfo | None:
        return self._agents.get(agent_id)


class _Searcher(KnowledgeSearcher):
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.results: list[SearchResult] = []

    async def search(
        self,
        *,
        tenant_id: int,
        query: str,
        knowledge_base_ids: list[str],
        knowledge_ids: list[str],
        tag_scopes: list[TagScope],
    ) -> list[SearchResult]:
        self.calls.append(
            {
                "tenant_id": tenant_id,
                "query": query,
                "knowledge_base_ids": list(knowledge_base_ids),
                "knowledge_ids": list(knowledge_ids),
                "tag_scopes": tag_scopes,
            }
        )
        return list(self.results)


class _Runner(QARunner):
    """Records the run call and emits a fixed event sequence."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def run(
        self,
        *,
        ctx: Context,
        session_id: str,
        request: object,
        agent: CustomAgentInfo | None,
        event_bus: EventBus,
    ) -> None:
        self.calls.append({"session_id": session_id, "agent": agent})
        await event_bus.emit(
            Event(
                type=EventType.AGENT_THOUGHT,
                session_id=session_id,
                data={"content": "thought", "done": False},
            )
        )
        await event_bus.emit(
            Event(
                type=EventType.AGENT_COMPLETE,
                session_id=session_id,
                data={"final_answer": "done"},
            )
        )


class _Gateway(MessageGateway):
    def __init__(self) -> None:
        self.user_messages: list[str] = []
        self.assistant_messages: list[str] = []
        self.completed: list[tuple[str, str]] = []

    async def create_user_message(self, *, session_id: str, query: str) -> str:
        self.user_messages.append(session_id)
        return f"user-{len(self.user_messages)}"

    async def create_assistant_message(
        self,
        *,
        session_id: str,
        request_id: str,
        agent: CustomAgentInfo | None,
        model_id: str,
    ) -> AssistantMessage:
        self.assistant_messages.append(session_id)
        return AssistantMessage(
            id=f"assistant-{len(self.assistant_messages)}",
            session_id=session_id,
        )

    async def complete_assistant_message(
        self,
        *,
        assistant_message_id: str,
        content: str,
        is_fallback: bool = False,
    ) -> None:
        del is_fallback
        self.completed.append((assistant_message_id, content))


class _QARequest:
    """Structural stand-in for the web request body."""

    def __init__(
        self,
        *,
        query: str,
        agent_id: str | None = None,
        agent_enabled: bool = False,
        summary_model_id: str | None = None,
        mentioned_items: list[object] | None = None,
    ) -> None:
        self.query = query
        self.knowledge_base_ids = None
        self.knowledge_ids = None
        self.agent_id = agent_id
        self.agent_enabled = agent_enabled
        self.web_search_enabled = False
        self.summary_model_id = summary_model_id
        self.mcp_service_ids = None
        self.skill_names = None
        self.tag_ids = None
        self.mentioned_items = mentioned_items
        self.disable_title = False
        self.images = None
        self.channel = None
        self.attachment_ids = None
        self.suggestion_attribution = None


def _agent(agent_id: str = "agent-1", *, mode: str = AGENT_MODE_SMART_REASONING) -> CustomAgentInfo:
    return CustomAgentInfo(
        id=agent_id,
        name="Test Agent",
        tenant_id=1,
        config={"agent_mode": mode},
        created_at=_NOW,
        updated_at=_NOW,
    )


def _service(
    *,
    agents: dict[str, CustomAgentInfo] | None = None,
    searcher: _Searcher | None = None,
    runner: _Runner | None = None,
    gateway: _Gateway | None = None,
) -> ChatService:
    return ChatService(
        tenant_id=7,
        user_id="user-1",
        request_id="req-1",
        agent_resolver=_AgentResolver(agents or {}),
        searcher=searcher or _Searcher(),
        knowledge_runner=runner or _Runner(),
        agent_runner=runner or _Runner(),
        message_gateway=gateway or _Gateway(),
    )


# ── Pure helpers ──────────────────────────────────────────────────────


def test_merge_knowledge_targets_deduplicates_and_merges_mentions() -> None:
    kb_ids, knowledge_ids = merge_knowledge_targets(
        knowledge_base_ids=["kb-1", "kb-1", "kb-2"],
        knowledge_ids=["k-1", "k-1"],
        mentioned_items=[_Item("kb-3", "kb"), _Item("k-2", "file"), _Item("k-1", "file")],
    )
    assert kb_ids == ["kb-1", "kb-2", "kb-3"]
    assert knowledge_ids == ["k-1", "k-2"]


def test_build_tag_scopes_scopes_request_tags_to_kbs() -> None:
    scopes = build_tag_scopes(
        tag_ids=["t1"],
        mentioned_items=[_Item("t2", "tag", kb_id="kb-9")],
        knowledge_base_ids=["kb-1", "kb-2"],
    )
    assert scopes == [
        TagScope(knowledge_base_id="kb-9", tag_ids=("t2",)),
        TagScope(knowledge_base_id="kb-1", tag_ids=("t1",)),
        TagScope(knowledge_base_id="kb-2", tag_ids=("t1",)),
    ]


def test_build_tag_scopes_rejects_orphan_tag_ids() -> None:
    with pytest.raises(ValidationError):
        build_tag_scopes(tag_ids=["t1"], mentioned_items=None, knowledge_base_ids=[])


def test_resolve_agent_mode_prefers_agent_config() -> None:
    assert (
        resolve_agent_mode(agent_enabled=False, agent=_agent(mode=AGENT_MODE_SMART_REASONING))
        is True
    )
    assert resolve_agent_mode(agent_enabled=True, agent=_agent(mode="quick-answer")) is False
    assert resolve_agent_mode(agent_enabled=True, agent=None) is True


# ── search_knowledge ──────────────────────────────────────────────────


async def test_search_knowledge_merges_legacy_single_kb() -> None:
    searcher = _Searcher()
    service = _service(searcher=searcher)

    await service.search_knowledge(
        query="hello",
        knowledge_base_id="kb-legacy",
    )

    call = searcher.calls[0]
    assert call["knowledge_base_ids"] == ["kb-legacy"]
    assert call["tenant_id"] == 7
    assert call["query"] == "hello"


async def test_search_knowledge_rejects_empty_query() -> None:
    service = _service()
    with pytest.raises(ValidationError) as exc:
        await service.search_knowledge(query="   ")
    assert exc.value.code == "chat.query_required"


async def test_search_knowledge_rejects_no_target() -> None:
    service = _service()
    with pytest.raises(ValidationError) as exc:
        await service.search_knowledge(query="hello")
    assert exc.value.code == "chat.search_target_required"


# ── stream_knowledge_qa / stream_agent_qa ─────────────────────────────


def test_answer_delta_reads_final_answer_content() -> None:
    event = Event(
        type=EventType.AGENT_FINAL_ANSWER,
        session_id="s1",
        data={"content": "月报"},
    )
    assert answer_delta(event) == "月报"
    assert answer_delta(Event(type=EventType.AGENT_THOUGHT, session_id="s1", data={})) == ""


async def test_stream_knowledge_qa_emits_agent_query_first() -> None:
    runner = _Runner()
    gateway = _Gateway()
    service = _service(runner=runner, gateway=gateway)
    stream = await service.stream_knowledge_qa(session_id="s1", request=_QARequest(query="hello"))

    events: list[Event] = []
    async for event in stream:
        events.append(event)

    assert events[0].type == EventType.AGENT_QUERY
    assert events[0].data is not None
    assert events[0].data["assistant_message_id"].startswith("assistant-")
    assert [e.type for e in events[1:]] == [EventType.AGENT_THOUGHT, EventType.AGENT_COMPLETE]
    assert gateway.completed == [("assistant-1", "")]


async def test_stream_knowledge_qa_rejects_blank_session() -> None:
    service = _service()
    with pytest.raises(ValidationError) as exc:
        await service.stream_knowledge_qa(session_id="  ", request=_QARequest(query="hello"))
    assert exc.value.code == "chat.session_required"


async def test_stream_agent_qa_requires_agent_id_in_agent_mode() -> None:
    service = _service()
    with pytest.raises(ValidationError) as exc:
        await service.stream_agent_qa(
            session_id="s1",
            request=_QARequest(query="x", agent_enabled=True),
        )
    assert exc.value.code == "chat.agent_required"


async def test_stream_agent_qa_allows_resolved_agent_in_agent_mode() -> None:
    runner = _Runner()
    service = _service(agents={"agent-1": _agent()}, runner=runner)
    stream = await service.stream_agent_qa(
        session_id="s1",
        request=_QARequest(query="x", agent_enabled=True, agent_id="agent-1"),
    )

    events: list[Event] = []
    async for _event in stream:
        events.append(_event)

    assert events and events[0].type == EventType.AGENT_QUERY
    assert runner.calls[0]["agent"] is not None


async def test_stream_agent_qa_forwards_agent_to_runner() -> None:
    runner = _Runner()
    service = _service(agents={"agent-1": _agent()}, runner=runner)
    stream = await service.stream_agent_qa(
        session_id="s1",
        request=_QARequest(query="x", agent_id="agent-1"),
    )
    async for _event in stream:
        pass
    assert runner.calls[0]["session_id"] == "s1"


__all__ = [
    "_Gateway",
    "_Item",
    "_QARequest",
    "_Runner",
    "_Searcher",
    "_service",
    "build_tag_scopes",
    "merge_knowledge_targets",
    "resolve_agent_mode",
]
