"""Unit tests for the knowledge-QA runner.

Covers model-id resolution, chunk hydration, and the into-chat + stream
path. All store and model seams are in-memory fakes.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from src.ai.llm.types import ChatOptions, ChatResponse, Message, ResponseType, StreamResponse
from src.common.exception import ValidationError
from src.common.json import JsonObject
from src.core.agents.types import CustomAgentInfo
from src.core.chat.bus import Event, EventBus
from src.core.chat.service import MentionedItemLike
from src.core.chat.sessions.knowledge_qa_runner import (
    KnowledgeQARunner,
    collect_document_ids,
    default_summary_config,
    load_chunk_hits,
    resolve_knowledge_qa_model_id,
    summary_config_for_agent,
)
from src.core.chat.types import EventType as ChatEventType
from src.core.contracts.knowledge import Knowledge
from src.core.knowledge.knowledge_bases.types import KnowledgeBaseInfo

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


class _Ctx:
    """Opaque execution context satisfying the ``Context`` protocol."""

    tenant_id = 1
    user_id = "test-user"
    request_id = "req-1"


@dataclass
class _Req:
    """Minimal knowledge-QA request body."""

    query: str
    knowledge_base_ids: Sequence[str] | None = None
    knowledge_ids: Sequence[str] | None = None
    agent_id: str | None = None
    summary_model_id: str | None = None
    mcp_service_ids: Sequence[str] | None = None
    skill_names: Sequence[str] | None = None
    tag_ids: Sequence[str] | None = None
    mentioned_items: Sequence[MentionedItemLike] | None = None
    disable_title: bool = False
    channel: str | None = None
    attachment_ids: Sequence[str] | None = None


@dataclass
class _Chunk:
    """In-memory text chunk matching ``TextChunkLike``."""

    id: str
    content: str
    knowledge_id: str
    knowledge_base_id: str
    chunk_index: int
    start_at: int = 0
    end_at: int = 10
    chunk_type: str = "text"
    parent_chunk_id: str | None = None
    image_info: str | None = None
    metadata: JsonObject | None = None
    is_enabled: bool = True


class _RecordingBus:
    """Real bus that records every streamed domain event."""

    def __init__(self) -> None:
        self.events: list[Event] = []
        self.bus = EventBus()
        for event_type in (
            ChatEventType.AGENT_REFERENCES,
            ChatEventType.AGENT_THOUGHT,
            ChatEventType.AGENT_FINAL_ANSWER,
            ChatEventType.AGENT_COMPLETE,
            ChatEventType.ERROR,
        ):
            self.bus.on(event_type, self._sink)

    async def _sink(self, event: Event) -> None:
        self.events.append(event)


class _FakeChat:
    """Chat client that streams a scripted answer."""

    def __init__(self) -> None:
        self.stream_calls: list[list[Message]] = []

    async def chat(self, messages: list[Message], opts: ChatOptions | None = None) -> ChatResponse:
        return ChatResponse(content="unused")

    def chat_stream(
        self, messages: list[Message], opts: ChatOptions | None = None
    ) -> AsyncIterator[StreamResponse]:
        self.stream_calls.append(messages)

        async def _gen() -> AsyncIterator[StreamResponse]:
            yield StreamResponse(
                response_type=ResponseType.ANSWER,
                content="公募基金",
                done=False,
            )
            yield StreamResponse(
                response_type=ResponseType.ANSWER,
                content="月报要点",
                done=True,
            )

        return _gen()

    def get_model_name(self) -> str:
        return "fake"

    def get_model_id(self) -> str:
        return "qa-1"


class _FakeChatModels:
    """In-memory ``ChatModelService`` stand-in."""

    def __init__(
        self,
        *,
        types: dict[str, str],
        chat: _FakeChat | None = None,
        first_id: str | None = None,
    ) -> None:
        self._types = types
        self.chat = chat or _FakeChat()
        self._first_id = first_id
        self.loaded: list[str] = []

    async def get_chat_model(self, *, tenant_id: int, model_id: str) -> _FakeChat:
        self.loaded.append(model_id)
        return self.chat

    async def get_model_type(self, *, tenant_id: int, model_id: str) -> str | None:
        return self._types.get(model_id)

    async def first_knowledge_qa_id(self, *, tenant_id: int) -> str | None:
        return self._first_id


class _FakeDocuments:
    """In-memory document loader."""

    def __init__(self, docs: list[Knowledge]) -> None:
        self._docs = docs

    async def get_documents(self, *, tenant_id: int, ids: list[str]) -> list[Knowledge]:
        wanted = set(ids)
        return [doc for doc in self._docs if doc.id in wanted]

    async def list_documents(self, *, tenant_id: int, knowledge_base_id: str) -> list[Knowledge]:
        return [doc for doc in self._docs if doc.knowledge_base_id == knowledge_base_id]


class _FakeChunks:
    """In-memory chunk loader."""

    def __init__(self, chunks: dict[str, list[_Chunk]]) -> None:
        self._chunks = chunks

    async def list_chunks_by_knowledge_id(
        self, *, tenant_id: int, knowledge_id: str
    ) -> Sequence[_Chunk]:
        return self._chunks.get(knowledge_id, [])


class _FakeKBs:
    """In-memory knowledge-base reader."""

    def __init__(self, by_id: dict[str, KnowledgeBaseInfo]) -> None:
        self._by_id = by_id

    async def get_knowledge_base_by_id_and_tenant(
        self, *, tenant_id: int, knowledge_base_id: str
    ) -> KnowledgeBaseInfo:
        kb = self._by_id.get(knowledge_base_id)
        if kb is None:
            raise ValidationError(
                code="kb.not_found",
                message=f"knowledge base {knowledge_base_id} not found",
            )
        return kb


def _document(*, doc_id: str = "doc-1", kb_id: str = "kb-1") -> Knowledge:
    return Knowledge(
        id=doc_id,
        tenant_id=1,
        knowledge_base_id=kb_id,
        type="document",
        title="公募基金9月月报",
        parse_status="completed",
        enable_status="enabled",
        file_name="fund.html",
        created_at=_NOW,
        updated_at=_NOW,
    )


def _kb(*, kb_id: str = "kb-1", summary_model_id: str = "") -> KnowledgeBaseInfo:
    return KnowledgeBaseInfo(
        id=kb_id,
        name="基金研报",
        tenant_id=1,
        summary_model_id=summary_model_id,
        created_at=_NOW,
        updated_at=_NOW,
    )


def test_summary_config_for_agent_uses_prompt_and_sampling() -> None:
    agent = _agent()
    agent = agent.model_copy(
        update={
            "config": {
                "model_id": "",
                "system_prompt": "只根据材料回答。",
                "temperature": 0.2,
                "max_completion_tokens": 512,
            }
        }
    )
    config = summary_config_for_agent(agent)
    assert config.prompt == "只根据材料回答。"
    assert config.temperature == 0.2
    assert config.max_completion_tokens == 512


def test_summary_config_for_agent_falls_back_when_prompt_blank() -> None:
    assert summary_config_for_agent(_agent()).prompt == default_summary_config().prompt


def _agent(*, model_id: str = "") -> CustomAgentInfo:
    return CustomAgentInfo(
        id="builtin-quick-answer",
        name="Quick",
        tenant_id=1,
        is_builtin=True,
        config={"model_id": model_id},
        created_at=_NOW,
        updated_at=_NOW,
    )


def _runner(
    *,
    types: dict[str, str] | None = None,
    first_id: str | None = None,
    docs: list[Knowledge] | None = None,
    chunks: dict[str, list[_Chunk]] | None = None,
    kbs: dict[str, KnowledgeBaseInfo] | None = None,
    chat: _FakeChat | None = None,
) -> tuple[KnowledgeQARunner, _FakeChat]:
    fake_chat = chat or _FakeChat()
    models = _FakeChatModels(
        types=types or {"qa-1": "KnowledgeQA"},
        chat=fake_chat,
        first_id=first_id,
    )
    runner = KnowledgeQARunner(
        chat_models=models,
        documents=_FakeDocuments(docs or [_document()]),
        chunks=_FakeChunks(chunks or {}),
        knowledge_bases=_FakeKBs(kbs or {"kb-1": _kb()}),
    )
    return runner, fake_chat


async def test_resolve_prefers_request_summary_model() -> None:
    models = _FakeChatModels(
        types={"override": "KnowledgeQA", "agent": "KnowledgeQA"},
        first_id="fallback",
    )
    chosen = await resolve_knowledge_qa_model_id(
        tenant_id=1,
        request=_Req(query="q", summary_model_id="override"),
        agent=_agent(model_id="agent"),
        knowledge_base_ids=["kb-1"],
        chat_models=models,
        knowledge_bases=_FakeKBs({"kb-1": _kb(summary_model_id="kb-model")}),
    )
    assert chosen == "override"


async def test_resolve_falls_through_invalid_override_to_tenant_model() -> None:
    models = _FakeChatModels(types={"bad": "Embedding", "qa-1": "KnowledgeQA"}, first_id="qa-1")
    chosen = await resolve_knowledge_qa_model_id(
        tenant_id=1,
        request=_Req(query="q", summary_model_id="bad"),
        agent=None,
        knowledge_base_ids=[],
        chat_models=models,
        knowledge_bases=_FakeKBs({}),
    )
    assert chosen == "qa-1"


async def test_resolve_uses_kb_summary_model() -> None:
    models = _FakeChatModels(types={"kb-qa": "KnowledgeQA"}, first_id=None)
    chosen = await resolve_knowledge_qa_model_id(
        tenant_id=1,
        request=_Req(query="q"),
        agent=None,
        knowledge_base_ids=["kb-1"],
        chat_models=models,
        knowledge_bases=_FakeKBs({"kb-1": _kb(summary_model_id="kb-qa")}),
    )
    assert chosen == "kb-qa"


async def test_resolve_raises_when_no_model() -> None:
    models = _FakeChatModels(types={}, first_id=None)
    with pytest.raises(ValidationError, match="No KnowledgeQA model"):
        await resolve_knowledge_qa_model_id(
            tenant_id=1,
            request=_Req(query="q"),
            agent=None,
            knowledge_base_ids=[],
            chat_models=models,
            knowledge_bases=_FakeKBs({}),
        )


async def test_collect_document_ids_prefers_explicit_files() -> None:
    docs = [_document(doc_id="doc-1"), _document(doc_id="doc-2")]
    ids = await collect_document_ids(
        tenant_id=1,
        knowledge_ids=["doc-1"],
        knowledge_base_ids=["kb-1"],
        documents=_FakeDocuments(docs),
    )
    assert ids == ["doc-1"]


async def test_load_chunk_hits_skips_disabled_and_caps() -> None:
    chunks = {
        "doc-1": [
            _Chunk(
                id="c-off",
                content="hidden",
                knowledge_id="doc-1",
                knowledge_base_id="kb-1",
                chunk_index=0,
                is_enabled=False,
            ),
            _Chunk(
                id="c-1",
                content="公募基金规模",
                knowledge_id="doc-1",
                knowledge_base_id="kb-1",
                chunk_index=1,
            ),
            _Chunk(
                id="c-2",
                content="extra",
                knowledge_id="doc-1",
                knowledge_base_id="kb-1",
                chunk_index=2,
            ),
        ]
    }
    hits = await load_chunk_hits(
        tenant_id=1,
        knowledge_ids=["doc-1"],
        documents=_FakeDocuments([_document()]),
        chunks=_FakeChunks(chunks),
        cap=1,
    )
    assert len(hits) == 1
    assert hits[0].id == "c-1"
    assert hits[0].knowledge_title == "公募基金9月月报"


async def test_runner_streams_chunk_grounded_answer_and_completes() -> None:
    chunks = {
        "doc-1": [
            _Chunk(
                id="c-1",
                content="公募基金9月规模变化",
                knowledge_id="doc-1",
                knowledge_base_id="kb-1",
                chunk_index=0,
            )
        ]
    }
    runner, chat = _runner(
        types={"qa-1": "KnowledgeQA"},
        first_id="qa-1",
        chunks=chunks,
    )
    bus = _RecordingBus()
    await runner.run(
        ctx=_Ctx(),
        session_id="sess-1",
        request=_Req(query="这份研报讲了什么", knowledge_ids=["doc-1"], summary_model_id="qa-1"),
        agent=_agent(),
        event_bus=bus.bus,
    )
    types = [event.type for event in bus.events]
    assert ChatEventType.AGENT_REFERENCES in types
    assert ChatEventType.AGENT_FINAL_ANSWER in types
    assert types[-1] == ChatEventType.AGENT_COMPLETE
    assert chat.stream_calls
    user_text = chat.stream_calls[0][-1].content
    assert "公募基金9月规模变化" in user_text
    assert "这份研报讲了什么" in user_text


async def test_runner_pure_chat_without_knowledge_scope() -> None:
    runner, chat = _runner(types={"qa-1": "KnowledgeQA"}, first_id="qa-1")
    bus = _RecordingBus()
    await runner.run(
        ctx=_Ctx(),
        session_id="sess-1",
        request=_Req(query="你好", summary_model_id="qa-1"),
        agent=None,
        event_bus=bus.bus,
    )
    types = [event.type for event in bus.events]
    assert ChatEventType.AGENT_REFERENCES not in types
    assert ChatEventType.AGENT_FINAL_ANSWER in types
    assert types[-1] == ChatEventType.AGENT_COMPLETE
    assert chat.stream_calls[0][-1].content == "你好"
