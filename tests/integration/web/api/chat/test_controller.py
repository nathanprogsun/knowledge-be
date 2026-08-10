"""Web integration tests for the chat endpoints (knowledge / agent / search).

Exercises the full HTTP path over ``TestClient`` against the real app.
The ``ChatService`` dependency is overridden with a fake-backed service
so no LLM / retrieval / message store is needed.

Endpoint coverage:

| Method | Path                          |
| ------ | ----------------------------- |
| POST   | /knowledge-chat/{session_id}  |
| POST   | /agent-chat/{session_id}      |
| POST   | /knowledge-search             |

Auth: header trio on the authed client; unauth tests build a bare
``TestClient`` and assert the 401 from ``require_auth``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.ai.retrieval.types import MatchType
from src.common.exception import ValidationError
from src.core.chat.bus import Event
from src.core.chat.pipeline.types import SearchResult
from src.core.chat.types import EventType
from src.web.deps.chat import get_chat_service

# ── Fake service backing the dep override ─────────────────────────────


class _FakeChatService:
    """In-memory ``ChatService`` stand-in controlling every turn's events.

    The real ``ChatService`` is constructed per request; this fake
    mirrors its public surface (``search_knowledge``, agent + knowledge
    stream methods) so the router layer is exercised without any
    LLM / retrieval / message infrastructure.
    """

    def __init__(self) -> None:
        #: Stubbed search hits returned by ``search_knowledge``.
        self.search_results: list[SearchResult] = []
        #: Events emitted by the QA stream methods.
        self.qa_events: list[Event] = []
        #: Validation errors to raise from the services.
        self.search_error: dict[str, str] | None = None
        self.qa_error: dict[str, str] | None = None
        self.agent_mode_error: bool = False
        self.calls: list[tuple[str, dict[str, Any]]] = []
        #: Request id stamped on streamed wire frames.
        self.request_id = "req-fake-1"

    def _record(self, method: str, **kwargs: Any) -> None:
        self.calls.append((method, kwargs))

    async def search_knowledge(
        self,
        *,
        query: str,
        knowledge_base_id: str | None = None,
        knowledge_base_ids: list[str] | None = None,
        knowledge_ids: list[str] | None = None,
        tag_ids: list[str] | None = None,
        mentioned_items: list[Any] | None = None,
    ) -> list[SearchResult]:
        self._record(
            "search_knowledge",
            query=query,
            knowledge_base_id=knowledge_base_id,
            knowledge_base_ids=knowledge_base_ids,
            knowledge_ids=knowledge_ids,
            tag_ids=tag_ids,
        )
        if self.search_error is not None:
            raise ValidationError(
                code=self.search_error["code"],
                message=self.search_error["message"],
            )
        return list(self.search_results)

    async def stream_knowledge_qa(
        self,
        *,
        session_id: str,
        request: object,
    ) -> AsyncIterator[Event]:
        self._record(
            "stream_knowledge_qa",
            session_id=session_id,
            query=getattr(request, "query", ""),
        )
        if self.qa_error is not None:
            raise ValidationError(
                code=self.qa_error["code"],
                message=self.qa_error["message"],
            )

        async def _events() -> AsyncIterator[Event]:
            yield Event(
                type=EventType.AGENT_QUERY,
                session_id=session_id,
                data={
                    "session_id": session_id,
                    "assistant_message_id": "msg-fake-1",
                },
            )
            for event in self.qa_events:
                yield event

        return _events()

    async def stream_agent_qa(
        self,
        *,
        session_id: str,
        request: object,
    ) -> AsyncIterator[Event]:
        self._record(
            "stream_agent_qa",
            session_id=session_id,
            query=getattr(request, "query", ""),
        )
        if self.agent_mode_error:
            raise ValidationError(
                code="chat.agent_required",
                message="agent_id is required when agent mode is enabled",
            )
        if self.qa_error is not None:
            raise ValidationError(
                code=self.qa_error["code"],
                message=self.qa_error["message"],
            )

        async def _events() -> AsyncIterator[Event]:
            yield Event(
                type=EventType.AGENT_QUERY,
                session_id=session_id,
                data={
                    "session_id": session_id,
                    "assistant_message_id": "msg-fake-1",
                },
            )
            for event in self.qa_events:
                yield event

        return _events()


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def fake_service() -> _FakeChatService:
    return _FakeChatService()


@pytest.fixture
def app(
    web_app: FastAPI,
    fake_service: _FakeChatService,
) -> FastAPI:
    """Override the per-request chat service factory with the fake."""

    def _override() -> Any:
        return fake_service

    web_app.dependency_overrides[get_chat_service] = _override
    return web_app


@pytest.fixture
def client(app: FastAPI, web_authed_client: TestClient) -> TestClient:
    return web_authed_client


@pytest.fixture
def anon_client(app: FastAPI) -> Iterator[TestClient]:
    """A ``TestClient`` without the auth header trio — 401 surface."""
    with TestClient(app=app) as c:
        yield c


def _hit(hit_id: str) -> SearchResult:
    return SearchResult(
        id=hit_id,
        content=f"content-{hit_id}",
        knowledge_id=f"kids-{hit_id}",
        knowledge_base_id="kb-1",
        knowledge_title=f"title-{hit_id}",
        score=0.9,
        match_type=MatchType.EMBEDDING,
    )


# ── /knowledge-search ────────────────────────────────────────────────


def test_knowledge_search_returns_envelope(
    client: TestClient,
    fake_service: _FakeChatService,
) -> None:
    """The search endpoint wraps yielded hits in the success envelope."""
    fake_service.search_results = [_hit("c1"), _hit("c2")]

    resp = client.post(
        "/knowledge-search",
        json={"query": "hello", "knowledge_base_ids": ["kb-1"]},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert [r["id"] for r in body["data"]] == ["c1", "c2"]
    assert body["data"][0]["knowledge_base_id"] == "kb-1"
    call = fake_service.calls[0][1]
    assert call["query"] == "hello"
    assert call["knowledge_base_ids"] == ["kb-1"]


def test_knowledge_search_merges_single_kb_id(
    client: TestClient,
    fake_service: _FakeChatService,
) -> None:
    """The backward-compat ``knowledge_base_id`` field is forwarded.

    The merge into ``knowledge_base_ids`` happens inside the service; the
    view forwards the raw single-KB field unchanged.
    """
    fake_service.search_results = [_hit("c1")]

    resp = client.post(
        "/knowledge-search",
        json={"query": "find", "knowledge_base_id": "kb-legacy"},
    )

    assert resp.status_code == 200
    call = fake_service.calls[0][1]
    assert call["knowledge_base_id"] == "kb-legacy"
    assert call["knowledge_base_ids"] is None


def test_knowledge_search_rejects_empty_query(client: TestClient) -> None:
    """A blank query fails validation at the Pydantic body boundary."""
    resp = client.post("/knowledge-search", json={"query": ""})
    assert resp.status_code == 422


def test_knowledge_search_rejects_no_target(
    client: TestClient,
    fake_service: _FakeChatService,
) -> None:
    """No knowledge scope (base / knowledge / tags) is rejected by the service."""
    fake_service.search_error = {
        "code": "chat.search_target_required",
        "message": (
            "At least one knowledge_base_id, knowledge_base_ids, "
            "knowledge_ids, or scoped tag must be provided"
        ),
    }
    resp = client.post("/knowledge-search", json={"query": "find"})
    assert resp.status_code == 422


# ── /knowledge-chat/{session_id} ──────────────────────────────────────


def test_knowledge_qa_streams_events(
    client: TestClient,
    fake_service: _FakeChatService,
) -> None:
    """The knowledge-QA endpoint streams agent_query, thinking, answer frames."""
    fake_service.qa_events = [
        Event(
            type=EventType.AGENT_THOUGHT,
            session_id="s1",
            data={"content": "thinking", "done": False},
        ),
        Event(
            type=EventType.AGENT_FINAL_ANSWER,
            session_id="s1",
            data={"content": "Answer text", "done": False},
        ),
        Event(
            type=EventType.AGENT_FINAL_ANSWER,
            session_id="s1",
            data={"content": "", "done": True},
        ),
        Event(
            type=EventType.AGENT_COMPLETE,
            session_id="s1",
            data={"final_answer": "Answer text"},
        ),
    ]

    chunks: list[str] = []
    with client.stream("POST", "/knowledge-chat/s1", json={"query": "hello"}) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        for chunk in resp.iter_text():
            chunks.append(chunk)
    body = "".join(chunks)

    assert "event: message" in body
    assert '"response_type": "agent_query"' in body
    assert '"response_type": "thinking"' in body
    assert '"content": "thinking"' in body
    assert '"response_type": "answer"' in body
    assert '"content": "Answer text"' in body
    assert '"response_type": "complete"' in body


def test_knowledge_qa_rejects_empty_session(client: TestClient) -> None:
    """A blank-session path is rejected before streaming."""
    resp = client.post("/knowledge-chat/%20%20", json={"query": "hello"})
    assert resp.status_code == 422


def test_knowledge_qa_rejects_empty_query(client: TestClient) -> None:
    """A blank query is rejected before the stream is opened."""
    resp = client.post("/knowledge-chat/s1", json={"query": "  "})
    assert resp.status_code == 422


# ── /agent-chat/{session_id} ──────────────────────────────────────────


def test_agent_qa_streams_events(
    client: TestClient,
    fake_service: _FakeChatService,
) -> None:
    """The agent-chat endpoint streams the same event vocabulary."""
    fake_service.qa_events = [
        Event(
            type=EventType.AGENT_THOUGHT,
            session_id="s1",
            data={"content": "thought", "done": False},
        ),
        Event(
            type=EventType.AGENT_TOOL_CALL,
            session_id="s1",
            data={"tool_name": "web_search", "tool_call_id": "tc1", "iteration": 0},
        ),
        Event(
            type=EventType.AGENT_TOOL_RESULT,
            session_id="s1",
            data={"tool_name": "web_search", "tool_call_id": "tc1", "success": True},
        ),
        Event(
            type=EventType.AGENT_FINAL_ANSWER,
            session_id="s1",
            data={"content": "final", "done": True},
        ),
    ]

    chunks: list[str] = []
    with client.stream("POST", "/agent-chat/s1", json={"query": "search this"}) as resp:
        assert resp.status_code == 200
        for chunk in resp.iter_text():
            chunks.append(chunk)
    body = "".join(chunks)

    assert '"response_type": "tool_call"' in body
    assert '"response_type": "tool_result"' in body
    assert '"response_type": "answer"' in body
    assert '"content": "final"' in body


def test_agent_qa_without_agent_id_is_rejected(
    client: TestClient,
    fake_service: _FakeChatService,
) -> None:
    """Agent mode without a resolvable agent id yields a validation error."""
    fake_service.agent_mode_error = True
    with client.stream(
        "POST",
        "/agent-chat/s1",
        json={"query": "x", "agent_enabled": True},
    ) as resp:
        assert resp.status_code == 422


# ── Auth gate ─────────────────────────────────────────────────────────


def test_unauthed_knowledge_search_returns_401(anon_client: TestClient) -> None:
    resp = anon_client.post(
        "/knowledge-search",
        json={"query": "x", "knowledge_base_ids": ["kb"]},
    )
    assert resp.status_code == 401


def test_unauthed_knowledge_qa_returns_401(anon_client: TestClient) -> None:
    resp = anon_client.post("/knowledge-chat/s1", json={"query": "x"})
    assert resp.status_code == 401


def test_unauthed_agent_qa_returns_401(anon_client: TestClient) -> None:
    resp = anon_client.post("/agent-chat/s1", json={"query": "x"})
    assert resp.status_code == 401


__all__ = [
    "_FakeChatService",
    "_hit",
    "anon_client",
    "app",
    "client",
    "fake_service",
]
