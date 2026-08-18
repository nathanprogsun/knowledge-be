"""Chat/agent web-layer contract tests (live).

Builds the real app via ``create_app`` and asserts the chat and agent
endpoints' request/response schemas against the reference API fixtures:

- the QA endpoints stream Server-Sent Events whose frames carry exactly
  the reference ``StreamResponse`` field set (``fixtures/
  chat_pipeline_responses.json`` -> ``contracts.StreamResponse``) and the
  reference ``response_type`` vocabulary;
- the knowledge-search endpoint answers the ``{success, data}`` envelope
  whose ``data`` rows carry the reference ``SearchResult`` field set;
- the agent endpoints answer the reference status codes and envelopes,
  and every ``data`` payload is valid against the frozen agent contract
  (``src.core.contracts.agents``) with exactly the contract's serialized
  field set.

The chat service dependency is overridden with a controllable fake so
the wire shape can be exercised without a live model / retrieval store;
the agent endpoints run against the real request-scoped service and the
real database.

The tests run against the real database. A failing assertion here means
the live web layer deviates from the reference wire shape — that is a
finding, not a test bug: the web layer is already merged, so deviations
are reported rather than silently fixed.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import NamedTuple

import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from src.ai.retrieval.types import MatchType
from src.core.chat.bus import Event
from src.core.chat.pipeline.types import SearchResult
from src.core.chat.types import EventType
from src.core.contracts import agents as agent_contracts
from src.core.tenants.member_service import ROLE_OWNER
from src.web.api.agents.views import SuggestedQuestionsData
from src.web.deps.chat import get_chat_service
from tests.contract.test_knowledge_invariants import model_wire_fields

_REFERENCE = "reference fixture"
_FIXTURE_PATH: Path = Path(__file__).parent / "fixtures" / "chat_pipeline_responses.json"

#: Tool-call object field set the reference ``StreamResponse`` permits.
_TOOL_CALL_FIELDS: frozenset[str] = frozenset({"id", "type", "function", "provider_metadata"})


class AgentSeed(NamedTuple):
    """A custom agent minted through the real service + database."""

    client: TestClient
    agent_id: str


class _FakeChatService:
    """In-memory ``ChatService`` stand-in controlling every turn's events.

    Mirrors the real service's public surface (``search_knowledge`` plus
    the two QA stream methods) so the router layer is exercised without
    any LLM / retrieval / message infrastructure. ``request_id`` is the
    id stamped on every streamed frame.
    """

    def __init__(self) -> None:
        #: Search hits returned by ``search_knowledge``.
        self.search_results: list[SearchResult] = []
        #: Events emitted by the QA stream methods (after the leading
        #: ``agent_query`` frame).
        self.qa_events: list[Event] = []
        #: Request id stamped on streamed wire frames.
        self.request_id = "req-contract-1"

    async def search_knowledge(
        self,
        *,
        query: str,
        knowledge_base_id: str | None = None,
        knowledge_base_ids: list[str] | None = None,
        knowledge_ids: list[str] | None = None,
        tag_ids: list[str] | None = None,
        mentioned_items: list[object] | None = None,
    ) -> list[SearchResult]:
        """Return the stubbed search hits unchanged."""
        return list(self.search_results)

    async def stream_knowledge_qa(
        self,
        *,
        session_id: str,
        request: object,
    ) -> AsyncIterator[Event]:
        """Stream the leading agent-query frame plus the stubbed events."""
        return self._stream(session_id)

    async def stream_agent_qa(
        self,
        *,
        session_id: str,
        request: object,
    ) -> AsyncIterator[Event]:
        """Stream the leading agent-query frame plus the stubbed events."""
        return self._stream(session_id)

    async def _stream(self, session_id: str) -> AsyncIterator[Event]:
        """Yield the leading agent-query event then every stubbed event."""
        yield Event(
            type=EventType.AGENT_QUERY,
            session_id=session_id,
            data={
                "session_id": session_id,
                "assistant_message_id": "msg-contract-1",
            },
        )
        for event in self.qa_events:
            yield event


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def fake_chat_service() -> _FakeChatService:
    """A fresh fake chat service per test."""
    return _FakeChatService()


@pytest_asyncio.fixture
async def chat_app(
    app: FastAPI,
    fake_chat_service: _FakeChatService,
) -> FastAPI:
    """Register the chat-service dependency override on the shared app."""
    app.dependency_overrides[get_chat_service] = lambda: fake_chat_service
    return app


@pytest_asyncio.fixture
async def chat_client(
    chat_app: FastAPI,
    admin_user: tuple[str, int],
) -> AsyncIterator[TestClient]:
    """An authed ``TestClient`` whose chat service is the fake."""
    user_id, tenant_id = admin_user
    client = TestClient(app=chat_app)
    client.headers.update(
        {
            "X-User-Id": user_id,
            "X-Tenant-ID": str(tenant_id),
            "X-Roles": ROLE_OWNER,
        }
    )
    with client:
        yield client


@pytest_asyncio.fixture
async def agent_seed(authed_client: TestClient) -> AgentSeed:
    """A custom agent minted through the real service + database."""
    response = authed_client.post("/api/v1/agents", json={"name": "contract-agent"})
    assert response.status_code == 201, response.text
    return AgentSeed(authed_client, response.json()["data"]["id"])


# ── Fixture helpers ───────────────────────────────────────────────────


def _fixture() -> dict[str, object]:
    """Load the chat/agent response fixture file."""
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


def _endpoint_spec(method: str, path: str) -> tuple[int, list[str]]:
    """Return ``(status, top-level keys)`` from the fixture for an endpoint."""
    endpoints = _fixture().get("endpoints", {})
    spec = endpoints.get(f"{method} {path}")
    assert isinstance(spec, dict), f"endpoint fixture missing: {method} {path}"
    status = spec.get("status")
    keys = spec.get("keys")
    assert isinstance(status, int) and isinstance(keys, list)
    return status, [k for k in keys if isinstance(k, str)]


def _contract_fields(name: str) -> list[str]:
    """Return the fixture's expected wire field set for a contract."""
    contracts_map = _fixture().get("contracts", {})
    assert isinstance(contracts_map, dict), "fixture 'contracts' section missing"
    fields = contracts_map.get(name)
    assert isinstance(fields, list), f"contract fixture missing: {name}"
    return [f for f in fields if isinstance(f, str)]


def _assert_keys(actual: dict[str, object], expected: list[str], label: str) -> None:
    """Assert a response body carries exactly the expected top-level keys."""
    actual_set = set(actual)
    expected_set = set(expected)
    assert actual_set == expected_set, (
        f"{label}: response keys diverge from the {_REFERENCE}.\n"
        f"  missing: {sorted(expected_set - actual_set)}\n"
        f"  extra:   {sorted(actual_set - expected_set)}"
    )


def _assert_payload(
    payload: object,
    model: type[BaseModel],
    label: str,
) -> None:
    """Assert a response ``data`` payload matches ``model`` exactly."""
    assert isinstance(payload, dict), f"{label}: expected a JSON object, got {type(payload)}"
    _assert_keys(payload, model_wire_fields(model), f"{label} data")
    model.model_validate(payload)


def _parse_sse_frames(body: str) -> list[dict[str, object]]:
    """Parse the ``event: message`` SSE blocks of a QA stream body."""
    frames: list[dict[str, object]] = []
    for block in body.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        assert block.startswith("event: message\n"), f"unexpected SSE block: {block!r}"
        data_line = block[len("event: message\n") :]
        assert data_line.startswith("data: "), f"unexpected SSE data line: {data_line!r}"
        raw = data_line[len("data: ") :]
        frames.append(json.loads(raw))
    return frames


def _assert_frame(frame: dict[str, object], label: str) -> None:
    """Assert one SSE frame carries only the reference ``StreamResponse`` keys."""
    stream_fields = set(_contract_fields("StreamResponse"))
    unknown = set(frame) - stream_fields
    assert not unknown, f"{label}: unexpected frame keys {sorted(unknown)}"
    for required in ("id", "response_type", "content", "done"):
        assert required in frame, f"{label}: missing required frame key {required!r}"


def _hit_dict(hit_id: str) -> dict[str, object]:
    """Build a JSON-shaped search hit (the dict form the view coerces)."""
    return {
        "id": hit_id,
        "content": f"content-{hit_id}",
        "knowledge_id": f"knowledge-{hit_id}",
        "knowledge_base_id": "kb-1",
        "knowledge_title": f"title-{hit_id}",
        "score": 0.9,
        "match_type": 0,
    }


# ── Knowledge QA (SSE) ───────────────────────────────────────────────


def test_knowledge_qa_stream_frames_match_reference(
    chat_client: TestClient,
    fake_chat_service: _FakeChatService,
) -> None:
    """The knowledge-QA endpoint streams reference-shaped frames."""
    fake_chat_service.qa_events = [
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
    status, _ = _endpoint_spec("POST", "/api/v1/knowledge-chat/{session_id}")
    chunks: list[str] = []
    with chat_client.stream("POST", "/api/v1/knowledge-chat/s1", json={"query": "hello"}) as resp:
        assert resp.status_code == status, resp.text
        assert resp.headers["content-type"].startswith("text/event-stream")
        for chunk in resp.iter_text():
            chunks.append(chunk)

    frames = _parse_sse_frames("".join(chunks))
    assert frames, "no SSE frames received"
    assert frames[0]["response_type"] == "agent_query"
    assert frames[0]["id"] == "req-contract-1"
    assert frames[0]["session_id"] == "s1"
    assert frames[0]["assistant_message_id"] == "msg-contract-1"
    response_types = [frame["response_type"] for frame in frames]
    assert "thinking" in response_types
    assert "answer" in response_types
    assert "complete" in response_types
    for index, frame in enumerate(frames):
        _assert_frame(frame, f"knowledge-QA frame {index}")


# ── Agent QA (SSE) ───────────────────────────────────────────────────


def test_agent_qa_stream_frames_match_reference(
    chat_client: TestClient,
    fake_chat_service: _FakeChatService,
) -> None:
    """The agent-chat endpoint streams the full reference frame vocabulary."""
    fake_chat_service.qa_events = [
        Event(
            type=EventType.AGENT_THOUGHT,
            session_id="s1",
            data={"content": "thought", "done": False},
        ),
        Event(
            type=EventType.AGENT_TOOL_CALL,
            session_id="s1",
            data={
                "tool_name": "web_search",
                "tool_call_id": "call_1",
                "iteration": 0,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "web_search", "arguments": "{}"},
                    }
                ],
            },
        ),
        Event(
            type=EventType.AGENT_TOOL_RESULT,
            session_id="s1",
            data={"tool_name": "web_search", "tool_call_id": "call_1", "success": True},
        ),
        Event(
            type=EventType.AGENT_REFERENCES,
            session_id="s1",
            data={"references": [_hit_dict("c1")]},
        ),
        Event(
            type=EventType.AGENT_FINAL_ANSWER,
            session_id="s1",
            data={"content": "final", "done": True},
        ),
        Event(
            type=EventType.AGENT_COMPLETE,
            session_id="s1",
            data={"final_answer": "final"},
        ),
    ]
    status, _ = _endpoint_spec("POST", "/agent-chat/{session_id}")
    chunks: list[str] = []
    with chat_client.stream("POST", "/agent-chat/s1", json={"query": "search this"}) as resp:
        assert resp.status_code == status, resp.text
        assert resp.headers["content-type"].startswith("text/event-stream")
        for chunk in resp.iter_text():
            chunks.append(chunk)

    frames = _parse_sse_frames("".join(chunks))
    assert frames, "no SSE frames received"
    response_types = [frame["response_type"] for frame in frames]
    assert response_types[0] == "agent_query"
    assert "thinking" in response_types
    assert "tool_call" in response_types
    assert "tool_result" in response_types
    assert "references" in response_types
    assert "answer" in response_types
    assert "complete" in response_types
    for index, frame in enumerate(frames):
        _assert_frame(frame, f"agent-QA frame {index}")

    search_fields = set(_contract_fields("SearchResult"))
    for frame in frames:
        if frame["response_type"] == "tool_call":
            calls = frame.get("tool_calls")
            assert isinstance(calls, list) and calls
            for call in calls:
                assert set(call) <= _TOOL_CALL_FIELDS
        elif frame["response_type"] == "references":
            refs = frame.get("knowledge_references")
            assert isinstance(refs, list) and refs
            for ref in refs:
                assert set(ref) <= search_fields


# ── Knowledge search (JSON envelope) ─────────────────────────────────


def test_knowledge_search_envelope_matches_reference(
    chat_client: TestClient,
    fake_chat_service: _FakeChatService,
) -> None:
    """The knowledge-search envelope carries reference-shaped hits."""
    fake_chat_service.search_results = [
        SearchResult(
            id="c1",
            content="content-c1",
            knowledge_id="knowledge-c1",
            knowledge_base_id="kb-1",
            knowledge_title="title-c1",
            score=0.9,
            match_type=MatchType.EMBEDDING,
        )
    ]
    resp = chat_client.post(
        "/api/v1/knowledge-search",
        json={"query": "hello", "knowledge_base_ids": ["kb-1"]},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    _assert_keys(body, ["success", "data"], "POST /knowledge-search")
    assert isinstance(body["data"], list) and body["data"]
    row = body["data"][0]
    assert set(row) == set(_contract_fields("SearchResult"))


# ── Agent endpoints (real service + database) ────────────────────────


def test_create_agent_matches_reference(authed_client: TestClient) -> None:
    """POST /agents answers 201 with the reference agent envelope."""
    status, keys = _endpoint_spec("POST", "/api/v1/agents")
    response = authed_client.post("/api/v1/agents", json={"name": "contract-agent-create"})
    assert response.status_code == status, response.text
    body = response.json()
    _assert_keys(body, keys, "POST /agents")
    _assert_payload(body["data"], agent_contracts.Agent, "POST /agents")


def test_get_agent_matches_reference(agent_seed: AgentSeed) -> None:
    """GET /agents/{id} answers 200 with the reference agent envelope."""
    status, keys = _endpoint_spec("GET", "/api/v1/agents/{id}")
    response = agent_seed.client.get(f"/api/v1/agents/{agent_seed.agent_id}")
    assert response.status_code == status, response.text
    body = response.json()
    _assert_keys(body, keys, "GET /agents/{id}")
    _assert_payload(body["data"], agent_contracts.Agent, "GET /agents/{id}")


def test_list_agents_matches_reference(agent_seed: AgentSeed) -> None:
    """GET /agents answers 200 with the reference list envelope."""
    status, keys = _endpoint_spec("GET", "/api/v1/agents")
    response = agent_seed.client.get("/api/v1/agents")
    assert response.status_code == status, response.text
    body = response.json()
    _assert_keys(body, keys, "GET /agents")
    assert isinstance(body["data"], list)
    for row in body["data"]:
        _assert_payload(row, agent_contracts.Agent, "GET /agents row")


def test_update_agent_matches_reference(agent_seed: AgentSeed) -> None:
    """PUT /agents/{id} answers 200 with the reference envelope."""
    status, keys = _endpoint_spec("PUT", "/api/v1/agents/{id}")
    response = agent_seed.client.put(
        f"/api/v1/agents/{agent_seed.agent_id}",
        json={"name": "renamed", "config": {}},
    )
    assert response.status_code == status, response.text
    _assert_keys(response.json(), keys, "PUT /agents/{id}")


def test_delete_agent_matches_reference(agent_seed: AgentSeed) -> None:
    """DELETE /agents/{id} answers 200 with the reference message envelope."""
    status, keys = _endpoint_spec("DELETE", "/api/v1/agents/{id}")
    response = agent_seed.client.delete(f"/api/v1/agents/{agent_seed.agent_id}")
    assert response.status_code == status, response.text
    _assert_keys(response.json(), keys, "DELETE /agents/{id}")


def test_copy_agent_matches_reference(agent_seed: AgentSeed) -> None:
    """POST /agents/{id}/copy answers 201 with the reference envelope."""
    status, keys = _endpoint_spec("POST", "/api/v1/agents/{id}/copy")
    response = agent_seed.client.post(f"/api/v1/agents/{agent_seed.agent_id}/copy")
    assert response.status_code == status, response.text
    body = response.json()
    _assert_keys(body, keys, "POST /agents/{id}/copy")
    _assert_payload(body["data"], agent_contracts.Agent, "POST /agents/{id}/copy")


def test_placeholders_match_reference(authed_client: TestClient) -> None:
    """GET /agents/placeholders answers 200 with the reference group shape."""
    status, keys = _endpoint_spec("GET", "/api/v1/agents/placeholders")
    response = authed_client.get("/api/v1/agents/placeholders")
    assert response.status_code == status, response.text
    body = response.json()
    _assert_keys(body, keys, "GET /agents/placeholders")
    _assert_payload(
        body["data"],
        agent_contracts.AgentPlaceholderGroup,
        "GET /agents/placeholders",
    )


def test_type_presets_match_reference(authed_client: TestClient) -> None:
    """GET /agents/type-presets answers 200 with the reference list shape."""
    status, keys = _endpoint_spec("GET", "/api/v1/agents/type-presets")
    response = authed_client.get("/api/v1/agents/type-presets")
    assert response.status_code == status, response.text
    body = response.json()
    _assert_keys(body, keys, "GET /agents/type-presets")
    assert isinstance(body["data"], list)


def test_suggested_questions_match_reference(agent_seed: AgentSeed) -> None:
    """GET /agents/{id}/suggested-questions answers the reference shape."""
    status, keys = _endpoint_spec("GET", "/api/v1/agents/{id}/suggested-questions")
    response = agent_seed.client.get(f"/api/v1/agents/{agent_seed.agent_id}/suggested-questions")
    assert response.status_code == status, response.text
    body = response.json()
    _assert_keys(body, keys, "GET /agents/{id}/suggested-questions")
    _assert_payload(
        body["data"],
        SuggestedQuestionsData,
        "GET /agents/{id}/suggested-questions",
    )


def test_skills_match_reference(authed_client: TestClient) -> None:
    """GET /skills answers 200 with the reference skills envelope."""
    status, keys = _endpoint_spec("GET", "/api/v1/skills")
    response = authed_client.get("/api/v1/skills")
    assert response.status_code == status, response.text
    body = response.json()
    _assert_keys(body, keys, "GET /skills")
    assert isinstance(body["data"], list)


__all__ = [
    "AgentSeed",
    "_FakeChatService",
    "agent_seed",
    "chat_app",
    "chat_client",
    "fake_chat_service",
]
