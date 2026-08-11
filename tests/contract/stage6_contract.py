"""Session/message web-layer contract tests (live).

Builds the real app via ``create_app`` and asserts the session and
message endpoints' request/response schemas against the reference API
fixtures:

- the session CRUD endpoints (create / get / list / update / delete /
  batch-delete / clear-messages / pin / unpin) answer the documented
  HTTP status and the documented top-level response keys
  (``fixtures/session_responses.json`` -> ``endpoints``);
- the response ``data`` payload is valid against the frozen wire
  contract (``src.core.contracts.sessions`` / the web view models) and
  carries exactly the contract's serialized field set;
- the message endpoints (load / delete / search / chat-history-stats)
  answer the reference status codes and envelopes, and the data
  payload matches the frozen message contract;
- the suggestion endpoints (get / ensure / record-event) answer the
  reference status codes and envelopes.

The tests run against the real database. A failing assertion here means
the live web layer deviates from the reference wire shape — that is a
finding, not a test bug: the web layer is already merged, so deviations
are reported rather than silently fixed.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from secrets import token_hex
from typing import NamedTuple

import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.core.contracts import sessions as contracts
from src.core.tenants.member_service import ROLE_OWNER
from src.db.models.message import Message as MessageRow
from src.db.models.session import Session as SessionRow
from src.web.api.chat.messages.views import (
    ChatHistoryStatsEnvelope,
    DeleteMessageResponse,
    MessageLoadEnvelope,
    SearchMessagesEnvelope,
    SuggestionEnvelope,
)
from src.web.api.chat.sessions.views import (
    DeleteSessionResponse,
    PinSessionEnvelope,
    SessionEnvelope,
    SessionListEnvelope,
)
from tests.contract.test_session_invariants import _model_wire_fields

_REFERENCE = "reference fixture"
_FIXTURE_PATH: Path = Path(__file__).parent / "fixtures" / "session_responses.json"


class SessionSeed(NamedTuple):
    client: TestClient
    tenant_id: int
    session_id: str


class MessageSeed(NamedTuple):
    client: TestClient
    tenant_id: int
    session_id: str
    message_id: str


# ── Fixture helpers ───────────────────────────────────────────────────


def _fixture() -> dict[str, object]:
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


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def session_seed(authed_client: TestClient) -> SessionSeed:
    """A session minted through the real service + database."""
    response = authed_client.post(
        "/sessions",
        json={"title": "contract-session", "description": "contract fixture"},
    )
    assert response.status_code == 201, response.text
    tenant_header = authed_client.headers.get("x-knowledge-tenant-id", "0")
    return SessionSeed(
        authed_client,
        int(tenant_header),
        response.json()["data"]["id"],
    )


@pytest_asyncio.fixture
async def message_seed(
    app: FastAPI,
    admin_user: tuple[str, int],
    _engine,
) -> AsyncIterator[MessageSeed]:
    """A session + one seeded message under a fresh principal.

    Messages carry request_id and content columns; we mint a session via
    the real service so the row's tenant/user ownership matches the
    authed client's headers, then INSERT one message row directly so the
    load endpoint has at least one row to return.

    Skips when the ``messages`` table is absent from the live schema —
    this happens before the messages migration lands on the shared DB.
    """
    from sqlalchemy import text

    user_id, tenant_id = admin_user
    async with _engine.session_factory() as probe:
        exists = (
            await probe.execute(
                text(
                    "SELECT 1 FROM information_schema.tables "
                    "WHERE table_name = 'messages'"
                )
            )
        ).first()
    if exists is None:
        pytest.skip("messages table not yet migrated on the shared DB")
    client = TestClient(app=app)
    client.headers.update(
        {
            "x-knowledge-user-id": user_id,
            "x-knowledge-tenant-id": str(tenant_id),
            "x-knowledge-roles": ROLE_OWNER,
        }
    )
    with client:
        response = client.post(
            "/sessions",
            json={"title": "contract-msg-session", "description": ""},
        )
        assert response.status_code == 201, response.text
        session_id = response.json()["data"]["id"]
        now = datetime.now(UTC)
        message_id = f"msg-contract-{token_hex(8)}"
        # Use the async engine directly for the INSERT.
        async with _engine.session_factory() as session, session.begin():
            row = MessageRow(
                id=message_id,
                request_id=f"req-{token_hex(8)}",
                session_id=session_id,
                role="user",
                content="Hello contract world",
                knowledge_references=[],
                agent_steps=None,
                is_completed=True,
                is_fallback=False,
                agent_duration_ms=0,
                rendered_content="",
                channel="",
                agent_id="",
                agent_tenant_id=0,
                model_id="",
                execution_context={},
                knowledge_id="",
                mentioned_items=[],
                images=[],
                attachments=[],
                created_at=now,
                updated_at=now,
            )
            session.add(row)
        yield MessageSeed(client, tenant_id, session_id, message_id)


@pytest_asyncio.fixture
async def require_messages_table(_engine) -> None:
    """Skip the test when the ``messages`` table is absent on the shared DB."""
    from sqlalchemy import text

    async with _engine.session_factory() as probe:
        exists = (
            await probe.execute(
                text(
                    "SELECT 1 FROM information_schema.tables "
                    "WHERE table_name = 'messages'"
                )
            )
        ).first()
    if exists is None:
        pytest.skip("messages table not yet migrated on the shared DB")


# ── Session endpoints ─────────────────────────────────────────────────


def test_create_session_matches_reference(session_seed: SessionSeed) -> None:
    """POST /sessions answers 201 with the reference session envelope."""
    status, keys = _endpoint_spec("POST", "/sessions")
    response = session_seed.client.post(
        "/sessions",
        json={"title": "create-contract", "description": ""},
    )
    assert response.status_code == status, response.text
    body = response.json()
    _assert_keys(body, keys, "POST /sessions")
    inner = body["data"]
    assert set(inner) == set(_model_wire_fields(contracts.Session)), (
        f"POST /sessions data: wire fields diverge from the contract.\n"
        f"  extra:   {sorted(set(inner) - set(_model_wire_fields(contracts.Session)))}"
    )
    contracts.Session.model_validate(inner)


def test_get_session_matches_reference(session_seed: SessionSeed) -> None:
    """GET /sessions/{id} answers 200 with the reference session envelope."""
    status, keys = _endpoint_spec("GET", "/sessions/{id}")
    response = session_seed.client.get(f"/sessions/{session_seed.session_id}")
    assert response.status_code == status, response.text
    body = response.json()
    _assert_keys(body, keys, "GET /sessions/{id}")
    contracts.Session.model_validate(body["data"])


def test_list_sessions_envelope_matches_reference(session_seed: SessionSeed) -> None:
    """GET /sessions answers 200 with the reference list envelope.

    The reference list endpoint flattens ``data`` to an array plus
    sibling ``total`` / ``page`` / ``page_size`` keys. The current web
    layer wraps the list in ``data.items`` (a ``SessionListResponse``
    paged payload). This assertion is the contract check — a divergence
    here is a reported finding.
    """
    status, keys = _endpoint_spec("GET", "/sessions")
    response = session_seed.client.get("/sessions")
    assert response.status_code == status, response.text
    body = response.json()
    _assert_keys(body, keys, "GET /sessions")


def test_list_sessions_payload_conforms_to_view_model(session_seed: SessionSeed) -> None:
    """The list endpoint's ``data`` payload matches the view-model shape."""
    response = session_seed.client.get("/sessions")
    assert response.status_code == 200, response.text
    body = response.json()
    assert isinstance(body, dict) and isinstance(body.get("data"), dict)
    SessionListEnvelope.model_validate(body)


def test_update_session_matches_reference(session_seed: SessionSeed) -> None:
    """PUT /sessions/{id} answers 200 with the reference envelope."""
    status, keys = _endpoint_spec("PUT", "/sessions/{id}")
    response = session_seed.client.put(
        f"/sessions/{session_seed.session_id}",
        json={"title": "renamed", "description": "updated"},
    )
    assert response.status_code == status, response.text
    body = response.json()
    _assert_keys(body, keys, "PUT /sessions/{id}")
    contracts.Session.model_validate(body["data"])


def test_delete_session_matches_reference(session_seed: SessionSeed) -> None:
    """DELETE /sessions/{id} answers 200 with the reference message envelope."""
    status, keys = _endpoint_spec("DELETE", "/sessions/{id}")
    response = session_seed.client.delete(f"/sessions/{session_seed.session_id}")
    assert response.status_code == status, response.text
    body = response.json()
    _assert_keys(body, keys, "DELETE /sessions/{id}")
    DeleteSessionResponse.model_validate(body)


def test_batch_delete_sessions_matches_reference(session_seed: SessionSeed) -> None:
    """DELETE /sessions/batch answers 200 with the reference envelope."""
    status, keys = _endpoint_spec("DELETE", "/sessions/batch")
    response = session_seed.client.request(
        "DELETE",
        "/sessions/batch",
        json={"ids": [session_seed.session_id], "delete_all": False},
    )
    assert response.status_code == status, response.text
    _assert_keys(response.json(), keys, "DELETE /sessions/batch")


def test_clear_session_messages_matches_reference(
    session_seed: SessionSeed,
    require_messages_table: None,
) -> None:
    """DELETE /sessions/{id}/messages answers 200 with the reference envelope."""
    status, keys = _endpoint_spec("DELETE", "/sessions/{id}/messages")
    response = session_seed.client.delete(f"/sessions/{session_seed.session_id}/messages")
    assert response.status_code == status, response.text
    _assert_keys(response.json(), keys, "DELETE /sessions/{id}/messages")


def test_pin_session_matches_reference(session_seed: SessionSeed) -> None:
    """POST /sessions/{id}/pin answers 200 with the reference pin envelope."""
    status, keys = _endpoint_spec("POST", "/sessions/{id}/pin")
    response = session_seed.client.post(f"/sessions/{session_seed.session_id}/pin")
    assert response.status_code == status, response.text
    body = response.json()
    _assert_keys(body, keys, "POST /sessions/{id}/pin")
    PinSessionEnvelope.model_validate(body)


def test_unpin_session_matches_reference(session_seed: SessionSeed) -> None:
    """DELETE /sessions/{id}/pin answers 200 with the reference pin envelope."""
    session_seed.client.post(f"/sessions/{session_seed.session_id}/pin")
    status, keys = _endpoint_spec("DELETE", "/sessions/{id}/pin")
    response = session_seed.client.delete(f"/sessions/{session_seed.session_id}/pin")
    assert response.status_code == status, response.text
    body = response.json()
    _assert_keys(body, keys, "DELETE /sessions/{id}/pin")
    PinSessionEnvelope.model_validate(body)


def test_session_envelope_matches_contract(session_seed: SessionSeed) -> None:
    """The session object carries exactly the frozen contract's wire fields."""
    response = session_seed.client.get(f"/sessions/{session_seed.session_id}")
    assert response.status_code == 200, response.text
    inner = response.json()["data"]
    assert set(inner) == set(_model_wire_fields(contracts.Session)), (
        f"GET /sessions/{{id}} data: wire fields diverge from the contract.\n"
        f"  missing: {sorted(set(_model_wire_fields(contracts.Session)) - set(inner))}\n"
        f"  extra:   {sorted(set(inner) - set(_model_wire_fields(contracts.Session)))}"
    )


# ── Message endpoints ─────────────────────────────────────────────────


def test_load_messages_matches_reference(message_seed: MessageSeed) -> None:
    """GET /messages/{session_id}/load answers 200 with the reference envelope."""
    status, keys = _endpoint_spec("GET", "/messages/{session_id}/load")
    response = message_seed.client.get(f"/messages/{message_seed.session_id}/load")
    assert response.status_code == status, response.text
    body = response.json()
    _assert_keys(body, keys, "GET /messages/{session_id}/load")
    MessageLoadEnvelope.model_validate(body)
    for row in body["data"]:
        contracts.Message.model_validate(row)


def test_delete_message_matches_reference(message_seed: MessageSeed) -> None:
    """DELETE /messages/{session_id}/{message_id} answers 200 with the reference envelope."""
    status, keys = _endpoint_spec("DELETE", "/messages/{session_id}/{message_id}")
    response = message_seed.client.delete(
        f"/messages/{message_seed.session_id}/{message_seed.message_id}",
    )
    assert response.status_code == status, response.text
    body = response.json()
    _assert_keys(body, keys, "DELETE /messages/{session_id}/{message_id}")
    DeleteMessageResponse.model_validate(body)


def test_search_messages_matches_reference(message_seed: MessageSeed) -> None:
    """POST /messages/search answers 200 with the reference search envelope."""
    status, keys = _endpoint_spec("POST", "/messages/search")
    response = message_seed.client.post(
        "/messages/search",
        json={"query": "Hello", "mode": "keyword", "limit": 20},
    )
    assert response.status_code == status, response.text
    body = response.json()
    _assert_keys(body, keys, "POST /messages/search")
    SearchMessagesEnvelope.model_validate(body)


def test_chat_history_stats_matches_reference(authed_client: TestClient) -> None:
    """GET /messages/chat-history-stats answers 200 with the reference envelope."""
    status, keys = _endpoint_spec("GET", "/messages/chat-history-stats")
    response = authed_client.get("/messages/chat-history-stats")
    assert response.status_code == status, response.text
    body = response.json()
    _assert_keys(body, keys, "GET /messages/chat-history-stats")
    ChatHistoryStatsEnvelope.model_validate(body)


# ── Suggestion endpoints ──────────────────────────────────────────────


def test_get_suggestions_matches_reference(message_seed: MessageSeed) -> None:
    """GET /sessions/{session_id}/messages/{message_id}/suggestions answers 200.

    The endpoint may legitimately 404 when no suggestion set exists for
    the message; the reference documents a 200 envelope shape that is
    exercised when a set is present. This assertion guards the envelope
    on the happy path by seeding a suggestion set directly via the
    service layer when available — otherwise the test asserts the
    envelope contract on a 200 response (the service's default
    ``GetFollowUps`` returns ``None`` which serializes as
    ``{success: true, data: null}``).
    """
    from src.core.chat.messages.suggestion_service import (
        MessageSuggestionSet as SuggestionSetRow,
        MessageSuggestionService,
    )

    # Seed a minimal suggestion set so the GET returns 200 with a non-null data.
    now = datetime.now(UTC)
    set_row = SuggestionSetRow(
        id=f"set-contract-{token_hex(8)}",
        tenant_id=message_seed.tenant_id,
        session_id=message_seed.session_id,
        assistant_message_id=message_seed.message_id,
        agent_id="",
        agent_tenant_id=0,
        placement="follow_up",
        config_hash=f"hash-{token_hex(8)}",
        locale="en-US",
        status="ready",
        allow_regenerate=False,
        suppression_reason="",
        questions=[],
        model_id="",
        prompt_tokens=0,
        completion_tokens=0,
        latency_ms=0,
        error_code="",
        lease_until=None,
        generated_at=None,
        created_at=now,
        updated_at=now,
    )

    class _StubService:
        async def get_follow_ups(
            self, *, session_id: str, assistant_message_id: str
        ) -> SuggestionSetRow | None:
            return set_row

    message_seed.client.app.dependency_overrides[MessageSuggestionService] = (
        lambda: _StubService()
    )
    try:
        status, keys = _endpoint_spec(
            "GET",
            "/sessions/{session_id}/messages/{message_id}/suggestions",
        )
        response = message_seed.client.get(
            f"/sessions/{message_seed.session_id}/messages/{message_seed.message_id}/suggestions"
        )
        assert response.status_code == status, response.text
        body = response.json()
        _assert_keys(body, keys, "GET /sessions/{session_id}/messages/{message_id}/suggestions")
        SuggestionEnvelope.model_validate(body)
    finally:
        message_seed.client.app.dependency_overrides.pop(MessageSuggestionService, None)


def test_record_suggestion_event_matches_reference(message_seed: MessageSeed) -> None:
    """POST /sessions/{session_id}/suggestion-events answers 204 with no body."""
    status, keys = _endpoint_spec("POST", "/sessions/{session_id}/suggestion-events")
    assert not keys, (
        f"reference says 204 returns no body, fixture declares keys={keys}"
    )
    response = message_seed.client.post(
        f"/sessions/{message_seed.session_id}/suggestion-events",
        json={
            "suggestion_set_id": "set-1",
            "question_id": "q-1",
            "event_type": "expose",
        },
    )
    assert response.status_code == status, response.text


__all__ = [
    "MessageSeed",
    "SessionSeed",
    "message_seed",
    "session_seed",
]
