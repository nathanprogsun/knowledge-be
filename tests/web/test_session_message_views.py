"""Web-layer tests for the session / message / suggestion routers.

Exercises the full HTTP path (routing, request validation, response
shapes) over ``TestClient`` against a standalone app that mounts the
three routers. Authentication and the RBAC gate are no-oped, and the
service dependencies are overridden with ``AsyncMock`` doubles, so
the tests never touch a database.

The wire shapes asserted here mirror the upstream handler envelopes:
``{"success": true, "data": ...}`` for reads, ``{"success": true,
"message": "..."}`` for deletes, ``{"success": true, "is_pinned":
bool}`` for the pin toggle.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock
from urllib.parse import quote

from fastapi import FastAPI
from fastapi.params import Depends
from fastapi.testclient import TestClient

from src.app_context import request_context
from src.common.exception import NotFoundError, ValidationError
from src.common.pagination import PaginationResponse
from src.core.chat.messages import (
    ChatHistoryKBStats,
    MessageSearchResult,
)
from src.db.models.message import Message as MessageRow
from src.db.models.message_suggestion import (
    SUGGESTION_STATUS_GENERATING,
    SUGGESTION_STATUS_READY,
    MessageSuggestionSet,
)
from src.db.models.session import Session as SessionRow
from src.web.api.chat.messages.router import (
    router as messages_router,
)
from src.web.api.chat.messages.router import (
    suggestion_router,
)
from src.web.api.chat.sessions.router import router as sessions_router
from src.web.deps import RoleViewerDep
from src.web.deps.chat_sessions import (
    get_message_service,
    get_message_suggestion_service,
    get_session_service,
)
from src.web.exception_handler import register_exception_handlers
from src.web.middleware.auth import require_auth

_NOW = datetime.now(UTC)


def _session_row(
    *,
    session_id: str = "sess-1",
    title: str = "title",
    description: str = "desc",
    tenant_id: int = 7,
    user_id: str = "u-1",
    is_pinned: bool = False,
) -> SessionRow:
    """Build a hydrated session row for service doubles."""
    return SessionRow(
        id=session_id,
        tenant_id=tenant_id,
        title=title,
        description=description,
        user_id=user_id,
        is_pinned=is_pinned,
        pinned_at=_NOW if is_pinned else None,
        created_at=_NOW,
        updated_at=_NOW,
        deleted_at=None,
    )


def _message_row(
    *,
    message_id: str = "msg-1",
    session_id: str = "sess-1",
    role: str = "user",
    content: str = "hello",
) -> MessageRow:
    """Build a hydrated message row for service doubles."""
    return MessageRow(
        id=message_id,
        session_id=session_id,
        request_id="req-1",
        role=role,
        content=content,
        created_at=_NOW,
        updated_at=_NOW,
        deleted_at=None,
    )


def _suggestion_set(
    *,
    set_id: str = "set-1",
    status: str = SUGGESTION_STATUS_READY,
    session_id: str = "sess-1",
) -> MessageSuggestionSet:
    """Build a hydrated suggestion-set row for service doubles."""
    return MessageSuggestionSet(
        id=set_id,
        tenant_id=7,
        session_id=session_id,
        assistant_message_id="msg-1",
        agent_id="",
        agent_tenant_id=0,
        placement="after_answer",
        config_hash="hash",
        locale="en",
        status=status,
        questions=[{"id": "q-1", "text": "follow up?", "category": "clarify"}],
        created_at=_NOW,
        updated_at=_NOW,
    )


def _noop_role_gates(app: FastAPI) -> None:
    """No-op the RBAC role gates so unauthenticated tests can pass."""

    def _noop() -> None:
        return None

    for dep in (RoleViewerDep,):
        for metadata in getattr(dep, "__metadata__", ()):
            if isinstance(metadata, Depends):
                app.dependency_overrides[metadata.dependency] = _noop


def _build_app(**service_overrides: object) -> FastAPI:
    """Build a standalone app with the three routers mounted.

    Authentication is a no-op that also seeds the request-context
    tenant / user so the message-context helpers resolve.
    """

    async def _noop_auth() -> None:
        request_context.set_tenant_id("7")
        request_context.set_user_id("u-1")
        return

    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(sessions_router, prefix="/api/v1")
    app.include_router(messages_router, prefix="/api/v1")
    app.include_router(suggestion_router, prefix="/api/v1")
    app.dependency_overrides[require_auth] = _noop_auth
    _noop_role_gates(app)
    if "session_service" in service_overrides:
        app.dependency_overrides[get_session_service] = lambda: service_overrides["session_service"]
    if "message_service" in service_overrides:
        app.dependency_overrides[get_message_service] = lambda: service_overrides["message_service"]
    if "suggestion_service" in service_overrides:
        app.dependency_overrides[get_message_suggestion_service] = lambda: service_overrides[
            "suggestion_service"
        ]
    return app


def _client(app: FastAPI) -> TestClient:
    """Wrap an app in a ``TestClient`` (with-block managed by the caller)."""
    return TestClient(app)


# ── Session endpoints ─────────────────────────────────────────────────


def test_create_session_returns_201_envelope() -> None:
    fake = AsyncMock()
    fake.tenant_id = 7
    fake.user_id = "u-1"
    row = _session_row(title="new chat")
    fake.create = AsyncMock(return_value=row)
    app = _build_app(session_service=fake)

    with _client(app) as client:
        response = client.post(
            "/api/v1/sessions",
            json={"title": "new chat", "description": "d"},
        )

    assert response.status_code == 201
    body = response.json()
    assert body["success"] is True
    assert body["data"]["id"] == "sess-1"
    assert body["data"]["title"] == "new chat"
    assert body["data"]["tenant_id"] == 7
    assert body["data"]["user_id"] == "u-1"
    assert "created_at" in body["data"]
    created = fake.create.await_args.args[0]
    assert created.tenant_id == 7
    assert created.title == "new chat"


def test_get_session_returns_envelope() -> None:
    fake = AsyncMock()
    fake.get = AsyncMock(return_value=_session_row())
    app = _build_app(session_service=fake)

    with _client(app) as client:
        response = client.get("/api/v1/sessions/sess-1")

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["id"] == "sess-1"
    fake.get.assert_awaited_once_with("sess-1")


def test_get_session_maps_not_found_to_404() -> None:
    fake = AsyncMock()
    fake.get = AsyncMock(side_effect=NotFoundError(code="session.not_found", message="gone"))
    app = _build_app(session_service=fake)

    with _client(app) as client:
        response = client.get("/api/v1/sessions/missing")

    assert response.status_code == 404
    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == "session.not_found"


def test_get_session_rejects_empty_id() -> None:
    fake = AsyncMock()
    app = _build_app(session_service=fake)

    with _client(app) as client:
        response = client.get("/api/v1/sessions/%20")

    assert response.status_code == 422


def test_list_sessions_returns_paged_envelope() -> None:
    fake = AsyncMock()
    fake.list_with_filters = AsyncMock(
        return_value=PaginationResponse(
            total=1,
            page=2,
            page_size=10,
            data=[_session_row(is_pinned=True)],
        )
    )
    app = _build_app(session_service=fake)

    with _client(app) as client:
        response = client.get("/api/v1/sessions?page=2&page_size=10&keyword=AI")

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["total"] == 1
    assert body["data"]["page"] == 2
    assert body["data"]["page_size"] == 10
    assert body["data"]["items"][0]["is_pinned"] is True
    query = fake.list_with_filters.await_args.args[0]
    assert query.keyword == "AI"
    assert query.page == 2


def test_update_session_returns_stored_row() -> None:
    fake = AsyncMock()
    fake.tenant_id = 7
    fake.update = AsyncMock(return_value=_session_row(session_id="sess-1", title="renamed"))
    app = _build_app(session_service=fake)

    with _client(app) as client:
        response = client.put(
            "/api/v1/sessions/sess-1",
            json={"title": "renamed", "description": "new desc"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["data"]["title"] == "renamed"
    updated = fake.update.await_args.args[0]
    assert updated.id == "sess-1"
    assert updated.title == "renamed"


def test_delete_session_returns_message() -> None:
    fake = AsyncMock()
    fake.delete = AsyncMock(return_value=True)
    app = _build_app(session_service=fake)

    with _client(app) as client:
        response = client.delete("/api/v1/sessions/sess-1")

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["message"] == "Session deleted successfully"
    fake.delete.assert_awaited_once_with("sess-1")


def test_delete_session_unknown_returns_404() -> None:
    fake = AsyncMock()
    fake.delete = AsyncMock(return_value=False)
    app = _build_app(session_service=fake)

    with _client(app) as client:
        response = client.delete("/api/v1/sessions/missing")

    assert response.status_code == 404


def test_batch_delete_with_ids() -> None:
    fake = AsyncMock()
    fake.batch_delete = AsyncMock(return_value=2)
    app = _build_app(session_service=fake)

    with _client(app) as client:
        response = client.request(
            "DELETE",
            "/api/v1/sessions/batch",
            json={"ids": ["sess-1", "sess-2"], "delete_all": False},
        )

    assert response.status_code == 200
    assert response.json()["message"] == "Sessions deleted successfully"
    fake.batch_delete.assert_awaited_once()


def test_batch_delete_with_delete_all() -> None:
    fake = AsyncMock()
    fake.delete_all = AsyncMock(return_value=3)
    app = _build_app(session_service=fake)

    with _client(app) as client:
        response = client.request(
            "DELETE",
            "/api/v1/sessions/batch",
            json={"delete_all": True},
        )

    assert response.status_code == 200
    assert response.json()["message"] == "All sessions deleted successfully"
    fake.delete_all.assert_awaited_once()


def test_batch_delete_rejects_empty_ids() -> None:
    fake = AsyncMock()
    app = _build_app(session_service=fake)

    with _client(app) as client:
        response = client.request("DELETE", "/api/v1/sessions/batch", json={"ids": []})

    assert response.status_code == 422


def test_pin_and_unpin_session() -> None:
    fake = AsyncMock()
    fake.set_pinned = AsyncMock(return_value=True)
    app = _build_app(session_service=fake)

    with _client(app) as client:
        pin_response = client.post("/api/v1/sessions/sess-1/pin")
        unpin_response = client.delete("/api/v1/sessions/sess-1/pin")

    assert pin_response.status_code == 200
    assert pin_response.json() == {"success": True, "is_pinned": True}
    assert unpin_response.status_code == 200
    assert unpin_response.json() == {"success": True, "is_pinned": False}
    assert fake.set_pinned.await_args_list[0].args == ("sess-1", True)
    assert fake.set_pinned.await_args_list[1].args == ("sess-1", False)


def test_pin_unknown_session_returns_404() -> None:
    fake = AsyncMock()
    fake.set_pinned = AsyncMock(return_value=False)
    app = _build_app(session_service=fake)

    with _client(app) as client:
        response = client.post("/api/v1/sessions/missing/pin")

    assert response.status_code == 404


def test_clear_session_messages() -> None:
    fake = AsyncMock()
    fake.clear_session_messages = AsyncMock(return_value=3)
    app = _build_app(message_service=fake)

    with _client(app) as client:
        response = client.delete("/api/v1/sessions/sess-1/messages")

    assert response.status_code == 200
    assert response.json()["message"] == "Session messages cleared successfully"
    assert fake.clear_session_messages.await_args.args[1] == "sess-1"


# ── Message endpoints ─────────────────────────────────────────────────


def test_load_messages_returns_recent_by_default() -> None:
    fake = AsyncMock()
    fake.get_recent_messages_by_session = AsyncMock(return_value=[_message_row()])
    app = _build_app(message_service=fake)

    with _client(app) as client:
        response = client.get("/api/v1/messages/sess-1/load")

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"][0]["id"] == "msg-1"
    assert body["data"][0]["role"] == "user"
    fake.get_recent_messages_by_session.assert_awaited_once()
    args = fake.get_recent_messages_by_session.await_args.args
    assert args[1] == "sess-1"
    assert args[2] == 20  # default limit


def test_load_messages_respects_limit_query() -> None:
    fake = AsyncMock()
    fake.get_recent_messages_by_session = AsyncMock(return_value=[])
    app = _build_app(message_service=fake)

    with _client(app) as client:
        response = client.get("/api/v1/messages/sess-1/load?limit=5")

    assert response.status_code == 200
    assert fake.get_recent_messages_by_session.await_args.args[2] == 5


def test_load_messages_with_before_time() -> None:
    fake = AsyncMock()
    fake.list_messages_by_session_before_time = AsyncMock(return_value=[_message_row()])
    app = _build_app(message_service=fake)

    cursor = (datetime.now(UTC) - timedelta(days=1)).isoformat()
    with _client(app) as client:
        response = client.get(f"/api/v1/messages/sess-1/load?before_time={quote(cursor)}")

    assert response.status_code == 200
    args = fake.list_messages_by_session_before_time.await_args.args
    assert args[1] == "sess-1"
    assert isinstance(args[2], datetime)


def test_load_messages_rejects_bad_before_time() -> None:
    fake = AsyncMock()
    app = _build_app(message_service=fake)

    with _client(app) as client:
        response = client.get("/api/v1/messages/sess-1/load?before_time=not-a-time")

    assert response.status_code == 422


def test_delete_message_returns_ack() -> None:
    fake = AsyncMock()
    fake.delete_message = AsyncMock(return_value=True)
    app = _build_app(message_service=fake)

    with _client(app) as client:
        response = client.delete("/api/v1/messages/sess-1/msg-1")

    assert response.status_code == 200
    assert response.json()["message"] == "Message deleted successfully"
    args = fake.delete_message.await_args.args
    assert args[1] == "sess-1"
    assert args[2] == "msg-1"


def test_delete_message_unknown_returns_404() -> None:
    fake = AsyncMock()
    fake.delete_message = AsyncMock(return_value=False)
    app = _build_app(message_service=fake)

    with _client(app) as client:
        response = client.delete("/api/v1/messages/sess-1/missing")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "message.not_found"


def test_search_messages_returns_envelope() -> None:
    fake = AsyncMock()
    result = MessageSearchResult(
        items=(),
        total=0,
    )
    fake.search_messages = AsyncMock(return_value=result)
    app = _build_app(message_service=fake)

    with _client(app) as client:
        response = client.post(
            "/api/v1/messages/search",
            json={"query": "彗星", "mode": "hybrid", "limit": 10},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"] == {"items": [], "total": 0}
    params = fake.search_messages.await_args.args[1]
    assert params.query == "彗星"
    assert params.mode.value == "hybrid"
    assert params.limit == 10


def test_search_messages_rejects_invalid_mode() -> None:
    fake = AsyncMock()
    app = _build_app(message_service=fake)

    with _client(app) as client:
        response = client.post(
            "/api/v1/messages/search",
            json={"query": "q", "mode": "bogus"},
        )

    assert response.status_code == 422


def test_chat_history_stats_returns_envelope() -> None:
    fake = AsyncMock()
    fake.get_chat_history_kb_stats = AsyncMock(
        return_value=ChatHistoryKBStats(enabled=True, indexed_message_count=4)
    )
    app = _build_app(message_service=fake)

    with _client(app) as client:
        response = client.get("/api/v1/messages/chat-history-stats")

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["enabled"] is True
    assert body["data"]["indexed_message_count"] == 4


# ── Suggestion endpoints ──────────────────────────────────────────────


def test_get_suggestions_returns_null_when_absent() -> None:
    fake = AsyncMock()
    fake.get_follow_ups = AsyncMock(return_value=None)
    app = _build_app(suggestion_service=fake)

    with _client(app) as client:
        response = client.get("/api/v1/sessions/sess-1/messages/msg-1/suggestions")

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"] is None


def test_get_suggestions_returns_set() -> None:
    fake = AsyncMock()
    fake.get_follow_ups = AsyncMock(return_value=_suggestion_set())
    app = _build_app(suggestion_service=fake)

    with _client(app) as client:
        response = client.get("/api/v1/sessions/sess-1/messages/msg-1/suggestions")

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["id"] == "set-1"
    assert body["data"]["status"] == SUGGESTION_STATUS_READY
    assert body["data"]["questions"][0]["text"] == "follow up?"


def test_ensure_suggestions_returns_200_for_ready() -> None:
    fake = AsyncMock()
    fake.ensure_follow_ups = AsyncMock(return_value=_suggestion_set())
    app = _build_app(suggestion_service=fake)

    with _client(app) as client:
        response = client.post(
            "/api/v1/sessions/sess-1/messages/msg-1/suggestions",
            json={"regenerate": True},
        )

    assert response.status_code == 200
    assert response.json()["data"]["id"] == "set-1"
    assert fake.ensure_follow_ups.await_args.kwargs["regenerate"] is True


def test_ensure_suggestions_returns_202_for_generating() -> None:
    fake = AsyncMock()
    fake.ensure_follow_ups = AsyncMock(
        return_value=_suggestion_set(status=SUGGESTION_STATUS_GENERATING)
    )
    app = _build_app(suggestion_service=fake)

    with _client(app) as client:
        response = client.post("/api/v1/sessions/sess-1/messages/msg-1/suggestions")

    assert response.status_code == 202


def test_record_suggestion_event_returns_204() -> None:
    fake = AsyncMock()
    fake.record_event = AsyncMock(return_value=None)
    app = _build_app(suggestion_service=fake)

    with _client(app) as client:
        response = client.post(
            "/api/v1/sessions/sess-1/suggestion-events",
            json={
                "suggestion_set_id": "set-1",
                "question_id": "q-1",
                "event_type": "click",
            },
        )

    assert response.status_code == 204
    assert response.content == b""
    assert fake.record_event.await_args.kwargs["event_type"] == "click"


# ── Validation boundary ───────────────────────────────────────────────


def test_search_messages_maps_service_validation_to_422() -> None:
    fake = AsyncMock()
    fake.search_messages = AsyncMock(
        side_effect=ValidationError(
            code="message.search_query_required",
            message="Search query cannot be empty",
        )
    )
    app = _build_app(message_service=fake)

    with _client(app) as client:
        response = client.post("/api/v1/messages/search", json={"query": ""})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "message.search_query_required"
