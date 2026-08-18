"""Unit tests for the embed-channel HTTP endpoints.

The admin endpoints are tested against a minimal app mounting the admin
routers; the public endpoints mount ``public_router``. The service layer
is replaced with fakes so the tests exercise request handling (embed
auth, wire-shape envelopes, error mapping, payload patching) without a
database or a live Redis.

The embed auth dependencies are also unit-tested directly (token
resolution, origin gating, session-handle verification) against fake
services.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from src.app_context import request_context
from src.common.exception import (
    ApplicationError,
    NotFoundError,
    PermissionDeniedError,
    ValidationError,
)
from src.core.channels.embed.service.embed_channel_service import (
    EmbedChannelCreateRequest,
    EmbedChannelUpdateRequest,
)
from src.core.channels.embed.session import (
    CreatedEmbedSession,
    sign_embed_session_handle,
)
from src.core.channels.embed.types import EmbedChannelInfo
from src.core.chat.bus import Event
from src.core.chat.types import EventType
from src.db.models.chunk import Chunk
from src.db.models.embed_channel import EmbedChannel
from src.web.api.channels.embed.router import (
    agents_router,
    public_router,
    router,
)
from src.web.api.channels.embed.views import (
    patch_embed_chat_payload,
    validate_allowed_origins,
)
from src.web.deps.chat_sessions import (
    get_message_suggestion_service,
    get_session_service,
)
from src.web.deps.embed_channels import (
    extract_embed_token,
    get_embed_channel,
    get_embed_channel_service,
    get_embed_chat_service,
    get_embed_chunk_service,
    get_embed_message_context,
    get_embed_message_service,
    get_embed_session_service,
    get_embed_webhook_dispatcher,
    require_embed_session,
)
from src.web.exception_handler import register_exception_handlers
from src.web.middleware.auth import require_auth

# ── Constants ─────────────────────────────────────────────────────────

_TENANT = 7
_USER = "u-1"
_CHANNEL_ID = "ch-1"
_AGENT_ID = "agent-1"
_PUBLISH_TOKEN = "em_secretpublish"
_SESSION_ID = "sess-1"
_MESSAGE_ID = "msg-1"
_ORIGIN = "https://example.com"


def _now() -> datetime:
    return datetime(2026, 8, 1, tzinfo=UTC)


# ── Fixture builders ──────────────────────────────────────────────────


def _channel_row(
    *,
    channel_id: str = _CHANNEL_ID,
    tenant_id: int = _TENANT,
    show_suggested_questions: bool = True,
    allow_web_search: bool = False,
    allow_file_upload: bool = False,
    allowed_origins: list[str] | None = None,
    webhook_secret: str = "",
    webhook_url: str = "",
) -> EmbedChannel:
    """Build a raw storage row the public auth / single-get use."""
    return EmbedChannel(
        id=channel_id,
        tenant_id=tenant_id,
        agent_id=_AGENT_ID,
        name="Support Widget",
        enabled=True,
        publish_token=_PUBLISH_TOKEN,
        allowed_origins=allowed_origins or [_ORIGIN],
        welcome_message="Hi there",
        rate_limit_per_minute=30,
        rate_limit_per_day=10000,
        primary_color="#fff",
        page_title="",
        header_title_mode="channel",
        show_suggested_questions=show_suggested_questions,
        widget_position="bottom-right",
        allow_web_search=allow_web_search,
        allow_file_upload=allow_file_upload,
        default_locale="",
        webhook_url=webhook_url,
        webhook_secret=webhook_secret,
        created_at=_now(),
        updated_at=_now(),
    )


def _channel_info(**overrides: Any) -> EmbedChannelInfo:
    """Build a service projection (no secrets)."""
    base: dict[str, Any] = {
        "id": _CHANNEL_ID,
        "tenant_id": _TENANT,
        "agent_id": _AGENT_ID,
        "name": "Support Widget",
        "enabled": True,
        "allowed_origins": [_ORIGIN],
        "welcome_message": "Hi there",
        "rate_limit_per_minute": 30,
        "rate_limit_per_day": 10000,
        "primary_color": "#fff",
        "page_title": "",
        "header_title_mode": "channel",
        "show_suggested_questions": True,
        "widget_position": "bottom-right",
        "allow_web_search": False,
        "allow_file_upload": False,
        "default_locale": "",
        "webhook_url": "",
        "created_at": _now(),
        "updated_at": _now(),
    }
    base.update(overrides)
    return EmbedChannelInfo(**base)


def _chunk_row(*, tenant_id: int = _TENANT) -> Chunk:
    """Build a chunk row for the embed chunk endpoint."""
    return Chunk(
        id="chunk-1",
        tenant_id=tenant_id,
        knowledge_base_id="kb-1",
        knowledge_id="know-1",
        content="hello",
        chunk_index=0,
        start_at=0,
        end_at=5,
        created_at=_now(),
        updated_at=_now(),
    )


def _session_handle(channel: EmbedChannel, session_id: str = _SESSION_ID) -> str:
    """Return a valid ``X-Embed-Session`` signature for ``channel``."""
    return sign_embed_session_handle(channel, session_id)


# ── Service fakes ─────────────────────────────────────────────────────


class _FakeEmbedService:
    """In-memory fake of ``EmbedChannelService`` (admin CRUD)."""

    def __init__(self) -> None:
        self.created: list[tuple[int, str, EmbedChannelCreateRequest]] = []
        self.updated: list[tuple[int, str, EmbedChannelUpdateRequest]] = []
        self.deleted: list[tuple[int, str]] = []
        self.rotated: list[tuple[int, str]] = []
        self.create_result: tuple[EmbedChannelInfo, str] = (_channel_info(), "em_newtoken")
        self.update_result: EmbedChannelInfo | None = None
        self.rotate_result: tuple[EmbedChannelInfo, str] = (_channel_info(), "em_rotated")
        self.list_by_agent: list[EmbedChannelInfo] = []
        self.list_by_tenant: list[EmbedChannelInfo] = []
        self.owned_row: EmbedChannel | None = None
        self.raise_on: ApplicationError | None = None

    async def create_channel(
        self, *, tenant_id: int, agent_id: str, request: EmbedChannelCreateRequest
    ) -> tuple[EmbedChannelInfo, str]:
        self.created.append((tenant_id, agent_id, request))
        self._maybe_raise()
        return self.create_result

    async def get_owned_channel(self, *, tenant_id: int, channel_id: str) -> EmbedChannel:
        self._maybe_raise()
        if self.owned_row is not None:
            return self.owned_row
        raise NotFoundError(
            code="embed.channel_not_found",
            message=f"embed channel {channel_id} not found",
        )

    async def list_channels_by_agent(self, *, tenant_id: int, agent_id: str) -> list[EmbedChannelInfo]:
        self._maybe_raise()
        return self.list_by_agent

    async def list_channels_by_tenant(self, *, tenant_id: int) -> list[EmbedChannelInfo]:
        self._maybe_raise()
        return self.list_by_tenant

    async def update_channel(
        self, *, tenant_id: int, channel_id: str, request: EmbedChannelUpdateRequest
    ) -> EmbedChannelInfo:
        self.updated.append((tenant_id, channel_id, request))
        self._maybe_raise()
        return self.update_result or _channel_info()

    async def delete_channel(self, *, tenant_id: int, channel_id: str) -> None:
        self.deleted.append((tenant_id, channel_id))
        self._maybe_raise()

    async def rotate_token(self, *, tenant_id: int, channel_id: str) -> tuple[EmbedChannelInfo, str]:
        self.rotated.append((tenant_id, channel_id))
        self._maybe_raise()
        return self.rotate_result

    def _maybe_raise(self) -> None:
        if self.raise_on is not None:
            raise self.raise_on


class _FakeSessionService:
    """In-memory fake of ``EmbedSessionService``."""

    def __init__(self) -> None:
        self.issued: list[str] = []
        self.created_sessions: list[tuple[str, str, str, str]] = []
        self.lookups: list[tuple[str, str]] = []
        self.preview_issued: list[str] = []
        self.raise_on_lookup: ApplicationError | None = None
        self.raise_on_issue: ApplicationError | None = None
        self.token_result: tuple[str, int] = ("ems_abc123", 1800)
        self.channel: EmbedChannel = _channel_row()

    async def issue_preview_session(self, *, channel_id: str) -> tuple[str, int]:
        self.preview_issued.append(channel_id)
        return self.token_result

    async def issue_session_token(self, *, channel_id: str) -> tuple[str, int]:
        self.issued.append(channel_id)
        if self.raise_on_issue is not None:
            raise self.raise_on_issue
        return self.token_result

    async def create_session(
        self,
        *,
        channel_id: str,
        token: str,
        origin: str,
        client_ip: str = "",
        title: str = "",
    ) -> CreatedEmbedSession:
        self.created_sessions.append((channel_id, token, origin, client_ip))
        return CreatedEmbedSession(session_id=_SESSION_ID, handle="sig-handle-1")

    async def lookup_for_embed(self, *, channel_id: str, token: str) -> EmbedChannel:
        self.lookups.append((channel_id, token))
        if self.raise_on_lookup is not None:
            raise self.raise_on_lookup
        return self.channel

    async def assert_origin_allowed(self, origin: str, allowed_origins: list[str]) -> None:
        return None

    async def enforce_rate_limits(self, *, channel: EmbedChannel, client_ip: str) -> None:
        return None


class _FakeWebhookDispatcher:
    """Fake of ``EmbedWebhookDispatcher`` that records dispatch calls."""

    def __init__(self) -> None:
        self.dispatched: list[tuple[str, str, dict[str, str]]] = []

    def dispatch(
        self,
        channel: EmbedChannel,
        *,
        event_type: str,
        session_id: str,
        payload: dict[str, str],
    ) -> asyncio.Task[None] | None:
        self.dispatched.append((event_type, session_id, payload))
        return None


class _FakeChunkService:
    """Fake of ``ChunkService`` (id-only lookup)."""

    def __init__(self) -> None:
        self.chunk: Chunk | None = None
        self.raise_on: ApplicationError | None = None

    async def get_chunk_by_id_only(self, *, id: str) -> Chunk:
        if self.raise_on is not None:
            raise self.raise_on
        if self.chunk is None:
            raise NotFoundError(
                code="chunk.not_found",
                message=f"chunk {id} not found",
            )
        return self.chunk


class _FakeMessageService:
    """Fake of ``MessageServiceImpl`` (history load)."""

    def __init__(self) -> None:
        self.rows: list[Any] = []

    async def get_recent_messages_by_session(
        self, ctx: Any, session_id: str, limit: int
    ) -> list[Any]:
        return self.rows

    async def list_messages_by_session_before_time(
        self, ctx: Any, session_id: str, before_time: datetime, limit: int
    ) -> list[Any]:
        return self.rows


class _FakeSuggestionService:
    """Fake of ``MessageSuggestionService``."""

    def __init__(self) -> None:
        self.recorded: list[tuple[str, str, str, str]] = []
        self.follow_ups_result: Any = None
        self.ensure_result: Any = None

    async def get_follow_ups(self, *, session_id: str, assistant_message_id: str) -> Any:
        return self.follow_ups_result

    async def ensure_follow_ups(
        self, *, session_id: str, assistant_message_id: str, regenerate: bool
    ) -> Any:
        return self.ensure_result

    async def record_event(
        self, *, session_id: str, suggestion_set_id: str, question_id: str, event_type: str
    ) -> None:
        self.recorded.append((session_id, suggestion_set_id, question_id, event_type))


class _FakeChatService:
    """Fake of ``ChatService`` that captures patched QA requests."""

    request_id = "req-embed-1"

    def __init__(self) -> None:
        self.knowledge_calls: list[tuple[str, Any]] = []
        self.agent_calls: list[tuple[str, Any]] = []

    async def stream_knowledge_qa(self, *, session_id: str, request: Any) -> Any:
        self.knowledge_calls.append((session_id, request))
        return _event_stream(session_id)

    async def stream_agent_qa(self, *, session_id: str, request: Any) -> Any:
        self.agent_calls.append((session_id, request))
        return _event_stream(session_id)


async def _event_stream(session_id: str) -> Any:
    """Yield one wire-mapped event so the SSE response is non-empty."""
    yield Event(
        type=EventType.AGENT_COMPLETE,
        session_id=session_id,
        data={"content": "hi", "done": True},
    )


# ── App fixtures ──────────────────────────────────────────────────────


def _fake_admin_auth(request: Request) -> None:
    """Stand-in for ``require_auth`` stamping an admin principal."""
    request.state.tenant_id = str(_TENANT)
    request.state.tenant_role = "admin"
    request.state.user_info = {
        "id": _USER,
        "username": "alice",
        "email": "alice@example.com",
        "is_active": "1",
        "can_access_all_tenants": "0",
        "is_system_admin": "0",
    }
    request.state.is_system_admin = False
    request.state.api_key_scope = None
    request_context.set_tenant_id(str(_TENANT))
    request_context.set_user_id(_USER)


@pytest.fixture
def embed_service() -> _FakeEmbedService:
    return _FakeEmbedService()


@pytest.fixture
def session_service() -> _FakeSessionService:
    return _FakeSessionService()


@pytest.fixture
def channel() -> EmbedChannel:
    return _channel_row()


def _holder() -> dict[str, Any]:
    """Per-test holder for the fake services (mirrors the org tests)."""
    return {}


@pytest.fixture
def admin_client(embed_service: _FakeEmbedService, session_service: _FakeSessionService) -> TestClient:
    """A minimal app with the admin embed routers and a fake service layer."""

    def _get_embed_service() -> _FakeEmbedService:
        return embed_service

    def _get_session_service() -> _FakeSessionService:
        return session_service

    def _get_chat_session_service() -> Any:
        class _Result:
            total = 4

        class _Fake:
            async def list_with_filters(self, query: Any) -> _Result:
                return _Result()

        return _Fake()

    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(agents_router, prefix="/api/v1")
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[require_auth] = _fake_admin_auth
    app.dependency_overrides[get_embed_channel_service] = _get_embed_service
    app.dependency_overrides[get_embed_session_service] = _get_session_service
    app.dependency_overrides[get_session_service] = _get_chat_session_service
    return TestClient(app)


class _MessageCtx:
    """Minimal pipeline ``Context`` carrying the embed channel's tenant."""

    tenant_id: int

    def __init__(self, *, tenant_id: int) -> None:
        self.tenant_id = tenant_id


@pytest.fixture
def public_client(
    channel: EmbedChannel,
    session_service: _FakeSessionService,
) -> TestClient:
    """A minimal app with the public embed router and fake services.

    ``get_embed_channel`` is overridden to return the fixture channel so
    the token / origin / rate-limit gates are bypassed for the view
    tests; the auth gates themselves are unit-tested separately.
    """
    webhook_dispatcher = _FakeWebhookDispatcher()
    chunk_service = _FakeChunkService()
    message_service = _FakeMessageService()
    suggestion_service = _FakeSuggestionService()
    chat_service = _FakeChatService()

    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(public_router, prefix="/api/v1")
    app.dependency_overrides[get_embed_channel] = lambda: channel
    app.dependency_overrides[get_embed_session_service] = lambda: session_service
    app.dependency_overrides[get_embed_webhook_dispatcher] = lambda: webhook_dispatcher
    app.dependency_overrides[get_embed_chunk_service] = lambda: chunk_service
    app.dependency_overrides[get_embed_message_service] = lambda: message_service
    app.dependency_overrides[get_embed_message_context] = lambda: _MessageCtx(tenant_id=_TENANT)
    app.dependency_overrides[get_embed_chat_service] = lambda: chat_service
    app.dependency_overrides[get_message_suggestion_service] = lambda: suggestion_service

    app.state.embed_fakes = {
        "webhook_dispatcher": webhook_dispatcher,
        "chunk_service": chunk_service,
        "message_service": message_service,
        "suggestion_service": suggestion_service,
        "chat_service": chat_service,
        "session_service": session_service,
        "channel": channel,
    }
    return TestClient(app)


def _fakes(client: TestClient) -> dict[str, Any]:
    """Return the per-test fake services stashed on the app."""
    return client.app.state.embed_fakes  # type: ignore[attr-defined]


# ── Admin: create / list / get / update / delete / rotate ─────────────


def test_admin_create_returns_201_with_publish_token(admin_client: TestClient) -> None:
    response = admin_client.post(
        f"/api/v1/agents/{_AGENT_ID}/embed-channels",
        json={
            "name": "Widget",
            "allowed_origins": [_ORIGIN],
            "rate_limit_per_minute": 60,
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["success"] is True
    data = payload["data"]
    assert data["id"] == _CHANNEL_ID
    assert data["publish_token"] == "em_newtoken"
    assert data["allowed_origins"] == [_ORIGIN]
    assert data["has_webhook_secret"] is False


def test_admin_create_rejects_empty_origin_allowlist(admin_client: TestClient) -> None:
    response = admin_client.post(
        f"/api/v1/agents/{_AGENT_ID}/embed-channels",
        json={"name": "Widget", "allowed_origins": []},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "embed.origin_required"


def test_admin_create_rejects_wildcard_in_production(
    admin_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GIN_MODE", "release")
    response = admin_client.post(
        f"/api/v1/agents/{_AGENT_ID}/embed-channels",
        json={"name": "Widget", "allowed_origins": ["*"]},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "embed.origin_wildcard_prohibited"


def test_admin_list_by_agent_returns_envelope(admin_client: TestClient) -> None:
    embed_service = admin_client.app.dependency_overrides[get_embed_channel_service]()
    embed_service.list_by_agent = [
        _channel_info(name="A"),
        _channel_info(name="B"),
    ]

    response = admin_client.get(f"/api/v1/agents/{_AGENT_ID}/embed-channels")

    assert response.status_code == 200
    data = response.json()["data"]
    assert [entry["name"] for entry in data] == ["A", "B"]
    assert all("publish_token" not in entry or entry["publish_token"] is None for entry in data)


def test_admin_list_all_returns_envelope(admin_client: TestClient) -> None:
    embed_service = admin_client.app.dependency_overrides[get_embed_channel_service]()
    embed_service.list_by_tenant = [_channel_info()]

    response = admin_client.get("/api/v1/embed-channels")

    assert response.status_code == 200
    assert len(response.json()["data"]) == 1


def test_admin_get_returns_publish_token_and_webhook_flag(
    admin_client: TestClient,
) -> None:
    embed_service = admin_client.app.dependency_overrides[get_embed_channel_service]()
    embed_service.owned_row = _channel_row(webhook_secret="secret-1")

    response = admin_client.get(f"/api/v1/embed-channels/{_CHANNEL_ID}")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["publish_token"] == _PUBLISH_TOKEN
    assert data["has_webhook_secret"] is True


def test_admin_update_returns_updated_record(admin_client: TestClient) -> None:
    embed_service = admin_client.app.dependency_overrides[get_embed_channel_service]()
    embed_service.update_result = _channel_info(name="Renamed")

    response = admin_client.put(
        f"/api/v1/embed-channels/{_CHANNEL_ID}",
        json={"name": "Renamed"},
    )

    assert response.status_code == 200
    assert response.json()["data"]["name"] == "Renamed"
    _, _, request = embed_service.updated[0]
    assert request.name == "Renamed"
    assert request.allowed_origins is None  # omitted means unchanged


def test_admin_update_validates_origin_allowlist_when_changed(
    admin_client: TestClient,
) -> None:
    embed_service = admin_client.app.dependency_overrides[get_embed_channel_service]()

    response = admin_client.put(
        f"/api/v1/embed-channels/{_CHANNEL_ID}",
        json={"name": "Renamed", "allowed_origins": []},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "embed.origin_required"
    assert embed_service.updated == []


def test_admin_delete_returns_success(admin_client: TestClient) -> None:
    embed_service = admin_client.app.dependency_overrides[get_embed_channel_service]()

    response = admin_client.delete(f"/api/v1/embed-channels/{_CHANNEL_ID}")

    assert response.status_code == 200
    assert response.json() == {"success": True}
    assert embed_service.deleted == [(_TENANT, _CHANNEL_ID)]


def test_admin_rotate_token_returns_new_token(admin_client: TestClient) -> None:
    embed_service = admin_client.app.dependency_overrides[get_embed_channel_service]()

    response = admin_client.post(f"/api/v1/embed-channels/{_CHANNEL_ID}/rotate-token")

    assert response.status_code == 200
    assert response.json()["data"]["publish_token"] == "em_rotated"
    assert embed_service.rotated == [(_TENANT, _CHANNEL_ID)]


def test_admin_preview_session_returns_session_token(admin_client: TestClient) -> None:
    session_service = admin_client.app.dependency_overrides[get_embed_session_service]()

    response = admin_client.post(f"/api/v1/embed-channels/{_CHANNEL_ID}/preview-session")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["session_token"] == "ems_abc123"
    assert data["expires_in"] == 1800
    assert session_service.preview_issued == [_CHANNEL_ID]


def test_admin_stats_returns_session_count(admin_client: TestClient) -> None:
    embed_service = admin_client.app.dependency_overrides[get_embed_channel_service]()
    embed_service.owned_row = _channel_row()

    response = admin_client.get(f"/api/v1/embed-channels/{_CHANNEL_ID}/stats")

    assert response.status_code == 200
    assert response.json()["data"]["session_count"] == 4


def test_admin_missing_tenant_context_is_401(admin_client: TestClient) -> None:
    from src.web.deps.context import get_tenant_id_dep

    admin_client.app.dependency_overrides[get_tenant_id_dep] = lambda: 0

    response = admin_client.get("/api/v1/embed-channels")

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "embed.tenant_context_missing"


# ── Public: config / exchange / sessions / suggested-questions ────────


def test_public_config_returns_public_config(public_client: TestClient) -> None:
    response = public_client.get(f"/api/v1/embed/{_CHANNEL_ID}/config")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["channel_id"] == _CHANNEL_ID
    assert data["name"] == "Support Widget"
    assert data["allowed_origins"] == [_ORIGIN]
    assert data["display_title"] == "Support Widget"
    assert "publish_token" not in data
    assert data["show_suggested_questions"] is True


def test_public_exchange_mints_session_token(public_client: TestClient) -> None:
    response = public_client.post(
        f"/api/v1/embed/{_CHANNEL_ID}/exchange",
        headers={"Authorization": f"Embed {_PUBLISH_TOKEN}"},
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["session_token"] == "ems_abc123"
    assert data["expires_in"] == 1800


def test_public_exchange_rejects_session_token(public_client: TestClient) -> None:
    response = public_client.post(
        f"/api/v1/embed/{_CHANNEL_ID}/exchange",
        headers={"Authorization": "Embed ems_sessiontoken"},
    )

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "embed.publish_token_required"


def test_public_create_session_returns_id_and_sig(public_client: TestClient) -> None:
    session_service = _fakes(public_client)["session_service"]

    response = public_client.post(
        f"/api/v1/embed/{_CHANNEL_ID}/sessions",
        headers={"Authorization": f"Embed {_PUBLISH_TOKEN}", "Origin": _ORIGIN},
    )

    assert response.status_code == 201
    data = response.json()["data"]
    assert data["id"] == _SESSION_ID
    assert data["sig"] == "sig-handle-1"
    assert session_service.created_sessions == [
        (_CHANNEL_ID, _PUBLISH_TOKEN, _ORIGIN, "testclient")
    ]


def test_public_suggested_questions_disabled_returns_empty(
    public_client: TestClient,
) -> None:
    public_client.app.dependency_overrides[get_embed_channel] = lambda: _channel_row(
        show_suggested_questions=False
    )

    response = public_client.get(f"/api/v1/embed/{_CHANNEL_ID}/suggested-questions")

    assert response.status_code == 200
    assert response.json()["data"]["questions"] == []


def test_public_suggested_questions_enabled_returns_empty_until_wired(
    public_client: TestClient,
) -> None:
    response = public_client.get(
        f"/api/v1/embed/{_CHANNEL_ID}/suggested-questions", params={"limit": "100"}
    )

    assert response.status_code == 200
    assert response.json()["data"]["questions"] == []


# ── Public: chunks ────────────────────────────────────────────────────


def test_public_chunk_returns_chunk(public_client: TestClient) -> None:
    _fakes(public_client)["chunk_service"].chunk = _chunk_row()

    response = public_client.get(f"/api/v1/embed/{_CHANNEL_ID}/chunks/chunk-1")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["id"] == "chunk-1"
    assert data["content"] == "hello"


def test_public_chunk_forbids_cross_tenant(public_client: TestClient) -> None:
    _fakes(public_client)["chunk_service"].chunk = _chunk_row(tenant_id=_TENANT + 1)

    response = public_client.get(f"/api/v1/embed/{_CHANNEL_ID}/chunks/chunk-1")

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "embed.chunk_forbidden"


def test_public_chunk_missing_is_404(public_client: TestClient) -> None:
    response = public_client.get(f"/api/v1/embed/{_CHANNEL_ID}/chunks/missing")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "embed.chunk_not_found"


# ── Public: message history ───────────────────────────────────────────


def test_public_load_messages_returns_envelope(public_client: TestClient) -> None:
    _fakes(public_client)["message_service"].rows = []

    response = public_client.get(
        f"/api/v1/embed/{_CHANNEL_ID}/messages/{_SESSION_ID}/load",
        headers={"X-Embed-Session": _session_handle(_channel_row())},
    )

    assert response.status_code == 200
    assert response.json() == {"success": True, "data": []}


def test_public_load_messages_requires_valid_session_handle(
    public_client: TestClient,
) -> None:
    response = public_client.get(
        f"/api/v1/embed/{_CHANNEL_ID}/messages/{_SESSION_ID}/load",
        headers={"X-Embed-Session": "bogus"},
    )

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "embed.session_signature_invalid"


# ── Public: webhook relay ─────────────────────────────────────────────


def test_public_webhook_relay_ack(public_client: TestClient) -> None:
    response = public_client.post(
        f"/api/v1/embed/{_CHANNEL_ID}/sessions/{_SESSION_ID}/events",
        headers={"X-Embed-Session": _session_handle(_channel_row())},
        json={
            "type": "message_sent",
            "query": "what is x",
            "content": "answering",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"success": True}
    event_type, session_id, payload = _fakes(public_client)["webhook_dispatcher"].dispatched[0]
    assert event_type == "message_sent"
    assert session_id == _SESSION_ID
    assert payload == {"query": "what is x", "content": "answering"}


def test_public_webhook_relay_rejects_unknown_type(public_client: TestClient) -> None:
    response = public_client.post(
        f"/api/v1/embed/{_CHANNEL_ID}/sessions/{_SESSION_ID}/events",
        headers={"X-Embed-Session": _session_handle(_channel_row())},
        json={"type": "message_edited"},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "embed.unsupported_event_type"


# ── Public: suggestions ───────────────────────────────────────────────


def test_public_get_suggestions_suppressed_when_disabled(
    public_client: TestClient,
) -> None:
    public_client.app.dependency_overrides[get_embed_channel] = lambda: _channel_row(
        show_suggested_questions=False
    )

    response = public_client.get(
        f"/api/v1/embed/{_CHANNEL_ID}/sessions/{_SESSION_ID}/messages/{_MESSAGE_ID}/suggestions",
        headers={"X-Embed-Session": _session_handle(_channel_row())},
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["status"] == "suppressed"
    assert data["suppression_reason"] == "channel_disabled"


def test_public_record_suggestion_event_returns_204(public_client: TestClient) -> None:
    response = public_client.post(
        f"/api/v1/embed/{_CHANNEL_ID}/sessions/{_SESSION_ID}/suggestion-events",
        headers={"X-Embed-Session": _session_handle(_channel_row())},
        json={
            "suggestion_set_id": "ss-1",
            "question_id": "q-1",
            "event_type": "click",
        },
    )

    assert response.status_code == 204


# ── Public: chat (SSE) ────────────────────────────────────────────────


def test_public_knowledge_chat_patches_payload(public_client: TestClient) -> None:
    chat_service = _fakes(public_client)["chat_service"]

    response = public_client.post(
        f"/api/v1/embed/{_CHANNEL_ID}/knowledge-chat/{_SESSION_ID}",
        headers={"X-Embed-Session": _session_handle(_channel_row())},
        json={"query": "hello", "web_search_enabled": True, "images": [{"data": "x"}]},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "hi" in response.text
    _, request = chat_service.knowledge_calls[0]
    assert request.query == "hello"
    # The channel forbids file uploads, so the images payload is stripped.
    assert request.images is None
    assert request.agent_enabled is False
    assert request.knowledge_base_ids == []


def test_public_agent_chat_sets_agent_mode(public_client: TestClient) -> None:
    chat_service = _fakes(public_client)["chat_service"]

    response = public_client.post(
        f"/api/v1/embed/{_CHANNEL_ID}/agent-chat/{_SESSION_ID}",
        headers={"X-Embed-Session": _session_handle(_channel_row())},
        json={"query": "hi"},
    )

    assert response.status_code == 200
    _, request = chat_service.agent_calls[0]
    assert request.agent_enabled is True


def test_public_chat_rejects_invalid_json(public_client: TestClient) -> None:
    response = public_client.post(
        f"/api/v1/embed/{_CHANNEL_ID}/knowledge-chat/{_SESSION_ID}",
        headers={"X-Embed-Session": _session_handle(_channel_row())},
        content=b"{not json",
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "embed.invalid_chat_json"


# ── Public: not-yet-wired capabilities ────────────────────────────────


def test_public_stop_session_capability_unavailable(public_client: TestClient) -> None:
    response = public_client.post(
        f"/api/v1/embed/{_CHANNEL_ID}/sessions/{_SESSION_ID}/stop",
        headers={"X-Embed-Session": _session_handle(_channel_row())},
    )

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "embed.capability_unavailable"


def test_public_mcp_oauth_and_files_unavailable(public_client: TestClient) -> None:
    headers = {"X-Embed-Session": _session_handle(_channel_row())}
    assert (
        public_client.post(
            f"/api/v1/embed/{_CHANNEL_ID}/sessions/{_SESSION_ID}/mcp-oauth-resolutions/p1",
            headers=headers,
        ).status_code
        == 502
    )
    assert (
        public_client.post(
            f"/api/v1/embed/{_CHANNEL_ID}/sessions/{_SESSION_ID}/tool-approvals/p1",
            headers=headers,
        ).status_code
        == 502
    )
    assert (
        public_client.get(f"/api/v1/embed/{_CHANNEL_ID}/files").status_code == 502
    )


# ── Public routes work without user auth ──────────────────────────────


def test_public_routes_require_no_auth_dependency(public_client: TestClient) -> None:
    """The public router must not depend on ``AuthDep``.

    The dependency override map keys only the embed-auth dependency; if
    any route had pulled in ``require_auth`` the TestClient would fail
    with a missing dependency override.
    """
    assert require_auth not in public_client.app.dependency_overrides
    response = public_client.get(f"/api/v1/embed/{_CHANNEL_ID}/config")
    assert response.status_code == 200


# ── Embed auth dependencies ───────────────────────────────────────────


def test_extract_embed_token_parses_header() -> None:
    assert extract_embed_token("Embed em_secret") == "em_secret"
    assert extract_embed_token("Bearer token") == ""
    assert extract_embed_token("Embed") == ""
    assert extract_embed_token(None) == ""


def test_get_embed_channel_resolves_token_and_stashes_channel(
    session_service: _FakeSessionService,
) -> None:
    channel = _channel_row()
    session_service.channel = channel
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/",
            "headers": [(b"origin", _ORIGIN.encode())],
            "client": ("1.2.3.4", 1234),
            "server": ("test", 80),
        }
    )

    resolved = asyncio.run(
        get_embed_channel(
            request=request,
            channel_id=_CHANNEL_ID,
            session_service=session_service,
            authorization=f"Embed {_PUBLISH_TOKEN}",
        )
    )

    assert resolved == channel
    assert session_service.lookups == [(_CHANNEL_ID, _PUBLISH_TOKEN)]
    assert request.state.embed_channel == channel
    assert request.state.embed_tenant_id == str(_TENANT)


def test_require_embed_session_accepts_valid_handle() -> None:
    channel = _channel_row()

    async def _run() -> None:
        request = Request(
            {"type": "http", "method": "GET", "headers": {}, "path": "/"}
        )
        await require_embed_session(
            request=request,
            session_id=_SESSION_ID,
            channel=channel,
            x_embed_session=_session_handle(channel),
        )

    asyncio.run(_run())


def test_require_embed_session_rejects_invalid_handle() -> None:
    channel = _channel_row()

    async def _run() -> None:
        request = Request(
            {"type": "http", "method": "GET", "headers": {}, "path": "/"}
        )
        with pytest.raises(PermissionDeniedError) as excinfo:
            await require_embed_session(
                request=request,
                session_id=_SESSION_ID,
                channel=channel,
                x_embed_session="forged",
            )
        assert excinfo.value.code == "embed.session_signature_invalid"

    asyncio.run(_run())


# ── Payload patching ──────────────────────────────────────────────────


def test_patch_embed_chat_payload_forces_channel_fields() -> None:
    channel = _channel_row()
    patched = patch_embed_chat_payload(
        b'{"query": "hi", "web_search_enabled": true}',
        channel,
        agent_mode=False,
    )

    assert patched["agent_id"] == _AGENT_ID
    assert patched["knowledge_base_ids"] == []
    assert patched["mcp_service_ids"] == []
    assert patched["agent_enabled"] is False
    # The channel does not allow web search, so the visitor toggle is off.
    assert patched["web_search_enabled"] is False
    assert patched["query"] == "hi"


def test_patch_embed_chat_payload_keeps_visitor_search_when_allowed() -> None:
    channel = _channel_row(allow_web_search=True)
    patched = patch_embed_chat_payload(
        b'{"web_search_enabled": true}',
        channel,
        agent_mode=True,
    )

    assert patched["web_search_enabled"] is True
    assert patched["agent_enabled"] is True


def test_patch_embed_chat_payload_strips_uploads_when_forbidden() -> None:
    channel = _channel_row()
    patched = patch_embed_chat_payload(
        b'{"query": "x", "images": [{"data": "a"}], "attachment_ids": ["1"]}',
        channel,
        agent_mode=False,
    )

    assert "images" not in patched
    assert "attachment_ids" not in patched


def test_patch_embed_chat_payload_rejects_invalid_json() -> None:
    with pytest.raises(ValidationError) as excinfo:
        patch_embed_chat_payload(b"nope", _channel_row(), agent_mode=False)
    assert excinfo.value.code == "embed.invalid_chat_json"


# ── Origin allowlist validation ───────────────────────────────────────


def test_validate_allowed_origins_accepts_https_and_subdomain() -> None:
    validate_allowed_origins(["https://example.com", "*.example.org"])


def test_validate_allowed_origins_rejects_empty() -> None:
    with pytest.raises(ValidationError) as excinfo:
        validate_allowed_origins([])
    assert excinfo.value.code == "embed.origin_required"


def test_validate_allowed_origins_rejects_non_http_scheme() -> None:
    with pytest.raises(ValidationError) as excinfo:
        validate_allowed_origins(["ftp://example.com"])
    assert excinfo.value.code == "embed.origin_invalid"


def test_validate_allowed_origins_rejects_wildcard_in_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GIN_MODE", "release")
    with pytest.raises(ValidationError) as excinfo:
        validate_allowed_origins(["*"])
    assert excinfo.value.code == "embed.origin_wildcard_prohibited"


__all__: list[Any] = []
