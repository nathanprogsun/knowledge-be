"""Unit tests for the IM-channel HTTP endpoints and slash-command dispatch.

The admin endpoints are tested against a minimal app mounting the
agent-scoped and tenant-wide routers; the callback endpoints mount the
callback router. The service layer and the platform adapter are replaced
with fakes so the tests exercise request handling (auth gates, wire-shape
envelopes, error mapping, command dispatch) without a database or a live
platform connection.

The command registry and its built-in handlers are also unit-tested
directly (registration, parsing, help/clear/stop/info/search execution).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any, ClassVar

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from src.app_context import request_context
from src.common.exception import (
    ApplicationError,
    ExternalServiceError,
    NotFoundError,
    UnauthorizedError,
    ValidationError,
)
from src.core.channels.im.adapter_base import (
    CallbackRequest,
    Context,
    IncomingMessage,
    ReplyMessage,
)
from src.core.channels.im.commands.handlers import (
    SearchService,
    build_default_registry,
)
from src.core.channels.im.commands.registry import (
    Command,
    CommandAction,
    CommandContext,
    CommandRegistry,
    CommandResult,
)
from src.core.channels.im.service.im_channel_service import (
    ChannelCreateRequest,
    ChannelUpdateRequest,
)
from src.core.channels.im.types import IMChannelInfo
from src.web.api.channels.im.router import (
    agents_router,
    callback_router,
    router,
)
from src.web.deps.context import get_tenant_id_dep
from src.web.deps.im_channels import (
    get_im_channel_service,
    get_im_command_registry,
)
from src.web.exception_handler import register_exception_handlers
from src.web.middleware.auth import require_auth

# ── Constants ─────────────────────────────────────────────────────────

_TENANT = 7
_USER = "u-1"
_CHANNEL_ID = "ch-1"
_AGENT_ID = "agent-1"


def _now() -> datetime:
    return datetime(2026, 8, 1, tzinfo=UTC)


# ── Fixture builders ──────────────────────────────────────────────────


def _channel_info(**overrides: Any) -> IMChannelInfo:
    """Build a service projection (no secrets)."""
    base: dict[str, Any] = {
        "id": _CHANNEL_ID,
        "tenant_id": _TENANT,
        "agent_id": _AGENT_ID,
        "platform": "slack",
        "name": "Support Bot",
        "enabled": True,
        "mode": "webhook",
        "output_mode": "stream",
        "knowledge_base_id": "kb-1",
        "bot_identity": "slack:app-1",
        "session_mode": "user",
        "credentials_configured": True,
        "created_at": _now(),
        "updated_at": _now(),
    }
    base.update(overrides)
    return IMChannelInfo(**base)


def _incoming(content: str = "/help", **overrides: Any) -> IncomingMessage:
    """Build a unified incoming message for callback tests."""
    base: dict[str, Any] = {
        "platform": "slack",
        "message_type": "text",
        "user_id": "u-1",
        "user_name": "alice",
        "chat_id": "c-1",
        "chat_type": "direct",
        "content": content,
        "message_id": "m-1",
    }
    base.update(overrides)
    return IncomingMessage(**base)


# ── Service fake ─────────────────────────────────────────────────────


class _FakeIMService:
    """In-memory fake of ``IMChannelService`` (CRUD + callback resolution)."""

    def __init__(self) -> None:
        self.created: list[tuple[int, ChannelCreateRequest]] = []
        self.updated: list[tuple[int, str, ChannelUpdateRequest]] = []
        self.deleted: list[tuple[int, str]] = []
        self.toggled: list[tuple[int, str]] = []
        self.create_result: IMChannelInfo = _channel_info()
        self.update_result: IMChannelInfo | None = None
        self.toggle_result: IMChannelInfo | None = None
        self.list_by_agent: list[IMChannelInfo] = []
        self.list_by_tenant: list[IMChannelInfo] = []
        self.channel: IMChannelInfo = _channel_info()
        self.adapter: Any = None
        self.raise_on: ApplicationError | None = None
        self.raise_on_ensure: ApplicationError | None = None

    async def create_channel(
        self, *, tenant_id: int, request: ChannelCreateRequest
    ) -> IMChannelInfo:
        self.created.append((tenant_id, request))
        self._maybe_raise()
        return self.create_result

    async def list_channels_by_agent(self, *, tenant_id: int, agent_id: str) -> list[IMChannelInfo]:
        self._maybe_raise()
        return self.list_by_agent

    async def list_channels(self, *, tenant_id: int) -> list[IMChannelInfo]:
        self._maybe_raise()
        return self.list_by_tenant

    async def get_channel(self, *, tenant_id: int, channel_id: str) -> IMChannelInfo:
        self._maybe_raise()
        return self.channel

    async def update_channel(
        self, *, tenant_id: int, channel_id: str, request: ChannelUpdateRequest
    ) -> IMChannelInfo:
        self.updated.append((tenant_id, channel_id, request))
        self._maybe_raise()
        return self.update_result or _channel_info()

    async def delete_channel(self, *, tenant_id: int, channel_id: str) -> None:
        self.deleted.append((tenant_id, channel_id))
        self._maybe_raise()

    async def toggle_channel_enabled(self, *, tenant_id: int, channel_id: str) -> IMChannelInfo:
        self.toggled.append((tenant_id, channel_id))
        self._maybe_raise()
        return self.toggle_result or _channel_info(enabled=False)

    async def ensure_channel_adapter(self, channel_id: str) -> tuple[Any, IMChannelInfo]:
        if self.raise_on_ensure is not None:
            raise self.raise_on_ensure
        return self.adapter, self.channel

    def _maybe_raise(self) -> None:
        if self.raise_on is not None:
            raise self.raise_on


# ── Adapter fake ─────────────────────────────────────────────────────


class _FakeAdapter:
    """Fake of the platform adapter surface the callback view drives."""

    def __init__(self) -> None:
        self.url_verification: bool = False
        self.verify_error: UnauthorizedError | None = None
        self.parsed: IncomingMessage | None = None
        self.parse_error: Exception | None = None
        self.sent: list[tuple[IncomingMessage, ReplyMessage]] = []

    def handle_url_verification(self, request: CallbackRequest) -> bool:
        return self.url_verification

    def verify_callback(self, request: CallbackRequest) -> None:
        if self.verify_error is not None:
            raise self.verify_error

    def parse_callback(self, request: CallbackRequest) -> IncomingMessage | None:
        if self.parse_error is not None:
            raise self.parse_error
        return self.parsed

    def send_reply(self, ctx: Context, incoming: IncomingMessage, reply: ReplyMessage) -> None:
        self.sent.append((incoming, reply))


# ── App fixtures ─────────────────────────────────────────────────────


def _make_fake_auth(role: str = "admin"):
    """Stand-in for ``require_auth`` stamping a principal with ``role``."""

    def _fake_auth(request: Request) -> None:
        request.state.tenant_id = str(_TENANT)
        request.state.tenant_role = role
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

    return _fake_auth


@pytest.fixture
def im_service() -> _FakeIMService:
    return _FakeIMService()


@pytest.fixture
def registry() -> CommandRegistry:
    return build_default_registry()


@pytest.fixture
def admin_client(im_service: _FakeIMService) -> TestClient:
    """A minimal app with the admin IM routers and a fake service layer."""

    def _get_service() -> _FakeIMService:
        return im_service

    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(agents_router, prefix="/api/v1")
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[require_auth] = _make_fake_auth("admin")
    app.dependency_overrides[get_im_channel_service] = _get_service
    return TestClient(app)


@pytest.fixture
def callback_client(
    im_service: _FakeIMService,
    registry: CommandRegistry,
) -> TestClient:
    """A minimal app with the callback router and fake service + adapter."""

    def _get_service() -> _FakeIMService:
        return im_service

    def _get_registry() -> CommandRegistry:
        return registry

    im_service.adapter = _FakeAdapter()

    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(callback_router, prefix="/api/v1")
    app.dependency_overrides[get_im_channel_service] = _get_service
    app.dependency_overrides[get_im_command_registry] = _get_registry
    return TestClient(app)


# ── Command registry ─────────────────────────────────────────────────


class _EchoCommand(Command):
    """Test command that echoes its args."""

    def name(self) -> str:
        return "echo"

    def description(self) -> str:
        return "echo the args"

    async def execute(self, ctx: CommandContext, args: list[str]) -> CommandResult:
        return CommandResult(content=" ".join(args))


def test_registry_register_and_parse() -> None:
    reg = CommandRegistry()
    reg.register(_EchoCommand())

    command, args, ok = reg.parse("/echo one two")

    assert ok is True
    assert command is not None
    assert command.name() == "echo"
    assert args == ["one", "two"]


def test_registry_parse_non_command_and_unknown() -> None:
    reg = CommandRegistry()
    reg.register(_EchoCommand())

    assert reg.parse("hello world") == (None, [], False)
    assert reg.parse("/nope") == (None, [], False)
    assert reg.parse("/") == (None, [], False)
    assert reg.parse("") == (None, [], False)


def test_registry_parse_is_case_insensitive() -> None:
    reg = CommandRegistry()
    reg.register(_EchoCommand())

    command, _args, ok = reg.parse("/ECHO x")

    assert ok is True
    assert command is not None
    assert command.name() == "echo"


def test_registry_register_duplicate_raises() -> None:
    reg = CommandRegistry()
    reg.register(_EchoCommand())

    with pytest.raises(ValueError, match="duplicate command registration: echo"):
        reg.register(_EchoCommand())


def test_registry_looks_like_command() -> None:
    reg = CommandRegistry()

    assert reg.looks_like_command("/help") is True
    assert reg.looks_like_command("/api/v2/users") is False
    assert reg.looks_like_command("hello") is False
    assert reg.looks_like_command("/") is False


def test_registry_is_registered_and_all() -> None:
    reg = CommandRegistry()
    reg.register(_EchoCommand())

    assert reg.is_registered("/echo") is True
    assert reg.is_registered("/nope") is False
    assert [c.name() for c in reg.all()] == ["echo"]


# ── Built-in handlers ─────────────────────────────────────────────────


def test_help_lists_all_commands(registry: CommandRegistry) -> None:
    result = asyncio.run(registry.get("help").execute(CommandContext(incoming=_incoming()), []))

    assert "**可用指令**" in result.content
    for name in ("help", "info", "search", "stop", "clear"):
        assert f"`/{name}`" in result.content
    assert result.action is CommandAction.NONE


def test_help_shows_one_command(registry: CommandRegistry) -> None:
    result = asyncio.run(
        registry.get("help").execute(CommandContext(incoming=_incoming()), ["clear"])
    )

    assert "**/clear**" in result.content
    assert "清空对话记忆" in result.content


def test_help_unknown_command(registry: CommandRegistry) -> None:
    result = asyncio.run(
        registry.get("help").execute(CommandContext(incoming=_incoming()), ["nope"])
    )

    assert "未知指令" in result.content


def test_clear_requests_reset_action(registry: CommandRegistry) -> None:
    result = asyncio.run(registry.get("clear").execute(CommandContext(incoming=_incoming()), []))

    assert result.action is CommandAction.CLEAR
    assert "对话已清空" in result.content


def test_stop_requests_stop_action(registry: CommandRegistry) -> None:
    result = asyncio.run(registry.get("stop").execute(CommandContext(incoming=_incoming()), []))

    assert result.action is CommandAction.STOP
    assert "中止" in result.content


def test_info_unbound_agent(registry: CommandRegistry) -> None:
    result = asyncio.run(registry.get("info").execute(CommandContext(incoming=_incoming()), []))

    assert "未绑定智能体" in result.content


def test_info_with_agent_renders_capabilities(registry: CommandRegistry) -> None:
    class _Agent:
        name = "Sales Bot"
        description = "Answers sales questions"

        class config:
            agent_mode = True
            kb_selection_mode = "selected"
            knowledge_base_ids: ClassVar[list[str]] = ["kb-1"]
            skills_selection_mode = "all"
            selected_skills: ClassVar[list[str]] = []
            mcp_selection_mode = "none"
            mcp_service_ids: ClassVar[list[str]] = []
            web_search_enabled = True

    ctx = CommandContext(
        incoming=_incoming(),
        agent_name="Sales Bot",
        custom_agent=_Agent(),  # type: ignore[arg-type]
        channel_output_mode="full",
    )
    result = asyncio.run(registry.get("info").execute(ctx, []))

    assert "**Sales Bot**" in result.content
    assert "Agent模式" in result.content
    assert "知识库" in result.content
    assert "全部启用" in result.content  # skills
    assert "已启用" in result.content  # web search
    assert "完整输出" in result.content  # output mode


def test_search_requires_query(registry: CommandRegistry) -> None:
    result = asyncio.run(registry.get("search").execute(CommandContext(incoming=_incoming()), []))

    assert "请输入搜索内容" in result.content


def test_search_not_wired_returns_hint(registry: CommandRegistry) -> None:
    result = asyncio.run(
        registry.get("search").execute(CommandContext(incoming=_incoming()), ["退款政策"])
    )

    assert "检索服务尚未接入" in result.content


def test_search_formats_results() -> None:
    class _Result:
        content = "退款政策：30 天内可全额退款。"
        knowledge_title = "退款政策文档"
        knowledge_id = "know-1"
        score = 0.73

    class _FakeSearch(SearchService):
        async def search(self, **kwargs: Any) -> list[Any]:
            return [_Result()]

    reg = build_default_registry(search_service=_FakeSearch())
    result = asyncio.run(
        reg.get("search").execute(CommandContext(incoming=_incoming()), ["退款政策"])
    )

    assert "找到 1 条结果" in result.content
    assert "退款政策文档" in result.content
    assert "匹配度：73%" in result.content


# ── Admin: create / list / update / delete / toggle ────────────────────


def test_admin_create_returns_200_with_record(admin_client: TestClient) -> None:
    response = admin_client.post(
        f"/api/v1/agents/{_AGENT_ID}/im-channels",
        json={"platform": "slack", "name": "Support Bot"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    data = payload["data"]
    assert data["id"] == _CHANNEL_ID
    assert data["platform"] == "slack"
    assert data["credentials_configured"] is True
    assert "credentials" not in data


def test_admin_create_passes_request_to_service(admin_client: TestClient) -> None:
    im_service = admin_client.app.dependency_overrides[get_im_channel_service]()

    response = admin_client.post(
        f"/api/v1/agents/{_AGENT_ID}/im-channels",
        json={"platform": "feishu", "credentials": {"app_id": "x"}},
    )

    assert response.status_code == 200
    tenant_id, request = im_service.created[0]
    assert tenant_id == _TENANT
    assert request.agent_id == _AGENT_ID
    assert request.platform == "feishu"
    assert request.credentials == {"app_id": "x"}
    assert request.enabled is True  # omitted means enabled


def test_admin_create_rejects_unsupported_platform(
    admin_client: TestClient,
) -> None:
    im_service = admin_client.app.dependency_overrides[get_im_channel_service]()
    im_service.raise_on = ValidationError(
        code="im.platform_unsupported",
        message="platform must be one of: ...",
    )

    response = admin_client.post(
        f"/api/v1/agents/{_AGENT_ID}/im-channels",
        json={"platform": "discord"},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "im.platform_unsupported"


def test_admin_list_by_agent_returns_envelope(admin_client: TestClient) -> None:
    im_service = admin_client.app.dependency_overrides[get_im_channel_service]()
    im_service.list_by_agent = [
        _channel_info(name="A"),
        _channel_info(name="B"),
    ]

    response = admin_client.get(f"/api/v1/agents/{_AGENT_ID}/im-channels")

    assert response.status_code == 200
    data = response.json()["data"]
    assert [entry["name"] for entry in data] == ["A", "B"]
    assert all("credentials" not in entry for entry in data)


def test_admin_list_all_returns_envelope(admin_client: TestClient) -> None:
    im_service = admin_client.app.dependency_overrides[get_im_channel_service]()
    im_service.list_by_tenant = [_channel_info()]

    response = admin_client.get("/api/v1/im-channels")

    assert response.status_code == 200
    assert len(response.json()["data"]) == 1


def test_admin_update_returns_updated_record(admin_client: TestClient) -> None:
    im_service = admin_client.app.dependency_overrides[get_im_channel_service]()
    im_service.update_result = _channel_info(name="Renamed")

    response = admin_client.put(
        f"/api/v1/im-channels/{_CHANNEL_ID}",
        json={"name": "Renamed"},
    )

    assert response.status_code == 200
    assert response.json()["data"]["name"] == "Renamed"
    _, channel_id, request = im_service.updated[0]
    assert channel_id == _CHANNEL_ID
    assert request.name == "Renamed"
    assert request.enabled is None  # omitted means unchanged


def test_admin_update_rejects_agent_transfer(admin_client: TestClient) -> None:
    im_service = admin_client.app.dependency_overrides[get_im_channel_service]()
    im_service.channel = _channel_info(agent_id=_AGENT_ID)

    response = admin_client.put(
        f"/api/v1/im-channels/{_CHANNEL_ID}",
        json={"agent_id": "agent-2"},
    )

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "im.agent_transfer_unavailable"
    assert im_service.updated == []


def test_admin_update_accepts_unchanged_agent_id(admin_client: TestClient) -> None:
    im_service = admin_client.app.dependency_overrides[get_im_channel_service]()
    im_service.channel = _channel_info(agent_id=_AGENT_ID)
    im_service.update_result = _channel_info(name="Renamed")

    response = admin_client.put(
        f"/api/v1/im-channels/{_CHANNEL_ID}",
        json={"agent_id": _AGENT_ID, "name": "Renamed"},
    )

    assert response.status_code == 200
    assert response.json()["data"]["name"] == "Renamed"


def test_admin_delete_returns_success(admin_client: TestClient) -> None:
    im_service = admin_client.app.dependency_overrides[get_im_channel_service]()

    response = admin_client.delete(f"/api/v1/im-channels/{_CHANNEL_ID}")

    assert response.status_code == 200
    assert response.json() == {"success": True}
    assert im_service.deleted == [(_TENANT, _CHANNEL_ID)]


def test_admin_toggle_returns_updated_record(admin_client: TestClient) -> None:
    im_service = admin_client.app.dependency_overrides[get_im_channel_service]()
    im_service.toggle_result = _channel_info(enabled=False)

    response = admin_client.post(f"/api/v1/im-channels/{_CHANNEL_ID}/toggle")

    assert response.status_code == 200
    assert response.json()["data"]["enabled"] is False
    assert im_service.toggled == [(_TENANT, _CHANNEL_ID)]


def test_admin_missing_tenant_context_is_401(admin_client: TestClient) -> None:
    admin_client.app.dependency_overrides[get_tenant_id_dep] = lambda: 0

    response = admin_client.get("/api/v1/im-channels")

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "im.tenant_context_missing"


def test_admin_viewer_can_list_but_not_mutate() -> None:
    im_service = _FakeIMService()
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(agents_router, prefix="/api/v1")
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[require_auth] = _make_fake_auth("viewer")
    app.dependency_overrides[get_im_channel_service] = lambda: im_service
    client = TestClient(app)

    assert client.get(f"/api/v1/agents/{_AGENT_ID}/im-channels").status_code == 200
    assert (
        client.post(
            f"/api/v1/agents/{_AGENT_ID}/im-channels", json={"platform": "slack"}
        ).status_code
        == 403
    )


# ── Callback: verification / parsing / ack ────────────────────────────


def test_callback_url_verification_ack(callback_client: TestClient) -> None:
    im_service = callback_client.app.dependency_overrides[get_im_channel_service]()
    adapter = _FakeAdapter()
    adapter.url_verification = True
    im_service.adapter = adapter

    response = callback_client.post(f"/api/v1/im/callback/{_CHANNEL_ID}", content=b"{}")

    assert response.status_code == 200
    assert response.json() == {"success": True}
    assert adapter.sent == []


def test_callback_verification_failure_is_403(callback_client: TestClient) -> None:
    im_service = callback_client.app.dependency_overrides[get_im_channel_service]()
    adapter = _FakeAdapter()
    adapter.verify_error = UnauthorizedError(
        code="im.verify_failed",
        message="slack signature verification failed",
    )
    im_service.adapter = adapter

    response = callback_client.post(f"/api/v1/im/callback/{_CHANNEL_ID}", content=b"{}")

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "im.verify_failed"


def test_callback_parse_failure_is_422(callback_client: TestClient) -> None:
    im_service = callback_client.app.dependency_overrides[get_im_channel_service]()
    adapter = _FakeAdapter()
    adapter.parse_error = ValueError("bad payload")
    im_service.adapter = adapter

    response = callback_client.post(f"/api/v1/im/callback/{_CHANNEL_ID}", content=b"{}")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "im.parse_failed"


def test_callback_non_message_event_ack(callback_client: TestClient) -> None:
    im_service = callback_client.app.dependency_overrides[get_im_channel_service]()
    adapter = _FakeAdapter()
    adapter.parsed = None
    im_service.adapter = adapter

    response = callback_client.post(f"/api/v1/im/callback/{_CHANNEL_ID}", content=b"{}")

    assert response.status_code == 200
    assert response.json() == {"success": True}
    assert adapter.sent == []


def test_callback_channel_not_found_is_404(callback_client: TestClient) -> None:
    im_service = callback_client.app.dependency_overrides[get_im_channel_service]()
    im_service.raise_on_ensure = NotFoundError(
        code="im.channel_not_found",
        message="im channel missing not found",
    )

    response = callback_client.post("/api/v1/im/callback/missing", content=b"{}")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "im.channel_not_found"


def test_callback_channel_disabled_is_502(callback_client: TestClient) -> None:
    im_service = callback_client.app.dependency_overrides[get_im_channel_service]()
    im_service.raise_on_ensure = ExternalServiceError(
        code="im.channel_disabled",
        message="channel is disabled",
    )

    response = callback_client.post(f"/api/v1/im/callback/{_CHANNEL_ID}", content=b"{}")

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "im.channel_disabled"


def test_callback_yunzhijia_ack_shape(callback_client: TestClient) -> None:
    im_service = callback_client.app.dependency_overrides[get_im_channel_service]()
    im_service.channel = _channel_info(platform="yunzhijia")
    adapter = _FakeAdapter()
    adapter.parsed = None
    im_service.adapter = adapter

    response = callback_client.post(f"/api/v1/im/callback/{_CHANNEL_ID}", content=b"{}")

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "data": {"type": 2, "content": ""},
    }


# ── Callback: slash-command dispatch ──────────────────────────────────


def test_callback_dispatches_help_command(callback_client: TestClient) -> None:
    im_service = callback_client.app.dependency_overrides[get_im_channel_service]()
    adapter = _FakeAdapter()
    adapter.parsed = _incoming(content="/help")
    im_service.adapter = adapter

    response = callback_client.post(f"/api/v1/im/callback/{_CHANNEL_ID}", content=b"{}")

    assert response.status_code == 200
    assert response.json() == {"success": True}
    assert len(adapter.sent) == 1
    _incoming_msg, reply = adapter.sent[0]
    assert reply.is_final is True
    assert "**可用指令**" in reply.content
    assert "`/help`" in reply.content


def test_callback_unknown_command_gets_help_hint(callback_client: TestClient) -> None:
    im_service = callback_client.app.dependency_overrides[get_im_channel_service]()
    adapter = _FakeAdapter()
    adapter.parsed = _incoming(content="/nope")
    im_service.adapter = adapter

    response = callback_client.post(f"/api/v1/im/callback/{_CHANNEL_ID}", content=b"{}")

    assert response.status_code == 200
    assert len(adapter.sent) == 1
    _incoming_msg, reply = adapter.sent[0]
    assert "未知指令" in reply.content


def test_callback_non_command_message_is_acked_without_reply(
    callback_client: TestClient,
) -> None:
    im_service = callback_client.app.dependency_overrides[get_im_channel_service]()
    adapter = _FakeAdapter()
    adapter.parsed = _incoming(content="hello there")
    im_service.adapter = adapter

    response = callback_client.post(f"/api/v1/im/callback/{_CHANNEL_ID}", content=b"{}")

    assert response.status_code == 200
    assert response.json() == {"success": True}
    assert adapter.sent == []


def test_callback_clear_command_requests_action_and_replies(
    callback_client: TestClient,
) -> None:
    im_service = callback_client.app.dependency_overrides[get_im_channel_service]()
    adapter = _FakeAdapter()
    adapter.parsed = _incoming(content="/clear")
    im_service.adapter = adapter

    response = callback_client.post(f"/api/v1/im/callback/{_CHANNEL_ID}", content=b"{}")

    assert response.status_code == 200
    assert len(adapter.sent) == 1
    _incoming_msg, reply = adapter.sent[0]
    assert "对话已清空" in reply.content


def test_callback_routes_require_no_auth_dependency(
    callback_client: TestClient,
) -> None:
    """The callback router must not depend on ``AuthDep``.

    The dependency override map keys only the service + registry; if any
    callback route had pulled in ``require_auth`` the TestClient would
    fail with a missing dependency override.
    """
    assert require_auth not in callback_client.app.dependency_overrides
    response = callback_client.get(f"/api/v1/im/callback/{_CHANNEL_ID}")
    assert response.status_code == 200


# ── Service seam: ensure_channel_adapter ──────────────────────────────


class _FakeRepo:
    """Fake repository exposing the durable-row lookup."""

    def __init__(self) -> None:
        self.row: Any = None

    async def get_by_id_global(self, channel_id: str) -> Any:
        return self.row


class _FakeSupervisor:
    """Fake supervisor exposing the runtime adapter map."""

    def __init__(self) -> None:
        self.active: tuple[Any, Any, bool] = (None, None, False)
        self.stopped: list[str] = []
        self.started: list[Any] = []

    def get_channel_adapter(self, channel_id: str) -> tuple[Any, Any, bool]:
        return self.active

    def stop_channel(self, channel_id: str) -> None:
        self.stopped.append(channel_id)

    async def start_channel(self, row: Any) -> None:
        self.started.append(row)
        self.active = (object(), row, True)


def _db_row(**overrides: Any) -> Any:
    """Build a raw storage row (with credentials) for the ensure seam."""
    from src.db.models.im_channel import IMChannel

    base: dict[str, Any] = {
        "id": _CHANNEL_ID,
        "tenant_id": _TENANT,
        "agent_id": _AGENT_ID,
        "platform": "slack",
        "name": "Support Bot",
        "enabled": True,
        "mode": "webhook",
        "output_mode": "stream",
        "knowledge_base_id": "kb-1",
        "bot_identity": "slack:app-1",
        "session_mode": "user",
        "credentials": {"bot_token": "x"},
        "created_at": _now(),
        "updated_at": _now(),
    }
    base.update(overrides)
    return IMChannel(**base)


def test_ensure_channel_adapter_starts_when_inactive() -> None:
    from src.core.channels.im.service.im_channel_service import IMChannelService

    repo = _FakeRepo()
    repo.row = _db_row()
    supervisor = _FakeSupervisor()
    service = IMChannelService(channel_repo=repo, supervisor=supervisor)  # type: ignore[arg-type]

    adapter, info = asyncio.run(service.ensure_channel_adapter(_CHANNEL_ID))

    assert supervisor.started == [repo.row]
    assert info.id == _CHANNEL_ID
    assert info.credentials_configured is True
    assert adapter is not None


def test_ensure_channel_adapter_raises_when_missing() -> None:
    from src.core.channels.im.service.im_channel_service import IMChannelService

    repo = _FakeRepo()
    repo.row = None
    supervisor = _FakeSupervisor()
    service = IMChannelService(channel_repo=repo, supervisor=supervisor)  # type: ignore[arg-type]

    with pytest.raises(NotFoundError) as excinfo:
        asyncio.run(service.ensure_channel_adapter(_CHANNEL_ID))

    assert excinfo.value.code == "im.channel_not_found"
    assert supervisor.stopped == [_CHANNEL_ID]


def test_ensure_channel_adapter_raises_when_disabled() -> None:
    from src.core.channels.im.service.im_channel_service import IMChannelService

    repo = _FakeRepo()
    repo.row = _db_row(enabled=False)
    supervisor = _FakeSupervisor()
    service = IMChannelService(channel_repo=repo, supervisor=supervisor)  # type: ignore[arg-type]

    with pytest.raises(ExternalServiceError) as excinfo:
        asyncio.run(service.ensure_channel_adapter(_CHANNEL_ID))

    assert excinfo.value.code == "im.channel_disabled"
    assert supervisor.stopped == [_CHANNEL_ID]


def test_ensure_channel_adapter_raises_when_start_fails() -> None:
    from src.core.channels.im.service.im_channel_service import IMChannelService

    class _FailingSupervisor(_FakeSupervisor):
        async def start_channel(self, row: Any) -> None:
            raise RuntimeError("adapter factory missing")

    repo = _FakeRepo()
    repo.row = _db_row()
    supervisor = _FailingSupervisor()
    service = IMChannelService(channel_repo=repo, supervisor=supervisor)  # type: ignore[arg-type]

    with pytest.raises(ExternalServiceError) as excinfo:
        asyncio.run(service.ensure_channel_adapter(_CHANNEL_ID))

    assert excinfo.value.code == "im.channel_not_available"


__all__: list[Any] = []
