"""Unit tests for ``IMChannelService`` and ``IMSupervisor``.

AAA-pattern, ``AsyncMock(spec=…)`` repositories, and a real
``IMSupervisor`` for the runtime / health-check paths (its only
state is the active-channel map and the adapter registry, neither
of which needs a database).

The service depends on a supervisor; the supervisor depends on
adapter factories. Tests stub each of those with concrete fakes
when they want behaviour, plain ``AsyncMock``s when they only
need the call shape.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import cast
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.common.exception import ConflictError, NotFoundError, ValidationError
from src.common.json import JsonObject
from src.core.channels.im.adapter_base import (
    MESSAGE_TYPE_TEXT,
    CallbackRequest,
    Context,
    IMAdapter,
    IncomingMessage,
    ReplyMessage,
    StreamSender,
)
from src.core.channels.im.service.im_channel_service import (
    ChannelCreateRequest,
    ChannelUpdateRequest,
    IMChannelService,
    build_im_channel_service,
    compute_bot_identity,
)
from src.core.channels.im.supervisor import (
    AdapterFactory,
    IMSupervisor,
    SupervisorConfig,
    get_default_supervisor,
    run_supervised,
)
from src.db.dao.im_channel_repository import IMChannelRepository
from src.db.models.im_channel import IMChannel
from tests.util.service_test import ServiceTest

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_TENANT_ID = 7
_AGENT_ID = "agent-001"


# ── Test fakes ───────────────────────────────────────────────────────


class _FakeAdapter(IMAdapter):
    """Minimal concrete adapter with recording for assertions.

    ``connect_returns_stop`` decides whether ``connect`` returns the
    default stop callable (``True``) or ``None`` (treated as a
    "no-cleanup-needed" connection). ``connected`` toggles the
    ``is_connected`` probe so health-check tests can flip it.
    """

    def __init__(
        self,
        *,
        platform: str,
        connect_returns_stop: bool = True,
        connected: bool = True,
    ) -> None:
        self._platform = platform
        self._connect_returns_stop = connect_returns_stop
        self._connected = connected
        self.connect_calls = 0
        self.stop_calls = 0
        self.disconnect_calls = 0

    def platform(self) -> str:
        return self._platform

    def verify_callback(self, request: CallbackRequest) -> None:
        return None

    def parse_callback(self, request: CallbackRequest) -> IncomingMessage | None:
        return None

    def send_reply(self, ctx: Context, incoming: IncomingMessage, reply: ReplyMessage) -> None:
        return None

    def handle_url_verification(self, request: CallbackRequest) -> bool:
        return False

    async def connect(self, ctx: Context):
        self.connect_calls += 1
        if not self._connect_returns_stop:
            return None

        def _stop() -> None:
            self.stop_calls += 1

        return _stop

    def disconnect(self) -> None:
        self.disconnect_calls += 1

    def is_connected(self) -> bool:
        return self._connected


def _make_repo() -> tuple[AsyncMock, dict[str, IMChannel]]:
    """Repository mock with closure-captured state.

    Mirrors the stateful_insert / lookup_by helper pair in
    ``tests/util/service_test.py`` but keeps the soft-delete /
    enabled-toggle / by-bot-identity lookups local so each test
    sees only the methods it exercises.
    """
    repo = AsyncMock(spec=IMChannelRepository)
    store: dict[str, IMChannel] = {}

    def _live() -> dict[str, IMChannel]:
        # Soft-delete filter mirrors the SQL ``deleted_at is null`` on
        # every repository read path.
        return {cid: r for cid, r in store.items() if r.deleted_at is None}

    async def _create(row: IMChannel) -> IMChannel:
        stored = row.model_copy()
        store[stored.id] = stored
        return stored

    async def _update(row: IMChannel) -> IMChannel:
        stored = row.model_copy()
        store[stored.id] = stored
        return stored

    async def _get_by_id(tenant_id: int, channel_id: str) -> IMChannel | None:
        row = _live().get(channel_id)
        if row is None or row.tenant_id != tenant_id:
            return None
        return row.model_copy()

    async def _get_by_id_global(channel_id: str) -> IMChannel | None:
        row = _live().get(channel_id)
        return row.model_copy() if row is not None else None

    async def _list_by_agent(tenant_id: int, agent_id: str) -> list[IMChannel]:
        return [
            r.model_copy()
            for r in _live().values()
            if r.tenant_id == tenant_id and r.agent_id == agent_id
        ]

    async def _list_by_tenant(tenant_id: int) -> list[IMChannel]:
        return [r.model_copy() for r in _live().values() if r.tenant_id == tenant_id]

    async def _list_enabled() -> list[IMChannel]:
        return [r.model_copy() for r in _live().values() if r.enabled]

    async def _find_by_bot_identity(bot_identity: str, *, exclude_id: str = "") -> IMChannel | None:
        for row in _live().values():
            if row.bot_identity != bot_identity:
                continue
            if exclude_id and row.id == exclude_id:
                continue
            return row.model_copy()
        return None

    async def _soft_delete(*, channel_id: str, tenant_id: int, now: datetime) -> bool:
        row = store.get(channel_id)
        if row is None or row.tenant_id != tenant_id:
            return False
        store[channel_id] = row.model_copy(update={"deleted_at": now})
        return True

    async def _soft_delete_by_agent(*, agent_id: str, tenant_id: int, now: datetime) -> int:
        to_delete = [
            cid for cid, r in store.items() if r.tenant_id == tenant_id and r.agent_id == agent_id
        ]
        for cid in to_delete:
            row = store[cid]
            store[cid] = row.model_copy(update={"deleted_at": now})
        return len(to_delete)

    async def _toggle_enabled(
        *, channel_id: str, tenant_id: int, now: datetime
    ) -> IMChannel | None:
        row = _live().get(channel_id)
        if row is None or row.tenant_id != tenant_id:
            return None
        toggled = row.model_copy(update={"enabled": not row.enabled, "updated_at": now})
        store[channel_id] = toggled
        return toggled.model_copy()

    repo.create.side_effect = _create
    repo.update.side_effect = _update
    repo.get_by_id.side_effect = _get_by_id
    repo.get_by_id_global.side_effect = _get_by_id_global
    repo.list_by_agent.side_effect = _list_by_agent
    repo.list_by_tenant.side_effect = _list_by_tenant
    repo.list_enabled.side_effect = _list_enabled
    repo.find_by_bot_identity.side_effect = _find_by_bot_identity
    repo.soft_delete.side_effect = _soft_delete
    repo.soft_delete_by_agent.side_effect = _soft_delete_by_agent
    repo.toggle_enabled.side_effect = _toggle_enabled
    return repo, store


def _seed_channel(
    *,
    channel_id: str,
    platform: str = "feishu",
    tenant_id: int = _TENANT_ID,
    agent_id: str = _AGENT_ID,
    enabled: bool = True,
    mode: str = "websocket",
    credentials: JsonObject | None = None,
) -> IMChannel:
    """Build a live channel row for in-memory seeding."""
    return IMChannel(
        id=channel_id,
        tenant_id=tenant_id,
        agent_id=agent_id,
        platform=platform,
        name=f"channel-{channel_id}",
        enabled=enabled,
        mode=mode,
        output_mode="stream",
        knowledge_base_id="",
        bot_identity="",
        session_mode="user",
        credentials=credentials if credentials is not None else {},
        created_at=_NOW,
        updated_at=_NOW,
    )


@pytest.fixture
def repo_and_store() -> tuple[AsyncMock, dict[str, IMChannel]]:
    return _make_repo()


@pytest.fixture
def repo(repo_and_store: tuple[AsyncMock, dict[str, IMChannel]]) -> AsyncMock:
    return repo_and_store[0]


@pytest.fixture
def store(repo_and_store: tuple[AsyncMock, dict[str, IMChannel]]) -> dict[str, IMChannel]:
    return repo_and_store[1]


@pytest.fixture
def supervisor() -> IMSupervisor:
    """Fresh supervisor per test — the runtime map should never leak."""
    return IMSupervisor(health_interval=0.01)


@pytest.fixture
def service(repo: AsyncMock, supervisor: IMSupervisor) -> IMChannelService:
    return IMChannelService(channel_repo=repo, supervisor=supervisor)


# ── compute_bot_identity ─────────────────────────────────────────────


class TestComputeBotIdentity(ServiceTest):
    def test_empty_credentials_returns_empty(self) -> None:
        assert compute_bot_identity("feishu", "websocket", {}) == ""

    def test_feishu_uses_app_id(self) -> None:
        assert compute_bot_identity("feishu", "websocket", {"app_id": "cli_1"}) == "feishu:cli_1"

    def test_lark_uses_app_id(self) -> None:
        assert compute_bot_identity("lark", "websocket", {"app_id": "cli_2"}) == "lark:cli_2"

    def test_wecom_websocket_uses_bot_id(self) -> None:
        assert compute_bot_identity("wecom", "websocket", {"bot_id": "bot-x"}) == "wecom:ws:bot-x"

    def test_wecom_webhook_uses_corp_and_agent(self) -> None:
        assert (
            compute_bot_identity(
                "wecom",
                "webhook",
                {"corp_id": "c", "corp_agent_id": "a"},
            )
            == "wecom:wh:c:a"
        )

    def test_telegram_uses_bot_token_prefix(self) -> None:
        assert (
            compute_bot_identity("telegram", "webhook", {"bot_token": "12345:abcdef"})
            == "telegram:12345"
        )

    def test_dingtalk_uses_client_id(self) -> None:
        assert compute_bot_identity("dingtalk", "webhook", {"client_id": "abc"}) == "dingtalk:abc"

    def test_mattermost_uses_outgoing_token(self) -> None:
        assert (
            compute_bot_identity("mattermost", "webhook", {"outgoing_token": "tok"})
            == "mattermost:wh:tok"
        )

    def test_wechat_uses_ilink_bot_id(self) -> None:
        assert compute_bot_identity("wechat", "longpoll", {"ilink_bot_id": "w"}) == "wechat:w"

    def test_qqbot_uses_app_id(self) -> None:
        assert compute_bot_identity("qqbot", "websocket", {"app_id": "qq-app"}) == "qqbot:qq-app"

    def test_yunzhijia_hashes_yzjtoken(self) -> None:
        identity = compute_bot_identity(
            "yunzhijia",
            "webhook",
            {"send_msg_url": "https://x.com/y?yzjtoken=secret"},
        )
        assert identity.startswith("yunzhijia:")
        # Same token → same identity (sha256 over the literal token).
        again = compute_bot_identity(
            "yunzhijia",
            "webhook",
            {"send_msg_url": "https://other.com/z?yzjtoken=secret"},
        )
        assert identity == again

    def test_unknown_platform_returns_empty(self) -> None:
        assert compute_bot_identity("other", "websocket", {"app_id": "x"}) == ""


# ── Create ───────────────────────────────────────────────────────────


class TestCreateChannel(ServiceTest):
    async def test_persists_with_defaults_and_returns_info(
        self, service: IMChannelService, store: dict[str, IMChannel]
    ) -> None:
        info = await service.create_channel(
            tenant_id=_TENANT_ID,
            request=ChannelCreateRequest(
                agent_id=_AGENT_ID,
                platform="feishu",
                name="  bot  ",
                credentials={"app_id": "cli_1"},
            ),
        )
        # Trimmed name; mode + output defaulted; bot_identity derived.
        assert info.name == "bot"
        assert info.mode == "websocket"
        assert info.output_mode == "stream"
        assert info.bot_identity == "feishu:cli_1"
        assert info.agent_id == _AGENT_ID
        assert info.platform == "feishu"
        # Persisted row reflects the same defaults.
        row = store[info.id]
        assert row.mode == "websocket"
        assert row.output_mode == "stream"
        assert row.bot_identity == "feishu:cli_1"
        assert row.session_mode == "user"

    async def test_mattermost_defaults_to_webhook(self, service: IMChannelService) -> None:
        info = await service.create_channel(
            tenant_id=_TENANT_ID,
            request=ChannelCreateRequest(
                agent_id=_AGENT_ID,
                platform="mattermost",
                name="m",
                credentials={"outgoing_token": "tok"},
            ),
        )
        assert info.mode == "webhook"
        assert info.bot_identity == "mattermost:wh:tok"

    async def test_yunzhijia_defaults_to_webhook(self, service: IMChannelService) -> None:
        info = await service.create_channel(
            tenant_id=_TENANT_ID,
            request=ChannelCreateRequest(
                agent_id=_AGENT_ID,
                platform="yunzhijia",
                name="y",
                credentials={"send_msg_url": "https://x.com/y?yzjtoken=secret"},
            ),
        )
        assert info.mode == "webhook"
        assert info.bot_identity.startswith("yunzhijia:")

    async def test_wechat_pins_longpoll_and_full(self, service: IMChannelService) -> None:
        info = await service.create_channel(
            tenant_id=_TENANT_ID,
            request=ChannelCreateRequest(
                agent_id=_AGENT_ID,
                platform="wechat",
                name="w",
                credentials={"ilink_bot_id": "bot"},
            ),
        )
        assert info.mode == "longpoll"
        assert info.output_mode == "full"

    async def test_duplicate_bot_raises_conflict(
        self, service: IMChannelService, store: dict[str, IMChannel]
    ) -> None:
        await service.create_channel(
            tenant_id=_TENANT_ID,
            request=ChannelCreateRequest(
                agent_id=_AGENT_ID,
                platform="feishu",
                name="first",
                credentials={"app_id": "cli_1"},
            ),
        )
        with pytest.raises(ConflictError) as excinfo:
            await service.create_channel(
                tenant_id=_TENANT_ID,
                request=ChannelCreateRequest(
                    agent_id=_AGENT_ID,
                    platform="feishu",
                    name="second",
                    credentials={"app_id": "cli_1"},
                ),
            )
        assert excinfo.value.code == "im.duplicate_bot"

    async def test_blank_bot_identity_skips_duplicate_check(
        self, service: IMChannelService
    ) -> None:
        # Two channels with no derivable bot identity → both succeed.
        info_a = await service.create_channel(
            tenant_id=_TENANT_ID,
            request=ChannelCreateRequest(agent_id=_AGENT_ID, platform="feishu", name="a"),
        )
        info_b = await service.create_channel(
            tenant_id=_TENANT_ID,
            request=ChannelCreateRequest(agent_id=_AGENT_ID, platform="feishu", name="b"),
        )
        assert info_a.id != info_b.id
        assert info_a.bot_identity == ""
        assert info_b.bot_identity == ""

    async def test_rejects_unknown_platform(self, service: IMChannelService) -> None:
        with pytest.raises(ValidationError) as excinfo:
            await service.create_channel(
                tenant_id=_TENANT_ID,
                request=ChannelCreateRequest(agent_id=_AGENT_ID, platform="other", name="x"),
            )
        assert excinfo.value.code == "im.platform_unsupported"

    async def test_rejects_invalid_session_mode(self, service: IMChannelService) -> None:
        with pytest.raises(ValidationError) as excinfo:
            await service.create_channel(
                tenant_id=_TENANT_ID,
                request=ChannelCreateRequest(
                    agent_id=_AGENT_ID,
                    platform="feishu",
                    name="x",
                    session_mode="shared",
                ),
            )
        assert excinfo.value.code == "im.session_mode_invalid"

    async def test_rejects_missing_tenant(self, service: IMChannelService) -> None:
        with pytest.raises(ValidationError) as excinfo:
            await service.create_channel(
                tenant_id=0,
                request=ChannelCreateRequest(agent_id=_AGENT_ID, platform="feishu", name="x"),
            )
        assert excinfo.value.code == "im.tenant_required"

    async def test_rejects_missing_agent(self, service: IMChannelService) -> None:
        with pytest.raises(ValidationError) as excinfo:
            await service.create_channel(
                tenant_id=_TENANT_ID,
                request=ChannelCreateRequest(agent_id="   ", platform="feishu", name="x"),
            )
        assert excinfo.value.code == "im.agent_id_required"


# ── Reads ────────────────────────────────────────────────────────────


class TestReads(ServiceTest):
    async def test_get_returns_info_when_present(
        self, service: IMChannelService, store: dict[str, IMChannel]
    ) -> None:
        created = await service.create_channel(
            tenant_id=_TENANT_ID,
            request=ChannelCreateRequest(agent_id=_AGENT_ID, platform="feishu", name="x"),
        )
        info = await service.get_channel(tenant_id=_TENANT_ID, channel_id=created.id)
        assert info.id == created.id

    async def test_get_is_tenant_scoped(
        self, service: IMChannelService, store: dict[str, IMChannel]
    ) -> None:
        created = await service.create_channel(
            tenant_id=_TENANT_ID,
            request=ChannelCreateRequest(agent_id=_AGENT_ID, platform="feishu", name="x"),
        )
        with pytest.raises(NotFoundError):
            await service.get_channel(tenant_id=_TENANT_ID + 1, channel_id=created.id)

    async def test_get_unknown_raises_not_found(self, service: IMChannelService) -> None:
        with pytest.raises(NotFoundError):
            await service.get_channel(tenant_id=_TENANT_ID, channel_id="missing")

    async def test_list_returns_only_that_tenants_channels(self, service: IMChannelService) -> None:
        await service.create_channel(
            tenant_id=_TENANT_ID,
            request=ChannelCreateRequest(agent_id=_AGENT_ID, platform="feishu", name="mine"),
        )
        await service.create_channel(
            tenant_id=_TENANT_ID + 1,
            request=ChannelCreateRequest(agent_id=_AGENT_ID, platform="feishu", name="theirs"),
        )
        infos = await service.list_channels(tenant_id=_TENANT_ID)
        assert [i.name for i in infos] == ["mine"]

    async def test_list_by_agent_filters(self, service: IMChannelService) -> None:
        await service.create_channel(
            tenant_id=_TENANT_ID,
            request=ChannelCreateRequest(agent_id=_AGENT_ID, platform="feishu", name="x"),
        )
        await service.create_channel(
            tenant_id=_TENANT_ID,
            request=ChannelCreateRequest(agent_id="other", platform="feishu", name="y"),
        )
        infos = await service.list_channels_by_agent(tenant_id=_TENANT_ID, agent_id=_AGENT_ID)
        assert [i.name for i in infos] == ["x"]


# ── Update ───────────────────────────────────────────────────────────


class TestUpdate(ServiceTest):
    async def test_applies_supplied_fields(
        self, service: IMChannelService, store: dict[str, IMChannel]
    ) -> None:
        created = await service.create_channel(
            tenant_id=_TENANT_ID,
            request=ChannelCreateRequest(agent_id=_AGENT_ID, platform="feishu", name="old"),
        )
        updated = await service.update_channel(
            tenant_id=_TENANT_ID,
            channel_id=created.id,
            request=ChannelUpdateRequest(
                name="new", knowledge_base_id="kb-1", session_mode="thread"
            ),
        )
        assert updated.name == "new"
        assert updated.knowledge_base_id == "kb-1"
        assert updated.session_mode == "thread"
        # Untouched fields preserved
        assert store[created.id].platform == "feishu"

    async def test_duplicate_bot_conflict_excludes_self(self, service: IMChannelService) -> None:
        # Same bot identity on the same row must not trip the guard.
        created = await service.create_channel(
            tenant_id=_TENANT_ID,
            request=ChannelCreateRequest(
                agent_id=_AGENT_ID,
                platform="feishu",
                name="x",
                credentials={"app_id": "cli_1"},
            ),
        )
        # Update with the same bot identity → no conflict.
        updated = await service.update_channel(
            tenant_id=_TENANT_ID,
            channel_id=created.id,
            request=ChannelUpdateRequest(name="renamed"),
        )
        assert updated.bot_identity == "feishu:cli_1"

    async def test_duplicate_bot_conflict_blocks_other(self, service: IMChannelService) -> None:
        a = await service.create_channel(
            tenant_id=_TENANT_ID,
            request=ChannelCreateRequest(
                agent_id=_AGENT_ID,
                platform="feishu",
                name="a",
                credentials={"app_id": "cli_1"},
            ),
        )
        await service.create_channel(
            tenant_id=_TENANT_ID,
            request=ChannelCreateRequest(
                agent_id=_AGENT_ID,
                platform="feishu",
                name="b",
                credentials={"app_id": "cli_2"},
            ),
        )
        with pytest.raises(ConflictError):
            await service.update_channel(
                tenant_id=_TENANT_ID,
                channel_id=a.id,
                request=ChannelUpdateRequest(credentials={"app_id": "cli_2"}, name="a"),
            )

    async def test_unknown_channel_raises_not_found(self, service: IMChannelService) -> None:
        with pytest.raises(NotFoundError):
            await service.update_channel(
                tenant_id=_TENANT_ID,
                channel_id="missing",
                request=ChannelUpdateRequest(name="x"),
            )


# ── Delete ───────────────────────────────────────────────────────────


class TestDelete(ServiceTest):
    async def test_soft_deletes_and_stops(
        self, service: IMChannelService, store: dict[str, IMChannel]
    ) -> None:
        created = await service.create_channel(
            tenant_id=_TENANT_ID,
            request=ChannelCreateRequest(agent_id=_AGENT_ID, platform="feishu", name="x"),
        )
        await service.delete_channel(tenant_id=_TENANT_ID, channel_id=created.id)
        assert store[created.id].deleted_at is not None
        with pytest.raises(NotFoundError):
            await service.get_channel(tenant_id=_TENANT_ID, channel_id=created.id)

    async def test_unknown_channel_raises_not_found(self, service: IMChannelService) -> None:
        with pytest.raises(NotFoundError):
            await service.delete_channel(tenant_id=_TENANT_ID, channel_id="missing")

    async def test_delete_by_agent_drains_all(
        self, service: IMChannelService, store: dict[str, IMChannel]
    ) -> None:
        for n in ("a", "b"):
            await service.create_channel(
                tenant_id=_TENANT_ID,
                request=ChannelCreateRequest(agent_id=_AGENT_ID, platform="feishu", name=n),
            )
        await service.create_channel(
            tenant_id=_TENANT_ID,
            request=ChannelCreateRequest(agent_id="other", platform="feishu", name="c"),
        )
        removed = await service.delete_channels_by_agent(tenant_id=_TENANT_ID, agent_id=_AGENT_ID)
        assert removed == 2
        for row in store.values():
            if row.agent_id == _AGENT_ID:
                assert row.deleted_at is not None
            else:
                assert row.deleted_at is None


# ── Toggle ───────────────────────────────────────────────────────────


class TestToggle(ServiceTest):
    async def test_flips_enabled_flag(
        self, service: IMChannelService, store: dict[str, IMChannel]
    ) -> None:
        created = await service.create_channel(
            tenant_id=_TENANT_ID,
            request=ChannelCreateRequest(agent_id=_AGENT_ID, platform="feishu", name="x"),
        )
        first = await service.toggle_channel_enabled(tenant_id=_TENANT_ID, channel_id=created.id)
        assert first.enabled is False
        second = await service.toggle_channel_enabled(tenant_id=_TENANT_ID, channel_id=created.id)
        assert second.enabled is True

    async def test_unknown_channel_raises_not_found(self, service: IMChannelService) -> None:
        with pytest.raises(NotFoundError):
            await service.toggle_channel_enabled(tenant_id=_TENANT_ID, channel_id="missing")


# ── Adapter registry ─────────────────────────────────────────────────


class TestAdapterRegistry(ServiceTest):
    def test_register_and_get_round_trip(self, service: IMChannelService) -> None:
        calls: list[IMChannel] = []

        def _factory(channel: IMChannel) -> IMAdapter:
            calls.append(channel)
            return _FakeAdapter(platform="feishu")

        service.register_adapter_factory("feishu", cast(AdapterFactory, _factory))
        retrieved = service.get_adapter_factory("feishu")
        assert retrieved is cast(AdapterFactory, _factory)

    def test_get_unknown_platform_raises_not_found(self, service: IMChannelService) -> None:
        with pytest.raises(NotFoundError) as excinfo:
            service.get_adapter_factory("other")
        assert excinfo.value.code == "im.adapter_not_found"

    def test_registered_platforms_sorted(self, service: IMChannelService) -> None:
        def _factory(channel: IMChannel) -> IMAdapter:
            return _FakeAdapter(platform="x")

        service.register_adapter_factory("wecom", cast(AdapterFactory, _factory))
        service.register_adapter_factory("feishu", cast(AdapterFactory, _factory))
        assert service.registered_platforms() == ["feishu", "wecom"]

    def test_re_register_replaces(self, service: IMChannelService) -> None:
        def _first(channel: IMChannel) -> IMAdapter:
            return _FakeAdapter(platform="feishu")

        def _second(channel: IMChannel) -> IMAdapter:
            return _FakeAdapter(platform="feishu")

        service.register_adapter_factory("feishu", cast(AdapterFactory, _first))
        service.register_adapter_factory("feishu", cast(AdapterFactory, _second))
        assert service.get_adapter_factory("feishu") is cast(AdapterFactory, _second)


# ── Supervisor lifecycle ─────────────────────────────────────────────


class TestSupervisorLifecycle(ServiceTest):
    async def test_start_creates_adapter_and_connects(
        self, repo: AsyncMock, supervisor: IMSupervisor, store: dict[str, IMChannel]
    ) -> None:
        service = IMChannelService(channel_repo=repo, supervisor=supervisor)
        row = _seed_channel(channel_id="ch-1")
        store["ch-1"] = row

        adapter_calls: list[IMChannel] = []

        def _factory(channel: IMChannel) -> IMAdapter:
            adapter_calls.append(channel)
            return _FakeAdapter(platform="feishu")

        service.register_adapter_factory("feishu", cast(AdapterFactory, _factory))

        await service.start_channel(tenant_id=_TENANT_ID, channel_id="ch-1")

        assert adapter_calls == [row]
        adapter, _, running = service.get_channel_adapter("ch-1")
        assert running is True
        assert adapter is not None
        assert adapter.platform() == "feishu"
        assert cast("_FakeAdapter", adapter).connect_calls == 1
        assert service.active_channel_count() == 1

    async def test_start_unknown_factory_raises_not_found(
        self, repo: AsyncMock, supervisor: IMSupervisor, store: dict[str, IMChannel]
    ) -> None:
        service = IMChannelService(channel_repo=repo, supervisor=supervisor)
        store["ch-1"] = _seed_channel(channel_id="ch-1", platform="wecom")
        with pytest.raises(NotFoundError) as excinfo:
            await service.start_channel(tenant_id=_TENANT_ID, channel_id="ch-1")
        assert excinfo.value.code == "im.adapter_not_found"
        assert service.active_channel_count() == 0

    async def test_re_start_replaces_active(
        self, repo: AsyncMock, supervisor: IMSupervisor, store: dict[str, IMChannel]
    ) -> None:
        service = IMChannelService(channel_repo=repo, supervisor=supervisor)
        store["ch-1"] = _seed_channel(channel_id="ch-1")
        factory_calls = [0]

        def _factory(channel: IMChannel) -> IMAdapter:
            factory_calls[0] += 1
            return _FakeAdapter(platform="feishu")

        service.register_adapter_factory("feishu", cast(AdapterFactory, _factory))

        await service.start_channel(tenant_id=_TENANT_ID, channel_id="ch-1")
        first = service.get_channel_adapter("ch-1")[0]
        await service.start_channel(tenant_id=_TENANT_ID, channel_id="ch-1")
        second = service.get_channel_adapter("ch-1")[0]
        assert first is not second
        assert cast("_FakeAdapter", first).disconnect_calls == 1
        assert factory_calls[0] == 2

    async def test_stop_channel_calls_disconnect_and_drops(
        self, repo: AsyncMock, supervisor: IMSupervisor, store: dict[str, IMChannel]
    ) -> None:
        service = IMChannelService(channel_repo=repo, supervisor=supervisor)
        store["ch-1"] = _seed_channel(channel_id="ch-1")

        def _factory(channel: IMChannel) -> IMAdapter:
            return _FakeAdapter(platform="feishu")

        service.register_adapter_factory("feishu", cast(AdapterFactory, _factory))

        await service.start_channel(tenant_id=_TENANT_ID, channel_id="ch-1")
        adapter = cast("_FakeAdapter", service.get_channel_adapter("ch-1")[0])
        assert adapter.connect_calls == 1

        service.stop_channel("ch-1")
        _, _, running = service.get_channel_adapter("ch-1")
        assert running is False
        assert adapter.disconnect_calls == 1
        assert service.active_channel_count() == 0


# ── Health check + reconnect ────────────────────────────────────────


class TestHealthCheck(ServiceTest):
    async def test_reconnects_an_unhealthy_adapter(
        self, repo: AsyncMock, supervisor: IMSupervisor, store: dict[str, IMChannel]
    ) -> None:
        service = IMChannelService(channel_repo=repo, supervisor=supervisor)
        store["ch-1"] = _seed_channel(channel_id="ch-1")
        factory_calls = [0]

        def _factory(channel: IMChannel) -> IMAdapter:
            factory_calls[0] += 1
            return _FakeAdapter(platform="feishu", connected=False)

        service.register_adapter_factory("feishu", cast(AdapterFactory, _factory))

        await service.start_channel(tenant_id=_TENANT_ID, channel_id="ch-1")
        first = cast("_FakeAdapter", service.get_channel_adapter("ch-1")[0])
        # Verify the first adapter is unhealthy before health check.
        assert first.is_connected() is False

        reconnected = await supervisor.health_check()

        assert reconnected == ["ch-1"]
        # A second adapter was constructed to replace the unhealthy one.
        assert factory_calls[0] == 2
        assert first.disconnect_calls == 1

    async def test_healthy_adapter_left_alone(
        self, repo: AsyncMock, supervisor: IMSupervisor, store: dict[str, IMChannel]
    ) -> None:
        service = IMChannelService(channel_repo=repo, supervisor=supervisor)
        store["ch-1"] = _seed_channel(channel_id="ch-1")
        factory_calls = [0]

        def _factory(channel: IMChannel) -> IMAdapter:
            factory_calls[0] += 1
            return _FakeAdapter(platform="feishu", connected=True)

        service.register_adapter_factory("feishu", cast(AdapterFactory, _factory))
        await service.start_channel(tenant_id=_TENANT_ID, channel_id="ch-1")

        reconnected = await supervisor.health_check()

        assert reconnected == []
        assert factory_calls[0] == 1

    async def test_health_check_swallows_factory_errors(
        self, repo: AsyncMock, supervisor: IMSupervisor, store: dict[str, IMChannel]
    ) -> None:
        service = IMChannelService(channel_repo=repo, supervisor=supervisor)
        store["ch-1"] = _seed_channel(channel_id="ch-1")
        calls = [0]

        def _factory(channel: IMChannel) -> IMAdapter:
            calls[0] += 1
            if calls[0] == 1:
                return _FakeAdapter(platform="feishu", connected=False)
            raise RuntimeError("boom")

        service.register_adapter_factory("feishu", cast(AdapterFactory, _factory))
        await service.start_channel(tenant_id=_TENANT_ID, channel_id="ch-1")

        # The health check finds the unhealthy adapter and rebuilds it;
        # the rebuild raises, and the channel is dropped so the loop
        # keeps running without raising out of health_check.
        reconnected = await supervisor.health_check()
        assert reconnected == []
        _, _, running = service.get_channel_adapter("ch-1")
        assert running is False
        assert calls[0] == 2


# ── run_supervised lifecycle ────────────────────────────────────────


class TestRunSupervised(ServiceTest):
    async def test_recycles_after_max_conn_age(
        self,
    ) -> None:
        connects: list[int] = [0]

        async def _connect():
            connects[0] += 1

            def _stop() -> None:
                pass

            return _stop

        cfg = SupervisorConfig(
            name="test",
            connect=_connect,
            max_conn_age=0.05,
            retry_delay=0.01,
        )
        stop_event = asyncio.Event()
        task = asyncio.create_task(run_supervised(stop_event, cfg))
        # Let the loop tick at least twice (recycle at 50ms), then stop.
        await asyncio.sleep(0.15)
        stop_event.set()
        await asyncio.wait_for(task, timeout=1.0)
        assert connects[0] >= 2

    async def test_retries_on_connect_failure_then_succeeds(
        self,
    ) -> None:
        attempts: list[int] = [0]

        async def _connect():
            attempts[0] += 1
            if attempts[0] < 3:
                raise RuntimeError("not yet")

            def _stop() -> None:
                pass

            return _stop

        cfg = SupervisorConfig(
            name="retry",
            connect=_connect,
            max_conn_age=0.2,
            retry_delay=0.01,
        )
        stop_event = asyncio.Event()
        task = asyncio.create_task(run_supervised(stop_event, cfg))
        # retry_delay=10ms → the first two failures retry within 20ms;
        # the third attempt succeeds and holds the connection.
        await asyncio.sleep(0.1)
        assert attempts[0] >= 3
        stop_event.set()
        await asyncio.wait_for(task, timeout=1.0)


# ── build_im_channel_service factory ────────────────────────────────


class TestFactory(ServiceTest):
    def test_builds_service_with_default_supervisor(self) -> None:
        # Cannot construct a real AsyncSession here, so we only assert
        # the factory's call shape via a stub.
        class _StubRepo:
            pass

        class _StubSession:
            def __init__(self) -> None:
                # The factory will pass this to ``IMChannelRepository``;
                # the real constructor would store it on the instance.
                self.repo = _StubRepo()

        session = cast(AsyncSession, _StubSession())
        svc = build_im_channel_service(session)
        assert isinstance(svc, IMChannelService)
        assert svc._channel_repo is not None  # type: ignore[attr-defined]
        assert svc._supervisor is get_default_supervisor()  # type: ignore[attr-defined]


# ── Optional capabilities (StreamSender) ────────────────────────────


class TestStreamSenderProbe(ServiceTest):
    def test_imadapter_does_not_satisfy_streamsender(self) -> None:
        adapter: IMAdapter = _FakeAdapter(platform="feishu")
        # The bare adapter does not implement the streaming protocol.
        assert not isinstance(adapter, StreamSender)

    def test_runtime_checkable_protocol_recognises_impl(self) -> None:
        class _StreamingAdapter(_FakeAdapter):
            async def start_stream(self, ctx: Context, incoming: IncomingMessage) -> str:
                return "stream-id"

            async def update_stream_content(
                self, ctx: Context, incoming: IncomingMessage, stream_id: str, full_content: str
            ) -> None:
                return None

            async def finalize_stream(
                self, ctx: Context, incoming: IncomingMessage, stream_id: str, final_content: str
            ) -> None:
                return None

            async def end_stream(
                self, ctx: Context, incoming: IncomingMessage, stream_id: str
            ) -> None:
                return None

        streaming: IMAdapter = _StreamingAdapter(platform="feishu")
        assert isinstance(streaming, StreamSender)


# ── Message shape sanity ─────────────────────────────────────────────


class TestMessageShapes(ServiceTest):
    def test_incoming_message_defaults(self) -> None:
        msg = IncomingMessage(platform="feishu", user_id="u")
        assert msg.message_type == MESSAGE_TYPE_TEXT
        assert msg.chat_type == "direct"
        assert msg.extra == {}
        assert msg.quote is None
        assert msg.file_size == 0

    def test_reply_message_defaults(self) -> None:
        reply = ReplyMessage(content="hi")
        assert reply.is_streaming is False
        assert reply.is_final is False
        assert reply.extra == {}


__all__ = []
