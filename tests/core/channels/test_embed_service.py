"""Unit tests for the embed channel service, session flow, and webhook.

The embed channel repository is replaced with an ``AsyncMock`` backed by
closure-captured live-row state; the session repository with a capturing
mock; the agent-ownership seam with a tiny in-memory stub. Redis-backed
seams (session-token store, rate limiter) use minimal fakes so the
service logic is exercised without a live Redis.

AAA pattern throughout: arrange state, act through the service, assert
on the returned projections / raised errors.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.common.exception import (
    NotFoundError,
    PermissionDeniedError,
    UnauthorizedError,
    ValidationError,
)
from src.core.channels.embed.service.embed_channel_service import (
    EMBED_DEFAULT_RATE_LIMIT_PER_MINUTE,
    EMBED_PUBLISH_TOKEN_PREFIX,
    EmbedChannelCreateRequest,
    EmbedChannelNotFoundError,
    EmbedChannelService,
    EmbedChannelUpdateRequest,
    generate_publish_token,
    normalize_default_locale,
    normalize_header_title_mode,
    normalize_widget_position,
)
from src.core.channels.embed.session import (
    EMBED_SESSION_TOKEN_PREFIX,
    CreatedEmbedSession,
    EmbedRateLimiter,
    EmbedSessionService,
    InMemoryRateLimiter,
    RateLimitExceededError,
    is_embed_session_token,
    origin_allowed,
    sign_embed_session_handle,
    verify_embed_session_handle,
)
from src.core.channels.embed.types import EMBED_SESSION_MARKER_PREFIX
from src.core.channels.embed.webhook import (
    SIGNATURE_HEADER_NAME,
    SIGNATURE_PREFIX,
    EmbedWebhookDispatcher,
    sign_embed_webhook_body,
    validate_embed_webhook_url,
    verify_embed_webhook_signature,
)
from src.db.dao.embed_channel_repository import EmbedChannelRepository
from src.db.dao.session_repository import SessionRepository
from src.db.models.embed_channel import EMBED_DEFAULT_AGENT_ID, EmbedChannel
from src.db.models.session import Session

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_TENANT = 7
_OTHER_TENANT = 99

_PUBLISH_TOKEN_PATTERN = re.compile(r"^em_[A-Za-z0-9_-]{43}$")
_SESSION_TOKEN_PATTERN = re.compile(r"^ems_[A-Za-z0-9_-]{43}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


# ── Helpers ──────────────────────────────────────────────────────────


def _channel_row(**overrides: object) -> dict[str, object]:
    """Full embed_channels row dict; the model validates it directly."""
    row: dict[str, object] = {
        "id": "ch-1",
        "tenant_id": _TENANT,
        "agent_id": "builtin-quick-answer",
        "name": "Web widget",
        "enabled": True,
        "publish_token": "em_existingtoken",
        "allowed_origins": ["https://example.com"],
        "welcome_message": "Hi!",
        "rate_limit_per_minute": 30,
        "rate_limit_per_day": 10000,
        "primary_color": "#1f6feb",
        "page_title": "Support",
        "header_title_mode": "channel",
        "show_suggested_questions": True,
        "widget_position": "bottom-right",
        "allow_web_search": False,
        "allow_file_upload": False,
        "default_locale": "",
        "webhook_url": "",
        "webhook_secret": "whsec_secret",
        "created_at": _NOW,
        "updated_at": _NOW,
        "deleted_at": None,
    }
    row.update(overrides)
    return row


def _channel(**overrides: object) -> EmbedChannel:
    return EmbedChannel.model_validate(_channel_row(**overrides))


def _seed_channel(rows: dict[str, EmbedChannel], **overrides: object) -> EmbedChannel:
    row = _channel(**overrides)
    rows[row.id] = row
    return row


class _FakeAgentOwnership:
    """Minimal seam; rejects agent ids not in ``owned``."""

    def __init__(self, owned: set[str] | None = None) -> None:
        self._owned = owned if owned is not None else {"builtin-quick-answer", "agent-1"}

    async def get_agent_by_id(self, *, tenant_id: int, agent_id: str) -> object:
        if agent_id not in self._owned:
            raise NotFoundError(code="agent.not_found", message="agent not found")
        return object()


# ── Repository mock ──────────────────────────────────────────────────


def _make_embed_repo() -> tuple[AsyncMock, dict[str, EmbedChannel]]:
    repo = AsyncMock(spec=EmbedChannelRepository)
    rows: dict[str, EmbedChannel] = {}

    def _live() -> dict[str, EmbedChannel]:
        return {i: r for i, r in rows.items() if r.deleted_at is None}

    async def _create(row: EmbedChannel) -> EmbedChannel:
        rows[row.id] = row
        return row

    async def _update(row: EmbedChannel) -> EmbedChannel:
        if row.id not in rows or rows[row.id].deleted_at is not None:
            raise ValueError(f"embed channel {row.id} not live")
        rows[row.id] = row
        return row

    async def _get_by_id(channel_id: str) -> EmbedChannel | None:
        return _live().get(channel_id)

    async def _list_by_agent(tenant_id: int, agent_id: str) -> list[EmbedChannel]:
        return [r for r in _live().values() if r.tenant_id == tenant_id and r.agent_id == agent_id]

    async def _list_by_tenant(tenant_id: int) -> list[EmbedChannel]:
        return [r for r in _live().values() if r.tenant_id == tenant_id]

    async def _soft_delete(*, channel_id: str, tenant_id: int, now: datetime) -> bool:
        existing = rows.get(channel_id)
        if existing is None or existing.tenant_id != tenant_id or existing.deleted_at is not None:
            return False
        rows[channel_id] = existing.model_copy(update={"deleted_at": now, "updated_at": now})
        return True

    repo.create.side_effect = _create
    repo.update.side_effect = _update
    repo.get_by_id.side_effect = _get_by_id
    repo.list_by_agent.side_effect = _list_by_agent
    repo.list_by_tenant.side_effect = _list_by_tenant
    repo.soft_delete.side_effect = _soft_delete
    return repo, rows


def _make_session_repo() -> tuple[AsyncMock, list[Session]]:
    repo = AsyncMock(spec=SessionRepository)
    created: list[Session] = []

    async def _create(row: Session) -> Session:
        created.append(row)
        return row

    repo.create.side_effect = _create
    return repo, created


def _make_service(repo: AsyncMock) -> EmbedChannelService:
    return EmbedChannelService(
        repo=repo,
        agent_ownership=_FakeAgentOwnership(),
    )


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def repo_rows() -> tuple[AsyncMock, dict[str, EmbedChannel]]:
    return _make_embed_repo()


@pytest.fixture
def embed_repo(repo_rows: tuple[AsyncMock, dict[str, EmbedChannel]]) -> AsyncMock:
    return repo_rows[0]


@pytest.fixture
def rows(repo_rows: tuple[AsyncMock, dict[str, EmbedChannel]]) -> dict[str, EmbedChannel]:
    return repo_rows[1]


@pytest.fixture
def service(embed_repo: AsyncMock) -> EmbedChannelService:
    return _make_service(embed_repo)


# ── Token generation ─────────────────────────────────────────────────


def test_generate_publish_token_format() -> None:
    token = generate_publish_token()
    assert token.startswith(EMBED_PUBLISH_TOKEN_PREFIX)
    assert _PUBLISH_TOKEN_PATTERN.fullmatch(token)


def test_generate_publish_token_unique() -> None:
    tokens = {generate_publish_token() for _ in range(100)}
    assert len(tokens) == 100


# ── Normalizers ──────────────────────────────────────────────────────


def test_normalize_widget_position_valid_and_default() -> None:
    assert normalize_widget_position("top-left") == "top-left"
    assert normalize_widget_position("  bottom-right ") == "bottom-right"
    assert normalize_widget_position("floating") == "bottom-right"
    assert normalize_widget_position("") == "bottom-right"


def test_normalize_header_title_mode() -> None:
    assert normalize_header_title_mode("session") == "session"
    assert normalize_header_title_mode("channel") == "channel"
    assert normalize_header_title_mode("weird") == "channel"
    assert normalize_header_title_mode("") == "channel"


def test_normalize_default_locale() -> None:
    assert normalize_default_locale("zh-CN") == "zh-CN"
    assert normalize_default_locale("en-US") == "en-US"
    assert normalize_default_locale("fr-FR") == ""
    assert normalize_default_locale("") == ""


# ── Create ───────────────────────────────────────────────────────────


async def test_create_channel_generates_token_and_persists(
    service: EmbedChannelService, rows: dict[str, EmbedChannel]
) -> None:
    info, token = await service.create_channel(
        tenant_id=_TENANT,
        agent_id="agent-1",
        request=EmbedChannelCreateRequest(
            name="  Support widget  ",
            allowed_origins=["https://app.example.com"],
            welcome_message="Hello!",
            rate_limit_per_minute=60,
            rate_limit_per_day=2000,
            primary_color="  #123456 ",
            page_title="  Help desk  ",
        ),
    )

    assert token.startswith(EMBED_PUBLISH_TOKEN_PREFIX)
    assert _PUBLISH_TOKEN_PATTERN.fullmatch(token)
    persisted = rows[info.id]
    assert persisted.publish_token == token
    assert persisted.name == "Support widget"
    assert persisted.primary_color == "#123456"
    assert persisted.page_title == "Help desk"
    assert persisted.allowed_origins == ["https://app.example.com"]
    assert persisted.tenant_id == _TENANT
    assert persisted.agent_id == "agent-1"
    # The projection never carries the token or secret.
    assert info.id == persisted.id
    assert not hasattr(info, "publish_token")
    assert not hasattr(info, "webhook_secret")


async def test_create_channel_applies_defaults(
    service: EmbedChannelService, rows: dict[str, EmbedChannel]
) -> None:
    info, _ = await service.create_channel(
        tenant_id=_TENANT,
        agent_id="agent-1",
        request=EmbedChannelCreateRequest(
            header_title_mode="floating",
            widget_position="floating",
            default_locale="fr-FR",
        ),
    )
    persisted = rows[info.id]
    assert persisted.rate_limit_per_minute == EMBED_DEFAULT_RATE_LIMIT_PER_MINUTE
    assert persisted.rate_limit_per_day == 10000
    assert persisted.header_title_mode == "channel"
    assert persisted.widget_position == "bottom-right"
    assert persisted.default_locale == ""


async def test_create_channel_defaults_blank_agent_id(
    service: EmbedChannelService, rows: dict[str, EmbedChannel]
) -> None:
    info, _ = await service.create_channel(
        tenant_id=_TENANT,
        agent_id="   ",
        request=EmbedChannelCreateRequest(name="Widget"),
    )
    assert rows[info.id].agent_id == EMBED_DEFAULT_AGENT_ID


async def test_create_channel_rejects_unowned_agent(
    service: EmbedChannelService, rows: dict[str, EmbedChannel]
) -> None:
    with pytest.raises(EmbedChannelNotFoundError):
        await service.create_channel(
            tenant_id=_TENANT,
            agent_id="not-owned",
            request=EmbedChannelCreateRequest(),
        )
    assert rows == {}


# ── Get / list ───────────────────────────────────────────────────────


async def test_get_channel_returns_projection(
    service: EmbedChannelService, rows: dict[str, EmbedChannel]
) -> None:
    _seed_channel(rows, id="ch-1")
    info = await service.get_channel(tenant_id=_TENANT, channel_id="ch-1")
    assert info.id == "ch-1"
    assert info.allowed_origins == ["https://example.com"]
    assert not hasattr(info, "publish_token")


async def test_get_channel_missing_raises(service: EmbedChannelService) -> None:
    with pytest.raises(EmbedChannelNotFoundError):
        await service.get_channel(tenant_id=_TENANT, channel_id="ch-ghost")


async def test_get_channel_other_tenant_raises(
    service: EmbedChannelService, rows: dict[str, EmbedChannel]
) -> None:
    _seed_channel(rows, id="ch-1", tenant_id=_OTHER_TENANT)
    with pytest.raises(EmbedChannelNotFoundError):
        await service.get_channel(tenant_id=_TENANT, channel_id="ch-1")


async def test_get_owned_channel_returns_raw_row_with_secrets(
    service: EmbedChannelService, rows: dict[str, EmbedChannel]
) -> None:
    _seed_channel(rows, id="ch-1", publish_token="em_secret-token")
    row = await service.get_owned_channel(tenant_id=_TENANT, channel_id="ch-1")
    assert isinstance(row, EmbedChannel)
    assert row.publish_token == "em_secret-token"
    assert row.webhook_secret == "whsec_secret"


async def test_list_channels_by_agent(
    service: EmbedChannelService, rows: dict[str, EmbedChannel]
) -> None:
    _seed_channel(rows, id="ch-1", agent_id="agent-1")
    _seed_channel(rows, id="ch-2", agent_id="agent-1")
    _seed_channel(rows, id="ch-3", agent_id="agent-2")
    infos = await service.list_channels_by_agent(tenant_id=_TENANT, agent_id="agent-1")
    assert {i.id for i in infos} == {"ch-1", "ch-2"}


async def test_list_channels_by_tenant(
    service: EmbedChannelService, rows: dict[str, EmbedChannel]
) -> None:
    _seed_channel(rows, id="ch-1")
    _seed_channel(rows, id="ch-2", agent_id="agent-2")
    _seed_channel(rows, id="ch-3", tenant_id=_OTHER_TENANT)
    infos = await service.list_channels_by_tenant(tenant_id=_TENANT)
    assert {i.id for i in infos} == {"ch-1", "ch-2"}


# ── Update ───────────────────────────────────────────────────────────


async def test_update_channel_applies_mutable_subset(
    service: EmbedChannelService, rows: dict[str, EmbedChannel]
) -> None:
    _seed_channel(rows, id="ch-1")
    info = await service.update_channel(
        tenant_id=_TENANT,
        channel_id="ch-1",
        request=EmbedChannelUpdateRequest(
            name="Renamed",
            enabled=False,
            rate_limit_per_minute=10,
            rate_limit_per_day=500,
            allowed_origins=["https://new.example.com"],
            widget_position="top-left",
            default_locale="en-US",
        ),
    )
    assert info.name == "Renamed"
    assert info.enabled is False
    assert info.rate_limit_per_minute == 10
    assert info.rate_limit_per_day == 500
    assert info.allowed_origins == ["https://new.example.com"]
    assert info.widget_position == "top-left"
    assert info.default_locale == "en-US"
    # Unchanged fields survive.
    assert info.welcome_message == "Hi!"
    assert info.primary_color == "#1f6feb"


async def test_update_channel_all_none_is_noop(
    service: EmbedChannelService, rows: dict[str, EmbedChannel]
) -> None:
    _seed_channel(rows, id="ch-1")
    info = await service.update_channel(
        tenant_id=_TENANT,
        channel_id="ch-1",
        request=EmbedChannelUpdateRequest(),
    )
    assert info.name == "Web widget"
    assert info.enabled is True
    assert info.allowed_origins == ["https://example.com"]


async def test_update_channel_normalizes_invalid_values(
    service: EmbedChannelService, rows: dict[str, EmbedChannel]
) -> None:
    _seed_channel(rows, id="ch-1")
    info = await service.update_channel(
        tenant_id=_TENANT,
        channel_id="ch-1",
        request=EmbedChannelUpdateRequest(
            widget_position="floating",
            header_title_mode="banner",
            default_locale="fr-FR",
        ),
    )
    assert info.widget_position == "bottom-right"
    assert info.header_title_mode == "channel"
    assert info.default_locale == ""


async def test_update_channel_webhook_url_validation(
    service: EmbedChannelService, rows: dict[str, EmbedChannel]
) -> None:
    _seed_channel(rows, id="ch-1")
    info = await service.update_channel(
        tenant_id=_TENANT,
        channel_id="ch-1",
        request=EmbedChannelUpdateRequest(
            webhook_url="https://hooks.example.com/embed",
            webhook_secret="new-secret",
        ),
    )
    assert info.webhook_url == "https://hooks.example.com/embed"

    with pytest.raises(ValidationError):
        await service.update_channel(
            tenant_id=_TENANT,
            channel_id="ch-1",
            request=EmbedChannelUpdateRequest(webhook_url="ftp://hooks.example.com"),
        )


async def test_update_channel_rejects_unowned_agent_change(
    service: EmbedChannelService, rows: dict[str, EmbedChannel]
) -> None:
    _seed_channel(rows, id="ch-1")
    with pytest.raises(EmbedChannelNotFoundError):
        await service.update_channel(
            tenant_id=_TENANT,
            channel_id="ch-1",
            request=EmbedChannelUpdateRequest(agent_id="not-owned"),
        )


async def test_update_channel_missing_raises(
    service: EmbedChannelService,
) -> None:
    with pytest.raises(EmbedChannelNotFoundError):
        await service.update_channel(
            tenant_id=_TENANT,
            channel_id="ch-ghost",
            request=EmbedChannelUpdateRequest(name="x"),
        )


# ── Delete / rotate ──────────────────────────────────────────────────


async def test_delete_channel_soft_deletes(
    service: EmbedChannelService, rows: dict[str, EmbedChannel]
) -> None:
    _seed_channel(rows, id="ch-1")
    await service.delete_channel(tenant_id=_TENANT, channel_id="ch-1")
    assert rows["ch-1"].deleted_at is not None
    with pytest.raises(EmbedChannelNotFoundError):
        await service.get_channel(tenant_id=_TENANT, channel_id="ch-1")
    assert await service.list_channels_by_tenant(tenant_id=_TENANT) == []


async def test_delete_channel_other_tenant_raises(
    service: EmbedChannelService, rows: dict[str, EmbedChannel]
) -> None:
    _seed_channel(rows, id="ch-1", tenant_id=_OTHER_TENANT)
    with pytest.raises(EmbedChannelNotFoundError):
        await service.delete_channel(tenant_id=_TENANT, channel_id="ch-1")
    assert rows["ch-1"].deleted_at is None


async def test_rotate_token_mints_new_token(
    service: EmbedChannelService, rows: dict[str, EmbedChannel]
) -> None:
    _seed_channel(rows, id="ch-1", publish_token="em_old-token")
    info, token = await service.rotate_token(tenant_id=_TENANT, channel_id="ch-1")
    assert token.startswith(EMBED_PUBLISH_TOKEN_PREFIX)
    assert token != "em_old-token"
    assert rows["ch-1"].publish_token == token
    assert info.id == "ch-1"
    assert not hasattr(info, "publish_token")


# ── Origin gating ────────────────────────────────────────────────────


def test_origin_allowed_cases() -> None:
    allowed = ["https://example.com", "*.trusted.io", "*"]
    assert origin_allowed("https://example.com", allowed) is True
    assert origin_allowed("HTTPS://EXAMPLE.COM", allowed) is True
    assert origin_allowed("https://sub.trusted.io", allowed) is True
    assert origin_allowed("https://evil.com", allowed) is True  # "*" blanket


def test_origin_allowed_empty_allowlist_rejects() -> None:
    assert origin_allowed("https://example.com", []) is False
    assert origin_allowed("", []) is False


def test_origin_allowed_no_wildcard_suffix() -> None:
    allowed = ["https://example.com", "*.trusted.io"]
    assert origin_allowed("https://nottrusted.io", allowed) is False
    assert origin_allowed("", allowed) is False
    # The "*.suffix" wildcard requires a subdomain label before the
    # apex; a bare apex does not match (mirrors the upstream suffix rule).
    assert origin_allowed("https://trusted.io", allowed) is False
    assert origin_allowed("https://sub.trusted.io", allowed) is True


async def test_session_origin_gate(embed_repo: AsyncMock, rows: dict[str, EmbedChannel]) -> None:
    _seed_channel(rows, id="ch-1", publish_token="em_pub")
    session_repo, _ = _make_session_repo()
    svc = EmbedSessionService(
        embed_channel_repo=embed_repo,
        session_repo=session_repo,
    )
    with pytest.raises(PermissionDeniedError):
        await svc.create_session(
            channel_id="ch-1",
            token="em_pub",
            origin="https://evil.example.com",
        )
    assert session_repo.create.await_count == 0


# ── Session service ──────────────────────────────────────────────────


async def test_lookup_for_embed_valid_publish_token(
    embed_repo: AsyncMock, rows: dict[str, EmbedChannel]
) -> None:
    _seed_channel(rows, id="ch-1", publish_token="em_pub-token")
    svc = EmbedSessionService(embed_channel_repo=embed_repo, session_repo=_make_session_repo()[0])
    channel = await svc.lookup_for_embed(channel_id="ch-1", token="em_pub-token")
    assert channel.id == "ch-1"


async def test_lookup_for_embed_invalid_token_raises(
    embed_repo: AsyncMock, rows: dict[str, EmbedChannel]
) -> None:
    _seed_channel(rows, id="ch-1", publish_token="em_real")
    svc = EmbedSessionService(embed_channel_repo=embed_repo, session_repo=_make_session_repo()[0])
    with pytest.raises(UnauthorizedError):
        await svc.lookup_for_embed(channel_id="ch-1", token="em_wrong")


async def test_lookup_for_embed_disabled_channel_raises(
    embed_repo: AsyncMock, rows: dict[str, EmbedChannel]
) -> None:
    _seed_channel(rows, id="ch-1", publish_token="em_pub", enabled=False)
    svc = EmbedSessionService(embed_channel_repo=embed_repo, session_repo=_make_session_repo()[0])
    with pytest.raises(PermissionDeniedError):
        await svc.lookup_for_embed(channel_id="ch-1", token="em_pub")


async def test_lookup_enabled_channel_raises_on_missing(
    embed_repo: AsyncMock, rows: dict[str, EmbedChannel]
) -> None:
    svc = EmbedSessionService(embed_channel_repo=embed_repo, session_repo=_make_session_repo()[0])
    with pytest.raises(UnauthorizedError):
        await svc.lookup_enabled_channel("ch-ghost")


def test_sign_and_verify_session_handle() -> None:
    channel = _channel(id="ch-1", publish_token="em_secret-token")
    sig = sign_embed_session_handle(channel, "session-1")
    assert sig
    assert verify_embed_session_handle(channel, "session-1", sig) is True
    assert verify_embed_session_handle(channel, "session-2", sig) is False
    assert verify_embed_session_handle(channel, "session-1", "tampered") is False
    assert verify_embed_session_handle(None, "session-1", sig) is False


def test_sign_handle_empty_session_id() -> None:
    channel = _channel(id="ch-1")
    assert sign_embed_session_handle(channel, "") == ""
    assert sign_embed_session_handle(None, "session-1") == ""


async def test_create_session_success(embed_repo: AsyncMock, rows: dict[str, EmbedChannel]) -> None:
    _seed_channel(rows, id="ch-1", publish_token="em_pub-token")
    session_repo, created = _make_session_repo()
    svc = EmbedSessionService(
        embed_channel_repo=embed_repo,
        session_repo=session_repo,
    )
    result = await svc.create_session(
        channel_id="ch-1",
        token="em_pub-token",
        origin="https://example.com",
        client_ip="1.2.3.4",
    )
    assert isinstance(result, CreatedEmbedSession)
    assert len(created) == 1
    session = created[0]
    assert session.id == result.session_id
    assert session.tenant_id == _TENANT
    assert session.description == EMBED_SESSION_MARKER_PREFIX + "ch-1"
    assert verify_embed_session_handle(rows["ch-1"], session.id, result.handle) is True


async def test_create_session_uses_session_token(
    embed_repo: AsyncMock, rows: dict[str, EmbedChannel]
) -> None:
    _seed_channel(rows, id="ch-1", publish_token="em_pub-token")
    session_repo, _ = _make_session_repo()

    class _FakeRedis:
        def __init__(self) -> None:
            self._store: dict[str, str] = {}

        async def set(self, key: str, value: str, *, ex: int) -> None:
            self._store[key] = value

        async def get(self, key: str) -> str | None:
            return self._store.get(key)

    fake_redis = _FakeRedis()
    svc = EmbedSessionService(
        embed_channel_repo=embed_repo,
        session_repo=session_repo,
        redis_client=fake_redis,
    )
    session_token, _ = await svc.issue_session_token(channel_id="ch-1")
    assert is_embed_session_token(session_token)
    assert _SESSION_TOKEN_PATTERN.fullmatch(session_token)

    resolved = await svc.resolve_session_token(session_token)
    assert resolved == "ch-1"

    result = await svc.create_session(
        channel_id="ch-1",
        token=session_token,
        origin="https://example.com",
    )
    assert result.session_id
    assert session_repo.create.await_count == 1


async def test_resolve_session_token_invalid(
    embed_repo: AsyncMock,
) -> None:
    svc = EmbedSessionService(embed_channel_repo=embed_repo, session_repo=_make_session_repo()[0])
    with pytest.raises(UnauthorizedError):
        await svc.resolve_session_token("not-a-session-token")


async def test_issue_session_token_without_redis_raises(
    embed_repo: AsyncMock, rows: dict[str, EmbedChannel]
) -> None:
    _seed_channel(rows, id="ch-1")
    svc = EmbedSessionService(embed_channel_repo=embed_repo, session_repo=_make_session_repo()[0])
    with pytest.raises(ValidationError):
        await svc.issue_session_token(channel_id="ch-1")


async def test_issue_preview_session(embed_repo: AsyncMock, rows: dict[str, EmbedChannel]) -> None:
    _seed_channel(rows, id="ch-1", publish_token="em_pub")
    session_repo, _ = _make_session_repo()

    class _FakeRedis:
        def __init__(self) -> None:
            self._store: dict[str, str] = {}

        async def set(self, key: str, value: str, *, ex: int) -> None:
            self._store[key] = value

        async def get(self, key: str) -> str | None:
            return self._store.get(key)

    svc = EmbedSessionService(
        embed_channel_repo=embed_repo,
        session_repo=session_repo,
        redis_client=_FakeRedis(),
    )
    token, ttl = await svc.issue_preview_session(channel_id="ch-1")
    assert token.startswith(EMBED_SESSION_TOKEN_PREFIX)
    assert ttl == 30 * 60


# ── Rate limiting ────────────────────────────────────────────────────


async def test_in_memory_rate_limiter_budget() -> None:
    limiter = InMemoryRateLimiter()
    for _ in range(3):
        await limiter.check(key="ch-1:1.2.3.4", limit=3, window_seconds=60)
    with pytest.raises(RateLimitExceededError):
        await limiter.check(key="ch-1:1.2.3.4", limit=3, window_seconds=60)
    # A separate key keeps its own budget.
    await limiter.check(key="ch-1:9.9.9.9", limit=3, window_seconds=60)


async def test_embed_rate_limiter_redis() -> None:
    class _FakeRedis:
        """Mimics the sliding-window Lua script for a single bucket."""

        def __init__(self) -> None:
            self._hits: dict[str, list[int]] = {}

        def register_script(self, script: str) -> Any:
            async def run(*, keys: list[str], args: list[object]) -> int:
                key = keys[0]
                now = int(args[0])
                window = int(args[1])
                max_req = int(args[2])
                live = [ts for ts in self._hits.get(key, []) if ts > now - window]
                if len(live) < max_req:
                    live.append(now)
                    self._hits[key] = live
                    return 1
                self._hits[key] = live
                return 0

            return run

    limiter = EmbedRateLimiter(_FakeRedis())
    for _ in range(2):
        await limiter.check(key="bucket", limit=2, window_seconds=60)
    with pytest.raises(RateLimitExceededError):
        await limiter.check(key="bucket", limit=2, window_seconds=60)


async def test_embed_rate_limiter_no_redis_fails_open() -> None:
    limiter = EmbedRateLimiter(None)
    await limiter.check(key="bucket", limit=1, window_seconds=60)


async def test_enforce_rate_limits_composes_buckets(
    embed_repo: AsyncMock, rows: dict[str, EmbedChannel]
) -> None:
    _seed_channel(rows, id="ch-1")
    session_repo, _ = _make_session_repo()
    recorded: list[tuple[str, int, int]] = []

    class _RecordingLimiter:
        async def check(self, *, key: str, limit: int, window_seconds: int) -> None:
            recorded.append((key, limit, window_seconds))

    svc = EmbedSessionService(
        embed_channel_repo=embed_repo,
        session_repo=session_repo,
        rate_limiter=_RecordingLimiter(),
    )
    await svc.enforce_rate_limits(channel=rows["ch-1"], client_ip="1.2.3.4")
    assert len(recorded) == 3
    keys = {key for key, _, _ in recorded}
    assert "embed:ratelimit:ch-1:1.2.3.4" in keys
    assert "embed:ratelimit:ch-1:__global" in keys
    assert "embed:ratelimit:day:ch-1" in keys
    # Global per-minute budget: max(30 * 20, 120) = 600.
    global_budget = [limit for key, limit, _ in recorded if key.endswith("__global")]
    assert global_budget == [600]


async def test_create_session_rate_limited(
    embed_repo: AsyncMock, rows: dict[str, EmbedChannel]
) -> None:
    _seed_channel(
        rows,
        id="ch-1",
        publish_token="em_pub-token",
        rate_limit_per_minute=1,
    )
    session_repo, _ = _make_session_repo()
    svc = EmbedSessionService(
        embed_channel_repo=embed_repo,
        session_repo=session_repo,
        rate_limiter=InMemoryRateLimiter(),
    )
    await svc.create_session(
        channel_id="ch-1",
        token="em_pub-token",
        origin="https://example.com",
        client_ip="1.2.3.4",
    )
    with pytest.raises(RateLimitExceededError):
        await svc.create_session(
            channel_id="ch-1",
            token="em_pub-token",
            origin="https://example.com",
            client_ip="1.2.3.4",
        )
    # The rejected call never created a session row.
    assert session_repo.create.await_count == 1


# ── Webhook ──────────────────────────────────────────────────────────


def test_sign_embed_webhook_body() -> None:
    raw = b'{"type":"session_created"}'
    sig = sign_embed_webhook_body("whsec_secret", raw)
    assert _HEX64.fullmatch(sig)
    assert sig != sign_embed_webhook_body("other-secret", raw)
    assert sign_embed_webhook_body("", raw) == ""


def test_verify_embed_webhook_signature() -> None:
    raw = b'{"type":"session_created"}'
    sig = sign_embed_webhook_body("whsec_secret", raw)
    assert verify_embed_webhook_signature("whsec_secret", raw, sig) is True
    assert verify_embed_webhook_signature("whsec_secret", raw, SIGNATURE_PREFIX + sig) is True
    assert verify_embed_webhook_signature("wrong", raw, sig) is False
    assert verify_embed_webhook_signature("whsec_secret", b"tampered", sig) is False
    assert verify_embed_webhook_signature("", raw, sig) is False
    assert verify_embed_webhook_signature("whsec_secret", raw, "") is False


async def test_validate_embed_webhook_url() -> None:
    await validate_embed_webhook_url("")  # empty allowed
    await validate_embed_webhook_url("https://hooks.example.com/x")
    await validate_embed_webhook_url("http://hooks.example.com/x")
    with pytest.raises(ValidationError):
        await validate_embed_webhook_url("ftp://hooks.example.com/x")
    with pytest.raises(ValidationError):
        await validate_embed_webhook_url("https://")
    with pytest.raises(ValidationError):
        await validate_embed_webhook_url("not a url")


async def test_validate_embed_webhook_url_ssrf_hook() -> None:
    called: list[str] = []

    async def _guard(raw: str) -> None:
        called.append(raw)

    await validate_embed_webhook_url("https://hooks.example.com", url_safety_check=_guard)
    assert called == ["https://hooks.example.com"]


async def test_dispatch_embed_webhook_sends_signed_post() -> None:
    class _FakeHTTPClient:
        def __init__(self) -> None:
            self.posts: list[dict[str, object]] = []

        async def post(
            self,
            url: str,
            *,
            content: bytes,
            headers: dict[str, str],
        ) -> None:
            self.posts.append({"url": url, "content": content, "headers": headers})

    client = _FakeHTTPClient()
    dispatcher = EmbedWebhookDispatcher(http_client=client)
    channel = _channel(
        id="ch-1",
        webhook_url="https://hooks.example.com/embed",
        webhook_secret="whsec_secret",
    )
    task = dispatcher.dispatch(
        channel,
        event_type="session_created",
        session_id="session-1",
        payload={"custom": "value"},
    )
    await task

    assert len(client.posts) == 1
    post = client.posts[0]
    assert post["url"] == "https://hooks.example.com/embed"
    headers = post["headers"]
    assert headers["Content-Type"] == "application/json"
    signature = headers[SIGNATURE_HEADER_NAME]
    assert signature.startswith(SIGNATURE_PREFIX)
    body = post["content"]
    assert isinstance(body, bytes)
    assert verify_embed_webhook_signature("whsec_secret", body, signature) is True
    import json

    payload = json.loads(body)
    assert payload["type"] == "session_created"
    assert payload["channel_id"] == "ch-1"
    assert payload["session_id"] == "session-1"
    assert payload["custom"] == "value"
    assert payload["timestamp"]


async def test_dispatch_embed_webhook_empty_url_is_noop() -> None:
    client = AsyncMock()
    dispatcher = EmbedWebhookDispatcher(http_client=client)
    channel = _channel(id="ch-1", webhook_url="", webhook_secret="")
    task = dispatcher.dispatch(channel, event_type="x", session_id="s", payload={})
    await task
    client.post.assert_not_called()


async def test_dispatch_embed_webhook_without_secret_omits_signature() -> None:
    class _FakeHTTPClient:
        def __init__(self) -> None:
            self.headers: dict[str, str] | None = None

        async def post(
            self,
            url: str,
            *,
            content: bytes,
            headers: dict[str, str],
        ) -> None:
            self.headers = headers

    client = _FakeHTTPClient()
    dispatcher = EmbedWebhookDispatcher(http_client=client)
    channel = _channel(id="ch-1", webhook_url="https://hooks.example.com/embed", webhook_secret="")
    task = dispatcher.dispatch(channel, event_type="x", session_id="s", payload={})
    await task
    assert SIGNATURE_HEADER_NAME not in (client.headers or {})


async def test_dispatch_embed_webhook_invalid_url_swallowed() -> None:
    client = AsyncMock()

    async def _guard(raw: str) -> None:
        raise ValidationError(code="oidc.ssrf_blocked", message="blocked")

    dispatcher = EmbedWebhookDispatcher(http_client=client, url_safety_check=_guard)
    channel = _channel(id="ch-1", webhook_url="https://hooks.example.com/embed", webhook_secret="")
    # The fire-and-forget task never raises; the POST never fires.
    task = dispatcher.dispatch(channel, event_type="x", session_id="s", payload={})
    await task
    client.post.assert_not_called()


def test_factory_builders_wire_services() -> None:
    from src.core.channels.embed.factory import (
        build_embed_channel_service,
        build_embed_session_service,
        build_embed_webhook_dispatcher,
    )

    # Repositories only retain the session reference, so a bare object is
    # a sufficient construction-time stand-in.
    session = object()
    assert isinstance(build_embed_channel_service(session), EmbedChannelService)
    session_service = build_embed_session_service(session)
    assert isinstance(session_service, EmbedSessionService)
    assert isinstance(build_embed_webhook_dispatcher(), EmbedWebhookDispatcher)
