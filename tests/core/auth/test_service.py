"""Unit tests for `AuthService` + `src.util.security`.

Per AGENTS.md §9, core services are tested with Protocol-based fakes
rather than against a real database. The session is a real
``AsyncSession`` against an in-memory SQLite engine; the protocol fakes
read/write from in-memory dicts and never touch the session, but the
session is still threaded through every service call to mirror the
real wire path.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.common.exception import UnauthorizedError
from src.core.auth.service import AuthService, LoginResult
from src.core.auth.types import UserDTO, UserPreferences
from src.db.dao.auth_tokens_repository import AuthTokenRepository
from src.db.dao.users_repository import UserRepository
from src.util.security import (
    TokenError,
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)

if TYPE_CHECKING:
    from src.db.models.auth.auth_tokens import AuthTokenRow


# ── In-memory session factory ────────────────────────────────────────


class _SessionFactory:
    """AsyncSession factory used by service tests."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._factory = async_sessionmaker(engine, expire_on_commit=False)

    def __call__(self) -> AsyncSession:
        return self._factory()


@pytest.fixture
async def session_factory() -> AsyncIterator[_SessionFactory]:
    engine: AsyncEngine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        yield _SessionFactory(engine)
    finally:
        await engine.dispose()


# ── In-memory fakes ───────────────────────────────────────────────────


class FakeUserRepository:
    """In-memory replacement for `UserRepository`."""

    def __init__(self) -> None:
        self.users: dict[str, UserDTO] = {}

    async def find_by_email(self, session: AsyncSession, email: str) -> UserDTO | None:
        for u in self.users.values():
            if u.email == email:
                return u
        return None

    async def find_by_id(self, session: AsyncSession, user_id: str) -> UserDTO | None:
        return self.users.get(user_id)

    async def insert(self, session: AsyncSession, row: object) -> None:
        return None


class FakeAuthTokenRepository:
    """In-memory replacement for `AuthTokenRepository`."""

    def __init__(self) -> None:
        self.tokens: dict[str, AuthTokenRow] = {}
        self.by_value: dict[str, str] = {}

    async def insert(self, session: AsyncSession, row: AuthTokenRow) -> None:
        self.tokens[row.id] = row
        self.by_value[row.token] = row.id

    async def find_by_token_value(self, session: AsyncSession, token: str) -> AuthTokenRow | None:
        token_id = self.by_value.get(token)
        if token_id is None:
            return None
        return self.tokens.get(token_id)

    async def revoke_all_for_user(self, session: AsyncSession, user_id: str) -> int:
        n = 0
        for tid, t in list(self.tokens.items()):
            if t.user_id == user_id and not t.is_revoked:
                self.tokens[tid] = t.model_copy(update={"is_revoked": True})
                n += 1
        return n

    async def revoke(self, session: AsyncSession, token_id: str) -> int:
        t = self.tokens.get(token_id)
        if t is None or t.is_revoked:
            return 0
        self.tokens[token_id] = t.model_copy(update={"is_revoked": True})
        return 1


# ── Helpers ──────────────────────────────────────────────────────────


def _seed_user(
    fake_users: FakeUserRepository,
    *,
    id: str = "usr-1",
    email: str = "alice@example.com",
    password: str = "correct-horse",
    is_active: bool = True,
    tenant_id: int | None = 7,
) -> UserDTO:
    user = UserDTO(
        id=id,
        username="alice",
        email=email,
        password_hash=hash_password(password),
        avatar=None,
        tenant_id=tenant_id,
        is_active=is_active,
        can_access_all_tenants=False,
        is_system_admin=False,
        preferences=UserPreferences(),
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    fake_users.users[user.id] = user
    return user


def _make_service(
    users: FakeUserRepository,
    tokens: FakeAuthTokenRepository,
) -> AuthService:
    return AuthService(
        user_repository=users,
        token_repository=tokens,
    )


# ── Password helpers ────────────────────────────────────────────────


def test_hash_password_round_trip() -> None:
    h = hash_password("hello")
    assert verify_password("hello", h)
    assert not verify_password("wrong", h)


def test_hash_password_distinct_salts() -> None:
    assert hash_password("hello") != hash_password("hello")


# ── JWT helpers ──────────────────────────────────────────────────────


def test_access_token_decodes() -> None:
    token, exp = create_access_token(user_id="u1", email="a@b", tenant_id=42)
    claims = decode_token(token)
    assert claims["user_id"] == "u1"
    assert claims["email"] == "a@b"
    assert claims["tenant_id"] == 42
    assert claims["type"] == "access"
    assert exp > datetime.now(UTC)


def test_refresh_token_decodes() -> None:
    token, _ = create_refresh_token(user_id="u1")
    claims = decode_token(token)
    assert claims["user_id"] == "u1"
    assert claims["type"] == "refresh"
    assert "email" not in claims


def test_decode_invalid_token_raises() -> None:
    with pytest.raises(TokenError):
        decode_token("not-a-jwt")


# ── Login ────────────────────────────────────────────────────────────


async def test_login_success(session_factory: _SessionFactory) -> None:
    users = FakeUserRepository()
    tokens = FakeAuthTokenRepository()
    _seed_user(users)
    svc = _make_service(users, tokens)

    async with session_factory() as session:
        result = await svc.login(session, email="alice@example.com", password="correct-horse")
        assert isinstance(result, LoginResult)
        assert result.user.id == "usr-1"
        assert result.access_token
        assert result.refresh_token
        assert len(tokens.tokens) == 2
        types = {t.token_type for t in tokens.tokens.values()}
        assert types == {"access_token", "refresh_token"}


async def test_login_wrong_email_raises(session_factory: _SessionFactory) -> None:
    users = FakeUserRepository()
    tokens = FakeAuthTokenRepository()
    _seed_user(users)
    svc = _make_service(users, tokens)

    async with session_factory() as session:
        with pytest.raises(UnauthorizedError):
            await svc.login(session, email="nobody@example.com", password="correct-horse")
        assert tokens.tokens == {}


async def test_login_wrong_password_raises(session_factory: _SessionFactory) -> None:
    users = FakeUserRepository()
    tokens = FakeAuthTokenRepository()
    _seed_user(users)
    svc = _make_service(users, tokens)

    async with session_factory() as session:
        with pytest.raises(UnauthorizedError):
            await svc.login(session, email="alice@example.com", password="wrong")
        assert tokens.tokens == {}


async def test_login_inactive_user_raises(session_factory: _SessionFactory) -> None:
    users = FakeUserRepository()
    tokens = FakeAuthTokenRepository()
    _seed_user(users, is_active=False)
    svc = _make_service(users, tokens)

    async with session_factory() as session:
        with pytest.raises(UnauthorizedError):
            await svc.login(session, email="alice@example.com", password="correct-horse")


# ── validate_token ───────────────────────────────────────────────────


async def test_validate_token_round_trip(session_factory: _SessionFactory) -> None:
    users = FakeUserRepository()
    tokens = FakeAuthTokenRepository()
    _seed_user(users, tenant_id=99)
    svc = _make_service(users, tokens)

    async with session_factory() as session:
        login = await svc.login(session, email="alice@example.com", password="correct-horse")
        user, tenant_id = await svc.validate_token(session, token=login.access_token)
        assert user.id == "usr-1"
        assert tenant_id == 99


async def test_validate_token_revoked_raises(session_factory: _SessionFactory) -> None:
    users = FakeUserRepository()
    tokens = FakeAuthTokenRepository()
    _seed_user(users)
    svc = _make_service(users, tokens)

    async with session_factory() as session:
        login = await svc.login(session, email="alice@example.com", password="correct-horse")
        revoked = await svc.revoke_token(session, token=login.access_token)
        assert revoked == 1

        with pytest.raises(UnauthorizedError):
            await svc.validate_token(session, token=login.access_token)


async def test_validate_token_with_refresh_raises(
    session_factory: _SessionFactory,
) -> None:
    users = FakeUserRepository()
    tokens = FakeAuthTokenRepository()
    _seed_user(users)
    svc = _make_service(users, tokens)

    async with session_factory() as session:
        login = await svc.login(session, email="alice@example.com", password="correct-horse")
        with pytest.raises(UnauthorizedError):
            await svc.validate_token(session, token=login.refresh_token)


async def test_validate_token_garbage_raises(session_factory: _SessionFactory) -> None:
    users = FakeUserRepository()
    tokens = FakeAuthTokenRepository()
    _seed_user(users)
    svc = _make_service(users, tokens)

    async with session_factory() as session:
        with pytest.raises(UnauthorizedError):
            await svc.validate_token(session, token="not-a-jwt")


# ── refresh ──────────────────────────────────────────────────────────


async def test_refresh_success(session_factory: _SessionFactory) -> None:
    users = FakeUserRepository()
    tokens = FakeAuthTokenRepository()
    _seed_user(users)
    svc = _make_service(users, tokens)

    async with session_factory() as session:
        login = await svc.login(session, email="alice@example.com", password="correct-horse")
        new = await svc.refresh(session, refresh_token=login.refresh_token)
        assert new.access_token != login.access_token
        assert new.refresh_token != login.refresh_token
        old = await tokens.find_by_token_value(session, login.refresh_token)
        assert old is not None and old.is_revoked is True
        assert len(tokens.tokens) == 4  # access+refresh x 2


async def test_refresh_revoked_raises(session_factory: _SessionFactory) -> None:
    users = FakeUserRepository()
    tokens = FakeAuthTokenRepository()
    _seed_user(users)
    svc = _make_service(users, tokens)

    async with session_factory() as session:
        login = await svc.login(session, email="alice@example.com", password="correct-horse")
        await svc.logout(session, token=login.access_token)

        with pytest.raises(UnauthorizedError):
            await svc.refresh(session, refresh_token=login.refresh_token)


async def test_refresh_with_access_token_raises(
    session_factory: _SessionFactory,
) -> None:
    users = FakeUserRepository()
    tokens = FakeAuthTokenRepository()
    _seed_user(users)
    svc = _make_service(users, tokens)

    async with session_factory() as session:
        login = await svc.login(session, email="alice@example.com", password="correct-horse")
        with pytest.raises(UnauthorizedError):
            await svc.refresh(session, refresh_token=login.access_token)


# ── logout ───────────────────────────────────────────────────────────


async def test_logout_revokes_all_tokens(session_factory: _SessionFactory) -> None:
    users = FakeUserRepository()
    tokens = FakeAuthTokenRepository()
    _seed_user(users)
    svc = _make_service(users, tokens)

    async with session_factory() as session:
        login = await svc.login(session, email="alice@example.com", password="correct-horse")
        n = await svc.logout(session, token=login.access_token)
        assert n == 2
        for t in tokens.tokens.values():
            assert t.is_revoked is True


async def test_logout_after_logout_is_noop(session_factory: _SessionFactory) -> None:
    users = FakeUserRepository()
    tokens = FakeAuthTokenRepository()
    _seed_user(users)
    svc = _make_service(users, tokens)

    async with session_factory() as session:
        login = await svc.login(session, email="alice@example.com", password="correct-horse")
        await svc.logout(session, token=login.access_token)
        n = await svc.logout(session, token=login.access_token)
        assert n == 0


async def test_logout_garbage_raises(session_factory: _SessionFactory) -> None:
    users = FakeUserRepository()
    tokens = FakeAuthTokenRepository()
    _seed_user(users)
    svc = _make_service(users, tokens)

    async with session_factory() as session:
        with pytest.raises(UnauthorizedError):
            await svc.logout(session, token="not-a-jwt")


# ── Default-repository instantiation ──────────────────────────────────


def test_default_repository_classes() -> None:
    """No fakes passed → service uses the concrete DAOs."""
    svc = AuthService()
    assert isinstance(svc._users, UserRepository)
    assert isinstance(svc._tokens, AuthTokenRepository)


__all__ = []
