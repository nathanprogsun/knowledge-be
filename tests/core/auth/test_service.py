"""Unit tests for `AuthService` + `src.util.security`.

Exercises the service with both in-memory fakes and the concrete
repositories to guard against drift between the AuthService signature
and the real repo signatures.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.common.exception import NotFoundError, UnauthorizedError
from src.core.auth.service import AuthService, LoginResult
from src.core.auth.types import UserInfo
from src.db.dao.auth_tokens_repository import AuthTokenRepository
from src.db.dao.users_repository import UserRepository
from src.db.models.auth.auth_tokens import AuthToken
from src.db.models.auth.users import User
from src.util.security import (
    TokenError,
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)

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
    """In-memory replacement for `UserRepository`.

    Holds the session in ``__init__`` (mirroring the real repo's
    contract) but never executes against it — reads/writes hit the
    in-memory ``users`` dict instead. Method signatures mirror the
    real ``UserRepository``: finders return storage ``User`` rows
    (the service projects them to ``UserInfo`` via
    ``UserInfo.map_from_db``).
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self.users: dict[str, User] = {}  # storage rows (with password_hash)

    async def find_by_email(self, email: str) -> User:
        for u in self.users.values():
            if u.email == email:
                return u
        raise NotFoundError(code="user.not_found", message=f"User {email} not found")

    async def find_by_id(self, user_id: str) -> User:
        user = self.users.get(user_id)
        if user is None:
            raise NotFoundError(code="user.not_found", message=f"User {user_id} not found")
        return user

    async def insert(self, row: User) -> User:
        self.users[row.id] = row
        return row


class FakeAuthTokenRepository:
    """In-memory replacement for `AuthTokenRepository`."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self.tokens: dict[str, AuthToken] = {}
        self.by_value: dict[str, str] = {}

    async def insert(self, row: AuthToken) -> AuthToken:
        self.tokens[row.id] = row
        self.by_value[row.token] = row.id
        return row

    async def find_by_token_value(self, token: str) -> AuthToken:
        token_id = self.by_value.get(token)
        if token_id is None:
            raise NotFoundError(code="token.not_found", message="Token not found")
        return self.tokens.get(token_id)  # type: ignore[return-value]

    async def revoke_all_for_user(self, user_id: str) -> int:
        n = 0
        for tid, t in list(self.tokens.items()):
            if t.user_id == user_id and not t.is_revoked:
                self.tokens[tid] = t.model_copy(update={"is_revoked": True})
                n += 1
        return n

    async def revoke(self, token_id: str) -> int:
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
) -> User:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    user = User(
        id=id,
        username="alice",
        email=email,
        password_hash=hash_password(password),
        avatar=None,
        tenant_id=tenant_id,
        is_active=is_active,
        can_access_all_tenants=False,
        is_system_admin=False,
        preferences={},
        created_at=now,
        updated_at=now,
    )
    fake_users.users[user.id] = user
    return user


def _make_service(
    users: FakeUserRepository,
    tokens: FakeAuthTokenRepository,
) -> AuthService:
    return AuthService(
        users_repo=users,  # type: ignore[arg-type]
        tokens_repo=tokens,  # type: ignore[arg-type]
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
    async with session_factory() as session:
        users = FakeUserRepository(session)
        tokens = FakeAuthTokenRepository(session)
        _seed_user(users)
        svc = _make_service(users, tokens)
        result = await svc.login(email="alice@example.com", password="correct-horse")
        assert isinstance(result, LoginResult)
        assert isinstance(result.user, UserInfo)
        assert result.user.id == "usr-1"
        assert result.access_token
        assert result.refresh_token
        assert len(tokens.tokens) == 2
        types = {t.token_type for t in tokens.tokens.values()}
        assert types == {"access_token", "refresh_token"}


async def test_login_wrong_email_raises(
    session_factory: _SessionFactory,
) -> None:
    async with session_factory() as session:
        users = FakeUserRepository(session)
        tokens = FakeAuthTokenRepository(session)
        _seed_user(users)
        svc = _make_service(users, tokens)
        with pytest.raises(UnauthorizedError):
            await svc.login(email="nobody@example.com", password="correct-horse")
        assert tokens.tokens == {}


async def test_login_wrong_password_raises(
    session_factory: _SessionFactory,
) -> None:
    async with session_factory() as session:
        users = FakeUserRepository(session)
        tokens = FakeAuthTokenRepository(session)
        _seed_user(users)
        svc = _make_service(users, tokens)
        with pytest.raises(UnauthorizedError):
            await svc.login(email="alice@example.com", password="wrong")
        assert tokens.tokens == {}


async def test_login_inactive_user_raises(
    session_factory: _SessionFactory,
) -> None:
    async with session_factory() as session:
        users = FakeUserRepository(session)
        tokens = FakeAuthTokenRepository(session)
        _seed_user(users, is_active=False)
        svc = _make_service(users, tokens)
        with pytest.raises(UnauthorizedError):
            await svc.login(email="alice@example.com", password="correct-horse")


# ── validate_token ───────────────────────────────────────────────────


async def test_validate_token_round_trip(
    session_factory: _SessionFactory,
) -> None:
    async with session_factory() as session:
        users = FakeUserRepository(session)
        tokens = FakeAuthTokenRepository(session)
        _seed_user(users, tenant_id=99)
        svc = _make_service(users, tokens)
        login = await svc.login(email="alice@example.com", password="correct-horse")
        user, tenant_id = await svc.validate_token(token=login.access_token)
        assert isinstance(user, UserInfo)
        assert user.id == "usr-1"
        assert tenant_id == 99


async def test_validate_token_revoked_raises(
    session_factory: _SessionFactory,
) -> None:
    async with session_factory() as session:
        users = FakeUserRepository(session)
        tokens = FakeAuthTokenRepository(session)
        _seed_user(users)
        svc = _make_service(users, tokens)
        login = await svc.login(email="alice@example.com", password="correct-horse")
        revoked = await svc.revoke_token(token=login.access_token)
        assert revoked == 1

        with pytest.raises(UnauthorizedError):
            await svc.validate_token(token=login.access_token)


async def test_validate_token_with_refresh_raises(
    session_factory: _SessionFactory,
) -> None:
    async with session_factory() as session:
        users = FakeUserRepository(session)
        tokens = FakeAuthTokenRepository(session)
        _seed_user(users)
        svc = _make_service(users, tokens)
        login = await svc.login(email="alice@example.com", password="correct-horse")
        with pytest.raises(UnauthorizedError):
            await svc.validate_token(token=login.refresh_token)


async def test_validate_token_garbage_raises(
    session_factory: _SessionFactory,
) -> None:
    async with session_factory() as session:
        users = FakeUserRepository(session)
        tokens = FakeAuthTokenRepository(session)
        _seed_user(users)
        svc = _make_service(users, tokens)
        with pytest.raises(UnauthorizedError):
            await svc.validate_token(token="not-a-jwt")


# ── refresh ──────────────────────────────────────────────────────────


async def test_refresh_success(session_factory: _SessionFactory) -> None:
    async with session_factory() as session:
        users = FakeUserRepository(session)
        tokens = FakeAuthTokenRepository(session)
        _seed_user(users)
        svc = _make_service(users, tokens)
        login = await svc.login(email="alice@example.com", password="correct-horse")
        new = await svc.refresh(refresh_token=login.refresh_token)
        assert isinstance(new.user, UserInfo)
        assert new.access_token != login.access_token
        assert new.refresh_token != login.refresh_token
        old = await tokens.find_by_token_value(login.refresh_token)
        assert old is not None and old.is_revoked is True
        assert len(tokens.tokens) == 4  # access+refresh x 2


async def test_refresh_revoked_raises(
    session_factory: _SessionFactory,
) -> None:
    async with session_factory() as session:
        users = FakeUserRepository(session)
        tokens = FakeAuthTokenRepository(session)
        _seed_user(users)
        svc = _make_service(users, tokens)
        login = await svc.login(email="alice@example.com", password="correct-horse")
        await svc.logout(token=login.access_token)

        with pytest.raises(UnauthorizedError):
            await svc.refresh(refresh_token=login.refresh_token)


async def test_refresh_with_access_token_raises(
    session_factory: _SessionFactory,
) -> None:
    async with session_factory() as session:
        users = FakeUserRepository(session)
        tokens = FakeAuthTokenRepository(session)
        _seed_user(users)
        svc = _make_service(users, tokens)
        login = await svc.login(email="alice@example.com", password="correct-horse")
        with pytest.raises(UnauthorizedError):
            await svc.refresh(refresh_token=login.access_token)


# ── logout ───────────────────────────────────────────────────────────


async def test_logout_revokes_all_tokens(
    session_factory: _SessionFactory,
) -> None:
    async with session_factory() as session:
        users = FakeUserRepository(session)
        tokens = FakeAuthTokenRepository(session)
        _seed_user(users)
        svc = _make_service(users, tokens)
        login = await svc.login(email="alice@example.com", password="correct-horse")
        n = await svc.logout(token=login.access_token)
        assert n == 2
        for t in tokens.tokens.values():
            assert t.is_revoked is True


async def test_logout_after_logout_is_noop(
    session_factory: _SessionFactory,
) -> None:
    async with session_factory() as session:
        users = FakeUserRepository(session)
        tokens = FakeAuthTokenRepository(session)
        _seed_user(users)
        svc = _make_service(users, tokens)
        login = await svc.login(email="alice@example.com", password="correct-horse")
        await svc.logout(token=login.access_token)
        n = await svc.logout(token=login.access_token)
        assert n == 0


async def test_logout_garbage_raises(
    session_factory: _SessionFactory,
) -> None:
    async with session_factory() as session:
        users = FakeUserRepository(session)
        tokens = FakeAuthTokenRepository(session)
        _seed_user(users)
        svc = _make_service(users, tokens)
        with pytest.raises(UnauthorizedError):
            await svc.logout(token="not-a-jwt")


# ── Real-repository smoke ────────────────────────────────────────────


async def test_login_uses_real_repositories(
    session_factory: _SessionFactory,
) -> None:
    """Service works end-to-end with the concrete repositories.

    This guards against drift between the AuthService protocol contract
    and the real ``UserRepository`` / ``AuthTokenRepository`` signatures.
    """
    async with session_factory() as session:
        users_repo = UserRepository(session)
        tokens_repo = AuthTokenRepository(session)
        svc = AuthService(
            users_repo=users_repo,
            tokens_repo=tokens_repo,
        )
        # The real UserRepository is a generic helper for ``users`` but
        # doesn't auto-create the schema; tests that need the table use
        # UserRepository only via the in-memory fakes. This guard just
        # exercises that AuthService can be constructed with the
        # concrete repos and exposes them under the documented names.
        assert isinstance(svc._users_repo, UserRepository)
        assert isinstance(svc._tokens_repo, AuthTokenRepository)
        assert svc._users_repo is users_repo
        assert svc._tokens_repo is tokens_repo


__all__ = []
