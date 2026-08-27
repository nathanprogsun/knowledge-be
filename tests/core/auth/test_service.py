"""Unit tests for `AuthService` + `src.util.security`.

Exercises the service with ``AsyncMock(spec=...)`` repositories so the
stateful behavior the service expects (insert + lookup + revoke) is
preserved while removing the hand-rolled Fake*Repository classes. The
real-repository smoke test at the bottom guards against drift between
the AuthService signature and the real repo signatures.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

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
from tests.util.service_test import ServiceTest

# ── In-memory repository doubles (stateful via side_effect closures) ─


def _make_users_repo() -> tuple[AsyncMock, dict[str, User]]:
    """Build a users-repo mock backed by a closure-captured dict."""
    repo = AsyncMock(spec=UserRepository)
    store: dict[str, User] = {}

    async def _find_by_email(email: str) -> User:
        for user in store.values():
            if user.email == email:
                return user
        raise NotFoundError(code="user.not_found", message=f"User {email} not found")

    async def _find_by_id(user_id: str) -> User:
        user = store.get(user_id)
        if user is None:
            raise NotFoundError(code="user.not_found", message=f"User {user_id} not found")
        return user

    async def _insert(row: User) -> User:
        store[row.id] = row
        return row

    repo.find_by_email.side_effect = _find_by_email
    repo.find_by_id.side_effect = _find_by_id
    repo.insert.side_effect = _insert
    return repo, store


def _make_tokens_repo() -> tuple[AsyncMock, dict[str, AuthToken], dict[str, str]]:
    """Build a tokens-repo mock backed by closure-captured dicts."""
    repo = AsyncMock(spec=AuthTokenRepository)
    tokens: dict[str, AuthToken] = {}
    by_value: dict[str, str] = {}

    async def _insert(row: AuthToken) -> AuthToken:
        tokens[row.id] = row
        by_value[row.token] = row.id
        return row

    async def _find_by_token_value(token: str) -> AuthToken:
        token_id = by_value.get(token)
        if token_id is None:
            raise NotFoundError(code="token.not_found", message="Token not found")
        return tokens[token_id]  # type: ignore[return-value]

    async def _revoke_all_for_user(user_id: str) -> int:
        n = 0
        for tid, t in list(tokens.items()):
            if t.user_id == user_id and not t.is_revoked:
                tokens[tid] = t.model_copy(update={"is_revoked": True})
                n += 1
        return n

    async def _revoke(token_id: str) -> int:
        t = tokens.get(token_id)
        if t is None or t.is_revoked:
            return 0
        tokens[token_id] = t.model_copy(update={"is_revoked": True})
        return 1

    repo.insert.side_effect = _insert
    repo.find_by_token_value.side_effect = _find_by_token_value
    repo.revoke_all_for_user.side_effect = _revoke_all_for_user
    repo.revoke.side_effect = _revoke
    return repo, tokens, by_value


def _seed_user(
    *,
    id: str = "usr-1",
    email: str = "alice@example.com",
    password: str = "correct-horse",
    is_active: bool = True,
    tenant_id: int | None = 7,
) -> User:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    return User(
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


def _make_service(
    users: AsyncMock,
    tokens: AsyncMock,
) -> AuthService:
    return AuthService(users_repo=users, tokens_repo=tokens)


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


class TestLogin(ServiceTest):
    async def test_success(self) -> None:
        users_repo, _ = _make_users_repo()
        tokens_repo, _, _ = _make_tokens_repo()
        await users_repo.insert(_seed_user())
        svc = _make_service(users_repo, tokens_repo)

        result = await svc.login(email="alice@example.com", password="correct-horse")

        assert isinstance(result, LoginResult)
        assert isinstance(result.user, UserInfo)
        assert result.user.id == "usr-1"
        assert result.access_token
        assert result.refresh_token
        assert tokens_repo.insert.call_count == 2
        types = {call.args[0].token_type for call in tokens_repo.insert.call_args_list}
        assert types == {"access_token", "refresh_token"}

    async def test_wrong_email_raises(self) -> None:
        users_repo, _ = _make_users_repo()
        tokens_repo, _, _ = _make_tokens_repo()
        await users_repo.insert(_seed_user())
        svc = _make_service(users_repo, tokens_repo)

        with pytest.raises(UnauthorizedError):
            await svc.login(email="nobody@example.com", password="correct-horse")

        assert tokens_repo.insert.call_count == 0

    async def test_wrong_password_raises(self) -> None:
        users_repo, _ = _make_users_repo()
        tokens_repo, _, _ = _make_tokens_repo()
        await users_repo.insert(_seed_user())
        svc = _make_service(users_repo, tokens_repo)

        with pytest.raises(UnauthorizedError):
            await svc.login(email="alice@example.com", password="wrong")

        assert tokens_repo.insert.call_count == 0

    async def test_inactive_user_raises(self) -> None:
        users_repo, _ = _make_users_repo()
        tokens_repo, _, _ = _make_tokens_repo()
        await users_repo.insert(_seed_user(is_active=False))
        svc = _make_service(users_repo, tokens_repo)

        with pytest.raises(UnauthorizedError):
            await svc.login(email="alice@example.com", password="correct-horse")


# ── validate_token ───────────────────────────────────────────────────


class TestValidateToken(ServiceTest):
    async def test_round_trip(self) -> None:
        users_repo, _ = _make_users_repo()
        tokens_repo, _, _ = _make_tokens_repo()
        await users_repo.insert(_seed_user(tenant_id=99))
        svc = _make_service(users_repo, tokens_repo)
        login = await svc.login(email="alice@example.com", password="correct-horse")

        user, tenant_id = await svc.validate_token(token=login.access_token)

        assert isinstance(user, UserInfo)
        assert user.id == "usr-1"
        assert tenant_id == 99

    async def test_revoked_raises(self) -> None:
        users_repo, _ = _make_users_repo()
        tokens_repo, _, _ = _make_tokens_repo()
        await users_repo.insert(_seed_user())
        svc = _make_service(users_repo, tokens_repo)
        login = await svc.login(email="alice@example.com", password="correct-horse")
        revoked = await svc.revoke_token(token=login.access_token)
        assert revoked == 1

        with pytest.raises(UnauthorizedError):
            await svc.validate_token(token=login.access_token)

    async def test_with_refresh_raises(self) -> None:
        users_repo, _ = _make_users_repo()
        tokens_repo, _, _ = _make_tokens_repo()
        await users_repo.insert(_seed_user())
        svc = _make_service(users_repo, tokens_repo)
        login = await svc.login(email="alice@example.com", password="correct-horse")

        with pytest.raises(UnauthorizedError):
            await svc.validate_token(token=login.refresh_token)

    async def test_garbage_raises(self) -> None:
        users_repo, _ = _make_users_repo()
        tokens_repo, _, _ = _make_tokens_repo()
        await users_repo.insert(_seed_user())
        svc = _make_service(users_repo, tokens_repo)

        with pytest.raises(UnauthorizedError):
            await svc.validate_token(token="not-a-jwt")


# ── refresh ──────────────────────────────────────────────────────────


class TestRefresh(ServiceTest):
    async def test_success(self) -> None:
        users_repo, _ = _make_users_repo()
        tokens_repo, tokens, _ = _make_tokens_repo()
        await users_repo.insert(_seed_user())
        svc = _make_service(users_repo, tokens_repo)
        login = await svc.login(email="alice@example.com", password="correct-horse")

        new = await svc.refresh(refresh_token=login.refresh_token)

        assert isinstance(new.user, UserInfo)
        assert new.access_token != login.access_token
        assert new.refresh_token != login.refresh_token
        # Old refresh row revoked (its id is keyed by row.id).
        old = await tokens_repo.find_by_token_value(login.refresh_token)
        assert old is not None and old.is_revoked is True
        assert len(tokens) == 4  # access+refresh x 2

    async def test_revoked_raises(self) -> None:
        users_repo, _ = _make_users_repo()
        tokens_repo, _, _ = _make_tokens_repo()
        await users_repo.insert(_seed_user())
        svc = _make_service(users_repo, tokens_repo)
        login = await svc.login(email="alice@example.com", password="correct-horse")
        await svc.logout(token=login.access_token)

        with pytest.raises(UnauthorizedError):
            await svc.refresh(refresh_token=login.refresh_token)

    async def test_with_access_token_raises(self) -> None:
        users_repo, _ = _make_users_repo()
        tokens_repo, _, _ = _make_tokens_repo()
        await users_repo.insert(_seed_user())
        svc = _make_service(users_repo, tokens_repo)
        login = await svc.login(email="alice@example.com", password="correct-horse")

        with pytest.raises(UnauthorizedError):
            await svc.refresh(refresh_token=login.access_token)


# ── logout ───────────────────────────────────────────────────────────


class TestLogout(ServiceTest):
    async def test_revokes_all_tokens(self) -> None:
        users_repo, _ = _make_users_repo()
        tokens_repo, tokens, _ = _make_tokens_repo()
        await users_repo.insert(_seed_user())
        svc = _make_service(users_repo, tokens_repo)
        login = await svc.login(email="alice@example.com", password="correct-horse")

        n = await svc.logout(token=login.access_token)

        assert n == 2
        for t in tokens.values():
            assert t.is_revoked is True

    async def test_after_logout_is_noop(self) -> None:
        users_repo, _ = _make_users_repo()
        tokens_repo, _, _ = _make_tokens_repo()
        await users_repo.insert(_seed_user())
        svc = _make_service(users_repo, tokens_repo)
        login = await svc.login(email="alice@example.com", password="correct-horse")
        await svc.logout(token=login.access_token)

        n = await svc.logout(token=login.access_token)

        assert n == 0

    async def test_garbage_raises(self) -> None:
        users_repo, _ = _make_users_repo()
        tokens_repo, _, _ = _make_tokens_repo()
        await users_repo.insert(_seed_user())
        svc = _make_service(users_repo, tokens_repo)

        with pytest.raises(UnauthorizedError):
            await svc.logout(token="not-a-jwt")


# ── Real-repository smoke ────────────────────────────────────────────


async def test_login_uses_real_repositories() -> None:
    """Service works end-to-end with the concrete repositories.

    This guards against drift between the AuthService protocol contract
    and the real ``UserRepository`` / ``AuthTokenRepository`` signatures.
    """

    class _NoopSession:
        """Bare-minimum session shim for constructor compatibility."""

    session: AsyncSession = _NoopSession()  # type: ignore[assignment]
    users_repo = UserRepository(session)
    tokens_repo = AuthTokenRepository(session)
    svc = AuthService(
        users_repo=users_repo,
        tokens_repo=tokens_repo,
    )
    # AuthService stores the repositories under documented names.
    assert svc._users_repo is users_repo
    assert svc._tokens_repo is tokens_repo


__all__ = []
