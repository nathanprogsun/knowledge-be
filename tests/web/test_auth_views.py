"""Web-layer tests for the auth router (login/refresh/logout).

Exercises the full HTTP path (routing, serialization, exception
handling) against the app with ``AuthService`` overridden to use
in-memory fake repos.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.app_context.lifespan import create_app
from src.core.auth.service import AuthService
from src.db.models.auth.auth_tokens import AuthToken
from src.db.models.auth.users import User
from src.util.security import hash_password
from src.web.deps import get_auth_service
from tests.fakes.auth_gates import override_auth_gates

# ── In-memory fakes ──────────────────────────────────────────────────


class _FakeUserRepo:
    """In-memory ``UserRepository`` replacement.

    Finders return storage ``User`` rows (the service projects them to
    ``UserInfo`` via ``UserInfo.map_from_db``), mirroring the real repo
    contract.
    """

    def __init__(self) -> None:
        self.users: dict[str, User] = {}

    async def find_by_email(self, email: str) -> User:
        for u in self.users.values():
            if u.email == email:
                return u
        from src.common.exception import NotFoundError

        raise NotFoundError(code="user.not_found", message=f"User {email} not found")

    async def find_by_id(self, user_id: str) -> User:
        user = self.users.get(user_id)
        if user is None:
            from src.common.exception import NotFoundError

            raise NotFoundError(code="user.not_found", message=f"User {user_id} not found")
        return user

    async def insert(self, row: User) -> User:
        self.users[row.id] = row
        return row


class _FakeTokenRepo:
    """In-memory ``AuthTokenRepository`` replacement."""

    def __init__(self) -> None:
        self.tokens: dict[str, AuthToken] = {}
        self._by_value: dict[str, str] = {}

    async def insert(self, row: AuthToken) -> AuthToken:
        self.tokens[row.id] = row
        self._by_value[row.token] = row.id
        return row

    async def find_by_token_value(self, token: str) -> AuthToken:
        tid = self._by_value.get(token)
        if tid is None:
            from src.common.exception import NotFoundError

            raise NotFoundError(code="token.not_found", message="Token not found")
        return self.tokens[tid]

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


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def fake_users() -> _FakeUserRepo:
    return _FakeUserRepo()


@pytest.fixture
def fake_tokens() -> _FakeTokenRepo:
    return _FakeTokenRepo()


@pytest.fixture
def app(fake_users: _FakeUserRepo, fake_tokens: _FakeTokenRepo) -> FastAPI:
    application = create_app()
    override_auth_gates(application)
    application.dependency_overrides[get_auth_service] = lambda: AuthService(
        users_repo=fake_users,  # type: ignore[arg-type]
        tokens_repo=fake_tokens,  # type: ignore[arg-type]
    )
    return application


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ── Helpers ─────────────────────────────────────────────────────────


def _seed_user(
    repo: _FakeUserRepo,
    *,
    email: str = "alice@example.com",
    password: str = "correct-horse",
    is_active: bool = True,
) -> User:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    user = User(
        id="usr-1",
        username="alice",
        email=email,
        password_hash=hash_password(password),
        avatar=None,
        tenant_id=7,
        is_active=is_active,
        can_access_all_tenants=False,
        is_system_admin=False,
        preferences={},
        created_at=now,
        updated_at=now,
    )
    repo.users[user.id] = user
    return user


# ── POST /auth/login ────────────────────────────────────────────────


async def test_login_success(client: AsyncClient, fake_users: _FakeUserRepo) -> None:
    _seed_user(fake_users)
    resp = await client.post(
        "/auth/login",
        json={"email": "alice@example.com", "password": "correct-horse"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["message"] == "Login successful"
    assert body["user"]["id"] == "usr-1"
    assert body["user"]["email"] == "alice@example.com"
    assert body["active_tenant"] is None
    assert body["memberships"] == []
    assert body["token"]
    assert body["refresh_token"]


async def test_login_wrong_password(client: AsyncClient, fake_users: _FakeUserRepo) -> None:
    _seed_user(fake_users)
    resp = await client.post(
        "/auth/login",
        json={"email": "alice@example.com", "password": "wrong"},
    )
    assert resp.status_code == 401
    body = resp.json()
    assert body["success"] is False
    assert body["error"]["code"] == "auth.invalid_credentials"


async def test_login_unknown_email(client: AsyncClient, fake_users: _FakeUserRepo) -> None:
    _seed_user(fake_users)
    resp = await client.post(
        "/auth/login",
        json={"email": "nobody@example.com", "password": "x"},
    )
    assert resp.status_code == 401


async def test_login_inactive_user(client: AsyncClient, fake_users: _FakeUserRepo) -> None:
    _seed_user(fake_users, is_active=False)
    resp = await client.post(
        "/auth/login",
        json={"email": "alice@example.com", "password": "correct-horse"},
    )
    assert resp.status_code == 401


# ── POST /auth/refresh ──────────────────────────────────────────────


async def test_refresh_success(client: AsyncClient, fake_users: _FakeUserRepo) -> None:
    _seed_user(fake_users)
    login = await client.post(
        "/auth/login",
        json={"email": "alice@example.com", "password": "correct-horse"},
    )
    refresh_token = login.json()["refresh_token"]

    resp = await client.post(
        "/auth/refresh",
        json={"refreshToken": refresh_token},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["access_token"]
    assert body["refresh_token"]
    assert body["refresh_token"] != refresh_token


async def test_refresh_with_access_token_fails(
    client: AsyncClient, fake_users: _FakeUserRepo
) -> None:
    _seed_user(fake_users)
    login = await client.post(
        "/auth/login",
        json={"email": "alice@example.com", "password": "correct-horse"},
    )
    access_token = login.json()["token"]

    resp = await client.post(
        "/auth/refresh",
        json={"refreshToken": access_token},
    )
    assert resp.status_code == 401


async def test_refresh_garbage_token(client: AsyncClient, fake_users: _FakeUserRepo) -> None:
    resp = await client.post(
        "/auth/refresh",
        json={"refreshToken": "not-a-jwt"},
    )
    assert resp.status_code == 401


# ── POST /auth/logout ───────────────────────────────────────────────


async def test_logout_success(client: AsyncClient, fake_users: _FakeUserRepo) -> None:
    _seed_user(fake_users)
    login = await client.post(
        "/auth/login",
        json={"email": "alice@example.com", "password": "correct-horse"},
    )
    token = login.json()["token"]

    resp = await client.post(
        "/auth/logout",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["message"] == "Logout successful"


async def test_logout_missing_header(client: AsyncClient) -> None:
    resp = await client.post("/auth/logout")
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"]["code"] == "auth.missing_authorization"


async def test_logout_invalid_header_format(client: AsyncClient) -> None:
    resp = await client.post(
        "/auth/logout",
        headers={"Authorization": "Basic abc"},
    )
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"]["code"] == "auth.invalid_authorization"


async def test_logout_garbage_token(client: AsyncClient, fake_users: _FakeUserRepo) -> None:
    _seed_user(fake_users)
    resp = await client.post(
        "/auth/logout",
        headers={"Authorization": "Bearer not-a-jwt"},
    )
    assert resp.status_code == 401


__all__ = []
