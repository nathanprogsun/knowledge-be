"""Web-layer tests for the added auth endpoints (register/me/validate/change-password).

Uses in-memory fake repos and a real JWT minted via the login flow, so
token-bearing endpoints exercise the actual decode + user lookup path.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.app_context.lifespan import create_app
from src.common.exception import NotFoundError
from src.core.auth.service import AuthService
from src.db.models.auth.auth_tokens import AuthToken
from src.db.models.auth.users import User
from src.util.security import hash_password, verify_password
from src.web.deps import get_auth_service
from tests.unit.fakes.auth_gates import override_auth_gates


class _FakeUserRepo:
    def __init__(self) -> None:
        self.users: dict[str, User] = {}

    async def find_by_email(self, email: str) -> User:
        for u in self.users.values():
            if u.email == email:
                return u
        raise NotFoundError(code="user.not_found", message=f"User {email} not found")

    async def find_by_username(self, username: str) -> User:
        for u in self.users.values():
            if u.username == username:
                return u
        raise NotFoundError(code="user.not_found", message=f"User {username} not found")

    async def find_by_id(self, user_id: str) -> User:
        user = self.users.get(user_id)
        if user is None:
            raise NotFoundError(code="user.not_found", message=f"User {user_id} not found")
        return user

    async def insert(self, row: User) -> User:
        if any(u.email == row.email for u in self.users.values()) or any(
            u.username == row.username for u in self.users.values()
        ):
            from src.common.exception import ConflictError

            raise ConflictError(code="user.exists", message="User already exists")
        self.users[row.id] = row
        return row

    async def update_by_primary_key(
        self, pk: dict[str, str], cols: dict[str, object]
    ) -> User | None:
        user_id = pk.get("id")
        if user_id is None or user_id not in self.users:
            return None
        updated = self.users[user_id].model_copy(update=cols)
        self.users[user_id] = updated
        return updated


class _FakeTokenRepo:
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
            raise NotFoundError(code="token.not_found", message="Token not found")
        return self.tokens[tid]

    async def revoke(self, token_id: str) -> int:
        t = self.tokens.get(token_id)
        if t is None or t.is_revoked:
            return 0
        self.tokens[token_id] = t.model_copy(update={"is_revoked": True})
        return 1

    async def revoke_all_for_user(self, user_id: str) -> int:
        n = 0
        for tid, t in list(self.tokens.items()):
            if t.user_id == user_id and not t.is_revoked:
                self.tokens[tid] = t.model_copy(update={"is_revoked": True})
                n += 1
        return n


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


def _seed_user(
    repo: _FakeUserRepo,
    *,
    email: str = "alice@example.com",
    username: str = "alice",
    password: str = "correct-horse",
    is_active: bool = True,
) -> User:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    user = User(
        id="usr-1",
        username=username,
        email=email,
        password_hash=hash_password(password),
        avatar=None,
        tenant_id=None,
        is_active=is_active,
        can_access_all_tenants=False,
        is_system_admin=False,
        preferences={},
        created_at=now,
        updated_at=now,
    )
    repo.users[user.id] = user
    return user


async def _login(
    client: AsyncClient, email: str = "alice@example.com", password: str = "correct-horse"
) -> str:
    resp = await client.post("/auth/login", json={"email": email, "password": password})
    assert resp.status_code == 200
    return str(resp.json()["token"])


# ── POST /auth/register ───────────────────────────────────────────────


async def test_register_success(client: AsyncClient, fake_users: _FakeUserRepo) -> None:
    resp = await client.post(
        "/auth/register",
        json={"username": "bob", "email": "bob@example.com", "password": "secret1"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["user"]["username"] == "bob"
    assert body["user"]["email"] == "bob@example.com"
    assert body["active_tenant"] is None


async def test_register_duplicate_email(client: AsyncClient, fake_users: _FakeUserRepo) -> None:
    _seed_user(fake_users)
    resp = await client.post(
        "/auth/register",
        json={"username": "alice2", "email": "alice@example.com", "password": "secret1"},
    )
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "user.exists"


async def test_register_empty_username(client: AsyncClient) -> None:
    resp = await client.post(
        "/auth/register",
        json={"username": "  ", "email": "x@example.com", "password": "secret1"},
    )
    assert resp.status_code == 422


# ── GET /auth/me ──────────────────────────────────────────────────────


async def test_me_success(client: AsyncClient, fake_users: _FakeUserRepo) -> None:
    _seed_user(fake_users)
    token = await _login(client)
    resp = await client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["user"]["email"] == "alice@example.com"


async def test_me_missing_header(client: AsyncClient) -> None:
    resp = await client.get("/auth/me")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "auth.missing_authorization"


# ── GET /auth/validate ────────────────────────────────────────────────


async def test_validate_valid_token(client: AsyncClient, fake_users: _FakeUserRepo) -> None:
    _seed_user(fake_users)
    token = await _login(client)
    resp = await client.get("/auth/validate", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["valid"] is True
    assert body["user_id"] == "usr-1"


async def test_validate_invalid_token(client: AsyncClient) -> None:
    resp = await client.get("/auth/validate", headers={"Authorization": "Bearer not-a-jwt"})
    assert resp.status_code == 200
    assert resp.json()["valid"] is False


# ── POST /auth/change-password ────────────────────────────────────────


async def test_change_password_success(client: AsyncClient, fake_users: _FakeUserRepo) -> None:
    _seed_user(fake_users)
    token = await _login(client)
    resp = await client.post(
        "/auth/change-password",
        headers={"Authorization": f"Bearer {token}"},
        json={"old_password": "correct-horse", "new_password": "new-secret"},
    )
    assert resp.status_code == 200
    assert resp.json()["success"] is True
    stored = fake_users.users["usr-1"]
    assert verify_password("new-secret", stored.password_hash)


async def test_change_password_wrong_old(client: AsyncClient, fake_users: _FakeUserRepo) -> None:
    _seed_user(fake_users)
    token = await _login(client)
    resp = await client.post(
        "/auth/change-password",
        headers={"Authorization": f"Bearer {token}"},
        json={"old_password": "wrong", "new_password": "new-secret"},
    )
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "auth.invalid_credentials"


__all__ = []
