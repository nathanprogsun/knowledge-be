"""Web-layer tests for the added auth endpoints (register/me/validate/change-password).

Uses ``AsyncMock(spec=...)`` repositories configured with stateful closures
so the registered user can be looked up by email / id on subsequent
requests. The login flow still runs through the real JWT pipeline.

Uses the shared ``web_app`` fixture (header-based auth) and applies
the service dep override on it; the real ``require_auth`` dep resolves
the principal via the ``X-User-Id/X-Tenant-ID/X-Roles`` header trio.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.common.exception import ConflictError, NotFoundError
from src.core.auth.service import AuthService
from src.db.dao.auth_tokens_repository import AuthTokenRepository
from src.db.dao.users_repository import UserRepository
from src.db.models.auth.auth_tokens import AuthToken
from src.db.models.auth.users import User
from src.util.security import hash_password, verify_password
from src.web.deps import get_auth_service


@pytest.fixture
def users_repo() -> AsyncMock:
    repo = AsyncMock(spec=UserRepository)
    store: dict[str, User] = {}

    async def _insert(row: User) -> User:
        for u in store.values():
            if u.email == row.email or u.username == row.username:
                raise ConflictError(code="user.exists", message="User already exists")
        store[row.id] = row
        return row

    async def _find_by_email(email: str) -> User:
        for u in store.values():
            if u.email == email:
                return u
        raise NotFoundError(code="user.not_found", message=f"User {email} not found")

    async def _find_by_username(username: str) -> User:
        for u in store.values():
            if u.username == username:
                return u
        raise NotFoundError(code="user.not_found", message=f"User {username} not found")

    async def _find_by_id(user_id: str) -> User:
        user = store.get(user_id)
        if user is None:
            raise NotFoundError(code="user.not_found", message=f"User {user_id} not found")
        return user

    async def _update_by_primary_key(pk: dict[str, str], cols: dict[str, object]) -> User | None:
        user_id = pk.get("id")
        if user_id is None or user_id not in store:
            return None
        updated = store[user_id].model_copy(update=cols)
        store[user_id] = updated
        return updated

    repo.insert.side_effect = _insert
    repo.find_by_email.side_effect = _find_by_email
    repo.find_by_username.side_effect = _find_by_username
    repo.find_by_id.side_effect = _find_by_id
    repo.update_by_primary_key.side_effect = _update_by_primary_key
    repo._store = store  # type: ignore[attr-defined]
    return repo


@pytest.fixture
def tokens_repo() -> AsyncMock:
    repo = AsyncMock(spec=AuthTokenRepository)
    store: dict[str, AuthToken] = {}
    by_value: dict[str, str] = {}

    async def _insert(row: AuthToken) -> AuthToken:
        store[row.id] = row
        by_value[row.token] = row.id
        return row

    async def _find_by_token_value(token: str) -> AuthToken:
        tid = by_value.get(token)
        if tid is None:
            raise NotFoundError(code="token.not_found", message="Token not found")
        return store[tid]

    async def _revoke(token_id: str) -> int:
        t = store.get(token_id)
        if t is None or t.is_revoked:
            return 0
        store[token_id] = t.model_copy(update={"is_revoked": True})
        return 1

    async def _revoke_all_for_user(user_id: str) -> int:
        n = 0
        for tid, t in list(store.items()):
            if t.user_id == user_id and not t.is_revoked:
                store[tid] = t.model_copy(update={"is_revoked": True})
                n += 1
        return n

    repo.insert.side_effect = _insert
    repo.find_by_token_value.side_effect = _find_by_token_value
    repo.revoke.side_effect = _revoke
    repo.revoke_all_for_user.side_effect = _revoke_all_for_user
    repo._store = store  # type: ignore[attr-defined]
    return repo


@pytest.fixture(autouse=True)
def _override_services(
    web_app: FastAPI,
    users_repo: AsyncMock,
    tokens_repo: AsyncMock,
) -> FastAPI:
    """Override the auth service dep on the shared web app (autouse)."""
    web_app.dependency_overrides[get_auth_service] = lambda: AuthService(
        users_repo=users_repo,
        tokens_repo=tokens_repo,
    )
    return web_app


def _seed_user(
    users_repo: AsyncMock,
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
    users_repo._store[user.id] = user  # type: ignore[attr-defined]
    return user


async def _login(
    web_authed_client: TestClient,
    email: str = "alice@example.com",
    password: str = "correct-horse",
) -> str:
    resp = web_authed_client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert resp.status_code == 200
    return str(resp.json()["token"])


# ── POST /auth/register ───────────────────────────────────────────────


async def test_register_success(web_authed_client: TestClient, users_repo: AsyncMock) -> None:
    resp = web_authed_client.post(
        "/api/v1/auth/register",
        json={"username": "bob", "email": "bob@example.com", "password": "secret1"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["user"]["username"] == "bob"
    assert body["user"]["email"] == "bob@example.com"
    assert body["tenant"] is None


async def test_register_duplicate_email(
    web_authed_client: TestClient, users_repo: AsyncMock
) -> None:
    _seed_user(users_repo)
    resp = web_authed_client.post(
        "/api/v1/auth/register",
        json={"username": "alice2", "email": "alice@example.com", "password": "secret1"},
    )
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "user.exists"


async def test_register_empty_username(web_authed_client: TestClient) -> None:
    resp = web_authed_client.post(
        "/api/v1/auth/register",
        json={"username": "  ", "email": "x@example.com", "password": "secret1"},
    )
    assert resp.status_code == 422


# ── GET /auth/me ──────────────────────────────────────────────────────


async def test_me_success(web_authed_client: TestClient, users_repo: AsyncMock) -> None:
    _seed_user(users_repo)
    token = await _login(web_authed_client)
    resp = web_authed_client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["user"]["email"] == "alice@example.com"


async def test_me_missing_header(web_authed_client: TestClient) -> None:
    resp = web_authed_client.get("/api/v1/auth/me")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "auth.missing_authorization"


# ── GET /auth/validate ────────────────────────────────────────────────


async def test_validate_valid_token(web_authed_client: TestClient, users_repo: AsyncMock) -> None:
    _seed_user(users_repo)
    token = await _login(web_authed_client)
    resp = web_authed_client.get(
        "/api/v1/auth/validate", headers={"Authorization": f"Bearer {token}"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["user"]["email"] == "alice@example.com"


async def test_validate_invalid_token(web_authed_client: TestClient) -> None:
    resp = web_authed_client.get(
        "/api/v1/auth/validate", headers={"Authorization": "Bearer not-a-jwt"}
    )
    assert resp.status_code == 200
    assert resp.json()["success"] is True
    assert resp.json()["user"] is None


# ── POST /auth/change-password ────────────────────────────────────────


async def test_change_password_success(
    web_authed_client: TestClient, users_repo: AsyncMock
) -> None:
    _seed_user(users_repo)
    token = await _login(web_authed_client)
    resp = web_authed_client.post(
        "/api/v1/auth/change-password",
        headers={"Authorization": f"Bearer {token}"},
        json={"old_password": "correct-horse", "new_password": "new-secret"},
    )
    assert resp.status_code == 200
    assert resp.json()["success"] is True
    stored = users_repo._store["usr-1"]  # type: ignore[attr-defined]
    assert verify_password("new-secret", stored.password_hash)


async def test_change_password_wrong_old(
    web_authed_client: TestClient, users_repo: AsyncMock
) -> None:
    _seed_user(users_repo)
    token = await _login(web_authed_client)
    resp = web_authed_client.post(
        "/api/v1/auth/change-password",
        headers={"Authorization": f"Bearer {token}"},
        json={"old_password": "wrong", "new_password": "new-secret"},
    )
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "auth.invalid_credentials"


__all__ = []
