"""Web-layer tests for the auth router (login/refresh/logout).

Exercises the full HTTP path (routing, serialization, exception
handling) against the app with ``AuthService`` overridden to use
``AsyncMock(spec=...)`` repositories configured with stateful closures.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from unittest.mock import AsyncMock

from src.common.exception import NotFoundError
from src.core.auth.service import AuthService
from src.db.dao.auth_tokens_repository import AuthTokenRepository
from src.db.dao.users_repository import UserRepository
from src.db.models.auth.auth_tokens import AuthToken
from src.db.models.auth.users import User
from src.util.security import hash_password
from src.web.deps import get_auth_service
from tests.util.service_test import lookup_by, stateful_insert

# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def users_repo() -> AsyncMock:
    """``AsyncMock(spec=UserRepository)`` with stateful ``insert`` + finders."""
    repo = AsyncMock(spec=UserRepository)
    store: dict[str, User] = {}
    stateful_insert(repo, store)

    async def _find_by_email(email: str) -> User:
        for u in store.values():
            if u.email == email:
                return u
        raise NotFoundError(code="user.not_found", message=f"User {email} not found")

    async def _find_by_id(user_id: str) -> User:
        user = store.get(user_id)
        if user is None:
            raise NotFoundError(code="user.not_found", message=f"User {user_id} not found")
        return user

    repo.find_by_email.side_effect = _find_by_email
    repo.find_by_id.side_effect = _find_by_id
    repo._store = store  # type: ignore[attr-defined]
    return repo


@pytest.fixture
def tokens_repo() -> AsyncMock:
    """``AsyncMock(spec=AuthTokenRepository)`` with stateful ``insert`` + finders."""
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
    web_app: FastAPI,  # noqa: ARG001 - resolved from the parent conftest
    users_repo: AsyncMock,
    tokens_repo: AsyncMock,
) -> FastAPI:
    """Override the auth service dep on the shared web app (autouse)."""
    web_app.dependency_overrides[get_auth_service] = lambda: AuthService(
        users_repo=users_repo,
        tokens_repo=tokens_repo,
    )
    return web_app


# ── Helpers ─────────────────────────────────────────────────────────


def _seed_user(
    users_repo: AsyncMock,
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
    users_repo._store[user.id] = user  # type: ignore[attr-defined]
    return user


# ── POST /auth/login ────────────────────────────────────────────────


async def test_login_success(
    web_authed_client: AsyncClient, users_repo: AsyncMock
) -> None:
    _seed_user(users_repo)
    resp = await web_authed_client.post(
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


async def test_login_wrong_password(
    web_authed_client: AsyncClient, users_repo: AsyncMock
) -> None:
    _seed_user(users_repo)
    resp = await web_authed_client.post(
        "/auth/login",
        json={"email": "alice@example.com", "password": "wrong"},
    )
    assert resp.status_code == 401
    body = resp.json()
    assert body["success"] is False
    assert body["error"]["code"] == "auth.invalid_credentials"


async def test_login_unknown_email(
    web_authed_client: AsyncClient, users_repo: AsyncMock
) -> None:
    _seed_user(users_repo)
    resp = await web_authed_client.post(
        "/auth/login",
        json={"email": "nobody@example.com", "password": "x"},
    )
    assert resp.status_code == 401


async def test_login_inactive_user(
    web_authed_client: AsyncClient, users_repo: AsyncMock
) -> None:
    _seed_user(users_repo, is_active=False)
    resp = await web_authed_client.post(
        "/auth/login",
        json={"email": "alice@example.com", "password": "correct-horse"},
    )
    assert resp.status_code == 401


# ── POST /auth/refresh ──────────────────────────────────────────────


async def test_refresh_success(
    web_authed_client: AsyncClient, users_repo: AsyncMock
) -> None:
    _seed_user(users_repo)
    login = await web_authed_client.post(
        "/auth/login",
        json={"email": "alice@example.com", "password": "correct-horse"},
    )
    refresh_token = login.json()["refresh_token"]

    resp = await web_authed_client.post(
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
    web_authed_client: AsyncClient, users_repo: AsyncMock
) -> None:
    _seed_user(users_repo)
    login = await web_authed_client.post(
        "/auth/login",
        json={"email": "alice@example.com", "password": "correct-horse"},
    )
    access_token = login.json()["token"]

    resp = await web_authed_client.post(
        "/auth/refresh",
        json={"refreshToken": access_token},
    )
    assert resp.status_code == 401


async def test_refresh_garbage_token(web_authed_client: AsyncClient) -> None:
    resp = await web_authed_client.post(
        "/auth/refresh",
        json={"refreshToken": "not-a-jwt"},
    )
    assert resp.status_code == 401


# ── POST /auth/logout ───────────────────────────────────────────────


async def test_logout_success(
    web_authed_client: AsyncClient, users_repo: AsyncMock
) -> None:
    _seed_user(users_repo)
    login = await web_authed_client.post(
        "/auth/login",
        json={"email": "alice@example.com", "password": "correct-horse"},
    )
    token = login.json()["token"]

    resp = await web_authed_client.post(
        "/auth/logout",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["message"] == "Logout successful"


async def test_logout_missing_header(web_authed_client: AsyncClient) -> None:
    resp = await web_authed_client.post("/auth/logout")
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"]["code"] == "auth.missing_authorization"


async def test_logout_invalid_header_format(web_authed_client: AsyncClient) -> None:
    resp = await web_authed_client.post(
        "/auth/logout",
        headers={"Authorization": "Basic abc"},
    )
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"]["code"] == "auth.invalid_authorization"


async def test_logout_garbage_token(
    web_authed_client: AsyncClient, users_repo: AsyncMock
) -> None:
    _seed_user(users_repo)
    resp = await web_authed_client.post(
        "/auth/logout",
        headers={"Authorization": "Bearer not-a-jwt"},
    )
    assert resp.status_code == 401


__all__ = []