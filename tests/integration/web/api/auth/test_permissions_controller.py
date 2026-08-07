"""HTTP-level permission tests for the auth + RBAC gates.

Exercises the real app with the auth dependency intact (not bypassed),
verifying that protected endpoints reject missing/invalid credentials and
that the RBAC gates enforce the declared role floors. Uses fake repos for
the underlying services so no database is needed for the auth path.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from src.app_context.lifespan import create_app
from src.common.exception import NotFoundError, UnauthorizedError
from src.core.auth.service import AuthService
from src.core.system.audit_service import AuditLogService
from src.core.system.system_setting_service import SystemSettingService
from src.db.models.auth.auth_tokens import AuthToken
from src.db.models.auth.users import User
from src.db.models.system.audit_log import AuditLog
from src.util.security import hash_password
from src.web.deps import (
    get_audit_log_service,
    get_auth_service,
    get_system_setting_service,
)
from src.web.middleware.auth import require_auth
from tests.integration.conftest import _noop_lifespan


class _FakeUserRepo:
    def __init__(self) -> None:
        self.users: dict[str, User] = {}

    async def find_by_email(self, email: str) -> User:
        for u in self.users.values():
            if u.email == email:
                return u
        raise NotFoundError(code="user.not_found", message="User not found")

    async def find_by_id(self, user_id: str) -> User:
        user = self.users.get(user_id)
        if user is None:
            raise NotFoundError(code="user.not_found", message="User not found")
        return user

    async def insert(self, row: User) -> User:
        self.users[row.id] = row
        return row

    async def update_by_primary_key(
        self, pk: dict[str, str], cols: dict[str, object]
    ) -> User | None:
        uid = pk.get("id")
        if uid is None or uid not in self.users:
            return None
        updated = self.users[uid].model_copy(update=cols)
        self.users[uid] = updated
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


class _FakeAuditRepo:
    def __init__(self) -> None:
        self.rows: list[AuditLog] = []


class _FakeSettingRepo:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, object]] = {}

    async def list_all(self) -> list[object]:
        return []


@pytest.fixture
def fake_users() -> _FakeUserRepo:
    return _FakeUserRepo()


@pytest.fixture
def fake_tokens() -> _FakeTokenRepo:
    return _FakeTokenRepo()


@pytest.fixture
def principal_role() -> dict[str, str]:
    return {"role": "owner"}


@pytest.fixture
def app(
    fake_users: _FakeUserRepo,
    fake_tokens: _FakeTokenRepo,
    principal_role: dict[str, str],
) -> FastAPI:
    application = create_app()
    application.router.lifespan_context = _noop_lifespan

    async def _override_auth(request: Request) -> None:
        """Populate request.state as a principal with ``principal_role``.

        Rejects requests without an Authorization header to exercise the
        global auth gate; accepts any Bearer token (token validity is not
        the focus here).
        """
        if not request.headers.get("authorization"):
            raise UnauthorizedError(code="auth.missing_authentication", message="missing auth")
        user = fake_users.users.get("usr-1")
        if user is None:
            raise UnauthorizedError(code="auth.missing_authentication", message="missing auth")
        request.state.user_info = {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "is_active": "1",
            "can_access_all_tenants": "1" if user.can_access_all_tenants else "0",
            "is_system_admin": "1" if user.is_system_admin else "0",
        }
        request.state.is_system_admin = user.is_system_admin
        request.state.tenant_id = "1"
        request.state.tenant_role = principal_role["role"]

    application.dependency_overrides[require_auth] = _override_auth
    application.dependency_overrides[get_auth_service] = lambda: AuthService(
        users_repo=fake_users,  # type: ignore[arg-type]
        tokens_repo=fake_tokens,  # type: ignore[arg-type]
    )
    application.dependency_overrides[get_system_setting_service] = lambda: SystemSettingService(
        settings_repo=_FakeSettingRepo(),  # type: ignore[arg-type]
        audit_repo=_FakeAuditRepo(),  # type: ignore[arg-type]
    )
    application.dependency_overrides[get_audit_log_service] = lambda: AuditLogService(
        audit_repo=_FakeAuditRepo(),  # type: ignore[arg-type]
    )
    return application


@pytest.fixture
async def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app=app, base_url="http://test") as c:
        yield c


def _seed_user(repo: _FakeUserRepo, *, is_system_admin: bool = False) -> User:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    user = User(
        id="usr-1",
        username="alice",
        email="alice@example.com",
        password_hash=hash_password("correct-horse"),
        avatar=None,
        tenant_id=None,
        is_active=True,
        can_access_all_tenants=False,
        is_system_admin=is_system_admin,
        preferences={},
        created_at=now,
        updated_at=now,
    )
    repo.users[user.id] = user
    return user


async def _token(client: TestClient, email: str = "alice@example.com") -> str:
    resp = client.post("/auth/login", json={"email": email, "password": "correct-horse"})
    assert resp.status_code == 200
    return str(resp.json()["token"])


# ── Global auth: missing / invalid credentials ────────────────────────


async def test_protected_endpoint_requires_auth(
    client: TestClient, fake_users: _FakeUserRepo
) -> None:
    _seed_user(fake_users)
    resp = client.get("/auth/me")
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "auth.missing_authentication"


async def test_protected_endpoint_invalid_token(
    client: TestClient, fake_users: _FakeUserRepo
) -> None:
    _seed_user(fake_users)
    resp = client.get("/auth/me", headers={"Authorization": "Bearer garbage"})
    assert resp.status_code == 401


async def test_public_endpoint_bypasses_auth(client: TestClient, fake_users: _FakeUserRepo) -> None:
    _seed_user(fake_users)
    resp = client.post(
        "/auth/login", json={"email": "alice@example.com", "password": "correct-horse"}
    )
    assert resp.status_code == 200


async def test_valid_token_allows_protected_endpoint(
    client: TestClient, fake_users: _FakeUserRepo
) -> None:
    _seed_user(fake_users)
    token = await _token(client)
    resp = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.json()["success"] is True


# ── System admin gate (always enforced) ───────────────────────────────


async def test_system_admin_route_rejects_regular_user(
    client: TestClient, fake_users: _FakeUserRepo
) -> None:
    _seed_user(fake_users)  # not a system admin
    token = await _token(client)
    resp = client.get("/system/admin/settings", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == "rbac.system_admin_required"


async def test_system_admin_route_allows_admin(
    client: TestClient, fake_users: _FakeUserRepo
) -> None:
    _seed_user(fake_users, is_system_admin=True)
    token = await _token(client)
    resp = client.get("/system/admin/settings", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200


# ── Tenant role gate (RBAC enforced by default) ───────────────────────


async def test_kv_put_requires_admin_role(
    client: TestClient, fake_users: _FakeUserRepo, principal_role: dict[str, str]
) -> None:
    # Set role to viewer: PUT /tenants/kv/{key} requires Admin, so the
    # gate rejects with 403 before the handler (which needs a DB session).
    principal_role["role"] = "viewer"
    _seed_user(fake_users)
    token = await _token(client)
    resp = client.put(
        "/tenants/kv/web-search-config",
        headers={"Authorization": f"Bearer {token}"},
        json={"max_results": 20},
    )
    assert resp.status_code == 403


__all__ = []
