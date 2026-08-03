"""Unit tests for ``src.core.auth.oidc.OidcService``.

Covers the PR-4 scope: authorization-URL building + signed-state
round-trip, and the existing-user bind path of ``login_with_oidc``.
New-user provisioning is asserted to raise
``ExternalServiceError(code="oidc.provisioning_unavailable")`` - it is
deferred until the Register flow + tenant services land.

The ``OidcClient`` is replaced by an in-memory fake so no HTTP is hit;
the real ``UserRepository`` / ``AuthTokenRepository`` are also faked
(mirroring ``tests/core/auth/test_service.py``). ``mint_token_pair`` runs
for real (real JWT signing against a test ``JWT_SECRET_KEY``) so the
issued access/refresh tokens are verifiable end-to-end.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

import pytest

from src.ai.oidc_client import OIDCTokenResponse, OIDCUserInfoClaims
from src.common.exception import (
    ExternalServiceError,
    NotFoundError,
    PermissionDeniedError,
    UnauthorizedError,
    ValidationError,
)
from src.core.auth.oidc import (
    OidcService,
    OIDCStatePayload,
    _sign_state,
    _verify_state,
)
from src.db.models.auth.auth_tokens import AuthToken
from src.db.models.auth.users import User
from src.settings import reset_settings_cache
from src.util.security import decode_token, hash_password, reset_secret_cache

# ── Fakes ────────────────────────────────────────────────────────────


class _FakeUsersRepo:
    """Minimal stand-in for ``UserRepository`` (``find_by_email`` only)."""

    def __init__(self) -> None:
        self.users: dict[str, User] = {}

    async def find_by_email(self, email: str) -> User:
        for user in self.users.values():
            if user.email == email:
                return user
        raise NotFoundError(code="user.not_found", message=f"User {email} not found")


class _FakeTokensRepo:
    """Minimal stand-in for ``AuthTokenRepository`` (``insert`` only)."""

    def __init__(self) -> None:
        self.inserted: list[AuthToken] = []

    async def insert(self, row: AuthToken) -> AuthToken:
        self.inserted.append(row)
        return row


class _FakeOidcClient:
    """In-memory ``OidcClient`` replacement.

    ``exchange_code`` / ``resolve_userinfo`` return the preloaded
    responses; ``discover_endpoints`` is unused (endpoints come from
    settings in the tests).
    """

    def __init__(
        self,
        *,
        token: OIDCTokenResponse,
        userinfo: OIDCUserInfoClaims,
    ) -> None:
        self._token = token
        self._userinfo = userinfo

    async def exchange_code(
        self,
        *,
        token_endpoint: str,
        client_id: str,
        client_secret: str,
        code: str,
        redirect_uri: str,
    ) -> OIDCTokenResponse:
        return self._token

    async def resolve_userinfo(
        self,
        *,
        user_info_endpoint: str,
        access_token: str,
        id_token: str,
        username_claim: str,
        email_claim: str,
    ) -> OIDCUserInfoClaims:
        return self._userinfo


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _oidc_env(monkeypatch: pytest.MonkeyPatch) -> object:
    """Enable OIDC + endpoints + a stable JWT secret for state/token signing."""
    monkeypatch.setenv("OIDC_ENABLE", "true")
    monkeypatch.setenv("OIDC_CLIENT_ID", "test-client")
    monkeypatch.setenv("OIDC_CLIENT_SECRET", "test-secret")
    monkeypatch.setenv("OIDC_PROVIDER_DISPLAY_NAME", "Test IdP")
    monkeypatch.setenv("OIDC_AUTHORIZATION_ENDPOINT", "https://idp.example.com/authorize")
    monkeypatch.setenv("OIDC_TOKEN_ENDPOINT", "https://idp.example.com/token")
    monkeypatch.setenv("OIDC_USER_INFO_ENDPOINT", "https://idp.example.com/userinfo")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-jwt-secret")
    reset_settings_cache()
    reset_secret_cache()
    yield
    reset_settings_cache()
    reset_secret_cache()


def _seed_user(
    users: _FakeUsersRepo,
    *,
    id: str = "usr-1",
    email: str = "alice@example.com",
    is_active: bool = True,
) -> User:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    user = User(
        id=id,
        username="alice",
        email=email,
        password_hash=hash_password("anything"),
        avatar=None,
        tenant_id=7,
        is_active=is_active,
        can_access_all_tenants=False,
        is_system_admin=False,
        preferences={},
        created_at=now,
        updated_at=now,
    )
    users.users[user.id] = user
    return user


def _make_service(
    users: _FakeUsersRepo,
    tokens: _FakeTokensRepo,
    client: _FakeOidcClient,
) -> OidcService:
    return OidcService(
        users_repo=users,  # type: ignore[arg-type]
        tokens_repo=tokens,  # type: ignore[arg-type]
        oidc_client=client,  # type: ignore[arg-type]
    )


def _userinfo(email: str, *, username: str = "Alice") -> OIDCUserInfoClaims:
    return OIDCUserInfoClaims(
        subject="sub-1",
        username=username,
        email=email,
        claims={"sub": "sub-1", "email": email, "name": username},
    )


# ── get_authorization_url ────────────────────────────────────────────


async def test_get_authorization_url_builds_query_and_state() -> None:
    users = _FakeUsersRepo()
    tokens = _FakeTokensRepo()
    client = _FakeOidcClient(
        token=OIDCTokenResponse(access_token="at", id_token="", token_type="Bearer"),
        userinfo=_userinfo("alice@example.com"),
    )
    service = _make_service(users, tokens, client)

    result = await service.get_authorization_url(redirect_uri="https://app.example.com/cb")

    assert result.provider_display_name == "Test IdP"
    assert "https://idp.example.com/authorize?" in result.authorization_url
    assert "response_type=code" in result.authorization_url
    assert "client_id=test-client" in result.authorization_url
    assert "redirect_uri=https%3A%2F%2Fapp.example.com%2Fcb" in result.authorization_url
    assert "scope=openid+profile+email" in result.authorization_url
    assert "state=" in result.authorization_url
    # State round-trips and carries the redirect_uri back.
    payload = _verify_state(result.state)
    assert payload.redirect_uri == "https://app.example.com/cb"
    assert payload.nonce == result.nonce


async def test_get_authorization_url_requires_redirect_uri() -> None:
    service = _make_service(
        _FakeUsersRepo(),
        _FakeTokensRepo(),
        _FakeOidcClient(
            token=OIDCTokenResponse(access_token="at", id_token="", token_type="Bearer"),
            userinfo=_userinfo("a@b.c"),
        ),
    )
    with pytest.raises(ValidationError, match="redirect_uri is required"):
        await service.get_authorization_url(redirect_uri="   ")


# ── login_with_oidc ──────────────────────────────────────────────────


async def test_login_with_oidc_existing_user_mints_tokens() -> None:
    users = _FakeUsersRepo()
    _seed_user(users, email="alice@example.com")
    tokens = _FakeTokensRepo()
    client = _FakeOidcClient(
        token=OIDCTokenResponse(access_token="at", id_token="idt", token_type="Bearer"),
        userinfo=_userinfo("alice@example.com"),
    )
    service = _make_service(users, tokens, client)

    result = await service.login_with_oidc(code="c", redirect_uri="https://app.example.com/cb")

    assert result.success is True
    assert result.message == "登录成功"
    assert result.is_new_user is False
    assert result.user is not None
    assert result.user.email == "alice@example.com"
    assert result.access_token
    assert result.refresh_token
    # Two token rows persisted (access + refresh).
    assert len(tokens.inserted) == 2
    types_inserted = {row.token_type for row in tokens.inserted}
    assert types_inserted == {"access_token", "refresh_token"}
    # The access token is a real, decodable JWT bound to the user.
    claims = decode_token(result.access_token)
    assert claims["type"] == "access"
    assert claims["user_id"] == "usr-1"


async def test_login_with_oidc_new_user_raises_provisioning_unavailable() -> None:
    users = _FakeUsersRepo()  # no seeded user
    tokens = _FakeTokensRepo()
    client = _FakeOidcClient(
        token=OIDCTokenResponse(access_token="at", id_token="idt", token_type="Bearer"),
        userinfo=_userinfo("nobody@example.com"),
    )
    service = _make_service(users, tokens, client)

    with pytest.raises(ExternalServiceError) as exc_info:
        await service.login_with_oidc(code="c", redirect_uri="https://app.example.com/cb")
    assert exc_info.value.code == "oidc.provisioning_unavailable"


async def test_login_with_oidc_missing_email_raises() -> None:
    users = _FakeUsersRepo()
    _seed_user(users, email="alice@example.com")
    tokens = _FakeTokensRepo()
    client = _FakeOidcClient(
        token=OIDCTokenResponse(access_token="at", id_token="idt", token_type="Bearer"),
        userinfo=_userinfo(email=""),
    )
    service = _make_service(users, tokens, client)

    with pytest.raises(ValidationError) as exc_info:
        await service.login_with_oidc(code="c", redirect_uri="https://app.example.com/cb")
    assert exc_info.value.code == "oidc.missing_email"


async def test_login_with_oidc_inactive_user_returns_failure() -> None:
    """Inactive user -> success=False (HTTP 200 body), not a raise.

    Mirrors upstream ``LoginWithOIDC`` returning
    ``{Success:false, Message:"Account is disabled"}`` with nil error.
    """
    users = _FakeUsersRepo()
    _seed_user(users, email="alice@example.com", is_active=False)
    tokens = _FakeTokensRepo()
    client = _FakeOidcClient(
        token=OIDCTokenResponse(access_token="at", id_token="idt", token_type="Bearer"),
        userinfo=_userinfo("alice@example.com"),
    )
    service = _make_service(users, tokens, client)

    result = await service.login_with_oidc(code="c", redirect_uri="https://app.example.com/cb")

    assert result.success is False
    assert result.message == "Account is disabled"
    assert result.user is None
    assert result.access_token == ""
    # No token rows minted for a disabled user.
    assert tokens.inserted == []


async def test_login_with_oidc_requires_code() -> None:
    service = _make_service(
        _FakeUsersRepo(),
        _FakeTokensRepo(),
        _FakeOidcClient(
            token=OIDCTokenResponse(access_token="at", id_token="", token_type="Bearer"),
            userinfo=_userinfo("a@b.c"),
        ),
    )
    with pytest.raises(ValidationError, match="code is required"):
        await service.login_with_oidc(code="   ", redirect_uri="https://app.example.com/cb")


async def test_login_with_oidc_disabled_when_oidc_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OIDC_ENABLE", "false")
    reset_settings_cache()
    service = _make_service(
        _FakeUsersRepo(),
        _FakeTokensRepo(),
        _FakeOidcClient(
            token=OIDCTokenResponse(access_token="at", id_token="", token_type="Bearer"),
            userinfo=_userinfo("a@b.c"),
        ),
    )
    with pytest.raises(PermissionDeniedError) as exc_info:
        await service.login_with_oidc(code="c", redirect_uri="https://app.example.com/cb")
    assert exc_info.value.code == "oidc.disabled"


# ── State signing ────────────────────────────────────────────────────


def test_state_round_trip() -> None:
    now = int(time.time())
    payload = OIDCStatePayload(nonce="n-1", redirect_uri="https://app.example.com/cb", iat=now)
    signed = _sign_state(payload)
    verified = _verify_state(signed)
    assert verified.nonce == "n-1"
    assert verified.redirect_uri == "https://app.example.com/cb"
    assert verified.iat == now


def test_state_tampered_signature_rejected() -> None:
    now = int(time.time())
    payload = OIDCStatePayload(nonce="n", redirect_uri="https://app.example.com/cb", iat=now)
    signed = _sign_state(payload)
    # Flip the last char of the signature segment.
    head, _, sig = signed.rpartition(".")
    tampered = head + "." + (sig[:-1] + ("A" if sig[-1] != "A" else "B"))
    with pytest.raises(UnauthorizedError, match="signature mismatch"):
        _verify_state(tampered)


def test_state_expired_rejected() -> None:
    # iat far in the past -> beyond the 10-minute TTL.
    payload = OIDCStatePayload(
        nonce="n",
        redirect_uri="https://app.example.com/cb",
        iat=1_700_000_000,
    )
    signed = _sign_state(payload)
    with pytest.raises(UnauthorizedError, match="expired"):
        _verify_state(signed)


def test_state_malformed_rejected() -> None:
    with pytest.raises(UnauthorizedError):
        _verify_state("not-a-valid-state")
