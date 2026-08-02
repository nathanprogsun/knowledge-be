"""Auth service — login, token lifecycle, logout.

Maps the auth-related methods on
``internal/application/service/user.go::userService`` (the upstream places
these on ``userService`` rather than a separate ``authService``).
Operations:

- ``login`` — verify email + bcrypt password, mint an access/refresh
  pair, persist both rows in ``auth_tokens``.
- ``validate_token`` — decode the JWT, reject revoked/refresh tokens,
  load the user, return ``(UserInfo, active_tenant_id)``.
- ``refresh`` — verify a refresh token, revoke the old row, mint a
  fresh pair.
- ``logout`` — bulk-revoke every outstanding token for the user.
- ``revoke_token`` — mark a single token revoked (by raw token value).

Session and repository ownership
--------------------------------

Per the cookiecutter-fastapi pattern (and AGENTS.md §7.2), the
service depends **only** on its repositories — it does not hold a
``AsyncSession``. Each repository owns its ``AsyncSession`` in
``__init__`` (all repos share the same per-request session); the
service calls ``self._users_repo.xxx(...)`` / ``self._tokens_repo.xxx(...)``
and never touches the session directly.

Construction flow (web layer, next PR):

    async with session_scope(session_factory) as session:
        users_repo = UserRepository(session)
        tokens_repo = AuthTokenRepository(session)
        svc = AuthService(
            users_repo=users_repo,
            tokens_repo=tokens_repo,
        )
        result = await svc.login(email=..., password=...)

Repository dependencies are passed in via the constructor — the service
never instantiates ``UserRepository()`` or ``AuthTokenRepository()``
itself. ``app_context/lifespan.py`` (next PR) builds the concrete
repos at startup; the request factory threads the shared session.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import UTC, datetime

from src.common.exception import NotFoundError, UnauthorizedError
from src.core.auth.types import UserInfo
from src.db.dao.auth_tokens_repository import (
    AuthTokenRepository,
)
from src.db.dao.users_repository import UserRepository
from src.db.models.auth.auth_tokens import AuthToken
from src.util.security import (
    TokenError,
    create_access_token,
    create_refresh_token,
    decode_token,
    verify_password,
)


@dataclass(frozen=True, slots=True)
class LoginResult:
    """Returned by ``AuthService.login`` and ``AuthService.refresh``."""

    user: UserInfo
    access_token: str
    access_expires_at: datetime
    refresh_token: str
    refresh_expires_at: datetime


class AuthService:
    """Stateless auth service — session is owned for the request lifetime.

    A new instance is constructed by the web layer per request via
    ``Depends(get_auth_service)``. Repository dependencies are injected;
    the service does not instantiate them.
    """

    def __init__(
        self,
        *,
        users_repo: UserRepository,
        tokens_repo: AuthTokenRepository,
    ) -> None:
        self._users_repo = users_repo
        self._tokens_repo = tokens_repo

    async def login(self, *, email: str, password: str) -> LoginResult:
        """Verify email + password, mint and persist an access/refresh pair."""
        try:
            user_row = await self._users_repo.find_by_email_with_credentials(email)
        except NotFoundError:
            # Same message as the wrong-password branch so a caller cannot
            # tell a missing email from a wrong password.
            raise UnauthorizedError(
                code="auth.invalid_credentials",
                message="Email or password is incorrect",
            ) from None
        if not user_row.is_active:
            raise UnauthorizedError(
                code="auth.invalid_credentials",
                message="Email or password is incorrect",
            )
        if not verify_password(password, user_row.password_hash):
            raise UnauthorizedError(
                code="auth.invalid_credentials",
                message="Email or password is incorrect",
            )
        info = UserInfo.model_validate(user_row.model_dump(exclude={"password_hash", "deleted_at"}))
        return await self._mint_pair(info)

    async def validate_token(self, *, token: str) -> tuple[UserInfo, int | None]:
        """Return ``(user, active_tenant_id)`` for a valid access token."""
        return await self._validate_token(token)

    async def refresh(self, *, refresh_token: str) -> LoginResult:
        """Verify a refresh token, revoke the old row, mint a fresh pair."""
        try:
            claims = decode_token(refresh_token)
        except TokenError as exc:
            raise UnauthorizedError(
                code="auth.invalid_refresh_token",
                message="Refresh token is invalid",
            ) from exc
        if claims.get("type") != "refresh":
            raise UnauthorizedError(
                code="auth.invalid_refresh_token",
                message="Not a refresh token",
            )
        user_id = claims.get("user_id")
        if not isinstance(user_id, str):
            raise UnauthorizedError(
                code="auth.invalid_refresh_token",
                message="Refresh token is missing user_id",
            )
        try:
            record = await self._tokens_repo.find_by_token_value(refresh_token)
        except NotFoundError:
            raise UnauthorizedError(
                code="auth.invalid_refresh_token",
                message="Refresh token is revoked",
            ) from None
        if record.is_revoked:
            raise UnauthorizedError(
                code="auth.invalid_refresh_token",
                message="Refresh token is revoked",
            )
        if record.token_type != "refresh_token":
            raise UnauthorizedError(
                code="auth.invalid_refresh_token",
                message="Not a refresh token",
            )
        # Revoke the old refresh row.
        await self._tokens_repo.revoke(record.id)
        # Load the user and mint a new pair.
        try:
            user = await self._users_repo.find_by_id(user_id)
        except NotFoundError:
            raise UnauthorizedError(
                code="auth.invalid_refresh_token",
                message="User no longer exists",
            ) from None
        if not user.is_active:
            raise UnauthorizedError(
                code="auth.invalid_refresh_token",
                message="User no longer exists",
            )
        return await self._mint_pair(user)

    async def logout(self, *, token: str) -> int:
        """Bulk-revoke every outstanding token for the token's owner.

        Returns the number of rows that flipped from active to revoked.
        Mirrors the upstream ``Logout`` semantics: any JWT (expired or
        not) is accepted as long as it decodes, so clients can end the
        session even after the access token TTL.
        """
        try:
            claims = decode_token(token)
        except TokenError as exc:
            raise UnauthorizedError(
                code="auth.invalid_token",
                message="Token is invalid",
            ) from exc
        user_id = claims.get("user_id")
        if not isinstance(user_id, str):
            raise UnauthorizedError(
                code="auth.invalid_token",
                message="Token is missing user_id",
            )
        return await self._tokens_repo.revoke_all_for_user(user_id)

    async def revoke_token(self, *, token: str) -> int:
        """Revoke a single token (looked up by raw value). Returns row count."""
        try:
            record = await self._tokens_repo.find_by_token_value(token)
        except NotFoundError:
            return 0
        return await self._tokens_repo.revoke(record.id)

    # ── Internal helpers ────────────────────────────────────────────

    async def _mint_pair(self, info: UserInfo) -> LoginResult:
        access, access_exp = create_access_token(
            user_id=info.id,
            email=info.email,
            tenant_id=info.tenant_id,
        )
        refresh, refresh_exp = create_refresh_token(user_id=info.id)
        now = datetime.now(UTC)
        await self._tokens_repo.insert(
            AuthToken(
                id=f"atk-{secrets.token_hex(8)}",
                user_id=info.id,
                token=access,
                token_type="access_token",
                expires_at=access_exp,
                is_revoked=False,
                created_at=now,
                updated_at=now,
            ),
        )
        await self._tokens_repo.insert(
            AuthToken(
                id=f"atk-{secrets.token_hex(8)}",
                user_id=info.id,
                token=refresh,
                token_type="refresh_token",
                expires_at=refresh_exp,
                is_revoked=False,
                created_at=now,
                updated_at=now,
            ),
        )
        return LoginResult(
            user=info,
            access_token=access,
            access_expires_at=access_exp,
            refresh_token=refresh,
            refresh_expires_at=refresh_exp,
        )

    async def _validate_token(self, token: str) -> tuple[UserInfo, int | None]:
        try:
            claims = decode_token(token)
        except TokenError as exc:
            raise UnauthorizedError(
                code="auth.invalid_token",
                message="Token is invalid",
            ) from exc
        if claims.get("type") != "access":
            raise UnauthorizedError(
                code="auth.invalid_token",
                message="Not an access token",
            )
        user_id = claims.get("user_id")
        if not isinstance(user_id, str):
            raise UnauthorizedError(
                code="auth.invalid_token",
                message="Token is missing user_id",
            )
        try:
            record = await self._tokens_repo.find_by_token_value(token)
        except NotFoundError:
            raise UnauthorizedError(
                code="auth.invalid_token",
                message="Token is revoked",
            ) from None
        if record.is_revoked:
            raise UnauthorizedError(
                code="auth.invalid_token",
                message="Token is revoked",
            )
        if record.token_type != "access_token":
            raise UnauthorizedError(
                code="auth.invalid_token",
                message="Not an access token",
            )
        try:
            user = await self._users_repo.find_by_id(user_id)
        except NotFoundError:
            raise UnauthorizedError(
                code="auth.invalid_token",
                message="User no longer exists",
            ) from None
        if not user.is_active:
            raise UnauthorizedError(
                code="auth.invalid_token",
                message="User no longer exists",
            )
        tenant_id_raw = claims.get("tenant_id")
        tenant_id = tenant_id_raw if isinstance(tenant_id_raw, int) else None
        return user, tenant_id


__all__ = ["AuthService", "LoginResult"]
