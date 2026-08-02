"""Auth service — login, token lifecycle, logout.

Maps the auth-related methods from
`internal/application/service/user.go::userService` (the upstream places
these on `userService` rather than a separate `authService`). Operations:

- ``login`` — verify email + bcrypt password, mint an access/refresh
  pair, persist both rows in ``auth_tokens``.
- ``validate_token`` — decode the JWT, reject revoked/refresh tokens,
  load the user, return ``(UserDTO, active_tenant_id)``.
- ``refresh`` — verify a refresh token, revoke the old row, mint a
  fresh pair.
- ``logout`` — bulk-revoke every outstanding token for the user.
- ``revoke_token`` — mark a single token revoked (by raw token value).

Service methods take ``AsyncSession`` per call (created by the web
layer's ``get_async_session`` dependency and committed via
``session_scope``). The service no longer holds a ``session_factory``;
the web layer owns the transactional boundary.

The service depends on Protocols (``UserLookup``, ``TokenStore``) rather
than the concrete repository classes so tests can swap in fakes
without touching the DB. Concrete ``UserRepository`` and
``AuthTokenRepository`` satisfy both Protocols via duck typing.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from sqlalchemy.ext.asyncio import AsyncSession

from src.common.exception import UnauthorizedError
from src.core.auth.types import (
    UserDTO,  # noqa: TC001  (used both as annotation and runtime attribute)
)
from src.db.dao.auth_tokens_repository import AuthTokenRepository
from src.db.dao.users_repository import UserRepository
from src.db.models.auth.auth_tokens import AuthTokenRow
from src.util.security import (
    TokenError,
    create_access_token,
    create_refresh_token,
    decode_token,
    verify_password,
)


class UserLookup(Protocol):
    """Read-side dependency for AuthService: a subset of UserRepository."""

    async def find_by_email(self, session: AsyncSession, email: str) -> UserDTO | None: ...

    async def find_by_id(self, session: AsyncSession, user_id: str) -> UserDTO | None: ...

    async def insert(self, session: AsyncSession, row: object) -> None: ...


class TokenStore(Protocol):
    """Read+write dependency for AuthService: a subset of AuthTokenRepository."""

    async def insert(self, session: AsyncSession, row: AuthTokenRow) -> None: ...

    async def find_by_token_value(
        self, session: AsyncSession, token: str
    ) -> AuthTokenRow | None: ...

    async def revoke_all_for_user(self, session: AsyncSession, user_id: str) -> int: ...

    async def revoke(self, session: AsyncSession, token_id: str) -> int: ...


@dataclass(frozen=True, slots=True)
class LoginResult:
    """Returned by ``AuthService.login`` and ``AuthService.refresh``."""

    user: UserDTO
    access_token: str
    access_expires_at: datetime
    refresh_token: str
    refresh_expires_at: datetime


class AuthService:
    """Stateless auth service — session is opened per request by the caller."""

    def __init__(
        self,
        *,
        user_repository: UserLookup | None = None,
        token_repository: TokenStore | None = None,
    ) -> None:
        self._users = user_repository or UserRepository()
        self._tokens = token_repository or AuthTokenRepository()

    async def login(self, session: AsyncSession, *, email: str, password: str) -> LoginResult:
        """Verify email + password, mint and persist an access/refresh pair."""
        user = await self._users.find_by_email(session, email)
        if user is None or not user.is_active:
            raise UnauthorizedError(
                code="auth.invalid_credentials",
                message="Email or password is incorrect",
            )
        if user.password_hash is None or not verify_password(password, user.password_hash):
            raise UnauthorizedError(
                code="auth.invalid_credentials",
                message="Email or password is incorrect",
            )
        return await self._mint_pair(session, user)

    async def validate_token(
        self, session: AsyncSession, *, token: str
    ) -> tuple[UserDTO, int | None]:
        """Return ``(user, active_tenant_id)`` for a valid access token."""
        return await self._validate_token(session, token)

    async def refresh(self, session: AsyncSession, *, refresh_token: str) -> LoginResult:
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
        record = await self._tokens.find_by_token_value(session, refresh_token)
        if record is None or record.is_revoked:
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
        await self._tokens.revoke(session, record.id)
        # Load the user and mint a new pair.
        user = await self._users.find_by_id(session, user_id)
        if user is None or not user.is_active:
            raise UnauthorizedError(
                code="auth.invalid_refresh_token",
                message="User no longer exists",
            )
        return await self._mint_pair(session, user)

    async def logout(self, session: AsyncSession, *, token: str) -> int:
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
        return await self._tokens.revoke_all_for_user(session, user_id)

    async def revoke_token(self, session: AsyncSession, *, token: str) -> int:
        """Revoke a single token (looked up by raw value). Returns row count."""
        record = await self._tokens.find_by_token_value(session, token)
        if record is None:
            return 0
        return await self._tokens.revoke(session, record.id)

    # ── Internal helpers ────────────────────────────────────────────

    async def _mint_pair(self, session: AsyncSession, user: UserDTO) -> LoginResult:
        access, access_exp = create_access_token(
            user_id=user.id,
            email=user.email,
            tenant_id=user.tenant_id,
        )
        refresh, refresh_exp = create_refresh_token(user_id=user.id)
        now = datetime.now(UTC)
        await self._tokens.insert(
            session,
            AuthTokenRow(
                id=f"atk-{secrets.token_hex(8)}",
                user_id=user.id,
                token=access,
                token_type="access_token",
                expires_at=access_exp,
                is_revoked=False,
                created_at=now,
                updated_at=now,
            ),
        )
        await self._tokens.insert(
            session,
            AuthTokenRow(
                id=f"atk-{secrets.token_hex(8)}",
                user_id=user.id,
                token=refresh,
                token_type="refresh_token",
                expires_at=refresh_exp,
                is_revoked=False,
                created_at=now,
                updated_at=now,
            ),
        )
        return LoginResult(
            user=user,
            access_token=access,
            access_expires_at=access_exp,
            refresh_token=refresh,
            refresh_expires_at=refresh_exp,
        )

    async def _validate_token(
        self, session: AsyncSession, token: str
    ) -> tuple[UserDTO, int | None]:
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
        record = await self._tokens.find_by_token_value(session, token)
        if record is None or record.is_revoked:
            raise UnauthorizedError(
                code="auth.invalid_token",
                message="Token is revoked",
            )
        if record.token_type != "access_token":
            raise UnauthorizedError(
                code="auth.invalid_token",
                message="Not an access token",
            )
        user = await self._users.find_by_id(session, user_id)
        if user is None or not user.is_active:
            raise UnauthorizedError(
                code="auth.invalid_token",
                message="User no longer exists",
            )
        tenant_id_raw = claims.get("tenant_id")
        tenant_id = tenant_id_raw if isinstance(tenant_id_raw, int) else None
        return user, tenant_id


__all__ = ["AuthService", "LoginResult", "TokenStore", "UserLookup"]
