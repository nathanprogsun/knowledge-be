"""Auth service - login, token lifecycle, logout."""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import UTC, datetime

from src.common.exception import (
    ConflictError,
    NotFoundError,
    UnauthorizedError,
    ValidationError,
)
from src.core.auth.types import UserInfo, UserPreferences
from src.db.dao.auth_tokens_repository import (
    AuthTokenRepository,
)
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


@dataclass(frozen=True, slots=True)
class LoginResult:
    """Returned by ``AuthService.login`` and ``AuthService.refresh``."""

    user: UserInfo
    access_token: str
    access_expires_at: datetime
    refresh_token: str
    refresh_expires_at: datetime


async def mint_token_pair(
    *,
    tokens_repo: AuthTokenRepository,
    info: UserInfo,
) -> LoginResult:
    """Mint an access/refresh JWT pair and persist both rows in ``auth_tokens``.

    Shared by ``AuthService`` (password login / refresh) and ``OidcService``
    (OIDC login): two ``auth_tokens`` rows (access + refresh) bound to
    ``info.id``.
    """
    access, access_exp = create_access_token(
        user_id=info.id,
        email=info.email,
        tenant_id=info.tenant_id,
    )
    refresh, refresh_exp = create_refresh_token(user_id=info.id)
    now = datetime.now(UTC)
    await tokens_repo.insert(
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
    await tokens_repo.insert(
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


class AuthService:
    """Stateless auth service - session is owned for the request lifetime."""

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
            user_row = await self._users_repo.find_by_email(email)
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
        return await self._mint_pair(UserInfo.map_from_db(user_row))

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
        return await self._mint_pair(UserInfo.map_from_db(user))

    async def logout(self, *, token: str) -> int:
        """Bulk-revoke every outstanding token for the token's owner.

        Returns the number of rows that flipped from active to revoked.
        Any JWT (expired or not) is accepted as long as it decodes, so
        clients can end the session even after the access token TTL.
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

    async def register(
        self,
        *,
        username: str,
        email: str,
        password: str,
    ) -> LoginResult:
        """Create a user and mint an access/refresh pair."""
        user_row = await self.create_user(
            username=username,
            email=email,
            password=password,
        )
        return await self._mint_pair(UserInfo.map_from_db(user_row))

    async def create_user(
        self,
        *,
        username: str,
        email: str,
        password: str,
    ) -> User:
        """Validate + insert a user row without minting tokens.

        Rejects empty / colliding usernames or emails with a
        ``ConflictError`` (the ``users`` table enforces uniqueness on
        both). The password is hashed at the configured cost.
        """
        clean_username = username.strip()
        clean_email = email.strip().lower()
        if not clean_username:
            raise ValidationError(
                code="auth.username_required",
                message="Username cannot be empty",
            )
        if not clean_email:
            raise ValidationError(
                code="auth.email_required",
                message="Email cannot be empty",
            )
        now = datetime.now(UTC)
        user_row = User(
            id=f"usr-{secrets.token_hex(8)}",
            username=clean_username,
            email=clean_email,
            password_hash=hash_password(password),
            is_active=True,
            created_at=now,
            updated_at=now,
        )
        try:
            return await self._users_repo.insert(user_row)
        except ConflictError as exc:
            raise ConflictError(
                code="user.exists",
                message="Email or username already registered",
            ) from exc

    async def change_password(
        self,
        *,
        user_id: str,
        old_password: str,
        new_password: str,
    ) -> None:
        """Verify ``old_password`` and replace it with ``new_password``."""
        try:
            user = await self._users_repo.find_by_id(user_id)
        except NotFoundError as exc:
            raise UnauthorizedError(
                code="auth.invalid_credentials",
                message="Email or password is incorrect",
            ) from exc
        if not verify_password(old_password, user.password_hash):
            raise UnauthorizedError(
                code="auth.invalid_credentials",
                message="Current password is incorrect",
            )
        updated = await self._users_repo.update_by_primary_key(
            {"id": user_id},
            {
                "password_hash": hash_password(new_password),
                "updated_at": datetime.now(UTC),
            },
        )
        if updated is None:
            raise NotFoundError(
                code="user.not_found",
                message=f"User {user_id} not found",
            )

    async def get_me(self, *, token: str) -> tuple[UserInfo, int | None]:
        """Resolve the authenticated user + active tenant from an access token."""
        return await self._validate_token(token)

    async def get_user_row_by_email(self, email: str) -> User | None:
        """Return the storage row for ``email`` or ``None`` (never raises)."""
        try:
            return await self._users_repo.find_by_email(email.strip().lower())
        except NotFoundError:
            return None

    async def get_user_by_id(self, user_id: str) -> UserInfo:
        """Return the user DTO for ``user_id``; raises ``NotFoundError``.

        Exists so the web layer (header-auth middleware) never touches
        ``UserRepository`` directly.
        """
        row = await self._users_repo.find_by_id(user_id)
        return UserInfo.map_from_db(row)

    async def mint_pair_for_user_row(self, user_row: User) -> LoginResult:
        """Mint + persist an access/refresh pair for an existing user row."""
        return await mint_token_pair(
            tokens_repo=self._tokens_repo,
            info=UserInfo.map_from_db(user_row),
        )

    async def update_home_tenant(self, user_id: str, tenant_id: int) -> User:
        """Point the user's home workspace at ``tenant_id``; returns the row."""
        updated = await self._users_repo.update_by_primary_key(
            {"id": user_id},
            {"tenant_id": tenant_id, "updated_at": datetime.now(UTC)},
        )
        if updated is None:
            raise NotFoundError(
                code="user.not_found",
                message=f"User {user_id} not found",
            )
        return updated

    async def update_my_preferences(
        self,
        *,
        user_id: str,
        patch: UserPreferences,
    ) -> UserPreferences:
        """PATCH-merge user preferences (only supplied keys are overwritten).

        ``last_active_tenant_id=0`` clears the preference (matches the
        Go PATCH semantics); ``None`` leaves the stored value untouched.
        """
        try:
            user = await self._users_repo.find_by_id(user_id)
        except NotFoundError as exc:
            raise UnauthorizedError(
                code="auth.invalid_credentials",
                message="User not found",
            ) from exc
        current = UserPreferences.from_json(user.preferences)
        merged = UserPreferences(
            last_active_tenant_id=(
                patch.last_active_tenant_id
                if patch.last_active_tenant_id is not None
                else current.last_active_tenant_id
            ),
        )
        await self._users_repo.update_by_primary_key(
            {"id": user_id},
            {
                "preferences": merged.model_dump(mode="json"),
                "updated_at": datetime.now(UTC),
            },
        )
        return merged

    # ── Internal helpers ────────────────────────────────────────────

    async def _mint_pair(self, info: UserInfo) -> LoginResult:
        return await mint_token_pair(tokens_repo=self._tokens_repo, info=info)

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
        return UserInfo.map_from_db(user), tenant_id


__all__ = ["AuthService", "LoginResult", "mint_token_pair"]
