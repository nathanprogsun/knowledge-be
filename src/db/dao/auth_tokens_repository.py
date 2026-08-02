"""Auth-token persistence — raw SQL only, no ORM.

Maps the methods declared in the upstream
``internal/types/interfaces/user.go::AuthTokenRepository`` interface.
``insert`` is inherited from ``GenericRepository[AuthTokenRow]``; the rest
are domain-specific. Every query uses named ``bindparams``.
``is_revoked = TRUE`` rows are kept (not soft-deleted) so audit logs
and replay-attack checks remain possible.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Final, cast

from sqlalchemy import text
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.dao.generic_repository import GenericRepository
from src.db.models.auth.auth_tokens import AuthTokenRow

_FIND_BY_TOKEN_SQL: Final = text(
    "SELECT id, user_id, token, token_type, expires_at, is_revoked, "
    "created_at, updated_at "
    "FROM auth_tokens WHERE token = :token"
)

_REVOKE_BY_USER_SQL: Final = text(
    "UPDATE auth_tokens SET is_revoked = TRUE, updated_at = :updated_at "
    "WHERE user_id = :user_id AND is_revoked = FALSE"
)

_REVOKE_BY_ID_SQL: Final = text(
    "UPDATE auth_tokens SET is_revoked = TRUE, updated_at = :updated_at "
    "WHERE id = :id AND is_revoked = FALSE"
)

_DELETE_EXPIRED_SQL: Final = text("DELETE FROM auth_tokens WHERE expires_at < :now")


class AuthTokenRepository(GenericRepository[AuthTokenRow]):
    """Auth-token SQL — domain-specific queries live here, common CRUD
    on the base class."""

    def __init__(self) -> None:
        super().__init__(AuthTokenRow)

    async def find_by_token_value(self, session: AsyncSession, token: str) -> AuthTokenRow | None:
        result = await session.execute(_FIND_BY_TOKEN_SQL.bindparams(token=token))
        row = result.mappings().first()
        if row is None:
            return None
        return AuthTokenRow.model_validate(dict(row))

    async def revoke_all_for_user(self, session: AsyncSession, user_id: str) -> int:
        """Revoke every outstanding token for ``user_id``.

        Returns the number of rows that flipped from active to revoked.
        """
        result = cast(
            CursorResult[Any],
            await session.execute(
                _REVOKE_BY_USER_SQL.bindparams(user_id=user_id, updated_at=datetime.now(UTC))
            ),
        )
        return result.rowcount or 0

    async def revoke(self, session: AsyncSession, token_id: str) -> int:
        """Revoke a single token by id. Returns the affected row count."""
        result = cast(
            CursorResult[Any],
            await session.execute(
                _REVOKE_BY_ID_SQL.bindparams(id=token_id, updated_at=datetime.now(UTC))
            ),
        )
        return result.rowcount or 0

    async def delete_expired(self, session: AsyncSession) -> int:
        """Remove tokens past their ``expires_at``. Returns the row count."""
        result = cast(
            CursorResult[Any],
            await session.execute(_DELETE_EXPIRED_SQL.bindparams(now=datetime.now(UTC))),
        )
        return result.rowcount or 0


__all__ = ["AuthTokenRepository"]
