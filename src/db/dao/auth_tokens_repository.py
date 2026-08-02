"""Auth-token persistence — raw SQL only, no ORM.

Maps the methods declared in the upstream
``internal/types/interfaces/user.go::AuthTokenRepository`` interface.
``insert`` is inherited from ``GenericRepository[AuthToken]``; the rest
are domain-specific. Every query uses named ``bindparams``.
``is_revoked = TRUE`` rows are kept (not soft-deleted) so audit logs
and replay-attack checks remain possible.

Session ownership
-----------------

Per the cookiecutter-fastapi pattern, the repository holds its
``AsyncSession`` in ``__init__``. Method signatures drop the
``session: AsyncSession`` parameter; the session is read from
``self._session`` instead.

Error semantics
---------------

``find_by_token_value`` raises ``NotFoundError(code="token.not_found")``
when no row matches — mirroring the upstream ``ErrTokenNotFound``
sentinel. Services translate this into domain errors (e.g. a revoked
or unknown refresh token becomes an ``UnauthorizedError``).

SQL style
---------

Statements are inlined in each method (``text(...).bindparams(...)``)
rather than hoisted to module constants — this mirrors the
cookiecutter-fastapi DAO style and keeps the SQL next to its logic.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

from sqlalchemy import text
from sqlalchemy.engine import CursorResult

from src.common.exception import NotFoundError
from src.db.dao.generic_repository import GenericRepository
from src.db.models.auth.auth_tokens import AuthToken


class AuthTokenRepository(GenericRepository[AuthToken]):
    """Auth-token SQL — domain-specific queries live here, common CRUD
    on the base class.
    """

    model_class = AuthToken

    def __init__(self, session) -> None:  # type: ignore[no-untyped-def]
        super().__init__(session)

    async def find_by_token_value(self, token: str) -> AuthToken:
        """Look up a token by its raw value.

        Raises ``NotFoundError`` when no row matches (including revoked
        rows — those are returned normally so the caller can check the
        ``is_revoked`` flag).
        """
        stmt = text(
            "SELECT id, user_id, token, token_type, expires_at, is_revoked, "
            "created_at, updated_at "
            "FROM auth_tokens WHERE token = :token"
        ).bindparams(token=token)
        row = (await self._session.execute(stmt)).mappings().first()
        if row is None:
            raise NotFoundError(
                code="token.not_found",
                message="Token not found",
            )
        return AuthToken.model_validate(dict(row))

    async def revoke_all_for_user(self, user_id: str) -> int:
        """Revoke every outstanding token for ``user_id``.

        Returns the number of rows that flipped from active to revoked.
        """
        stmt = text(
            "UPDATE auth_tokens SET is_revoked = TRUE, updated_at = :updated_at "
            "WHERE user_id = :user_id AND is_revoked = FALSE"
        ).bindparams(user_id=user_id, updated_at=datetime.now(UTC))
        result = cast(
            CursorResult[Any],
            await self._session.execute(stmt),
        )
        return result.rowcount or 0

    async def revoke(self, token_id: str) -> int:
        """Revoke a single token by id. Returns the affected row count."""
        stmt = text(
            "UPDATE auth_tokens SET is_revoked = TRUE, updated_at = :updated_at "
            "WHERE id = :id AND is_revoked = FALSE"
        ).bindparams(id=token_id, updated_at=datetime.now(UTC))
        result = cast(
            CursorResult[Any],
            await self._session.execute(stmt),
        )
        return result.rowcount or 0

    async def delete_expired(self) -> int:
        """Remove tokens past their ``expires_at``. Returns the row count."""
        stmt = text("DELETE FROM auth_tokens WHERE expires_at < :now").bindparams(
            now=datetime.now(UTC)
        )
        result = cast(
            CursorResult[Any],
            await self._session.execute(stmt),
        )
        return result.rowcount or 0


__all__ = ["AuthTokenRepository"]
