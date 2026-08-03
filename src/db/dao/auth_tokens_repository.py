"""Auth-token persistence — raw SQL only, no ORM.

``insert`` and ``find_by_token_value`` lean on ``GenericRepository``;
``revoke`` / ``revoke_all_for_user`` / ``delete_expired`` use
conditional UPDATE / DELETE that return row counts. ``is_revoked = TRUE``
rows are kept (not soft-deleted) so audit logs remain possible.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import cast

from sqlalchemy import text
from sqlalchemy.engine import CursorResult, RowMapping

from src.common.exception import NotFoundError
from src.db.dao.generic_repository import GenericRepository
from src.db.models.auth.auth_tokens import AuthToken


class AuthTokenRepository(GenericRepository[AuthToken]):
    """Auth-token SQL — common CRUD on the base class, domain
    revocation / cleanup queries here.
    """

    model_class = AuthToken

    async def find_by_token_value(self, token: str) -> AuthToken:
        """Look up a token by its raw value.

        Revoked rows are returned normally so the caller can inspect
        the ``is_revoked`` flag; only an absent row raises
        ``NotFoundError``.
        """
        result = await self.find_unique_by_column_values(
            {"token": token},
            exclude_deleted_or_archived=False,
        )
        if result is None:
            raise NotFoundError(
                code="token.not_found",
                message="Token not found",
            )
        return result

    async def revoke_all_for_user(self, user_id: str) -> int:
        """Revoke every outstanding token for ``user_id``.

        Returns the number of rows that flipped from active to revoked.
        """
        stmt = text(
            "UPDATE auth_tokens SET is_revoked = TRUE, updated_at = :updated_at "
            "WHERE user_id = :user_id AND is_revoked = FALSE"
        ).bindparams(user_id=user_id, updated_at=datetime.now(UTC))
        result = cast(
            CursorResult[RowMapping],
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
            CursorResult[RowMapping],
            await self._session.execute(stmt),
        )
        return result.rowcount or 0

    async def delete_expired(self) -> int:
        """Remove tokens past their ``expires_at``. Returns the row count."""
        stmt = text("DELETE FROM auth_tokens WHERE expires_at < :now").bindparams(
            now=datetime.now(UTC)
        )
        result = cast(
            CursorResult[RowMapping],
            await self._session.execute(stmt),
        )
        return result.rowcount or 0


__all__ = ["AuthTokenRepository"]
