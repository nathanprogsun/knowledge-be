"""User persistence — raw SQL only, no ORM.

Domain-named thin wrappers over ``GenericRepository[User]``
(``find_by_email``, ``find_by_username``, ``insert``, ``list``).
Soft-deleted rows are filtered out on every read.

Mutations are not re-implemented here: callers use
``GenericRepository.update_by_primary_key`` directly with an explicit
column dict. Which columns are mutable, and the ``updated_at``
timestamp, are service decisions, not repository ones.
"""

from __future__ import annotations

from src.common.exception import ConflictError
from src.db.dao.generic_repository import GenericRepository
from src.db.models.auth.users import User


class UserRepository(GenericRepository[User]):
    """User-table SQL — domain wrappers on the generic CRUD base.
    No update / soft-delete methods here; those go through the base.
    """

    model_class = User

    # ── Reads ───────────────────────────────────────────────────────

    async def find_by_email(self, email: str) -> User:
        return await self.find_unique_by_column_values_or_fail(
            {"email": email},
            not_found_code="user.not_found",
            not_found_message=f"User with email {email} not found",
        )

    async def find_by_username(self, username: str) -> User:
        return await self.find_unique_by_column_values_or_fail(
            {"username": username},
            not_found_code="user.not_found",
            not_found_message=f"User with username {username} not found",
        )

    # ── Inserts ─────────────────────────────────────────────────────

    async def insert(self, row: User) -> User:
        """Insert a user; raise ``ConflictError`` on email/username clash.

        Emits ``ON CONFLICT DO NOTHING`` without a target column so the
        statement suppresses conflicts on **any** unique constraint (the
        ``users`` table has separate unique constraints on ``email`` and
        ``username`` — Postgres cannot target both in a single
        ``ON CONFLICT`` clause). A ``None`` result therefore means the
        user already exists under email or username.
        """
        result = await self.insert_or_none(row)
        if result is None:
            raise ConflictError(
                code="user.exists",
                message="User already exists",
            )
        return result

    # ── Listing ─────────────────────────────────────────────────────

    async def list(
        self,
        *,
        limit: int,
        offset: int,
    ) -> list[User]:
        """Return paginated users ordered by creation time (newest first)."""
        return await self.find_all(
            limit=limit,
            offset=offset,
            order_by="created_at desc",
        )


__all__ = ["UserRepository"]
