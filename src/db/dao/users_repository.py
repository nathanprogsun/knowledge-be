"""User persistence — raw SQL only, no ORM.

Maps the methods declared in the upstream
``internal/types/interfaces/user.go::UserRepository`` interface. Every
query uses named ``bindparams``; soft-deleted rows
(``deleted_at IS NOT NULL``) are filtered out on every read.

Only domain-named thin wrappers over ``GenericRepository[User]`` live
here (``find_by_email``, ``find_by_username``, ``insert``, ``list``).
Mutations (``update``, ``update_password``, ``soft_delete``) are **not**
re-implemented — callers use ``GenericRepository.update_by_primary_key``
directly with an explicit column dict. This keeps the db layer free of
business logic (AGENTS.md §1): which columns are mutable, and the
``updated_at`` timestamp, are service decisions, not repository ones.

Session ownership
-----------------

Per the cookiecutter-fastapi pattern, the repository holds its
``AsyncSession`` in ``__init__`` (inherited from ``GenericRepository``).

Error semantics
---------------

- ``find_by_*`` raise ``NotFoundError(code="user.not_found")`` when no
  row matches — mirroring the upstream ``ErrUserNotFound`` sentinel.
- ``insert`` raises ``ConflictError(code="user.exists")`` on a unique
  constraint conflict (email or username). The base ``insert_or_none``
  uses ``ON CONFLICT DO NOTHING`` so this is a clean ``if None`` check,
  not an ``IntegrityError`` catch.

Boundary translation
--------------------

This repository returns storage ``User`` rows. The wire-side
``UserInfo`` projection (no ``password_hash`` / ``deleted_at``) is
produced by ``UserInfo.map_from_db`` — per AGENTS.md §1, db layer holds
no business logic.
"""

from __future__ import annotations

from src.common.exception import ConflictError
from src.db.dao.generic_repository import GenericRepository
from src.db.models.auth.users import User


class UserRepository(GenericRepository[User]):
    """User-table SQL — domain-named wrappers on top of the generic
    CRUD base class. Mutations go through ``update_by_primary_key``
    directly; this class defines no update/soft-delete methods.
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
        ``ON CONFLICT`` clause). When no row was inserted, the user
        already exists under email or username, so raise
        ``ConflictError``.

        Faster and more precise than catching ``IntegrityError``: a
        genuine data error that isn't a unique-constraint violation
        (e.g. an FK breach) still surfaces as the original
        ``IntegrityError`` rather than being masked as a user-exists
        conflict.
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
