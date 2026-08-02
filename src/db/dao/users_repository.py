"""User persistence — raw SQL only, no ORM.

Maps the methods declared in the upstream
``internal/types/interfaces/user.go::UserRepository`` interface. Every
query uses named ``bindparams``; soft-deleted rows
(``deleted_at IS NOT NULL``) are filtered out on every read.

``insert`` and ``find_by_id`` are inherited from
``GenericRepository[User]``. Everything else here is
domain-specific (email/username lookups, password rotation, soft
delete, listing).

Session ownership
-----------------

Per the cookiecutter-fastapi pattern, the repository holds its
``AsyncSession`` in ``__init__``. Method signatures drop the
``session: AsyncSession`` parameter; the session is read from
``self._session`` instead.

Boundary translation
--------------------

Read methods return the wire-side ``UserInfo`` (no
``password_hash``) so callers never see the bcrypt hash. The
boundary translation lives in ``_to_info``.

Error semantics
---------------

Lookup methods raise ``NotFoundError(code="user.not_found")`` when no
row matches — mirroring the upstream ``ErrUserNotFound`` sentinel.
Services translate this into domain errors (e.g. a failed login
becomes an ``UnauthorizedError`` so the caller cannot distinguish a
missing email from a wrong password).

SQL style
---------

Statements are inlined in each method (``text(...).bindparams(...)``)
rather than hoisted to module constants — this mirrors the
cookiecutter-fastapi DAO style and keeps the SQL next to its logic.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, cast

from sqlalchemy import text
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import IntegrityError

from src.common.exception import ConflictError, NotFoundError
from src.core.auth.types import UserInfo, UserPreferences
from src.db.dao.generic_repository import GenericRepository
from src.db.models.auth.users import User


def _to_info(row: User) -> UserInfo:
    """Hydrate a wire-side ``UserInfo`` from a storage ``User`` row.

    ``password_hash`` and ``deleted_at`` are stripped — the former is
    sensitive (never crosses the service boundary), the latter is
    a storage-only soft-delete flag.
    """
    record = row.model_dump()
    record.pop("password_hash", None)
    record.pop("deleted_at", None)
    prefs = record.get("preferences") or {}
    if isinstance(prefs, str):
        # Defensive: some drivers surface JSON as a raw string.
        prefs = json.loads(prefs)
    record["preferences"] = UserPreferences.model_validate(prefs)
    return UserInfo.model_validate(record)


class UserRepository(GenericRepository[User]):
    """User-table SQL — domain-specific queries live here, common CRUD
    on the base class.
    """

    model_class = User

    def __init__(self, session) -> None:  # type: ignore[no-untyped-def]
        super().__init__(session)

    async def find_by_id(  # type: ignore[override]
        self, user_id: str
    ) -> UserInfo:
        row = await super().find_by_id(user_id)
        return _to_info(row)

    async def find_by_email(self, email: str) -> UserInfo:
        stmt = text(
            "SELECT id, username, email, password_hash, avatar, tenant_id, "
            "is_active, can_access_all_tenants, is_system_admin, preferences, "
            "created_at, updated_at, deleted_at "
            "FROM users WHERE email = :email AND deleted_at IS NULL"
        ).bindparams(email=email)
        row = (await self._session.execute(stmt)).mappings().first()
        if row is None:
            raise NotFoundError(
                code="user.not_found",
                message=f"User with email {email} not found",
            )
        return _to_info(User.model_validate(dict(row)))

    async def find_by_username(self, username: str) -> UserInfo:
        stmt = text(
            "SELECT id, username, email, password_hash, avatar, tenant_id, "
            "is_active, can_access_all_tenants, is_system_admin, preferences, "
            "created_at, updated_at, deleted_at "
            "FROM users WHERE username = :username AND deleted_at IS NULL"
        ).bindparams(username=username)
        row = (await self._session.execute(stmt)).mappings().first()
        if row is None:
            raise NotFoundError(
                code="user.not_found",
                message=f"User with username {username} not found",
            )
        return _to_info(User.model_validate(dict(row)))

    async def find_by_email_with_credentials(self, email: str) -> User:
        """Return the full storage row (including ``password_hash``).

        Only the auth service should call this — the bcrypt verify
        step needs the hash, but the hash must never leave the auth
        service boundary.

        Raises ``NotFoundError`` when no active user matches the email.
        """
        stmt = text(
            "SELECT id, username, email, password_hash, avatar, tenant_id, "
            "is_active, can_access_all_tenants, is_system_admin, preferences, "
            "created_at, updated_at, deleted_at "
            "FROM users WHERE email = :email AND deleted_at IS NULL"
        ).bindparams(email=email)
        row = (await self._session.execute(stmt)).mappings().first()
        if row is None:
            raise NotFoundError(
                code="user.not_found",
                message=f"User with email {email} not found",
            )
        return User.model_validate(dict(row))

    async def insert(self, row: User) -> None:
        try:
            await super().insert(row)
        except IntegrityError as exc:
            raise ConflictError(
                code="user.exists",
                message="User already exists",
            ) from exc

    async def update(self, dto: UserInfo) -> None:
        now = datetime.now(UTC)
        stmt = text(
            "UPDATE users SET "
            "username = :username, email = :email, password_hash = :password_hash, "
            "avatar = :avatar, tenant_id = :tenant_id, "
            "is_active = :is_active, can_access_all_tenants = :can_access_all_tenants, "
            "is_system_admin = :is_system_admin, preferences = :preferences, "
            "updated_at = :updated_at "
            "WHERE id = :id AND deleted_at IS NULL"
        ).bindparams(
            id=dto.id,
            username=dto.username,
            email=dto.email,
            password_hash="",  # never updated via this path
            avatar=dto.avatar,
            tenant_id=dto.tenant_id,
            is_active=dto.is_active,
            can_access_all_tenants=dto.can_access_all_tenants,
            is_system_admin=dto.is_system_admin,
            preferences=dto.preferences.model_dump(),
            updated_at=now,
        )
        result = cast(
            CursorResult[Any],
            await self._session.execute(stmt),
        )
        if result.rowcount == 0:
            raise NotFoundError(
                code="user.not_found",
                message=f"User {dto.id} not found",
            )

    async def update_password(self, user_id: str, password_hash: str) -> None:
        now = datetime.now(UTC)
        stmt = text(
            "UPDATE users SET password_hash = :password_hash, updated_at = :updated_at "
            "WHERE id = :user_id AND deleted_at IS NULL"
        ).bindparams(user_id=user_id, password_hash=password_hash, updated_at=now)
        result = cast(
            CursorResult[Any],
            await self._session.execute(stmt),
        )
        if result.rowcount == 0:
            raise NotFoundError(
                code="user.not_found",
                message=f"User {user_id} not found",
            )

    async def soft_delete(self, user_id: str) -> None:
        now = datetime.now(UTC)
        stmt = text(
            "UPDATE users SET deleted_at = :deleted_at, updated_at = :updated_at "
            "WHERE id = :user_id AND deleted_at IS NULL"
        ).bindparams(user_id=user_id, deleted_at=now, updated_at=now)
        result = cast(
            CursorResult[Any],
            await self._session.execute(stmt),
        )
        if result.rowcount == 0:
            raise NotFoundError(
                code="user.not_found",
                message=f"User {user_id} not found",
            )

    async def list(
        self,
        *,
        limit: int,
        offset: int,
    ) -> list[UserInfo]:
        stmt = text(
            "SELECT id, username, email, password_hash, avatar, tenant_id, "
            "is_active, can_access_all_tenants, is_system_admin, preferences, "
            "created_at, updated_at, deleted_at "
            "FROM users WHERE deleted_at IS NULL "
            "ORDER BY created_at DESC LIMIT :limit OFFSET :offset"
        ).bindparams(limit=limit, offset=offset)
        rows = (await self._session.execute(stmt)).mappings().all()
        return [_to_info(User.model_validate(dict(row))) for row in rows]


__all__ = ["UserRepository"]
