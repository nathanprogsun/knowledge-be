"""User persistence — raw SQL only, no ORM.

Maps the methods declared in the upstream
``internal/types/interfaces/user.go::UserRepository`` interface. Every
query uses named ``bindparams``; soft-deleted rows (``deleted_at IS NOT
NULL``) are filtered out on every read.

``insert`` and ``find_by_id`` are inherited from
``GenericRepository[UserRow]``. Everything else here is domain-specific
(email/username lookups, password rotation, soft delete, listing).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Final, cast

from sqlalchemy import text
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.common.exception import ConflictError, NotFoundError
from src.core.auth.types import UserDTO
from src.db.dao.generic_repository import GenericRepository
from src.db.models.auth.users import UserRow

_FIND_BY_EMAIL_SQL: Final = text(
    "SELECT id, username, email, password_hash, avatar, tenant_id, "
    "is_active, can_access_all_tenants, is_system_admin, preferences, "
    "created_at, updated_at, deleted_at "
    "FROM users WHERE email = :email AND deleted_at IS NULL"
)

_FIND_BY_USERNAME_SQL: Final = text(
    "SELECT id, username, email, password_hash, avatar, tenant_id, "
    "is_active, can_access_all_tenants, is_system_admin, preferences, "
    "created_at, updated_at, deleted_at "
    "FROM users WHERE username = :username AND deleted_at IS NULL"
)

_UPDATE_SQL: Final = text(
    "UPDATE users SET "
    "username = :username, email = :email, password_hash = :password_hash, "
    "avatar = :avatar, tenant_id = :tenant_id, "
    "is_active = :is_active, can_access_all_tenants = :can_access_all_tenants, "
    "is_system_admin = :is_system_admin, preferences = :preferences, "
    "updated_at = :updated_at "
    "WHERE id = :id AND deleted_at IS NULL"
)

_UPDATE_PASSWORD_SQL: Final = text(
    "UPDATE users SET password_hash = :password_hash, updated_at = :updated_at "
    "WHERE id = :user_id AND deleted_at IS NULL"
)

_SOFT_DELETE_SQL: Final = text(
    "UPDATE users SET deleted_at = :deleted_at, updated_at = :updated_at "
    "WHERE id = :user_id AND deleted_at IS NULL"
)

_LIST_SQL: Final = text(
    "SELECT id, username, email, password_hash, avatar, tenant_id, "
    "is_active, can_access_all_tenants, is_system_admin, preferences, "
    "created_at, updated_at, deleted_at "
    "FROM users WHERE deleted_at IS NULL "
    "ORDER BY created_at DESC LIMIT :limit OFFSET :offset"
)


def _to_dto(row: UserRow) -> UserDTO:
    """Hydrate a ``UserDTO`` from a ``UserRow``.

    ``preferences`` is a JSONB dict on Postgres; downstream services
    want a ``UserPreferences`` Pydantic model. Re-validate here so the
    repository owns the boundary translation.
    """
    return UserDTO.model_validate(row.model_dump())


class UserRepository(GenericRepository[UserRow]):
    """User-table SQL — domain-specific queries live here, common CRUD
    on the base class."""

    def __init__(self) -> None:
        super().__init__(UserRow)

    async def find_by_id(  # type: ignore[override]
        self, session: AsyncSession, user_id: str
    ) -> UserDTO | None:
        row = await super().find_by_id(session, user_id)
        if row is None:
            return None
        return _to_dto(row)

    async def find_by_email(self, session: AsyncSession, email: str) -> UserDTO | None:
        row = (await session.execute(_FIND_BY_EMAIL_SQL.bindparams(email=email))).mappings().first()
        if row is None:
            return None
        return _to_dto(UserRow.model_validate(dict(row)))

    async def find_by_username(self, session: AsyncSession, username: str) -> UserDTO | None:
        row = (
            (await session.execute(_FIND_BY_USERNAME_SQL.bindparams(username=username)))
            .mappings()
            .first()
        )
        if row is None:
            return None
        return _to_dto(UserRow.model_validate(dict(row)))

    async def insert(self, session: AsyncSession, row: UserRow) -> None:
        try:
            await super().insert(session, row)
        except IntegrityError as exc:
            raise ConflictError(
                code="user.exists",
                message="User already exists",
            ) from exc

    async def update(self, session: AsyncSession, dto: UserDTO) -> None:
        now = datetime.now(UTC)
        result = cast(
            CursorResult[Any],
            await session.execute(
                _UPDATE_SQL.bindparams(
                    id=dto.id,
                    username=dto.username,
                    email=dto.email,
                    password_hash=dto.password_hash,
                    avatar=dto.avatar,
                    tenant_id=dto.tenant_id,
                    is_active=dto.is_active,
                    can_access_all_tenants=dto.can_access_all_tenants,
                    is_system_admin=dto.is_system_admin,
                    preferences=dto.preferences.model_dump(),
                    updated_at=now,
                )
            ),
        )
        if result.rowcount == 0:
            raise NotFoundError(
                code="user.not_found",
                message=f"User {dto.id} not found",
            )

    async def update_password(
        self, session: AsyncSession, user_id: str, password_hash: str
    ) -> None:
        now = datetime.now(UTC)
        result = cast(
            CursorResult[Any],
            await session.execute(
                _UPDATE_PASSWORD_SQL.bindparams(
                    user_id=user_id, password_hash=password_hash, updated_at=now
                )
            ),
        )
        if result.rowcount == 0:
            raise NotFoundError(
                code="user.not_found",
                message=f"User {user_id} not found",
            )

    async def soft_delete(self, session: AsyncSession, user_id: str) -> None:
        now = datetime.now(UTC)
        result = cast(
            CursorResult[Any],
            await session.execute(
                _SOFT_DELETE_SQL.bindparams(user_id=user_id, deleted_at=now, updated_at=now)
            ),
        )
        if result.rowcount == 0:
            raise NotFoundError(
                code="user.not_found",
                message=f"User {user_id} not found",
            )

    async def list(
        self,
        session: AsyncSession,
        *,
        limit: int,
        offset: int,
    ) -> list[UserDTO]:
        rows = (
            (await session.execute(_LIST_SQL.bindparams(limit=limit, offset=offset)))
            .mappings()
            .all()
        )
        return [_to_dto(UserRow.model_validate(dict(row))) for row in rows]


__all__ = ["UserRepository"]
