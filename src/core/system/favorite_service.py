"""User-resource favorite service — input validation + repo delegation.

Thin orchestration over :class:`UserResourceFavoriteRepository`. The
service enforces two contract rules that the handler must not have to
repeat:

- ``resource_type`` must be in the allowlist
  (:data:`FAVORITE_RESOURCE_TYPES`). The upstream handler maps a
  non-allowlist value to ``400 Bad Request``; the service raises
  :class:`ValidationError` with code ``favorite.invalid_type`` and the
  global exception handler translates it.
- ``resource_id`` must be non-empty (after trim). A blank id would
  collide with the composite PK and mean a user starred "the empty
  resource" — meaningless in the domain model. Mapped to
  ``favorite.empty_id`` (also ``400``).

Favoriting is a non-business action: it does not emit audit events,
does not write cross-aggregate state, and does not need caching. The
service is therefore intentionally a thin pass-through — there is no
value in adding business rules that only make the handler's life
harder.
"""

from __future__ import annotations

from src.common.exception import ValidationError
from src.core.system.types import FavoriteInfo
from src.db.dao.user_resource_favorite_repository import UserResourceFavoriteRepository
from src.db.models.user_resource_favorite import (
    FAVORITE_RESOURCE_TYPES,
)


class UserResourceFavoriteService:
    """Per-user starred-resource operations.

    The service is request-scoped; the repository is built on the same
    :class:`AsyncSession` so a successful ``add`` + the caller's own
    audit row (if any) commit atomically.
    """

    def __init__(self, *, repo: UserResourceFavoriteRepository) -> None:
        self._repo = repo

    # ── List ────────────────────────────────────────────────────────

    async def list_favorites(
        self,
        *,
        user_id: str,
        tenant_id: int,
        resource_type: str,
    ) -> list[FavoriteInfo]:
        """Return the caller's starred resources of one type, newest first.

        Raises :class:`ValidationError` (``favorite.invalid_type``)
        when ``resource_type`` is not in the allowlist — the handler
        maps it to ``400 Bad Request``.
        """
        self._ensure_valid_resource_type(resource_type)
        rows = await self._repo.list_by_user(
            user_id=user_id,
            tenant_id=tenant_id,
            resource_type=resource_type,
        )
        return [FavoriteInfo.map_from_db(row) for row in rows]

    # ── Add ─────────────────────────────────────────────────────────

    async def add_favorite(
        self,
        *,
        user_id: str,
        tenant_id: int,
        resource_type: str,
        resource_id: str,
    ) -> FavoriteInfo | None:
        """Star one resource. Idempotent — a duplicate no-ops.

        Returns the persisted row, or ``None`` when the favorite was
        already present. Both outcomes are success paths from the
        handler's perspective (200 with empty body).
        """
        self._ensure_valid_resource_type(resource_type)
        self._ensure_non_empty_id(resource_id)
        row = await self._repo.add(
            user_id=user_id,
            tenant_id=tenant_id,
            resource_type=resource_type,
            resource_id=resource_id,
        )
        return FavoriteInfo.map_from_db(row) if row is not None else None

    # ── Remove ──────────────────────────────────────────────────────

    async def remove_favorite(
        self,
        *,
        user_id: str,
        tenant_id: int,
        resource_type: str,
        resource_id: str,
    ) -> bool:
        """Unstar one resource. Idempotent — a missing target no-ops.

        Returns whether a row was actually deleted. A ``False`` return
        is not an error; the handler still answers ``200`` so a
        double-tap unstar is safe to retry.
        """
        self._ensure_valid_resource_type(resource_type)
        self._ensure_non_empty_id(resource_id)
        return await self._repo.remove(
            user_id=user_id,
            tenant_id=tenant_id,
            resource_type=resource_type,
            resource_id=resource_id,
        )

    # ── Internal ────────────────────────────────────────────────────

    @staticmethod
    def _ensure_valid_resource_type(resource_type: str) -> None:
        """Reject resource types outside the allowlist.

        Mirrors the upstream ``IsValidFavoriteResourceType`` check so
        the same set of strings works on both sides of the wire.
        """
        if resource_type not in FAVORITE_RESOURCE_TYPES:
            raise ValidationError(
                code="favorite.invalid_type",
                message=f"invalid favorite resource type {resource_type!r}",
            )

    @staticmethod
    def _ensure_non_empty_id(resource_id: str) -> None:
        """Reject blank resource ids.

        A blank id would (1) collide with the composite PK in
        unpredictable ways, and (2) mean the user starred "no
        resource" — a meaningless operation. Trimmed before the
        check so whitespace-only values are caught.
        """
        if not resource_id or not resource_id.strip():
            raise ValidationError(
                code="favorite.empty_id",
                message="favorite resource id is required",
            )


__all__ = ["UserResourceFavoriteService"]
