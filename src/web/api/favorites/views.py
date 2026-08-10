"""Wire-shape conversion for the user-favorites endpoints.

Projects the storage row :class:`UserResourceFavorite` onto the public
contract ``{success, data}`` envelope. Field names mirror the upstream
Go struct so the JSON wire format stays aligned: ``user_id``,
``tenant_id``, ``resource_type``, ``resource_id``, ``created_at``.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from src.db.models.user_resource_favorite import UserResourceFavorite


class FavoriteEntry(BaseModel):
    """One favorited resource on the wire."""

    model_config = ConfigDict(frozen=True)

    user_id: str
    tenant_id: int
    resource_type: str
    resource_id: str
    created_at: datetime


class FavoritesListResponse(BaseModel):
    """``{"success": true, "data": [FavoriteEntry, ...]}``.

    Used by ``GET /user/favorites?type=<kb|agent>``. The envelope shape
    matches the rest of the per-user API; the data array is empty when
    the user has not starred anything of the requested type.
    """

    model_config = ConfigDict(frozen=True)

    success: bool
    data: list[FavoriteEntry]


class FavoriteMutationResponse(BaseModel):
    """``{"success": true}`` for the star / unstar endpoints.

    The upstream handler returns no row on mutation (it only embeds
    the favorite list on the read path). Mirroring that keeps the
    toggle-on-click UX symmetric for the frontend.
    """

    model_config = ConfigDict(frozen=True)

    success: bool


def favorite_to_response(row: UserResourceFavorite) -> FavoriteEntry:
    """Project one storage row to the wire shape."""
    return FavoriteEntry(
        user_id=row.user_id,
        tenant_id=row.tenant_id,
        resource_type=row.resource_type,
        resource_id=row.resource_id,
        created_at=row.created_at,
    )


def favorites_list_response(rows: list[UserResourceFavorite]) -> FavoritesListResponse:
    """Wrap a list of storage rows into the standard envelope."""
    return FavoritesListResponse(success=True, data=[favorite_to_response(r) for r in rows])


__all__ = [
    "FavoriteEntry",
    "FavoriteMutationResponse",
    "FavoritesListResponse",
    "favorite_to_response",
    "favorites_list_response",
]
