"""User-favorite HTTP endpoints — list, star, unstar.

Maps the upstream user-favorite handler:

=============================================  ====
Route                                           Action
=============================================  ====
``GET    /user/favorites``                       List my favorites of one type
``POST   /user/favorites``                       Star a resource
``DELETE /user/favorites/{type}/{id}``           Unstar a resource
=============================================  ====

Authorization model: a user can only manipulate *their own*
favorites in the workspace they're currently scoped into. The
handler resolves ``user_id`` and ``tenant_id`` from the request
context (the auth middleware populates ``request.state.user_info`` and
``request.state.tenant_id``), so callers cannot pass a different
identity via query string or body — cross-user or cross-workspace
access is not supported by design.

The service layer enforces the resource-type allowlist and the
non-empty resource id. Both checks raise ``ValidationError`` which the
global exception handler maps to ``400 Bad Request``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query
from pydantic import BaseModel, ConfigDict

from src.common.exception import UnauthorizedError
from src.web.api.favorites.views import (
    FavoriteMutationResponse,
    FavoritesListResponse,
    favorites_list_response,
)
from src.web.deps import AuthDep, FavoriteServiceDep
from src.web.deps.context import get_tenant_id_dep, get_user_id_dep

# Function-arg-style principal deps. ``_PrincipalUser`` is a string
# (auth-middleware-populated ``user_info.id``) so a request without
# a user context reads as ``None``; the router rejects ``None``
# explicitly because favorites are always per-user.
_PrincipalTenant = Annotated[int, Depends(get_tenant_id_dep)]
_PrincipalUser = Annotated[str | None, Depends(get_user_id_dep)]


router = APIRouter(prefix="/user/favorites", tags=["user-favorites"])


class AddFavoriteRequest(BaseModel):
    """Body for ``POST /user/favorites``.

    Symmetric with the DELETE path parameters so the frontend can
    build a single API helper that switches verb on toggle. Field
    names mirror the Go request struct (``type`` / ``id``).
    """

    model_config = ConfigDict(frozen=True)

    type: str
    id: str


def _require_principal(user_id: str | None, tenant_id: int) -> tuple[str, int]:
    """Return the (user_id, tenant_id) pair, or fail.

    Favorites are always scoped to the current principal; without a
    authenticated user there is no safe default (the empty string
    would collide with PK entries, ``tenant_id == 0`` is the
    system-scope sentinel). The handler raises ``401`` so the frontend
    knows to re-authenticate rather than retry.
    """
    if not user_id or tenant_id == 0:
        raise UnauthorizedError(
            code="auth.principal_context_missing",
            message="unauthorized: principal context missing",
        )
    return user_id, tenant_id


@router.get("", response_model=FavoritesListResponse)
async def list_favorites(
    _auth: AuthDep,
    service: FavoriteServiceDep,
    user_id: _PrincipalUser,
    tenant_id: _PrincipalTenant,
    type: str = Query(
        ...,
        description="Resource type to list favorites for (kb | agent).",
    ),
) -> FavoritesListResponse:
    """List this user's favorites of one resource type.

    ``type`` is required and must be in the service's allowlist; an
    unsupported value yields ``400``. The response is empty when the
    user has not starred anything of the requested type.
    """
    uid, tid = _require_principal(user_id, tenant_id)
    rows = await service.list_favorites(user_id=uid, tenant_id=tid, resource_type=type)
    return favorites_list_response(rows)


@router.post("", response_model=FavoriteMutationResponse, status_code=200)
async def add_favorite(
    _auth: AuthDep,
    service: FavoriteServiceDep,
    user_id: _PrincipalUser,
    tenant_id: _PrincipalTenant,
    body: AddFavoriteRequest,
) -> FavoriteMutationResponse:
    """Star a resource for the current user.

    Idempotent: starring an already-starred resource returns
    ``200`` with no body change, matching the upstream
    ``FirstOrCreate`` semantics.
    """
    uid, tid = _require_principal(user_id, tenant_id)
    await service.add_favorite(
        user_id=uid,
        tenant_id=tid,
        resource_type=body.type,
        resource_id=body.id,
    )
    return FavoriteMutationResponse(success=True)


@router.delete(
    "/{type}/{id}",
    response_model=FavoriteMutationResponse,
)
async def remove_favorite(
    _auth: AuthDep,
    service: FavoriteServiceDep,
    user_id: _PrincipalUser,
    tenant_id: _PrincipalTenant,
    type: str = Path(..., description="Resource type (kb | agent)"),
    id: str = Path(..., description="Resource id"),
) -> FavoriteMutationResponse:
    """Unstar a resource for the current user.

    Idempotent: unstarring a non-starred resource returns ``200``
    with no error, so a double-tap toggle is safe to retry.
    """
    uid, tid = _require_principal(user_id, tenant_id)
    await service.remove_favorite(
        user_id=uid,
        tenant_id=tid,
        resource_type=type,
        resource_id=id,
    )
    return FavoriteMutationResponse(success=True)


__all__ = ["router"]
