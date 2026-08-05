"""Web-search system-level routes (the builtin-provider catalog).

Maps ``internal/router/routes_infra.go::RegisterWebSearchRoutes`` — a
singleton ``GET /web-search/providers`` that returns the system-level
catalog of enabled providers.

This router is mounted under ``/web-search`` (NOT ``/web-search-providers``
which hosts the tenant CRUD endpoints in ``router.py``). The split
mirrors the upstream Go registration so the URL paths the Go client
expects line up 1:1.
"""

from __future__ import annotations

from fastapi import APIRouter

from src.web.api.infra.web_search.views import (
    WebSearchBuiltinProviderListEnvelope,
    builtin_provider_list_envelope,
)
from src.web.deps.rbac import RoleViewerDep
from src.web.middleware.auth import AuthDep

router = APIRouter(prefix="/web-search", tags=["web-search"])


@router.get("/providers", response_model=WebSearchBuiltinProviderListEnvelope)
async def list_builtin_providers(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
) -> WebSearchBuiltinProviderListEnvelope:
    """Return the system-level list of enabled builtin providers.

    Independent of any tenant's saved configurations; powers the
    "available providers" picker in the management UI.
    """
    return builtin_provider_list_envelope()


__all__ = ["router"]
