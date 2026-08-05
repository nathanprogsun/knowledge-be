"""Web-search HTTP endpoints — provider CRUD + connectivity tests.

Maps the routes declared in ``internal/router/routes_infra.go::RegisterWebSearchProviderRoutes``.

Provider rows hold external service credentials (Bing, Tavily, Google,
etc.). Reads are Viewer+; all mutations and connection tests (which
probe external systems with stored credentials) are Admin+. Tenant
isolation is enforced at the service layer (every read/write carries
``tenant_id``); the handler only forwards the caller's active tenant.

The ``provider_id`` for a new row is generated server-side via ``uuid4``
so callers never see a UUID-before-creation race. The wire response
includes the freshly generated id.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter

from src.app_context.request_context import get_tenant_id
from src.common.exception import ValidationError
from src.common.json import JsonObject
from src.core.contracts.infra import (
    CreateWebSearchProviderRequest,
    TestWebSearchProviderRequest,
    UpdateWebSearchProviderRequest,
)
from src.core.infra.web_search.provider_service import (
    WebSearchClient,
    WebSearchClientRegistry,
)
from src.web.api.infra.web_search.views import (
    WebSearchProviderDeleteResponse,
    WebSearchProviderEnvelope,
    WebSearchProviderListEnvelope,
    WebSearchProviderTypeListEnvelope,
    provider_envelope,
    provider_list_envelope,
    provider_type_list_envelope,
)
from src.web.deps.infra_web_search import WebSearchProviderServiceDep
from src.web.deps.rbac import (
    RoleAdminDep,
    RoleViewerDep,
)
from src.web.middleware.auth import AuthDep

router = APIRouter(prefix="/web-search-providers", tags=["web-search"])


# ── /web-search-providers/types — type metadata (Viewer) ─────────────


@router.get("/types", response_model=WebSearchProviderTypeListEnvelope)
async def list_provider_types(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
) -> WebSearchProviderTypeListEnvelope:
    """Return the registry metadata for every supported provider type."""
    return provider_type_list_envelope()


# ── /web-search-providers — CRUD (writes are Admin) ─────────────────


@router.post("", response_model=WebSearchProviderEnvelope, status_code=201)
async def create_provider(
    _auth: AuthDep,
    _admin: RoleAdminDep,
    body: CreateWebSearchProviderRequest,
    provider_service: WebSearchProviderServiceDep,
) -> WebSearchProviderEnvelope:
    """Create a new provider for the current tenant."""
    info = await provider_service.create_provider(
        tenant_id=_require_context_tenant(),
        name=body.name,
        provider=body.provider,
        description=body.description,
        parameters=body.parameters,
        is_default=bool(body.is_default),
        provider_id=str(uuid.uuid4()),
    )
    return provider_envelope(info)


@router.get("", response_model=WebSearchProviderListEnvelope)
async def list_providers(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
    provider_service: WebSearchProviderServiceDep,
) -> WebSearchProviderListEnvelope:
    """Return every provider saved by the current tenant, oldest first."""
    infos = await provider_service.list_providers(_require_context_tenant())
    return provider_list_envelope(infos)


@router.get("/{provider_id}", response_model=WebSearchProviderEnvelope)
async def get_provider(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
    provider_id: str,
    provider_service: WebSearchProviderServiceDep,
) -> WebSearchProviderEnvelope:
    """Return one provider by id."""
    info = await provider_service.get_provider(_require_context_tenant(), provider_id)
    return provider_envelope(info)


@router.put("/{provider_id}", response_model=WebSearchProviderEnvelope)
async def update_provider(
    _auth: AuthDep,
    _admin: RoleAdminDep,
    provider_id: str,
    body: UpdateWebSearchProviderRequest,
    provider_service: WebSearchProviderServiceDep,
) -> WebSearchProviderEnvelope:
    """Update mutable fields of an existing provider.

    The ``provider`` field is immutable post-creation; the request DTO
    does not carry it. The service enforces the same invariant.
    """
    info = await provider_service.update_provider(
        tenant_id=_require_context_tenant(),
        provider_id=provider_id,
        name=body.name,
        description=body.description,
        parameters=body.parameters,
        is_default=body.is_default,
    )
    return provider_envelope(info)


@router.delete("/{provider_id}", response_model=WebSearchProviderDeleteResponse)
async def delete_provider(
    _auth: AuthDep,
    _admin: RoleAdminDep,
    provider_id: str,
    provider_service: WebSearchProviderServiceDep,
) -> WebSearchProviderDeleteResponse:
    """Soft-delete a provider by id."""
    await provider_service.delete_provider(_require_context_tenant(), provider_id)
    return WebSearchProviderDeleteResponse()


# ── /web-search-providers/test — raw credentials (Admin) ─────────────


@router.post("/test", response_model=WebSearchProviderDeleteResponse)
async def test_provider_raw(
    _auth: AuthDep,
    _admin: RoleAdminDep,
    body: TestWebSearchProviderRequest,
    provider_service: WebSearchProviderServiceDep,
) -> WebSearchProviderDeleteResponse:
    """Test connectivity with raw (unsaved) credentials.

    Implemented as a typed ack: the service raises a ``ValidationError``
    when the upstream search returns no results; the global exception
    handler renders it as ``{"success": false, "error": "..."}``. On
    success the endpoint answers with the standard ack shape.
    """
    await provider_service.test_provider_raw(
        provider=body.provider,
        parameters=body.parameters,
        registry=_NoopClientRegistry(),
    )
    return WebSearchProviderDeleteResponse()


@router.post("/{provider_id}/test", response_model=WebSearchProviderDeleteResponse)
async def test_provider_by_id(
    _auth: AuthDep,
    _admin: RoleAdminDep,
    provider_id: str,
    provider_service: WebSearchProviderServiceDep,
) -> WebSearchProviderDeleteResponse:
    """Test connectivity against the saved configuration."""
    await provider_service.test_provider_by_id(
        tenant_id=_require_context_tenant(),
        provider_id=provider_id,
        registry=_NoopClientRegistry(),
    )
    return WebSearchProviderDeleteResponse()


# ── Helpers ────────────────────────────────────────────────────────


def _require_context_tenant() -> int:
    """Return the current tenant id from request context, or raise."""
    raw = get_tenant_id()
    if raw is None:
        raise ValidationError(
            code="tenant.context_missing",
            message="No active workspace in request context",
        )
    try:
        return int(raw)
    except ValueError as exc:
        raise ValidationError(
            code="tenant.context_invalid",
            message="Active workspace id is invalid",
        ) from exc


class _NoopClientRegistry(WebSearchClientRegistry):
    """Stub registry for the test endpoints.

    The real HTTP client registry (the dispatcher module) is not yet
    wired into these endpoints, so a successful connectivity test
    cannot be observed here. The endpoints still exercise full
    validation + lookup so the route surface is complete.
    """

    def create_provider(
        self,
        provider_type: str,
        params: JsonObject,
    ) -> WebSearchClient:
        raise ValidationError(
            code="web_search_provider.test_unavailable",
            message="Connectivity test requires the upstream HTTP client registry; not yet wired",
        )


__all__ = ["router"]
