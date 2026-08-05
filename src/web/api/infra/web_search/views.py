"""Wire-shape conversion for the web-search endpoints.

Renders the service-side DTOs as the frozen wire contracts in
``src.core.contracts.infra``. The ``api_key`` field is intentionally
NOT exposed — it lives behind the (future) credentials subresource.
``parameters`` is sent as-is minus ``api_key``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from src.core.contracts.infra import (
    WebSearchBuiltinProvider,
    WebSearchProvider,
    WebSearchProviderParameters,
    WebSearchProviderTypeInfo,
)
from src.core.infra.web_search.types import (
    BUILTIN_PROVIDERS,
    PROVIDER_TYPES,
    WebSearchProviderInfo,
)

# ── Response envelopes ───────────────────────────────────────────────


class WebSearchProviderEnvelope(BaseModel):
    """``{"success": true, "data": {...}}`` — single-provider responses."""

    model_config = ConfigDict(frozen=True)

    success: bool = True
    data: WebSearchProvider


class WebSearchProviderListEnvelope(BaseModel):
    """``{"success": true, "data": [...]}`` — list responses."""

    model_config = ConfigDict(frozen=True)

    success: bool = True
    data: list[WebSearchProvider]


class WebSearchProviderTypeListEnvelope(BaseModel):
    """``{"success": true, "data": [...]}`` — type-metadata responses."""

    model_config = ConfigDict(frozen=True)

    success: bool = True
    data: list[WebSearchProviderTypeInfo]


class WebSearchBuiltinProviderListEnvelope(BaseModel):
    """``{"success": true, "data": [...]}`` — builtin provider list."""

    model_config = ConfigDict(frozen=True)

    success: bool = True
    data: list[WebSearchBuiltinProvider]


class WebSearchProviderDeleteResponse(BaseModel):
    """``{"success": true}`` — simple ack response."""

    model_config = ConfigDict(frozen=True)

    success: bool = True


# ── Conversion helpers ───────────────────────────────────────────────


def _mask_parameters(
    params: WebSearchProviderParameters | None,
) -> WebSearchProviderParameters:
    """Return a copy of ``params`` with ``api_key`` redacted.

    The wire contract forbids leaking ``api_key`` through this endpoint —
    credentials live behind a dedicated subresource (PR-19+ followup).
    ``api_key=None`` keeps the field present in the JSON so clients can
    tell that the slot exists.
    """
    if params is None:
        return WebSearchProviderParameters()
    return WebSearchProviderParameters(
        api_key=None,
        cx=params.cx,
        base_url=params.base_url,
        proxy_url=params.proxy_url,
        extra_config=params.extra_config,
    )


def info_to_wire(info: WebSearchProviderInfo) -> WebSearchProvider:
    """Project the service DTO onto the wire contract."""
    return WebSearchProvider(
        id=info.id,
        tenant_id=info.tenant_id,
        name=info.name,
        provider=info.provider,
        description=info.description,
        is_default=info.is_default,
        parameters=_mask_parameters(info.parameters).model_dump(exclude_none=True),
        created_at=info.created_at,
        updated_at=info.updated_at,
        deleted_at=info.deleted_at,
    )


def provider_envelope(info: WebSearchProviderInfo) -> WebSearchProviderEnvelope:
    """Wrap one provider in the success envelope."""
    return WebSearchProviderEnvelope(data=info_to_wire(info))


def provider_list_envelope(
    infos: list[WebSearchProviderInfo],
) -> WebSearchProviderListEnvelope:
    """Wrap a list of providers in the success envelope."""
    return WebSearchProviderListEnvelope(data=[info_to_wire(i) for i in infos])


def provider_type_list_envelope() -> WebSearchProviderTypeListEnvelope:
    """Wrap the registry metadata in the success envelope."""
    return WebSearchProviderTypeListEnvelope(data=list(PROVIDER_TYPES))


def builtin_provider_list_envelope() -> WebSearchBuiltinProviderListEnvelope:
    """Wrap the builtin-provider list in the success envelope."""
    return WebSearchBuiltinProviderListEnvelope(data=list(BUILTIN_PROVIDERS))


__all__ = [
    "WebSearchBuiltinProviderListEnvelope",
    "WebSearchProviderDeleteResponse",
    "WebSearchProviderEnvelope",
    "WebSearchProviderListEnvelope",
    "WebSearchProviderTypeListEnvelope",
    "builtin_provider_list_envelope",
    "info_to_wire",
    "provider_envelope",
    "provider_list_envelope",
    "provider_type_list_envelope",
]
