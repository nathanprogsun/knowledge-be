"""Wire-shape conversion for the tenant endpoints.

``include_secrets`` defaults to ``False`` because role resolution is not
wired up yet: the four secret-bearing config blobs (``web_search_config``,
``parser_engine_config``, ``credentials``, ``storage_engine_config``) are
redacted in every response until a caller is allowed to see them.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from src.core.contracts.tenants import (
    RetrieverEngineEntry,
    RetrieverEnginesConfig,
    Tenant,
    TenantList,
)
from src.core.tenants.types import RetrieverEngines, TenantInfo


class TenantEnvelope(BaseModel):
    """``{"success": true, "data": {...}}`` - single-tenant responses."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: Tenant


class TenantListEnvelope(BaseModel):
    """``{"success": true, "data": {"items": [...]}}`` - list responses."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: TenantList


class DeleteTenantResponse(BaseModel):
    """``{"success": true, "message": "..."}`` - simple ack response."""

    model_config = ConfigDict(frozen=True)

    success: bool
    message: str


def _to_engines_config(engines: RetrieverEngines) -> RetrieverEnginesConfig:
    return RetrieverEnginesConfig(
        engines=[
            RetrieverEngineEntry(
                retriever_type=entry.retriever_type,
                retriever_engine_type=entry.retriever_engine_type,
            )
            for entry in engines.engines
        ]
    )


def tenant_info_to_contract(info: TenantInfo, *, include_secrets: bool = False) -> Tenant:
    """Project the service DTO onto the frozen wire contract."""
    return Tenant(
        id=info.id,
        name=info.name,
        description=info.description,
        status=info.status,
        retriever_engines=_to_engines_config(info.retriever_engines),
        business=info.business,
        storage_quota=info.storage_quota,
        storage_used=info.storage_used,
        context_config=info.context_config,
        chat_history_config=info.chat_history_config,
        retrieval_config=info.retrieval_config,
        web_search_config=info.web_search_config if include_secrets else None,
        parser_engine_config=info.parser_engine_config if include_secrets else None,
        credentials=info.credentials if include_secrets else None,
        storage_engine_config=info.storage_engine_config if include_secrets else None,
        created_at=info.created_at,
        updated_at=info.updated_at,
        deleted_at=info.deleted_at,
    )


def tenant_envelope(info: TenantInfo) -> TenantEnvelope:
    """Wrap one tenant in the success envelope."""
    return TenantEnvelope(success=True, data=tenant_info_to_contract(info))


def tenant_list_envelope(
    infos: list[TenantInfo],
    *,
    total: int | None = None,
    page: int | None = None,
    page_size: int | None = None,
) -> TenantListEnvelope:
    """Wrap a tenant page in the success envelope.

    ``total`` / ``page`` / ``page_size`` stay ``None`` for the unpaginated
    list endpoint, which answers with ``items`` alone.
    """
    return TenantListEnvelope(
        success=True,
        data=TenantList(
            items=[tenant_info_to_contract(info) for info in infos],
            total=total,
            page=page,
            page_size=page_size,
        ),
    )


__all__ = [
    "DeleteTenantResponse",
    "TenantEnvelope",
    "TenantListEnvelope",
    "tenant_envelope",
    "tenant_info_to_contract",
    "tenant_list_envelope",
]
