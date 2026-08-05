"""Wire-shape conversion for the vector-store endpoints.

The view models match the frozen ``VectorStore`` / ``VectorStoreResponse``
contracts in ``src/core/contracts/infra.py``; the service layer hands
them a ``VectorStoreInfo`` (a typed DTO) and we translate to the wire
shape here so the boundary translation lives next to the endpoint.

The Go response wrapper includes ``source`` (``"user"`` / ``"env"``)
and ``readonly`` columns on every row; the Python wire contract reuses
the same two fields on the same envelope. Env-driven virtual entries
appear via the same list endpoint, but they are synthesised at the
service layer from ``RETRIEVE_DRIVER`` and never persisted.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from src.common.json import JsonObject
from src.core.contracts.infra import (
    CreateVectorStoreRequest,
    TestVectorStoreResponse,
    UpdateVectorStoreRequest,
    VectorStore,
    VectorStoreTypeInfo,
)
from src.core.infra.vector_stores.types import VectorStoreInfo, mask_sensitive_fields

# ── View models (wire shape) ─────────────────────────────────────────


class VectorStoreListResponse(BaseModel):
    """Wire shape for ``GET /vector-stores``."""

    model_config = ConfigDict(frozen=True)

    success: bool = True
    data: list[VectorStore]


class VectorStoreEnvelope(BaseModel):
    """Wire shape for single-vector-store responses (post / get / put)."""

    model_config = ConfigDict(frozen=True)

    success: bool = True
    data: VectorStore


class VectorStoreTypesResponse(BaseModel):
    """Wire shape for ``GET /vector-stores/types``."""

    model_config = ConfigDict(frozen=True)

    success: bool = True
    data: list[VectorStoreTypeInfo]


class DeleteVectorStoreResponse(BaseModel):
    """Wire shape for ``DELETE /vector-stores/{id}``."""

    model_config = ConfigDict(frozen=True)

    success: bool = True


# ── Conversion helpers ──────────────────────────────────────────────


def vector_store_to_contract(
    info: VectorStoreInfo,
    *,
    source: str | None = None,
    readonly: bool | None = None,
) -> VectorStore:
    """Project the service DTO onto the frozen wire contract.

    ``source`` and ``readonly`` default to the values carried on the
    row. The web layer overrides them for env-store virtual entries
    so the response matches the Go contract.
    """
    return VectorStore(
        id=info.id,
        tenant_id=info.tenant_id,
        name=info.name,
        engine_type=info.engine_type,
        connection_config=mask_sensitive_fields(info.connection_config),
        index_config=info.index_config,
        source=source if source is not None else info.source,
        readonly=readonly if readonly is not None else info.readonly,
        created_at=info.created_at,
        updated_at=info.updated_at,
        deleted_at=info.deleted_at,
    )


def vector_store_envelope(
    info: VectorStoreInfo,
    *,
    source: str | None = None,
    readonly: bool | None = None,
) -> VectorStoreEnvelope:
    """Wrap a single vector store in the success envelope."""
    return VectorStoreEnvelope(
        success=True,
        data=vector_store_to_contract(info, source=source, readonly=readonly),
    )


def vector_store_list_envelope(
    infos: list[VectorStoreInfo],
    *,
    sources: list[str] | None = None,
    readonlys: list[bool] | None = None,
) -> VectorStoreListResponse:
    """Wrap a list of stores in the success envelope.

    ``sources`` and ``readonlys`` are optional parallel arrays that
    override the per-row defaults; the web layer uses them to mark
    env-store virtual entries before the response is sent.
    """
    n = len(infos)
    if sources is not None:
        sources_iter: list[str | None] = list(sources) + [None] * (n - len(sources))
    else:
        sources_iter = [None] * n
    if readonlys is not None:
        readonlys_iter: list[bool | None] = list(readonlys) + [None] * (n - len(readonlys))
    else:
        readonlys_iter = [None] * n
    return VectorStoreListResponse(
        success=True,
        data=[
            vector_store_to_contract(
                info,
                source=sources_iter[i],
                readonly=readonlys_iter[i],
            )
            for i, info in enumerate(infos)
        ],
    )


# ── Request models (raw / test) ─────────────────────────────────────


class TestVectorStoreRawRequest(BaseModel):
    """Optional override for the frozen ``TestVectorStoreRequest``.

    The frozen contract already matches this shape exactly; the
    wrapper exists so the router can declare a single canonical
    request body and the validation layer (Pydantic) reuses the
    contract type.
    """

    model_config = ConfigDict(frozen=True)

    engine_type: str
    connection_config: JsonObject = Field(default_factory=dict)


# ── Helpers ──────────────────────────────────────────────────────────


def request_to_create(body: CreateVectorStoreRequest) -> CreateVectorStoreRequest:
    """Identity dispatcher for the wire-shaped create request.

    The wire contract already matches the service-layer input, so the
    conversion is a no-op. The function is kept for symmetry with
    patterns in other domains and to keep the router declarative.
    """
    return body


def request_to_update(body: UpdateVectorStoreRequest) -> UpdateVectorStoreRequest:
    """Identity dispatcher for the wire-shaped update request."""
    return body


def to_test_response(
    response: TestVectorStoreResponse,
) -> TestVectorStoreResponse:
    """Identity dispatcher for the wire-shaped test response.

    The service returns the frozen contract already, so the
    conversion is a no-op. The function is kept for symmetry with
    other domain views (e.g. the source/readonly overrides on the
    store envelope).
    """
    return response


# ── Env-store synthesis ──────────────────────────────────────────────


def vector_store_datetime_from_unix(_: int | None) -> datetime | None:
    """Reserved for env-store timestamp synthesis.

    The Python rewrite builds env-store entries from ``RETRIEVE_DRIVER``
    in the service layer; the timestamps are stubbed at creation time
    so the wire contract always carries a valid datetime. This helper
    is intentionally a no-op until the env-store builder lands.
    """
    return None


__all__ = [
    "DeleteVectorStoreResponse",
    "TestVectorStoreRawRequest",
    "VectorStoreEnvelope",
    "VectorStoreListResponse",
    "VectorStoreTypesResponse",
    "request_to_create",
    "request_to_update",
    "to_test_response",
    "vector_store_datetime_from_unix",
    "vector_store_envelope",
    "vector_store_list_envelope",
    "vector_store_to_contract",
]
