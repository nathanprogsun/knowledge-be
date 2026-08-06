"""VectorStore HTTP endpoints - CRUD + connectivity probes.

Maps the eight endpoints declared in the corresponding upstream handler
and ``docs/api/vector-store.md``:

| Method | Path                          | Role gate                  |
| ------ | ----------------------------- | -------------------------- |
| GET    | `/vector-stores/types`        | Viewer                     |
| POST   | `/vector-stores/test`         | Admin                      |
| POST   | `/vector-stores`              | Admin                      |
| GET    | `/vector-stores`              | Viewer                     |
| GET    | `/vector-stores/{id}`         | Viewer                     |
| PUT    | `/vector-stores/{id}`         | Admin                      |
| DELETE | `/vector-stores/{id}`         | Admin                      |
| POST   | `/vector-stores/{id}/test`    | Admin                      |

The upstream handler merges env-store virtual entries (synthesised
from ``RETRIEVE_DRIVER``) into the list / get responses. The Python
rewrite keeps that behaviour in the service layer so the web layer
stays declarative: the list endpoint returns DB rows + env entries,
with ``source`` / ``readonly`` overridden on the wire.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Annotated, cast

from fastapi import APIRouter, Depends, Path

from src.common.exception import NotFoundError, ValidationError
from src.common.json import JsonObject
from src.core.contracts.infra import (
    CreateVectorStoreRequest,
    TestVectorStoreResponse,
    UpdateVectorStoreRequest,
    VectorStoreTypeInfo,
)
from src.core.infra.vector_stores.types import (
    VectorStoreInfo,
    vector_store_types,
)
from src.web.api.infra.vector_stores.views import (
    DeleteVectorStoreResponse,
    TestVectorStoreRawRequest,
    VectorStoreEnvelope,
    VectorStoreListResponse,
    VectorStoreTypesResponse,
    vector_store_envelope,
    vector_store_list_envelope,
)
from src.web.deps import (
    AuthDep,
    RoleAdminDep,
    RoleViewerDep,
)
from src.web.deps.context import get_tenant_id_dep
from src.web.deps.infra_vector_stores import VectorStoreServiceDep

# Function-arg-style principal dep alias.
_PrincipalTenant = Annotated[int, Depends(get_tenant_id_dep)]


router = APIRouter(prefix="/vector-stores", tags=["vector-stores"])

# Constants for env-store synthesis. The upstream side reads
# ``RETRIEVE_DRIVER`` via ``os.Getenv``; the Python rewrite mirrors
# that.
_ENV_STORE_ID_PREFIX = "__env_"


def _require_tenant_id(tenant_id: int) -> int:
    """Return the current tenant id, or raise when missing."""
    if tenant_id == 0:
        raise ValidationError(
            code="tenant.context_missing",
            message="No active workspace in request context",
        )
    return tenant_id


# ── Env-store synthesis ──────────────────────────────────────────────


def _is_env_store_id(store_id: str) -> bool:
    """Mirror of the upstream ``types.IsEnvStoreID``."""
    return store_id.startswith(_ENV_STORE_ID_PREFIX)


def _build_env_store_entries(retrieve_driver: str) -> list[VectorStoreInfo]:
    """Synthesise env-store virtual entries from ``RETRIEVE_DRIVER``.

    Mirrors the same shape as the upstream ``types.BuildEnvVectorStores``.
    Drivers with no store (postgres, sqlite) are still surfaced so the
    UI can render "System default" badges.
    """
    entries: list[VectorStoreInfo] = []
    if not retrieve_driver:
        return entries
    now = _env_now()
    for driver in retrieve_driver.split(","):
        driver = driver.strip()
        if not driver:
            continue
        if driver == "postgres":
            entries.append(
                VectorStoreInfo(
                    id=f"{_ENV_STORE_ID_PREFIX}postgres__",
                    tenant_id=0,
                    name="PostgreSQL",
                    engine_type="postgres",
                    connection_config={"use_default_connection": True},
                    index_config=None,
                    source="env",
                    readonly=True,
                    created_at=now,
                    updated_at=now,
                )
            )
        elif driver == "sqlite":
            entries.append(
                VectorStoreInfo(
                    id=f"{_ENV_STORE_ID_PREFIX}sqlite__",
                    tenant_id=0,
                    name="SQLite",
                    engine_type="sqlite",
                    connection_config=None,
                    index_config=None,
                    source="env",
                    readonly=True,
                    created_at=now,
                    updated_at=now,
                )
            )
        elif driver == "elasticsearch_v8":
            entries.append(
                VectorStoreInfo(
                    id=f"{_ENV_STORE_ID_PREFIX}elasticsearch_v8__",
                    tenant_id=0,
                    name="Elasticsearch v8",
                    engine_type="elasticsearch",
                    connection_config={
                        "addr": os.environ.get("ELASTICSEARCH_ADDR"),
                        "username": os.environ.get("ELASTICSEARCH_USERNAME"),
                        "password": os.environ.get("ELASTICSEARCH_PASSWORD"),
                    },
                    index_config={
                        "index_name": os.environ.get("ELASTICSEARCH_INDEX"),
                    },
                    source="env",
                    readonly=True,
                    created_at=now,
                    updated_at=now,
                )
            )
        elif driver == "elasticsearch_v7":
            entries.append(
                VectorStoreInfo(
                    id=f"{_ENV_STORE_ID_PREFIX}elasticsearch_v7__",
                    tenant_id=0,
                    name="Elasticsearch v7",
                    engine_type="elasticsearch",
                    connection_config={
                        "addr": os.environ.get("ELASTICSEARCH_ADDR"),
                        "username": os.environ.get("ELASTICSEARCH_USERNAME"),
                        "password": os.environ.get("ELASTICSEARCH_PASSWORD"),
                    },
                    index_config={
                        "index_name": os.environ.get("ELASTICSEARCH_INDEX"),
                    },
                    source="env",
                    readonly=True,
                    created_at=now,
                    updated_at=now,
                )
            )
        elif driver == "opensearch":
            entries.append(
                VectorStoreInfo(
                    id=f"{_ENV_STORE_ID_PREFIX}opensearch__",
                    tenant_id=0,
                    name="OpenSearch",
                    engine_type="opensearch",
                    connection_config={
                        "addr": os.environ.get("OPENSEARCH_ADDR"),
                        "username": os.environ.get("OPENSEARCH_USERNAME"),
                        "password": os.environ.get("OPENSEARCH_PASSWORD"),
                        "insecure_skip_verify": _truthy(
                            os.environ.get("OPENSEARCH_INSECURE_SKIP_VERIFY")
                        ),
                    },
                    index_config={
                        "index_name": os.environ.get("OPENSEARCH_INDEX"),
                    },
                    source="env",
                    readonly=True,
                    created_at=now,
                    updated_at=now,
                )
            )
        elif driver == "qdrant":
            entries.append(
                VectorStoreInfo(
                    id=f"{_ENV_STORE_ID_PREFIX}qdrant__",
                    tenant_id=0,
                    name="Qdrant",
                    engine_type="qdrant",
                    connection_config={
                        "host": os.environ.get("QDRANT_HOST"),
                        "api_key": os.environ.get("QDRANT_API_KEY"),
                    },
                    index_config=None,
                    source="env",
                    readonly=True,
                    created_at=now,
                    updated_at=now,
                )
            )
        elif driver == "milvus":
            entries.append(
                VectorStoreInfo(
                    id=f"{_ENV_STORE_ID_PREFIX}milvus__",
                    tenant_id=0,
                    name="Milvus",
                    engine_type="milvus",
                    connection_config={
                        "addr": os.environ.get("MILVUS_ADDRESS"),
                        "username": os.environ.get("MILVUS_USERNAME"),
                        "password": os.environ.get("MILVUS_PASSWORD"),
                    },
                    index_config=None,
                    source="env",
                    readonly=True,
                    created_at=now,
                    updated_at=now,
                )
            )
        elif driver == "tencent_vectordb":
            entries.append(
                VectorStoreInfo(
                    id=f"{_ENV_STORE_ID_PREFIX}tencent_vectordb__",
                    tenant_id=0,
                    name="Tencent VectorDB",
                    engine_type="tencent_vectordb",
                    connection_config={
                        "addr": os.environ.get("TENCENT_VECTORDB_ADDR"),
                        "username": os.environ.get("TENCENT_VECTORDB_USERNAME"),
                        "api_key": os.environ.get("TENCENT_VECTORDB_API_KEY"),
                        "database": os.environ.get("TENCENT_VECTORDB_DATABASE"),
                    },
                    index_config={
                        "collection_name": os.environ.get("TENCENT_VECTORDB_COLLECTION"),
                    },
                    source="env",
                    readonly=True,
                    created_at=now,
                    updated_at=now,
                )
            )
        elif driver == "weaviate":
            entries.append(
                VectorStoreInfo(
                    id=f"{_ENV_STORE_ID_PREFIX}weaviate__",
                    tenant_id=0,
                    name="Weaviate",
                    engine_type="weaviate",
                    connection_config={
                        "host": os.environ.get("WEAVIATE_HOST"),
                        "grpc_address": os.environ.get("WEAVIATE_GRPC_ADDRESS"),
                        "scheme": os.environ.get("WEAVIATE_SCHEME"),
                        "api_key": os.environ.get("WEAVIATE_API_KEY"),
                    },
                    index_config=None,
                    source="env",
                    readonly=True,
                    created_at=now,
                    updated_at=now,
                )
            )
        elif driver == "doris":
            http_port_raw = os.environ.get("DORIS_HTTP_PORT")
            try:
                http_port = int(http_port_raw) if http_port_raw else 0
            except ValueError:
                http_port = 0
            entries.append(
                VectorStoreInfo(
                    id=f"{_ENV_STORE_ID_PREFIX}doris__",
                    tenant_id=0,
                    name="Apache Doris",
                    engine_type="doris",
                    connection_config={
                        "addr": os.environ.get("DORIS_ADDR"),
                        "http_port": http_port,
                        "database": os.environ.get("DORIS_DATABASE"),
                        "username": os.environ.get("DORIS_USERNAME"),
                        "password": os.environ.get("DORIS_PASSWORD"),
                    },
                    index_config={
                        "collection_prefix": os.environ.get("DORIS_TABLE_PREFIX"),
                    },
                    source="env",
                    readonly=True,
                    created_at=now,
                    updated_at=now,
                )
            )
    return entries


def _env_now() -> datetime:
    """Return the current UTC datetime for env-store timestamp synthesis."""
    return datetime.now(UTC)


def _truthy(value: str | None) -> bool:
    """Return True for environment values like ``true`` / ``1``."""
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


# ── Endpoints ────────────────────────────────────────────────────────


@router.get("/types", response_model=VectorStoreTypesResponse)
async def list_vector_store_types(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
) -> VectorStoreTypesResponse:
    """Return the registry metadata for every supported engine type.

    Mirrors the upstream ``GET /vector-stores/types`` endpoint. The
    answer is static (mirrors ``types.GetVectorStoreTypes``) so no
    service call is needed.
    """
    types = cast("list[VectorStoreTypeInfo]", list(vector_store_types()))
    return VectorStoreTypesResponse(success=True, data=types)


@router.post("/test", response_model=TestVectorStoreResponse)
async def test_vector_store_raw(
    _auth: AuthDep,
    _admin: RoleAdminDep,
    body: TestVectorStoreRawRequest,
    service: VectorStoreServiceDep,
) -> TestVectorStoreResponse:
    """Test a raw user-supplied connection config (no persistence).

    A validation failure (SSRF block, missing field) answers 200 with
    ``success=false`` - upstream ``TestRawConnection`` keeps the HTTP
    status at 200 and reports the error in the body.
    """
    try:
        return await service.test_raw(
            engine_type=body.engine_type,
            connection_config=body.connection_config,
        )
    except ValidationError as exc:
        return TestVectorStoreResponse(success=False, version=None, error=exc.message)


@router.post("", response_model=VectorStoreEnvelope, status_code=201)
async def create_vector_store(
    _auth: AuthDep,
    _admin: RoleAdminDep,
    body: CreateVectorStoreRequest,
    service: VectorStoreServiceDep,
    tenant_id: _PrincipalTenant,
) -> VectorStoreEnvelope:
    """Create a new vector store for the active workspace."""
    tenant_id = _require_tenant_id(tenant_id)
    info = await service.create_store(tenant_id=tenant_id, body=body)
    return vector_store_envelope(info)


@router.get("", response_model=VectorStoreListResponse)
async def list_vector_stores(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
    service: VectorStoreServiceDep,
    tenant_id: _PrincipalTenant,
) -> VectorStoreListResponse:
    """List every vector store visible to the active workspace.

    Env-store virtual entries (synthesised from ``RETRIEVE_DRIVER``)
    appear first, followed by the tenant's DB-managed rows. Each
    entry carries ``source`` and ``readonly`` so the UI can render
    the right badges.
    """
    tenant_id = _require_tenant_id(tenant_id)
    env_entries = _build_env_store_entries(os.environ.get("RETRIEVE_DRIVER", ""))
    db_entries = await service.list_stores(tenant_id)
    combined = env_entries + db_entries
    sources = [info.source for info in combined]
    readonlys = [info.readonly for info in combined]
    return vector_store_list_envelope(
        combined,
        sources=sources,
        readonlys=readonlys,
    )


@router.get("/{store_id}", response_model=VectorStoreEnvelope)
async def get_vector_store(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
    service: VectorStoreServiceDep,
    tenant_id: _PrincipalTenant,
    store_id: str = Path(...),
) -> VectorStoreEnvelope:
    """Return one vector store by id; env-store virtual entries are allowed.

    Env-store entries surface with ``source="env"`` and ``readonly=True``
    so the UI can render the same badge shape as the list endpoint.
    """
    tenant_id = _require_tenant_id(tenant_id)
    if _is_env_store_id(store_id):
        env_entries = _build_env_store_entries(os.environ.get("RETRIEVE_DRIVER", ""))
        for entry in env_entries:
            if entry.id == store_id:
                return vector_store_envelope(entry)
        raise NotFoundError(
            code="vector_store.not_found",
            message=f"vector store {store_id} not found",
        )
    info = await service.get_store(tenant_id, store_id)
    return vector_store_envelope(info)


@router.put("/{store_id}", response_model=VectorStoreEnvelope)
async def update_vector_store(
    _auth: AuthDep,
    _admin: RoleAdminDep,
    service: VectorStoreServiceDep,
    body: UpdateVectorStoreRequest,
    tenant_id: _PrincipalTenant,
    store_id: str = Path(...),
) -> VectorStoreEnvelope:
    """Update the mutable ``name`` field of a vector store.

    Env-store entries and unknown ids are rejected.
    """
    tenant_id = _require_tenant_id(tenant_id)
    if _is_env_store_id(store_id):
        raise ValidationError(
            code="vector_store.env_store_readonly",
            message="environment-configured vector stores cannot be modified via API",
        )
    info = await service.update_store(
        tenant_id=tenant_id,
        store_id=store_id,
        body=body,
    )
    return vector_store_envelope(info)


@router.delete("/{store_id}", response_model=DeleteVectorStoreResponse)
async def delete_vector_store(
    _auth: AuthDep,
    _admin: RoleAdminDep,
    service: VectorStoreServiceDep,
    tenant_id: _PrincipalTenant,
    store_id: str = Path(...),
) -> DeleteVectorStoreResponse:
    """Soft-delete a vector store. Env-store entries are rejected.

    An unknown id answers 404 - upstream handler checks ownership
    first (the upstream ``DeleteVectorStore`` handler).
    """
    tenant_id = _require_tenant_id(tenant_id)
    if _is_env_store_id(store_id):
        raise ValidationError(
            code="vector_store.env_store_readonly",
            message="environment-configured vector stores cannot be modified via API",
        )
    deleted = await service.delete_store(tenant_id, store_id)
    if not deleted:
        raise NotFoundError(
            code="vector_store.not_found",
            message=f"vector store {store_id} not found",
        )
    return DeleteVectorStoreResponse(success=True)


@router.post("/{store_id}/test", response_model=TestVectorStoreResponse)
async def test_vector_store_by_id(
    _auth: AuthDep,
    _admin: RoleAdminDep,
    service: VectorStoreServiceDep,
    tenant_id: _PrincipalTenant,
    store_id: str = Path(...),
) -> TestVectorStoreResponse:
    """Run the connectivity probe against an existing store.

    Env-store entries are probed in place (the same probe path used
    by the upstream handler). The DB-row path uses the stored config.
    """
    tenant_id = _require_tenant_id(tenant_id)
    try:
        if _is_env_store_id(store_id):
            env_entries = _build_env_store_entries(os.environ.get("RETRIEVE_DRIVER", ""))
            for entry in env_entries:
                if entry.id == store_id:
                    return await service.test_raw(
                        engine_type=entry.engine_type,
                        connection_config=cast("JsonObject", entry.connection_config or {}),
                    )
            raise ValidationError(
                code="vector_store.not_found",
                message=f"vector store {store_id} not found",
            )
        return await service.test_by_id(tenant_id, store_id)
    except ValidationError as exc:
        # Upstream ``TestStoreByID`` answers 200 with the error in the body.
        return TestVectorStoreResponse(success=False, version=None, error=exc.message)


# Re-export the wire-shape VectorStore so the router module stays
# self-contained (used by the response_model ``TestVectorStoreResponse``
# chain via the contract type).
__all__ = ["router"]
