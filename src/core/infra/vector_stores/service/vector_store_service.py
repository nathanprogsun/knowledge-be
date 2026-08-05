"""VectorStoreService — CRUD + connectivity probe for tenant-scoped stores.

Mirrors ``internal/application/service/vectorstore.go::vectorStoreService``
on the Go side, scoped to the parts that do not require the engine
factory / KB binding guard (those land in later PRs once the engine
registry and the knowledge-base domain are in scope).

The service depends only on the vector-store repository. The web layer
constructs a fresh repo + service per request (via
``factory.build_vector_store_service``).

Behaviour parity notes:

- ``create_store`` validates the engine type against
  ``SUPPORTED_ENGINE_TYPES`` and validates required connection fields
  against the engine-specific rules in ``healthcheck``. The duplicate
  guard (same tenant + engine + endpoint + index) fails
  with :class:`ConflictError` so the web layer renders a 409.
- ``update_store`` only mutates ``name``; engine_type / connection /
  index are immutable post-creation, mirroring the Go service.
- ``delete_store`` rejects unknown ids with :class:`NotFoundError` so
  the web layer renders a 404. The KB-binding guard lives on the Go
  service (``s.kbRepo.CountByVectorStoreID``) and will be wired in via
  the knowledge-base domain.
- ``test_by_id`` runs the connectivity probe against the stored
  config; ``test_raw`` validates user-supplied raw config before
  delegating to the probe (matches the Go
  ``TestRawConnection`` -> ``TestConnection`` split).
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import cast

from src.common.exception import ApplicationError, ConflictError, NotFoundError, ValidationError
from src.common.json import JsonObject, JsonValue
from src.common.oidc_client import validate_ssrf_safe_url
from src.core.contracts.infra import (
    CreateVectorStoreRequest,
    TestVectorStoreResponse,
    UpdateVectorStoreRequest,
)
from src.core.infra.vector_stores.healthcheck import (
    is_valid_engine_type,
    test_connection_async,
    validate_connection_config,
)
from src.core.infra.vector_stores.types import (
    SUPPORTED_ENGINE_TYPES,
    VectorStoreInfo,
)
from src.db.dao.vector_store_repository import (
    VectorStoreRepository,
    _endpoint_of,
    _index_name_of,
)
from src.db.models.infra.vector_store import VectorStore

_NOT_FOUND_CODE = "vector_store.not_found"


def _now() -> datetime:
    """Return a timezone-aware ``now`` for stamping rows."""
    return datetime.now(UTC)


def _new_id() -> str:
    """Generate a UUID for a freshly created vector-store row."""
    return str(uuid.uuid4())


def _require_tenant_id(tenant_id: int) -> None:
    if not isinstance(tenant_id, int) or tenant_id <= 0:
        raise ValidationError(
            code="vector_store.tenant_required",
            message="tenant ID is required",
        )


def _require_engine_type(engine_type: str) -> None:
    if engine_type not in SUPPORTED_ENGINE_TYPES:
        raise ValidationError(
            code="vector_store.invalid_engine_type",
            message=f"unsupported engine type: {engine_type}",
        )


def _require_name(name: str) -> str:
    clean = name.strip()
    if not clean:
        raise ValidationError(
            code="vector_store.name_required",
            message="name is required",
        )
    return clean


class VectorStoreService:
    """Stateless vector-store service, constructed per request."""

    def __init__(
        self,
        *,
        vector_store_repo: VectorStoreRepository,
    ) -> None:
        self._vector_store_repo = vector_store_repo

    # ── Reads ───────────────────────────────────────────────────────

    async def list_stores(self, tenant_id: int) -> list[VectorStoreInfo]:
        """Return every live vector store of the tenant, newest first."""
        _require_tenant_id(tenant_id)
        rows = await self._vector_store_repo.list_for_tenant(tenant_id)
        return [VectorStoreInfo.map_from_db(row) for row in rows]

    async def get_store(self, tenant_id: int, store_id: str) -> VectorStoreInfo:
        """Return one store by id within the tenant scope, or raise."""
        _require_tenant_id(tenant_id)
        row = await self._vector_store_repo.get_by_id(tenant_id, store_id)
        if row is None:
            raise NotFoundError(
                code=_NOT_FOUND_CODE,
                message=f"vector store {store_id} not found",
            )
        return VectorStoreInfo.map_from_db(row)

    async def get_stores_by_ids(
        self,
        tenant_id: int,
        store_ids: list[str],
    ) -> list[VectorStoreInfo]:
        """Return stores for the given ids within the tenant scope.

        Missing ids are silently dropped; the web layer never surfaces
        an error from a batch resolve.
        """
        _require_tenant_id(tenant_id)
        out: list[VectorStoreInfo] = []
        for sid in store_ids:
            row = await self._vector_store_repo.get_by_id(tenant_id, sid)
            if row is not None:
                out.append(VectorStoreInfo.map_from_db(row))
        return out

    # ── Create ──────────────────────────────────────────────────────

    async def create_store(
        self,
        *,
        tenant_id: int,
        body: CreateVectorStoreRequest,
    ) -> VectorStoreInfo:
        """Insert a new vector-store row.

        Required fields are validated at the boundary; the duplicate
        guard fails with :class:`ConflictError` on a collision.
        Connection-config validation mirrors the Go
        ``validateConnectionConfig`` set, but the connectivity probe
        itself is opt-in via the test endpoints (the Go service
        probes on create; the Python rewrite keeps the probe separate
        so the test endpoint can stand alone for the UI).
        """
        _require_tenant_id(tenant_id)
        _require_engine_type(body.engine_type)
        clean_name = _require_name(body.name)
        self._validate_connection_config(body.engine_type, body.connection_config)
        endpoint = _endpoint_of(cast("JsonObject | None", body.connection_config))
        index_name = _index_name_of(body.index_config, body.engine_type)
        if await self._vector_store_repo.exists_by_engine_type_endpoint_index(
            tenant_id=tenant_id,
            engine_type=body.engine_type,
            endpoint=endpoint,
            index_name=index_name,
        ):
            raise ConflictError(
                code="vector_store.duplicate",
                message="a vector store with the same endpoint and index already exists",
            )
        now = _now()
        row = VectorStore(
            id=_new_id(),
            tenant_id=tenant_id,
            name=clean_name,
            engine_type=body.engine_type,
            connection_config=_normalise_connection_config(body.connection_config),
            index_config=_normalise_index_config(body.index_config),
            source="user",
            readonly=False,
            created_at=now,
            updated_at=now,
        )
        try:
            persisted = await self._vector_store_repo.insert(row)
        except Exception as exc:  # pragma: no cover - rare race path
            # The DB-level unique constraint (tenant + name) is a second
            # line of defence behind the endpoint+index pre-check; a
            # same-name store with a different endpoint collides here.
            name = exc.__class__.__name__
            if "UniqueViolation" in name and "tenant_name" in str(exc):
                raise ConflictError(
                    code="vector_store.duplicate_name",
                    message=(
                        f"a vector store named {clean_name!r} already exists in this workspace"
                    ),
                ) from exc
            raise
        return VectorStoreInfo.map_from_db(persisted)

    # ── Update ──────────────────────────────────────────────────────

    async def update_store(
        self,
        *,
        tenant_id: int,
        store_id: str,
        body: UpdateVectorStoreRequest,
    ) -> VectorStoreInfo:
        """Update the mutable fields of an existing store.

        Only ``name`` is mutable; the service rejects requests that
        try to mutate the engine type, connection config, or index
        config (the request body itself is frozen and does not expose
        those fields, so the protection is mostly defence-in-depth).
        """
        _require_tenant_id(tenant_id)
        existing = await self._vector_store_repo.get_by_id(tenant_id, store_id)
        if existing is None:
            raise ValidationError(
                code=_NOT_FOUND_CODE,
                message=f"vector store {store_id} not found",
            )
        clean_name = _require_name(body.name)
        now = _now()
        updated = await self._vector_store_repo.update_by_primary_key(
            {"id": store_id},
            {"name": clean_name, "updated_at": now},
        )
        if updated is None:
            raise NotFoundError(
                code=_NOT_FOUND_CODE,
                message=f"vector store {store_id} not found",
            )
        return VectorStoreInfo.map_from_db(updated)

    # ── Delete ──────────────────────────────────────────────────────

    async def delete_store(self, tenant_id: int, store_id: str) -> bool:
        """Soft-delete a store. Returns ``True`` if a row was deleted.

        Mirrors the Go service's idempotent semantics: deleting an
        unknown or already-deleted store is reported as ``False``
        rather than a 404, so the web layer can render a uniform
        200/204. The KB-binding guard lives on the Go service and
        will be wired in later.
        """
        _require_tenant_id(tenant_id)
        existing = await self._vector_store_repo.get_by_id(tenant_id, store_id)
        if existing is None:
            return False
        now = _now()
        updated = await self._vector_store_repo.update_by_primary_key(
            {"id": store_id},
            {"deleted_at": now, "updated_at": now},
            exclude_deleted_or_archived=False,
        )
        return updated is not None

    async def require_store(self, tenant_id: int, store_id: str) -> VectorStoreInfo:
        """Return one store or raise :class:`NotFoundError` (404-style)."""
        _require_tenant_id(tenant_id)
        row = await self._vector_store_repo.get_by_id(tenant_id, store_id)
        if row is None:
            raise NotFoundError(
                code=_NOT_FOUND_CODE,
                message=f"vector store {store_id} not found",
            )
        return VectorStoreInfo.map_from_db(row)

    # ── Connect probes ──────────────────────────────────────────────

    async def test_by_id(
        self,
        tenant_id: int,
        store_id: str,
    ) -> TestVectorStoreResponse:
        """Run the connectivity probe against the stored config."""
        info = await self.require_store(tenant_id, store_id)
        return await self._run_test(
            engine_type=info.engine_type,
            connection_config=info.connection_config or {},
            allowlist_only=False,
        )

    async def test_raw(
        self,
        engine_type: str,
        connection_config: JsonObject,
    ) -> TestVectorStoreResponse:
        """Run the connectivity probe against raw user-supplied config.

        Mirrors the Go ``TestRawConnection`` path: the engine-type
        allowlist + required-field checks run before the probe so a
        raw postgres probe against the application's own DB host
        cannot be used as a credential oracle.
        """
        return await self._run_test(
            engine_type=engine_type,
            connection_config=connection_config,
            allowlist_only=True,
        )

    # ── Internal helpers ────────────────────────────────────────────

    async def _run_test(
        self,
        *,
        engine_type: str,
        connection_config: Mapping[str, JsonValue],
        allowlist_only: bool,
    ) -> TestVectorStoreResponse:
        if allowlist_only and not is_valid_engine_type(engine_type):
            raise ValidationError(
                code="vector_store.unsupported_engine",
                message=f"connection test is not supported for engine type: {engine_type}",
            )
        self._validate_connection_config(engine_type, connection_config)
        # SSRF guard on the probe target, mirroring the storage-backend
        # test endpoints: the probe must not be usable as an internal
        # port scanner. ``addr`` (elasticsearch/milvus/...) or ``host``
        # (qdrant) carries the endpoint.
        endpoint = connection_config.get("addr") or connection_config.get("host")
        if isinstance(endpoint, str) and endpoint.strip():
            try:
                await validate_ssrf_safe_url(endpoint)
            except ApplicationError as exc:
                raise ValidationError(
                    code="vector_store.endpoint_ssrf_blocked",
                    message="vector store endpoint failed SSRF validation",
                    details={"reason": exc.message},
                ) from exc
        version, error = await test_connection_async(engine_type, connection_config)
        if error is not None:
            return TestVectorStoreResponse(
                success=False,
                version=None,
                error=error.message,
            )
        return TestVectorStoreResponse(success=True, version=version, error=None)

    @staticmethod
    def _validate_connection_config(
        engine_type: str,
        config: Mapping[str, JsonValue],
    ) -> None:
        """Apply the engine-specific required-field checks.

        Coerces the wire ``JsonObject`` to a plain ``Mapping`` for the
        helper so callers do not need to know the wire contravariant
        type.
        """
        validate_connection_config(engine_type, config)


# ── Free helpers ─────────────────────────────────────────────────────


def _normalise_connection_config(
    raw: JsonObject,
) -> JsonObject:
    """Strip ``None`` values from the connection config blob.

    Mirrors the Go ``omitempty`` JSON tags on ``ConnectionConfig`` so the
    stored shape is stable across rewrites.
    """
    return {k: v for k, v in raw.items() if v is not None}


def _normalise_index_config(
    raw: JsonObject | None,
) -> JsonObject | None:
    """Strip ``None`` values from the index config blob."""
    if raw is None:
        return None
    return {k: v for k, v in raw.items() if v is not None}


__all__ = ["VectorStoreService"]
