"""Storage-backend registry service — CRUD + connectivity + resolution.

Mirrors ``internal/application/service/service/storagebackend.go::StorageBackendService``.
The Go methods map one-for-one:

- ``create`` — validate, SSRF-check the endpoint, probe connectivity, insert.
- ``update`` — env rows are read-only, provider and physical location are
  immutable, redacted secrets are preserved, disabling a referenced
  backend is refused; then re-validate, re-probe, persist.
- ``delete`` — soft-delete, refusing the workspace default, a backend with
  bound knowledge bases or active resources, an env row, or a legacy alias.
- ``set_default`` — point ``tenants.default_storage_backend_id`` at an
  active backend.
- ``test`` — run the provider probe; ``local`` creates its directory first.
- ``resolve_backend`` — backend id wins, provider is a legacy-alias
  fallback, then the workspace default.

``resolve_file_service`` from Go is deliberately absent: it returns a
``FileService`` (upload/download), which arrives with the file domain.
``resolve_backend`` — the half that selects the instance — is
implemented here because the registry owns that decision.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from src.ai.storage.base import StorageAdapter
from src.ai.storage.cos_backend import CosStorageAdapter
from src.ai.storage.local_backend import LocalStorageAdapter
from src.ai.storage.minio_backend import MINIO_MODE_DOCKER, MinioStorageAdapter
from src.ai.storage.obs_backend import ObsStorageAdapter
from src.ai.storage.s3_backend import S3StorageAdapter
from src.common.exception import (
    ApplicationError,
    ConflictError,
    NotFoundError,
    ValidationError,
)
from src.common.json import BindParams
from src.common.oidc_client import validate_ssrf_safe_url
from src.core.infra.storage_backends.types import (
    StorageBackendConfigInfo,
    StorageBackendInfo,
    StorageBackendListResult,
    StorageConnectivityResult,
    allowed_providers,
    is_provider_allowed,
)
from src.db.dao.storage_backend_repository import StorageBackendRepository
from src.db.models.storage_backend import (
    STORAGE_BACKEND_SOURCE_ENV,
    STORAGE_BACKEND_SOURCE_USER,
    STORAGE_BACKEND_STATUS_ACTIVE,
    STORAGE_BACKEND_STATUS_DISABLED,
    StorageBackend,
)

# Providers whose config carries no reachable endpoint to SSRF-check:
# ``local`` is a filesystem path and ``cos`` is addressed by region.
_ENDPOINT_EXEMPT_PROVIDERS: frozenset[str] = frozenset({"local", "cos"})

_VALID_STATUSES: frozenset[str] = frozenset(
    {STORAGE_BACKEND_STATUS_ACTIVE, STORAGE_BACKEND_STATUS_DISABLED}
)

# Providers served by the generic S3 adapter, with the label each reports.
_S3_COMPATIBLE_LABELS: dict[str, str] = {
    "s3": "S3",
    "tos": "TOS",
    "oss": "OSS",
    "ks3": "KS3",
}


class StorageBackendService:
    """Workspace-scoped registry of concrete storage instances."""

    def __init__(self, *, backend_repo: StorageBackendRepository) -> None:
        self._repo = backend_repo

    # ── Reads ───────────────────────────────────────────────────────

    async def list_backends(self, tenant_id: int) -> StorageBackendListResult:
        """Return the workspace's backends (credentials masked) + default id."""
        rows = await self._repo.list_for_tenant(tenant_id)
        default_id = await self._repo.get_default_backend_id(tenant_id)
        return StorageBackendListResult(
            backends=[StorageBackendInfo.map_from_db(row).masked() for row in rows],
            default_storage_backend_id=default_id,
        )

    async def get_backend(self, *, tenant_id: int, id: str) -> StorageBackendInfo:
        """Return one backend with credentials masked.

        Raises ``NotFoundError`` when the id does not belong to a live row
        of this workspace.
        """
        return (await self._require_backend(tenant_id=tenant_id, id=id)).masked()

    def list_provider_types(self) -> list[str]:
        """Return the provider names permitted by ``STORAGE_ALLOW_LIST``."""
        return list(allowed_providers())

    # ── Create ──────────────────────────────────────────────────────

    async def create(
        self,
        *,
        tenant_id: int,
        name: str,
        provider: str,
        config: StorageBackendConfigInfo,
        status: str = "",
    ) -> StorageBackendInfo:
        """Validate, probe and persist a new backend.

        The connectivity probe runs *before* the insert (as in Go), so a
        misconfigured backend never reaches the table. A duplicate name
        surfaces as ``ConflictError``.
        """
        normalized_name = name.strip()
        normalized_provider = provider.strip().lower()
        resolved_status = status.strip() or STORAGE_BACKEND_STATUS_ACTIVE
        self._validate(
            tenant_id=tenant_id,
            name=normalized_name,
            provider=normalized_provider,
            config=config,
            status=resolved_status,
        )
        await self._validate_endpoint(provider=normalized_provider, config=config)
        await self._probe(provider=normalized_provider, config=config)

        existing = await self._repo.find_by_name(tenant_id=tenant_id, name=normalized_name)
        if existing is not None:
            raise ConflictError(
                code="storage_backend.duplicate_name",
                message="a storage backend with this name already exists",
            )
        now = datetime.now(UTC)
        row = StorageBackend(
            id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            name=normalized_name,
            provider=normalized_provider,
            config=config.to_json(),
            source=STORAGE_BACKEND_SOURCE_USER,
            status=resolved_status,
            legacy_alias=False,
            created_at=now,
            updated_at=now,
        )
        # Mask before returning so POST responses never carry plaintext
        # credentials. Mirrors ``NewStorageBackendResponse`` in Go, which
        # always routes through ``MaskSensitiveFields`` — the same
        # invariant ``list_backends`` / ``get_backend`` already enforce.
        return StorageBackendInfo.map_from_db(await self._repo.create(row)).masked()

    # ── Update ──────────────────────────────────────────────────────

    async def update(
        self,
        *,
        tenant_id: int,
        id: str,
        name: str | None = None,
        config: StorageBackendConfigInfo | None = None,
        status: str | None = None,
    ) -> StorageBackendInfo:
        """Update a backend's mutable fields.

        ``provider`` is not a parameter: Go overwrites any incoming value
        with the stored one, making it immutable. The physical location
        (endpoint / region / bucket / path prefix) is immutable too —
        moving data is a migration, not an edit.
        """
        existing = await self._require_backend(tenant_id=tenant_id, id=id)
        if existing.source == STORAGE_BACKEND_SOURCE_ENV:
            raise ValidationError(
                code="storage_backend.env_read_only",
                message="environment storage backend is read-only",
            )

        merged_config = (config or existing.config).merge_secrets(existing.config)
        if merged_config.location_key(existing.provider) != existing.config.location_key(
            existing.provider
        ):
            raise ValidationError(
                code="storage_backend.immutable_location",
                message=(
                    "endpoint, region, bucket and path prefix are immutable; "
                    "use storage migration instead"
                ),
            )

        resolved_name = (name or existing.name).strip()
        resolved_status = (status or existing.status).strip() or existing.status
        if resolved_status == STORAGE_BACKEND_STATUS_DISABLED and not existing.is_disabled:
            await self._reject_if_referenced(
                tenant_id=tenant_id,
                id=id,
                message="a default or bound storage backend cannot be disabled",
            )

        self._validate(
            tenant_id=tenant_id,
            name=resolved_name,
            provider=existing.provider,
            config=merged_config,
            status=resolved_status,
        )
        await self._validate_endpoint(provider=existing.provider, config=merged_config)
        await self._probe(provider=existing.provider, config=merged_config)

        columns: BindParams = {
            "name": resolved_name,
            "config": merged_config.to_json(),
            "status": resolved_status,
            "updated_at": datetime.now(UTC),
        }
        updated = await self._repo.update_columns(tenant_id=tenant_id, id=id, columns=columns)
        if updated is None:
            raise NotFoundError(
                code="storage_backend.not_found",
                message="storage backend not found",
            )
        # Mask before returning so PUT responses never carry plaintext
        # credentials. Mirrors ``NewStorageBackendResponse`` in Go, which
        # always routes through ``MaskSensitiveFields`` — the same
        # invariant ``list_backends`` / ``get_backend`` already enforce.
        return StorageBackendInfo.map_from_db(updated).masked()

    # ── Delete / default ────────────────────────────────────────────

    async def delete(self, *, tenant_id: int, id: str) -> None:
        """Soft-delete a backend that nothing depends on.

        Refused for an env-sourced row, the workspace default, a backend
        with bound knowledge bases or active resources, and a legacy alias
        (old file paths may still name it).
        """
        existing = await self._require_backend(tenant_id=tenant_id, id=id)
        if existing.source == STORAGE_BACKEND_SOURCE_ENV:
            raise ValidationError(
                code="storage_backend.env_read_only",
                message="environment storage backend is read-only",
            )
        default_id = await self._repo.get_default_backend_id(tenant_id)
        if default_id == id:
            raise ValidationError(
                code="storage_backend.is_default",
                message="default storage backend cannot be deleted",
            )
        kb_count = await self._repo.count_knowledge_base_references(tenant_id=tenant_id, id=id)
        if kb_count > 0:
            raise ValidationError(
                code="storage_backend.knowledge_bases_bound",
                message=f"storage backend still has {kb_count} knowledge base(s) bound to it",
            )
        resource_count = await self._repo.count_active_resource_references(
            tenant_id=tenant_id, id=id
        )
        if resource_count > 0:
            raise ValidationError(
                code="storage_backend.resources_active",
                message=f"storage backend still has {resource_count} active resource(s)",
            )
        if existing.legacy_alias:
            raise ValidationError(
                code="storage_backend.legacy_alias",
                message=(
                    "legacy storage backend cannot be deleted while old file paths may reference it"
                ),
            )
        await self._repo.soft_delete(tenant_id=tenant_id, id=id)

    async def set_default(self, *, tenant_id: int, id: str) -> None:
        """Make ``id`` the workspace default. Only an active row qualifies."""
        existing = await self._require_backend(tenant_id=tenant_id, id=id)
        if not existing.is_active:
            raise ValidationError(
                code="storage_backend.not_active",
                message="only an active storage backend can be the default",
            )
        await self._repo.set_default_backend_id(tenant_id=tenant_id, id=id)

    # ── Connectivity ────────────────────────────────────────────────

    async def test_config(
        self,
        *,
        tenant_id: int,
        name: str,
        provider: str,
        config: StorageBackendConfigInfo,
    ) -> StorageConnectivityResult:
        """Probe an unsaved configuration.

        A validation failure propagates (the request itself is wrong); a
        connectivity failure is returned as ``success=false`` with a
        sanitized message, matching Go's ``TestRaw`` which keeps the HTTP
        status at 200.
        """
        normalized_provider = provider.strip().lower()
        self._validate(
            tenant_id=tenant_id,
            name=name.strip(),
            provider=normalized_provider,
            config=config,
            status=STORAGE_BACKEND_STATUS_ACTIVE,
        )
        await self._validate_endpoint(provider=normalized_provider, config=config)
        return await self._probe_to_result(provider=normalized_provider, config=config)

    async def test_backend(self, *, tenant_id: int, id: str) -> StorageConnectivityResult:
        """Probe a saved backend using its stored credentials."""
        existing = await self._require_backend(tenant_id=tenant_id, id=id)
        await self._validate_endpoint(provider=existing.provider, config=existing.config)
        return await self._probe_to_result(provider=existing.provider, config=existing.config)

    # ── Resolution ──────────────────────────────────────────────────

    async def resolve_backend(
        self,
        *,
        tenant_id: int,
        backend_id: str = "",
        provider: str = "",
    ) -> StorageBackendInfo | None:
        """Select the instance a read/write should target.

        Precedence follows Go's ``ResolveBackend``: an explicit
        ``backend_id`` wins; a bare ``provider`` falls back to that
        provider's legacy alias; otherwise the workspace default is used.
        ``None`` means "no instance registered" — the caller then falls
        back to the workspace's legacy singleton config.
        """
        normalized_id = backend_id.strip()
        normalized_provider = provider.strip().lower()
        if not normalized_id and normalized_provider:
            alias = await self._repo.find_legacy_alias(
                tenant_id=tenant_id, provider=normalized_provider
            )
            if alias is not None:
                return StorageBackendInfo.map_from_db(alias)
        if not normalized_id:
            normalized_id = (await self._repo.get_default_backend_id(tenant_id) or "").strip()
        if not normalized_id:
            return None
        row = await self._repo.get_by_id(tenant_id=tenant_id, id=normalized_id)
        if row is None:
            raise NotFoundError(
                code="storage_backend.not_found",
                message="storage backend not found",
            )
        info = StorageBackendInfo.map_from_db(row)
        if not info.is_active:
            raise ValidationError(
                code="storage_backend.not_active",
                message="storage backend is not active",
            )
        return info

    # ── Internals ───────────────────────────────────────────────────

    async def _require_backend(self, *, tenant_id: int, id: str) -> StorageBackendInfo:
        row = await self._repo.get_by_id(tenant_id=tenant_id, id=id)
        if row is None:
            raise NotFoundError(
                code="storage_backend.not_found",
                message="storage backend not found",
            )
        return StorageBackendInfo.map_from_db(row)

    async def _reject_if_referenced(self, *, tenant_id: int, id: str, message: str) -> None:
        """Raise when the backend is the default or still referenced."""
        if await self._repo.get_default_backend_id(tenant_id) == id:
            raise ValidationError(code="storage_backend.in_use", message=message)
        if await self._repo.count_knowledge_base_references(tenant_id=tenant_id, id=id) > 0:
            raise ValidationError(code="storage_backend.in_use", message=message)
        if await self._repo.count_active_resource_references(tenant_id=tenant_id, id=id) > 0:
            raise ValidationError(code="storage_backend.in_use", message=message)

    def _validate(
        self,
        *,
        tenant_id: int,
        name: str,
        provider: str,
        config: StorageBackendConfigInfo,
        status: str,
    ) -> None:
        """Mirror ``StorageBackend.Validate`` field-for-field."""
        if tenant_id == 0:
            raise ValidationError(
                code="storage_backend.tenant_required",
                message="tenant_id is required",
            )
        if not name:
            raise ValidationError(
                code="storage_backend.name_required",
                message="name is required",
            )
        if not is_provider_allowed(provider):
            raise ValidationError(
                code="storage_backend.provider_not_allowed",
                message=f'storage provider "{provider}" is not allowed',
            )
        if provider not in allowed_providers():
            raise ValidationError(
                code="storage_backend.unsupported_provider",
                message=f"unsupported storage provider: {provider}",
            )
        if status not in _VALID_STATUSES:
            raise ValidationError(
                code="storage_backend.invalid_status",
                message="status must be active or disabled",
            )
        config.validate_for_provider(provider)

    async def _validate_endpoint(self, *, provider: str, config: StorageBackendConfigInfo) -> None:
        """SSRF-check the configured endpoint.

        Mirrors ``validateStorageBackendEndpoint``: ``local`` and a
        docker-mode ``minio`` have no row-supplied endpoint, ``cos`` is
        region-addressed, and a blank endpoint is left to the per-provider
        required-field validation. A scheme-less endpoint gains one from
        ``use_ssl`` before the check so ``host:9000`` is validated as a URL.
        """
        if provider == "local" or (provider == "minio" and config.mode == MINIO_MODE_DOCKER):
            return
        endpoint = config.endpoint.strip()
        if provider in _ENDPOINT_EXEMPT_PROVIDERS or not endpoint:
            return
        if "://" not in endpoint:
            scheme = "http" if provider == "minio" and not config.use_ssl else "https"
            endpoint = f"{scheme}://{endpoint}"
        try:
            await validate_ssrf_safe_url(endpoint)
        except ApplicationError as exc:
            raise ValidationError(
                code="storage_backend.endpoint_ssrf_blocked",
                message="storage endpoint failed SSRF validation",
                details={"reason": exc.message},
            ) from exc

    async def _probe(self, *, provider: str, config: StorageBackendConfigInfo) -> None:
        """Run the provider probe, wrapping a failure as a bad request.

        Go returns ``NewBadRequestError("storage connection test failed")``
        with the sanitized cause in the details, so a create/update refusal
        is a 4xx and never leaks a hostname or port.
        """
        try:
            await self._adapter_for(provider=provider, config=config).check_connectivity()
        except ApplicationError as exc:
            raise ValidationError(
                code="storage_backend.connection_test_failed",
                message="storage connection test failed",
                details={"reason": exc.message},
            ) from exc

    async def _probe_to_result(
        self, *, provider: str, config: StorageBackendConfigInfo
    ) -> StorageConnectivityResult:
        """Run the probe, returning the failure as data rather than raising."""
        try:
            await self._adapter_for(provider=provider, config=config).check_connectivity()
        except ApplicationError as exc:
            return StorageConnectivityResult(success=False, error=exc.message)
        return StorageConnectivityResult(success=True)

    def _adapter_for(self, *, provider: str, config: StorageBackendConfigInfo) -> StorageAdapter:
        """Build the adapter for ``provider`` from the normalized config."""
        if provider == "local":
            return _LocalProbe(config.path_prefix)
        if provider == "minio":
            return MinioStorageAdapter(
                endpoint=config.endpoint,
                access_key_id=config.access_key_id,
                secret_access_key=config.secret_access_key,
                bucket_name=config.bucket_name,
                use_ssl=config.use_ssl,
                region=config.region,
                mode=config.mode or "remote",
            )
        if provider == "cos":
            return CosStorageAdapter(
                region=config.region,
                access_key_id=config.access_key_id,
                secret_access_key=config.secret_access_key,
                bucket_name=config.bucket_name,
                app_id=config.app_id,
            )
        if provider == "obs":
            return ObsStorageAdapter(
                endpoint=config.endpoint,
                region=config.region,
                access_key_id=config.access_key_id,
                secret_access_key=config.secret_access_key,
                bucket_name=config.bucket_name,
                use_ssl=config.use_ssl,
            )
        label = _S3_COMPATIBLE_LABELS.get(provider)
        if label is None:
            raise ValidationError(
                code="storage_backend.unsupported_provider",
                message=f"unsupported storage provider: {provider}",
            )
        return S3StorageAdapter(
            endpoint=config.endpoint,
            region=config.region,
            access_key_id=config.access_key_id,
            secret_access_key=config.secret_access_key,
            bucket_name=config.bucket_name,
            use_ssl=config.use_ssl,
            force_path_style=config.force_path_style,
            provider_label=label,
        )


class _LocalProbe:
    """Local adapter that creates its directory before probing.

    Go's ``Test`` does the ``MkdirAll`` inline before dispatching to the
    local file service; keeping it in one object means the service's
    dispatch table stays a pure adapter lookup.
    """

    def __init__(self, path_prefix: str) -> None:
        self._adapter = LocalStorageAdapter(path_prefix=path_prefix)

    async def check_connectivity(self) -> None:
        """Create the directory tree, then verify it is usable."""
        await self._adapter.ensure_directory()
        await self._adapter.check_connectivity()


__all__ = ["StorageBackendService"]
