"""Unit tests for ``StorageBackendService`` and its config value type.

The connectivity probe is patched out in every test that is not about
probing: the service dials a real endpoint otherwise, and the domain rules
under test (validation, immutability, delete guards, resolution
precedence) are independent of whether a bucket answers.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest

from src.common.exception import (
    ConflictError,
    NotFoundError,
    StorageBackendError,
    ValidationError,
)
from src.core.infra.storage_backends.service.storage_backend_service import (
    StorageBackendService,
)
from src.core.infra.storage_backends.types import (
    REDACTED_SECRET_PLACEHOLDER,
    STORAGE_ALLOW_LIST_ENV,
    SUPPORTED_PROVIDERS,
    StorageBackendConfigInfo,
    StorageBackendInfo,
    allowed_providers,
    is_provider_allowed,
)
from src.db.models.storage_backend import (
    STORAGE_BACKEND_SOURCE_ENV,
    STORAGE_BACKEND_SOURCE_USER,
    STORAGE_BACKEND_STATUS_ACTIVE,
    STORAGE_BACKEND_STATUS_DISABLED,
    StorageBackend,
)
from tests.fakes.storage_backends import FakeStorageBackendRepository

_TENANT_ID = 7
_OTHER_TENANT_ID = 8
_NOW = datetime(2026, 1, 1, tzinfo=UTC)

# A minio config that passes per-provider validation.
_MINIO_CONFIG = StorageBackendConfigInfo(
    mode="remote",
    endpoint="storage.example.com:9000",
    access_key_id="AKIA_EXAMPLE",
    secret_access_key="secret-example",
    bucket_name="documents",
)


@pytest.fixture
def repo() -> FakeStorageBackendRepository:
    return FakeStorageBackendRepository()


@pytest.fixture
def service(repo: FakeStorageBackendRepository) -> StorageBackendService:
    return StorageBackendService(backend_repo=repo)  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def _stub_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralise outbound I/O: the adapter probe and the SSRF check.

    ``_adapter_for`` is replaced rather than ``_probe`` so both probe
    call-paths (the raising one used by create/update and the
    result-returning one used by the test endpoints) go through the stub.
    """

    async def _ok(**_kwargs: object) -> None:
        return None

    monkeypatch.setattr(
        StorageBackendService,
        "_adapter_for",
        lambda _self, **_kwargs: _PassingAdapter(),
    )
    monkeypatch.setattr(
        StorageBackendService,
        "_validate_endpoint",
        lambda _self, **_kwargs: _ok(),
    )


@pytest.fixture
def clean_allow_list() -> Iterator[None]:
    """Ensure ``STORAGE_ALLOW_LIST`` does not leak between tests."""
    previous = os.environ.get(STORAGE_ALLOW_LIST_ENV)
    os.environ.pop(STORAGE_ALLOW_LIST_ENV, None)
    yield
    if previous is None:
        os.environ.pop(STORAGE_ALLOW_LIST_ENV, None)
    else:
        os.environ[STORAGE_ALLOW_LIST_ENV] = previous


def _seed(
    repo: FakeStorageBackendRepository,
    *,
    id: str = "backend-1",
    tenant_id: int = _TENANT_ID,
    name: str = "Primary MinIO",
    provider: str = "minio",
    config: StorageBackendConfigInfo | None = None,
    source: str = STORAGE_BACKEND_SOURCE_USER,
    status: str = STORAGE_BACKEND_STATUS_ACTIVE,
    legacy_alias: bool = False,
) -> StorageBackend:
    """Insert a row directly, bypassing service-side validation."""
    row = StorageBackend(
        id=id,
        tenant_id=tenant_id,
        name=name,
        provider=provider,
        config=(config or _MINIO_CONFIG).to_json(),
        source=source,
        status=status,
        legacy_alias=legacy_alias,
        created_at=_NOW,
        updated_at=_NOW,
    )
    repo.rows[id] = row
    return row


# ── Provider allow-list ─────────────────────────────────────────────


def test_allowed_providers_defaults_to_every_supported_provider(
    clean_allow_list: None,
) -> None:
    assert allowed_providers() == SUPPORTED_PROVIDERS


def test_allowed_providers_respects_env_and_keeps_canonical_order(
    clean_allow_list: None,
) -> None:
    os.environ[STORAGE_ALLOW_LIST_ENV] = "obs,minio"

    assert allowed_providers() == ("minio", "obs")


def test_allowed_providers_accepts_alternative_separators(
    clean_allow_list: None,
) -> None:
    os.environ[STORAGE_ALLOW_LIST_ENV] = "minio; s3 |cos"

    assert allowed_providers() == ("minio", "cos", "s3")


def test_empty_provider_is_treated_as_allowed(clean_allow_list: None) -> None:
    os.environ[STORAGE_ALLOW_LIST_ENV] = "minio"

    assert is_provider_allowed("") is True
    assert is_provider_allowed("obs") is False


def test_list_provider_types_returns_allowed_names(
    service: StorageBackendService,
    clean_allow_list: None,
) -> None:
    os.environ[STORAGE_ALLOW_LIST_ENV] = "local,minio"

    assert service.list_provider_types() == ["local", "minio"]


# ── Config value type ───────────────────────────────────────────────


def test_mask_sensitive_fields_replaces_only_populated_secrets() -> None:
    masked = _MINIO_CONFIG.mask_sensitive_fields()

    assert masked.access_key_id == REDACTED_SECRET_PLACEHOLDER
    assert masked.secret_access_key == REDACTED_SECRET_PLACEHOLDER
    assert masked.bucket_name == "documents"


def test_mask_sensitive_fields_leaves_blank_secrets_blank() -> None:
    masked = StorageBackendConfigInfo(bucket_name="b").mask_sensitive_fields()

    assert masked.access_key_id == ""
    assert masked.secret_access_key == ""


def test_merge_secrets_preserves_stored_value_when_redacted() -> None:
    incoming = _MINIO_CONFIG.model_copy(
        update={
            "access_key_id": REDACTED_SECRET_PLACEHOLDER,
            "secret_access_key": "",
        }
    )

    merged = incoming.merge_secrets(_MINIO_CONFIG)

    assert merged.access_key_id == "AKIA_EXAMPLE"
    assert merged.secret_access_key == "secret-example"


def test_merge_secrets_accepts_a_real_rotation() -> None:
    incoming = _MINIO_CONFIG.model_copy(update={"secret_access_key": "rotated"})

    merged = incoming.merge_secrets(_MINIO_CONFIG)

    assert merged.secret_access_key == "rotated"


def test_location_key_ignores_credentials() -> None:
    rotated = _MINIO_CONFIG.model_copy(update={"secret_access_key": "rotated"})

    assert rotated.location_key("minio") == _MINIO_CONFIG.location_key("minio")


def test_location_key_changes_with_bucket() -> None:
    moved = _MINIO_CONFIG.model_copy(update={"bucket_name": "elsewhere"})

    assert moved.location_key("minio") != _MINIO_CONFIG.location_key("minio")


def test_location_key_defaults_minio_mode_to_remote() -> None:
    blank_mode = _MINIO_CONFIG.model_copy(update={"mode": ""})

    assert blank_mode.location_key("minio") == _MINIO_CONFIG.location_key("minio")


def test_from_json_tolerates_none_and_nulls() -> None:
    assert StorageBackendConfigInfo.from_json(None) == StorageBackendConfigInfo()
    assert StorageBackendConfigInfo.from_json({"endpoint": None}).endpoint == ""


def test_from_json_parses_raw_json_text() -> None:
    parsed = StorageBackendConfigInfo.from_json('{"bucket_name": "b"}')

    assert parsed.bucket_name == "b"


@pytest.mark.parametrize("prefix", ["/absolute", "..", "../escape", "nested/../.."])
def test_validate_rejects_absolute_or_traversing_path_prefix(prefix: str) -> None:
    config = StorageBackendConfigInfo(path_prefix=prefix)

    with pytest.raises(ValidationError) as exc:
        config.validate_for_provider("local")

    assert exc.value.code == "storage_backend.invalid_path_prefix"


def test_validate_local_requires_nothing_else() -> None:
    StorageBackendConfigInfo(path_prefix="tenant/7").validate_for_provider("local")


def test_validate_minio_docker_mode_only_requires_bucket() -> None:
    StorageBackendConfigInfo(mode="docker", bucket_name="b").validate_for_provider("minio")


def test_validate_minio_remote_mode_requires_endpoint_and_credentials() -> None:
    with pytest.raises(ValidationError) as exc:
        StorageBackendConfigInfo(mode="remote", bucket_name="b").validate_for_provider("minio")

    assert exc.value.code == "storage_backend.missing_config_field"


def test_validate_cos_requires_region_not_endpoint() -> None:
    StorageBackendConfigInfo(
        region="ap-guangzhou",
        access_key_id="id",
        secret_access_key="key",
        bucket_name="b",
    ).validate_for_provider("cos")


def test_validate_s3_requires_endpoint_and_region() -> None:
    with pytest.raises(ValidationError) as exc:
        StorageBackendConfigInfo(
            access_key_id="id", secret_access_key="key", bucket_name="b"
        ).validate_for_provider("s3")

    assert exc.value.code == "storage_backend.missing_config_field"


# ── create ──────────────────────────────────────────────────────────


async def test_create_persists_a_user_sourced_active_backend(
    service: StorageBackendService,
    repo: FakeStorageBackendRepository,
) -> None:
    info = await service.create(
        tenant_id=_TENANT_ID,
        name="  Primary MinIO  ",
        provider="MinIO",
        config=_MINIO_CONFIG,
    )

    assert info.name == "Primary MinIO"
    assert info.provider == "minio"
    assert info.source == STORAGE_BACKEND_SOURCE_USER
    assert info.status == STORAGE_BACKEND_STATUS_ACTIVE
    assert info.legacy_alias is False
    assert repo.rows[info.id].tenant_id == _TENANT_ID


async def test_create_rejects_a_duplicate_name(
    service: StorageBackendService,
    repo: FakeStorageBackendRepository,
) -> None:
    _seed(repo, name="Primary MinIO")

    with pytest.raises(ConflictError) as exc:
        await service.create(
            tenant_id=_TENANT_ID,
            name="Primary MinIO",
            provider="minio",
            config=_MINIO_CONFIG,
        )

    assert exc.value.code == "storage_backend.duplicate_name"


async def test_create_rejects_a_blank_name(service: StorageBackendService) -> None:
    with pytest.raises(ValidationError) as exc:
        await service.create(
            tenant_id=_TENANT_ID, name="   ", provider="minio", config=_MINIO_CONFIG
        )

    assert exc.value.code == "storage_backend.name_required"


async def test_create_rejects_a_zero_tenant(service: StorageBackendService) -> None:
    with pytest.raises(ValidationError) as exc:
        await service.create(tenant_id=0, name="Primary", provider="minio", config=_MINIO_CONFIG)

    assert exc.value.code == "storage_backend.tenant_required"


async def test_create_rejects_an_unsupported_provider(
    service: StorageBackendService,
) -> None:
    with pytest.raises(ValidationError) as exc:
        await service.create(
            tenant_id=_TENANT_ID,
            name="Nope",
            provider="dropbox",
            config=_MINIO_CONFIG,
        )

    assert exc.value.code == "storage_backend.provider_not_allowed"


async def test_create_rejects_an_invalid_status(service: StorageBackendService) -> None:
    with pytest.raises(ValidationError) as exc:
        await service.create(
            tenant_id=_TENANT_ID,
            name="Primary",
            provider="minio",
            config=_MINIO_CONFIG,
            status="paused",
        )

    assert exc.value.code == "storage_backend.invalid_status"


async def test_create_refuses_when_the_probe_fails(
    service: StorageBackendService,
    repo: FakeStorageBackendRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        StorageBackendService,
        "_adapter_for",
        lambda _self, **_kw: _FailingAdapter(),
    )

    with pytest.raises(ValidationError) as exc:
        await service.create(
            tenant_id=_TENANT_ID,
            name="Primary",
            provider="minio",
            config=_MINIO_CONFIG,
        )

    assert exc.value.code == "storage_backend.connection_test_failed"
    # Nothing was persisted — the probe runs before the insert.
    assert repo.rows == {}


# ── get / list ──────────────────────────────────────────────────────


async def test_get_masks_credentials(
    service: StorageBackendService,
    repo: FakeStorageBackendRepository,
) -> None:
    _seed(repo)

    info = await service.get_backend(tenant_id=_TENANT_ID, id="backend-1")

    assert info.config.access_key_id == REDACTED_SECRET_PLACEHOLDER
    assert info.config.secret_access_key == REDACTED_SECRET_PLACEHOLDER


async def test_get_does_not_cross_tenants(
    service: StorageBackendService,
    repo: FakeStorageBackendRepository,
) -> None:
    _seed(repo, tenant_id=_OTHER_TENANT_ID)

    with pytest.raises(NotFoundError) as exc:
        await service.get_backend(tenant_id=_TENANT_ID, id="backend-1")

    assert exc.value.code == "storage_backend.not_found"


async def test_list_returns_masked_rows_and_the_default_id(
    service: StorageBackendService,
    repo: FakeStorageBackendRepository,
) -> None:
    _seed(repo, id="backend-1", name="A")
    _seed(repo, id="backend-2", name="B")
    repo.default_backend_id[_TENANT_ID] = "backend-2"

    result = await service.list_backends(_TENANT_ID)

    assert [b.id for b in result.backends] == ["backend-1", "backend-2"]
    assert result.default_storage_backend_id == "backend-2"
    assert all(b.config.secret_access_key == REDACTED_SECRET_PLACEHOLDER for b in result.backends)


async def test_list_excludes_other_tenants(
    service: StorageBackendService,
    repo: FakeStorageBackendRepository,
) -> None:
    _seed(repo, id="mine")
    _seed(repo, id="theirs", tenant_id=_OTHER_TENANT_ID)

    result = await service.list_backends(_TENANT_ID)

    assert [b.id for b in result.backends] == ["mine"]


# ── update ──────────────────────────────────────────────────────────


async def test_update_renames_and_keeps_stored_secrets(
    service: StorageBackendService,
    repo: FakeStorageBackendRepository,
) -> None:
    _seed(repo)
    redacted = _MINIO_CONFIG.mask_sensitive_fields()

    info = await service.update(
        tenant_id=_TENANT_ID, id="backend-1", name="Renamed", config=redacted
    )

    assert info.name == "Renamed"
    assert info.config.access_key_id == "AKIA_EXAMPLE"
    assert info.config.secret_access_key == "secret-example"


async def test_update_accepts_a_credential_rotation(
    service: StorageBackendService,
    repo: FakeStorageBackendRepository,
) -> None:
    _seed(repo)
    rotated = _MINIO_CONFIG.model_copy(update={"secret_access_key": "rotated"})

    info = await service.update(tenant_id=_TENANT_ID, id="backend-1", config=rotated)

    assert info.config.secret_access_key == "rotated"


async def test_update_rejects_a_location_change(
    service: StorageBackendService,
    repo: FakeStorageBackendRepository,
) -> None:
    _seed(repo)
    moved = _MINIO_CONFIG.model_copy(update={"bucket_name": "elsewhere"})

    with pytest.raises(ValidationError) as exc:
        await service.update(tenant_id=_TENANT_ID, id="backend-1", config=moved)

    assert exc.value.code == "storage_backend.immutable_location"


async def test_update_rejects_an_env_sourced_backend(
    service: StorageBackendService,
    repo: FakeStorageBackendRepository,
) -> None:
    _seed(repo, source=STORAGE_BACKEND_SOURCE_ENV)

    with pytest.raises(ValidationError) as exc:
        await service.update(tenant_id=_TENANT_ID, id="backend-1", name="Renamed")

    assert exc.value.code == "storage_backend.env_read_only"


async def test_update_can_disable_an_unreferenced_backend(
    service: StorageBackendService,
    repo: FakeStorageBackendRepository,
) -> None:
    _seed(repo)

    info = await service.update(
        tenant_id=_TENANT_ID, id="backend-1", status=STORAGE_BACKEND_STATUS_DISABLED
    )

    assert info.status == STORAGE_BACKEND_STATUS_DISABLED


async def test_update_cannot_disable_the_workspace_default(
    service: StorageBackendService,
    repo: FakeStorageBackendRepository,
) -> None:
    _seed(repo)
    repo.default_backend_id[_TENANT_ID] = "backend-1"

    with pytest.raises(ValidationError) as exc:
        await service.update(
            tenant_id=_TENANT_ID, id="backend-1", status=STORAGE_BACKEND_STATUS_DISABLED
        )

    assert exc.value.code == "storage_backend.in_use"


async def test_update_cannot_disable_a_backend_with_bound_knowledge_bases(
    service: StorageBackendService,
    repo: FakeStorageBackendRepository,
) -> None:
    _seed(repo)
    repo.knowledge_base_references = 2

    with pytest.raises(ValidationError) as exc:
        await service.update(
            tenant_id=_TENANT_ID, id="backend-1", status=STORAGE_BACKEND_STATUS_DISABLED
        )

    assert exc.value.code == "storage_backend.in_use"


async def test_update_of_an_already_disabled_backend_skips_the_guard(
    service: StorageBackendService,
    repo: FakeStorageBackendRepository,
) -> None:
    _seed(repo, status=STORAGE_BACKEND_STATUS_DISABLED)
    repo.knowledge_base_references = 5

    info = await service.update(
        tenant_id=_TENANT_ID, id="backend-1", status=STORAGE_BACKEND_STATUS_DISABLED
    )

    assert info.status == STORAGE_BACKEND_STATUS_DISABLED


async def test_update_of_a_missing_backend_raises_not_found(
    service: StorageBackendService,
) -> None:
    with pytest.raises(NotFoundError) as exc:
        await service.update(tenant_id=_TENANT_ID, id="ghost", name="X")

    assert exc.value.code == "storage_backend.not_found"


# ── delete ──────────────────────────────────────────────────────────


async def test_delete_soft_deletes_an_unreferenced_backend(
    service: StorageBackendService,
    repo: FakeStorageBackendRepository,
) -> None:
    _seed(repo)

    await service.delete(tenant_id=_TENANT_ID, id="backend-1")

    assert repo.rows["backend-1"].deleted_at is not None
    assert await repo.get_by_id(tenant_id=_TENANT_ID, id="backend-1") is None


async def test_delete_refuses_the_workspace_default(
    service: StorageBackendService,
    repo: FakeStorageBackendRepository,
) -> None:
    _seed(repo)
    repo.default_backend_id[_TENANT_ID] = "backend-1"

    with pytest.raises(ValidationError) as exc:
        await service.delete(tenant_id=_TENANT_ID, id="backend-1")

    assert exc.value.code == "storage_backend.is_default"


async def test_delete_refuses_a_backend_with_bound_knowledge_bases(
    service: StorageBackendService,
    repo: FakeStorageBackendRepository,
) -> None:
    _seed(repo)
    repo.knowledge_base_references = 3

    with pytest.raises(ValidationError) as exc:
        await service.delete(tenant_id=_TENANT_ID, id="backend-1")

    assert exc.value.code == "storage_backend.knowledge_bases_bound"
    assert "3 knowledge base(s)" in exc.value.message


async def test_delete_refuses_a_backend_with_active_resources(
    service: StorageBackendService,
    repo: FakeStorageBackendRepository,
) -> None:
    _seed(repo)
    repo.active_resource_references = 4

    with pytest.raises(ValidationError) as exc:
        await service.delete(tenant_id=_TENANT_ID, id="backend-1")

    assert exc.value.code == "storage_backend.resources_active"


async def test_delete_refuses_an_env_sourced_backend(
    service: StorageBackendService,
    repo: FakeStorageBackendRepository,
) -> None:
    _seed(repo, source=STORAGE_BACKEND_SOURCE_ENV)

    with pytest.raises(ValidationError) as exc:
        await service.delete(tenant_id=_TENANT_ID, id="backend-1")

    assert exc.value.code == "storage_backend.env_read_only"


async def test_delete_refuses_a_legacy_alias(
    service: StorageBackendService,
    repo: FakeStorageBackendRepository,
) -> None:
    _seed(repo, legacy_alias=True)

    with pytest.raises(ValidationError) as exc:
        await service.delete(tenant_id=_TENANT_ID, id="backend-1")

    assert exc.value.code == "storage_backend.legacy_alias"


# ── set_default ─────────────────────────────────────────────────────


async def test_set_default_points_the_workspace_at_an_active_backend(
    service: StorageBackendService,
    repo: FakeStorageBackendRepository,
) -> None:
    _seed(repo)

    await service.set_default(tenant_id=_TENANT_ID, id="backend-1")

    assert repo.default_backend_id[_TENANT_ID] == "backend-1"


async def test_set_default_refuses_a_disabled_backend(
    service: StorageBackendService,
    repo: FakeStorageBackendRepository,
) -> None:
    _seed(repo, status=STORAGE_BACKEND_STATUS_DISABLED)

    with pytest.raises(ValidationError) as exc:
        await service.set_default(tenant_id=_TENANT_ID, id="backend-1")

    assert exc.value.code == "storage_backend.not_active"


async def test_set_default_of_a_missing_backend_raises_not_found(
    service: StorageBackendService,
) -> None:
    with pytest.raises(NotFoundError):
        await service.set_default(tenant_id=_TENANT_ID, id="ghost")


# ── connectivity probes ─────────────────────────────────────────────


async def test_test_config_returns_success_when_the_probe_passes(
    service: StorageBackendService,
) -> None:
    result = await service.test_config(
        tenant_id=_TENANT_ID,
        name="Primary",
        provider="minio",
        config=_MINIO_CONFIG,
    )

    assert result.success is True
    assert result.error is None


async def test_test_config_returns_the_failure_as_data(
    service: StorageBackendService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        StorageBackendService,
        "_adapter_for",
        lambda _self, **_kw: _FailingAdapter(),
    )

    result = await service.test_config(
        tenant_id=_TENANT_ID,
        name="Primary",
        provider="minio",
        config=_MINIO_CONFIG,
    )

    assert result.success is False
    assert result.error == "bucket unavailable"


async def test_test_config_still_rejects_an_invalid_request(
    service: StorageBackendService,
) -> None:
    with pytest.raises(ValidationError) as exc:
        await service.test_config(
            tenant_id=_TENANT_ID, name="", provider="minio", config=_MINIO_CONFIG
        )

    assert exc.value.code == "storage_backend.name_required"


async def test_test_backend_probes_a_saved_row(
    service: StorageBackendService,
    repo: FakeStorageBackendRepository,
) -> None:
    _seed(repo)

    result = await service.test_backend(tenant_id=_TENANT_ID, id="backend-1")

    assert result.success is True


async def test_test_backend_of_a_missing_row_raises_not_found(
    service: StorageBackendService,
) -> None:
    with pytest.raises(NotFoundError):
        await service.test_backend(tenant_id=_TENANT_ID, id="ghost")


# ── resolve_backend ─────────────────────────────────────────────────


async def test_resolve_prefers_an_explicit_backend_id(
    service: StorageBackendService,
    repo: FakeStorageBackendRepository,
) -> None:
    _seed(repo, id="explicit")
    _seed(repo, id="alias", name="Alias", legacy_alias=True)
    repo.default_backend_id[_TENANT_ID] = "alias"

    resolved = await service.resolve_backend(
        tenant_id=_TENANT_ID, backend_id="explicit", provider="minio"
    )

    assert resolved is not None
    assert resolved.id == "explicit"


async def test_resolve_falls_back_to_the_legacy_alias_for_a_bare_provider(
    service: StorageBackendService,
    repo: FakeStorageBackendRepository,
) -> None:
    _seed(repo, id="alias", name="Alias", legacy_alias=True)
    repo.default_backend_id[_TENANT_ID] = "other"

    resolved = await service.resolve_backend(tenant_id=_TENANT_ID, provider="minio")

    assert resolved is not None
    assert resolved.id == "alias"


async def test_resolve_uses_the_workspace_default_when_nothing_is_specified(
    service: StorageBackendService,
    repo: FakeStorageBackendRepository,
) -> None:
    _seed(repo, id="default-backend")
    repo.default_backend_id[_TENANT_ID] = "default-backend"

    resolved = await service.resolve_backend(tenant_id=_TENANT_ID)

    assert resolved is not None
    assert resolved.id == "default-backend"


async def test_resolve_returns_none_when_no_instance_is_registered(
    service: StorageBackendService,
) -> None:
    assert await service.resolve_backend(tenant_id=_TENANT_ID) is None


async def test_resolve_raises_when_the_named_backend_is_absent(
    service: StorageBackendService,
) -> None:
    with pytest.raises(NotFoundError) as exc:
        await service.resolve_backend(tenant_id=_TENANT_ID, backend_id="ghost")

    assert exc.value.code == "storage_backend.not_found"


async def test_resolve_raises_when_the_resolved_backend_is_disabled(
    service: StorageBackendService,
    repo: FakeStorageBackendRepository,
) -> None:
    _seed(repo, status=STORAGE_BACKEND_STATUS_DISABLED)

    with pytest.raises(ValidationError) as exc:
        await service.resolve_backend(tenant_id=_TENANT_ID, backend_id="backend-1")

    assert exc.value.code == "storage_backend.not_active"


# ── DTO projection ──────────────────────────────────────────────────


def test_map_from_db_drops_the_soft_delete_marker_and_hydrates_config() -> None:
    row = StorageBackend(
        id="backend-1",
        tenant_id=_TENANT_ID,
        name="Primary",
        provider="minio",
        config=_MINIO_CONFIG.to_json(),
        source=STORAGE_BACKEND_SOURCE_USER,
        status=STORAGE_BACKEND_STATUS_ACTIVE,
        created_at=_NOW,
        updated_at=_NOW,
        deleted_at=_NOW,
    )

    info = StorageBackendInfo.map_from_db(row)

    assert "deleted_at" not in info.model_dump()
    assert info.config.bucket_name == "documents"
    assert info.is_active is True


class _PassingAdapter:
    """Adapter double whose probe always succeeds."""

    async def check_connectivity(self) -> None:
        return None


class _FailingAdapter:
    """Adapter double whose probe always reports a sanitized failure."""

    async def check_connectivity(self) -> None:
        raise StorageBackendError(
            code="storage_backend.bucket_unavailable", message="bucket unavailable"
        )


__all__: list[str] = []
