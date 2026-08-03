"""Unit tests for `TenantAPIKeyService` and its token helpers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.common.exception import NotFoundError, ValidationError
from src.core.tenants.api_key_service import (
    SCOPE_PLATFORM,
    SCOPE_TENANT,
    TenantAPIKeyService,
    generate_api_key_token,
    hash_api_key_token,
    normalize_capabilities,
    normalize_knowledge_base_ids,
    normalize_scope_type,
)
from src.db.models.tenants.tenant_api_keys import TenantAPIKey
from tests.fakes.tenant_api_keys import FakeTenantAPIKeyRepository

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_TENANT_ID = 7


@pytest.fixture
def repo() -> FakeTenantAPIKeyRepository:
    return FakeTenantAPIKeyRepository()


@pytest.fixture
def service(repo: FakeTenantAPIKeyRepository) -> TenantAPIKeyService:
    return TenantAPIKeyService(api_keys_repo=repo)  # type: ignore[arg-type]


# ── token helpers ───────────────────────────────────────────────────


def test_generate_token_is_prefixed_and_unpadded() -> None:
    token = generate_api_key_token()

    assert token.startswith("sk-")
    assert "=" not in token


def test_generate_token_is_unique_per_call() -> None:
    assert generate_api_key_token() != generate_api_key_token()


def test_hash_is_stable_sha256_hex() -> None:
    digest = hash_api_key_token("sk-example")

    assert digest == hash_api_key_token("sk-example")
    assert len(digest) == 64


def test_normalize_scope_type_defaults_to_tenant() -> None:
    assert normalize_scope_type(None) == SCOPE_TENANT
    assert normalize_scope_type("  nonsense ") == SCOPE_TENANT
    assert normalize_scope_type(" Platform ") == SCOPE_PLATFORM


def test_normalize_capabilities_drops_unknown_and_duplicates() -> None:
    assert normalize_capabilities(["Chat", "chat", "nope", " ingest "]) == ["chat", "ingest"]


def test_normalize_knowledge_base_ids_keeps_case_and_order() -> None:
    assert normalize_knowledge_base_ids([" KB-1 ", "kb-1", "KB-1", ""]) == ["KB-1", "kb-1"]


# ── create_api_key ──────────────────────────────────────────────────


async def test_create_api_key_returns_token_once_and_stores_only_its_hash(
    service: TenantAPIKeyService,
    repo: FakeTenantAPIKeyRepository,
) -> None:
    result = await service.create_api_key(name="ci", tenant_id=_TENANT_ID)

    stored = repo.rows[result.key.id]
    assert result.token.startswith("sk-")
    assert stored.key_hash == hash_api_key_token(result.token)
    assert set(result.key.model_dump()).isdisjoint({"key_hash", "api_key"})


async def test_create_api_key_trims_name(service: TenantAPIKeyService) -> None:
    result = await service.create_api_key(name="  ci  ", tenant_id=_TENANT_ID)

    assert result.key.name == "ci"


async def test_create_api_key_rejects_blank_name(service: TenantAPIKeyService) -> None:
    with pytest.raises(ValidationError) as excinfo:
        await service.create_api_key(name="   ", tenant_id=_TENANT_ID)

    assert excinfo.value.code == "tenant_api_key.name_required"


async def test_create_tenant_key_requires_a_tenant_id(service: TenantAPIKeyService) -> None:
    with pytest.raises(ValidationError) as excinfo:
        await service.create_api_key(name="ci")

    assert excinfo.value.code == "tenant_api_key.tenant_required"


async def test_create_platform_key_rejects_full_access(service: TenantAPIKeyService) -> None:
    with pytest.raises(ValidationError) as excinfo:
        await service.create_api_key(
            name="ci",
            scope_type=SCOPE_PLATFORM,
            full_access=True,
            capabilities=["system_tenants_read"],
        )

    assert excinfo.value.code == "tenant_api_key.platform_full_access"


async def test_create_platform_key_requires_capabilities(service: TenantAPIKeyService) -> None:
    with pytest.raises(ValidationError) as excinfo:
        await service.create_api_key(name="ci", scope_type=SCOPE_PLATFORM)

    assert excinfo.value.code == "tenant_api_key.capabilities_required"


async def test_create_platform_key_has_no_tenant(service: TenantAPIKeyService) -> None:
    result = await service.create_api_key(
        name="ci",
        tenant_id=_TENANT_ID,
        scope_type=SCOPE_PLATFORM,
        capabilities=["system_tenants_read"],
    )

    assert result.key.tenant_id is None
    assert result.key.scope_type == SCOPE_PLATFORM


async def test_create_full_access_key_clears_narrowing_lists(
    service: TenantAPIKeyService,
) -> None:
    result = await service.create_api_key(
        name="ci",
        tenant_id=_TENANT_ID,
        full_access=True,
        knowledge_base_ids=["kb-1"],
        capabilities=["chat"],
    )

    assert result.key.knowledge_base_ids == []
    assert result.key.capabilities == []


async def test_create_scoped_key_normalizes_its_lists(service: TenantAPIKeyService) -> None:
    result = await service.create_api_key(
        name="ci",
        tenant_id=_TENANT_ID,
        knowledge_base_ids=["kb-1", "kb-1", " "],
        capabilities=["chat", "bogus", "CHAT"],
    )

    assert result.key.knowledge_base_ids == ["kb-1"]
    assert result.key.capabilities == ["chat"]


async def test_create_api_key_normalizes_expiry_to_utc(service: TenantAPIKeyService) -> None:
    expires = datetime(2026, 6, 1, 12, tzinfo=UTC).astimezone()

    result = await service.create_api_key(
        name="ci",
        tenant_id=_TENANT_ID,
        expires_at=expires,
    )

    assert result.key.expires_at is not None
    assert result.key.expires_at.utcoffset() == timedelta(0)


# ── authenticate ────────────────────────────────────────────────────


async def test_authenticate_resolves_a_live_key(
    service: TenantAPIKeyService,
) -> None:
    created = await service.create_api_key(name="ci", tenant_id=_TENANT_ID)

    found = await service.authenticate(created.token)

    assert found.id == created.key.id
    assert found.tenant_id == _TENANT_ID


async def test_authenticate_trims_the_token(service: TenantAPIKeyService) -> None:
    created = await service.create_api_key(name="ci", tenant_id=_TENANT_ID)

    found = await service.authenticate(f"  {created.token}  ")

    assert found.id == created.key.id


@pytest.mark.parametrize("token", ["", "   "])
async def test_authenticate_rejects_empty_token(
    service: TenantAPIKeyService,
    token: str,
) -> None:
    with pytest.raises(NotFoundError) as excinfo:
        await service.authenticate(token)

    assert excinfo.value.code == "tenant_api_key.not_found"


async def test_authenticate_rejects_unknown_token(service: TenantAPIKeyService) -> None:
    with pytest.raises(NotFoundError):
        await service.authenticate("sk-never-issued")


async def test_authenticate_rejects_revoked_key(service: TenantAPIKeyService) -> None:
    created = await service.create_api_key(name="ci", tenant_id=_TENANT_ID)
    await service.revoke_api_key(created.key.id, tenant_id=_TENANT_ID)

    with pytest.raises(NotFoundError):
        await service.authenticate(created.token)


async def test_authenticate_rejects_expired_key(
    service: TenantAPIKeyService,
) -> None:
    created = await service.create_api_key(
        name="ci",
        tenant_id=_TENANT_ID,
        expires_at=datetime.now(UTC) - timedelta(seconds=1),
    )

    with pytest.raises(NotFoundError):
        await service.authenticate(created.token)


async def test_authenticate_stamps_last_used_on_first_use(
    service: TenantAPIKeyService,
    repo: FakeTenantAPIKeyRepository,
) -> None:
    created = await service.create_api_key(name="ci", tenant_id=_TENANT_ID)

    await service.authenticate(created.token)

    assert repo.rows[created.key.id].last_used_at is not None


async def test_authenticate_throttles_repeated_last_used_writes(
    service: TenantAPIKeyService,
    repo: FakeTenantAPIKeyRepository,
) -> None:
    created = await service.create_api_key(name="ci", tenant_id=_TENANT_ID)
    await service.authenticate(created.token)
    first_touch = repo.rows[created.key.id].last_used_at

    await service.authenticate(created.token)

    assert repo.rows[created.key.id].last_used_at == first_touch


async def test_authenticate_refreshes_last_used_after_the_interval(
    service: TenantAPIKeyService,
    repo: FakeTenantAPIKeyRepository,
) -> None:
    created = await service.create_api_key(name="ci", tenant_id=_TENANT_ID)
    stale = datetime.now(UTC) - timedelta(minutes=5)
    repo.rows[created.key.id] = repo.rows[created.key.id].model_copy(update={"last_used_at": stale})

    await service.authenticate(created.token)

    refreshed = repo.rows[created.key.id].last_used_at
    assert refreshed is not None
    assert refreshed > stale


# ── listing ─────────────────────────────────────────────────────────


async def test_list_api_keys_returns_only_that_tenants_live_keys(
    service: TenantAPIKeyService,
) -> None:
    mine = await service.create_api_key(name="mine", tenant_id=_TENANT_ID)
    await service.create_api_key(name="theirs", tenant_id=_TENANT_ID + 1)
    revoked = await service.create_api_key(name="revoked", tenant_id=_TENANT_ID)
    await service.revoke_api_key(revoked.key.id, tenant_id=_TENANT_ID)

    keys = await service.list_api_keys(_TENANT_ID)

    assert [k.id for k in keys] == [mine.key.id]


async def test_list_platform_api_keys_filters_by_scope(
    service: TenantAPIKeyService,
) -> None:
    platform = await service.create_api_key(
        name="ops",
        scope_type=SCOPE_PLATFORM,
        capabilities=["system_tenants_read"],
    )
    await service.create_api_key(name="tenant-scoped", tenant_id=_TENANT_ID)

    keys = await service.list_platform_api_keys()

    assert [k.id for k in keys] == [platform.key.id]


# ── revocation ──────────────────────────────────────────────────────


async def test_revoke_api_key_stamps_revoked_at(
    service: TenantAPIKeyService,
    repo: FakeTenantAPIKeyRepository,
) -> None:
    created = await service.create_api_key(name="ci", tenant_id=_TENANT_ID)

    await service.revoke_api_key(created.key.id, tenant_id=_TENANT_ID)

    assert repo.rows[created.key.id].revoked_at is not None


async def test_revoke_api_key_rejects_another_tenants_key(
    service: TenantAPIKeyService,
) -> None:
    created = await service.create_api_key(name="ci", tenant_id=_TENANT_ID)

    with pytest.raises(NotFoundError):
        await service.revoke_api_key(created.key.id, tenant_id=_TENANT_ID + 1)


async def test_revoke_api_key_twice_raises_not_found(service: TenantAPIKeyService) -> None:
    created = await service.create_api_key(name="ci", tenant_id=_TENANT_ID)
    await service.revoke_api_key(created.key.id, tenant_id=_TENANT_ID)

    with pytest.raises(NotFoundError):
        await service.revoke_api_key(created.key.id, tenant_id=_TENANT_ID)


async def test_revoke_platform_api_key_rejects_a_tenant_key(
    service: TenantAPIKeyService,
) -> None:
    created = await service.create_api_key(name="ci", tenant_id=_TENANT_ID)

    with pytest.raises(NotFoundError):
        await service.revoke_platform_api_key(created.key.id)


# ── backfill ────────────────────────────────────────────────────────


async def _seed_placeholder(
    repo: FakeTenantAPIKeyRepository,
    *,
    api_key: str = "sk-legacy",
) -> TenantAPIKey:
    return await repo.insert(
        TenantAPIKey(
            tenant_id=_TENANT_ID,
            name="legacy",
            key_hash=f"migrated-tenant-{_TENANT_ID}",
            api_key=api_key,
            full_access=True,
            created_at=_NOW,
            updated_at=_NOW,
        )
    )


async def test_backfill_replaces_placeholder_hashes(
    service: TenantAPIKeyService,
    repo: FakeTenantAPIKeyRepository,
) -> None:
    legacy = await _seed_placeholder(repo)

    backfilled = await service.backfill_missing_key_hashes()

    assert backfilled == 1
    assert repo.rows[legacy.id].key_hash == hash_api_key_token("sk-legacy")


async def test_backfilled_key_authenticates(
    service: TenantAPIKeyService,
    repo: FakeTenantAPIKeyRepository,
) -> None:
    legacy = await _seed_placeholder(repo)
    await service.backfill_missing_key_hashes()

    found = await service.authenticate("sk-legacy")

    assert found.id == legacy.id


async def test_backfill_skips_rows_without_a_stored_token(
    service: TenantAPIKeyService,
    repo: FakeTenantAPIKeyRepository,
) -> None:
    await _seed_placeholder(repo, api_key="   ")

    assert await service.backfill_missing_key_hashes() == 0


async def test_backfill_is_a_noop_without_placeholders(
    service: TenantAPIKeyService,
) -> None:
    await service.create_api_key(name="ci", tenant_id=_TENANT_ID)

    assert await service.backfill_missing_key_hashes() == 0
