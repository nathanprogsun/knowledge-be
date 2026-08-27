"""Unit tests for `TenantAPIKeyService` and its token helpers.

The service is exercised against an ``AsyncMock(spec=TenantAPIKeyRepository)``
with closure-captured state. Revoked rows are filtered from reads so
``list_for_tenant`` and friends behave like the SQL they mirror.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

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
from src.db.dao.tenant_api_keys_repository import (
    PLACEHOLDER_KEY_HASH_PREFIX,
    TenantAPIKeyRepository,
)
from src.db.models.tenants.tenant_api_keys import TenantAPIKey
from tests.util.service_test import ServiceTest

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_TENANT_ID = 7


def _make_repo() -> tuple[AsyncMock, dict[int, TenantAPIKey]]:
    """Tenant-API-key repo mock with closure-captured state."""
    repo = AsyncMock(spec=TenantAPIKeyRepository)
    rows: dict[int, TenantAPIKey] = {}
    _next_id = {"value": 0}

    def _live() -> dict[int, TenantAPIKey]:
        return {i: r for i, r in rows.items() if r.revoked_at is None}

    @staticmethod
    def _sorted(rs: list[TenantAPIKey]) -> list[TenantAPIKey]:
        return sorted(rs, key=lambda r: r.created_at, reverse=True)

    async def _insert(row: TenantAPIKey) -> TenantAPIKey:
        _next_id["value"] += 1
        stored = row.model_copy(update={"id": _next_id["value"]})
        rows[stored.id] = stored
        return stored

    async def _revoke(key_id: int, *, tenant_id: int, revoked_at: datetime) -> None:
        row = _live().get(key_id)
        if row is None or row.tenant_id != tenant_id:
            raise NotFoundError(code="tenant_api_key.not_found", message="Tenant API key not found")
        rows[key_id] = row.model_copy(update={"revoked_at": revoked_at})

    async def _revoke_platform(key_id: int, *, revoked_at: datetime) -> None:
        row = _live().get(key_id)
        if row is None or row.scope_type != "platform":
            raise NotFoundError(code="tenant_api_key.not_found", message="Tenant API key not found")
        rows[key_id] = row.model_copy(update={"revoked_at": revoked_at})

    async def _touch_last_used(key_id: int, *, used_at: datetime) -> int:
        row = _live().get(key_id)
        if row is None:
            return 0
        rows[key_id] = row.model_copy(update={"last_used_at": used_at})
        return 1

    async def _update_hash(key_id: int, *, key_hash: str) -> int:
        row = _live().get(key_id)
        if row is None:
            return 0
        rows[key_id] = row.model_copy(update={"key_hash": key_hash})
        return 1

    async def _find_by_hash(key_hash: str) -> TenantAPIKey:
        for row in _live().values():
            if row.key_hash == key_hash:
                return row
        raise NotFoundError(code="tenant_api_key.not_found", message="Tenant API key not found")

    async def _list_for_tenant(tenant_id: int) -> list[TenantAPIKey]:
        return _sorted([r for r in _live().values() if r.tenant_id == tenant_id])

    async def _list_platform() -> list[TenantAPIKey]:
        return _sorted([r for r in _live().values() if r.scope_type == "platform"])

    async def _list_with_placeholder_hash() -> list[TenantAPIKey]:
        return _sorted(
            [r for r in _live().values() if r.key_hash.startswith(PLACEHOLDER_KEY_HASH_PREFIX)]
        )

    async def _has_placeholder_hash() -> bool:
        return bool(await _list_with_placeholder_hash())

    repo.insert.side_effect = _insert
    repo.revoke.side_effect = _revoke
    repo.revoke_platform.side_effect = _revoke_platform
    repo.touch_last_used.side_effect = _touch_last_used
    repo.update_hash.side_effect = _update_hash
    repo.find_by_hash.side_effect = _find_by_hash
    repo.list_for_tenant.side_effect = _list_for_tenant
    repo.list_platform.side_effect = _list_platform
    repo.list_with_placeholder_hash.side_effect = _list_with_placeholder_hash
    repo.has_placeholder_hash.side_effect = _has_placeholder_hash
    return repo, rows


@pytest.fixture
def repo_and_rows() -> tuple[AsyncMock, dict[int, TenantAPIKey]]:
    return _make_repo()


@pytest.fixture
def repo(repo_and_rows: tuple[AsyncMock, dict[int, TenantAPIKey]]) -> AsyncMock:
    return repo_and_rows[0]


@pytest.fixture
def rows(repo_and_rows: tuple[AsyncMock, dict[int, TenantAPIKey]]) -> dict[int, TenantAPIKey]:
    return repo_and_rows[1]


@pytest.fixture
def service(repo: AsyncMock) -> TenantAPIKeyService:
    return TenantAPIKeyService(api_keys_repo=repo)


# ── token helpers ───────────────────────────────────────────────────


class TestTokenHelpers(ServiceTest):
    def test_generate_token_is_prefixed_and_unpadded(self) -> None:
        token = generate_api_key_token()
        assert token.startswith("sk-")
        assert "=" not in token

    def test_generate_token_is_unique_per_call(self) -> None:
        assert generate_api_key_token() != generate_api_key_token()

    def test_hash_is_stable_sha256_hex(self) -> None:
        digest = hash_api_key_token("sk-example")
        assert digest == hash_api_key_token("sk-example")
        assert len(digest) == 64

    def test_normalize_scope_type_defaults_to_tenant(self) -> None:
        assert normalize_scope_type(None) == SCOPE_TENANT
        assert normalize_scope_type("  nonsense ") == SCOPE_TENANT
        assert normalize_scope_type(" Platform ") == SCOPE_PLATFORM

    def test_normalize_capabilities_drops_unknown_and_duplicates(self) -> None:
        assert normalize_capabilities(["Chat", "chat", "nope", " ingest "]) == ["chat", "ingest"]

    def test_normalize_knowledge_base_ids_keeps_case_and_order(self) -> None:
        assert normalize_knowledge_base_ids([" KB-1 ", "kb-1", "KB-1", ""]) == ["KB-1", "kb-1"]


# ── create_api_key ──────────────────────────────────────────────────


class TestCreateApiKey(ServiceTest):
    async def test_returns_token_once_and_stores_only_its_hash(
        self, service: TenantAPIKeyService, rows: dict[int, TenantAPIKey]
    ) -> None:
        result = await service.create_api_key(name="ci", tenant_id=_TENANT_ID, full_access=True)

        stored = rows[result.key.id]
        assert result.token.startswith("sk-")
        assert stored.key_hash == hash_api_key_token(result.token)
        assert set(result.key.model_dump()).isdisjoint({"key_hash", "api_key"})

    async def test_trims_name(self, service: TenantAPIKeyService) -> None:
        result = await service.create_api_key(name="  ci  ", tenant_id=_TENANT_ID, full_access=True)
        assert result.key.name == "ci"

    async def test_rejects_blank_name(self, service: TenantAPIKeyService) -> None:
        with pytest.raises(ValidationError) as excinfo:
            await service.create_api_key(name="   ", tenant_id=_TENANT_ID, full_access=True)
        assert excinfo.value.code == "tenant_api_key.name_required"

    async def test_tenant_key_requires_a_tenant_id(self, service: TenantAPIKeyService) -> None:
        with pytest.raises(ValidationError):
            await service.create_api_key(name="ci", full_access=False)

    async def test_tenant_key_requires_capabilities_when_not_full_access(
        self, service: TenantAPIKeyService
    ) -> None:
        with pytest.raises(ValidationError) as excinfo:
            await service.create_api_key(name="ci", tenant_id=_TENANT_ID, capabilities=[])
        assert excinfo.value.code == "tenant_api_key.capabilities_required"

    async def test_platform_key_rejects_full_access(self, service: TenantAPIKeyService) -> None:
        with pytest.raises(ValidationError) as excinfo:
            await service.create_api_key(
                name="ops", scope_type=SCOPE_PLATFORM, full_access=True, capabilities=["chat"]
            )
        assert excinfo.value.code == "tenant_api_key.platform_full_access"

    async def test_platform_key_requires_capabilities(self, service: TenantAPIKeyService) -> None:
        with pytest.raises(ValidationError) as excinfo:
            await service.create_api_key(name="ops", scope_type=SCOPE_PLATFORM, capabilities=[])
        assert excinfo.value.code == "tenant_api_key.capabilities_required"

    async def test_full_access_clears_narrowing_lists(
        self, service: TenantAPIKeyService, rows: dict[int, TenantAPIKey]
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


# ── authentication ──────────────────────────────────────────────────


class TestAuthenticate(ServiceTest):
    async def test_resolves_a_valid_token(self, service: TenantAPIKeyService) -> None:
        created = await service.create_api_key(name="ci", tenant_id=_TENANT_ID, full_access=True)

        found = await service.authenticate(created.token)

        assert found.id == created.key.id

    async def test_rejects_empty_token(self, service: TenantAPIKeyService) -> None:
        with pytest.raises(NotFoundError):
            await service.authenticate("   ")

    async def test_rejects_unknown_token(self, service: TenantAPIKeyService) -> None:
        with pytest.raises(NotFoundError):
            await service.authenticate("sk-never-issued")

    async def test_rejects_revoked_key(self, service: TenantAPIKeyService) -> None:
        created = await service.create_api_key(name="ci", tenant_id=_TENANT_ID, full_access=True)
        await service.revoke_api_key(created.key.id, tenant_id=_TENANT_ID)

        with pytest.raises(NotFoundError):
            await service.authenticate(created.token)

    async def test_rejects_expired_key(self, service: TenantAPIKeyService) -> None:
        created = await service.create_api_key(
            name="ci",
            tenant_id=_TENANT_ID,
            full_access=True,
            expires_at=datetime.now(UTC) - timedelta(seconds=1),
        )

        with pytest.raises(NotFoundError):
            await service.authenticate(created.token)

    async def test_stamps_last_used_on_first_use(
        self, service: TenantAPIKeyService, rows: dict[int, TenantAPIKey]
    ) -> None:
        created = await service.create_api_key(name="ci", tenant_id=_TENANT_ID, full_access=True)

        await service.authenticate(created.token)

        assert rows[created.key.id].last_used_at is not None

    async def test_throttles_repeated_last_used_writes(
        self, service: TenantAPIKeyService, rows: dict[int, TenantAPIKey]
    ) -> None:
        created = await service.create_api_key(name="ci", tenant_id=_TENANT_ID, full_access=True)
        await service.authenticate(created.token)
        first_touch = rows[created.key.id].last_used_at

        await service.authenticate(created.token)

        assert rows[created.key.id].last_used_at == first_touch

    async def test_refreshes_last_used_after_the_interval(
        self, service: TenantAPIKeyService, rows: dict[int, TenantAPIKey]
    ) -> None:
        created = await service.create_api_key(name="ci", tenant_id=_TENANT_ID, full_access=True)
        stale = datetime.now(UTC) - timedelta(minutes=5)
        rows[created.key.id] = rows[created.key.id].model_copy(update={"last_used_at": stale})

        await service.authenticate(created.token)

        refreshed = rows[created.key.id].last_used_at
        assert refreshed is not None
        assert refreshed > stale


# ── listing ─────────────────────────────────────────────────────────


class TestListing(ServiceTest):
    async def test_returns_only_that_tenants_live_keys(self, service: TenantAPIKeyService) -> None:
        mine = await service.create_api_key(name="mine", tenant_id=_TENANT_ID, full_access=True)
        await service.create_api_key(name="theirs", tenant_id=_TENANT_ID + 1, full_access=True)
        revoked = await service.create_api_key(
            name="revoked", tenant_id=_TENANT_ID, full_access=True
        )
        await service.revoke_api_key(revoked.key.id, tenant_id=_TENANT_ID)

        keys = await service.list_api_keys(_TENANT_ID)

        assert [k.id for k in keys] == [mine.key.id]

    async def test_list_platform_filters_by_scope(self, service: TenantAPIKeyService) -> None:
        platform = await service.create_api_key(
            name="ops",
            scope_type=SCOPE_PLATFORM,
            capabilities=["system_tenants_read"],
        )
        await service.create_api_key(name="tenant-scoped", tenant_id=_TENANT_ID, full_access=True)

        keys = await service.list_platform_api_keys()

        assert [k.id for k in keys] == [platform.key.id]


# ── revocation ──────────────────────────────────────────────────────


class TestRevocation(ServiceTest):
    async def test_stamps_revoked_at(
        self, service: TenantAPIKeyService, rows: dict[int, TenantAPIKey]
    ) -> None:
        created = await service.create_api_key(name="ci", tenant_id=_TENANT_ID, full_access=True)

        await service.revoke_api_key(created.key.id, tenant_id=_TENANT_ID)

        assert rows[created.key.id].revoked_at is not None

    async def test_rejects_another_tenants_key(self, service: TenantAPIKeyService) -> None:
        created = await service.create_api_key(name="ci", tenant_id=_TENANT_ID, full_access=True)

        with pytest.raises(NotFoundError):
            await service.revoke_api_key(created.key.id, tenant_id=_TENANT_ID + 1)

    async def test_twice_raises_not_found(self, service: TenantAPIKeyService) -> None:
        created = await service.create_api_key(name="ci", tenant_id=_TENANT_ID, full_access=True)
        await service.revoke_api_key(created.key.id, tenant_id=_TENANT_ID)

        with pytest.raises(NotFoundError):
            await service.revoke_api_key(created.key.id, tenant_id=_TENANT_ID)

    async def test_revoke_platform_rejects_a_tenant_key(self, service: TenantAPIKeyService) -> None:
        created = await service.create_api_key(name="ci", tenant_id=_TENANT_ID, full_access=True)

        with pytest.raises(NotFoundError):
            await service.revoke_platform_api_key(created.key.id)


# ── backfill ────────────────────────────────────────────────────────


async def _seed_placeholder(
    rows: dict[int, TenantAPIKey],
    *,
    api_key: str = "sk-legacy",
) -> TenantAPIKey:
    next_id = max(rows.keys(), default=0) + 1
    row = TenantAPIKey(
        id=next_id,
        tenant_id=_TENANT_ID,
        name="legacy",
        key_hash=f"migrated-tenant-{_TENANT_ID}",
        api_key=api_key,
        full_access=True,
        created_at=_NOW,
        updated_at=_NOW,
    )
    rows[next_id] = row
    return row


class TestBackfill(ServiceTest):
    async def test_replaces_placeholder_hashes(
        self, service: TenantAPIKeyService, rows: dict[int, TenantAPIKey]
    ) -> None:
        legacy = await _seed_placeholder(rows)

        backfilled = await service.backfill_missing_key_hashes()

        assert backfilled == 1
        assert rows[legacy.id].key_hash == hash_api_key_token("sk-legacy")

    async def test_backfilled_key_authenticates(
        self, service: TenantAPIKeyService, rows: dict[int, TenantAPIKey]
    ) -> None:
        legacy = await _seed_placeholder(rows)
        await service.backfill_missing_key_hashes()

        found = await service.authenticate("sk-legacy")

        assert found.id == legacy.id

    async def test_skips_rows_without_a_stored_token(
        self, service: TenantAPIKeyService, rows: dict[int, TenantAPIKey]
    ) -> None:
        await _seed_placeholder(rows, api_key="   ")

        assert await service.backfill_missing_key_hashes() == 0

    async def test_is_a_noop_without_placeholders(self, service: TenantAPIKeyService) -> None:
        await service.create_api_key(name="ci", tenant_id=_TENANT_ID, full_access=True)

        assert await service.backfill_missing_key_hashes() == 0
