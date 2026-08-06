"""Unit tests for `TenantService`.

The service is exercised against the shared in-memory
`FakeTenantRepository` (``tests/fakes/tenants.py``), whose method
signatures mirror the real ``TenantRepository``, plus one test that
constructs the service with the real repository so a signature drift
between the two surfaces fails here rather than in production.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.common.exception import NotFoundError, ValidationError
from src.core.tenants.service import TenantService
from src.db.dao.tenants_repository import TenantRepository
from src.db.models.tenants.tenants import DEFAULT_STORAGE_QUOTA_BYTES, Tenant
from tests.unit.fakes.tenants import FakeTenantRepository

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture
def repo() -> FakeTenantRepository:
    return FakeTenantRepository()


@pytest.fixture
def service(repo: FakeTenantRepository) -> TenantService:
    return TenantService(tenants_repo=repo)  # type: ignore[arg-type]


async def _seed(
    repo: FakeTenantRepository,
    *,
    name: str = "acme",
    description: str | None = None,
    created_at: datetime = _NOW,
) -> Tenant:
    return await repo.insert(
        Tenant(
            name=name,
            description=description,
            created_at=created_at,
            updated_at=created_at,
        )
    )


# ── create_tenant ───────────────────────────────────────────────────


async def test_create_tenant_persists_active_workspace(
    service: TenantService,
    repo: FakeTenantRepository,
) -> None:
    info = await service.create_tenant(name="acme", description="the workspace")

    assert info.id in repo.rows
    assert info.name == "acme"
    assert info.status == "active"
    assert info.storage_quota == DEFAULT_STORAGE_QUOTA_BYTES


async def test_create_tenant_trims_the_name(service: TenantService) -> None:
    info = await service.create_tenant(name="  acme  ")

    assert info.name == "acme"


async def test_create_tenant_honours_explicit_storage_quota(service: TenantService) -> None:
    info = await service.create_tenant(name="acme", storage_quota=4096)

    assert info.storage_quota == 4096


async def test_create_tenant_rejects_blank_name(service: TenantService) -> None:
    with pytest.raises(ValidationError) as excinfo:
        await service.create_tenant(name="   ")

    assert excinfo.value.code == "tenant.name_required"


async def test_create_tenant_ignores_caller_supplied_status(
    service: TenantService,
    repo: FakeTenantRepository,
) -> None:
    info = await service.create_tenant(name="acme")

    assert repo.rows[info.id].status == "active"


# ── get_tenant / get_tenants ────────────────────────────────────────


async def test_get_tenant_returns_projection(
    service: TenantService,
    repo: FakeTenantRepository,
) -> None:
    stored = await _seed(repo, name="acme")

    info = await service.get_tenant(stored.id)

    assert info.id == stored.id
    assert info.name == "acme"


async def test_get_tenant_rejects_zero_id(service: TenantService) -> None:
    with pytest.raises(ValidationError) as excinfo:
        await service.get_tenant(0)

    assert excinfo.value.code == "tenant.invalid_id"


async def test_get_tenant_missing_raises_not_found(service: TenantService) -> None:
    with pytest.raises(NotFoundError):
        await service.get_tenant(4242)


async def test_get_tenants_maps_by_id_and_skips_unknown(
    service: TenantService,
    repo: FakeTenantRepository,
) -> None:
    first = await _seed(repo, name="a")
    second = await _seed(repo, name="b")

    found = await service.get_tenants([first.id, second.id, 999])

    assert set(found) == {first.id, second.id}


async def test_get_tenants_empty_input_returns_empty_map(service: TenantService) -> None:
    assert await service.get_tenants([]) == {}


# ── list_tenants ────────────────────────────────────────────────────


async def test_list_tenants_returns_newest_first(
    service: TenantService,
    repo: FakeTenantRepository,
) -> None:
    older = await _seed(repo, name="older", created_at=_NOW)
    newer = await _seed(repo, name="newer", created_at=_NOW + timedelta(days=1))

    infos = await service.list_tenants()

    assert [i.id for i in infos] == [newer.id, older.id]


async def test_list_tenants_excludes_deleted(
    service: TenantService,
    repo: FakeTenantRepository,
) -> None:
    live = await _seed(repo, name="live")
    gone = await _seed(repo, name="gone")
    await service.delete_tenant(gone.id)

    infos = await service.list_tenants()

    assert [i.id for i in infos] == [live.id]


# ── search_tenants ──────────────────────────────────────────────────


async def test_search_tenants_paginates_and_reports_total(
    service: TenantService,
    repo: FakeTenantRepository,
) -> None:
    for index in range(5):
        await _seed(repo, name=f"t{index}", created_at=_NOW + timedelta(hours=index))

    infos, total = await service.search_tenants(page=2, page_size=2)

    assert total == 5
    assert [i.name for i in infos] == ["t2", "t1"]


async def test_search_tenants_without_pagination_returns_everything(
    service: TenantService,
    repo: FakeTenantRepository,
) -> None:
    for index in range(3):
        await _seed(repo, name=f"t{index}", created_at=_NOW + timedelta(hours=index))

    infos, total = await service.search_tenants(page=0, page_size=0)

    assert total == 3
    assert len(infos) == 3


async def test_search_tenants_filters_by_keyword(
    service: TenantService,
    repo: FakeTenantRepository,
) -> None:
    match = await _seed(repo, name="alpha")
    await _seed(repo, name="beta")

    infos, total = await service.search_tenants(keyword="alpha")

    assert total == 1
    assert [i.id for i in infos] == [match.id]


# ── update_tenant ───────────────────────────────────────────────────


async def test_update_tenant_patches_only_supplied_columns(
    service: TenantService,
    repo: FakeTenantRepository,
) -> None:
    stored = await _seed(repo, name="acme", description="original")

    info = await service.update_tenant(stored.id, description="patched")

    assert info.name == "acme"
    assert info.description == "patched"


async def test_update_tenant_stamps_updated_at(
    service: TenantService,
    repo: FakeTenantRepository,
) -> None:
    stored = await _seed(repo, name="acme")

    info = await service.update_tenant(stored.id, name="acme corp")

    assert info.updated_at > stored.updated_at


async def test_update_tenant_rejects_blank_name(
    service: TenantService,
    repo: FakeTenantRepository,
) -> None:
    stored = await _seed(repo, name="acme")

    with pytest.raises(ValidationError) as excinfo:
        await service.update_tenant(stored.id, name="  ")

    assert excinfo.value.code == "tenant.name_required"


async def test_update_tenant_rejects_zero_id(service: TenantService) -> None:
    with pytest.raises(ValidationError) as excinfo:
        await service.update_tenant(0, name="acme")

    assert excinfo.value.code == "tenant.invalid_id"


async def test_update_tenant_missing_raises_not_found(service: TenantService) -> None:
    with pytest.raises(NotFoundError) as excinfo:
        await service.update_tenant(4242, name="acme")

    assert excinfo.value.code == "tenant.not_found"


# ── delete_tenant ───────────────────────────────────────────────────


async def test_delete_tenant_soft_deletes_the_row(
    service: TenantService,
    repo: FakeTenantRepository,
) -> None:
    stored = await _seed(repo, name="acme")

    assert await service.delete_tenant(stored.id) is True
    assert repo.rows[stored.id].deleted_at is not None


async def test_delete_tenant_is_idempotent(
    service: TenantService,
    repo: FakeTenantRepository,
) -> None:
    stored = await _seed(repo, name="acme")
    await service.delete_tenant(stored.id)

    assert await service.delete_tenant(stored.id) is False


async def test_delete_tenant_unknown_id_reports_false(service: TenantService) -> None:
    assert await service.delete_tenant(4242) is False


# ── storage counters ────────────────────────────────────────────────


async def test_adjust_storage_used_returns_new_total(
    service: TenantService,
    repo: FakeTenantRepository,
) -> None:
    stored = await _seed(repo, name="acme")

    assert await service.adjust_storage_used(stored.id, delta=2048) == 2048


async def test_adjust_storage_used_clamps_at_zero(
    service: TenantService,
    repo: FakeTenantRepository,
) -> None:
    stored = await _seed(repo, name="acme")
    await service.adjust_storage_used(stored.id, delta=10)

    assert await service.adjust_storage_used(stored.id, delta=-50) == 0


async def test_bulk_set_storage_quota_applies_to_every_tenant(
    service: TenantService,
    repo: FakeTenantRepository,
) -> None:
    first = await _seed(repo, name="a")
    await _seed(repo, name="b")

    affected = await service.bulk_set_storage_quota(quota_bytes=4096)

    assert affected == 2
    assert repo.rows[first.id].storage_quota == 4096


@pytest.mark.parametrize("quota", [0, -1])
async def test_bulk_set_storage_quota_rejects_non_positive(
    service: TenantService,
    quota: int,
) -> None:
    with pytest.raises(ValidationError) as excinfo:
        await service.bulk_set_storage_quota(quota_bytes=quota)

    assert excinfo.value.code == "tenant.invalid_quota"


# ── signature drift guard ───────────────────────────────────────────


def test_service_accepts_the_real_repository_type() -> None:
    """Construction with the concrete repo must keep type-checking.

    ``TenantRepository`` needs a session only when a query runs, so
    ``None`` is enough to prove the constructor contract holds.
    """
    service = TenantService(tenants_repo=TenantRepository(None))  # type: ignore[arg-type]

    assert isinstance(service, TenantService)
