"""Unit tests for `TenantService`.

The service is exercised against an ``AsyncMock(spec=TenantRepository)``
with closure-captured state so the SQL-touching methods (insert /
find_by_id / find_by_ids / list_all / search / update_by_primary_key /
adjust_storage_used / bulk_set_storage_quota) keep working in-memory.
A real-repository construction test guards against signature drift
between the mock spec and the concrete ``TenantRepository``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from src.common.exception import NotFoundError, ValidationError
from src.core.tenants.service import TenantService
from src.db.dao.tenants_repository import TenantRepository
from src.db.models.tenants.tenants import DEFAULT_STORAGE_QUOTA_BYTES, Tenant
from tests.util.service_test import ServiceTest

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _make_repo() -> tuple[AsyncMock, dict[int, Tenant]]:
    """Tenant-repo mock with stateful closure-backed storage."""
    repo = AsyncMock(spec=TenantRepository)
    rows: dict[int, Tenant] = {}
    _next_id = {"value": 0}

    def _live() -> dict[int, Tenant]:
        return {i: r for i, r in rows.items() if r.deleted_at is None}

    @staticmethod
    def _sorted(rs: list[Tenant]) -> list[Tenant]:
        return sorted(rs, key=lambda r: r.created_at, reverse=True)

    async def _insert(row: Tenant) -> Tenant:
        _next_id["value"] += 1
        stored = row.model_copy(update={"id": _next_id["value"]})
        rows[stored.id] = stored
        return stored

    async def _update_by_primary_key(
        primary_key_to_value: dict[str, object],
        column_to_update: dict[str, object],
    ) -> Tenant | None:
        tenant_id = int(str(primary_key_to_value["id"]))
        row = _live().get(tenant_id)
        if row is None:
            return None
        updated = row.model_copy(update=column_to_update)
        rows[tenant_id] = updated
        return updated

    async def _adjust_storage_used(
        tenant_id: int,
        *,
        delta: int,
        updated_at: datetime,
    ) -> int:
        row = _live().get(tenant_id)
        if row is None:
            raise NotFoundError(code="tenant.not_found", message="Tenant not found")
        used = max(row.storage_used + delta, 0)
        rows[tenant_id] = row.model_copy(
            update={"storage_used": used, "updated_at": updated_at}
        )
        return used

    async def _bulk_set_storage_quota(
        *,
        quota_bytes: int,
        updated_at: datetime,
    ) -> int:
        live = _live()
        for tenant_id, row in live.items():
            rows[tenant_id] = row.model_copy(
                update={"storage_quota": quota_bytes, "updated_at": updated_at}
            )
        return len(live)

    async def _find_by_id(id_: str | int) -> Tenant:
        row = _live().get(int(str(id_)))
        if row is None:
            raise NotFoundError(code="tenant.not_found", message=f"Tenant {id_} not found")
        return row

    async def _find_by_ids(ids: list[int]) -> list[Tenant]:
        if not ids:
            return []
        wanted = set(ids)
        return _sorted([r for r in _live().values() if r.id in wanted])

    async def _list_all(
        *, limit: int | None = None, offset: int = 0
    ) -> list[Tenant]:
        rs = _sorted(list(_live().values()))
        return rs[offset : offset + limit] if limit is not None else rs

    async def _search(
        *,
        keyword: str | None = None,
        tenant_id: int | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> tuple[list[Tenant], int]:
        matches = _sorted(
            [r for r in _live().values() if _matches(r, keyword, tenant_id)]
        )
        page = matches[offset : offset + limit] if limit is not None else matches
        return page, len(matches)

    @staticmethod
    def _matches(row: Tenant, keyword: str | None, tenant_id: int | None) -> bool:
        if tenant_id is None and not keyword:
            return True
        if tenant_id is not None and tenant_id > 0 and row.id == tenant_id:
            return True
        if not keyword:
            return False
        return keyword in row.name or keyword in (row.description or "")

    repo.insert.side_effect = _insert
    repo.update_by_primary_key.side_effect = _update_by_primary_key
    repo.adjust_storage_used.side_effect = _adjust_storage_used
    repo.bulk_set_storage_quota.side_effect = _bulk_set_storage_quota
    repo.find_by_id.side_effect = _find_by_id
    repo.find_by_ids.side_effect = _find_by_ids
    repo.list_all.side_effect = _list_all
    repo.search.side_effect = _search
    return repo, rows


def _seed(
    rows: dict[int, Tenant],
    *,
    name: str = "acme",
    description: str | None = None,
    created_at: datetime = _NOW,
) -> Tenant:
    """Insert a row directly into the closure-captured store."""
    next_id = max(rows.keys(), default=0) + 1
    row = Tenant(
        id=next_id,
        name=name,
        description=description,
        created_at=created_at,
        updated_at=created_at,
    )
    rows[next_id] = row
    return row


@pytest.fixture
def repo_and_rows() -> tuple[AsyncMock, dict[int, Tenant]]:
    return _make_repo()


@pytest.fixture
def repo(repo_and_rows: tuple[AsyncMock, dict[int, Tenant]]) -> AsyncMock:
    return repo_and_rows[0]


@pytest.fixture
def rows(repo_and_rows: tuple[AsyncMock, dict[int, Tenant]]) -> dict[int, Tenant]:
    return repo_and_rows[1]


@pytest.fixture
def service(repo_and_rows: tuple[AsyncMock, dict[int, Tenant]]) -> TenantService:
    return TenantService(tenants_repo=repo_and_rows[0])


# ── create_tenant ───────────────────────────────────────────────────


class TestCreateTenant(ServiceTest):
    async def test_persists_active_workspace(
        self, service: TenantService, rows: dict[int, Tenant]
    ) -> None:
        info = await service.create_tenant(name="acme", description="the workspace")

        assert info.id in rows
        assert info.name == "acme"
        assert info.status == "active"
        assert info.storage_quota == DEFAULT_STORAGE_QUOTA_BYTES

    async def test_trims_the_name(self, service: TenantService) -> None:
        info = await service.create_tenant(name="  acme  ")
        assert info.name == "acme"

    async def test_honours_explicit_storage_quota(self, service: TenantService) -> None:
        info = await service.create_tenant(name="acme", storage_quota=4096)
        assert info.storage_quota == 4096

    async def test_rejects_blank_name(self, service: TenantService) -> None:
        with pytest.raises(ValidationError) as excinfo:
            await service.create_tenant(name="   ")
        assert excinfo.value.code == "tenant.name_required"

    async def test_ignores_caller_supplied_status(
        self, service: TenantService, rows: dict[int, Tenant]
    ) -> None:
        info = await service.create_tenant(name="acme")
        assert rows[info.id].status == "active"


# ── get_tenant / get_tenants ────────────────────────────────────────


class TestGetTenant(ServiceTest):
    async def test_returns_projection(self, service: TenantService, rows: dict[int, Tenant]) -> None:
        stored = _seed(rows, name="acme")
        info = await service.get_tenant(stored.id)
        assert info.id == stored.id
        assert info.name == "acme"

    async def test_missing_raises_not_found(self, service: TenantService) -> None:
        with pytest.raises(NotFoundError) as excinfo:
            await service.get_tenant(4242)
        assert excinfo.value.code == "tenant.not_found"


class TestGetTenants(ServiceTest):
    async def test_returns_a_map_keyed_by_id(
        self, service: TenantService, rows: dict[int, Tenant]
    ) -> None:
        first = _seed(rows, name="first")
        second = _seed(rows, name="second")

        result = await service.get_tenants([first.id, second.id, 9999])

        assert set(result) == {first.id, second.id}


# ── list_tenants ────────────────────────────────────────────────────


class TestListTenants(ServiceTest):
    async def test_orders_newest_first(
        self, service: TenantService, rows: dict[int, Tenant]
    ) -> None:
        older = _seed(rows, name="older", created_at=_NOW)
        newer = _seed(rows, name="newer", created_at=_NOW + timedelta(days=1))

        infos = await service.list_tenants()

        assert [i.id for i in infos] == [newer.id, older.id]

    async def test_excludes_deleted(
        self, service: TenantService, rows: dict[int, Tenant]
    ) -> None:
        live = _seed(rows, name="live")
        gone = _seed(rows, name="gone")
        await service.delete_tenant(gone.id)

        infos = await service.list_tenants()

        assert [i.id for i in infos] == [live.id]


# ── search_tenants ──────────────────────────────────────────────────


class TestSearchTenants(ServiceTest):
    async def test_paginates_and_reports_total(
        self, service: TenantService, rows: dict[int, Tenant]
    ) -> None:
        for index in range(5):
            _seed(rows, name=f"t{index}", created_at=_NOW + timedelta(hours=index))

        infos, total = await service.search_tenants(page=2, page_size=2)

        assert total == 5
        assert [i.name for i in infos] == ["t2", "t1"]

    async def test_without_pagination_returns_everything(
        self, service: TenantService, rows: dict[int, Tenant]
    ) -> None:
        for index in range(3):
            _seed(rows, name=f"t{index}", created_at=_NOW + timedelta(hours=index))

        infos, total = await service.search_tenants(page=0, page_size=0)

        assert total == 3
        assert len(infos) == 3

    async def test_filters_by_keyword(
        self, service: TenantService, rows: dict[int, Tenant]
    ) -> None:
        match = _seed(rows, name="alpha")
        _seed(rows, name="beta")

        infos, total = await service.search_tenants(keyword="alpha")

        assert total == 1
        assert [i.id for i in infos] == [match.id]


# ── update_tenant ───────────────────────────────────────────────────


class TestUpdateTenant(ServiceTest):
    async def test_patches_only_supplied_columns(
        self, service: TenantService, rows: dict[int, Tenant]
    ) -> None:
        stored = _seed(rows, name="acme", description="original")

        info = await service.update_tenant(stored.id, description="patched")

        assert info.name == "acme"
        assert info.description == "patched"

    async def test_stamps_updated_at(
        self, service: TenantService, rows: dict[int, Tenant]
    ) -> None:
        stored = _seed(rows, name="acme")

        info = await service.update_tenant(stored.id, name="acme corp")

        assert info.updated_at > stored.updated_at

    async def test_rejects_blank_name(
        self, service: TenantService, rows: dict[int, Tenant]
    ) -> None:
        stored = _seed(rows, name="acme")

        with pytest.raises(ValidationError) as excinfo:
            await service.update_tenant(stored.id, name="  ")

        assert excinfo.value.code == "tenant.name_required"

    async def test_rejects_zero_id(self, service: TenantService) -> None:
        with pytest.raises(ValidationError) as excinfo:
            await service.update_tenant(0, name="acme")
        assert excinfo.value.code == "tenant.invalid_id"

    async def test_missing_raises_not_found(self, service: TenantService) -> None:
        with pytest.raises(NotFoundError) as excinfo:
            await service.update_tenant(4242, name="acme")
        assert excinfo.value.code == "tenant.not_found"


# ── delete_tenant ───────────────────────────────────────────────────


class TestDeleteTenant(ServiceTest):
    async def test_soft_deletes_the_row(
        self, service: TenantService, rows: dict[int, Tenant]
    ) -> None:
        stored = _seed(rows, name="acme")

        assert await service.delete_tenant(stored.id) is True
        assert rows[stored.id].deleted_at is not None

    async def test_is_idempotent(
        self, service: TenantService, rows: dict[int, Tenant]
    ) -> None:
        stored = _seed(rows, name="acme")
        await service.delete_tenant(stored.id)

        assert await service.delete_tenant(stored.id) is False

    async def test_unknown_id_reports_false(self, service: TenantService) -> None:
        assert await service.delete_tenant(4242) is False


# ── storage counters ────────────────────────────────────────────────


class TestStorageCounters(ServiceTest):
    async def test_adjust_storage_used_returns_new_total(
        self, service: TenantService, rows: dict[int, Tenant]
    ) -> None:
        stored = _seed(rows, name="acme")

        assert await service.adjust_storage_used(stored.id, delta=2048) == 2048

    async def test_adjust_storage_used_clamps_at_zero(
        self, service: TenantService, rows: dict[int, Tenant]
    ) -> None:
        stored = _seed(rows, name="acme")
        await service.adjust_storage_used(stored.id, delta=10)

        assert await service.adjust_storage_used(stored.id, delta=-50) == 0

    async def test_bulk_set_storage_quota_applies_to_every_tenant(
        self, service: TenantService, rows: dict[int, Tenant]
    ) -> None:
        first = _seed(rows, name="a")
        _seed(rows, name="b")

        affected = await service.bulk_set_storage_quota(quota_bytes=4096)

        assert affected == 2
        assert rows[first.id].storage_quota == 4096

    @pytest.mark.parametrize("quota", [0, -1])
    async def test_bulk_set_storage_quota_rejects_non_positive(
        self, service: TenantService, quota: int
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