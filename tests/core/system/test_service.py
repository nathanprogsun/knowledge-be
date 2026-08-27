"""Unit tests for ``SystemSettingService`` + ``AuditLogService``.

Per AGENTS.md §9, core services are tested with Protocol-based mocks
where they materially reduce test setup. The mocks mirror the real
repository contracts (finders return storage rows; the service projects
them to ``SystemSettingInfo`` / ``AuditLogInfo`` via ``map_from_db``).
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from src.common.exception import ValidationError
from src.core.system.audit_actions import AuditAction, AuditOutcome
from src.core.system.audit_service import AuditLogService
from src.core.system.system_setting_service import SystemSettingService
from src.db.dao.audit_log_repository import AuditLogRepository
from src.db.dao.system_setting_repository import SystemSettingRepository
from src.db.models.system.audit_log import AuditLog
from src.db.models.system.system_setting import SystemSetting
from tests.util.service_test import ServiceTest

# ── In-memory repository doubles (stateful via side_effect closures) ─


def _make_audit_repo() -> tuple[AsyncMock, list[AuditLog]]:
    """Audit-repo mock with insertion tracking + cursor list + dedup count."""
    repo = AsyncMock(spec=AuditLogRepository)
    rows: list[AuditLog] = []
    _next_id = {"value": 0}

    async def _create(entry: AuditLog) -> AuditLog:
        _next_id["value"] += 1
        persisted = entry.model_copy(update={"id": _next_id["value"]})
        rows.append(persisted)
        return persisted

    async def _list_for_tenant(
        *,
        tenant_id: int,
        after_id: int = 0,
        limit: int = 50,
        action: str | None = None,
        outcome: str | None = None,
        actor_user_id: str | None = None,
        scope_type: str | None = None,
        scope_id: str | None = None,
        unscoped_only: bool = False,
    ) -> list[AuditLog]:
        filtered = [r for r in rows if r.tenant_id == tenant_id]
        if after_id > 0:
            filtered = [r for r in filtered if r.id < after_id]
        if action:
            filtered = [r for r in filtered if r.action == action]
        if outcome:
            filtered = [r for r in filtered if r.outcome == outcome]
        if actor_user_id:
            filtered = [r for r in filtered if r.actor_user_id == actor_user_id]
        if scope_type:
            filtered = [r for r in filtered if r.scope_type == scope_type]
        if scope_id:
            filtered = [r for r in filtered if r.scope_id == scope_id]
        if unscoped_only:
            filtered = [r for r in filtered if not r.scope_type]
        filtered = sorted(filtered, key=lambda r: r.id, reverse=True)
        return filtered[:limit]

    async def _count_since_for_dedup(
        *,
        tenant_id: int,
        actor_user_id: str,
        action: str,
        request_path: str,
        since: datetime,
    ) -> int:
        return sum(
            1
            for r in rows
            if r.tenant_id == tenant_id
            and r.actor_user_id == actor_user_id
            and r.action == action
            and r.request_path == request_path
            and r.created_at >= since
        )

    async def _delete_older_than(cutoff: datetime) -> int:
        before = len(rows)
        rows[:] = [r for r in rows if r.created_at >= cutoff]
        return before - len(rows)

    repo.create.side_effect = _create
    repo.list_for_tenant.side_effect = _list_for_tenant
    repo.count_since_for_dedup.side_effect = _count_since_for_dedup
    repo.delete_older_than.side_effect = _delete_older_than
    return repo, rows


def _make_settings_repo() -> tuple[AsyncMock, dict[str, SystemSetting]]:
    """Settings-repo mock with key-indexed upsert + lookup + list."""
    repo = AsyncMock(spec=SystemSettingRepository)
    rows: dict[str, SystemSetting] = {}
    _next_id = {"value": 0}

    async def _get_by_key(key: str) -> SystemSetting | None:
        return rows.get(key)

    async def _list_all() -> list[SystemSetting]:
        return sorted(rows.values(), key=lambda r: (r.category, r.key))

    async def _upsert(setting: SystemSetting) -> SystemSetting:
        existing = rows.get(setting.key)
        if existing is not None:
            persisted = existing.model_copy(
                update={
                    "value": setting.value,
                    "value_type": setting.value_type,
                    "category": setting.category,
                    "description": setting.description,
                    "is_secret": setting.is_secret,
                    "requires_restart": setting.requires_restart,
                    "last_modified_by": setting.last_modified_by,
                    "updated_at": setting.updated_at,
                }
            )
        else:
            _next_id["value"] += 1
            persisted = setting.model_copy(update={"id": _next_id["value"]})
        rows[setting.key] = persisted
        return persisted

    async def _delete_by_key(key: str) -> int:
        if key in rows:
            del rows[key]
            return 1
        return 0

    repo.get_by_key.side_effect = _get_by_key
    repo.list_all.side_effect = _list_all
    repo.upsert.side_effect = _upsert
    repo.delete_by_key.side_effect = _delete_by_key
    return repo, rows


# ── SystemSettingService ────────────────────────────────────────────


def _make_setting_svc() -> tuple[SystemSettingService, dict[str, SystemSetting], list[AuditLog]]:
    sr, sr_rows = _make_settings_repo()
    ar, ar_rows = _make_audit_repo()
    return (
        SystemSettingService(settings_repo=sr, audit_repo=ar),
        sr_rows,
        ar_rows,
    )


class TestSystemSettingService(ServiceTest):
    async def test_list_includes_virtual_rows_for_unpersisted_keys(self) -> None:
        svc, _, _ = _make_setting_svc()

        infos = await svc.list_settings()
        keys = {i.key for i in infos}
        # Registry keys appear even though nothing is persisted.
        assert "auth.registration_mode" in keys
        assert "tenant.default_storage_quota_gb" in keys
        # Virtual rows have id=0.
        vm = next(i for i in infos if i.key == "auth.registration_mode")
        assert vm.id == 0
        assert vm.value == "self_serve"
        assert vm.enum == ["self_serve", "invite_only"]

    async def test_list_merges_persisted_rows_with_registry(self) -> None:
        svc, _, _ = _make_setting_svc()
        await svc.update(key="auth.registration_mode", raw_value="invite_only")

        infos = await svc.list_settings()
        row = next(i for i in infos if i.key == "auth.registration_mode")
        assert row.value == "invite_only"
        assert row.id != 0
        assert row.enum == ["self_serve", "invite_only"]

    async def test_get_unknown_key_raises(self) -> None:
        svc, _, _ = _make_setting_svc()
        with pytest.raises(ValidationError):
            await svc.get("nope.does_not_exist")

    async def test_get_returns_virtual_row_when_unpersisted(self) -> None:
        svc, _, _ = _make_setting_svc()
        info = await svc.get("auth.registration_mode")
        assert info.id == 0
        assert info.value == "self_serve"

    async def test_update_unknown_key_raises(self) -> None:
        svc, _, _ = _make_setting_svc()
        with pytest.raises(ValidationError):
            await svc.update(key="nope", raw_value=1)

    async def test_update_type_mismatch_raises(self) -> None:
        svc, _, _ = _make_setting_svc()
        with pytest.raises(ValidationError):
            await svc.update(key="auth.registration_mode", raw_value=123)

    async def test_update_enum_violation_raises(self) -> None:
        svc, _, _ = _make_setting_svc()
        with pytest.raises(ValidationError):
            await svc.update(key="auth.registration_mode", raw_value="bogus")

    async def test_update_persists_and_emits_audit(self) -> None:
        svc, sr_rows, ar_rows = _make_setting_svc()
        info = await svc.update(
            key="auth.registration_mode",
            raw_value="invite_only",
            actor_user_id="usr-admin",
        )
        assert info.value == "invite_only"
        assert info.last_modified_by == "usr-admin"
        # Row persisted.
        persisted = sr_rows["auth.registration_mode"]
        assert persisted.value == "invite_only"
        # Audit row emitted.
        assert len(ar_rows) == 1
        audit = ar_rows[0]
        assert audit.action == AuditAction.SYSTEM_SETTING_CHANGED
        assert audit.tenant_id == 0
        assert audit.target_id == "auth.registration_mode"
        assert audit.outcome == AuditOutcome.SUCCESS
        assert audit.actor_user_id == "usr-admin"

    async def test_reset_unknown_key_raises(self) -> None:
        svc, _, _ = _make_setting_svc()
        with pytest.raises(ValidationError):
            await svc.reset("nope")

    async def test_reset_idempotent_on_unpersisted(self) -> None:
        svc, _, _ = _make_setting_svc()
        await svc.reset("auth.registration_mode")  # no error

    async def test_reset_deletes_persisted_row(self) -> None:
        svc, sr_rows, _ = _make_setting_svc()
        await svc.update(key="auth.registration_mode", raw_value="invite_only")
        assert sr_rows.get("auth.registration_mode") is not None
        await svc.reset("auth.registration_mode")
        assert sr_rows.get("auth.registration_mode") is None


# ── 3-tier resolver ─────────────────────────────────────────────────


class TestResolver(ServiceTest):
    async def test_get_int_falls_back_to_default(self) -> None:
        svc, _, _ = _make_setting_svc()
        val = await svc.get_int("tenant.max_owned_per_user", "", 99)
        assert val == 99  # no DB row, no env → default

    async def test_get_int_reads_db_row(self) -> None:
        svc, _, _ = _make_setting_svc()
        await svc.update(key="tenant.max_owned_per_user", raw_value=42)
        val = await svc.get_int("tenant.max_owned_per_user", "", 99)
        assert val == 42

    async def test_get_bool_reads_db_row(self) -> None:
        svc, _, _ = _make_setting_svc()
        await svc.update(key="tenant.self_service_creation_enabled", raw_value=False)
        val = await svc.get_bool("tenant.self_service_creation_enabled", "", True)
        assert val is False


# ── AuditLogService ─────────────────────────────────────────────────


def _make_audit_svc() -> tuple[AuditLogService, list[AuditLog]]:
    repo, rows = _make_audit_repo()
    return AuditLogService(audit_repo=repo), rows


class TestAuditLogService(ServiceTest):
    async def test_log_writes_entry(self) -> None:
        svc, rows = _make_audit_svc()
        entry = AuditLog(
            id=0,
            tenant_id=7,
            actor_user_id="usr-1",
            action=AuditAction.MEMBER_ADDED,
            outcome=AuditOutcome.SUCCESS,
            created_at=datetime.now(UTC),
        )

        persisted = await svc.log(entry)

        assert persisted.id != 0
        assert len(rows) == 1

    async def test_log_denied_dedup_skips_recent(self) -> None:
        svc, rows = _make_audit_svc()
        await svc.log_denied(
            tenant_id=7,
            actor_user_id="usr-1",
            actor_role="viewer",
            action=AuditAction.ACCESS_DENIED,
            request_path="/api/v1/tenants/7/members",
        )
        assert len(rows) == 1
        # Second call within 1 minute is deduped.
        result = await svc.log_denied(
            tenant_id=7,
            actor_user_id="usr-1",
            actor_role="viewer",
            action=AuditAction.ACCESS_DENIED,
            request_path="/api/v1/tenants/7/members",
        )
        assert result is None
        assert len(rows) == 1

    async def test_list_returns_newest_first_with_cursor(self) -> None:
        svc, _ = _make_audit_svc()
        for _ in range(5):
            await svc.log(
                AuditLog(
                    id=0,
                    tenant_id=7,
                    actor_user_id="usr-1",
                    action=AuditAction.MEMBER_ADDED,
                    outcome=AuditOutcome.SUCCESS,
                    created_at=datetime.now(UTC),
                )
            )

        page1 = await svc.list_entries(tenant_id=7, limit=2)
        assert len(page1.entries) == 2
        assert page1.next_cursor != 0

        page2 = await svc.list_entries(tenant_id=7, limit=2, after_id=page1.next_cursor)
        assert len(page2.entries) == 2
        # All ids are strictly decreasing across pages.
        assert page2.entries[0].id < page1.entries[-1].id

    async def test_purge_noop_when_retention_zero(self) -> None:
        svc, _ = _make_audit_svc()
        await svc.log(
            AuditLog(
                id=0,
                tenant_id=7,
                actor_user_id="usr-1",
                action=AuditAction.MEMBER_ADDED,
                outcome=AuditOutcome.SUCCESS,
                created_at=datetime.now(UTC),
            )
        )

        deleted = await svc.purge(0)

        assert deleted == 0


__all__ = []
