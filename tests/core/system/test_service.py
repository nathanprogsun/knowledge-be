"""Unit tests for ``SystemSettingService`` + ``AuditLogService``.

Per AGENTS.md §9, core services are tested with Protocol-based fakes
where they materially reduce test setup. The fakes mirror the real
repository contracts (finders return storage rows; the service projects
them to ``SystemSettingInfo`` / ``AuditLogInfo`` via ``map_from_db``).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.common.exception import ValidationError
from src.core.system.audit_actions import AuditAction, AuditOutcome
from src.core.system.audit_service import AuditLogService
from src.core.system.system_setting_service import SystemSettingService
from src.db.models.system.audit_log import AuditLog
from src.db.models.system.system_setting import SystemSetting

# ── In-memory fakes ──────────────────────────────────────────────────


class _FakeAuditLogRepo:
    """In-memory ``AuditLogRepository`` replacement."""

    def __init__(self) -> None:
        self.rows: list[AuditLog] = []
        self._next_id: int = 1

    async def create(self, entry: AuditLog) -> AuditLog:
        persisted = entry.model_copy(update={"id": self._next_id})
        self._next_id += 1
        self.rows.append(persisted)
        return persisted

    async def list_for_tenant(
        self,
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
        rows = [r for r in self.rows if r.tenant_id == tenant_id]
        if after_id > 0:
            rows = [r for r in rows if r.id < after_id]
        if action:
            rows = [r for r in rows if r.action == action]
        if outcome:
            rows = [r for r in rows if r.outcome == outcome]
        if actor_user_id:
            rows = [r for r in rows if r.actor_user_id == actor_user_id]
        if scope_type:
            rows = [r for r in rows if r.scope_type == scope_type]
        if scope_id:
            rows = [r for r in rows if r.scope_id == scope_id]
        if unscoped_only:
            rows = [r for r in rows if not r.scope_type]
        rows = sorted(rows, key=lambda r: r.id, reverse=True)
        return rows[:limit]

    async def count_since_for_dedup(
        self,
        *,
        tenant_id: int,
        actor_user_id: str,
        action: str,
        request_path: str,
        since: datetime,
    ) -> int:
        return sum(
            1
            for r in self.rows
            if r.tenant_id == tenant_id
            and r.actor_user_id == actor_user_id
            and r.action == action
            and r.request_path == request_path
            and r.created_at >= since
        )

    async def delete_older_than(self, cutoff: datetime) -> int:
        before = len(self.rows)
        self.rows = [r for r in self.rows if r.created_at >= cutoff]
        return before - len(self.rows)


class _FakeSystemSettingRepo:
    """In-memory ``SystemSettingRepository`` replacement."""

    def __init__(self) -> None:
        self.rows: dict[str, SystemSetting] = {}
        self._next_id: int = 1

    async def get_by_key(self, key: str) -> SystemSetting | None:
        return self.rows.get(key)

    async def list_all(self) -> list[SystemSetting]:
        return sorted(self.rows.values(), key=lambda r: (r.category, r.key))

    async def upsert(self, setting: SystemSetting) -> SystemSetting:
        existing = self.rows.get(setting.key)
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
            persisted = setting.model_copy(update={"id": self._next_id})
            self._next_id += 1
        self.rows[setting.key] = persisted
        return persisted

    async def delete_by_key(self, key: str) -> int:
        if key in self.rows:
            del self.rows[key]
            return 1
        return 0


# ── SystemSettingService ────────────────────────────────────────────


def _make_setting_svc(
    *,
    settings_repo: _FakeSystemSettingRepo | None = None,
    audit_repo: _FakeAuditLogRepo | None = None,
) -> tuple[SystemSettingService, _FakeSystemSettingRepo, _FakeAuditLogRepo]:
    sr = settings_repo or _FakeSystemSettingRepo()
    ar = audit_repo or _FakeAuditLogRepo()
    return (
        SystemSettingService(
            settings_repo=sr,  # type: ignore[arg-type]
            audit_repo=ar,  # type: ignore[arg-type]
        ),
        sr,
        ar,
    )


async def test_list_includes_virtual_rows_for_unpersisted_keys() -> None:
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


async def test_list_merges_persisted_rows_with_registry() -> None:
    svc, sr, _ = _make_setting_svc()
    now = datetime.now(UTC)
    await sr.upsert(
        SystemSetting(
            id=0,
            key="auth.registration_mode",
            value="invite_only",
            value_type="string",
            category="auth",
            description="",
            is_secret=False,
            requires_restart=False,
            last_modified_by="usr-1",
            created_at=now,
            updated_at=now,
        )
    )
    infos = await svc.list_settings()
    row = next(i for i in infos if i.key == "auth.registration_mode")
    assert row.value == "invite_only"
    assert row.id != 0
    assert row.enum == ["self_serve", "invite_only"]


async def test_get_unknown_key_raises() -> None:
    svc, _, _ = _make_setting_svc()
    with pytest.raises(ValidationError):
        await svc.get("nope.does_not_exist")


async def test_get_returns_virtual_row_when_unpersisted() -> None:
    svc, _, _ = _make_setting_svc()
    info = await svc.get("auth.registration_mode")
    assert info.id == 0
    assert info.value == "self_serve"


async def test_update_unknown_key_raises() -> None:
    svc, _, _ = _make_setting_svc()
    with pytest.raises(ValidationError):
        await svc.update(key="nope", raw_value=1)


async def test_update_type_mismatch_raises() -> None:
    svc, _, _ = _make_setting_svc()
    with pytest.raises(ValidationError):
        await svc.update(key="auth.registration_mode", raw_value=123)


async def test_update_enum_violation_raises() -> None:
    svc, _, _ = _make_setting_svc()
    with pytest.raises(ValidationError):
        await svc.update(key="auth.registration_mode", raw_value="bogus")


async def test_update_persists_and_emits_audit() -> None:
    svc, sr, ar = _make_setting_svc()
    info = await svc.update(
        key="auth.registration_mode",
        raw_value="invite_only",
        actor_user_id="usr-admin",
    )
    assert info.value == "invite_only"
    assert info.last_modified_by == "usr-admin"
    # Row persisted.
    persisted = await sr.get_by_key("auth.registration_mode")
    assert persisted is not None
    assert persisted.value == "invite_only"
    # Audit row emitted.
    assert len(ar.rows) == 1
    audit = ar.rows[0]
    assert audit.action == AuditAction.SYSTEM_SETTING_CHANGED
    assert audit.tenant_id == 0
    assert audit.target_id == "auth.registration_mode"
    assert audit.outcome == AuditOutcome.SUCCESS
    assert audit.actor_user_id == "usr-admin"


async def test_reset_unknown_key_raises() -> None:
    svc, _, _ = _make_setting_svc()
    with pytest.raises(ValidationError):
        await svc.reset("nope")


async def test_reset_idempotent_on_unpersisted() -> None:
    svc, _, _ = _make_setting_svc()
    await svc.reset("auth.registration_mode")  # no error


async def test_reset_deletes_persisted_row() -> None:
    svc, sr, _ = _make_setting_svc()
    await svc.update(key="auth.registration_mode", raw_value="invite_only")
    assert await sr.get_by_key("auth.registration_mode") is not None
    await svc.reset("auth.registration_mode")
    assert await sr.get_by_key("auth.registration_mode") is None


# ── 3-tier resolver ─────────────────────────────────────────────────


async def test_get_int_falls_back_to_default() -> None:
    svc, _, _ = _make_setting_svc()
    val = await svc.get_int("tenant.max_owned_per_user", "", 99)
    assert val == 99  # no DB row, no env → default


async def test_get_int_reads_db_row() -> None:
    svc, _, _ = _make_setting_svc()
    await svc.update(key="tenant.max_owned_per_user", raw_value=42)
    val = await svc.get_int("tenant.max_owned_per_user", "", 99)
    assert val == 42


async def test_get_bool_reads_db_row() -> None:
    svc, _, _ = _make_setting_svc()
    await svc.update(key="tenant.self_service_creation_enabled", raw_value=False)
    val = await svc.get_bool("tenant.self_service_creation_enabled", "", True)
    assert val is False


# ── AuditLogService ─────────────────────────────────────────────────


def _make_audit_svc(
    *, repo: _FakeAuditLogRepo | None = None
) -> tuple[AuditLogService, _FakeAuditLogRepo]:
    r = repo or _FakeAuditLogRepo()
    return AuditLogService(audit_repo=r), r  # type: ignore[arg-type]


async def test_log_writes_entry() -> None:
    svc, repo = _make_audit_svc()
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
    assert len(repo.rows) == 1


async def test_log_denied_dedup_skips_recent() -> None:
    svc, repo = _make_audit_svc()
    await svc.log_denied(
        tenant_id=7,
        actor_user_id="usr-1",
        actor_role="viewer",
        action=AuditAction.ACCESS_DENIED,
        request_path="/api/v1/tenants/7/members",
    )
    assert len(repo.rows) == 1
    # Second call within 1 minute is deduped.
    result = await svc.log_denied(
        tenant_id=7,
        actor_user_id="usr-1",
        actor_role="viewer",
        action=AuditAction.ACCESS_DENIED,
        request_path="/api/v1/tenants/7/members",
    )
    assert result is None
    assert len(repo.rows) == 1


async def test_list_returns_newest_first_with_cursor() -> None:
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


async def test_purge_noop_when_retention_zero() -> None:
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
