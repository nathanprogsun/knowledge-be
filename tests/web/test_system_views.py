"""Web-layer tests for the system-admin router (settings CRUD + audit).

Per AGENTS.md §9, web routers are tested via ``httpx.AsyncClient``
against the app. The service dependencies are overridden with
service-backed fakes so the tests exercise the full HTTP path (routing,
serialization, exception handling) without touching a real database.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from src.core.system.audit_service import AuditLogService
from src.core.system.system_setting_service import SystemSettingService
from src.db.models.system.audit_log import AuditLog
from src.db.models.system.system_setting import SystemSetting
from src.web.deps import (
    get_audit_log_service,
    get_system_setting_service,
)
from src.web.deps.rbac import require_system_admin_dep

# ── In-memory fakes (reuse the service-test fakes' shape) ───────────


class _FakeAuditLogRepo:
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
        return 0

    async def delete_older_than(self, cutoff: datetime) -> int:
        before = len(self.rows)
        self.rows = [r for r in self.rows if r.created_at >= cutoff]
        return before - len(self.rows)


class _FakeSystemSettingRepo:
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


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def fake_audit_repo() -> _FakeAuditLogRepo:
    return _FakeAuditLogRepo()


@pytest.fixture
def fake_settings_repo() -> _FakeSystemSettingRepo:
    return _FakeSystemSettingRepo()


@pytest.fixture(autouse=True)
def _override_services(
    web_app: FastAPI,  # noqa: ARG001 - resolved from the parent conftest
    fake_audit_repo: _FakeAuditLogRepo,
    fake_settings_repo: _FakeSystemSettingRepo,
) -> FastAPI:
    """Override system service deps on the shared web app (autouse).

    The system-admin router gates on ``SystemAdminDep``; the header
    channel uses role ``owner`` (not ``system_admin``), so we bypass
    the dep locally — the test is exercising settings CRUD, not RBAC.
    """
    web_app.dependency_overrides[require_system_admin_dep] = lambda: None
    web_app.dependency_overrides[get_system_setting_service] = lambda: SystemSettingService(
        settings_repo=fake_settings_repo,  # type: ignore[arg-type]
        audit_repo=fake_audit_repo,  # type: ignore[arg-type]
    )
    web_app.dependency_overrides[get_audit_log_service] = lambda: AuditLogService(
        audit_repo=fake_audit_repo,  # type: ignore[arg-type]
    )
    return web_app


# ── GET /system/admin/settings ──────────────────────────────────────


async def test_list_settings_returns_registry_keys(
    web_authed_client: AsyncClient,
) -> None:
    resp = await web_authed_client.get("/system/admin/settings")
    assert resp.status_code == 200
    body = resp.json()
    keys = {r["key"] for r in body}
    assert "auth.registration_mode" in keys
    assert "tenant.default_storage_quota_gb" in keys
    # Virtual rows have id=0.
    vm = next(r for r in body if r["key"] == "auth.registration_mode")
    assert vm["id"] == 0
    assert vm["value"] == "self_serve"


# ── GET /system/admin/settings/{key} ────────────────────────────────


async def test_get_setting_returns_virtual_row(web_authed_client: AsyncClient) -> None:
    resp = await web_authed_client.get("/system/admin/settings/auth.registration_mode")
    assert resp.status_code == 200
    body = resp.json()
    assert body["key"] == "auth.registration_mode"
    assert body["enum"] == ["self_serve", "invite_only"]


async def test_get_unknown_setting_returns_422(web_authed_client: AsyncClient) -> None:
    resp = await web_authed_client.get("/system/admin/settings/nope.does_not_exist")
    assert resp.status_code == 422


# ── PUT /system/admin/settings/{key} ────────────────────────────────


async def test_update_setting_persists_and_audits(
    web_authed_client: AsyncClient,
    fake_audit_repo: _FakeAuditLogRepo,
) -> None:
    resp = await web_authed_client.put(
        "/system/admin/settings/auth.registration_mode",
        json={"value": "invite_only"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["value"] == "invite_only"
    # Audit row emitted.
    assert len(fake_audit_repo.rows) == 1
    assert fake_audit_repo.rows[0].action == "system.setting_changed"


async def test_update_setting_type_mismatch_returns_422(
    web_authed_client: AsyncClient,
) -> None:
    resp = await web_authed_client.put(
        "/system/admin/settings/auth.registration_mode",
        json={"value": 123},
    )
    assert resp.status_code == 422


async def test_update_setting_enum_violation_returns_422(
    web_authed_client: AsyncClient,
) -> None:
    resp = await web_authed_client.put(
        "/system/admin/settings/auth.registration_mode",
        json={"value": "bogus"},
    )
    assert resp.status_code == 422


# ── DELETE /system/admin/settings/{key} ─────────────────────────────


async def test_reset_setting_idempotent(web_authed_client: AsyncClient) -> None:
    resp = await web_authed_client.delete("/system/admin/settings/auth.registration_mode")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True


async def test_reset_unknown_setting_returns_422(web_authed_client: AsyncClient) -> None:
    resp = await web_authed_client.delete("/system/admin/settings/nope")
    assert resp.status_code == 422


# ── GET /system/admin/audit-log ─────────────────────────────────────


async def test_system_audit_log_returns_newest_first(
    web_authed_client: AsyncClient,
    fake_audit_repo: _FakeAuditLogRepo,
) -> None:
    now = datetime.now(UTC)
    for i in range(3):
        fake_audit_repo.rows.append(
            AuditLog(
                id=10 - i,
                tenant_id=0,
                actor_user_id="usr-admin",
                action="system.setting_changed",
                outcome="success",
                created_at=now,
            )
        )
    fake_audit_repo._next_id = 100
    resp = await web_authed_client.get("/system/admin/audit-log?limit=2")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert len(body["data"]) == 2
    # Newest first.
    assert body["data"][0]["id"] > body["data"][1]["id"]
    assert body["next_cursor"] == body["data"][1]["id"]


async def test_system_audit_log_filter_by_action(
    web_authed_client: AsyncClient,
    fake_audit_repo: _FakeAuditLogRepo,
) -> None:
    now = datetime.now(UTC)
    fake_audit_repo.rows.append(
        AuditLog(
            id=1,
            tenant_id=0,
            actor_user_id="usr-admin",
            action="system.setting_changed",
            outcome="success",
            created_at=now,
        )
    )
    fake_audit_repo.rows.append(
        AuditLog(
            id=2,
            tenant_id=0,
            actor_user_id="usr-admin",
            action="system.admin_promoted",
            outcome="success",
            created_at=now,
        )
    )
    resp = await web_authed_client.get("/system/admin/audit-log?action=system.admin_promoted")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["data"]) == 1
    assert body["data"][0]["action"] == "system.admin_promoted"


__all__ = []