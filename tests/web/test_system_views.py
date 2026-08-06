"""Web-layer tests for the system-admin router (settings CRUD + audit).

Exercises the router over HTTP via ``httpx.AsyncClient`` against the
app. Service deps are overridden with the real services backed by
``AsyncMock(spec=...)`` repositories configured with stateful closures,
so the tests exercise the full HTTP path (routing, serialization,
exception handling) without touching a real database.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from src.core.system.audit_service import AuditLogService
from src.core.system.system_setting_service import SystemSettingService
from src.db.dao.audit_log_repository import AuditLogRepository
from src.db.dao.system_setting_repository import SystemSettingRepository
from src.db.models.system.audit_log import AuditLog
from src.db.models.system.system_setting import SystemSetting
from src.web.deps import (
    get_audit_log_service,
    get_system_setting_service,
)
from src.web.deps.rbac import require_system_admin_dep
from tests.util.service_test import stateful_insert


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def audit_repo() -> AsyncMock:
    repo = AsyncMock(spec=AuditLogRepository)
    rows: list[AuditLog] = []
    counter = [0]

    async def _create(entry: AuditLog) -> AuditLog:
        counter[0] += 1
        persisted = entry.model_copy(update={"id": counter[0]})
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
        out = [r for r in rows if r.tenant_id == tenant_id]
        if after_id > 0:
            out = [r for r in out if r.id < after_id]
        if action:
            out = [r for r in out if r.action == action]
        out = sorted(out, key=lambda r: r.id, reverse=True)
        return out[:limit]

    async def _count_since_for_dedup(
        *,
        tenant_id: int,
        actor_user_id: str,
        action: str,
        request_path: str,
        since: datetime,
    ) -> int:
        return 0

    async def _delete_older_than(cutoff: datetime) -> int:
        before = len(rows)
        rows[:] = [r for r in rows if r.created_at >= cutoff]
        return before - len(rows)

    repo.create.side_effect = _create
    repo.list_for_tenant.side_effect = _list_for_tenant
    repo.count_since_for_dedup.side_effect = _count_since_for_dedup
    repo.delete_older_than.side_effect = _delete_older_than
    repo._rows = rows  # type: ignore[attr-defined]
    repo._counter = counter  # type: ignore[attr-defined]
    return repo


@pytest.fixture
def settings_repo() -> AsyncMock:
    repo = AsyncMock(spec=SystemSettingRepository)
    rows: dict[str, SystemSetting] = {}
    counter = [0]

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
            counter[0] += 1
            persisted = setting.model_copy(update={"id": counter[0]})
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
    repo._rows = rows  # type: ignore[attr-defined]
    return repo


@pytest.fixture(autouse=True)
def _override_services(
    web_app: FastAPI,  # noqa: ARG001 - resolved from the parent conftest
    audit_repo: AsyncMock,
    settings_repo: AsyncMock,
) -> FastAPI:
    """Override system service deps on the shared web app (autouse).

    The system-admin router gates on ``SystemAdminDep``; the header
    channel uses role ``owner`` (not ``system_admin``), so we bypass
    the dep locally — the test is exercising settings CRUD, not RBAC.
    """
    web_app.dependency_overrides[require_system_admin_dep] = lambda: None
    web_app.dependency_overrides[get_system_setting_service] = lambda: SystemSettingService(
        settings_repo=settings_repo,
        audit_repo=audit_repo,
    )
    web_app.dependency_overrides[get_audit_log_service] = lambda: AuditLogService(
        audit_repo=audit_repo,
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
    audit_repo: AsyncMock,
) -> None:
    resp = await web_authed_client.put(
        "/system/admin/settings/auth.registration_mode",
        json={"value": "invite_only"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["value"] == "invite_only"
    # Audit row emitted.
    rows = audit_repo._rows  # type: ignore[attr-defined]
    assert len(rows) == 1
    assert rows[0].action == "system.setting_changed"


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
    audit_repo: AsyncMock,
) -> None:
    now = datetime.now(UTC)
    rows = audit_repo._rows  # type: ignore[attr-defined]
    for i in range(3):
        rows.append(
            AuditLog(
                id=10 - i,
                tenant_id=0,
                actor_user_id="usr-admin",
                action="system.setting_changed",
                outcome="success",
                created_at=now,
            )
        )
    audit_repo._counter[0] = 100  # type: ignore[attr-defined]
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
    audit_repo: AsyncMock,
) -> None:
    now = datetime.now(UTC)
    rows = audit_repo._rows  # type: ignore[attr-defined]
    rows.append(
        AuditLog(
            id=1,
            tenant_id=0,
            actor_user_id="usr-admin",
            action="system.setting_changed",
            outcome="success",
            created_at=now,
        )
    )
    rows.append(
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