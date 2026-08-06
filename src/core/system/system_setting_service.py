"""System-setting service — 3-tier resolver + CRUD + audit.

Operations:

- ``list`` — return every known setting (DB-persisted + registry-backed
  virtual rows for keys that have no DB row yet).
- ``get`` — return one setting by key (virtual row when not persisted).
- ``get_int`` / ``get_string`` / ``get_bool`` / ``get_string_list`` —
  the 3-tier resolver (DB > ENV > built-in default) used by feature
  services at runtime.
- ``update`` — validate against the registry, upsert the DB row, emit
  an audit row (``system.setting_changed``).
- ``reset`` — delete the DB override so the resolver falls back to
  ENV / built-in default.

The Redis pubsub cache invalidation, the SSRF whitelist side-effect
bridge, and the model-concurrency limiter push are stubbed; they land
alongside the services they configure.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import cast

from src.common.exception import ValidationError
from src.common.json import JsonObject, JsonValue
from src.core.system.audit_actions import AuditAction, AuditOutcome
from src.core.system.registry import SettingSpec, all_keys, get_spec
from src.core.system.types import SystemSettingInfo
from src.db.dao.audit_log_repository import AuditLogRepository
from src.db.dao.system_setting_repository import SystemSettingRepository
from src.db.models.system.audit_log import AuditLog
from src.db.models.system.system_setting import SystemSetting


class SystemSettingService:
    """Platform-wide tunables service — 3-tier resolver + admin CRUD."""

    def __init__(
        self,
        *,
        settings_repo: SystemSettingRepository,
        audit_repo: AuditLogRepository,
    ) -> None:
        self._settings_repo = settings_repo
        self._audit_repo = audit_repo

    # ── Admin CRUD (mounted under /system/admin/settings) ────────────

    async def list_settings(self) -> list[SystemSettingInfo]:
        """Return every known setting.

        DB-persisted rows are enriched with the registry's ``enum``;
        keys that exist in the registry but have no DB row yet are
        synthesised as virtual rows so the management UI renders the
        full set.
        """
        persisted = await self._settings_repo.list_all()
        persisted_by_key = {row.key: row for row in persisted}
        infos: list[SystemSettingInfo] = []
        for key in all_keys():
            spec = get_spec(key)
            assert spec is not None
            row = persisted_by_key.get(key)
            if row is not None:
                infos.append(SystemSettingInfo.map_from_db(row, enum=list(spec.enum)))
            else:
                infos.append(self._virtual_setting(key, spec))
        return infos

    async def get(self, key: str) -> SystemSettingInfo:
        """Return one setting by key.

        Raises ``ValidationError`` for unknown keys. Returns a virtual
        row when the key is registered but not yet persisted.
        """
        spec = get_spec(key)
        if spec is None:
            raise ValidationError(
                code="system.unknown_setting_key",
                message=f"unknown setting key {key!r}",
            )
        row = await self._settings_repo.get_by_key(key)
        if row is None:
            return self._virtual_setting(key, spec)
        return SystemSettingInfo.map_from_db(row, enum=list(spec.enum))

    async def update(
        self,
        *,
        key: str,
        raw_value: int | str | bool | list[str],
        actor_user_id: str = "",
    ) -> SystemSettingInfo:
        """Validate and persist a new value for ``key``.

        Raises ``ValidationError`` for unknown keys, type mismatches,
        or enum violations. Emits an audit row
        (``system.setting_changed``) on success.
        """
        spec = get_spec(key)
        if spec is None:
            raise ValidationError(
                code="system.unknown_setting_key",
                message=f"unknown setting key {key!r}",
            )
        encoded = _encode_for_type(spec, raw_value)
        _validate_enum(spec, raw_value)

        prev = await self._settings_repo.get_by_key(key)
        category = prev.category if prev is not None else (spec.category or "general")
        description = prev.description if prev is not None else spec.description
        is_secret = prev.is_secret if prev is not None else False
        requires_restart = prev.requires_restart if prev is not None else spec.requires_restart

        now = datetime.now(UTC)
        row = SystemSetting(
            id=prev.id if prev is not None else 0,
            key=key,
            value=encoded,
            value_type=spec.value_type,
            category=category,
            description=description,
            is_secret=is_secret,
            requires_restart=requires_restart,
            last_modified_by=actor_user_id,
            created_at=prev.created_at if prev is not None else now,
            updated_at=now,
        )
        persisted = await self._settings_repo.upsert(row)

        # Audit row — tenant_id=0 is the system-scope convention.
        old_value = prev.value if prev is not None else None
        await self._audit_repo.create(
            AuditLog(
                id=0,
                tenant_id=0,
                actor_user_id=actor_user_id,
                actor_role="system_admin",
                action=AuditAction.SYSTEM_SETTING_CHANGED,
                target_type="system_setting",
                target_id=key,
                outcome=AuditOutcome.SUCCESS,
                details={
                    "old_value": old_value,
                    "new_value": encoded,
                    "value_type": spec.value_type,
                },
                created_at=now,
            )
        )
        return SystemSettingInfo.map_from_db(persisted, enum=list(spec.enum))

    async def reset(self, key: str) -> None:
        """Delete the DB override for ``key``.

        Idempotent — deleting a key that was never persisted is a
        no-op (no audit row written). Raises ``ValidationError`` for
        unknown keys so a typo on the URL doesn't silently pretend it
        cleared something.
        """
        spec = get_spec(key)
        if spec is None:
            raise ValidationError(
                code="system.unknown_setting_key",
                message=f"unknown setting key {key!r}",
            )
        await self._settings_repo.delete_by_key(key)

    # ── 3-tier resolver (DB > ENV > default) ──────────────────────────

    async def get_int(self, key: str, env_name: str, default: int) -> int:
        """Resolve ``key`` as ``int`` via DB > ENV > default."""
        raw = await self._resolve_raw(key)
        if raw is not None:
            if isinstance(raw, bool):
                return default
            if isinstance(raw, int):
                return raw
            if isinstance(raw, str):
                try:
                    return int(raw)
                except ValueError:
                    return default
            return default
        if env_name:
            env_raw = os.environ.get(env_name)
            if env_raw is not None and env_raw.strip():
                try:
                    return int(env_raw)
                except ValueError:
                    pass
        return default

    async def get_string(self, key: str, env_name: str, default: str) -> str:
        """Resolve ``key`` as ``str`` via DB > ENV > default."""
        raw = await self._resolve_raw(key)
        if raw is not None:
            return str(raw)
        if env_name:
            env_val = os.environ.get(env_name)
            if env_val is not None:
                return env_val
        return default

    async def get_bool(self, key: str, env_name: str, default: bool) -> bool:
        """Resolve ``key`` as ``bool`` via DB > ENV > default."""
        raw = await self._resolve_raw(key)
        if raw is not None:
            return bool(raw)
        if env_name:
            env_val = os.environ.get(env_name)
            if env_val is not None:
                return env_val.strip().lower() in ("1", "true", "yes", "on")
        return default

    async def get_string_list(self, key: str, env_name: str, default: list[str]) -> list[str]:
        """Resolve ``key`` as ``list[str]`` via DB > ENV > default."""
        raw = await self._resolve_raw(key)
        if raw is not None:
            if isinstance(raw, list):
                return [str(x) for x in raw]
            if isinstance(raw, str):
                return [s.strip() for s in raw.split(",") if s.strip()]
            return default
        if env_name:
            env_val = os.environ.get(env_name)
            if env_val is not None and env_val.strip():
                return [s.strip() for s in env_val.split(",") if s.strip()]
        return default

    # ── Internal helpers ─────────────────────────────────────────────

    async def _resolve_raw(self, key: str) -> int | str | bool | list[str] | None:
        """Return the DB-persisted value for ``key``, or ``None``."""
        row = await self._settings_repo.get_by_key(key)
        if row is None:
            return None
        return cast("int | str | bool | list[str]", row.value)

    @staticmethod
    def _virtual_setting(key: str, spec: SettingSpec) -> SystemSettingInfo:
        """Build a registry-backed virtual row for an unpersisted key."""
        now = datetime.now(UTC)
        return SystemSettingInfo(
            id=0,
            key=key,
            value=spec.default,
            value_type=spec.value_type,
            category=spec.category,
            description=spec.description,
            is_secret=False,
            requires_restart=spec.requires_restart,
            last_modified_by="",
            created_at=now,
            updated_at=now,
            enum=list(spec.enum),
            last_modified_by_name="",
        )


def _encode_for_type(
    spec: SettingSpec,
    raw_value: int | str | bool | list[str],
) -> JsonObject | list[JsonValue] | str | int | bool:
    """Normalise ``raw_value`` against the registry's declared type.

    Raises ``ValidationError`` on type mismatch.
    """
    if spec.value_type == "int":
        if isinstance(raw_value, bool) or not isinstance(raw_value, int):
            raise ValidationError(
                code="system.setting_type_mismatch",
                message=f"expected int for {spec.value_type!r}, got {type(raw_value).__name__}",
            )
        return raw_value
    if spec.value_type == "string":
        if not isinstance(raw_value, str):
            raise ValidationError(
                code="system.setting_type_mismatch",
                message=f"expected string for {spec.value_type!r}, got {type(raw_value).__name__}",
            )
        return raw_value
    if spec.value_type == "bool":
        if not isinstance(raw_value, bool):
            raise ValidationError(
                code="system.setting_type_mismatch",
                message=f"expected bool for {spec.value_type!r}, got {type(raw_value).__name__}",
            )
        return raw_value
    if spec.value_type == "string_list":
        if not isinstance(raw_value, list) or not all(isinstance(x, str) for x in raw_value):
            raise ValidationError(
                code="system.setting_type_mismatch",
                message=f"expected list[str] for {spec.value_type!r}, got {type(raw_value).__name__}",
            )
        return list(raw_value)
    raise ValidationError(
        code="system.setting_unknown_type",
        message=f"unknown value_type {spec.value_type!r} for key",
    )


def _validate_enum(spec: SettingSpec, raw_value: int | str | bool | list[str]) -> None:
    """Reject values outside the registry-declared enum (string only)."""
    if not spec.enum or spec.value_type != "string":
        return
    if not isinstance(raw_value, str) or raw_value not in spec.enum:
        raise ValidationError(
            code="system.setting_enum_violation",
            message=f"value {raw_value!r} not in {list(spec.enum)}",
        )


__all__ = ["SystemSettingService"]
