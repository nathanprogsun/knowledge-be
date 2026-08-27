"""System-admin user operations — promote/revoke, password reset, audit.

The ``/system/admin`` endpoints are gated to system admins at the web
layer; this service owns the security-critical persistence: user
lookup, the promote/revoke state transitions, password hashing plus
session revocation, and the audit row each operation emits.

The revoke guards mirror the upstream repository semantics:

- the caller cannot revoke their own privileges,
- revoking the last remaining system admin is rejected,
- revoking an already-non-admin user is an idempotent success (the
  audit row is still written, marked ``changed=false``).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Final

from src.common.exception import NotFoundError, ValidationError
from src.common.json import JsonObject
from src.core.auth.types import UserInfo
from src.core.system.audit_actions import AuditAction, AuditOutcome
from src.db.dao.audit_log_repository import AuditLogRepository
from src.db.dao.auth_tokens_repository import AuthTokenRepository
from src.db.dao.users_repository import UserRepository
from src.db.models.auth.users import User
from src.db.models.system.audit_log import AuditLog
from src.util.security import hash_password

# Password policy shared with the registration form: 8-32 characters,
# at least one letter and one digit (mirrors the upstream policy).
_MIN_PASSWORD_LENGTH: Final = 8
_MAX_PASSWORD_LENGTH: Final = 32


def validate_password_policy(password: str) -> None:
    """Raise ``ValidationError`` when ``password`` fails the policy."""
    if not (_MIN_PASSWORD_LENGTH <= len(password) <= _MAX_PASSWORD_LENGTH):
        raise ValidationError(
            code="auth.password_policy",
            message="Password must be 8-32 characters long",
        )
    if not any(ch.isalpha() for ch in password) or not any(ch.isdigit() for ch in password):
        raise ValidationError(
            code="auth.password_policy",
            message="Password must contain at least one letter and one number",
        )


class SystemAdminService:
    """System-admin user operations plus their audit trail."""

    def __init__(
        self,
        *,
        users_repo: UserRepository,
        tokens_repo: AuthTokenRepository,
        audit_repo: AuditLogRepository,
    ) -> None:
        self._users_repo = users_repo
        self._tokens_repo = tokens_repo
        self._audit_repo = audit_repo

    # ── Promote ─────────────────────────────────────────────────────

    async def promote(self, *, user_id: str, email: str, actor_id: str) -> UserInfo:
        """Promote a user to system admin; idempotent when already admin.

        The target is identified by ``user_id`` (priority when both are
        given) or ``email``; a missing target raises ``NotFoundError``.
        """
        user = await self._find_user(user_id=user_id, email=email)
        if user.is_system_admin:
            idempotent = True
        else:
            user = await self._set_is_system_admin(user, True)
            idempotent = False
        await self._emit_admin_audit(
            action=AuditAction.SYSTEM_ADMIN_PROMOTED,
            user=user,
            actor_id=actor_id,
            details={"idempotent": idempotent},
        )
        return UserInfo.map_from_db(user)

    # ── Revoke ──────────────────────────────────────────────────────

    async def revoke(self, *, user_id: str, actor_id: str) -> UserInfo:
        """Revoke system-admin privileges with the self/last-admin guards."""
        if user_id == actor_id:
            raise ValidationError(
                code="system.cannot_revoke_self",
                message="Cannot revoke your own system admin privileges",
            )
        user = await self._find_user(user_id=user_id)
        if not user.is_system_admin:
            # Idempotent no-op — privileges were already absent. The audit
            # row is still written with changed=false so probing leaves a
            # forensic trail.
            await self._emit_admin_audit(
                action=AuditAction.SYSTEM_ADMIN_REVOKED,
                user=user,
                actor_id=actor_id,
                details={"changed": False},
            )
            return UserInfo.map_from_db(user)
        admin_count = len(
            await self._users_repo.find_all_by_column_values(
                {"is_system_admin": True},
            )
        )
        if admin_count <= 1:
            raise ValidationError(
                code="system.last_system_admin",
                message="Cannot revoke the last remaining system administrator",
            )
        user = await self._set_is_system_admin(user, False)
        await self._emit_admin_audit(
            action=AuditAction.SYSTEM_ADMIN_REVOKED,
            user=user,
            actor_id=actor_id,
            details={"changed": True},
        )
        return UserInfo.map_from_db(user)

    async def list_system_admins(
        self,
        *,
        offset: int,
        limit: int,
    ) -> tuple[list[UserInfo], int]:
        """Paginated list of system administrators (oldest first).

        Mirrors the upstream ``ListSystemAdmins``: returns the total
        count plus the requested page, newest first.
        """
        admins = await self._users_repo.find_all_by_column_values(
            {"is_system_admin": True},
        )
        total = len(admins)
        page = admins[offset : offset + limit]
        return [UserInfo.map_from_db(u) for u in page], total

    # ── Password reset ──────────────────────────────────────────────

    async def reset_password(
        self,
        *,
        email: str,
        new_password: str,
        actor_id: str,
    ) -> None:
        """Replace another user's password and revoke their sessions.

        The cannot-reset-self rule and the password policy are enforced
        here; the caller (the web layer) already gated on system admin.
        """
        validate_password_policy(new_password)
        user = await self._find_user(email=email)
        if user.id == actor_id:
            raise ValidationError(
                code="system.cannot_reset_self",
                message="Cannot reset your own password here",
            )
        updated = await self._users_repo.update_by_primary_key(
            {"id": user.id},
            {
                "password_hash": hash_password(new_password),
                "updated_at": datetime.now(UTC),
            },
        )
        if updated is None:
            raise NotFoundError(code="user.not_found", message="User not found")
        # A stolen token must not survive a password rotation.
        await self._tokens_repo.revoke_all_for_user(user.id)
        await self._emit_admin_audit(
            action=AuditAction.SYSTEM_USER_PASSWORD_RESET,
            user=user,
            actor_id=actor_id,
            details={"sessions_revoked": True},
        )

    # ── Internal helpers ────────────────────────────────────────────

    async def _find_user(self, *, user_id: str = "", email: str = "") -> User:
        """Resolve a user by id (priority) or email; both funnel to one 404."""
        try:
            if user_id:
                return await self._users_repo.find_by_id(
                    user_id,
                    not_found_code="user.not_found",
                    not_found_message="User not found",
                )
            return await self._users_repo.find_by_email(email)
        except NotFoundError as exc:
            # Normalise the message so a missing id and a missing email
            # look identical to the caller.
            raise NotFoundError(
                code="user.not_found",
                message="User not found",
            ) from exc

    async def _set_is_system_admin(self, user: User, value: bool) -> User:
        updated = await self._users_repo.update_by_primary_key(
            {"id": user.id},
            {
                "is_system_admin": value,
                "updated_at": datetime.now(UTC),
            },
        )
        if updated is None:
            raise NotFoundError(code="user.not_found", message="User not found")
        return updated

    async def _emit_admin_audit(
        self,
        *,
        action: str,
        user: User,
        actor_id: str,
        details: JsonObject,
    ) -> None:
        """Write a system-scope audit row (``tenant_id=0``)."""
        await self._audit_repo.create(
            AuditLog(
                id=0,
                tenant_id=0,
                actor_user_id=actor_id,
                actor_role="system_admin",
                action=action,
                target_type="user",
                target_id=user.id,
                target_user_id=user.id,
                outcome=AuditOutcome.SUCCESS,
                details={
                    "target_email": user.email,
                    "target_username": user.username,
                    **details,
                },
                created_at=datetime.now(UTC),
            )
        )


__all__ = [
    "SystemAdminService",
    "validate_password_policy",
]
