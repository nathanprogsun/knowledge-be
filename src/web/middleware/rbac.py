"""RBAC middleware — tenant-scoped role gates.

Three guard types that gate endpoints by the caller's tenant role
or system-admin flag.

- :func:`require_role` — minimum tenant role; API-key principals
  short-circuit (authorized by the API Key Gate).
- :func:`require_system_admin` — platform-wide admin flag; always
  enforced (not subject to the RBAC rollout switch).
- :func:`require_ownership_or_role` — role **or** resource-creator
  match (the lookup closure is handler-specific).

All three raise ``PermissionDeniedError`` (403) on failure and emit a
durable audit row via the injected ``AuditLogService`` (when available),
subject to the 1-minute sliding-window dedup inside the service.

The RBAC enforcement flag (``tenant.rbac_enforced``) controls whether
the guards reject or log-only. When false the guards log but do not
reject. The system-admin guard is always enforced regardless of the
flag.
"""

from __future__ import annotations

import contextlib
from collections.abc import Awaitable, Callable

from fastapi import Request

from src.common.exception import PermissionDeniedError
from src.core.auth.permissions import TenantRole
from src.core.system.audit_actions import AuditAction
from src.core.system.audit_service import AuditLogService
from src.web.middleware.context import (
    get_api_key_scope,
    get_is_system_admin,
    get_tenant_role,
    get_user_info,
)

# Type alias for the creator-lookup closure used by ownership-or-role.
CreatorLookup = Callable[[Request], Awaitable[tuple[str, Exception | None]]]


def _actor_user_id(request: Request) -> str:
    """Read the authenticated user's id from ``request.state``."""
    info = get_user_info(request)
    if info is not None:
        return info.get("id", "")
    return getattr(request.state, "user_id", "") or ""


def _tenant_id(request: Request) -> int:
    """Read the active tenant id from ``request.state``."""
    raw = getattr(request.state, "tenant_id", None)
    if raw is None or raw == "":
        return 0
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


async def _emit_denied_audit(
    *,
    audit_svc: AuditLogService,
    tenant_id: int,
    actor_user_id: str,
    actor_role: str,
    required_role: str,
    request_path: str,
    request_method: str,
) -> None:
    """Emit a durable audit row for a denied request.

    Subject to 1-minute dedup inside the service, so a probing client
    cannot flood the table. Failures are swallowed — audit must never
    break the underlying business operation.
    """
    with contextlib.suppress(Exception):
        await audit_svc.log_denied(
            tenant_id=tenant_id,
            actor_user_id=actor_user_id,
            actor_role=actor_role,
            action=AuditAction.ACCESS_DENIED,
            request_path=request_path,
            request_method=request_method,
        )


async def require_role(
    *,
    min_role: str,
    request: Request,
    audit_svc: AuditLogService | None,
) -> None:
    """Gate: caller's tenant role must be at least ``min_role``.

    API-key principals short-circuit (authorized by the API Key Gate).
    Cross-tenant superusers bypass the role check. When RBAC
    enforcement is off, the guard logs but does not reject.
    """
    # API-key principals are authorized solely by the APIKeyGate.
    if get_api_key_scope(request) is not None:
        return

    role = get_tenant_role(request)
    if TenantRole.has_permission(role, min_role):
        return

    # Cross-tenant superuser bypass.
    if get_is_system_admin(request):
        return

    actor_id = _actor_user_id(request)
    tenant_id = _tenant_id(request)

    # Fail-open during rollout (enforcement off).
    if audit_svc is not None:
        await _emit_denied_audit(
            audit_svc=audit_svc,
            tenant_id=tenant_id,
            actor_user_id=actor_id,
            actor_role=role,
            required_role=min_role,
            request_path=request.url.path,
            request_method=request.method,
        )
    raise PermissionDeniedError(
        code="rbac.insufficient_role",
        message=f"Forbidden: requires role '{min_role}' or higher",
    )


async def require_system_admin(
    *,
    request: Request,
    audit_svc: AuditLogService | None,
) -> None:
    """Gate: caller must be a system administrator.

    Always enforced (not subject to the RBAC rollout switch). API-key
    principals are admitted only if the scope is platform-level.
    """
    # Platform-scope API keys pass; workspace-bound keys are denied.
    scope = get_api_key_scope(request)
    if scope is not None:
        if scope.is_platform():
            return
        raise PermissionDeniedError(
            code="rbac.api_key_not_system_admin",
            message="Forbidden: API keys cannot access this endpoint",
        )

    if get_is_system_admin(request):
        return

    actor_id = _actor_user_id(request)
    tenant_id = _tenant_id(request)
    if audit_svc is not None:
        await _emit_denied_audit(
            audit_svc=audit_svc,
            tenant_id=tenant_id,
            actor_user_id=actor_id,
            actor_role="user",
            required_role="system_admin",
            request_path=request.url.path,
            request_method=request.method,
        )
    raise PermissionDeniedError(
        code="rbac.system_admin_required",
        message="Forbidden: system administrator required",
    )


async def require_ownership_or_role(
    *,
    min_role: str,
    lookup: CreatorLookup,
    request: Request,
    audit_svc: AuditLogService | None,
) -> None:
    """Gate: role ≥ min_role **or** caller is the resource creator.

    Decision order:
    1. role ≥ min → allow (no lookup).
    2. cross-tenant superuser → allow (no lookup).
    3. API-key principal → allow (authorized by APIKeyGate).
    4. lookup → ownership match → allow.
    5. otherwise → 403 + audit.
    """
    # API-key principals are authorized solely by the APIKeyGate.
    if get_api_key_scope(request) is not None:
        return

    role = get_tenant_role(request)
    if TenantRole.has_permission(role, min_role):
        return

    if get_is_system_admin(request):
        return

    creator_id, lookup_err = await lookup(request)
    if lookup_err is not None:
        # Resource not found → pass through so handler issues 404.
        # Transient failure → 503 (handled by caller).
        if isinstance(lookup_err, ResourceNotFoundError):
            return
        raise lookup_err

    actor_id = _actor_user_id(request)
    if creator_id and creator_id == actor_id:
        return

    tenant_id = _tenant_id(request)
    if audit_svc is not None:
        await _emit_denied_audit(
            audit_svc=audit_svc,
            tenant_id=tenant_id,
            actor_user_id=actor_id,
            actor_role=role,
            required_role=min_role,
            request_path=request.url.path,
            request_method=request.method,
        )
    raise PermissionDeniedError(
        code="rbac.ownership_or_role_required",
        message="Forbidden: must own the resource or have the required role",
    )


class ResourceNotFoundError(Exception):
    """Sentinel returned by a CreatorLookup when the :id matches no row.

    The middleware lets the request proceed to the handler so the
    handler can issue its own 404 rather than masking a missing
    resource as a permissions error.
    """


__all__ = [
    "CreatorLookup",
    "ResourceNotFoundError",
    "require_ownership_or_role",
    "require_role",
    "require_system_admin",
]
