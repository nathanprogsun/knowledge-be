"""Audit middleware — inject AuditLogService into the request state.

Maps ``internal/middleware/audit_provider.go``. The middleware stashes
the ``AuditLogService`` on ``request.state`` so RBAC guards can pull it
out without needing the service threaded into their signatures.

Wiring is centralised in the app factory so each request gets the same
instance for the lifetime of the process. The middleware is a no-op
when ``svc`` is ``None`` (e.g. lite mode where audit isn't configured)
so the RBAC reject path degrades gracefully.
"""

from __future__ import annotations

from fastapi import Request

from src.core.system.audit_service import AuditLogService


def set_audit_service(request: Request, svc: AuditLogService | None) -> None:
    """Stash the audit service on ``request.state``.

    Called by the audit middleware on every request. RBAC guards read
    it via :func:`get_audit_service`.
    """
    request.state.audit_service = svc


def get_audit_service(request: Request) -> AuditLogService | None:
    """Return the audit service stashed by the audit middleware.

    Returns ``None`` when no provider was wired upstream. Callers MUST
    nil-check before invoking — audit failure must never break the
    underlying business operation.
    """
    return getattr(request.state, "audit_service", None)


async def audit_provider(
    *,
    request: Request,
    audit_svc: AuditLogService | None,
) -> None:
    """Inject the audit service into the request state.

    This is a FastAPI dependency (not ASGI middleware) so it runs per
    request and can access the resolved ``AuditLogService``. When
    ``audit_svc`` is ``None`` the middleware is a no-op.
    """
    set_audit_service(request, audit_svc)


__all__ = [
    "audit_provider",
    "get_audit_service",
    "set_audit_service",
]
