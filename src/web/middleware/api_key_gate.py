"""API Key Gate middleware — route-policy-based API-key authorization.

API-key authorization is
a **separate authority** from the JWT role/ownership guards. Every
API-key-accessible route declares one policy in the
``APIKeyRouteAuthorizer`` registry; the gate is the single place that
enforces it. Routes that declare no policy are denied for API keys by
default (fail-closed).

JWT principals pass straight through — the gate only authorizes
requests carrying an ``X-API-Key`` header that the auth middleware has
already resolved into a ``TenantAPIKeyScope`` on ``request.state``.

The authorizer is populated at router-construction time (single
threaded) and only read at request time, so it needs no locking.
"""

from __future__ import annotations

from fastapi import Request

from src.common.exception import PermissionDeniedError
from src.core.auth.permissions import APIKeyRoutePolicy, TenantAPIKeyScope


class APIKeyRouteAuthorizer:
    """Registry of per-route API-key policies.

    Populated at router-construction time. Keyed by HTTP method, then
    by the FastAPI route path (e.g. ``"/api/v1/knowledge-bases/{id}"``).
    """

    def __init__(self) -> None:
        self._policies: dict[str, dict[str, APIKeyRoutePolicy]] = {}

    def register(
        self,
        method: str,
        path: str,
        policy: APIKeyRoutePolicy,
    ) -> None:
        """Record the policy for ``(method, path)``.

        ``path`` MUST be the FastAPI route template (e.g.
        ``"/api/v1/knowledge-bases/{id}"``), not a concrete URL.
        """
        method_upper = method.upper().strip()
        path_norm = _normalize_path(path)
        if method_upper not in self._policies:
            self._policies[method_upper] = {}
        self._policies[method_upper][path_norm] = policy

    def lookup(self, method: str, path: str) -> APIKeyRoutePolicy | None:
        """Return the policy for ``(method, path)``, or ``None``."""
        by_path = self._policies.get(method.upper())
        if by_path is None:
            return None
        return by_path.get(_normalize_path(path))

    def registered_routes(self) -> dict[str, list[str]]:
        """Return every ``(method, path)`` pair the authorizer knows."""
        result: dict[str, list[str]] = {}
        for method, by_path in self._policies.items():
            result[method] = sorted(by_path.keys())
        return result


async def api_key_gate(
    *,
    request: Request,
    authorizer: APIKeyRouteAuthorizer,
) -> None:
    """Gate: authorize API-key principals against the route policy.

    JWT principals (no ``api_key_scope`` on ``request.state``) pass
    through. API-key principals are authorized purely from the declared
    policy table; absent policy → default deny.
    """
    scope: TenantAPIKeyScope | None = getattr(request.state, "api_key_scope", None)
    if scope is None:
        return  # JWT principal — not our jurisdiction.

    policy = authorizer.lookup(request.method, request.url.path)
    if policy is None:
        raise PermissionDeniedError(
            code="api_key.route_not_allowed",
            message="Forbidden: API key scope does not allow this operation",
        )

    _authorize(scope, policy)


def _authorize(scope: TenantAPIKeyScope, policy: APIKeyRoutePolicy) -> None:
    """Apply the declared policy to an API-key scope.

    Absent policy → default deny (caller checks this before calling
    here). ``PlatformOnly`` rejects non-platform keys. Full-access keys
    pass unless ``PlatformOnly`` is set. Scoped keys pass when any
    declared capability matches.
    """
    if policy.platform_only and not scope.is_platform():
        raise PermissionDeniedError(
            code="api_key.platform_only",
            message="Forbidden: API key scope does not allow this operation",
        )

    if policy.platform_only and not policy.capabilities:
        raise PermissionDeniedError(
            code="api_key.platform_only_no_caps",
            message="Forbidden: API key scope does not allow this operation",
        )

    if scope.full_access:
        return

    for cap in policy.capabilities:
        if scope.has_capability(cap):
            return

    if not policy.require_full_access and not policy.capabilities:
        return

    raise PermissionDeniedError(
        code="api_key.scope_forbidden",
        message="Forbidden: API key scope does not allow this operation",
    )


def _normalize_path(path: str) -> str:
    """Collapse duplicate slashes and trim trailing slash."""
    if not path:
        return ""
    while "//" in path:
        path = path.replace("//", "/")
    if len(path) > 1:
        path = path.rstrip("/")
    return path


__all__ = ["APIKeyRouteAuthorizer", "api_key_gate"]
