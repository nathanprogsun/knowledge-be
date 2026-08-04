"""HTTP middleware for the knowledge-be API.

Modules:

- ``rbac`` — tenant-scoped role gates (RequireRole / RequireSystemAdmin /
  RequireOwnershipOrRole).
- ``api_key_gate`` — route-policy-based API-key authorization.
- ``kb_access`` — KB membership / share-fallback guard (stub).
- ``embed_auth`` — embed-channel publish-token authentication (stub).
- ``audit`` — injects the AuditLogService into the request context.
"""

from __future__ import annotations

__all__: list[str] = []
