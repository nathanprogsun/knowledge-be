"""KB Access middleware — KB membership / share-fallback guard.

The guard resolves whether the caller may access a specific
knowledge base:

1. KB belongs to caller's tenant → grant own access.
2. Org-shared KB → grant min(share, role) cap.
3. Shared agent carries the KB → grant Viewer (read-only).

**Stub**: the full implementation depends on the KnowledgeBase,
Organization, and AgentShare domains. Until those are available, this
module provides only the data structures and a placeholder guard that
raises ``NotImplementedError``.

The ``KBAccess`` data class and ``KBIDResolver`` type alias are
defined now so future code can reference them without redefining the
contract.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from fastapi import Request


@dataclass(frozen=True, slots=True)
class KBAccess:
    """Result of a successful KB access resolution.

    Stashed on ``request.state`` so handlers that need the resolved KB
    or permission can pull it without re-running the resolution.
    """

    knowledge_base_id: str
    effective_tenant_id: int
    permission: str


# Resolver: reads the KB id from the request (URL param, query, body).
KBIDResolver = Callable[[Request], Awaitable[str]]


async def require_kb_access(
    *,
    resolver: KBIDResolver,
    request: Request,
) -> KBAccess:
    """Gate: resolve and check KB access for the caller.

    Stub — raises ``NotImplementedError``. The full resolution (tenant
    ownership → org share → agent share) will be implemented alongside
    the KnowledgeBase domain.
    """
    raise NotImplementedError(
        "KB access guard is not yet implemented; it requires the KnowledgeBase domain service."
    )


__all__ = ["KBAccess", "KBIDResolver", "require_kb_access"]
