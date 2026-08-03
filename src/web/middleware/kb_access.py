"""KB Access middleware — KB membership / share-fallback guard.

Maps ``internal/middleware/kb_access.go``. The guard resolves whether
the caller may access a specific knowledge base:

1. KB belongs to caller's tenant → grant own access.
2. Org-shared KB → grant min(share, role) cap.
3. Shared agent carries the KB → grant Viewer (read-only).

PR-12 scope: **stub**. The full implementation depends on the
KnowledgeBase domain (stage 4), the Organization domain (stage 7), and
the AgentShare domain (stage 7). Until those land, this module
provides only the data structures and a placeholder guard that
raises ``NotImplementedError`` so callers know the guard is not yet
wired.

The ``KBAccess`` data class and ``KBIDResolver`` type alias are
defined now so downstream PRs can reference them without redefining the
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

    PR-12: stub — raises ``NotImplementedError``. The full resolution
    (tenant ownership → org share → agent share) lands in stage 4
    alongside the KnowledgeBase domain.
    """
    raise NotImplementedError(
        "KB access guard is a stub in PR-12; it will be implemented "
        "in stage 4 (KnowledgeBase domain)."
    )


__all__ = ["KBAccess", "KBIDResolver", "require_kb_access"]
