"""Embed Auth middleware — embed-channel publish-token authentication.

Maps ``internal/middleware/embed_auth.go``. Validates embed publish
tokens and injects a scoped tenant context for embed routes
(``/embed/{channel_id}/...``). Includes per-IP and global rate limiting
via Redis.

PR-12 scope: **stub**. The full implementation depends on
``EmbedChannelService`` (PR-120, stage 7), ``TenantService`` (already
available), and a Redis-backed rate limiter (PR-129, stage 8). Until
those land, this module provides only the data structures and a
placeholder guard that raises ``NotImplementedError``.
"""

from __future__ import annotations

from fastapi import Request


async def embed_auth(
    *,
    request: Request,
) -> str:
    """Gate: validate embed publish token and return the channel id.

    PR-12: stub — raises ``NotImplementedError``. The full validation
    (token lookup → channel resolution → tenant context injection →
    rate limit) lands in stage 7 alongside the EmbedChannel domain.
    """
    raise NotImplementedError(
        "Embed auth guard is a stub in PR-12; it will be implemented "
        "in stage 7 (EmbedChannel domain)."
    )


__all__ = ["embed_auth"]
