"""Embed Auth middleware — embed-channel publish-token authentication.

Validates embed publish tokens and injects a scoped tenant context
for embed routes (``/embed/{channel_id}/...``). Includes per-IP and
global rate limiting via Redis.

**Stub**: the full implementation depends on ``EmbedChannelService``
and a Redis-backed rate limiter. Until those are available, this module
provides only a placeholder guard that raises ``NotImplementedError``.
"""

from __future__ import annotations

from fastapi import Request


async def embed_auth(
    *,
    request: Request,
) -> str:
    """Gate: validate embed publish token and return the channel id.

    Stub — raises ``NotImplementedError``. The full validation (token
    lookup → channel resolution → tenant context injection → rate limit)
    will be implemented alongside the EmbedChannel domain.
    """
    raise NotImplementedError(
        "Embed auth guard is not yet implemented; it requires the EmbedChannel domain service."
    )


__all__ = ["embed_auth"]
