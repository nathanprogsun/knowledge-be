"""ARQ RedisSettings helpers that keep passworded DSNs usable.

``RedisSettings.from_dsn`` keeps an empty username as ``""`` and leaves
percent-encoding on the password. Redis then rejects AUTH against a
password-only URL such as ``redis://:secret@localhost:6379/0``.
"""

from __future__ import annotations

from dataclasses import replace
from urllib.parse import unquote

from arq.connections import RedisSettings


def redis_settings_from_url(redis_url: str) -> RedisSettings:
    """Parse ``redis_url`` and normalize username / password for AUTH."""
    parsed = RedisSettings.from_dsn(redis_url)
    password = unquote(parsed.password) if parsed.password else None
    return replace(
        parsed,
        username=parsed.username or None,
        password=password,
    )


__all__ = ["redis_settings_from_url"]
