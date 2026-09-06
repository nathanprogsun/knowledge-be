"""ARQ worker settings — env-driven configuration for the background pool.

Loaded from environment variables prefixed with ``WORKER_`` (plus the
optional ``.env`` file). Read via ``get_worker_settings()``, which
memoizes a process-wide singleton via ``functools.lru_cache`` (no
module-level mutable globals), mirroring ``src.settings``.

``redis_settings`` derives an :class:`arq.connections.RedisSettings`
from ``redis_url`` and applies the pool-tuning knobs
(``max_connections``, ``conn_timeout``, ``conn_retries``,
``conn_retry_delay``).
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

from arq.connections import RedisSettings
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.common.arq_redis import redis_settings_from_url
from src.settings import get_settings

_DEFAULT_REDIS_URL = "redis://localhost:6379"


class WorkerSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="WORKER_",
        extra="ignore",
        case_sensitive=False,
    )

    # Redis connection.
    redis_url: str = _DEFAULT_REDIS_URL
    # Pool tuning applied on top of the parsed URL.
    max_connections: int = 10
    conn_timeout: int = 1
    conn_retries: int = 5
    conn_retry_delay: int = 1

    # Queue + job execution.
    queue_name: str = "arq:queue"
    max_jobs: int = 10
    job_timeout: int = 300
    max_tries: int = 5
    poll_delay: float = 0.5
    keep_result: int = 3600

    # Health + process behaviour.
    health_check_interval: int = 3600
    burst: bool = False
    handle_signals: bool = True
    log_results: bool = True

    @property
    def redis_settings(self) -> RedisSettings:
        """arq RedisSettings derived from ``redis_url`` plus pool tuning."""
        base = redis_settings_from_url(self.redis_url)
        return replace(
            base,
            max_connections=self.max_connections,
            conn_timeout=self.conn_timeout,
            conn_retries=self.conn_retries,
            conn_retry_delay=self.conn_retry_delay,
        )


@lru_cache(maxsize=1)
def get_worker_settings() -> WorkerSettings:
    """Return the process-wide WorkerSettings singleton.

    Lazy-initialized on first call and cached via ``lru_cache``. Tests
    can reset with ``reset_worker_settings_cache()``.
    """
    settings = WorkerSettings()
    if settings.redis_url != _DEFAULT_REDIS_URL:
        return settings
    app_redis = get_settings().redis_url
    if not app_redis or app_redis == settings.redis_url:
        return settings
    return settings.model_copy(update={"redis_url": app_redis})


def reset_worker_settings_cache() -> None:
    """Drop the memoized singleton. Used by tests that mutate env vars."""
    get_worker_settings.cache_clear()


__all__ = [
    "WorkerSettings",
    "get_worker_settings",
    "reset_worker_settings_cache",
]
