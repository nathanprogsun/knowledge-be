"""Application settings — single source of truth for runtime configuration.

Loaded from environment variables prefixed `KNOWLEDGE_BE_` and an optional
`.env` file. Read via `get_settings()` which memoizes a process-wide
singleton via `functools.lru_cache` (no module-level mutable globals).
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):  # type: ignore[explicit-any]
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="KNOWLEDGE_BE_",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "knowledge-be"
    environment: str = "development"

    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/knowledge_be"
    redis_url: str = "redis://localhost:6379"

    jwt_secret_key: str = "change-me"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 10080
    refresh_token_expire_days: int = 7

    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_echo: bool = False
    db_conn_prewarm: bool = True

    cors_allow_origins: list[str] = ["*"]
    default_main_thread_pool_size: int = 40

    docreader_addr: str = "localhost:50051"
    docreader_transport: str = "grpc"

    system_aes_key: str = ""


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide Settings singleton.

    Lazy-initialized on first call and cached via `lru_cache`. Tests can
    reset with `reset_settings_cache()`.
    """
    return Settings()


def reset_settings_cache() -> None:
    """Drop the memoized singleton. Used by tests that mutate env vars."""
    get_settings.cache_clear()
