"""Unit tests for passworded Redis DSN normalization."""

from __future__ import annotations

from src.common.arq_redis import redis_settings_from_url


def test_password_only_url_drops_empty_username_and_unquotes() -> None:
    settings = redis_settings_from_url("redis://:redis123%21%40%23@localhost:6379/0")
    assert settings.host == "localhost"
    assert settings.port == 6379
    assert settings.database == 0
    assert settings.username is None
    assert settings.password == "redis123!@#"


def test_username_and_password_round_trip() -> None:
    settings = redis_settings_from_url("redis://user:pass@cache:6380/3")
    assert settings.username == "user"
    assert settings.password == "pass"
    assert settings.host == "cache"
    assert settings.port == 6380
    assert settings.database == 3
