"""Tests for the chat factory: config mapping and source routing."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.ai.llm.anthropic import AnthropicChat
from src.ai.llm.base import (
    MODEL_SOURCE_LOCAL,
    MODEL_SOURCE_REMOTE,
    config_from_model,
    new_chat,
    new_remote_chat,
)
from src.ai.llm.concurrency import ConcurrencyChat
from src.ai.llm.remote_api import RemoteAPIChat
from src.ai.llm.types import Chat, ChatConfig
from src.core.contracts.infra import Model, ModelParameters


def _model(
    *,
    source: str = MODEL_SOURCE_REMOTE,
    provider: str = "deepseek",
    name: str = "deepseek-chat",
    base_url: str = "",
    max_concurrency: int = 0,
    extra_config: dict[str, str] | None = None,
) -> Model:
    return Model(
        id="model-1",
        tenant_id=1,
        name=name,
        type="chat",
        source=source,
        parameters=ModelParameters(
            provider=provider,
            base_url=base_url,
            api_key="sk-123",
            max_concurrency=max_concurrency,
            extra_config=extra_config,
            custom_headers={"X-Trace": "t1"},
        ),
        created_at=datetime(2024, 1, 1, tzinfo=UTC),
        updated_at=datetime(2024, 1, 1, tzinfo=UTC),
    )


@pytest.fixture(autouse=True)
def _ssrf_whitelist(monkeypatch: pytest.MonkeyPatch) -> None:
    # Lets the SSRF validation accept the fake endpoint without real DNS.
    monkeypatch.setenv("SSRF_WHITELIST", "llm.test")


# ── config_from_model ────────────────────────────────────────────────


def test_config_from_model_maps_all_fields() -> None:
    model = _model(
        source=MODEL_SOURCE_REMOTE,
        provider="openai",
        name="gpt-4o",
        max_concurrency=3,
        extra_config={"api_version": "2024-01-01"},
    )
    config = config_from_model(model, app_id="app-1", app_secret="secret-1")
    assert config is not None
    assert config.model_id == "model-1"
    assert config.api_key == "sk-123"
    assert config.base_url == ""
    assert config.model_name == "gpt-4o"
    assert config.source == MODEL_SOURCE_REMOTE
    assert config.provider == "openai"
    assert config.max_concurrency == 3
    assert config.extra_config == {"api_version": "2024-01-01"}
    assert config.custom_headers == {"X-Trace": "t1"}
    assert config.app_id == "app-1"
    assert config.app_secret == "secret-1"


def test_config_from_model_none() -> None:
    assert config_from_model(None) is None


# ── new_remote_chat ──────────────────────────────────────────────────


def test_new_remote_chat_returns_remote_api_chat() -> None:
    chat = new_remote_chat(
        ChatConfig(source=MODEL_SOURCE_REMOTE, provider="deepseek", model_name="deepseek-chat")
    )
    assert isinstance(chat, RemoteAPIChat)
    assert chat.get_model_name() == "deepseek-chat"


def test_new_remote_chat_anthropic_returns_anthropic_chat() -> None:
    chat = new_remote_chat(
        ChatConfig(
            source=MODEL_SOURCE_REMOTE,
            provider="anthropic",
            model_name="claude-3-5-sonnet",
            api_key="sk-ant-test",
        )
    )
    assert isinstance(chat, AnthropicChat)
    assert chat.get_model_name() == "claude-3-5-sonnet"


def test_new_remote_chat_anthropic_requires_api_key() -> None:
    with pytest.raises(ValueError, match="API key"):
        new_remote_chat(
            ChatConfig(source=MODEL_SOURCE_REMOTE, provider="anthropic", model_name="claude")
        )


def test_new_remote_chat_detects_provider_from_url() -> None:
    chat = new_remote_chat(
        ChatConfig(
            source=MODEL_SOURCE_REMOTE,
            base_url="http://llm.test",
            model_name="m",
        )
    )
    assert isinstance(chat, RemoteAPIChat)
    # No provider configured and no known host, so detection falls back to generic.
    assert chat.get_provider() == "generic"


# ── new_chat ─────────────────────────────────────────────────────────


def test_new_chat_routes_remote_and_wraps_concurrency() -> None:
    chat = new_chat(
        ChatConfig(
            source=MODEL_SOURCE_REMOTE,
            provider="deepseek",
            model_name="deepseek-chat",
            max_concurrency=2,
        )
    )
    assert isinstance(chat, ConcurrencyChat)
    assert chat.get_model_id() == ""
    assert chat.get_model_name() == "deepseek-chat"


def test_new_chat_anthropic_wraps_concurrency() -> None:
    chat = new_chat(
        ChatConfig(
            source=MODEL_SOURCE_REMOTE,
            provider="anthropic",
            model_name="claude-3-5-sonnet",
            api_key="sk-ant-test",
            max_concurrency=2,
        )
    )
    assert isinstance(chat, ConcurrencyChat)
    assert isinstance(chat._inner, AnthropicChat)
    assert chat.get_model_name() == "claude-3-5-sonnet"


def test_new_chat_local_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="Ollama"):
        new_chat(ChatConfig(source=MODEL_SOURCE_LOCAL, model_name="qwen2"))


def test_new_chat_unsupported_source() -> None:
    with pytest.raises(ValueError, match="unsupported chat model source"):
        new_chat(ChatConfig(source="bogus", model_name="m"))


def test_new_chat_returns_chat_protocol() -> None:
    chat = new_chat(ChatConfig(source=MODEL_SOURCE_REMOTE, provider="deepseek", model_name="m"))
    assert isinstance(chat, Chat)
