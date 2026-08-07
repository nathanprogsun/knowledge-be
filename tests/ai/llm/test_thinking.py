"""Tests for thinking-strategy request shaping."""

from __future__ import annotations

from src.ai.llm.thinking import (
    ChatTemplateKwargs,
    EnableThinking,
    NoThinking,
    ThinkingTypeField,
    parse_thinking_override,
    thinking_strategy_name,
)
from src.ai.llm.types import ChatOptions
from src.common.json import JsonObject


def _opts(thinking: bool | None = None) -> ChatOptions:
    return ChatOptions(thinking=thinking)


# ── noThinking ───────────────────────────────────────────────────────


def test_no_thinking_returns_standard_request() -> None:
    strategy = NoThinking()
    body, use_raw = strategy.apply({"model": "m"}, None, False)
    assert body is None
    assert use_raw is False


# ── enableThinking ───────────────────────────────────────────────────


def test_enable_thinking_emits_nothing_when_unset_and_not_forced() -> None:
    strategy = EnableThinking()
    body, use_raw = strategy.apply({"model": "m"}, None, False)
    assert body is None
    assert use_raw is False


def test_enable_thinking_true() -> None:
    strategy = EnableThinking()
    body, use_raw = strategy.apply({"model": "m"}, _opts(True), True)
    assert body == {"model": "m", "enable_thinking": True}
    assert use_raw is True


def test_enable_thinking_false() -> None:
    strategy = EnableThinking()
    body, _use_raw = strategy.apply({"model": "m"}, _opts(False), True)
    assert body == {"model": "m", "enable_thinking": False}


def test_enable_thinking_always_send_pins_when_unset() -> None:
    strategy = EnableThinking(always_send=True)
    body, _use_raw = strategy.apply({"model": "m"}, None, True)
    assert body == {"model": "m", "enable_thinking": False}


def test_enable_thinking_disabled_on_non_stream() -> None:
    strategy = EnableThinking(always_send=True, disable_on_non_stream=True)
    body, _use_raw = strategy.apply({"model": "m"}, _opts(True), False)
    assert body == {"model": "m", "enable_thinking": False}
    stream_body, _ = strategy.apply({"model": "m"}, _opts(True), True)
    assert stream_body == {"model": "m", "enable_thinking": True}


# ── thinkingTypeField ────────────────────────────────────────────────


def test_thinking_type_field_emits_nothing_when_unset() -> None:
    strategy = ThinkingTypeField()
    body, use_raw = strategy.apply({"model": "m"}, None, False)
    assert body is None
    assert use_raw is False


def test_thinking_type_field_enabled() -> None:
    strategy = ThinkingTypeField()
    body, use_raw = strategy.apply({"model": "m"}, _opts(True), True)
    assert body == {"model": "m", "thinking": {"type": "enabled"}}
    assert use_raw is True


def test_thinking_type_field_disabled() -> None:
    strategy = ThinkingTypeField()
    body, _ = strategy.apply({"model": "m"}, _opts(False), True)
    assert body == {"model": "m", "thinking": {"type": "disabled"}}


# ── chatTemplateKwargs ───────────────────────────────────────────────


def test_chat_template_kwargs_emits_nothing_when_unset() -> None:
    strategy = ChatTemplateKwargs()
    body, use_raw = strategy.apply({"model": "m"}, None, False)
    assert body is None
    assert use_raw is False


def test_chat_template_kwargs_enabled() -> None:
    strategy = ChatTemplateKwargs()
    req: JsonObject = {"model": "m"}
    body, use_raw = strategy.apply(req, _opts(True), True)
    assert body is req
    assert body == {"model": "m", "chat_template_kwargs": {"enable_thinking": True}}
    assert use_raw is True


# ── parseThinkingOverride ────────────────────────────────────────────


def test_parse_thinking_override_none_when_unset() -> None:
    assert parse_thinking_override(None) is None
    assert parse_thinking_override({}) is None
    assert parse_thinking_override({"thinking_control": ""}) is None


def test_parse_thinking_override_none() -> None:
    strategy = parse_thinking_override({"thinking_control": "none"})
    assert isinstance(strategy, NoThinking)


def test_parse_thinking_override_enable_thinking() -> None:
    strategy = parse_thinking_override({"thinking_control": "enable_thinking"})
    assert isinstance(strategy, EnableThinking)


def test_parse_thinking_override_thinking_type() -> None:
    strategy = parse_thinking_override({"thinking_control": "thinking_type"})
    assert isinstance(strategy, ThinkingTypeField)


def test_parse_thinking_override_unknown_falls_back_to_template() -> None:
    strategy = parse_thinking_override({"thinking_control": "bogus"})
    assert isinstance(strategy, ChatTemplateKwargs)


def test_parse_thinking_override_is_case_insensitive() -> None:
    strategy = parse_thinking_override({"thinking_control": "  Enable_Thinking "})
    assert isinstance(strategy, EnableThinking)


# ── thinkingStrategyName ─────────────────────────────────────────────


def test_thinking_strategy_name() -> None:
    assert thinking_strategy_name(NoThinking()) == "none"
    assert thinking_strategy_name(EnableThinking()) == "enable_thinking"
    assert thinking_strategy_name(ThinkingTypeField()) == "thinking_type"
    assert thinking_strategy_name(ChatTemplateKwargs()) == "chat_template_kwargs"
