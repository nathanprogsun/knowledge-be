"""Mapping of ``ChatOptions.thinking`` onto provider HTTP request fields.

``ThinkingStrategy`` is the interface: ``apply`` returns ``(custom_body,
use_raw_http)``. A ``None`` body means "send the standard request unchanged"
(no raw HTTP required); a non-``None`` body must be sent verbatim over raw
HTTP because it carries fields the SDK would otherwise strip.

The strategies mirror the reference implementation:

- ``NoThinking`` sends no thinking fields at all.
- ``EnableThinking`` encodes thinking via Qwen's ``enable_thinking`` boolean,
  optionally pinning it on every request and/or forcing it off for
  non-stream calls.
- ``ThinkingTypeField`` encodes thinking via ``{ "thinking": { "type": ... } }``
  (LKEAP / Volcengine style providers).
- ``ChatTemplateKwargs`` encodes thinking via the standard request's
  ``chat_template_kwargs.enable_thinking`` (vLLM / NVIDIA / generic local
  deployments).

``parse_thinking_override`` reads the ``thinking_control`` extra-config key the
frontend writes; an unrecognized non-empty value falls back to
``ChatTemplateKwargs``, preserving the legacy default-mode behavior.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.ai.llm.types import ChatOptions
from src.common.json import JsonObject

#: ``parameters.extra_config`` key selecting how thinking is translated.
EXTRA_CONFIG_THINKING_CONTROL = "thinking_control"


@runtime_checkable
class ThinkingStrategy(Protocol):
    """Encodes ``ChatOptions.Thinking`` for one provider's wire format."""

    def apply(
        self,
        req: JsonObject,
        opts: ChatOptions | None,
        is_stream: bool,
    ) -> tuple[JsonObject | None, bool]:
        """Return ``(custom_body, use_raw_http)`` for ``req``.

        A ``None`` body keeps the standard request; a non-``None`` body must
        be sent verbatim over raw HTTP.
        """
        ...


class NoThinking:
    """Sends no thinking-related fields at all."""

    def apply(
        self, req: JsonObject, opts: ChatOptions | None, is_stream: bool
    ) -> tuple[JsonObject | None, bool]:
        return None, False


class EnableThinking:
    """Encodes thinking via Qwen's ``enable_thinking`` boolean.

    - ``always_send`` pins the field even when ``opts.thinking`` is unset
      (Aliyun Qwen thinking models require it on every request).
    - ``disable_on_non_stream`` forces ``enable_thinking=False`` for non-stream
      requests (Qwen3 rejects thinking in non-stream mode).
    """

    def __init__(
        self, *, always_send: bool = False, disable_on_non_stream: bool = False
    ) -> None:
        self._always_send = always_send
        self._disable_on_non_stream = disable_on_non_stream

    def apply(
        self, req: JsonObject, opts: ChatOptions | None, is_stream: bool
    ) -> tuple[JsonObject | None, bool]:
        thinking = False
        if opts is not None and opts.thinking is not None:
            thinking = opts.thinking
        elif not self._always_send:
            return None, False
        if self._disable_on_non_stream and not is_stream:
            thinking = False
        custom = dict(req)
        custom["enable_thinking"] = thinking
        return custom, True


class ThinkingTypeField:
    """Encodes thinking via ``{ "thinking": { "type": "enabled"|"disabled" } }``."""

    def apply(
        self, req: JsonObject, opts: ChatOptions | None, is_stream: bool
    ) -> tuple[JsonObject | None, bool]:
        if opts is None or opts.thinking is None:
            return None, False
        custom = dict(req)
        custom["thinking"] = {"type": "enabled" if opts.thinking else "disabled"}
        return custom, True


class ChatTemplateKwargs:
    """Encodes thinking via ``chat_template_kwargs.enable_thinking``."""

    def apply(
        self, req: JsonObject, opts: ChatOptions | None, is_stream: bool
    ) -> tuple[JsonObject | None, bool]:
        if opts is None or opts.thinking is None:
            return None, False
        req["chat_template_kwargs"] = {"enable_thinking": opts.thinking}
        return req, True


def parse_thinking_override(
    extra_config: dict[str, str] | None,
) -> ThinkingStrategy | None:
    """Return the strategy selected by ``thinking_control``, or ``None``.

    An unrecognized non-empty value falls back to ``ChatTemplateKwargs``.
    """
    if not extra_config:
        return None
    value = (extra_config.get(EXTRA_CONFIG_THINKING_CONTROL) or "").strip().lower()
    if value == "":
        return None
    if value == "none":
        return NoThinking()
    if value == "enable_thinking":
        return EnableThinking()
    if value == "thinking_type":
        return ThinkingTypeField()
    return ChatTemplateKwargs()


def thinking_strategy_name(strategy: ThinkingStrategy) -> str:
    """Return the frontend label for ``strategy`` (``"none"`` fallback)."""
    if isinstance(strategy, EnableThinking):
        return "enable_thinking"
    if isinstance(strategy, ThinkingTypeField):
        return "thinking_type"
    if isinstance(strategy, ChatTemplateKwargs):
        return "chat_template_kwargs"
    return "none"


__all__ = [
    "EXTRA_CONFIG_THINKING_CONTROL",
    "ChatTemplateKwargs",
    "EnableThinking",
    "NoThinking",
    "ThinkingStrategy",
    "ThinkingTypeField",
    "parse_thinking_override",
    "thinking_strategy_name",
]
