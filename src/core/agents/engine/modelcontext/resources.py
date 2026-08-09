"""Durable-resource half of the model-context registry.

Assigns request-local ``res://NNNN`` handles to stored resource references and
restores them before application code consumes model output. Wiki
``summary/<uuid>`` slugs share the exact failure mode the registry exists to
prevent — a high-entropy identifier the model must reproduce verbatim inside
``[[slug|display]]`` links — so they are aliased to the same low-entropy token.
Entity slugs (``entity/<readable-title>``) are low-entropy and semantically
meaningful, so they are deliberately left untouched.
"""

from __future__ import annotations

import re

from src.ai.llm.types import LLMToolCall, Message
from src.core.agents.engine.modelcontext.handle_table import HandleTable as _HandleTable

#: Also recognizes legacy physical references. New writes persist ``resource://``
#: handles, but old chunks and message history can still contain a provider URL;
#: giving both forms the same request-local handle makes rollout safe without a
#: blocking full-table content rewrite.
_STORED_REF_RE = re.compile(
    r"resource://[0-9A-Za-z_-]{22}"
    r"|(?:storage://[0-9A-Za-z_-]+/)?"
    r"(?:local|minio|cos|tos|s3|oss|ks3|obs)://[^\s)\]>\"']+"
    r"|summary/[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)

#: Matches the handle syntax produced by ``encode_text``. Used to spot
#: handle-shaped tokens the registry cannot map back (a hallucinated reference
#: or a coincidental collision) and by the stream-side orphan filter.
_RESOURCE_HANDLE_SHAPE_RE = re.compile(r"res://\d+")


class ResourceRegistry:
    """Request-local codec between stored resource references and handles.

    Safe to reuse across all rounds of one agent execution.
    """

    def __init__(self) -> None:
        self._table: _HandleTable[None] = _HandleTable[None]("res://", 4, 1)

    def encode_text(self, value: str) -> str:
        """Replace stored references with compact, stable handles."""
        if value == "":
            return value
        return _STORED_REF_RE.sub(
            lambda match: self._table.register(match.group(0), match.group(0), None, None),
            value,
        )

    def decode_text(self, value: str) -> str:
        """Restore every handle currently known to the registry.

        Handles are replaced longest-first with no word-boundary check — this
        plain substring behavior is load-bearing for handles adjacent to
        punctuation in Markdown.
        """
        if value == "":
            return value
        pairs = sorted(self._table.pairs(), key=lambda item: len(item.handle), reverse=True)
        for item in pairs:
            value = value.replace(item.handle, item.value)
        return value

    def strip_orphan_handles(self, value: str) -> str:
        """Remove handle-shaped tokens after all known handles are restored.

        Use only on model output; tool arguments must retain unknown handles
        long enough for the codec to reject the call.
        """
        if value == "":
            return value
        return _RESOURCE_HANDLE_SHAPE_RE.sub("", value)

    def encode_messages(self, messages: list[Message]) -> list[Message]:
        """Return a copied message list with textual references compacted.

        Binary/image content fields are intentionally left untouched.
        """
        if not messages:
            return messages
        out = list(messages)
        for i, message in enumerate(out):
            out[i] = message.model_copy(
                update={
                    "content": self.encode_text(message.content),
                    "reasoning_content": self.encode_text(message.reasoning_content),
                    "multi_content": [
                        part.model_copy(update={"text": self.encode_text(part.text)})
                        for part in message.multi_content
                    ],
                    "tool_calls": [
                        call.model_copy(
                            update={
                                "function": call.function.model_copy(
                                    update={"arguments": self.encode_text(call.function.arguments)}
                                )
                            }
                        )
                        for call in message.tool_calls
                    ],
                }
            )
        return out

    def decode_tool_calls(self, tool_calls: list[LLMToolCall]) -> None:
        """Restore handles in tool-call JSON arguments."""
        for call in tool_calls:
            call.function = call.function.model_copy(
                update={"arguments": self.decode_text(call.function.arguments)}
            )

    def orphan_handles(self, decoded: str) -> list[str]:
        """Report distinct handle-shaped tokens the registry cannot resolve.

        A non-empty result means the model emitted a reference no real resource
        backs (hallucination) or the user text happened to collide with the
        handle syntax.
        """
        if decoded == "":
            return []
        orphans: list[str] = []
        seen: set[str] = set()
        for match in _RESOURCE_HANDLE_SHAPE_RE.findall(decoded):
            if self._table.has(match):
                continue
            if match in seen:
                continue
            seen.add(match)
            orphans.append(match)
        return orphans

    def handles(self) -> list[str]:
        """Return the handle tokens currently known to this registry."""
        return [item.handle for item in self._table.pairs()]


__all__ = ["ResourceRegistry"]
