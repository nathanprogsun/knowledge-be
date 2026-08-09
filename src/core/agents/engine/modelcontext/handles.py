"""Exported word-boundary-aware handle space wrapper.

``HandleTable`` is a typed, invocation-local bidirectional mapping used when a
prompt needs compact handles for durable values (wiki issue ``iN`` handles,
ingest ``ref-N`` slug handles, ``c000`` citation-batch handles). Handles are
never persisted; ``resolve`` converts model output back to the durable value
first.
"""

from __future__ import annotations

import re

from src.core.agents.engine.modelcontext.handle_table import HandleTable as _HandleTable


class HandleTable:
    """Bidirectional mapping between durable values and invocation-local handles.

    Unlike the resource codec, text decoding is word-boundary-aware so short
    handles such as ``i1`` cannot fire inside ordinary prose tokens.
    """

    def __init__(self, prefix: str, width: int, start: int) -> None:
        self._table: _HandleTable[None] = _HandleTable[None](prefix, width, start)

    @property
    def prefix(self) -> str:
        """The immutable handle prefix used by streaming decoders."""
        return self._table.prefix

    def register(self, value: str) -> str:
        """Return the stable handle assigned to ``value`` in this table."""
        value = value.strip()
        if value == "":
            return ""
        return self._table.register(value, value, None, None)

    def handle(self, value: str) -> str | None:
        """Return an already-registered handle without creating one."""
        return self._table.handle_for_key(value)

    def resolve(self, handle: str) -> str | None:
        """Convert a known model handle back to its durable value."""
        result = self._table.resolve(handle.strip())
        if result is None:
            return None
        value, _meta = result
        return value

    def empty(self) -> bool:
        """Report whether no handles have been allocated."""
        return self._table.size() == 0

    def len(self) -> int:
        """Return the number of registered handles."""
        return self._table.size()

    def encode_known_text(self, value: str) -> str:
        """Replace already-registered durable values with their handles.

        Longer values are processed first to avoid substring shadowing.
        """
        if value == "":
            return value
        pairs = sorted(self._table.pairs(), key=lambda item: len(item.value), reverse=True)
        for item in pairs:
            value = value.replace(item.value, item.handle)
        return value

    def decode_known_text(self, value: str) -> str:
        """Restore registered handles in complete text on word boundaries.

        A durable value may legally contain ``$``; using a literal callback for
        the replacement avoids regexp-expansion interpretation.
        """
        if value == "":
            return value
        pairs = sorted(self._table.pairs(), key=lambda item: len(item.handle), reverse=True)
        for item in pairs:
            pattern = item.word_bounded
            if pattern is None:
                continue
            durable = item.value

            def _replace(_match: re.Match[str], durable: str = durable) -> str:
                return durable

            value = pattern.sub(_replace, value)
        return value


__all__ = ["HandleTable"]
