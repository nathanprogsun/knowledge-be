"""Generic bidirectional handle table behind every model-handle space.

The table maps a durable dedup key to a sequentially allocated handle and
stores the durable value plus optional metadata, so a resolvable handle can
never be observed without its metadata. The key and the value may differ
(web references dedup on a canonicalized URL but decode back to the original
raw URL). Entries are never removed, which keeps the counter equivalent to the
historical ``len(map)+1`` allocation and handle numbering stable for the
lifetime of the table.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

M = TypeVar("M")


@dataclass(frozen=True)
class HandleEntry(Generic[M]):
    """Durable value plus metadata stored for one allocated handle."""

    value: str
    meta: M
    word_bounded: re.Pattern[str] | None = None


@dataclass(frozen=True)
class HandlePair:
    """Snapshot row for the text codecs (compaction and decoding)."""

    value: str
    handle: str
    word_bounded: re.Pattern[str] | None = None


class HandleTable(Generic[M]):
    """Typed, invocation-local bidirectional mapping of durable keys to handles.

    ``prefix`` / ``width`` / ``start`` shape the handle space, e.g. ``c1``
    (``prefix="c", width=0, start=1``), ``c000`` (``prefix="c", width=3,
    start=0``) or ``res://0001`` (``prefix="res://", width=4, start=1``).
    """

    def __init__(self, prefix: str, width: int, start: int) -> None:
        self.prefix = prefix
        self._width = width
        self._next = start
        self._handle_by_key: dict[str, str] = {}
        self._entry_by_handle: dict[str, HandleEntry[M]] = {}

    def register(
        self,
        key: str,
        value: str,
        meta: M,
        merge: Callable[[M, M], M] | None = None,
    ) -> str:
        """Return the handle for ``key``, allocating the next one on first use.

        ``merge`` folds the metadata of a repeated registration into the
        existing entry; the allocated handle is never renumbered.
        """
        if key == "":
            return ""
        existing = self._handle_by_key.get(key)
        if existing is not None:
            if merge is not None:
                entry = self._entry_by_handle[existing]
                self._entry_by_handle[existing] = HandleEntry(
                    value=entry.value,
                    meta=merge(entry.meta, meta),
                    word_bounded=entry.word_bounded,
                )
            return existing
        number = f"{self._next:0{self._width}d}" if self._width > 0 else f"{self._next}"
        self._next += 1
        handle = f"{self.prefix}{number}"
        self._handle_by_key[key] = handle
        self._entry_by_handle[handle] = HandleEntry(
            value=value,
            meta=meta,
            word_bounded=re.compile(rf"\b{re.escape(handle)}\b"),
        )
        return handle

    def handle_for_key(self, key: str) -> str | None:
        """Return an already-registered handle without allocating one."""
        return self._handle_by_key.get(key)

    def resolve(self, handle: str) -> tuple[str, M] | None:
        """Return the durable value and metadata snapshot for a known handle."""
        entry = self._entry_by_handle.get(handle)
        if entry is None:
            return None
        return (entry.value, entry.meta)

    def has(self, handle: str) -> bool:
        """Report whether ``handle`` exists in this table."""
        return handle in self._entry_by_handle

    def size(self) -> int:
        """Return the number of allocated handles."""
        return len(self._entry_by_handle)

    def pairs(self) -> list[HandlePair]:
        """Return a value/handle snapshot for the text codecs.

        Callers own ordering (e.g. longest-value-first for substring
        compaction). The word-bounded pattern is compiled once at registration
        because decoding runs on every streamed chunk.
        """
        pairs: list[HandlePair] = []
        for handle, entry in self._entry_by_handle.items():
            pairs.append(
                HandlePair(value=entry.value, handle=handle, word_bounded=entry.word_bounded)
            )
        return pairs


__all__ = ["HandleEntry", "HandlePair", "HandleTable"]
