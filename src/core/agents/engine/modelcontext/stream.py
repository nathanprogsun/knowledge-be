"""Streaming-safe handle decoding.

Owns the shared suffix-hold primitive, the per-space decoders built on it, and
the composed ``StreamDecoder`` that applies every stage in the only safe order:
resource restoration, source-citation expansion, issue-handle decoding, and
orphan filtering.
"""

from __future__ import annotations

from collections.abc import Callable

from src.core.agents.engine.modelcontext.citations import CitationStreamExpander
from src.core.agents.engine.modelcontext.handles import HandleTable
from src.core.agents.engine.modelcontext.resources import (
    _RESOURCE_HANDLE_SHAPE_RE,
    ResourceRegistry,
)

_RESOURCE_PREFIX = "res://"


class StreamHold:
    """Shared primitive for decoders that must never emit a partial handle.

    Each ``feed`` withholds a trailing run that could still grow into a handle
    in the next provider chunk and applies the space-specific decode to
    everything released. ``flush`` decides what happens to a suffix still held
    when the stream closes.
    """

    def __init__(
        self,
        hold_len: Callable[[str], int],
        emit: Callable[[str], str],
        flush: Callable[[str], str],
    ) -> None:
        self._hold_len = hold_len
        self._emit = emit
        self._flush = flush
        self._pending = ""

    def feed(self, chunk: str) -> str:
        combined = self._pending + chunk
        self._pending = ""
        if combined == "":
            return ""
        hold = self._hold_len(combined)
        if hold > 0 and hold <= len(combined):
            self._pending = combined[len(combined) - hold :]
            combined = combined[: len(combined) - hold]
        return self._emit(combined)

    def flush(self) -> str:
        pending = self._pending
        self._pending = ""
        return self._flush(pending)


class ResourceStreamDecoder:
    """Restores ``res://`` handles split across provider chunks."""

    def __init__(self, registry: ResourceRegistry) -> None:
        handles = registry.handles()

        def hold_len(combined: str) -> int:
            hold = 0
            for handle in handles:
                # A provider may split the token at any byte boundary, including
                # "re" + "s://0001". Holding at most the short matching suffix
                # is the only way to guarantee a request-local handle never leaks.
                for n in range(1, len(handle)):
                    if n > hold and combined.endswith(handle[:n]):
                        hold = n
            return hold

        self._hold = StreamHold(hold_len, registry.decode_text, registry.decode_text)

    def feed(self, chunk: str) -> str:
        return self._hold.feed(chunk)

    def flush(self) -> str:
        return self._hold.flush()


class HandleStreamDecoder:
    """Restores ``HandleTable`` values without leaking a split handle."""

    def __init__(self, table: HandleTable) -> None:
        self._table = table
        prefix = table.prefix

        def hold_len(combined: str) -> int:
            start = len(combined)
            while start > 0 and is_handle_token_byte(combined[start - 1]):
                start -= 1
            tail = combined[start:]
            # The prefix is immutable after construction.
            if could_be_numeric_handle(tail, prefix):
                return len(tail)
            return 0

        self._hold = StreamHold(hold_len, table.decode_known_text, table.decode_known_text)

    def feed(self, chunk: str) -> str:
        return self._hold.feed(chunk)

    def flush(self) -> str:
        return self._hold.flush()


def is_handle_token_byte(value: str) -> bool:
    """Return whether ``value`` is a single ASCII handle-token byte."""
    return "a" <= value <= "z" or "A" <= value <= "Z" or "0" <= value <= "9" or value in "-_"


def could_be_numeric_handle(value: str, prefix: str) -> bool:
    """Return whether ``value`` is the prefix plus a (possibly empty) digit run."""
    if value == "" or prefix == "":
        return False
    if prefix.startswith(value):
        return True
    if not value.startswith(prefix) or len(value) == len(prefix):
        return False
    return all("0" <= char <= "9" for char in value[len(prefix) :])


class OrphanResourceStreamFilter:
    """Removes unresolved resource handles only after known ones are restored.

    Buffers a possible handle suffix so provider chunk boundaries cannot leak a
    partial internal token to the UI.
    """

    def __init__(self) -> None:
        self._hold = StreamHold(
            orphan_resource_hold_len,
            lambda released: _RESOURCE_HANDLE_SHAPE_RE.sub("", released),
            orphan_resource_flush,
        )

    def feed(self, chunk: str) -> str:
        return self._hold.feed(chunk)

    def flush(self) -> str:
        return self._hold.flush()


def orphan_resource_hold_len(combined: str) -> int:
    """Bytes to withhold so unknown ``res://`` handles are safe across splits.

    This may defer at most a few ordinary characters until the next chunk;
    ``flush`` preserves them when they are normal prose.
    """
    hold_at = -1
    for n in range(1, len(_RESOURCE_PREFIX)):
        if combined.endswith(_RESOURCE_PREFIX[:n]):
            hold_at = len(combined) - n
    idx = combined.rfind(_RESOURCE_PREFIX)
    if idx >= 0:
        suffix = combined[idx + len(_RESOURCE_PREFIX) :]
        if suffix == "" or all_digits(suffix):
            hold_at = idx
    if hold_at < 0:
        return 0
    return len(combined) - hold_at


def orphan_resource_flush(pending: str) -> str:
    """A stream that ends mid-token must not surface the protocol fragment.

    Preserve ordinary ``r``/``re``/``res`` prose, but discard any suffix that
    has already crossed into the reserved URL-like syntax.
    """
    if _RESOURCE_PREFIX.startswith(pending) and len(pending) >= len("res:"):
        return ""
    if pending.startswith(_RESOURCE_PREFIX):
        digits = pending[len(_RESOURCE_PREFIX) :]
        if digits == "" or all_digits(digits):
            return ""
    return _RESOURCE_HANDLE_SHAPE_RE.sub("", pending)


def all_digits(value: str) -> bool:
    """Return whether every character of ``value`` is an ASCII digit."""
    if value == "":
        return False
    return all("0" <= char <= "9" for char in value)


class StreamDecoder:
    """Composed decoder applying every stage in the only safe order."""

    def __init__(
        self,
        resources: ResourceStreamDecoder,
        sources: CitationStreamExpander,
        issues: HandleStreamDecoder,
        orphans: OrphanResourceStreamFilter,
    ) -> None:
        self._resources = resources
        self._sources = sources
        self._issues = issues
        self._orphans = orphans

    def feed(self, chunk: str) -> str:
        if self._resources is not None:
            chunk = self._resources.feed(chunk)
        if self._sources is not None:
            chunk = self._sources.feed(chunk)
        if self._issues is not None:
            chunk = self._issues.feed(chunk)
        if self._orphans is not None:
            chunk = self._orphans.feed(chunk)
        return chunk

    def flush(self) -> str:
        # Each stage's tail must be fed THROUGH the later stages before those
        # stages flush their own pending suffix, otherwise a handle completed by
        # an earlier stage's tail would bypass later decoding.
        tail = ""
        if self._resources is not None:
            tail = self._resources.flush()
        if self._sources is not None:
            tail = self._sources.feed(tail) + self._sources.flush()
        if self._issues is not None:
            tail = self._issues.feed(tail) + self._issues.flush()
        if self._orphans is not None:
            tail = self._orphans.feed(tail) + self._orphans.flush()
        return tail


__all__ = [
    "HandleStreamDecoder",
    "OrphanResourceStreamFilter",
    "ResourceStreamDecoder",
    "StreamDecoder",
    "StreamHold",
    "all_digits",
    "could_be_numeric_handle",
    "is_handle_token_byte",
]
