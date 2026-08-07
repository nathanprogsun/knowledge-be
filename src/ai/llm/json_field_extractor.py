"""Incremental extraction of a string field from streaming JSON fragments.

The reference implementation feeds tool-call argument deltas through a small
state machine that skips the JSON prefix and surfaces the target field's value
as it arrives. For ``field_name="thought"`` and arguments shaped like
``{"thought":"..."}``, each ``feed`` call returns the newly completed content
and defers emission until a safe boundary (so an escape sequence that is still
streaming is never split).
"""

from __future__ import annotations

_SIMPLE_ESCAPES = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "b": "\b",
    "f": "\f",
}


def _find_field_value_start(buf: str, field_name: str) -> int:
    """Byte/code-point offset where the field value content begins.

    Returns -1 when the opening quote has not been seen yet.
    """
    key = f'"{field_name}"'
    idx = buf.find(key)
    if idx < 0:
        return -1
    pos = idx + len(key)
    while pos < len(buf):
        ch = buf[pos]
        if ch == ":" or ch in " \t\n\r":
            pos += 1
            continue
        if ch == '"':
            return pos + 1
        return -1
    return -1


def _find_safe_end(value: str, from_: int) -> tuple[int, bool]:
    """Return ``(safe_end, finished)`` for emission within ``value``.

    ``safe_end`` is the furthest offset that can be emitted without splitting
    an escape sequence; ``finished`` is true when the closing quote was found.
    """
    i = from_
    n = len(value)
    while i < n:
        ch = value[i]
        if ch == "\\":
            if i + 1 >= n:
                return i, False
            if value[i + 1] == "u":
                if i + 5 >= n:
                    return i, False
                i += 6
            else:
                i += 2
        elif ch == '"':
            return i, True
        else:
            i += 1
    return i, False


def _unescape_json_string(s: str) -> str:
    """Convert JSON string escape sequences to their actual characters."""
    if "\\" not in s:
        return s
    out: list[str] = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch == "\\" and i + 1 < n:
            nxt = s[i + 1]
            if nxt == "u":
                if i + 5 < n:
                    try:
                        out.append(chr(int(s[i + 2 : i + 6], 16)))
                    except ValueError:
                        # Unpaired surrogate: keep the escape verbatim.
                        out.append(s[i : i + 6])
                    i += 6
                else:
                    out.append(s[i])
                    i += 1
            elif nxt in _SIMPLE_ESCAPES:
                out.append(_SIMPLE_ESCAPES[nxt])
                i += 2
            else:
                out.append(s[i])
                i += 1
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


class JSONFieldExtractor:
    """State machine extracting one string field from argument deltas."""

    def __init__(self, field_name: str) -> None:
        self._field_name = field_name
        self._buffer = ""
        self._value_start = -1
        self._last_emit = 0
        self._done = False

    def feed(self, args_delta: str) -> str:
        """Process a new argument delta; return newly emitted content."""
        if self._done:
            return ""
        self._buffer += args_delta

        if self._value_start < 0:
            idx = _find_field_value_start(self._buffer, self._field_name)
            if idx < 0:
                return ""
            self._value_start = idx
            self._last_emit = 0

        value_content = self._buffer[self._value_start :]
        safe_end, finished = _find_safe_end(value_content, self._last_emit)
        if safe_end <= self._last_emit:
            if finished:
                self._done = True
            return ""

        raw_chunk = value_content[self._last_emit : safe_end]
        self._last_emit = safe_end
        if finished:
            self._done = True
        return _unescape_json_string(raw_chunk)

    def is_done(self) -> bool:
        """True once the closing quote has been observed."""
        return self._done


__all__ = ["JSONFieldExtractor"]
