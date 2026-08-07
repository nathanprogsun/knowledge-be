"""Context-preserving table header tracking for the adaptive text chunker.

When a large Markdown table is split across multiple chunks, each chunk after
the first would lose the table header context. This tracker detects table
headers and signals the merge logic to prepend them to subsequent chunks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Final

# Marks a Markdown table: header row + separator row (e.g. "| A | B |\n| --- | --- |\n").
_TABLE_HEADER_START_PATTERN: Final = re.compile(
    r"^\s*(?:\|[^|\n]*)+[\r\n]+\s*(?:\|\s*:?-{3,}:?\s*)+\|?[\r\n]+$",
    re.DOTALL | re.IGNORECASE,
)
# Empty/whitespace line or a line that doesn't start with | or whitespace.
_TABLE_HEADER_END_PATTERN: Final = re.compile(r"^\s*$|^\s*[^|\s].*$", re.DOTALL | re.IGNORECASE)
# Matches a single Markdown table row: "| cell | cell | ... |\n".
_TABLE_ROW_PATTERN: Final = re.compile(r"^\s*(?:\|[^|\n]*)+\|\s*$", re.MULTILINE)

_MARKDOWN_TABLE_HOOK_PRIORITY: Final = 15


@dataclass(frozen=True)
class HeaderTrackerHook:
    """Pattern pair for detecting a contextual header.

    When ``start_pattern`` matches a unit's text, that text becomes an
    "active header". The header stays active until ``end_pattern`` matches a
    subsequent unit.
    """

    start_pattern: re.Pattern[str]
    end_pattern: re.Pattern[str]
    priority: int


def _default_header_hooks() -> list[HeaderTrackerHook]:
    return [
        HeaderTrackerHook(
            start_pattern=_TABLE_HEADER_START_PATTERN,
            end_pattern=_TABLE_HEADER_END_PATTERN,
            priority=_MARKDOWN_TABLE_HOOK_PRIORITY,
        ),
    ]


@dataclass
class HeaderTracker:
    """Maintains the state of active headers across split units.

    The tracker is a stateful scan helper (mirroring the reference
    algorithm): ``update`` advances it one split unit at a time and
    ``get_headers`` reports the currently active headers.
    """

    hooks: list[HeaderTrackerHook] = field(default_factory=_default_header_hooks)
    active_headers: dict[int, str] = field(default_factory=dict)  # priority -> header text
    ended_headers: dict[int, bool] = field(default_factory=dict)  # priorities that ended
    pending_extend: dict[int, bool] = field(default_factory=dict)  # headers awaiting first data row
    # Set when a table row unit ends with a paragraph break (the blank line
    # between tables is consumed by \n\n splitting). The header stays active
    # until the next unit is seen so we can detect a new table.
    pending_table_break: bool = False
    # Tells mergeUnits to flush before the current unit when a new table
    # starts (column mismatch or pendingTableBreak + table row).
    header_ended_this_unit: bool = False

    def update(self, split: str) -> None:
        self.header_ended_this_unit = False

        if self.pending_table_break:
            self.pending_table_break = False
            if _MARKDOWN_TABLE_HOOK_PRIORITY in self.active_headers:
                if first_table_row_column_count(split) > 0:
                    self.clear_table_header()
                    self.header_ended_this_unit = True
                else:
                    self.clear_table_header()

        # 1. Check for header-end markers among currently active headers.
        for hook in self.hooks:
            if hook.priority not in self.active_headers:
                continue
            if hook.end_pattern.search(split) is not None:
                self.ended_headers[hook.priority] = True
                del self.active_headers[hook.priority]
                self.pending_extend.pop(hook.priority, None)

        # 1b. Paragraph splits consume the blank line between tables. Mark a
        # break after "| last row |\n\n" and resolve on the next unit; also
        # end when a new table row has a different column count than the
        # active header.
        if _MARKDOWN_TABLE_HOOK_PRIORITY in self.active_headers and not self.pending_extend.get(
            _MARKDOWN_TABLE_HOOK_PRIORITY, False
        ):
            if split_ends_with_paragraph_break(split):
                self.pending_table_break = True
            else:
                self.end_table_header_on_column_mismatch(split)

        # 2. If a header has an empty column-name row (e.g. "||"), replace it
        #    with a proper Markdown table header using the first data row as
        #    column names.
        #
        #    Before: "||"           + "| --- | --- |\n"
        #    After:  "| col1 | col2 |\n" + "| --- | --- |\n"
        for priority in list(self.pending_extend):
            if priority in self.active_headers and _TABLE_ROW_PATTERN.search(split) is not None:
                sep = extract_separator_line(self.active_headers[priority])
                self.active_headers[priority] = split + sep
            del self.pending_extend[priority]

        # 3. Check for new header-start markers (only for hooks that are
        #    neither active nor ended).
        for hook in self.hooks:
            if hook.priority in self.active_headers:
                continue
            if hook.priority in self.ended_headers:
                continue
            m = hook.start_pattern.search(split)
            if m is not None:
                self.active_headers[hook.priority] = m.group(0)
                if is_empty_table_header_row(m.group(0)):
                    self.pending_extend[hook.priority] = True

        # 4. If all headers ended, clear the ended set so future tables can be
        #    tracked.
        if not self.active_headers:
            self.ended_headers.clear()

    def get_headers(self) -> str:
        """All active headers concatenated, sorted by priority descending."""
        if not self.active_headers:
            return ""
        entries = sorted(self.active_headers.items(), key=lambda item: item[0], reverse=True)
        return "\n".join(text for _, text in entries)

    def clear_table_header(self) -> None:
        self.ended_headers[_MARKDOWN_TABLE_HOOK_PRIORITY] = True
        self.active_headers.pop(_MARKDOWN_TABLE_HOOK_PRIORITY, None)
        self.pending_extend.pop(_MARKDOWN_TABLE_HOOK_PRIORITY, None)

    def end_table_header_on_column_mismatch(self, split: str) -> None:
        header = self.active_headers.get(_MARKDOWN_TABLE_HOOK_PRIORITY)
        if header is None:
            return
        row_cols = first_table_row_column_count(split)
        header_cols = header_table_column_count(header)
        if row_cols > 0 and header_cols > 0 and row_cols != header_cols:
            self.clear_table_header()
            self.header_ended_this_unit = True


def is_empty_table_header_row(header: str) -> bool:
    """True when the header row (line before the separator) contains only pipes.

    Common with converters that produce tables like ``||`` / ``| --- | --- |``
    where the real column names appear in the first data row.
    """
    idx = header.find("\n")
    if idx < 0:
        return False
    row = header[:idx].strip()
    return all(ch in "|\t " for ch in row)


def extract_separator_line(header: str) -> str:
    """Return the separator line (e.g. ``"| --- | --- |\n"``) from a header."""
    for line in header.split("\n"):
        if "---" in line:
            return line + "\n"
    return ""


def split_ends_with_paragraph_break(split: str) -> bool:
    trimmed = split.rstrip(" \t\r")
    return trimmed.endswith(("\n\n", "\r\n\r\n"))


def table_row_column_count(line: str) -> int:
    line = line.strip()
    if not line.startswith("|"):
        return 0
    parts = line.split("|")
    if parts and parts[0].strip() == "":
        parts = parts[1:]
    if parts and parts[-1].strip() == "":
        parts = parts[:-1]
    return len(parts)


def first_table_row_column_count(text: str) -> int:
    for line in text.split("\n"):
        line = line.strip()
        if line != "" and _TABLE_ROW_PATTERN.search(line) is not None:
            return table_row_column_count(line)
    return 0


def header_table_column_count(header: str) -> int:
    for line in header.split("\n"):
        line = line.strip()
        if line == "" or "---" in line:
            continue
        count = table_row_column_count(line)
        if count > 0:
            return count
    return 0


def header_column_mismatch(headers: str, next_unit: str) -> bool:
    """True when the next unit starts a table whose width differs from the header."""
    header_cols = header_table_column_count(headers)
    row_cols = first_table_row_column_count(next_unit)
    return header_cols > 0 and row_cols > 0 and header_cols != row_cols


__all__ = [
    "HeaderTracker",
    "header_column_mismatch",
    "header_table_column_count",
]
