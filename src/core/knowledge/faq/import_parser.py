# ruff: noqa: RUF001  # Chinese user-facing messages use fullwidth punctuation.

"""CSV / Excel row parsing for FAQ batch imports.

The tabular import format is one row per FAQ entry. The column set is
fixed by the import template (the same shape the export path produces):
tag name, standard question, similar questions, negative questions,
answers, and three boolean toggles. Multi-value columns separate items
with ``##``.

The parser is deliberately structural: it checks the header, the cell
count, the boolean tokens, and cell coercion — the *format* of the file.
Semantic per-entry validation (required standard question / answers,
intra-entry duplicate rules) is left to ``sanitize_faq_content`` and the
batch duplicate detection to the import pipeline, so the parser has no
dependency on the persistence layer and can be unit-tested in isolation.

Excel support loads ``openpyxl`` lazily so the module imports without it;
when the dependency is missing the parser raises a clear error instead of
failing at import time.
"""

from __future__ import annotations

import csv
import importlib
import io
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Protocol, cast

from src.common.exception import ValidationError
from src.core.contracts.knowledge import FAQEntryPayload

# A raw spreadsheet cell: the parser coerces every cell to ``str`` before
# interpreting it, so any cell type ``openpyxl`` / ``csv`` may produce is
# accepted here.
CellValue = str | int | float | bool | datetime | None

# The import template header. Column positions are the format contract:
# 0 tag, 1 standard question, 2 similar questions, 3 negative questions,
# 4 answers, 5 answer-all toggle, 6 disabled toggle, 7 recommend toggle.
IMPORT_HEADERS: tuple[str, ...] = (
    "分类(必填)",
    "问题(必填)",
    "相似问题(选填-多个用##分隔)",
    "反例问题(选填-多个用##分隔)",
    "机器人回答(必填-多个用##分隔)",
    "是否全部回复(选填-默认FALSE)",
    "是否停用(选填-默认FALSE)",
    "是否禁止被推荐(选填-默认False 可被推荐)",
)

# Separator for multi-value columns (similar / negative / answers).
VALUE_SEPARATOR = "##"

# Minimum number of header columns the template requires. The two optional
# trailing toggles may be omitted by a caller.
_MIN_HEADER_COLUMNS = 6

_TRUE_TOKENS: frozenset[str] = frozenset({"true", "1", "yes", "y", "t", "是"})
_FALSE_TOKENS: frozenset[str] = frozenset({"false", "0", "no", "n", "f", "否"})


@dataclass(frozen=True)
class ImportRowError:
    """A structural row-level failure (bad column count / boolean)."""

    row_number: int
    code: str
    message: str


@dataclass(frozen=True)
class ParsedEntry:
    """One successfully parsed row plus its 1-based row number.

    The row number is the file position (header row is 1, the first data
    row is 2) so failed-entry reports can point the user at the offending
    line.
    """

    row_number: int
    payload: FAQEntryPayload


@dataclass(frozen=True)
class ParsedImport:
    """Result of parsing a FAQ import file.

    ``entries`` holds the structurally valid rows in file order,
    ``errors`` the structural failures, and ``skipped_rows`` the count of
    blank data rows that were ignored. ``total`` is the number of
    non-blank data rows read from the file.
    """

    entries: list[ParsedEntry] = field(default_factory=list)
    errors: list[ImportRowError] = field(default_factory=list)
    skipped_rows: int = 0

    @property
    def total(self) -> int:
        """Number of data rows read (valid entries plus structural errors)."""
        return len(self.entries) + len(self.errors)


def parse_import_file(data: bytes, *, filename: str) -> ParsedImport:
    """Parse ``data`` as CSV or Excel, dispatching on the file extension."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        return parse_csv(data)
    if suffix in {".xlsx", ".xls"}:
        return parse_excel(data)
    raise ValidationError(
        code="faq.import_unsupported_file",
        message=f"不支持的 FAQ 导入文件格式: {suffix or '未知'} (仅支持 CSV / Excel)",
    )


def parse_csv(data: bytes) -> ParsedImport:
    """Parse UTF-8 CSV bytes into entries. A leading BOM is tolerated."""
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValidationError(
            code="faq.import_decode_error",
            message="FAQ CSV 文件编码不是 UTF-8",
        ) from exc
    reader = csv.reader(io.StringIO(text), dialect="excel")
    return _parse_rows(reader, source_label="CSV")


def parse_excel(data: bytes) -> ParsedImport:
    """Parse Excel workbook bytes into entries (first worksheet)."""
    workbook = _load_workbook(data)
    sheet = workbook.active
    return _parse_rows(sheet.iter_rows(values_only=True), source_label="Excel")


# ── Row pipeline ──────────────────────────────────────────────────────


def _parse_rows(
    rows: Iterable[Sequence[CellValue]],
    *,
    source_label: str,
) -> ParsedImport:
    """Consume the row stream: locate the header, then map data rows."""
    entries: list[ParsedEntry] = []
    errors: list[ImportRowError] = []
    skipped_rows = 0
    row_number = 0
    header_seen = False
    for cells in rows:
        row_number += 1
        normalized = _coerce_cells(cells)
        if not header_seen:
            if not _is_blank(normalized):
                header_seen = True
                if not _header_matches(normalized):
                    raise ValidationError(
                        code="faq.import_invalid_header",
                        message=(
                            f"FAQ 导入模板表头不匹配（{source_label} 第 {row_number} 行），"
                            "请使用 FAQ 导入模板文件"
                        ),
                    )
            continue
        if _is_blank(normalized):
            skipped_rows += 1
            continue
        payload, error = _row_to_entry(normalized, row_number=row_number)
        if error is not None:
            errors.append(error)
        elif payload is not None:
            entries.append(ParsedEntry(row_number=row_number, payload=payload))

    if not header_seen:
        raise ValidationError(
            code="faq.import_empty",
            message="FAQ 导入文件为空，没有可解析的行",
        )
    return ParsedImport(entries=entries, errors=errors, skipped_rows=skipped_rows)


def _header_matches(cells: list[str]) -> bool:
    """Return whether ``cells`` is the template header (or a valid prefix).

    At least the required columns must match position-for-position; extra
    trailing columns beyond the template are tolerated.
    """
    width = min(len(cells), len(IMPORT_HEADERS))
    if width < _MIN_HEADER_COLUMNS:
        return False
    return cells[:width] == list(IMPORT_HEADERS[:width])


def _row_to_entry(
    cells: list[str],
    *,
    row_number: int,
) -> tuple[FAQEntryPayload | None, ImportRowError | None]:
    """Map one data row to a payload, reporting structural failures.

    Only the format is checked here (column count, boolean tokens);
    required-field and duplicate semantics are enforced downstream by the
    content sanitizer.
    """
    if len(cells) < _MIN_HEADER_COLUMNS:
        return None, _row_error(
            row_number,
            "faq.import_invalid_row",
            f"第 {row_number} 行列数不足，需要至少 {_MIN_HEADER_COLUMNS} 列",
        )

    answer_all, error = _parse_bool(cells[5], row_number=row_number, column="是否全部回复")
    if error is not None:
        return None, error
    is_disabled, error = _parse_bool(
        cells[6] if len(cells) > 6 else "",
        row_number=row_number,
        column="是否停用",
    )
    if error is not None:
        return None, error
    not_recommended, error = _parse_bool(
        cells[7] if len(cells) > 7 else "",
        row_number=row_number,
        column="是否禁止被推荐",
    )
    if error is not None:
        return None, error

    payload = FAQEntryPayload(
        standard_question=cells[1].strip(),
        similar_questions=_split_multi(cells[2]),
        negative_questions=_split_multi(cells[3]),
        answers=_split_multi(cells[4]),
        answer_strategy="all" if answer_all else "random",
        tag_name=cells[0].strip() or None,
        is_enabled=not is_disabled,
        is_recommended=not not_recommended,
    )
    return payload, None


def _row_error(row_number: int, code: str, message: str) -> ImportRowError:
    """Build the structural row error used by the parse helpers."""
    return ImportRowError(row_number=row_number, code=code, message=message)


def _parse_bool(
    raw: str,
    *,
    row_number: int,
    column: str,
) -> tuple[bool, ImportRowError | None]:
    """Parse a boolean toggle cell; empty means the documented default."""
    token = raw.strip().lower()
    if not token:
        return False, None
    if token in _TRUE_TOKENS:
        return True, None
    if token in _FALSE_TOKENS:
        return False, None
    return False, _row_error(
        row_number,
        "faq.import_invalid_boolean",
        f"第 {row_number} 行「{column}」取值无效: {raw.strip()!r}",
    )


def _split_multi(raw: str) -> list[str]:
    """Split a ``##``-separated column, trimming and dropping empties."""
    return [part.strip() for part in raw.split(VALUE_SEPARATOR) if part.strip()]


def _coerce_cells(cells: Sequence[CellValue]) -> list[str]:
    """Normalise a raw row (which may carry numbers / dates / None) to strings."""
    return ["" if cell is None else str(cell) for cell in cells]


def _is_blank(cells: list[str]) -> bool:
    """Return whether the row is entirely whitespace / empty."""
    return not any(cell.strip() for cell in cells)


class _Worksheet(Protocol):
    """Minimal worksheet surface the parser needs from ``openpyxl``."""

    def iter_rows(self, *, values_only: bool = True) -> Iterable[Sequence[CellValue]]: ...


class _Workbook(Protocol):
    """Minimal workbook surface the parser needs from ``openpyxl``."""

    active: _Worksheet


def _load_workbook(data: bytes) -> _Workbook:
    """Load an Excel workbook via a lazy ``openpyxl`` dependency probe.

    ``openpyxl`` is optional at runtime; ``importlib.import_module`` keeps
    the dependency lazy without a runtime import statement.
    """
    try:
        openpyxl = importlib.import_module("openpyxl")
    except ImportError as exc:
        raise ValidationError(
            code="faq.import_excel_unsupported",
            message="当前环境缺少 openpyxl，无法解析 Excel 文件",
        ) from exc
    return cast(_Workbook, openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True))


__all__ = [
    "IMPORT_HEADERS",
    "VALUE_SEPARATOR",
    "ImportRowError",
    "ParsedEntry",
    "ParsedImport",
    "parse_csv",
    "parse_excel",
    "parse_import_file",
]
