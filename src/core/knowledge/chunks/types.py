"""Domain constants and helpers for document chunks.

Mirrors the chunk domain contract: ``ChunkType`` values, ``ChunkStatus``
levels, and the ``ChunkFlags`` bit field. The flag helpers operate on the
``flags`` integer column; the Go model keeps the same bits as a small
value type and we translate that to plain functions.

The wire projection of a chunk row lives in ``src/core/contracts/knowledge.py``
(frozen); this module holds the domain-side vocabulary only.
"""

from __future__ import annotations

# ── Chunk types ──────────────────────────────────────────────────────

CHUNK_TYPE_TEXT = "text"
CHUNK_TYPE_PARENT_TEXT = "parent_text"
CHUNK_TYPE_IMAGE_OCR = "image_ocr"
CHUNK_TYPE_IMAGE_CAPTION = "image_caption"
CHUNK_TYPE_SUMMARY = "summary"
CHUNK_TYPE_ENTITY = "entity"
CHUNK_TYPE_RELATIONSHIP = "relationship"
CHUNK_TYPE_FAQ = "faq"
CHUNK_TYPE_WEB_SEARCH = "web_search"
CHUNK_TYPE_TABLE_SUMMARY = "table_summary"
CHUNK_TYPE_TABLE_COLUMN = "table_column"
CHUNK_TYPE_WIKI_PAGE = "wiki_page"

# ── Chunk status ─────────────────────────────────────────────────────

CHUNK_STATUS_DEFAULT = 0
CHUNK_STATUS_STORED = 1
CHUNK_STATUS_INDEXED = 2

# ── Chunk flags (bit field) ──────────────────────────────────────────

# Recommended flag: when set, the chunk may be surfaced to users. It is
# the default for new chunks (the ``flags`` column defaults to 1).
CHUNK_FLAG_RECOMMENDED = 1 << 0


def has_flag(flags: int, flag: int) -> bool:
    """Return whether ``flag`` is set in the ``flags`` bit field."""
    return (flags & flag) != 0


def set_flag(flags: int, flag: int) -> int:
    """Return ``flags`` with ``flag`` set (a new value, never mutates)."""
    return flags | flag


def clear_flag(flags: int, flag: int) -> int:
    """Return ``flags`` with ``flag`` cleared (a new value, never mutates)."""
    return flags & ~flag


def toggle_flag(flags: int, flag: int) -> int:
    """Return ``flags`` with ``flag`` flipped (a new value, never mutates)."""
    return flags ^ flag


__all__ = [
    "CHUNK_FLAG_RECOMMENDED",
    "CHUNK_STATUS_DEFAULT",
    "CHUNK_STATUS_INDEXED",
    "CHUNK_STATUS_STORED",
    "CHUNK_TYPE_ENTITY",
    "CHUNK_TYPE_FAQ",
    "CHUNK_TYPE_IMAGE_CAPTION",
    "CHUNK_TYPE_IMAGE_OCR",
    "CHUNK_TYPE_PARENT_TEXT",
    "CHUNK_TYPE_RELATIONSHIP",
    "CHUNK_TYPE_SUMMARY",
    "CHUNK_TYPE_TABLE_COLUMN",
    "CHUNK_TYPE_TABLE_SUMMARY",
    "CHUNK_TYPE_TEXT",
    "CHUNK_TYPE_WEB_SEARCH",
    "CHUNK_TYPE_WIKI_PAGE",
    "clear_flag",
    "has_flag",
    "set_flag",
    "toggle_flag",
]
