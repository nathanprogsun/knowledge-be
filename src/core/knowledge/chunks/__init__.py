"""Chunk domain: types, status, flags, revisions, and retrieval questions."""

from __future__ import annotations

from src.core.knowledge.chunks.types import (
    CHUNK_FLAG_RECOMMENDED,
    CHUNK_STATUS_DEFAULT,
    CHUNK_STATUS_INDEXED,
    CHUNK_STATUS_STORED,
    CHUNK_TYPE_ENTITY,
    CHUNK_TYPE_FAQ,
    CHUNK_TYPE_IMAGE_CAPTION,
    CHUNK_TYPE_IMAGE_OCR,
    CHUNK_TYPE_PARENT_TEXT,
    CHUNK_TYPE_RELATIONSHIP,
    CHUNK_TYPE_SUMMARY,
    CHUNK_TYPE_TABLE_COLUMN,
    CHUNK_TYPE_TABLE_SUMMARY,
    CHUNK_TYPE_TEXT,
    CHUNK_TYPE_WEB_SEARCH,
    CHUNK_TYPE_WIKI_PAGE,
    clear_flag,
    has_flag,
    set_flag,
    toggle_flag,
)

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
