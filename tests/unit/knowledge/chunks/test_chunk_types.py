"""Unit tests for the chunk domain vocabulary in
``src.core.knowledge.chunks.types``.

Pure constant + helper tests: chunk-type strings, status levels, and the
``flags`` bit-field helpers.
"""

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


def test_chunk_type_values_match_contract() -> None:
    assert CHUNK_TYPE_TEXT == "text"
    assert CHUNK_TYPE_PARENT_TEXT == "parent_text"
    assert CHUNK_TYPE_IMAGE_OCR == "image_ocr"
    assert CHUNK_TYPE_IMAGE_CAPTION == "image_caption"
    assert CHUNK_TYPE_SUMMARY == "summary"
    assert CHUNK_TYPE_ENTITY == "entity"
    assert CHUNK_TYPE_RELATIONSHIP == "relationship"
    assert CHUNK_TYPE_FAQ == "faq"
    assert CHUNK_TYPE_WEB_SEARCH == "web_search"
    assert CHUNK_TYPE_TABLE_SUMMARY == "table_summary"
    assert CHUNK_TYPE_TABLE_COLUMN == "table_column"
    assert CHUNK_TYPE_WIKI_PAGE == "wiki_page"


def test_chunk_status_levels() -> None:
    assert CHUNK_STATUS_DEFAULT == 0
    assert CHUNK_STATUS_STORED == 1
    assert CHUNK_STATUS_INDEXED == 2


def test_chunk_flag_recommended_is_first_bit() -> None:
    assert CHUNK_FLAG_RECOMMENDED == 1


def test_has_flag() -> None:
    assert has_flag(1, CHUNK_FLAG_RECOMMENDED) is True
    assert has_flag(0, CHUNK_FLAG_RECOMMENDED) is False
    assert has_flag(2, CHUNK_FLAG_RECOMMENDED) is False


def test_set_flag() -> None:
    assert set_flag(0, CHUNK_FLAG_RECOMMENDED) == 1
    assert set_flag(1, CHUNK_FLAG_RECOMMENDED) == 1
    assert set_flag(2, CHUNK_FLAG_RECOMMENDED) == 3


def test_clear_flag() -> None:
    assert clear_flag(1, CHUNK_FLAG_RECOMMENDED) == 0
    assert clear_flag(0, CHUNK_FLAG_RECOMMENDED) == 0
    assert clear_flag(3, CHUNK_FLAG_RECOMMENDED) == 2


def test_toggle_flag() -> None:
    assert toggle_flag(0, CHUNK_FLAG_RECOMMENDED) == 1
    assert toggle_flag(1, CHUNK_FLAG_RECOMMENDED) == 0
    assert toggle_flag(2, CHUNK_FLAG_RECOMMENDED) == 3


def test_flag_helpers_never_mutate_input() -> None:
    original = 3
    assert set_flag(original, 0) == 3
    assert clear_flag(original, CHUNK_FLAG_RECOMMENDED) == 2
    assert toggle_flag(original, 0) == 3
    # The input integer is untouched throughout.
    assert original == 3
