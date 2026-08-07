"""Unit tests for wiki domain types and validation helpers."""

# ruff: noqa: RUF001  # Chinese test data uses fullwidth punctuation.

from __future__ import annotations

from src.core.knowledge.wiki.types import (
    WIKI_CATEGORY_MAX_DEPTH,
    WIKI_EDIT_SOURCE_AGENT,
    WIKI_EDIT_SOURCE_PIPELINE,
    WIKI_EDIT_SOURCE_REVERT,
    WIKI_EDIT_SOURCE_USER,
    WIKI_FOLDER_ROOT_ID,
    WIKI_MAX_REVISIONS_HARD_CAP,
    WIKI_MAX_REVISIONS_PER_PAGE,
    WIKI_PAGE_STATUS_ARCHIVED,
    WIKI_PAGE_STATUS_DRAFT,
    WIKI_PAGE_STATUS_PUBLISHED,
    WIKI_PAGE_TYPE_COMPARISON,
    WIKI_PAGE_TYPE_CONCEPT,
    WIKI_PAGE_TYPE_ENTITY,
    WIKI_PAGE_TYPE_INDEX,
    WIKI_PAGE_TYPE_SUMMARY,
    WIKI_PAGE_TYPE_SYNTHESIS,
    WIKI_PRUNABLE_EDIT_SOURCES,
    WikiPageListFilter,
    clean_category_part,
    clean_category_path,
    is_valid_page_status,
    is_valid_page_type,
    normalize_edit_source,
    split_page_types,
)

# ── page type / status validation ────────────────────────────────────


def test_is_valid_page_type_accepts_every_known_type() -> None:
    for page_type in (
        WIKI_PAGE_TYPE_SUMMARY,
        WIKI_PAGE_TYPE_ENTITY,
        WIKI_PAGE_TYPE_CONCEPT,
        WIKI_PAGE_TYPE_INDEX,
        WIKI_PAGE_TYPE_SYNTHESIS,
        WIKI_PAGE_TYPE_COMPARISON,
    ):
        assert is_valid_page_type(page_type) is True


def test_is_valid_page_type_rejects_unknown() -> None:
    assert is_valid_page_type("glossary") is False
    assert is_valid_page_type("") is False


def test_is_valid_page_status_accepts_known_statuses() -> None:
    assert is_valid_page_status(WIKI_PAGE_STATUS_DRAFT) is True
    assert is_valid_page_status(WIKI_PAGE_STATUS_PUBLISHED) is True
    assert is_valid_page_status(WIKI_PAGE_STATUS_ARCHIVED) is True


def test_is_valid_page_status_rejects_unknown() -> None:
    assert is_valid_page_status("deleted") is False


# ── edit source normalisation ────────────────────────────────────────


def test_normalize_edit_source_keeps_known_sources() -> None:
    for source in (
        WIKI_EDIT_SOURCE_PIPELINE,
        WIKI_EDIT_SOURCE_AGENT,
        WIKI_EDIT_SOURCE_USER,
        WIKI_EDIT_SOURCE_REVERT,
    ):
        assert normalize_edit_source(source) == source


def test_normalize_edit_source_maps_unknown_to_pipeline() -> None:
    assert normalize_edit_source("") == WIKI_EDIT_SOURCE_PIPELINE
    assert normalize_edit_source("hand_edited") == WIKI_EDIT_SOURCE_PIPELINE


def test_prunable_edit_sources_include_legacy_empty() -> None:
    assert "" in WIKI_PRUNABLE_EDIT_SOURCES
    assert WIKI_EDIT_SOURCE_PIPELINE in WIKI_PRUNABLE_EDIT_SOURCES


def test_retention_constants_are_two_tiered() -> None:
    assert WIKI_MAX_REVISIONS_PER_PAGE < WIKI_MAX_REVISIONS_HARD_CAP


# ── page type splitting ──────────────────────────────────────────────


def test_split_page_types_splits_and_dedupes() -> None:
    assert split_page_types("entity,concept,entity") == ["entity", "concept"]


def test_split_page_types_trims_and_drops_blanks() -> None:
    assert split_page_types(" entity , ,concept ") == ["entity", "concept"]


def test_split_page_types_none_for_blank() -> None:
    assert split_page_types("") is None
    assert split_page_types("   ") is None


# ── category part cleaning ───────────────────────────────────────────


def test_clean_category_part_splits_embedded_separators() -> None:
    assert clean_category_part("AI/LLM") == ["AI", "LLM"]
    assert clean_category_part("AI／LLM") == ["AI", "LLM"]
    assert clean_category_part("AI｜LLM") == ["AI", "LLM"]


def test_clean_category_part_strips_wrapping_quotes_and_brackets() -> None:
    assert clean_category_part('"RAG"') == ["RAG"]
    assert clean_category_part("（人物）") == ["人物"]
    assert clean_category_part("[人物]") == ["人物"]


def test_clean_category_part_drops_type_labels() -> None:
    assert clean_category_part("entity") == []
    assert clean_category_part("实体") == []
    assert clean_category_part("概念") == []


def test_clean_category_part_blank_returns_empty() -> None:
    assert clean_category_part("") == []
    assert clean_category_part("   ") == []


def test_clean_category_path_dedupes_and_caps_at_max_depth() -> None:
    path = clean_category_path(["AI", "AI", "LLM 应用", "RAG", "深入"])

    assert path == ["AI", "LLM 应用", "RAG"]
    assert len(path) == WIKI_CATEGORY_MAX_DEPTH


def test_clean_category_path_drops_noise_segments() -> None:
    assert clean_category_path(["entity", "AI", "概念"]) == ["AI"]


def test_clean_category_path_empty_input() -> None:
    assert clean_category_path([]) == []


# ── list filter model ────────────────────────────────────────────────


def test_wiki_page_list_filter_defaults() -> None:
    filt = WikiPageListFilter(knowledge_base_id="kb-1")

    assert filt.page_size == 20
    assert filt.page == 1
    assert filt.category_path == []
    assert filt.folder_id is None


def test_wiki_folder_root_sentinel_is_empty() -> None:
    assert WIKI_FOLDER_ROOT_ID == ""
