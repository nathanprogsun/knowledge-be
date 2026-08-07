"""Unit tests for the knowledge-base domain types."""

from __future__ import annotations

from datetime import UTC, datetime

from src.core.knowledge.knowledge_bases.types import (
    FAQ_INDEX_MODE_QUESTION_ANSWER,
    FAQ_INDEX_MODE_QUESTION_ONLY,
    FAQ_QUESTION_INDEX_MODE_COMBINED,
    FAQ_QUESTION_INDEX_MODE_SEPARATE,
    KNOWLEDGE_BASE_TYPE_DOCUMENT,
    KNOWLEDGE_BASE_TYPE_FAQ,
    KNOWLEDGE_BASE_TYPE_WIKI,
    KNOWLEDGE_BASE_TYPES,
    KnowledgeBaseInfo,
    _decode_json,
)
from src.db.models.knowledge_base import KnowledgeBase

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _row(**overrides: object) -> KnowledgeBase:
    defaults: dict[str, object] = {
        "id": "kb-1",
        "name": "docs",
        "tenant_id": 7,
        "created_at": _NOW,
        "updated_at": _NOW,
    }
    return KnowledgeBase.model_validate({**defaults, **overrides})


def test_map_from_db_renames_cos_config_to_storage_config() -> None:
    info = KnowledgeBaseInfo.map_from_db(_row(cos_config={"provider": "cos"}))

    assert info.storage_config == {"provider": "cos"}


def test_map_from_db_preserves_json_config_columns() -> None:
    info = KnowledgeBaseInfo.map_from_db(
        _row(
            chunking_config={"chunk_size": 512},
            indexing_strategy={"vector_enabled": True},
        )
    )

    assert info.chunking_config == {"chunk_size": 512}
    assert info.indexing_strategy == {"vector_enabled": True}


def test_map_from_db_defaults_response_only_fields() -> None:
    info = KnowledgeBaseInfo.map_from_db(_row())

    assert info.knowledge_count == 0
    assert info.chunk_count == 0
    assert info.share_count == 0
    assert info.is_processing is False
    assert info.is_pinned is False
    assert info.pinned_at is None


def test_decode_json_handles_none_empty_and_invalid() -> None:
    assert _decode_json(None) is None
    assert _decode_json("") is None
    assert _decode_json("not-json") is None
    assert _decode_json('["a"]') is None  # non-object JSON yields None


def test_decode_json_parses_text_and_passthrough_dicts() -> None:
    assert _decode_json('{"enabled": true, "model_id": "m"}') == {
        "enabled": True,
        "model_id": "m",
    }
    assert _decode_json({"a": 1}) == {"a": 1}


def test_knowledge_base_type_constants() -> None:
    assert {
        KNOWLEDGE_BASE_TYPE_DOCUMENT,
        KNOWLEDGE_BASE_TYPE_FAQ,
        KNOWLEDGE_BASE_TYPE_WIKI,
    } == KNOWLEDGE_BASE_TYPES
    assert KNOWLEDGE_BASE_TYPE_DOCUMENT == "document"
    assert KNOWLEDGE_BASE_TYPE_FAQ == "faq"
    assert KNOWLEDGE_BASE_TYPE_WIKI == "wiki"
    assert FAQ_INDEX_MODE_QUESTION_ONLY == "question_only"
    assert FAQ_INDEX_MODE_QUESTION_ANSWER == "question_answer"
    assert FAQ_QUESTION_INDEX_MODE_COMBINED == "combined"
    assert FAQ_QUESTION_INDEX_MODE_SEPARATE == "separate"
