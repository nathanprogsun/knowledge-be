"""Unit tests for the tag domain DTOs and table models."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.common.exception import ValidationError
from src.core.knowledge.tags.types import TagInfo
from src.db.models.knowledge_tag import DocumentTag, KnowledgeTag

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _db_row(**overrides: object) -> KnowledgeTag:
    defaults: dict[str, object] = {
        "id": "tag-abc",
        "seq_id": 10000001,
        "tenant_id": 7,
        "knowledge_base_id": "kb-123",
        "name": "infrastructure",
        "color": "#ff0000",
        "sort_order": 3,
        "created_at": _NOW,
        "updated_at": _NOW,
    }
    return KnowledgeTag.model_validate({**defaults, **overrides})


# ── KnowledgeTag model metadata ───────────────────────────────────────


def test_knowledge_tag_insert_excludes_db_generated_seq_id() -> None:
    # ``id`` participates in INSERT (caller-assigned); ``seq_id`` is
    # assigned by the DB sequence and read back via RETURNING.
    columns = KnowledgeTag.insert_sql_column_list()

    assert "id" in columns
    assert "seq_id" not in columns


def test_knowledge_tag_ordered_primary_keys() -> None:
    assert KnowledgeTag.ordered_primary_keys() == ("id",)


def test_document_tag_uses_composite_primary_key() -> None:
    assert DocumentTag.ordered_primary_keys() == ("knowledge_id", "tag_id")


def test_document_tag_primary_key_validation_rejects_missing_part() -> None:
    with pytest.raises(ValidationError) as exc_info:
        DocumentTag.validate_contains_all_primary_keys(["knowledge_id"])
    assert exc_info.value.code == "db.missing_primary_key"


# ── TagInfo.map_from_db ───────────────────────────────────────────────


def test_map_from_db_copies_row_fields() -> None:
    info = TagInfo.map_from_db(_db_row())

    assert info.id == "tag-abc"
    assert info.seq_id == 10000001
    assert info.tenant_id == 7
    assert info.knowledge_base_id == "kb-123"
    assert info.name == "infrastructure"
    assert info.color == "#ff0000"
    assert info.sort_order == 3
    assert info.created_at == _NOW
    assert info.updated_at == _NOW


def test_map_from_db_defaults_counts_to_zero() -> None:
    info = TagInfo.map_from_db(_db_row())

    assert info.knowledge_count == 0
    assert info.chunk_count == 0


def test_map_from_db_accepts_usage_counts() -> None:
    info = TagInfo.map_from_db(_db_row(), knowledge_count=3, chunk_count=5)

    assert info.knowledge_count == 3
    assert info.chunk_count == 5


def test_map_from_db_defaults_optional_columns() -> None:
    info = TagInfo.map_from_db(_db_row(color=None, sort_order=0))

    assert info.color is None
    assert info.sort_order == 0
