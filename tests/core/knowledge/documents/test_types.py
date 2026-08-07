"""Unit tests for :mod:`src.core.knowledge.documents.types`.

Pins the document-domain constants and the ``DocumentListFilter`` shape:
field names mirror the upstream list-filter struct, so the frozen model
is part of the public contract.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from src.core.knowledge.documents.types import (
    CHANNEL_WEB,
    KNOWLEDGE_TYPE_MANUAL,
    PARSE_STATUS_DELETING,
    PARSE_STATUS_PENDING,
    PARSE_STATUSES,
    SUMMARY_STATUS_NONE,
    SUMMARY_STATUSES,
    DocumentListFilter,
)

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


# ── Status constants ────────────────────────────────────────────────


def test_parse_status_constant_set_is_exhaustive() -> None:
    assert PARSE_STATUS_PENDING in PARSE_STATUSES
    assert PARSE_STATUS_DELETING in PARSE_STATUSES
    assert len(PARSE_STATUSES) == 7


def test_summary_status_constant_set_is_exhaustive() -> None:
    assert SUMMARY_STATUS_NONE in SUMMARY_STATUSES
    assert len(SUMMARY_STATUSES) == 5


# ── DocumentListFilter ──────────────────────────────────────────────


def test_document_list_filter_defaults_are_off() -> None:
    f = DocumentListFilter()
    assert f.tag_ids == []
    assert f.keyword is None
    assert f.file_type is None
    assert f.parse_status is None
    assert f.source is None
    assert f.updated_from is None
    assert f.updated_to is None


def test_document_list_filter_accepts_every_dimension() -> None:
    f = DocumentListFilter(
        tag_ids=["tag-1", "tag-2"],
        keyword="budget",
        file_type=KNOWLEDGE_TYPE_MANUAL,
        parse_status=PARSE_STATUS_PENDING,
        source=CHANNEL_WEB,
        updated_from=_NOW,
        updated_to=_NOW,
    )

    assert f.tag_ids == ["tag-1", "tag-2"]
    assert f.keyword == "budget"
    assert f.file_type == "manual"
    assert f.parse_status == "pending"
    assert f.source == "web"
    assert f.updated_from == _NOW
    assert f.updated_to == _NOW


def test_document_list_filter_is_frozen() -> None:
    f = DocumentListFilter(keyword="budget")
    with pytest.raises(ValidationError):
        f.keyword = "replaced"
