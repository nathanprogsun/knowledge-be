"""Unit tests for document download / preview helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from src.common.exception import NotFoundError
from src.common.json import JsonObject
from src.core.contracts.knowledge import Knowledge
from src.web.api.knowledge.documents.document_reads import stream_document

_NOW = datetime(2026, 9, 5, tzinfo=UTC)


def _knowledge(*, type: str, metadata: JsonObject | None = None) -> Knowledge:
    return Knowledge(
        id="doc-1",
        tenant_id=7,
        knowledge_base_id="kb-1",
        type=type,
        title="Note",
        parse_status="completed",
        enable_status="enabled",
        file_name="note.md",
        file_path=None,
        metadata=metadata,
        created_at=_NOW,
        updated_at=_NOW,
    )


@pytest.mark.asyncio
async def test_stream_manual_document_returns_markdown() -> None:
    service = AsyncMock()
    service.get_document = AsyncMock(
        return_value=_knowledge(
            type="manual",
            metadata={"content": "# hello", "status": "draft"},
        )
    )

    response = await stream_document(
        service=service,
        session=AsyncMock(),
        tenant_id=7,
        knowledge_id="doc-1",
        disposition="attachment",
    )

    assert response.media_type == "text/markdown; charset=utf-8"
    assert response.body == b"# hello"
    assert "attachment" in response.headers["content-disposition"]
    assert response.headers["x-content-type-options"] == "nosniff"


@pytest.mark.asyncio
async def test_stream_url_document_without_file_raises() -> None:
    service = AsyncMock()
    service.get_document = AsyncMock(return_value=_knowledge(type="url"))

    with pytest.raises(NotFoundError) as exc_info:
        await stream_document(
            service=service,
            session=AsyncMock(),
            tenant_id=7,
            knowledge_id="doc-1",
            disposition="inline",
        )

    assert exc_info.value.code == "knowledge.file_unavailable"
