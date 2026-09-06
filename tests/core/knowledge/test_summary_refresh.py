"""Unit tests for the regenerate-summary refresher."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from src.common.exception import NotFoundError, ValidationError
from src.core.knowledge.documents.summary_refresh import DocumentSummaryRefresher
from src.core.knowledge.knowledge_bases.types import KnowledgeBaseInfo
from src.db.models.knowledge import Document

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _row() -> Document:
    return Document(
        id="doc-1",
        tenant_id=1,
        knowledge_base_id="kb-1",
        type="file",
        title="t",
        description="",
        source="upload",
        channel="web",
        parse_status="completed",
        summary_status="none",
        enable_status="enabled",
        created_at=_NOW,
        updated_at=_NOW,
    )


@pytest.mark.asyncio
async def test_refresh_raises_when_document_missing() -> None:
    knowledge_repo = AsyncMock()
    knowledge_repo.get_by_id.return_value = None
    refresher = DocumentSummaryRefresher(
        knowledge_repo=knowledge_repo,
        chunk_repo=AsyncMock(),
        kb_service=AsyncMock(),
        chat_models=AsyncMock(),
    )
    with pytest.raises(NotFoundError) as exc:
        await refresher.refresh(tenant_id=1, knowledge_id="missing")
    assert exc.value.code == "knowledge.not_found"


@pytest.mark.asyncio
async def test_refresh_raises_when_no_chat_model() -> None:
    knowledge_repo = AsyncMock()
    knowledge_repo.get_by_id.return_value = _row()
    kb_service = AsyncMock()
    kb_service.get_knowledge_base_by_id_and_tenant.return_value = KnowledgeBaseInfo(
        id="kb-1",
        tenant_id=1,
        name="kb",
        description="",
        type="document",
        created_at=_NOW,
        updated_at=_NOW,
    )
    chat_models = AsyncMock()
    chat_models.first_knowledge_qa_id.return_value = None
    refresher = DocumentSummaryRefresher(
        knowledge_repo=knowledge_repo,
        chunk_repo=AsyncMock(),
        kb_service=kb_service,
        chat_models=chat_models,
    )
    with pytest.raises(ValidationError) as exc:
        await refresher.refresh(tenant_id=1, knowledge_id="doc-1")
    assert exc.value.code == "knowledge.summary_model_not_configured"
