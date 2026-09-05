"""Unit tests for keyword knowledge search over stored chunks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from src.common.json import JsonObject
from src.core.chat.sessions.keyword_searcher import KeywordKnowledgeSearcher
from src.core.contracts.knowledge import Knowledge

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


@dataclass
class _Chunk:
    id: str
    content: str
    knowledge_id: str
    knowledge_base_id: str
    chunk_index: int
    start_at: int = 0
    end_at: int = 10
    chunk_type: str = "text"
    parent_chunk_id: str | None = None
    image_info: str | None = None
    metadata: JsonObject | None = None
    is_enabled: bool = True


def _doc(doc_id: str, *, title: str) -> Knowledge:
    return Knowledge(
        id=doc_id,
        tenant_id=1,
        knowledge_base_id="kb-1",
        type="file",
        title=title,
        description="",
        source="upload",
        channel="web",
        parse_status="completed",
        enable_status="enabled",
        created_at=_NOW,
        updated_at=_NOW,
    )


class _Docs:
    def __init__(self, rows: list[Knowledge]) -> None:
        self._rows = rows

    async def get_documents(self, *, tenant_id: int, ids: list[str]) -> list[Knowledge]:
        del tenant_id
        wanted = set(ids)
        return [row for row in self._rows if row.id in wanted]

    async def list_documents(self, *, tenant_id: int, knowledge_base_id: str) -> list[Knowledge]:
        del tenant_id
        return [row for row in self._rows if row.knowledge_base_id == knowledge_base_id]


class _Chunks:
    def __init__(self, rows: list[_Chunk]) -> None:
        self._rows = rows

    async def list_chunks_by_knowledge_id(
        self, *, tenant_id: int, knowledge_id: str
    ) -> list[_Chunk]:
        del tenant_id
        return [row for row in self._rows if row.knowledge_id == knowledge_id]


@pytest.mark.asyncio
async def test_keyword_searcher_returns_matching_chunks() -> None:
    searcher = KeywordKnowledgeSearcher(
        documents=_Docs([_doc("doc-1", title="基金研报")]),
        chunks=_Chunks(
            [
                _Chunk(
                    id="c-1",
                    content="本周基金净值上涨",
                    knowledge_id="doc-1",
                    knowledge_base_id="kb-1",
                    chunk_index=0,
                ),
                _Chunk(
                    id="c-2",
                    content="天气很好",
                    knowledge_id="doc-1",
                    knowledge_base_id="kb-1",
                    chunk_index=1,
                ),
            ]
        ),
    )

    hits = await searcher.search(
        tenant_id=1,
        query="基金",
        knowledge_base_ids=["kb-1"],
        knowledge_ids=[],
        tag_scopes=[],
    )

    assert [hit.id for hit in hits] == ["c-1"]
    assert hits[0].knowledge_title == "基金研报"
    assert "基金" in (hits[0].matched_content or hits[0].content)


@pytest.mark.asyncio
async def test_keyword_searcher_empty_query_returns_nothing() -> None:
    searcher = KeywordKnowledgeSearcher(documents=_Docs([]), chunks=_Chunks([]))
    hits = await searcher.search(
        tenant_id=1,
        query="   ",
        knowledge_base_ids=["kb-1"],
        knowledge_ids=[],
        tag_scopes=[],
    )
    assert hits == []
