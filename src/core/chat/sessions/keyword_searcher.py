"""Keyword knowledge search over stored text chunks.

Hybrid vector search is still unwired. Command palette and
``POST /knowledge-search`` need hits from documents the tenant already
parsed. This searcher loads enabled chunks in the named scope and keeps
rows whose content contains the query.
"""

from __future__ import annotations

from src.core.chat.pipeline.types import SearchResult
from src.core.chat.service import TagScope
from src.core.chat.sessions.knowledge_qa_runner import (
    ChunkLoader,
    DocumentLoader,
    chunk_to_search_result,
    collect_document_ids,
)
from src.core.contracts.knowledge import Knowledge

_HIT_CAP: int = 30
_DOC_SCAN_CAP: int = 80
_SNIPPET_RADIUS: int = 80


def _needles(query: str) -> list[str]:
    """Prefer the full query, then whitespace tokens, all lowercased."""
    stripped = query.strip()
    if not stripped:
        return []
    full = stripped.lower()
    parts: list[str] = [full]
    for token in stripped.split():
        lower = token.lower()
        if lower and lower not in parts:
            parts.append(lower)
    return parts


def _score(content: str, needles: list[str]) -> float:
    """Return 0 when nothing matches, else the fraction of needles found."""
    if not needles:
        return 0.0
    hay = content.lower()
    hits = sum(1 for needle in needles if needle in hay)
    if hits == 0:
        return 0.0
    return min(1.0, hits / len(needles))


def _snippet(content: str, needles: list[str]) -> str:
    """Return a short window around the first needle hit."""
    hay = content.lower()
    index = -1
    for needle in needles:
        index = hay.find(needle)
        if index >= 0:
            break
    if index < 0:
        return content[: _SNIPPET_RADIUS * 2]
    start = max(0, index - _SNIPPET_RADIUS)
    end = min(len(content), index + _SNIPPET_RADIUS)
    return content[start:end]


class KeywordKnowledgeSearcher:
    """``KnowledgeSearcher`` that ranks stored chunks by keyword overlap."""

    def __init__(self, *, documents: DocumentLoader, chunks: ChunkLoader) -> None:
        self._documents = documents
        self._chunks = chunks

    async def search(
        self,
        *,
        tenant_id: int,
        query: str,
        knowledge_base_ids: list[str],
        knowledge_ids: list[str],
        tag_scopes: list[TagScope],
    ) -> list[SearchResult]:
        del tag_scopes
        needles = _needles(query)
        if not needles:
            return []
        target_ids = await collect_document_ids(
            tenant_id=tenant_id,
            knowledge_ids=knowledge_ids,
            knowledge_base_ids=knowledge_base_ids,
            documents=self._documents,
        )
        hits = await self._collect_hits(tenant_id, target_ids, needles)
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:_HIT_CAP]

    async def _collect_hits(
        self,
        tenant_id: int,
        target_ids: list[str],
        needles: list[str],
    ) -> list[SearchResult]:
        """Scan enabled chunks until the hit or document cap is reached."""
        hits: list[SearchResult] = []
        docs = await self._documents.get_documents(
            tenant_id=tenant_id,
            ids=target_ids[:_DOC_SCAN_CAP],
        )
        by_id: dict[str, Knowledge] = {doc.id: doc for doc in docs}
        for knowledge_id in target_ids[:_DOC_SCAN_CAP]:
            rows = await self._chunks.list_chunks_by_knowledge_id(
                tenant_id=tenant_id,
                knowledge_id=knowledge_id,
            )
            document = by_id.get(knowledge_id)
            for row in rows:
                if not row.is_enabled or not row.content.strip():
                    continue
                score = _score(row.content, needles)
                if score <= 0:
                    continue
                base = chunk_to_search_result(row, document)
                hits.append(
                    base.model_copy(
                        update={
                            "score": score,
                            "matched_content": _snippet(row.content, needles),
                        }
                    )
                )
                if len(hits) >= _HIT_CAP:
                    return hits
        return hits


__all__ = ["KeywordKnowledgeSearcher"]
