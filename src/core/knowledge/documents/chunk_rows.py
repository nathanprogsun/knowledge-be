"""Convert parsed chunks into ``chunks`` table rows.

Lives beside the process pipeline so the orchestrator stays under the
file-size bound while the parent/child link rules stay in one place.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from src.core.knowledge.documents.chunk_pipeline import ParsedChunk
from src.db.models.chunk import Chunk

_CHUNK_TYPE_TEXT = "text"
_CHUNK_TYPE_PARENT_TEXT = "parent_text"


def _link_sequence(rows: list[Chunk]) -> list[Chunk]:
    """Return new rows with prev/next ids wired between consecutive entries.

    The input rows are frozen and never mutated; links are applied via
    copies.
    """
    if len(rows) < 2:
        return rows
    linked: list[Chunk] = []
    for i, row in enumerate(rows):
        updates: dict[str, str | None] = {}
        if i > 0:
            updates["pre_chunk_id"] = rows[i - 1].id
        if i < len(rows) - 1:
            updates["next_chunk_id"] = rows[i + 1].id
        linked.append(row.model_copy(update=updates) if updates else row)
    return linked


def _chunk_row(
    *,
    tenant_id: int,
    knowledge_id: str,
    knowledge_base_id: str,
    now: datetime,
    content: str,
    chunk_index: int,
    start_at: int,
    end_at: int,
    chunk_type: str,
    context_header: str = "",
    parent_chunk_id: str | None = None,
) -> Chunk:
    return Chunk(
        id=str(uuid.uuid4()),
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        knowledge_id=knowledge_id,
        content=content,
        chunk_index=chunk_index,
        is_enabled=True,
        start_at=start_at,
        end_at=end_at,
        chunk_type=chunk_type,
        parent_chunk_id=parent_chunk_id,
        context_header=context_header,
        created_at=now,
        updated_at=now,
    )


def _parent_rows(
    *,
    tenant_id: int,
    knowledge_id: str,
    knowledge_base_id: str,
    parent_chunks: list[ParsedChunk],
    now: datetime,
) -> list[Chunk]:
    parents = [
        _chunk_row(
            tenant_id=tenant_id,
            knowledge_id=knowledge_id,
            knowledge_base_id=knowledge_base_id,
            now=now,
            content=pc.content,
            chunk_index=pc.seq,
            start_at=pc.start,
            end_at=pc.end,
            chunk_type=_CHUNK_TYPE_PARENT_TEXT,
            context_header=pc.context_header,
        )
        for pc in parent_chunks
    ]
    return _link_sequence(parents)


def _child_rows(
    *,
    tenant_id: int,
    knowledge_id: str,
    knowledge_base_id: str,
    chunks: list[ParsedChunk],
    parents: list[Chunk],
    now: datetime,
) -> list[Chunk]:
    rows: list[Chunk] = []
    for pc in chunks:
        if not pc.content.strip():
            continue
        parent_id: str | None = None
        if 0 <= pc.parent_index < len(parents):
            parent_id = parents[pc.parent_index].id
        rows.append(
            _chunk_row(
                tenant_id=tenant_id,
                knowledge_id=knowledge_id,
                knowledge_base_id=knowledge_base_id,
                now=now,
                content=pc.content,
                chunk_index=pc.seq,
                start_at=pc.start,
                end_at=pc.end,
                chunk_type=_CHUNK_TYPE_TEXT,
                context_header=pc.context_header,
                parent_chunk_id=parent_id,
            )
        )
    return rows


def _flat_rows(
    *,
    tenant_id: int,
    knowledge_id: str,
    knowledge_base_id: str,
    chunks: list[ParsedChunk],
    now: datetime,
) -> list[Chunk]:
    rows: list[Chunk] = []
    for pc in chunks:
        if not pc.content.strip():
            continue
        rows.append(
            _chunk_row(
                tenant_id=tenant_id,
                knowledge_id=knowledge_id,
                knowledge_base_id=knowledge_base_id,
                now=now,
                content=pc.content,
                chunk_index=pc.seq,
                start_at=pc.start,
                end_at=pc.end,
                chunk_type=_CHUNK_TYPE_TEXT,
                context_header=pc.context_header,
            )
        )
    return rows


def build_chunk_rows(
    *,
    tenant_id: int,
    knowledge_id: str,
    knowledge_base_id: str,
    chunks: list[ParsedChunk],
    parent_chunks: list[ParsedChunk],
    now: datetime,
) -> tuple[list[Chunk], list[Chunk]]:
    """Convert parsed chunks into storage rows.

    Parent rows (parent-child mode) are stored as ``parent_text`` and
    never indexed; text rows are stored as ``text`` and carry prev/next
    links only in flat mode, mirroring the upstream chunk-write. In
    parent-child mode each text row references its parent and only
    parented rows join the index subset. Returns ``(all_rows, text_rows)``
    with ``all_rows`` in document order and ``text_rows`` the subset that
    feeds the vector index.
    """
    has_parent_child = bool(parent_chunks)
    if has_parent_child:
        parents = _parent_rows(
            tenant_id=tenant_id,
            knowledge_id=knowledge_id,
            knowledge_base_id=knowledge_base_id,
            parent_chunks=parent_chunks,
            now=now,
        )
        rows = parents + _child_rows(
            tenant_id=tenant_id,
            knowledge_id=knowledge_id,
            knowledge_base_id=knowledge_base_id,
            chunks=chunks,
            parents=parents,
            now=now,
        )
    else:
        rows = _flat_rows(
            tenant_id=tenant_id,
            knowledge_id=knowledge_id,
            knowledge_base_id=knowledge_base_id,
            chunks=chunks,
            now=now,
        )
    ordered = sorted(rows, key=lambda row: row.chunk_index)
    if not has_parent_child:
        ordered = _link_sequence(ordered)
    text_rows = [
        row
        for row in ordered
        if row.chunk_type == _CHUNK_TYPE_TEXT
        and (not has_parent_child or row.parent_chunk_id is not None)
    ]
    return ordered, text_rows


__all__ = ["build_chunk_rows"]
