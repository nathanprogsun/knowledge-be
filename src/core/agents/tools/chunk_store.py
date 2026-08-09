"""Paged chunk listing store for the list-knowledge-chunks tool.

``PagedChunkStore`` is the seam the tool executes against; the concrete
``SqlPagedChunkStore`` reads the ``chunks`` table directly over an
``AsyncSession`` with the same chunk-type filter the retrieval tools use
(text + FAQ), so the totals reported match what the tool can page over.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, cast, runtime_checkable

from sqlalchemy import text
from sqlalchemy.engine import RowMapping
from sqlalchemy.ext.asyncio import AsyncSession

from src.common.json import SqlValue
from src.db.models.chunk import Chunk

_TABLE_NAME = "chunks"

#: Chunk types eligible for paged listing (text + FAQ).
_LISTED_CHUNK_TYPES = ("text", "faq")


@runtime_checkable
class PagedChunkStore(Protocol):
    """Paged listing of a document's text + FAQ chunks."""

    async def list_paged_chunks(
        self,
        *,
        tenant_id: int,
        knowledge_id: str,
        page: int,
        page_size: int,
        enabled_only: bool = True,
    ) -> tuple[list[Chunk], int]: ...


class SqlPagedChunkStore:
    """``chunks``-table implementation over an ``AsyncSession``.

    The count and the rows share the same filters so ``total`` stays
    consistent with what paging can return.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list_paged_chunks(
        self,
        *,
        tenant_id: int,
        knowledge_id: str,
        page: int,
        page_size: int,
        enabled_only: bool = True,
    ) -> tuple[list[Chunk], int]:
        offset = max((page - 1) * page_size, 0)
        enabled_filter = " and is_enabled = :enabled" if enabled_only else ""
        base_params: dict[str, SqlValue] = {
            "tenant_id": tenant_id,
            "knowledge_id": knowledge_id,
            "ct_text": _LISTED_CHUNK_TYPES[0],
            "ct_faq": _LISTED_CHUNK_TYPES[1],
        }
        if enabled_only:
            base_params["enabled"] = True

        count_sql = text(
            f"select count(*) from {_TABLE_NAME} "
            "where tenant_id = :tenant_id and knowledge_id = :knowledge_id "
            "and chunk_type in (:ct_text, :ct_faq) and deleted_at is null"
            f"{enabled_filter}"
        ).bindparams(**base_params)
        count_result = await self._session.execute(count_sql)
        total = count_result.scalar_one()
        total_int = int(total) if total is not None else 0

        rows_sql = text(
            f"select * from {_TABLE_NAME} "
            "where tenant_id = :tenant_id and knowledge_id = :knowledge_id "
            "and chunk_type in (:ct_text, :ct_faq) and deleted_at is null"
            f"{enabled_filter} "
            "order by chunk_index asc limit :limit offset :offset"
        ).bindparams(
            **base_params,
            limit=page_size,
            offset=offset,
        )
        rows_result = await self._session.execute(rows_sql)
        return [self._to_chunk(mapping) for mapping in rows_result.mappings().all()], total_int

    def _to_chunk(self, mapping: RowMapping) -> Chunk:
        row = cast("Mapping[str, SqlValue]", mapping)
        return cast("Chunk", Chunk.from_row(row))


__all__ = ["PagedChunkStore", "SqlPagedChunkStore"]
