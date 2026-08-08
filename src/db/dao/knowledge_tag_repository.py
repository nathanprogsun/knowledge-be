"""Knowledge-tag persistence — raw SQL only, no ORM.

Implements the ``tags`` and ``document_tags`` surfaces: tag CRUD
(scoped by ``tenant_id``), paginated keyword filtering within a
knowledge base, document-tag bind/unbind, and the per-tag reference
counts (live documents / chunks) that back list stats and the
delete guard.

Tags are hard-deleted (no ``deleted_at`` column on the table), so
reads need no soft-delete filter and deletes remove the row outright.
The association table's composite primary key makes re-binding a
document idempotent: ``set_knowledge_tags`` replaces the whole binding
set in one delete + insert, skipping empty and duplicate ids.

Every query is ``sqlalchemy.text()`` with named ``bindparams``; user
input reaches only bindparam slots, never the SQL string.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import NamedTuple, cast

from sqlalchemy import CursorResult, text

from src.common.exception import DataError
from src.common.json import BindParams, SqlValue
from src.db.dao.generic_repository import GenericRepository
from src.db.models.knowledge_tag import KnowledgeTag

# LIKE treats these as wildcards / escape; a user-supplied keyword
# must have them neutralised before interpolation into `%...%`.
_LIKE_ESCAPE_CHAR = "\\"
_LIKE_SPECIAL_CHARS = ("\\", "%", "_")


def escape_like_pattern(term: str) -> str:
    """Escape LIKE wildcards so the search term matches literally."""
    escaped = term
    for char in _LIKE_SPECIAL_CHARS:
        escaped = escaped.replace(char, _LIKE_ESCAPE_CHAR + char)
    return escaped


class TagReferenceCounts(NamedTuple):
    """Aggregate live-document and live-chunk references for one tag."""

    knowledge_count: int
    chunk_count: int


class TagRepository(GenericRepository[KnowledgeTag]):
    """`tags`-table SQL — tag CRUD + document-tag bind/unbind + filtering."""

    model_class = KnowledgeTag

    # ── Tag CRUD ────────────────────────────────────────────────────

    async def create(self, row: KnowledgeTag) -> KnowledgeTag:
        """Insert a tag and return the persisted row (seq_id assigned by DB)."""
        return await self.insert(row)

    async def update(self, row: KnowledgeTag) -> KnowledgeTag:
        """Overwrite the mutable columns of the row, returning the result.

        ``id`` / ``seq_id`` / ``tenant_id`` / ``knowledge_base_id`` /
        ``created_at`` are immutable by contract, so they stay out of the
        SET clause.
        """
        immutable = {"id", "seq_id", "tenant_id", "knowledge_base_id", "created_at"}
        updates = {k: v for k, v in row.model_dump().items() if k not in immutable}
        persisted = await self.update_by_primary_key({"id": row.id}, updates)
        if persisted is None:
            raise DataError(
                code="tag.update_no_row",
                message=f"tag {row.id} not found for update",
            )
        return persisted

    async def get_by_id(self, tenant_id: int, id: str) -> KnowledgeTag | None:
        """Return the tag for ``id`` scoped by tenant, or ``None``."""
        return await self.find_unique_by_column_values({"id": id, "tenant_id": tenant_id})

    async def get_by_ids(self, tenant_id: int, ids: list[str]) -> list[KnowledgeTag]:
        """Return every tenant-scoped tag whose id is in ``ids``."""
        if not ids:
            return []
        placeholders = ", ".join(f":id{i}" for i in range(len(ids)))
        params: BindParams = {f"id{i}": vid for i, vid in enumerate(ids)}
        params["tenant_id"] = tenant_id
        stmt = text(
            f"select * from tags where tenant_id = :tenant_id and id in ({placeholders})"
        ).bindparams(**params)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    async def get_by_seq_id(self, tenant_id: int, seq_id: int) -> KnowledgeTag | None:
        """Return the tag for ``seq_id`` scoped by tenant, or ``None``."""
        return await self.find_unique_by_column_values({"seq_id": seq_id, "tenant_id": tenant_id})

    async def get_by_seq_ids(self, tenant_id: int, seq_ids: list[int]) -> list[KnowledgeTag]:
        """Return every tenant-scoped tag whose seq_id is in ``seq_ids``."""
        if not seq_ids:
            return []
        placeholders = ", ".join(f":sid{i}" for i in range(len(seq_ids)))
        params: BindParams = {f"sid{i}": sid for i, sid in enumerate(seq_ids)}
        params["tenant_id"] = tenant_id
        stmt = text(
            f"select * from tags where tenant_id = :tenant_id and seq_id in ({placeholders})"
        ).bindparams(**params)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    async def get_by_name(
        self,
        tenant_id: int,
        knowledge_base_id: str,
        name: str,
    ) -> KnowledgeTag | None:
        """Return the tag with ``name`` inside a knowledge base, or ``None``.

        ``(tenant_id, knowledge_base_id, name)`` is unique, so at most
        one row can match.
        """
        return await self.find_unique_by_column_values(
            {
                "tenant_id": tenant_id,
                "knowledge_base_id": knowledge_base_id,
                "name": name,
            }
        )

    async def list_by_kb(
        self,
        *,
        tenant_id: int,
        knowledge_base_id: str,
        page: int = 1,
        page_size: int = 20,
        keyword: str = "",
    ) -> tuple[list[KnowledgeTag], int]:
        """Return one page of the knowledge base's tags plus the total.

        ``keyword`` filters ``name`` with LIKE (wildcards neutralised).
        Ordering mirrors the upstream query: ``sort_order`` ascending,
        then ``created_at`` / ``seq_id`` descending so OFFSET pagination
        stays stable when the first two keys collide.
        """
        keyword = keyword.strip()
        where = "tenant_id = :tenant_id and knowledge_base_id = :kb_id"
        params: BindParams = {"tenant_id": tenant_id, "kb_id": knowledge_base_id}
        if keyword:
            where += f" and name like :keyword escape '{_LIKE_ESCAPE_CHAR}'"
            params["keyword"] = f"%{escape_like_pattern(keyword)}%"

        count_stmt = text(f"select count(*) from tags where {where}").bindparams(**params)
        total = (await self._session.execute(count_stmt)).scalar_one()

        offset = (page - 1) * page_size
        stmt = text(
            f"select * from tags where {where} "
            "order by sort_order asc, created_at desc, seq_id desc "
            "limit :limit offset :offset"
        ).bindparams(**params, limit=page_size, offset=offset)
        result = await self._session.execute(stmt)
        rows = [self._hydrate(m) for m in result.mappings().all()]
        return rows, int(total)

    async def delete(self, *, tenant_id: int, id: str) -> bool:
        """Hard-delete a tag scoped by tenant. Returns whether a row was removed."""
        result = await self._session.execute(
            text("delete from tags where tenant_id = :tenant_id and id = :id").bindparams(
                tenant_id=tenant_id,
                id=id,
            )
        )
        return (cast("CursorResult[SqlValue]", result).rowcount or 0) > 0

    # ── Reference counts ────────────────────────────────────────────

    async def count_references(
        self,
        *,
        tenant_id: int,
        knowledge_base_id: str,
        tag_id: str,
    ) -> TagReferenceCounts:
        """Return the live-document and live-chunk counts for one tag.

        Documents come from ``document_tags`` joined to live
        ``documents`` rows; chunks carry their own ``tag_id`` column.
        Both sides filter soft-deleted rows, mirroring the upstream
        count semantics.
        """
        knowledge_stmt = text(
            "select count(*) from document_tags dt "
            "join documents d on dt.knowledge_id = d.id and d.deleted_at is null "
            "where d.tenant_id = :tenant_id and d.knowledge_base_id = :kb_id "
            "and dt.tag_id = :tag_id"
        ).bindparams(tenant_id=tenant_id, kb_id=knowledge_base_id, tag_id=tag_id)
        knowledge_total = (await self._session.execute(knowledge_stmt)).scalar_one()

        chunk_stmt = text(
            "select count(*) from chunks "
            "where tenant_id = :tenant_id and knowledge_base_id = :kb_id "
            "and tag_id = :tag_id and deleted_at is null"
        ).bindparams(tenant_id=tenant_id, kb_id=knowledge_base_id, tag_id=tag_id)
        chunk_total = (await self._session.execute(chunk_stmt)).scalar_one()
        return TagReferenceCounts(
            knowledge_count=int(knowledge_total),
            chunk_count=int(chunk_total),
        )

    async def batch_count_references(
        self,
        *,
        tenant_id: int,
        knowledge_base_id: str,
        tag_ids: list[str],
    ) -> dict[str, TagReferenceCounts]:
        """Return per-tag reference counts in two aggregate queries.

        Every requested id is present in the result, zero-filled when
        nothing references it.
        """
        result: dict[str, TagReferenceCounts] = {
            tag_id: TagReferenceCounts(0, 0) for tag_id in tag_ids
        }
        if not tag_ids:
            return result
        placeholders = ", ".join(f":tag{i}" for i in range(len(tag_ids)))
        params: BindParams = {f"tag{i}": tag_id for i, tag_id in enumerate(tag_ids)}
        params["tenant_id"] = tenant_id
        params["kb_id"] = knowledge_base_id

        knowledge_stmt = text(
            "select dt.tag_id, count(*) as total from document_tags dt "
            "join documents d on dt.knowledge_id = d.id and d.deleted_at is null "
            "where d.tenant_id = :tenant_id and d.knowledge_base_id = :kb_id "
            f"and dt.tag_id in ({placeholders}) "
            "group by dt.tag_id"
        ).bindparams(**params)
        for row in (await self._session.execute(knowledge_stmt)).mappings().all():
            current = result.get(row["tag_id"])
            if current is not None:
                result[row["tag_id"]] = TagReferenceCounts(int(row["total"]), current.chunk_count)

        chunk_stmt = text(
            "select tag_id, count(*) as total from chunks "
            "where tenant_id = :tenant_id and knowledge_base_id = :kb_id "
            f"and tag_id in ({placeholders}) and deleted_at is null "
            "group by tag_id"
        ).bindparams(**params)
        for row in (await self._session.execute(chunk_stmt)).mappings().all():
            current = result.get(row["tag_id"])
            if current is not None:
                result[row["tag_id"]] = TagReferenceCounts(
                    current.knowledge_count, int(row["total"])
                )
        return result

    # ── Document-tag bind / unbind ──────────────────────────────────

    async def set_knowledge_tags(self, *, knowledge_id: str, tag_ids: list[str]) -> None:
        """Replace a document's tag bindings in one delete + insert.

        Empty and duplicate tag ids are skipped. The caller owns the
        surrounding transaction (a session-level commit).
        """
        await self._session.execute(
            text("delete from document_tags where knowledge_id = :knowledge_id").bindparams(
                knowledge_id=knowledge_id
            )
        )
        unique_ids: list[str] = []
        seen: set[str] = set()
        for tag_id in tag_ids:
            if tag_id == "" or tag_id in seen:
                continue
            seen.add(tag_id)
            unique_ids.append(tag_id)
        if not unique_ids:
            return
        now = datetime.now(UTC)
        placeholders = ", ".join(f"(:k{i}, :t{i}, :c{i})" for i in range(len(unique_ids)))
        params: BindParams = {}
        for i, tag_id in enumerate(unique_ids):
            params[f"k{i}"] = knowledge_id
            params[f"t{i}"] = tag_id
            params[f"c{i}"] = now
        stmt = text(
            f"insert into document_tags (knowledge_id, tag_id, created_at) values {placeholders}"
        ).bindparams(**params)
        await self._session.execute(stmt)

    async def get_knowledge_tags(self, knowledge_ids: list[str]) -> dict[str, list[KnowledgeTag]]:
        """Return each document's tags, keyed by ``knowledge_id``."""
        result: dict[str, list[KnowledgeTag]] = {}
        if not knowledge_ids:
            return result
        placeholders = ", ".join(f":kid{i}" for i in range(len(knowledge_ids)))
        params: BindParams = {f"kid{i}": kid for i, kid in enumerate(knowledge_ids)}
        stmt = text(
            "select t.id, t.seq_id, t.tenant_id, t.knowledge_base_id, t.name, t.color, "
            "t.sort_order, t.created_at, t.updated_at, dk.knowledge_id "
            "from document_tags dk join tags t on dk.tag_id = t.id "
            f"where dk.knowledge_id in ({placeholders})"
        ).bindparams(**params)
        rows = (await self._session.execute(stmt)).mappings().all()
        for row in rows:
            tag = self._hydrate(row)
            result.setdefault(row["knowledge_id"], []).append(tag)
        return result

    async def delete_knowledge_tag_relations(self, knowledge_id: str) -> int:
        """Remove every tag binding of a document. Returns the row count."""
        result = await self._session.execute(
            text("delete from document_tags where knowledge_id = :knowledge_id").bindparams(
                knowledge_id=knowledge_id
            )
        )
        return cast("CursorResult[SqlValue]", result).rowcount or 0


__all__ = ["TagReferenceCounts", "TagRepository", "escape_like_pattern"]
