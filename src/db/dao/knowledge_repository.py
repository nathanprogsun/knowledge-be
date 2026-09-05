"""Document persistence — raw SQL only, no ORM.

Maps the persistence layer for knowledge entries: create, tenant-scoped
read, list (paged and unpaged), full-row update, soft delete (single and
batch), batch read, status counts, and column-scoped updates. Each method
scopes by ``tenant_id`` so a caller can never read or mutate another
workspace's rows.

Reads filter ``deleted_at IS NULL`` so a soft-deleted row behaves as if
it no longer exists.

The full-row :meth:`update` deliberately never writes ``deleted_at`` or
``pending_subtasks_count``: both are owned elsewhere (the delete pipeline
and the parse-finalize counter respectively), and a blind overwrite
would clobber concurrent orchestration.
"""

from __future__ import annotations

from datetime import datetime
from typing import cast

from sqlalchemy import JSON, bindparam, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import CursorResult, RowMapping

from src.common.exception import DataError
from src.common.json import BindParams, SqlValue
from src.db.dao.generic_repository import GenericRepository
from src.db.models.knowledge import Document

# JSONB on Postgres, JSON on other dialects (e.g. SQLite in tests).
_JSON = JSON().with_variant(JSONB(), "postgresql")

# Table name as a module-level literal; user input is bound via
# ``bindparams``, never interpolated.
_TABLE = "documents"
_KB_TABLE = "knowledge_bases"
_KB_TYPE_DOCUMENT = "document"

# Full-row updates never touch these columns: ``deleted_at`` belongs to
# the soft-delete pipeline, ``pending_subtasks_count`` is the exclusive
# counter of the parse-finalize orchestration.
_OMIT_ON_UPDATE: frozenset[str] = frozenset({"deleted_at", "pending_subtasks_count"})

# Rows mid-deletion stay out of default lists so an async delete does not
# linger as a normal entry; a dead-lettered delete is flipped to ``failed``
# and becomes visible again as an actionable error.
_STATUS_DELETING = "deleting"


def _escape_like(keyword: str) -> str:
    """Escape SQL LIKE wildcards (``%`` / ``_``) in a keyword.

    Backslash is escaped first so a literal backslash in the keyword does
    not neutralise the escapes added here.
    """
    return keyword.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _list_where(
    tenant_id: int,
    knowledge_base_id: str,
    *,
    keyword: str | None,
    file_type: str | None,
    parse_status: str | None,
    source: str | None,
    updated_from: datetime | None,
    updated_to: datetime | None,
) -> tuple[str, BindParams]:
    """Build the shared WHERE clause + bindparams for paged listing.

    Both the count and the page query use the exact same clause so the
    total always matches the returned slice. When ``parse_status`` is
    unset, rows mid-deletion (``parse_status = 'deleting'``) are hidden.
    """
    where_parts = [
        "tenant_id = :tenant_id",
        "knowledge_base_id = :knowledge_base_id",
    ]
    params: BindParams = {
        "tenant_id": tenant_id,
        "knowledge_base_id": knowledge_base_id,
    }
    if keyword:
        params["kw"] = f"%{_escape_like(keyword.lower())}%"
        where_parts.append("(lower(file_name) like :kw or lower(title) like :kw)")
    if file_type:
        if file_type in ("manual", "url"):
            where_parts.append("type = :file_type")
        else:
            where_parts.append("file_type = :file_type")
        params["file_type"] = file_type
    if parse_status:
        where_parts.append("parse_status = :parse_status")
        params["parse_status"] = parse_status
    else:
        where_parts.append("parse_status <> :status_deleting")
        params["status_deleting"] = _STATUS_DELETING
    if source:
        if source in ("manual", "url"):
            where_parts.append("type = :source")
        else:
            where_parts.append("channel = :source")
        params["source"] = source
    if updated_from is not None:
        where_parts.append("updated_at >= :updated_from")
        params["updated_from"] = updated_from
    if updated_to is not None:
        where_parts.append("updated_at <= :updated_to")
        params["updated_to"] = updated_to
    where_parts.append("deleted_at is null")
    return " and ".join(where_parts), params


def _search_file_type_clause(file_types: list[str], params: BindParams) -> str | None:
    """OR-together url/html aliases and bound file-type matches."""
    include_url = False
    extensions: list[str] = []
    seen: set[str] = set()
    for raw in file_types:
        normalized = raw.strip().lstrip(".").lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        if normalized in ("url", "html"):
            include_url = True
        else:
            extensions.append(normalized)
    clauses: list[str] = []
    if include_url:
        clauses.append("(d.type = 'url' or d.file_type in ('html', 'url'))")
    if extensions:
        placeholders = ", ".join(f":ft_{i}" for i in range(len(extensions)))
        clauses.append(f"d.file_type in ({placeholders})")
        for index, ext in enumerate(extensions):
            params[f"ft_{index}"] = ext
    if not clauses:
        return None
    return "(" + " or ".join(clauses) + ")"


def _search_where(
    tenant_id: int,
    *,
    keyword: str,
    file_types: list[str],
) -> tuple[str, BindParams]:
    """Tenant-scoped search across document-type knowledge bases."""
    where_parts = [
        "d.tenant_id = :tenant_id",
        "d.deleted_at is null",
        "d.parse_status <> :status_deleting",
        "kb.deleted_at is null",
        "kb.type = :kb_type",
    ]
    params: BindParams = {
        "tenant_id": tenant_id,
        "status_deleting": _STATUS_DELETING,
        "kb_type": _KB_TYPE_DOCUMENT,
    }
    if keyword:
        params["kw"] = f"%{_escape_like(keyword.lower())}%"
        where_parts.append("(lower(d.file_name) like :kw or lower(d.title) like :kw)")
    type_clause = _search_file_type_clause(file_types, params)
    if type_clause is not None:
        where_parts.append(type_clause)
    return " and ".join(where_parts), params


class KnowledgeRepository(GenericRepository[Document]):
    """`documents`-table SQL — CRUD + list + count + status queries."""

    model_class = Document

    # ── Writes ──────────────────────────────────────────────────────

    async def create(self, row: Document) -> Document:
        """Insert a document and return the persisted row."""
        return await self.insert(row)

    async def update(self, row: Document) -> Document:
        """Overwrite every mutable column of the row, returning the result.

        ``id`` / ``tenant_id`` / ``knowledge_base_id`` / ``created_at``
        are immutable by contract, and ``deleted_at`` /
        ``pending_subtasks_count`` are owned by their own pipelines, so
        they stay out of the SET clause.
        """
        immutable = {"id", "tenant_id", "knowledge_base_id", "created_at"} | _OMIT_ON_UPDATE
        updates = {k: v for k, v in row.model_dump().items() if k not in immutable}
        persisted = await self.update_by_primary_key({"id": row.id}, updates)
        if persisted is None:
            raise DataError(
                code="document.update_no_row",
                message=f"document {row.id} not found for update",
            )
        return persisted

    async def update_columns(self, id: str, values: BindParams) -> Document | None:
        """Write several columns of one row in a single statement.

        Only affects a live (non-deleted) row. Returns the refreshed row,
        or ``None`` when the id did not match.
        """
        if not values:
            return None
        return await self.update_by_primary_key({"id": id}, values)

    async def update_active_deleting_columns(
        self,
        id: str,
        values: BindParams,
    ) -> bool:
        """Write columns only while the row is still in ``deleting`` state.

        Guards against a late status flip resurrecting a row that has
        already moved on. Returns whether any row was affected.
        """
        if not values:
            return False
        self.model_class.validate_in_columns(values)
        set_clause = ", ".join(f'"{c}" = :u_{c}' for c in values)
        update_params: BindParams = {f"u_{c}": v for c, v in values.items()}
        bps = [bindparam(f"u_{c}", type_=_JSON) for c in values if c in self._json_columns]
        stmt = text(
            f"update {_TABLE} set {set_clause} "
            "where id = :id and parse_status = :deleting and deleted_at is null"
        ).bindparams(*bps, **update_params, id=id, deleting=_STATUS_DELETING)
        result = await self._session.execute(stmt)
        return (cast("CursorResult[SqlValue]", result).rowcount or 0) > 0

    async def soft_delete(self, *, tenant_id: int, id: str, now: datetime) -> bool:
        """Mark one row deleted. Returns whether a live row was affected."""
        stmt = text(
            f"update {_TABLE} set deleted_at = :now, updated_at = :now "
            "where tenant_id = :tenant_id and id = :id and deleted_at is null"
        ).bindparams(tenant_id=tenant_id, id=id, now=now)
        result = await self._session.execute(stmt)
        return (cast("CursorResult[SqlValue]", result).rowcount or 0) > 0

    async def soft_delete_list(
        self,
        *,
        tenant_id: int,
        ids: list[str],
        now: datetime,
    ) -> int:
        """Mark a batch of rows deleted. Returns the number affected."""
        if not ids:
            return 0
        placeholders = ", ".join(f":id_{i}" for i in range(len(ids)))
        params: BindParams = {
            "tenant_id": tenant_id,
            "now": now,
            **{f"id_{i}": v for i, v in enumerate(ids)},
        }
        stmt = text(
            f"update {_TABLE} set deleted_at = :now, updated_at = :now "
            f"where tenant_id = :tenant_id and id in ({placeholders}) and deleted_at is null"
        ).bindparams(**params)
        result = await self._session.execute(stmt)
        return cast("CursorResult[SqlValue]", result).rowcount or 0

    # ── Reads ───────────────────────────────────────────────────────

    async def get_by_id(self, tenant_id: int, id: str) -> Document | None:
        """Return one live document by primary key + tenant scope."""
        return await self.find_unique_by_column_values({"id": id, "tenant_id": tenant_id})

    async def get_by_id_only(self, id: str) -> Document | None:
        """Return one live document by id alone (no tenant filter)."""
        return await self.find_by_primary_key({"id": id})

    async def get_batch(self, tenant_id: int, ids: list[str]) -> list[Document]:
        """Return every live document whose id is in ``ids``."""
        if not ids:
            return []
        placeholders = ", ".join(f":id_{i}" for i in range(len(ids)))
        params: BindParams = {
            "tenant_id": tenant_id,
            **{f"id_{i}": v for i, v in enumerate(ids)},
        }
        stmt = text(
            f"select * from {_TABLE} where tenant_id = :tenant_id "
            f"and id in ({placeholders}) and deleted_at is null"
        ).bindparams(**params)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    async def list_by_knowledge_base(
        self,
        tenant_id: int,
        knowledge_base_id: str,
    ) -> list[Document]:
        """Return every live document of a knowledge base, newest first."""
        stmt = text(
            f"select * from {_TABLE} where tenant_id = :tenant_id "
            "and knowledge_base_id = :kb_id and deleted_at is null "
            "order by created_at desc"
        ).bindparams(tenant_id=tenant_id, kb_id=knowledge_base_id)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    async def list_paged_by_knowledge_base(
        self,
        tenant_id: int,
        knowledge_base_id: str,
        *,
        limit: int,
        offset: int,
        keyword: str | None = None,
        file_type: str | None = None,
        parse_status: str | None = None,
        source: str | None = None,
        updated_from: datetime | None = None,
        updated_to: datetime | None = None,
    ) -> tuple[list[Document], int]:
        """Return a page of the knowledge base's documents plus the total.

        ``keyword`` matches ``file_name`` / ``title`` case-insensitively.
        ``file_type`` and ``source`` route the special values ``manual`` /
        ``url`` onto the ``type`` column, matching the upstream filter
        semantics. When ``parse_status`` is unset, mid-deletion rows are
        hidden. Tag-based filtering is handled by the tag-relation layer.
        """
        where, params = _list_where(
            tenant_id,
            knowledge_base_id,
            keyword=keyword,
            file_type=file_type,
            parse_status=parse_status,
            source=source,
            updated_from=updated_from,
            updated_to=updated_to,
        )
        count_stmt = text(f"select count(*) from {_TABLE} where {where}").bindparams(**params)
        total = (await self._session.execute(count_stmt)).scalar_one()
        page_params: BindParams = {**params, "limit": limit, "offset": offset}
        stmt = text(
            f"select * from {_TABLE} where {where} "
            "order by created_at desc limit :limit offset :offset"
        ).bindparams(**page_params)
        result = await self._session.execute(stmt)
        rows = [self._hydrate(m) for m in result.mappings().all()]
        return rows, int(total) if total is not None else 0

    async def search_across_document_kbs(
        self,
        tenant_id: int,
        *,
        keyword: str,
        offset: int,
        limit: int,
        file_types: list[str],
    ) -> tuple[list[tuple[Document, str]], int]:
        """Page live documents across document-type knowledge bases."""
        where, params = _search_where(tenant_id, keyword=keyword, file_types=file_types)
        count_stmt = text(
            f"select count(*) from {_TABLE} d "
            f"inner join {_KB_TABLE} kb on kb.id = d.knowledge_base_id "
            f"and kb.tenant_id = d.tenant_id where {where}"
        ).bindparams(**params)
        total = (await self._session.execute(count_stmt)).scalar_one()
        page_params: BindParams = {**params, "limit": limit, "offset": offset}
        stmt = text(
            f"select d.*, kb.name as knowledge_base_name from {_TABLE} d "
            f"inner join {_KB_TABLE} kb on kb.id = d.knowledge_base_id "
            f"and kb.tenant_id = d.tenant_id where {where} "
            "order by d.updated_at desc limit :limit offset :offset"
        ).bindparams(**page_params)
        result = await self._session.execute(stmt)
        pairs = [self._document_and_kb_name(mapping) for mapping in result.mappings().all()]
        return pairs, int(total) if total is not None else 0

    def _document_and_kb_name(self, mapping: RowMapping) -> tuple[Document, str]:
        """Hydrate a document row and peel off the joined KB name."""
        payload = dict(mapping)
        raw_name = payload.pop("knowledge_base_name", None)
        name = raw_name if isinstance(raw_name, str) else ""
        return self._hydrate(cast("RowMapping", payload)), name

    # ── Counts ──────────────────────────────────────────────────────

    async def count_by_knowledge_base(self, tenant_id: int, knowledge_base_id: str) -> int:
        """Count live documents in a knowledge base."""
        stmt = text(
            f"select count(*) from {_TABLE} "
            "where tenant_id = :tenant_id and knowledge_base_id = :kb_id "
            "and deleted_at is null"
        ).bindparams(tenant_id=tenant_id, kb_id=knowledge_base_id)
        total = (await self._session.execute(stmt)).scalar_one()
        return int(total) if total is not None else 0

    async def count_by_status(
        self,
        tenant_id: int,
        knowledge_base_id: str,
        parse_statuses: list[str],
    ) -> int:
        """Count live documents whose parse status is any of ``parse_statuses``."""
        if not parse_statuses:
            return 0
        placeholders = ", ".join(f":ps_{i}" for i in range(len(parse_statuses)))
        params: BindParams = {
            "tenant_id": tenant_id,
            "knowledge_base_id": knowledge_base_id,
            **{f"ps_{i}": s for i, s in enumerate(parse_statuses)},
        }
        stmt = text(
            f"select count(*) from {_TABLE} "
            "where tenant_id = :tenant_id and knowledge_base_id = :knowledge_base_id "
            f"and parse_status in ({placeholders}) and deleted_at is null"
        ).bindparams(**params)
        total = (await self._session.execute(stmt)).scalar_one()
        return int(total) if total is not None else 0


__all__ = ["KnowledgeRepository"]
