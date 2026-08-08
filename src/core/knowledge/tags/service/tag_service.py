# ruff: noqa: RUF001  # Chinese API messages use fullwidth punctuation.

"""Tag service — knowledge-base scoped tag operations.

Request-scoped: constructed per request by ``build_tag_service``; the
repositories own the per-request session. Mirrors the upstream tag
service semantics: tenant-scoped CRUD, name-conflict detection, the
unclassified-tag sort pin, the reference-guarded delete, and the
document-tag association operations.

The content cascade (deleting the chunks / documents under a tag when
``force`` or ``content_only`` is set) runs on the async task layer and
is out of scope here — the service only decides whether the tag row
itself is removed. Knowledge-base existence and tenant ownership are
validated whenever a knowledge-base repository is injected; wiring
layers that already resolved the scope may omit it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from src.common.exception import ConflictError, NotFoundError, ValidationError
from src.common.json import BindParams
from src.common.pagination import PaginationResponse
from src.core.knowledge.tags.types import UNTAGGED_TAG_NAME, TagInfo
from src.db.dao.knowledge_base_repository import KnowledgeBaseRepository
from src.db.dao.knowledge_tag_repository import TagReferenceCounts, TagRepository
from src.db.models.knowledge_tag import KnowledgeTag


class TagService:
    """Stateless tag service, constructed per request."""

    def __init__(
        self,
        *,
        tag_repo: TagRepository,
        kb_repo: KnowledgeBaseRepository | None = None,
    ) -> None:
        self._tag_repo = tag_repo
        self._kb_repo = kb_repo

    # ── List ────────────────────────────────────────────────────────

    async def list_tags(
        self,
        *,
        tenant_id: int,
        knowledge_base_id: str,
        page: int = 1,
        page_size: int = 20,
        keyword: str = "",
    ) -> PaginationResponse[TagInfo]:
        """Return one page of the knowledge base's tags plus usage stats.

        ``keyword`` filters tag names with LIKE (wildcards neutralised
        by the repository). Every listed tag carries its live-document
        and live-chunk counts.
        """
        if not knowledge_base_id:
            raise ValidationError(
                code="tag.kb_id_required",
                message="知识库ID不能为空",
            )
        await self._require_kb(tenant_id=tenant_id, knowledge_base_id=knowledge_base_id)
        rows, total = await self._tag_repo.list_by_kb(
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            page=page,
            page_size=page_size,
            keyword=keyword,
        )
        if not rows:
            return PaginationResponse(
                total=total,
                page=page,
                page_size=page_size,
                data=[],
            )
        counts = await self._tag_repo.batch_count_references(
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            tag_ids=[row.id for row in rows],
        )
        data = [self._to_info(row, counts=counts[row.id]) for row in rows]
        return PaginationResponse(total=total, page=page, page_size=page_size, data=data)

    # ── Create ──────────────────────────────────────────────────────

    async def create_tag(
        self,
        *,
        tenant_id: int,
        knowledge_base_id: str,
        name: str,
        color: str | None = None,
        sort_order: int = 0,
    ) -> TagInfo:
        """Create a tag under the knowledge base, returning its shape.

        A tag whose name equals the unclassified name is pinned to the
        front of the list (sort order ``-1``). A duplicate name within
        the knowledge base is a conflict.
        """
        clean_name = self._require_create_scope(
            knowledge_base_id=knowledge_base_id,
            name=name,
        )
        await self._require_kb(tenant_id=tenant_id, knowledge_base_id=knowledge_base_id)
        existing = await self._tag_repo.get_by_name(
            tenant_id,
            knowledge_base_id,
            clean_name,
        )
        if existing is not None:
            raise ConflictError(
                code="tag.name_conflict",
                message="标签名称已存在",
            )
        now = datetime.now(UTC)
        row = KnowledgeTag(
            id=str(uuid4()),
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            name=clean_name,
            color=color.strip() if color is not None else None,
            sort_order=-1 if clean_name == UNTAGGED_TAG_NAME else sort_order,
            created_at=now,
            updated_at=now,
        )
        persisted = await self._tag_repo.create(row)
        return TagInfo.map_from_db(persisted)

    # ── Find-or-create ──────────────────────────────────────────────

    async def find_or_create_tag_by_name(
        self,
        *,
        tenant_id: int,
        knowledge_base_id: str,
        name: str,
    ) -> TagInfo:
        """Return the existing tag with ``name``, or create it when absent."""
        clean_name = self._require_create_scope(
            knowledge_base_id=knowledge_base_id,
            name=name,
        )
        await self._require_kb(tenant_id=tenant_id, knowledge_base_id=knowledge_base_id)
        existing = await self._tag_repo.get_by_name(
            tenant_id,
            knowledge_base_id,
            clean_name,
        )
        if existing is not None:
            return TagInfo.map_from_db(existing)
        return await self.create_tag(
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            name=clean_name,
            color="",
            sort_order=0,
        )

    # ── Update ──────────────────────────────────────────────────────

    async def update_tag(
        self,
        *,
        tenant_id: int,
        tag_id: str,
        name: str | None = None,
        color: str | None = None,
        sort_order: int | None = None,
    ) -> TagInfo:
        """Patch the tag's mutable fields, returning the new shape.

        Omitted fields are left untouched; ``name`` is trimmed and must
        stay non-empty when supplied.
        """
        if not tag_id:
            raise ValidationError(
                code="tag.tag_id_required",
                message="标签ID不能为空",
            )
        row = await self._get_owned_tag(tenant_id=tenant_id, tag_id=tag_id)
        updates: BindParams = {"updated_at": datetime.now(UTC)}
        if name is not None:
            clean_name = name.strip()
            if not clean_name:
                raise ValidationError(
                    code="tag.name_required",
                    message="标签名称不能为空",
                )
            updates["name"] = clean_name
        if color is not None:
            updates["color"] = color.strip()
        if sort_order is not None:
            updates["sort_order"] = sort_order
        updated = row.model_copy(update=updates)
        persisted = await self._tag_repo.update(updated)
        return TagInfo.map_from_db(persisted)

    # ── Delete ──────────────────────────────────────────────────────

    async def delete_tag(
        self,
        *,
        tenant_id: int,
        tag_id: str,
        force: bool = False,
        content_only: bool = False,
        exclude_ids: list[str] | None = None,
    ) -> bool:
        """Delete a tag, returning whether the tag row was removed.

        ``content_only`` keeps the tag and clears only its content
        (the async cascade is out of scope here). ``force`` bypasses
        the reference guard; a non-empty ``exclude_ids`` keeps the tag
        because excluded content still references it.
        """
        if not tag_id:
            raise ValidationError(
                code="tag.tag_id_required",
                message="标签ID不能为空",
            )
        tag = await self._get_owned_tag(tenant_id=tenant_id, tag_id=tag_id)
        knowledge_count, chunk_count = await self._tag_repo.count_references(
            tenant_id=tenant_id,
            knowledge_base_id=tag.knowledge_base_id,
            tag_id=tag_id,
        )
        if content_only:
            return False
        if not force and (knowledge_count > 0 or chunk_count > 0):
            raise ValidationError(
                code="tag.has_references",
                message="标签仍有知识或FAQ条目引用，无法删除",
            )
        if exclude_ids:
            return False
        return await self._tag_repo.delete(tenant_id=tenant_id, id=tag_id)

    # ── Document-tag bind / unbind ──────────────────────────────────

    async def set_knowledge_tags(
        self,
        *,
        knowledge_id: str,
        tag_ids: list[str],
    ) -> None:
        """Replace a document's tag bindings in one delete + insert."""
        await self._tag_repo.set_knowledge_tags(
            knowledge_id=knowledge_id,
            tag_ids=tag_ids,
        )

    async def get_knowledge_tags(
        self,
        knowledge_ids: list[str],
    ) -> dict[str, list[TagInfo]]:
        """Return each document's tags, keyed by knowledge id."""
        grouped = await self._tag_repo.get_knowledge_tags(knowledge_ids)
        return {
            knowledge_id: [TagInfo.map_from_db(row) for row in rows]
            for knowledge_id, rows in grouped.items()
        }

    async def delete_knowledge_tag_relations(self, knowledge_id: str) -> int:
        """Remove every tag binding of a document; return the row count."""
        return await self._tag_repo.delete_knowledge_tag_relations(knowledge_id)

    # ── Shared helpers ──────────────────────────────────────────────

    @staticmethod
    def _require_create_scope(*, knowledge_base_id: str, name: str) -> str:
        """Validate and return the trimmed name for a create operation."""
        clean_name = name.strip()
        if not knowledge_base_id or not clean_name:
            raise ValidationError(
                code="tag.kb_id_and_name_required",
                message="知识库ID和标签名称不能为空",
            )
        return clean_name

    async def _require_kb(self, *, tenant_id: int, knowledge_base_id: str) -> None:
        """Raise when the knowledge base is absent or not owned by the tenant.

        A no-op when no knowledge-base repository is injected — wiring
        that already resolved the scope relies on its own guard.
        """
        if self._kb_repo is None:
            return
        kb = await self._kb_repo.get_by_id_and_tenant(knowledge_base_id, tenant_id)
        if kb is None:
            raise NotFoundError(
                code="tag.kb_not_found",
                message="知识库不存在",
            )

    async def _get_owned_tag(self, *, tenant_id: int, tag_id: str) -> KnowledgeTag:
        """Return one tenant-scoped tag; raise ``tag.not_found`` when absent."""
        row = await self._tag_repo.get_by_id(tenant_id, tag_id)
        if row is None:
            raise NotFoundError(
                code="tag.not_found",
                message="标签不存在",
            )
        return row

    @staticmethod
    def _to_info(row: KnowledgeTag, *, counts: TagReferenceCounts) -> TagInfo:
        """Project a row onto the service shape with reference counts."""
        return TagInfo.map_from_db(
            row,
            knowledge_count=counts.knowledge_count,
            chunk_count=counts.chunk_count,
        )


__all__ = ["TagService"]
