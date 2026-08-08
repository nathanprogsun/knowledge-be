"""Wire-shape conversion for the knowledge-tag endpoints.

Projects the service DTO (``TagInfo``) onto the frozen tag contracts
(``Tag`` / ``TagList``) and defines the endpoint envelopes plus the
delete-request body. The tag wire shape carries usage counts; the
storage ``seq_id`` stays off the wire.

The delete-request body mirrors the upstream shape (``exclude_ids`` are
chunk seq ids). Chunk-level exclusion resolution is a domain-layer
concern; the service treats a non-empty exclude set as keep-tag, so the
view passes the raw ids through without resolving them.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src.common.pagination import PaginationResponse
from src.core.contracts.knowledge import Tag, TagList
from src.core.knowledge.tags.types import TagInfo


class DeleteTagRequest(BaseModel):
    """``{"exclude_ids": [...]}`` - delete options body.

    ``exclude_ids`` lists chunk seq ids to exclude from the delete.
    """

    model_config = ConfigDict(frozen=True)

    exclude_ids: list[int] = Field(default_factory=list)


class TagEnvelope(BaseModel):
    """``{"success": true, "data": {...}}`` - single-tag responses."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: Tag


class TagListEnvelope(BaseModel):
    """``{"success": true, "data": {total, page, page_size, data}}`` - list responses."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: TagList


class DeleteTagResponse(BaseModel):
    """``{"success": true}`` - delete acknowledgement."""

    model_config = ConfigDict(frozen=True)

    success: bool


def tag_to_contract(info: TagInfo) -> Tag:
    """Project a tag DTO onto the frozen wire contract."""
    return Tag(
        id=info.id,
        tenant_id=info.tenant_id,
        knowledge_base_id=info.knowledge_base_id,
        name=info.name,
        color=info.color,
        sort_order=info.sort_order,
        knowledge_count=info.knowledge_count,
        chunk_count=info.chunk_count,
        created_at=info.created_at,
        updated_at=info.updated_at,
    )


def tag_envelope(info: TagInfo) -> TagEnvelope:
    """Wrap one tag in the success envelope."""
    return TagEnvelope(success=True, data=tag_to_contract(info))


def tag_list_envelope(page: PaginationResponse[TagInfo]) -> TagListEnvelope:
    """Wrap one tag page in the success envelope."""
    return TagListEnvelope(
        success=True,
        data=TagList(
            total=page.total,
            page=page.page,
            page_size=page.page_size,
            data=[tag_to_contract(info) for info in page.data],
        ),
    )


__all__ = [
    "DeleteTagRequest",
    "DeleteTagResponse",
    "TagEnvelope",
    "TagListEnvelope",
    "tag_envelope",
    "tag_list_envelope",
    "tag_to_contract",
]
