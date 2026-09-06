# Chinese API messages use fullwidth punctuation.

"""Create a manual Markdown knowledge entry.

Standalone create-variant: validates the payload (content cleanliness and
length, title, status), persists a ``manual`` knowledge row through the
merged repositories with the manual metadata JSON, and attaches tag
bindings. Publish rows are stamped ``pending`` ready for the async
processing seam; draft rows stay ``draft``.

Behaviour mirrors the upstream create-manual service:

- content is markdown-cleaned and must be non-empty and within the
  length limit;
- the title is input-validated and defaults to a timestamped name when
  blank;
- the status must be ``draft`` or ``publish`` (blank means ``draft``);
- the manual metadata (content, format, status, version, updated_at) is
  stored in the row's JSON metadata column, with any process overrides
  embedded for publish rows.
"""

from __future__ import annotations

from datetime import UTC, datetime

from src.common.exception import ValidationError
from src.common.json import JsonObject
from src.core.contracts.knowledge import Knowledge
from src.core.knowledge.documents.create_common import (
    ENABLE_STATUS_DISABLED,
    MANUAL_CONTENT_MAX_LENGTH,
    build_document_row,
    clean_markdown,
    default_channel,
    ensure_manual_file_name,
    require_knowledge_base_id,
    require_tenant_id,
    to_knowledge,
    validate_input,
    validate_knowledge_tags,
)
from src.core.knowledge.documents.types import (
    KNOWLEDGE_TYPE_MANUAL,
    MANUAL_KNOWLEDGE_FORMAT_MARKDOWN,
    MANUAL_KNOWLEDGE_STATUS_DRAFT,
    MANUAL_KNOWLEDGE_STATUS_PUBLISH,
    PARSE_STATUS_PENDING,
)
from src.core.knowledge.knowledge_bases.service.kb_service import KBService
from src.db.dao.knowledge_repository import KnowledgeRepository
from src.db.dao.knowledge_tag_repository import TagRepository


def normalize_manual_status(status: str | None) -> str:
    """Normalise a manual-knowledge status; blank defaults to ``draft``."""
    normalized = (status or "").strip().lower()
    if not normalized:
        return MANUAL_KNOWLEDGE_STATUS_DRAFT
    if normalized not in (MANUAL_KNOWLEDGE_STATUS_DRAFT, MANUAL_KNOWLEDGE_STATUS_PUBLISH):
        raise ValidationError(
            code="knowledge.status_invalid",
            message="状态仅支持 draft 或 publish",
        )
    return normalized


async def create_knowledge_from_manual(
    *,
    tenant_id: int,
    kb_id: str,
    title: str,
    content: str,
    status: str | None = None,
    tag_ids: list[str] | None = None,
    channel: str | None = None,
    process_overrides: JsonObject | None = None,
    knowledge_repo: KnowledgeRepository,
    kb_service: KBService,
    tag_repo: TagRepository | None = None,
    now: datetime | None = None,
) -> Knowledge:
    """Create a manual Markdown knowledge entry."""
    require_tenant_id(tenant_id)
    require_knowledge_base_id(kb_id)
    clean_content = clean_markdown(content)
    if not clean_content.strip():
        raise ValidationError(
            code="knowledge.content_required",
            message="内容不能为空",
        )
    if len(clean_content) > MANUAL_CONTENT_MAX_LENGTH:
        raise ValidationError(
            code="knowledge.content_too_long",
            message=f"内容长度超出限制（最多{MANUAL_CONTENT_MAX_LENGTH}个字符）",
        )
    safe_title, ok = validate_input(title)
    if not ok:
        raise ValidationError(
            code="knowledge.title_invalid",
            message="标题包含非法字符或超出长度限制",
        )
    status = normalize_manual_status(status)
    kb = await kb_service.get_knowledge_base_by_id(knowledge_base_id=kb_id)
    stamp = now or datetime.now(UTC)
    display_title = safe_title or f"Knowledge-{stamp:%Y%m%d-%H%M%S}"
    file_name = ensure_manual_file_name(display_title, stamp)
    metadata: JsonObject = {
        "content": clean_content,
        "format": MANUAL_KNOWLEDGE_FORMAT_MARKDOWN,
        "status": status,
        "version": 1,
        "updated_at": stamp.isoformat(),
    }
    if status == MANUAL_KNOWLEDGE_STATUS_PUBLISH and process_overrides is not None:
        metadata["process_overrides"] = process_overrides
    parse_status = (
        PARSE_STATUS_PENDING
        if status == MANUAL_KNOWLEDGE_STATUS_PUBLISH
        else MANUAL_KNOWLEDGE_STATUS_DRAFT
    )
    row = build_document_row(
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
        type=KNOWLEDGE_TYPE_MANUAL,
        title=display_title,
        description="",
        source=KNOWLEDGE_TYPE_MANUAL,
        channel=default_channel(channel),
        parse_status=parse_status,
        enable_status=ENABLE_STATUS_DISABLED,
        embedding_model_id=kb.embedding_model_id,
        file_name=file_name,
        file_type=KNOWLEDGE_TYPE_MANUAL,
        metadata=metadata,
        now=stamp,
    )
    # Validate tag bindings before the insert so an invalid tag id fails
    # fast without leaving a dangling knowledge row.
    validated_tag_ids: list[str] = []
    if tag_ids:
        if tag_repo is None:
            raise ValidationError(
                code="knowledge.tag_repo_required",
                message="tag_ids requires a tag repository",
            )
        validated_tag_ids = await validate_knowledge_tags(
            tenant_id=tenant_id,
            kb_id=kb_id,
            tag_ids=tag_ids,
            tag_repo=tag_repo,
        )
    persisted = await knowledge_repo.create(row)
    if validated_tag_ids and tag_repo is not None:
        await tag_repo.set_knowledge_tags(
            knowledge_id=persisted.id,
            tag_ids=validated_tag_ids,
        )
    return to_knowledge(persisted)


__all__ = ["create_knowledge_from_manual", "normalize_manual_status"]
