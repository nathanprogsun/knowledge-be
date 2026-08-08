# ruff: noqa: RUF001  # Chinese API messages use fullwidth punctuation.

"""Create a knowledge entry from a URL source.

Standalone create-variant: routes a URL import to the direct-file path
when the URL (or the caller's ``file_name`` / ``file_type`` hints) points
at a downloadable file, otherwise creates a web ``url`` knowledge row.
The record is persisted through the merged repositories; the file
download, storage wiring, and the async processing enqueue are deferred
seams that land with the storage and worker domains.

Behaviour mirrors the upstream create-from-URL service:

- a URL whose path carries a supported import extension (or carries
  user hints) is treated as a file download (``file_url`` row type) and
  is rejected for FAQ knowledge bases;
- plain URLs become ``url`` rows with ``file_type`` ``html``;
- validation rejects non-http(s) URLs, unsafe / SSRF-blocked hosts, and
  unsupported file types;
- a duplicate within the knowledge base refreshes the existing row's
  timestamps and surfaces as ``ConflictError``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol

from src.common.exception import ConflictError, ValidationError
from src.common.oidc_client import validate_ssrf_safe_url
from src.core.contracts.knowledge import Knowledge
from src.core.knowledge.documents.create_common import (
    ENABLE_STATUS_DISABLED,
    build_document_row,
    calculate_str,
    default_channel,
    extract_file_name_from_url,
    find_duplicate_document,
    get_file_type,
    is_file_url,
    is_safe_url,
    is_valid_http_url,
    normalize_file_extension,
    refresh_duplicate_timestamps,
    require_knowledge_base_id,
    require_tenant_id,
    to_knowledge,
    validate_import_file_type,
    validate_input,
    validate_knowledge_tags,
)
from src.core.knowledge.documents.types import PARSE_STATUS_PENDING
from src.core.knowledge.knowledge_bases.service.kb_service import KBService
from src.core.knowledge.knowledge_bases.types import KNOWLEDGE_BASE_TYPE_FAQ
from src.db.dao.knowledge_repository import KnowledgeRepository
from src.db.dao.knowledge_tag_repository import TagRepository
from src.db.models.knowledge import Document

_INVALID_URL_MESSAGE = "无效或非安全的URL"


class URLGuard(Protocol):
    """SSRF guard injected by callers; defaults to the shared validator."""

    async def __call__(self, url: str) -> None: ...


async def _guard_ssrf(url: str, guard: URLGuard | None) -> None:
    """Run the SSRF guard, classifying a rejection as an invalid URL."""
    checker = guard if guard is not None else validate_ssrf_safe_url
    try:
        await checker(url)
    except ValidationError as exc:
        raise ValidationError(
            code="knowledge.invalid_url",
            message=_INVALID_URL_MESSAGE,
        ) from exc


def _require_url(url: str) -> None:
    """Reject a blank URL at the create boundary."""
    if not url or not url.strip():
        raise ValidationError(
            code="knowledge.url_required",
            message="URL不能为空",
        )


async def _raise_duplicate(
    existing: Document,
    *,
    code: str,
    message: str,
    knowledge_repo: KnowledgeRepository,
    now: datetime,
) -> None:
    """Refresh the duplicate row's timestamps, then raise the collision."""
    refreshed = await refresh_duplicate_timestamps(
        knowledge_repo=knowledge_repo,
        knowledge_id=existing.id,
        now=now,
    )
    target = refreshed or existing
    raise ConflictError(
        code=code,
        message=message,
        details={"knowledge_id": target.id, "title": target.title},
    )


async def _attach_tags(
    tenant_id: int,
    kb_id: str,
    row: Document,
    tag_ids: list[str] | None,
    tag_repo: TagRepository | None,
) -> None:
    """Validate and attach tag bindings after the row is persisted."""
    if not tag_ids:
        return
    if tag_repo is None:
        raise ValidationError(
            code="knowledge.tag_repo_required",
            message="tag_ids requires a tag repository",
        )
    validated = await validate_knowledge_tags(
        tenant_id=tenant_id,
        kb_id=kb_id,
        tag_ids=tag_ids,
        tag_repo=tag_repo,
    )
    if validated:
        await tag_repo.set_knowledge_tags(knowledge_id=row.id, tag_ids=validated)


async def _create_from_web_url(
    *,
    tenant_id: int,
    kb_id: str,
    url: str,
    title: str | None,
    tag_ids: list[str] | None,
    channel: str | None,
    knowledge_repo: KnowledgeRepository,
    kb_service: KBService,
    tag_repo: TagRepository | None,
    url_guard: URLGuard | None,
    now: datetime,
) -> Knowledge:
    """Create a web ``url`` knowledge row."""
    kb = await kb_service.get_knowledge_base_by_id(knowledge_base_id=kb_id)
    if not is_valid_http_url(url) or not is_safe_url(url):
        raise ValidationError(code="knowledge.invalid_url", message=_INVALID_URL_MESSAGE)
    await _guard_ssrf(url, url_guard)
    file_hash = calculate_str(url)
    existing = await find_duplicate_document(
        tenant_id=tenant_id,
        kb_id=kb_id,
        doc_type="url",
        source=url,
        file_hash=file_hash,
        knowledge_repo=knowledge_repo,
    )
    if existing is not None:
        await _raise_duplicate(
            existing,
            code="knowledge.duplicate_url",
            message=f"URL already exists: {existing.source}",
            knowledge_repo=knowledge_repo,
            now=now,
        )
    row = build_document_row(
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
        type="url",
        title=title or "",
        source=url,
        channel=default_channel(channel),
        parse_status=PARSE_STATUS_PENDING,
        enable_status=ENABLE_STATUS_DISABLED,
        embedding_model_id=kb.embedding_model_id,
        file_type="html",
        file_hash=file_hash,
        now=now,
    )
    persisted = await knowledge_repo.create(row)
    await _attach_tags(tenant_id, kb_id, persisted, tag_ids, tag_repo)
    return to_knowledge(persisted)


async def _create_from_file_url(
    *,
    tenant_id: int,
    kb_id: str,
    file_url: str,
    file_name: str | None,
    file_type: str | None,
    title: str | None,
    tag_ids: list[str] | None,
    channel: str | None,
    knowledge_repo: KnowledgeRepository,
    kb_service: KBService,
    tag_repo: TagRepository | None,
    url_guard: URLGuard | None,
    now: datetime,
) -> Knowledge:
    """Create a ``file_url`` knowledge row for a downloadable URL."""
    kb = await kb_service.get_knowledge_base_by_id(knowledge_base_id=kb_id)
    if kb.type == KNOWLEDGE_BASE_TYPE_FAQ:
        raise ValidationError(
            code="knowledge.faq_file_unsupported",
            message="FAQ 知识库不支持文件上传，请使用 FAQ 导入功能",
        )
    if not is_valid_http_url(file_url) or not is_safe_url(file_url):
        raise ValidationError(code="knowledge.invalid_url", message=_INVALID_URL_MESSAGE)
    await _guard_ssrf(file_url, url_guard)
    resolved_name = file_name or extract_file_name_from_url(file_url)
    if resolved_name:
        safe_name, ok = validate_input(resolved_name)
        if not ok:
            raise ValidationError(
                code="knowledge.file_name_invalid",
                message="文件名包含非法字符",
            )
        resolved_name = safe_name
    resolved_type = normalize_file_extension(file_type or get_file_type(resolved_name))
    display_name = resolved_name or title or extract_file_name_from_url(file_url) or file_url
    file_hash = calculate_str(file_url)
    existing = await find_duplicate_document(
        tenant_id=tenant_id,
        kb_id=kb_id,
        doc_type="file_url",
        source=file_url,
        file_hash=file_hash,
        knowledge_repo=knowledge_repo,
    )
    if existing is not None:
        await _raise_duplicate(
            existing,
            code="knowledge.duplicate_file",
            message=f"File already exists: {existing.file_name or existing.title}",
            knowledge_repo=knowledge_repo,
            now=now,
        )
    validate_import_file_type(resolved_type)
    row = build_document_row(
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
        type="file_url",
        title=title or display_name,
        source=file_url,
        channel=default_channel(channel),
        parse_status=PARSE_STATUS_PENDING,
        enable_status=ENABLE_STATUS_DISABLED,
        embedding_model_id=kb.embedding_model_id,
        file_name=display_name,
        file_type=resolved_type,
        file_hash=file_hash,
        now=now,
    )
    persisted = await knowledge_repo.create(row)
    await _attach_tags(tenant_id, kb_id, persisted, tag_ids, tag_repo)
    return to_knowledge(persisted)


async def create_knowledge_from_url(
    *,
    tenant_id: int,
    kb_id: str,
    url: str,
    file_name: str | None = None,
    file_type: str | None = None,
    enable_multimodel: bool | None = None,
    title: str | None = None,
    tag_ids: list[str] | None = None,
    channel: str | None = None,
    knowledge_repo: KnowledgeRepository,
    kb_service: KBService,
    tag_repo: TagRepository | None = None,
    url_guard: URLGuard | None = None,
    now: datetime | None = None,
) -> Knowledge:
    """Create a knowledge entry from a URL, routing file URLs separately.

    ``enable_multimodel`` is reserved for the deferred process-config
    resolution and does not affect the persisted row.
    """
    require_tenant_id(tenant_id)
    require_knowledge_base_id(kb_id)
    _require_url(url)
    stamp = now or datetime.now(UTC)
    if is_file_url(url, file_name or "", file_type or ""):
        return await _create_from_file_url(
            tenant_id=tenant_id,
            kb_id=kb_id,
            file_url=url,
            file_name=file_name,
            file_type=file_type,
            title=title,
            tag_ids=tag_ids,
            channel=channel,
            knowledge_repo=knowledge_repo,
            kb_service=kb_service,
            tag_repo=tag_repo,
            url_guard=url_guard,
            now=stamp,
        )
    return await _create_from_web_url(
        tenant_id=tenant_id,
        kb_id=kb_id,
        url=url,
        title=title,
        tag_ids=tag_ids,
        channel=channel,
        knowledge_repo=knowledge_repo,
        kb_service=kb_service,
        tag_repo=tag_repo,
        url_guard=url_guard,
        now=stamp,
    )


__all__ = ["URLGuard", "create_knowledge_from_url"]
