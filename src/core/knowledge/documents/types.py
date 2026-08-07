"""Domain value types for the document domain.

Mirrors the upstream knowledge-entity constants and the list-filter
shape. The storage row lives in ``src.db.models.knowledge.Document``; the
service-facing wire projection is added with the service layer.

What this module adds:

- knowledge-type constants (``manual`` / ``faq``)
- parse-status and summary-status constants
- ingestion-channel constants
- manual-knowledge format / status constants
- ``DocumentListFilter`` — the optional dimensions for paged document
  listing (empty / ``None`` means "no filter on that dimension").

Field and JSON names match the upstream entity exactly.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict

# ── Knowledge type ───────────────────────────────────────────────────

KNOWLEDGE_TYPE_MANUAL = "manual"
KNOWLEDGE_TYPE_FAQ = "faq"

# ── Parse status ─────────────────────────────────────────────────────
# ``pending`` -> ``processing`` -> ``finalizing`` -> ``completed``.
# ``finalizing`` means the primary parse has finished but enrichment
# subtasks (summary, question generation, graph extract) are still in
# flight; the row is queryable for vector search. ``deleting`` and
# ``cancelled`` are short-circuit states for in-flight / queued tasks.

PARSE_STATUS_PENDING = "pending"
PARSE_STATUS_PROCESSING = "processing"
PARSE_STATUS_FINALIZING = "finalizing"
PARSE_STATUS_COMPLETED = "completed"
PARSE_STATUS_FAILED = "failed"
PARSE_STATUS_DELETING = "deleting"
PARSE_STATUS_CANCELLED = "cancelled"

PARSE_STATUSES: frozenset[str] = frozenset(
    {
        PARSE_STATUS_PENDING,
        PARSE_STATUS_PROCESSING,
        PARSE_STATUS_FINALIZING,
        PARSE_STATUS_COMPLETED,
        PARSE_STATUS_FAILED,
        PARSE_STATUS_DELETING,
        PARSE_STATUS_CANCELLED,
    }
)

# ── Summary status ───────────────────────────────────────────────────

SUMMARY_STATUS_NONE = "none"
SUMMARY_STATUS_PENDING = "pending"
SUMMARY_STATUS_PROCESSING = "processing"
SUMMARY_STATUS_COMPLETED = "completed"
SUMMARY_STATUS_FAILED = "failed"

SUMMARY_STATUSES: frozenset[str] = frozenset(
    {
        SUMMARY_STATUS_NONE,
        SUMMARY_STATUS_PENDING,
        SUMMARY_STATUS_PROCESSING,
        SUMMARY_STATUS_COMPLETED,
        SUMMARY_STATUS_FAILED,
    }
)

# ── Ingestion channel ────────────────────────────────────────────────

CHANNEL_WEB = "web"
CHANNEL_API = "api"
CHANNEL_BROWSER_EXTENSION = "browser_extension"
CHANNEL_WECHAT = "wechat"
CHANNEL_WECOM = "wecom"
CHANNEL_FEISHU = "feishu"
CHANNEL_DINGTALK = "dingtalk"
CHANNEL_SLACK = "slack"
CHANNEL_IM = "im"
CHANNEL_NOTION = "notion"
CHANNEL_YUQUE = "yuque"
CHANNEL_RSS = "rss"

# ── Manual-knowledge format / status ─────────────────────────────────

MANUAL_KNOWLEDGE_FORMAT_MARKDOWN = "markdown"
MANUAL_KNOWLEDGE_STATUS_DRAFT = "draft"
MANUAL_KNOWLEDGE_STATUS_PUBLISH = "publish"


class DocumentListFilter(BaseModel):
    """Optional dimensions for paged document listing.

    All fields are optional; empty / ``None`` means "no filter on that
    dimension". Mirrors the upstream list-filter struct field-for-field.

    ``file_type`` and ``source`` share the same special-case routing onto
    the ``type`` column for the values ``manual`` / ``url``, so callers
    can filter "manually created" / "URL imported" entries with either
    control.
    """

    model_config = ConfigDict(frozen=True)

    tag_ids: list[str] = []
    keyword: str | None = None
    file_type: str | None = None
    parse_status: str | None = None
    source: str | None = None
    updated_from: datetime | None = None
    updated_to: datetime | None = None


__all__ = [
    "CHANNEL_API",
    "CHANNEL_BROWSER_EXTENSION",
    "CHANNEL_DINGTALK",
    "CHANNEL_FEISHU",
    "CHANNEL_IM",
    "CHANNEL_NOTION",
    "CHANNEL_RSS",
    "CHANNEL_SLACK",
    "CHANNEL_WEB",
    "CHANNEL_WECHAT",
    "CHANNEL_WECOM",
    "CHANNEL_YUQUE",
    "KNOWLEDGE_TYPE_FAQ",
    "KNOWLEDGE_TYPE_MANUAL",
    "MANUAL_KNOWLEDGE_FORMAT_MARKDOWN",
    "MANUAL_KNOWLEDGE_STATUS_DRAFT",
    "MANUAL_KNOWLEDGE_STATUS_PUBLISH",
    "PARSE_STATUSES",
    "PARSE_STATUS_CANCELLED",
    "PARSE_STATUS_COMPLETED",
    "PARSE_STATUS_DELETING",
    "PARSE_STATUS_FAILED",
    "PARSE_STATUS_FINALIZING",
    "PARSE_STATUS_PENDING",
    "PARSE_STATUS_PROCESSING",
    "SUMMARY_STATUSES",
    "SUMMARY_STATUS_COMPLETED",
    "SUMMARY_STATUS_FAILED",
    "SUMMARY_STATUS_NONE",
    "SUMMARY_STATUS_PENDING",
    "SUMMARY_STATUS_PROCESSING",
    "DocumentListFilter",
]
