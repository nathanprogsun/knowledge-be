"""Audit action constants — dot-namespaced string literals.

Mirrors ``internal/types/audit_log.go::AuditAction``. Actions are
grouped by area (``rbac.*``, ``vector_store.*``, ``system.*``,
``kb.*``, ``knowledge.*``, ``tag.*``, ``datasource.*``, ``kb.share_*``,
``wiki.*``, ``faq.*``, ``opensearch.*``) so each area can be extended
without colliding with existing events.

Only ``system.setting_changed`` is emitted today (via
``SystemSettingService.Update``); the full set is defined here so
consumers reference ``AuditAction.KB_CREATED`` etc. without redefining
the enum.
"""

from __future__ import annotations

from typing import Final


class AuditAction:
    """Dot-namespaced audit action constants (no enum — plain ``str``)."""

    # ── RBAC ───────────────────────────────────────────────────────────
    MEMBER_ADDED: Final[str] = "rbac.member_added"
    MEMBER_REMOVED: Final[str] = "rbac.member_removed"
    MEMBER_ROLE_CHANGED: Final[str] = "rbac.member_role_changed"
    MEMBER_LEFT: Final[str] = "rbac.member_left"
    ACCESS_DENIED: Final[str] = "rbac.access_denied"
    INVITATION_SENT: Final[str] = "rbac.invitation_sent"
    INVITATION_ACCEPTED: Final[str] = "rbac.invitation_accepted"
    INVITATION_DECLINED: Final[str] = "rbac.invitation_declined"
    INVITATION_REVOKED: Final[str] = "rbac.invitation_revoked"
    INVITATION_EXPIRED: Final[str] = "rbac.invitation_expired"

    # ── VectorStore ────────────────────────────────────────────────────
    VECTOR_STORE_CREATED: Final[str] = "vector_store.created"
    VECTOR_STORE_UPDATED: Final[str] = "vector_store.updated"
    VECTOR_STORE_DELETED: Final[str] = "vector_store.deleted"

    # ── OpenSearch ─────────────────────────────────────────────────────
    OPENSEARCH_INDEX_CREATED: Final[str] = "opensearch.index_created"
    OPENSEARCH_INDEX_DELETED: Final[str] = "opensearch.index_deleted"
    OPENSEARCH_REINDEX_EXECUTED: Final[str] = "opensearch.reindex_executed"

    # ── System ─────────────────────────────────────────────────────────
    SYSTEM_SETTING_CHANGED: Final[str] = "system.setting_changed"
    SYSTEM_ADMIN_PROMOTED: Final[str] = "system.admin_promoted"
    SYSTEM_ADMIN_REVOKED: Final[str] = "system.admin_revoked"
    SYSTEM_USER_PASSWORD_RESET: Final[str] = "system.user_password_reset"
    SYSTEM_API_KEY_CREATED: Final[str] = "system.api_key_created"
    SYSTEM_API_KEY_REVOKED: Final[str] = "system.api_key_revoked"
    SYSTEM_QUEUE_TASK_RETRIED: Final[str] = "system.queue_task_retried"
    SYSTEM_QUEUE_TASK_DELETED: Final[str] = "system.queue_task_deleted"
    SYSTEM_QUEUE_TASK_RUN_NOW: Final[str] = "system.queue_task_run_now"
    SYSTEM_QUEUE_TASK_CANCELLED: Final[str] = "system.queue_task_cancelled"
    SYSTEM_QUEUE_ARCHIVED_PURGED: Final[str] = "system.queue_archived_purged"

    # ── Knowledge base ─────────────────────────────────────────────────
    KB_CREATED: Final[str] = "kb.created"
    KB_UPDATED: Final[str] = "kb.updated"
    KB_DELETED: Final[str] = "kb.deleted"
    KB_DUPLICATED: Final[str] = "kb.duplicated"
    KB_CLONE_STARTED: Final[str] = "kb.clone_started"
    KB_CLONE_COMPLETED: Final[str] = "kb.clone_completed"
    KB_CLONE_FAILED: Final[str] = "kb.clone_failed"
    KB_SHARE_ADDED: Final[str] = "kb.share_added"
    KB_SHARE_PERMISSION_CHANGED: Final[str] = "kb.share_permission_changed"
    KB_SHARE_REMOVED: Final[str] = "kb.share_removed"

    # ── Knowledge item ─────────────────────────────────────────────────
    KNOWLEDGE_CREATED: Final[str] = "knowledge.created"
    KNOWLEDGE_UPDATED: Final[str] = "knowledge.updated"
    KNOWLEDGE_DELETED: Final[str] = "knowledge.deleted"
    KNOWLEDGE_BATCH_DELETED: Final[str] = "knowledge.batch_deleted"
    KNOWLEDGE_REPARSE_STARTED: Final[str] = "knowledge.reparse_started"
    KNOWLEDGE_PARSE_CANCELED: Final[str] = "knowledge.parse_canceled"
    KNOWLEDGE_MOVE_STARTED: Final[str] = "knowledge.move_started"
    KNOWLEDGE_MOVE_COMPLETED: Final[str] = "knowledge.move_completed"
    KNOWLEDGE_MOVE_FAILED: Final[str] = "knowledge.move_failed"

    # ── Tag ────────────────────────────────────────────────────────────
    TAG_CREATED: Final[str] = "tag.created"
    TAG_UPDATED: Final[str] = "tag.updated"
    TAG_DELETED: Final[str] = "tag.deleted"

    # ── DataSource ─────────────────────────────────────────────────────
    DATASOURCE_CREATED: Final[str] = "datasource.created"
    DATASOURCE_UPDATED: Final[str] = "datasource.updated"
    DATASOURCE_DELETED: Final[str] = "datasource.deleted"
    DATASOURCE_SYNC_STARTED: Final[str] = "datasource.sync_started"
    DATASOURCE_SYNC_COMPLETED: Final[str] = "datasource.sync_completed"
    DATASOURCE_SYNC_FAILED: Final[str] = "datasource.sync_failed"
    DATASOURCE_PAUSED: Final[str] = "datasource.paused"
    DATASOURCE_RESUMED: Final[str] = "datasource.resumed"

    # ── Wiki / FAQ ─────────────────────────────────────────────────────
    WIKI_CONTENT_CHANGED: Final[str] = "wiki.content_changed"
    FAQ_IMPORT_STARTED: Final[str] = "faq.import_started"
    FAQ_IMPORT_COMPLETED: Final[str] = "faq.import_completed"
    FAQ_IMPORT_FAILED: Final[str] = "faq.import_failed"


class AuditOutcome:
    """Audit outcome constants (``internal/types/audit_log.go``)."""

    SUCCESS: Final[str] = "success"
    ACCEPTED: Final[str] = "accepted"
    DENIED: Final[str] = "denied"
    FAILED: Final[str] = "failed"
    PARTIAL: Final[str] = "partial"
    CANCELED: Final[str] = "canceled"


__all__ = ["AuditAction", "AuditOutcome"]
