"""Unit tests for the PR-156 TableModel additions.

Covers the six new tables (``tenant_disabled_shared_agents``,
``user_kb_pins``, ``user_resource_favorites``, ``wiki_log_entries``,
``wiki_page_issues``, ``wiki_page_revisions``) and the ``api_key``
column added to ``Tenant``. Each model is exercised against its
declared ``ClassVar`` metadata (``table``, ``primary_keys``,
``json_columns``, ``db_generated_columns``) and one round-trip from a
SQL-shaped row dict.
"""

from __future__ import annotations

from datetime import UTC, datetime

from src.db.models.auth.user_kb_pins import UserKBPin
from src.db.models.auth.user_resource_favorites import UserResourceFavorite
from src.db.models.tenants.tenant_disabled_shared_agents import (
    TenantDisabledSharedAgent,
)
from src.db.models.tenants.tenants import Tenant
from src.db.models.wiki_log_entry import WikiLogEntry
from src.db.models.wiki_page_issue import WikiPageIssue
from src.db.models.wiki_page_revision import WikiPageRevision

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


# ── tenant_disabled_shared_agents ────────────────────────────────────────


def test_tenant_disabled_shared_agent_metadata() -> None:
    assert TenantDisabledSharedAgent.table == "tenant_disabled_shared_agents"
    assert TenantDisabledSharedAgent.primary_keys == (
        "tenant_id",
        "agent_id",
        "source_tenant_id",
    )
    assert TenantDisabledSharedAgent.json_columns == ()


def test_tenant_disabled_shared_agent_round_trip() -> None:
    row = {
        "tenant_id": 7,
        "agent_id": "agent-1",
        "source_tenant_id": 9,
        "created_at": _NOW,
    }
    model = TenantDisabledSharedAgent.from_row(row)
    assert model.tenant_id == 7
    assert model.agent_id == "agent-1"
    assert model.source_tenant_id == 9
    assert model.created_at == _NOW


# ── user_kb_pins ─────────────────────────────────────────────────────────


def test_user_kb_pin_metadata() -> None:
    assert UserKBPin.table == "user_kb_pins"
    assert UserKBPin.primary_keys == ("tenant_id", "user_id", "kb_id")
    assert UserKBPin.json_columns == ()


def test_user_kb_pin_round_trip() -> None:
    row = {
        "tenant_id": 1,
        "user_id": "user-1",
        "kb_id": "kb-1",
        "pinned_at": _NOW,
    }
    model = UserKBPin.from_row(row)
    assert model.tenant_id == 1
    assert model.user_id == "user-1"
    assert model.kb_id == "kb-1"
    assert model.pinned_at == _NOW


# ── user_resource_favorites ──────────────────────────────────────────────


def test_user_resource_favorite_metadata() -> None:
    assert UserResourceFavorite.table == "user_resource_favorites"
    assert UserResourceFavorite.primary_keys == (
        "user_id",
        "tenant_id",
        "resource_type",
        "resource_id",
    )
    assert UserResourceFavorite.json_columns == ()
    # ``created_at`` carries a DB default; excluded from INSERT.
    assert UserResourceFavorite.insert_sql_column_list() == (
        "user_id",
        "tenant_id",
        "resource_type",
        "resource_id",
    )


def test_user_resource_favorite_round_trip() -> None:
    row = {
        "user_id": "user-1",
        "tenant_id": 1,
        "resource_type": "kb",
        "resource_id": "kb-1",
        "created_at": _NOW,
    }
    model = UserResourceFavorite.from_row(row)
    assert model.user_id == "user-1"
    assert model.resource_type == "kb"
    assert model.created_at == _NOW


# ── wiki_log_entries ─────────────────────────────────────────────────────


def test_wiki_log_entry_metadata() -> None:
    assert WikiLogEntry.table == "wiki_log_entries"
    assert WikiLogEntry.primary_keys == ("id",)
    assert WikiLogEntry.json_columns == ("pages_affected",)
    # ``id`` is DB-assigned.
    assert WikiLogEntry.insert_sql_column_list() == (
        "tenant_id",
        "knowledge_base_id",
        "action",
        "knowledge_id",
        "doc_title",
        "summary",
        "pages_affected",
        "created_at",
    )


def test_wiki_log_entry_round_trip() -> None:
    row = {
        "id": 42,
        "tenant_id": 1,
        "knowledge_base_id": "kb-1",
        "action": "ingest",
        "knowledge_id": "k-1",
        "doc_title": "Acme",
        "summary": "ingested",
        "pages_affected": ["page-1", "page-2"],
        "created_at": _NOW,
    }
    model = WikiLogEntry.from_row(row)
    assert model.id == 42
    assert model.action == "ingest"
    assert model.pages_affected == ["page-1", "page-2"]


# ── wiki_page_issues ─────────────────────────────────────────────────────


def test_wiki_page_issue_metadata() -> None:
    assert WikiPageIssue.table == "wiki_page_issues"
    assert WikiPageIssue.primary_keys == ("id",)
    assert WikiPageIssue.json_columns == ("suspected_knowledge_ids",)
    # ``id`` is caller-assigned (UUID); all columns are inserted.
    assert "id" in WikiPageIssue.insert_sql_column_list()


def test_wiki_page_issue_round_trip() -> None:
    row = {
        "id": "issue-1",
        "tenant_id": 1,
        "knowledge_base_id": "kb-1",
        "slug": "entity/acme",
        "issue_type": "stale_link",
        "description": "broken ref",
        "suspected_knowledge_ids": ["k-1", "k-2"],
        "status": "pending",
        "reported_by": "user-1",
        "created_at": _NOW,
        "updated_at": _NOW,
        "deleted_at": None,
    }
    model = WikiPageIssue.from_row(row)
    assert model.id == "issue-1"
    assert model.issue_type == "stale_link"
    assert model.suspected_knowledge_ids == ["k-1", "k-2"]
    assert model.deleted_at is None


# ── wiki_page_revisions ──────────────────────────────────────────────────


def test_wiki_page_revision_metadata() -> None:
    assert WikiPageRevision.table == "wiki_page_revisions"
    assert WikiPageRevision.primary_keys == ("id",)
    assert WikiPageRevision.json_columns == ("aliases",)


def test_wiki_page_revision_round_trip() -> None:
    row = {
        "id": "rev-1",
        "tenant_id": 1,
        "knowledge_base_id": "kb-1",
        "page_id": "page-1",
        "slug": "entity/acme",
        "version": 2,
        "title": "Acme",
        "page_type": "entity",
        "status": "published",
        "content": "body",
        "summary": "summary",
        "aliases": ["acme"],
        "edit_source": "user",
        "editor_id": "user-1",
        "edited_at": _NOW,
        "created_at": _NOW,
    }
    model = WikiPageRevision.from_row(row)
    assert model.id == "rev-1"
    assert model.version == 2
    assert model.aliases == ["acme"]
    assert model.edit_source == "user"


# ── tenants.api_key ──────────────────────────────────────────────────────


def test_tenant_api_key_defaults_to_empty_string() -> None:
    row = {
        "id": 1,
        "name": "Acme",
        "created_at": _NOW,
        "updated_at": _NOW,
    }
    model = Tenant.from_row(row)
    assert model.api_key == ""


def test_tenant_api_key_round_trip() -> None:
    row = {
        "id": 1,
        "name": "Acme",
        "api_key": "secret-key",
        "created_at": _NOW,
        "updated_at": _NOW,
    }
    model = Tenant.from_row(row)
    assert model.api_key == "secret-key"