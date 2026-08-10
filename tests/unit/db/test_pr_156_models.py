"""Unit tests for the storage rows added in PR-156.

These tests pin the column shape, primary-key composition, JSON
column declarations, and ``db_generated_columns`` carve-out of the
six new tables plus the ``api_key`` column added to ``tenants``.
The shape of every test mirrors ``tests/unit/util/test_table_model.py``
(AAA, frozen, fail-fast on schema drift) so a future drift lands
here before it reaches SQL.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import ClassVar

import pytest

from src.common.json import JsonObject
from src.db.models.tenant_disabled_shared_agent import TenantDisabledSharedAgent
from src.db.models.tenants.tenants import Tenant
from src.db.models.user_kb_pin import UserKBPin
from src.db.models.user_resource_favorite import UserResourceFavorite
from src.db.models.wiki_log_entry import WikiLogEntry
from src.db.models.wiki_page import WikiPageIssue, WikiPageRevision

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


# ── tenant_disabled_shared_agents ─────────────────────────────────────


def test_tenant_disabled_shared_agent_metadata() -> None:
    assert TenantDisabledSharedAgent.table == "tenant_disabled_shared_agents"
    assert TenantDisabledSharedAgent.primary_keys == (
        "tenant_id",
        "agent_id",
        "source_tenant_id",
    )
    assert TenantDisabledSharedAgent.json_columns == ()


def test_tenant_disabled_shared_agent_pk_extraction() -> None:
    row = TenantDisabledSharedAgent(
        tenant_id=1,
        agent_id="agent-1",
        source_tenant_id=2,
        created_at=_NOW,
    )
    assert row.primary_key_to_value() == {
        "tenant_id": 1,
        "agent_id": "agent-1",
        "source_tenant_id": 2,
    }


# ── user_kb_pins ──────────────────────────────────────────────────────


def test_user_kb_pin_metadata() -> None:
    assert UserKBPin.table == "user_kb_pins"
    assert UserKBPin.primary_keys == ("tenant_id", "user_id", "kb_id")
    assert UserKBPin.json_columns == ()
    # No DB-generated columns; the application supplies every value.
    assert UserKBPin.db_generated_columns == ()


def test_user_kb_pin_round_trip() -> None:
    row = UserKBPin(
        tenant_id=1,
        user_id="user-1",
        kb_id="kb-1",
        pinned_at=_NOW,
    )
    hydrated = UserKBPin.from_row(row.model_dump())
    assert hydrated.tenant_id == 1
    assert hydrated.user_id == "user-1"
    assert hydrated.kb_id == "kb-1"
    assert hydrated.pinned_at == _NOW


# ── user_resource_favorites ──────────────────────────────────────────


def test_user_resource_favorite_metadata() -> None:
    assert UserResourceFavorite.table == "user_resource_favorites"
    assert UserResourceFavorite.primary_keys == (
        "user_id",
        "tenant_id",
        "resource_type",
        "resource_id",
    )
    assert UserResourceFavorite.json_columns == ()


def test_user_resource_favorite_accepts_known_resource_types() -> None:
    # 'kb' and 'agent' are the current enum values; we do not pin a
    # value here — the column is open-ended so future resource
    # types need no migration.
    row = UserResourceFavorite(
        user_id="user-1",
        tenant_id=1,
        resource_type="kb",
        resource_id="kb-1",
        created_at=_NOW,
    )
    assert row.resource_type == "kb"


# ── wiki_log_entries ─────────────────────────────────────────────────


def test_wiki_log_entry_metadata() -> None:
    assert WikiLogEntry.table == "wiki_log_entries"
    assert WikiLogEntry.primary_keys == ("id",)
    assert WikiLogEntry.json_columns == ("pages_affected",)
    # ``id`` is BIGSERIAL — DB-assigned, so it is excluded from INSERT.
    assert "id" in WikiLogEntry.db_generated_columns


def test_wiki_log_entry_insert_columns_exclude_id() -> None:
    assert "id" not in WikiLogEntry.insert_sql_column_list()
    assert "pages_affected" in WikiLogEntry.insert_sql_column_list()


def test_wiki_log_entry_defaults() -> None:
    row = WikiLogEntry(
        tenant_id=1,
        knowledge_base_id="kb-1",
        action="ingest",
        created_at=_NOW,
    )
    # Caller-omitted columns fall back to the upstream default of '' / [].
    assert row.knowledge_id == ""
    assert row.doc_title == ""
    assert row.summary == ""
    assert row.pages_affected == []


# ── wiki_page_issues ──────────────────────────────────────────────────


def test_wiki_page_issue_metadata() -> None:
    assert WikiPageIssue.table == "wiki_page_issues"
    assert WikiPageIssue.primary_keys == ("id",)
    assert WikiPageIssue.json_columns == ("suspected_knowledge_ids",)
    # Caller-assigned UUID; the DB does not mint it.
    assert WikiPageIssue.db_generated_columns == ()


def test_wiki_page_issue_default_status_is_pending() -> None:
    row = WikiPageIssue(
        id="issue-1",
        tenant_id=1,
        knowledge_base_id="kb-1",
        slug="entity/acme",
        issue_type="inaccurate",
        description="mismatch with source",
        reported_by="user-1",
        created_at=_NOW,
        updated_at=_NOW,
    )
    assert row.status == "pending"
    assert row.suspected_knowledge_ids is None


def test_wiki_page_issue_round_trip_with_ids() -> None:
    row = WikiPageIssue(
        id="issue-1",
        tenant_id=1,
        knowledge_base_id="kb-1",
        slug="entity/acme",
        issue_type="broken_link",
        description="dead link",
        reported_by="pipeline",
        created_at=_NOW,
        updated_at=_NOW,
        suspected_knowledge_ids=["kg-1", "kg-2"],
    )
    assert row.suspected_knowledge_ids == ["kg-1", "kg-2"]
    assert row.id == "issue-1"


# ── wiki_page_revisions ──────────────────────────────────────────────


def test_wiki_page_revision_metadata() -> None:
    assert WikiPageRevision.table == "wiki_page_revisions"
    assert WikiPageRevision.primary_keys == ("id",)
    assert WikiPageRevision.json_columns == ("aliases",)
    assert WikiPageRevision.db_generated_columns == ()


def test_wiki_page_revision_defaults() -> None:
    row = WikiPageRevision(
        id="rev-1",
        tenant_id=1,
        knowledge_base_id="kb-1",
        page_id="page-1",
        slug="entity/acme",
        version=2,
        edited_at=_NOW,
        created_at=_NOW,
    )
    # Mirrors wiki_pages defaults: title/page_type/status/content/summary
    # all have server-side defaults of '' / 'summary' / 'published'.
    assert row.title == ""
    assert row.page_type == "summary"
    assert row.status == "published"
    assert row.content == ""
    assert row.summary == ""
    assert row.aliases == []
    assert row.edit_source == ""
    assert row.editor_id == ""


def test_wiki_page_revision_full_round_trip() -> None:
    payload: dict[str, object] = {
        "id": "rev-1",
        "tenant_id": 1,
        "knowledge_base_id": "kb-1",
        "page_id": "page-1",
        "slug": "entity/acme",
        "version": 3,
        "title": "Acme Corp",
        "page_type": "entity",
        "status": "published",
        "content": "Body text",
        "summary": "Summary",
        "aliases": ["acme", "acme-corp"],
        "edit_source": "user",
        "editor_id": "user-1",
        "edited_at": _NOW,
        "created_at": _NOW,
    }
    hydrated = WikiPageRevision.from_row(payload)
    assert hydrated.version == 3
    assert hydrated.aliases == ["acme", "acme-corp"]
    assert hydrated.editor_id == "user-1"


# ── tenants.api_key (added in this PR) ───────────────────────────────


def test_tenant_carries_api_key_field() -> None:
    row = Tenant(
        id=1,
        name="Acme",
        business="acme",
        created_at=_NOW,
        updated_at=_NOW,
    )
    # Default is empty string; the SQL column is NOT NULL DEFAULT ''.
    assert row.api_key == ""


def test_tenant_api_key_overridable() -> None:
    row = Tenant(
        id=1,
        name="Acme",
        business="acme",
        api_key="secret",
        created_at=_NOW,
        updated_at=_NOW,
    )
    assert row.api_key == "secret"


# ── Common: freeze + immutability (smoke) ─────────────────────────────


@pytest.mark.parametrize(
    "factory",
    [
        lambda: TenantDisabledSharedAgent(
            tenant_id=1,
            agent_id="a",
            source_tenant_id=2,
            created_at=_NOW,
        ),
        lambda: UserKBPin(
            tenant_id=1,
            user_id="u",
            kb_id="kb",
            pinned_at=_NOW,
        ),
        lambda: UserResourceFavorite(
            user_id="u",
            tenant_id=1,
            resource_type="kb",
            resource_id="kb-1",
            created_at=_NOW,
        ),
        lambda: WikiLogEntry(
            tenant_id=1,
            knowledge_base_id="kb-1",
            action="ingest",
            created_at=_NOW,
        ),
        lambda: WikiPageIssue(
            id="i-1",
            tenant_id=1,
            knowledge_base_id="kb-1",
            slug="x",
            issue_type="inaccurate",
            description="d",
            reported_by="u",
            created_at=_NOW,
            updated_at=_NOW,
        ),
        lambda: WikiPageRevision(
            id="r-1",
            tenant_id=1,
            knowledge_base_id="kb-1",
            page_id="p-1",
            slug="x",
            version=1,
            edited_at=_NOW,
            created_at=_NOW,
        ),
    ],
)
def test_models_are_frozen(factory: ClassVar) -> None:
    from pydantic import ValidationError as PydanticValidationError

    row = factory()
    field_name = next(iter(type(row).model_fields))
    with pytest.raises(PydanticValidationError):
        # Every model inherits the TableModel ``frozen=True`` config.
        setattr(row, field_name, "mutated")  # type: ignore[arg-type]
