"""Unit tests for the `tenants` domain DTOs."""

from __future__ import annotations

from datetime import UTC, datetime

from src.core.tenants.types import RetrieverEngines, TenantInfo
from src.db.models.tenants.tenants import Tenant

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _db_row(**overrides: object) -> Tenant:
    defaults: dict[str, object] = {
        "id": 7,
        "name": "acme",
        "description": "acme workspace",
        "business": "saas",
        "created_at": _NOW,
        "updated_at": _NOW,
    }
    return Tenant.model_validate({**defaults, **overrides})


# ── RetrieverEngines.from_json ──────────────────────────────────────


def test_from_json_reads_object_form() -> None:
    engines = RetrieverEngines.from_json(
        {"engines": [{"retriever_type": "keywords", "retriever_engine_type": "postgres"}]}
    )

    assert [e.retriever_type for e in engines.engines] == ["keywords"]


def test_from_json_reads_legacy_bare_array_form() -> None:
    engines = RetrieverEngines.from_json(
        [{"retriever_type": "vector", "retriever_engine_type": "milvus"}]
    )

    assert [e.retriever_engine_type for e in engines.engines] == ["milvus"]


def test_from_json_parses_raw_json_string() -> None:
    engines = RetrieverEngines.from_json(
        '{"engines": [{"retriever_type": "vector", "retriever_engine_type": "qdrant"}]}'
    )

    assert [e.retriever_engine_type for e in engines.engines] == ["qdrant"]


def test_from_json_returns_empty_for_none_and_empty_string() -> None:
    assert RetrieverEngines.from_json(None).engines == []
    assert RetrieverEngines.from_json("").engines == []


# ── TenantInfo.map_from_db ──────────────────────────────────────────


def test_map_from_db_copies_wire_fields() -> None:
    info = TenantInfo.map_from_db(_db_row())

    assert info.id == 7
    assert info.name == "acme"
    assert info.description == "acme workspace"
    assert info.business == "saas"
    assert info.created_at == _NOW


def test_map_from_db_hydrates_retriever_engines() -> None:
    row = _db_row(
        retriever_engines={
            "engines": [{"retriever_type": "keywords", "retriever_engine_type": "postgres"}]
        }
    )

    info = TenantInfo.map_from_db(row)

    assert info.retriever_engines.engines[0].retriever_engine_type == "postgres"


def test_map_from_db_drops_secret_and_storage_only_columns() -> None:
    row = _db_row(
        api_principal_config={"mode": "signed_token", "hmac_secret": "s3cret"},
        agent_config={"max_iterations": 3},
        conversation_config={"temperature": 0.5},
        default_storage_backend_id="bk-1",
    )

    fields = set(TenantInfo.map_from_db(row).model_dump())

    assert "api_principal_config" not in fields
    assert "agent_config" not in fields
    assert "conversation_config" not in fields
    assert "default_storage_backend_id" not in fields


def test_map_from_db_keeps_soft_delete_marker() -> None:
    info = TenantInfo.map_from_db(_db_row(deleted_at=_NOW))

    assert info.deleted_at == _NOW
