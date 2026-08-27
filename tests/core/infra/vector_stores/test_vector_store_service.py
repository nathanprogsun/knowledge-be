"""Unit tests for ``VectorStoreService`` + ``VectorStoreRepository``.

Per AGENTS.md §9, core services are tested with Protocol-based mocks
where they materially reduce test setup. The mocks mirror the real
repository contracts so the service exercises the same surface as it
would in production (finders return storage rows; the service projects
them to ``VectorStoreInfo`` via ``map_from_db``).

The probes that go through ``test_connection_async`` are exercised
against the in-memory probes (the test fixtures pre-populate a fake
TCP / HTTP transport via monkeypatching the probe module so we never
hit a real network in unit tests).
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from src.common.exception import ConflictError, NotFoundError, ValidationError
from src.core.contracts.infra import (
    CreateVectorStoreRequest,
    UpdateVectorStoreRequest,
)
from src.core.contracts.infra import (
    TestVectorStoreRequest as _RawTestRequest,
)
from src.core.infra.vector_stores import healthcheck as healthcheck_module
from src.core.infra.vector_stores.service import vector_store_service as service_module
from src.core.infra.vector_stores.service.vector_store_service import (
    VectorStoreService,
)
from src.core.infra.vector_stores.types import (
    SUPPORTED_ENGINE_TYPES,
    VectorStoreInfo,
    vector_store_types,
)
from src.db.dao.vector_store_repository import VectorStoreRepository
from src.db.models.infra.vector_store import VectorStore

# Silence unused-import warnings for the aliased wire model.
__ = (_RawTestRequest,)


# ── VectorStore repository mock (stateful via side_effect closures) ──


def _make_repo() -> tuple[AsyncMock, dict[str, VectorStore]]:
    """VectorStore-repo mock with closure-captured storage."""
    repo = AsyncMock(spec=VectorStoreRepository)
    rows: dict[str, VectorStore] = {}

    async def _insert(row: VectorStore) -> VectorStore:
        if row.id in rows:
            raise ValueError(f"duplicate id: {row.id}")
        rows[row.id] = row
        return row

    async def _get_by_id(tenant_id: int, store_id: str) -> VectorStore | None:
        for row in rows.values():
            if row.id == store_id and row.tenant_id == tenant_id and row.deleted_at is None:
                return row
        return None

    async def _list_for_tenant(tenant_id: int) -> list[VectorStore]:
        out = [
            row for row in rows.values() if row.tenant_id == tenant_id and row.deleted_at is None
        ]
        out.sort(key=lambda r: r.created_at, reverse=True)
        return out

    async def _exists_by_engine_type_endpoint_index(
        *,
        tenant_id: int,
        engine_type: str,
        endpoint: str,
        index_name: str,
    ) -> bool:
        from src.db.dao.vector_store_repository import (
            _endpoint_of,
            _index_name_of,
        )

        for row in rows.values():
            if (
                row.tenant_id == tenant_id
                and row.engine_type == engine_type
                and row.deleted_at is None
                and _endpoint_of(row.connection_config) == endpoint
                and _index_name_of(row.index_config, engine_type) == index_name
            ):
                return True
        return False

    async def _update_by_primary_key(
        primary_key_to_value: dict[str, object],
        column_to_update: dict[str, object],
        *,
        exclude_deleted_or_archived: bool = True,
    ) -> VectorStore | None:
        sid = primary_key_to_value.get("id")
        if not isinstance(sid, str):
            return None
        row = rows.get(sid)
        if row is None:
            return None
        tid = primary_key_to_value.get("tenant_id")
        if isinstance(tid, int) and row.tenant_id != tid:
            return None
        if exclude_deleted_or_archived and row.deleted_at is not None:
            return None
        updated = row.model_copy(update=column_to_update)
        rows[sid] = updated
        return updated

    repo.insert.side_effect = _insert
    repo.get_by_id.side_effect = _get_by_id
    repo.list_for_tenant.side_effect = _list_for_tenant
    repo.exists_by_engine_type_endpoint_index.side_effect = _exists_by_engine_type_endpoint_index
    repo.update_by_primary_key.side_effect = _update_by_primary_key
    return repo, rows


@pytest.fixture(autouse=True)
def _ssrf_whitelist(monkeypatch: pytest.MonkeyPatch) -> None:
    """Allow the fake ``es``/``es-b``/``es-c`` probe hosts through the
    SSRF policy (the whitelist is re-read on every call)."""
    monkeypatch.setenv("SSRF_WHITELIST", "es,es-b,es-c")


# ── Probe monkeypatch ────────────────────────────────────────────────


@pytest.fixture
def fake_probe(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[tuple[str, dict[str, object]]]]:
    """Capture every probe call and return a fake ``(version, error)``.

    Default behaviour: succeed with empty version. The fixture returns
    a list that tests can inspect to assert on the probe inputs.

    Patches both the healthcheck module and the service module so the
    service's already-imported name resolves to the fake.
    """
    calls: list[tuple[str, dict[str, object]]] = []

    async def fake_test(
        engine_type: str,
        config: dict[str, object],
    ) -> tuple[str, None] | tuple[None, ValidationError]:
        calls.append((engine_type, dict(config)))
        return ("", None)

    monkeypatch.setattr(healthcheck_module, "test_connection_async", fake_test)
    monkeypatch.setattr(service_module, "test_connection_async", fake_test)
    yield calls


async def _noop_ssrf(_url: str) -> None:
    """Test-only SSRF guard bypass (the guard has its own tests)."""


# ── Service-level tests ──────────────────────────────────────────────


class _Timestamps:
    """Datetime fixtures used across the service tests."""

    @staticmethod
    def now() -> datetime:
        return datetime(2026, 1, 1, tzinfo=UTC)


def _sample_create_body(
    *,
    name: str = "es-hot",
    engine_type: str = "elasticsearch",
    connection_config: dict[str, object] | None = None,
    index_config: dict[str, object] | None = None,
) -> CreateVectorStoreRequest:
    """Build a sample ``CreateVectorStoreRequest`` for tests."""
    if connection_config is None:
        connection_config = {"addr": "http://es:9200"}
    if index_config is None:
        index_config = {}
    return CreateVectorStoreRequest(
        name=name,
        engine_type=engine_type,
        connection_config=connection_config,  # type: ignore[arg-type]
        index_config=index_config,  # type: ignore[arg-type]
    )


def _sample_update_body(name: str = "es-hot-renamed") -> UpdateVectorStoreRequest:
    """Build a sample ``UpdateVectorStoreRequest`` for tests."""
    return UpdateVectorStoreRequest(name=name)


async def test_create_store_persists_row(
    fake_probe: list[tuple[str, dict[str, object]]],
) -> None:
    """The create path inserts a row with a fresh UUID and stamps the timestamps."""
    repo, rows = _make_repo()
    service = VectorStoreService(vector_store_repo=repo)
    info = await service.create_store(
        tenant_id=1,
        body=_sample_create_body(),
    )
    assert info.id != ""
    assert info.tenant_id == 1
    assert info.name == "es-hot"
    assert info.engine_type == "elasticsearch"
    assert info.source == "user"
    assert info.readonly is False
    assert info.created_at is not None
    assert info.updated_at == info.created_at
    assert len(rows) == 1


async def test_create_store_rejects_empty_name(
    fake_probe: list[tuple[str, dict[str, object]]],
) -> None:
    """A blank name is rejected with a typed validation error."""
    repo, rows = _make_repo()
    service = VectorStoreService(vector_store_repo=repo)
    with pytest.raises(ValidationError) as exc:
        await service.create_store(
            tenant_id=1,
            body=_sample_create_body(name="   "),
        )
    assert exc.value.code == "vector_store.name_required"
    assert rows == {}


async def test_create_store_rejects_unsupported_engine(
    fake_probe: list[tuple[str, dict[str, object]]],
) -> None:
    """An engine type outside the registry is rejected before any write."""
    repo, rows = _make_repo()
    service = VectorStoreService(vector_store_repo=repo)
    with pytest.raises(ValidationError) as exc:
        await service.create_store(
            tenant_id=1,
            body=_sample_create_body(engine_type="postgres"),
        )
    assert exc.value.code == "vector_store.invalid_engine_type"
    assert rows == {}


async def test_create_store_rejects_missing_required_field(
    fake_probe: list[tuple[str, dict[str, object]]],
) -> None:
    """The service rejects an ES store without an ``addr`` field."""
    repo, _ = _make_repo()
    service = VectorStoreService(vector_store_repo=repo)
    with pytest.raises(ValidationError) as exc:
        await service.create_store(
            tenant_id=1,
            body=_sample_create_body(connection_config={}),
        )
    assert exc.value.code == "vector_store.addr_required"


async def test_create_store_rejects_duplicate(
    fake_probe: list[tuple[str, dict[str, object]]],
) -> None:
    """Two stores with the same engine+endpoint+index are rejected."""
    repo, _ = _make_repo()
    service = VectorStoreService(vector_store_repo=repo)
    body = _sample_create_body()
    await service.create_store(tenant_id=1, body=body)
    with pytest.raises(ConflictError) as exc:
        await service.create_store(tenant_id=1, body=body)
    assert exc.value.code == "vector_store.duplicate"


async def test_create_store_allows_different_tenant(
    fake_probe: list[tuple[str, dict[str, object]]],
) -> None:
    """Two stores with the same endpoint in different tenants coexist."""
    repo, rows = _make_repo()
    service = VectorStoreService(vector_store_repo=repo)
    body = _sample_create_body()
    await service.create_store(tenant_id=1, body=body)
    info = await service.create_store(tenant_id=2, body=body)
    assert info.tenant_id == 2
    assert len(rows) == 2


async def test_list_stores_returns_tenant_scoped_rows(
    fake_probe: list[tuple[str, dict[str, object]]],
) -> None:
    """The list endpoint returns only the active tenant's live rows, newest first."""
    repo, _ = _make_repo()
    service = VectorStoreService(vector_store_repo=repo)
    older = await service.create_store(
        tenant_id=1,
        body=_sample_create_body(name="es-a"),
    )
    newer = await service.create_store(
        tenant_id=1,
        body=_sample_create_body(
            name="es-b",
            connection_config={"addr": "http://es-b:9200"},
        ),
    )
    # A row from a different tenant.
    await service.create_store(
        tenant_id=2,
        body=_sample_create_body(
            name="es-c",
            connection_config={"addr": "http://es-c:9200"},
        ),
    )
    listed = await service.list_stores(tenant_id=1)
    ids = [info.id for info in listed]
    assert ids == [newer.id, older.id]


async def test_get_store_returns_match(
    fake_probe: list[tuple[str, dict[str, object]]],
) -> None:
    """``get_store`` returns the row matching the (id, tenant) pair."""
    repo, _ = _make_repo()
    service = VectorStoreService(vector_store_repo=repo)
    info = await service.create_store(
        tenant_id=1,
        body=_sample_create_body(),
    )
    fetched = await service.get_store(tenant_id=1, store_id=info.id)
    assert fetched.id == info.id


async def test_get_store_raises_on_unknown_id(
    fake_probe: list[tuple[str, dict[str, object]]],
) -> None:
    """``get_store`` raises ``NotFoundError`` for an unknown id."""
    repo, _ = _make_repo()
    service = VectorStoreService(vector_store_repo=repo)
    with pytest.raises(NotFoundError) as exc:
        await service.get_store(tenant_id=1, store_id="missing")
    assert exc.value.code == "vector_store.not_found"


async def test_get_store_raises_for_other_tenant(
    fake_probe: list[tuple[str, dict[str, object]]],
) -> None:
    """``get_store`` does not leak rows across tenant boundaries."""
    repo, _ = _make_repo()
    service = VectorStoreService(vector_store_repo=repo)
    info = await service.create_store(
        tenant_id=1,
        body=_sample_create_body(),
    )
    with pytest.raises(NotFoundError) as exc:
        await service.get_store(tenant_id=2, store_id=info.id)
    assert exc.value.code == "vector_store.not_found"


async def test_update_store_renames_row(
    fake_probe: list[tuple[str, dict[str, object]]],
) -> None:
    """``update_store`` only mutates the ``name`` column."""
    repo, rows = _make_repo()
    service = VectorStoreService(vector_store_repo=repo)
    info = await service.create_store(
        tenant_id=1,
        body=_sample_create_body(),
    )
    original_updated_at = info.updated_at
    updated = await service.update_store(
        tenant_id=1,
        store_id=info.id,
        body=_sample_update_body(name="renamed"),
    )
    assert updated.name == "renamed"
    assert updated.updated_at is not None
    assert original_updated_at is not None
    assert updated.updated_at >= original_updated_at
    assert rows[info.id].engine_type == "elasticsearch"  # immutable


async def test_update_store_rejects_empty_name(
    fake_probe: list[tuple[str, dict[str, object]]],
) -> None:
    """An empty name on update is rejected with a typed error."""
    repo, _ = _make_repo()
    service = VectorStoreService(vector_store_repo=repo)
    info = await service.create_store(
        tenant_id=1,
        body=_sample_create_body(),
    )
    with pytest.raises(ValidationError) as exc:
        await service.update_store(
            tenant_id=1,
            store_id=info.id,
            body=_sample_update_body(name="  "),
        )
    assert exc.value.code == "vector_store.name_required"


async def test_update_store_raises_on_unknown_id(
    fake_probe: list[tuple[str, dict[str, object]]],
) -> None:
    """An unknown id on update is rejected with a typed error."""
    repo, _ = _make_repo()
    service = VectorStoreService(vector_store_repo=repo)
    with pytest.raises(ValidationError) as exc:
        await service.update_store(
            tenant_id=1,
            store_id="missing",
            body=_sample_update_body(),
        )
    assert exc.value.code == "vector_store.not_found"


async def test_delete_store_soft_deletes(
    fake_probe: list[tuple[str, dict[str, object]]],
) -> None:
    """``delete_store`` soft-deletes the row and removes it from list results."""
    repo, rows = _make_repo()
    service = VectorStoreService(vector_store_repo=repo)
    info = await service.create_store(
        tenant_id=1,
        body=_sample_create_body(),
    )
    deleted = await service.delete_store(tenant_id=1, store_id=info.id)
    assert deleted is True
    listed = await service.list_stores(tenant_id=1)
    assert listed == []
    # The row is still in the repo, with ``deleted_at`` set.
    stored = rows[info.id]
    assert stored.deleted_at is not None


async def test_delete_store_idempotent_for_unknown_id(
    fake_probe: list[tuple[str, dict[str, object]]],
) -> None:
    """Deleting an unknown id returns ``False`` (idempotent semantics)."""
    repo, _ = _make_repo()
    service = VectorStoreService(vector_store_repo=repo)
    deleted = await service.delete_store(tenant_id=1, store_id="missing")
    assert deleted is False


async def test_require_store_raises_not_found(
    fake_probe: list[tuple[str, dict[str, object]]],
) -> None:
    """``require_store`` surfaces a :class:`NotFoundError` on miss."""
    repo, _ = _make_repo()
    service = VectorStoreService(vector_store_repo=repo)
    with pytest.raises(NotFoundError) as exc:
        await service.require_store(tenant_id=1, store_id="missing")
    assert exc.value.code == "vector_store.not_found"


async def test_test_by_id_invokes_probe_with_stored_config(
    fake_probe: list[tuple[str, dict[str, object]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``test_by_id`` runs the probe with the stored config and returns a success response."""
    repo, _ = _make_repo()
    service = VectorStoreService(vector_store_repo=repo)
    monkeypatch.setattr(service_module, "validate_ssrf_safe_url", _noop_ssrf)
    info = await service.create_store(
        tenant_id=1,
        body=_sample_create_body(
            connection_config={"addr": "http://es:9200"},
        ),
    )
    result = await service.test_by_id(tenant_id=1, store_id=info.id)
    assert result.success is True
    assert result.error is None
    assert len(fake_probe) == 1
    engine_type, _config = fake_probe[0]
    assert engine_type == "elasticsearch"


async def test_test_by_id_returns_probe_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``test_by_id`` surfaces a failed probe as ``success=False``."""
    repo, _ = _make_repo()
    service = VectorStoreService(vector_store_repo=repo)
    monkeypatch.setattr(service_module, "validate_ssrf_safe_url", _noop_ssrf)
    info = await service.create_store(
        tenant_id=1,
        body=_sample_create_body(),
    )

    async def fake_test(
        engine_type: str,
        config: dict[str, object],
    ) -> tuple[str, None] | tuple[None, ValidationError]:
        return (
            None,
            ValidationError(code="vector_store.connection_failed", message="refused"),
        )

    monkeypatch.setattr(healthcheck_module, "test_connection_async", fake_test)
    monkeypatch.setattr(service_module, "test_connection_async", fake_test)
    result = await service.test_by_id(tenant_id=1, store_id=info.id)
    assert result.success is False
    assert result.error == "refused"


async def test_test_raw_rejects_unsupported_engine(
    fake_probe: list[tuple[str, dict[str, object]]],
) -> None:
    """``test_raw`` rejects engines outside the allowlist before probing."""
    repo, _ = _make_repo()
    service = VectorStoreService(vector_store_repo=repo)
    body = _RawTestRequest(
        engine_type="postgres",
        connection_config={"use_default_connection": True},
    )
    with pytest.raises(ValidationError) as exc:
        await service.test_raw(
            engine_type=body.engine_type,
            connection_config=body.connection_config,
        )
    assert exc.value.code == "vector_store.unsupported_engine"


async def test_test_raw_validates_required_fields(
    fake_probe: list[tuple[str, dict[str, object]]],
) -> None:
    """``test_raw`` rejects a missing required field before probing."""
    repo, _ = _make_repo()
    service = VectorStoreService(vector_store_repo=repo)
    with pytest.raises(ValidationError) as exc:
        await service.test_raw(
            engine_type="elasticsearch",
            connection_config={},
        )
    assert exc.value.code == "vector_store.addr_required"


async def test_test_raw_returns_probe_success(
    fake_probe: list[tuple[str, dict[str, object]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``test_raw`` returns the probe's success payload on the happy path."""
    repo, _ = _make_repo()
    service = VectorStoreService(vector_store_repo=repo)
    # The SSRF guard is exercised by its own tests; bypass it here so the
    # probe path can run against the sample internal endpoint.
    monkeypatch.setattr(service_module, "validate_ssrf_safe_url", _noop_ssrf)
    result = await service.test_raw(
        engine_type="elasticsearch",
        connection_config={"addr": "http://es:9200"},
    )
    assert result.success is True
    assert result.error is None


async def test_test_raw_returns_probe_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``test_raw`` passes the detected version through on success."""
    repo, _ = _make_repo()
    service = VectorStoreService(vector_store_repo=repo)

    async def fake_test(
        engine_type: str,
        config: dict[str, object],
    ) -> tuple[str, None] | tuple[None, ValidationError]:
        return ("7.10.1", None)

    monkeypatch.setattr(healthcheck_module, "test_connection_async", fake_test)
    monkeypatch.setattr(service_module, "test_connection_async", fake_test)
    # The SSRF guard is exercised by its own tests; bypass it here so the
    # probe path can run against the sample internal endpoint.
    monkeypatch.setattr(service_module, "validate_ssrf_safe_url", _noop_ssrf)
    result = await service.test_raw(
        engine_type="elasticsearch",
        connection_config={"addr": "http://es:9200"},
    )
    assert result.success is True
    assert result.version == "7.10.1"


# ── Registry / type tests ────────────────────────────────────────────


def test_registry_covers_seven_engines() -> None:
    """The registry exposes exactly the seven engine types that pass the allowlist."""
    types = vector_store_types()
    assert len(types) == 7
    assert {info.type for info in types} == {
        "elasticsearch",
        "qdrant",
        "milvus",
        "tencent_vectordb",
        "weaviate",
        "doris",
        "opensearch",
    }
    assert (
        frozenset(
            {info.type for info in types},
        )
        == SUPPORTED_ENGINE_TYPES
    )


def test_vector_store_info_map_from_db_round_trip() -> None:
    """The DTO projection keeps every field of the storage row."""
    now = _Timestamps.now()
    row = VectorStore(
        id="vs-1",
        tenant_id=1,
        name="es-hot",
        engine_type="elasticsearch",
        connection_config={"addr": "http://es:9200"},
        index_config={"index_name": "kb"},
        source="user",
        readonly=False,
        created_at=now,
        updated_at=now,
    )
    info = VectorStoreInfo.map_from_db(row)
    assert info.id == "vs-1"
    assert info.tenant_id == 1
    assert info.name == "es-hot"
    assert info.engine_type == "elasticsearch"
    assert info.connection_config == {"addr": "http://es:9200"}
    assert info.index_config == {"index_name": "kb"}


def test_mask_sensitive_fields_redacts_secrets() -> None:
    """The masking helper replaces ``password`` / ``api_key`` with the placeholder."""
    from src.core.infra.vector_stores.types import (
        REDACTED_SECRET_PLACEHOLDER,
        mask_sensitive_fields,
    )

    masked = mask_sensitive_fields(
        {"addr": "http://es:9200", "password": "secret", "api_key": "k"},
    )
    assert masked is not None
    assert masked["addr"] == "http://es:9200"
    assert masked["password"] == REDACTED_SECRET_PLACEHOLDER
    assert masked["api_key"] == REDACTED_SECRET_PLACEHOLDER


def test_mask_sensitive_fields_keeps_empty_secrets_visible() -> None:
    """Empty secrets stay empty so the UI can distinguish "set" from "not set"."""
    from src.core.infra.vector_stores.types import mask_sensitive_fields

    masked = mask_sensitive_fields({"password": ""})
    assert masked is not None
    assert masked["password"] == ""


# ── Healthcheck probe tests ──────────────────────────────────────────


def test_is_valid_engine_type_filters_unsupported() -> None:
    """Only the seven registry engines are accepted as ``DB-store`` types."""
    assert healthcheck_module.is_valid_engine_type("elasticsearch") is True
    assert healthcheck_module.is_valid_engine_type("postgres") is False
    assert healthcheck_module.is_valid_engine_type("sqlite") is False
    assert healthcheck_module.is_valid_engine_type("nope") is False


def test_validate_connection_config_required_fields() -> None:
    """Required fields raise a typed ``ValidationError`` when missing."""
    # ES
    with pytest.raises(ValidationError) as exc:
        healthcheck_module.validate_connection_config("elasticsearch", {})
    assert exc.value.code == "vector_store.addr_required"
    # Qdrant
    with pytest.raises(ValidationError) as exc:
        healthcheck_module.validate_connection_config("qdrant", {})
    assert exc.value.code == "vector_store.host_required"
    # Milvus
    with pytest.raises(ValidationError) as exc:
        healthcheck_module.validate_connection_config("milvus", {})
    assert exc.value.code == "vector_store.addr_required"
    # Tencent VectorDB requires addr, username, api_key
    with pytest.raises(ValidationError) as exc:
        healthcheck_module.validate_connection_config("tencent_vectordb", {"addr": "x"})
    assert exc.value.code == "vector_store.username_required"
    with pytest.raises(ValidationError) as exc:
        healthcheck_module.validate_connection_config(
            "tencent_vectordb",
            {"addr": "x", "username": "u"},
        )
    assert exc.value.code == "vector_store.api_key_required"
    # Weaviate
    with pytest.raises(ValidationError) as exc:
        healthcheck_module.validate_connection_config("weaviate", {})
    assert exc.value.code == "vector_store.host_required"
    # Doris
    with pytest.raises(ValidationError) as exc:
        healthcheck_module.validate_connection_config("doris", {"addr": "x"})
    assert exc.value.code == "vector_store.database_required"
    # OpenSearch
    with pytest.raises(ValidationError) as exc:
        healthcheck_module.validate_connection_config("opensearch", {})
    assert exc.value.code == "vector_store.addr_required"


def test_validate_connection_config_passes_when_valid() -> None:
    """A fully populated config passes the engine-specific check."""
    healthcheck_module.validate_connection_config("elasticsearch", {"addr": "http://es:9200"})
    healthcheck_module.validate_connection_config("qdrant", {"host": "localhost"})
    healthcheck_module.validate_connection_config("milvus", {"addr": "localhost:19530"})
    healthcheck_module.validate_connection_config(
        "tencent_vectordb",
        {"addr": "http://x", "username": "u", "api_key": "k"},
    )
    healthcheck_module.validate_connection_config("weaviate", {"host": "weaviate:8080"})
    healthcheck_module.validate_connection_config("doris", {"addr": "x", "database": "kb"})
    healthcheck_module.validate_connection_config("opensearch", {"addr": "https://os:9200"})


def test_test_connection_rejects_unknown_engine() -> None:
    """An unknown engine returns a typed ``ValidationError``."""
    version, error = healthcheck_module.test_connection("nope", {})
    assert version is None
    assert error is not None
    assert error.code == "vector_store.unsupported_engine"


__all__ = []
