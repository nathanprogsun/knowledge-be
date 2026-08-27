"""Unit tests for ``CloudService`` — save + status.

The tenant repository is replaced by the shared ``FakeTenantRepository``
(``tests/fakes/tenants.py``) so the credential merge is exercised against
the same method signatures the real repo exposes. The upstream
``/api/v1/health`` probe is driven by an ``httpx.MockTransport``, keeping
the signing path real while no network call happens.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import httpx
import pytest

from src.common.exception import ExternalServiceError, ValidationError
from src.core.infra.models.service.provider_service import (
    ENC_PREFIX,
    KB_CLOUD_BASE_URL,
    CloudService,
    is_kb_cloud_doc_reader_addr,
    sign_request_headers,
)
from src.db.dao.tenants_repository import TenantRepository
from src.db.models.tenants.tenants import Tenant


def _make_repo() -> tuple[AsyncMock, dict[int, Tenant]]:
    """``AsyncMock(spec=TenantRepository)`` with closure-captured state.

    Only the small surface exercised by the credential merge
    (insert / find_by_id / update_by_primary_key) is implemented.
    """
    repo = AsyncMock(spec=TenantRepository)
    rows: dict[int, Tenant] = {}
    _next_id = {"value": 0}
    repo.rows = rows  # type: ignore[attr-defined]

    async def _insert(row: Tenant) -> Tenant:
        _next_id["value"] += 1
        stored = row.model_copy(update={"id": _next_id["value"]})
        rows[stored.id] = stored
        return stored

    async def _find_by_id(id_: str | int) -> Tenant:
        row = rows.get(int(str(id_)))
        if row is None or row.deleted_at is not None:
            from src.common.exception import NotFoundError

            raise NotFoundError(code="tenant.not_found", message=f"Tenant {id_} not found")
        return row

    async def _update_by_primary_key(
        primary_key_to_value: dict[str, object],
        column_to_update: dict[str, object],
    ) -> Tenant | None:
        tenant_id = int(str(primary_key_to_value["id"]))
        row = rows.get(tenant_id)
        if row is None or row.deleted_at is not None:
            return None
        updated = row.model_copy(update=column_to_update)
        rows[tenant_id] = updated
        return updated

    repo.insert.side_effect = _insert
    repo.find_by_id.side_effect = _find_by_id
    repo.update_by_primary_key.side_effect = _update_by_primary_key
    return repo, rows


_HEALTH_URL = f"{KB_CLOUD_BASE_URL}/api/v1/health"


def _client(status_code: int = 200) -> httpx.AsyncClient:
    """Return a client whose health probe answers with ``status_code``."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == _HEALTH_URL
        return httpx.Response(status_code)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _raising_client(exc: httpx.HTTPError) -> httpx.AsyncClient:
    """Return a client whose health probe raises a transport error."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise exc

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def _seed_tenant(repo: AsyncMock, **overrides: object) -> Tenant:
    now = datetime.now(UTC)
    return await repo.insert(Tenant(name="acme", created_at=now, updated_at=now, **overrides))


def _service(repo: AsyncMock, client: httpx.AsyncClient) -> CloudService:
    return CloudService(tenants_repo=repo, http_client=client)


def _make() -> tuple[AsyncMock, dict[int, Tenant]]:
    """Fixture-style helper for tests that want to share a single repo."""
    return _make_repo()


# ── Signing ─────────────────────────────────────────────────────────


def test_sign_request_headers_returns_the_six_upstream_headers() -> None:
    # Arrange / Act
    headers = sign_request_headers(app_id="app", app_secret="secret", request_id="req-1")

    # Assert
    assert set(headers) == {
        "X-APPID",
        "X-API-Key",
        "X-Request-ID",
        "X-Timestamp",
        "X-Nonce",
        "X-Signature",
    }
    assert headers["X-APPID"] == "app"
    assert headers["X-API-Key"] == "secret"
    assert headers["X-Request-ID"] == "req-1"
    assert len(headers["X-Nonce"]) == 16
    assert len(headers["X-Signature"]) == 32


def test_sign_request_headers_signature_varies_with_the_nonce() -> None:
    # Arrange / Act
    first = sign_request_headers(app_id="app", app_secret="secret", request_id="req-1")
    second = sign_request_headers(app_id="app", app_secret="secret", request_id="req-1")

    # Assert — the random nonce enters the signed param set
    assert first["X-Signature"] != second["X-Signature"]


def test_is_kb_cloud_doc_reader_addr_ignores_trailing_slash() -> None:
    # Arrange
    addr = f"{KB_CLOUD_BASE_URL}/api/v1/doc/reader/"

    # Act / Assert
    assert is_kb_cloud_doc_reader_addr(addr) is True
    assert is_kb_cloud_doc_reader_addr("  " + addr.rstrip("/") + " ") is True
    assert is_kb_cloud_doc_reader_addr("https://evil.example/api/v1/doc/reader") is False


# ── save_credentials ────────────────────────────────────────────────


async def test_save_credentials_persists_the_pair_under_cloud() -> None:
    # Arrange
    repo, _ = _make()
    tenant = await _seed_tenant(repo)
    async with _client() as client:
        service = _service(repo, client)

        # Act
        await service.save_credentials(tenant_id=tenant.id, app_id="app", app_secret="secret")

    # Assert
    assert repo.rows[tenant.id].credentials == {"cloud": {"app_id": "app", "app_secret": "secret"}}


async def test_save_credentials_preserves_other_credential_providers() -> None:
    # Arrange
    repo, _ = _make()
    tenant = await _seed_tenant(repo, credentials={"other": {"token": "keep-me"}})
    async with _client() as client:
        service = _service(repo, client)

        # Act
        await service.save_credentials(tenant_id=tenant.id, app_id="app", app_secret="secret")

    # Assert
    stored = repo.rows[tenant.id].credentials
    assert stored is not None
    assert stored["other"] == {"token": "keep-me"}
    assert stored["cloud"] == {"app_id": "app", "app_secret": "secret"}


@pytest.mark.parametrize(
    ("app_id", "app_secret", "code"),
    [
        ("", "secret", "cloud.app_id_required"),
        ("app", "", "cloud.app_secret_required"),
    ],
)
async def test_save_credentials_rejects_blank_fields(
    app_id: str,
    app_secret: str,
    code: str,
) -> None:
    # Arrange
    repo, _ = _make()
    tenant = await _seed_tenant(repo)
    async with _client() as client:
        service = _service(repo, client)

        # Act / Assert
        with pytest.raises(ValidationError) as excinfo:
            await service.save_credentials(
                tenant_id=tenant.id,
                app_id=app_id,
                app_secret=app_secret,
            )
    assert excinfo.value.code == code
    assert repo.rows[tenant.id].credentials is None


@pytest.mark.parametrize("status_code", [401, 403])
async def test_save_credentials_rejects_unauthorized_upstream(status_code: int) -> None:
    # Arrange
    repo, _ = _make()
    tenant = await _seed_tenant(repo)
    async with _client(status_code) as client:
        service = _service(repo, client)

        # Act / Assert
        with pytest.raises(ExternalServiceError) as excinfo:
            await service.save_credentials(
                tenant_id=tenant.id,
                app_id="app",
                app_secret="wrong",
            )
    assert excinfo.value.code == "cloud.invalid_credentials"
    assert repo.rows[tenant.id].credentials is None


async def test_save_credentials_rejects_unexpected_upstream_status() -> None:
    # Arrange
    repo, _ = _make()
    tenant = await _seed_tenant(repo)
    async with _client(503) as client:
        service = _service(repo, client)

        # Act / Assert
        with pytest.raises(ExternalServiceError) as excinfo:
            await service.save_credentials(
                tenant_id=tenant.id,
                app_id="app",
                app_secret="secret",
            )
    assert excinfo.value.code == "cloud.verification_failed"


async def test_save_credentials_reports_an_unreachable_service() -> None:
    # Arrange
    repo, _ = _make()
    tenant = await _seed_tenant(repo)
    async with _raising_client(httpx.ConnectError("refused")) as client:
        service = _service(repo, client)

        # Act / Assert
        with pytest.raises(ExternalServiceError) as excinfo:
            await service.save_credentials(
                tenant_id=tenant.id,
                app_id="app",
                app_secret="secret",
            )
    assert excinfo.value.code == "cloud.service_unreachable"
    assert repo.rows[tenant.id].credentials is None


# ── check_status ────────────────────────────────────────────────────


async def test_check_status_reports_no_models_without_credentials() -> None:
    # Arrange
    repo, _ = _make()
    tenant = await _seed_tenant(repo)
    async with _client() as client:
        service = _service(repo, client)

        # Act
        status = await service.check_status(tenant_id=tenant.id)

    # Assert
    assert status.has_models is False
    assert status.needs_reinit is False
    assert status.reason is None


async def test_check_status_reports_no_models_for_an_unknown_tenant() -> None:
    # Arrange
    repo, _ = _make()
    async with _client() as client:
        service = _service(repo, client)

        # Act
        status = await service.check_status(tenant_id=999)

    # Assert — Go swallows the missing tenant rather than erroring
    assert status.has_models is False
    assert status.needs_reinit is False


async def test_check_status_reports_no_models_when_a_field_is_blank() -> None:
    # Arrange
    repo, _ = _make()
    tenant = await _seed_tenant(
        repo,
        credentials={"cloud": {"app_id": "app", "app_secret": ""}},
    )
    async with _client() as client:
        service = _service(repo, client)

        # Act
        status = await service.check_status(tenant_id=tenant.id)

    # Assert — empty credentials resolve to the "not configured" shape
    assert status.has_models is False
    assert status.needs_reinit is False


async def test_check_status_is_healthy_with_a_decrypted_secret() -> None:
    # Arrange
    repo, _ = _make()
    tenant = await _seed_tenant(
        repo,
        credentials={"cloud": {"app_id": "app", "app_secret": "plaintext"}},
    )
    async with _client() as client:
        service = _service(repo, client)

        # Act
        status = await service.check_status(tenant_id=tenant.id)

    # Assert
    assert status.has_models is True
    assert status.needs_reinit is False
    assert status.reason is None


async def test_check_status_requests_reinit_for_an_undecrypted_secret() -> None:
    # Arrange
    repo, _ = _make()
    tenant = await _seed_tenant(
        repo,
        credentials={"cloud": {"app_id": "app", "app_secret": f"{ENC_PREFIX}blob"}},
    )
    async with _client() as client:
        service = _service(repo, client)

        # Act
        status = await service.check_status(tenant_id=tenant.id)

    # Assert
    assert status.has_models is True
    assert status.needs_reinit is True
    assert status.reason is not None
    assert "APPSECRET" in status.reason
