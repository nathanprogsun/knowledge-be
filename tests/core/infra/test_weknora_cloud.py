"""Unit tests for ``WeKnoraCloudService`` — save + status.

The tenant repository is replaced by the shared ``FakeTenantRepository``
(``tests/fakes/tenants.py``) so the credential merge is exercised against
the same method signatures the real repo exposes. The upstream
``/api/v1/health`` probe is driven by an ``httpx.MockTransport``, keeping
the signing path real while no network call happens.
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

from src.common.exception import ExternalServiceError, ValidationError
from src.core.infra.models.service.weknora_cloud_service import (
    ENC_PREFIX,
    WEKNORA_CLOUD_BASE_URL,
    WeKnoraCloudService,
    is_weknora_cloud_doc_reader_addr,
    sign_request_headers,
)
from src.db.models.tenants.tenants import Tenant
from tests.fakes.tenants import FakeTenantRepository

_HEALTH_URL = f"{WEKNORA_CLOUD_BASE_URL}/api/v1/health"


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


async def _seed_tenant(repo: FakeTenantRepository, **overrides: object) -> Tenant:
    now = datetime.now(UTC)
    return await repo.insert(
        Tenant(name="acme", created_at=now, updated_at=now, **overrides)  # type: ignore[arg-type]
    )


def _service(repo: FakeTenantRepository, client: httpx.AsyncClient) -> WeKnoraCloudService:
    return WeKnoraCloudService(tenants_repo=repo, http_client=client)  # type: ignore[arg-type]


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


def test_is_weknora_cloud_doc_reader_addr_ignores_trailing_slash() -> None:
    # Arrange
    addr = f"{WEKNORA_CLOUD_BASE_URL}/api/v1/doc/reader/"

    # Act / Assert
    assert is_weknora_cloud_doc_reader_addr(addr) is True
    assert is_weknora_cloud_doc_reader_addr("  " + addr.rstrip("/") + " ") is True
    assert is_weknora_cloud_doc_reader_addr("https://evil.example/api/v1/doc/reader") is False


# ── save_credentials ────────────────────────────────────────────────


async def test_save_credentials_persists_the_pair_under_weknoracloud() -> None:
    # Arrange
    repo = FakeTenantRepository()
    tenant = await _seed_tenant(repo)
    async with _client() as client:
        service = _service(repo, client)

        # Act
        await service.save_credentials(tenant_id=tenant.id, app_id="app", app_secret="secret")

    # Assert
    assert repo.rows[tenant.id].credentials == {
        "weknoracloud": {"app_id": "app", "app_secret": "secret"}
    }


async def test_save_credentials_preserves_other_credential_providers() -> None:
    # Arrange
    repo = FakeTenantRepository()
    tenant = await _seed_tenant(repo, credentials={"other": {"token": "keep-me"}})
    async with _client() as client:
        service = _service(repo, client)

        # Act
        await service.save_credentials(tenant_id=tenant.id, app_id="app", app_secret="secret")

    # Assert
    stored = repo.rows[tenant.id].credentials
    assert stored is not None
    assert stored["other"] == {"token": "keep-me"}
    assert stored["weknoracloud"] == {"app_id": "app", "app_secret": "secret"}


@pytest.mark.parametrize(
    ("app_id", "app_secret", "code"),
    [
        ("", "secret", "weknoracloud.app_id_required"),
        ("app", "", "weknoracloud.app_secret_required"),
    ],
)
async def test_save_credentials_rejects_blank_fields(
    app_id: str,
    app_secret: str,
    code: str,
) -> None:
    # Arrange
    repo = FakeTenantRepository()
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
    repo = FakeTenantRepository()
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
    assert excinfo.value.code == "weknoracloud.invalid_credentials"
    assert repo.rows[tenant.id].credentials is None


async def test_save_credentials_rejects_unexpected_upstream_status() -> None:
    # Arrange
    repo = FakeTenantRepository()
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
    assert excinfo.value.code == "weknoracloud.verification_failed"


async def test_save_credentials_reports_an_unreachable_service() -> None:
    # Arrange
    repo = FakeTenantRepository()
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
    assert excinfo.value.code == "weknoracloud.service_unreachable"
    assert repo.rows[tenant.id].credentials is None


# ── check_status ────────────────────────────────────────────────────


async def test_check_status_reports_no_models_without_credentials() -> None:
    # Arrange
    repo = FakeTenantRepository()
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
    repo = FakeTenantRepository()
    async with _client() as client:
        service = _service(repo, client)

        # Act
        status = await service.check_status(tenant_id=999)

    # Assert — Go swallows the missing tenant rather than erroring
    assert status.has_models is False
    assert status.needs_reinit is False


async def test_check_status_reports_no_models_when_a_field_is_blank() -> None:
    # Arrange
    repo = FakeTenantRepository()
    tenant = await _seed_tenant(
        repo,
        credentials={"weknoracloud": {"app_id": "app", "app_secret": ""}},
    )
    async with _client() as client:
        service = _service(repo, client)

        # Act
        status = await service.check_status(tenant_id=tenant.id)

    # Assert — mirrors ``CredentialsConfig.GetWeKnoraCloud`` returning nil
    assert status.has_models is False
    assert status.needs_reinit is False


async def test_check_status_is_healthy_with_a_decrypted_secret() -> None:
    # Arrange
    repo = FakeTenantRepository()
    tenant = await _seed_tenant(
        repo,
        credentials={"weknoracloud": {"app_id": "app", "app_secret": "plaintext"}},
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
    repo = FakeTenantRepository()
    tenant = await _seed_tenant(
        repo,
        credentials={"weknoracloud": {"app_id": "app", "app_secret": f"{ENC_PREFIX}blob"}},
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
