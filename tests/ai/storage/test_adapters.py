"""Unit tests for the storage file-service adapters.

Every object-store call is mocked at the ``httpx`` transport layer with
``respx``; no real object storage is contacted. The suite covers the
provider:// path conventions, temp-bucket semantics, cross-backend copy
refusals, the factory dispatch and the scoping / resource-catalog
decorators.
"""

from __future__ import annotations

import hashlib
import urllib.parse
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import pytest
import respx

from src.ai.storage.backend_scoped import BackendScopedFileService
from src.ai.storage.base import (
    FileUpload,
    build_storage_backend_path,
    parse_storage_backend_path,
    parse_tenant_id_from_storage_path,
)
from src.ai.storage.cos_backend import CosStorageAdapter
from src.ai.storage.dummy_backend import DummyFileService
from src.ai.storage.errors import CrossBackendCopyError
from src.ai.storage.factory import new_file_service_from_storage_config
from src.ai.storage.ks3_backend import KS3FileService
from src.ai.storage.local_backend import LocalStorageAdapter
from src.ai.storage.minio_backend import MinioStorageAdapter
from src.ai.storage.obs_backend import ObsStorageAdapter
from src.ai.storage.oss_backend import OssFileService
from src.ai.storage.resource_catalog import (
    ResourceCatalogFileService,
    ResourceRegistration,
    new_resource_catalog_file_service,
)
from src.ai.storage.s3_backend import S3StorageAdapter
from src.ai.storage.tos_backend import TosFileService
from src.common.exception import StorageBackendError, ValidationError

_TENANT = 7
_KB = "kb1"

_UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"


# ── Shared fakes ───────────────────────────────────────────────────────


class FakeUpload:
    """A ``FileUpload`` that serves in-memory bytes."""

    def __init__(self, filename: str, data: bytes, content_type: str = "") -> None:
        self.filename = filename
        self.size = len(data)
        self.content_type = content_type
        self._data = data

    async def read(self) -> bytes:
        return self._data


class RecordingFileService:
    """A ``FileService`` that records calls instead of touching storage."""

    def __init__(self) -> None:
        self.saved_files: list[tuple[str, int]] = []
        self.saved_bytes: list[tuple[bytes, str, bool]] = []
        self.deleted: list[str] = []
        self.copied: list[tuple[str, int, str]] = []
        self.get_file_url_calls: list[str] = []

    async def check_connectivity(self) -> None:
        return None

    async def save_file(
        self, *, file: FileUpload, tenant_id: int, knowledge_id: str
    ) -> str:
        self.saved_files.append((file.filename, file.size))
        return f"local://{tenant_id}/{knowledge_id}/photo.png"

    async def save_bytes(
        self, *, data: bytes, tenant_id: int, file_name: str, temp: bool
    ) -> str:
        self.saved_bytes.append((data, file_name, temp))
        return f"local://{tenant_id}/exports/out.txt"

    async def get_file(self, file_path: str) -> BinaryIO:
        return BytesIO(b"content")

    async def get_file_url(self, file_path: str) -> str:
        self.get_file_url_calls.append(file_path)
        return f"https://files.example.com/{file_path}"

    async def delete_file(self, file_path: str) -> None:
        self.deleted.append(file_path)

    async def copy_file(self, src_path: str, tenant_id: int, knowledge_id: str) -> str:
        self.copied.append((src_path, tenant_id, knowledge_id))
        return f"local://{tenant_id}/{knowledge_id}/copy.png"


class FakeCatalog:
    """A ``ResourceCatalog`` that records registrations and grants."""

    def __init__(self) -> None:
        self.registered: list[tuple[int, str, ResourceRegistration]] = []
        self.bindings: list[tuple[str, str, str, str]] = []
        self.deleted: list[str] = []
        self.grant_tokens: list[str] = []
        self._resolve: dict[str, tuple[str, object | None]] = {}
        self._next_ref = 1

    async def register(
        self, *, tenant_id: int, physical_path: str, meta: ResourceRegistration
    ) -> str:
        self.registered.append((tenant_id, physical_path, meta))
        ref = f"resource://{self._next_ref}"
        self._next_ref += 1
        self._resolve[ref] = (physical_path, object())
        return ref

    async def resolve_path(self, value: str) -> tuple[str, object | None]:
        return self._resolve.get(value, (value, None))

    async def bind(self, reference: str, owner_type: str, owner_id: str, relation: str) -> None:
        self.bindings.append((reference, owner_type, owner_id, relation))

    async def mark_deleted(self, reference: str) -> None:
        self.deleted.append(reference)

    async def create_access_grant(self, reference: str, ttl_seconds: int) -> str:
        self.grant_tokens.append(reference)
        return f"token-{len(self.grant_tokens)}"


class FailingCatalog(FakeCatalog):
    """A catalog whose ``register`` always refuses."""

    async def register(
        self, *, tenant_id: int, physical_path: str, meta: ResourceRegistration
    ) -> str:
        raise StorageBackendError(code="catalog.full", message="no space")


@dataclass(frozen=True)
class StorageConfigStub:
    """Structural stand-in for the normalized storage config."""

    default_provider: str = ""
    mode: str = ""
    endpoint: str = ""
    region: str = ""
    access_key_id: str = ""
    secret_access_key: str = ""
    bucket_name: str = ""
    path_prefix: str = ""
    app_id: str = ""
    use_ssl: bool = False
    force_path_style: bool = False
    use_temp_bucket: bool = False
    temp_bucket_name: str = ""
    temp_region: str = ""


def _key_regex(prefix: str, scope: str, ext: str = r"\.png$") -> str:
    """A regex matching ``{prefix}{tenant}/{scope}/{uuid}{ext}``."""
    return rf"/{prefix}{_TENANT}/{scope}/{_UUID}{ext}"


# ── Local backend ──────────────────────────────────────────────────────


async def test_local_save_and_read_roundtrip(tmp_path: Path) -> None:
    adapter = LocalStorageAdapter(base_dir=str(tmp_path))
    path = await adapter.save_file(
        file=FakeUpload("report.png", b"hello", "image/png"),
        tenant_id=_TENANT,
        knowledge_id=_KB,
    )
    assert path.startswith(f"local://{_TENANT}/{_KB}/")
    handle = await adapter.get_file(path)
    try:
        assert handle.read() == b"hello"
    finally:
        handle.close()

    url = await adapter.get_file_url(path)
    assert url == path

    await adapter.delete_file(path)
    with pytest.raises(StorageBackendError):
        await adapter.get_file(path)


async def test_local_save_bytes_lands_in_exports(tmp_path: Path) -> None:
    adapter = LocalStorageAdapter(base_dir=str(tmp_path))
    path = await adapter.save_bytes(
        data=b"data", tenant_id=_TENANT, file_name="out.csv", temp=False
    )
    assert path.startswith(f"local://{_TENANT}/exports/")
    assert await adapter.get_file_url(path) == path


async def test_local_save_bytes_rejects_traversal(tmp_path: Path) -> None:
    adapter = LocalStorageAdapter(base_dir=str(tmp_path))
    with pytest.raises(ValidationError):
        await adapter.save_bytes(
            data=b"x", tenant_id=_TENANT, file_name="a/../..", temp=False
        )
    with pytest.raises(ValidationError):
        await adapter.save_bytes(data=b"x", tenant_id=_TENANT, file_name="..", temp=False)


async def test_local_copy_same_backend_and_cross_backend(tmp_path: Path) -> None:
    adapter = LocalStorageAdapter(base_dir=str(tmp_path))
    src = await adapter.save_file(
        file=FakeUpload("a.txt", b"abc"), tenant_id=_TENANT, knowledge_id=_KB
    )
    copied = await adapter.copy_file(src, tenant_id=_TENANT, knowledge_id="kb2")
    assert copied.startswith("local://")
    assert copied != src
    handle = await adapter.get_file(copied)
    try:
        assert handle.read() == b"abc"
    finally:
        handle.close()

    with pytest.raises(CrossBackendCopyError):
        await adapter.copy_file("s3://bucket/7/kb1/a.txt", tenant_id=_TENANT, knowledge_id="kb2")


async def test_local_get_file_url_presigns_with_external_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SYSTEM_AES_KEY", "k" * 32)
    monkeypatch.setenv("APP_EXTERNAL_URL", "https://app.example.com/")
    adapter = LocalStorageAdapter(base_dir=str(tmp_path))
    path = await adapter.save_file(
        file=FakeUpload("img.png", b"img", "image/png"), tenant_id=_TENANT, knowledge_id=_KB
    )
    url = await adapter.get_file_url(path)
    assert url.startswith("https://app.example.com/api/v1/files/presigned?")
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
    assert query["file_path"] == [path]
    assert query["tenant_id"] == [str(_TENANT)]
    assert "sig" in query


async def test_local_probe_and_ensure_directory(tmp_path: Path) -> None:
    adapter = LocalStorageAdapter(base_dir=str(tmp_path), path_prefix="nested")
    await adapter.ensure_directory()
    await adapter.check_connectivity()


# ── S3 backend ─────────────────────────────────────────────────────────


async def test_s3_save_get_delete_copy_roundtrip() -> None:
    adapter = S3StorageAdapter(
        endpoint="s3.example.com",
        region="us-east-1",
        access_key_id="AKID",
        secret_access_key="secret",
        bucket_name="documents",
        use_ssl=True,
        force_path_style=False,
        path_prefix="weknora/",
    )
    with respx.mock(base_url="https://documents.s3.example.com") as router:
        put_route = router.put(path__regex=_key_regex("weknora/", _KB))
        put_route.respond(200)
        get_route = router.get(path__regex=_key_regex("weknora/", _KB))
        get_route.respond(200, content=b"hello")
        delete_route = router.delete(path__regex=_key_regex("weknora/", _KB))
        delete_route.respond(204)

        path = await adapter.save_file(
            file=FakeUpload("report.png", b"hello", "image/png"),
            tenant_id=_TENANT,
            knowledge_id=_KB,
        )
        assert path.startswith(f"s3://documents/weknora/{_TENANT}/{_KB}/")
        assert path.endswith(".png")
        assert put_route.calls.last.request.content == b"hello"

        handle = await adapter.get_file(path)
        try:
            assert handle.read() == b"hello"
        finally:
            handle.close()

        url = await adapter.get_file_url(path)
        assert url.startswith("https://documents.s3.example.com/")
        assert "X-Amz-Expires=86400" in url
        assert "X-Amz-Signature=" in url

        await adapter.delete_file(path)
        assert delete_route.called


async def test_s3_copy_copies_into_knowledge_layout() -> None:
    adapter = S3StorageAdapter(
        endpoint="s3.example.com",
        region="us-east-1",
        access_key_id="AKID",
        secret_access_key="secret",
        bucket_name="documents",
        use_ssl=True,
        path_prefix="weknora/",
    )
    src = f"s3://documents/weknora/{_TENANT}/kb-old/abc.png"
    with respx.mock(base_url="https://documents.s3.example.com") as router:
        copy_route = router.put(path__regex=_key_regex("weknora/", _KB))
        copy_route.respond(200)
        copied = await adapter.copy_file(src, tenant_id=_TENANT, knowledge_id=_KB)
    assert copied.startswith(f"s3://documents/weknora/{_TENANT}/{_KB}/")
    assert (
        copy_route.calls.last.request.headers["x-amz-copy-source"]
        == f"documents/weknora/{_TENANT}/kb-old/abc.png"
    )


async def test_s3_copy_cross_backend_refused() -> None:
    adapter = S3StorageAdapter(
        endpoint="s3.example.com",
        region="us-east-1",
        access_key_id="AKID",
        secret_access_key="secret",
        bucket_name="documents",
        use_ssl=True,
    )
    with pytest.raises(CrossBackendCopyError):
        await adapter.copy_file("minio://documents/7/kb1/a.png", tenant_id=_TENANT, knowledge_id=_KB)


async def test_s3_probe_ok_and_missing_bucket() -> None:
    adapter = S3StorageAdapter(
        endpoint="s3.example.com",
        region="us-east-1",
        access_key_id="AKID",
        secret_access_key="secret",
        bucket_name="documents",
        use_ssl=True,
    )
    with respx.mock(base_url="https://documents.s3.example.com") as router:
        router.head(path="/").respond(200)
        await adapter.check_connectivity()
    with respx.mock(base_url="https://documents.s3.example.com") as router:
        router.head(path="/").respond(404)
        with pytest.raises(StorageBackendError):
            await adapter.check_connectivity()


# ── MinIO backend ──────────────────────────────────────────────────────


async def test_minio_save_and_save_bytes_paths() -> None:
    adapter = MinioStorageAdapter(
        endpoint="minio.example.com:9000",
        access_key_id="AKID",
        secret_access_key="secret",
        bucket_name="documents",
    )
    with respx.mock(base_url="http://minio.example.com:9000") as router:
        router.put(path__regex=rf"/documents/{_TENANT}/{_KB}/{_UUID}\.png$").respond(200)
        path = await adapter.save_file(
            file=FakeUpload("a.png", b"img"), tenant_id=_TENANT, knowledge_id=_KB
        )
        assert path.startswith(f"minio://documents/{_TENANT}/{_KB}/")

        router.put(path__regex=rf"/documents/{_TENANT}/exports/{_UUID}\.txt$").respond(200)
        bytes_path = await adapter.save_bytes(
            data=b"out", tenant_id=_TENANT, file_name="out.txt", temp=False
        )
        assert bytes_path.startswith(f"minio://documents/{_TENANT}/exports/")


async def test_minio_get_url_and_delete() -> None:
    adapter = MinioStorageAdapter(
        endpoint="minio.example.com:9000",
        access_key_id="AKID",
        secret_access_key="secret",
        bucket_name="documents",
    )
    path = f"minio://documents/{_TENANT}/{_KB}/x.png"
    with respx.mock(base_url="http://minio.example.com:9000") as router:
        router.get(path__regex=rf"/documents/{_TENANT}/{_KB}/x\.png$").respond(200, content=b"data")
        handle = await adapter.get_file(path)
        try:
            assert handle.read() == b"data"
        finally:
            handle.close()
        url = await adapter.get_file_url(path)
        assert url.startswith("http://minio.example.com:9000/documents/")
        assert "X-Amz-Signature=" in url
        router.delete(path__regex=rf"/documents/{_TENANT}/{_KB}/x\.png$").respond(204)
        await adapter.delete_file(path)


# ── TOS backend ────────────────────────────────────────────────────────


async def test_tos_temp_bucket_save_bytes() -> None:
    svc = TosFileService(
        endpoint="tos.example.com",
        region="cn-beijing",
        access_key="AK",
        secret_key="SK",
        bucket_name="tos-main",
        path_prefix="weknora",
        temp_bucket_name="tos-temp",
    )
    with respx.mock(base_url="https://tos-main.tos.example.com") as router:
        router.put(path__regex=_key_regex("weknora/", "exports", r"\.json$")).respond(200)
        path = await svc.save_bytes(
            data=b"data", tenant_id=_TENANT, file_name="out.json", temp=False
        )
        assert path.startswith(f"tos://tos-main/weknora/{_TENANT}/exports/")
    with respx.mock(base_url="https://tos-temp.tos.example.com") as router:
        router.put(path__regex=rf"/exports/{_TENANT}/{_UUID}\.json$").respond(200)
        temp_path = await svc.save_bytes(
            data=b"data", tenant_id=_TENANT, file_name="out.json", temp=True
        )
        assert temp_path.startswith(f"tos://tos-temp/exports/{_TENANT}/")


async def test_tos_copy_cross_backend_refused() -> None:
    svc = TosFileService(
        endpoint="tos.example.com",
        region="cn-beijing",
        access_key="AK",
        secret_key="SK",
        bucket_name="tos-main",
    )
    with pytest.raises(CrossBackendCopyError):
        await svc.copy_file("oss://other/7/kb1/a.png", tenant_id=_TENANT, knowledge_id=_KB)


# ── OSS backend ────────────────────────────────────────────────────────


async def test_oss_temp_bucket_routes_reads_by_bucket() -> None:
    svc = OssFileService(
        endpoint="oss.example.com",
        region="cn-shanghai",
        access_key="AK",
        secret_key="SK",
        bucket_name="oss-main",
        path_prefix="weknora/",
        temp_bucket_name="oss-temp",
    )
    with respx.mock(base_url="https://oss-main.oss.example.com") as router:
        router.put(path__regex=_key_regex("weknora/", _KB)).respond(200)
        path = await svc.save_file(
            file=FakeUpload("a.png", b"img"), tenant_id=_TENANT, knowledge_id=_KB
        )
        assert path.startswith(f"oss://oss-main/weknora/{_TENANT}/{_KB}/")

    temp_path = f"oss://oss-temp/exports/{_TENANT}/x.json"
    with respx.mock(base_url="https://oss-temp.oss.example.com") as router:
        router.get(path__regex=rf"/exports/{_TENANT}/x\.json$").respond(200, content=b"temp")
        handle = await svc.get_file(temp_path)
        try:
            assert handle.read() == b"temp"
        finally:
            handle.close()


# ── KS3 backend ────────────────────────────────────────────────────────


async def test_ks3_save_and_copy() -> None:
    svc = KS3FileService(
        endpoint="ks3.example.com",
        region="cn-north-1",
        access_key="AK",
        secret_key="SK",
        bucket_name="ks3-main",
        path_prefix="weknora",
    )
    with respx.mock(base_url="https://ks3-main.ks3.example.com") as router:
        router.put(path__regex=_key_regex("weknora/", _KB)).respond(200)
        path = await svc.save_file(
            file=FakeUpload("a.png", b"img"), tenant_id=_TENANT, knowledge_id=_KB
        )
        assert path.startswith(f"ks3://ks3-main/weknora/{_TENANT}/{_KB}/")

        router.put(path__regex=_key_regex("weknora/", _KB)).respond(200)
        copied = await svc.copy_file(
            f"ks3://ks3-main/weknora/{_TENANT}/old/a.png", tenant_id=_TENANT, knowledge_id=_KB
        )
        assert copied.startswith(f"ks3://ks3-main/weknora/{_TENANT}/{_KB}/")


# ── COS backend ────────────────────────────────────────────────────────


async def test_cos_save_path_contains_bucket_and_region() -> None:
    adapter = CosStorageAdapter(
        region="ap-guangzhou",
        access_key_id="AK",
        secret_access_key="SK",
        bucket_name="cos-bucket",
        path_prefix="weknora",
    )
    with respx.mock(base_url="https://cos-bucket.cos.ap-guangzhou.myqcloud.com") as router:
        router.put(path__regex=_key_regex("weknora/", _KB)).respond(200)
        path = await adapter.save_file(
            file=FakeUpload("a.png", b"img"), tenant_id=_TENANT, knowledge_id=_KB
        )
    assert path.startswith(f"cos://cos-bucket/ap-guangzhou/weknora/{_TENANT}/{_KB}/")


async def test_cos_temp_bucket_returns_legacy_url() -> None:
    adapter = CosStorageAdapter(
        region="ap-guangzhou",
        access_key_id="AK",
        secret_access_key="SK",
        bucket_name="cos-bucket",
        path_prefix="weknora",
        temp_bucket_name="cos-temp",
        temp_region="ap-shanghai",
    )
    with respx.mock(base_url="https://cos-temp.cos.ap-shanghai.myqcloud.com") as router:
        router.put(path__regex=rf"/exports/{_TENANT}/{_UUID}\.txt$").respond(200)
        temp_path = await adapter.save_bytes(
            data=b"data", tenant_id=_TENANT, file_name="out.txt", temp=True
        )
        assert temp_path.startswith("https://cos-temp.cos.ap-shanghai.myqcloud.com/exports/")

    # GetFileURL for the temp URL presigns against the temp bucket (no HTTP).
    url = await adapter.get_file_url(temp_path)
    assert url.startswith("https://cos-temp.cos.ap-shanghai.myqcloud.com/")
    assert "X-Amz-Signature=" in url


async def test_cos_copy_cross_backend_refused() -> None:
    adapter = CosStorageAdapter(
        region="ap-guangzhou",
        access_key_id="AK",
        secret_access_key="SK",
        bucket_name="cos-bucket",
    )
    with pytest.raises(CrossBackendCopyError):
        await adapter.copy_file("tos://other/7/kb1/a.png", tenant_id=_TENANT, knowledge_id=_KB)


# ── OBS backend ────────────────────────────────────────────────────────


async def test_obs_path_style_and_proxy_domain() -> None:
    adapter = ObsStorageAdapter(
        endpoint="https://obs.cn-north-4.example.com",
        region="cn-north-4",
        access_key_id="AK",
        secret_access_key="SK",
        bucket_name="obs-bucket",
        path_prefix="weknora",
    )
    with respx.mock(base_url="https://obs.cn-north-4.example.com") as router:
        router.put(
            path__regex=rf"/obs-bucket/weknora/{_TENANT}/{_KB}/{_UUID}\.png$"
        ).respond(200)
        path = await adapter.save_file(
            file=FakeUpload("a.png", b"img"), tenant_id=_TENANT, knowledge_id=_KB
        )
        assert path.startswith(f"obs://obs-bucket/weknora/{_TENANT}/{_KB}/")
        url = await adapter.get_file_url(path)
        assert url.startswith("https://obs.cn-north-4.example.com/obs-bucket/weknora/")


async def test_obs_proxy_domain_path_format() -> None:
    adapter = ObsStorageAdapter(
        endpoint="https://obs.cn-north-4.example.com",
        region="cn-north-4",
        access_key_id="AK",
        secret_access_key="SK",
        bucket_name="obs-bucket",
        path_prefix="weknora",
        proxy_domain="https://files.example.com",
    )
    with respx.mock(base_url="https://obs.cn-north-4.example.com") as router:
        router.put(
            path__regex=rf"/obs-bucket/weknora/{_TENANT}/{_KB}/{_UUID}\.png$"
        ).respond(200)
        path = await adapter.save_file(
            file=FakeUpload("a.png", b"img"), tenant_id=_TENANT, knowledge_id=_KB
        )
        assert path.startswith("https://files.example.com/weknora/")
        url = await adapter.get_file_url(path)
        assert url.startswith("https://files.example.com/weknora/")

    with respx.mock(base_url="https://obs.cn-north-4.example.com") as router:
        router.delete(
            path__regex=rf"/obs-bucket/weknora/{_TENANT}/{_KB}/{_UUID}\.png$"
        ).respond(204)
        await adapter.delete_file(path)


# ── Dummy backend ──────────────────────────────────────────────────────


async def test_dummy_service_semantics() -> None:
    svc = DummyFileService()
    await svc.check_connectivity()
    path = await svc.save_file(file=FakeUpload("a.png", b"x"), tenant_id=_TENANT, knowledge_id=_KB)
    assert path.startswith(f"dummy://{_TENANT}/")
    bytes_path = await svc.save_bytes(
        data=b"x", tenant_id=_TENANT, file_name="a.txt", temp=False
    )
    assert bytes_path.startswith(f"dummy://{_TENANT}/")
    assert await svc.get_file_url(path) == path
    await svc.delete_file(path)
    assert await svc.copy_file(path, tenant_id=_TENANT, knowledge_id=_KB) == path
    with pytest.raises(StorageBackendError):
        await svc.get_file(path)


# ── Factory ────────────────────────────────────────────────────────────


async def test_factory_dispatch_all_providers() -> None:
    cases = [
        ("s3", S3StorageAdapter),
        ("minio", MinioStorageAdapter),
        ("cos", CosStorageAdapter),
        ("obs", ObsStorageAdapter),
        ("tos", TosFileService),
        ("oss", OssFileService),
        ("ks3", KS3FileService),
        ("local", LocalStorageAdapter),
    ]
    for provider, expected in cases:
        endpoint = "storage.example.com" if provider not in ("local", "cos") else ""
        config = StorageConfigStub(
            endpoint=endpoint,
            region="r",
            access_key_id="AK",
            secret_access_key="SK",
            bucket_name="b",
        )
        service, resolved = new_file_service_from_storage_config(provider, config)
        assert resolved == provider
        assert isinstance(service, expected)


async def test_factory_empty_provider_falls_back_to_default() -> None:
    config = StorageConfigStub(
        default_provider="minio",
        endpoint="minio.example.com",
        access_key_id="AK",
        secret_access_key="SK",
        bucket_name="b",
    )
    service, resolved = new_file_service_from_storage_config("", config)
    assert resolved == "minio"
    assert isinstance(service, MinioStorageAdapter)


async def test_factory_empty_provider_refused() -> None:
    with pytest.raises(StorageBackendError):
        new_file_service_from_storage_config("")


async def test_factory_unsupported_provider_refused() -> None:
    with pytest.raises(StorageBackendError):
        new_file_service_from_storage_config("gcs", StorageConfigStub())


async def test_factory_incomplete_config_refused() -> None:
    with pytest.raises(StorageBackendError):
        new_file_service_from_storage_config("s3", StorageConfigStub())


# ── Backend scoping decorator ──────────────────────────────────────────


async def test_backend_scoped_wraps_and_unwraps_paths() -> None:
    inner = RecordingFileService()
    scoped = BackendScopedFileService(backend_id="backend-a", inner=inner)

    path = await scoped.save_file(
        file=FakeUpload("a.png", b"x"), tenant_id=_TENANT, knowledge_id=_KB
    )
    assert path == f"storage://backend-a/local://{_TENANT}/{_KB}/photo.png"

    url = await scoped.get_file_url(path)
    assert url == f"https://files.example.com/local://{_TENANT}/{_KB}/photo.png"
    assert inner.get_file_url_calls == [f"local://{_TENANT}/{_KB}/photo.png"]

    copied = await scoped.copy_file(path, tenant_id=_TENANT, knowledge_id="kb2")
    assert copied.startswith("storage://backend-a/local://")

    await scoped.delete_file(path)
    assert inner.deleted == [f"local://{_TENANT}/{_KB}/photo.png"]


async def test_backend_scoped_rejects_foreign_backend() -> None:
    scoped = BackendScopedFileService(backend_id="backend-a", inner=RecordingFileService())
    with pytest.raises(StorageBackendError):
        await scoped.delete_file("storage://backend-b/local://7/kb1/x.png")


async def test_backend_scoped_re_signs_local_presigned_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SYSTEM_AES_KEY", "k" * 32)

    class PresignedInner(RecordingFileService):
        async def get_file_url(self, file_path: str) -> str:
            self.get_file_url_calls.append(file_path)
            base = "https://app.example.com/api/v1/files/presigned"
            return f"{base}?file_path={file_path}&tenant_id={_TENANT}&expires=9999999999&sig=abc"

    scoped = BackendScopedFileService(backend_id="backend-a", inner=PresignedInner())
    url = await scoped.get_file_url("storage://backend-a/local://7/kb1/photo.png")
    assert url.startswith("https://app.example.com/api/v1/files/presigned?")
    assert "file_path=storage%3A%2F%2Fbackend-a%2Flocal%3A%2F%2F7%2Fkb1%2Fphoto.png" in url
    assert "sig=" in url


# ── Resource catalog decorator ─────────────────────────────────────────


async def test_resource_catalog_registers_and_binds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_EXTERNAL_URL", "https://app.example.com")
    inner = RecordingFileService()
    catalog = FakeCatalog()
    svc = new_resource_catalog_file_service(inner, catalog)
    assert isinstance(svc, ResourceCatalogFileService)

    ref = await svc.save_file(
        file=FakeUpload("report.pdf", b"pdf"), tenant_id=_TENANT, knowledge_id=_KB
    )
    assert ref.startswith("resource://")
    assert catalog.bindings == [(ref, "knowledge", _KB, "source_file")]
    assert catalog.registered[0][1] == f"local://{_TENANT}/{_KB}/photo.png"
    assert catalog.registered[0][2].kind == "file"
    assert catalog.registered[0][2].mime_type == "application/pdf"

    url = await svc.get_file_url(ref)
    assert url == "https://app.example.com/r/token-1"

    await svc.delete_file(ref)
    assert catalog.deleted == [ref]
    assert inner.deleted == [f"local://{_TENANT}/{_KB}/photo.png"]


async def test_resource_catalog_save_bytes_registers_content_hash() -> None:
    inner = RecordingFileService()
    catalog = FakeCatalog()
    svc = ResourceCatalogFileService(inner=inner, catalog=catalog)

    ref = await svc.save_bytes(
        data=b"payload", tenant_id=_TENANT, file_name="out.txt", temp=True
    )
    assert ref.startswith("resource://")
    meta = catalog.registered[0][2]
    assert meta.temporary is True
    assert meta.size == len(b"payload")
    assert meta.content_hash == hashlib.sha256(b"payload").hexdigest()


async def test_resource_catalog_copy_registers_new_resource() -> None:
    inner = RecordingFileService()
    catalog = FakeCatalog()
    svc = ResourceCatalogFileService(inner=inner, catalog=catalog)

    ref = await svc.copy_file(
        f"local://{_TENANT}/{_KB}/photo.png", tenant_id=_TENANT, knowledge_id="kb2"
    )
    assert ref.startswith("resource://")
    assert inner.copied == [(f"local://{_TENANT}/{_KB}/photo.png", _TENANT, "kb2")]
    assert catalog.bindings == [(ref, "knowledge", "kb2", "source_file")]


async def test_resource_catalog_register_failure_deletes_physical() -> None:
    inner = RecordingFileService()
    svc = ResourceCatalogFileService(inner=inner, catalog=FailingCatalog())
    with pytest.raises(StorageBackendError):
        await svc.save_file(
            file=FakeUpload("a.png", b"x"), tenant_id=_TENANT, knowledge_id=_KB
        )
    assert inner.deleted == [f"local://{_TENANT}/{_KB}/photo.png"]


async def test_resource_catalog_returns_inner_when_unconfigured() -> None:
    inner = RecordingFileService()
    assert new_resource_catalog_file_service(inner, None) is inner
    assert new_resource_catalog_file_service(None, FakeCatalog()) is None


# ── Path helpers ───────────────────────────────────────────────────────


async def test_storage_backend_path_helpers() -> None:
    wrapped = build_storage_backend_path("backend-a", "cos://b/r/k")
    assert wrapped == "storage://backend-a/cos://b/r/k"
    assert parse_storage_backend_path(wrapped) == ("backend-a", "cos://b/r/k")
    assert parse_storage_backend_path("local://7/kb/x.png") is None
    assert parse_tenant_id_from_storage_path("local://7/kb/x.png") == 7
    assert (
        parse_tenant_id_from_storage_path("storage://backend-a/s3://documents/7/kb/x.png") == 7
    )
    assert parse_tenant_id_from_storage_path("s3://documents/kb/x.png") == 0


async def test_factory_obs_reads_env_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OBS_ENDPOINT", "https://obs.example.com")
    monkeypatch.setenv("OBS_ACCESS_KEY", "AK")
    monkeypatch.setenv("OBS_SECRET_KEY", "SK")
    monkeypatch.setenv("OBS_BUCKET_NAME", "obs-bucket")
    service, resolved = new_file_service_from_storage_config("obs", None)
    assert resolved == "obs"
    assert isinstance(service, ObsStorageAdapter)
