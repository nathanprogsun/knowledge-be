"""Tests for the docreader gRPC client.

The gRPC transport is mocked at the stub / call level (``_FakeStub`` and
``_FakeStreamCall``) so no live server is ever contacted. The tests assert:

- auth configuration loading from the environment and dial-option building,
- token attachment as a per-RPC ``authorization`` header,
- unary Read (parse) and ListEngines round-trips,
- the server-streaming ReadStream (progress) including cancellation,
- ``progress`` decoding of the meta-then-images frame sequence,
- gRPC ``RpcError`` failures surfacing as ``ExternalServiceError``.
"""

# mypy: disable-error-code="import-untyped"

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import grpc
import pytest

from src.ai.docreader.client import (
    AuthConfig,
    DocReaderClient,
    ImageRefInfo,
    build_dial_options,
    get_image_refs_from_response,
    get_max_message_size,
    load_auth_config_from_env,
    new_client,
    new_client_with_auth,
)
from src.ai.docreader.proto import docreader_pb2 as pb2
from src.common.exception import ExternalServiceError


class _FakeChannel:
    """Minimal stand-in for ``grpc.Channel``; only used by ``close``."""

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeStreamCall:
    """Stand-in for the gRPC server-streaming iterator with ``cancel``."""

    def __init__(self) -> None:
        self.cancelled = False
        self.frames: list[pb2.ReadStreamResponse] = []

    def __iter__(self) -> Iterator[pb2.ReadStreamResponse]:
        return iter(self.frames)

    def cancel(self) -> None:
        self.cancelled = True


class _RaisingStream:
    """Iterable whose iteration raises a gRPC ``RpcError`` (stream failure)."""

    def __iter__(self) -> Iterator[Any]:
        raise _fake_rpc_error("UNAVAILABLE", "stream unavailable")


class _FakeStub:
    """Captures RPC kwargs and returns canned results / errors."""

    def __init__(self) -> None:
        self.read_calls: list[tuple[Any, dict[str, Any]]] = []
        self.stream_calls: list[tuple[Any, dict[str, Any]]] = []
        self.engines_calls: list[tuple[Any, dict[str, Any]]] = []
        self.read_result: Any = pb2.ReadResponse()
        self.read_error: Any = None
        self.stream_result: Any = None
        self.stream_error: Any = None
        self.engines_result: Any = pb2.ListEnginesResponse()
        self.engines_error: Any = None

    def Read(self, request: Any, **kwargs: Any) -> Any:
        self.read_calls.append((request, kwargs))
        if self.read_error is not None:
            raise self.read_error
        return self.read_result

    def ReadStream(self, request: Any, **kwargs: Any) -> Any:
        self.stream_calls.append((request, kwargs))
        if self.stream_error is not None:
            raise self.stream_error
        return self.stream_result

    def ListEngines(self, request: Any, **kwargs: Any) -> Any:
        self.engines_calls.append((request, kwargs))
        if self.engines_error is not None:
            raise self.engines_error
        return self.engines_result


def _fake_rpc_error(code: str, detail: str) -> grpc.RpcError:
    err = grpc.RpcError()
    err.code = lambda: code
    err.details = lambda: detail
    return err


def _client(stub: _FakeStub, **auth: Any) -> DocReaderClient:
    return DocReaderClient(_FakeChannel(), stub=stub, auth=AuthConfig(**auth))


# ─────────────────────────────────────────────────────────────────────
# Auth configuration & dial options
# ─────────────────────────────────────────────────────────────────────


def test_load_auth_config_from_env_reads_knobs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRPC_TLS_ENABLED", "true")
    monkeypatch.setenv("GRPC_TLS_CERT", "/certs/cert.pem")
    monkeypatch.setenv("GRPC_TLS_KEY", "/certs/key.pem")
    monkeypatch.setenv("GRPC_TLS_CA", "/certs/ca.pem")
    monkeypatch.setenv("GRPC_TLS_SERVER_NAME", "docreader.example.com")
    monkeypatch.setenv("GRPC_AUTH_TOKEN", "secret-token")

    config = load_auth_config_from_env()

    assert config == AuthConfig(
        tls_enabled=True,
        cert_file="/certs/cert.pem",
        key_file="/certs/key.pem",
        ca_file="/certs/ca.pem",
        server_name="docreader.example.com",
        auth_token="secret-token",
    )


def test_load_auth_config_from_env_defaults_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "GRPC_TLS_ENABLED",
        "GRPC_TLS_CERT",
        "GRPC_TLS_KEY",
        "GRPC_TLS_CA",
        "GRPC_TLS_SERVER_NAME",
        "GRPC_AUTH_TOKEN",
    ):
        monkeypatch.delenv(key, raising=False)

    assert load_auth_config_from_env() == AuthConfig()


def test_get_max_message_size_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MAX_FILE_SIZE_MB", raising=False)
    assert get_max_message_size() == 50 * 1024 * 1024


def test_get_max_message_size_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_FILE_SIZE_MB", "10")
    assert get_max_message_size() == 10 * 1024 * 1024


def test_get_max_message_size_ignores_invalid_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_FILE_SIZE_MB", "not-a-number")
    assert get_max_message_size() == 50 * 1024 * 1024


def test_build_dial_options_insecure_defaults() -> None:
    options, credentials = build_dial_options(2048)

    assert credentials is None
    assert ("grpc.max_receive_message_length", 2048) in options
    assert ("grpc.max_send_message_length", 2048) in options
    assert ('{"loadBalancingPolicy":"round_robin"}' in options[0][1])


def test_build_dial_options_tls_with_ca(tmp_path: Path) -> None:
    ca = tmp_path / "ca.pem"
    ca.write_bytes(b"fake-ca-pem")

    options, credentials = build_dial_options(
        2048,
        AuthConfig(tls_enabled=True, ca_file=str(ca), server_name="docreader.example.com"),
    )

    assert credentials is not None
    assert ("grpc.ssl_target_name_override", "docreader.example.com") in options


def test_build_dial_options_tls_requires_cert_and_key_pair(tmp_path: Path) -> None:
    cert = tmp_path / "cert.pem"
    cert.write_bytes(b"fake-cert")

    with pytest.raises(ExternalServiceError) as excinfo:
        build_dial_options(2048, AuthConfig(tls_enabled=True, cert_file=str(cert)))

    assert "must be set together" in excinfo.value.message


def test_build_dial_options_tls_missing_ca_file_raises() -> None:
    with pytest.raises(ExternalServiceError) as excinfo:
        build_dial_options(2048, AuthConfig(tls_enabled=True, ca_file="/no/such/ca.pem"))

    assert "failed to read TLS file" in excinfo.value.message


def test_new_client_with_auth_and_new_client_build_and_close() -> None:
    client = new_client_with_auth("localhost:50051")
    assert isinstance(client, DocReaderClient)
    client.close()

    client = new_client("localhost:50051")
    assert isinstance(client, DocReaderClient)
    client.close()


# ─────────────────────────────────────────────────────────────────────
# Unary Read (parse)
# ─────────────────────────────────────────────────────────────────────


def test_read_passes_request_and_returns_response() -> None:
    stub = _FakeStub()
    stub.read_result = pb2.ReadResponse(markdown_content="# parsed")
    client = _client(stub)

    response = client.read(pb2.ReadRequest(file_name="doc.pdf", file_type="pdf"))

    assert response.markdown_content == "# parsed"
    request, kwargs = stub.read_calls[0]
    assert request.file_name == "doc.pdf"
    assert kwargs["metadata"] == []


def test_read_attaches_bearer_token_header() -> None:
    stub = _FakeStub()
    client = _client(stub, auth_token="tok")

    client.read(pb2.ReadRequest())

    _, kwargs = stub.read_calls[0]
    assert ("authorization", "Bearer tok") in kwargs["metadata"]


def test_read_merges_caller_metadata_with_token() -> None:
    stub = _FakeStub()
    client = _client(stub, auth_token="tok")

    client.read(pb2.ReadRequest(), metadata=[("x-trace", "abc")])

    _, kwargs = stub.read_calls[0]
    assert ("x-trace", "abc") in kwargs["metadata"]
    assert ("authorization", "Bearer tok") in kwargs["metadata"]


def test_read_wraps_rpc_error_as_service_error() -> None:
    stub = _FakeStub()
    stub.read_error = _fake_rpc_error("UNAVAILABLE", "service down")
    client = _client(stub)

    with pytest.raises(ExternalServiceError) as excinfo:
        client.read(pb2.ReadRequest())

    assert excinfo.value.code == "docreader.rpc_error"
    assert "UNAVAILABLE" in excinfo.value.message
    assert "service down" in excinfo.value.message


def test_read_rejects_unexpected_response_type() -> None:
    stub = _FakeStub()
    stub.read_result = "not-a-message"
    client = _client(stub)

    with pytest.raises(ExternalServiceError) as excinfo:
        client.read(pb2.ReadRequest())

    assert "unexpected response type" in excinfo.value.message


# ─────────────────────────────────────────────────────────────────────
# Streaming ReadStream (progress) & cancellation
# ─────────────────────────────────────────────────────────────────────


def test_read_stream_yields_raw_frames() -> None:
    stub = _FakeStub()
    frames = [pb2.ReadStreamResponse(meta=pb2.ReadStreamMeta(markdown_content="# md"))]
    stub.stream_result = iter(frames)
    client = _client(stub)

    stream = client.read_stream(pb2.ReadRequest(file_name="scan.pdf"))

    assert list(stream) == frames
    request, kwargs = stub.stream_calls[0]
    assert request.file_name == "scan.pdf"
    assert kwargs["metadata"] == []


def test_read_stream_cancel_aborts_in_flight_call() -> None:
    stub = _FakeStub()
    call = _FakeStreamCall()
    stub.stream_result = call
    client = _client(stub)

    stream = client.read_stream(pb2.ReadRequest())
    assert call.cancelled is False
    stream.cancel()
    assert call.cancelled is True
    # Cancelling via the client and twice is a no-op.
    client.cancel(stream)
    assert call.cancelled is True


def test_read_stream_wraps_iteration_rpc_error() -> None:
    stub = _FakeStub()
    stub.stream_result = _RaisingStream()
    client = _client(stub)

    with pytest.raises(ExternalServiceError) as excinfo:
        list(client.read_stream(pb2.ReadRequest()))

    assert excinfo.value.code == "docreader.rpc_error"
    assert "stream unavailable" in excinfo.value.message


def test_progress_decodes_meta_then_images() -> None:
    stub = _FakeStub()
    stub.stream_result = iter(
        [
            pb2.ReadStreamResponse(
                meta=pb2.ReadStreamMeta(markdown_content="# doc", image_count=1)
            ),
            pb2.ReadStreamResponse(
                image=pb2.ImageRef(
                    filename="fig1.png",
                    original_ref="http://cdn/fig1.png",
                    mime_type="image/png",
                    storage_key="sk-fig1",
                )
            ),
        ]
    )
    client = _client(stub)

    events = list(client.progress(pb2.ReadRequest()))

    assert isinstance(events[0], pb2.ReadStreamMeta)
    assert events[0].markdown_content == "# doc"
    assert events[0].image_count == 1
    assert events[1] == ImageRefInfo(
        filename="fig1.png",
        original_ref="http://cdn/fig1.png",
        mime_type="image/png",
        storage_key="sk-fig1",
    )


def test_progress_with_no_images_yields_meta_only() -> None:
    stub = _FakeStub()
    stub.stream_result = iter(
        [pb2.ReadStreamResponse(meta=pb2.ReadStreamMeta(markdown_content="# doc"))]
    )
    client = _client(stub)

    events = list(client.progress(pb2.ReadRequest()))

    assert len(events) == 1
    assert isinstance(events[0], pb2.ReadStreamMeta)


# ─────────────────────────────────────────────────────────────────────
# ListEngines
# ─────────────────────────────────────────────────────────────────────


def test_list_engines_returns_engine_info() -> None:
    stub = _FakeStub()
    stub.engines_result = pb2.ListEnginesResponse(
        engines=[pb2.ParserEngineInfo(name="markitdown", available=True)]
    )
    client = _client(stub)

    response = client.list_engines(pb2.ListEnginesRequest())

    assert response.engines[0].name == "markitdown"
    assert response.engines[0].available is True


# ─────────────────────────────────────────────────────────────────────
# Image ref extraction
# ─────────────────────────────────────────────────────────────────────


def test_get_image_refs_from_response_extracts_refs() -> None:
    response = pb2.ReadResponse(
        image_refs=[
            pb2.ImageRef(
                filename="a.png",
                original_ref="orig-a",
                mime_type="image/png",
                storage_key="sk-a",
            ),
            pb2.ImageRef(filename="b.png", storage_key="sk-b"),
        ]
    )

    refs = get_image_refs_from_response(response)

    assert refs == [
        ImageRefInfo(
            filename="a.png",
            original_ref="orig-a",
            mime_type="image/png",
            storage_key="sk-a",
        ),
        ImageRefInfo(filename="b.png", original_ref="", mime_type="", storage_key="sk-b"),
    ]


def test_get_image_refs_from_response_empty_and_none() -> None:
    assert get_image_refs_from_response(None) == []
    assert get_image_refs_from_response(pb2.ReadResponse()) == []


# ─────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────


def test_log_suppresses_debug_unless_enabled(caplog: pytest.LogCaptureFixture) -> None:
    client = _client(_FakeStub())
    with caplog.at_level(logging.DEBUG, logger="src.ai.docreader"):
        client.log("DEBUG", "hidden line")
    assert not any("hidden line" in record.getMessage() for record in caplog.records)

    client.set_debug(True)
    with caplog.at_level(logging.DEBUG, logger="src.ai.docreader"):
        client.log("DEBUG", "visible line")
    assert any("visible line" in record.getMessage() for record in caplog.records)
