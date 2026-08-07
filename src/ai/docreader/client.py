"""gRPC client for the document-reader service.

Maps the upstream docreader client package (transport + auth wiring) to
Python. ``DocReaderClient`` exposes the unary Read RPC (parse), the
server-streaming ReadStream RPC (progress: one metadata frame followed by
one frame per extracted image) and the ListEngines RPC, with TLS /
bearer-token authentication driven by the same environment variables as
the upstream client. The generated stubs live in ``.proto``; this module
adds channel construction, auth configuration and the typed helpers
(image-ref extraction, stream decoding) on top.

Cancellation is exposed on the live stream returned by
:meth:`DocReaderClient.read_stream` through :meth:`ReadStream.cancel`
(and ``DocReaderClient.cancel``), matching gRPC context cancellation.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import grpc  # type: ignore[import-untyped]

from src.ai.docreader.proto import docreader_pb2 as pb2
from src.ai.docreader.proto import docreader_pb2_grpc as pb2_grpc
from src.common.exception import ExternalServiceError

logger = logging.getLogger(__name__)

_DEFAULT_MAX_FILE_SIZE_MB = 50
_DEFAULT_MAX_MESSAGE_SIZE = _DEFAULT_MAX_FILE_SIZE_MB * 1024 * 1024
_MAX_FILE_SIZE_ENV = "MAX_FILE_SIZE_MB"

# Environment keys match the upstream auth contract.
_ENV_TLS_ENABLED = "GRPC_TLS_ENABLED"
_ENV_TLS_CERT = "GRPC_TLS_CERT"
_ENV_TLS_KEY = "GRPC_TLS_KEY"
_ENV_TLS_CA = "GRPC_TLS_CA"
_ENV_TLS_SERVER_NAME = "GRPC_TLS_SERVER_NAME"
_ENV_AUTH_TOKEN = "GRPC_AUTH_TOKEN"

_AUTH_HEADER = "authorization"
_BEARER_PREFIX = "Bearer "
_ROUND_ROBIN_SERVICE_CONFIG = '{"loadBalancingPolicy":"round_robin"}'


@dataclass(slots=True)
class ImageRefInfo:
    """Image reference extracted from a converted document."""

    filename: str
    original_ref: str
    mime_type: str
    storage_key: str


@dataclass(slots=True)
class AuthConfig:
    """TLS / bearer-token configuration for the docreader gRPC channel."""

    tls_enabled: bool = False
    cert_file: str = ""
    key_file: str = ""
    ca_file: str = ""
    server_name: str = ""
    auth_token: str = ""


def load_auth_config_from_env() -> AuthConfig:
    """Read the docreader auth knobs from the process environment."""
    return AuthConfig(
        tls_enabled=os.getenv(_ENV_TLS_ENABLED) == "true",
        cert_file=os.getenv(_ENV_TLS_CERT, ""),
        key_file=os.getenv(_ENV_TLS_KEY, ""),
        ca_file=os.getenv(_ENV_TLS_CA, ""),
        server_name=os.getenv(_ENV_TLS_SERVER_NAME, ""),
        auth_token=os.getenv(_ENV_AUTH_TOKEN, ""),
    )


def get_max_message_size() -> int:
    """Return the configured max gRPC message size in bytes (default 50 MiB)."""
    raw = os.getenv(_MAX_FILE_SIZE_ENV)
    if raw:
        try:
            size = int(raw)
        except ValueError:
            size = 0
        if size > 0:
            return size * 1024 * 1024
    return _DEFAULT_MAX_MESSAGE_SIZE


def build_dial_options(
    max_msg_size: int,
    auth: AuthConfig | None = None,
) -> tuple[list[tuple[str, Any]], grpc.ChannelCredentials | None]:
    """Build gRPC channel options and transport credentials.

    ``credentials`` is ``None`` when TLS is disabled — callers must then use
    an insecure channel. The server-name override (SNI / certificate host
    check) is expressed as the ``grpc.ssl_target_name_override`` channel
    option, which is how gRPC Python applies it.
    """
    config = auth if auth is not None else AuthConfig()
    options: list[tuple[str, Any]] = [
        ("grpc.default_service_config", _ROUND_ROBIN_SERVICE_CONFIG),
        ("grpc.max_receive_message_length", max_msg_size),
        ("grpc.max_send_message_length", max_msg_size),
    ]
    if config.server_name:
        options.append(("grpc.ssl_target_name_override", config.server_name))
    credentials = _build_tls_credentials(config) if config.tls_enabled else None
    return options, credentials


def _build_tls_credentials(auth: AuthConfig) -> grpc.ChannelCredentials:
    if bool(auth.cert_file) != bool(auth.key_file):
        raise ExternalServiceError(
            code="docreader.tls_config",
            message="GRPC_TLS_CERT and GRPC_TLS_KEY must be set together for mTLS",
        )
    return grpc.ssl_channel_credentials(
        root_certificates=_read_file(auth.ca_file) if auth.ca_file else None,
        private_key=_read_file(auth.key_file) if auth.key_file else None,
        certificate_chain=_read_file(auth.cert_file) if auth.cert_file else None,
    )


def _read_file(path: str) -> bytes:
    try:
        with open(path, "rb") as fh:
            return fh.read()
    except OSError as exc:
        raise ExternalServiceError(
            code="docreader.tls_config",
            message=f"failed to read TLS file {path!r}: {exc}",
        ) from exc


def new_client(addr: str) -> DocReaderClient:
    """Create a docreader client using auth configuration from the environment."""
    return new_client_with_auth(addr, load_auth_config_from_env())


def new_client_with_auth(addr: str, auth: AuthConfig | None = None) -> DocReaderClient:
    """Create a docreader client connected to ``addr`` with the given auth config."""
    config = auth if auth is not None else AuthConfig()
    options, credentials = build_dial_options(get_max_message_size(), config)
    target = f"dns:///{addr}"
    if credentials is not None:
        channel = grpc.secure_channel(target, credentials, options=options)
        logger.info("docreader: TLS enabled for %s", addr)
    else:
        channel = grpc.insecure_channel(target, options=options)
    if config.auth_token:
        logger.info(
            "docreader: token authentication enabled for %s (TLS=%s)",
            addr,
            config.tls_enabled,
        )
    return DocReaderClient(channel, auth=config)


class DocReaderClient:
    """gRPC client for the document-reader service.

    Equivalent to the upstream ``Client``: :meth:`read` maps the unary Read
    RPC (parse), :meth:`read_stream` maps the server-streaming ReadStream RPC
    (progress) and :meth:`list_engines` maps the ListEngines RPC. The bearer
    token, when configured, is attached to every call as an ``authorization``
    metadata header (gRPC Python has no channel-level per-RPC credentials for
    plaintext transports; the upstream Go client emits the same header via
    per-RPC credentials).
    """

    def __init__(
        self,
        channel: grpc.Channel,
        *,
        stub: Any = None,
        auth: AuthConfig | None = None,
        debug: bool = False,
    ) -> None:
        self._channel = channel
        self._stub: Any = (
            pb2_grpc.DocReaderStub(channel)  # type: ignore[no-untyped-call]
            if stub is None
            else stub
        )
        self._auth = auth if auth is not None else AuthConfig()
        self._debug = debug

    def close(self) -> None:
        """Close the underlying channel."""
        self._channel.close()

    def set_debug(self, debug: bool) -> None:
        """Enable/disable DEBUG log lines."""
        self._debug = debug

    def log(self, level: str, message: str) -> None:
        """Emit a client log line; DEBUG lines are dropped unless debug is on."""
        if level == "DEBUG" and not self._debug:
            return
        logger.log(getattr(logging, level.upper(), logging.INFO), message)

    def read(
        self,
        request: pb2.ReadRequest,
        *,
        timeout: float | None = None,
        metadata: Sequence[tuple[str, str]] | None = None,
    ) -> pb2.ReadResponse:
        """Parse a document (unary Read RPC) and return the raw response."""
        try:
            response = self._stub.Read(
                request,
                timeout=timeout,
                metadata=self._auth_metadata(metadata),
            )
        except grpc.RpcError as exc:
            raise _rpc_error("read", exc) from exc
        if not isinstance(response, pb2.ReadResponse):
            raise ExternalServiceError(
                code="docreader.rpc_error",
                message=(
                    "docreader read returned unexpected response type "
                    f"{type(response).__name__}"
                ),
            )
        return response

    def read_stream(
        self,
        request: pb2.ReadRequest,
        *,
        timeout: float | None = None,
        metadata: Sequence[tuple[str, str]] | None = None,
    ) -> ReadStream:
        """Start a server-streaming Read call (parse with progress frames).

        The returned stream yields raw ``ReadStreamResponse`` frames: the
        first carries ``meta`` (parse metadata / progress header), every
        subsequent frame carries one ``image``. Cancel an in-flight call with
        :meth:`ReadStream.cancel` or :meth:`DocReaderClient.cancel`.
        """
        call = self._stub.ReadStream(
            request,
            timeout=timeout,
            metadata=self._auth_metadata(metadata),
        )
        return ReadStream(call)

    def list_engines(
        self,
        request: pb2.ListEnginesRequest,
        *,
        timeout: float | None = None,
        metadata: Sequence[tuple[str, str]] | None = None,
    ) -> pb2.ListEnginesResponse:
        """List available parser engines."""
        try:
            response = self._stub.ListEngines(
                request,
                timeout=timeout,
                metadata=self._auth_metadata(metadata),
            )
        except grpc.RpcError as exc:
            raise _rpc_error("list_engines", exc) from exc
        if not isinstance(response, pb2.ListEnginesResponse):
            raise ExternalServiceError(
                code="docreader.rpc_error",
                message=(
                    "docreader list_engines returned unexpected response type "
                    f"{type(response).__name__}"
                ),
            )
        return response

    def progress(
        self,
        request: pb2.ReadRequest,
        *,
        timeout: float | None = None,
        metadata: Sequence[tuple[str, str]] | None = None,
    ) -> Iterator[pb2.ReadStreamMeta | ImageRefInfo]:
        """Decode a ReadStream into progress events.

        Yields the parse metadata frame first, then one :class:`ImageRefInfo`
        per extracted image.
        """
        stream = self.read_stream(request, timeout=timeout, metadata=metadata)
        for frame in stream:
            which = frame.WhichOneof("payload")
            if which == "meta":
                yield frame.meta
            elif which == "image":
                yield _image_ref_info(frame.image)

    def cancel(self, stream: ReadStream) -> None:
        """Cancel an in-flight :meth:`read_stream` call."""
        stream.cancel()

    def _auth_metadata(
        self,
        extra: Sequence[tuple[str, str]] | None = None,
    ) -> list[tuple[str, str]]:
        merged = list(extra) if extra is not None else []
        if self._auth.auth_token:
            merged.append((_AUTH_HEADER, f"{_BEARER_PREFIX}{self._auth.auth_token}"))
        return merged


class ReadStream:
    """A live server-streaming Read call.

    Iterate to receive raw ``ReadStreamResponse`` frames: the first carries
    ``meta`` (parse metadata / progress header), every subsequent frame
    carries one ``image``. Call :meth:`cancel` to abort the call in flight.
    """

    def __init__(self, call: Any) -> None:
        self._call = call

    def __iter__(self) -> Iterator[pb2.ReadStreamResponse]:
        try:
            for frame in self._call:
                if isinstance(frame, pb2.ReadStreamResponse):
                    yield frame
                else:
                    raise ExternalServiceError(
                        code="docreader.rpc_error",
                        message=(
                            "docreader read_stream yielded unexpected frame type "
                            f"{type(frame).__name__}"
                        ),
                    )
        except grpc.RpcError as exc:
            raise _rpc_error("read_stream", exc) from exc

    def cancel(self) -> None:
        """Abort the in-flight stream; further iteration surfaces cancellation."""
        cancel = getattr(self._call, "cancel", None)
        if cancel is not None:
            cancel()


def get_image_refs_from_response(response: pb2.ReadResponse | None) -> list[ImageRefInfo]:
    """Extract image references from a Read response (empty when none)."""
    if response is None:
        return []
    return [_image_ref_info(ref) for ref in response.image_refs]


def _image_ref_info(ref: pb2.ImageRef) -> ImageRefInfo:
    return ImageRefInfo(
        filename=ref.filename,
        original_ref=ref.original_ref,
        mime_type=ref.mime_type,
        storage_key=ref.storage_key,
    )


def _rpc_error(method: str, exc: grpc.RpcError) -> ExternalServiceError:
    code_fn = getattr(exc, "code", None)
    details_fn = getattr(exc, "details", None)
    status = code_fn() if callable(code_fn) else None
    detail = details_fn() if callable(details_fn) else None
    message = f"docreader {method} RPC failed"
    if status is not None:
        message += f": {status}"
    if detail:
        message += f": {detail}"
    return ExternalServiceError(
        code="docreader.rpc_error",
        message=message,
        details={"status": str(status)} if status is not None else None,
    )
