"""Connectivity probe for vector store engines.

The upstream path issues driver-specific probes (qdrant-go, tcvectordb SDK,
Weaviate client, MySQL-stream for Doris, plain HTTP for ES /
OpenSearch); for the Python rewrite the probes are intentionally
lightweight stubs that exercise the same set of engines without
requiring the per-driver SDK set.

Each ``test_*`` function returns a tuple ``(version, error)``:
``version`` is the detected server version (empty string when unknown),
``error`` is ``None`` on success. Connection failures surface as
:class:`ValidationError` so the service layer can surface the right
``code`` and the web layer can render a 4xx response.

The ES / OpenSearch probes use a raw HTTP GET against the root path
(via :func:`urllib.request`) so the dependency footprint stays limited.
For the synchronous-stubs world, a short-lived synchronous HTTP call
is acceptable inside an async function; the service layer wraps the
probe in :func:`asyncio.to_thread` when it needs to be cleanly
non-blocking.
"""

from __future__ import annotations

import base64
import json
import socket
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping

from src.common.exception import ValidationError
from src.common.json import JsonValue

# Engine-types that are valid for the DB-managed ``vector_stores`` table.
# Mirrors ``types.IsValidEngineType`` on the Go side; ``postgres`` and
# ``sqlite`` are excluded because they require the app's default DB
# connection and never appear as user-managed rows.
VALID_ENGINE_TYPES: frozenset[str] = frozenset(
    {
        "elasticsearch",
        "qdrant",
        "milvus",
        "tencent_vectordb",
        "weaviate",
        "doris",
        "opensearch",
    },
)

# Connection timeout for the probes — matches the Go
# ``connectionTestTimeout`` constant (10s).
DEFAULT_TIMEOUT_SECONDS: float = 10.0


def is_valid_engine_type(engine_type: str) -> bool:
    """Return True if ``engine_type`` is valid for the DB-managed store table."""
    return engine_type in VALID_ENGINE_TYPES


def validate_connection_config(engine_type: str, config: Mapping[str, JsonValue]) -> None:
    """Validate the required fields per engine type.

    Mirrors ``validateConnectionConfig`` in the Go service. The probe
    guards share the same required-field semantics so an empty field
    never falls through to a driver-level default.
    """
    if engine_type == "elasticsearch":
        if not _get_str(config, "addr"):
            raise ValidationError(
                code="vector_store.addr_required",
                message="addr is required for elasticsearch",
            )
    elif engine_type == "qdrant":
        if not _get_str(config, "host"):
            raise ValidationError(
                code="vector_store.host_required",
                message="host is required for qdrant",
            )
    elif engine_type == "milvus":
        if not _get_str(config, "addr"):
            raise ValidationError(
                code="vector_store.addr_required",
                message="addr is required for milvus",
            )
    elif engine_type == "tencent_vectordb":
        if not _get_str(config, "addr"):
            raise ValidationError(
                code="vector_store.addr_required",
                message="addr is required for tencent_vectordb",
            )
        if not _get_str(config, "username"):
            raise ValidationError(
                code="vector_store.username_required",
                message="username is required for tencent_vectordb",
            )
        if not _get_str(config, "api_key"):
            raise ValidationError(
                code="vector_store.api_key_required",
                message="api_key is required for tencent_vectordb",
            )
    elif engine_type == "weaviate":
        if not _get_str(config, "host"):
            raise ValidationError(
                code="vector_store.host_required",
                message="host is required for weaviate",
            )
    elif engine_type == "doris":
        if not _get_str(config, "addr"):
            raise ValidationError(
                code="vector_store.addr_required",
                message="addr is required for doris (FE MySQL host:port)",
            )
        if not _get_str(config, "database"):
            raise ValidationError(
                code="vector_store.database_required",
                message="database is required for doris",
            )
    elif engine_type == "opensearch" and not _get_str(config, "addr"):
        raise ValidationError(
            code="vector_store.addr_required",
            message="addr is required for opensearch",
        )


def test_connection(
    engine_type: str,
    config: Mapping[str, JsonValue],
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> tuple[str, None] | tuple[None, ValidationError]:
    """Probe the engine and return ``(version, None)`` on success.

    On failure returns ``(None, ValidationError)`` so the service layer
    can translate the failure into the right error code without
    propagating a generic exception.
    """
    if engine_type == "elasticsearch":
        return _test_elasticsearch(config, timeout=timeout)
    if engine_type == "qdrant":
        return _test_tcp(config, default_port=6334, timeout=timeout)
    if engine_type == "milvus":
        return _test_tcp(config, default_port=19530, timeout=timeout)
    if engine_type == "tencent_vectordb":
        return _test_http_root(config, timeout=timeout)
    if engine_type == "weaviate":
        return _test_weaviate(config, timeout=timeout)
    if engine_type == "doris":
        return _test_tcp(config, default_port=9030, timeout=timeout)
    if engine_type == "opensearch":
        return _test_opensearch(config, timeout=timeout)
    return (
        None,
        ValidationError(
            code="vector_store.unsupported_engine",
            message=f"connection test not supported for engine type: {engine_type}",
        ),
    )


# ── Probes ───────────────────────────────────────────────────────────


def _test_elasticsearch(
    config: Mapping[str, JsonValue],
    *,
    timeout: float,
) -> tuple[str, None] | tuple[None, ValidationError]:
    """Probe ES via a plain HTTP GET against the root path.

    The Go code uses a raw HTTP request (rather than the go-elasticsearch
    SDK) because the v8 SDK's TypedClient performs a product check that
    rejects ES v7 servers. We mirror the same shape: a raw GET that
    parses the ``version.number`` field off the JSON response.
    """
    addr = _get_str(config, "addr")
    if not addr:
        return (
            None,
            ValidationError(
                code="vector_store.addr_required",
                message="failed to create elasticsearch request: invalid address",
            ),
        )
    username = _get_str(config, "username")
    password = _get_str(config, "password")
    try:
        request = urllib.request.Request(addr, method="GET")
        if username:
            token = _basic_auth_token(username, password)
            request.add_header("Authorization", f"Basic {token}")
        with urllib.request.urlopen(  # value comes from validated config
            request, timeout=timeout
        ) as response:
            body = response.read(4096)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError):
        return (
            None,
            ValidationError(
                code="vector_store.connection_failed",
                message="failed to connect to elasticsearch: connection refused or authentication failed",
            ),
        )
    try:
        payload = json.loads(body)
    except (ValueError, json.JSONDecodeError):
        return ("", None)
    version_obj = payload.get("version") if isinstance(payload, dict) else None
    if isinstance(version_obj, dict):
        number = version_obj.get("number")
        if isinstance(number, str):
            return (number, None)
    return ("", None)


def _test_opensearch(
    config: Mapping[str, JsonValue],
    *,
    timeout: float,
) -> tuple[str, None] | tuple[None, ValidationError]:
    """Probe OpenSearch via a raw HTTP GET.

    OpenSearch exposes the same root endpoint shape as ES with an
    optional ``distribution`` field; we do not enforce version on this
    probe (the Go driver does), the implementation only verifies the
    server is reachable.
    """
    addr = _get_str(config, "addr")
    if not addr:
        return (
            None,
            ValidationError(
                code="vector_store.addr_required",
                message="failed to create opensearch connection: addr is required",
            ),
        )
    username = _get_str(config, "username")
    password = _get_str(config, "password")
    try:
        request = urllib.request.Request(addr, method="GET")
        if username:
            token = _basic_auth_token(username, password)
            request.add_header("Authorization", f"Basic {token}")
        with urllib.request.urlopen(  # value comes from validated config
            request, timeout=timeout
        ) as response:
            response.read(4096)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError):
        return (
            None,
            ValidationError(
                code="vector_store.connection_failed",
                message=(
                    "failed to connect to opensearch: check address, credentials, "
                    "version (>= 2.4), and that the k-NN plugin is installed"
                ),
            ),
        )
    return ("", None)


def _test_weaviate(
    config: Mapping[str, JsonValue],
    *,
    timeout: float,
) -> tuple[str, None] | tuple[None, ValidationError]:
    """Probe Weaviate via the ``/v1/.well-known/ready`` endpoint.

    The Go SDK exposes a ``ReadyChecker``; we use the standard
    lightweight probe so a Python rewrite does not need to drag the
    Weaviate client in.
    """
    host = _get_str(config, "host") or "weaviate:8080"
    scheme = _get_str(config, "scheme") or "http"
    url = f"{scheme}://{host}/v1/.well-known/ready"
    try:
        with urllib.request.urlopen(  # value comes from validated config
            url, timeout=timeout
        ) as response:
            response.read(2048)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError):
        return (
            None,
            ValidationError(
                code="vector_store.connection_failed",
                message="failed to connect to weaviate: server not ready or authentication failed",
            ),
        )
    # Weaviate version detection is optional; the Go path hits /v1/meta.
    # We keep the probe simple and skip the /v1/meta round-trip so the
    # rewrite does not depend on the Weaviate client.
    return ("", None)


def _test_http_root(
    config: Mapping[str, JsonValue],
    *,
    timeout: float,
) -> tuple[str, None] | tuple[None, ValidationError]:
    """Probe an HTTP-root-style endpoint (Tencent VectorDB).

    Version detection is left out because the real Go SDK path uses
    the ListDatabase RPC; the rewrite only verifies the address is
    reachable.
    """
    addr = _get_str(config, "addr")
    if not addr:
        return (
            None,
            ValidationError(
                code="vector_store.addr_required",
                message="failed to connect to tencent vectordb: invalid address",
            ),
        )
    try:
        with urllib.request.urlopen(  # value comes from validated config
            addr, timeout=timeout
        ) as response:
            response.read(2048)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError):
        return (
            None,
            ValidationError(
                code="vector_store.connection_failed",
                message="failed to connect to tencent vectordb: connection refused or authentication failed",
            ),
        )
    return ("", None)


def _test_tcp(
    config: Mapping[str, JsonValue],
    *,
    default_port: int,
    timeout: float,
) -> tuple[str, None] | tuple[None, ValidationError]:
    """Open a TCP connection to verify the engine is reachable.

    Used for engines whose wire protocol is driver-private (Qdrant gRPC,
    Milvus gRPC, Doris MySQL). The probe only verifies reachability —
    version detection lives in the driver.
    """
    addr = _resolve_host_port(config, default_port=default_port)
    if not addr:
        return (
            None,
            ValidationError(
                code="vector_store.address_required",
                message="address is required for connection test",
            ),
        )
    try:
        with socket.create_connection(addr, timeout=timeout):
            pass
    except (OSError, TimeoutError):
        return (
            None,
            ValidationError(
                code="vector_store.connection_failed",
                message="failed to connect to engine: connection refused or server unreachable",
            ),
        )
    return ("", None)


# ── Helpers ──────────────────────────────────────────────────────────


def _resolve_host_port(
    config: Mapping[str, JsonValue],
    *,
    default_port: int,
) -> tuple[str, int] | None:
    """Return ``(host, port)`` from the connection config, or ``None``."""
    host = _get_str(config, "host")
    if host:
        port = _get_int(config, "port") or default_port
        return (host, port)
    addr = _get_str(config, "addr")
    if not addr:
        return None
    if ":" in addr:
        host_part, _, port_part = addr.rpartition(":")
        try:
            port = int(port_part)
        except ValueError:
            port = default_port
        return (host_part, port)
    return (addr, default_port)


def _get_str(config: Mapping[str, JsonValue], key: str) -> str:
    value = config.get(key)
    if isinstance(value, str):
        return value
    return ""


def _get_int(config: Mapping[str, JsonValue], key: str) -> int:
    value = config.get(key)
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    return 0


def _basic_auth_token(username: str, password: str) -> str:
    raw = f"{username}:{password}".encode()
    return base64.b64encode(raw).decode("ascii")


# ── Helpers callable from the service layer ──────────────────────────


Probe = Callable[[str, Mapping[str, JsonValue]], tuple[str, None] | tuple[None, ValidationError]]


async def test_connection_async(
    engine_type: str,
    config: Mapping[str, JsonValue],
) -> tuple[str, None] | tuple[None, ValidationError]:
    """Async wrapper over :func:`test_connection`.

    The probes are CPU-cheap and short-lived (a TCP or HTTP round-trip),
    so the function is sync; the ``async`` shape mirrors the Go sign of
    the service method and keeps the service layer signature
    consistent.
    """
    return test_connection(engine_type, config)


__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "VALID_ENGINE_TYPES",
    "is_valid_engine_type",
    "test_connection",
    "test_connection_async",
    "validate_connection_config",
]
