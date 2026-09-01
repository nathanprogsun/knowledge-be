"""OpenTelemetry tracing setup.

Single entry point: ``setup_tracing(app)``, called from the FastAPI
lifespan. Disabled unless ``settings.otel_enabled`` — when disabled the
function is a no-op and imports of the OTel SDK still happen at module
top (per repo import rules), which is cheap because the SDK performs no
I/O until a provider is registered.

Log correlation: ``configure_logging`` (src/app_logging.py) patches
loguru records with the current ``trace_id``/``span_id`` from the OTel
context, so every log line emitted inside a request span carries the
same identifiers as the trace — grep a trace_id across logs and the
collector and you get the full request story.

Frontend correlation: the SPA attaches a W3C ``traceparent`` header to
every axios request; the FastAPI instrumentation extracts it, so browser
actions become the root context of server traces.

Exporter selection (``otel_exporter`` setting):
- ``console`` (default): stdout JSON, used for smoke tests
- ``otlp``: OTLP/HTTP to ``otel_exporter_otlp_endpoint``
- ``file``: newline-delimited JSON to ``otel_file_dir/traces.jsonl``
  so an agent can ``jq`` / grep a trace locally without a collector
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace.span import INVALID_SPAN_CONTEXT
from opentelemetry.util.types import AttributeValue
from sqlalchemy.ext.asyncio import AsyncEngine

from src.common.json import JsonValue
from src.settings import Settings, get_settings


class JsonlFileSpanExporter(SpanExporter):
    """Append each exported span as a single JSON line.

    Designed for offline agent debugging: ``jq . traces.jsonl`` or
    ``grep <trace_id> traces.jsonl`` reveal the request's full shape
    without needing a collector running. File path defaults to
    ``<otel_file_dir>/traces.jsonl``; the directory is created on first
    write. Spans are line-buffered (the file is opened with buffering
    disabled per export), so a process crash loses at most one span.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        with self._path.open("a", encoding="utf-8", buffering=1) as fp:
            for span in spans:
                fp.write(json.dumps(_serialise_span(span), default=str) + "\n")
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:  # pragma: no cover — nothing to flush
        return None


def _json_attrs(attrs: Mapping[str, AttributeValue] | None) -> dict[str, JsonValue]:
    """Coerce an OTel attribute map to a plain ``JsonValue`` dict.

    OTel types attribute values as ``Sequence`` for arrays; the wire
    shape needs concrete lists, and the recursive ``JsonValue`` type
    does not cover the SDK's ``AttributeValue`` union.
    """
    out: dict[str, JsonValue] = {}
    for key, value in (attrs or {}).items():
        if isinstance(value, (str, bool, int, float)):
            out[key] = value
        else:
            out[key] = list(value)
    return out


def _serialise_span(span: ReadableSpan) -> dict[str, JsonValue]:
    """Render a span as a JSON-serialisable dict.

    Mirrors the OTel ``Span.to_json`` shape with a couple of project-
    specific additions: ``name``, ``kind``, ``timestamp`` / ``duration_ms``
    for quick ``jq`` filtering. Hex-encoded trace/span IDs come straight
    from the SDK; we keep the field names lowercase to match OTel
    spec output that downstream tools expect.
    """
    ctx = span.get_span_context() or INVALID_SPAN_CONTEXT
    parent = span.parent
    start_ns = span.start_time or 0
    end_ns = span.end_time or start_ns
    duration_ms = (end_ns - start_ns) / 1_000_000 if end_ns else 0
    return {
        "name": span.name,
        "trace_id": f"{ctx.trace_id:032x}",
        "span_id": f"{ctx.span_id:016x}",
        "parent_span_id": (f"{parent.span_id:016x}" if parent is not None else ""),
        "kind": span.kind.name if span.kind is not None else "INTERNAL",
        "timestamp": datetime.fromtimestamp(start_ns / 1e9, tz=UTC).isoformat(),
        "duration_ms": duration_ms,
        "status": {
            "status_code": span.status.status_code.name,
            "description": span.status.description or "",
        },
        "attributes": _json_attrs(span.attributes),
        "events": [
            {
                "name": e.name,
                "timestamp": e.timestamp,
                "attributes": _json_attrs(e.attributes),
            }
            for e in span.events
        ],
        "resource": _json_attrs(span.resource.attributes),
    }


def _build_exporter(settings: Settings) -> SpanExporter:
    """Pick an exporter based on ``settings.otel_exporter``.

    ``file`` is preferred over ``otlp`` when explicitly selected
    because it lets an agent inspect traces without standing up a
    collector. ``console`` (the previous default) stays useful for
    quick smoke tests where you want spans on stdout.
    """
    mode = getattr(settings, "otel_exporter", "console").lower()
    if mode == "file":
        return JsonlFileSpanExporter(
            Path(getattr(settings, "otel_file_dir", "traces")) / "traces.jsonl"
        )
    if mode == "otlp":
        endpoint = getattr(settings, "otel_exporter_otlp_endpoint", None)
        if endpoint:
            return OTLPSpanExporter(endpoint=f"{endpoint.rstrip('/')}/v1/traces")
    return ConsoleSpanExporter()


def is_tracing_enabled() -> bool:
    """Return ``True`` iff OTel was activated for this process."""
    return bool(getattr(get_settings(), "otel_enabled", False))


def is_file_exporter() -> bool:
    """Return ``True`` iff the JSONL file exporter is the active mode.

    Surfaced through ``/api/v1/meta/capabilities`` so agents / operators
    can confirm traces are landing on disk without poking the directory.
    """
    return getattr(get_settings(), "otel_exporter", "console").lower() == "file"


def setup_tracing(app: FastAPI) -> bool:
    """Register the tracer provider and instrument the ASGI app + httpx.

    Must run inside ``create_app`` — after the app object exists but
    before uvicorn builds the middleware stack; instrumenting from the
    lifespan is too late and silently produces no spans.

    Returns True when tracing was activated.
    """
    settings = get_settings()
    if not settings.otel_enabled:
        return False

    provider = TracerProvider(
        resource=Resource.create({"service.name": settings.otel_service_name}),
    )
    provider.add_span_processor(BatchSpanProcessor(_build_exporter(settings)))
    trace.set_tracer_provider(provider)

    FastAPIInstrumentor.instrument_app(app)
    HTTPXClientInstrumentor().instrument()
    return True


def instrument_engine(engine: AsyncEngine) -> None:
    """Attach SQLAlchemy span emission to the engine.

    Called from the lifespan (the engine is built there); safe to call
    with tracing disabled — the instrumentor short-circuits when no
    tracer provider is registered, but we skip the call entirely to
    keep the disabled path side-effect free.
    """
    if not get_settings().otel_enabled:
        return
    SQLAlchemyInstrumentor().instrument(engine=engine.sync_engine)


__all__ = ["instrument_engine", "is_file_exporter", "is_tracing_enabled", "setup_tracing"]
