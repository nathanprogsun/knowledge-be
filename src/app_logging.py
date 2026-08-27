"""Process-wide logging configuration.

Uses loguru. Output is human-readable in development; switch to JSON via
`serialize=True` when deploying behind structured log aggregators.

Every record is patched with the current OpenTelemetry ``trace_id`` /
``span_id`` (zeros when no span is active, e.g. tracing disabled or
code running outside a request), so log lines correlate with traces in
the collector without changing call sites.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from loguru import logger
from opentelemetry import trace

from src.settings import get_settings

if TYPE_CHECKING:
    from loguru import Record


def _inject_trace_context(record: Record) -> None:
    """Stamp the active OTel span context onto the loguru record."""
    span_context = trace.get_current_span().get_span_context()
    record["extra"]["trace_id"] = format(span_context.trace_id, "032x")
    record["extra"]["span_id"] = format(span_context.span_id, "016x")


def configure_logging() -> None:
    settings = get_settings()
    level = "DEBUG" if settings.environment == "development" else "INFO"
    logger.remove()
    logger.configure(patcher=_inject_trace_context)
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
            "trace={extra[trace_id]} span={extra[span_id]} | "
            "{name}:{function}:{line} | {message}"
        ),
        backtrace=False,
        diagnose=False,
    )


__all__ = ["configure_logging", "logger"]
