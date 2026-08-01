"""Process-wide logging configuration.

Uses loguru. Output is human-readable in development; switch to JSON via
`serialize=True` when deploying behind structured log aggregators.
"""

from __future__ import annotations

import sys

from loguru import logger

from src.settings import get_settings


def configure_logging() -> None:
    settings = get_settings()
    level = "DEBUG" if settings.environment == "development" else "INFO"
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}"
        ),
        backtrace=False,
        diagnose=False,
    )


__all__ = ["configure_logging", "logger"]
