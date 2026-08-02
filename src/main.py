"""Process entry point.

`uvicorn src.main:app` is wired up via `app` re-exported from
`src.app_context.lifespan` so the FastAPI instance lives next to its
lifespan wiring (DI registration happens at startup).
"""

from __future__ import annotations

import uvicorn

from src.app_context.lifespan import app  # noqa: F401 — re-exported for uvicorn
from src.app_logging import configure_logging
from src.settings import get_settings


def run() -> None:
    """Start uvicorn. Used by both `python -m src.main` and gunicorn."""
    configure_logging()
    settings = get_settings()
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.environment == "development",
        log_config=None,  # loguru owns logging
    )


if __name__ == "__main__":
    run()
