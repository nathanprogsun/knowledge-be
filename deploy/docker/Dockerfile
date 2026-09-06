# syntax=docker/dockerfile:1
# Repository-root build context.
#   docker build -f deploy/docker/Dockerfile --target api -t knowledge-be-api .
#   docker build -f deploy/docker/Dockerfile --target worker -t knowledge-be-worker .

FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

COPY --from=ghcr.io/astral-sh/uv:0.10.9 /uv /bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY alembic ./alembic
COPY alembic.ini ./

RUN uv sync --frozen --no-dev --no-editable

ARG WITH_ODL=0
RUN if [ "$WITH_ODL" = "1" ]; then \
      apt-get update && apt-get install -y --no-install-recommends openjdk-17-jre-headless \
      && rm -rf /var/lib/apt/lists/* \
      && uv pip install --python /app/.venv/bin/python "opendataloader-pdf[hybrid]>=2.4.7"; \
    fi

FROM runtime AS api

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM runtime AS worker

CMD ["python", "-m", "src.workers.main"]
