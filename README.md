# knowledge-be

Backend service for knowledge management and AI-powered Q&A. FastAPI +
SQLAlchemy (async) + ARQ task queue.

## Features

- Multi-tenant knowledge bases, documents, chunks, tags, FAQ, wiki.
- Chat sessions, messages, agents, embeddings, vector stores.
- Instant messaging channels, web search, MCP services, evaluation.
- Strict layered architecture (see `AGENTS.md`).

## Quick start

```bash
# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync deps
uv sync --all-extras

# Copy and edit environment
cp .env.example .env

# Run linters / type checkers / tests
make lint
make typecheck
make test

# Start the API (requires Postgres + Redis at the URLs in .env)
make dev-app
```

## Layout

```
knowledge-be/
├── src/
│   ├── main.py         ← uvicorn entry
│   ├── settings.py     ← pydantic-settings singleton
│   ├── app_logging.py  ← loguru config
│   ├── app_context/    ← lifespan + DI registry + request contextvars
│   ├── common/         ← exception hierarchy, pagination, TableModel, session
│   └── db/             ← DatabaseEngine (async pool)
├── alembic/            ← async migrations (env + 0000 placeholder)
├── tests/              ← pytest, asyncio mode auto
├── pyproject.toml      ← deps + tool config
├── ruff.toml           ← linter + formatter
├── mypy.ini            ← strict type checker
├── pyrightconfig.json  ← LSP type-checker config (points at .venv)
├── Makefile            ← dev targets
├── AGENTS.md           ← project conventions
├── .env.example        ← environment template
└── README.md
```

## Configuration

Settings are loaded from environment variables prefixed `KNOWLEDGE_BE_`
or from a `.env` file in the project root. See `src/settings.py` for
the full schema and defaults.

## License

Proprietary.