# knowledge-be

Backend service for knowledge management and AI-powered Q&A. FastAPI +
SQLAlchemy (async) + ARQ task queue.

> Conventions: see `AGENTS.md` (single source of truth: layered arch,
> DI scope, error hierarchy, async purity, agent workflow).
> Frontend conventions: `frontend/AGENTS.md`. Product surface map:
> `.agents/feature-map/ui.md`.
> AI contributors: read `AGENTS.md` §12 before opening a PR.

## Features

- Multi-tenant knowledge bases, documents, chunks, tags, FAQ, wiki.
- Chat sessions, messages, agents, embeddings, vector stores.
- Instant messaging channels, web search, MCP services, evaluation.
- OpenAPI-first API contract: `make openapi` regenerates frontend types.
- Strict layered architecture enforced by `make check-layer`.

## Quick start

```bash
# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh
# uv version is pinned in pyproject.toml [tool.uv]; keep it aligned with CI.

# Sync deps
uv sync --all-extras

# Copy and edit environment
cp .env.example .env
# Postgres and Redis must be reachable at DB_* and REDIS_URL.
# Optional local stack (reads deploy/env/api.env.example and
# deploy/env/worker.env.example; compose overrides DB_HOST and REDIS_URL):
#   docker compose -f deploy/docker/docker-compose.yml up -d postgres redis

# Lint / typecheck / tests
make lint
make typecheck
make test

# Format gate (CI fails on unformatted)
make format          # --check only
make format-fix      # rewrite in place

# Anti-drift gates (layer / DI / endpoint / schema / imports / sql /
# pr-leak / map_from_db / exception-types / agent-notes). Run before
# declaring a non-trivial change "done". See AGENTS.md §9.
make check

# Apply DB migrations (requires Postgres at $DB_HOST:$DB_PORT)
make migrate

# Regenerate OpenAPI JSON + frontend TypeScript types from FastAPI
make openapi

# Start the API (requires Postgres + Redis at the URLs in .env)
make dev-app

# Background worker (same Redis as REDIS_URL unless WORKER_REDIS_URL is set)
make dev-worker
# or: uv run python -m src.workers.main
```

`make typecheck` is full mypy; CI uses the mypy ratchet (`make check-mypy-baseline`).
`make test` runs the full pytest tree, including live-Postgres suites; CI ignores those suites.
Document parsing needs a docreader gRPC process at `DOCREADER_ADDR`; compose does not start it yet.

### Frontend dev loop

The Vue 3 + Vite + TS package lives in `frontend/`. All `make frontend-*`
targets run inside that directory:

```bash
make frontend-install     # npm ci
make frontend-typecheck   # vue-tsc
make frontend-test        # tsx --test
make frontend-build       # vite build
```

The dev proxy is `/api` → `http://localhost:8000` (override with
`VITE_DEV_PROXY_TARGET`). See `frontend/README.md` for frontend-only docs.

### Pre-commit / pre-push hooks

```bash
pre-commit install --hook-type pre-commit --hook-type pre-push
```

Pre-commit runs ruff + verify-agent-notes; pre-push adds full mypy
strict. See `.pre-commit-config.yaml` for the exact hook set.

## Layout

```
knowledge-be/
├── frontend/             ← Vue 3 + Vite + TS UI (`frontend/AGENTS.md`)
├── docs/                 ← migration baselines + endpoint inventory
├── .agents/
│   ├── notes/            ← RFC-style decision records (AGENTS.md §12)
│   ├── skills/           ← reusable agent workflows (SKILL.md)
│   └── feature-map/      ← generated.json + ui.md
├── .github/workflows/    ← ci.yml: lint, hygiene, format, typecheck, …
├── src/
│   ├── main.py           ← uvicorn entry
│   ├── settings.py       ← pydantic-settings singleton
│   ├── app_logging.py    ← loguru config
│   ├── app_context/      ← lifespan + DI registry + request contextvars
│   ├── ai/               ← embedding / LLM / graph / docreader clients
│   ├── common/           ← exception hierarchy, pagination, TableModel, …
│   ├── core/             ← domain services (auth, chat, agents, knowledge, …)
│   ├── db/               ← DatabaseEngine, DAO, row models
│   ├── util/             ← shared utilities
│   ├── web/              ← FastAPI routers, deps, middleware
│   │   ├── api/          ← HTTP handlers grouped by domain
│   │   ├── deps/         ← REQUEST-scope DI factories
│   │   ├── middleware/   ← auth, request-context, error handling
│   │   └── exception_handler.py
│   └── workers/          ← ARQ task handlers
├── alembic/              ← async migrations (`make migrate` to upgrade)
├── tests/                ← pytest, asyncio mode auto
├── scripts/              ← anti-drift checks (run via `make check`)
├── .env.example          ← environment template (see "Configuration")
├── .pre-commit-config.yaml
├── pyproject.toml        ← deps + tool config
├── ruff.toml             ← linter + formatter
├── mypy.ini              ← strict type checker
├── pyrightconfig.json    ← LSP type-checker config (points at .venv)
├── Makefile              ← dev + CI targets (see "Quick start")
├── AGENTS.md             ← project conventions. READ THIS FIRST
└── README.md             ← this file
```

## AI contributor checklist

1. Read `AGENTS.md` (especially §1 layer rules, §3 DI scope, §5 errors,
   §9 CI gates, §12 agent workflow).
2. Run `make help` to see all dev targets.
3. Before declaring "done": `make check` + `make test` + (if you touched
   the API surface) `make openapi` + (if you made a non-trivial decision)
   add or update a note under `.agents/notes/implemented/`.

## Configuration

`src/settings.py` is the schema. `.env.example` is the template. Copy it
to `.env` and edit for your machine (no prefix; `pydantic-settings`
reads the canonical names).

| Group | Variable | Notes |
| --- | --- | --- |
| Runtime | `APP_NAME`, `ENVIRONMENT` | |
| Database | `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_DRIVER` | Auto-composed into the computed `database_url`. Set `DATABASE_URL_OVERRIDE` (Settings field `database_url_override`) to replace that URL. A bare `DATABASE_URL` is ignored. |
| Redis | `REDIS_URL` | ARQ broker + cache. Worker falls back to this unless `WORKER_REDIS_URL` is set. |
| Auth | `JWT_SECRET_KEY`, `JWT_ALGORITHM`, `JWT_EXPIRE_MINUTES`, `REFRESH_TOKEN_EXPIRE_DAYS` | `JWT_SECRET_KEY` is must-set for real deploys. |
| Secrets | `SYSTEM_AES_KEY` | Raw 32-byte UTF-8 string (exactly 32 bytes). Must-set for real deploys. |
| RBAC | `RBAC_ENFORCED`, `CROSS_TENANT_ACCESS_ENABLED` | Role guards and cross-tenant superuser routes. |
| Telemetry | `OTEL_ENABLED`, `OTEL_EXPORTER_OTLP_ENDPOINT` | See `src/common/telemetry.py`. Zero overhead when disabled. |

See `.env.example` for optional OIDC, header-auth, and other keys.

## License

Proprietary.
