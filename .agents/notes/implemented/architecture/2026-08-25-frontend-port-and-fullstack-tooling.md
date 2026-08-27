# Agent Note: Frontend port and full-stack tooling (OpenAPI codegen, OTel, frontend CI)
Tags: frontend, tooling, openapi, otel
Related files: frontend/, scripts/export_openapi.py, src/common/telemetry.py, .github/workflows/ci.yml

Status: implemented
Date: 2026-08-25
Scope: governs how the Vue frontend lives in this repo, how frontend–backend contracts stay synchronized, and how observability is wired

## Context

The Knowledge Base Vue frontend was ported into `frontend/` to make this a
full-stack repo. Three gaps surfaced immediately: frontend API types
were hand-written against the backend (drift-prone), there was no
tracing/log correlation, and the frontend had no CI gate.

## Decision

- Frontend lives at `frontend/` as an independent package (own
  `package.json`, own toolchain). No pnpm workspace — a single TS
  package gains nothing from workspace hoisting.
- The FastAPI OpenAPI schema is the single source of truth for
  frontend wire types. `make openapi` runs
  `scripts/export_openapi.py` (no DB needed) and regenerates
  `frontend/src/api/__generated__/schema.ts` via openapi-typescript.
  Hand-written response interfaces in `frontend/src/api/**` are being
  replaced by `Schema[...]` aliases (auth module migrated as the
  reference pattern). View-models that normalize wire data
  (`UserInfo`, `TenantInfo`) stay local but must derive from the
  generated shape (`Partial<Omit<...>>` pattern).
- Tracing: `src/common/telemetry.py` (`setup_tracing` in `create_app`,
  `instrument_engine` in the lifespan) wires FastAPI + SQLAlchemy +
  httpx instrumentation, gated by `OTEL_ENABLED`. loguru records carry
  `trace_id`/`span_id` via a patcher in `src/app_logging.py`. The SPA
  attaches a W3C `traceparent` header to every axios request, so
  browser actions root the server-side trace.
- The frontend is a blocking CI lane (`frontend` job in ci.yml):
  install → contract-drift check (regenerate schema.ts, `git diff
  --exit-code`) → vue-tsc → tsx tests → vite build. The aggregator
  requires it.

## Alternatives considered

- **pnpm workspace monorepo** — rejected: one TS package, no shared
  dependencies, added root tooling complexity for zero dedupe gain.
- **Instrument FastAPI from the lifespan** — rejected: uvicorn has
  already built the middleware stack by then; instrumentation silently
  produces no spans. Must run inside `create_app`.
- **Full OTel SDK in the browser** — rejected: heavyweight for the
  current need; a random `traceparent` per request gives end-to-end
  correlation with zero dependencies.

## Consequences

Backend contract changes now surface as frontend type errors after
`make openapi`, and CI fails if the generated file is stale or
hand-edited. Local dev and tests carry zero OTel overhead unless
explicitly enabled. `vue-tsc` needs a raised heap
(`NODE_OPTIONS=--max-old-space-size=6144` in CI) on large trees.

## Required verification

- `make openapi` regenerates `docs/api/openapi.json` and
  `schema.ts`; CI `frontend` job enforces no drift.
- `uv run pytest -q tests/` unaffected (OTel disabled by default).
- Manual: `OTEL_ENABLED=true uv run uvicorn src.main:app` + a request
  with a `traceparent` header prints spans with that trace id.
