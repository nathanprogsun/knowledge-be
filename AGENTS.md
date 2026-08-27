# AGENTS.md — knowledge-be

> Conventions for human and AI contributors. Keep ≤ 200 lines; refresh
> whenever conventions evolve.

## 1. Layered architecture

```
web      → core, common, app_context
core     → db, common, util
db       → common, db/models, db/dbengine
workers  → core, common, util
*        → common, util
frontend → backend only over HTTP (/api/v1); never imports src/
```

Forbidden:
- `web` importing `db` — anywhere, including `web/deps` and
  `web/middleware`. Repositories are instantiated only inside
  `core/<domain>/factory.py` `build_*_service(session, ...)` factories;
  `web/deps` factories are one-line forwarders. Enforced by
  `scripts/check_layer_violation.py` (`make check-layer`).
- `db` is called only by repositories; repositories only by services.
- `core` handling HTTP concerns (`Request`, `Response`, `HTTPException`).
- `db` containing business logic.
- `workers` calling `db` directly.
- Function-level imports anywhere.
- `Any` / `object` type annotations.
- Module-level mutable globals.

## 2. Code conventions

- Python ≥ 3.11 (use `match`; `type X = ...` and PEP 695 generics only
  once the floor moves to ≥ 3.12 — until then use plain `TypeAlias`).
- All imports at file top, in `stdlib → third-party → first-party` groups.
- Every variable, parameter, return type, class attribute must have an
  explicit annotation.
- Use `T | None` over `Optional[T]`.
- Files: 200–400 lines typical, 800 max.
- Functions: < 50 lines.
- Comments carry **business rationale only** — never restate the
  code, and never reference internal PR numbers or migration-plan
  stages (this repo is public-facing).

## 3. Service registration (DI)

DI follows **scope layering**, not blanket singletons:

- **APP scope** — stateless + expensive to construct (`DatabaseEngine`,
  `OidcClient`/pooled `httpx.AsyncClient`, settings): created once in the
  FastAPI lifespan, stored on `LifeSpanService`, obtained via
  `get_xxx_from_lifespan(request.app)`. Close long-lived resources in
  the lifespan shutdown (e.g. `OidcClient.aclose()`).
- **REQUEST scope** — anything holding the per-request `AsyncSession`
  (repositories, and services binding repositories): constructed per
  request in `web/deps/` factories. The factory composes fresh repos
  with APP-scope singletons pulled from the lifespan registry.
- Request-scoped services MUST NOT be registered on `LifeSpanService`
  (enforced by `scripts/check_service_singleton.py`). Test conftests
  under `tests/integration/` are exempt (each test owns its request
  lifetime via the test app's lifespan) — the only sanctioned path for
  session-scoped real services outside `src/`.
- Never use class-level Singleton bases/metaclasses: they break test
  transport injection, bind to the wrong event loop, and hide
  dependencies. "One instance per app" = construct once in the
  lifespan, not type-level tricks.
- Services receive deps via constructor injection — never via runtime
  `Depends` inside `core`. Services MUST NOT hold request-scoped state;
  request data is read from `src.app_context.request_context`.

## 4. Persistence

- Use `TableModel` (Pydantic, frozen) as row shape.
- All SQL is raw text via `sqlalchemy.text()` with named `bindparams`.
  Never f-string or `%`-format SQL. Never ORM `.select(Model).where(...)`.
- DAOs accept an `AsyncSession` and never commit/rollback. Transactions
  are managed by service layer via `session_scope`.
- Soft-deleted tables: filter `WHERE deleted_at IS NULL` on every read.

## 5. Errors

- `core`, `db`, `ai`, `workers` raise `ApplicationError` subclasses only.
- `web` translates `ApplicationError` subclasses to HTTP status via a
  single exception handler.
- Never raise `fastapi.HTTPException` from non-`web` layers.

Hierarchy: `ApplicationError → NotFoundError | ConflictError | ValidationError
| PermissionDeniedError | UnauthorizedError | ExternalServiceError →
{AIProviderError, VectorStoreError, StorageBackendError} | DataError`.## 6. API contracts: OpenAPI is the source of truth

- The FastAPI OpenAPI schema is the single contract definition. There is
  no separately governed frozen-contract layer; `src/core/contracts/`
  is an ordinary shared-types module (frozen by convention, not by gate).
- Frontend wire types are **generated**, never hand-written:
  `make openapi` runs `scripts/export_openapi.py` (no DB needed) and
  regenerates `frontend/src/api/__generated__/schema.ts` via
  openapi-typescript. CI fails on drift (`git diff --exit-code` after
  regeneration).
- Frontend view-models that normalize wire data (e.g. `UserInfo`,
  `TenantInfo`) stay local but must derive from the generated schema
  types (`Partial<Omit<...>>` pattern), not redefine fields.

## 7. Frontend

- `frontend/` is an independent Vue 3 + Vite + TS package (own
  `package.json`); no JS workspace/monorepo tooling at the repo root.
- Dev proxy: `/api` → `http://localhost:8000` (override with
  `VITE_DEV_PROXY_TARGET`). Backend API prefix is `/api/v1`.
- Frontend gates live in the `frontend` CI lane: `npm ci` → contract
  drift check → `vue-tsc` → `tsx --test` → `vite build`. `vue-tsc`
  needs `NODE_OPTIONS=--max-old-space-size=6144`.
- `npm` commands run inside `frontend/` only; never install JS
  dependencies at the repo root.

## 8. Observability

- Tracing is OpenTelemetry, gated by `OTEL_ENABLED` (default off;
  zero overhead when disabled). `src/common/telemetry.py`:
  `setup_tracing(app)` MUST run inside `create_app` (instrumenting
  from the lifespan is too late — no spans); `instrument_engine()`
  runs in the lifespan for SQLAlchemy.
- Exporter: OTLP/HTTP when `OTEL_EXPORTER_OTLP_ENDPOINT` is set, else
  console.
- loguru records carry `trace_id`/`span_id` via the patcher in
  `src/app_logging.py` — do not bypass `configure_logging`.
- The SPA attaches a W3C `traceparent` header to every axios request;
  keep it intact when touching `frontend/src/utils/request.ts`.

## 9. CI gates

Pre-merge gates:
- `make lint` — ruff check
- `make format` — ruff format --check
- `make typecheck` — `uv run mypy` (strict from mypy.ini). Always run
  via `uv run` — a bare `mypy` on PATH may resolve to a global install
  and produce false failures.
- `make test` — pytest
- `make check` — anti-drift scripts (layer directionality, DI scope,
  endpoint coverage, DB schema compatibility, top-level imports)
- CI `frontend` lane — see §7

A change that fails any gate cannot merge.

## 10. Async purity

- All I/O is `async`; services and DAOs are `async def`.
- Blocking calls via `asyncio.to_thread` or workers.

## 11. DB-row → service-DTO projection (`map_from_db`)

Service-output DTOs that mirror a storage row expose a
`map_from_db(cls, db: <StorageModel>) -> Self` classmethod performing
the boundary translation. The db layer never references the wire DTO
(§1); the service calls ``XxxInfo.map_from_db(row)``.

- Defined on the DTO (in ``core``), not on the ``TableModel``.
- Strips sensitive / storage-only columns (``password_hash``,
  ``deleted_at``); the exclude-set is a module-level ``frozenset``.
- Hydrates nested sub-models from JSON columns via a ``from_json``
  classmethod accepting both ``dict`` and raw ``str``.
- ``web`` receives the projected DTO; it never calls ``map_from_db``.

## 12. Agent workflow

- ``.agents/notes/`` holds **Agent Notes** — RFC-style decision records
  preserving the *why* and *what we gave up*. Every non-trivial change
  MUST add or update one (mechanical/local edits exempt). Format
  enforced by ``scripts/verify_agent_notes.py`` (``make
  check-agent-notes``; also a pre-commit hook).
- ``.agents/skills/`` holds reusable agent workflows (``SKILL.md``);
  on machines with Claude Code, ``.claude/skills`` symlinks here.
- The wording rules of §2 apply to everything under ``.agents/``.

## 13. Pre-commit / pre-push

Pre-commit hooks (intentionally narrow): ``ruff check --fix`` and
``ruff-format`` on staged files, plus ``verify-agent-notes`` (full repo
scan, fast). The ``mypy`` hook runs at **pre-push** — full strict
typecheck of ``src/`` and ``tests/`` before the push lands.

Install both stages once:

```sh
pre-commit install --hook-type pre-commit --hook-type pre-push
```

## 14. CI gate follow-ups

- **Coverage threshold not yet enforced.** CI runs ``pytest --cov=src``
  for visibility only; tighten once a baseline is observed.
- **Real-DB integration tests are not yet wired into CI.** Either add
  Postgres/Redis ``services:`` to the CI job, or add a DB reachability
  probe in ``tests/integration/conftest.py`` that ``pytest.skip``s the
  integration suites when no DB is reachable.
- **Frontend contract sweep incomplete.** Only ``api/auth`` is fully
  migrated to generated types; the remaining ``frontend/src/api/``
  modules still carry hand-written interfaces to be converted per §6.
- **Pre-existing check-layer failures.** ``src/web/deps/embed_channels.py``,
  ``src/web/deps/rbac.py``, ``src/web/middleware/auth.py`` carry layer /
  `Any` violations predating the full-stack change; fix separately.
