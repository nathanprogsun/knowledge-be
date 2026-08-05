# AGENTS.md — knowledge-be

> Project conventions for human and AI contributors. Keep this file ≤ 200
> lines; refresh whenever the conventions evolve.

## 1. Layered architecture

```
web      → core, common, app_context
core     → db, common, util
db       → common, db/models, db/dbengine
workers  → core, common, util
*        → common, util
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
  `get_xxx_from_lifespan(request.app)`. Long-lived resources must be
  closed in the lifespan shutdown (e.g. `OidcClient.aclose()`).
- **REQUEST scope** — anything holding the per-request `AsyncSession`
  (repositories, and services binding repositories): constructed per
  request in `web/deps/` factories. The factory composes fresh repos
  with APP-scope singletons pulled from the lifespan registry.
- Request-scoped services MUST NOT be registered on `LifeSpanService`
  (enforced by `scripts/check_service_singleton.py`).
- Never use class-level Singleton bases/metaclasses: they break test
  transport injection, bind to the wrong event loop, and hide
  dependencies. "One instance per app" is achieved by constructing once
  in the lifespan — not by type-level tricks.
- Services receive their own deps via constructor injection — never via
  runtime `Depends` inside `core`.
- Services MUST NOT hold request-scoped state; request data is read
  from `src.app_context.request_context`.

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
{AIProviderError, VectorStoreError, StorageBackendError} | DataError`.

## 6. Contracts

- `src/core/contracts/` holds frozen Pydantic models
  (`model_config = ConfigDict(frozen=True)`). Modules fill in implementations
  but never alter schemas or field names.

## 7. CI gates

Pre-merge gates:
- `make lint` — ruff check
- `make format` — ruff format --check
- `make typecheck` — mypy --strict
- `make test` — pytest
- `make check` — anti-drift scripts (layer directionality, DI scope,
  endpoint coverage, contract invariants, top-level imports)

A change that fails any gate cannot merge.

## 8. Async purity

- All I/O is `async`.
- Services and DAOs are `async def`.
- Blocking calls via `asyncio.to_thread` or workers.

## 9. DB-row → service-DTO projection (`map_from_db`)

Service-output DTOs that mirror a storage row expose a
`map_from_db(cls, db: <StorageModel>) -> Self` classmethod performing
the boundary translation. The db layer never references the wire DTO
(§1); the service calls ``XxxInfo.map_from_db(row)``.

- Defined on the DTO (in ``core``), not on the ``TableModel``.
- Strips sensitive / storage-only columns (e.g. ``password_hash``,
  ``deleted_at``); the exclude-set is a module-level ``frozenset`` next
  to the DTO.
- Hydrates nested typed sub-models from JSON-backed columns via a
  ``from_json`` classmethod that accepts both parsed ``dict`` and raw
  JSON ``str`` (driver-portability).
- ``web`` receives the projected DTO; it never calls ``map_from_db``.