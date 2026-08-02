# AGENTS.md — knowledge-be

> Project conventions for human and AI contributors. Keep this file ≤ 200
> lines; refresh whenever the conventions evolve.

## 1. Layered architecture

```
web      → core, common, app_context
core     → db, ai, common, util
db       → common, db/models, db/dbengine
ai       → common, util
workers  → core, common, util
*        → common, util
```

Forbidden:
- `web` calling `db` directly.
- `core` handling HTTP concerns (`Request`, `Response`, `HTTPException`).
- `db` containing business logic or calling `ai`.
- `ai` calling `core` or `db`.
- `workers` calling `db` or `ai` directly.
- Function-level imports anywhere.
- `Any` / `object` type annotations.
- Module-level mutable globals.

## 2. Code conventions

- Python ≥ 3.11 (use `match`, `type X = ...`, PEP 695 generics).
- All imports at file top, in `stdlib → third-party → first-party` groups.
- Every variable, parameter, return type, class attribute must have an
  explicit annotation.
- Use `T | None` over `Optional[T]`.
- Files: 200–400 lines typical, 800 max.
- Functions: < 50 lines.

## 3. Service registration (DI)

- Every service is a singleton registered once in `LifeSpanService`
  during FastAPI lifespan startup.
- Web routers obtain services via
  `Annotated[T, Depends(get_xxx_from_lifespan)]`.
- Services receive their own deps via constructor injection at
  registration time — never via runtime `Depends`.
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

A change that fails any gate cannot merge.

## 8. Async purity

- All I/O is `async`.
- Services and DAOs are `async def`.
- Blocking calls via `asyncio.to_thread` or workers.