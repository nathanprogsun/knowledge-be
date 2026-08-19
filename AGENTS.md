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
- Test-time conftests under `tests/integration/` are exempt from the
  REQUEST-scope rule: they may hold session-scoped real services and
  repositories on the test fixture graph because each test owns its
  own request lifetime via the test app's lifespan. This exemption
  is the only sanctioned path for session-scoped real services
  outside `src/`.
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
- `make typecheck` — `uv run mypy` (strict from mypy.ini). Always run
  through the project venv via `uv run` — a bare `mypy` on PATH may
  resolve to a global install and produce false failures.
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

## 10. Agent workflow

- ``.agents/notes/`` holds **Agent Notes** — RFC-style decision records
  preserving the *why* and *what we gave up*. Every non-trivial change
  MUST add or update at least one Agent Note in the same change (only
  mechanical/local edits are exempt). The path scheme and format are
  enforced by ``scripts/verify_agent_notes.py``
  (``make check-agent-notes``; also a pre-commit hook).
- ``.agents/skills/`` holds reusable agent workflows (``SKILL.md``):
  ``kb-pre-push-checks``, ``kb-code-review``,
  ``kb-contract-alignment``, ``kb-trim-cot-leakage``, ``kb-agent-note``.
  On machines with Claude Code, ``.claude/skills`` symlinks here.
- The public-facing wording rules of §2 apply to everything under
  ``.agents/``.

## 11. Pre-commit / pre-push

Pre-commit hooks (run on every commit) are intentionally narrow:
``ruff check --fix`` and ``ruff-format`` on staged files, plus
``verify-agent-notes`` (full repo scan, fast). The ``mypy`` hook is
moved to the **pre-push** stage so commit-time stays fast — full
strict typecheck of ``src/`` and ``tests/`` runs before the push lands.

Install both stages once:

```sh
pre-commit install --hook-type pre-commit --hook-type pre-push
```

## 12. CI gate follow-ups

- **Coverage threshold not yet enforced.** CI runs ``pytest --cov=src``
  for visibility only; no ``fail_under`` until a baseline is observed
  over a release cycle. The audit recommends tightening the threshold
  once the real number is known.
- **Anti-drift ``check-endpoint`` is fail-closed.** ``scripts/check_endpoint_coverage.py``
  now errors out when the docs root is missing instead of passing
  silently; bundle the upstream API endpoint tables in ``docs/api/*.md``
  (or pass ``--docs-root`` in CI) before merging the change that
  exposed this gate.
- **Real-DB integration tests are not yet wired into CI.** ``docker-compose.test.yml``
  lists Postgres and Redis as commented-out placeholders. Either uncomment
  them and add a ``services:`` block to the CI job, or add a DB
  reachability probe in ``tests/integration/conftest.py`` that
  ``pytest.skip``s the integration suites when no DB is reachable.