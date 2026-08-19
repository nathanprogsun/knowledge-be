---
name: kb-code-review
description: Use when reviewing a pull request in the knowledge-be repo.
  Orients the reviewer to this codebase's standards (AGENTS.md conventions,
  layered architecture, error hierarchy, DI scope, contracts) and the
  review-specific checks that code alone cannot show.
---

# kb-code-review

Review a knowledge-be change against the conventions in AGENTS.md
(§1–§9).

## Layering

- `web` must not import `db`; repositories are instantiated only in
  `core/<domain>/factory.py`; `web/deps` factories are one-line
  forwarders.
- `db` is called only by repositories; repositories only by services.
- `core` must not handle HTTP concerns (`Request`, `Response`,
  `HTTPException`).
- No function-level imports; no `Any`/`object` annotations; no
  module-level mutable globals.

## Errors

- `core`/`db`/`ai`/`workers` raise `ApplicationError` subclasses only;
  `web` translates them via a single exception handler. Never raise
  `fastapi.HTTPException` from non-`web` layers.

## Contracts

- Frozen Pydantic models under `src/core/contracts/`; modules fill in
  implementations but never alter schemas or field names.
- Verify field names and shapes match the upstream contract, including
  JSON serialization names (see `kb-contract-alignment`).

## DI scope

- APP-scope singletons live on `LifeSpanService`; REQUEST-scoped
  services must not be registered there.
- No class-level Singleton bases/metaclasses.
- Request data is read from `src.app_context.request_context`.

## Persistence

- Raw SQL via `sqlalchemy.text()` with named `bindparams` only; never
  f-strings, `%`-format, or ORM `.select()`.
- DAOs accept an `AsyncSession` and never commit/rollback; transactions
  live in the service layer.
- Soft-deleted tables filter `WHERE deleted_at IS NULL` on every read.

## Checks that code alone cannot show

- New/changed endpoints are reflected in the endpoint inventory.
- Service-output DTOs expose `map_from_db` and strip storage-only
  columns (`password_hash`, `deleted_at`).
- A non-trivial change carries an Agent Note with `## Alternatives
  considered`.
- Committed prose is resolvable at HEAD (`kb-trim-cot-leakage`).
