# Agent Note: Test isolation boundaries
Tags: testing, boundaries
Related files: tests/, pyproject.toml

Status: implemented
Date: 2026-08-19
Scope: what tests mock versus run real

## Context

Service and integration tests need an async database and
network-free determinism, but must stay cheap and container-free
in CI.

## Decision

- Service tests bind an in-memory SQLAlchemy engine (`aiosqlite`)
  instead of spinning up a container.
- SSE/transport tests mock the `httpx` transport with `respx` so they
  run without a live server.
- Mock only the expensive or non-deterministic boundary (engine,
  transport, clock); keep everything downstream real.
- The pytest unraisable-exception plugin is disabled: asyncpg
  connections are torn down at GC when the sync-def session engine is
  collected, which would otherwise fail the session under
  `filterwarnings=error`.

## Alternatives considered

- **Dockerized test database** — rejected: container startup cost per
  run is unacceptable for the local loop.
- **Hand-rolled transport stand-in** — rejected: a custom fake proves
  the bridge moves bytes, not that the shipping client behaves.

## Consequences

Tests run fast and deterministically; behavior below the mocked
boundary is exercised for real.

## Required verification

`make test` must stay green.
