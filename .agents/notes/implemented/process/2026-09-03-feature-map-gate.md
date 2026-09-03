# Agent Note: Generated feature map gate

Status: implemented
Tags: feature-map, anti-drift
Date: 2026-09-03
Scope: how agents discover HTTP, service factories, and worker tasks
Related files: scripts/build_feature_map.py, scripts/check_feature_map.py, Makefile

## Context

Agents could follow `AGENTS.md` layer rules but still had no inventory
of product surfaces. Router prefixes, `build_*_service` factories, and
ARQ task names lived in three trees with no join key.

## Decision

`scripts/build_feature_map.py` walks `src/web/api/**/router.py`,
`src/core/**/factory.py`, and `src/workers/tasks/*.py` and writes
`.agents/feature-map/generated.json`. `make check-feature-map` diffs a
fresh generation against the committed file.

Favorites stay mapped as their own HTTP domain even though the factory
lives under `core/system` (`system+favorites` on service entries).

## Alternatives considered

- **Generate from the live OpenAPI app** — rejected: the map must
  build without booting FastAPI or a database.
- **Hand-written markdown only** — rejected: it would drift the same
  way the old "covers every domain" comment did.

## Consequences

A new `POST /api/v1/sessions`-class route or `build_*_service` that is
not regenerated fails `make check`. UI tab mapping is still a later
manual file.

## Required verification

- `uv run pytest tests/scripts/test_feature_map.py`
- `make check-feature-map`
- generated map contains `POST /api/v1/sessions` and `build_session_service`
