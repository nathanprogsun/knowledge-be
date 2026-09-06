# Agent Note: WeKnora-shaped compose profiles

Status: implemented
Date: 2026-09-07
Scope: local compose matches WeKnora's default + optional profile model for clients this repo already has
Related files: compose.yaml, deploy/env/compose.env, Makefile, README.md

## Context

WeKnora's compose defaults to Postgres, Redis, app, frontend, and a docreader sidecar, with MinIO / Neo4j / Qdrant / Milvus / Weaviate / SearXNG / Doris behind profiles. knowledge-be's stack only started Postgres and Redis plus app processes, with no shared file volume and no optional sidecars, so uploads and alternate retrieve/graph/search paths could not be exercised the WeKnora way.

## Decision

`compose.yaml` keeps the default stack (postgres, redis, migrate, api, worker, web) and adds a shared `files_data` volume at `/data/files` with `LOCAL_STORAGE_BASE_DIR` and `RETRIEVE_DRIVER=postgres`. Optional profiles reuse WeKnora names: `minio`, `neo4j`, `qdrant`, `milvus`, `weaviate`, `searxng`, `doris`, plus `full` = minio+neo4j+qdrant+searxng. Connection env for those sidecars is always present on api/worker so a matching profile can be used after a UI/API backend row is created. WeKnora's docreader image stays out (parse is in-process / ODL). Sandbox, MCP, Dex, Langfuse, and ODL-hybrid are wired in a follow-up note (2026-09-07-weknora-optional-profiles-wired).

## Alternatives considered

- **Copy WeKnora compose verbatim including docreader** — rejected: this product must not depend on the WeKnora reader image; builtin parse is the default.
- **Make MinIO the default storage** — rejected: WeKnora itself defaults to local disk; MinIO stays a profile.
- **Auto-seed a local storage_backends row on AUTO_SETUP** — deferred: still needed for zero-click upload, but is an app change, not a compose service.
- **Put milvus/weaviate/doris into `full`** — rejected: WeKnora also keeps those out of `full` because they are heavy.

## Consequences

`docker compose up` is enough for Postgres-backed retrieve and local files once an operator creates a `local` storage backend. `docker compose --profile full up` brings the common optional sidecars. Containers alone do not switch retrieve/graph on; `NEO4J_ENABLE=true` or a vector-store / storage-backend row is still required. Langfuse remains unavailable as a product integration.

## Required verification

- `docker compose config --services` lists postgres redis migrate api worker web
- `docker compose --profile full config --services` also lists minio neo4j qdrant searxng
- `python scripts/verify_agent_notes.py --repo-root .`
