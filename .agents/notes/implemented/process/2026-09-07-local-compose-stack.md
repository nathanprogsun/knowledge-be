# Agent Note: Local compose starts the product stack

Status: implemented
Date: 2026-09-07
Scope: `docker compose up` from the repo root starts Postgres, Redis, migrate, API, worker, and the Vite SPA
Related files: compose.yaml, deploy/docker/Dockerfile, deploy/docker/Dockerfile.worker, deploy/docker/docker-compose.yml, deploy/env/compose.env, .dockerignore, docker-compose.test.yml, Makefile, README.md

## Context

The previous compose file listed api and worker but loaded `deploy/env/*.env.example` placeholders that cannot boot (`<required>` is not a 32-byte AES key). The SPA was missing. Document parse no longer needs a reader container, yet README still told operators to wait for one. A host that already bound 5432/6379 could not start this stack; that conflict is gone once those containers are removed.

## Decision

The canonical file is `compose.yaml` at the repository root. `deploy/env/compose.env` holds local-only secrets that actually start the process (Postgres password `postgres`, AES key 32 hex chars, `AUTO_SETUP_ENABLED=true`). API and worker images share `deploy/docker/Dockerfile` stages installed with `uv sync --frozen --no-dev`. A one-shot `migrate` service runs `alembic upgrade head` before api/worker. `web` is Node 22 running Vite with `VITE_DEV_PROXY_TARGET=http://api:8000`. `deploy/docker/docker-compose.yml` includes the root file so older `-f` paths still work. No reader service is defined.

## Alternatives considered

- **Keep api.env.example as the compose env_file** — rejected: `<required>` cannot satisfy AES or JWT; compose would create containers that exit on boot.
- **Nginx production frontend image as the default web service** — rejected: that image needs a pre-built `dist/` and defaults `APP_HOST=app:8080`, which is the wrong API address here. Vite matches the documented local loop.
- **Add a reader container** — rejected: parse is in-process; this product does not take a third-party reader image as a dependency.
- **Interpolate host `.env` into compose** — rejected: a passworded `REDIS_URL` on the host would not match a passwordless compose Redis.

## Consequences

`docker compose up -d --build` is the local stack. Host `make dev-app` against that Postgres must use `DB_PASSWORD=postgres` and `REDIS_URL=redis://localhost:6379`, not a leftover password from another product. Compose auto-setup is on; a public deploy must not copy `compose.env`. The API image still vendors every vector-store client because they are project dependencies, so the first build is large.

## Required verification

- `docker compose config --services` lists postgres, redis, migrate, api, worker, web
- `docker compose up -d --build` then `GET http://127.0.0.1:8000/health` and `GET http://127.0.0.1:5173/login`
- `python scripts/verify_agent_notes.py --repo-root .`
