# Agent Note: Env example follows Settings

Status: implemented
Date: 2026-09-06
Scope: `.env.example` is the onboarding template for `src.settings.Settings`; a check fails when they drift
Related files: .env.example, README.md, scripts/check_env_example.py, tests/scripts/test_env_example.py, Makefile, deploy/env/api.env.example, deploy/env/worker.env.example, deploy/k8s/api-secret.yaml.example

## Context

The README and `.env.example` taught `DATABASE_URL` as the compose override. Settings reads `DATABASE_URL_OVERRIDE` and ignores unknown keys. `SYSTEM_AES_KEY` was documented as `openssl rand -base64 32`, which is 44 characters; credential encryption only accepts exactly 32 UTF-8 bytes. Compose and preflight pointed at `deploy/env/*.example` files that were not in the tree. Local `.env` carried `RBAC_ENFORCED` and `CROSS_TENANT_ACCESS_ENABLED` that the template omitted.

## Decision

`.env.example` lists every Settings field as an assignment or a commented assignment. The override name is `DATABASE_URL_OVERRIDE`. `SYSTEM_AES_KEY` is a 32-character ASCII secret (`secrets.token_hex(16)`). RBAC rollout flags are active local keys. OTEL, OIDC, and header-auth stay commented. `scripts/check_env_example.py` imports `Settings` (class only, no instance) and treats `# KEY=` as present. `DATABASE_URL` in the template is a dedicated failure pointing at `DATABASE_URL_OVERRIDE`. WorkerSettings `WORKER_*` keys are not required. The script is part of `make check` via `uv run python` so CI's system interpreter does not need pydantic on `PATH`. Helper tests use tempfile examples so they do not depend on a live template rewrite. Worker start is `make dev-worker` (`python -m src.workers.main`). Compose env templates live under `deploy/env/`. Vector-store and MinIO `os.getenv` keys stay out of the template; they are not Settings fields and a `.env` file does not populate `os.environ`.

## Alternatives considered

- **Alias `DATABASE_URL` onto `database_url_override`** — rejected: the field name is already honest. An alias would keep the false docs working.
- **Put every `os.getenv` key into `.env.example`** — rejected: those keys never load from `.env` through Settings. Dumping them would teach a mechanism that does not work.
- **Required-key pydantic validators for JWT and AES in production** — rejected: this change is the template and the drift gate. Startup policy is a separate decision.

## Consequences

A new clone can copy `.env.example` and name the URL override correctly. AES comments match `src/util/crypto.py`. `make check` fails when someone adds a Settings field and forgets the template. Retrieval env vars still need a real process environment or a later Settings promotion.

## Required verification

- `make check-env-example` (`uv run python scripts/check_env_example.py --repo-root .`)
- `uv run pytest tests/scripts/test_env_example.py`
- `make help` lists `dev-app`, `dev-worker`, `openapi`
- `python scripts/verify_agent_notes.py --repo-root .`
