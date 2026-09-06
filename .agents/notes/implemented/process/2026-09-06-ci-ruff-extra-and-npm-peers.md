# Agent Note: CI ruff extra and frontend peer override

Status: implemented
Date: 2026-09-06
Scope: make the GitHub Actions lint, format, test, and frontend jobs install the tools they invoke
Related files: .github/workflows/ci.yml, frontend/.npmrc, .gitignore, .agents/feature-map/generated.json

## Context

PR CI on the operator stack failed before any product gate ran. `uv run ruff` built the default extra-less venv, then could not spawn `ruff` because the binary lives in the `dev` extra. `npm ci` rejected `typescript@6` against `openapi-typescript`'s `^5` peer. The workflow already claimed `frontend/.npmrc` carried `legacy-peer-deps`, but that file was never committed. `uv sync` on the test job could pick the runner's 3.12 even after `uv python install 3.11`, and jieba's unescaped regex is a parse-time SyntaxError on 3.12. The feature map was stale after the wiki issue and revision routes landed.

## Decision

Lint and format invoke `uv run --extra dev ruff`. The test job sets `UV_PYTHON=3.11` so sync and pytest use the installed 3.11. `frontend/.npmrc` commits `legacy-peer-deps=true` so `npm ci` matches local installs. The repo-wide `.*` ignore had been hiding that file; `.gitignore` now re-includes `.npmrc`. The feature map is regenerated from the current routers.

## Alternatives considered

- **Move ruff into default project dependencies** — rejected: lint tools stay in the `dev` extra. The job must ask for that extra.
- **Pin TypeScript to 5.x** — rejected: the SPA already types on 6. The peer conflict is only at install time.
- **`npm ci --legacy-peer-deps` only in the workflow** — rejected: local `npm ci` would still fail. The override belongs in `.npmrc`.
- **Leave the test job on the runner default Python** — rejected: jieba 0.42.1 cannot import on 3.12 under `filterwarnings=error`.

## Consequences

CI lint, format, and frontend install can reach their real gates. Test collection stays on 3.11 until jieba is replaced. The same workflow still lives on the lower stack PRs until those heads rebase onto this change.

## Required verification

- GitHub Actions `lint`, `format`, `frontend`, `test`, and `anti-drift check` on `feat/kb-op-9-wiki-issues`
- `python scripts/check_feature_map.py --repo-root .`
- `python scripts/verify_agent_notes.py --repo-root .`
