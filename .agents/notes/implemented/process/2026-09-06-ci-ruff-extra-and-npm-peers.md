# Agent Note: CI ruff extra and frontend peer override

Status: implemented
Date: 2026-09-06
Scope: make the GitHub Actions lint, format, test, and frontend jobs install the tools they invoke and reach their real gates
Related files: .github/workflows/ci.yml, frontend/.npmrc, .gitignore, src/common/jieba_compat.py, frontend/src/components/AttachmentUpload.vue, .agents/feature-map/generated.json

## Context

PR CI on the operator stack failed before any product gate ran. `uv run ruff` built the default extra-less venv, then could not spawn `ruff` because the binary lives in the `dev` extra. `npm ci` rejected `typescript@6` against `openapi-typescript`'s `^5` peer. The workflow already claimed `frontend/.npmrc` carried `legacy-peer-deps`, but that file was never committed. After those install fixes, the real gates failed: ruff reported UP017/UP035/B009, `uv run` from `frontend/` looked for `frontend/scripts/export_openapi.py`, and pytest collection died on jieba 0.42.1's unescaped regex literals. The feature map was stale after the wiki issue and revision routes landed.

## Decision

Lint and format invoke `uv run --extra dev ruff`. The test job sets `UV_PYTHON=3.11` and `PYTHONWARNINGS=ignore::SyntaxWarning`. `frontend/.npmrc` commits `legacy-peer-deps=true` so `npm ci` matches local installs. The repo-wide `.*` ignore had been hiding that file; `.gitignore` now re-includes `.npmrc`. The OpenAPI export runs from the repo root. `src/common/jieba_compat.py` rewrites jieba's `re.compile` literals to raw strings before import. The unit job also ignores live-Postgres trees (`tests/contract`, `tests/db`, `test_integration_*`). The feature map is regenerated from the current routers.

## Alternatives considered

- **Move ruff into default project dependencies** — rejected: lint tools stay in the `dev` extra. The job must ask for that extra.
- **Pin TypeScript to 5.x** — rejected: the SPA already types on 6. The peer conflict is only at install time.
- **`npm ci --legacy-peer-deps` only in the workflow** — rejected: local `npm ci` would still fail. The override belongs in `.npmrc`.
- **Leave the test job on the runner default Python** — rejected: jieba 0.42.1 cannot import on 3.12 under `filterwarnings=error`.
- **Pin jieba to a git fork** — rejected: upstream master still ships the same literals. A local rewrite stays on the locked 0.42.1 sdist.

## Consequences

CI lint, format, and frontend install can reach their real gates. Test collection can import the retrieval stack on runners that promote jieba's literals to SyntaxError. The same workflow still lives on the lower stack PRs until those heads rebase onto this change.

## Required verification

- GitHub Actions `lint`, `format`, `frontend`, `test`, and `anti-drift check` on `feat/kb-op-9-wiki-issues`
- `python scripts/check_feature_map.py --repo-root .`
- `python scripts/verify_agent_notes.py --repo-root .`
