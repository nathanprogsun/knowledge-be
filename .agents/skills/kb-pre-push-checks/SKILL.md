---
name: kb-pre-push-checks
description: Use before pushing, force-pushing, marking ready for review, or
  claiming checks pass on a knowledge-be branch. Selects the narrowest local
  checks that cover the outgoing change; CI owns exhaustive coverage.
---

# kb-pre-push-checks

Before pushing a knowledge-be branch, run the checks that would fail
for the changed behavior — not the full suite on every push.

## Gates for any change

- `make lint` and `make format` — ruff
- `make typecheck` — mypy via `uv run` (project venv only; a bare
  `mypy` on PATH may resolve to a global install and produce false
  failures)
- `make test` — pytest over `tests/`
- `make check` — anti-drift gates (layer directionality, DI scope,
  endpoint coverage, contract invariants, top-level imports, SQL
  format, PR-leak scan)

## Narrowing

- When only `src/core/<domain>/` changed, run that domain's tests first
  (`pytest tests/core/<domain>/`), then the full `make check`.
- Contract-bearing files (`src/common/pagination.py`,
  `src/common/exception.py`, `src/core/contracts/*.py`, domain view
  models) additionally require `make check-contract` and the
  field-alignment review in `kb-contract-alignment`.
- A non-trivial change must carry an Agent Note (`kb-agent-note`).

## Never

- Claim checks pass without running them.
- Push a non-trivial change without an Agent Note.
- Reference internal PR ids or stage/checkpoint labels in commit text.
