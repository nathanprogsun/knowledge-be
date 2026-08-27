# Agent Note: Agent-collaboration layout aligned with deepseek-harness patterns
Tags: agents, process, governance
Related files: .agents/, .github/, AGENTS.md, .rgignore

Status: implemented
Date: 2026-08-25
Scope: governs .agents/ taxonomy, Claude integration symlinks, and .github templates/CI lanes

## Context

The repo already had `.agents/notes` + `.agents/skills`, but the
taxonomy was partial, archived notes polluted search results, the
contract-alignment skill described a dissolved governance regime, and
`.github/` had only the CI workflow. deepseek-harness (deepseek-ai)
demonstrates a mature "agents as first-class collaborators" layout;
this change adopts the parts that fit a Python+Vue repo.

## Decision

- Notes taxonomy completed: `implemented/{architecture,bug-fix,feature,
  process,testing}` (bug-fix/feature added; the verifier's closed class
  set already accepted them).
- `.rgignore` excludes `.agents/notes/archived/` from ripgrep so stale
  decisions stop surfacing in search.
- Root `CLAUDE.md` is a symlink to `AGENTS.md` — one instruction file
  serves all agents. `.claude/skills` already symlinked to
  `.agents/skills`.
- `kb-contract-alignment` rewritten for the OpenAPI-as-truth regime
  (`make openapi` + frontend type-check as the drift signal);
  `kb-pre-push-checks` updated to drop the removed `check-contract`
  gate.
- `.github/` gains `PULL_REQUEST_TEMPLATE.md` (gates checklist + Agent
  Note field), `ISSUE_TEMPLATE/{bug_report,feature_request}.yml`, and
  a `hygiene` CI lane (`git diff --check`, full-history checkout)
  wired into the `all-checks-passed` aggregator.

## Alternatives considered

- **lefthook (deepseek uses it)** — rejected: this repo standardized
  on the Python `pre-commit` framework with a pre-push mypy stage;
  two hook managers would fight.
- **Bilingual notes (`.zh.md` + `.i18n.yaml` pairs)** — rejected:
  single-language team; pairing machinery is overhead without a
  second-language readership.
- **jscpd/knip-style dead-code gates** — deferred: value unclear until
  the frontend contract sweep (AGENTS.md §14) finishes.

## Consequences

Agents get a complete note taxonomy and current skills; contributors
get PR/issue structure; CI blocks whitespace/conflict-marker debris.
Empty `bug-fix`/`feature` dirs appear once the first note lands in
them (git does not track empty directories).

## Required verification

- `python3 scripts/verify_agent_notes.py --repo-root .` passes.
- `uv run python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"`
  passes.
- `CLAUDE.md` resolves: `head -1 CLAUDE.md` prints the AGENTS.md title.
