# Agent Note: Minimum verify-flow skill

Status: implemented
Tags: agent-ergonomics, verify-flow
Date: 2026-09-03
Scope: how agents drive login and the knowledge-base list
Related files: .agents/skills/kb-verify-flow/SKILL.md

## Context

Agents could lint and typecheck a change and still not know whether
login or the knowledge-base list was reachable on a local checkout.

## Decision

`.agents/skills/kb-verify-flow` is the loop: Launch, Doctor, Drive,
Evidence, Cleanup. Feature notes for those two surfaces live under
`features/`. `helpers/run_loop.py all` writes JSON/markdown under
`evidence/` and cleanup removes only `evidence/tmp/`.

The loop does not start Postgres or `make dev-app`. A down API is a
Doctor finding. Authenticated KB-list probes stay skipped unless
`KB_VERIFY_TOKEN` is set.

## Alternatives considered

- **Browser-only driving** — rejected: a local checkout often has no
  live SPA, and the first loop must still produce evidence.
- **Require a logged-in session** — rejected: storing credentials in
  the skill tree is worse than an explicit skip.

## Consequences

An agent can run one command and leave a `last-run.md` that survives
cleanup. Live-stack login remains a later, credentialed step.

## Required verification

- Feature files contain the four required headings
- `python3 .agents/skills/kb-verify-flow/helpers/run_loop.py all`
- `evidence/last-run.md` exists after cleanup
