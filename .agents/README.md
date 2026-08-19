# Agent Workflows and Agent Notes

Agent-driven engineering infrastructure for this repository:

- `notes/` — **Agent Notes**: RFC-style decision records written by
  agents. They capture the *why* and *what we gave up* — the parts code
  and docs cannot carry.
- `skills/` — **reusable agent workflows** (`SKILL.md` files): code
  review, pre-push checks, contract alignment, CoT-leakage trimming,
  and Agent Note authoring, packaged for AI coding agents.

## Public-facing constraint

This repository is public. Everything under `.agents/` is committed and
public-facing; it follows the same wording rules as the rest of the
codebase (AGENTS.md §2): comments and prose carry business rationale
only, never internal PR ids, stage/checkpoint labels, or the upstream
project name. Use neutral terms ("upstream contract", "domain model",
"storage column").
