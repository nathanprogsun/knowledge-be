# Agent Note: Agent Notes system
Tags: agents, notes, process
Related files: .agents/notes/, scripts/verify_agent_notes.py

Status: implemented
Date: 2026-08-19
Scope: decision-record discipline for agent-driven changes

## Context

Agent-driven changes accumulate decisions in conversations that later
disappear. Without a durable record of *why* and *what we gave up*,
decisions get re-litigated or silently forgotten.

## Decision

Introduce `.agents/notes/` as the canonical decision-record store,
modeled on RFC-style notes:

- one note per decision, path-encoded
  `{lifecycle}/{class}/yyyy-mm-dd-topic-title.md`
- every non-trivial change adds or updates at least one note
- `## Alternatives considered` is mandatory for shipped/declined notes
- the path scheme and file format are enforced by
  `scripts/verify_agent_notes.py`

## Alternatives considered

- **Keep decisions in chat history only** — rejected: no shared,
  reviewable record; lost when the session ends.
- **Free-form markdown under `docs/`** — rejected: `docs/` is
  user-facing documentation; notes carry maintainer rationale and need
  a machine-checkable format.

## Consequences

Every change carries its rationale into the repository; reviewers and
future contributors can reconstruct why a decision was made.

## Required verification

`make check-agent-notes` and the `verify-agent-notes` pre-commit hook
must stay green.
