---
name: kb-agent-note
description: Use when making a non-trivial change to knowledge-be. Writes or
  updates an Agent Note under .agents/notes/ per the path scheme and format
  enforced by scripts/verify_agent_notes.py.
---

# kb-agent-note

Every non-trivial change MUST add or update at least one Agent Note in
the same change. A change is non-trivial when it alters behavior,
architecture, a contract shared across files or packages, process or
tooling, testing strategy, an on-disk/wire/config format, or any other
decision a maintainer would need the reason for. Only purely
mechanical/local edits are exempt.

## Path

```
.agents/notes/{lifecycle}/{class}/yyyy-mm-dd-topic-title.md
```

- lifecycle: `proposed` / `implemented` / `rejected` / `archived`
- class (closed set): `feature`, `bug-fix`, `simplification`,
  `architecture`, `process`, `testing`

## Format

Start from `.agents/notes/_template.md`. Mandatory elements:

- first line `# Agent Note: <topic title>`
- `Status:` line matching the lifecycle folder
- sections: `## Context`, `## Decision`,
  `## Alternatives considered`, `## Consequences`,
  `## Required verification`

`## Alternatives considered` is mandatory for `implemented`/`rejected`
notes — record each genuine alternative and why it lost.

## Enforcement

Run before committing:

```
python scripts/verify_agent_notes.py --repo-root .
```

The `verify-agent-notes` pre-commit hook runs the same check on every
commit; `make check-agent-notes` runs it in the full gate chain.
