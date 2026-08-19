# Agent Notes

Agent Notes are RFC-style decision records written by agents: durable
proposals and decision records that preserve the *why* and *what we
gave up* — the parts code and docs cannot carry.

## Where notes live

```
.agents/notes/{lifecycle}/{class}/yyyy-mm-dd-topic-title.md
```

Two axes, both encoded in the **path**:

- **lifecycle**: `proposed/` → `implemented/` | `rejected/` → `archived/`
- **class** — a closed set, enforced by `scripts/verify_agent_notes.py`:
  `feature`, `bug-fix`, `simplification`, `architecture`, `process`,
  `testing`

## When to write one

Every non-trivial change MUST add or update at least one Agent Note in
the same change. A change is non-trivial when it alters:

- behavior, architecture, or a contract shared across files/packages
- process or tooling
- testing strategy
- an on-disk, wire, or configuration format
- any decision a maintainer would need to know the reason for

Only purely mechanical/local edits are exempt.

## File format

Start from `_template.md`. Mandatory elements:

- first line: `# Agent Note: <topic title>`
- a `Status:` line matching the lifecycle folder
  (`proposed` / `implemented` / `rejected` / `archived`)
- sections: `## Context`, `## Decision`,
  `## Alternatives considered`, `## Consequences`,
  `## Required verification`

`## Alternatives considered` is mandatory for `implemented` and
`rejected` notes: record each genuine alternative and why it lost. A
decision recorded without what it beat invites re-litigation.

`scripts/verify_agent_notes.py` enforces the closed class set, the path
scheme, the `Status:` line, and the alternatives section.

## Lifecycle

- `proposed/` — under consideration; not yet accepted.
- `implemented/` — shipped. Keep it current with what actually shipped
  (facts only — paths, names, structure — not the decision itself).
- `rejected/` — considered and declined; records why.
- `archived/` — permanently frozen. Do not edit, move, or delete
  archived notes, and do not treat them as authority for current
  behavior.

## Skills

Reusable workflows live in `.agents/skills/` (`SKILL.md`). Relevant to
notes: `kb-agent-note` (authoring) and `kb-pre-push-checks` (gates).
