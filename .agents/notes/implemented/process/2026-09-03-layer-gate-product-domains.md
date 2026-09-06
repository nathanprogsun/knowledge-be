# Agent Note: Layer gate covers product domains

Status: implemented
Tags: check-layer, makefile, domains
Date: 2026-09-03
Scope: which domains `make check-layer` actually scans
Related files: Makefile, AGENTS.md

## Context

`make check-layer` claimed to cover every shipped domain while only
scanning auth/tenant plus infra. Product domains (chat, favorites,
embed, sharing, …) could regress behind a green gate.

## Decision

`check-layer` now scans auth/tenant, infra, and the product domain
list: `favorites,chat,organizations,channels,knowledge,knowledge_bases,agents,evaluation,sharing,me,files,cloud`.

`ai` and `workers` stay out of this gate. The retrieval package still
carries a large `Any` backlog that would drown the signal.

AGENTS.md §14 states the scoped coverage instead of implying a full
scan.

## Alternatives considered

- **Scan every domain including `ai` and `workers`** — rejected: the
  retrieval `Any` debt would make the gate unusable until a separate
  cleanup lands.
- **Leave the Makefile list unchanged and only fix the comment** —
  rejected: a truthful comment without a wider scan still lets product
  layers drift.

## Consequences

A new `web -> db.models` import or `dict[str, object]` annotation in
a product domain fails `make check`. Retrieval / worker typing debt
is still invisible to this target.

## Required verification

- `make check-layer` passes.
- `make check` passes.
- `rg "covers every shipped domain" Makefile` has no matches.
