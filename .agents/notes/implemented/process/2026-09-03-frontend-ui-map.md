# Agent Note: Frontend UI feature map

Status: implemented
Tags: feature-map, frontend, agent-ergonomics
Date: 2026-09-03
Scope: how agents navigate Vue routes, settings sections, and KB tabs
Related files: frontend/AGENTS.md, frontend/README.md, .agents/feature-map/ui.md

## Context

The generated HTTP map lists routers and factories but not the SPA
surfaces that call them. Agents opened `KnowledgeBase.vue` or
`Settings.vue` and walked thousands of lines to find a tab.

## Decision

`frontend/AGENTS.md` is the frontend boot file (router, Pinia,
generated types, large-component boundaries). `frontend/README.md`
lists the local npm commands. `.agents/feature-map/ui.md` maps each
top-level route, settings section, and knowledge-base tab to a Vue
file, an API module, and a `generated.json` `METHOD path` key.

Rows that call routes outside `**/router.py` use the nearest key that
is present. Large shells stay unsplit.

## Alternatives considered

- **Generate the UI map from the Vue router AST** — rejected: settings
  sections and KB tabs are query keys, not routes, and the join to
  HTTP keys still needs a human table.
- **Put frontend rules in the root AGENTS.md** — rejected: the root
  file is already at the 200-line cap; the SPA needs its own boot.

## Consequences

Agents start at `frontend/AGENTS.md` and `ui.md` instead of reading
shell components. Root `AGENTS.md` §12 points at both files. The
generated map gate is unchanged.

## Required verification

- `test -f frontend/README.md`
- `wc -l frontend/AGENTS.md` is at most 200
- `rg "WikiBrowser.vue" .agents/feature-map/ui.md`
- `make check-feature-map`
