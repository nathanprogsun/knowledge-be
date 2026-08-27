# Agent Note: D-05 — "我创建" tab folds on 1-agent case (tracing only)
Tags: testing, tracing, agent
Related files: .agents/skills/, tests/

Status: implemented
Date: 2026-08-22
Scope: D-05 (P1 UX) — the "我创建的" agent tab renders collapsed (only the
section header bar visible) when the user owns exactly one agent. The
section re-expands after a full page refresh.

## What we know

- `frontend/src/views/agent/AgentList.vue` renders the "我创建" section
  header conditionally on `showShareGroupHeaders && agent.isMine &&
  !agent.is_builtin && isMyAgent(agent)` plus a "first in the slice"
  guard. The agent cards inside the section are gated by
  `v-show="!isAgentRowHidden(agent)"`.
- `isAgentRowHidden(item)` returns true when
  `isAgentSectionCollapsed(agentSectionOf(item))` is true. The collapse
  state lives in `collapsedAgentSections`, a `ref<Set<AgentSectionKey>>`
  initialised as `new Set()` (empty → all sections expanded by
  convention; the inline comment notes "ephemeral, only valid in the
  current session").
- `applyAgentListData` (called from the list fetch) just assigns to
  `agents.value` and runs `checkAndOpenEditModal()`. It does **not**
  mutate `collapsedAgentSections`.
- `toggleAgentSection` is the only mutator of `collapsedAgentSections`
  and only fires on a header click.

## Why we are not yet sure

The static read says the initial state is "all expanded", which
contradicts the user-reported symptom. Two hypotheses remain open:

1. The Set's reactive proxy is being shared across re-mounts of the
   component (e.g. via the route-keyed keep-alive on the agent view),
   so the second mount inherits the first mount's collapsed set. The
   page refresh would reset that.
2. A future section code path (e.g. when `spaceSelection === 'mine'`,
   line 311+) carries its own collapse state that the "all" view's
   empty-Set default doesn't reset.

## What we changed (tracing)

Three `console.debug` call sites added to `AgentList.vue` so a browser
reproduction can confirm or refute the hypothesis without further code
changes:

- `isAgentSectionCollapsed` — logs every lookup with the key, result,
  and current set contents.
- `toggleAgentSection` — logs every toggle with the new set state.
- `applyAgentListData` — logs total / mine / collapsed-at-load after
  every list fetch.

All logs are prefixed `[agent-section]` / `[agent-list]` so a single
`console` filter pulls them out.

## Open follow-ups (after browser repro)

- If hypothesis 1 holds: switch `collapsedAgentSections` from a single
  module-level ref to a per-view lifecycle (e.g. inside `<KeepAlive>`'s
  onActivated hook, or simply re-init on every `onBeforeMount`).
- If hypothesis 2 holds: unify the "all" and "mine" views' collapse
  state via a shared store, or document them as intentionally separate.

Either fix is one-line. Holding off until the console output pins down
which side of the wire the bug lives on.


## Alternatives considered

- **Skip the trace and fix the most likely root cause directly**
  (reset `collapsedAgentSections` on every mount). Rejected: the
  symptom is described by the user, not reproduced locally; fixing
  the wrong hypothesis would leave the bug in place and add dead
  code. Tracing first lets the browser console pick the right fix.
- **Move `collapsedAgentSections` to a Pinia store.** Rejected for
  now: it would force every other section-collapsing view to share
  state, which is not what we want. Per-view lifecycle is the
  smaller blast radius.
- **Drop the section-collapse UI entirely on the '1 agent' edge
  case.** Rejected: it papers over the symptom for a single count
  and does not address the underlying stale-state hypothesis.
## Verification

- `pytest tests/contract/ tests/core/{chat,agents,knowledge,auth}/` is
  unaffected (frontend-only change). Expected: still
  `2088 passed, 7 xfailed, 13 failed`.
- `make check-agent-notes` accepts this note under
  `implemented/testing/`.
