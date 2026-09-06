# frontend — agent boot

Vue 3 + Vite + TypeScript SPA. Talk to the backend only over HTTP
(`/api/v1`). Never import `src/` from this package.

## Entry points

- Router: `src/router/index.ts`. Product routes hang under `/platform`.
- App shell: `src/views/platform/index.vue`.
- Public embed chat is not a route in this SPA. Admin embed lives under
  Settings `integration-embed` (`AgentEmbedChannelPanel.vue`).
- UI surface map (tabs, settings sections): `../.agents/feature-map/ui.md`.
- Generated HTTP inventory: `../.agents/feature-map/generated.json`.

## Pinia

Stores live in `src/stores/`. Keep request/session state in Pinia, not
module-level mutables. Auth token + tenant scope: `stores/auth.ts`.
Do not add a new store when an existing one already owns the resource
(`knowledge.ts`, `chatResources.ts`, `organization.ts`).

## Generated types

`src/api/__generated__/schema.ts` is produced by `npm run gen:api`
(`make openapi` from the repo root). New or changed API clients must
use `components["schemas"]`. Local view-models, when needed, are
`Partial<Omit<...>>` — do not re-declare wire fields by hand.

`src/api/auth` is the reference client. Other `src/api/*` modules still
carry hand-written interfaces; migrate those instead of copying them.

## Large components

These files are multi-feature shells. Do not add a fourth concern
inside them; extract a child or a store instead.

- `src/views/knowledge/KnowledgeBase.vue` — documents / wiki / graph
  via `?tab=`. Wiki UI lives in `src/views/knowledge/wiki/WikiBrowser.vue`.
- `src/views/agent/AgentEditorModal.vue` — agent create/edit.
- `src/views/knowledge/components/FAQEntryManager.vue` — FAQ editor.

This round does not split those files. Change the smallest child that
already owns the behavior.

## Commands

Run inside `frontend/` only. Never `npm install` at the repo root.

```bash
npm run dev
npm run type-check
npm test
npm run gen:api
```
