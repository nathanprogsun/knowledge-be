# Knowledge base list

Default post-login landing page.

## Sub-features

- Owned / shared / favorites / recents filters.
- Create knowledge base (contributor+).
- Jump into `/platform/knowledge-bases/:kbId`.

## How to get to it

After login the router sends `/` to `/platform/knowledge-bases`
(`frontend/src/views/knowledge/KnowledgeBaseList.vue`). Map row:
`.agents/feature-map/ui.md` → `GET /api/v1/knowledge-bases`.

## Driving it with

1. Require a session. Skip the authenticated GET unless
   `KB_VERIFY_TOKEN` is set.
2. Probe: `GET {KB_API_BASE}/api/v1/knowledge-bases` with
   `Authorization: Bearer $KB_VERIFY_TOKEN`.
3. Client: `frontend/src/api/knowledge-base/index.ts`.

## Gotchas

- Empty states differ by filter (`all` / `favorites` / `recents` /
  `mine`). An empty list is not a failed probe.
- `/knowledgeBase` is a legacy alias of the KB editor, not this list.
- List cards mix owned and shared rows; do not count duplicates as
  extra knowledge bases.
