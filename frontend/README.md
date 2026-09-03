# knowledge-be frontend

Vue 3 + Vite + TypeScript SPA for the knowledge-be API.

The backend prefix is `/api/v1`. In dev, Vite proxies `/api` to
`http://localhost:8000` (override with `VITE_DEV_PROXY_TARGET`).

Agent conventions: [`AGENTS.md`](./AGENTS.md). Product surface map:
[`../.agents/feature-map/ui.md`](../.agents/feature-map/ui.md).

## Commands

Run these inside `frontend/` only. Do not install JS dependencies at
the repo root.

```bash
npm ci
npm run dev
npm run type-check
npm test
npm run gen:api
npm run build
```

`npm run gen:api` reads `../docs/api/openapi.json` and writes
`src/api/__generated__/schema.ts`. From the repo root, `make openapi`
exports the schema and runs the same codegen.

Repo-root wrappers:

```bash
make frontend-install
make frontend-typecheck
make frontend-test
make frontend-build
```

`vue-tsc` needs `NODE_OPTIONS=--max-old-space-size=6144` in CI.
