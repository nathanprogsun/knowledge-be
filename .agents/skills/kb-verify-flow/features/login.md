# Login

Unauthenticated entry. Invite-register reuses the same Vue file.

## Sub-features

- Password login (`POST /api/v1/auth/login`).
- Invite-register on `/register?token=`.
- OIDC start when `GET /api/v1/auth/config` says it is enabled.

## How to get to it

Open `/login` on the SPA (`frontend/src/views/auth/Login.vue`).
Router: `frontend/src/router/index.ts`. Map row:
`.agents/feature-map/ui.md` → `POST /api/v1/auth/login`.

## Driving it with

1. Doctor: `GET {KB_WEB_BASE}/login` and `GET {KB_API_BASE}/api/v1/auth/config`.
2. Optional live login only when the user supplied credentials in
   this session. Never write them into evidence.
3. Client: `frontend/src/api/auth/index.ts`.

## Gotchas

- `/register` is the same component; it switches on `?token=`.
- A 401 on `/api/v1/auth/login` with empty body is expected. Do not
  treat it as a down API if `/auth/config` returned 200.
- OIDC callback is `/api/v1/auth/oidc/callback`, not this page.
