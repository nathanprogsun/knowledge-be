---
name: kb-verify-flow
description: Drive login and knowledge-base-list through a short Launch /
  Doctor / Drive / Evidence / Cleanup loop. Use when checking that a
  local knowledge-be checkout can still reach those two surfaces.
---

# kb-verify-flow

Minimum executable loop for the two boot surfaces. Do not treat a
green unit suite as a substitute. Feature notes live in `features/`.

Default URLs: API `http://127.0.0.1:8000`, SPA `http://127.0.0.1:5173`.
Override with `KB_API_BASE` and `KB_WEB_BASE`.

Evidence directory: `.agents/skills/kb-verify-flow/evidence/`.
Cleanup deletes only `evidence/tmp/`. Reports stay.

## Launch

1. Prefer already-running processes. Do not start Postgres/Redis or
   `make dev-app` unless the user asked for a live stack.
2. If the API is down, record that in Doctor and continue. The loop
   still writes evidence.
3. Optional start (only when asked): `make dev-app` from the repo
   root, and `npm run dev` inside `frontend/`.

Helper: `python3 .agents/skills/kb-verify-flow/helpers/run_loop.py launch`

## Doctor

Check, in order:

1. Repo root has `.env` (or `.env.example` as a stand-in).
2. `frontend/node_modules` exists if a SPA probe is planned.
3. Every `features/*.md` (except README) has the four required
   headings.
4. `GET {KB_API_BASE}/api/v1/auth/config` — public, no session.
5. `GET {KB_WEB_BASE}/login` — SPA login route.

Write `evidence/doctor.json`. A down API is a Doctor finding, not a
script crash.

Helper: `python3 .agents/skills/kb-verify-flow/helpers/run_loop.py doctor`

## Drive

For each file in `features/` other than README:

1. Read **How to get to it** and **Driving it with**.
2. Hit the listed public probe (login page or auth config).
3. Do not invent credentials. Authenticated KB-list probes stay
   skipped unless `KB_VERIFY_TOKEN` is set.

Write `evidence/drive.json`.

Helper: `python3 .agents/skills/kb-verify-flow/helpers/run_loop.py drive`

## Evidence

After Drive, write `evidence/last-run.md` summarizing Launch, Doctor,
Drive, and what Cleanup will keep. Keep the file after Cleanup.

Helper: `python3 .agents/skills/kb-verify-flow/helpers/run_loop.py evidence`

## Cleanup

Delete `evidence/tmp/` only. Leave `doctor.json`, `drive.json`, and
`last-run.md`.

Helper: `python3 .agents/skills/kb-verify-flow/helpers/run_loop.py cleanup`

## Helpers

`helpers/run_loop.py` is the single entry. `python3 …/run_loop.py all`
runs Launch → Doctor → Drive → Evidence → Cleanup.

`helpers/http_probe.py` is the GET wrapper used by Doctor and Drive.

## Never

- Store passwords or live tokens in evidence files.
- Delete `evidence/*.md` or `evidence/*.json` during Cleanup.
- Claim a feature is driven if Doctor could not reach its probe and
  no skip reason was recorded.
