# Operator completion plan

This program closes every SPA click that still 404s, 5xxs, or lies.
The operator is the person who logs in and runs knowledge, chat, and workspace.
The rule is a happy-path click must show a true state.
The PR ids in order are kb-op-1 through kb-op-9.
Owners build the stack. The operator lands it.

## How to read this

One box is one unit of work. Every box names the evidence that checks it. A nested box is a sub-step of the box above it. Check a box only when its evidence exists, a file, a log line, a screenshot, a test run, or a SHA. The body is a how-to. The appendices explain and record.

The program runs `pstack/skills/poteto-mode/playbooks/autopilot-stack.md`. The operator lands kb-op-1 through kb-op-9. Owners stop at STACK-READY.

Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

## Program checklist

### Arm the program

- [ ] State the protocol and this plan to the operator, then stop. Start execution only on her explicit go.
- [ ] On her go, arm a `/goal` with this exact text. ".agents/docs/operator-completion-plan.md. PR ids kb-op-1 kb-op-2 kb-op-3 kb-op-4 kb-op-5 kb-op-6 kb-op-7 kb-op-8 kb-op-9. Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked. The operator lands the stack. Done when those nine PRs are merged and the live SPA happy path shows no 4xx, 5xx, or lying pending."
- [ ] Read these from trunk at program start. Re-read them at every tick.
  - [ ] `git show origin/main:pstack/skills/poteto-mode/playbooks/autopilot-stack.md`
  - [ ] `git show origin/main:pstack/skills/swarm/SKILL.md`
  - [ ] `git show origin/main:cursor-team-kit/skills/control-ui/SKILL.md`
  - [ ] `git show origin/main:pstack/skills/poteto-mode/playbooks/opening-a-pr.md`
  - [ ] `git show origin/main:pstack/skills/how/SKILL.md`
  - [ ] `git show origin/main:pstack/skills/interrogate/SKILL.md`
  - [ ] `git show origin/main:pstack/skills/show-me-your-work/SKILL.md`
  - [ ] Also read this repo trunk at `origin/master`. Skills live in the plugin cache when they are missing from the git tree.
- [ ] Arm the 30-minute audit tick. In a local session, a real terminal `/loop`. In a cloud root, a cloud-sleeper wake chain. Never leave the cadence to memory.
- [ ] Use this tick prompt, verbatim. "Re-read the execution playbook from trunk and the armed /goal. Audit the operation against both and fix drift in this tick. Probe every active lane and judge progress by side effects only. Stand down a stuck lane and dispatch its replacement now. Then send the operator a status message, whether or not anything changed, with the queue table of PR, owner, state, and head SHA, the verdicts since the last tick, what merged, open operator gates, and blockers."
- [ ] On the operator's hold or stand-down, send every owner a zero-writes order at once.

### Spawn owners

- [ ] Spawn one owner per PR with the full lifecycle the execution playbook names.
- [ ] Follow this dependency graph. Start dependent work only after its parent merges, or base it on the parent branch when the execution playbook stacks.
  - [ ] kb-op-1 is first. Branch it from `master`.
  - [ ] kb-op-2 after kb-op-1.
  - [ ] kb-op-3 after kb-op-2.
  - [ ] kb-op-4 after kb-op-3.
  - [ ] kb-op-5 after kb-op-4.
  - [ ] kb-op-6 after kb-op-5.
  - [ ] kb-op-7 after kb-op-6.
  - [ ] kb-op-8 after kb-op-7.
  - [ ] kb-op-9 after kb-op-8.
- [ ] Hold the file boundaries. kb-op-1 is the leftover working tree only. Leave out `.agents/docs/ai-native-ergonomics-plan.md` and `.agents/skills/kb-verify-flow/evidence/*`. Later PRs stay inside the globs in their Files boxes.
- [ ] Hold the review gate. kb-op-1 through kb-op-9 change an interaction. They wait for the operator's review in chat with screenshots and a video before merge.

### PR mechanics, for every PR

- [ ] Resolve the forge once. Default to `gh`; if `command -v origin` succeeds and Origin can resolve the repository, use `origin pr` for every PR operation. Record any fallback to `gh`. Never require `gt`.
- [ ] Open the PR ready, never draft, with `origin pr create --status open --base <base-branch>` or `gh pr create --base <base-branch>` according to the resolved forge. A stack child targets its parent branch.
- [ ] Run the repo's lint and typecheck once before the PR-facing push. Push with hooks on.
- [ ] Run `/deslop` before each commit and `/no-comments` before review.
- [ ] Triage every Bugbot and security-reviewer comment per `../references/bugbot-triage.md`.
- [ ] Rebase onto current trunk before babysit and again before the merge-ready report.

### Verdict and merge, for every PR

- [ ] At the merge-ready head SHA, run the swarm per `pstack/skills/swarm/SKILL.md`. One gates lane. The ten live lanes from the PR's **Verify, live** block. The perf lane from its **Verify, perf** block. One audit lane that reads the diff and the receipts and distrusts the PR body.
- [ ] Clean only when every lane is `PASS`. Findings go back to the owner. A new head gets a fresh swarm and a fresh verdict.
- [ ] Root appends the PR to the linear base-branch stack after a clean verdict. The operator lands bottom-up. After rebase, compare `git patch-id` of the base-to-head diff to the verdict patch-id. Re-verify when the patch-id changes. Keep the code verdict and re-run CI when it does not.

### Boot recipe, for every live lane

Each live lane runs on its own cloud VM at the PR head. Drive through `control-ui` or `control-cli` from `cursor-team-kit`.

- [ ] `git fetch origin <head-branch> && git checkout <head SHA>`.
- [ ] Start Postgres, Redis, and docreader from the local compose. Start API with `AUTO_SETUP_ENABLED=true uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000`. Start the SPA with `npm run dev` in `frontend/`. Start the worker with `uv run python -m src.workers.main`. Wait until `/api/v1/system/health` and `http://localhost:5173` answer.
- [ ] Deliver input only through `control-ui` commands. Read-only diagnostics are `browser_snapshot`, `browser_console_messages`, and API `fetch` in the page.
- [ ] Save every screenshot to `/tmp/swarm-<pr-id>/worker-<n>/<slug>.png` and return the paths with the report.

## Land leftover operator routes (kb-op-1)

**Depends on.** None.

**Files.**

- [ ] Edit the leftover working tree already on `master`. Do not add new behavior.
- [ ] Edit `.agents/notes/implemented/bug-fix/2026-09-04-*.md` and `2026-09-05-*.md` that already record this wave.
- [ ] Delete nothing under `.agents/skills/kb-verify-flow/evidence/`. Leave that evidence unstaged.
- [ ] Edit `docs/api/openapi.json`, `frontend/src/api/__generated__/schema.ts`, `scripts/endpoint_inventory.json`, and `tests/integration/web/test_routers.py` once at the end after `make openapi && make check-endpoint`.

**Build.**

- [ ] Keep `PersistentMessageGateway`, `PUT /knowledge-bases/{id}/pin`, `POST /sessions/{id}/stop`, `/me/invitations*`, manual fields on `PUT /knowledge/{id}`, and `GET /knowledge/{id}/download` plus `/preview` as they stand.

**You see.**

- [ ] The KB list pin button returns 200 and the row shows `is_pinned`.
- [ ] Stop during a stream toasts success.
- [ ] The invitations dialog lists rows.
- [ ] Manual editor save returns 200 and keeps `content`.
- [ ] A stored file downloads. A URL row without bytes returns `knowledge.file_unavailable`.

**Verify, unit.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] `tests/web/test_session_message_views.py` stop cases. Run `uv run pytest tests/web/test_session_message_views.py tests/web/test_me_invitations.py tests/web/test_document_reads.py tests/core/knowledge/test_kb_service_crud.py tests/integration/web/test_routers.py tests/integration/web/api/knowledge/documents/test_controller.py tests/integration/web/api/knowledge_bases/test_controller.py`.

**Verify, live.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked. Ten lanes on `grok-4.6-fast-xhigh` at the PR head, per the boot recipe.

- [ ] Lane 1. Regression lane against trunk. Run login and open `/platform/knowledge-bases` at trunk and head. Trunk lacks this leftover wave. Gate pin, stop, inbox, manual save, and download on head, plus the KB list the user waits for. Save `kb-op-1-regression.png`. Pass when the list renders and those five actions return a true 2xx or the honest file-unavailable code.
- [ ] Lane 2. Toggle pin twice on one KB. Save `kb-op-1-pin.png`. Pass when the list `is_pinned` matches the last toggle.
- [ ] Lane 3. Start knowledge-chat and click Stop. Save `kb-op-1-stop.png`. Pass when the toast is success and the status is not 404.
- [ ] Lane 4. Open the invitations dialog. Save `kb-op-1-inbox.png`. Pass when `GET /me/invitations` is 200.
- [ ] Lane 5. Edit a manual document and save. Save `kb-op-1-manual.png`. Pass when reload shows the new markdown.
- [ ] Lane 6. Download a file document. Save `kb-op-1-download.png`. Pass when the browser receives bytes.
- [ ] Lane 7. Preview a URL document with empty `file_path`. Save `kb-op-1-url-preview.png`. Pass when the code is `knowledge.file_unavailable`.
- [ ] Lane 8. Finish one knowledge-chat turn and wait for suggestion chips. Save `kb-op-1-suggest.png`. Pass when `POST` suggestions is 200 and chips render.
- [ ] Lane 9. Reload the session. Save `kb-op-1-history.png`. Pass when `GET /messages/{session}/load` returns the user and assistant rows.
- [ ] Lane 10. Type `@` in the composer. Save `kb-op-1-search.png`. Pass when `GET /knowledge/search` lists the test KB docs.

**Verify, perf.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] Metric. Time from KB list navigation to the first row paint, at trunk and at head. Also time from pin click to the updated list on head, plus the list the user waits for.
- [ ] Probe. `control-ui` navigation plus Performance marks around the list XHR, run at trunk and at the head, interleaved. Both sides must produce the list paint time.
- [ ] Baseline. Record the trunk list paint time first.
- [ ] Rule. Head list paint must stay within 20 percent of trunk. Pin click to updated list must finish within 2000 ms. Fail either number.

**Review gate.** The operator reviews before merge.

- [ ] Copy lane 2 and lane 5 screenshots into `.audit/media/kb-op-1-review-pin.png` and `.audit/media/kb-op-1-review-manual.png`.
- [ ] Record a 30 to 60 second video of the change on a lane VM. Save it as `.audit/media/kb-op-1-review.mp4`.
- [ ] Post the screenshots and the video in chat. Stop at merge-ready. Wait for the operator's click.

**Merge.**

- [ ] Root's clean verdict at the exact head SHA.
- [ ] Bugbot triage done.
- [ ] Rebased onto current trunk after the verdict, patch-id unchanged.
- [ ] The root appends kb-op-1 as the stack root targeting `master`. The operator lands it.

## Share StreamManager so stop cancels (kb-op-2)

**Depends on.** kb-op-1.

**Files.**

- [ ] Edit `src/app_context/registry.py`.
- [ ] Edit `src/app_context/lifespan.py`.
- [ ] Edit `src/core/chat/sessions/factory.py`.
- [ ] Edit `src/core/chat/factory.py`.
- [ ] Edit `src/core/chat/service.py`.
- [ ] Edit `src/web/deps/chat_sessions.py`.
- [ ] Edit `src/web/deps/chat.py`.
- [ ] Edit `src/web/deps/embed_channels.py`.
- [ ] Create `.agents/notes/implemented/bug-fix/2026-09-05-shared-stream-manager.md`.

**Build.**

- [ ] Put one `MemoryStreamManager` on `LifeSpanService`. Inject that same instance into `StopStreamService` and `ChatService`, including embed chat. Race `_stream_qa` against `wait_cancelled`. Do not add Redis. Do not wire `continue_stream.py`.

**You see.**

- [ ] Clicking Stop ends new tokens. Reload does not grow the assistant row after stop.

**Verify, unit.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] `tests/core/chat/test_factory.py` and `tests/web/test_session_message_views.py` gain a shared-manager cancel case. Run `uv run pytest tests/core/chat/test_factory.py tests/web/test_session_message_views.py tests/core/chat/test_service.py`.

**Verify, live.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked. Ten lanes on `grok-4.6-fast-xhigh` at the PR head, per the boot recipe.

- [ ] Lane 1. Regression lane against trunk. Run a full knowledge-chat answer at trunk and head. Gate stop-cancels-tokens on head plus the finished answer the user waits for on both. Save `kb-op-2-regression.png`. Pass when an unstopped turn still completes on head.
- [ ] Lane 2. Stop mid-stream on a long answer. Save `kb-op-2-stop.png`. Pass when the model log stops and no new SSE tokens arrive.
- [ ] Lane 3. Reload after stop. Save `kb-op-2-reload.png`. Pass when the assistant text matches the stopped buffer.
- [ ] Lane 4. Stop then send a new question. Save `kb-op-2-next-turn.png`. Pass when the new turn streams.
- [ ] Lane 5. Two sessions, stop only one. Save `kb-op-2-two-sessions.png`. Pass when the other session still finishes.
- [ ] Lane 6. Stop with a missing `message_id`. Save `kb-op-2-no-message-id.png`. Pass when the response is 200 or a typed 4xx, never 500.
- [ ] Lane 7. Agent-chat stop. Save `kb-op-2-agent-stop.png`. Pass when agent-chat tokens stop.
- [ ] Lane 8. Suggestions after a stopped turn. Save `kb-op-2-suggest-after-stop.png`. Pass when suggestions are 200 or empty, never 501.
- [ ] Lane 9. Stop from the logged-in composer after an embed-style ChatService build. Save `kb-op-2-embed-dep.png`. Pass when tokens still stop. Do not add continue-stream.
- [ ] Lane 10. Ghost thinking row after stop. Save `kb-op-2-ghost.png`. Pass when no extra 正在思考 row remains.

**Verify, perf.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] Metric. Time from Stop click to the last SSE chunk, at head. Time for an unstopped short answer at trunk and head.
- [ ] Probe. Mark Stop click and last `EventSource` message, run at trunk and at the head, interleaved. Both sides must produce the unstopped-answer time.
- [ ] Baseline. Record the trunk unstopped-answer time first.
- [ ] Rule. Head unstopped-answer time must stay within 20 percent of trunk. Stop to last chunk must finish within 2000 ms. Fail either number.

**Review gate.** The operator reviews before merge.

- [ ] Copy lane 2 screenshots into `.audit/media/kb-op-2-review-stop.png`.
- [ ] Record a 30 to 60 second video of the change on a lane VM. Save it as `.audit/media/kb-op-2-review.mp4`.
- [ ] Post the screenshots and the video in chat. Stop at merge-ready. Wait for the operator's click.

**Merge.**

- [ ] Root's clean verdict at the exact head SHA.
- [ ] Bugbot triage done.
- [ ] Rebased onto current trunk after the verdict, patch-id unchanged.
- [ ] The root appends kb-op-2 onto kb-op-1. The operator lands it.

## Persist manual tags without fake parse (kb-op-3)

**Depends on.** kb-op-2.

**Files.**

- [ ] Edit `src/core/contracts/knowledge.py`.
- [ ] Edit `src/core/knowledge/documents/service/knowledge_service.py`.
- [ ] Edit `src/core/knowledge/documents/create_manual.py`.
- [ ] Edit `src/core/knowledge/documents/documents_orchestrator.py`.
- [ ] Edit `src/core/knowledge/documents/factory.py`.
- [ ] Edit `src/web/api/knowledge/documents/router.py`.
- [ ] Edit `src/web/api/knowledge/documents/document_reads.py`.
- [ ] Edit `src/db/dao/knowledge_tag_repository.py` only if `set_knowledge_tags` needs a caller from `KnowledgeService`.
- [ ] Edit `frontend/src/api/knowledge-base/index.ts` and `frontend/src/components/manual-knowledge-editor.vue` only if the body field names still drift.
- [ ] Create `.agents/notes/implemented/bug-fix/2026-09-05-manual-tags-publish.md`.

**Build.**

- [ ] Accept `tag_ids` and `process_config` on create and on `PUT /knowledge/{id}`. Pass them through the create router, not only `tag_id`. Keep publish as draft or raise a typed `ValidationError`. Do not enqueue `manual_process`. That task still raises `NotImplementedError`. Do not stamp `parse_status` pending.

**You see.**

- [ ] Saving tags in the manual editor persists them.
- [ ] Publish does not leave a pending spinner. The row stays draft or the API returns a typed error.

**Verify, unit.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] `tests/core/knowledge/test_create_variants.py` and `tests/core/knowledge/test_doc_service_crud.py` gain tag and publish cases. Run `uv run pytest tests/core/knowledge/test_create_variants.py tests/core/knowledge/test_doc_service_crud.py tests/core/knowledge/test_arq_enqueue.py`.

**Verify, live.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked. Ten lanes on `grok-4.6-fast-xhigh` at the PR head, per the boot recipe.

- [ ] Lane 1. Regression lane against trunk. Open the manual editor, change title only, save. Gate tag persist on head plus the saved title the user waits for. Save `kb-op-3-regression.png`. Pass when title-only save still works.
- [ ] Lane 2. Set tags and save as draft. Save `kb-op-3-tags.png`. Pass when reload shows the same tag ids.
- [ ] Lane 3. Click publish. Save `kb-op-3-publish.png`. Pass when parse_status is not pending and no `manual_process` job is queued.
- [ ] Lane 4. Publish again after a refresh. Save `kb-op-3-publish-again.png`. Pass when the typed reject or draft state is stable.
- [ ] Lane 5. Put manual fields on a non-manual file. Save `kb-op-3-non-manual.png`. Pass when `knowledge.manual_fields_unsupported` or tags-only apply, never silent drop of file bytes.
- [ ] Lane 6. Clear tags and save. Save `kb-op-3-clear-tags.png`. Pass when the document has no tags.
- [ ] Lane 7. Create manual with `tag_ids` and `process_config`. Save `kb-op-3-create.png`. Pass when create keeps both.
- [ ] Lane 8. Open the document detail after a tagged save. Save `kb-op-3-detail.png`. Pass when spans or tags routes are 200.
- [ ] Lane 9. Search `@` for the tagged page. Save `kb-op-3-search.png`. Pass when the title is in the picker.
- [ ] Lane 10. Ask knowledge-chat about the tagged page. Save `kb-op-3-chat.png`. Pass when the turn still streams.

**Verify, perf.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] Metric. Time from draft save to 200, at trunk and head. Time from publish click to the typed response on head, plus the editor the user waits for.
- [ ] Probe. Mark save XHR start and response, run at trunk and at the head, interleaved. Both sides must produce the draft-save time.
- [ ] Baseline. Record the trunk draft-save time first.
- [ ] Rule. Head draft-save must stay within 20 percent of trunk. Publish to typed response must finish within 2000 ms. Fail either number.

**Review gate.** The operator reviews before merge.

- [ ] Copy lane 2 and lane 3 screenshots into `.audit/media/kb-op-3-review-tags.png` and `.audit/media/kb-op-3-review-publish.png`.
- [ ] Record a 30 to 60 second video of the change on a lane VM. Save it as `.audit/media/kb-op-3-review.mp4`.
- [ ] Post the screenshots and the video in chat. Stop at merge-ready. Wait for the operator's click.

**Merge.**

- [ ] Root's clean verdict at the exact head SHA.
- [ ] Bugbot triage done.
- [ ] Rebased onto current trunk after the verdict, patch-id unchanged.
- [ ] The root appends kb-op-3 onto kb-op-2. The operator lands it.

## Store file-url document bytes (kb-op-4)

**Depends on.** kb-op-3.

**Files.**

- [ ] Edit `src/core/knowledge/documents/process_document.py`.
- [ ] Edit `src/core/knowledge/documents/process_runtime.py`.
- [ ] Edit `src/core/knowledge/documents/factory.py`.
- [ ] Edit `src/workers/tasks/document_process.py`.
- [ ] Edit `src/web/deps/knowledge_documents.py` only to reuse `_BackendFileServiceResolver` in the worker.
- [ ] Create `.agents/notes/implemented/bug-fix/2026-09-05-url-store-bytes.md`.

**Build.**

- [ ] When the row is `file_url` and `file_path` is empty, download, call `FileService.save_bytes`, and persist `file_path` on the row. Leave ordinary `url` rows without bytes.

**You see.**

- [ ] A `file_url` document previews and downloads after the worker runs.
- [ ] The Sina web `url` row still returns `knowledge.file_unavailable`.

**Verify, unit.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] `tests/core/knowledge/test_create_variants.py` and `tests/workers/test_document_process.py` gain a SaveBytes case. Run `uv run pytest tests/core/knowledge/test_create_variants.py tests/workers/test_document_process.py tests/web/test_document_reads.py`.

**Verify, live.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked. Ten lanes on `grok-4.6-fast-xhigh` at the PR head, per the boot recipe.

- [ ] Lane 1. Regression lane against trunk. Open the existing Sina web `url` doc. Both sides return `knowledge.file_unavailable`. Gate `file_url` preview bytes on head plus the detail page the user waits for. Save `kb-op-4-regression.png`. Pass when the Sina detail page still opens on both.
- [ ] Lane 2. Import a `file_url` document and wait for the worker. Save `kb-op-4-file-url.png`. Pass when `file_path` is set.
- [ ] Lane 3. Preview that `file_url` doc. Save `kb-op-4-preview.png`. Pass when the preview returns bytes.
- [ ] Lane 4. Download that `file_url` doc. Save `kb-op-4-download.png`. Pass when the attachment has a non-empty body.
- [ ] Lane 5. Preview the Sina web `url` doc. Save `kb-op-4-sina.png`. Pass when the code is still `knowledge.file_unavailable`.
- [ ] Lane 6. Preview a manual doc. Save `kb-op-4-manual.png`. Pass when markdown still streams.
- [ ] Lane 7. Preview a stored file doc. Save `kb-op-4-file.png`. Pass when the file still streams.
- [ ] Lane 8. Import a `file_url` that fails to fetch. Save `kb-op-4-bad-url.png`. Pass when the error is typed, not a silent pending.
- [ ] Lane 9. Contributor download, Viewer preview. Save `kb-op-4-roles.png`. Pass when those role gates still hold.
- [ ] Lane 10. Chat about the `file_url` doc after bytes land. Save `kb-op-4-chat.png`. Pass when the answer cites that page.

**Verify, perf.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] Metric. Time from preview click to first byte, at head. Time to open the document detail at trunk and head.
- [ ] Probe. Mark preview request and first response byte, run at trunk and at the head, interleaved. Both sides must produce the detail-open time.
- [ ] Baseline. Record the trunk detail-open time first.
- [ ] Rule. Head detail-open must stay within 20 percent of trunk. Preview first byte must finish within 3000 ms. Fail either number.

**Review gate.** The operator reviews before merge.

- [ ] Copy lane 2 screenshots into `.audit/media/kb-op-4-review-preview.png`.
- [ ] Record a 30 to 60 second video of the change on a lane VM. Save it as `.audit/media/kb-op-4-review.mp4`.
- [ ] Post the screenshots and the video in chat. Stop at merge-ready. Wait for the operator's click.

**Merge.**

- [ ] Root's clean verdict at the exact head SHA.
- [ ] Bugbot triage done.
- [ ] Rebased onto current trunk after the verdict, patch-id unchanged.
- [ ] The root appends kb-op-4 onto kb-op-3. The operator lands it.

## Wire tenant members and invite admin (kb-op-5)

**Depends on.** kb-op-4.

**Files.**

- [ ] Edit `src/web/api/tenants/router.py`.
- [ ] Edit `src/core/tenants/factory.py` only as the existing `build_tenant_member_service` and `build_tenant_invitation_service` callers.
- [ ] Edit `src/core/tenants/invitation_service.py` only if list-for-tenant envelopes still need a view.
- [ ] Edit `frontend/src/api/tenant/members.ts` and `frontend/src/api/tenant/invitations.ts` only if envelopes drift.
- [ ] Edit `frontend/src/views` callers in `TenantMembers.vue` and `TenantInfo.vue` only if leave or role envelopes drift.
- [ ] Create `.agents/notes/implemented/bug-fix/2026-09-05-tenant-members.md`.

**Build.**

- [ ] Add `/tenants/{id}/members`, `/tenants/{id}/leave`, `/tenants/{id}/invitations`, and `/tenants/{id}/invite-links` on `TenantMemberService` and `TenantInvitationService`. Wrap leave as `remove_member` for the caller. Project email and name if the table join already has them. Do not remap org members.

**You see.**

- [ ] The workspace members page lists, invites, accepts, revokes, and changes roles.

**Verify, unit.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] Extend `tests/core/tenants/test_invitation_service.py` and add `tests/web/test_tenant_members.py`. Run `uv run pytest tests/core/tenants/test_invitation_service.py tests/web/test_tenant_members.py tests/web/test_me_invitations.py`.

**Verify, live.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked. Ten lanes on `grok-4.6-fast-xhigh` at the PR head, per the boot recipe.

- [ ] Lane 1. Regression lane against trunk. Open workspace members. Trunk 404s. Gate list 200 on head plus the members table the user waits for. Save `kb-op-5-regression.png`. Pass when `/me/invitations` still works on both.
- [ ] Lane 2. List members. Save `kb-op-5-list.png`. Pass when the current user is in the table.
- [ ] Lane 3. Create an invitation. Save `kb-op-5-invite.png`. Pass when the invitee inbox shows it.
- [ ] Lane 4. Accept from `/me/invitations`. Save `kb-op-5-accept.png`. Pass when membership appears.
- [ ] Lane 5. Decline a second invite. Save `kb-op-5-decline.png`. Pass when status is declined.
- [ ] Lane 6. Revoke a pending invite. Save `kb-op-5-revoke.png`. Pass when the admin list drops it.
- [ ] Lane 7. Create an invite-link. Save `kb-op-5-link.png`. Pass when POST is 200 and a URL returns.
- [ ] Lane 8. Change a member role. Save `kb-op-5-role.png`. Pass when reload shows the new role.
- [ ] Lane 9. Leave the workspace from `TenantInfo.vue`. Save `kb-op-5-leave.png`. Pass when `POST /tenants/{id}/leave` is 200 or a typed 4xx for the last owner.
- [ ] Lane 10. Inbox badge count after accept. Save `kb-op-5-badge.png`. Pass when pending-count drops.

**Verify, perf.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] Metric. Time from members page open to first row, at head. Time to open `/me` pending-count at trunk and head.
- [ ] Probe. Mark the members XHR, run at trunk and at the head, interleaved. Both sides must produce the pending-count time.
- [ ] Baseline. Record the trunk pending-count time first.
- [ ] Rule. Head pending-count must stay within 20 percent of trunk. Members first row must finish within 2000 ms. Fail either number.

**Review gate.** The operator reviews before merge.

- [ ] Copy lane 2 and lane 3 screenshots into `.audit/media/kb-op-5-review-list.png` and `.audit/media/kb-op-5-review-invite.png`.
- [ ] Record a 30 to 60 second video of the change on a lane VM. Save it as `.audit/media/kb-op-5-review.mp4`.
- [ ] Post the screenshots and the video in chat. Stop at merge-ready. Wait for the operator's click.

**Merge.**

- [ ] Root's clean verdict at the exact head SHA.
- [ ] Bugbot triage done.
- [ ] Rebased onto current trunk after the verdict, patch-id unchanged.
- [ ] The root appends kb-op-5 onto kb-op-4. The operator lands it.

## Persist session attachments (kb-op-6)

**Depends on.** kb-op-5.

**Files.**

- [ ] Edit `src/web/api/chat/sessions/router.py`.
- [ ] Edit `src/core/knowledge/documents/temporary_document.py`.
- [ ] Edit `src/core/knowledge/documents/factory.py` to add `build_temporary_document_service`.
- [ ] Edit `src/web/deps/knowledge_documents.py` or a one-line chat dep that forwards that factory.
- [ ] Edit `src/workers/tasks/temporary_document.py` only if parse-after-upload already needs a caller.
- [ ] Create `.agents/notes/implemented/bug-fix/2026-09-05-session-attachments.md`.

**Build.**

- [ ] Expose `POST`, `GET`, and `DELETE /sessions/{id}/attachments` plus preview on `TemporaryDocumentService`. Do not inject attachment ids into `KnowledgeQARunner`. That bind is a later how.

**You see.**

- [ ] The composer paperclip uploads, previews, and deletes without 404.

**Verify, unit.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] Add `tests/web/test_session_attachments.py`. Run `uv run pytest tests/web/test_session_attachments.py tests/web/test_session_message_views.py tests/integration/web/test_routers.py`.

**Verify, live.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked. Ten lanes on `grok-4.6-fast-xhigh` at the PR head, per the boot recipe.

- [ ] Lane 1. Regression lane against trunk. Open composer paperclip. Trunk 404s. Gate upload 200 on head plus the composer the user waits for. Save `kb-op-6-regression.png`. Pass when the composer still sends text chat on both.
- [ ] Lane 2. Upload a small png. Save `kb-op-6-upload.png`. Pass when the list shows the file.
- [ ] Lane 3. Preview that file. Save `kb-op-6-preview.png`. Pass when preview returns bytes.
- [ ] Lane 4. Delete that file. Save `kb-op-6-delete.png`. Pass when the list is empty.
- [ ] Lane 5. Upload then send a question. Save `kb-op-6-chat.png`. Pass when the turn completes. The model does not have to read the file.
- [ ] Lane 6. Upload on a session you do not own. Save `kb-op-6-other-session.png`. Pass when the status is 404.
- [ ] Lane 7. Upload over the size cap. Save `kb-op-6-too-big.png`. Pass when the error is typed 4xx.
- [ ] Lane 8. Refresh after upload. Save `kb-op-6-refresh.png`. Pass when the attachment is still listed.
- [ ] Lane 9. Stop during a turn that has an attachment. Save `kb-op-6-stop.png`. Pass when stop still cancels.
- [ ] Lane 10. Viewer cannot upload, Contributor can. Save `kb-op-6-roles.png`. Pass when those gates hold.

**Verify, perf.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] Metric. Time from upload submit to list refresh, at head. Time to send a text-only chat at trunk and head.
- [ ] Probe. Mark upload XHR, run at trunk and at the head, interleaved. Both sides must produce the text-chat time.
- [ ] Baseline. Record the trunk text-chat time first.
- [ ] Rule. Head text-chat time must stay within 20 percent of trunk. Upload to list refresh must finish within 3000 ms. Fail either number.

**Review gate.** The operator reviews before merge.

- [ ] Copy lane 2 screenshots into `.audit/media/kb-op-6-review-upload.png`.
- [ ] Record a 30 to 60 second video of the change on a lane VM. Save it as `.audit/media/kb-op-6-review.mp4`.
- [ ] Post the screenshots and the video in chat. Stop at merge-ready. Wait for the operator's click.

**Merge.**

- [ ] Root's clean verdict at the exact head SHA.
- [ ] Bugbot triage done.
- [ ] Rebased onto current trunk after the verdict, patch-id unchanged.
- [ ] The root appends kb-op-6 onto kb-op-5. The operator lands it.

## Close FAQ batch and search holes (kb-op-7)

**Depends on.** kb-op-6.

**Files.**

- [ ] Edit `src/web/api/knowledge/faq/router.py`.
- [ ] Edit `src/core/knowledge/faq/import_runner.py` only if JSON upsert cannot reuse `FAQImportRunner`.
- [ ] Edit `src/core/knowledge/faq/service/faq_service.py`.
- [ ] Edit `frontend/src/api/knowledge-base/index.ts` only if the JSON upsert envelope still drifts.
- [ ] Create `.agents/notes/implemented/bug-fix/2026-09-05-faq-batch-search.md`.

**Build.**

- [ ] Add FAQ field update, tag update, search, last-result display, and a JSON `{entries, mode}` upsert next to the existing multipart `POST /entries`. Keep CSV and Excel on `FAQImportRunner`. Do not treat file upload as the SPA JSON POST.

**You see.**

- [ ] FAQ batch tag, field edit, search, and JSON import return 200 and change rows.

**Verify, unit.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] Extend `tests/integration/web/api/knowledge/faq/test_controller.py`. Run `uv run pytest tests/integration/web/api/knowledge/faq/test_controller.py`.

**Verify, live.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked. Ten lanes on `grok-4.6-fast-xhigh` at the PR head, per the boot recipe.

- [ ] Lane 1. Regression lane against trunk. Open FAQ list. Gate JSON import and search on head plus the list the user waits for. Save `kb-op-7-regression.png`. Pass when FAQ list still loads on both.
- [ ] Lane 2. Search FAQ entries. Save `kb-op-7-search.png`. Pass when `POST .../faq/search` is 200.
- [ ] Lane 3. Batch update tags. Save `kb-op-7-tags.png`. Pass when selected rows show the new tags.
- [ ] Lane 4. Batch update fields. Save `kb-op-7-fields.png`. Pass when `PUT .../faq/entries/fields` is 200.
- [ ] Lane 5. JSON batch upsert from `FAQEntryManager.vue`. Save `kb-op-7-json.png`. Pass when new entries appear.
- [ ] Lane 6. CSV import still works. Save `kb-op-7-csv.png`. Pass when multipart `POST /entries` still imports.
- [ ] Lane 7. Toggle last-result display. Save `kb-op-7-display.png`. Pass when `PUT .../import/last-result/display` is 200.
- [ ] Lane 8. Import progress GET. Save `kb-op-7-progress.png`. Pass when progress is 200.
- [ ] Lane 9. Knowledge-chat after FAQ import. Save `kb-op-7-chat.png`. Pass when chat still streams.
- [ ] Lane 10. Viewer cannot mutate FAQ. Save `kb-op-7-viewer.png`. Pass when Viewer gets 403 on PUT.

**Verify, perf.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] Metric. Time from FAQ search submit to rows, at head. Time to open FAQ list at trunk and head.
- [ ] Probe. Mark the search XHR, run at trunk and at the head, interleaved. Both sides must produce the FAQ list time.
- [ ] Baseline. Record the trunk FAQ list time first.
- [ ] Rule. Head FAQ list time must stay within 20 percent of trunk. Search must finish within 2000 ms. Fail either number.

**Review gate.** The operator reviews before merge.

- [ ] Copy lane 2 and lane 5 screenshots into `.audit/media/kb-op-7-review-search.png` and `.audit/media/kb-op-7-review-json.png`.
- [ ] Record a 30 to 60 second video of the change on a lane VM. Save it as `.audit/media/kb-op-7-review.mp4`.
- [ ] Post the screenshots and the video in chat. Stop at merge-ready. Wait for the operator's click.

**Merge.**

- [ ] Root's clean verdict at the exact head SHA.
- [ ] Bugbot triage done.
- [ ] Rebased onto current trunk after the verdict, patch-id unchanged.
- [ ] The root appends kb-op-7 onto kb-op-6. The operator lands it.

## Wire KB shares and org lists (kb-op-8)

**Depends on.** kb-op-7.

**Files.**

- [ ] Create KB share routes next to `src/web/api/knowledge_bases/router.py` or a sibling router under that package.
- [ ] Edit `src/core/sharing/kb_share_service.py` and `src/core/sharing/factory.py`. List and `can_access_knowledge_base` already run. Mutations still raise `NotImplementedError`.
- [ ] Edit `src/web/api/organizations` only for `GET /organizations/{id}/shared-knowledge-bases` and `GET /organizations/{id}/shares`.
- [ ] Create `.agents/notes/implemented/bug-fix/2026-09-05-kb-shares.md`.

**Build.**

- [ ] Add `POST`, `GET`, `PUT`, and `DELETE /knowledge-bases/{id}/shares`. Add the org-scoped KB share lists. Run how on org permission caps before the mutation methods. Leave agent-share org twins for a later program.

**You see.**

- [ ] Share dialog creates a share. Settings list and revoke it. Org picker shows shared KBs.

**Verify, unit.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] Add `tests/web/test_kb_shares.py`. Run `uv run pytest tests/web/test_kb_shares.py tests/integration/web/api/knowledge_bases/test_controller.py tests/integration/web/test_routers.py`.

**Verify, live.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked. Ten lanes on `grok-4.6-fast-xhigh` at the PR head, per the boot recipe.

- [ ] Lane 1. Regression lane against trunk. Open the share dialog. Trunk 404s. Gate list 200 on head plus the dialog the user waits for. Save `kb-op-8-regression.png`. Pass when the KB list still loads on both.
- [ ] Lane 2. Create a share. Save `kb-op-8-create.png`. Pass when `POST .../shares` is 200.
- [ ] Lane 3. List shares on the KB. Save `kb-op-8-list.png`. Pass when the new share is in the table.
- [ ] Lane 4. Update a share. Save `kb-op-8-update.png`. Pass when `PUT` is 200.
- [ ] Lane 5. Delete a share. Save `kb-op-8-delete.png`. Pass when the table drops it.
- [ ] Lane 6. Open org shared-knowledge-bases. Save `kb-op-8-org-list.png`. Pass when `GET /organizations/{id}/shared-knowledge-bases` is 200.
- [ ] Lane 7. Open org shares settings. Save `kb-op-8-org-shares.png`. Pass when `GET /organizations/{id}/shares` is 200.
- [ ] Lane 8. Pin still works after share. Save `kb-op-8-pin.png`. Pass when pin toggles.
- [ ] Lane 9. Viewer cannot create a share. Save `kb-op-8-viewer.png`. Pass when Viewer gets 403.
- [ ] Lane 10. Knowledge-chat after a share. Save `kb-op-8-chat.png`. Pass when chat still streams.

**Verify, perf.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] Metric. Time from share-dialog open to list, at head. Time to open the KB list at trunk and head.
- [ ] Probe. Mark the shares XHR, run at trunk and at the head, interleaved. Both sides must produce the KB list time.
- [ ] Baseline. Record the trunk KB list time first.
- [ ] Rule. Head KB list time must stay within 20 percent of trunk. Share list must finish within 2000 ms. Fail either number.

**Review gate.** The operator reviews before merge.

- [ ] Copy lane 2 screenshots into `.audit/media/kb-op-8-review-create.png`.
- [ ] Record a 30 to 60 second video of the change on a lane VM. Save it as `.audit/media/kb-op-8-review.mp4`.
- [ ] Post the screenshots and the video in chat. Stop at merge-ready. Wait for the operator's click.

**Merge.**

- [ ] Root's clean verdict at the exact head SHA.
- [ ] Bugbot triage done.
- [ ] Rebased onto current trunk after the verdict, patch-id unchanged.
- [ ] The root appends kb-op-8 onto kb-op-7. The operator lands it.

## Wire wiki issues and revisions (kb-op-9)

**Depends on.** kb-op-8.

**Files.**

- [ ] Edit `src/web/api/knowledge/wiki/router.py`.
- [ ] Edit `src/core/knowledge/wiki/issues.py` and add the missing `WikiPageIssueRepository` DAO.
- [ ] Edit `src/core/knowledge/wiki/factory.py` to build issue and revision services.
- [ ] Edit `src/db/models/wiki_page.py` only as the `WikiPageRevision` row shape. Do not invent a second table.
- [ ] Create `.agents/notes/implemented/bug-fix/2026-09-05-wiki-issues-revisions.md`.

**Build.**

- [ ] Add issues list and status first on the existing `issues.py` functions. Then run how on revision snapshot, prune, and revert before writing those routes. `WikiPageService` only bumps a version counter today. Keep `wiki.kb_wiki_not_enabled` when wiki is off. Do not treat lint or auto-fix as this feature.

**You see.**

- [ ] With wiki on, the revision drawer and issues panel stop 404ing.

**Verify, unit.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] Extend `tests/integration/web/api/knowledge/wiki/test_controller.py`. Run `uv run pytest tests/integration/web/api/knowledge/wiki/test_controller.py`.

**Verify, live.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked. Ten lanes on `grok-4.6-fast-xhigh` at the PR head, per the boot recipe.

- [ ] Lane 1. Regression lane against trunk. Open wiki with wiki off. Both sides keep `wiki.kb_wiki_not_enabled`. Save `kb-op-9-regression.png`. Pass when the 422 code is unchanged.
- [ ] Lane 2. Turn wiki on and open a page. Save `kb-op-9-page.png`. Pass when the existing page route is still 200.
- [ ] Lane 3. Open the revision drawer. Save `kb-op-9-revisions.png`. Pass when `GET .../revisions/{slug}` is 200.
- [ ] Lane 4. Revert if a prior revision exists. Save `kb-op-9-revert.png`. Pass when revert is 200 or the drawer has no older revision.
- [ ] Lane 5. Open issues. Save `kb-op-9-issues.png`. Pass when `GET .../wiki/issues` is 200.
- [ ] Lane 6. Change one issue status. Save `kb-op-9-issue-status.png`. Pass when PUT is 200.
- [ ] Lane 7. Wiki search still works. Save `kb-op-9-search.png`. Pass when search is 200.
- [ ] Lane 8. FAQ list after wiki work. Save `kb-op-9-faq.png`. Pass when FAQ list is 200.
- [ ] Lane 9. Knowledge-chat after wiki on. Save `kb-op-9-chat.png`. Pass when chat still streams.
- [ ] Lane 10. Turn wiki off again. Save `kb-op-9-off.png`. Pass when the 422 code returns.

**Verify, perf.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] Metric. Time to open the issues panel, at head. Time to open the KB with wiki off at trunk and head.
- [ ] Probe. Mark the issues XHR, run at trunk and at the head, interleaved. Both sides must produce the wiki-off KB open time.
- [ ] Baseline. Record the trunk wiki-off KB open time first.
- [ ] Rule. Head wiki-off KB open must stay within 20 percent of trunk. Issues panel must finish within 2000 ms. Fail either number.

**Review gate.** The operator reviews before merge.

- [ ] Copy lane 3 and lane 5 screenshots into `.audit/media/kb-op-9-review-revisions.png` and `.audit/media/kb-op-9-review-issues.png`.
- [ ] Record a 30 to 60 second video of the change on a lane VM. Save it as `.audit/media/kb-op-9-review.mp4`.
- [ ] Post the screenshots and the video in chat. Stop at merge-ready. Wait for the operator's click.

**Merge.**

- [ ] Root's clean verdict at the exact head SHA.
- [ ] Bugbot triage done.
- [ ] Rebased onto current trunk after the verdict, patch-id unchanged.
- [ ] The root appends kb-op-9 onto kb-op-8. The operator lands it.

## Close the program

- [ ] Every box above is checked with its evidence.
- [ ] Reply to the operator with the report the execution playbook names.

## Appendix A. Prototype evidence

No new prototype branch. The leftover operator loop already answered the first 404 questions on the live SPA. Evidence sits in `.audit/e2e-operator-loop.tsv` rows iter1 through iter8. There is no commit SHA yet because that wave is still uncommitted.

[Split uncommitted clusters](a3ac05b7-a55a-4c6a-9ffb-2f169502089f) proved the leftover wave must land as one PR. Shared hubs are `src/core/chat/factory.py`, `document_reads.py`, `knowledge_service.py`, and `kb_service.py`.

[Map remaining seams](fa25e009-1e6a-453f-852b-2bd433855421) proved stop needs one APP-scope `MemoryStreamManager`, publish must not enqueue `manual_process`, and SaveBytes belongs only on `file_url`.

[Inventory SPA 404s](71c9c261-ebaa-42eb-b9b5-901aba4fdb9a) proved hybrid-search has no SPA client. The next operator 404s are members, attachments, FAQ batch, KB shares, and wiki issues or revisions.

[Park later capabilities](d3ccb204-5bba-4e77-b9a2-34d03d370735) proved which later seams already have a service and which still need a how. Attachments reuse `TemporaryDocumentService`. Share list already runs. Share mutations and wiki revisions do not.

Still unproven, so each later PR must prove it on the live SPA.

- Whether a shared `MemoryStreamManager` cancels tokens across two HTTP requests.
- Whether a `file_url` worker write fills `file_path` and preview streams it.
- Whether JSON FAQ upsert can sit next to multipart `POST /entries` without breaking CSV.

## Appendix B. Alternatives rejected

Wiring HTTP hybrid-search in this program. Rejected. No file under `frontend/src/api` calls it. The command palette already uses `POST /knowledge-search`.

Enqueue `manual_process` on publish. Rejected. `src/workers/tasks/manual_process.py` still raises `NotImplementedError`. A pending spinner would lie.

SaveBytes on ordinary web `url` rows. Rejected. Only the file-URL worker path persists bytes. The Sina `url` row stays `knowledge.file_unavailable`.

A four-PR split of the leftover wave. Rejected. The hubs already merged the code. [Split uncommitted clusters](a3ac05b7-a55a-4c6a-9ffb-2f169502089f) said one land is cheaper.

ReAct agent-chat in this stack. Rejected. Tool-approval cards do not show until a tool loop exists. Knowledge-chat already answers.

Injecting attachments into QA in kb-op-6. Rejected. Upload and preview close the 404. Binding ids into a turn needs a how.

Treating wiki lint as page issues. Rejected. Lint and auto-fix already ship. Issues and revisions are a different table.

Autopilot-full. Rejected. The work is sequenced. Merge authority stays with the operator.

Orchestrate. Rejected. Nine stacked PRs fit one program owner.

## Appendix C. Risks

kb-op-1 is a large uncommitted tree. The owner watches `make openapi` drift and layer gates. Do not add features while landing it. Do not stage the ergonomics plan or kb-verify-flow evidence.

kb-op-2 can register a request-scoped object on `LifeSpanService` by mistake. The owner watches `scripts/check_service_singleton.py`. `StreamManager` is not a `*Service`. Miss `embed_channels.py` and Stop only works on the logged-in composer.

kb-op-3 can stamp pending after a failed enqueue. The owner watches that publish never writes pending.

kb-op-4 can persist a path only on the in-memory payload. The owner writes `file_path` on the row.

Parked, not in these nine PRs. After this stack, the small honesty leftovers can start without a design. Web-search test swaps `_NoopClientRegistry` for the lifespan registry. Session title uses `build_session_service_with_title`. Suggestion events need a new table. The rest need a how first.

- HTTP hybrid-search empty `[]`. One engine versus three doors.
- ReAct agent-chat and in-chat `/agent/tool-approvals`. MCP settings approvals already ship.
- Continue-stream. Same process-local manager hole as stop.
- Attachment ids inside knowledge-chat.
- Settings credentials PUT on models, MCP, web-search, and datasource.
- KB activity, rebuild-index, and tenant audit-log.
- Org-scoped shared-agents lists.
- Web-search test button.
- Graph extract stubs.
- Suggestion analytics persist.
- Session title generation. `KnowledgeBase.vue` imports it and never calls it.
- `ai/retrieval` `Any` cleanup.
- The agent-ergonomics local merge series in `.agents/docs/ai-native-ergonomics-plan.md`.

## Appendix D. Links and reading list

Read `AGENTS.md`, `frontend/AGENTS.md`, `.agents/feature-map/ui.md`, and `.claude/skills/kb-verify-flow/SKILL.md` before editing.

kb-op-2 runs `pstack/skills/how/SKILL.md` before build. kb-op-8 runs it on org permission caps before share mutations. kb-op-9 runs it on revision snapshot, prune, and revert before those routes. kb-op-8 also runs `pstack/skills/interrogate/SKILL.md` if the org cap fork is contested.

The trail is `.audit/e2e-operator-loop.tsv` per `pstack/skills/show-me-your-work/SKILL.md`. Append one row per PR verdict.

Control skill is `cursor-team-kit` `control-ui`.
Forge is `gh`. Trunk is `origin/master`.
