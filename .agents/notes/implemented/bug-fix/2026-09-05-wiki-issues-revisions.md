# Agent Note: Wiki page issues and revision history

Status: implemented
Date: 2026-09-05
Scope: wire wiki issues and page revision history on the existing tables
Related files: src/core/knowledge/wiki/issues.py, src/core/knowledge/wiki/page_service.py, src/core/knowledge/wiki/revisions.py, src/db/dao/wiki_page_issue_repository.py, src/db/dao/wiki_page_revision_repository.py, src/web/api/knowledge/wiki/history_router.py, tests/integration/web/api/knowledge/wiki/test_controller.py

## Context

The revision drawer and issues panel already called HTTP that the wiki router did not declare. `wiki_page_issues` and `wiki_page_revisions` already exist. The page service already bumped `version` on a user-visible edit. The issues protocol already used `pending` / `ignored` / `resolved`.

## Decision

Issues persist through `WikiPageIssueRepository` and `WikiPageIssueStore`. HTTP is `GET /knowledgebase/{kb_id}/wiki/issues` (Viewer+, `data` is the list) and `PUT .../issues/{issue_id}/status` (Admin+). Status values are the protocol vocabulary, including `ignored`. Lint and auto-fix do not create issues. `pending_issue_count` feeds `get_stats` so the badge is not a lying 0 once rows exist.

A user-visible edit snapshots the pre-edit page as `(page_id, old version)` before the rewrite. Unique `(page_id, version)` makes a retry a no-op. Bookkeeping writes do not snapshot. The table is append-only. There is no prune job. List is `GET .../wiki/revisions/{slug}` newest first, `content` omitted, envelope `{revisions, total, current_version}`. `?version=N` returns one snapshot with content. `POST .../wiki/revert` `{slug, version}` snapshots the current page, copies the stored snapshot onto the page, bumps version, and sets `last_edit_source` to `revert`. Missing version is 404. Wiki-off stays `wiki.kb_wiki_not_enabled`.

## Alternatives considered

- **Prune job / soft or hard caps** — rejected: the table is append-only. Bound the list with `limit` / `offset`.
- **Treat lint findings as issues** — rejected: lint and auto-fix stay their own routes. Issues are review flags, not lint output.
- **Force-delete the current row on revert** — rejected: a revert is an edit. The pre-revert page is snapshotted so the revert is itself revertable.
- **A second revision table** — rejected: `wiki_page_revisions` already holds snapshots. Current version stays on `wiki_pages`.

## Consequences

The issues panel can list and ignore rows. The revision drawer can list snapshots, fetch one body, and revert. Ingest-task counts in stats stay 0. Historical pages from before this wire have no snapshots until the next user-visible edit.

## Required verification

- `uv run pytest tests/integration/web/api/knowledge/wiki/test_controller.py tests/unit/knowledge/wiki/test_wiki_types.py tests/integration/web/test_routers.py tests/core/knowledge/test_wiki_crud.py`
- `make openapi` and `make check-endpoint`
- `uv run python scripts/check_layer_violation.py --domains auth,tenant,infra,knowledge,chat`
- `python scripts/verify_agent_notes.py --repo-root .`
