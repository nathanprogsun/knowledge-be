# Agent Note: Tenant members leave invitations and invite-links

Status: implemented
Date: 2026-09-05
Scope: HTTP for workspace members, self-leave, invitations, and share links
Related files: src/web/api/tenants/router.py, src/web/api/tenants/views.py, tests/web/test_tenant_members.py, docs/api/tenants.md

## Context

The members settings page and workspace info page already call
`/tenants/{id}/members`, `/leave`, `/invitations`, and `/invite-links`.
Those routes were missing on the tenants router. Membership and
invitation services already enforced uniqueness, last-owner, and the
inbox accept path. The hole was HTTP only.

## Decision

Wire the existing `TenantMemberService` and `TenantInvitationService`
on the tenants router. Leave is `remove_member` for the caller, so the
last-owner conflict stays one typed error. Add and invite resolve email
through `AuthService.get_user_row_by_email` and 404 when the address is
not a registered user. Member list fields `email`, `username`, and
`avatar` stay empty unless the caller already has them. The member
search join filters on `users` but still selects `tenant_members.*`, so
the list does not invent a second user projection. Share-link create
returns a host-relative `/register?token=...` URL. Revoke hides rows
from another workspace as not found.

## Alternatives considered

- **Alias organization member routes** — rejected: org members are
  tenant-to-org links, not workspace user memberships, and the page
  already speaks the tenant envelope.
- **Hydrate every list row through `AuthService.get_user_by_id`** —
  rejected: the list join does not return user columns, and an N+1
  lookup is a new fetch path the services do not own.
- **Return invite URLs on the tenant invitation list** — rejected: the
  invitation DTO drops the share-link token, and re-emitting it would
  need a service change the list envelope does not require.
- **Silent no-op when the last owner leaves** — rejected: the service
  already raises `tenant_member.last_owner`, and the page treats 409 as
  that case.

## Consequences

The members page can list, invite, change role, remove, and copy a
share link. Leave is a typed last-owner conflict. An invitee created
here appears on `/me/invitations` without a second inbox. List rows
show an em-dash when email and name were not on the membership DTO.
Share-link copy after a later list refresh has no URL until create
returns one again.

## Required verification

- `uv run pytest tests/core/tenants/test_invitation_service.py tests/web/test_tenant_members.py tests/web/test_me_invitations.py`
- `make openapi && make check-endpoint`
- `python scripts/verify_agent_notes.py --repo-root .`
- `uv run python scripts/check_layer_violation.py --domains auth,tenant,infra,knowledge,chat`
- `uv run python scripts/check_mypy_baseline.py`
