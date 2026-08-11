#!/usr/bin/env bash
# Roll back to a previous release tag.
# Usage: ./deploy/scripts/rollback.sh <previous-release-tag>
set -euo pipefail

log() { printf '[rollback] %s\n' "$*"; }
fail() { printf '[rollback] ERROR: %s\n' "$*" >&2; exit 1; }

PREV_TAG="${1:-}"
if [ -z "$PREV_TAG" ]; then
  fail "usage: $0 <previous-release-tag>"
fi

if ! git rev-parse "$PREV_TAG" >/dev/null 2>&1; then
  fail "tag not found: $PREV_TAG"
fi

# ── Checkout previous release ───────────────────────────────────────────
log "checking out $PREV_TAG"
git checkout "$PREV_TAG"

# ── Reinstall dependencies ──────────────────────────────────────────────
log "syncing dependencies"
uv sync

# ── Roll back migrations to the previous release's head ────────────────
# The previous tag carries its own alembic state; downgrade to it.
log "rolling back database migrations"
uv run alembic downgrade -1

log "rollback to $PREV_TAG complete"
log "NOTE: verify the API and worker pods are restarted with the previous image"
