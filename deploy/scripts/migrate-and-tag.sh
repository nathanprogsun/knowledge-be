#!/usr/bin/env bash
# Run database migrations and tag the release.
# Usage: ./deploy/scripts/migrate-and-tag.sh <release-tag>
set -euo pipefail

log() { printf '[migrate-and-tag] %s\n' "$*"; }
fail() { printf '[migrate-and-tag] ERROR: %s\n' "$*" >&2; exit 1; }

RELEASE_TAG="${1:-}"
if [ -z "$RELEASE_TAG" ]; then
  fail "usage: $0 <release-tag>"
fi

# ── Load env ────────────────────────────────────────────────────────────
if [ -f deploy/env/api.env.example ]; then
  set -a
  # shellcheck disable=SC1091
  source deploy/env/api.env.example
  set +a
fi

# ── Run migrations ──────────────────────────────────────────────────────
log "running alembic migrations"
uv run alembic upgrade head

# ── Verify single head ──────────────────────────────────────────────────
HEADS=$(uv run alembic heads | wc -l | tr -d ' ')
if [ "$HEADS" -ne 1 ]; then
  fail "expected 1 alembic head, found $HEADS"
fi
log "alembic single head confirmed"

# ── Tag the release ─────────────────────────────────────────────────────
if git rev-parse "$RELEASE_TAG" >/dev/null 2>&1; then
  log "tag $RELEASE_TAG already exists — skipping"
else
  git tag -a "$RELEASE_TAG" -m "release $RELEASE_TAG"
  log "tagged $RELEASE_TAG"
fi

log "migrations applied and release tagged: $RELEASE_TAG"
