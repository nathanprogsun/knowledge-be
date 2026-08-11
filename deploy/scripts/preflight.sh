#!/usr/bin/env bash
# Preflight checks before a release cutover.
# Verifies the runtime prerequisites are present and reachable.
set -euo pipefail

log() { printf '[preflight] %s\n' "$*"; }
fail() { printf '[preflight] ERROR: %s\n' "$*" >&2; exit 1; }

# ── Runtime binaries ───────────────────────────────────────────────────
for bin in python3 uv docker psql redis-cli; do
  if ! command -v "$bin" >/dev/null 2>&1; then
    fail "missing required binary: $bin"
  fi
  log "found $bin: $(command -v "$bin")"
done

# ── Python version ─────────────────────────────────────────────────────
PY_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
PY_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 11 ]; }; then
  fail "python3 >= 3.11 required (found $PY_MAJOR.$PY_MINOR)"
fi
log "python3 $PY_MAJOR.$PY_MINOR OK"

# ── Environment file ────────────────────────────────────────────────────
if [ -z "${ENV_FILE:-}" ]; then
  ENV_FILE="deploy/env/api.env.example"
fi
if [ ! -f "$ENV_FILE" ]; then
  fail "env file not found: $ENV_FILE (set ENV_FILE)"
fi
log "env file: $ENV_FILE"

# ── Database reachability ───────────────────────────────────────────────
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
if ! pg_isready -h "$DB_HOST" -p "$DB_PORT" >/dev/null 2>&1; then
  fail "postgres not reachable at $DB_HOST:$DB_PORT"
fi
log "postgres reachable at $DB_HOST:$DB_PORT"

# ── Redis reachability ─────────────────────────────────────────────────
REDIS_URL="${REDIS_URL:-redis://localhost:6379}"
REDIS_HOST=$(echo "$REDIS_URL" | sed -E 's|redis://([^:/]+).*|\1|')
REDIS_PORT=$(echo "$REDIS_URL" | sed -E 's|redis://[^:/]+:([0-9]+).*|\1|')
if ! redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping >/dev/null 2>&1; then
  fail "redis not reachable at $REDIS_HOST:$REDIS_PORT"
fi
log "redis reachable at $REDIS_HOST:$REDIS_PORT"

log "preflight OK — all prerequisites satisfied"
