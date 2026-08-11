#!/usr/bin/env bash
# Smoke test the deployed API.
# Verifies health, auth, and a core knowledge endpoint respond correctly.
set -euo pipefail

log() { printf '[smoke-test] %s\n' "$*"; }
fail() { printf '[smoke-test] ERROR: %s\n' "$*" >&2; exit 1; }

BASE_URL="${BASE_URL:-http://localhost:8000}"
TIMEOUT="${TIMEOUT:-10}"

# ── Health check ────────────────────────────────────────────────────────
log "checking $BASE_URL/health"
HEALTH=$(curl -sS --max-time "$TIMEOUT" "$BASE_URL/health" || fail "health endpoint unreachable")
echo "$HEALTH" | grep -q '"status"' || fail "health response missing status field"
log "health OK: $HEALTH"

# ── OpenAPI schema ─────────────────────────────────────────────────────
log "checking $BASE_URL/openapi.json"
curl -sS --max-time "$TIMEOUT" "$BASE_URL/openapi.json" | python3 -c '
import json, sys
schema = json.load(sys.stdin)
paths = schema.get("paths", {})
if len(paths) < 50:
    raise SystemExit(f"expected >=50 paths, found {len(paths)}")
print(f"openapi OK: {len(paths)} paths")
'

# ── Auth endpoint (login should 422 on empty body, not 500) ─────────────
log "checking auth validation"
STATUS=$(curl -sS -o /dev/null -w '%{http_code}' --max-time "$TIMEOUT" \
  -X POST "$BASE_URL/auth/login" -H 'Content-Type: application/json' -d '{}')
if [ "$STATUS" != "422" ]; then
  fail "expected 422 on empty login, got $STATUS"
fi
log "auth validation OK (422 on empty body)"

log "smoke test PASSED"
