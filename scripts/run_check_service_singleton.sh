#!/bin/bash
# Wrapper that always invokes the local scripts/check_service_singleton.py,
# bypassing the symlinked copy at .ai-context/scripts/check_service_singleton.py
# (which points at the older WeKnora upstream version and predates this
# repo's scope-discipline semantics).
#
# Usage: scripts/run_check_service_singleton.sh [args passed through]
set -euo pipefail
exec python3 "$(dirname "$0")/check_service_singleton.py" "$@"