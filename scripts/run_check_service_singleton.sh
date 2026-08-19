#!/bin/bash
# Wrapper that always invokes the local scripts/check_service_singleton.py,
# bypassing the older copy linked from the project's private agent
# workspace (which predates this repo's scope-discipline semantics).
#
# Usage: scripts/run_check_service_singleton.sh [args passed through]
set -euo pipefail
exec python3 "$(dirname "$0")/check_service_singleton.py" "$@"