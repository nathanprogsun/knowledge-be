#!/bin/bash
# Run all anti-drift checks. Exit 1 if any fail.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_ROOT="${1:-$(pwd)}"

echo "Running anti-drift checks..."
echo "Source root: $SRC_ROOT"
echo ""

FAILED=0

for script in check_layer_violation check_service_singleton check_endpoint_coverage check_schema_compatibility check_contract_invariants check_imports; do
    echo "--- $script ---"
    if python3 "$SCRIPT_DIR/$script.py" --src-root "$SRC_ROOT"; then
        echo "[PASS] $script"
    else
        echo "[FAIL] $script"
        FAILED=1
    fi
    echo ""
done

if [ $FAILED -eq 0 ]; then
    echo "=== All checks passed ==="
    exit 0
else
    echo "=== Some checks failed ==="
    exit 1
fi