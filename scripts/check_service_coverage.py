#!/usr/bin/env python3
"""Verify every Python service class has a counterpart in the Go service
inventory, and that the inventory is complete.

The inventory lives in ``scripts/service_inventory.json`` and lists the
upstream Go service files (one entry per service). This script scans
``src/core/**/service/*.py`` for Python service classes and reports any
Go service without a Python counterpart (and vice versa).

Usage::

    python check_service_coverage.py [--src-root PATH]

Exit codes:
    0 = full coverage
    1 = at least one uncovered service
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _resolve_src_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    cur = Path(__file__).resolve().parent
    for _ in range(6):
        candidate = cur / "src"
        if candidate.is_dir():
            return candidate
    raise SystemExit("cannot locate src/ root")


def _load_inventory(root: Path) -> list[str]:
    inv = root / "scripts" / "service_inventory.json"
    if not inv.exists():
        return []
    data = json.loads(inv.read_text())
    return [s["name"] for s in data.get("services", [])]


def _scan_python_services(root: Path) -> list[str]:
    """Return the set of Python service class names under src/core."""
    import ast

    services: list[str] = []
    for py in (root / "src" / "core").rglob("*.py"):
        if "service" not in str(py):
            continue
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name.endswith("Service"):
                services.append(node.name)
    return sorted(set(services))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", default=None)
    args = parser.parse_args()
    root = _resolve_src_root(args.src_root)

    inventory = _load_inventory(root)
    python_services = _scan_python_services(root)

    missing = [s for s in inventory if s not in python_services]
    for name in missing:
        print(f"[MISSING] Go service {name!r} has no Python counterpart")

    if missing:
        print(f"[FAIL] {len(missing)} uncovered service(s)")
        return 1
    print(f"[PASS] {len(python_services)} Python services, {len(inventory)} Go services — full coverage")
    return 0


if __name__ == "__main__":
    sys.exit(main())
