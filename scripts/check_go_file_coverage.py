#!/usr/bin/env python3
"""Verify Go service and handler files have Python counterparts.

Scans the upstream Go source tree (``internal/service/`` and
``internal/handler/``) and reports any Go file whose domain has no
corresponding module under ``src/core/`` or ``src/web/api/``.

The mapping is by domain keyword: a Go file ``knowledge.go`` maps to a
Python module containing ``knowledge`` in its path. This is a heuristic
coverage check, not a 1:1 file matcher.

Usage::

    python check_go_file_coverage.py [--go-root PATH] [--src-root PATH]

Exit codes:
    0 = every Go domain has a Python counterpart
    1 = at least one Go domain is uncovered
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def _resolve(explicit: str | None, default: Path) -> Path:
    if explicit:
        return Path(explicit).resolve()
    return default


def _go_domains(go_root: Path) -> list[str]:
    """Extract domain keywords from Go service/handler file names."""
    domains: set[str] = set()
    for sub in ("service", "handler"):
        d = go_root / "internal" / sub
        if not d.is_dir():
            continue
        for go in d.glob("*.go"):
            stem = go.stem
            # Strip common suffixes: _service, _handler, _impl
            stem = re.sub(r"_(service|handler|impl)$", "", stem)
            domains.add(stem)
    return sorted(domains)


def _python_domains(src_root: Path) -> set[str]:
    """Collect domain keywords from Python module paths."""
    domains: set[str] = set()
    for py in (src_root / "src").rglob("*.py"):
        parts = py.relative_to(src_root / "src").parts
        for part in parts:
            if part in {"core", "web", "api", "service", "views", "router"}:
                continue
            if part.endswith(".py"):
                part = part[:-3]
            domains.add(part)
    return domains


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--go-root", default=None)
    parser.add_argument("--src-root", default=None)
    args = parser.parse_args()

    go_root = _resolve(args.go_root, Path("/Users/jung/pro/Knowledge Base"))
    src_root = _resolve(args.src_root, Path(__file__).resolve().parent.parent)

    go_domains = _go_domains(go_root)
    py_domains = _python_domains(src_root)

    uncovered = [d for d in go_domains if d not in py_domains]
    for d in uncovered:
        print(f"[UNCOVERED] Go domain {d!r} has no Python module")

    if uncovered:
        print(f"[FAIL] {len(uncovered)} uncovered Go domain(s)")
        return 1
    print(f"[PASS] all {len(go_domains)} Go domains covered by Python modules")
    return 0


if __name__ == "__main__":
    sys.exit(main())
