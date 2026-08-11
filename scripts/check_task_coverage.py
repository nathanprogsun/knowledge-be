#!/usr/bin/env python3
"""Verify all 19 upstream worker task types have a registered Python handler.

The upstream asynq task type constants are the source of truth. This script
imports every ``src/workers/tasks/*`` module (which triggers registration)
and compares the registered task names against the expected set.

Usage::

    python check_task_coverage.py [--src-root PATH]

Exit codes:
    0 = all tasks registered
    1 = at least one task missing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# The 19 upstream asynq task type constants (verbatim).
EXPECTED_TASKS: frozenset[str] = frozenset(
    {
        "chunk:extract",
        "datasource_sync",
        "datatable:summary",
        "document_process",
        "faq:import",
        "image_multimodal",
        "index:delete",
        "kb:clone",
        "kb:delete",
        "knowledge:list_delete",
        "knowledge:list_reparse",
        "knowledge:move",
        "knowledge:post_process",
        "manual_process",
        "question:generation",
        "summary:generation",
        "temporary_document:process",
        "wiki:finalize",
        "wiki:ingest",
    }
)


def _resolve_src_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    cur = Path(__file__).resolve().parent
    for _ in range(6):
        candidate = cur / "src"
        if candidate.is_dir():
            return candidate
    raise SystemExit("cannot locate src/ root")


def _registered_tasks(root: Path) -> set[str]:
    """Import every task module and return the registered task names."""
    import importlib
    import pkgutil

    import src.workers.tasks as tasks_pkg

    for mod in pkgutil.iter_modules(tasks_pkg.__path__):
        importlib.import_module(f"{tasks_pkg.__name__}.{mod.name}")

    from src.workers.registry import all_tasks

    return set(all_tasks().keys())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", default=None)
    args = parser.parse_args()
    root = _resolve_src_root(args.src_root)
    sys.path.insert(0, str(root))

    registered = _registered_tasks(root)
    missing = sorted(EXPECTED_TASKS - registered)
    extra = sorted(registered - EXPECTED_TASKS)

    for name in missing:
        print(f"[MISSING] task {name!r} is not registered")
    for name in extra:
        print(f"[EXTRA] task {name!r} is registered but not in the expected set")

    if missing or extra:
        print(f"[FAIL] {len(missing)} missing, {len(extra)} extra")
        return 1
    print(f"[PASS] all {len(EXPECTED_TASKS)} tasks registered")
    return 0


if __name__ == "__main__":
    sys.exit(main())
